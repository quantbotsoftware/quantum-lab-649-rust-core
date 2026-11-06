//! Verificarea încrucișată Rust ↔ Python ÎNGHEȚAT (protocol S19 §2–§4, criteriile B2–B7).
//!
//! Intrări: referința `REFERINTA_PARITATE_<UTC>.json` scrisă de `tu4/s19_paritate_ref.py` (fiecare float și în hex,
//! decodat EXACT de [`parse_hex_f64`]), emisiile sigilate pentru ținta dată (citite read-only) și fișierul S17.
//! Toleranțe absolute fixate în protocol: 1e-12 pentru toate mărimile float, Δ = 0 pentru numărări, 1e-10 pentru
//! FrFT față de NumPy/LAPACK și pentru F_1 față de DFT. Nicio emisie, niciun bilet nou, nicio decontare.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::fisher::{density_report, inclusion, info_d, pair_inclusion, softmax};
use super::h1_mass::{mass_correlation, H1Mass};
use super::h3_aero::{cosine_basis, H3Aero, K_AD};
use super::spectral::admissible::{admissible, enumerate_all, top_ticket, LinearTable};
use super::spectral::frft::{complex_matmul, dft_matrix, identity_residual, max_abs_diff, unitarity_residual, Frft};
use super::spectral::{label_dft, neumaier_sum, pair_correlation, CountState, UrnSpec};
use crate::parser::{self, Game};

pub const TOL: f64 = 1e-12;
pub const TOL_FRFT: f64 = 1e-10;
pub const A_GRID: [f64; 7] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9];

/// Decodează EXACT formatul `float.hex()` din Python (`[-]0x1.hhhhp±e`, `0x0.0p+0`, subnormale, `inf`, `nan`).
pub fn parse_hex_f64(s: &str) -> Result<f64, String> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let sign = if neg { -1.0 } else { 1.0 };
    match body {
        "inf" => return Ok(sign * f64::INFINITY),
        "nan" => return Ok(f64::NAN),
        _ => {}
    }
    let body = body.strip_prefix("0x").ok_or_else(|| format!("hex invalid: {s}"))?;
    let (mant, exp) = body.split_once('p').ok_or_else(|| format!("hex fără exponent: {s}"))?;
    let exp: i32 = exp.parse().map_err(|_| format!("exponent invalid: {s}"))?;
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    if fp.len() > 13 {
        return Err(format!("prea multe cifre hex: {s}"));
    }
    let lead: u64 = match ip {
        "0" => 0,
        "1" => 1,
        _ => return Err(format!("parte întreagă invalidă: {s}")),
    };
    let mut frac: u64 = 0;
    for ch in fp.chars() {
        frac = frac << 4 | u64::from(ch.to_digit(16).ok_or_else(|| format!("cifră hex invalidă: {s}"))?);
    }
    frac <<= 4 * (13 - fp.len() as u32);
    let bits = if lead == 1 {
        let biased = exp + 1023;
        if !(1..=2046).contains(&biased) {
            return Err(format!("exponent în afara domeniului normal: {s}"));
        }
        (biased as u64) << 52 | frac
    } else if frac == 0 {
        0
    } else if exp == -1022 {
        frac
    } else {
        return Err(format!("subnormal necanonic: {s}"));
    };
    let v = f64::from_bits(bits);
    Ok(if neg { -v } else { v })
}

#[derive(Debug, Clone, Deserialize)]
pub struct HexArr {
    pub hex: Vec<String>,
}

impl HexArr {
    pub fn values(&self) -> Result<Vec<f64>, String> {
        self.hex.iter().map(|h| parse_hex_f64(h)).collect()
    }
    pub fn scalar(&self) -> Result<f64, String> {
        self.values()?.first().copied().ok_or_else(|| "scalar gol".to_string())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRef {
    pub logw: HexArr,
    pub lambda: HexArr,
    #[serde(rename = "D")]
    pub d: HexArr,
    pub pi: HexArr,
    #[serde(rename = "Pi")]
    pub pi2: HexArr,
    pub q: Option<HexArr>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaseRef {
    pub name: String,
    pub kind: String,
    pub n: usize,
    pub k: usize,
    #[serde(rename = "N")]
    pub big_n: u64,
    pub game: Option<String>,
    pub urn: String,
    pub source: Option<Value>,
    pub draws: Option<Vec<Vec<u8>>>,
    pub counts: Vec<u64>,
    pub pairs: Vec<u64>,
    pub theta: HexArr,
    pub x2: HexArr,
    pub basis: HexArr,
    pub h1: ModelRef,
    pub h3: ModelRef,
    #[serde(rename = "R")]
    pub r: HexArr,
    pub z: HexArr,
    pub dft_re: HexArr,
    pub dft_im: HexArr,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FrftRef {
    pub n: usize,
    pub a: f64,
    pub re: HexArr,
    pub im: HexArr,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Reference {
    pub schema: String,
    pub run_utc: String,
    pub cases: Vec<CaseRef>,
    pub frft: Vec<FrftRef>,
}

/// O comparație raportată.
#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub group: &'static str,
    pub case: String,
    pub quantity: String,
    pub n_values: usize,
    pub n_bitexact: usize,
    pub max_abs_delta: f64,
    pub tol: f64,
    pub pass: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

#[derive(Debug, Default)]
pub struct Checks(pub Vec<Check>);

impl Checks {
    fn floats(&mut self, group: &'static str, case: &str, q: &str, got: &[f64], want: &[f64], tol: f64) {
        let ok_len = got.len() == want.len();
        let mut worst = if ok_len { 0.0f64 } else { f64::INFINITY };
        let mut bit = 0usize;
        for (a, b) in got.iter().zip(want) {
            let d = (a - b).abs();
            worst = if d.is_nan() { f64::INFINITY } else { worst.max(d) };
            bit += usize::from(a.to_bits() == b.to_bits() || (*a == 0.0 && *b == 0.0));
        }
        self.0.push(Check {
            group,
            case: case.to_string(),
            quantity: q.to_string(),
            n_values: want.len(),
            n_bitexact: bit,
            max_abs_delta: worst,
            tol,
            pass: ok_len && worst <= tol,
            detail: None,
        });
    }

    fn exact<T: PartialEq>(&mut self, group: &'static str, case: &str, q: &str, got: &[T], want: &[T]) {
        let same = got.iter().zip(want).filter(|(a, b)| a == b).count();
        let ok = got.len() == want.len() && same == want.len();
        self.0.push(Check {
            group,
            case: case.to_string(),
            quantity: q.to_string(),
            n_values: want.len(),
            n_bitexact: same,
            max_abs_delta: if ok { 0.0 } else { f64::INFINITY },
            tol: 0.0,
            pass: ok,
            detail: None,
        });
    }

    fn flag(&mut self, group: &'static str, case: &str, q: &str, pass: bool, detail: Value) {
        self.0.push(Check {
            group,
            case: case.to_string(),
            quantity: q.to_string(),
            n_values: 1,
            n_bitexact: usize::from(pass),
            max_abs_delta: if pass { 0.0 } else { f64::INFINITY },
            tol: 0.0,
            pass,
            detail: Some(detail),
        });
    }

    fn bound(&mut self, group: &'static str, case: &str, q: &str, value: f64, tol: f64) {
        self.0.push(Check {
            group,
            case: case.to_string(),
            quantity: q.to_string(),
            n_values: 1,
            n_bitexact: 0,
            max_abs_delta: value,
            tol,
            pass: value.is_finite() && value <= tol,
            detail: None,
        });
    }
}

// ------------------------------------------------------------------ calea pe numărări reale (pentru W3-S19)

/// H1 pe numărări reale c (aceeași ordine a operațiilor ca `H1Mass::from_counts`). Întoarce (ln ŵ, λ).
pub fn h1_float(c: &[f64], big_n: f64, k: usize) -> (Vec<f64>, f64) {
    let n = c.len();
    let (nf, p) = (n as f64, k as f64 / n as f64);
    let shrink = (nf - 1.0) / nf;
    let theta: Vec<f64> = c.iter().map(|ci| shrink * (ci / big_n - p) / (p * (1.0 - p))).collect();
    let dev2 = neumaier_sum(c.iter().map(|ci| {
        let d = ci - big_n * p;
        d * d
    }));
    let x2 = (nf - 1.0) / nf * dev2 / (big_n * p * (1.0 - p));
    let lambda = if x2 > 0.0 && x2.is_finite() { (1.0 - (nf - 3.0) / x2).max(0.0) } else { 0.0 };
    (theta.iter().map(|t| lambda * t).collect(), lambda)
}

/// H3 pe numărări reale c (aceeași ordine a operațiilor ca `H3Aero::from_counts`). Întoarce (ln ŵ, λ_s).
pub fn h3_float(c: &[f64], big_n: f64, k: usize) -> (Vec<f64>, f64) {
    let n = c.len();
    let (nf, p) = (n as f64, k as f64 / n as f64);
    let shrink = (nf - 1.0) / nf;
    let theta: Vec<f64> = c.iter().map(|ci| shrink * (ci / big_n - p) / (p * (1.0 - p))).collect();
    let basis = cosine_basis(n, K_AD).unwrap_or_default();
    let scale = ((nf - 1.0) / nf).sqrt();
    let sd = (big_n * p * (1.0 - p)).sqrt();
    let z: Vec<f64> = c.iter().map(|ci| scale * (ci - big_n * p) / sd).collect();
    let dot = |a: &[f64], b: &[f64]| neumaier_sum(a.iter().zip(b).map(|(x, y)| x * y));
    let scores: Vec<f64> = basis.iter().map(|b| dot(&z, b)).collect();
    let x2s = neumaier_sum(scores.iter().map(|s| s * s));
    let coef: Vec<f64> = basis.iter().map(|b| dot(&theta, b)).collect();
    let ths: Vec<f64> = (0..n).map(|i| neumaier_sum(basis.iter().zip(&coef).map(|(b, cf)| b[i] * cf))).collect();
    let lambda = if x2s > 0.0 && x2s.is_finite() { (1.0 - (K_AD as f64 - 2.0) / x2s).max(0.0) } else { 0.0 };
    (ths.iter().map(|t| lambda * t).collect(), lambda)
}

// ------------------------------------------------------------------ recalculul Rust al unui caz

/// Mărimile calculate de Rust pentru un caz.
pub struct RustCase {
    pub state: CountState,
    pub h1: H1Mass,
    pub h3: H3Aero,
}

fn game_of(tag: &str) -> Result<Game, String> {
    Game::from_tag(tag).ok_or_else(|| format!("joc necunoscut {tag}"))
}

/// Construiește starea numărărilor: din extragerile referinței (fixturi) sau parsând arhiva LIVE (cazuri reale).
pub fn build_state(case: &CaseRef, root: &Path) -> Result<CountState, String> {
    if let Some(draws) = &case.draws {
        let mut s = CountState::new(UrnSpec::new(case.n, case.k).map_err(|e| e.to_string())?);
        for d in draws {
            s.push(d).map_err(|e| format!("{}: {e}", case.name))?;
        }
        return Ok(s);
    }
    let src = case.source.as_ref().ok_or("caz real fără sursă")?;
    let rel = src.get("file").and_then(Value::as_str).ok_or("sursă fără fișier")?;
    let bytes = std::fs::read(root.join(rel)).map_err(|e| format!("{rel}: {e}"))?;
    let parsed = parser::parse_bytes(&bytes, None);
    if !parsed.rejects.is_empty() {
        return Err(format!("{rel}: {} respingeri", parsed.rejects.len()));
    }
    let game = game_of(case.game.as_deref().unwrap_or(""))?;
    if case.urn == "ball" {
        CountState::from_bonus(game, &parsed.records).map_err(|e| e.to_string())?.ok_or_else(|| "fără bilă".to_string())
    } else {
        CountState::from_main(game, &parsed.records).map_err(|e| e.to_string())
    }
}

pub fn compute_case(case: &CaseRef, root: &Path) -> Result<RustCase, String> {
    let state = build_state(case, root)?;
    let h1 = H1Mass::from_counts(&state).map_err(|e| e.to_string())?;
    let h3 = H3Aero::from_counts(&state, K_AD).map_err(|e| e.to_string())?;
    Ok(RustCase { state, h1, h3 })
}

fn model_checks(
    ck: &mut Checks,
    name: &str,
    tag: &str,
    logw: &[f64],
    lambda: f64,
    k: usize,
    m: &ModelRef,
) -> Result<(), String> {
    ck.floats("B3", name, &format!("{tag}.logw"), logw, &m.logw.values()?, TOL);
    ck.floats("B3", name, &format!("{tag}.lambda"), &[lambda], &[m.lambda.scalar()?], TOL);
    ck.floats("B4", name, &format!("{tag}.D"), &[info_d(logw, k)], &[m.d.scalar()?], TOL);
    ck.floats("B4", name, &format!("{tag}.pi"), &inclusion(logw, k), &m.pi.values()?, TOL);
    ck.floats("B4", name, &format!("{tag}.Pi"), &pair_inclusion(logw, k), &m.pi2.values()?, TOL);
    if let Some(q) = &m.q {
        ck.floats("B4", name, &format!("{tag}.q"), &softmax(logw), &q.values()?, TOL);
    }
    if k >= 2 {
        let r = density_report(logw, k).map_err(|e| e.to_string())?;
        ck.bound("B4", name, &format!("{tag}.rho.|trace-1|"), (r.trace - 1.0).abs(), TOL);
        ck.bound("B4", name, &format!("{tag}.rho.max(0,-min_eig)"), (-r.min_eigenvalue).max(0.0), TOL);
    }
    Ok(())
}

/// B2–B5 pentru un caz.
pub fn check_case(ck: &mut Checks, case: &CaseRef, rc: &RustCase) -> Result<(), String> {
    let name = case.name.as_str();
    let (n, k) = (case.n, case.k);
    let s = &rc.state;
    ck.exact("B2", name, "N", &[s.draws()], &[case.big_n]);
    ck.exact("B2", name, "counts", &s.counts(), &case.counts);
    let pairs: Vec<u64> = (0..n * n).map(|idx| s.pair(idx / n, idx % n)).collect();
    ck.exact("B2", name, "pairs", &pairs, &case.pairs);
    ck.floats("B3", name, "theta", &rc.h1.theta, &case.theta.values()?, TOL);
    ck.floats("B3", name, "x2", &[rc.h1.x2], &[case.x2.scalar()?], TOL);
    let basis = cosine_basis(n, K_AD).map_err(|e| e.to_string())?;
    let flat: Vec<f64> = (0..n).flat_map(|i| basis.iter().map(move |b| b[i])).collect();
    ck.floats("B3", name, "basis", &flat, &case.basis.values()?, TOL);
    model_checks(ck, name, "h1", &rc.h1.log_w, rc.h1.lambda, k, &case.h1)?;
    model_checks(ck, name, "h3", &rc.h3.log_w, rc.h3.lambda_s, k, &case.h3)?;
    let r = pair_correlation(s).map_err(|e| e.to_string())?;
    let rflat: Vec<f64> = (0..n * n).map(|idx| r.get(idx / n, idx % n).re).collect();
    ck.floats("B5", name, "R", &rflat, &case.r.values()?, TOL);
    let imag_max = (0..n * n).map(|idx| r.get(idx / n, idx % n).im.abs()).fold(0.0, f64::max);
    ck.bound("B5", name, "R.hermiticity_residual+max|Im|", r.hermiticity_residual() + imag_max, 0.0);
    let mc = mass_correlation(s, &rc.h1).map_err(|e| e.to_string())?;
    ck.bound("B5", name, "R.trace_vs_correlation", (mc.trace - r.trace()).abs(), TOL);
    ck.floats("B5", name, "z", &rc.h3.z, &case.z.values()?, TOL);
    let spec = label_dft(&rc.h3.z);
    let re: Vec<f64> = spec.iter().map(|c| c.re).collect();
    let im: Vec<f64> = spec.iter().map(|c| c.im).collect();
    ck.floats("B5", name, "dft.re", &re, &case.dft_re.values()?, TOL);
    ck.floats("B5", name, "dft.im", &im, &case.dft_im.values()?, TOL);
    Ok(())
}

// ------------------------------------------------------------------ emisiile sigilate și S17

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}

fn u8s(v: &Value) -> Option<Vec<u8>> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u8)).collect())
}

pub fn emission_path(root: &Path, game: &str, target: &str) -> PathBuf {
    root.join("tu4").join("tribunal").join("emisii").join(game).join(format!("{target}.json"))
}

pub fn load_json(p: &Path) -> Result<Value, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))
}

/// B2/B3/B4 față de câmpurile emisiei — NUMAI dacă emisia a folosit aceleași înregistrări (număr + ultima cheie) ca
/// arhiva curentă; altfel se raportează doar indicatorul `aceeasi_arhiva_ca_emisia` = false.
pub fn check_emission(ck: &mut Checks, case: &CaseRef, rc: &RustCase, em: &Value) {
    let name = format!("{}~emisie", case.name);
    let src = case.source.as_ref();
    let same = src.and_then(|s| s.get("records")).and_then(Value::as_u64) == em["archive"]["records"].as_u64()
        && src.and_then(|s| s.get("last_key")).and_then(Value::as_str) == em["archive"]["last_key"].as_str();
    ck.flag("B2", &name, "aceeasi_arhiva_ca_emisia", true, serde_json::json!({"aceleasi_inregistrari": same}));
    if !same {
        // Arhiva LIVE s-a schimbat după emisie: comparațiile cu emisia nu sunt definite (protocol §4 B2/B3); se raportează.
        return;
    }
    let k = case.k;
    if case.urn == "ball" {
        {
            let want: Vec<u64> = em["warm_start"]["ball_counts"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            ck.exact("B2", &name, "ball_counts=warm_start", &rc.state.counts(), &want);
        }
        for (tag, logw, lam, x2) in
            [("H1", &rc.h1.log_w, rc.h1.lambda, rc.h1.x2), ("H3", &rc.h3.log_w, rc.h3.lambda_s, rc.h3.x2_s)]
        {
            let b = &em["models"][tag]["ball"];
            ck.floats("B3", &name, &format!("{tag}.ball.logw"), logw, &f64s(&b["logw"]), TOL);
            ck.floats(
                "B3",
                &name,
                &format!("{tag}.ball.lambda"),
                &[lam],
                &[b["lambda"].as_f64().unwrap_or(f64::NAN)],
                TOL,
            );
            ck.floats("B3", &name, &format!("{tag}.ball.X2"), &[x2], &[b["X2"].as_f64().unwrap_or(f64::NAN)], TOL);
            ck.floats(
                "B4",
                &name,
                &format!("{tag}.ball.D_nats"),
                &[info_d(logw, k)],
                &[b["D_nats"].as_f64().unwrap_or(f64::NAN)],
                TOL,
            );
            ck.floats("B4", &name, &format!("{tag}.ball.q"), &softmax(logw), &f64s(&b["q"]), TOL);
        }
        return;
    }
    {
        let want: Vec<u64> = em["warm_start"]["counts"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        ck.exact("B2", &name, "counts=warm_start", &rc.state.counts(), &want);
        ck.exact("B2", &name, "N=warm_start", &[rc.state.draws()], &[em["warm_start"]["N"].as_u64().unwrap_or(0)]);
    }
    for (tag, logw, lam, x2) in
        [("H1", &rc.h1.log_w, rc.h1.lambda, rc.h1.x2), ("H3", &rc.h3.log_w, rc.h3.lambda_s, rc.h3.x2_s)]
    {
        let b = &em["models"][tag];
        ck.floats("B3", &name, &format!("{tag}.logw"), logw, &f64s(&b["logw"]), TOL);
        ck.floats("B3", &name, &format!("{tag}.lambda"), &[lam], &[b["lambda"].as_f64().unwrap_or(f64::NAN)], TOL);
        ck.floats("B3", &name, &format!("{tag}.X2"), &[x2], &[b["X2"].as_f64().unwrap_or(f64::NAN)], TOL);
        ck.floats(
            "B4",
            &name,
            &format!("{tag}.D_nats"),
            &[info_d(logw, k)],
            &[b["D_nats"].as_f64().unwrap_or(f64::NAN)],
            TOL,
        );
        ck.floats("B4", &name, &format!("{tag}.pi"), &inclusion(logw, k), &f64s(&b["pi"]), TOL);
    }
}

/// B7 (A1 + A3): heap-ul și enumerarea Rust pe ln ŵ din emisii, față de fișierul S17.
pub fn check_admissibility(ck: &mut Checks, emissions: &BTreeMap<String, Value>, s17: &Value) -> Vec<Value> {
    let mut enums = Vec::new();
    for (n, b) in [(49usize, 6usize), (40, 5), (45, 5)] {
        let e = enumerate_all(&vec![0.0; n], b);
        let t = LinearTable::new(n, b);
        let closed = super::spectral::admissible::ap_count_closed(n, b);
        let ok =
            e.as_ref().is_some_and(|e| e.n_linear == closed && e.n_combos == binom(n, b)) && t.len() as u64 == closed;
        ck.flag(
            "B7",
            &format!("{n}/{b}"),
            "A1_progresii",
            ok,
            serde_json::json!({
            "enumerat": e.as_ref().map(|e| e.n_linear), "forma_inchisa": closed, "tabela": t.len(),
            "total": e.as_ref().map(|e| e.n_combos), "C(n,b)": binom(n, b), "secunde": e.as_ref().map(|e| e.seconds)}),
        );
    }
    for (g, em) in emissions {
        let b = em["ticket_size"].as_u64().unwrap_or(0) as usize;
        for m in ["H1", "H3"] {
            let key = format!("{g}/{m}");
            let rec = &s17["streams"][&key];
            let blk = &em["models"][m];
            let lw = f64s(&blk["logw"]);
            let a = admissible(&lw, b);
            let want_top = u8s(&rec["top_ticket_emis"]).unwrap_or_default();
            let mut fields = vec![
                ("stare", serde_json::json!(a.stare.as_str()) == rec["stare"]),
                ("top_ticket_emis", a.top_ticket == want_top),
                (
                    "top=ip_core.ticket",
                    a.top_ticket == top_ticket(&lw, b) && Some(a.top_ticket.clone()) == u8s(&blk["top_ticket"]),
                ),
                ("top_emis_liniar", serde_json::json!(a.top_liniar) == rec["top_emis_liniar"]),
                ("bilet_afisare", serde_json::json!(a.bilet_afisare) == rec["bilet_afisare"]),
                ("rang", serde_json::json!(a.rang) == rec.get("rang").cloned().unwrap_or(Value::Null)),
                (
                    "noduri_heap",
                    serde_json::json!(a.noduri_heap) == rec.get("noduri_heap").cloned().unwrap_or(Value::Null),
                ),
            ];
            if let Some(jb) = blk.get("ball").filter(|_| em.get("ball_n").and_then(Value::as_u64).is_some()) {
                let blw = f64s(&jb["logw"]);
                let active = blw.iter().any(|v| v.abs() != 0.0);
                let (stare, afis) = if active {
                    ("ACTIVA", serde_json::json!(top_ticket(&blw, 1)[0]))
                } else {
                    ("NU_PARIAZA", Value::Null)
                };
                fields.push(("bila_stare", serde_json::json!(stare) == rec["bila_stare"]));
                fields.push(("bila_afisare", afis == rec["bila_afisare"]));
            }
            let all_ok = fields.iter().all(|f| f.1);
            ck.flag(
                "B7",
                &key,
                "A3_S17_campuri",
                all_ok,
                serde_json::json!({
                "rust": a, "campuri": fields.iter().map(|(n, ok)| (n.to_string(), *ok)).collect::<BTreeMap<_, _>>()}),
            );
            if let (Some(t), Some(ad), Some(r)) = (a.ln_q_u_top, a.ln_q_u_admisibil, a.q_adm_pe_q_top) {
                ck.floats("B7", &key, "ln_q_u_top", &[t], &[rec["ln_q_u_top"].as_f64().unwrap_or(f64::NAN)], TOL);
                ck.floats(
                    "B7",
                    &key,
                    "ln_q_u_admisibil",
                    &[ad],
                    &[rec["ln_q_u_admisibil"].as_f64().unwrap_or(f64::NAN)],
                    TOL,
                );
                ck.floats(
                    "B7",
                    &key,
                    "q_adm_pe_q_top",
                    &[r],
                    &[rec["q_adm_pe_q_top"].as_f64().unwrap_or(f64::NAN)],
                    TOL,
                );
                if let Some(e) = enumerate_all(&lw, b) {
                    let ok = Some(e.top.clone()) == Some(a.top_ticket.clone())
                        && Some(e.admissible.clone()) == a.bilet_afisare
                        && Some(e.rank) == a.rang
                        && serde_json::json!(e.rank) == rec["enumerare"]["rank"];
                    ck.flag("B7", &key, "enumerare_completa", ok, serde_json::json!(e));
                    enums.push(serde_json::json!({"stream": key, "enumerare": e}));
                }
            }
        }
    }
    enums
}

fn binom(n: usize, k: usize) -> u64 {
    (0..k).fold(1u64, |c, i| c * (n - i) as u64 / (i + 1) as u64)
}

// ------------------------------------------------------------------ FrFT (B6)

pub fn check_frft(ck: &mut Checks, refs: &[FrftRef]) -> Result<(), String> {
    let mut plans: BTreeMap<usize, Frft> = BTreeMap::new();
    for n in [20usize, 40, 45, 49] {
        let f = Frft::new(n).map_err(|e| e.to_string())?;
        let case = format!("n={n}");
        ck.bound("B6", &case, "W0.F0-I", identity_residual(&f.matrix(0.0), n), TOL);
        ck.bound("B6", &case, "W0.F1-DFT", max_abs_diff(&f.matrix(1.0), &dft_matrix(n)), TOL_FRFT);
        for (a, b) in [(0.3, 0.4), (0.5, 0.5), (0.25, 1.5)] {
            let d = max_abs_diff(&complex_matmul(&f.matrix(a), &f.matrix(b), n), &f.matrix(a + b));
            ck.bound("B6", &case, &format!("W0.F{a}F{b}-F{}", a + b), d, TOL);
        }
        for a in A_GRID {
            ck.bound("B6", &case, &format!("W1.unitar(a={a})"), unitarity_residual(&f.matrix(a), n), TOL);
        }
        ck.bound("B6", &case, "jacobi.reziduu", f.eigen_residual(), TOL);
        plans.insert(n, f);
    }
    for r in refs {
        let f = plans.get(&r.n).ok_or("n FrFT neașteptat")?;
        let want: Vec<Complex64> =
            r.re.values()?.into_iter().zip(r.im.values()?).map(|(a, b)| Complex64::new(a, b)).collect();
        ck.bound(
            "B6",
            &format!("n={}", r.n),
            &format!("incrucisat_numpy(a={})", r.a),
            max_abs_diff(&f.matrix(r.a), &want),
            TOL_FRFT,
        );
    }
    Ok(())
}

/// W2 + W3-S19 pe un caz real cu modelul activ: energia, ln ŵ și biletul admisibil după dus-întorsul FrFT.
pub fn check_frft_round_trip(ck: &mut Checks, case: &CaseRef, rc: &RustCase, b: usize) -> Result<(), String> {
    let n = case.n;
    let k = case.k;
    let f = Frft::new(n).map_err(|e| e.to_string())?;
    let big_n = rc.state.draws() as f64;
    let p = k as f64 / n as f64;
    let c: Vec<f64> = rc.state.counts().iter().map(|&v| v as f64).collect();
    let (lw1, l1) = h1_float(&c, big_n, k);
    let (lw3, l3) = h3_float(&c, big_n, k);
    ck.floats("B6", &case.name, "cale_float=cale_intreaga.h1", &lw1, &rc.h1.log_w, 0.0);
    ck.floats("B6", &case.name, "cale_float=cale_intreaga.h3", &lw3, &rc.h3.log_w, 0.0);
    let ez = neumaier_sum(rc.h3.z.iter().map(|v| v * v));
    for a in A_GRID {
        let y = f.apply_real(a, &rc.h3.z).map_err(|e| e.to_string())?;
        let e1 = neumaier_sum(y.iter().map(|v| v.norm_sqr()));
        ck.bound("B6", &case.name, &format!("W2.energie_rel(a={a})"), (e1 - ez).abs() / ez.max(f64::MIN_POSITIVE), TOL);
        let dev: Vec<f64> = c.iter().map(|ci| ci - big_n * p).collect();
        let back = f.apply_inverse(a, &f.apply_real(a, &dev).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let c2: Vec<f64> = back.iter().map(|v| big_n * p + v.re).collect();
        for (tag, active, lw0, l0, (lw2, l2)) in [
            ("h1", rc.h1.lambda > 0.0, &rc.h1.log_w, l1, h1_float(&c2, big_n, k)),
            ("h3", rc.h3.lambda_s > 0.0, &rc.h3.log_w, l3, h3_float(&c2, big_n, k)),
        ] {
            if !active {
                continue;
            }
            ck.floats("B6", &case.name, &format!("W3.{tag}.logw(a={a})"), &lw2, lw0, TOL);
            ck.floats("B6", &case.name, &format!("W3.{tag}.lambda(a={a})"), &[l2], &[l0], TOL);
            if k > 1 {
                let a0 = admissible(lw0, b);
                let a2 = admissible(&lw2, b);
                ck.flag(
                    "B6",
                    &case.name,
                    &format!("W3.{tag}.admisibil_neschimbat(a={a})"),
                    a0.bilet_afisare == a2.bilet_afisare && a0.stare == a2.stare,
                    serde_json::json!({"inainte": a0.bilet_afisare, "dupa": a2.bilet_afisare}),
                );
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ rularea completă

/// Timpul UTC curent în formatul ISO (fără dependențe; algoritmul „days from civil” al lui H. Hinnant).
pub fn utc_now_iso() -> String {
    let secs =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Rulează B2–B7 și întoarce raportul JSON.
pub fn run(reference: &Path, root: &Path, target: &str, s17_path: &Path) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    let reff: Reference =
        serde_json::from_slice(&std::fs::read(reference).map_err(|e| format!("{}: {e}", reference.display()))?)
            .map_err(|e| format!("referință: {e}"))?;
    if reff.schema != "ql649-s19-paritate-ref/1" {
        return Err(format!("schema necunoscută {}", reff.schema));
    }
    let mut ck = Checks::default();
    let mut emissions = BTreeMap::new();
    for g in ["649", "540", "joker"] {
        emissions.insert(g.to_string(), load_json(&emission_path(root, g, target))?);
    }
    let mut timings = BTreeMap::new();
    for case in &reff.cases {
        let tc = std::time::Instant::now();
        let rc = compute_case(case, root)?;
        check_case(&mut ck, case, &rc)?;
        if case.kind == "real" {
            let g = case.game.clone().unwrap_or_default();
            if let Some(em) = emissions.get(&g) {
                check_emission(&mut ck, case, &rc, em);
                let b = em["ticket_size"].as_u64().unwrap_or(case.k as u64) as usize;
                if rc.h1.lambda > 0.0 || rc.h3.lambda_s > 0.0 {
                    check_frft_round_trip(&mut ck, case, &rc, if case.urn == "ball" { 1 } else { b })?;
                }
            }
        }
        timings.insert(case.name.clone(), tc.elapsed().as_secs_f64());
    }
    let tf = std::time::Instant::now();
    check_frft(&mut ck, &reff.frft)?;
    timings.insert("FrFT".to_string(), tf.elapsed().as_secs_f64());
    let s17 = load_json(s17_path)?;
    let ta = std::time::Instant::now();
    let enums = check_admissibility(&mut ck, &emissions, &s17);
    timings.insert("admisibilitate".to_string(), ta.elapsed().as_secs_f64());
    let mut groups: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for c in &ck.0 {
        let e = groups.entry(c.group).or_default();
        e.0 += 1;
        e.1 += usize::from(c.pass);
    }
    let summary: BTreeMap<String, Value> = groups
        .iter()
        .map(|(g, (t, p))| {
            (
                g.to_string(),
                serde_json::json!({"checks": t, "pass": p, "verdict": if t == p { "PASS" } else { "FAIL" }}),
            )
        })
        .collect();
    let floats: Vec<&Check> = ck.0.iter().filter(|c| c.tol > 0.0 && c.n_values > 0).collect();
    let all_pass = ck.0.iter().all(|c| c.pass);
    Ok(serde_json::json!({
        "tool": "rust-singularity-core parity",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "TU4-S19 (tu4/PROTOCOL_SUPERQUANTUM_S19.md)",
        "run_utc": utc_now_iso(),
        "reference": reference.display().to_string(),
        "reference_run_utc": reff.run_utc,
        "target": target,
        "tolerances": {"float_abs": TOL, "frft": TOL_FRFT, "counts": "exact"},
        "n_checks": ck.0.len(),
        "n_pass": ck.0.iter().filter(|c| c.pass).count(),
        "verdict_B2_B7": if all_pass { "PASS" } else { "FAIL" },
        "groups": summary,
        "max_abs_delta_float": floats.iter().map(|c| c.max_abs_delta).fold(0.0, f64::max),
        "values_compared_float": floats.iter().map(|c| c.n_values).sum::<usize>(),
        "values_bitexact_float": floats.iter().map(|c| c.n_bitexact).sum::<usize>(),
        "failures": ck.0.iter().filter(|c| !c.pass).collect::<Vec<_>>(),
        "checks": ck.0,
        "enumerations": enums,
        "timings_seconds": timings,
        "total_seconds": t0.elapsed().as_secs_f64(),
        "nota": "paritate numerica si filtru de afisare; nu este predictie, emisie sau decontare",
        "positive_claims_allowed": crate::POSITIVE_CLAIMS_ALLOWED,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spectral::tests::synthetic;

    #[test]
    fn hex_parser_is_exact() {
        for (s, v) in [
            ("0x1.0000000000000p+0", 1.0),
            ("-0x1.8000000000000p+1", -3.0),
            ("0x0.0p+0", 0.0),
            ("0x1.999999999999ap-4", 0.1),
            ("0x1.fffffffffffffp+1023", f64::MAX),
            ("0x0.0000000000001p-1022", f64::from_bits(1)),
            ("0x1.0p-1022", f64::MIN_POSITIVE),
            ("inf", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
        ] {
            assert_eq!(parse_hex_f64(s).unwrap().to_bits(), v.to_bits(), "{s}");
        }
        assert_eq!(parse_hex_f64("-0x0.0p+0").unwrap().to_bits(), (-0.0f64).to_bits());
        assert!(parse_hex_f64("nan").unwrap().is_nan());
        for bad in ["1.0", "0x2.0p+0", "0x1.0", "0x1.zp+0", "0x1.0p+5000", "0x0.1p+0"] {
            assert!(parse_hex_f64(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn float_path_equals_integer_path_bitwise() {
        for (n, k, t, seed) in [(49usize, 6usize, 600usize, 3u64), (45, 5, 500, 4), (20, 1, 300, 5)] {
            let s = synthetic(n, k, t, seed);
            let h1 = H1Mass::from_counts(&s).unwrap();
            let h3 = H3Aero::from_counts(&s, K_AD).unwrap();
            let c: Vec<f64> = s.counts().iter().map(|&v| v as f64).collect();
            let (lw1, l1) = h1_float(&c, s.draws() as f64, k);
            let (lw3, l3) = h3_float(&c, s.draws() as f64, k);
            assert_eq!((lw1, l1.to_bits()), (h1.log_w.clone(), h1.lambda.to_bits()));
            assert_eq!((lw3, l3.to_bits()), (h3.log_w.clone(), h3.lambda_s.to_bits()));
        }
    }

    #[test]
    fn checks_record_bitexact_and_failures() {
        let mut ck = Checks::default();
        ck.floats("B3", "c", "q", &[1.0, 2.0], &[1.0, 2.0 + 1e-13], TOL);
        ck.floats("B3", "c", "q2", &[1.0], &[1.1], TOL);
        ck.exact("B2", "c", "e", &[1u64, 2], &[1, 2]);
        ck.exact("B2", "c", "e2", &[1u64], &[1, 2]);
        assert_eq!((ck.0[0].pass, ck.0[0].n_bitexact), (true, 1));
        assert!(!ck.0[1].pass && ck.0[2].pass && !ck.0[3].pass);
        let iso = utc_now_iso();
        assert!(iso.len() == 20 && iso.ends_with('Z') && iso.starts_with("20"), "{iso}");
    }
}

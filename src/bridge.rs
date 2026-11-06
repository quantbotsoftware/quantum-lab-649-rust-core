//! Puntea read-only dintre nucleul Rust v2 și stratul Tauri v2 (protocoalele S19–S22 + KJ-V5 + HS-V4).
//!
//! Funcțiile de aici sunt apelate de comenzile `#[tauri::command]` din `src/main.rs`; sunt testate fără fereastră.
//! Nicio funcție din `bridge` nu scrie pe disc, nu emite și nu decontează: citește arhiva LIVE curentă (indicată de
//! `tu3/date/live/CURRENT_LIVE.json`), emisiile sigilate (pentru filtrul de afișare S17) și ultimul raport de paritate.
//! Densitățile descriu legea MODELULUI (H1/H3); sub urna reală uniformă toate biletele au aceeași șansă.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::integrity::{self, parse_target, CertifiedEmission, GAMES};
use crate::models::fisher::{density_report, inclusion, info_d, softmax};
use crate::models::h1_mass::H1Mass;
use crate::models::h3_aero::{H3Aero, K_AD};
use crate::models::parity::load_json;
use crate::models::spectral::admissible::{admissible, enumerate_all, top_ticket, AdmState, Admissible};
use crate::models::spectral::frft::{unitarity_residual, Frft};
use crate::models::spectral::{label_dft, neumaier_sum, parseval_relative_residual, CountState};
use crate::parser::{self, Game};
use crate::POSITIVE_CLAIMS_ALLOWED;

/// Rădăcina pistei QL649 (suprascrisă de variabila de mediu `QL649_ROOT`).
pub const DEFAULT_ROOT: &str = "./data/quantum_lab_649";
/// Rădăcina depozitului `LOTO_RO`.
pub const DEFAULT_REPO_ROOT: &str = "./data";
/// Ținta verificării de paritate S19 (`parity`, legată de fișierul S17 `BILETE_ADMISIBILE_20261028T001924Z.json`).
pub const S19_PARITY_TARGET: &str = "2026-10-29";
pub const NOTA: &str = "audit spectral al legii modelului; nu este predicție, emisie sau sfat de joc; \
sub urna uniformă reală toate biletele au aceeași șansă";

pub fn ql_root() -> PathBuf {
    std::env::var_os("QL649_ROOT").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_ROOT))
}

pub fn repo_root() -> PathBuf {
    std::env::var_os("LOTO_RO_ROOT").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_REPO_ROOT))
}

/// Versiunea LIVE curentă: (directorul versiunii, CURRENT_LIVE.json, MANIFEST_LIVE.json).
pub fn current_live(root: &Path) -> Result<(PathBuf, Value, Value), String> {
    let live = root.join("tu3").join("date").join("live");
    let cur = load_json(&live.join("CURRENT_LIVE.json"))?;
    let vdir = live.join(cur["version_dir"].as_str().ok_or("CURRENT_LIVE fără version_dir")?);
    let man = load_json(&vdir.join("MANIFEST_LIVE.json"))?;
    if man["database_sha256"] != cur["database_sha256"] {
        return Err("database_sha256 diferit între CURRENT_LIVE și manifest".into());
    }
    Ok((vdir, cur, man))
}

fn archive_file(vdir: &Path, man: &Value, game: Game) -> Result<PathBuf, String> {
    let f = man["games"][game.tag()]["file"].as_str().ok_or_else(|| format!("manifest fără fișier pentru {game}"))?;
    Ok(vdir.join(f))
}

fn parse_game(tag: &str) -> Result<Game, String> {
    Game::from_tag(tag).ok_or_else(|| format!("joc necunoscut „{tag}” (649, 540, joker)"))
}

/// Inventarul celor 3 arhive LIVE (parsare nativă tokio + memmap2), comparat cu CURRENT_LIVE.json.
pub async fn inventory_live(root: &Path) -> Result<Value, String> {
    let (vdir, cur, man) = current_live(root)?;
    let games = [Game::Loto649, Game::Loto540, Game::Joker];
    let paths: Vec<PathBuf> = games.iter().map(|&g| archive_file(&vdir, &man, g)).collect::<Result<_, _>>()?;
    let res = parser::load_many(paths.clone(), None).await;
    let mut out = Vec::new();
    for (g, r) in games.iter().zip(res) {
        let a = r.map_err(|e| e.to_string())?;
        let s = a.summary();
        let want = &cur["games"][g.tag()];
        out.push(serde_json::json!({
            "joc": g.tag(),
            "rezumat": s,
            "potrivire_current_live": want["records"].as_u64() == Some(a.records.len() as u64)
                && want["last_key"].as_str() == a.last_key.as_deref() && a.rejects.is_empty(),
        }));
    }
    Ok(serde_json::json!({
        "version_dir": cur["version_dir"], "updated_utc": cur["updated_utc"], "arhive": out,
        "kind": "INVENTAR_DATE_NU_EXPERIMENT", "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

/// Starea numărărilor (urna principală și, la Joker, bila) din arhiva LIVE curentă.
pub async fn live_state(root: &Path, game: Game) -> Result<(CountState, Option<CountState>, Value), String> {
    let (vdir, cur, man) = current_live(root)?;
    let a = parser::load_archive(archive_file(&vdir, &man, game)?, Some(game)).await.map_err(|e| e.to_string())?;
    if !a.rejects.is_empty() {
        return Err(format!("{} respingeri în arhiva {game}", a.rejects.len()));
    }
    let main = CountState::from_main(game, &a.records).map_err(|e| e.to_string())?;
    let bonus = CountState::from_bonus(game, &a.records).map_err(|e| e.to_string())?;
    let meta =
        serde_json::json!({"version_dir": cur["version_dir"], "last_key": a.last_key, "records": a.records.len()});
    Ok((main, bonus, meta))
}

fn model_logw(state: &CountState, model: &str) -> Result<(Vec<f64>, f64, f64, usize), String> {
    match model {
        "H1" => {
            let h = H1Mass::from_counts(state).map_err(|e| e.to_string())?;
            Ok((h.log_w, h.lambda, h.x2, state.spec().n - 1))
        }
        "H3" => {
            let h = H3Aero::from_counts(state, K_AD).map_err(|e| e.to_string())?;
            Ok((h.log_w, h.lambda_s, h.x2_s, K_AD))
        }
        _ => Err(format!("model necunoscut „{model}” (H1, H3)")),
    }
}

fn z_of(state: &CountState) -> Vec<f64> {
    let spec = state.spec();
    let (n, p, big_n) = (spec.n as f64, spec.p(), state.draws() as f64);
    let scale = ((n - 1.0) / n).sqrt();
    let sd = (big_n * p * (1.0 - p)).sqrt();
    state.counts().iter().map(|&c| scale * (c as f64 - big_n * p) / sd).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct BallView {
    pub n: usize,
    pub draws: u64,
    pub counts: Vec<u64>,
    pub lambda: f64,
    pub x2: f64,
    pub logw: Vec<f64>,
    pub q: Vec<f64>,
    pub d_nats: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DensityView {
    pub joc: String,
    pub model: String,
    pub n: usize,
    pub k: usize,
    pub draws: u64,
    pub meta: Value,
    pub counts: Vec<u64>,
    pub lambda: f64,
    pub x2: f64,
    pub df: usize,
    pub bet_active: bool,
    pub logw: Vec<f64>,
    pub pi: Vec<f64>,
    pub pi_uniform: f64,
    pub d_nats: f64,
    pub rho: Vec<f64>,
    pub rho_trace: f64,
    pub rho_purity: f64,
    pub rho_purity_uniform: f64,
    pub rho_min_eigenvalue: f64,
    pub rho_max_eigenvalue: f64,
    pub z: Vec<f64>,
    pub dft_abs: Vec<f64>,
    pub parseval_residual: f64,
    pub ball: Option<BallView>,
    pub nota: &'static str,
    pub positive_claims_allowed: bool,
}

pub async fn densities(root: &Path, game: &str, model: &str) -> Result<DensityView, String> {
    let g = parse_game(game)?;
    let (state, bonus, meta) = live_state(root, g).await?;
    let spec = state.spec();
    let (n, k) = (spec.n, spec.k);
    let (logw, lambda, x2, df) = model_logw(&state, model)?;
    let rep = density_report(&logw, k).map_err(|e| e.to_string())?;
    let z = z_of(&state);
    let spectrum = label_dft(&z);
    let ball = match bonus {
        Some(b) => {
            let (blw, bl, bx2, _) = model_logw(&b, model)?;
            Some(BallView {
                n: b.spec().n,
                draws: b.draws(),
                counts: b.counts(),
                lambda: bl,
                x2: bx2,
                q: softmax(&blw),
                d_nats: info_d(&blw, 1),
                logw: blw,
            })
        }
        None => None,
    };
    let (nf, kf) = (n as f64, k as f64);
    Ok(DensityView {
        joc: g.tag().to_string(),
        model: model.to_string(),
        n,
        k,
        draws: state.draws(),
        meta,
        counts: state.counts(),
        lambda,
        x2,
        df,
        bet_active: lambda > 0.0,
        pi: inclusion(&logw, k),
        pi_uniform: kf / nf,
        d_nats: info_d(&logw, k),
        rho_trace: rep.trace,
        rho_purity: rep.purity,
        rho_purity_uniform: 1.0 / nf + (kf - 1.0) * (kf - 1.0) / (nf * (nf - 1.0)),
        rho_min_eigenvalue: rep.min_eigenvalue,
        rho_max_eigenvalue: rep.max_eigenvalue,
        rho: rep.rho,
        parseval_residual: parseval_relative_residual(&z, &spectrum),
        dft_abs: spectrum.iter().map(|c| c.norm()).collect(),
        z,
        logw,
        ball,
        nota: NOTA,
        positive_claims_allowed: POSITIVE_CLAIMS_ALLOWED,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct FrftView {
    pub joc: String,
    pub n: usize,
    pub a: f64,
    pub z: Vec<f64>,
    pub abs: Vec<f64>,
    pub arg: Vec<f64>,
    pub energy_z: f64,
    pub energy_fa: f64,
    pub energy_rel_residual: f64,
    pub unitarity_residual: f64,
    pub nota: &'static str,
}

pub async fn frft_view(root: &Path, game: &str, a: f64) -> Result<FrftView, String> {
    if !a.is_finite() || a.abs() > 4.0 {
        return Err(format!("ordinul a = {a} în afara [−4, 4]"));
    }
    let g = parse_game(game)?;
    let (state, _, _) = live_state(root, g).await?;
    let z = z_of(&state);
    let n = z.len();
    let f = Frft::new(n).map_err(|e| e.to_string())?;
    let y = f.apply_real(a, &z).map_err(|e| e.to_string())?;
    let ez = neumaier_sum(z.iter().map(|v| v * v));
    let ey = neumaier_sum(y.iter().map(|c| c.norm_sqr()));
    Ok(FrftView {
        joc: g.tag().to_string(),
        n,
        a,
        abs: y.iter().map(|c| c.norm()).collect(),
        arg: y.iter().map(|c| c.arg()).collect(),
        energy_rel_residual: (ey - ez).abs() / ez.max(f64::MIN_POSITIVE),
        unitarity_residual: unitarity_residual(&f.matrix(a), n),
        energy_z: ez,
        energy_fa: ey,
        z,
        nota:
            "FrFT Candan–Kutay–Ozaktas: transformare unitară; energia (deci X²) este invariantă; nu selectează bilete",
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamAdmissibility {
    pub joc: String,
    pub model: String,
    pub tinta: String,
    pub stare_emisie: &'static str,
    pub sha256: Option<String>,
    pub emis_utc: Value,
    pub lambda: Value,
    pub b: Option<usize>,
    pub filtru: Option<Admissible>,
    pub bila_stare: Option<&'static str>,
    pub bila_afisare: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct StreamWithLogw {
    pub view: StreamAdmissibility,
    pub logw: Option<Vec<f64>>,
}

fn emission_streams(c: &CertifiedEmission, target: &str) -> Result<Vec<StreamWithLogw>, String> {
    let em = &c.doc;
    let b = em["ticket_size"].as_u64().ok_or("emisie fără ticket_size")? as usize;
    let mut out = Vec::new();
    for m in ["H1", "H3"] {
        let blk = &em["models"][m];
        let lw: Vec<f64> = blk["logw"].as_array().ok_or("emisie fără logw")?.iter().filter_map(Value::as_f64).collect();
        let (bila_stare, bila_afisare) = match blk.get("ball").filter(|_| em["ball_n"].as_u64().is_some()) {
            Some(jb) => {
                let blw: Vec<f64> =
                    jb["logw"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
                if blw.iter().any(|v| v.abs() != 0.0) {
                    (Some("ACTIVA"), Some(top_ticket(&blw, 1)[0]))
                } else {
                    (Some("NU_PARIAZA"), None)
                }
            }
            None => (None, None),
        };
        out.push(StreamWithLogw {
            view: StreamAdmissibility {
                joc: c.game.clone(),
                model: m.to_string(),
                tinta: target.to_string(),
                stare_emisie: "CERTIFICATA",
                sha256: Some(c.sha256.clone()),
                emis_utc: em["emitted_utc"].clone(),
                lambda: blk["lambda"].clone(),
                b: Some(b),
                filtru: Some(admissible(&lw, b)),
                bila_stare,
                bila_afisare,
            },
            logw: Some(lw),
        });
    }
    Ok(out)
}

pub fn admissibility_streams(root: &Path, target: &str) -> Result<Vec<StreamWithLogw>, String> {
    let t = parse_target(target)?;
    let iso = t.iso();
    let lant = integrity::load_lant(root)?;
    let mut out = Vec::new();
    for g in GAMES {
        match integrity::certify_emission(root, &lant, g, &t)? {
            Some(c) => out.extend(emission_streams(&c, &iso)?),
            None => {
                for m in ["H1", "H3"] {
                    out.push(StreamWithLogw {
                        view: StreamAdmissibility {
                            joc: g.to_string(),
                            model: m.to_string(),
                            tinta: iso.clone(),
                            stare_emisie: "EMISIE_ABSENTA",
                            sha256: None,
                            emis_utc: Value::Null,
                            lambda: Value::Null,
                            b: None,
                            filtru: None,
                            bila_stare: None,
                            bila_afisare: None,
                        },
                        logw: None,
                    });
                }
            }
        }
    }
    Ok(out)
}

pub fn targets_view(root: &Path) -> Result<Value, String> {
    let tl = integrity::target_list(root)?;
    let mut v = serde_json::to_value(&tl).map_err(|e| e.to_string())?;
    v["positive_claims_allowed"] = Value::Bool(POSITIVE_CLAIMS_ALLOWED);
    Ok(v)
}

pub fn default_target(root: &Path) -> Result<String, String> {
    Ok(integrity::target_list(root)?.implicita)
}

pub fn admissibility_view(root: &Path, target: &str) -> Result<Value, String> {
    let streams = admissibility_streams(root, target)?;
    let certified = streams.iter().filter(|s| s.view.stare_emisie == "CERTIFICATA").count();
    Ok(serde_json::json!({
        "tinta": parse_target(target)?.iso(),
        "fluxuri_certificate": certified,
        "fluxuri": streams.into_iter().map(|s| s.view).collect::<Vec<_>>(),
        "nota": "strat de AFIȘARE pe emisiile sigilate (S17): nu schimbă nicio probabilitate, nu intră în judecător, \
    nu este emisie nouă; sub urna uniformă toate biletele au aceeași șansă; nu este sfat de joc",
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

pub fn enumeration_check(s: &StreamWithLogw) -> Option<Value> {
    let (f, lw, b) = (s.view.filtru.as_ref()?, s.logw.as_ref()?, s.view.b?);
    if f.stare == AdmState::NuPariaza {
        return None;
    }
    let e = enumerate_all(lw, b);
    let ok = e.as_ref().is_some_and(|e| {
        e.top == f.top_ticket && Some(&e.admissible) == f.bilet_afisare.as_ref() && Some(e.rank) == f.rang
    });
    Some(serde_json::json!({
        "joc": s.view.joc, "model": s.view.model, "ok": ok, "enumerare": e,
        "heap": {"top": f.top_ticket, "admisibil": f.bilet_afisare, "rang": f.rang},
    }))
}

/// Rezumatul celui mai recent raport `tu4/rezultate/S19/PARITATE_RUST_*.json` + verificarea independentă Python `KJ-V5`.
pub fn parity_latest(root: &Path) -> Result<Value, String> {
    let dir = root.join("tu4").join("rezultate").join("S19");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("PARITATE_RUST_") && f.ends_with(".json"))
        })
        .collect();
    files.sort();
    let Some(p) = files.last() else {
        return Ok(serde_json::json!({"disponibil": false}));
    };
    let r = load_json(p)?;
    let ver_kjv5_path = root.join("kj_v5/rezultate/VERIFICARE_KJV5.json");
    let ver_kjv5 = if ver_kjv5_path.exists() { load_json(&ver_kjv5_path).unwrap_or(Value::Null) } else { Value::Null };
    let v3_rez = root.join("v3_upgrade/rezultate");
    let opt_json = |name: &str| -> Value {
        let fp = v3_rez.join(name);
        if fp.exists() { load_json(&fp).unwrap_or(Value::Null) } else { Value::Null }
    };
    let v3_verificari = serde_json::json!({
        "gate_a0_parity": opt_json("GATE_A0_PARITY.json"),
        "s1_cdv3": opt_json("VERIFICARE_S1_CDV3.json"),
        "s2_pmv3": opt_json("VERIFICARE_S2_PMV3.json"),
        "s3_portofoliu": opt_json("VERIFICARE_S3_PORTOFOLIU.json"),
        "s4_ckv6": opt_json("VERIFICARE_S4_CKV6.json"),
        "gate_a6_registre": opt_json("GATE_A6_REGISTRE_SI_DECONTARE.json"),
    });
    Ok(serde_json::json!({
        "disponibil": true,
        "fisier": p.file_name().and_then(|f| f.to_str()),
        "run_utc": r["run_utc"], "verdict_B2_B7": r["verdict_B2_B7"], "groups": r["groups"],
        "n_checks": r["n_checks"], "n_pass": r["n_pass"], "max_abs_delta_float": r["max_abs_delta_float"],
        "values_compared_float": r["values_compared_float"], "values_bitexact_float": r["values_bitexact_float"],
        "tolerances": r["tolerances"],
        "verificare_kjv5": ver_kjv5,
        "v3_verificari": v3_verificari,
    }))
}

/// Informații statice pentru interfața Rust Core v3.
pub fn info() -> Value {
    serde_json::json!({
        "tool": "rust-singularity-core-v3",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "PROTOCOL_RUST_CORE_V3 (S1–S5, Porțile A0–A7) + KJV5-RCV2-P1 + HSV4-UI + TU4 S18–S22",
        "calcul": "clasic HPC: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.9 sm_120, 8 GB GDDR7) + Intel Core i7-14650HX (24 fire AVX2+FMA+POPCNT) + Vault zero-copy mmap pe ./ssd_data",
        "nota": NOTA,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root_available() -> Option<PathBuf> {
        let r = ql_root();
        r.join("tu3/date/live/CURRENT_LIVE.json").exists().then_some(r)
    }

    #[tokio::test]
    async fn bridge_commands_on_live_data() {
        let Some(root) = root_available() else { return };
        let inv = inventory_live(&root).await.unwrap();
        assert_eq!(inv["arhive"].as_array().unwrap().len(), 3);
        assert!(inv["arhive"].as_array().unwrap().iter().all(|a| a["potrivire_current_live"] == true));
        for (g, m) in [("649", "H1"), ("540", "H3"), ("joker", "H3")] {
            let d = densities(&root, g, m).await.unwrap();
            assert_eq!(d.rho.len(), d.n * d.n);
            assert!((d.rho_trace - 1.0).abs() <= 1e-12 && d.rho_min_eigenvalue >= -1e-12);
            assert!((d.pi.iter().sum::<f64>() - d.k as f64).abs() <= 1e-12);
            assert!(d.parseval_residual <= 1e-12);
            assert_eq!(d.ball.is_some(), g == "joker");
            assert!(!d.positive_claims_allowed);
        }
        assert!(densities(&root, "999", "H1").await.is_err());
        assert!(densities(&root, "649", "H9").await.is_err());
        let f = frft_view(&root, "649", 0.5).await.unwrap();
        assert!(f.energy_rel_residual <= 1e-12 && f.unitarity_residual <= 1e-12);
        assert!(frft_view(&root, "649", f64::NAN).await.is_err());
        let adm = admissibility_view(&root, S19_PARITY_TARGET).unwrap();
        assert_eq!(adm["fluxuri"].as_array().unwrap().len(), 6);
        assert!(parity_latest(&root).is_ok());
        assert_eq!(info()["positive_claims_allowed"], false);
    }

    #[test]
    fn b1_target_20261029_matches_s17_file() {
        let Some(root) = root_available() else { return };
        let s17 = load_json(&root.join("tu4/rezultate/S17/BILETE_ADMISIBILE_20261028T001924Z.json")).unwrap();
        let streams = admissibility_streams(&root, "2026-10-29").unwrap();
        assert_eq!(streams.len(), 6);
        for s in &streams {
            let v = &s.view;
            assert_eq!(v.stare_emisie, "CERTIFICATA");
            let r = &s17["streams"][format!("{}/{}", v.joc, v.model)];
            let f = v.filtru.as_ref().unwrap();
            assert_eq!(r["stare"].as_str(), Some(f.stare.as_str()), "{}/{}", v.joc, v.model);
            let labels =
                |x: &Value| x.as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|u| u as u8).collect());
            assert_eq!(labels(&r["top_ticket_emis"]), Some(f.top_ticket.clone()));
            assert_eq!(labels(&r["bilet_afisare"]), f.bilet_afisare.clone());
            assert_eq!(r["rang"].as_u64(), f.rang);
            assert_eq!(r["noduri_heap"].as_u64(), f.noduri_heap);
            for (k, mine) in [("ln_q_u_top", f.ln_q_u_top), ("ln_q_u_admisibil", f.ln_q_u_admisibil)] {
                match (r[k].as_f64(), mine) {
                    (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-12, "{k}: {a} vs {b}"),
                    (None, None) => {}
                    other => panic!("{k}: {other:?}"),
                }
            }
        }
    }
}

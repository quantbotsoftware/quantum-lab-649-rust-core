//! `KJ-V5` — Evaluarea cauzală Walk-Forward (`WARMUP = 100`, `s = 100 .. S-1`) pe loteriile din România
//! (`649`, `540`, `joker` + studiul de ablație pe 4 configurații `K1`, `K2`, `K1_K2`, `Full_KJV5` + multipletul
//! post-arhivă exact pe 24 fire CPU) și pe cele `10` fluxuri internaționale mecanice & electronice RNG
//! (`PROTOCOL_KJV5_RUST_V2.md` §4.2).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::kjv5::krawtchouk::binom_f64;
use crate::kjv5::multiplet::{
    cond_poisson_log_z_and_marginals, evaluate_portfolio_exact_full_enum, find_sigma_alpha_fast, hypergeom_probs,
    ticket_to_draw_marginals, top_k_1based, PortfolioFullEval,
};
use crate::kjv5::{
    compute_prior_weights, decompose_channel_masses, AblationConfig, CanonicalStep, KjV5Space, KjV5State,
    KjV5StepOperators, StepExpertPack, ABLATION_CONFIGS, EXPERT_GRID, NUM_EXPERTS, W0_ABSTAIN_PRIOR,
};
use crate::npz::downsample_indices;
use crate::POSITIVE_CLAIMS_ALLOWED;

pub const WARMUP_STEPS: usize = 100;
pub const EW_HALF_LIFE: f64 = 26.0;
pub const EPISTEMIC_NOTICE: &str = "Minime de energie / stări statistice condiționate de istoric (multiplet ortogonal \
Coulomb–Jastrow), NU predicții cu avantaj demonstrat și fără promisiuni de câștig. positive_claims_allowed=false.";

#[derive(Debug, Clone, Copy)]
pub struct StreamSpec {
    pub code: &'static str,
    pub name: &'static str,
    pub group: &'static str,
    pub n: usize,
    pub r: usize,
    pub k: usize,
    pub m_bonus: Option<usize>,
}

pub const RO_SPECS: [StreamSpec; 3] = [
    StreamSpec { code: "649", name: "Loto 6/49 România", group: "ro_live", n: 49, r: 6, k: 6, m_bonus: None },
    StreamSpec { code: "540", name: "Loto 5/40 România", group: "ro_live", n: 40, r: 6, k: 5, m_bonus: None },
    StreamSpec {
        code: "joker",
        name: "Joker România (5/45 + 1/20)",
        group: "ro_live",
        n: 45,
        r: 5,
        k: 5,
        m_bonus: Some(20),
    },
];

pub const INTL_SPECS: [StreamSpec; 10] = [
    StreamSpec {
        code: "fr_loto",
        name: "FR Loto (FDJ 5/49 + 1/10)",
        group: "mechanical_external",
        n: 49,
        r: 5,
        k: 5,
        m_bonus: Some(10),
    },
    StreamSpec {
        code: "hu_hatoslotto",
        name: "HU Hatoslottó (6/45)",
        group: "mechanical_external",
        n: 45,
        r: 6,
        k: 6,
        m_bonus: None,
    },
    StreamSpec {
        code: "hu_otoslotto",
        name: "HU Ötöslottó (5/90)",
        group: "mechanical_external",
        n: 90,
        r: 5,
        k: 5,
        m_bonus: None,
    },
    StreamSpec {
        code: "us_powerball",
        name: "US Powerball (5/69 + 1/26)",
        group: "mechanical_external",
        n: 69,
        r: 5,
        k: 5,
        m_bonus: Some(26),
    },
    StreamSpec {
        code: "hu_skandinav_mech",
        name: "HU Skandináv Lottó — Tragerea 1 Mecanică (Kézi 7/35)",
        group: "mechanical_external",
        n: 35,
        r: 7,
        k: 7,
        m_bonus: None,
    },
    StreamSpec {
        code: "fdj_eurodreams",
        name: "FDJ EuroDreams (6/40 + 1/5 e-Draw RNG)",
        group: "electronic_rng",
        n: 40,
        r: 6,
        k: 6,
        m_bonus: Some(5),
    },
    StreamSpec {
        code: "hu_skandinav_rng",
        name: "HU Skandináv Lottó — Tragerea 2 Electronică RNG (Gépi 7/35)",
        group: "electronic_rng",
        n: 35,
        r: 7,
        k: 7,
        m_bonus: None,
    },
    StreamSpec {
        code: "ny_take5",
        name: "US NY Take 5 (5/39 Daily)",
        group: "electronic_rng",
        n: 39,
        r: 5,
        k: 5,
        m_bonus: None,
    },
    StreamSpec {
        code: "ny_cash4life",
        name: "US NY Cash4Life (5/60 + 1/4 Daily)",
        group: "electronic_rng",
        n: 60,
        r: 5,
        k: 5,
        m_bonus: Some(4),
    },
    StreamSpec {
        code: "ny_pick10",
        name: "US NY Pick 10 (10/80 din 20/80 extrase)",
        group: "electronic_rng",
        n: 80,
        r: 20,
        k: 10,
        m_bonus: None,
    },
];

#[derive(Debug, Deserialize)]
struct RawDrawLine {
    draw_date: String,
    numbers: Vec<usize>,
    #[serde(default)]
    joker: Option<usize>,
    #[serde(default)]
    bonus: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LoadedStream {
    pub spec: StreamSpec,
    pub file_path: PathBuf,
    pub file_sha256: String,
    pub version_dir: Option<String>,
    pub manifest_sha256: Option<String>,
    pub n_draws_total: usize,
    pub double_draw_days: usize,
    pub steps: Vec<CanonicalStep>,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha256_bytes(&b))
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn round8(v: f64) -> f64 {
    (v * 1e8).round() / 1e8
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

fn utc_now_iso() -> String {
    let secs =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Parsează un fișier `.jsonl` în pași canonici ordonați cronologic (și lexicografic intra-zi).
pub fn parse_jsonl_to_steps(
    spec: StreamSpec,
    path: &Path,
    expected_sha: Option<&str>,
    version_dir: Option<String>,
    manifest_sha256: Option<String>,
) -> Result<LoadedStream, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let file_sha256 = sha256_bytes(&bytes);
    if let Some(exp) = expected_sha {
        if file_sha256 != exp {
            return Err(format!("SHA-256 invalid pentru {}: {file_sha256} != {exp}", path.display()));
        }
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{} UTF-8 invalid: {e}", path.display()))?;
    let mut by_date: BTreeMap<String, Vec<(Vec<usize>, Option<usize>)>> = BTreeMap::new();
    let mut n_draws_total = 0usize;

    for (ln_no, raw_ln) in text.lines().enumerate() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        let rec: RawDrawLine =
            serde_json::from_str(ln).map_err(|e| format!("{}:{}: {e}", path.display(), ln_no + 1))?;
        if rec.numbers.len() != spec.r {
            return Err(format!(
                "{}:{}: așteptat {} bile, găsit {}",
                path.display(),
                ln_no + 1,
                spec.r,
                rec.numbers.len()
            ));
        }
        for &b in &rec.numbers {
            if b == 0 || b > spec.n {
                return Err(format!("{}:{}: bilă {b} în afara 1..={}", path.display(), ln_no + 1, spec.n));
            }
        }
        let jb = rec.joker.or(rec.bonus);
        if let Some(mb) = spec.m_bonus {
            let v = jb.ok_or_else(|| format!("{}:{}: bonus lipsă", path.display(), ln_no + 1))?;
            if v == 0 || v > mb {
                return Err(format!("{}:{}: bonus {v} în afara 1..={mb}", path.display(), ln_no + 1));
            }
        }
        by_date.entry(rec.draw_date).or_default().push((rec.numbers, spec.m_bonus.and(jb)));
        n_draws_total += 1;
    }

    let space = KjV5Space::new(spec.n, spec.r, spec.k, spec.m_bonus);
    let mut double_draw_days = 0usize;
    let mut steps = Vec::with_capacity(by_date.len());
    for (dt, drs) in by_date {
        if drs.len() > 1 {
            double_draw_days += 1;
        }
        steps.push(CanonicalStep::from_1based(&space, dt, &drs));
    }

    Ok(LoadedStream {
        spec,
        file_path: path.to_path_buf(),
        file_sha256,
        version_dir,
        manifest_sha256,
        n_draws_total,
        double_draw_days,
        steps,
    })
}

/// Încarcă o arhivă RO LIVE (`649`, `540`, `joker`) verificată SHA-256 față de `CURRENT_LIVE.json` + `MANIFEST_LIVE.json`.
pub fn load_ro_live_stream(ql_root: &Path, game_code: &str) -> Result<LoadedStream, String> {
    let spec = RO_SPECS
        .iter()
        .copied()
        .find(|s| s.code == game_code)
        .ok_or_else(|| format!("joc RO necunoscut: {game_code}"))?;
    let live_dir = ql_root.join("tu3/date/live");
    let cur_bytes = std::fs::read(live_dir.join("CURRENT_LIVE.json")).map_err(|e| format!("CURRENT_LIVE.json: {e}"))?;
    let cur: Value = serde_json::from_slice(&cur_bytes).map_err(|e| e.to_string())?;
    let vdir_rel = cur["version_dir"].as_str().ok_or("CURRENT_LIVE fără version_dir")?.to_string();
    let vdir = live_dir.join(&vdir_rel);
    let man_bytes = std::fs::read(vdir.join("MANIFEST_LIVE.json")).map_err(|e| format!("MANIFEST_LIVE.json: {e}"))?;
    let man_sha = sha256_bytes(&man_bytes);
    if let Some(exp_msha) = cur["manifest_sha256"].as_str() {
        if man_sha != exp_msha {
            return Err(format!("manifest_sha256 diferit față de CURRENT_LIVE: {man_sha} != {exp_msha}"));
        }
    }
    let man: Value = serde_json::from_slice(&man_bytes).map_err(|e| e.to_string())?;
    if man["database_sha256"] != cur["database_sha256"] {
        return Err("database_sha256 diferit între CURRENT_LIVE și MANIFEST_LIVE".into());
    }
    let rel_file = man["games"][game_code]["file"]
        .as_str()
        .ok_or_else(|| format!("MANIFEST_LIVE fără fișier pentru {game_code}"))?;
    let exp_sha = cur["games"][game_code]["sha256"]
        .as_str()
        .ok_or_else(|| format!("CURRENT_LIVE fără sha256 pentru {game_code}"))?;
    parse_jsonl_to_steps(spec, &vdir.join(rel_file), Some(exp_sha), Some(vdir_rel), Some(man_sha))
}

/// Încarcă toate cele `10` fluxuri internaționale verificate SHA-256 față de `PROVENIENTA_INTL.json`.
pub fn load_all_intl_streams(ql_root: &Path) -> Result<(String, Vec<LoadedStream>), String> {
    let prov_path = ql_root.join("hs_v4/date_intl/PROVENIENTA_INTL.json");
    let prov_bytes = std::fs::read(&prov_path).map_err(|e| format!("{}: {e}", prov_path.display()))?;
    let prov_sha = sha256_bytes(&prov_bytes);
    let prov: Value = serde_json::from_slice(&prov_bytes).map_err(|e| e.to_string())?;
    let datasets = prov["datasets"].as_object().ok_or("PROVENIENTA_INTL fără datasets")?;

    let mut out = Vec::with_capacity(INTL_SPECS.len());
    for spec in INTL_SPECS {
        let ds = datasets.get(spec.code).ok_or_else(|| format!("dataset lipsă în PROVENIENTA_INTL: {}", spec.code))?;
        let rel_f = ds["jsonl_file"].as_str().ok_or("jsonl_file lipsă")?;
        let exp_sha = ds["jsonl_sha256"].as_str().ok_or("jsonl_sha256 lipsă")?;
        let s = parse_jsonl_to_steps(spec, &ql_root.join(rel_f), Some(exp_sha), None, None)?;
        out.push(s);
    }
    Ok((prov_sha, out))
}

/// Selectează exact biletul `σ_α` la pasul `s`, eliminând constanta uniformă `w_0 * P_U` din arborele BnB
/// când `1 - w_0` devine foarte mic astfel încât tăierea pe intervale rămâne exactă și instantanee.
fn select_sigma_alpha_step(
    space: &KjV5Space,
    ops: &KjV5StepOperators,
    pack: &StepExpertPack,
    w_port: &[f64; NUM_EXPERTS],
    pi_port: &[f64],
) -> (Vec<usize>, Option<usize>, f64) {
    let unif_joint = 1.0 / (space.total_joint_states as f64);
    let s_act: f64 = w_port[1..].iter().sum();
    if s_act <= 1e-15 {
        let tkt: Vec<usize> = (1..=space.k).collect();
        return (tkt, space.m_bonus.map(|_| 1), unif_joint.ln());
    }
    // Dacă pasul anterior are extrageri multiple pe un spațiu mare (> 12M), folosim bazinul primelor 15 bile
    // după marginalele exacte π_port (de ex. cele 5 zile duble din 3630 pe HU Ötöslottó 5/90).
    if ops.prev_draws.len() != 1 && space.total_joint_states > 12_000_000 {
        let mut w_act = [0.0f64; NUM_EXPERTS];
        for m in 1..NUM_EXPERTS {
            w_act[m] = w_port[m] / s_act;
        }
        let pool_m = (space.k + 9).min(space.n).min(16);
        let mut ord: Vec<usize> = (0..space.n).collect();
        ord.sort_by(|&a, &b| pi_port[b].total_cmp(&pi_port[a]).then_with(|| a.cmp(&b)));
        let mut pool = ord[..pool_m].to_vec();
        pool.sort_unstable();

        // Construim un spațiu temporar sau evaluăm direct prin find_sigma_alpha_fast pe primul prev_draw?
        // Nu: evaluăm exact probabilitatea portofoliului pe toate combinațiile C(pool_m, k) (<= 4368 combinații)!
        let mut best_tkt: Vec<usize> = pool[..space.k].iter().map(|&x| x + 1).collect();
        let mut best_jb: Option<usize> = space.m_bonus.map(|_| 1);
        let mut best_p = -1.0f64;
        let m_prev_f = ops.prev_draws.len() as f64;

        let mut idx_c: Vec<usize> = (0..space.k).collect();
        loop {
            let sig0: Vec<usize> = idx_c.iter().map(|&ix| pool[ix]).collect();
            if let Some(mb) = space.m_bonus {
                let mut b_tot = vec![w_port[0] * unif_joint; mb];
                for m in 1..NUM_EXPERTS {
                    if w_port[m] <= 1e-18 {
                        continue;
                    }
                    let pot = ops.expert_potential(space, m);
                    for (d_idx, (d_prev, _)) in ops.prev_draws.iter().enumerate() {
                        let mut dm = 0u128;
                        for &x in d_prev {
                            dm |= 1u128 << x;
                        }
                        let w_main = crate::kjv5::partition::exact_ticket_weight_main(&pot, dm, &sig0);
                        let p_main = w_port[m] * (w_main / pack.z_packs[m][d_idx].z_main) / m_prev_f;
                        for (b_i, slot) in b_tot.iter_mut().enumerate() {
                            *slot += p_main * pack.z_packs[m][d_idx].pi_bonus[b_i];
                        }
                    }
                }
                for (b_i, &p) in b_tot.iter().enumerate() {
                    if p > best_p + 1e-13 * best_p.abs().max(1e-300) {
                        best_p = p;
                        best_tkt = sig0.iter().map(|&x| x + 1).collect();
                        best_jb = Some(b_i + 1);
                    }
                }
            } else {
                let mut p_tot = w_port[0] * unif_joint;
                for m in 1..NUM_EXPERTS {
                    if w_port[m] <= 1e-18 {
                        continue;
                    }
                    let pot = ops.expert_potential(space, m);
                    let mut p_ex = 0.0f64;
                    for (d_idx, (d_prev, _)) in ops.prev_draws.iter().enumerate() {
                        let mut dm = 0u128;
                        for &x in d_prev {
                            dm |= 1u128 << x;
                        }
                        let w_main = crate::kjv5::partition::exact_ticket_weight_main(&pot, dm, &sig0);
                        p_ex += (w_main / pack.z_packs[m][d_idx].z_main) / m_prev_f;
                    }
                    p_tot += w_port[m] * p_ex;
                }
                if p_tot > best_p + 1e-13 * best_p.abs().max(1e-300) {
                    best_p = p_tot;
                    best_tkt = sig0.iter().map(|&x| x + 1).collect();
                }
            }

            let mut pos = space.k;
            while pos > 0 && idx_c[pos - 1] == pool_m - space.k + (pos - 1) {
                pos -= 1;
            }
            if pos == 0 {
                break;
            }
            idx_c[pos - 1] += 1;
            for j in pos..space.k {
                idx_c[j] = idx_c[j - 1] + 1;
            }
        }
        return (best_tkt, best_jb, best_p.max(1e-300).ln());
    }

    // Când s_act < 1e-6, normăm partea activă (w_0 = 0) pentru ca BnB pe intervale să taie instantaneu arborele,
    // apoi reconstruim exact probabilitatea mixturii complete w_0 * P_U + s_act * P_act(σ_α)!
    if s_act < 1e-6 {
        let mut w_act = [0.0f64; NUM_EXPERTS];
        let mut pi_act = vec![0.0f64; space.n];
        for m in 1..NUM_EXPERTS {
            let wm = w_port[m] / s_act;
            w_act[m] = wm;
            for i in 0..space.n {
                pi_act[i] += wm * pack.pi_m_main[m][i];
            }
        }
        let (tkt, jb, lp_act, _) = find_sigma_alpha_fast(space, ops, &pack.z_packs, &w_act, &pi_act);
        let p_full = w_port[0] * unif_joint + s_act * lp_act.exp();
        // Dacă diferența relativă față de [1..=k] în P_port este sub pragul de departajare 1e-13, păstrăm [1..=k]
        if (p_full - unif_joint).abs() <= 1e-13 * unif_joint {
            let base_tkt: Vec<usize> = (1..=space.k).collect();
            return (base_tkt, space.m_bonus.map(|_| 1), p_full.max(1e-300).ln());
        }
        return (tkt, jb, p_full.max(1e-300).ln());
    }

    let (tkt, jb, lp, _) = find_sigma_alpha_fast(space, ops, &pack.z_packs, w_port, pi_port);
    (tkt, jb, lp)
}

/// Construiește multipletul post-arhivă `(α, β, γ)`: enumerare paralelă exactă pe 24 fire CPU când spațiul `<= 50M`
/// (inclusiv toate cele 3 jocuri RO `649`, `540`, `joker`!), sau bazinul primelor bile după `π_port` pe spații uriașe (`Pick 10` / `Powerball`).
fn compute_terminal_multiplet(
    space: &KjV5Space,
    ops: &KjV5StepOperators,
    pack: &StepExpertPack,
    w_port: &[f64; NUM_EXPERTS],
    pi_port: &[f64],
    force_full_enum: bool,
) -> PortfolioFullEval {
    if force_full_enum || space.total_joint_states <= 25_000_000 {
        return evaluate_portfolio_exact_full_enum(space, ops, &pack.z_packs, w_port);
    }
    // Pe spații > 25M din loteriile internaționale (`Powerball` 292M, `Pick 10` 1.65e12, `Ötöslottó` 43.9M),
    // construim multipletul Coulomb–Jastrow din primele `pool_m = min(max(3*k, 16), 18)` bile ordonate după `π_port`.
    let pool_m = (3 * space.k).clamp(16, 18).min(space.n);
    let mut ord: Vec<usize> = (0..space.n).collect();
    ord.sort_by(|&a, &b| pi_port[b].total_cmp(&pi_port[a]).then_with(|| a.cmp(&b)));
    let mut pool = ord[..pool_m].to_vec();
    pool.sort_unstable();

    let (alpha_tkt, alpha_jb, alpha_lp) = select_sigma_alpha_step(space, ops, pack, w_port, pi_port);
    // Construim variantele ortogonale β și γ din bazinul disjunct/minim suprapus ordonat după π_port
    let mut used: BTreeSet<usize> = alpha_tkt.iter().copied().collect();
    let pick_next = |used_set: &mut BTreeSet<usize>, alt_jb_offset: usize| -> (Vec<usize>, Option<usize>) {
        let mut chosen_1b: Vec<usize> =
            ord.iter().map(|&i| i + 1).filter(|x| !used_set.contains(x)).take(space.k).collect();
        if chosen_1b.len() < space.k {
            for &i in &ord {
                let x = i + 1;
                if !chosen_1b.contains(&x) {
                    chosen_1b.push(x);
                    if chosen_1b.len() == space.k {
                        break;
                    }
                }
            }
        }
        chosen_1b.sort_unstable();
        for &x in &chosen_1b {
            used_set.insert(x);
        }
        let jb = space.m_bonus.map(|mb| ((alpha_jb.unwrap_or(1) - 1 + alt_jb_offset) % mb) + 1);
        (chosen_1b, jb)
    };
    let (beta_tkt, beta_jb) = pick_next(&mut used, 1);
    let (gamma_tkt, gamma_jb) = pick_next(&mut used, 2);

    let log_u = -(space.total_joint_states as f64).ln();
    let eval_lp = |tkt_1b: &[usize], jb_1b: Option<usize>| -> f64 {
        let sig0: Vec<usize> = tkt_1b.iter().map(|&x| x - 1).collect();
        let unif_j = 1.0 / (space.total_joint_states as f64);
        let m_prev_f = ops.prev_draws.len().max(1) as f64;
        let mut p_tot = w_port[0] * unif_j;
        for m in 1..NUM_EXPERTS {
            if w_port[m] <= 1e-18 {
                continue;
            }
            let pot = ops.expert_potential(space, m);
            let mut p_ex = 0.0f64;
            for (d_idx, (d_prev, _)) in ops.prev_draws.iter().enumerate() {
                let mut dm = 0u128;
                for &x in d_prev {
                    dm |= 1u128 << x;
                }
                let w_main = crate::kjv5::partition::exact_ticket_weight_main(&pot, dm, &sig0);
                let b_mult = match (jb_1b, space.m_bonus) {
                    (Some(jb), Some(_)) => pack.z_packs[m][d_idx].pi_bonus[jb - 1],
                    _ => 1.0,
                };
                p_ex += (w_main / pack.z_packs[m][d_idx].z_main) * b_mult / m_prev_f;
            }
            p_tot += w_port[m] * p_ex;
        }
        p_tot.max(1e-300).ln()
    };

    let beta_lp = eval_lp(&beta_tkt, beta_jb);
    let gamma_lp = eval_lp(&gamma_tkt, gamma_jb);

    let denom_ov = (space.k + usize::from(space.m_bonus.is_some())) as f64;
    let ov = |a: &[usize], aj: Option<usize>, b: &[usize], bj: Option<usize>| -> f64 {
        let mut c = a.iter().filter(|x| b.contains(x)).count();
        if space.m_bonus.is_some() && aj == bj {
            c += 1;
        }
        (c as f64) / denom_ov
    };
    let gab = ov(&alpha_tkt, alpha_jb, &beta_tkt, beta_jb);
    let gac = ov(&alpha_tkt, alpha_jb, &gamma_tkt, gamma_jb);
    let gbc = ov(&beta_tkt, beta_jb, &gamma_tkt, gamma_jb);
    let det_g3 = (1.0 + 2.0 * gab * gbc * gac - (gab * gab + gbc * gbc + gac * gac)).max(0.0);
    let core_1b: Vec<usize> = alpha_tkt.iter().copied().take(3.min(space.k)).collect();

    PortfolioFullEval {
        sigma_alpha_1based: alpha_tkt.clone(),
        bonus_alpha_1based: alpha_jb,
        alpha_log_prob: alpha_lp,
        runner_up_gap: (alpha_lp - beta_lp).max(0.0),
        variants: vec![
            crate::kjv5::multiplet::MultipletVariant {
                label: "alpha",
                ticket_1based: alpha_tkt,
                bonus_1based: alpha_jb,
                log_prob: alpha_lp,
                log_lift_vs_uniform: alpha_lp - log_u,
            },
            crate::kjv5::multiplet::MultipletVariant {
                label: "beta",
                ticket_1based: beta_tkt,
                bonus_1based: beta_jb,
                log_prob: beta_lp,
                log_lift_vs_uniform: beta_lp - log_u,
            },
            crate::kjv5::multiplet::MultipletVariant {
                label: "gamma",
                ticket_1based: gamma_tkt,
                bonus_1based: gamma_jb,
                log_prob: gamma_lp,
                log_lift_vs_uniform: gamma_lp - log_u,
            },
        ],
        gram_matrix_3x3: [[1.0, gab, gac], [gab, 1.0, gbc], [gac, gbc, 1.0]],
        det_g3,
        deterministic_core_1based: core_1b,
        exact_global_certified: pool.len() == space.n,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryPoint {
    pub eval_offset: usize,
    pub step: usize,
    pub date: String,
    pub e_s: f64,
    pub log10_e_s: f64,
    pub w_0: f64,
    pub w_k1: f64,
    pub w_k2: f64,
    pub w_k3: f64,
}

fn summarize_window(
    steps_idx: &[usize],
    draw_indices: &[usize],
    expected_hits_u: f64,
    hits_kj: &[usize],
    hits_b: &[usize],
    hits_r: &[usize],
    log_step_kj: &[f64],
    log_step_u: &[f64],
    log_step_b: &[f64],
    log_step_r: &[f64],
    brier_kj: &[f64],
    brier_u: &[f64],
    brier_b: &[f64],
    brier_r: &[f64],
) -> Value {
    let n_steps = steps_idx.len();
    let n_draws = draw_indices.len();
    let h_kj: usize = draw_indices.iter().map(|&i| hits_kj[i]).sum();
    let h_b: usize = draw_indices.iter().map(|&i| hits_b[i]).sum();
    let h_r: usize = draw_indices.iter().map(|&i| hits_r[i]).sum();
    let mean_kj = if n_draws > 0 { (h_kj as f64) / (n_draws as f64) } else { 0.0 };
    let ls_kj: f64 = steps_idx.iter().map(|&i| log_step_kj[i]).sum();
    let ls_u: f64 = steps_idx.iter().map(|&i| log_step_u[i]).sum();
    let ls_b: f64 = steps_idx.iter().map(|&i| log_step_b[i]).sum();
    let ls_r: f64 = steps_idx.iter().map(|&i| log_step_r[i]).sum();
    let br_kj: f64 = draw_indices.iter().map(|&i| brier_kj[i]).sum();
    let br_u: f64 = draw_indices.iter().map(|&i| brier_u[i]).sum();
    let br_b: f64 = draw_indices.iter().map(|&i| brier_b[i]).sum();
    let br_r: f64 = draw_indices.iter().map(|&i| brier_r[i]).sum();

    json!({
        "n_steps": n_steps,
        "n_draws": n_draws,
        "hits_KJV5": h_kj,
        "hits_B": h_b,
        "hits_R": h_r,
        "mean_hits_KJV5": round6(mean_kj),
        "expected_mean_hits_U": round6(expected_hits_u),
        "excess_hits_vs_U": round6(mean_kj - expected_hits_u),
        "log_score_KJV5": round6(ls_kj),
        "log_score_U": round6(ls_u),
        "log_score_B": round6(ls_b),
        "log_score_R": round6(ls_r),
        "brier_KJV5": round6(br_kj),
        "brier_U": round6(br_u),
        "brier_B": round6(br_b),
        "brier_R": round6(br_r),
    })
}

fn build_categories(
    spec: StreamSpec,
    main_hits_kj: &[usize],
    main_hits_b: &[usize],
    main_hits_r: &[usize],
    bonus_hits_kj: &[usize],
    bonus_hits_b: &[usize],
    bonus_hits_r: &[usize],
) -> Value {
    let n_draws = main_hits_kj.len();
    let n_f = n_draws as f64;
    let p_hg = hypergeom_probs(spec.n, spec.r, spec.k);
    let m_max = spec.r.min(spec.k);

    let mut main_hist = serde_json::Map::new();
    for h in 0..=m_max {
        let lbl = format!("{h}/{}", spec.k);
        let c_kj = main_hits_kj.iter().filter(|&&x| x == h).count();
        let c_b = main_hits_b.iter().filter(|&&x| x == h).count();
        let c_r = main_hits_r.iter().filter(|&&x| x == h).count();
        main_hist.insert(
            lbl,
            json!({
                "KJV5": c_kj,
                "B": c_b,
                "R": c_r,
                "expected_U": round4(n_f * p_hg[h]),
            }),
        );
    }

    let mut winning_tiers = serde_json::Map::new();
    for min_h in [3usize, 4, 5, 6] {
        if min_h > m_max {
            continue;
        }
        let lbl_ge = if min_h == m_max { format!("{min_h}/{}", spec.k) } else { format!("{min_h}+") };
        let c_kj = main_hits_kj.iter().filter(|&&x| x >= min_h).count();
        let c_b = main_hits_b.iter().filter(|&&x| x >= min_h).count();
        let c_r = main_hits_r.iter().filter(|&&x| x >= min_h).count();
        let p_ge: f64 = p_hg[min_h..].iter().sum();
        winning_tiers.insert(
            lbl_ge,
            json!({
                "KJV5": c_kj,
                "B": c_b,
                "R": c_r,
                "expected_U": round4(n_f * p_ge),
            }),
        );
        if min_h < m_max {
            let lbl_ex = format!("exact_{min_h}/{}", spec.k);
            winning_tiers.insert(
                lbl_ex,
                json!({
                    "KJV5": main_hits_kj.iter().filter(|&&x| x == min_h).count(),
                    "B": main_hits_b.iter().filter(|&&x| x == min_h).count(),
                    "R": main_hits_r.iter().filter(|&&x| x == min_h).count(),
                    "expected_U": round4(n_f * p_hg[min_h]),
                }),
            );
        }
    }

    let mut out = json!({
        "n_target_draws": n_draws,
        "main_histogram": main_hist,
        "winning_tiers": winning_tiers,
    });

    if let Some(mb) = spec.m_bonus {
        let pj1 = 1.0 / (mb as f64);
        let pj0 = 1.0 - pj1;
        out["bonus_ball_hits"] = json!({
            "KJV5": bonus_hits_kj.iter().filter(|&&x| x == 1).count(),
            "B": bonus_hits_b.iter().filter(|&&x| x == 1).count(),
            "R": bonus_hits_r.iter().filter(|&&x| x == 1).count(),
            "expected_U": round4(n_f * pj1),
        });
        let mut joint = serde_json::Map::new();
        for h in (0..=m_max).rev() {
            for jh in [1usize, 0] {
                if h == 0 && jh == 0 {
                    continue;
                }
                let lbl = if jh == 1 { format!("{h}+Bonus") } else { format!("{h}+0") };
                let p_j = p_hg[h] * if jh == 1 { pj1 } else { pj0 };
                let c_kj = (0..n_draws).filter(|&i| main_hits_kj[i] == h && bonus_hits_kj[i] == jh).count();
                let c_b = (0..n_draws).filter(|&i| main_hits_b[i] == h && bonus_hits_b[i] == jh).count();
                let c_r = (0..n_draws).filter(|&i| main_hits_r[i] == h && bonus_hits_r[i] == jh).count();
                joint.insert(lbl, json!({ "KJV5": c_kj, "B": c_b, "R": c_r, "expected_U": round4(n_f * p_j) }));
            }
        }
        out["joint_bonus_tiers"] = Value::Object(joint);
    }

    out
}

fn evaluate_gates(
    win_tot: &Value,
    win_h1: &Value,
    win_h2: &Value,
    win_last100: &Value,
    e_curve: &[f64],
) -> (Vec<Value>, usize) {
    let final_e = *e_curve.last().unwrap_or(&1.0);
    let min_e = e_curve.iter().copied().fold(f64::INFINITY, f64::min);
    let max_e = e_curve.iter().copied().fold(0.0f64, f64::max);
    let log10_e = final_e.max(1e-300).log10();

    let ex = win_tot["excess_hits_vs_U"].as_f64().unwrap_or(0.0);
    let h_kj = win_tot["hits_KJV5"].as_u64().unwrap_or(0);
    let h_b = win_tot["hits_B"].as_u64().unwrap_or(0);
    let h_r = win_tot["hits_R"].as_u64().unwrap_or(0);

    let ls_kj = win_tot["log_score_KJV5"].as_f64().unwrap_or(0.0);
    let ls_u = win_tot["log_score_U"].as_f64().unwrap_or(0.0);
    let ls_b = win_tot["log_score_B"].as_f64().unwrap_or(0.0);
    let ls_r = win_tot["log_score_R"].as_f64().unwrap_or(0.0);

    let br_kj = win_tot["brier_KJV5"].as_f64().unwrap_or(0.0);
    let br_u = win_tot["brier_U"].as_f64().unwrap_or(0.0);
    let br_b = win_tot["brier_B"].as_f64().unwrap_or(0.0);
    let br_r = win_tot["brier_R"].as_f64().unwrap_or(0.0);

    let h1_diff = win_h1["log_score_KJV5"].as_f64().unwrap_or(0.0) - win_h1["log_score_B"].as_f64().unwrap_or(0.0);
    let h2_diff = win_h2["log_score_KJV5"].as_f64().unwrap_or(0.0) - win_h2["log_score_B"].as_f64().unwrap_or(0.0);
    let l100_diff =
        win_last100["log_score_KJV5"].as_f64().unwrap_or(0.0) - win_last100["log_score_B"].as_f64().unwrap_or(0.0);

    let g1 = ex >= 0.20;
    let g2 = h_kj > h_b.max(h_r);
    let g3 = ls_kj > ls_u.max(ls_b).max(ls_r);
    let g4 = br_kj < br_u.min(br_b).min(br_r);
    let g5 = h1_diff > 0.0 && h2_diff > 0.0 && l100_diff > 0.0;
    let g6 = final_e > 1.0;

    let gates = vec![
        json!({
            "id": "G1",
            "rule": "Exces mediu de potriviri σ_α peste media hipergeometrică uniformă >= +0.20",
            "pass": g1,
            "value": ex,
            "threshold": 0.20,
        }),
        json!({
            "id": "G2",
            "rule": "Hits(KJ-V5) > max(Hits(B), Hits(R))",
            "pass": g2,
            "value": { "KJV5": h_kj, "B": h_b, "R": h_r },
        }),
        json!({
            "id": "G3",
            "rule": "LogScore(KJ-V5) > max(LogScore(U), LogScore(B), LogScore(R))",
            "pass": g3,
            "value": {
                "KJV5": ls_kj,
                "U": ls_u,
                "B": ls_b,
                "R": ls_r,
                "diff_vs_U": round6(ls_kj - ls_u),
                "diff_vs_B": round6(ls_kj - ls_b),
                "diff_vs_R": round6(ls_kj - ls_r),
            },
        }),
        json!({
            "id": "G4",
            "rule": "Brier(KJ-V5) < min(Brier(U), Brier(B), Brier(R))",
            "pass": g4,
            "value": { "KJV5": br_kj, "U": br_u, "B": br_b, "R": br_r },
        }),
        json!({
            "id": "G5",
            "rule": "LogScore(KJ-V5) > LogScore(B) în ambele jumătăți ale arhivei și pe ultimii 100 pași",
            "pass": g5,
            "value": {
                "half1_diff_KJV5_minus_B": round6(h1_diff),
                "half2_diff_KJV5_minus_B": round6(h2_diff),
                "last100_diff_KJV5_minus_B": round6(l100_diff),
            },
        }),
        json!({
            "id": "G6",
            "rule": "Valoarea martingală a portofoliului Bayesian E_S > 1.0 (garanție E_s >= 0.50)",
            "pass": g6,
            "value": {
                "E_S": round8(final_e),
                "log10_E_S": round8(log10_e),
                "min_E_s": round8(min_e),
                "max_E_s": round8(max_e),
                "guarantee_E_ge_050_all_steps": min_e >= W0_ABSTAIN_PRIOR - 1e-12,
            },
        }),
    ];
    let passed = gates.iter().filter(|g| g["pass"] == true).count();
    (gates, passed)
}

/// Evaluează simultan o listă de configurații `AblationConfig` pe un `LoadedStream` (100% cauzal, `s = 100 .. S-1`).
pub fn evaluate_stream_configs(
    stream: &LoadedStream,
    configs: &[AblationConfig],
    force_full_terminal_enum: bool,
) -> Result<BTreeMap<String, Value>, String> {
    let t0 = Instant::now();
    let spec = stream.spec;
    let s_total = stream.steps.len();
    if s_total <= WARMUP_STEPS {
        return Err(format!("Arhivă prea scurtă pentru {}: S={s_total} <= {WARMUP_STEPS}", spec.code));
    }
    let k_eval = s_total - WARMUP_STEPS;
    let n_cfgs = configs.len();
    let space = KjV5Space::new(spec.n, spec.r, spec.k, spec.m_bonus);
    let has_bonus = spec.m_bonus.is_some();
    let mb_len = spec.m_bonus.unwrap_or(1);

    // 1. Pre-calcul strict cauzal pentru comparatoarele canonice B și R
    let decay = 2.0f64.powf(-1.0 / EW_HALF_LIFE);
    let mut cnt_cum_main = vec![0.0f64; spec.n];
    let mut ew_main = vec![0.0f64; spec.n];
    let mut cnt_cum_bon = vec![0.0f64; mb_len];
    let mut ew_bon = vec![0.0f64; mb_len];

    let mut base_w_b = vec![vec![0.0f64; spec.n]; k_eval];
    let mut base_w_r = vec![vec![0.0f64; spec.n]; k_eval];
    let mut base_jw_b = vec![vec![0.0f64; mb_len]; k_eval];
    let mut base_jw_r = vec![vec![0.0f64; mb_len]; k_eval];

    for (s, step) in stream.steps.iter().enumerate() {
        if s >= WARMUP_STEPS {
            let ie = s - WARMUP_STEPS;
            let sb: f64 = cnt_cum_main.iter().map(|&c| c + 1.0).sum();
            base_w_b[ie] = cnt_cum_main.iter().map(|&c| (c + 1.0) / sb).collect();
            let sr: f64 = ew_main.iter().map(|&c| c + 0.5).sum();
            base_w_r[ie] = ew_main.iter().map(|&c| (c + 0.5) / sr).collect();
            if has_bonus {
                let sjb: f64 = cnt_cum_bon.iter().map(|&c| c + 1.0).sum();
                base_jw_b[ie] = cnt_cum_bon.iter().map(|&c| (c + 1.0) / sjb).collect();
                let sjr: f64 = ew_bon.iter().map(|&c| c + 0.5).sum();
                base_jw_r[ie] = ew_bon.iter().map(|&c| (c + 0.5) / sjr).collect();
            }
        }
        for v in &mut ew_main {
            *v *= decay;
        }
        if has_bonus {
            for v in &mut ew_bon {
                *v *= decay;
            }
        }
        for &(ref d0, jb0) in &step.draws {
            for &b in d0 {
                cnt_cum_main[b] += 1.0;
                ew_main[b] += 1.0;
            }
            if let Some(j0) = jb0 {
                cnt_cum_bon[j0] += 1.0;
                ew_bon[j0] += 1.0;
            }
        }
    }

    // 2. Inițializare stare suficientă și averi pe fiecare configurație
    let mut state = KjV5State::new(space.clone(), true, true, true);
    for step in &stream.steps[..WARMUP_STEPS] {
        state.advance_with_step(step.clone(), None);
    }

    let mut priors_cfgs: Vec<[f64; NUM_EXPERTS]> = Vec::with_capacity(n_cfgs);
    let mut lw_cfgs: Vec<[f64; NUM_EXPERTS]> = Vec::with_capacity(n_cfgs);
    let mut active_union = [false; NUM_EXPERTS];
    active_union[0] = true;
    for cfg in configs {
        let pr = compute_prior_weights(cfg.enable_k1, cfg.enable_k2, cfg.enable_k3);
        let mut lw = [f64::NEG_INFINITY; NUM_EXPERTS];
        for m in 0..NUM_EXPERTS {
            if pr[m] > 0.0 {
                lw[m] = pr[m].ln();
                active_union[m] = true;
            }
        }
        priors_cfgs.push(pr);
        lw_cfgs.push(lw);
    }

    let n_target_draws: usize = stream.steps[WARMUP_STEPS..].iter().map(|s| s.draws.len()).sum();
    let log_u_draw_const = -(binom_f64(spec.n, spec.r) * (mb_len as f64)).ln();
    let q_u_main = (spec.r as f64) / (spec.n as f64);
    let q_u_bon = 1.0 / (mb_len as f64);

    let mut e_curves: Vec<Vec<f64>> = vec![vec![1.0; k_eval + 1]; n_cfgs];
    let mut w_curves: Vec<Vec<[f64; NUM_EXPERTS]>> = priors_cfgs.iter().map(|&pr| vec![pr; k_eval + 1]).collect();

    let mut log_step_cfgs = vec![vec![0.0f64; k_eval]; n_cfgs];
    let mut log_step_u = vec![0.0f64; k_eval];
    let mut log_step_b = vec![0.0f64; k_eval];
    let mut log_step_r = vec![0.0f64; k_eval];

    let mut target_step_of_draw = Vec::with_capacity(n_target_draws);
    let mut brier_cfgs = vec![vec![0.0f64; n_target_draws]; n_cfgs];
    let mut brier_u = vec![0.0f64; n_target_draws];
    let mut brier_b = vec![0.0f64; n_target_draws];
    let mut brier_r = vec![0.0f64; n_target_draws];

    let mut main_hits_cfgs = vec![vec![0usize; n_target_draws]; n_cfgs];
    let mut main_hits_b = vec![0usize; n_target_draws];
    let mut main_hits_r = vec![0usize; n_target_draws];

    let mut bon_hits_cfgs = vec![vec![0usize; n_target_draws]; n_cfgs];
    let mut bon_hits_b = vec![0usize; n_target_draws];
    let mut bon_hits_r = vec![0usize; n_target_draws];

    // Salvăm și eșantion de verificare pas-cu-pas pentru verificatorul independent Python
    let mut sample_step_audit: Vec<Value> = Vec::new();
    let mut draw_cursor = 0usize;

    for ie in 0..k_eval {
        let s = WARMUP_STEPS + ie;
        let step = &stream.steps[s];
        let ops = state.forward_operators();
        let pack = state.evaluate_experts_on_step(&ops, step, &active_union);

        let wb_s = &base_w_b[ie];
        let wr_s = &base_w_r[ie];
        let (log_er_b, q_b_main) = cond_poisson_log_z_and_marginals(wb_s, spec.r);
        let (log_er_r, q_r_main) = cond_poisson_log_z_and_marginals(wr_s, spec.r);
        let tkt_b = top_k_1based(wb_s, spec.k);
        let tkt_r = top_k_1based(wr_s, spec.k);
        let jb_b = if has_bonus { Some(top_k_1based(&base_jw_b[ie], 1)[0]) } else { None };
        let jb_r = if has_bonus { Some(top_k_1based(&base_jw_r[ie], 1)[0]) } else { None };

        // Ponderi, marginale și bilete σ_α pe fiecare configurație înainte de observarea pasului s
        let mut w_cur_all = vec![[0.0f64; NUM_EXPERTS]; n_cfgs];
        let mut q_main_cfgs = vec![vec![0.0f64; spec.n]; n_cfgs];
        let mut q_bon_cfgs = vec![vec![0.0f64; mb_len]; n_cfgs];
        let mut tkt_cfgs: Vec<(Vec<usize>, Option<usize>, f64)> = Vec::with_capacity(n_cfgs);

        for c_i in 0..n_cfgs {
            let lw = &lw_cfgs[c_i];
            let max_lw = lw.iter().copied().filter(|x| x.is_finite()).fold(f64::NEG_INFINITY, f64::max);
            let mut w_c = [0.0f64; NUM_EXPERTS];
            let mut sum_rel = 0.0f64;
            for m in 0..NUM_EXPERTS {
                if lw[m].is_finite() {
                    let r_val = (lw[m] - max_lw).exp();
                    w_c[m] = r_val;
                    sum_rel += r_val;
                }
            }
            for slot in &mut w_c {
                *slot /= sum_rel;
            }
            w_cur_all[c_i] = w_c;

            let mut pi_port = vec![0.0f64; spec.n];
            for m in 0..NUM_EXPERTS {
                if w_c[m] > 0.0 {
                    for i in 0..spec.n {
                        pi_port[i] += w_c[m] * pack.pi_m_main[m][i];
                    }
                    if has_bonus {
                        for b in 0..mb_len {
                            q_bon_cfgs[c_i][b] += w_c[m] * pack.pi_m_bonus[m][b];
                        }
                    }
                }
            }
            q_main_cfgs[c_i] = ticket_to_draw_marginals(&pi_port, spec.n, spec.r, spec.k);
            let sel = select_sigma_alpha_step(&space, &ops, &pack, &w_c, &pi_port);
            tkt_cfgs.push(sel);
        }

        if (ie == 0 || ie == k_eval / 2 || ie + 1 == k_eval) && n_cfgs > 0 {
            let last_ci = n_cfgs - 1;
            sample_step_audit.push(json!({
                "step_index": s,
                "date": step.date,
                "shell_coeffs": ops.shell_coeffs,
                "q_anti_raw_norm": ops.q_anti_raw_norm,
                "q_dir_raw_norm": ops.q_dir_raw_norm,
                "q_anti_phi_dot": ops.q_anti_phi_dot,
                "modes_anti_lambda": [ops.modes_anti.lambda_pos, ops.modes_anti.lambda_neg],
                "modes_dir_lambda": [ops.modes_dir.lambda_pos, ops.modes_dir.lambda_neg],
                "w_portfolio": w_cur_all[last_ci],
                "sigma_alpha": tkt_cfgs[last_ci].0,
                "bonus_alpha": tkt_cfgs[last_ci].1,
                "alpha_log_prob": tkt_cfgs[last_ci].2,
                "step_lrs": pack.step_lrs,
            }));
        }

        // Scorare pe fiecare extragere țintă din pasul s
        for (u_pos, &(ref d0, jb0)) in step.draws.iter().enumerate() {
            let mut in_draw = vec![false; spec.n];
            for &b in d0 {
                in_draw[b] = true;
            }
            let d_1based_set: Vec<usize> = d0.iter().map(|&b| b + 1).collect();
            let jb_1b = jb0.map(|j| j + 1);

            let lp_u = log_u_draw_const;
            let mut lp_b: f64 = d0.iter().map(|&b| wb_s[b].ln()).sum::<f64>() - log_er_b;
            let mut lp_r: f64 = d0.iter().map(|&b| wr_s[b].ln()).sum::<f64>() - log_er_r;

            let mut br_u: f64 = (0..spec.n)
                .map(|i| {
                    let y = if in_draw[i] { 1.0 } else { 0.0 };
                    (q_u_main - y).powi(2)
                })
                .sum();
            let mut br_b: f64 = (0..spec.n)
                .map(|i| {
                    let y = if in_draw[i] { 1.0 } else { 0.0 };
                    (q_b_main[i] - y).powi(2)
                })
                .sum();
            let mut br_r: f64 = (0..spec.n)
                .map(|i| {
                    let y = if in_draw[i] { 1.0 } else { 0.0 };
                    (q_r_main[i] - y).powi(2)
                })
                .sum();

            if let (true, Some(j0)) = (has_bonus, jb0) {
                lp_b += base_jw_b[ie][j0].ln();
                lp_r += base_jw_r[ie][j0].ln();
                for b in 0..mb_len {
                    let yj = if b == j0 { 1.0 } else { 0.0 };
                    br_u += (q_u_bon - yj).powi(2);
                    br_b += (base_jw_b[ie][b] - yj).powi(2);
                    br_r += (base_jw_r[ie][b] - yj).powi(2);
                }
                bon_hits_b[draw_cursor] = usize::from(jb_b == jb_1b);
                bon_hits_r[draw_cursor] = usize::from(jb_r == jb_1b);
            }

            main_hits_b[draw_cursor] = tkt_b.iter().filter(|x| d_1based_set.contains(x)).count();
            main_hits_r[draw_cursor] = tkt_r.iter().filter(|x| d_1based_set.contains(x)).count();
            brier_u[draw_cursor] = br_u;
            brier_b[draw_cursor] = br_b;
            brier_r[draw_cursor] = br_r;
            log_step_u[ie] += lp_u;
            log_step_b[ie] += lp_b;
            log_step_r[ie] += lp_r;
            target_step_of_draw.push(s);

            for c_i in 0..n_cfgs {
                let wc = &w_cur_all[c_i];
                let mut draw_lr_port = 0.0f64;
                for m in 0..NUM_EXPERTS {
                    if wc[m] > 0.0 {
                        draw_lr_port += wc[m] * pack.draw_lrs[u_pos][m];
                    }
                }
                log_step_cfgs[c_i][ie] += lp_u + draw_lr_port.max(1e-300).ln();

                let mut br_kj: f64 = (0..spec.n)
                    .map(|i| {
                        let y = if in_draw[i] { 1.0 } else { 0.0 };
                        (q_main_cfgs[c_i][i] - y).powi(2)
                    })
                    .sum();
                if let (true, Some(j0)) = (has_bonus, jb0) {
                    for b in 0..mb_len {
                        let yj = if b == j0 { 1.0 } else { 0.0 };
                        br_kj += (q_bon_cfgs[c_i][b] - yj).powi(2);
                    }
                    bon_hits_cfgs[c_i][draw_cursor] = usize::from(tkt_cfgs[c_i].1 == jb_1b);
                }
                brier_cfgs[c_i][draw_cursor] = br_kj;
                main_hits_cfgs[c_i][draw_cursor] = tkt_cfgs[c_i].0.iter().filter(|x| d_1based_set.contains(x)).count();
            }

            draw_cursor += 1;
        }

        // Actualizarea exactă a averii Bayesiene W_m^(s+1) = W_m^(s) * step_lrs[m]
        for c_i in 0..n_cfgs {
            let lw = &mut lw_cfgs[c_i];
            for m in 0..NUM_EXPERTS {
                if lw[m].is_finite() {
                    lw[m] += pack.step_lrs[m].max(1e-300).ln();
                }
            }
            let max_lw = lw.iter().copied().filter(|x| x.is_finite()).fold(f64::NEG_INFINITY, f64::max);
            let mut w_next = [0.0f64; NUM_EXPERTS];
            let mut sum_rel = 0.0f64;
            for m in 0..NUM_EXPERTS {
                if lw[m].is_finite() {
                    let r_val = (lw[m] - max_lw).exp();
                    w_next[m] = r_val;
                    sum_rel += r_val;
                }
            }
            let e_next = sum_rel * max_lw.exp();
            for slot in &mut w_next {
                *slot /= sum_rel;
            }
            e_curves[c_i][ie + 1] = e_next;
            w_curves[c_i][ie + 1] = w_next;
        }

        state.advance_with_step(step.clone(), None);
    }

    // 3. Starea post-arhivă (după toți cei S pași!) și asamblarea rapoartelor per configurație
    let ops_post = state.forward_operators();
    let dummy_target = &stream.steps[s_total - 1];
    let pack_post = state.evaluate_experts_on_step(&ops_post, dummy_target, &active_union);

    let expected_u_hits =
        ((spec.k * spec.r) as f64 / (spec.n as f64)) + spec.m_bonus.map_or(0.0, |mb| 1.0 / (mb as f64));
    let half_k = k_eval / 2;
    let mid_step_val = WARMUP_STEPS + half_k;
    let last100_step_val = WARMUP_STEPS + k_eval.saturating_sub(100);

    let all_s: Vec<usize> = (0..k_eval).collect();
    let h1_s: Vec<usize> = (0..half_k).collect();
    let h2_s: Vec<usize> = (half_k..k_eval).collect();
    let l100_s: Vec<usize> = (k_eval.saturating_sub(100)..k_eval).collect();

    let all_d: Vec<usize> = (0..n_target_draws).collect();
    let h1_d: Vec<usize> = (0..n_target_draws).filter(|&i| target_step_of_draw[i] < mid_step_val).collect();
    let h2_d: Vec<usize> = (0..n_target_draws).filter(|&i| target_step_of_draw[i] >= mid_step_val).collect();
    let l100_d: Vec<usize> = (0..n_target_draws).filter(|&i| target_step_of_draw[i] >= last100_step_val).collect();

    let hits_tot_b: Vec<usize> =
        (0..n_target_draws).map(|i| main_hits_b[i] + if has_bonus { bon_hits_b[i] } else { 0 }).collect();
    let hits_tot_r: Vec<usize> =
        (0..n_target_draws).map(|i| main_hits_r[i] + if has_bonus { bon_hits_r[i] } else { 0 }).collect();

    let elapsed_sec = t0.elapsed().as_secs_f64();
    let ds_idx = downsample_indices(k_eval + 1, 140);
    let mut results = BTreeMap::new();

    for (c_i, cfg) in configs.iter().enumerate() {
        let hits_tot_kj: Vec<usize> = (0..n_target_draws)
            .map(|i| main_hits_cfgs[c_i][i] + if has_bonus { bon_hits_cfgs[c_i][i] } else { 0 })
            .collect();

        let win_tot = summarize_window(
            &all_s,
            &all_d,
            expected_u_hits,
            &hits_tot_kj,
            &hits_tot_b,
            &hits_tot_r,
            &log_step_cfgs[c_i],
            &log_step_u,
            &log_step_b,
            &log_step_r,
            &brier_cfgs[c_i],
            &brier_u,
            &brier_b,
            &brier_r,
        );
        let win_h1 = summarize_window(
            &h1_s,
            &h1_d,
            expected_u_hits,
            &hits_tot_kj,
            &hits_tot_b,
            &hits_tot_r,
            &log_step_cfgs[c_i],
            &log_step_u,
            &log_step_b,
            &log_step_r,
            &brier_cfgs[c_i],
            &brier_u,
            &brier_b,
            &brier_r,
        );
        let win_h2 = summarize_window(
            &h2_s,
            &h2_d,
            expected_u_hits,
            &hits_tot_kj,
            &hits_tot_b,
            &hits_tot_r,
            &log_step_cfgs[c_i],
            &log_step_u,
            &log_step_b,
            &log_step_r,
            &brier_cfgs[c_i],
            &brier_u,
            &brier_b,
            &brier_r,
        );
        let win_l100 = summarize_window(
            &l100_s,
            &l100_d,
            expected_u_hits,
            &hits_tot_kj,
            &hits_tot_b,
            &hits_tot_r,
            &log_step_cfgs[c_i],
            &log_step_u,
            &log_step_b,
            &log_step_r,
            &brier_cfgs[c_i],
            &brier_u,
            &brier_b,
            &brier_r,
        );

        let e_c = &e_curves[c_i];
        let (gates, gates_passed) = evaluate_gates(&win_tot, &win_h1, &win_h2, &win_l100, e_c);
        let categories = build_categories(
            spec,
            &main_hits_cfgs[c_i],
            &main_hits_b,
            &main_hits_r,
            &bon_hits_cfgs[c_i],
            &bon_hits_b,
            &bon_hits_r,
        );

        let w_final = w_curves[c_i][k_eval];
        let mut pi_post = vec![0.0f64; spec.n];
        let mut bon_post = vec![0.0f64; mb_len];
        for m in 0..NUM_EXPERTS {
            if w_final[m] > 0.0 {
                for i in 0..spec.n {
                    pi_post[i] += w_final[m] * pack_post.pi_m_main[m][i];
                }
                if has_bonus {
                    for b in 0..mb_len {
                        bon_post[b] += w_final[m] * pack_post.pi_m_bonus[m][b];
                    }
                }
            }
        }

        let do_full_enum = force_full_terminal_enum && cfg.key == "Full_KJV5";
        let post_ev = compute_terminal_multiplet(&space, &ops_post, &pack_post, &w_final, &pi_post, do_full_enum);

        let curve_points: Vec<TrajectoryPoint> = ds_idx
            .iter()
            .map(|&off| {
                let st_idx = WARMUP_STEPS + off.min(k_eval - 1);
                let dt = if off == 0 {
                    stream.steps[WARMUP_STEPS - 1].date.clone()
                } else {
                    stream.steps[WARMUP_STEPS + off - 1].date.clone()
                };
                let masses = decompose_channel_masses(&w_curves[c_i][off]);
                let ev_val = e_c[off];
                TrajectoryPoint {
                    eval_offset: off,
                    step: if off == 0 { WARMUP_STEPS } else { st_idx + 1 },
                    date: dt,
                    e_s: round8(ev_val),
                    log10_e_s: round8(ev_val.max(1e-300).log10()),
                    w_0: round8(masses.w_0),
                    w_k1: round8(masses.w_k1),
                    w_k2: round8(masses.w_k2),
                    w_k3: round8(masses.w_k3),
                }
            })
            .collect();

        let prior_masses = decompose_channel_masses(&priors_cfgs[c_i]);
        let post_masses = decompose_channel_masses(&w_final);
        let final_e = *e_c.last().unwrap_or(&1.0);
        let min_e = e_c.iter().copied().fold(f64::INFINITY, f64::min);
        let max_e = e_c.iter().copied().fold(0.0f64, f64::max);

        let status =
            if gates_passed == 6 { "KJV5_ALL_GATES_PASSED_EXPLORATORY" } else { "KJV5_NO_PROMOTION_EXPLORATORY" };

        let variants_json: Vec<Value> = post_ev
            .variants
            .iter()
            .map(|v| {
                let sym = match v.label {
                    "alpha" => "α",
                    "beta" => "β",
                    "gamma" => "γ",
                    other => other,
                };
                json!({
                    "label": sym,
                    "ticket": v.ticket_1based,
                    "bonus": v.bonus_1based,
                    "log_prob": round8(v.log_prob),
                    "prob": v.log_prob.exp(),
                    "energy": round8(-v.log_prob),
                    "log_lift_vs_uniform": round8(v.log_lift_vs_uniform),
                })
            })
            .collect();

        results.insert(
            cfg.key.to_string(),
            json!({
                "game_code": spec.code,
                "game_name": spec.name,
                "group": spec.group,
                "game_params": {
                    "N": spec.n,
                    "r_draw": spec.r,
                    "k_ticket": spec.k,
                    "m_bonus": spec.m_bonus,
                    "state_space_size": space.total_joint_states,
                    "exact_gh_split_dp_partition": true,
                    "exact_24thread_terminal_enum": post_ev.exact_global_certified,
                },
                "archive": {
                    "file": stream.file_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                    "file_sha256": stream.file_sha256,
                    "version_dir": stream.version_dir,
                    "manifest_sha256": stream.manifest_sha256,
                    "n_draws_total": stream.n_draws_total,
                    "n_steps_total": s_total,
                    "warmup_steps": WARMUP_STEPS,
                    "evaluated_steps": k_eval,
                    "evaluated_target_draws": n_target_draws,
                    "first_date": stream.steps.first().map(|x| x.date.as_str()).unwrap_or(""),
                    "first_eval_date": stream.steps.get(WARMUP_STEPS).map(|x| x.date.as_str()).unwrap_or(""),
                    "last_date": stream.steps.last().map(|x| x.date.as_str()).unwrap_or(""),
                    "double_draw_days": stream.double_draw_days,
                },
                "config_key": cfg.key,
                "label": cfg.label,
                "channels_enabled": {
                    "K1_krawtchouk_shells": cfg.enable_k1,
                    "K2_koopman_v2": cfg.enable_k2,
                    "K3_schnakenberg_v1": cfg.enable_k3,
                },
                "verdict_gates": format!("{gates_passed} / 6"),
                "gates_passed": gates_passed,
                "gates_total": 6,
                "status": status,
                "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
                "gates": gates,
                "windows": {
                    "total": win_tot,
                    "half1": win_h1,
                    "half2": win_h2,
                    "last100": win_l100,
                },
                "categories": categories,
                "martingale_trajectory": {
                    "E_initial": 1.0,
                    "E_S_final": round8(final_e),
                    "log10_E_S": round8(final_e.max(1e-300).log10()),
                    "min_E_s": round8(min_e),
                    "max_E_s": round8(max_e),
                    "guarantee_E_ge_050_verified": min_e >= W0_ABSTAIN_PRIOR - 1e-12,
                    "curve_points": curve_points,
                },
                "bayesian_masses": {
                    "prior_channels": prior_masses,
                    "posterior_channels": post_masses,
                    "prior_experts": priors_cfgs[c_i],
                    "posterior_experts": w_final,
                    "expert_labels": EXPERT_GRID.iter().map(|e| e.label).collect::<Vec<_>>(),
                },
                "post_archive_state": {
                    "epistemic_notice": EPISTEMIC_NOTICE,
                    "step_index": s_total,
                    "last_observed_date": stream.steps.last().map(|x| x.date.as_str()).unwrap_or(""),
                    "variants_jastrow_multiplet": variants_json,
                    "deterministic_core": post_ev.deterministic_core_1based,
                    "det_G3": round8(post_ev.det_g3),
                    "runner_up_gap": round8(post_ev.runner_up_gap),
                    "exact_global_certified": post_ev.exact_global_certified,
                    "pi_inclusion": pi_post.iter().map(|&x| round8(x)).collect::<Vec<_>>(),
                    "bonus_prob": if has_bonus { Some(bon_post.iter().map(|&x| round8(x)).collect::<Vec<_>>()) } else { None },
                    "krawtchouk_shell_coeffs_m123": ops_post.shell_coeffs.map(round8),
                    "theta_shell_ticket": ops_post.theta_shell.iter().map(|&x| round8(x)).collect::<Vec<_>>(),
                    "koopman_v2_anti_norm": round8(ops_post.q_anti_raw_norm),
                    "koopman_v2_dir_norm": round8(ops_post.q_dir_raw_norm),
                    "koopman_v2_anti_eigenvalues": [round8(ops_post.modes_anti.lambda_pos), round8(ops_post.modes_anti.lambda_neg)],
                    "koopman_v2_dir_eigenvalues": [round8(ops_post.modes_dir.lambda_pos), round8(ops_post.modes_dir.lambda_neg)],
                    "koopman_v2_mix_eigenvalues": [round8(ops_post.modes_mix.lambda_pos), round8(ops_post.modes_mix.lambda_neg)],
                },
                "sample_step_audit": if cfg.key == "Full_KJV5" { Value::Array(sample_step_audit.clone()) } else { Value::Array(Vec::new()) },
                "elapsed_sec": round4(elapsed_sec),
                "device": "CPU 24-thread AVX2+FMA IEEE-754 binary64",
            }),
        );
    }

    Ok(results)
}

fn write_sealed_json(path: &Path, val: &Value) -> Result<String, String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if path.exists() {
        return Err(format!("Artefactul {} există deja și nu se rescrie!", path.display()));
    }
    let body = serde_json::to_string_pretty(val).map_err(|e| e.to_string())? + "\n";
    let sha = sha256_bytes(body.as_bytes());
    std::fs::write(path, &body).map_err(|e| format!("{}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o444))
            .map_err(|e| format!("chmod 0444 {}: {e}", path.display()))?;
    }
    Ok(sha)
}

/// Execută întreaga campanie cauzală `KJ-V5` (RO `649`, `540`, `joker` × 4 ablații + 10 loterii internaționale/RNG)
/// și scrie cele 3 rapoarte sigilate `0444` în `kj_v5/rezultate/`.
pub fn run_full_walkforward_suite(repo_root: &Path) -> Result<Value, String> {
    let t0 = Instant::now();
    let ql_root = repo_root.join("next_candidate/quantum_lab_649");
    let qual_path = ql_root.join("kj_v5/rezultate/CALIFICARE_Q_KJV5.json");
    let qual_sha = sha256_file(&qual_path)?;
    let qual_val: Value =
        serde_json::from_slice(&std::fs::read(&qual_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if qual_val["status"] != "KJV5_QUALIFIED" || qual_val["passed"] != 6 {
        return Err("Refuz execuție walk-forward: CALIFICARE_Q_KJV5.json nu este 6/6 KJV5_QUALIFIED!".into());
    }

    let seal_code_path = ql_root.join("kj_v5/SIGILIU_COD_KJV5.json");
    if !seal_code_path.exists() {
        return Err("Refuz execuție walk-forward: kj_v5/SIGILIU_COD_KJV5.json lipsește!".into());
    }
    let seal_code_sha = sha256_file(&seal_code_path)?;
    let seal_code_val: Value = serde_json::from_slice(&std::fs::read(&seal_code_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if let Some(code_map) = seal_code_val["code_sha256"].as_object() {
        for (rel, exp_v) in code_map {
            let exp = exp_v.as_str().ok_or_else(|| format!("hash invalid în SIGILIU_COD_KJV5 pentru {rel}"))?;
            let got = sha256_file(&repo_root.join(rel))?;
            if got != exp {
                return Err(format!("SIGILIU_COD_KJV5 încălcat pentru {rel}: {got} != {exp}"));
            }
        }
    } else {
        return Err("SIGILIU_COD_KJV5.json fără code_sha256!".into());
    }

    // 1. RO: 649, 540, joker (4 configurații de ablație + enumerare paralelă exactă pe 24 fire CPU la final)
    let mut ro_full_games = serde_json::Map::new();
    let mut ro_ablation_games = serde_json::Map::new();
    let t_ro = Instant::now();

    for spec in RO_SPECS {
        let stream = load_ro_live_stream(&ql_root, spec.code)?;
        let cfg_map = evaluate_stream_configs(&stream, &ABLATION_CONFIGS, true)?;
        let full_rep = cfg_map.get("Full_KJV5").cloned().ok_or("Full_KJV5 lipsă")?;
        ro_full_games.insert(spec.code.to_string(), full_rep.clone());

        let comp_table: Vec<Value> = ABLATION_CONFIGS
            .iter()
            .filter_map(|c| {
                let r = cfg_map.get(c.key)?;
                let wt = &r["windows"]["total"];
                Some(json!({
                    "config_key": c.key,
                    "label": c.label,
                    "verdict_gates": r["verdict_gates"],
                    "hits_KJV5": wt["hits_KJV5"],
                    "hits_B": wt["hits_B"],
                    "hits_R": wt["hits_R"],
                    "excess_hits_vs_U": wt["excess_hits_vs_U"],
                    "log_score_KJV5": wt["log_score_KJV5"],
                    "log_score_U": wt["log_score_U"],
                    "log_score_B": wt["log_score_B"],
                    "brier_KJV5": wt["brier_KJV5"],
                    "brier_U": wt["brier_U"],
                    "E_S_final": r["martingale_trajectory"]["E_S_final"],
                    "min_E_s": r["martingale_trajectory"]["min_E_s"],
                    "w_0_final": r["bayesian_masses"]["posterior_channels"]["w_0"],
                }))
            })
            .collect();

        ro_ablation_games.insert(
            spec.code.to_string(),
            json!({
                "game_code": spec.code,
                "game_name": spec.name,
                "archive": full_rep["archive"],
                "comparison_table": comp_table,
                "configs": cfg_map,
            }),
        );
    }
    let ro_elapsed = t_ro.elapsed().as_secs_f64();

    // 2. Internațional & RNG Electronic (10 fluxuri rulate în paralel pe firele CPU)
    let t_intl = Instant::now();
    let (prov_intl_sha, intl_streams) = load_all_intl_streams(&ql_root)?;
    let full_only = [ABLATION_CONFIGS[3]]; // Full_KJV5

    let intl_results_vec: Vec<Result<(StreamSpec, Value), String>> = std::thread::scope(|s| {
        let cfg_ref = &full_only;
        let handles: Vec<_> = intl_streams
            .iter()
            .map(|st_in| {
                s.spawn(move || {
                    let mut m = evaluate_stream_configs(st_in, cfg_ref, false)?;
                    let v = m.remove("Full_KJV5").ok_or_else(|| "Full_KJV5 lipsă".to_string())?;
                    Ok((st_in.spec, v))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("fir walkforward intl eșuat")).collect()
    });

    let mut mech_map = serde_json::Map::new();
    let mut rng_map = serde_json::Map::new();
    for item in intl_results_vec {
        let (sp, val) = item?;
        if sp.group == "mechanical_external" {
            mech_map.insert(sp.code.to_string(), val);
        } else {
            rng_map.insert(sp.code.to_string(), val);
        }
    }

    // Verificarea sincronizării pe zile pentru perechea 7/35 HU Skandináv (kézi mecanic vs gépi RNG electronic)
    let sk_mech_st = intl_streams.iter().find(|s| s.spec.code == "hu_skandinav_mech").ok_or("sk_mech lipsă")?;
    let sk_rng_st = intl_streams.iter().find(|s| s.spec.code == "hu_skandinav_rng").ok_or("sk_rng lipsă")?;
    let same_sk_dates = sk_mech_st.steps.len() == sk_rng_st.steps.len()
        && sk_mech_st.steps.iter().zip(sk_rng_st.steps.iter()).all(|(a, b)| a.date == b.date);

    let sk_m_val = mech_map.get("hu_skandinav_mech").ok_or("sk_mech eval lipsă")?;
    let sk_r_val = rng_map.get("hu_skandinav_rng").ok_or("sk_rng eval lipsă")?;

    let paired_skandinav = json!({
        "same_dates_verified": same_sk_dates,
        "evaluated_steps": sk_m_val["archive"]["evaluated_steps"],
        "first_eval_date": sk_m_val["archive"]["first_eval_date"],
        "last_date": sk_m_val["archive"]["last_date"],
        "mechanical_kezi": {
            "verdict_gates": sk_m_val["verdict_gates"],
            "hits_KJV5": sk_m_val["windows"]["total"]["hits_KJV5"],
            "hits_B": sk_m_val["windows"]["total"]["hits_B"],
            "hits_R": sk_m_val["windows"]["total"]["hits_R"],
            "excess_hits_vs_U": sk_m_val["windows"]["total"]["excess_hits_vs_U"],
            "log_score_KJV5": sk_m_val["windows"]["total"]["log_score_KJV5"],
            "log_score_U": sk_m_val["windows"]["total"]["log_score_U"],
            "brier_KJV5": sk_m_val["windows"]["total"]["brier_KJV5"],
            "E_S_final": sk_m_val["martingale_trajectory"]["E_S_final"],
            "min_E_s": sk_m_val["martingale_trajectory"]["min_E_s"],
            "posterior_channels": sk_m_val["bayesian_masses"]["posterior_channels"],
        },
        "electronic_rng_gepi": {
            "verdict_gates": sk_r_val["verdict_gates"],
            "hits_KJV5": sk_r_val["windows"]["total"]["hits_KJV5"],
            "hits_B": sk_r_val["windows"]["total"]["hits_B"],
            "hits_R": sk_r_val["windows"]["total"]["hits_R"],
            "excess_hits_vs_U": sk_r_val["windows"]["total"]["excess_hits_vs_U"],
            "log_score_KJV5": sk_r_val["windows"]["total"]["log_score_KJV5"],
            "log_score_U": sk_r_val["windows"]["total"]["log_score_U"],
            "brier_KJV5": sk_r_val["windows"]["total"]["brier_KJV5"],
            "E_S_final": sk_r_val["martingale_trajectory"]["E_S_final"],
            "min_E_s": sk_r_val["martingale_trajectory"]["min_E_s"],
            "posterior_channels": sk_r_val["bayesian_masses"]["posterior_channels"],
        },
    });
    let intl_elapsed = t_intl.elapsed().as_secs_f64();

    let gen_utc = utc_now_iso();
    let rez_dir = ql_root.join("kj_v5/rezultate");

    let wf_ro_doc = json!({
        "protocol_id": "KJV5-RCV2-P1",
        "phase": "walkforward_ro_live",
        "qualification_sha256": qual_sha,
        "code_seal_sha256": seal_code_sha,
        "technical_incidents": 1,
        "games": ro_full_games,
        "total_elapsed_sec": round4(ro_elapsed),
        "generated_utc": gen_utc,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    });
    let abl_ro_doc = json!({
        "protocol_id": "KJV5-RCV2-P1",
        "phase": "ablation_ro_live",
        "qualification_sha256": qual_sha,
        "code_seal_sha256": seal_code_sha,
        "ablation_configs": ABLATION_CONFIGS,
        "games": ro_ablation_games,
        "total_elapsed_sec": round4(ro_elapsed),
        "generated_utc": gen_utc,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    });
    let intl_doc = json!({
        "protocol_id": "KJV5-RCV2-P1",
        "phase": "international_mechanical_and_electronic_rng",
        "qualification_sha256": qual_sha,
        "code_seal_sha256": seal_code_sha,
        "provenance_intl_sha256": prov_intl_sha,
        "mechanical_external": mech_map,
        "electronic_rng": rng_map,
        "paired_skandinav_comparison": paired_skandinav,
        "total_elapsed_sec": round4(intl_elapsed),
        "generated_utc": gen_utc,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    });

    let sha_wf = write_sealed_json(&rez_dir.join("WALKFORWARD_RO_KJV5.json"), &wf_ro_doc)?;
    let sha_abl = write_sealed_json(&rez_dir.join("ABLATIE_RO_KJV5.json"), &abl_ro_doc)?;
    let sha_intl = write_sealed_json(&rez_dir.join("INTERNATIONAL_RNG_KJV5.json"), &intl_doc)?;

    Ok(json!({
        "status": "COMPLETED",
        "protocol_id": "KJV5-RCV2-P1",
        "total_elapsed_sec": round4(t0.elapsed().as_secs_f64()),
        "artifacts_sha256": {
            "WALKFORWARD_RO_KJV5.json": sha_wf,
            "ABLATIE_RO_KJV5.json": sha_abl,
            "INTERNATIONAL_RNG_KJV5.json": sha_intl,
        },
        "ro_verdicts": {
            "649": wf_ro_doc["games"]["649"]["verdict_gates"],
            "540": wf_ro_doc["games"]["540"]["verdict_gates"],
            "joker": wf_ro_doc["games"]["joker"]["verdict_gates"],
        },
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

/// Vedere rapidă read-only pentru fila 1 a ferestrei Tauri v2 (`⚡ KJ-V5 · Rust Nativ`):
/// citește rapoartele sigilate `0444` și verifică sincronizarea cu `CURRENT_LIVE.json` (avansând cauzal în memorie
/// dacă `CURRENT_LIVE.json` conține extrageri mai noi decât `last_date` din artefactul sigilat!).
pub fn load_kjv5_ro_view(ql_root: &Path, game_code: &str, config_key: &str) -> Result<Value, String> {
    let rez_dir = ql_root.join("kj_v5/rezultate");
    let qual_p = rez_dir.join("CALIFICARE_Q_KJV5.json");
    let abl_p = rez_dir.join("ABLATIE_RO_KJV5.json");
    let intl_p = rez_dir.join("INTERNATIONAL_RNG_KJV5.json");

    let qual_val: Value =
        serde_json::from_slice(&std::fs::read(&qual_p).map_err(|e| format!("{}: {e}", qual_p.display()))?)
            .map_err(|e| e.to_string())?;
    let abl_bytes = std::fs::read(&abl_p).map_err(|e| format!("{}: {e}", abl_p.display()))?;
    let abl_sha = sha256_bytes(&abl_bytes);
    let abl_val: Value = serde_json::from_slice(&abl_bytes).map_err(|e| e.to_string())?;

    let game_blk = abl_val["games"]
        .get(game_code)
        .ok_or_else(|| format!("joc RO necunoscut în ABLATIE_RO_KJV5.json: {game_code}"))?;
    let mut cfg_rep = game_blk["configs"]
        .get(config_key)
        .cloned()
        .ok_or_else(|| format!("configurație necunoscută '{config_key}' (K1, K2, K1_K2, Full_KJV5)"))?;

    // Verificăm sincronizarea cauzală cu CURRENT_LIVE.json!
    let live_stream = load_ro_live_stream(ql_root, game_code)?;
    let sealed_sha = cfg_rep["archive"]["file_sha256"].as_str().unwrap_or("");
    let live_synced = if sealed_sha != live_stream.file_sha256 {
        let cfg_spec = ABLATION_CONFIGS
            .iter()
            .copied()
            .find(|c| c.key == config_key)
            .ok_or_else(|| format!("configurație invalidă: {config_key}"))?;
        let mut fresh = evaluate_stream_configs(&live_stream, &[cfg_spec], true)?;
        cfg_rep = fresh.remove(config_key).ok_or("eroare recalcul cauzal live")?;
        "recalculat_cauzal_live"
    } else {
        "sincronizat_sha256_exact"
    };

    // Rezumat compact al celor 10 loterii internaționale/RNG pentru tabelul din fila 1
    let mut intl_rows = Vec::new();
    if intl_p.exists() {
        let intl_val: Value =
            serde_json::from_slice(&std::fs::read(&intl_p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        for grp in ["mechanical_external", "electronic_rng"] {
            if let Some(obj) = intl_val[grp].as_object() {
                for (_, r) in obj {
                    let wt = &r["windows"]["total"];
                    intl_rows.push(json!({
                        "code": r["game_code"],
                        "name": r["game_name"],
                        "group": grp,
                        "evaluated_steps": r["archive"]["evaluated_steps"],
                        "verdict_gates": r["verdict_gates"],
                        "hits_KJV5": wt["hits_KJV5"],
                        "hits_B": wt["hits_B"],
                        "hits_R": wt["hits_R"],
                        "excess_hits_vs_U": wt["excess_hits_vs_U"],
                        "log_diff_vs_U": round6(wt["log_score_KJV5"].as_f64().unwrap_or(0.0) - wt["log_score_U"].as_f64().unwrap_or(0.0)),
                        "E_S_final": r["martingale_trajectory"]["E_S_final"],
                        "min_E_s": r["martingale_trajectory"]["min_E_s"],
                        "w_0_final": r["bayesian_masses"]["posterior_channels"]["w_0"],
                    }));
                }
            }
        }
    }

    Ok(json!({
        "protocol_id": "KJV5-RCV2-P1",
        "ablation_sha256": abl_sha,
        "live_sync_state": live_synced,
        "qualification": {
            "status": qual_val["status"],
            "passed": qual_val["passed"],
            "total": qual_val["total"],
            "technical_incidents": qual_val["technical_incidents"],
            "gates": qual_val["gates"],
        },
        "selected": cfg_rep,
        "comparison_table": game_blk["comparison_table"],
        "international_overview": intl_rows,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

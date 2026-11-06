//! Simulatorul CUDA `C4-v2` cu `1,000,000` de Traiectorii Multi-An (`1 / 5 / 10 ani` = `104 / 520 / 1,040` extrageri)
//! (`Sesiunea 3 / Poarta A4` din `PROTOCOL_RUST_CORE_V3.md`).
//! - PRNG Philox-4x32-10 fără coliziuni pe `RTX 5060 sm_120`;
//! - Reporturi dinamice pe categorii + șocuri dinamice de vânzări + câștigători `NB2` ai mulțimii;
//! - Recurență Markov analitică exactă în `u128_s96` pentru validarea Porții A4 în `±4 SE` (`1 extragere` și `1 / 5 / 10 ani`);
//! - Cuantile exacte `[q01, q05, q25, q50, q75, q95, q99]` pe toate cele `1,000,000` de traiectorii.
//! Guvernanță științifică: `positive_claims_allowed = false`.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::c3v2::{
    encode_v3bin_header_64, load_crowd_params_from_pmv3_report, make_writable_if_exists,
    sha256_file_c3, upsert_vault_manifest_record, C3V2PortfolioReport, C3V2_V3BIN_NAME,
};
use crate::cdv3::{mask_to_numbers_1based, unpack_joker_ticket};
use crate::cuda_ffi::{
    check_rss_guard, gpu_c3v2_eval_540, gpu_c3v2_eval_649, gpu_c3v2_eval_joker,
    gpu_c4v2_compute_rates_raw, gpu_c4v2_prep_hits_joker, gpu_c4v2_prep_hits_u32,
    gpu_c4v2_sim_chunk_raw, GpuCrowdTables540, GpuCrowdTables649, GpuCrowdTablesJoker,
    ResidentGpuUniverse,
};
use crate::pmv3::{is_patterned_ticket, CdV3QRelabeledReport, PmV3CalibrationReport};
use crate::vault::{
    MappedVaultUniverse, VaultFileRecord, COUNT_540_DRAWS6, COUNT_649,
    COUNT_JOKER_FULL24M, VAULT_HEADER_BYTES, VAULT_MAGIC,
};

pub const TABLE_KIND_C4V2_MC1M: u32 = 8;
pub const C4V2_V3BIN_NAME: &str = "c4v2_mc1m_summary.v3bin";
pub const C4V2_MC_TRAJECTORIES: usize = 1_000_000;
pub const C4V2_CHUNK_LEN: usize = 250_000;
pub const C4V2_SEED: u64 = 20_261_104;
pub const HORIZON_YEARS: [u32; 3] = [1, 5, 10];
pub const HORIZON_DRAWS: [usize; 3] = [104, 520, 1_040];

/// Construiește grila deterministă de 16 șocuri multiplicative de vânzări `xi_l > 0` cu media exactă `1.0`.
pub fn build_sales_shock_xi16() -> [f64; 16] {
    // Cuantile Gaussiene simetrice pe (l + 0.5) / 16 cu sigma = 0.14, normalizate exact la media 1.0
    const Z16: [f64; 16] = [
        -1.8627318674216514,
        -1.3180108974099655,
        -1.0099901679706244,
        -0.776421760665102,
        -0.5791321620444457,
        -0.40225006539246635,
        -0.23720210932049704,
        -0.07841240562624398,
        0.07841240562624398,
        0.23720210932049704,
        0.40225006539246635,
        0.5791321620444457,
        0.776421760665102,
        1.0099901679706244,
        1.3180108974099655,
        1.8627318674216514,
    ];
    let sigma = 0.14f64;
    let mut raw = [0.0f64; 16];
    let mut s = 0.0f64;
    for l in 0..16 {
        raw[l] = (sigma * Z16[l]).exp();
        s += raw[l];
    }
    let mean = s / 16.0;
    for x in &mut raw {
        *x /= mean;
    }
    raw
}

/// Evaluează recurența Markov analitică exactă de ordinul 1 și 2 (medie + varianță cu covarianță temporală de resetare)
/// pe `1`, `104`, `520` și `1,040` extrageri din vectorul de rate `rates` (`n_cats * 12`) al unui portofoliu pe un joc.
pub fn compute_exact_markov_horizon_moments(
    game: &str,
    rates_cat12: &[f64],
    rep0_8: &[f64; 8],
    fnew_8: &[f64; 8],
    var_d0_total_exact: f64,
) -> (f64, [f64; 3], [f64; 3]) {
    let n_shared = match game {
        "649" | "540" => 3usize,
        "joker" => 8usize,
        _ => unreachable!(),
    };
    let (fixed_iv_mean, fixed_iv_m2) = if game == "649" {
        (rates_cat12[3 * 12], rates_cat12[3 * 12 + 6])
    } else {
        (0.0, 0.0)
    };
    let var_fixed_iv = (fixed_iv_m2 - fixed_iv_mean * fixed_iv_mean).max(0.0);

    let mut draw0_exp = fixed_iv_mean;
    let mut hor_exp = [
        104.0 * fixed_iv_mean,
        520.0 * fixed_iv_mean,
        1_040.0 * fixed_iv_mean,
    ];
    let mut hor_var_cats = [
        104.0 * var_fixed_iv,
        520.0 * var_fixed_iv,
        1_040.0 * var_fixed_iv,
    ];
    let mut sum_var_c_d0 = var_fixed_iv;

    for c in 0..n_shared {
        let b = c * 12;
        let psi0 = rates_cat12[b];
        let rho0 = rates_cat12[b + 1];
        let psi_bar = rates_cat12[b + 2];
        let eta_unit = rates_cat12[b + 3];
        let rho_bar = rates_cat12[b + 4];
        let beta_unit = rates_cat12[b + 5];
        let h2_0 = rates_cat12[b + 6];
        let h2_bar0 = rates_cat12[b + 7];
        let h2_bar1 = rates_cat12[b + 8];
        let h2_bar2 = rates_cat12[b + 9];
        let gamma_unit = rates_cat12[b + 10];
        let f = fnew_8[c];

        let mut er = rep0_8[c];
        let mut er2 = er * er;
        let mut ewin = [0.0f64; 1_040];
        let mut ewin2 = [0.0f64; 1_040];

        for s in 0..1_040usize {
            if s == 0 {
                let a0 = er + f;
                let a0_sq = a0 * a0;
                ewin[0] = a0 * psi0;
                ewin2[0] = a0_sq * h2_0;
                er = a0 * rho0;
                er2 = a0_sq * rho0;
            } else {
                ewin[s] = er * psi_bar + f * eta_unit;
                ewin2[s] = er2 * h2_bar0 + 2.0 * f * er * h2_bar1 + f * f * h2_bar2;
                let er_next = er * rho_bar + f * beta_unit;
                let er2_next = er2 * rho_bar + 2.0 * f * er * beta_unit + f * f * gamma_unit;
                er = er_next;
                er2 = er2_next;
            }
        }

        // Traiectoria analitică de după resetarea reportului (R = 0 la pasul >= 1)
        let mut er0 = 0.0f64;
        let mut m0 = [0.0f64; 1_040];
        for tau in 0..1_040usize {
            m0[tau] = er0 * psi_bar + f * eta_unit;
            er0 = er0 * rho_bar + f * beta_unit;
        }

        draw0_exp += ewin[0];
        sum_var_c_d0 += (ewin2[0] - ewin[0] * ewin[0]).max(0.0);

        for (hi, &h_len) in HORIZON_DRAWS.iter().enumerate() {
            let mut c_mean = 0.0f64;
            let mut c_var = 0.0f64;
            for s in 0..h_len {
                c_mean += ewin[s];
                c_var += (ewin2[s] - ewin[s] * ewin[s]).max(0.0);
                let mut cov_tail = 0.0f64;
                for t in (s + 1)..h_len {
                    cov_tail += m0[t - s - 1] - ewin[t];
                }
                c_var += 2.0 * ewin[s] * cov_tail;
            }
            hor_exp[hi] += c_mean;
            hor_var_cats[hi] += c_var.max(0.0);
        }
    }

    // Covarianța intra-extragere între categorii (zero/ușor negativă pt K=1, pozitivă pt roți mari K>1)
    let cov_cross_d0 = var_d0_total_exact - sum_var_c_d0;
    let mut hor_var = [0.0f64; 3];
    for (hi, &h_len) in HORIZON_DRAWS.iter().enumerate() {
        hor_var[hi] = (hor_var_cats[hi] + (h_len as f64) * cov_cross_d0).max(0.0);
    }

    (draw0_exp, hor_exp, hor_var)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct C4NetQuantiles {
    pub q01: f64,
    pub q05: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub q95: f64,
    pub q99: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct C4HorizonMetrics {
    pub years: u32,
    pub draws: usize,
    pub total_stake_lei: f64,
    pub gross_mean_sim_lei: f64,
    pub gross_mean_analytic_lei: f64,
    pub net_mean_sim_lei: f64,
    pub net_mean_analytic_lei: f64,
    pub net_per_draw_sim_lei: f64,
    pub net_per_draw_analytic_lei: f64,
    pub net_se_sim_lei: f64,
    pub net_se_markov_exact_lei: f64,
    pub net_se_reference_lei: f64,
    pub z_score_vs_analytic: f64,
    pub within_4se_analytic: bool,
    pub return_ratio_rho_sim: f64,
    pub return_ratio_rho_analytic: f64,
    pub prob_net_profit_ge_0: f64,
    pub prob_any_single_prize_ge_1000: f64,
    pub net_profit_quantiles_lei: C4NetQuantiles,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct C4StrategyResult {
    pub strategy_id: String,
    pub category_group: String,
    pub games_label: String,
    pub k_649: usize,
    pub k_540: usize,
    pub k_joker: usize,
    pub stake_per_draw_lei: f64,
    pub p_win_single_draw_exact: f64,
    pub p_win_single_draw_sim: f64,
    pub p_win_se: f64,
    pub p_win_z_score: f64,
    pub p_win_within_4se: bool,
    pub draw0_gross_mean_sim_lei: f64,
    pub draw0_gross_mean_analytic_lei: f64,
    pub draw0_gross_mean_c3v2_lei: f64,
    pub draw0_analytic_vs_c3v2_abs_diff: f64,
    pub draw0_se_sample_lei: f64,
    pub draw0_se_exact_lei: f64,
    pub draw0_z_score_exact_se: f64,
    pub draw0_within_4se: bool,
    pub markov_rates_649: Option<Vec<f64>>,
    pub markov_rates_540: Option<Vec<f64>>,
    pub markov_rates_joker: Option<Vec<f64>>,
    pub initial_ticket_preview: Vec<String>,
    pub horizons: Vec<C4HorizonMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct C4V2SimulationReport {
    pub artifact: String,
    pub version: String,
    pub positive_claims_allowed: bool,
    pub prng_algorithm: String,
    pub seed: u64,
    pub trajectories_m: usize,
    pub chunk_trajectories: usize,
    pub horizon_years: Vec<u32>,
    pub horizon_draws: Vec<usize>,
    pub sales_shock_xi16: [f64; 16],
    pub rep0_by_game: serde_json::Value,
    pub fnew_by_game: serde_json::Value,
    pub source_pmv3_sha256: String,
    pub source_cdv3q_sha256: String,
    pub source_c3v2_sha256: String,
    pub vault_summary_v3bin: String,
    pub vault_summary_sha256: String,
    pub vault_summary_bytes: u64,
    pub vault_summary_records: u64,
    pub gpu_sim_649_ms: f64,
    pub gpu_sim_540_ms: f64,
    pub gpu_sim_joker_ms: f64,
    pub total_physical_draws_simulated: u64,
    pub all_draw0_within_4se: bool,
    pub all_1y_within_4se: bool,
    pub all_5y_10y_within_4se: bool,
    pub all_p_win_within_4se: bool,
    pub all_analytic_match_c3v2_lt_1e11: bool,
    pub rss_current_bytes: u64,
    pub rss_peak_hwm_bytes: u64,
    pub rss_limit_bytes: u64,
    pub rss_guard_ok: bool,
    pub build_seconds: f64,
    pub c4v2_gate_pass: bool,
    pub strategies: Vec<C4StrategyResult>,
}

#[inline]
fn splitmix64_c4(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pick_canonical_uniform_ticket(game: &str, host_univ: &[u64], seed: u64) -> u64 {
    let mut rng = seed;
    let gcols = if game == "649" { 7usize } else { 5usize };
    loop {
        let idx = (splitmix64_c4(&mut rng) as usize) % host_univ.len();
        let tk = host_univ[idx];
        let bm = if game == "joker" {
            tk & ((1u64 << 45) - 1)
        } else {
            tk
        };
        let nums = mask_to_numbers_1based(bm);
        if !is_patterned_ticket(&nums, gcols) {
            return tk;
        }
    }
}

fn extract_rep0_and_fnew(
    pm_rep: &PmV3CalibrationReport,
    game: &str,
) -> Result<([f64; 8], [f64; 8]), String> {
    let wf = pm_rep
        .walk_forward_games
        .iter()
        .find(|w| w.game == game)
        .ok_or_else(|| format!("joc lipsa in PM_V3_CALIBRARE.json: {game}"))?;
    let cats: &[&str] = match game {
        "649" | "540" => &["I", "II", "III"],
        "joker" => &["I", "II", "III", "IV", "V", "VI", "VII", "VIII"],
        _ => unreachable!(),
    };
    let mut rep0 = [0.0f64; 8];
    let mut fnew = [0.0f64; 8];
    for (i, &cn) in cats.iter().enumerate() {
        rep0[i] = wf.terminal_state.report_in.get(cn).copied().unwrap_or(0.0);
        fnew[i] = wf
            .terminal_state
            .fund_hat_new
            .get(cn)
            .copied()
            .unwrap_or(0.0);
    }
    Ok((rep0, fnew))
}

#[derive(Clone)]
struct StrategySpec {
    strategy_id: String,
    category_group: String,
    games_label: String,
    leg_649: Option<usize>,
    leg_540: Option<usize>,
    leg_jok: Option<usize>,
    stake_per_draw: f64,
}

fn format_ticket_preview(
    t649: Option<&[u64]>,
    t540: Option<&[u64]>,
    tjok: Option<&[u64]>,
) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(s) = t649 {
        for &m in s.iter().take(2) {
            out.push(format!("6/49:{:?}", mask_to_numbers_1based(m)));
        }
    }
    if let Some(s) = t540 {
        for &m in s.iter().take(2) {
            out.push(format!("5/40:{:?}", mask_to_numbers_1based(m)));
        }
    }
    if let Some(s) = tjok {
        for &pk in s.iter().take(2) {
            let jt = unpack_joker_ticket(pk);
            out.push(format!("Joker:{:?}+J{}", jt.numbers, jt.joker));
        }
    }
    out
}

pub fn run_full_c4v2_pipeline(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    pmv3_json_path: &Path,
    cdv3q_json_path: &Path,
    c3v2_json_path: &Path,
    out_c4v2_json_path: &Path,
) -> Result<C4V2SimulationReport, String> {
    let t0 = Instant::now();
    check_rss_guard(0)?;

    let (_, pm_sha) = sha256_file_c3(pmv3_json_path)?;
    let (_, cdq_sha) = sha256_file_c3(cdv3q_json_path)?;
    let (_, c3_sha) = sha256_file_c3(c3v2_json_path)?;

    let pm_rep: PmV3CalibrationReport = serde_json::from_str(
        &fs::read_to_string(pmv3_json_path)
            .map_err(|e| format!("read {}: {e}", pmv3_json_path.display()))?,
    )
    .map_err(|e| format!("parse PM_V3_CALIBRARE.json: {e}"))?;
    let cdq_rep: CdV3QRelabeledReport = serde_json::from_str(
        &fs::read_to_string(cdv3q_json_path)
            .map_err(|e| format!("read {}: {e}", cdv3q_json_path.display()))?,
    )
    .map_err(|e| format!("parse CD_V3Q_REETICHETAT.json: {e}"))?;
    let c3_rep: C3V2PortfolioReport = serde_json::from_str(
        &fs::read_to_string(c3v2_json_path)
            .map_err(|e| format!("read {}: {e}", c3v2_json_path.display()))?,
    )
    .map_err(|e| format!("parse C3_V2_PORTOFOLII.json: {e}"))?;

    // Mapăm `c3v2_pareto_portfolios.v3bin` pentru extragerea exactă a biletelor fiecărui portofoliu C3-v2
    let c3_v3bin_path = vault.vault_dir.join(C3V2_V3BIN_NAME);
    let c3_f =
        File::open(&c3_v3bin_path).map_err(|e| format!("open {}: {e}", c3_v3bin_path.display()))?;
    let c3_mm = unsafe { memmap2::Mmap::map(&c3_f) }
        .map_err(|e| format!("mmap {}: {e}", c3_v3bin_path.display()))?;
    let c3_tks: &[u64] = unsafe {
        std::slice::from_raw_parts(
            c3_mm[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
            c3_rep.vault_portfolios_records as usize,
        )
    };

    // 1. Construim listele de portofolii per joc (`<= 16` portofolii pe fiecare joc) + cele 30 strategii canonice
    let mut ports_649: Vec<Vec<u64>> = Vec::with_capacity(14);
    let mut ports_540: Vec<Vec<u64>> = Vec::with_capacity(13);
    let mut ports_jok: Vec<Vec<u64>> = Vec::with_capacity(14);
    let mut strat_specs: Vec<StrategySpec> = Vec::with_capacity(30);

    // 1.A Baselines uniforme cu 1 bilet (și U_3 mixt)
    let u649_tk = pick_canonical_uniform_ticket("649", vault.masks_649(), C4V2_SEED ^ 0x649);
    let u540_tk = pick_canonical_uniform_ticket("540", vault.tickets_540(), C4V2_SEED ^ 0x540);
    let ujok_tk = pick_canonical_uniform_ticket("joker", vault.full_joker(), C4V2_SEED ^ 0x4520);
    ports_649.push(vec![u649_tk]); // idx 0
    ports_540.push(vec![u540_tk]); // idx 0
    ports_jok.push(vec![ujok_tk]); // idx 0

    strat_specs.push(StrategySpec {
        strategy_id: "U_649".to_string(),
        category_group: "uniform_baseline".to_string(),
        games_label: "649".to_string(),
        leg_649: Some(0),
        leg_540: None,
        leg_jok: None,
        stake_per_draw: 8.0,
    });
    strat_specs.push(StrategySpec {
        strategy_id: "U_540".to_string(),
        category_group: "uniform_baseline".to_string(),
        games_label: "540".to_string(),
        leg_649: None,
        leg_540: Some(0),
        leg_jok: None,
        stake_per_draw: 5.0,
    });
    strat_specs.push(StrategySpec {
        strategy_id: "U_joker".to_string(),
        category_group: "uniform_baseline".to_string(),
        games_label: "joker".to_string(),
        leg_649: None,
        leg_540: None,
        leg_jok: Some(0),
        stake_per_draw: 7.0,
    });
    strat_specs.push(StrategySpec {
        strategy_id: "U_3".to_string(),
        category_group: "uniform_baseline".to_string(),
        games_label: "649+540+joker".to_string(),
        leg_649: Some(0),
        leg_540: Some(0),
        leg_jok: Some(0),
        stake_per_draw: 20.0,
    });

    // 1.B Portofolii optime C3-v2 pe joc individual la B in {40, 100, 500} x {Max-EV (mu=0), Max-P_any (mu=inf)}
    for game in ["649", "540", "joker"] {
        for budget in [40u32, 100, 500] {
            let fr = c3_rep
                .single_game_frontiers
                .iter()
                .find(|f| f.game == game && f.budget_lei == budget)
                .ok_or_else(|| format!("lipseste frontiera C3-v2 ({game}, B={budget})"))?;
            for &(pt_idx, reg_tag) in &[(0usize, "MAX_EV"), (fr.points.len() - 1, "MAX_PANY")] {
                let pt = &fr.points[pt_idx];
                let off = pt.vault_offset_records as usize;
                let len = pt.vault_len_records as usize;
                let tks = c3_tks[off..off + len].to_vec();
                let (l649, l540, ljok) = match game {
                    "649" => {
                        let idx = ports_649.len();
                        ports_649.push(tks);
                        (Some(idx), None, None)
                    }
                    "540" => {
                        let idx = ports_540.len();
                        ports_540.push(tks);
                        (None, Some(idx), None)
                    }
                    "joker" => {
                        let idx = ports_jok.len();
                        ports_jok.push(tks);
                        (None, None, Some(idx))
                    }
                    _ => unreachable!(),
                };
                strat_specs.push(StrategySpec {
                    strategy_id: format!("C3V2_{}_B{}_{}", game.to_uppercase(), budget, reg_tag),
                    category_group: "c3v2_single_game".to_string(),
                    games_label: game.to_string(),
                    leg_649: l649,
                    leg_540: l540,
                    leg_jok: ljok,
                    stake_per_draw: pt.actual_cost_lei,
                });
            }
        }
    }

    // 1.C Portofolii optime mixte 3 jocuri C3-v2 la B in {40, 100, 500} x {Max-EV, Max-P_any}
    for budget in [40u32, 100, 500] {
        let mix_b: Vec<_> = c3_rep
            .mixed_3game_portfolios
            .iter()
            .filter(|m| m.budget_lei == budget)
            .collect();
        if mix_b.len() < 2 {
            return Err(format!("portofolii mixte insuficiente pentru B={budget}"));
        }
        for &(mp, reg_tag) in &[(mix_b[0], "MAX_EV"), (*mix_b.last().unwrap(), "MAX_PANY")] {
            let o1 = mp.vault_offset_649 as usize;
            let o2 = mp.vault_offset_540 as usize;
            let o3 = mp.vault_offset_joker as usize;
            let i1 = ports_649.len();
            ports_649.push(c3_tks[o1..o1 + mp.k_649].to_vec());
            let i2 = ports_540.len();
            ports_540.push(c3_tks[o2..o2 + mp.k_540].to_vec());
            let i3 = ports_jok.len();
            ports_jok.push(c3_tks[o3..o3 + mp.k_joker].to_vec());

            strat_specs.push(StrategySpec {
                strategy_id: format!("C3V2_MIX3_B{}_{}", budget, reg_tag),
                category_group: "c3v2_mixed_3game".to_string(),
                games_label: "649+540+joker".to_string(),
                leg_649: Some(i1),
                leg_540: Some(i2),
                leg_jok: Some(i3),
                stake_per_draw: mp.actual_cost_lei,
            });
        }
    }

    // 1.D Roțile garantate 100% QAP din CD-V3Q (`CDV3_649_163_QAP` și `CDV3_JOKER_QAP`)
    let md_649 = cdq_rep
        .main_designs
        .iter()
        .find(|d| d.id == "CDV3_649_163_QAP")
        .ok_or_else(|| "lipseste CDV3_649_163_QAP".to_string())?;
    let w649_tks: Vec<u64> = md_649
        .tickets_numbers
        .iter()
        .map(|v| v.iter().fold(0u64, |a, &x| a | (1u64 << (x - 1))))
        .collect();
    let idx_w649 = ports_649.len();
    let cost_w649 = (w649_tks.len() as f64) * 8.0;
    ports_649.push(w649_tks);
    strat_specs.push(StrategySpec {
        strategy_id: "CDV3_649_163_QAP".to_string(),
        category_group: "cdv3q_guaranteed_wheel".to_string(),
        games_label: "649".to_string(),
        leg_649: Some(idx_w649),
        leg_540: None,
        leg_jok: None,
        stake_per_draw: cost_w649,
    });

    let md_jok = cdq_rep
        .main_designs
        .iter()
        .find(|d| d.id == "CDV3_JOKER_QAP")
        .ok_or_else(|| "lipseste CDV3_JOKER_QAP".to_string())?;
    let wjok_tks: Vec<u64> = md_jok
        .joker_tickets
        .as_ref()
        .ok_or_else(|| "joker_tickets lipsa pe CDV3_JOKER_QAP".to_string())?
        .iter()
        .map(|x| x.packed_u64)
        .collect();
    let idx_wjok = ports_jok.len();
    let cost_wjok = (wjok_tks.len() as f64) * 7.0;
    ports_jok.push(wjok_tks);
    strat_specs.push(StrategySpec {
        strategy_id: "CDV3_JOKER_QAP".to_string(),
        category_group: "cdv3q_guaranteed_wheel".to_string(),
        games_label: "joker".to_string(),
        leg_649: None,
        leg_540: None,
        leg_jok: Some(idx_wjok),
        stake_per_draw: cost_wjok,
    });

    assert!(ports_649.len() <= 16 && ports_540.len() <= 16 && ports_jok.len() <= 16);

    let (p649, p540, pjok) = load_crowd_params_from_pmv3_report(&pm_rep)?;
    let (rep0_649, fnew_649) = extract_rep0_and_fnew(&pm_rep, "649")?;
    let (rep0_540, fnew_540) = extract_rep0_and_fnew(&pm_rep, "540")?;
    let (rep0_jok, fnew_jok) = extract_rep0_and_fnew(&pm_rep, "joker")?;
    let xi16 = build_sales_shock_xi16();

    let n_strats = strat_specs.len();
    // Alocăm bufferele exacte per traiectorie pentru toate cele 30 strategii (~944 MiB total RAM)
    let mut strat_d0_gross = vec![0.0f64; n_strats * C4V2_MC_TRAJECTORIES];
    let mut strat_1y_gross = vec![0.0f64; n_strats * C4V2_MC_TRAJECTORIES];
    let mut strat_5y_gross = vec![0.0f64; n_strats * C4V2_MC_TRAJECTORIES];
    let mut strat_10y_gross = vec![0.0f64; n_strats * C4V2_MC_TRAJECTORIES];
    let mut strat_flags = vec![0u8; n_strats * C4V2_MC_TRAJECTORIES];
    check_rss_guard(0)?;

    // Funcție helper pentru aplatizarea portofoliilor unui joc
    let flatten_ports = |ports: &[Vec<u64>]| -> (Vec<u64>, Vec<u32>, Vec<u32>) {
        let mut flat = Vec::new();
        let mut offs = Vec::with_capacity(ports.len());
        let mut lens = Vec::with_capacity(ports.len());
        for p in ports {
            offs.push(flat.len() as u32);
            lens.push(p.len() as u32);
            flat.extend_from_slice(p);
        }
        (flat, offs, lens)
    };

    // --- JOCUL 1: LOTO 6/49 (14 portofolii x 1,000,000 traiectorii x 1,040 extrageri) ---
    let (rates_649, evals_649, gpu_649_ms) = {
        let qt649 = GpuCrowdTables649::build(&gpu_univ.d_masks_649, &p649)?;
        let mut evals = Vec::with_capacity(ports_649.len());
        for p in &ports_649 {
            evals.push(gpu_c3v2_eval_649(&gpu_univ.d_masks_649, &qt649, &p649, p)?);
        }
        let (flat, offs, lens) = flatten_ports(&ports_649);
        let d_hits = gpu_c4v2_prep_hits_u32(0, &gpu_univ.d_masks_649, &flat, &offs, &lens)?;
        let rates = gpu_c4v2_compute_rates_raw(
            0,
            std::ptr::null(),
            qt649.d_q6.as_ptr(),
            qt649.d_q5.as_ptr(),
            qt649.d_q4.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            d_hits.as_ptr() as *const std::ffi::c_void,
            &p649,
            &xi16,
            ports_649.len(),
        )?;

        let np = ports_649.len();
        let mut chunk_pay = vec![0.0f64; C4V2_CHUNK_LEN * np * 4];
        let mut chunk_flg = vec![0u8; C4V2_CHUNK_LEN * np];
        let mut sim_ms = 0.0f64;

        for c0 in (0..C4V2_MC_TRAJECTORIES).step_by(C4V2_CHUNK_LEN) {
            let clen = C4V2_CHUNK_LEN.min(C4V2_MC_TRAJECTORIES - c0);
            sim_ms += gpu_c4v2_sim_chunk_raw(
                0,
                std::ptr::null(),
                qt649.d_q6.as_ptr(),
                qt649.d_q5.as_ptr(),
                qt649.d_q4.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                d_hits.as_ptr() as *const std::ffi::c_void,
                &p649,
                &xi16,
                &rep0_649,
                &fnew_649,
                np,
                c0 as u32,
                clen as u32,
                C4V2_SEED,
                &mut chunk_pay[..clen * np * 4],
                &mut chunk_flg[..clen * np],
            )?;
            for (si, sp) in strat_specs.iter().enumerate() {
                if let Some(p_idx) = sp.leg_649 {
                    let dst_base = si * C4V2_MC_TRAJECTORIES + c0;
                    for t in 0..clen {
                        let base_p = t * np + p_idx;
                        let o4 = base_p * 4;
                        strat_d0_gross[dst_base + t] += chunk_pay[o4];
                        strat_1y_gross[dst_base + t] += chunk_pay[o4 + 1];
                        strat_5y_gross[dst_base + t] += chunk_pay[o4 + 2];
                        strat_10y_gross[dst_base + t] += chunk_pay[o4 + 3];
                        strat_flags[dst_base + t] |= chunk_flg[base_p];
                    }
                }
            }
        }
        check_rss_guard(0)?;
        (rates, evals, sim_ms)
    };

    // --- JOCUL 2: LOTO 5/40 (13 portofolii x 1,000,000 traiectorii x 1,040 extrageri) ---
    let (rates_540, evals_540, gpu_540_ms) = {
        let qt540 =
            GpuCrowdTables540::build(&gpu_univ.d_tickets_540, &gpu_univ.d_draws6_540, &p540)?;
        let mut evals = Vec::with_capacity(ports_540.len());
        for p in &ports_540 {
            evals.push(gpu_c3v2_eval_540(&gpu_univ.d_draws6_540, &qt540, &p540, p)?);
        }
        let (flat, offs, lens) = flatten_ports(&ports_540);
        let d_hits = gpu_c4v2_prep_hits_u32(1, &gpu_univ.d_draws6_540, &flat, &offs, &lens)?;
        let rates = gpu_c4v2_compute_rates_raw(
            1,
            gpu_univ.d_draws6_540.as_ptr(),
            qt540.d_p5.as_ptr(),
            qt540.d_t5.as_ptr(),
            qt540.d_qiii.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            d_hits.as_ptr() as *const std::ffi::c_void,
            &p540,
            &xi16,
            ports_540.len(),
        )?;

        let np = ports_540.len();
        let mut chunk_pay = vec![0.0f64; C4V2_CHUNK_LEN * np * 4];
        let mut chunk_flg = vec![0u8; C4V2_CHUNK_LEN * np];
        let mut sim_ms = 0.0f64;

        for c0 in (0..C4V2_MC_TRAJECTORIES).step_by(C4V2_CHUNK_LEN) {
            let clen = C4V2_CHUNK_LEN.min(C4V2_MC_TRAJECTORIES - c0);
            sim_ms += gpu_c4v2_sim_chunk_raw(
                1,
                gpu_univ.d_draws6_540.as_ptr(),
                qt540.d_p5.as_ptr(),
                qt540.d_t5.as_ptr(),
                qt540.d_qiii.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                d_hits.as_ptr() as *const std::ffi::c_void,
                &p540,
                &xi16,
                &rep0_540,
                &fnew_540,
                np,
                c0 as u32,
                clen as u32,
                C4V2_SEED,
                &mut chunk_pay[..clen * np * 4],
                &mut chunk_flg[..clen * np],
            )?;
            for (si, sp) in strat_specs.iter().enumerate() {
                if let Some(p_idx) = sp.leg_540 {
                    let dst_base = si * C4V2_MC_TRAJECTORIES + c0;
                    for t in 0..clen {
                        let base_p = t * np + p_idx;
                        let o4 = base_p * 4;
                        strat_d0_gross[dst_base + t] += chunk_pay[o4];
                        strat_1y_gross[dst_base + t] += chunk_pay[o4 + 1];
                        strat_5y_gross[dst_base + t] += chunk_pay[o4 + 2];
                        strat_10y_gross[dst_base + t] += chunk_pay[o4 + 3];
                        strat_flags[dst_base + t] |= chunk_flg[base_p];
                    }
                }
            }
        }
        check_rss_guard(0)?;
        (rates, evals, sim_ms)
    };

    // --- JOCUL 3: JOKER (14 portofolii x 1,000,000 traiectorii x 1,040 extrageri) ---
    let (rates_jok, evals_jok, gpu_jok_ms) = {
        let qtjok = GpuCrowdTablesJoker::build(&gpu_univ.d_base_joker, &pjok)?;
        let mut evals = Vec::with_capacity(ports_jok.len());
        for p in &ports_jok {
            evals.push(gpu_c3v2_eval_joker(&gpu_univ.d_base_joker, &qtjok, &pjok, p)?);
        }
        let (flat, offs, lens) = flatten_ports(&ports_jok);
        let d_hits = gpu_c4v2_prep_hits_joker(&gpu_univ.d_base_joker, &flat, &offs, &lens)?;
        let rates = gpu_c4v2_compute_rates_raw(
            2,
            std::ptr::null(),
            qtjok.d_q5.as_ptr(),
            qtjok.d_q4.as_ptr(),
            qtjok.d_q3.as_ptr(),
            qtjok.d_q2.as_ptr(),
            qtjok.d_q1.as_ptr(),
            d_hits.as_ptr() as *const std::ffi::c_void,
            &pjok,
            &xi16,
            ports_jok.len(),
        )?;

        let np = ports_jok.len();
        let mut chunk_pay = vec![0.0f64; C4V2_CHUNK_LEN * np * 4];
        let mut chunk_flg = vec![0u8; C4V2_CHUNK_LEN * np];
        let mut sim_ms = 0.0f64;

        for c0 in (0..C4V2_MC_TRAJECTORIES).step_by(C4V2_CHUNK_LEN) {
            let clen = C4V2_CHUNK_LEN.min(C4V2_MC_TRAJECTORIES - c0);
            sim_ms += gpu_c4v2_sim_chunk_raw(
                2,
                std::ptr::null(),
                qtjok.d_q5.as_ptr(),
                qtjok.d_q4.as_ptr(),
                qtjok.d_q3.as_ptr(),
                qtjok.d_q2.as_ptr(),
                qtjok.d_q1.as_ptr(),
                d_hits.as_ptr() as *const std::ffi::c_void,
                &pjok,
                &xi16,
                &rep0_jok,
                &fnew_jok,
                np,
                c0 as u32,
                clen as u32,
                C4V2_SEED,
                &mut chunk_pay[..clen * np * 4],
                &mut chunk_flg[..clen * np],
            )?;
            for (si, sp) in strat_specs.iter().enumerate() {
                if let Some(p_idx) = sp.leg_jok {
                    let dst_base = si * C4V2_MC_TRAJECTORIES + c0;
                    for t in 0..clen {
                        let base_p = t * np + p_idx;
                        let o4 = base_p * 4;
                        strat_d0_gross[dst_base + t] += chunk_pay[o4];
                        strat_1y_gross[dst_base + t] += chunk_pay[o4 + 1];
                        strat_5y_gross[dst_base + t] += chunk_pay[o4 + 2];
                        strat_10y_gross[dst_base + t] += chunk_pay[o4 + 3];
                        strat_flags[dst_base + t] |= chunk_flg[base_p];
                    }
                }
            }
        }
        check_rss_guard(0)?;
        (rates, evals, sim_ms)
    };

    // 2. Agregare exactă pe 24 fire CPU (medii, SE, z-scores analitice, cuantile exacte pe 1M traiectorii)
    let m_f64 = C4V2_MC_TRAJECTORIES as f64;
    let strat_results: Vec<C4StrategyResult> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_strats);
        let mut rem_d0 = strat_d0_gross.as_mut_slice();
        let mut rem_1y = strat_1y_gross.as_mut_slice();
        let mut rem_5y = strat_5y_gross.as_mut_slice();
        let mut rem_10y = strat_10y_gross.as_mut_slice();
        let mut rem_flg = strat_flags.as_slice();

        for sp in &strat_specs {
            let (s_d0, tail_d0) = rem_d0.split_at_mut(C4V2_MC_TRAJECTORIES);
            let (s_1y, tail_1y) = rem_1y.split_at_mut(C4V2_MC_TRAJECTORIES);
            let (s_5y, tail_5y) = rem_5y.split_at_mut(C4V2_MC_TRAJECTORIES);
            let (s_10y, tail_10y) = rem_10y.split_at_mut(C4V2_MC_TRAJECTORIES);
            let (s_flg, tail_flg) = rem_flg.split_at(C4V2_MC_TRAJECTORIES);
            rem_d0 = tail_d0;
            rem_1y = tail_1y;
            rem_5y = tail_5y;
            rem_10y = tail_10y;
            rem_flg = tail_flg;

            let r649_ref = &rates_649;
            let r540_ref = &rates_540;
            let rjok_ref = &rates_jok;
            let e649_ref = &evals_649;
            let e540_ref = &evals_540;
            let ejok_ref = &evals_jok;
            let p649_v = &ports_649;
            let p540_v = &ports_540;
            let pjok_v = &ports_jok;

            handles.push(s.spawn(move || {
                // Valori analitice exacte (medie + varianță Markov dinamică) și P_any exacte pe cele 3 componente
                let mut d0_analyt = 0.0f64;
                let mut d0_c3v2 = 0.0f64;
                let mut var_d0_exact = 0.0f64;
                let mut hor_analyt = [0.0f64; 3];
                let mut hor_var_markov = [0.0f64; 3];
                let mut miss_prob = 1.0f64;

                let mut mr_649 = None;
                let mut mr_540 = None;
                let mut mr_jok = None;

                if let Some(i1) = sp.leg_649 {
                    let sub_r = &r649_ref[i1 * 48..(i1 + 1) * 48];
                    mr_649 = Some(sub_r.to_vec());
                    let ev_o = &e649_ref[i1];
                    let v0_g =
                        (ev_o.second_moment_draw0 - ev_o.ev_portfolio * ev_o.ev_portfolio).max(0.0);
                    let (d0_g, h_g, hv_g) = compute_exact_markov_horizon_moments(
                        "649",
                        sub_r,
                        &rep0_649,
                        &fnew_649,
                        v0_g,
                    );
                    d0_analyt += d0_g;
                    for h in 0..3 {
                        hor_analyt[h] += h_g[h];
                        hor_var_markov[h] += hv_g[h];
                    }
                    d0_c3v2 += ev_o.ev_portfolio;
                    var_d0_exact += v0_g;
                    let pa = (ev_o.win_states as f64) / (COUNT_649 as f64);
                    miss_prob *= (1.0 - pa).max(0.0);
                }

                if let Some(i2) = sp.leg_540 {
                    let sub_r = &r540_ref[i2 * 36..(i2 + 1) * 36];
                    mr_540 = Some(sub_r.to_vec());
                    let ev_o = &e540_ref[i2];
                    let v0_g =
                        (ev_o.second_moment_draw0 - ev_o.ev_portfolio * ev_o.ev_portfolio).max(0.0);
                    let (d0_g, h_g, hv_g) = compute_exact_markov_horizon_moments(
                        "540",
                        sub_r,
                        &rep0_540,
                        &fnew_540,
                        v0_g,
                    );
                    d0_analyt += d0_g;
                    for h in 0..3 {
                        hor_analyt[h] += h_g[h];
                        hor_var_markov[h] += hv_g[h];
                    }
                    d0_c3v2 += ev_o.ev_portfolio;
                    var_d0_exact += v0_g;
                    let pa = (ev_o.win_states as f64) / (COUNT_540_DRAWS6 as f64);
                    miss_prob *= (1.0 - pa).max(0.0);
                }

                if let Some(i3) = sp.leg_jok {
                    let sub_r = &rjok_ref[i3 * 96..(i3 + 1) * 96];
                    mr_jok = Some(sub_r.to_vec());
                    let ev_o = &ejok_ref[i3];
                    let v0_g =
                        (ev_o.second_moment_draw0 - ev_o.ev_portfolio * ev_o.ev_portfolio).max(0.0);
                    let (d0_g, h_g, hv_g) = compute_exact_markov_horizon_moments(
                        "joker",
                        sub_r,
                        &rep0_jok,
                        &fnew_jok,
                        v0_g,
                    );
                    d0_analyt += d0_g;
                    for h in 0..3 {
                        hor_analyt[h] += h_g[h];
                        hor_var_markov[h] += hv_g[h];
                    }
                    d0_c3v2 += ev_o.ev_portfolio;
                    var_d0_exact += v0_g;
                    let pa = (ev_o.win_states as f64) / (COUNT_JOKER_FULL24M as f64);
                    miss_prob *= (1.0 - pa).max(0.0);
                }

                let p_win_exact = 1.0 - miss_prob;

                // Statistici draw 0 + flaguri
                let mut win_d0_cnt = 0u64;
                let mut big_cnt = [0u64; 3];
                for &fl in s_flg {
                    if (fl & 1) != 0 {
                        win_d0_cnt += 1;
                    }
                    if (fl & 2) != 0 {
                        big_cnt[0] += 1;
                    }
                    if (fl & 4) != 0 {
                        big_cnt[1] += 1;
                    }
                    if (fl & 8) != 0 {
                        big_cnt[2] += 1;
                    }
                }
                let p_win_sim = (win_d0_cnt as f64) / m_f64;
                let p_win_se = ((p_win_exact * (1.0 - p_win_exact)).max(0.0) / m_f64).sqrt();
                let p_win_z = if p_win_se > 1.0e-15 {
                    (p_win_sim - p_win_exact) / p_win_se
                } else if (p_win_sim - p_win_exact).abs() == 0.0 {
                    0.0
                } else {
                    1.0e9
                };

                let mut sum_d0 = 0.0f64;
                for &v in s_d0.iter() {
                    sum_d0 += v;
                }
                let d0_mean_sim = sum_d0 / m_f64;
                let mut var_d0_samp = 0.0f64;
                for &v in s_d0.iter() {
                    let d = v - d0_mean_sim;
                    var_d0_samp += d * d;
                }
                let d0_se_samp = (var_d0_samp / (m_f64 * (m_f64 - 1.0))).sqrt();
                let d0_se_exact = (var_d0_exact / m_f64).sqrt();
                let d0_se_ref = d0_se_exact.max(d0_se_samp).max(1.0e-12);
                let d0_z = (d0_mean_sim - d0_analyt) / d0_se_ref;

                // Orizonturile multi-an 1y (104), 5y (520), 10y (1,040)
                let mut hor_slices = [s_1y, s_5y, s_10y];
                let mut horizons = Vec::with_capacity(3);
                let q_indices: [usize; 7] = [
                    10_000,  // q01
                    50_000,  // q05
                    250_000, // q25
                    500_000, // q50
                    750_000, // q75
                    950_000, // q95
                    990_000, // q99
                ];

                for h_idx in 0..3 {
                    let years = HORIZON_YEARS[h_idx];
                    let draws = HORIZON_DRAWS[h_idx];
                    let stake_tot = sp.stake_per_draw * (draws as f64);
                    let arr = &mut hor_slices[h_idx];

                    let mut gross_sum = 0.0f64;
                    let mut prof_cnt = 0u64;
                    for v in arr.iter_mut() {
                        gross_sum += *v;
                        let net_v = *v - stake_tot;
                        if net_v >= -1.0e-12 {
                            prof_cnt += 1;
                        }
                        *v = net_v; // transformăm in-place în profit net pentru sortarea cuantilelor exacte
                    }
                    let gross_mean_sim = gross_sum / m_f64;
                    let net_mean_sim = gross_mean_sim - stake_tot;
                    let gross_mean_an = hor_analyt[h_idx];
                    let net_mean_an = gross_mean_an - stake_tot;

                    let mut sq_diff = 0.0f64;
                    for &nv in arr.iter() {
                        let d = nv - net_mean_sim;
                        sq_diff += d * d;
                    }
                    let net_se_sim = (sq_diff / (m_f64 * (m_f64 - 1.0))).sqrt();
                    let net_se_markov = (hor_var_markov[h_idx] / m_f64).sqrt();
                    let net_se_ref = net_se_sim.max(net_se_markov).max(1.0e-12);
                    let z_hor = (net_mean_sim - net_mean_an) / net_se_ref;

                    arr.sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let quants = C4NetQuantiles {
                        q01: arr[q_indices[0]],
                        q05: arr[q_indices[1]],
                        q25: arr[q_indices[2]],
                        q50: 0.5 * (arr[q_indices[3] - 1] + arr[q_indices[3]]),
                        q75: arr[q_indices[4]],
                        q95: arr[q_indices[5]],
                        q99: arr[q_indices[6]],
                    };

                    horizons.push(C4HorizonMetrics {
                        years,
                        draws,
                        total_stake_lei: stake_tot,
                        gross_mean_sim_lei: gross_mean_sim,
                        gross_mean_analytic_lei: gross_mean_an,
                        net_mean_sim_lei: net_mean_sim,
                        net_mean_analytic_lei: net_mean_an,
                        net_per_draw_sim_lei: net_mean_sim / (draws as f64),
                        net_per_draw_analytic_lei: net_mean_an / (draws as f64),
                        net_se_sim_lei: net_se_sim,
                        net_se_markov_exact_lei: net_se_markov,
                        net_se_reference_lei: net_se_ref,
                        z_score_vs_analytic: z_hor,
                        within_4se_analytic: z_hor.abs() <= 4.0,
                        return_ratio_rho_sim: gross_mean_sim / stake_tot,
                        return_ratio_rho_analytic: gross_mean_an / stake_tot,
                        prob_net_profit_ge_0: (prof_cnt as f64) / m_f64,
                        prob_any_single_prize_ge_1000: (big_cnt[h_idx] as f64) / m_f64,
                        net_profit_quantiles_lei: quants,
                    });
                }

                let k_649 = sp.leg_649.map(|i| p649_v[i].len()).unwrap_or(0);
                let k_540 = sp.leg_540.map(|i| p540_v[i].len()).unwrap_or(0);
                let k_jok = sp.leg_jok.map(|i| pjok_v[i].len()).unwrap_or(0);
                let preview = format_ticket_preview(
                    sp.leg_649.map(|i| p649_v[i].as_slice()),
                    sp.leg_540.map(|i| p540_v[i].as_slice()),
                    sp.leg_jok.map(|i| pjok_v[i].as_slice()),
                );

                C4StrategyResult {
                    strategy_id: sp.strategy_id.clone(),
                    category_group: sp.category_group.clone(),
                    games_label: sp.games_label.clone(),
                    k_649,
                    k_540,
                    k_joker: k_jok,
                    stake_per_draw_lei: sp.stake_per_draw,
                    p_win_single_draw_exact: p_win_exact,
                    p_win_single_draw_sim: p_win_sim,
                    p_win_se,
                    p_win_z_score: p_win_z,
                    p_win_within_4se: p_win_z.abs() <= 4.0,
                    draw0_gross_mean_sim_lei: d0_mean_sim,
                    draw0_gross_mean_analytic_lei: d0_analyt,
                    draw0_gross_mean_c3v2_lei: d0_c3v2,
                    draw0_analytic_vs_c3v2_abs_diff: (d0_analyt - d0_c3v2).abs(),
                    draw0_se_sample_lei: d0_se_samp,
                    draw0_se_exact_lei: d0_se_exact,
                    draw0_z_score_exact_se: d0_z,
                    draw0_within_4se: d0_z.abs() <= 4.0,
                    markov_rates_649: mr_649,
                    markov_rates_540: mr_540,
                    markov_rates_joker: mr_jok,
                    initial_ticket_preview: preview,
                    horizons,
                }
            }));
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Eliberăm imediat cele ~944 MiB de traiectorii brute
    drop(strat_d0_gross);
    drop(strat_1y_gross);
    drop(strat_5y_gross);
    drop(strat_10y_gross);
    drop(strat_flags);

    // 3. Scriem tabelul binar `c4v2_mc1m_summary.v3bin` în Vault-ul SSD (16 x f64 = 128 octeți per record)
    let v3bin_path = vault.vault_dir.join(C4V2_V3BIN_NAME);
    make_writable_if_exists(&v3bin_path);
    let rec_cnt = (strat_results.len() * 3) as u64;
    let hdr = encode_v3bin_header_64(TABLE_KIND_C4V2_MC1M, 49, 6, rec_cnt, 128);
    let mut summary_f64: Vec<f64> = Vec::with_capacity((rec_cnt as usize) * 16);
    for sr in &strat_results {
        for hm in &sr.horizons {
            let q = &hm.net_profit_quantiles_lei;
            summary_f64.extend_from_slice(&[
                hm.years as f64,
                hm.draws as f64,
                hm.total_stake_lei,
                hm.net_mean_sim_lei,
                hm.net_se_sim_lei,
                hm.net_mean_analytic_lei,
                hm.return_ratio_rho_sim,
                hm.prob_net_profit_ge_0,
                hm.prob_any_single_prize_ge_1000,
                q.q01,
                q.q05,
                q.q25,
                q.q50,
                q.q75,
                q.q95,
                q.q99,
            ]);
        }
    }
    {
        let mut wf =
            File::create(&v3bin_path).map_err(|e| format!("create {}: {e}", v3bin_path.display()))?;
        wf.write_all(&hdr)
            .map_err(|e| format!("write hdr {}: {e}", v3bin_path.display()))?;
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                summary_f64.as_ptr() as *const u8,
                summary_f64.len() * 8,
            )
        };
        wf.write_all(raw_bytes)
            .map_err(|e| format!("write payload {}: {e}", v3bin_path.display()))?;
        wf.sync_all()
            .map_err(|e| format!("sync {}: {e}", v3bin_path.display()))?;
    }
    let (v3bin_bytes, v3bin_sha) = sha256_file_c3(&v3bin_path)?;

    upsert_vault_manifest_record(
        &vault.vault_dir,
        VaultFileRecord {
            filename: C4V2_V3BIN_NAME.to_string(),
            table_kind: TABLE_KIND_C4V2_MC1M,
            n_pool: 49,
            k_pick: 6,
            record_count: rec_cnt,
            file_bytes: v3bin_bytes,
            sha256: v3bin_sha.clone(),
        },
    )?;

    let all_d0_ok = strat_results.iter().all(|s| s.draw0_within_4se);
    let all_1y_ok = strat_results
        .iter()
        .all(|s| s.horizons[0].within_4se_analytic);
    let all_5y_10y_ok = strat_results
        .iter()
        .all(|s| s.horizons[1].within_4se_analytic && s.horizons[2].within_4se_analytic);
    let all_pwin_ok = strat_results.iter().all(|s| s.p_win_within_4se);
    let all_c3_match = strat_results
        .iter()
        .all(|s| s.draw0_analytic_vs_c3v2_abs_diff < 1.0e-11);

    let (rss_cur, rss_hwm) = check_rss_guard(0)?;
    let rss_lim = crate::cuda_ffi::MAX_SAFE_RSS_BYTES;
    let rss_ok = rss_cur < rss_lim && rss_hwm < rss_lim;

    let gate_pass = all_d0_ok
        && all_1y_ok
        && all_5y_10y_ok
        && all_pwin_ok
        && all_c3_match
        && rss_ok;

    let report = C4V2SimulationReport {
        artifact: "C4_V2_SIMULARE_1M".to_string(),
        version: "3.0.0-S3".to_string(),
        positive_claims_allowed: false,
        prng_algorithm: "Philox-4x32-10_per_trajectory_per_draw_collision_free".to_string(),
        seed: C4V2_SEED,
        trajectories_m: C4V2_MC_TRAJECTORIES,
        chunk_trajectories: C4V2_CHUNK_LEN,
        horizon_years: HORIZON_YEARS.to_vec(),
        horizon_draws: HORIZON_DRAWS.to_vec(),
        sales_shock_xi16: xi16,
        rep0_by_game: serde_json::json!({
            "649": &rep0_649[..3],
            "540": &rep0_540[..3],
            "joker": &rep0_jok[..8],
        }),
        fnew_by_game: serde_json::json!({
            "649": &fnew_649[..3],
            "540": &fnew_540[..3],
            "joker": &fnew_jok[..8],
        }),
        source_pmv3_sha256: pm_sha,
        source_cdv3q_sha256: cdq_sha,
        source_c3v2_sha256: c3_sha,
        vault_summary_v3bin: v3bin_path.display().to_string(),
        vault_summary_sha256: v3bin_sha,
        vault_summary_bytes: v3bin_bytes,
        vault_summary_records: rec_cnt,
        gpu_sim_649_ms: gpu_649_ms,
        gpu_sim_540_ms: gpu_540_ms,
        gpu_sim_joker_ms: gpu_jok_ms,
        total_physical_draws_simulated: 3u64 * (C4V2_MC_TRAJECTORIES as u64) * 1_040u64,
        all_draw0_within_4se: all_d0_ok,
        all_1y_within_4se: all_1y_ok,
        all_5y_10y_within_4se: all_5y_10y_ok,
        all_p_win_within_4se: all_pwin_ok,
        all_analytic_match_c3v2_lt_1e11: all_c3_match,
        rss_current_bytes: rss_cur,
        rss_peak_hwm_bytes: rss_hwm,
        rss_limit_bytes: rss_lim,
        rss_guard_ok: rss_ok,
        build_seconds: t0.elapsed().as_secs_f64(),
        c4v2_gate_pass: gate_pass,
        strategies: strat_results,
    };

    if let Some(p) = out_c4v2_json_path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
    }
    fs::write(
        out_c4v2_json_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .map_err(|e| format!("write {}: {e}", out_c4v2_json_path.display()))?;

    if !report.c4v2_gate_pass {
        return Err(format!(
            "c4v2_gate_pass=false (d0={all_d0_ok}, 1y={all_1y_ok}, 5y10y={all_5y_10y_ok}, pwin={all_pwin_ok}, c3match={all_c3_match}, rss={rss_ok})"
        ));
    }

    Ok(report)
}

/// Verificare read-only `c4v2-verify` (SHA-256 `.v3bin`, monotonie cuantile, recurență Markov independentă și `±4 SE`).
pub fn verify_existing_c4v2(
    vault: &MappedVaultUniverse,
    c4v2_json_path: &Path,
) -> Result<bool, String> {
    let raw = fs::read_to_string(c4v2_json_path)
        .map_err(|e| format!("read {}: {e}", c4v2_json_path.display()))?;
    let rep: C4V2SimulationReport =
        serde_json::from_str(&raw).map_err(|e| format!("parse C4_V2_SIMULARE_1M.json: {e}"))?;

    if rep.positive_claims_allowed
        || !rep.c4v2_gate_pass
        || rep.trajectories_m != C4V2_MC_TRAJECTORIES
        || !rep.all_draw0_within_4se
        || !rep.all_1y_within_4se
        || !rep.all_5y_10y_within_4se
        || !rep.all_p_win_within_4se
        || !rep.all_analytic_match_c3v2_lt_1e11
    {
        return Err("flaguri guvernanta/poarta C4-v2 invalide in JSON".to_string());
    }

    let v3bin_path = vault.vault_dir.join(C4V2_V3BIN_NAME);
    let (sz, sha) = sha256_file_c3(&v3bin_path)?;
    if sz != rep.vault_summary_bytes || sha != rep.vault_summary_sha256 {
        return Err(format!(
            "SHA-256/bytes mismatch pe {}: ({sz}, {sha}) != ({}, {})",
            C4V2_V3BIN_NAME, rep.vault_summary_bytes, rep.vault_summary_sha256
        ));
    }
    let f = File::open(&v3bin_path).map_err(|e| format!("open {}: {e}", v3bin_path.display()))?;
    let mm = unsafe { memmap2::Mmap::map(&f) }
        .map_err(|e| format!("mmap {}: {e}", v3bin_path.display()))?;
    if mm.len() != VAULT_HEADER_BYTES + (rep.vault_summary_records as usize) * 128
        || &mm[0..8] != VAULT_MAGIC
    {
        return Err("header c4v2_mc1m_summary.v3bin invalid".to_string());
    }

    for sr in &rep.strategies {
        if !sr.p_win_within_4se || !sr.draw0_within_4se {
            return Err(format!("draw0 / p_win in afara ±4 SE pe {}", sr.strategy_id));
        }
        for hm in &sr.horizons {
            if !hm.within_4se_analytic {
                return Err(format!(
                    "orizont {}y in afara ±4 SE pe {}",
                    hm.years, sr.strategy_id
                ));
            }
            let q = &hm.net_profit_quantiles_lei;
            if !(q.q01 <= q.q05
                && q.q05 <= q.q25
                && q.q25 <= q.q50
                && q.q50 <= q.q75
                && q.q75 <= q.q95
                && q.q95 <= q.q99)
            {
                return Err(format!(
                    "cuantile nemonotone la {} ({}y)",
                    sr.strategy_id, hm.years
                ));
            }
        }
    }

    check_rss_guard(0)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sales_shock_xi16_exact_unit_mean_and_monotone() {
        let xi = build_sales_shock_xi16();
        let mean: f64 = xi.iter().sum::<f64>() / 16.0;
        assert!((mean - 1.0).abs() < 1.0e-15);
        for w in xi.windows(2) {
            assert!(w[1] > w[0] && w[0] > 0.0);
        }
    }

    #[test]
    fn test_exact_markov_horizon_recurrence_identities() {
        let mut rates = [0.0f64; 48];
        rates[0] = 0.01; // psi0 cat I
        rates[1] = 0.50; // rho0 cat I
        rates[2] = 0.01; // psi_bar cat I
        rates[3] = 0.01; // eta_unit cat I
        rates[4] = 0.50; // rho_bar cat I
        rates[5] = 0.50; // beta_unit cat I
        rates[6] = 0.008; // h2_0 cat I
        rates[7] = 0.008; // h2_bar0 cat I
        rates[8] = 0.008; // h2_bar1 cat I
        rates[9] = 0.008; // h2_bar2 cat I
        rates[10] = 0.50; // gamma_unit cat I
        rates[36] = 1.25; // fixed_iv mean
        rates[42] = 900.0 * 0.02; // fixed_iv m2
        let mut rep0 = [0.0f64; 8];
        let mut fnew = [0.0f64; 8];
        rep0[0] = 1_000.0;
        fnew[0] = 500.0;
        let var_d0 = (1500.0 * 1500.0 * 0.008 - 15.0 * 15.0) + (18.0 - 1.25 * 1.25);

        let (d0, hor, hvar) =
            compute_exact_markov_horizon_moments("649", &rates, &rep0, &fnew, var_d0);
        assert!((d0 - (1500.0 * 0.01 + 1.25)).abs() < 1.0e-12);
        assert!(hor[0] > 104.0 * 1.25 && hor[2] > hor[1] && hor[1] > hor[0]);
        assert!(hvar[0] > 0.0 && hvar[2] > hvar[1] && hvar[1] > hvar[0]);
    }
}

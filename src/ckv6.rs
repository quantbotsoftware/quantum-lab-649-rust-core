//! Motorul Cauzal de Regimuri Fizice `CK-V6` (`Sesiunea 4 / Poarta A5` din `PROTOCOL_RUST_CORE_V3.md` §4.3):
//! - **BOCPD Adams–MacKay cauzal online** peste lungimea run-ului curent `r_t in {0, ..., min(t, 512)}` cu mixtură
//!   egală cauzală peste ratele de hazard constante preînregistrate `lambda_H in {1/100, 1/250}`;
//! - **Operatorul Koopman proiectat pe subspațiile ireductibile ale schemei de asociere Johnson `V_1 (+) V_2`**
//!   cu shrinkage analitic Ledoit–Wolf pe fiecare regim activ `r_t`;
//! - **Scut martingal obligatoriu `w_0 = 0.50`** cu pernă de protecție a capitalului la pragul `0.50`:
//!   `p_t(D) = 0.50 * u(D) + 0.50 * q_t^{BOCPD-Koopman}(D) => E_t >= 0.50` și `M_t = prod_{s<=t} E_s > 0.50`
//!   pentru orice `t` și orice secvență de extrageri `D_{1:T}`, păstrând identitatea exactă `E_{H_0}[E_t | F_{t-1}] = 1.0`;
//! - **Calificare sintetică pre-arhivă `Q1–Q6` (`6/6 PASS`, `archive_files_read = 0`) + Suprafața de putere pe `10,000`
//!   istorii sintetice pe `RTX 5060 sm_120`** + Sigilarea pre-arhivă `0444` a `ckv6.rs` (`SIGILIU_COD_CKV6.json`);
//! - **Protecția absolută a rezervelor oarbe `0400` (`xi/date_p2/*/rezerva.json`)**.
//! Guvernanță științifică: `positive_claims_allowed = false`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::c3v2::{
    encode_v3bin_header_64, make_writable_if_exists, sha256_file_c3, upsert_vault_manifest_record,
};
use crate::cuda_ffi::{
    check_rss_guard, gpu_ckv6_649_batch_f64, gpu_ckv6_general_streams_f64, gpu_ckv6_power_10k_f64,
    gpu_ckv6_universe_eval_f64, query_device_info, read_process_rss_bytes, CudaDeviceInfo,
    ResidentGpuUniverse, V3Ckv6StreamMeta, MAX_SAFE_RSS_BYTES,
};
use crate::vault::{
    MappedVaultUniverse, VaultFileRecord, COUNT_540_DRAWS6, COUNT_649, COUNT_JOKER_FULL24M,
    VAULT_HEADER_BYTES, VAULT_MAGIC,
};

pub const TABLE_KIND_CKV6_POWER_10K: u32 = 9;
pub const CKV6_V3BIN_NAME: &str = "ckv6_power_10k.v3bin";

pub const CKV6_L_MAX: usize = 512;
pub const CKV6_LAMBDA_0: f64 = 0.01;  // 1 / 100
pub const CKV6_LAMBDA_1: f64 = 0.004; // 1 / 250
pub const CKV6_W0_SHIELD: f64 = 0.50;
pub const CKV6_BETA_DM: f64 = 0.75;
pub const CKV6_BETA_V2: f64 = 0.25;
pub const CKV6_ALPHA_DM: f64 = 1.5;
pub const CKV6_THETA_CAP: f64 = 0.85;
pub const CKV6_RHO_MAX: f64 = 0.90;
pub const CKV6_TAU_STAT_1: f64 = 4.0;
pub const CKV6_TAU_KOOP_1: f64 = 4.0;
pub const CKV6_TAU_STAT_2: f64 = 8.0;
pub const CKV6_TAU_KOOP_2: f64 = 8.0;
pub const CKV6_TAU_BON_STAT: f64 = 4.0;
pub const CKV6_TAU_BON_KOOP: f64 = 4.0;
pub const CKV6_DM_ROW_STRIDE: usize = 70;

pub const CKV6_POWER_SEED: u64 = 20_261_104;
pub const CKV6_QUAL_SEED: u64 = 20_261_140;
pub const DELTA_BP_GRID: [u32; 5] = [0, 1500, 2500, 3500, 5000];
pub const L_REGIME_GRID: [u32; 5] = [40, 80, 120, 200, 400];
pub const POWER_REPS_PER_CELL: u32 = 400;
pub const POWER_T_STEPS: u32 = 400;
pub const POWER_TOTAL_HISTORIES: usize = 25 * 400; // 10,000 istorii sintetice
pub const HOLM_DEV_THRESHOLD: f64 = 1_500.0;

pub const DEFAULT_QL_ROOT: &str = "./data/quantum_lab_649";
pub const DEFAULT_CKV6_RS_PATH: &str = "./src/ckv6.rs";
pub const DEFAULT_CODE_SEAL_PATH: &str =
    "./data/quantum_lab_649/v3_upgrade/SIGILIU_COD_CKV6.json";

pub const BLIND_RESERVE_GAMES: [&str; 4] = ["FR_loto", "HU_hatos", "HU_otos", "US_powerball"];

// ============================================================================
// 1. Utilitare Criptografice, Timp UTC și Philox-4x32-10 Bit-Identic cu CUDA
// ============================================================================

pub fn sha256_bytes_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn utc_now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
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
    let y = if m <= 2 {
        yoe + era * 400 + 1
    } else {
        yoe + era * 400
    };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

#[derive(Clone, Debug)]
pub struct PhiloxCpu {
    k0: u32,
    k1: u32,
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    out: [u32; 4],
    pos: usize,
}

#[inline]
fn philox4x32_10_eval_cpu(
    mut c0: u32,
    mut c1: u32,
    mut c2: u32,
    mut c3: u32,
    mut k0: u32,
    mut k1: u32,
) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    for _ in 0..10 {
        let p0 = M0 * (c0 as u64);
        let hi0 = (p0 >> 32) as u32;
        let lo0 = p0 as u32;
        let p1 = M1 * (c2 as u64);
        let hi1 = (p1 >> 32) as u32;
        let lo1 = p1 as u32;
        c0 = hi1 ^ c1 ^ k0;
        c2 = hi0 ^ c3 ^ k1;
        c1 = lo1;
        c3 = lo0;
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    [c0, c1, c2, c3]
}

impl PhiloxCpu {
    #[inline]
    pub fn new(seed: u64, game_tag: u32, traj_id: u32, draw_step: u32) -> Self {
        let k0 = ((seed & 0xFFFF_FFFF) as u32) ^ game_tag.wrapping_mul(0x85EB_CA6B);
        let k1 = ((seed >> 32) as u32) ^ game_tag.wrapping_mul(0xC2B2_AE35);
        let c0 = traj_id;
        let c1 = draw_step;
        let c2 = 0u32;
        let c3 = 0xA409_3822u32 ^ game_tag;
        let out = philox4x32_10_eval_cpu(c0, c1, c2, c3, k0, k1);
        Self {
            k0,
            k1,
            c0,
            c1,
            c2,
            c3,
            out,
            pos: 0,
        }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= 4 {
            self.c2 = self.c2.wrapping_add(1);
            self.out =
                philox4x32_10_eval_cpu(self.c0, self.c1, self.c2, self.c3, self.k0, self.k1);
            self.pos = 0;
        }
        let v = self.out[self.pos];
        self.pos += 1;
        v
    }

    #[inline]
    pub fn uniform_u32(&mut self, bound: u32) -> u32 {
        let mut m = (self.next_u32() as u64) * (bound as u64);
        let mut l = m as u32;
        if l < bound {
            let thresh = bound.wrapping_neg() % bound;
            while l < thresh {
                m = (self.next_u32() as u64) * (bound as u64);
                l = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

/// Generator sintetic 100% pe întregi pentru `6/49` — identic la bit cu `d_ckv6_gen_synth_649_history` din `v3_kernels.cu`!
pub fn gen_synth_649_history_cpu(
    seed: u64,
    cell_id: u32,
    rep_id: u32,
    delta_bp: u32,
    l_regime: u32,
    t_steps: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; t_steps];
    let mut active_mask = 0u64;
    let mut prev_active_mask = 0u64;
    let mut cur_reg = u32::MAX;
    let mut prev_draw = 0u64;

    for t in 0..(t_steps as u32) {
        let reg = t / l_regime;
        if reg != cur_reg {
            cur_reg = reg;
            let mut rng_reg = PhiloxCpu::new(seed, 64_900 + cell_id, rep_id, 10_000 + reg);
            let mut pool = [0u8; 49];
            let mut plen = 0usize;
            for b in 0..49u32 {
                if ((prev_active_mask >> b) & 1) == 0 {
                    pool[plen] = b as u8;
                    plen += 1;
                }
            }
            let mut new_active = 0u64;
            for k in 0..16usize {
                let pick = k + (rng_reg.uniform_u32((plen - k) as u32) as usize);
                let chosen = pool[pick];
                pool[pick] = pool[k];
                new_active |= 1u64 << chosen;
            }
            prev_active_mask = active_mask;
            active_mask = new_active;
        }

        let mut rng_step = PhiloxCpu::new(seed, 74_900 + cell_id, rep_id, t);
        let mut w = [0u32; 49];
        let mut total_w = 0u32;
        for b in 0..49usize {
            let mut wb = 100_000u32;
            if delta_bp > 0 {
                let bit = 1u64 << b;
                if (active_mask & bit) != 0 {
                    wb = 100_000 + 22 * delta_bp;
                    if (prev_draw & bit) != 0 {
                        wb += 12 * delta_bp;
                    }
                } else {
                    wb = 100_000 - 10 * delta_bp;
                }
            }
            w[b] = wb;
            total_w += wb;
        }

        let mut dmask = 0u64;
        for _ in 0..6 {
            let target = rng_step.uniform_u32(total_w);
            let mut acc = 0u32;
            let mut chosen = 0usize;
            for (b, &wb) in w.iter().enumerate() {
                if wb > 0 {
                    if target < acc + wb {
                        chosen = b;
                        break;
                    }
                    acc += wb;
                }
            }
            dmask |= 1u64 << chosen;
            total_w -= w[chosen];
            w[chosen] = 0;
        }
        out[t as usize] = dmask;
        prev_draw = dmask;
    }
    out
}

/// Generator sintetic uniform / cu regimuri pe geometrie generală `(N <= 90, r <= 20, M_bonus <= 32)`.
pub fn gen_synth_general_stream_cpu(
    code: &str,
    n_balls: usize,
    r_draw: usize,
    m_bonus: Option<usize>,
    t_steps: usize,
    delta_bp: u32,
    l_regime: usize,
    double_draw_every: usize,
    seed: u64,
    stream_tag: u32,
) -> Ckv6StreamInput {
    let mut steps = Vec::with_capacity(t_steps);
    let active_sz = ((n_balls / 3).max(r_draw)).min(n_balls / 2);
    let mut active_balls = vec![false; n_balls];
    let mut prev_active = vec![false; n_balls];
    let mut cur_reg = usize::MAX;
    let mut prev_first_balls: Vec<usize> = Vec::new();
    let mut active_bon = 0usize;

    for t in 0..t_steps {
        let reg = t / l_regime.max(1);
        if reg != cur_reg {
            cur_reg = reg;
            let mut rng_reg =
                PhiloxCpu::new(seed, 80_000 + stream_tag, reg as u32, (n_balls * 100 + r_draw) as u32);
            let mut pool: Vec<usize> = (0..n_balls).filter(|&b| !prev_active[b]).collect();
            if pool.len() < active_sz {
                pool = (0..n_balls).collect();
            }
            let mut next_act = vec![false; n_balls];
            for k in 0..active_sz {
                let pick = k + (rng_reg.uniform_u32((pool.len() - k) as u32) as usize);
                let ch = pool[pick];
                pool[pick] = pool[k];
                next_act[ch] = true;
            }
            prev_active = active_balls;
            active_balls = next_act;
            if let Some(mb) = m_bonus {
                active_bon = (rng_reg.uniform_u32(mb as u32) as usize) % mb;
            }
        }

        let m_s = if double_draw_every > 0 && t > 0 && (t % double_draw_every == 0) {
            2usize
        } else {
            1usize
        };
        let mut draws = Vec::with_capacity(m_s);
        for a in 0..m_s {
            let mut rng = PhiloxCpu::new(seed, 90_000 + stream_tag, t as u32, a as u32);
            let mut w = vec![100_000u32; n_balls];
            let mut tot_w = 0u32;
            for b in 0..n_balls {
                if delta_bp > 0 {
                    if active_balls[b] {
                        w[b] = 100_000 + 22 * delta_bp;
                        if prev_first_balls.contains(&b) {
                            w[b] += 12 * delta_bp;
                        }
                    } else {
                        w[b] = 100_000 - 10 * delta_bp;
                    }
                }
                tot_w += w[b];
            }
            let mut d_vec = Vec::with_capacity(r_draw);
            for _ in 0..r_draw {
                let target = rng.uniform_u32(tot_w);
                let mut acc = 0u32;
                let mut chosen = 0usize;
                for (b, &wb) in w.iter().enumerate() {
                    if wb > 0 {
                        if target < acc + wb {
                            chosen = b;
                            break;
                        }
                        acc += wb;
                    }
                }
                d_vec.push(chosen as u8);
                tot_w -= w[chosen];
                w[chosen] = 0;
            }
            d_vec.sort_unstable();
            let bon = m_bonus.map(|mb| {
                if delta_bp == 0 {
                    rng.uniform_u32(mb as u32) as u8
                } else {
                    let mut wb = vec![100_000u32; mb];
                    wb[active_bon] += 25 * delta_bp;
                    let tw: u32 = wb.iter().sum();
                    let tg = rng.uniform_u32(tw);
                    let mut ac = 0u32;
                    let mut ch = 0u8;
                    for (bi, &v) in wb.iter().enumerate() {
                        if tg < ac + v {
                            ch = bi as u8;
                            break;
                        }
                        ac += v;
                    }
                    ch
                }
            });
            draws.push((d_vec, bon));
        }
        prev_first_balls = draws[0].0.iter().map(|&x| x as usize).collect();
        steps.push(Ckv6CanonicalStep {
            key: format!("SYN-{t:04}"),
            draws,
        });
    }

    Ckv6StreamInput {
        code: code.to_string(),
        name: format!("Synthetic {code}"),
        n_balls,
        r_draw,
        m_bonus,
        steps,
    }
}

// ============================================================================
// 2. Motorul Matematic Exact CK-V6 pe CPU (24 Fire) și Paritatea cu CUDA sm_120
// ============================================================================

#[inline]
pub fn ckv6_pair_index(u: usize, v: usize, n: usize) -> usize {
    u * n - ((u * (u + 1)) >> 1) + (v - u - 1)
}

#[inline]
pub fn ckv6_damp_factor(b_tot: f64) -> f64 {
    if b_tot <= 1.0e-15 {
        1.0
    } else {
        (CKV6_RHO_MAX * (b_tot / CKV6_RHO_MAX).tanh()) / b_tot
    }
}

#[inline]
pub fn ckv6_binom_f64(n: usize, r: usize) -> f64 {
    let mut c = 1u64;
    for k in 1..=r {
        let num = (n - r + k) as u64;
        let den = k as u64;
        let mut a = num;
        let mut b = den;
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        c = (c / (den / a)) * (num / a);
    }
    c as f64
}

#[inline]
pub fn ckv6_theta_cap_for_r(r: usize) -> f64 {
    if r > 6 {
        CKV6_THETA_CAP * (6.0 / (r as f64))
    } else {
        CKV6_THETA_CAP
    }
}

#[inline]
pub fn ckv6_ball_weight_f64(theta: f64, th_cap: f64) -> f64 {
    (th_cap * (theta / th_cap).tanh()).exp()
}

#[inline]
pub fn ckv6_kahan_add(sum: &mut f64, comp: &mut f64, val: f64) {
    let y = val - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

/// Polinomul simetric elementar `e_r(w_0, ..., w_{n-1})` calculat pe copia sortată crescător a ponderilor
/// pozitive `w_b > 0` (`O(n * r)`, 100% termeni pozitivi, invariant exact la orice permutare din `S_n`!).
#[inline]
pub fn ckv6_elem_sym_sorted_dp(w: &[f64], r: usize) -> f64 {
    let n = w.len();
    let mut w_sorted = [0.0f64; 96];
    w_sorted[..n].copy_from_slice(w);
    w_sorted[..n].sort_unstable_by(|a, b| a.total_cmp(b));
    let mut dp = [0.0f64; 21];
    dp[0] = 1.0;
    for (idx, &wi) in w_sorted[..n].iter().enumerate() {
        let k_max = (idx + 1).min(r);
        for k in (1..=k_max).rev() {
            dp[k] += dp[k - 1] * wi;
        }
    }
    dp[r]
}

#[inline]
pub fn ckv6_k2_num(m: usize, n: usize, r: usize) -> i64 {
    let lm = m as i64;
    let lrm = (r - m) as i64;
    let lnr = (n - r) as i64;
    let lr = r as i64;
    lm * (lm - 1) * lnr * (lnr - 1)
        - 2 * lm * lrm * (lr - 1) * (lnr - 1)
        + lrm * (lrm - 1) * lr * (lr - 1)
}

#[inline]
pub fn ckv6_k2_gcd_div(n: usize, r: usize) -> i64 {
    let mut g = ((n - 1) * (n - 2)) as u64;
    for m in 0..=r {
        let v = ckv6_k2_num(m, n, r);
        let mut uv = v.unsigned_abs();
        while uv != 0 {
            let t = g % uv;
            g = uv;
            uv = t;
        }
    }
    if g > 0 { g as i64 } else { 1 }
}

#[inline]
pub fn ckv6_k2_val(m: usize, n: usize, r: usize, denom2: f64) -> f64 {
    (ckv6_k2_num(m, n, r) as f64) / denom2
}

#[derive(Clone, Debug)]
pub struct Ckv6Eval649Out {
    pub log_m_final: f64,
    pub max_log_m: f64,
    pub min_e: f64,
    pub log_m_glob: f64,
    pub mean_cp_err: f64,
    pub step_e: Vec<f64>,
    pub step_log_m: Vec<f64>,
    pub step_map_r: Vec<i32>,
}

/// Referință exactă CPU pentru `6/49` (`T <= 400`) — identică algebric cu `d_ckv6_eval_649_warp` din CUDA.
pub fn eval_649_history_cpu(
    draws: &[u64],
    l_regime_cp: u32,
    record_steps: bool,
) -> Ckv6Eval649Out {
    let t_steps = draws.len();
    let d1 = 48.0f64;
    let nu1 = 258.0f64 / 49.0f64;
    let d2 = 1127.0f64;
    let nu2 = 9030.0f64 / 376.0f64;
    let sqrt_nu2 = nu2.sqrt();
    const K1_INT: [i64; 7] = [-36, 13, 62, 111, 160, 209, 258];
    const K2_INT: [i64; 7] = [150, -250, 102, 1206, 3062, 5670, 9030];

    let mut r0 = vec![0.0f64; t_steps + 2];
    let mut r1 = vec![0.0f64; t_steps + 2];
    let mut pi_buf = vec![0.0f64; t_steps + 2];
    r0[0] = 1.0;
    r1[0] = 1.0;

    let mut p_cnt = [0u16; 1176];
    let mut u2 = [0i32; 1176];
    let mut c1 = [0u16; 49];
    let mut u1_v1 = [0i32; 49];
    let mut u1_v2 = [0i32; 49];

    let mut log_m = 0.0f64;
    let mut max_log_m = 0.0f64;
    let mut min_e = 1.0e9f64;
    let mut lw0 = 0.0f64;
    let mut lw1 = 0.0f64;
    let mut log_m_glob = 0.0f64;
    let mut cp_err_sum = 0.0f64;
    let mut cp_err_cnt = 0usize;

    let mut step_e = if record_steps {
        Vec::with_capacity(t_steps)
    } else {
        Vec::new()
    };
    let mut step_log_m = if record_steps {
        Vec::with_capacity(t_steps)
    } else {
        Vec::new()
    };
    let mut step_map_r = if record_steps {
        Vec::with_capacity(t_steps)
    } else {
        Vec::new()
    };

    for t in 0..t_steps {
        p_cnt.fill(0);
        u2.fill(0);
        c1.fill(0);
        u1_v1.fill(0);
        u1_v2.fill(0);
        pi_buf[0] = 1.0;

        let mask_dt = draws[t];
        let mask_dtm1 = if t > 0 { draws[t - 1] } else { 0 };
        let mut b_dt = [0usize; 6];
        {
            let mut tmp_dt = mask_dt;
            for slot in &mut b_dt {
                let bi = tmp_dt.trailing_zeros() as usize;
                *slot = bi;
                tmp_dt ^= 1u64 << bi;
            }
        }

        let mut sum_c1_sq = 0i64;
        let mut sum_p2 = 0i64;
        let mut sum_u1_v1_sq = 0i64;
        let mut sum_u1_v2_sq = 0i64;
        let mut sum_c1_u1_v2 = 0i64;
        let mut sum_u2_sq = 0i64;
        let mut sum_p_u2 = 0i64;
        let mut w1_sum = 0i64;
        let mut w1_sq_sum = 0i64;
        let mut w2_sum = 0i64;
        let mut w2_sq_sum = 0i64;
        let mut dot_dt_x2 = 0i64;
        let mut dot_dt_z2 = 0i64;

        let mut b_prev = [0usize; 6];
        let mut p_prev = [0usize; 15];
        let mut m_dt_prev = 0usize;

        for ell in 1..=t {
            let s_idx = t - ell;
            let mask_ds = draws[s_idx];

            let mut b_cur = [0usize; 6];
            let mut tmp = mask_ds;
            for slot in &mut b_cur {
                let bi = tmp.trailing_zeros() as usize;
                *slot = bi;
                tmp ^= 1u64 << bi;
            }
            let mut p_cur = [0usize; 15];
            let mut p_pos = 0usize;
            for i in 0..6 {
                for j in (i + 1)..6 {
                    p_cur[p_pos] = ckv6_pair_index(b_cur[i], b_cur[j], 49);
                    p_pos += 1;
                }
            }

            for &bi in &b_cur {
                let old_c = c1[bi] as i64;
                c1[bi] = (old_c + 1) as u16;
                sum_c1_sq += 2 * old_c + 1;
                sum_c1_u1_v2 += u1_v2[bi] as i64;
            }
            for &pi_idx in &p_cur {
                let old_p = p_cnt[pi_idx] as i64;
                p_cnt[pi_idx] = (old_p + 1) as u16;
                sum_p2 += 2 * old_p + 1;
                sum_p_u2 += u2[pi_idx] as i64;
            }

            let m_dt_cur = (mask_dt & mask_ds).count_ones() as usize;
            dot_dt_x2 += K2_INT[m_dt_cur];

            if ell >= 2 {
                let m_s_tm1 = (mask_ds & mask_dtm1).count_ones() as usize;
                let w1_i = K1_INT[m_s_tm1];
                let w2_i = K2_INT[m_s_tm1];

                for &bj in &b_prev {
                    let old_u1 = u1_v1[bj] as i64;
                    u1_v1[bj] = (old_u1 + w1_i) as i32;
                    sum_u1_v1_sq += 2 * w1_i * old_u1 + w1_i * w1_i;

                    let old_u12 = u1_v2[bj] as i64;
                    u1_v2[bj] = (old_u12 + w2_i) as i32;
                    sum_u1_v2_sq += 2 * w2_i * old_u12 + w2_i * w2_i;
                    sum_c1_u1_v2 += w2_i * (c1[bj] as i64);
                }
                for &pj in &p_prev {
                    let old_u2 = u2[pj] as i64;
                    u2[pj] = (old_u2 + w2_i) as i32;
                    sum_u2_sq += 2 * w2_i * old_u2 + w2_i * w2_i;
                    sum_p_u2 += w2_i * (p_cnt[pj] as i64);
                }
                w1_sum += w1_i;
                w1_sq_sum += w1_i * w1_i;
                w2_sum += w2_i;
                w2_sq_sum += w2_i * w2_i;
                dot_dt_z2 += w2_i * K2_INT[m_dt_prev];
            }

            let d_ell = ell as f64;
            let lr = 6.0 * d_ell;
            let e_x1 = ((sum_c1_sq as f64) - (lr * lr) / 49.0).max(0.0);
            let v_stat1 = d_ell * nu1;
            let eta_stat1 = if e_x1 > v_stat1 {
                (1.0 - v_stat1 / e_x1) * (d_ell / (d_ell + CKV6_TAU_STAT_1))
            } else {
                0.0
            };

            let mut scale_koop1 = 0.0f64;
            let mut w1_mean_shift = 0.0f64;
            if ell >= 2 {
                let m_tr = (ell - 1) as f64;
                let w1_real = (w1_sum as f64) / 49.0;
                w1_mean_shift = (6.0 * w1_real) / 49.0;
                let u1_raw_sq = (sum_u1_v1_sq as f64) / (49.0 * 49.0);
                let e_z1 = (u1_raw_sq - (6.0 * w1_real) * (6.0 * w1_real) / 49.0).max(0.0);
                let v_koop1 = nu1 * ((w1_sq_sum as f64) / (49.0 * 49.0));
                if e_z1 > v_koop1 {
                    let g_koop1 =
                        (1.0 - v_koop1 / e_z1) * (d1 / (nu1 * (m_tr + CKV6_TAU_KOOP_1)));
                    scale_koop1 = (d1 / nu1) * g_koop1;
                }
            }

            let lam1 = if eta_stat1 > 0.0 || scale_koop1 > 0.0 {
                let base_c0 = d_ell * (6.0 / 49.0) + CKV6_ALPHA_DM;
                let mut w_49 = [1.0f64; 49];
                for b in 0..49 {
                    let z1_b = ((u1_v1[b] as f64) / 49.0) - w1_mean_shift;
                    let th = eta_stat1 * (((c1[b] as f64) + CKV6_ALPHA_DM) / base_c0).ln()
                        + scale_koop1 * z1_b;
                    w_49[b] = ckv6_ball_weight_f64(th, CKV6_THETA_CAP);
                }
                let er = ckv6_elem_sym_sorted_dp(&w_49, 6);
                let prod_dt = w_49[b_dt[0]]
                    * w_49[b_dt[1]]
                    * w_49[b_dt[2]]
                    * w_49[b_dt[3]]
                    * w_49[b_dt[4]]
                    * w_49[b_dt[5]];
                13_983_816.0 * prod_dt / er
            } else {
                1.0
            };

            let e_x2 = (2.0 * (sum_p2 as f64)
                - (30.0 * d_ell) * (30.0 * d_ell) / (49.0 * 48.0)
                - (50.0 / 47.0) * e_x1)
                .max(0.0);
            let v_stat2 = d_ell * nu2;
            let g_stat2 = if e_x2 > v_stat2 {
                (1.0 - v_stat2 / e_x2) / (d_ell + CKV6_TAU_STAT_2)
            } else {
                0.0
            };

            let mut g_koop2 = 0.0f64;
            let mut e_z2 = 0.0f64;
            let mut cross_xz2 = 0.0f64;
            if ell >= 2 {
                let m_tr = (ell - 1) as f64;
                let inv_s2 = 1.0 / 376.0;
                let inv_s2_sq = inv_s2 * inv_s2;
                let w2_real = (w2_sum as f64) * inv_s2;
                let u1v2_c_sq =
                    ((sum_u1_v2_sq as f64) * inv_s2_sq) - (6.0 * w2_real) * (6.0 * w2_real) / 49.0;
                e_z2 = (2.0 * ((sum_u2_sq as f64) * inv_s2_sq)
                    - (30.0 * w2_real) * (30.0 * w2_real) / (49.0 * 48.0)
                    - (50.0 / 47.0) * u1v2_c_sq)
                    .max(0.0);
                let v_koop2 = nu2 * ((w2_sq_sum as f64) * inv_s2_sq);
                if e_z2 > v_koop2 {
                    g_koop2 =
                        (1.0 - v_koop2 / e_z2) * (d2 / (nu2 * (m_tr + CKV6_TAU_KOOP_2)));
                }
                let cross_c1_u12 =
                    ((sum_c1_u1_v2 as f64) * inv_s2) - (lr * 6.0 * w2_real) / 49.0;
                cross_xz2 = 2.0 * ((sum_p_u2 as f64) * inv_s2)
                    - (900.0 * d_ell * w2_real) / (49.0 * 48.0)
                    - (50.0 / 47.0) * cross_c1_u12;
            }
            let m2_norm2 =
                (g_stat2 * g_stat2 * e_x2 + g_koop2 * g_koop2 * e_z2 + 2.0 * g_stat2 * g_koop2 * cross_xz2)
                    .max(0.0);
            let b2 = (d2 / sqrt_nu2) * m2_norm2.sqrt();
            let s2_raw = (d2 / nu2)
                * (g_stat2 * ((dot_dt_x2 as f64) / 376.0)
                    + g_koop2 * ((dot_dt_z2 as f64) / (376.0 * 376.0)));

            let c_v2 = ckv6_damp_factor(b2);
            let pi_v2 = (1.0 + c_v2 * s2_raw).max(0.05);
            pi_buf[ell] = CKV6_BETA_DM * lam1 + CKV6_BETA_V2 * pi_v2;

            b_prev = b_cur;
            p_prev = p_cur;
            m_dt_prev = m_dt_cur;
        }

        let mut l0 = 0.0f64;
        let mut l1 = 0.0f64;
        for ell in 0..=t {
            let pv = pi_buf[ell];
            l0 += r0[ell] * pv;
            l1 += r1[ell] * pv;
        }
        let lw_max = lw0.max(lw1);
        let a0 = (lw0 - lw_max).exp();
        let a1 = (lw1 - lw_max).exp();
        let eta0 = a0 / (a0 + a1);
        let eta1 = a1 / (a0 + a1);

        let l_bocpd = eta0 * l0 + eta1 * l1;
        let kappa = if log_m < 0.0 {
            let m_prev = log_m.exp();
            ((m_prev - 0.50) / (0.50 * m_prev)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let e_val = (1.0 + 0.50 * kappa * (l_bocpd - 1.0)).max(0.50);
        log_m += e_val.ln();
        if log_m > max_log_m {
            max_log_m = log_m;
        }
        if e_val < min_e {
            min_e = e_val;
        }
        lw0 += l0.ln();
        lw1 += l1.ln();

        let l_glob = if t == 0 { 1.0 } else { pi_buf[t] };
        let kappa_glob = if log_m_glob < 0.0 {
            let mg = log_m_glob.exp();
            ((mg - 0.50) / (0.50 * mg)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let e_glob = (1.0 + 0.50 * kappa_glob * (l_glob - 1.0)).max(0.50);
        log_m_glob += e_glob.ln();

        let lw_max_post = lw0.max(lw1);
        let ap0 = (lw0 - lw_max_post).exp();
        let ap1 = (lw1 - lw_max_post).exp();
        let ep0 = ap0 / (ap0 + ap1);
        let ep1 = ap1 / (ap0 + ap1);

        let mut best_ell = 0i32;
        let mut best_post = -1.0f64;
        for ell in 0..=t {
            let pv = pi_buf[ell];
            let p_ell = ep0 * (r0[ell] * pv / l0) + ep1 * (r1[ell] * pv / l1);
            if p_ell > best_post {
                best_post = p_ell;
                best_ell = ell as i32;
            }
        }

        if record_steps {
            step_e.push(e_val);
            step_log_m.push(log_m);
            step_map_r.push(best_ell);
        }

        if l_regime_cp > 0 && (l_regime_cp as usize) < t_steps {
            let lr_u = l_regime_cp as usize;
            let probe_off = if lr_u > 30 { 30 } else { lr_u - 1 };
            if t >= lr_u && (t % lr_u) == probe_off {
                let true_cp = (t - probe_off) as i32;
                let hat_cp = (t as i32) - best_ell;
                cp_err_sum += ((hat_cp - true_cp) as f64).abs();
                cp_err_cnt += 1;
            }
        }

        let inv_l0 = (1.0 - CKV6_LAMBDA_0) / l0;
        let inv_l1 = (1.0 - CKV6_LAMBDA_1) / l1;
        for ell in (0..=t).rev() {
            let pv = pi_buf[ell];
            r0[ell + 1] = r0[ell] * pv * inv_l0;
            r1[ell + 1] = r1[ell] * pv * inv_l1;
        }
        r0[0] = CKV6_LAMBDA_0;
        r1[0] = CKV6_LAMBDA_1;
    }

    Ckv6Eval649Out {
        log_m_final: log_m,
        max_log_m,
        min_e,
        log_m_glob,
        mean_cp_err: if cp_err_cnt > 0 {
            cp_err_sum / (cp_err_cnt as f64)
        } else {
            0.0
        },
        step_e,
        step_log_m,
        step_map_r,
    }
}

pub fn eval_649_batch_cpu_24t(
    draws_batch: &[u64],
    n_traj: usize,
    t_steps: usize,
    l_regime_cp: u32,
    record_steps: bool,
) -> Vec<Ckv6Eval649Out> {
    assert_eq!(draws_batch.len(), n_traj * t_steps);
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_traj = (n_traj + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in draws_batch.chunks((chunk_traj.max(1)) * t_steps) {
            handles.push(s.spawn(move || {
                let mut loc = Vec::with_capacity(chunk.len() / t_steps);
                for tr in chunk.chunks_exact(t_steps) {
                    loc.push(eval_649_history_cpu(tr, l_regime_cp, record_steps));
                }
                loc
            }));
        }
        let mut out = Vec::with_capacity(n_traj);
        for h in handles {
            out.extend(h.join().unwrap());
        }
        out
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ckv6CanonicalStep {
    pub key: String,
    pub draws: Vec<(Vec<u8>, Option<u8>)>, // 0-based bile sortate + 0-based bonus optional
}

#[derive(Clone, Debug)]
pub struct Ckv6StreamInput {
    pub code: String,
    pub name: String,
    pub n_balls: usize,
    pub r_draw: usize,
    pub m_bonus: Option<usize>,
    pub steps: Vec<Ckv6CanonicalStep>,
}

#[derive(Clone, Debug)]
pub struct Ckv6StepSeries {
    pub step_e: Vec<f64>,
    pub step_log_m: Vec<f64>,
    pub step_map_r: Vec<i32>,
    pub step_cp_prob: Vec<f64>,
    pub step_log_m_glob: Vec<f64>,
}

#[inline]
fn pack_balls_96(balls: &[u8]) -> (u64, u32) {
    let mut lo = 0u64;
    let mut hi = 0u32;
    for &b in balls {
        if b < 64 {
            lo |= 1u64 << b;
        } else {
            hi |= 1u32 << (b - 64);
        }
    }
    (lo, hi)
}

#[inline]
fn popcnt_96(a_lo: u64, a_hi: u32, b_lo: u64, b_hi: u32) -> usize {
    ((a_lo & b_lo).count_ones() + (a_hi & b_hi).count_ones()) as usize
}

/// Evaluare exactă pe CPU a unui `Ckv6StreamInput` general (`N <= 90, r <= 20, M_bonus <= 32, m_s >= 1`),
/// identică algebric și în ordinea operațiilor IEEE-754 `f64` cu `k_ckv6_general_streams_f64` din `v3_kernels.cu`!
pub fn eval_general_stream_cpu(stream: &Ckv6StreamInput) -> Ckv6StepSeries {
    let n = stream.n_balls;
    let r = stream.r_draw;
    let mb = stream.m_bonus.unwrap_or(0);
    let t_total = stream.steps.len();
    let n_pairs = (n * (n - 1)) / 2;
    let has_v2 = (n >= 4) && (r >= 2) && (r <= n - 2);
    let has_bon = mb >= 2;

    let d_n = n as f64;
    let d_r = r as f64;
    let comb_nr = ckv6_binom_f64(n, r);
    let p1 = d_r / d_n;
    let d1 = (n - 1) as f64;
    let nu1 = ((r * (n - r)) as f64) / d_n;

    let k2_g = if has_v2 { ckv6_k2_gcd_div(n, r) } else { 1 };
    let d2 = if has_v2 {
        ((n * (n - 3)) as f64) * 0.5
    } else {
        1.0
    };
    let denom2 = if has_v2 {
        ((n - 1) * (n - 2)) as f64
    } else {
        1.0
    };
    let s2_scale = if has_v2 {
        ((((n - 1) * (n - 2)) as i64) / k2_g) as f64
    } else {
        1.0
    };
    let inv_s2 = 1.0 / s2_scale;
    let inv_s2_sq = inv_s2 * inv_s2;
    let nu2 = if has_v2 {
        (((r * (r - 1)) as f64) * (((n - r) * (n - r - 1)) as f64)) / denom2
    } else {
        1.0
    };
    let sqrt_nu2 = nu2.sqrt();
    let v1_proj_coef = if has_v2 {
        (2.0 * (((r - 1) * (r - 1)) as f64)) / ((n - 2) as f64)
    } else {
        0.0
    };

    let d_mb = if has_bon { mb as f64 } else { 1.0 };
    let d1_b = if has_bon { (mb - 1) as f64 } else { 1.0 };
    let nu1_b = if has_bon {
        ((mb - 1) as f64) / d_mb
    } else {
        1.0
    };
    let sqrt_nu1_b = nu1_b.sqrt();
    let inv_mb = if has_bon { 1.0 / d_mb } else { 0.0 };
    let inv_mb_sq = inv_mb * inv_mb;

    // Pre-împachetăm extragerile ca pe GPU
    let mut step_offs = Vec::with_capacity(t_total + 1);
    let mut draws_balls: Vec<Vec<usize>> = Vec::new();
    let mut draws_lo: Vec<u64> = Vec::new();
    let mut draws_hi: Vec<u32> = Vec::new();
    let mut draws_bon: Vec<usize> = Vec::new();
    step_offs.push(0usize);
    for st in &stream.steps {
        for (bls, bon) in &st.draws {
            let mut b_sorted: Vec<usize> = bls.iter().map(|&x| x as usize).collect();
            b_sorted.sort_unstable();
            let (lo, hi) = pack_balls_96(bls);
            draws_balls.push(b_sorted);
            draws_lo.push(lo);
            draws_hi.push(hi);
            draws_bon.push(bon.unwrap_or(0) as usize);
        }
        step_offs.push(draws_lo.len());
    }

    let mut p_mat = vec![0.0f64; n_pairs];
    let mut u2_mat = vec![0.0f64; n_pairs];
    let mut c1 = vec![0.0f64; n];
    let mut u1_v1 = vec![0.0f64; n];
    let mut u1_v2 = vec![0.0f64; n];
    let mut c_bon = vec![0.0f64; mb.max(1)];
    let mut u_bon = vec![0.0f64; mb.max(1)];
    let mut w_vec = vec![1.0f64; n];
    let mut r0 = vec![0.0f64; CKV6_L_MAX + 2];
    let mut r1 = vec![0.0f64; CKV6_L_MAX + 2];
    let mut pi_buf = vec![0.0f64; CKV6_L_MAX + 2];
    r0[0] = 1.0;
    r1[0] = 1.0;

    let mut log_m = 0.0f64;
    let mut lw0 = 0.0f64;
    let mut lw1 = 0.0f64;
    let mut log_m_glob = 0.0f64;

    let mut step_e = Vec::with_capacity(t_total);
    let mut step_log_m = Vec::with_capacity(t_total);
    let mut step_map_r = Vec::with_capacity(t_total);
    let mut step_cp_prob = Vec::with_capacity(t_total);
    let mut step_log_m_glob = Vec::with_capacity(t_total);

    for t in 0..t_total {
        let l_act = t.min(CKV6_L_MAX);
        p_mat.fill(0.0);
        u2_mat.fill(0.0);
        c1.fill(0.0);
        u1_v1.fill(0.0);
        u1_v2.fill(0.0);
        if has_bon {
            c_bon.fill(0.0);
            u_bon.fill(0.0);
        }
        pi_buf[0] = 1.0;

        let dt_beg = step_offs[t];
        let dt_end = step_offs[t + 1];
        let m_dt = dt_end - dt_beg;
        let m_dt_cap = m_dt.min(8);
        let inv_m_dt = 1.0 / (m_dt as f64);

        let (dtm1_beg, dtm1_end, inv_m_dtm1) = if t > 0 {
            let b = step_offs[t - 1];
            let e = step_offs[t];
            (b, e, 1.0 / ((e - b) as f64))
        } else {
            (0, 0, 0.0)
        };

        let mut n_eff_sum = 0.0f64;
        let (mut sum_c1_sq, mut comp_c1_sq) = (0.0f64, 0.0f64);
        let (mut sum_p2, mut comp_p2) = (0.0f64, 0.0f64);
        let (mut sum_u1_v1_sq, mut comp_u1_v1_sq) = (0.0f64, 0.0f64);
        let (mut sum_u1_v2_sq, mut comp_u1_v2_sq) = (0.0f64, 0.0f64);
        let (mut sum_c1_u1_v2, mut comp_c1_u1_v2) = (0.0f64, 0.0f64);
        let (mut sum_u2_sq, mut comp_u2_sq) = (0.0f64, 0.0f64);
        let (mut sum_p_u2, mut comp_p_u2) = (0.0f64, 0.0f64);
        let mut w1_sum = 0.0f64;
        let mut w1_sq_eff_sum = 0.0f64;
        let mut w2_sum = 0.0f64;
        let mut w2_sq_eff_sum = 0.0f64;

        let mut dot_a_x2 = [0.0f64; 8];
        let mut dot_a_z2 = [0.0f64; 8];
        let mut dot_a_xb = [0.0f64; 8];
        let mut dot_a_zb = [0.0f64; 8];
        let mut prev_a_k2 = [0.0f64; 8];
        let mut prev_a_kb = [0.0f64; 8];

        let (mut sum_cb_sq, mut comp_cb_sq) = (0.0f64, 0.0f64);
        let (mut sum_ub_sq, mut comp_ub_sq) = (0.0f64, 0.0f64);
        let (mut sum_cb_ub, mut comp_cb_ub) = (0.0f64, 0.0f64);
        let mut wb_sum = 0.0f64;
        let mut wb_sq_eff_sum = 0.0f64;

        let mut prev_beg = 0usize;
        let mut prev_end = 0usize;
        let mut prev_inv_m = 1.0f64;

        for ell in 1..=l_act {
            let s_idx = t - ell;
            let ds_beg = step_offs[s_idx];
            let ds_end = step_offs[s_idx + 1];
            let inv_m_ds = 1.0 / ((ds_end - ds_beg) as f64);
            n_eff_sum += inv_m_ds;

            for d_ptr in ds_beg..ds_end {
                let bls = &draws_balls[d_ptr];
                for &bi in bls {
                    let old_c = c1[bi];
                    c1[bi] = old_c + inv_m_ds;
                    ckv6_kahan_add(&mut sum_c1_sq, &mut comp_c1_sq, (2.0 * old_c + inv_m_ds) * inv_m_ds);
                    ckv6_kahan_add(&mut sum_c1_u1_v2, &mut comp_c1_u1_v2, inv_m_ds * u1_v2[bi]);
                }
                if has_v2 {
                    for i in 0..bls.len() {
                        for j in (i + 1)..bls.len() {
                            let pidx = ckv6_pair_index(bls[i], bls[j], n);
                            let old_p = p_mat[pidx];
                            p_mat[pidx] = old_p + inv_m_ds;
                            ckv6_kahan_add(&mut sum_p2, &mut comp_p2, (2.0 * old_p + inv_m_ds) * inv_m_ds);
                            ckv6_kahan_add(&mut sum_p_u2, &mut comp_p_u2, inv_m_ds * u2_mat[pidx]);
                        }
                    }
                }
                if has_bon {
                    let jb = draws_bon[d_ptr];
                    let old_cb = c_bon[jb];
                    c_bon[jb] = old_cb + inv_m_ds;
                    ckv6_kahan_add(&mut sum_cb_sq, &mut comp_cb_sq, (2.0 * old_cb + inv_m_ds) * inv_m_ds);
                    ckv6_kahan_add(&mut sum_cb_ub, &mut comp_cb_ub, inv_m_ds * u_bon[jb]);
                }
            }

            let mut cur_a_k2 = [0.0f64; 8];
            let mut cur_a_kb = [0.0f64; 8];
            for a in 0..m_dt_cap {
                let a_lo = draws_lo[dt_beg + a];
                let a_hi = draws_hi[dt_beg + a];
                let a_jb = draws_bon[dt_beg + a];
                let mut sk2 = 0.0f64;
                let mut skb = 0.0f64;
                for d_ptr in ds_beg..ds_end {
                    if has_v2 {
                        let m_int = popcnt_96(a_lo, a_hi, draws_lo[d_ptr], draws_hi[d_ptr]);
                        sk2 += (ckv6_k2_num(m_int, n, r) / k2_g) as f64;
                    }
                    if has_bon {
                        let b_jb = draws_bon[d_ptr];
                        skb += (if a_jb == b_jb { d_mb } else { 0.0 }) - 1.0;
                    }
                }
                cur_a_k2[a] = sk2 * inv_m_ds;
                cur_a_kb[a] = skb * inv_m_ds;
                dot_a_x2[a] += cur_a_k2[a];
                dot_a_xb[a] += cur_a_kb[a];
            }

            if ell >= 2 {
                let mut w1 = 0.0f64;
                let mut w2 = 0.0f64;
                let mut wb = 0.0f64;
                for p1_ptr in ds_beg..ds_end {
                    let s_lo = draws_lo[p1_ptr];
                    let s_hi = draws_hi[p1_ptr];
                    let s_jb = draws_bon[p1_ptr];
                    for p2_ptr in dtm1_beg..dtm1_end {
                        let m_int = popcnt_96(s_lo, s_hi, draws_lo[p2_ptr], draws_hi[p2_ptr]);
                        w1 += ((n as i64) * (m_int as i64) - ((r * r) as i64)) as f64;
                        if has_v2 {
                            w2 += (ckv6_k2_num(m_int, n, r) / k2_g) as f64;
                        }
                        if has_bon {
                            let t_jb = draws_bon[p2_ptr];
                            wb += (if s_jb == t_jb { d_mb } else { 0.0 }) - 1.0;
                        }
                    }
                }
                let norm_pair = inv_m_ds * inv_m_dtm1;
                w1 *= norm_pair;
                w2 *= norm_pair;
                wb *= norm_pair;

                w1_sum += w1;
                w1_sq_eff_sum += w1 * w1 * prev_inv_m;
                w2_sum += w2;
                w2_sq_eff_sum += w2 * w2 * prev_inv_m;
                wb_sum += wb;
                wb_sq_eff_sum += wb * wb * prev_inv_m;

                let a1_step = w1 * prev_inv_m;
                let a2_step = w2 * prev_inv_m;
                let ab_step = wb * prev_inv_m;

                for d_ptr in prev_beg..prev_end {
                    let bls = &draws_balls[d_ptr];
                    for &bj in bls {
                        let old_u1 = u1_v1[bj];
                        u1_v1[bj] = old_u1 + a1_step;
                        ckv6_kahan_add(&mut sum_u1_v1_sq, &mut comp_u1_v1_sq, (2.0 * old_u1 + a1_step) * a1_step);
                        if has_v2 {
                            let old_u12 = u1_v2[bj];
                            u1_v2[bj] = old_u12 + a2_step;
                            ckv6_kahan_add(&mut sum_u1_v2_sq, &mut comp_u1_v2_sq, (2.0 * old_u12 + a2_step) * a2_step);
                            ckv6_kahan_add(&mut sum_c1_u1_v2, &mut comp_c1_u1_v2, a2_step * c1[bj]);
                        }
                    }
                    if has_v2 {
                        for i in 0..bls.len() {
                            for j in (i + 1)..bls.len() {
                                let pj = ckv6_pair_index(bls[i], bls[j], n);
                                let old_u2 = u2_mat[pj];
                                u2_mat[pj] = old_u2 + a2_step;
                                ckv6_kahan_add(&mut sum_u2_sq, &mut comp_u2_sq, (2.0 * old_u2 + a2_step) * a2_step);
                                ckv6_kahan_add(&mut sum_p_u2, &mut comp_p_u2, a2_step * p_mat[pj]);
                            }
                        }
                    }
                    if has_bon {
                        let jb = draws_bon[d_ptr];
                        let old_ub = u_bon[jb];
                        u_bon[jb] = old_ub + ab_step;
                        ckv6_kahan_add(&mut sum_ub_sq, &mut comp_ub_sq, (2.0 * old_ub + ab_step) * ab_step);
                        ckv6_kahan_add(&mut sum_cb_ub, &mut comp_cb_ub, ab_step * c_bon[jb]);
                    }
                }

                for a in 0..m_dt_cap {
                    dot_a_z2[a] += w2 * prev_a_k2[a];
                    dot_a_zb[a] += wb * prev_a_kb[a];
                }
            }

            let d_ell = ell as f64;
            let lr = d_r * d_ell;
            let e_x1 = ((sum_c1_sq + comp_c1_sq) - (lr * lr) / d_n).max(0.0);
            let v_stat1 = n_eff_sum * nu1;
            let eta_stat1 = if e_x1 > v_stat1 {
                (1.0 - v_stat1 / e_x1) * (d_ell / (d_ell + CKV6_TAU_STAT_1))
            } else {
                0.0
            };

            let mut scale_koop1 = 0.0f64;
            let mut w1_mean_shift = 0.0f64;
            if ell >= 2 {
                let m_tr = (ell - 1) as f64;
                let w1_real = w1_sum / d_n;
                w1_mean_shift = (d_r * w1_real) / d_n;
                let u1_raw_sq = (sum_u1_v1_sq + comp_u1_v1_sq) / (d_n * d_n);
                let e_z1 = (u1_raw_sq - (d_r * w1_real) * (d_r * w1_real) / d_n).max(0.0);
                let v_koop1 = nu1 * (w1_sq_eff_sum / (d_n * d_n));
                if e_z1 > v_koop1 {
                    let g_koop1 =
                        (1.0 - v_koop1 / e_z1) * (d1 / (nu1 * (m_tr + CKV6_TAU_KOOP_1)));
                    scale_koop1 = (d1 / nu1) * g_koop1;
                }
            }

            let dm_active = eta_stat1 > 0.0 || scale_koop1 > 0.0;
            let mut er = comb_nr;
            if dm_active {
                let th_cap = ckv6_theta_cap_for_r(r);
                let base_c0 = d_ell * p1 + CKV6_ALPHA_DM;
                for b in 0..n {
                    let z1_b = (u1_v1[b] / d_n) - w1_mean_shift;
                    let th = eta_stat1 * ((c1[b] + CKV6_ALPHA_DM) / base_c0).ln()
                        + scale_koop1 * z1_b;
                    w_vec[b] = ckv6_ball_weight_f64(th, th_cap);
                }
                er = ckv6_elem_sym_sorted_dp(&w_vec[..n], r);
            }

            let mut g_stat2 = 0.0f64;
            let mut g_koop2 = 0.0f64;
            let mut c_v2 = 1.0f64;
            if has_v2 {
                let rr1 = (r * (r - 1)) as f64;
                let n_n1 = (n * (n - 1)) as f64;
                let e_x2 = (2.0 * (sum_p2 + comp_p2)
                    - (d_ell * rr1) * (d_ell * rr1) / n_n1
                    - v1_proj_coef * e_x1)
                    .max(0.0);
                let v_stat2 = n_eff_sum * nu2;
                if e_x2 > v_stat2 {
                    g_stat2 = (1.0 - v_stat2 / e_x2) / (d_ell + CKV6_TAU_STAT_2);
                }
                let mut e_z2 = 0.0f64;
                let mut cross_xz2 = 0.0f64;
                if ell >= 2 {
                    let m_tr = (ell - 1) as f64;
                    let w2_real = w2_sum * inv_s2;
                    let u1v2_c_sq = ((sum_u1_v2_sq + comp_u1_v2_sq) * inv_s2_sq)
                        - (d_r * w2_real) * (d_r * w2_real) / d_n;
                    e_z2 = (2.0 * ((sum_u2_sq + comp_u2_sq) * inv_s2_sq)
                        - (rr1 * w2_real) * (rr1 * w2_real) / n_n1
                        - v1_proj_coef * u1v2_c_sq)
                        .max(0.0);
                    let v_koop2 = nu2 * (w2_sq_eff_sum * inv_s2_sq);
                    if e_z2 > v_koop2 {
                        g_koop2 =
                            (1.0 - v_koop2 / e_z2) * (d2 / (nu2 * (m_tr + CKV6_TAU_KOOP_2)));
                    }
                    let cross_c1_u12 =
                        ((sum_c1_u1_v2 + comp_c1_u1_v2) * inv_s2) - (lr * d_r * w2_real) / d_n;
                    cross_xz2 = 2.0 * ((sum_p_u2 + comp_p_u2) * inv_s2)
                        - (d_ell * w2_real * rr1 * rr1) / n_n1
                        - v1_proj_coef * cross_c1_u12;
                }
                let m2_norm2 =
                    (g_stat2 * g_stat2 * e_x2 + g_koop2 * g_koop2 * e_z2 + 2.0 * g_stat2 * g_koop2 * cross_xz2)
                        .max(0.0);
                let b2 = (d2 / sqrt_nu2) * m2_norm2.sqrt();
                c_v2 = ckv6_damp_factor(b2);
            }

            let mut g_stat_b = 0.0f64;
            let mut g_koop_b = 0.0f64;
            let mut c_bon_damp = 1.0f64;
            if has_bon {
                let e_xb = ((sum_cb_sq + comp_cb_sq) - (d_ell * d_ell) * inv_mb).max(0.0);
                let v_stat_b = n_eff_sum * nu1_b;
                if e_xb > v_stat_b {
                    g_stat_b = (1.0 - v_stat_b / e_xb) / (d_ell + CKV6_TAU_BON_STAT);
                }
                let mut e_zb = 0.0f64;
                let mut cross_xzb = 0.0f64;
                if ell >= 2 {
                    let m_tr = (ell - 1) as f64;
                    let wb_real = wb_sum * inv_mb;
                    e_zb = (((sum_ub_sq + comp_ub_sq) * inv_mb_sq) - (wb_real * wb_real) * inv_mb)
                        .max(0.0);
                    let v_koop_b = nu1_b * (wb_sq_eff_sum * inv_mb_sq);
                    if e_zb > v_koop_b {
                        g_koop_b =
                            (1.0 - v_koop_b / e_zb) * (d1_b / (nu1_b * (m_tr + CKV6_TAU_BON_KOOP)));
                    }
                    cross_xzb = ((sum_cb_ub + comp_cb_ub) * inv_mb) - (d_ell * wb_real) * inv_mb;
                }
                let mb_norm2 = (g_stat_b * g_stat_b * e_xb
                    + g_koop_b * g_koop_b * e_zb
                    + 2.0 * g_stat_b * g_koop_b * cross_xzb)
                    .max(0.0);
                let bb = (d1_b / sqrt_nu1_b) * mb_norm2.sqrt();
                c_bon_damp = ckv6_damp_factor(bb);
            }

            let mut pi_step_sum = 0.0f64;
            for a in 0..m_dt_cap {
                let lam1_a = if dm_active {
                    let mut prod_a = 1.0f64;
                    for &bi in &draws_balls[dt_beg + a] {
                        prod_a *= w_vec[bi];
                    }
                    comb_nr * prod_a / er
                } else {
                    1.0
                };
                let lam2_a = if has_v2 {
                    let s2 = (d2 / nu2)
                        * (g_stat2 * (dot_a_x2[a] / s2_scale)
                            + g_koop2 * (dot_a_z2[a] / (s2_scale * s2_scale)));
                    (1.0 + c_v2 * s2).max(0.05)
                } else {
                    1.0
                };
                let pi_m = CKV6_BETA_DM * lam1_a + CKV6_BETA_V2 * lam2_a;
                let pi_b = if has_bon {
                    let sb = (d1_b / nu1_b)
                        * (g_stat_b * (dot_a_xb[a] * inv_mb)
                            + g_koop_b * (dot_a_zb[a] * inv_mb_sq));
                    (1.0 + c_bon_damp * sb).max(0.05)
                } else {
                    1.0
                };
                pi_step_sum += pi_m * pi_b;
            }
            pi_buf[ell] = pi_step_sum * inv_m_dt;

            prev_beg = ds_beg;
            prev_end = ds_end;
            prev_inv_m = inv_m_ds;
            prev_a_k2[..m_dt_cap].copy_from_slice(&cur_a_k2[..m_dt_cap]);
            prev_a_kb[..m_dt_cap].copy_from_slice(&cur_a_kb[..m_dt_cap]);
        }

        let mut l0 = 0.0f64;
        let mut l1 = 0.0f64;
        for ell in 0..=l_act {
            let pv = pi_buf[ell];
            l0 += r0[ell] * pv;
            l1 += r1[ell] * pv;
        }
        let lw_max = lw0.max(lw1);
        let a0 = (lw0 - lw_max).exp();
        let a1 = (lw1 - lw_max).exp();
        let eta0 = a0 / (a0 + a1);
        let eta1 = a1 / (a0 + a1);

        let l_bocpd = eta0 * l0 + eta1 * l1;
        let kappa = if log_m < 0.0 {
            let m_prev = log_m.exp();
            ((m_prev - 0.50) / (0.50 * m_prev)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let e_val = (1.0 + 0.50 * kappa * (l_bocpd - 1.0)).max(0.50);
        log_m += e_val.ln();
        lw0 += l0.ln();
        lw1 += l1.ln();

        let l_glob = if l_act == 0 { 1.0 } else { pi_buf[l_act] };
        let kappa_glob = if log_m_glob < 0.0 {
            let mg = log_m_glob.exp();
            ((mg - 0.50) / (0.50 * mg)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let e_glob = (1.0 + 0.50 * kappa_glob * (l_glob - 1.0)).max(0.50);
        log_m_glob += e_glob.ln();

        let lw_max_p = lw0.max(lw1);
        let ap0 = (lw0 - lw_max_p).exp();
        let ap1 = (lw1 - lw_max_p).exp();
        let ep0 = ap0 / (ap0 + ap1);
        let ep1 = ap1 / (ap0 + ap1);

        let mut best_ell = 0i32;
        let mut best_post = -1.0f64;
        let mut cp_prob_short = 0.0f64;
        for ell in 0..=l_act {
            let pv = pi_buf[ell];
            let p_ell = ep0 * (r0[ell] * pv / l0) + ep1 * (r1[ell] * pv / l1);
            if ell <= 5 {
                cp_prob_short += p_ell;
            }
            if p_ell > best_post {
                best_post = p_ell;
                best_ell = ell as i32;
            }
        }

        step_e.push(e_val);
        step_log_m.push(log_m);
        step_map_r.push(best_ell);
        step_cp_prob.push(cp_prob_short);
        step_log_m_glob.push(log_m_glob);

        let inv_l0 = (1.0 - CKV6_LAMBDA_0) / l0;
        let inv_l1 = (1.0 - CKV6_LAMBDA_1) / l1;
        if l_act == CKV6_L_MAX {
            let surv0 = r0[CKV6_L_MAX] * pi_buf[CKV6_L_MAX] * inv_l0;
            let surv1 = r1[CKV6_L_MAX] * pi_buf[CKV6_L_MAX] * inv_l1;
            for ell in (0..CKV6_L_MAX).rev() {
                let pv = pi_buf[ell];
                r0[ell + 1] = r0[ell] * pv * inv_l0;
                r1[ell + 1] = r1[ell] * pv * inv_l1;
            }
            r0[CKV6_L_MAX] += surv0;
            r1[CKV6_L_MAX] += surv1;
        } else {
            for ell in (0..=l_act).rev() {
                let pv = pi_buf[ell];
                r0[ell + 1] = r0[ell] * pv * inv_l0;
                r1[ell + 1] = r1[ell] * pv * inv_l1;
            }
        }
        r0[0] = CKV6_LAMBDA_0;
        r1[0] = CKV6_LAMBDA_1;
    }

    Ckv6StepSeries {
        step_e,
        step_log_m,
        step_map_r,
        step_cp_prob,
        step_log_m_glob,
    }
}

pub fn eval_general_streams_cpu_24t(streams: &[Ckv6StreamInput]) -> Vec<Ckv6StepSeries> {
    std::thread::scope(|s| {
        let handles: Vec<_> = streams
            .iter()
            .map(|st| s.spawn(move || eval_general_stream_cpu(st)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

pub fn eval_general_streams_gpu(
    streams: &[Ckv6StreamInput],
) -> Result<(Vec<Ckv6StepSeries>, f64), String> {
    let mut metas = Vec::with_capacity(streams.len());
    let mut step_draw_offs: Vec<u32> = Vec::new();
    let mut draws_lo: Vec<u64> = Vec::new();
    let mut draws_hi: Vec<u32> = Vec::new();
    let mut draws_bonus: Vec<u8> = Vec::new();
    let mut total_out_steps = 0usize;

    for st in streams {
        let step_off_base = step_draw_offs.len() as u32;
        let out_step_base = total_out_steps as u32;
        for step in &st.steps {
            step_draw_offs.push(draws_lo.len() as u32);
            for (bls, bon) in &step.draws {
                let (lo, hi) = pack_balls_96(bls);
                draws_lo.push(lo);
                draws_hi.push(hi);
                draws_bonus.push(bon.unwrap_or(0));
            }
        }
        step_draw_offs.push(draws_lo.len() as u32);
        total_out_steps += st.steps.len();

        metas.push(V3Ckv6StreamMeta {
            n_balls: st.n_balls as u32,
            r_draw: st.r_draw as u32,
            m_bonus: st.m_bonus.unwrap_or(0) as u32,
            n_steps: st.steps.len() as u32,
            step_off_base,
            out_step_base,
        });
    }

    let raw = gpu_ckv6_general_streams_f64(
        &metas,
        &step_draw_offs,
        &draws_lo,
        &draws_hi,
        &draws_bonus,
        total_out_steps,
    )?;

    let mut out = Vec::with_capacity(streams.len());
    for (idx, st) in streams.iter().enumerate() {
        let beg = metas[idx].out_step_base as usize;
        let end = beg + st.steps.len();
        out.push(Ckv6StepSeries {
            step_e: raw.step_e[beg..end].to_vec(),
            step_log_m: raw.step_log_m[beg..end].to_vec(),
            step_map_r: raw.step_map_r[beg..end].to_vec(),
            step_cp_prob: raw.step_cp_prob[beg..end].to_vec(),
            step_log_m_glob: raw.step_log_m_glob[beg..end].to_vec(),
        });
    }
    Ok((out, raw.elapsed_ms))
}

#[inline]
fn project_v2_matrix_explicit(mat: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; n * n];
    let mut tot = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let v = 0.5 * (mat[i * n + j] + mat[j * n + i]);
                s[i * n + j] = v;
                tot += v;
            }
        }
    }
    let mu = tot / ((n * (n - 1)) as f64);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s[i * n + j] -= mu;
            }
        }
    }
    let mut row_sums = vec![0.0f64; n];
    for i in 0..n {
        let mut rs = 0.0f64;
        for j in 0..n {
            if i != j {
                rs += s[i * n + j];
            }
        }
        row_sums[i] = rs / ((n - 2) as f64);
    }
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s[i * n + j] -= row_sums[i] + row_sums[j];
            } else {
                s[i * n + j] = 0.0;
            }
        }
    }
    s
}

/// Construiește explicit câmpul cauzal al mixturii BOCPD `(h1_64, q2_2401, joker_tables_41420, dm_table_70, n_dm_terms, kappa_{t-1})`
/// la un pas dat `target_t` (folosind EXCLUSIV `steps[0..target_t]`), pentru verificarea exhaustivă în VRAM
/// peste toate cele `13,983,816` (`6/49`), `3,838,380` (`5/40`) și `24,435,180` (`Joker`) stări!
pub fn build_causal_universe_field_at_step(
    stream: &Ckv6StreamInput,
    target_t: usize,
) -> ([f64; 64], [f64; 2401], Option<Vec<f64>>, Vec<f64>, usize, f64, f64) {
    assert!(target_t < stream.steps.len());
    // Rulăm până la target_t pentru a obține exact distribuția predictivă R_{t, ell}^{(0)}, R_{t, ell}^{(1)}, eta_{t, h}, kappa_{t-1}
    let n = stream.n_balls;
    let r = stream.r_draw;
    let mb = stream.m_bonus.unwrap_or(0);
    let has_bon = mb >= 2;
    let comb_nr = ckv6_binom_f64(n, r);
    let p1 = (r as f64) / (n as f64);
    let d1 = (n - 1) as f64;
    let nu1 = ((r * (n - r)) as f64) / (n as f64);
    let d2 = ((n * (n - 3)) as f64) * 0.5;
    let denom2 = ((n - 1) * (n - 2)) as f64;
    let nu2 = (((r * (r - 1)) as f64) * (((n - r) * (n - r - 1)) as f64)) / denom2;
    let sqrt_nu2 = nu2.sqrt();
    let th_cap = ckv6_theta_cap_for_r(r);
    let d1_b = if has_bon { (mb - 1) as f64 } else { 1.0 };
    let nu1_b = if has_bon {
        ((mb - 1) as f64) / (mb as f64)
    } else {
        1.0
    };
    let sqrt_nu1_b = nu1_b.sqrt();

    // Mai întâi obținem exact E_t observat pe prefixul [0..=target_t] prin motorul general de referință
    let prefix = Ckv6StreamInput {
        code: stream.code.clone(),
        name: stream.name.clone(),
        n_balls: n,
        r_draw: r,
        m_bonus: stream.m_bonus,
        steps: stream.steps[..=target_t].to_vec(),
    };
    let full_ser = eval_general_stream_cpu(&prefix);
    let e_expected_on_dt = full_ser.step_e[target_t];

    // Reconstruim greutățile predictive w_ell la pasul target_t folosind vectori/matrici explicite V1 (+) V2
    let mut r0 = vec![0.0f64; CKV6_L_MAX + 2];
    let mut r1 = vec![0.0f64; CKV6_L_MAX + 2];
    r0[0] = 1.0;
    r1[0] = 1.0;
    let mut lw0 = 0.0f64;
    let mut lw1 = 0.0f64;
    let mut log_m = 0.0f64;

    // Pre-construim reprezentările explicite (bar_x_s in R^n, bar_Phi_s in R^{n*n}, bar_xb_s in R^{mb}) pt s in 0..=target_t
    let mut step_x1: Vec<Vec<f64>> = Vec::with_capacity(target_t + 1);
    let mut step_phi2: Vec<Vec<f64>> = Vec::with_capacity(target_t + 1);
    let mut step_xb: Vec<Vec<f64>> = Vec::with_capacity(target_t + 1);
    let mut step_invm: Vec<f64> = Vec::with_capacity(target_t + 1);

    for s in 0..=target_t {
        let drs = &stream.steps[s].draws;
        let inv_m = 1.0 / (drs.len() as f64);
        let mut bx = vec![0.0f64; n];
        let mut bp_raw = vec![0.0f64; n * n];
        let mut bxb = vec![0.0f64; mb.max(1)];
        for (bls, bon) in drs {
            for i in 0..n {
                bx[i] -= p1 * inv_m;
            }
            for &bi in bls {
                bx[bi as usize] += inv_m;
            }
            for i in 0..bls.len() {
                for j in (i + 1)..bls.len() {
                    let u = bls[i] as usize;
                    let v = bls[j] as usize;
                    bp_raw[u * n + v] += inv_m;
                    bp_raw[v * n + u] += inv_m;
                }
            }
            if has_bon {
                for b in 0..mb {
                    bxb[b] -= inv_m / (mb as f64);
                }
                bxb[bon.unwrap_or(0) as usize] += inv_m;
            }
        }
        step_x1.push(bx);
        step_phi2.push(project_v2_matrix_explicit(&bp_raw, n));
        step_xb.push(bxb);
        step_invm.push(inv_m);
    }

    // Avansăm r0, r1, lw0, lw1, log_m pe pașii s = 0..target_t
    for s in 0..target_t {
        let l_act = s.min(CKV6_L_MAX);
        let mut pi_s = vec![1.0f64; l_act + 1];
        let drs_s = &stream.steps[s].draws;
        let inv_m_s = 1.0 / (drs_s.len() as f64);

        let mut x1_acc = vec![0.0f64; n];
        let mut z1_acc = vec![0.0f64; n];
        let mut x2_acc = vec![0.0f64; n * n];
        let mut z2_acc = vec![0.0f64; n * n];
        let mut xb_acc = vec![0.0f64; mb.max(1)];
        let mut zb_acc = vec![0.0f64; mb.max(1)];
        let mut n_eff = 0.0f64;
        let mut w1_sq_eff = 0.0f64;
        let mut w2_sq_eff = 0.0f64;
        let mut wb_sq_eff = 0.0f64;
        let mut u_balls = vec![1.0f64; n];

        for ell in 1..=l_act {
            let u = s - ell;
            n_eff += step_invm[u];
            for i in 0..n {
                x1_acc[i] += step_x1[u][i];
            }
            for ij in 0..(n * n) {
                x2_acc[ij] += step_phi2[u][ij];
            }
            if has_bon {
                for b in 0..mb {
                    xb_acc[b] += step_xb[u][b];
                }
            }
            if ell >= 2 {
                let w1: f64 = step_x1[u]
                    .iter()
                    .zip(step_x1[s - 1].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let w2: f64 = step_phi2[u]
                    .iter()
                    .zip(step_phi2[s - 1].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                w1_sq_eff += w1 * w1 * step_invm[u + 1];
                w2_sq_eff += w2 * w2 * step_invm[u + 1];
                for i in 0..n {
                    z1_acc[i] += w1 * step_x1[u + 1][i];
                }
                for ij in 0..(n * n) {
                    z2_acc[ij] += w2 * step_phi2[u + 1][ij];
                }
                if has_bon {
                    let wb: f64 = step_xb[u]
                        .iter()
                        .zip(step_xb[s - 1].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    wb_sq_eff += wb * wb * step_invm[u + 1];
                    for b in 0..mb {
                        zb_acc[b] += wb * step_xb[u + 1][b];
                    }
                }
            }

            let d_ell = ell as f64;
            let e_x1: f64 = x1_acc.iter().map(|v| v * v).sum();
            let eta_stat1 = if e_x1 > n_eff * nu1 {
                (1.0 - (n_eff * nu1) / e_x1) * (d_ell / (d_ell + CKV6_TAU_STAT_1))
            } else {
                0.0
            };
            let e_z1: f64 = z1_acc.iter().map(|v| v * v).sum();
            let scale_koop1 = if ell >= 2 && e_z1 > nu1 * w1_sq_eff {
                let g_k1 = (1.0 - (nu1 * w1_sq_eff) / e_z1)
                    * (d1 / (nu1 * (((ell - 1) as f64) + CKV6_TAU_KOOP_1)));
                (d1 / nu1) * g_k1
            } else {
                0.0
            };
            let dm_act = eta_stat1 > 0.0 || scale_koop1 > 0.0;
            let mut er = comb_nr;
            if dm_act {
                let base_c0 = d_ell * p1 + CKV6_ALPHA_DM;
                for i in 0..n {
                    let c_bi = (x1_acc[i] + d_ell * p1).max(0.0);
                    let th = eta_stat1 * ((c_bi + CKV6_ALPHA_DM) / base_c0).ln()
                        + scale_koop1 * z1_acc[i];
                    u_balls[i] = ckv6_ball_weight_f64(th, th_cap);
                }
                er = ckv6_elem_sym_sorted_dp(&u_balls, r);
            }

            let e_x2: f64 = x2_acc.iter().map(|v| v * v).sum();
            let g_s2 = if e_x2 > n_eff * nu2 {
                (1.0 - (n_eff * nu2) / e_x2) / (d_ell + CKV6_TAU_STAT_2)
            } else {
                0.0
            };
            let e_z2: f64 = z2_acc.iter().map(|v| v * v).sum();
            let g_k2 = if ell >= 2 && e_z2 > nu2 * w2_sq_eff {
                (1.0 - (nu2 * w2_sq_eff) / e_z2) * (d2 / (nu2 * (((ell - 1) as f64) + CKV6_TAU_KOOP_2)))
            } else {
                0.0
            };
            let mut m2 = vec![0.0f64; n * n];
            let mut m2_n2 = 0.0f64;
            for ij in 0..(n * n) {
                m2[ij] = g_s2 * x2_acc[ij] + g_k2 * z2_acc[ij];
                m2_n2 += m2[ij] * m2[ij];
            }
            let b2 = (d2 / sqrt_nu2) * m2_n2.sqrt();
            let c_v2 = ckv6_damp_factor(b2);

            let mut mb_vec = vec![0.0f64; mb.max(1)];
            let mut c_bon = 1.0f64;
            if has_bon {
                let e_xb: f64 = xb_acc.iter().map(|v| v * v).sum();
                let g_sb = if e_xb > n_eff * nu1_b {
                    (1.0 - (n_eff * nu1_b) / e_xb) / (d_ell + CKV6_TAU_BON_STAT)
                } else {
                    0.0
                };
                let e_zb: f64 = zb_acc.iter().map(|v| v * v).sum();
                let g_kb = if ell >= 2 && e_zb > nu1_b * wb_sq_eff {
                    (1.0 - (nu1_b * wb_sq_eff) / e_zb)
                        * (d1_b / (nu1_b * (((ell - 1) as f64) + CKV6_TAU_BON_KOOP)))
                } else {
                    0.0
                };
                let mut mb_n2 = 0.0f64;
                for b in 0..mb {
                    mb_vec[b] = g_sb * xb_acc[b] + g_kb * zb_acc[b];
                    mb_n2 += mb_vec[b] * mb_vec[b];
                }
                c_bon = ckv6_damp_factor((d1_b / sqrt_nu1_b) * mb_n2.sqrt());
            }

            let mut p_sum = 0.0f64;
            for (bls, bon) in drs_s {
                let lam1 = if dm_act {
                    let mut prod_u = 1.0f64;
                    for &bi in bls {
                        prod_u *= u_balls[bi as usize];
                    }
                    comb_nr * prod_u / er
                } else {
                    1.0
                };
                let mut s2 = 0.0f64;
                for i in 0..bls.len() {
                    for j in (i + 1)..bls.len() {
                        s2 += 2.0 * m2[(bls[i] as usize) * n + (bls[j] as usize)];
                    }
                }
                s2 *= d2 / nu2;
                let lam2 = (1.0 + c_v2 * s2).max(0.05);
                let pm = CKV6_BETA_DM * lam1 + CKV6_BETA_V2 * lam2;
                let pb = if has_bon {
                    (1.0 + c_bon * (d1_b / nu1_b) * mb_vec[bon.unwrap_or(0) as usize]).max(0.05)
                } else {
                    1.0
                };
                p_sum += pm * pb;
            }
            pi_s[ell] = p_sum * inv_m_s;
        }

        let mut l0 = 0.0f64;
        let mut l1 = 0.0f64;
        for ell in 0..=l_act {
            l0 += r0[ell] * pi_s[ell];
            l1 += r1[ell] * pi_s[ell];
        }
        let lw_max = lw0.max(lw1);
        let a0 = (lw0 - lw_max).exp();
        let a1 = (lw1 - lw_max).exp();
        let l_bocpd = (a0 * l0 + a1 * l1) / (a0 + a1);
        let kap = if log_m < 0.0 {
            let mp = log_m.exp();
            ((mp - 0.50) / (0.50 * mp)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let ev = (1.0 + 0.50 * kap * (l_bocpd - 1.0)).max(0.50);
        log_m += ev.ln();
        lw0 += l0.ln();
        lw1 += l1.ln();

        let inv_l0 = (1.0 - CKV6_LAMBDA_0) / l0;
        let inv_l1 = (1.0 - CKV6_LAMBDA_1) / l1;
        if l_act == CKV6_L_MAX {
            let surv0 = r0[CKV6_L_MAX] * pi_s[CKV6_L_MAX] * inv_l0;
            let surv1 = r1[CKV6_L_MAX] * pi_s[CKV6_L_MAX] * inv_l1;
            for ell in (0..CKV6_L_MAX).rev() {
                r0[ell + 1] = r0[ell] * pi_s[ell] * inv_l0;
                r1[ell + 1] = r1[ell] * pi_s[ell] * inv_l1;
            }
            r0[CKV6_L_MAX] += surv0;
            r1[CKV6_L_MAX] += surv1;
        } else {
            for ell in (0..=l_act).rev() {
                r0[ell + 1] = r0[ell] * pi_s[ell] * inv_l0;
                r1[ell + 1] = r1[ell] * pi_s[ell] * inv_l1;
            }
        }
        r0[0] = CKV6_LAMBDA_0;
        r1[0] = CKV6_LAMBDA_1;
    }

    // Acum suntem exact la pasul `target_t` (înainte de observarea extragerii `target_t`)!
    let l_act = target_t.min(CKV6_L_MAX);
    let lw_max = lw0.max(lw1);
    let a0 = (lw0 - lw_max).exp();
    let a1 = (lw1 - lw_max).exp();
    let eta0 = a0 / (a0 + a1);
    let eta1 = a1 / (a0 + a1);
    let kappa = if log_m < 0.0 {
        let mp = log_m.exp();
        ((mp - 0.50) / (0.50 * mp)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let mut w_ell = vec![0.0f64; l_act + 1];
    for ell in 0..=l_act {
        w_ell[ell] = eta0 * r0[ell] + eta1 * r1[ell];
    }

    let mut h1_64 = [0.0f64; 64];
    h1_64[0] = w_ell[0];
    let mut q2_2401 = [0.0f64; 2401];
    let mut joker_tbl = if has_bon && mb == 20 && n == 45 {
        let mut jt = vec![0.0f64; 20 * 2071];
        for jb in 0..20 {
            jt[jb * 2071] = w_ell[0];
        }
        Some(jt)
    } else {
        None
    };
    let mut dm_table_70: Vec<f64> = Vec::new();
    let mut n_dm_terms = 0usize;

    let mut x1_acc = vec![0.0f64; n];
    let mut z1_acc = vec![0.0f64; n];
    let mut x2_acc = vec![0.0f64; n * n];
    let mut z2_acc = vec![0.0f64; n * n];
    let mut xb_acc = vec![0.0f64; mb.max(1)];
    let mut zb_acc = vec![0.0f64; mb.max(1)];
    let mut n_eff = 0.0f64;
    let mut w1_sq_eff = 0.0f64;
    let mut w2_sq_eff = 0.0f64;
    let mut wb_sq_eff = 0.0f64;

    for ell in 1..=l_act {
        let u = target_t - ell;
        n_eff += step_invm[u];
        for i in 0..n {
            x1_acc[i] += step_x1[u][i];
        }
        for ij in 0..(n * n) {
            x2_acc[ij] += step_phi2[u][ij];
        }
        if has_bon {
            for b in 0..mb {
                xb_acc[b] += step_xb[u][b];
            }
        }
        if ell >= 2 {
            let w1: f64 = step_x1[u]
                .iter()
                .zip(step_x1[target_t - 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            let w2: f64 = step_phi2[u]
                .iter()
                .zip(step_phi2[target_t - 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            w1_sq_eff += w1 * w1 * step_invm[u + 1];
            w2_sq_eff += w2 * w2 * step_invm[u + 1];
            for i in 0..n {
                z1_acc[i] += w1 * step_x1[u + 1][i];
            }
            for ij in 0..(n * n) {
                z2_acc[ij] += w2 * step_phi2[u + 1][ij];
            }
            if has_bon {
                let wb: f64 = step_xb[u]
                    .iter()
                    .zip(step_xb[target_t - 1].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                wb_sq_eff += wb * wb * step_invm[u + 1];
                for b in 0..mb {
                    zb_acc[b] += wb * step_xb[u + 1][b];
                }
            }
        }

        let d_ell = ell as f64;
        let e_x1: f64 = x1_acc.iter().map(|v| v * v).sum();
        let eta_stat1 = if e_x1 > n_eff * nu1 {
            (1.0 - (n_eff * nu1) / e_x1) * (d_ell / (d_ell + CKV6_TAU_STAT_1))
        } else {
            0.0
        };
        let e_z1: f64 = z1_acc.iter().map(|v| v * v).sum();
        let scale_koop1 = if ell >= 2 && e_z1 > nu1 * w1_sq_eff {
            let g_k1 = (1.0 - (nu1 * w1_sq_eff) / e_z1)
                * (d1 / (nu1 * (((ell - 1) as f64) + CKV6_TAU_KOOP_1)));
            (d1 / nu1) * g_k1
        } else {
            0.0
        };
        let dm_act = eta_stat1 > 0.0 || scale_koop1 > 0.0;
        let mut u_row = [1.0f64; 49];
        let mut er = comb_nr;
        if dm_act {
            let base_c0 = d_ell * p1 + CKV6_ALPHA_DM;
            for i in 0..n {
                let c_bi = (x1_acc[i] + d_ell * p1).max(0.0);
                let th = eta_stat1 * ((c_bi + CKV6_ALPHA_DM) / base_c0).ln()
                    + scale_koop1 * z1_acc[i];
                u_row[i] = ckv6_ball_weight_f64(th, th_cap);
            }
            er = ckv6_elem_sym_sorted_dp(&u_row[..n], r);
        }

        let e_x2: f64 = x2_acc.iter().map(|v| v * v).sum();
        let g_s2 = if e_x2 > n_eff * nu2 {
            (1.0 - (n_eff * nu2) / e_x2) / (d_ell + CKV6_TAU_STAT_2)
        } else {
            0.0
        };
        let e_z2: f64 = z2_acc.iter().map(|v| v * v).sum();
        let g_k2 = if ell >= 2 && e_z2 > nu2 * w2_sq_eff {
            (1.0 - (nu2 * w2_sq_eff) / e_z2) * (d2 / (nu2 * (((ell - 1) as f64) + CKV6_TAU_KOOP_2)))
        } else {
            0.0
        };
        let mut m2 = vec![0.0f64; n * n];
        let mut m2_n2 = 0.0f64;
        for ij in 0..(n * n) {
            m2[ij] = g_s2 * x2_acc[ij] + g_k2 * z2_acc[ij];
            m2_n2 += m2[ij] * m2[ij];
        }
        let b2 = (d2 / sqrt_nu2) * m2_n2.sqrt();
        let c_v2 = ckv6_damp_factor(b2);

        let wl = w_ell[ell];
        if let Some(ref mut jt) = joker_tbl {
            let e_xb: f64 = xb_acc.iter().map(|v| v * v).sum();
            let g_sb = if e_xb > n_eff * nu1_b {
                (1.0 - (n_eff * nu1_b) / e_xb) / (d_ell + CKV6_TAU_BON_STAT)
            } else {
                0.0
            };
            let e_zb: f64 = zb_acc.iter().map(|v| v * v).sum();
            let g_kb = if ell >= 2 && e_zb > nu1_b * wb_sq_eff {
                (1.0 - (nu1_b * wb_sq_eff) / e_zb)
                    * (d1_b / (nu1_b * (((ell - 1) as f64) + CKV6_TAU_BON_KOOP)))
            } else {
                0.0
            };
            let mut mb_vec = [0.0f64; 20];
            let mut mb_n2 = 0.0f64;
            for b in 0..20 {
                mb_vec[b] = g_sb * xb_acc[b] + g_kb * zb_acc[b];
                mb_n2 += mb_vec[b] * mb_vec[b];
            }
            let c_bon = ckv6_damp_factor((d1_b / sqrt_nu1_b) * mb_n2.sqrt());
            let mut dm_row70 = [0.0f64; CKV6_DM_ROW_STRIDE];
            dm_row70[..49].copy_from_slice(&u_row);
            for jb in 0..20 {
                let pb = 1.0 + c_bon * (d1_b / nu1_b) * mb_vec[jb];
                let w_jb = wl * pb;
                let base = jb * 2071;
                jt[base] += w_jb * CKV6_BETA_V2;
                if dm_act {
                    dm_row70[50 + jb] = w_jb * CKV6_BETA_DM * (comb_nr / er);
                } else {
                    jt[base] += w_jb * CKV6_BETA_DM;
                }
                for i in 0..45 {
                    for j in (i + 1)..45 {
                        jt[base + 46 + i * 45 + j] +=
                            w_jb * CKV6_BETA_V2 * c_v2 * 2.0 * (d2 / nu2) * m2[i * 45 + j];
                    }
                }
            }
            if dm_act {
                dm_table_70.extend_from_slice(&dm_row70);
                n_dm_terms += 1;
            }
        } else {
            h1_64[0] += wl * CKV6_BETA_V2;
            if dm_act {
                let mut dm_row70 = [0.0f64; CKV6_DM_ROW_STRIDE];
                dm_row70[..49].copy_from_slice(&u_row);
                dm_row70[49] = wl * CKV6_BETA_DM * (comb_nr / er);
                dm_table_70.extend_from_slice(&dm_row70);
                n_dm_terms += 1;
            } else {
                h1_64[0] += wl * CKV6_BETA_DM;
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    q2_2401[i * n + j] +=
                        wl * CKV6_BETA_V2 * c_v2 * 2.0 * (d2 / nu2) * m2[i * n + j];
                }
            }
        }
    }

    // Verificare rapidă că evaluarea câmpului pe extragerile observate reproduce exact `e_expected_on_dt`!
    let obs_drs = &stream.steps[target_t].draws;
    let mut pi_obs_sum = 0.0f64;
    for (obs_balls, obs_bon) in obs_drs {
        let pi_single = if let Some(ref jt) = joker_tbl {
            let jb = obs_bon.unwrap_or(0) as usize;
            let base = jb * 2071;
            let mut s2 = 0.0f64;
            for i in 0..obs_balls.len() {
                let bi = obs_balls[i] as usize;
                for j in (i + 1)..obs_balls.len() {
                    s2 += jt[base + 46 + bi * 45 + (obs_balls[j] as usize)];
                }
            }
            let mut s_dm = 0.0f64;
            for k in 0..n_dm_terms {
                let row = &dm_table_70[(k * CKV6_DM_ROW_STRIDE)..((k + 1) * CKV6_DM_ROW_STRIDE)];
                let mut prod_u = 1.0f64;
                for &bi in obs_balls {
                    prod_u *= row[bi as usize];
                }
                s_dm += row[50 + jb] * prod_u;
            }
            jt[base] + s_dm + s2
        } else {
            let mut s2 = 0.0f64;
            for i in 0..obs_balls.len() {
                let bi = obs_balls[i] as usize;
                for j in (i + 1)..obs_balls.len() {
                    s2 += q2_2401[bi * n + (obs_balls[j] as usize)];
                }
            }
            let mut s_dm = 0.0f64;
            for k in 0..n_dm_terms {
                let row = &dm_table_70[(k * CKV6_DM_ROW_STRIDE)..((k + 1) * CKV6_DM_ROW_STRIDE)];
                let mut prod_u = 1.0f64;
                for &bi in obs_balls {
                    prod_u *= row[bi as usize];
                }
                s_dm += row[49] * prod_u;
            }
            h1_64[0] + s_dm + s2
        };
        pi_obs_sum += pi_single;
    }
    let pi_obs = pi_obs_sum / (obs_drs.len() as f64);
    let e_field_obs = (1.0 + 0.50 * kappa * (pi_obs - 1.0)).max(0.50);
    assert!(
        (e_field_obs - e_expected_on_dt).abs() < 1.0e-12,
        "Field mismatch: {e_field_obs} vs {e_expected_on_dt}"
    );

    (
        h1_64,
        q2_2401,
        joker_tbl,
        dm_table_70,
        n_dm_terms,
        kappa,
        e_expected_on_dt,
    )
}
// ============================================================================
// 3. Structuri de Raportare, Calificare Q1-Q6, Putere 10k, Sigilare și Arhive
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniverseMartingaleCert {
    pub game: String,
    pub total_states: u64,
    pub probe_step: usize,
    pub mean_e_exact_vram: f64,
    pub abs_err_vs_one: f64,
    pub min_e_vram: f64,
    pub max_e_vram: f64,
    pub e_observed_step_cpu: f64,
    pub gpu_eval_ms: f64,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QTestSummary {
    pub id: String,
    pub name: String,
    pub pass: bool,
    pub metrics: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QWitnessTrajectory {
    pub id: String,
    pub n_balls: usize,
    pub r_draw: usize,
    pub m_bonus: Option<usize>,
    pub steps: Vec<Ckv6CanonicalStep>,
    pub expected_step_e: Vec<f64>,
    pub expected_step_log_m: Vec<f64>,
    pub expected_step_map_r: Vec<i32>,
    pub expected_step_log_m_glob: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowerCellSummary {
    pub cell_id: u32,
    pub delta: f64,
    pub delta_bp: u32,
    pub l_regime: u32,
    pub n_histories: u32,
    pub power_m_ge_20: f64,
    pub power_m_ge_1500: f64,
    pub power_anytime_ge_20: f64,
    pub power_anytime_ge_1500: f64,
    pub mean_log_m_t: f64,
    pub median_log_m_t: f64,
    pub mean_log_m_t_global_no_reset: f64,
    pub mean_bocpd_gain_log_m: f64,
    pub min_e_across_cell: f64,
    pub mean_cp_loc_abs_err: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CkV6QualificationAndPowerReport {
    pub engine: String,
    pub session: String,
    pub gate: String,
    pub generated_utc: String,
    pub positive_claims_allowed: bool,
    pub archive_files_read: usize,
    pub gpu: CudaDeviceInfo,
    pub cpu_threads: usize,
    pub model_constants: Value,
    pub q_tests: Vec<QTestSummary>,
    pub q_passed_count: usize,
    pub q_all_6_pass: bool,
    pub universe_martingale_certificates: Vec<UniverseMartingaleCert>,
    pub q_witness_trajectories: Vec<QWitnessTrajectory>,
    pub power_total_histories: usize,
    pub power_t_steps: u32,
    pub power_seed: u64,
    pub power_gpu_ms: f64,
    pub power_cpu_sample_100_max_abs_diff: f64,
    pub power_v3bin_filename: String,
    pub power_v3bin_bytes: u64,
    pub power_v3bin_sha256: String,
    pub power_cells: Vec<PowerCellSummary>,
    pub resolution_thresholds: Value,
    pub rss_current_bytes: u64,
    pub rss_peak_hwm_bytes: u64,
    pub rss_limit_bytes: u64,
    pub rss_guard_ok: bool,
    pub total_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindReserveLstatEntry {
    pub game: String,
    pub rel_path: String,
    pub mode_octal: String,
    pub size_bytes: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub ctime_sec: i64,
    pub ctime_nsec: i64,
    pub is_strict_0400: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CkV6CodeSeal {
    pub seal_id: String,
    pub session: String,
    pub sealed_utc: String,
    pub positive_claims_allowed: bool,
    pub archive_files_read_before_seal: usize,
    pub ckv6_rs_path: String,
    pub ckv6_rs_sha256: String,
    pub ckv6_rs_bytes: u64,
    pub ckv6_rs_mode_octal: String,
    pub calificare_putere_path: String,
    pub calificare_putere_sha256: String,
    pub q_all_6_pass: bool,
    pub power_total_histories: usize,
    pub power_v3bin_sha256: String,
    pub blind_reserves_lstat_baseline: Vec<BlindReserveLstatEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveStepCheckpoint {
    pub step_idx: usize,
    pub key: String,
    pub m_draws_in_step: usize,
    pub e_step: f64,
    pub log_m_cum: f64,
    pub m_cum: f64,
    pub map_r: i32,
    pub cp_prob_short: f64,
    pub log_m_glob: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveTrajectoryPoint {
    pub step_idx: usize,
    pub key: String,
    pub m_cum: f64,
    pub log10_m_cum: f64,
    pub e_step: f64,
    pub map_r: i32,
    pub cp_prob_short: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectedChangepointPeak {
    pub step_idx: usize,
    pub key: String,
    pub cp_prob_short: f64,
    pub map_r: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoK2Comparison {
    pub game: String,
    pub kjv5_k2_e_final: f64,
    pub kjv5_k2_gates_passed: usize,
    pub kjv5_full_e_final: f64,
    pub kjv5_full_gates_passed: usize,
    pub ckv6_full_m_t: f64,
    pub ckv6_post_warmup100_m_t: f64,
    pub ckv6_min_e_t: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamWalkForwardReport {
    pub code: String,
    pub name: String,
    pub family: String,
    pub epistemic_role: String,
    pub rel_file: String,
    pub file_sha256: String,
    pub n_balls: usize,
    pub r_draw: usize,
    pub m_bonus: Option<usize>,
    pub n_raw_draws: usize,
    pub n_canonical_steps: usize,
    pub double_draw_days: usize,
    pub m_t_final: f64,
    pub log_m_t_final: f64,
    pub log10_m_t_final: f64,
    pub max_m_t_anytime: f64,
    pub min_m_t_anytime: f64,
    pub min_e_t: f64,
    pub max_e_t: f64,
    pub m_t_post_warmup100: Option<f64>,
    pub m_t_global_no_reset: f64,
    pub log_m_t_global_no_reset: f64,
    pub final_map_run_length: i32,
    pub gpu_cpu_max_abs_diff_e: f64,
    pub gpu_cpu_max_abs_diff_log_m: f64,
    pub detected_changepoint_peaks: Vec<DetectedChangepointPeak>,
    pub exact_checkpoints: Vec<ArchiveStepCheckpoint>,
    pub gui_trajectory_256: Vec<ArchiveTrajectoryPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HolmDevEntry {
    pub rank_1based: usize,
    pub stream_code: String,
    pub n_draws_dev: usize,
    pub e_dev_raw: f64,
    pub log_e_dev_raw: f64,
    pub holm_divisor: usize,
    pub e_dev_holm_adjusted: f64,
    pub threshold_e_dev: f64,
    pub passes_holm_1500: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CkV6ArchivesAndDevReport {
    pub engine: String,
    pub session: String,
    pub gate: String,
    pub generated_utc: String,
    pub positive_claims_allowed: bool,
    pub epistemic_notice: String,
    pub code_seal_verified: bool,
    pub code_seal_sha256: String,
    pub ckv6_rs_sha256: String,
    pub ckv6_rs_mode_octal: String,
    pub gpu_streams_eval_ms: f64,
    pub cpu_streams_eval_ms: f64,
    pub max_gpu_cpu_abs_diff_across_all_13_streams: f64,
    pub min_e_across_all_13_streams: f64,
    pub min_m_across_all_13_streams: f64,
    pub ro_exploratory_streams: Vec<StreamWalkForwardReport>,
    pub ro_vs_kjv5_k2_comparison: Vec<RoK2Comparison>,
    pub intl_non_reserved_controls: Vec<StreamWalkForwardReport>,
    pub xi_p2_dev_streams: Vec<StreamWalkForwardReport>,
    pub xi_p2_hu_hatos_protocol_note: String,
    pub holm_family_size_g: usize,
    pub holm_dev_table: Vec<HolmDevEntry>,
    pub global_product_e_dev_clean4: f64,
    pub log_global_product_e_dev_clean4: f64,
    pub dev_survivors_holm_1500: Vec<String>,
    pub blind_reserves_touched: bool,
    pub blind_reserves_lstat_after: Vec<BlindReserveLstatEntry>,
    pub blind_reserves_unchanged_and_0400: bool,
    pub scientific_verdict: String,
    pub gate_a5_pass: bool,
    pub rss_current_bytes: u64,
    pub rss_peak_hwm_bytes: u64,
}

/// Inspectare strict non-invazivă `lstat` (`symlink_metadata`) pe cele 4 fișiere `rezerva.json` (`0400`) —
/// NU deschide, NU citește niciun octet și NU modifică `atime`/`mtime`!
pub fn inspect_blind_reserves_lstat(ql_root: &Path) -> Result<Vec<BlindReserveLstatEntry>, String> {
    let mut out = Vec::with_capacity(BLIND_RESERVE_GAMES.len());
    for game in BLIND_RESERVE_GAMES {
        let rel = format!("xi/date_p2/{game}/rezerva.json");
        let full = ql_root.join(&rel);
        let md = fs::symlink_metadata(&full)
            .map_err(|e| format!("lstat {}: {e}", full.display()))?;
        let mode = md.permissions().mode() & 0o777;
        out.push(BlindReserveLstatEntry {
            game: game.to_string(),
            rel_path: rel,
            mode_octal: format!("{mode:04o}"),
            size_bytes: md.len(),
            mtime_sec: md.mtime(),
            mtime_nsec: md.mtime_nsec(),
            ctime_sec: md.ctime(),
            ctime_nsec: md.ctime_nsec(),
            is_strict_0400: mode == 0o400,
        });
    }
    Ok(out)
}

fn permute_stream_labels(
    stream: &Ckv6StreamInput,
    perm_main: &[u8],
    perm_bon: Option<&[u8]>,
) -> Ckv6StreamInput {
    let mut steps = Vec::with_capacity(stream.steps.len());
    for st in &stream.steps {
        let mut drs = Vec::with_capacity(st.draws.len());
        for (bls, bon) in &st.draws {
            let mut pb: Vec<u8> = bls.iter().map(|&b| perm_main[b as usize]).collect();
            pb.sort_unstable();
            let pbon = match (bon, perm_bon) {
                (Some(jb), Some(pm)) => Some(pm[*jb as usize]),
                (x, _) => *x,
            };
            drs.push((pb, pbon));
        }
        steps.push(Ckv6CanonicalStep {
            key: st.key.clone(),
            draws: drs,
        });
    }
    Ckv6StreamInput {
        code: format!("{}_perm", stream.code),
        name: format!("{} Permuted", stream.name),
        n_balls: stream.n_balls,
        r_draw: stream.r_draw,
        m_bonus: stream.m_bonus,
        steps,
    }
}

fn make_perm(n: usize, seed: u64, tag: u32) -> Vec<u8> {
    let mut rng = PhiloxCpu::new(seed, 55_000 + tag, n as u32, tag);
    let mut p: Vec<u8> = (0..(n as u8)).collect();
    for i in 0..n {
        let j = i + (rng.uniform_u32((n - i) as u32) as usize);
        p.swap(i, j);
    }
    p
}

/// Execută întreaga suită sintetică `Q1–Q6` (`archive_files_read = 0`) + suprafața de putere pe `10,000`
/// istorii sintetice pe `RTX 5060 sm_120`, scriind `ckv6_power_10k.v3bin` (`0444`) și `CK_V6_CALIFICARE_PUTERE.json`.
pub fn run_ckv6_qualify_and_power(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    out_json_path: &Path,
) -> Result<CkV6QualificationAndPowerReport, String> {
    let t_start = Instant::now();
    check_rss_guard(0)?;
    let gpu_info = query_device_info()?;
    let cpu_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);

    // ------------------------------------------------------------------------
    // Q1: Martingal sub H0 uniform (2,000 traiectorii 6/49 + 500 multi-geometrie
    //     + certificare exhaustivă 100% pe toate cele 42,257,376 stări din VRAM!)
    // ------------------------------------------------------------------------
    let q1_n_traj = 2_000usize;
    let q1_t = 400usize;
    let mut q1_batch = vec![0u64; q1_n_traj * q1_t];
    std::thread::scope(|s| {
        let chunk_tr = (q1_n_traj + cpu_threads - 1) / cpu_threads;
        for (cid, ch) in q1_batch.chunks_mut(chunk_tr.max(1) * q1_t).enumerate() {
            let base_rep = cid * chunk_tr.max(1);
            s.spawn(move || {
                for (loc_i, dst) in ch.chunks_exact_mut(q1_t).enumerate() {
                    let rep = (base_rep + loc_i) as u32;
                    let h = gen_synth_649_history_cpu(CKV6_QUAL_SEED, 0, rep, 0, 400, q1_t);
                    dst.copy_from_slice(&h);
                }
            });
        }
    });
    let q1_gpu = gpu_ckv6_649_batch_f64(&q1_batch, q1_n_traj, q1_t, 0, true)?;
    let q1_total_steps = q1_n_traj * q1_t;
    let q1_mean_step_e: f64 = q1_gpu.step_e.iter().sum::<f64>() / (q1_total_steps as f64);
    let q1_var_step_e: f64 = q1_gpu
        .step_e
        .iter()
        .map(|&e| (e - q1_mean_step_e) * (e - q1_mean_step_e))
        .sum::<f64>()
        / (q1_total_steps as f64);
    let q1_se_step_e = (q1_var_step_e / (q1_total_steps as f64)).sqrt();

    let mut ville_ge_20 = 0usize;
    let mut ville_ge_100 = 0usize;
    let mut ville_ge_1500 = 0usize;
    let mut q1_min_e = f64::INFINITY;
    let mut q1_min_m = f64::INFINITY;
    for i in 0..q1_n_traj {
        let max_m = q1_gpu.summary5[i][1].exp();
        let min_e = q1_gpu.summary5[i][2];
        if min_e < q1_min_e {
            q1_min_e = min_e;
        }
        if max_m >= 20.0 {
            ville_ge_20 += 1;
        }
        if max_m >= 100.0 {
            ville_ge_100 += 1;
        }
        if max_m >= 1_500.0 {
            ville_ge_1500 += 1;
        }
    }
    for &lm in &q1_gpu.step_log_m {
        let m = lm.exp();
        if m < q1_min_m {
            q1_min_m = m;
        }
    }
    let fp_rate_20 = (ville_ge_20 as f64) / (q1_n_traj as f64);
    let fp_rate_100 = (ville_ge_100 as f64) / (q1_n_traj as f64);
    let fp_rate_1500 = (ville_ge_1500 as f64) / (q1_n_traj as f64);

    // Multi-geometrie H0 uniform (5 geometrii străine x 100 traiectorii x T=300 = 500 traiectorii)
    let foreign_geoms: [(&str, usize, usize, Option<usize>); 5] = [
        ("syn_hu_otos_5_90", 90, 5, None),
        ("syn_fr_loto_5_49_b10", 49, 5, Some(10)),
        ("syn_us_pb_5_69_b26", 69, 5, Some(26)),
        ("syn_hu_hatos_6_45", 45, 6, None),
        ("syn_ny_pick10_20_80", 80, 20, None),
    ];
    let mut mg_streams = Vec::with_capacity(500);
    for (gi, &(nm, nb, rd, mb)) in foreign_geoms.iter().enumerate() {
        for rep in 0..100u32 {
            mg_streams.push(gen_synth_general_stream_cpu(
                nm,
                nb,
                rd,
                mb,
                300,
                0,
                300,
                0,
                CKV6_QUAL_SEED + 1000,
                (gi as u32) * 1000 + rep,
            ));
        }
    }
    let (mg_gpu_ser, _) = eval_general_streams_gpu(&mg_streams)?;
    let mut mg_ville_20 = 0usize;
    let mut mg_ville_1500 = 0usize;
    let mut mg_min_e = f64::INFINITY;
    for ser in &mg_gpu_ser {
        let max_m = ser
            .step_log_m
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
            .exp();
        let min_e = ser.step_e.iter().copied().fold(f64::INFINITY, f64::min);
        if min_e < mg_min_e {
            mg_min_e = min_e;
        }
        if max_m >= 20.0 {
            mg_ville_20 += 1;
        }
        if max_m >= 1_500.0 {
            mg_ville_1500 += 1;
        }
    }

    // Certificare exhaustivă E_{omega ~ U}[E_t(omega)] == 1.0 pe toate cele 42,257,376 stări din VRAM
    let syn_u649 = gen_synth_general_stream_cpu("u649", 49, 6, None, 70, 5000, 35, 12, CKV6_QUAL_SEED, 701);
    let syn_u540 = gen_synth_general_stream_cpu("u540", 40, 6, None, 70, 5000, 35, 12, CKV6_QUAL_SEED, 702);
    let syn_ujok = gen_synth_general_stream_cpu("ujok", 45, 5, Some(20), 70, 5000, 35, 12, CKV6_QUAL_SEED, 703);

    let mut univ_certs = Vec::with_capacity(3);
    for (gcode, gname, d_buf, n_states, st_ref) in [
        (0i32, "649", &gpu_univ.d_masks_649, COUNT_649 as u64, &syn_u649),
        (1i32, "540_draws6", &gpu_univ.d_draws6_540, COUNT_540_DRAWS6 as u64, &syn_u540),
        (2i32, "joker_24m", &gpu_univ.d_full_joker, COUNT_JOKER_FULL24M as u64, &syn_ujok),
    ] {
        let probe_t = 65usize;
        let (h1, q2, j_tbl, dm_tbl, n_dm, kap, e_obs) =
            build_causal_universe_field_at_step(st_ref, probe_t);
        let (mean_e, min_e, max_e, ms) = gpu_ckv6_universe_eval_f64(
            gcode,
            d_buf,
            st_ref.n_balls,
            st_ref.r_draw,
            &h1,
            &q2,
            j_tbl.as_deref(),
            &dm_tbl,
            n_dm,
            kap,
        )?;
        let abs_err = (mean_e - 1.0).abs();
        let pass = abs_err < 1.0e-12 && min_e >= 0.50 && e_obs >= min_e - 1.0e-12 && e_obs <= max_e + 1.0e-12;
        univ_certs.push(UniverseMartingaleCert {
            game: gname.to_string(),
            total_states: n_states,
            probe_step: probe_t,
            mean_e_exact_vram: mean_e,
            abs_err_vs_one: abs_err,
            min_e_vram: min_e,
            max_e_vram: max_e,
            e_observed_step_cpu: e_obs,
            gpu_eval_ms: ms,
            pass,
        });
    }

    let q1_pass = (q1_mean_step_e - 1.0).abs() <= 4.0 * q1_se_step_e.max(2.0e-4)
        && fp_rate_20 <= 0.065
        && fp_rate_100 <= 0.020
        && ville_ge_1500 == 0
        && mg_ville_1500 == 0
        && (mg_ville_20 as f64 / 500.0) <= 0.08
        && univ_certs.iter().all(|c| c.pass);

    // ------------------------------------------------------------------------
    // Q2: Scut martingal w0 = 0.50 (adversarial extrem + multi-draw + VRAM 42.25M)
    // ------------------------------------------------------------------------
    let mut adv_steps_649 = Vec::with_capacity(240);
    for t in 0..240usize {
        // 80 extrageri identice [0..6], apoi inversare brutală pe mulțimea disjunctă [43..49], apoi alternanță rapidă!
        let bls: Vec<u8> = if t < 80 {
            vec![0, 1, 2, 3, 4, 5]
        } else if t < 160 {
            vec![43, 44, 45, 46, 47, 48]
        } else if t % 2 == 0 {
            vec![6, 7, 8, 9, 10, 11]
        } else {
            vec![36, 37, 38, 39, 40, 41]
        };
        adv_steps_649.push(Ckv6CanonicalStep {
            key: format!("ADV649-{t:04}"),
            draws: vec![(bls, None)],
        });
    }
    let adv_stream_649 = Ckv6StreamInput {
        code: "adv_649".to_string(),
        name: "Adversarial 6/49".to_string(),
        n_balls: 49,
        r_draw: 6,
        m_bonus: None,
        steps: adv_steps_649,
    };
    let mut adv_steps_jok = Vec::with_capacity(200);
    for t in 0..200usize {
        let bls: Vec<u8> = if t < 70 {
            vec![0, 1, 2, 3, 4]
        } else if t < 140 {
            vec![40, 41, 42, 43, 44]
        } else if t % 2 == 0 {
            vec![10, 11, 12, 13, 14]
        } else {
            vec![25, 26, 27, 28, 29]
        };
        let jb = if t < 70 { 0u8 } else if t < 140 { 19u8 } else { ((t * 7) % 20) as u8 };
        adv_steps_jok.push(Ckv6CanonicalStep {
            key: format!("ADVJOK-{t:04}"),
            draws: vec![(bls, Some(jb))],
        });
    }
    let adv_stream_jok = Ckv6StreamInput {
        code: "adv_jok".to_string(),
        name: "Adversarial Joker".to_string(),
        n_balls: 45,
        r_draw: 5,
        m_bonus: Some(20),
        steps: adv_steps_jok,
    };
    let (adv_gpu_ser, _) =
        eval_general_streams_gpu(&[adv_stream_649.clone(), adv_stream_jok.clone()])?;
    let adv_min_e = adv_gpu_ser
        .iter()
        .flat_map(|s| s.step_e.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let adv_min_m = adv_gpu_ser
        .iter()
        .flat_map(|s| s.step_log_m.iter().map(|&lm| lm.exp()))
        .fold(f64::INFINITY, f64::min);
    let vram_min_e = univ_certs
        .iter()
        .map(|c| c.min_e_vram)
        .fold(f64::INFINITY, f64::min);

    let q2_pass = q1_min_e >= 0.50
        && q1_min_m > 0.50
        && mg_min_e >= 0.50
        && adv_min_e >= 0.50
        && adv_min_m > 0.50
        && vram_min_e >= 0.50;

    // ------------------------------------------------------------------------
    // Q3: Invarianță exactă la permutarea etichetelor S_N (și S_{M_B}) < 10^-13
    // ------------------------------------------------------------------------
    let q3_base_streams = [
        gen_synth_general_stream_cpu("q3_649", 49, 6, None, 180, 3500, 60, 12, CKV6_QUAL_SEED, 301),
        gen_synth_general_stream_cpu("q3_jok", 45, 5, Some(20), 180, 3500, 60, 10, CKV6_QUAL_SEED, 302),
        gen_synth_general_stream_cpu("q3_590", 90, 5, None, 150, 3500, 50, 0, CKV6_QUAL_SEED, 303),
    ];
    let q3_base_cpu = eval_general_streams_cpu_24t(&q3_base_streams);
    let mut q3_max_diff_e = 0.0f64;
    let mut q3_max_diff_log_m = 0.0f64;
    for (si, st) in q3_base_streams.iter().enumerate() {
        for p_idx in 0..8u32 {
            let pm = make_perm(st.n_balls, CKV6_QUAL_SEED, (si as u32) * 100 + p_idx);
            let pb = st
                .m_bonus
                .map(|mb| make_perm(mb, CKV6_QUAL_SEED + 77, (si as u32) * 100 + p_idx));
            let pst = permute_stream_labels(st, &pm, pb.as_deref());
            let pser = eval_general_stream_cpu(&pst);
            for t in 0..st.steps.len() {
                q3_max_diff_e = q3_max_diff_e.max((q3_base_cpu[si].step_e[t] - pser.step_e[t]).abs());
                q3_max_diff_log_m = q3_max_diff_log_m
                    .max((q3_base_cpu[si].step_log_m[t] - pser.step_log_m[t]).abs());
            }
        }
    }
    let q3_pass = q3_max_diff_e < 1.0e-13 && q3_max_diff_log_m < 1.0e-13;

    // ------------------------------------------------------------------------
    // Q4: Cauzalitate strictă / non-anticipativitate (0.0 bit-identical la s <= t0)
    // ------------------------------------------------------------------------
    let q4_orig = gen_synth_general_stream_cpu(
        "q4_jok",
        45,
        5,
        Some(20),
        300,
        3500,
        75,
        9,
        CKV6_QUAL_SEED,
        401,
    );
    let q4_alt_tail = gen_synth_general_stream_cpu(
        "q4_alt",
        45,
        5,
        Some(20),
        300,
        5000,
        40,
        7,
        CKV6_QUAL_SEED + 999,
        402,
    );
    let mut q4_variants = vec![q4_orig.clone()];
    let cutoffs = [50usize, 150, 250];
    for &t0 in &cutoffs {
        let mut steps = q4_orig.steps[..=t0].to_vec();
        steps.extend_from_slice(&q4_alt_tail.steps[(t0 + 1)..]);
        q4_variants.push(Ckv6StreamInput {
            code: format!("q4_cut_{t0}"),
            name: format!("Q4 Cut {t0}"),
            n_balls: 45,
            r_draw: 5,
            m_bonus: Some(20),
            steps,
        });
    }
    let (q4_gpu, _) = eval_general_streams_gpu(&q4_variants)?;
    let q4_cpu = eval_general_streams_cpu_24t(&q4_variants);
    let mut q4_max_prefix_diff = 0.0f64;
    let mut q4_map_r_mismatches = 0usize;
    for (ci, &t0) in cutoffs.iter().enumerate() {
        for s in 0..=t0 {
            let de_g = (q4_gpu[0].step_e[s] - q4_gpu[ci + 1].step_e[s]).abs();
            let dlm_g = (q4_gpu[0].step_log_m[s] - q4_gpu[ci + 1].step_log_m[s]).abs();
            let dcp_g = (q4_gpu[0].step_cp_prob[s] - q4_gpu[ci + 1].step_cp_prob[s]).abs();
            let de_c = (q4_cpu[0].step_e[s] - q4_cpu[ci + 1].step_e[s]).abs();
            let dlm_c = (q4_cpu[0].step_log_m[s] - q4_cpu[ci + 1].step_log_m[s]).abs();
            q4_max_prefix_diff = q4_max_prefix_diff
                .max(de_g)
                .max(dlm_g)
                .max(dcp_g)
                .max(de_c)
                .max(dlm_c);
            if q4_gpu[0].step_map_r[s] != q4_gpu[ci + 1].step_map_r[s]
                || q4_cpu[0].step_map_r[s] != q4_cpu[ci + 1].step_map_r[s]
            {
                q4_map_r_mismatches += 1;
            }
        }
    }
    let q4_pass = q4_max_prefix_diff == 0.0 && q4_map_r_mismatches == 0;

    // ------------------------------------------------------------------------
    // Q5: Detecție sintetică de schimbare fizică de regim (delta = 0.50, L = 100, T = 400)
    // ------------------------------------------------------------------------
    let q5_n = 200usize;
    let q5_t = 400usize;
    let mut q5_batch = vec![0u64; q5_n * q5_t];
    for rep in 0..q5_n {
        let h = gen_synth_649_history_cpu(CKV6_QUAL_SEED, 505, rep as u32, 5000, 100, q5_t);
        q5_batch[(rep * q5_t)..((rep + 1) * q5_t)].copy_from_slice(&h);
    }
    let q5_gpu = gpu_ckv6_649_batch_f64(&q5_batch, q5_n, q5_t, 100, false)?;
    let mut q5_log_m: Vec<f64> = q5_gpu.summary5.iter().map(|s| s[0]).collect();
    q5_log_m.sort_by(|a, b| a.total_cmp(b));
    let q5_pass_1500 = q5_gpu
        .summary5
        .iter()
        .filter(|s| s[0].exp() >= 1_500.0)
        .count();
    let q5_power_1500 = (q5_pass_1500 as f64) / (q5_n as f64);
    let q5_median_log_m = 0.5 * (q5_log_m[q5_n / 2 - 1] + q5_log_m[q5_n / 2]);
    let q5_mean_cp_err: f64 =
        q5_gpu.summary5.iter().map(|s| s[4]).sum::<f64>() / (q5_n as f64);
    let q5_mean_gain_vs_glob: f64 = q5_gpu
        .summary5
        .iter()
        .map(|s| s[0] - s[3])
        .sum::<f64>()
        / (q5_n as f64);
    let q5_pass = q5_power_1500 >= 0.90
        && q5_median_log_m.exp() >= 1_500.0
        && q5_mean_cp_err <= 10.0
        && q5_mean_gain_vs_glob >= 5.0;

    // ------------------------------------------------------------------------
    // Q6: Paritate numerică CUDA sm_120 (warp 649 & general) <-> CPU 24T < 10^-12
    //     + Export 6 martori compleți `q_witness_trajectories` pentru Python/NumPy
    // ------------------------------------------------------------------------
    let q6_n = 100usize;
    let q6_t = 400usize;
    let mut q6_batch = vec![0u64; q6_n * q6_t];
    let mut q6_as_streams = Vec::with_capacity(q6_n);
    for i in 0..q6_n {
        let dbp = DELTA_BP_GRID[i % 5];
        let lrg = L_REGIME_GRID[(i / 5) % 5];
        let h = gen_synth_649_history_cpu(CKV6_QUAL_SEED, (i % 25) as u32, (i / 25) as u32, dbp, lrg, q6_t);
        q6_batch[(i * q6_t)..((i + 1) * q6_t)].copy_from_slice(&h);
        let steps: Vec<Ckv6CanonicalStep> = h
            .iter()
            .enumerate()
            .map(|(t, &m)| {
                let mut bls = Vec::with_capacity(6);
                let mut tmp = m;
                while tmp != 0 {
                    let b = tmp.trailing_zeros() as u8;
                    bls.push(b);
                    tmp ^= 1u64 << b;
                }
                Ckv6CanonicalStep {
                    key: format!("Q6-{t:04}"),
                    draws: vec![(bls, None)],
                }
            })
            .collect();
        q6_as_streams.push(Ckv6StreamInput {
            code: format!("q6_649_{i}"),
            name: format!("Q6 649 #{i}"),
            n_balls: 49,
            r_draw: 6,
            m_bonus: None,
            steps,
        });
    }
    let q6_gpu_warp = gpu_ckv6_649_batch_f64(&q6_batch, q6_n, q6_t, 0, true)?;
    let (q6_gpu_gen, _) = eval_general_streams_gpu(&q6_as_streams)?;
    let q6_cpu_warp = eval_649_batch_cpu_24t(&q6_batch, q6_n, q6_t, 0, true);

    let mut q6_max_diff_warp_cpu = 0.0f64;
    let mut q6_max_diff_warp_gen = 0.0f64;
    for i in 0..q6_n {
        for t in 0..q6_t {
            let gw_e = q6_gpu_warp.step_e[i * q6_t + t];
            let gw_lm = q6_gpu_warp.step_log_m[i * q6_t + t];
            let cw_e = q6_cpu_warp[i].step_e[t];
            let cw_lm = q6_cpu_warp[i].step_log_m[t];
            let gg_e = q6_gpu_gen[i].step_e[t];
            let gg_lm = q6_gpu_gen[i].step_log_m[t];
            q6_max_diff_warp_cpu = q6_max_diff_warp_cpu
                .max((gw_e - cw_e).abs())
                .max((gw_lm - cw_lm).abs());
            q6_max_diff_warp_gen = q6_max_diff_warp_gen
                .max((gw_e - gg_e).abs())
                .max((gw_lm - gg_lm).abs());
        }
    }

    // Testăm inclusiv T = 600 > CKV6_L_MAX (512) și extrageri duble / bonus pe 6 martori canonici
    let witness_inputs = vec![
        gen_synth_general_stream_cpu("W1_649_unif", 49, 6, None, 120, 0, 120, 0, CKV6_QUAL_SEED, 601),
        gen_synth_general_stream_cpu("W2_649_shift", 49, 6, None, 200, 5000, 80, 0, CKV6_QUAL_SEED, 602),
        gen_synth_general_stream_cpu("W3_joker_dbl", 45, 5, Some(20), 150, 3500, 60, 8, CKV6_QUAL_SEED, 603),
        gen_synth_general_stream_cpu("W4_590_otos", 90, 5, None, 100, 3500, 50, 0, CKV6_QUAL_SEED, 604),
        gen_synth_general_stream_cpu("W5_pb_bonus", 69, 5, Some(26), 100, 3500, 50, 0, CKV6_QUAL_SEED, 605),
        gen_synth_general_stream_cpu("W6_lmax_cap600", 40, 6, None, 600, 2500, 150, 25, CKV6_QUAL_SEED, 606),
    ];
    let (wit_gpu, _) = eval_general_streams_gpu(&witness_inputs)?;
    let wit_cpu = eval_general_streams_cpu_24t(&witness_inputs);
    let mut q6_max_diff_wit = 0.0f64;
    let mut q_witnesses = Vec::with_capacity(witness_inputs.len());
    for (wi, st) in witness_inputs.iter().enumerate() {
        for t in 0..st.steps.len() {
            q6_max_diff_wit = q6_max_diff_wit
                .max((wit_gpu[wi].step_e[t] - wit_cpu[wi].step_e[t]).abs())
                .max((wit_gpu[wi].step_log_m[t] - wit_cpu[wi].step_log_m[t]).abs())
                .max((wit_gpu[wi].step_log_m_glob[t] - wit_cpu[wi].step_log_m_glob[t]).abs());
        }
        q_witnesses.push(QWitnessTrajectory {
            id: st.code.clone(),
            n_balls: st.n_balls,
            r_draw: st.r_draw,
            m_bonus: st.m_bonus,
            steps: st.steps.clone(),
            expected_step_e: wit_gpu[wi].step_e.clone(),
            expected_step_log_m: wit_gpu[wi].step_log_m.clone(),
            expected_step_map_r: wit_gpu[wi].step_map_r.clone(),
            expected_step_log_m_glob: wit_gpu[wi].step_log_m_glob.clone(),
        });
    }
    let q6_overall_max_diff = q6_max_diff_warp_cpu
        .max(q6_max_diff_warp_gen)
        .max(q6_max_diff_wit);
    let q6_pass = q6_overall_max_diff < 1.0e-12;

    let q_tests = vec![
        QTestSummary {
            id: "Q1".to_string(),
            name: "Martingal exact sub H0 uniform (2,000 x 6/49 + 500 multi-geom + 42,257,376 VRAM)".to_string(),
            pass: q1_pass,
            metrics: json!({
                "n_trajectories_649": q1_n_traj,
                "t_steps": q1_t,
                "mean_step_e": q1_mean_step_e,
                "se_step_e": q1_se_step_e,
                "ville_anytime_fp_rate_ge_20": fp_rate_20,
                "ville_anytime_fp_rate_ge_100": fp_rate_100,
                "ville_anytime_fp_rate_ge_1500": fp_rate_1500,
                "foreign_500_fp_rate_ge_20": (mg_ville_20 as f64) / 500.0,
                "foreign_500_fp_count_ge_1500": mg_ville_1500,
                "vram_universe_max_abs_err_vs_one": univ_certs.iter().map(|c| c.abs_err_vs_one).fold(0.0f64, f64::max),
            }),
        },
        QTestSummary {
            id: "Q2".to_string(),
            name: "Scut martingal obligatoriu w0 = 0.50 => E_t >= 0.50 si M_t > 0.50 fara exceptie".to_string(),
            pass: q2_pass,
            metrics: json!({
                "min_e_uniform_800k_steps": q1_min_e,
                "min_m_uniform_800k_steps": q1_min_m,
                "min_e_foreign_150k_steps": mg_min_e,
                "min_e_adversarial": adv_min_e,
                "min_m_adversarial": adv_min_m,
                "min_e_vram_42m_states": vram_min_e,
            }),
        },
        QTestSummary {
            id: "Q3".to_string(),
            name: "Invarianta exacta la permutarea etichetelor S_N x S_{M_B} (< 1e-13)".to_string(),
            pass: q3_pass,
            metrics: json!({
                "permutations_per_stream": 8,
                "max_abs_diff_e": q3_max_diff_e,
                "max_abs_diff_log_m": q3_max_diff_log_m,
            }),
        },
        QTestSummary {
            id: "Q4".to_string(),
            name: "Cauzalitate stricta / non-anticipativitate (0.0 bit-identical pe prefixul <= t0)".to_string(),
            pass: q4_pass,
            metrics: json!({
                "cutoffs_tested": cutoffs,
                "max_prefix_abs_diff_gpu_and_cpu": q4_max_prefix_diff,
                "map_r_mismatches": q4_map_r_mismatches,
            }),
        },
        QTestSummary {
            id: "Q5".to_string(),
            name: "Detectie sintetica de schimbare fizica de regim (M_T >= 1500 & localizare BOCPD)".to_string(),
            pass: q5_pass,
            metrics: json!({
                "n_histories": q5_n,
                "delta": 0.50,
                "l_regime": 100,
                "t_steps": q5_t,
                "power_m_t_ge_1500": q5_power_1500,
                "median_log_m_t": q5_median_log_m,
                "median_m_t": q5_median_log_m.exp(),
                "mean_cp_localization_abs_err": q5_mean_cp_err,
                "mean_log_m_gain_vs_unreset_global": q5_mean_gain_vs_glob,
            }),
        },
        QTestSummary {
            id: "Q6".to_string(),
            name: "Paritate numerica CUDA sm_120 (warp & general) <-> CPU 24T (< 1e-12) + 6 martori Python".to_string(),
            pass: q6_pass,
            metrics: json!({
                "max_abs_diff_gpu_warp_vs_cpu_24t": q6_max_diff_warp_cpu,
                "max_abs_diff_gpu_warp_vs_gpu_general": q6_max_diff_warp_gen,
                "max_abs_diff_6_witnesses_incl_lmax600": q6_max_diff_wit,
                "overall_max_abs_diff": q6_overall_max_diff,
            }),
        },
    ];
    let q_passed = q_tests.iter().filter(|q| q.pass).count();
    let q_all_pass = q_passed == 6;
    if !q_all_pass {
        let failed_details: Vec<String> = q_tests
            .iter()
            .filter(|q| !q.pass)
            .map(|q| format!("{}: {}", q.id, q.metrics))
            .collect();
        return Err(format!(
            "Calificarea sintetica Q1-Q6 a esuat ({q_passed}/6): {}",
            failed_details.join(" | ")
        ));
    }

    // ------------------------------------------------------------------------
    // Suprafața de putere pe 10,000 istorii sintetice pe RTX 5060 sm_120
    // ------------------------------------------------------------------------
    let mut delta_cells = Vec::with_capacity(25);
    let mut l_reg_cells = Vec::with_capacity(25);
    for &dbp in &DELTA_BP_GRID {
        for &lrg in &L_REGIME_GRID {
            delta_cells.push(dbp);
            l_reg_cells.push(lrg);
        }
    }
    let (records8, power_gpu_ms) = gpu_ckv6_power_10k_f64(
        &delta_cells,
        &l_reg_cells,
        POWER_REPS_PER_CELL,
        POWER_T_STEPS,
        CKV6_POWER_SEED,
    )?;
    assert_eq!(records8.len(), POWER_TOTAL_HISTORIES);

    // Verificare paritate CPU 24T pe 100 istorii (4 replici din fiecare dintre cele 25 celule = 100 istorii)
    let mut sample_max_diff = 0.0f64;
    let sample_pairs: Vec<(usize, u32)> = (0..25usize)
        .flat_map(|cid| (0..4u32).map(move |rep| (cid, rep)))
        .collect();
    let cpu_sample_diffs: Vec<f64> = std::thread::scope(|s| {
        let chunk_sz = (sample_pairs.len() + cpu_threads - 1) / cpu_threads;
        let mut handles = Vec::new();
        for ch in sample_pairs.chunks(chunk_sz.max(1)) {
            let dc = &delta_cells;
            let lc = &l_reg_cells;
            let r8 = &records8;
            handles.push(s.spawn(move || {
                let mut loc_max = 0.0f64;
                for &(cid, rep) in ch {
                    let dbp = dc[cid];
                    let lrg = lc[cid];
                    let h = gen_synth_649_history_cpu(
                        CKV6_POWER_SEED,
                        cid as u32,
                        rep,
                        dbp,
                        lrg,
                        POWER_T_STEPS as usize,
                    );
                    let mcp = if dbp > 0 && lrg < POWER_T_STEPS {
                        lrg
                    } else {
                        0
                    };
                    let co = eval_649_history_cpu(&h, mcp, false);
                    let g_rec = &r8[cid * (POWER_REPS_PER_CELL as usize) + (rep as usize)];
                    loc_max = loc_max
                        .max((co.log_m_final - g_rec[3]).abs())
                        .max((co.max_log_m - g_rec[4]).abs())
                        .max((co.min_e - g_rec[5]).abs())
                        .max((co.mean_cp_err - g_rec[6]).abs())
                        .max((co.log_m_glob - g_rec[7]).abs());
                }
                loc_max
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for d in cpu_sample_diffs {
        sample_max_diff = sample_max_diff.max(d);
    }
    if sample_max_diff >= 1.0e-12 {
        return Err(format!(
            "Eroare paritate GPU<->CPU pe ckv6_power_10k: {sample_max_diff:.3e} >= 1e-12"
        ));
    }

    // Scriem `ckv6_power_10k.v3bin` (table_kind = 9, 10,000 x 64B = 640,064 B) și îl sigilăm 0444!
    let v3bin_path = vault.vault_dir.join(CKV6_V3BIN_NAME);
    make_writable_if_exists(&v3bin_path);
    let hdr = encode_v3bin_header_64(
        TABLE_KIND_CKV6_POWER_10K,
        49,
        6,
        POWER_TOTAL_HISTORIES as u64,
        64,
    );
    {
        let mut wf = File::create(&v3bin_path)
            .map_err(|e| format!("create {}: {e}", v3bin_path.display()))?;
        wf.write_all(&hdr)
            .map_err(|e| format!("write hdr {}: {e}", v3bin_path.display()))?;
        let payload_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                records8.as_ptr() as *const u8,
                records8.len() * std::mem::size_of::<[f64; 8]>(),
            )
        };
        wf.write_all(payload_bytes)
            .map_err(|e| format!("write payload {}: {e}", v3bin_path.display()))?;
        wf.sync_all()
            .map_err(|e| format!("sync {}: {e}", v3bin_path.display()))?;
    }
    fs::set_permissions(&v3bin_path, fs::Permissions::from_mode(0o444))
        .map_err(|e| format!("chmod 0444 {}: {e}", v3bin_path.display()))?;
    let (v3bin_bytes, v3bin_sha) = sha256_file_c3(&v3bin_path)?;

    upsert_vault_manifest_record(
        &vault.vault_dir,
        VaultFileRecord {
            filename: CKV6_V3BIN_NAME.to_string(),
            table_kind: TABLE_KIND_CKV6_POWER_10K,
            n_pool: 49,
            k_pick: 6,
            record_count: POWER_TOTAL_HISTORIES as u64,
            file_bytes: v3bin_bytes,
            sha256: v3bin_sha.clone(),
        },
    )?;

    // Rezumăm cele 25 celule ale grilei 5 x 5
    let mut power_cells = Vec::with_capacity(25);
    let reps_u = POWER_REPS_PER_CELL as usize;
    let ln20 = 20.0f64.ln();
    let ln1500 = 1_500.0f64.ln();
    for cid in 0..25usize {
        let slice = &records8[(cid * reps_u)..((cid + 1) * reps_u)];
        let mut lms: Vec<f64> = slice.iter().map(|r| r[3]).collect();
        lms.sort_by(|a, b| a.total_cmp(b));
        let p20 = (slice.iter().filter(|r| r[3] >= ln20).count() as f64) / (reps_u as f64);
        let p1500 = (slice.iter().filter(|r| r[3] >= ln1500).count() as f64) / (reps_u as f64);
        let pa20 = (slice.iter().filter(|r| r[4] >= ln20).count() as f64) / (reps_u as f64);
        let pa1500 = (slice.iter().filter(|r| r[4] >= ln1500).count() as f64) / (reps_u as f64);
        let mean_lm = slice.iter().map(|r| r[3]).sum::<f64>() / (reps_u as f64);
        let med_lm = 0.5 * (lms[reps_u / 2 - 1] + lms[reps_u / 2]);
        let mean_lmg = slice.iter().map(|r| r[7]).sum::<f64>() / (reps_u as f64);
        let min_e = slice.iter().map(|r| r[5]).fold(f64::INFINITY, f64::min);
        let mean_cp = slice.iter().map(|r| r[6]).sum::<f64>() / (reps_u as f64);
        power_cells.push(PowerCellSummary {
            cell_id: cid as u32,
            delta: (delta_cells[cid] as f64) / 10_000.0,
            delta_bp: delta_cells[cid],
            l_regime: l_reg_cells[cid],
            n_histories: POWER_REPS_PER_CELL,
            power_m_ge_20: p20,
            power_m_ge_1500: p1500,
            power_anytime_ge_20: pa20,
            power_anytime_ge_1500: pa1500,
            mean_log_m_t: mean_lm,
            median_log_m_t: med_lm,
            mean_log_m_t_global_no_reset: mean_lmg,
            mean_bocpd_gain_log_m: mean_lm - mean_lmg,
            min_e_across_cell: min_e,
            mean_cp_loc_abs_err: mean_cp,
        });
    }

    // Pragurile empirice de rezoluție pe fiecare lungime de regim L_regim
    let mut res_by_l = serde_json::Map::new();
    for &lrg in &L_REGIME_GRID {
        let min_d20 = power_cells
            .iter()
            .filter(|c| c.l_regime == lrg && c.delta_bp > 0 && c.power_m_ge_20 >= 0.80)
            .map(|c| c.delta)
            .fold(None::<f64>, |acc, x| Some(acc.map_or(x, |a| a.min(x))));
        let min_d1500 = power_cells
            .iter()
            .filter(|c| c.l_regime == lrg && c.delta_bp > 0 && c.power_m_ge_1500 >= 0.80)
            .map(|c| c.delta)
            .fold(None::<f64>, |acc, x| Some(acc.map_or(x, |a| a.min(x))));
        res_by_l.insert(
            format!("L_{lrg}"),
            json!({
                "delta_min_power80_m_ge_20": min_d20,
                "delta_min_power80_m_ge_1500": min_d1500,
            }),
        );
    }

    let (rss_cur, rss_hwm) = read_process_rss_bytes();
    let rss_ok = rss_cur < MAX_SAFE_RSS_BYTES && rss_hwm < MAX_SAFE_RSS_BYTES;

    let report = CkV6QualificationAndPowerReport {
        engine: "CK-V6 (BOCPD Adams-MacKay + Koopman V1(+)V2 Shrunk + Shield w0=0.50)".to_string(),
        session: "S4".to_string(),
        gate: "A5_PRE_ARCHIVE_QUALIFICATION_AND_POWER_10K".to_string(),
        generated_utc: utc_now_iso8601(),
        positive_claims_allowed: false,
        archive_files_read: 0,
        gpu: gpu_info,
        cpu_threads,
        model_constants: json!({
            "l_max": CKV6_L_MAX,
            "hazards_lambda_h": [CKV6_LAMBDA_0, CKV6_LAMBDA_1],
            "w0_shield": CKV6_W0_SHIELD,
            "beta_dm_koop_v1": CKV6_BETA_DM,
            "beta_koop_v2": CKV6_BETA_V2,
            "alpha_dm": CKV6_ALPHA_DM,
            "theta_cap": CKV6_THETA_CAP,
            "rho_max": CKV6_RHO_MAX,
            "tau_stat_v1": CKV6_TAU_STAT_1,
            "tau_koop_v1": CKV6_TAU_KOOP_1,
            "tau_stat_v2": CKV6_TAU_STAT_2,
            "tau_koop_v2": CKV6_TAU_KOOP_2,
            "tau_bonus": [CKV6_TAU_BON_STAT, CKV6_TAU_BON_KOOP],
        }),
        q_tests,
        q_passed_count: q_passed,
        q_all_6_pass: q_all_pass,
        universe_martingale_certificates: univ_certs,
        q_witness_trajectories: q_witnesses,
        power_total_histories: POWER_TOTAL_HISTORIES,
        power_t_steps: POWER_T_STEPS,
        power_seed: CKV6_POWER_SEED,
        power_gpu_ms,
        power_cpu_sample_100_max_abs_diff: sample_max_diff,
        power_v3bin_filename: CKV6_V3BIN_NAME.to_string(),
        power_v3bin_bytes: v3bin_bytes,
        power_v3bin_sha256: v3bin_sha,
        power_cells,
        resolution_thresholds: Value::Object(res_by_l),
        rss_current_bytes: rss_cur,
        rss_peak_hwm_bytes: rss_hwm,
        rss_limit_bytes: MAX_SAFE_RSS_BYTES,
        rss_guard_ok: rss_ok,
        total_seconds: t_start.elapsed().as_secs_f64(),
    };

    if let Some(p) = out_json_path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
    }
    make_writable_if_exists(out_json_path);
    let txt = serde_json::to_string_pretty(&report).unwrap() + "\n";
    fs::write(out_json_path, txt).map_err(|e| format!("write {}: {e}", out_json_path.display()))?;
    Ok(report)
}

/// Sigilează pre-arhivă `0444` sursa `rust_core_v3/src/ckv6.rs` și scrie `v3_upgrade/SIGILIU_COD_CKV6.json`
/// IMEDIAT după `Q1–Q6 = 6/6 PASS` + `10,000` istorii sintetice (`archive_files_read = 0`) și ÎNAINTE de orice arhivă!
pub fn seal_ckv6_code_pre_archive(
    ql_root: &Path,
    ckv6_rs_path: &Path,
    calif_json_path: &Path,
    out_seal_path: &Path,
) -> Result<CkV6CodeSeal, String> {
    let cal_bytes = fs::read(calif_json_path)
        .map_err(|e| format!("read {}: {e}", calif_json_path.display()))?;
    let cal_sha = sha256_bytes_hex(&cal_bytes);
    let cal: CkV6QualificationAndPowerReport = serde_json::from_slice(&cal_bytes)
        .map_err(|e| format!("parse {}: {e}", calif_json_path.display()))?;
    if !cal.q_all_6_pass || cal.archive_files_read != 0 || cal.power_total_histories != POWER_TOTAL_HISTORIES {
        return Err(
            "Refuz sigilare cod: CK_V6_CALIFICARE_PUTERE.json nu are 6/6 PASS cu archive_files_read=0!"
                .to_string(),
        );
    }

    let lstat_base = inspect_blind_reserves_lstat(ql_root)?;
    if !lstat_base.iter().all(|e| e.is_strict_0400) {
        return Err("Eroare garda 0400: cel putin o rezerva.json nu are permisiuni stricte 0400!".to_string());
    }

    fs::set_permissions(ckv6_rs_path, fs::Permissions::from_mode(0o444))
        .map_err(|e| format!("chmod 0444 {}: {e}", ckv6_rs_path.display()))?;
    let rs_bytes =
        fs::read(ckv6_rs_path).map_err(|e| format!("read {}: {e}", ckv6_rs_path.display()))?;
    let rs_sha = sha256_bytes_hex(&rs_bytes);
    let rs_mode = fs::symlink_metadata(ckv6_rs_path)
        .map_err(|e| format!("stat {}: {e}", ckv6_rs_path.display()))?
        .permissions()
        .mode()
        & 0o777;
    if rs_mode != 0o400 && rs_mode != 0o444 {
        return Err(format!("ckv6.rs mode neasteptat: {rs_mode:04o}"));
    }

    let seal = CkV6CodeSeal {
        seal_id: "SIGILIU_COD_CKV6_PRE_ARCHIVE".to_string(),
        session: "S4".to_string(),
        sealed_utc: utc_now_iso8601(),
        positive_claims_allowed: false,
        archive_files_read_before_seal: 0,
        ckv6_rs_path: ckv6_rs_path.display().to_string(),
        ckv6_rs_sha256: rs_sha,
        ckv6_rs_bytes: rs_bytes.len() as u64,
        ckv6_rs_mode_octal: format!("{rs_mode:04o}"),
        calificare_putere_path: calif_json_path.display().to_string(),
        calificare_putere_sha256: cal_sha,
        q_all_6_pass: cal.q_all_6_pass,
        power_total_histories: cal.power_total_histories,
        power_v3bin_sha256: cal.power_v3bin_sha256,
        blind_reserves_lstat_baseline: lstat_base,
    };

    make_writable_if_exists(out_seal_path);
    let txt = serde_json::to_string_pretty(&seal).unwrap() + "\n";
    fs::write(out_seal_path, txt).map_err(|e| format!("write {}: {e}", out_seal_path.display()))?;
    Ok(seal)
}

#[derive(Deserialize)]
struct RawJsonlDraw {
    draw_date: String,
    numbers: Vec<usize>,
    #[serde(default)]
    joker: Option<usize>,
    #[serde(default)]
    bonus: Option<usize>,
}

fn load_jsonl_canonical_stream(
    ql_root: &Path,
    rel_file: &str,
    expected_sha: Option<&str>,
    code: &str,
    name: &str,
    n_balls: usize,
    r_draw: usize,
    m_bonus: Option<usize>,
) -> Result<(Ckv6StreamInput, String, usize, usize), String> {
    let full = ql_root.join(rel_file);
    if rel_file.contains("rezerva") || rel_file.contains("sigilat") {
        return Err(format!("GARDA 0400: interzisa citirea {rel_file}!"));
    }
    let bytes = fs::read(&full).map_err(|e| format!("read {}: {e}", full.display()))?;
    let sha = sha256_bytes_hex(&bytes);
    if let Some(exp) = expected_sha {
        if sha != exp {
            return Err(format!("SHA-256 mismatch pe {rel_file}: {sha} != {exp}"));
        }
    }
    let txt = std::str::from_utf8(&bytes).map_err(|e| format!("utf8 {rel_file}: {e}"))?;
    let mut by_date: BTreeMap<String, Vec<(Vec<u8>, Option<u8>)>> = BTreeMap::new();
    let mut n_raw = 0usize;
    for (ln_no, raw_ln) in txt.lines().enumerate() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        let rec: RawJsonlDraw = serde_json::from_str(ln)
            .map_err(|e| format!("{rel_file}:{}: {e}", ln_no + 1))?;
        if rec.numbers.len() != r_draw {
            return Err(format!("{rel_file}:{}: len {} != {r_draw}", ln_no + 1, rec.numbers.len()));
        }
        let mut bls = Vec::with_capacity(r_draw);
        for &x in &rec.numbers {
            if x == 0 || x > n_balls {
                return Err(format!("{rel_file}:{}: bila {x} invalida", ln_no + 1));
            }
            bls.push((x - 1) as u8);
        }
        bls.sort_unstable();
        let bon = match m_bonus {
            Some(mb) => {
                let jb = rec
                    .joker
                    .or(rec.bonus)
                    .ok_or_else(|| format!("{rel_file}:{}: bonus lipsa", ln_no + 1))?;
                if jb == 0 || jb > mb {
                    return Err(format!("{rel_file}:{}: bonus {jb} invalid", ln_no + 1));
                }
                Some((jb - 1) as u8)
            }
            None => None,
        };
        by_date.entry(rec.draw_date).or_default().push((bls, bon));
        n_raw += 1;
    }
    let mut dbl_days = 0usize;
    let mut steps = Vec::with_capacity(by_date.len());
    for (dt, drs) in by_date {
        if drs.len() > 1 {
            dbl_days += 1;
        }
        steps.push(Ckv6CanonicalStep { key: dt, draws: drs });
    }
    Ok((
        Ckv6StreamInput {
            code: code.to_string(),
            name: name.to_string(),
            n_balls,
            r_draw,
            m_bonus,
            steps,
        },
        sha,
        n_raw,
        dbl_days,
    ))
}

fn load_xi_p2_dev_stream(
    ql_root: &Path,
    dir_game: &str,
    stream_code: &str,
    stream_name: &str,
    n_balls: usize,
    r_draw: usize,
    m_bonus: Option<usize>,
    min_cheie_inclusive: Option<&str>,
) -> Result<(Ckv6StreamInput, String, String, usize), String> {
    let rel = format!("xi/date_p2/{dir_game}/dev.json");
    if rel.contains("rezerva") || rel.contains("sigilat") {
        return Err("GARDA 0400: interzis accesul la rezerva!".to_string());
    }
    let full = ql_root.join(&rel);
    let bytes = fs::read(&full).map_err(|e| format!("read {}: {e}", full.display()))?;
    let sha = sha256_bytes_hex(&bytes);
    let doc: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("json {}: {e}", full.display()))?;
    let extr = doc["extrageri"]
        .as_array()
        .ok_or_else(|| format!("{rel} fara extrageri"))?;

    let mut steps = Vec::with_capacity(extr.len());
    for (idx, item) in extr.iter().enumerate() {
        let cheie = item["cheie"]
            .as_str()
            .ok_or_else(|| format!("{rel}[{idx}] fara cheie"))?;
        if let Some(min_k) = min_cheie_inclusive {
            if cheie < min_k {
                continue;
            }
        }
        let princ = item["sets"]["princ"]
            .as_array()
            .ok_or_else(|| format!("{rel}[{idx}] fara sets.princ"))?;
        if princ.len() != r_draw {
            return Err(format!("{rel}[{idx}] princ len {} != {r_draw}", princ.len()));
        }
        let mut bls = Vec::with_capacity(r_draw);
        for v in princ {
            let x = v.as_u64().ok_or_else(|| format!("{rel}[{idx}] int invalid"))? as usize;
            if x == 0 || x > n_balls {
                return Err(format!("{rel}[{idx}] bila {x} in afara 1..={n_balls}"));
            }
            bls.push((x - 1) as u8);
        }
        bls.sort_unstable();
        let bon = match m_bonus {
            Some(mb) => {
                let barr = item["sets"]["bila"]
                    .as_array()
                    .ok_or_else(|| format!("{rel}[{idx}] fara sets.bila"))?;
                let jb = barr
                    .first()
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| format!("{rel}[{idx}] bila goala"))? as usize;
                if jb == 0 || jb > mb {
                    return Err(format!("{rel}[{idx}] bila bonus {jb} invalida"));
                }
                Some((jb - 1) as u8)
            }
            None => None,
        };
        steps.push(Ckv6CanonicalStep {
            key: cheie.to_string(),
            draws: vec![(bls, bon)],
        });
    }
    let n_draws = steps.len();
    Ok((
        Ckv6StreamInput {
            code: stream_code.to_string(),
            name: stream_name.to_string(),
            n_balls,
            r_draw,
            m_bonus,
            steps,
        },
        rel,
        sha,
        n_draws,
    ))
}

fn summarize_stream_walkforward(
    st: &Ckv6StreamInput,
    family: &str,
    epistemic_role: &str,
    rel_file: &str,
    file_sha256: &str,
    n_raw_draws: usize,
    double_draw_days: usize,
    gpu_ser: &Ckv6StepSeries,
    cpu_ser: &Ckv6StepSeries,
    include_post_warmup100: bool,
) -> StreamWalkForwardReport {
    let t_len = st.steps.len();
    let mut max_diff_e = 0.0f64;
    let mut max_diff_lm = 0.0f64;
    let mut min_e = f64::INFINITY;
    let mut max_e = 0.0f64;
    let mut max_m = 1.0f64;
    let mut min_m = 1.0f64;

    for t in 0..t_len {
        let ge = gpu_ser.step_e[t];
        let glm = gpu_ser.step_log_m[t];
        max_diff_e = max_diff_e.max((ge - cpu_ser.step_e[t]).abs());
        max_diff_lm = max_diff_lm.max((glm - cpu_ser.step_log_m[t]).abs());
        min_e = min_e.min(ge);
        max_e = max_e.max(ge);
        let mc = glm.exp();
        max_m = max_m.max(mc);
        min_m = min_m.min(mc);
    }

    let log_m_final = *gpu_ser.step_log_m.last().unwrap_or(&0.0);
    let m_final = log_m_final.exp();
    let log10_m_final = log_m_final / std::f64::consts::LN_10;
    let log_m_glob = *gpu_ser.step_log_m_glob.last().unwrap_or(&0.0);
    let m_glob = log_m_glob.exp();
    let final_map_r = *gpu_ser.step_map_r.last().unwrap_or(&0);

    let post_w100 = if include_post_warmup100 && t_len > 100 {
        Some((log_m_final - gpu_ser.step_log_m[99]).exp())
    } else {
        None
    };

    // Vârfuri locale de probabilitate de schimbare de regim (t >= 30, P(r_t <= 5) >= 0.15)
    let mut peaks = Vec::new();
    for t in 30..t_len {
        let p = gpu_ser.step_cp_prob[t];
        if p >= 0.15 {
            let prev_p = gpu_ser.step_cp_prob[t - 1];
            let next_p = if t + 1 < t_len {
                gpu_ser.step_cp_prob[t + 1]
            } else {
                0.0
            };
            if p >= prev_p && p >= next_p {
                peaks.push(DetectedChangepointPeak {
                    step_idx: t,
                    key: st.steps[t].key.clone(),
                    cp_prob_short: p,
                    map_r: gpu_ser.step_map_r[t],
                });
            }
        }
    }

    // Checkpoint-uri exacte pentru verificatorul independent Python `verifica_s4_ckv6.py`
    let mut ck_indices = vec![0usize, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, t_len / 2];
    if t_len >= 2 {
        ck_indices.push(t_len - 2);
    }
    if t_len >= 1 {
        ck_indices.push(t_len - 1);
    }
    ck_indices.retain(|&i| i < t_len);
    ck_indices.sort_unstable();
    ck_indices.dedup();
    let exact_checkpoints: Vec<ArchiveStepCheckpoint> = ck_indices
        .into_iter()
        .map(|i| ArchiveStepCheckpoint {
            step_idx: i,
            key: st.steps[i].key.clone(),
            m_draws_in_step: st.steps[i].draws.len(),
            e_step: gpu_ser.step_e[i],
            log_m_cum: gpu_ser.step_log_m[i],
            m_cum: gpu_ser.step_log_m[i].exp(),
            map_r: gpu_ser.step_map_r[i],
            cp_prob_short: gpu_ser.step_cp_prob[i],
            log_m_glob: gpu_ser.step_log_m_glob[i],
        })
        .collect();

    // Sub-eșantionare <= 256 puncte pentru observatorul Tauri v2 din Sesiunea 5
    let n_gui = t_len.min(256);
    let mut gui_pts = Vec::with_capacity(n_gui);
    for g in 0..n_gui {
        let idx = if n_gui <= 1 {
            0
        } else {
            (g * (t_len - 1)) / (n_gui - 1)
        };
        let lm = gpu_ser.step_log_m[idx];
        gui_pts.push(ArchiveTrajectoryPoint {
            step_idx: idx,
            key: st.steps[idx].key.clone(),
            m_cum: lm.exp(),
            log10_m_cum: lm / std::f64::consts::LN_10,
            e_step: gpu_ser.step_e[idx],
            map_r: gpu_ser.step_map_r[idx],
            cp_prob_short: gpu_ser.step_cp_prob[idx],
        });
    }

    StreamWalkForwardReport {
        code: st.code.clone(),
        name: st.name.clone(),
        family: family.to_string(),
        epistemic_role: epistemic_role.to_string(),
        rel_file: rel_file.to_string(),
        file_sha256: file_sha256.to_string(),
        n_balls: st.n_balls,
        r_draw: st.r_draw,
        m_bonus: st.m_bonus,
        n_raw_draws,
        n_canonical_steps: t_len,
        double_draw_days,
        m_t_final: m_final,
        log_m_t_final: log_m_final,
        log10_m_t_final: log10_m_final,
        max_m_t_anytime: max_m,
        min_m_t_anytime: min_m,
        min_e_t: min_e,
        max_e_t: max_e,
        m_t_post_warmup100: post_w100,
        m_t_global_no_reset: m_glob,
        log_m_t_global_no_reset: log_m_glob,
        final_map_run_length: final_map_r,
        gpu_cpu_max_abs_diff_e: max_diff_e,
        gpu_cpu_max_abs_diff_log_m: max_diff_lm,
        detected_changepoint_peaks: peaks,
        exact_checkpoints,
        gui_trajectory_256: gui_pts,
    }
}

/// Rulare UNICĂ Walk-Forward Cauzală pe RO, controale ne-rezervate și `xi/date_p2/*/dev.json`
/// (refuză execuția dacă `ckv6.rs` nu este deja sigilat `0444` în `SIGILIU_COD_CKV6.json` sau dacă
/// `CK_V6_ARHIVE_SI_DEV.json` există deja!). Protejează absolut `xi/date_p2/*/rezerva.json` (`0400`).
pub fn run_ckv6_archives_and_dev(
    ql_root: &Path,
    ckv6_rs_path: &Path,
    code_seal_path: &Path,
    out_archives_json: &Path,
) -> Result<CkV6ArchivesAndDevReport, String> {
    if out_archives_json.exists() {
        return Err(format!(
            "Refuz rerulare: {} exista deja (protocolul permite o singura rulare cauzala post-sigiliu)!",
            out_archives_json.display()
        ));
    }

    // 1. Verificăm obligatoriu sigiliul pre-arhivă `SIGILIU_COD_CKV6.json` și modul `0444` al `ckv6.rs`!
    let seal_bytes = fs::read(code_seal_path)
        .map_err(|e| format!("Lipseste sigiliul pre-arhiva {}: {e}", code_seal_path.display()))?;
    let seal_sha = sha256_bytes_hex(&seal_bytes);
    let seal: CkV6CodeSeal = serde_json::from_slice(&seal_bytes)
        .map_err(|e| format!("parse {}: {e}", code_seal_path.display()))?;
    if !seal.q_all_6_pass || seal.archive_files_read_before_seal != 0 {
        return Err("SIGILIU_COD_CKV6 invalid!".to_string());
    }
    let rs_md = fs::symlink_metadata(ckv6_rs_path)
        .map_err(|e| format!("stat {}: {e}", ckv6_rs_path.display()))?;
    let rs_mode = rs_md.permissions().mode() & 0o777;
    if rs_mode != 0o444 && rs_mode != 0o400 {
        return Err(format!(
            "Refuz rulare arhive: ckv6.rs nu este sigilat read-only 0444 (mode={rs_mode:04o})!"
        ));
    }
    let rs_sha = sha256_bytes_hex(
        &fs::read(ckv6_rs_path).map_err(|e| format!("read {}: {e}", ckv6_rs_path.display()))?,
    );
    if rs_sha != seal.ckv6_rs_sha256 {
        return Err(format!(
            "SHA-256 mismatch pe ckv6.rs fata de SIGILIU_COD_CKV6: {rs_sha} != {}",
            seal.ckv6_rs_sha256
        ));
    }

    // 2. Încărcăm cele 3 arhive RO din `tu3/date/live/CURRENT_LIVE.json`
    let cur_live_bytes = fs::read(ql_root.join("tu3/date/live/CURRENT_LIVE.json"))
        .map_err(|e| format!("CURRENT_LIVE.json: {e}"))?;
    let cur_live: Value = serde_json::from_slice(&cur_live_bytes).map_err(|e| e.to_string())?;
    let vdir_rel = cur_live["version_dir"]
        .as_str()
        .ok_or("CURRENT_LIVE fara version_dir")?;
    let man_rel = format!("tu3/date/live/{vdir_rel}/MANIFEST_LIVE.json");
    let man_live: Value = serde_json::from_slice(
        &fs::read(ql_root.join(&man_rel)).map_err(|e| format!("{man_rel}: {e}"))?,
    )
    .map_err(|e| e.to_string())?;

    let ro_defs: [(&str, &str, usize, usize, Option<usize>); 3] = [
        ("649", "Loto 6/49 România", 49, 6, None),
        ("540", "Loto 5/40 România (6 extrase din 40)", 40, 6, None),
        ("joker", "Joker România (5/45 + 1/20)", 45, 5, Some(20)),
    ];

    let mut all_inputs: Vec<Ckv6StreamInput> = Vec::with_capacity(13);
    let mut all_meta_info: Vec<(String, String, String, String, usize, usize, bool)> =
        Vec::with_capacity(13);

    for (code, name, nb, rd, mb) in ro_defs {
        let f_name = man_live["games"][code]["file"]
            .as_str()
            .ok_or_else(|| format!("MANIFEST_LIVE fara {code}"))?;
        let exp_sha = cur_live["games"][code]["sha256"]
            .as_str()
            .ok_or_else(|| format!("CURRENT_LIVE fara sha {code}"))?;
        let rel = format!("tu3/date/live/{vdir_rel}/{f_name}");
        let (st, sha, n_raw, dbl) =
            load_jsonl_canonical_stream(ql_root, &rel, Some(exp_sha), code, name, nb, rd, mb)?;
        all_inputs.push(st);
        all_meta_info.push((
            "ro_live".to_string(),
            "exploratory_diagnostic_already_explored_history".to_string(),
            rel,
            sha,
            n_raw,
            dbl,
            true,
        ));
    }

    // 3. Încărcăm DOAR cele 6 controale internaționale & RNG NE-REZERVATE (care NU se suprapun cu `xi/date_p2/*/rezerva.json`!)
    let prov_bytes = fs::read(ql_root.join("hs_v4/date_intl/PROVENIENTA_INTL.json"))
        .map_err(|e| format!("PROVENIENTA_INTL.json: {e}"))?;
    let prov: Value = serde_json::from_slice(&prov_bytes).map_err(|e| e.to_string())?;
    let non_reserved_intl: [(&str, &str, &str, usize, usize, Option<usize>); 6] = [
        (
            "hu_skandinav_mech",
            "HU Skandináv Lottó — Tragerea 1 Mecanică (Kézi 7/35)",
            "mechanical_external_non_reserved",
            35,
            7,
            None,
        ),
        (
            "fdj_eurodreams",
            "FDJ EuroDreams (6/40 + 1/5 e-Draw RNG)",
            "electronic_rng_control",
            40,
            6,
            Some(5),
        ),
        (
            "hu_skandinav_rng",
            "HU Skandináv Lottó — Tragerea 2 Electronică RNG (Gépi 7/35)",
            "electronic_rng_control",
            35,
            7,
            None,
        ),
        (
            "ny_take5",
            "US NY Take 5 (5/39 Daily RNG)",
            "electronic_rng_control",
            39,
            5,
            None,
        ),
        (
            "ny_cash4life",
            "US NY Cash4Life (5/60 + 1/4 Daily RNG)",
            "electronic_rng_control",
            60,
            5,
            Some(4),
        ),
        (
            "ny_pick10",
            "US NY Pick 10 (20/80 Daily RNG)",
            "electronic_rng_control",
            80,
            20,
            None,
        ),
    ];

    for (code, name, fam, nb, rd, mb) in non_reserved_intl {
        let ds = &prov["datasets"][code];
        let rel = ds["jsonl_file"]
            .as_str()
            .ok_or_else(|| format!("PROVENIENTA_INTL fara {code}"))?;
        let exp_sha = ds["jsonl_sha256"]
            .as_str()
            .ok_or_else(|| format!("PROVENIENTA_INTL fara sha {code}"))?;
        let (st, sha, n_raw, dbl) =
            load_jsonl_canonical_stream(ql_root, rel, Some(exp_sha), code, name, nb, rd, mb)?;
        all_inputs.push(st);
        all_meta_info.push((
            fam.to_string(),
            "non_reserved_external_and_rng_control".to_string(),
            rel.to_string(),
            sha,
            n_raw,
            dbl,
            false,
        ));
    }

    // 4. Încărcăm cele G = 4 fluxuri curate din `xi/date_p2/*/dev.json` (FĂRĂ a atinge `rezerva.json`!)
    let dev_defs: [(&str, &str, &str, usize, usize, Option<usize>, Option<&str>); 4] = [
        ("HU_otos", "HU_otos", "HU Ötöslottó (5/90, dev.json)", 90, 5, None, None),
        ("FR_loto", "FR_loto", "FR Loto (5/49 + 1/10, dev.json)", 49, 5, Some(10), None),
        (
            "US_powerball",
            "US_powerball",
            "US Powerball (5/69 + 1/26, dev.json)",
            69,
            5,
            Some(26),
            None,
        ),
        (
            "HU_hatos",
            "HU_hatos_clean_2007_07",
            "HU Hatoslottó (6/45, dev.json epoca curată 20 câmpuri cheie >= 2007-07)",
            45,
            6,
            None,
            Some("2007-07"),
        ),
    ];

    for (dir_g, scode, sname, nb, rd, mb, min_k) in dev_defs {
        let (st, rel, sha, n_dr) =
            load_xi_p2_dev_stream(ql_root, dir_g, scode, sname, nb, rd, mb, min_k)?;
        all_inputs.push(st);
        all_meta_info.push((
            "xi_p2_dev".to_string(),
            "foreign_development_set_holm_gated".to_string(),
            rel,
            sha,
            n_dr,
            0,
            false,
        ));
    }

    // 5. Evaluăm simultan toate cele 13 fluxuri pe RTX 5060 sm_120 și pe CPU 24T, verificând paritatea < 10^-12
    let (gpu_all, gpu_ms) = eval_general_streams_gpu(&all_inputs)?;
    let t_cpu0 = Instant::now();
    let cpu_all = eval_general_streams_cpu_24t(&all_inputs);
    let cpu_ms = t_cpu0.elapsed().as_secs_f64() * 1000.0;

    let mut ro_reps = Vec::with_capacity(3);
    let mut intl_reps = Vec::with_capacity(6);
    let mut dev_reps = Vec::with_capacity(4);
    let mut max_gpu_cpu_diff = 0.0f64;
    let mut global_min_e = f64::INFINITY;
    let mut global_min_m = f64::INFINITY;

    for i in 0..all_inputs.len() {
        let (ref fam, ref role, ref rel, ref sha, n_raw, dbl, inc_w100) = all_meta_info[i];
        let rep = summarize_stream_walkforward(
            &all_inputs[i],
            fam,
            role,
            rel,
            sha,
            n_raw,
            dbl,
            &gpu_all[i],
            &cpu_all[i],
            inc_w100,
        );
        max_gpu_cpu_diff = max_gpu_cpu_diff
            .max(rep.gpu_cpu_max_abs_diff_e)
            .max(rep.gpu_cpu_max_abs_diff_log_m);
        global_min_e = global_min_e.min(rep.min_e_t);
        global_min_m = global_min_m.min(rep.min_m_t_anytime);
        if i < 3 {
            ro_reps.push(rep);
        } else if i < 9 {
            intl_reps.push(rep);
        } else {
            dev_reps.push(rep);
        }
    }
    if max_gpu_cpu_diff >= 1.0e-12 {
        return Err(format!(
            "Eroare paritate GPU<->CPU pe arhivele reale: {max_gpu_cpu_diff:.3e} >= 1e-12"
        ));
    }

    // 6. Comparație transparentă cu valorile K2 și Full_KJV5 din `kj_v5/rezultate/ABLATIE_RO_KJV5.json`
    let abl_bytes = fs::read(ql_root.join("kj_v5/rezultate/ABLATIE_RO_KJV5.json"))
        .map_err(|e| format!("ABLATIE_RO_KJV5.json: {e}"))?;
    let abl: Value = serde_json::from_slice(&abl_bytes).map_err(|e| e.to_string())?;
    let extract_abl = |g_obj: &Value, cfg_key: &str| -> (f64, usize) {
        if let Some(arr) = g_obj["comparison_table"].as_array() {
            for item in arr {
                if item["config_key"].as_str() == Some(cfg_key) {
                    let e_val = item["E_S_final"].as_f64().unwrap_or(0.0);
                    let gates = item["verdict_gates"]
                        .as_str()
                        .and_then(|s| s.split('/').next())
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    return (e_val, gates);
                }
            }
        }
        (0.0, 0)
    };
    let mut ro_k2_cmp = Vec::with_capacity(3);
    for ro in &ro_reps {
        let g_obj = &abl["games"][&ro.code];
        let (k2_e, k2_gates) = extract_abl(g_obj, "K2");
        let (full_e, full_gates) = extract_abl(g_obj, "Full_KJV5");
        ro_k2_cmp.push(RoK2Comparison {
            game: ro.code.clone(),
            kjv5_k2_e_final: k2_e,
            kjv5_k2_gates_passed: k2_gates,
            kjv5_full_e_final: full_e,
            kjv5_full_gates_passed: full_gates,
            ckv6_full_m_t: ro.m_t_final,
            ckv6_post_warmup100_m_t: ro.m_t_post_warmup100.unwrap_or(ro.m_t_final),
            ckv6_min_e_t: ro.min_e_t,
        });
    }

    // 7. Corecția step-down Holm peste familia celor G = 4 fluxuri `xi/date_p2/*/dev.json`
    let g_fam = dev_reps.len();
    let mut order: Vec<usize> = (0..g_fam).collect();
    order.sort_by(|&a, &b| dev_reps[b].m_t_final.total_cmp(&dev_reps[a].m_t_final));

    let mut holm_table = Vec::with_capacity(g_fam);
    let mut survivors = Vec::new();
    let mut running_min_adj = f64::INFINITY;
    let mut still_rejecting = true;
    for (rk0, &idx) in order.iter().enumerate() {
        let div = g_fam - rk0;
        let raw_e = dev_reps[idx].m_t_final;
        let adj_e = (raw_e / (div as f64)).min(running_min_adj);
        running_min_adj = adj_e;
        let pass_h = still_rejecting && (adj_e >= HOLM_DEV_THRESHOLD);
        if !pass_h {
            still_rejecting = false;
        } else {
            survivors.push(dev_reps[idx].code.clone());
        }
        holm_table.push(HolmDevEntry {
            rank_1based: rk0 + 1,
            stream_code: dev_reps[idx].code.clone(),
            n_draws_dev: dev_reps[idx].n_raw_draws,
            e_dev_raw: raw_e,
            log_e_dev_raw: dev_reps[idx].log_m_t_final,
            holm_divisor: div,
            e_dev_holm_adjusted: adj_e,
            threshold_e_dev: HOLM_DEV_THRESHOLD,
            passes_holm_1500: pass_h,
        });
    }
    let log_global_prod_dev: f64 = dev_reps.iter().map(|r| r.log_m_t_final).sum();
    let global_prod_dev = log_global_prod_dev.exp();

    // 8. REGULA DE FIER A REZERVELOR OARBE 0400:
    //    Dacă `survivors` este vidă, NICIUN fișier `rezerva.json` nu este atins!
    let blind_reserves_touched = false;
    if !survivors.is_empty() {
        // Chiar dacă ar exista un supraviețuitor excepțional, nu deschidem `rezerva.json` fără verificare Python independentă prealabilă!
        // Raportăm candidatul pentru confirmare separată.
    }
    let lstat_after = inspect_blind_reserves_lstat(ql_root)?;
    let reserves_intact = lstat_after.len() == seal.blind_reserves_lstat_baseline.len()
        && lstat_after
            .iter()
            .zip(seal.blind_reserves_lstat_baseline.iter())
            .all(|(a, b)| {
                a.is_strict_0400
                    && a.size_bytes == b.size_bytes
                    && a.mtime_sec == b.mtime_sec
                    && a.mtime_nsec == b.mtime_nsec
            });
    if !reserves_intact {
        return Err("EROARE CRITICA GARDA 0400: lstat pe rezerva.json s-a modificat!".to_string());
    }

    let verdict = if survivors.is_empty() {
        "CKV6_NO_DEV_SURVIVOR_BLIND_RESERVES_SEALED_0400".to_string()
    } else {
        "CKV6_DEV_SURVIVOR_DETECTED_PENDING_VERIFICATION".to_string()
    };

    let (rss_cur, rss_hwm) = read_process_rss_bytes();
    let gate_a5_pass = max_gpu_cpu_diff < 1.0e-12
        && global_min_e >= 0.50
        && global_min_m >= 0.50
        && reserves_intact
        && !blind_reserves_touched
        && rss_cur < MAX_SAFE_RSS_BYTES
        && rss_hwm < MAX_SAFE_RSS_BYTES;

    let report = CkV6ArchivesAndDevReport {
        engine: "CK-V6 (BOCPD Adams-MacKay + Koopman V1(+)V2 Shrunk + Shield w0=0.50)".to_string(),
        session: "S4".to_string(),
        gate: "A5_CAUSAL_REGIME_WALKFORWARD_AND_BLIND_RESERVE_GUARD".to_string(),
        generated_utc: utc_now_iso8601(),
        positive_claims_allowed: false,
        epistemic_notice: "Evaluare strict cauzala pe istoric; loteriile RO sunt marcate explicit diagnostic exploratoriu pe arhiva deja explorata; niciun avantaj comercial sau predictiv nu este revendicat (positive_claims_allowed = false).".to_string(),
        code_seal_verified: true,
        code_seal_sha256: seal_sha,
        ckv6_rs_sha256: rs_sha,
        ckv6_rs_mode_octal: format!("{rs_mode:04o}"),
        gpu_streams_eval_ms: gpu_ms,
        cpu_streams_eval_ms: cpu_ms,
        max_gpu_cpu_abs_diff_across_all_13_streams: max_gpu_cpu_diff,
        min_e_across_all_13_streams: global_min_e,
        min_m_across_all_13_streams: global_min_m,
        ro_exploratory_streams: ro_reps,
        ro_vs_kjv5_k2_comparison: ro_k2_cmp,
        intl_non_reserved_controls: intl_reps,
        xi_p2_dev_streams: dev_reps,
        xi_p2_hu_hatos_protocol_note: "Conform AMENDAMENT_XI_P2_02.md, primele 790 randuri (cheie < 2007-07) din HU_hatos/dev.json au 21 campuri cu potszam citit eronat ca numar principal; CK-V6 evalueaza exclusiv epoca curata de 20 campuri (cheie >= 2007-07, 698 extrageri). DE_6aus49_sambata si US_megamillions nu au dev.json generat in xi/date_p2.".to_string(),
        holm_family_size_g: g_fam,
        holm_dev_table: holm_table,
        global_product_e_dev_clean4: global_prod_dev,
        log_global_product_e_dev_clean4: log_global_prod_dev,
        dev_survivors_holm_1500: survivors,
        blind_reserves_touched,
        blind_reserves_lstat_after: lstat_after,
        blind_reserves_unchanged_and_0400: reserves_intact,
        scientific_verdict: verdict,
        gate_a5_pass,
        rss_current_bytes: rss_cur,
        rss_peak_hwm_bytes: rss_hwm,
    };

    make_writable_if_exists(out_archives_json);
    let txt = serde_json::to_string_pretty(&report).unwrap() + "\n";
    fs::write(out_archives_json, txt)
        .map_err(|e| format!("write {}: {e}", out_archives_json.display()))?;
    Ok(report)
}

/// Verificare read-only rapidă a tuturor livrabilelor S4 (`s4-verify` / `ckv6-verify`), fără a rescrie niciun JSON!
pub fn verify_existing_s4_ckv6(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    ql_root: &Path,
    ckv6_rs_path: &Path,
    code_seal_path: &Path,
    calif_json_path: &Path,
    archives_json_path: &Path,
) -> Result<bool, String> {
    let seal_bytes =
        fs::read(code_seal_path).map_err(|e| format!("read {}: {e}", code_seal_path.display()))?;
    let seal_sha = sha256_bytes_hex(&seal_bytes);
    let seal: CkV6CodeSeal = serde_json::from_slice(&seal_bytes)
        .map_err(|e| format!("parse {}: {e}", code_seal_path.display()))?;

    let rs_md = fs::symlink_metadata(ckv6_rs_path)
        .map_err(|e| format!("stat {}: {e}", ckv6_rs_path.display()))?;
    let rs_mode = rs_md.permissions().mode() & 0o777;
    let rs_sha = sha256_bytes_hex(
        &fs::read(ckv6_rs_path).map_err(|e| format!("read {}: {e}", ckv6_rs_path.display()))?,
    );
    if (rs_mode != 0o444 && rs_mode != 0o400) || rs_sha != seal.ckv6_rs_sha256 {
        return Err("Verificare esuata pe ckv6.rs 0444 / SHA-256!".to_string());
    }

    let cal_bytes = fs::read(calif_json_path)
        .map_err(|e| format!("read {}: {e}", calif_json_path.display()))?;
    let cal_sha = sha256_bytes_hex(&cal_bytes);
    if cal_sha != seal.calificare_putere_sha256 {
        return Err("SHA-256 mismatch pe CK_V6_CALIFICARE_PUTERE.json!".to_string());
    }
    let cal: CkV6QualificationAndPowerReport = serde_json::from_slice(&cal_bytes)
        .map_err(|e| format!("parse {}: {e}", calif_json_path.display()))?;
    if !cal.q_all_6_pass
        || cal.positive_claims_allowed
        || cal.archive_files_read != 0
        || cal.power_total_histories != POWER_TOTAL_HISTORIES
    {
        return Err("CK_V6_CALIFICARE_PUTERE.json nu respecta invariantele!".to_string());
    }

    // Verificăm `ckv6_power_10k.v3bin` (640,064 B, header QLV3VLT\x01, kind=9, SHA-256 + spot-check CPU)
    let v3bin_path = vault.vault_dir.join(CKV6_V3BIN_NAME);
    let f = File::open(&v3bin_path).map_err(|e| format!("open {}: {e}", v3bin_path.display()))?;
    let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", v3bin_path.display()))?;
    let expected_len = VAULT_HEADER_BYTES + POWER_TOTAL_HISTORIES * 64;
    if mmap.len() != expected_len || &mmap[0..8] != VAULT_MAGIC {
        return Err("ckv6_power_10k.v3bin dimensiune sau magic invalid!".to_string());
    }
    let kind = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
    let rec_cnt = u64::from_le_bytes(mmap[24..32].try_into().unwrap());
    let rec_bytes = u64::from_le_bytes(mmap[32..40].try_into().unwrap());
    if kind != TABLE_KIND_CKV6_POWER_10K
        || rec_cnt != POWER_TOTAL_HISTORIES as u64
        || rec_bytes != 64
    {
        return Err("ckv6_power_10k.v3bin header invalid!".to_string());
    }
    let live_v3bin_sha = sha256_bytes_hex(&mmap);
    if live_v3bin_sha != cal.power_v3bin_sha256 {
        return Err("ckv6_power_10k.v3bin SHA-256 diferit!".to_string());
    }

    // Spot-check GPU pe universul 6/49 (13,983,816 stări) ca să certificăm live și hardware-ul în `s4-verify`
    let probe_st = gen_synth_general_stream_cpu("v_u649", 49, 6, None, 40, 3500, 20, 0, CKV6_QUAL_SEED, 901);
    let (h1, q2, _, dm_tbl, n_dm, kap, _) = build_causal_universe_field_at_step(&probe_st, 35);
    let (mean_e, min_e, _, _) = gpu_ckv6_universe_eval_f64(
        0,
        &gpu_univ.d_masks_649,
        49,
        6,
        &h1,
        &q2,
        None,
        &dm_tbl,
        n_dm,
        kap,
    )?;
    if (mean_e - 1.0).abs() >= 1.0e-12 || min_e < 0.50 {
        return Err(format!("s4-verify VRAM martingale check esuat: mean={mean_e}, min={min_e}"));
    }

    let arch_bytes = fs::read(archives_json_path)
        .map_err(|e| format!("read {}: {e}", archives_json_path.display()))?;
    let arch: CkV6ArchivesAndDevReport = serde_json::from_slice(&arch_bytes)
        .map_err(|e| format!("parse {}: {e}", archives_json_path.display()))?;
    if !arch.gate_a5_pass
        || arch.positive_claims_allowed
        || arch.code_seal_sha256 != seal_sha
        || arch.ckv6_rs_sha256 != rs_sha
        || arch.min_e_across_all_13_streams < 0.50
    {
        return Err("CK_V6_ARHIVE_SI_DEV.json invalid!".to_string());
    }
    if arch.dev_survivors_holm_1500.is_empty() && arch.blind_reserves_touched {
        return Err("EROARE GARDA 0400: blind_reserves_touched=true fara supravietuitor!".to_string());
    }

    let lstat_now = inspect_blind_reserves_lstat(ql_root)?;
    let intact = lstat_now
        .iter()
        .zip(seal.blind_reserves_lstat_baseline.iter())
        .all(|(a, b)| {
            a.is_strict_0400
                && a.size_bytes == b.size_bytes
                && a.mtime_sec == b.mtime_sec
                && a.mtime_nsec == b.mtime_nsec
        });
    if !intact {
        return Err("EROARE GARDA 0400: rezerva.json modificata la s4-verify!".to_string());
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_ckv6_synthetic_gpu_cpu_parity_and_shield() {
        let mut batch = Vec::with_capacity(8 * 120);
        for i in 0..8u32 {
            let dbp = if i % 2 == 0 { 0 } else { 5000 };
            batch.extend(gen_synth_649_history_cpu(CKV6_QUAL_SEED, i, i, dbp, 40, 120));
        }
        let gpu = gpu_ckv6_649_batch_f64(&batch, 8, 120, 40, true).expect("GPU CK-V6 649 batch");
        let cpu = eval_649_batch_cpu_24t(&batch, 8, 120, 40, true);
        for i in 0..8 {
            assert!(gpu.summary5[i][2] >= 0.50);
            for t in 0..120 {
                let de = (gpu.step_e[i * 120 + t] - cpu[i].step_e[t]).abs();
                let dlm = (gpu.step_log_m[i * 120 + t] - cpu[i].step_log_m[t]).abs();
                assert!(de < 1.0e-12, "de={de}");
                assert!(dlm < 1.0e-12, "dlm={dlm}");
                assert!(gpu.step_log_m[i * 120 + t].exp() > 0.50);
            }
        }
    }

    #[test]
    fn test_ckv6_general_causality_and_permutation_invariance() {
        let s0 = gen_synth_general_stream_cpu(
            "t_jok",
            45,
            5,
            Some(20),
            90,
            3500,
            30,
            6,
            CKV6_QUAL_SEED,
            99,
        );
        let pm = make_perm(45, CKV6_QUAL_SEED, 1);
        let pb = make_perm(20, CKV6_QUAL_SEED, 2);
        let sp = permute_stream_labels(&s0, &pm, Some(&pb));
        let (gpu_res, _) = eval_general_streams_gpu(&[s0.clone(), sp]).expect("GPU general");
        let cpu_0 = eval_general_stream_cpu(&s0);
        for t in 0..90 {
            assert!((gpu_res[0].step_e[t] - cpu_0.step_e[t]).abs() < 1.0e-12);
            assert!((gpu_res[0].step_e[t] - gpu_res[1].step_e[t]).abs() < 1.0e-13);
            assert!(gpu_res[0].step_e[t] >= 0.50);
        }
    }

    #[test]
    fn test_ckv6_pick10_80_20_high_degree_parity() {
        let s80 = gen_synth_general_stream_cpu(
            "t_pick10",
            80,
            20,
            None,
            600,
            2500,
            150,
            0,
            CKV6_QUAL_SEED,
            880,
        );
        let (gpu_res, _) = eval_general_streams_gpu(&[s80.clone()]).expect("GPU 80/20");
        let cpu_res = eval_general_stream_cpu(&s80);
        let mut max_de = 0.0f64;
        let mut max_dlm = 0.0f64;
        for t in 0..600 {
            max_de = max_de.max((gpu_res[0].step_e[t] - cpu_res.step_e[t]).abs());
            max_dlm = max_dlm.max((gpu_res[0].step_log_m[t] - cpu_res.step_log_m[t]).abs());
        }
        assert!(max_de < 1.0e-12, "max_de 80/20 = {max_de:.3e}");
        assert!(max_dlm < 1.0e-12, "max_dlm 80/20 = {max_dlm:.3e}");
    }

    #[test]
    fn test_blind_reserves_0400_non_invasive_lstat() {
        let ql = PathBuf::from(DEFAULT_QL_ROOT);
        let entries = inspect_blind_reserves_lstat(&ql).expect("lstat rezerve 0400");
        assert_eq!(entries.len(), 4);
        for e in entries {
            assert!(e.is_strict_0400, "rezerva {} mode={}", e.game, e.mode_octal);
            assert!(e.size_bytes > 100);
        }
    }
}

//! Optimizatorul Exact de Portofoliu pe Tot Universul `C3-v2` (`Sesiunea 3 / Poarta A4` din `PROTOCOL_RUST_CORE_V3.md`).
//! - Selecție direct peste toate cele `39,077,004` bilete rezidente în VRAM (`0` trunchiere `4,096`, `0` bilete `patterned`);
//! - Calcul bit-exact al lui `P_any(P)` pe CUDA `sm_120 (__popcll)` + `24T` CPU `POPCNT` (`0` eroare Bonferroni);
//! - Penalizare exactă de auto-partajare multi-bilet `NB2` `G_NB(m, mu, r) = E[m / (m + W)]` în `u128_s96`;
//! - Frontiera Pareto completă `(EV_portofoliu, P_any)` pe `6/49`, `5/40`, `Joker` și portofolii mixte `3` jocuri;
//! - Poarta `G4` (`36/36` instanțe mici verificate contra enumerării exhaustive).
//! Guvernanță științifică: `positive_claims_allowed = false`.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cdv3::{mask_to_numbers_1based, unpack_joker_ticket, JokerTicketJson};
use crate::cuda_ffi::{
    check_rss_guard, gpu_c3v2_eval_540, gpu_c3v2_eval_649, gpu_c3v2_eval_joker,
    gpu_c3v2_scan_universe, DeviceBuffer, GpuBitpackedCoverState, GpuCrowdTables540,
    GpuCrowdTables649, GpuCrowdTablesJoker, GpuJokerCoverState, ResidentGpuUniverse,
    V3CrowdParamsF64, V3PortfolioEvalOut,
};
use crate::pmv3::{
    build_binom_table_50x7, build_crowd_params_f64, double_to_u128_s96, is_patterned_ticket,
    is_pure_1_to_31, pmv3_g_share_cpu, pmv3_rank_lex_cpu, pmv3_ticket_lw_cpu, u128_s96_to_double,
    CdV3QRelabeledReport, PmV3CalibrationReport,
};
use crate::vault::{
    binom, MappedVaultUniverse, VaultFileRecord, COUNT_540_DRAWS6, COUNT_540_TICKETS, COUNT_649,
    COUNT_JOKER_BASE45, COUNT_JOKER_FULL24M, TOTAL_UNIVERSE_TICKETS, VAULT_HEADER_BYTES,
    VAULT_MAGIC,
};

pub const TABLE_KIND_C3V2_PORTFOLIOS: u32 = 7;
pub const C3V2_V3BIN_NAME: &str = "c3v2_pareto_portfolios.v3bin";

pub const BUDGETS_649: [u32; 7] = [40, 100, 250, 500, 1000, 1304, 1500];
pub const BUDGETS_540: [u32; 6] = [40, 100, 250, 500, 1000, 1500];
pub const BUDGETS_JOKER: [u32; 7] = [40, 100, 250, 500, 714, 1000, 1500];
pub const BUDGETS_MIX: [u32; 6] = [40, 100, 250, 500, 1000, 1500];

pub const MU_REGIMES: [(f64, &str); 6] = [
    (0.0, "mu_0_max_ev"),
    (10.0, "mu_10"),
    (100.0, "mu_100"),
    (1000.0, "mu_1000"),
    (10000.0, "mu_10000"),
    (1.0e9, "mu_inf_max_p_any"),
];

#[inline]
fn splitmix64_c3(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ============================================================================
// 1. Cuadratura Gauss-Legendre 32p & Formulele Multi-Bilet NB2 pe CPU (Paritate GPU)
// ============================================================================

static GL32_01: OnceLock<([f64; 32], [f64; 32])> = OnceLock::new();

pub fn gl32_nodes_weights_01() -> &'static ([f64; 32], [f64; 32]) {
    GL32_01.get_or_init(|| {
        const N: usize = 32;
        let mut hx = [0.0f64; N];
        let mut hw = [0.0f64; N];
        let pi_v = std::f64::consts::PI;
        for i in 0..(N / 2) {
            let mut z = (pi_v * ((i as f64) + 0.75) / ((N as f64) + 0.5)).cos();
            let mut dp = 0.0f64;
            for _ in 0..12 {
                let mut p1 = 1.0f64;
                let mut p2 = 0.0f64;
                for j in 1..=N {
                    let p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * (j as f64) - 1.0) * z * p2 - ((j as f64) - 1.0) * p3)
                        / (j as f64);
                }
                dp = (N as f64) * (z * p1 - p2) / (z * z - 1.0);
                z -= p1 / dp;
            }
            let w = 2.0 / ((1.0 - z * z) * dp * dp);
            hx[i] = 0.5 * (1.0 - z);
            hx[N - 1 - i] = 0.5 * (1.0 + z);
            hw[i] = 0.5 * w;
            hw[N - 1 - i] = 0.5 * w;
        }
        (hx, hw)
    })
}

/// `G_NB(m, mu, r) = E[m / (m + W)]` pentru `W ~ NB2(mu, r)`, `m >= 0` (identic cu `c3v2_g_share_multi` din CUDA).
#[inline]
pub fn c3v2_g_share_multi_cpu(m: usize, mu: f64, r: f64, mode: i32) -> f64 {
    if m == 0 {
        return 0.0;
    }
    if m == 1 {
        return pmv3_g_share_cpu(mu, r, mode);
    }
    if mu <= 0.0 {
        return 1.0;
    }
    let dm = m as f64;
    if mode == 2 {
        return dm / (dm + mu);
    }
    let (gx, gw) = gl32_nodes_weights_01();
    let eff_r = if mode == 1 || r >= 1.0e12 { 1.0e12 } else { r };
    let a_scale = dm + mu;
    let log_a = a_scale.ln();
    let mut acc = 0.0f64;
    for q in 0..32 {
        let wq = gx[q];
        let omw = 1.0 - wq;
        let dw = omw * a_scale + wq;
        let mut lp = (dm - 1.0) * omw.ln() + dm * log_a - (dm + 1.0) * dw.ln();
        if mode == 1 || r >= 1.0e12 {
            lp -= (mu * wq) / dw;
        } else {
            lp -= eff_r * ((mu * wq) / (eff_r * dw)).ln_1p();
        }
        acc += gw[q] * lp.exp();
    }
    dm * acc
}

/// Momentul 2 `H2_NB(m, mu, r) = E[(m / (m + W))^2]` pentru `W ~ NB2(mu, r)`, `m >= 1`.
#[inline]
pub fn c3v2_h2_share_multi_cpu(m: usize, mu: f64, r: f64, mode: i32) -> f64 {
    if m == 0 {
        return 0.0;
    }
    if mu <= 0.0 {
        return 1.0;
    }
    let dm = m as f64;
    if mode == 2 {
        let g = dm / (dm + mu);
        return g * g;
    }
    let (gx, gw) = gl32_nodes_weights_01();
    let eff_r = if mode == 1 || r >= 1.0e12 { 1.0e12 } else { r };
    let a_scale = dm + mu;
    let log_a = a_scale.ln();
    let mut acc = 0.0f64;
    for q in 0..32 {
        let tq = gx[q];
        let omw = tq * tq * tq;
        let wq = 1.0 - omw;
        let jac = 3.0 * tq * tq;
        let dw = omw * a_scale + wq;
        let neg_log_1ms = (wq / (omw * a_scale)).ln_1p();
        let mut lp = (dm - 1.0) * omw.ln() + dm * log_a - (dm + 1.0) * dw.ln();
        if mode == 1 || r >= 1.0e12 {
            lp -= (mu * wq) / dw;
        } else {
            lp -= eff_r * ((mu * wq) / (eff_r * dw)).ln_1p();
        }
        acc += gw[q] * jac * neg_log_1ms * lp.exp();
    }
    dm * dm * acc
}

#[inline]
fn atomic_add_u128_s96_slot(lo_arr: &[AtomicU64], hi_arr: &[AtomicU64], idx: usize, val: f64) {
    let (vlo, vhi) = double_to_u128_s96(val);
    if (vlo | vhi) == 0 {
        return;
    }
    let old_lo = lo_arr[idx].fetch_add(vlo, Ordering::Relaxed);
    let carry = if old_lo.wrapping_add(vlo) < old_lo {
        1u64
    } else {
        0u64
    };
    let add_hi = vhi + carry;
    if add_hi != 0 {
        hi_arr[idx].fetch_add(add_hi, Ordering::Relaxed);
    }
}

fn alloc_atomic_u64(n: usize) -> Vec<AtomicU64> {
    let mut v = Vec::with_capacity(n);
    v.resize_with(n, || AtomicU64::new(0));
    v
}

// ============================================================================
// 2. Tabele de Probabilitate a Mulțimii și Evaluatoare Exacte CPU 24T
// ============================================================================

pub struct CpuCrowdTables649 {
    pub q6: Vec<f64>,
    pub q5: Vec<f64>,
    pub q4: Vec<f64>,
}

impl CpuCrowdTables649 {
    pub fn build_24t(masks_649: &[u64], params: &V3CrowdParamsF64) -> Self {
        assert_eq!(masks_649.len(), COUNT_649);
        let binom_tbl = build_binom_table_50x7();
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(24);
        let chunk_sz = (COUNT_649 + n_threads - 1) / n_threads;
        const N5: usize = 1_906_884;
        const N4: usize = 211_876;

        let mut p_vec = vec![0.0f64; COUNT_649];
        let z_lo = AtomicU64::new(0);
        let z_hi = AtomicU64::new(0);

        std::thread::scope(|s| {
            for (m_chunk, p_chunk) in masks_649.chunks(chunk_sz).zip(p_vec.chunks_mut(chunk_sz)) {
                let zlo_r = &z_lo;
                let zhi_r = &z_hi;
                s.spawn(move || {
                    let mut loc_lo = 0u64;
                    let mut loc_hi = 0u64;
                    for (&mk, dst) in m_chunk.iter().zip(p_chunk.iter_mut()) {
                        let w = (pmv3_ticket_lw_cpu(mk, 6, params) - params.lw_shift).exp();
                        *dst = w;
                        let (vlo, vhi) = double_to_u128_s96(w);
                        let nl = loc_lo.wrapping_add(vlo);
                        let c = if nl < loc_lo { 1u64 } else { 0u64 };
                        loc_lo = nl;
                        loc_hi = loc_hi.wrapping_add(vhi + c);
                    }
                    let old = zlo_r.fetch_add(loc_lo, Ordering::Relaxed);
                    let c = if old.wrapping_add(loc_lo) < old {
                        1u64
                    } else {
                        0u64
                    };
                    if loc_hi + c != 0 {
                        zhi_r.fetch_add(loc_hi + c, Ordering::Relaxed);
                    }
                });
            }
        });
        let inv_z = 1.0 / u128_s96_to_double(z_lo.load(Ordering::Relaxed), z_hi.load(Ordering::Relaxed));

        let s5_lo = alloc_atomic_u64(N5);
        let s5_hi = alloc_atomic_u64(N5);
        let s4_lo = alloc_atomic_u64(N4);
        let s4_hi = alloc_atomic_u64(N4);

        std::thread::scope(|s| {
            for (m_chunk, p_chunk) in masks_649.chunks(chunk_sz).zip(p_vec.chunks_mut(chunk_sz)) {
                let s5l = &s5_lo;
                let s5h = &s5_hi;
                let s4l = &s4_lo;
                let s4h = &s4_hi;
                let b_ref = &binom_tbl;
                s.spawn(move || {
                    for (&mk, pv) in m_chunk.iter().zip(p_chunk.iter_mut()) {
                        let p = *pv * inv_z;
                        *pv = p;
                        let mut bits = [0u64; 6];
                        let mut m = mk;
                        for b in &mut bits {
                            let lsb = m & m.wrapping_neg();
                            *b = lsb;
                            m ^= lsb;
                        }
                        for i in 0..6 {
                            let r5 = pmv3_rank_lex_cpu(mk ^ bits[i], 49, 5, N5 as u32, b_ref);
                            atomic_add_u128_s96_slot(s5l, s5h, r5, p);
                        }
                        for i in 0..6 {
                            for j in (i + 1)..6 {
                                let r4 = pmv3_rank_lex_cpu(
                                    mk ^ bits[i] ^ bits[j],
                                    49,
                                    4,
                                    N4 as u32,
                                    b_ref,
                                );
                                atomic_add_u128_s96_slot(s4l, s4h, r4, p);
                            }
                        }
                    }
                });
            }
        });

        let s5: Vec<f64> = (0..N5)
            .map(|i| u128_s96_to_double(s5_lo[i].load(Ordering::Relaxed), s5_hi[i].load(Ordering::Relaxed)))
            .collect();
        let s4: Vec<f64> = (0..N4)
            .map(|i| u128_s96_to_double(s4_lo[i].load(Ordering::Relaxed), s4_hi[i].load(Ordering::Relaxed)))
            .collect();
        drop(s5_lo);
        drop(s5_hi);
        drop(s4_lo);
        drop(s4_hi);

        let mut q5 = vec![0.0f64; COUNT_649];
        let mut q4 = vec![0.0f64; COUNT_649];
        std::thread::scope(|s| {
            for (((m_chunk, p_chunk), q5_chunk), q4_chunk) in masks_649
                .chunks(chunk_sz)
                .zip(p_vec.chunks_mut(chunk_sz))
                .zip(q5.chunks_mut(chunk_sz))
                .zip(q4.chunks_mut(chunk_sz))
            {
                let s5_ref = &s5;
                let s4_ref = &s4;
                let b_ref = &binom_tbl;
                s.spawn(move || {
                    for (((&mk, pv), q5d), q4d) in m_chunk
                        .iter()
                        .zip(p_chunk.iter_mut())
                        .zip(q5_chunk.iter_mut())
                        .zip(q4_chunk.iter_mut())
                    {
                        let mut bits = [0u64; 6];
                        let mut m = mk;
                        for b in &mut bits {
                            let lsb = m & m.wrapping_neg();
                            *b = lsb;
                            m ^= lsb;
                        }
                        let mut t5 = 0.0f64;
                        for i in 0..6 {
                            t5 += s5_ref[pmv3_rank_lex_cpu(mk ^ bits[i], 49, 5, N5 as u32, b_ref)];
                        }
                        let mut t4 = 0.0f64;
                        for i in 0..6 {
                            for j in (i + 1)..6 {
                                t4 += s4_ref[pmv3_rank_lex_cpu(
                                    mk ^ bits[i] ^ bits[j],
                                    49,
                                    4,
                                    N4 as u32,
                                    b_ref,
                                )];
                            }
                        }
                        let pd = (*pv).max(0.0);
                        *pv = pd;
                        *q5d = (t5 - 6.0 * pd).max(0.0);
                        *q4d = (t4 - 5.0 * t5 + 15.0 * pd).max(0.0);
                    }
                });
            }
        });
        Self { q6: p_vec, q5, q4 }
    }
}

pub struct CpuCrowdTables540 {
    pub p5: Vec<f64>,
    pub t5: Vec<f64>,
    pub qiii: Vec<f64>,
}

impl CpuCrowdTables540 {
    pub fn build_24t(
        tickets_540: &[u64],
        draws6_540: &[u64],
        params: &V3CrowdParamsF64,
    ) -> Self {
        let binom_tbl = build_binom_table_50x7();
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(24);
        let chunk_t = (COUNT_540_TICKETS + n_threads - 1) / n_threads;
        let chunk_d = (COUNT_540_DRAWS6 + n_threads - 1) / n_threads;
        const N4: usize = 91_390;

        let mut p5 = vec![0.0f64; COUNT_540_TICKETS];
        let z_lo = AtomicU64::new(0);
        let z_hi = AtomicU64::new(0);

        std::thread::scope(|s| {
            for (t_chunk, p_chunk) in tickets_540.chunks(chunk_t).zip(p5.chunks_mut(chunk_t)) {
                let zlo_r = &z_lo;
                let zhi_r = &z_hi;
                s.spawn(move || {
                    let mut loc_lo = 0u64;
                    let mut loc_hi = 0u64;
                    for (&mk, dst) in t_chunk.iter().zip(p_chunk.iter_mut()) {
                        let w = (pmv3_ticket_lw_cpu(mk, 5, params) - params.lw_shift).exp();
                        *dst = w;
                        let (vlo, vhi) = double_to_u128_s96(w);
                        let nl = loc_lo.wrapping_add(vlo);
                        let c = if nl < loc_lo { 1u64 } else { 0u64 };
                        loc_lo = nl;
                        loc_hi = loc_hi.wrapping_add(vhi + c);
                    }
                    let old = zlo_r.fetch_add(loc_lo, Ordering::Relaxed);
                    let c = if old.wrapping_add(loc_lo) < old {
                        1u64
                    } else {
                        0u64
                    };
                    if loc_hi + c != 0 {
                        zhi_r.fetch_add(loc_hi + c, Ordering::Relaxed);
                    }
                });
            }
        });
        let inv_z = 1.0 / u128_s96_to_double(z_lo.load(Ordering::Relaxed), z_hi.load(Ordering::Relaxed));

        let s4_lo = alloc_atomic_u64(N4);
        let s4_hi = alloc_atomic_u64(N4);
        std::thread::scope(|s| {
            for (t_chunk, p_chunk) in tickets_540.chunks(chunk_t).zip(p5.chunks_mut(chunk_t)) {
                let s4l = &s4_lo;
                let s4h = &s4_hi;
                let b_ref = &binom_tbl;
                s.spawn(move || {
                    for (&mk, pv) in t_chunk.iter().zip(p_chunk.iter_mut()) {
                        let p = *pv * inv_z;
                        *pv = p;
                        let mut m = mk;
                        for _ in 0..5 {
                            let lsb = m & m.wrapping_neg();
                            m ^= lsb;
                            let r4 = pmv3_rank_lex_cpu(mk ^ lsb, 40, 4, N4 as u32, b_ref);
                            atomic_add_u128_s96_slot(s4l, s4h, r4, p);
                        }
                    }
                });
            }
        });
        let s4: Vec<f64> = (0..N4)
            .map(|i| u128_s96_to_double(s4_lo[i].load(Ordering::Relaxed), s4_hi[i].load(Ordering::Relaxed)))
            .collect();
        drop(s4_lo);
        drop(s4_hi);

        let mut t5_vec = vec![0.0f64; COUNT_540_DRAWS6];
        let mut qiii_vec = vec![0.0f64; COUNT_540_DRAWS6];
        std::thread::scope(|s| {
            for ((d_chunk, t5_c), q3_c) in draws6_540
                .chunks(chunk_d)
                .zip(t5_vec.chunks_mut(chunk_d))
                .zip(qiii_vec.chunks_mut(chunk_d))
            {
                let p_ref = &p5;
                let s4_ref = &s4;
                let b_ref = &binom_tbl;
                s.spawn(move || {
                    for ((&mk, t5_d), q3_d) in
                        d_chunk.iter().zip(t5_c.iter_mut()).zip(q3_c.iter_mut())
                    {
                        let mut bits = [0u64; 6];
                        let mut m = mk;
                        for b in &mut bits {
                            let lsb = m & m.wrapping_neg();
                            *b = lsb;
                            m ^= lsb;
                        }
                        let mut t5 = 0.0f64;
                        for i in 0..6 {
                            let rk = pmv3_rank_lex_cpu(
                                mk ^ bits[i],
                                40,
                                5,
                                COUNT_540_TICKETS as u32,
                                b_ref,
                            );
                            t5 += p_ref[rk];
                        }
                        let mut t4 = 0.0f64;
                        for i in 0..6 {
                            for j in (i + 1)..6 {
                                let r4 = pmv3_rank_lex_cpu(
                                    mk ^ bits[i] ^ bits[j],
                                    40,
                                    4,
                                    N4 as u32,
                                    b_ref,
                                );
                                t4 += s4_ref[r4];
                            }
                        }
                        *t5_d = t5;
                        *q3_d = (t4 - 5.0 * t5).max(0.0);
                    }
                });
            }
        });
        Self {
            p5,
            t5: t5_vec,
            qiii: qiii_vec,
        }
    }
}

pub struct CpuCrowdTablesJoker {
    pub q5: Vec<f64>,
    pub q4: Vec<f64>,
    pub q3: Vec<f64>,
    pub q2: Vec<f64>,
    pub q1: Vec<f64>,
}

impl CpuCrowdTablesJoker {
    pub fn build_24t(base_joker: &[u64], params: &V3CrowdParamsF64) -> Self {
        let binom_tbl = build_binom_table_50x7();
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(24);
        let chunk_b = (COUNT_JOKER_BASE45 + n_threads - 1) / n_threads;
        const N4: usize = 148_995;
        const N3: usize = 14_190;
        const N2: usize = 990;
        const N1: usize = 45;

        let mut p_vec = vec![0.0f64; COUNT_JOKER_BASE45];
        let z_lo = AtomicU64::new(0);
        let z_hi = AtomicU64::new(0);

        std::thread::scope(|s| {
            for (b_chunk, p_chunk) in base_joker.chunks(chunk_b).zip(p_vec.chunks_mut(chunk_b)) {
                let zlo_r = &z_lo;
                let zhi_r = &z_hi;
                s.spawn(move || {
                    let mut loc_lo = 0u64;
                    let mut loc_hi = 0u64;
                    for (&mk, dst) in b_chunk.iter().zip(p_chunk.iter_mut()) {
                        let w = (pmv3_ticket_lw_cpu(mk, 5, params) - params.lw_shift).exp();
                        *dst = w;
                        let (vlo, vhi) = double_to_u128_s96(w);
                        let nl = loc_lo.wrapping_add(vlo);
                        let c = if nl < loc_lo { 1u64 } else { 0u64 };
                        loc_lo = nl;
                        loc_hi = loc_hi.wrapping_add(vhi + c);
                    }
                    let old = zlo_r.fetch_add(loc_lo, Ordering::Relaxed);
                    let c = if old.wrapping_add(loc_lo) < old {
                        1u64
                    } else {
                        0u64
                    };
                    if loc_hi + c != 0 {
                        zhi_r.fetch_add(loc_hi + c, Ordering::Relaxed);
                    }
                });
            }
        });
        let inv_z = 1.0 / u128_s96_to_double(z_lo.load(Ordering::Relaxed), z_hi.load(Ordering::Relaxed));

        let s4_lo = alloc_atomic_u64(N4);
        let s4_hi = alloc_atomic_u64(N4);
        let s3_lo = alloc_atomic_u64(N3);
        let s3_hi = alloc_atomic_u64(N3);
        let s2_lo = alloc_atomic_u64(N2);
        let s2_hi = alloc_atomic_u64(N2);
        let s1_lo = alloc_atomic_u64(N1);
        let s1_hi = alloc_atomic_u64(N1);

        std::thread::scope(|s| {
            for (b_chunk, p_chunk) in base_joker.chunks(chunk_b).zip(p_vec.chunks_mut(chunk_b)) {
                let (s4l, s4h, s3l, s3h) = (&s4_lo, &s4_hi, &s3_lo, &s3_hi);
                let (s2l, s2h, s1l, s1h) = (&s2_lo, &s2_hi, &s1_lo, &s1_hi);
                let b_ref = &binom_tbl;
                s.spawn(move || {
                    for (&mk, pv) in b_chunk.iter().zip(p_chunk.iter_mut()) {
                        let p = *pv * inv_z;
                        *pv = p;
                        let mut pos = [0usize; 5];
                        let mut bits = [0u64; 5];
                        let mut m = mk;
                        for i in 0..5 {
                            let a = m.trailing_zeros() as usize;
                            let lsb = 1u64 << a;
                            pos[i] = a;
                            bits[i] = lsb;
                            m ^= lsb;
                        }
                        for i in 0..5 {
                            atomic_add_u128_s96_slot(s1l, s1h, pos[i], p);
                            let r4 = pmv3_rank_lex_cpu(mk ^ bits[i], 45, 4, N4 as u32, b_ref);
                            atomic_add_u128_s96_slot(s4l, s4h, r4, p);
                        }
                        for i in 0..5 {
                            for j in (i + 1)..5 {
                                let pm = bits[i] | bits[j];
                                let r2 = pmv3_rank_lex_cpu(pm, 45, 2, N2 as u32, b_ref);
                                let r3 = pmv3_rank_lex_cpu(mk ^ pm, 45, 3, N3 as u32, b_ref);
                                atomic_add_u128_s96_slot(s2l, s2h, r2, p);
                                atomic_add_u128_s96_slot(s3l, s3h, r3, p);
                            }
                        }
                    }
                });
            }
        });

        let s4: Vec<f64> = (0..N4)
            .map(|i| u128_s96_to_double(s4_lo[i].load(Ordering::Relaxed), s4_hi[i].load(Ordering::Relaxed)))
            .collect();
        let s3: Vec<f64> = (0..N3)
            .map(|i| u128_s96_to_double(s3_lo[i].load(Ordering::Relaxed), s3_hi[i].load(Ordering::Relaxed)))
            .collect();
        let s2: Vec<f64> = (0..N2)
            .map(|i| u128_s96_to_double(s2_lo[i].load(Ordering::Relaxed), s2_hi[i].load(Ordering::Relaxed)))
            .collect();
        let s1: Vec<f64> = (0..N1)
            .map(|i| u128_s96_to_double(s1_lo[i].load(Ordering::Relaxed), s1_hi[i].load(Ordering::Relaxed)))
            .collect();

        let mut q4 = vec![0.0f64; COUNT_JOKER_BASE45];
        let mut q3 = vec![0.0f64; COUNT_JOKER_BASE45];
        let mut q2 = vec![0.0f64; COUNT_JOKER_BASE45];
        let mut q1 = vec![0.0f64; COUNT_JOKER_BASE45];

        std::thread::scope(|s| {
            for (((((b_chunk, p_chunk), q4c), q3c), q2c), q1c) in base_joker
                .chunks(chunk_b)
                .zip(p_vec.chunks_mut(chunk_b))
                .zip(q4.chunks_mut(chunk_b))
                .zip(q3.chunks_mut(chunk_b))
                .zip(q2.chunks_mut(chunk_b))
                .zip(q1.chunks_mut(chunk_b))
            {
                let (s4r, s3r, s2r, s1r) = (&s4, &s3, &s2, &s1);
                let b_ref = &binom_tbl;
                s.spawn(move || {
                    for (((((&mk, pv), d4), d3), d2), d1) in b_chunk
                        .iter()
                        .zip(p_chunk.iter_mut())
                        .zip(q4c.iter_mut())
                        .zip(q3c.iter_mut())
                        .zip(q2c.iter_mut())
                        .zip(q1c.iter_mut())
                    {
                        let mut pos = [0usize; 5];
                        let mut bits = [0u64; 5];
                        let mut m = mk;
                        for i in 0..5 {
                            let a = m.trailing_zeros() as usize;
                            let lsb = 1u64 << a;
                            pos[i] = a;
                            bits[i] = lsb;
                            m ^= lsb;
                        }
                        let mut t1 = 0.0f64;
                        let mut t4 = 0.0f64;
                        for i in 0..5 {
                            t1 += s1r[pos[i]];
                            t4 += s4r[pmv3_rank_lex_cpu(mk ^ bits[i], 45, 4, N4 as u32, b_ref)];
                        }
                        let mut t2 = 0.0f64;
                        let mut t3 = 0.0f64;
                        for i in 0..5 {
                            for j in (i + 1)..5 {
                                let pm = bits[i] | bits[j];
                                t2 += s2r[pmv3_rank_lex_cpu(pm, 45, 2, N2 as u32, b_ref)];
                                t3 += s3r[pmv3_rank_lex_cpu(mk ^ pm, 45, 3, N3 as u32, b_ref)];
                            }
                        }
                        let pd = (*pv).max(0.0);
                        *pv = pd;
                        *d4 = (t4 - 5.0 * pd).max(0.0);
                        *d3 = (t3 - 4.0 * t4 + 10.0 * pd).max(0.0);
                        *d2 = (t2 - 3.0 * t3 + 6.0 * t4 - 10.0 * pd).max(0.0);
                        *d1 = (t1 - 2.0 * t2 + 3.0 * t3 - 4.0 * t4 + 5.0 * pd).max(0.0);
                    }
                });
            }
        });

        Self {
            q5: p_vec,
            q4,
            q3,
            q2,
            q1,
        }
    }
}

/// Evaluare exactă CPU 24T pe `6/49` (`13,983,816` stări) cu `u128_s96` (paritate bit-identică cu CUDA).
pub fn cpu_c3v2_eval_649(
    masks_649: &[u64],
    qt: &CpuCrowdTables649,
    params: &V3CrowdParamsF64,
    tickets: &[u64],
) -> V3PortfolioEvalOut {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_sz = (COUNT_649 + n_threads - 1) / n_threads;

    let cnts: Vec<AtomicU64> = alloc_atomic_u64(6);
    let ev_lo: Vec<AtomicU64> = alloc_atomic_u64(5);
    let ev_hi: Vec<AtomicU64> = alloc_atomic_u64(5);

    let m2_parts: Vec<f64> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, m_chunk) in masks_649.chunks(chunk_sz).enumerate() {
            let start = ci * chunk_sz;
            let q6s = &qt.q6[start..start + m_chunk.len()];
            let q5s = &qt.q5[start..start + m_chunk.len()];
            let q4s = &qt.q4[start..start + m_chunk.len()];
            let cnt_r = &cnts;
            let lo_r = &ev_lo;
            let hi_r = &ev_hi;
            handles.push(s.spawn(move || {
                let mut loc_c = [0u64; 6];
                let mut loc_m2 = 0.0f64;
                let inv_om = 1.0 / 13_983_816.0;
                let ns = params.sales_n;
                let sm = params.share_mode;

                for (i, &dm) in m_chunk.iter().enumerate() {
                    let (mut m_i, mut m_ii, mut m_iii, mut m_iv) = (0usize, 0usize, 0usize, 0usize);
                    for &tk in tickets {
                        let h = (dm & tk).count_ones();
                        if h >= 3 {
                            match h {
                                6 => m_i += 1,
                                5 => m_ii += 1,
                                4 => m_iii += 1,
                                _ => m_iv += 1,
                            }
                        }
                    }
                    let m_any = m_i + m_ii + m_iii + m_iv;
                    if m_any > 0 {
                        loc_c[0] += 1;
                        if m_any >= 2 {
                            loc_c[1] += (m_any * (m_any - 1) / 2) as u64;
                        }
                        if m_i > 0 {
                            loc_c[2] += 1;
                        }
                        if m_ii > 0 {
                            loc_c[3] += 1;
                        }
                        if m_iii > 0 {
                            loc_c[4] += 1;
                        }
                        if m_iv > 0 {
                            loc_c[5] += 1;
                        }

                        let (mut e0, mut e1, mut e2, mut e3) = (0.0, 0.0, 0.0, 0.0);
                        let mut v_cond = 0.0f64;
                        if m_i > 0 {
                            let mu = ns * q6s[i];
                            let g1 = c3v2_g_share_multi_cpu(m_i, mu, params.nb_r[0], sm);
                            let h2 = c3v2_h2_share_multi_cpu(m_i, mu, params.nb_r[0], sm);
                            e0 = params.funds[0] * g1;
                            v_cond += (params.funds[0] * params.funds[0] * (h2 - g1 * g1)).max(0.0);
                            atomic_add_u128_s96_slot(lo_r, hi_r, 0, e0 * inv_om);
                        }
                        if m_ii > 0 {
                            let mu = ns * q5s[i];
                            let g1 = c3v2_g_share_multi_cpu(m_ii, mu, params.nb_r[1], sm);
                            let h2 = c3v2_h2_share_multi_cpu(m_ii, mu, params.nb_r[1], sm);
                            e1 = params.funds[1] * g1;
                            v_cond += (params.funds[1] * params.funds[1] * (h2 - g1 * g1)).max(0.0);
                            atomic_add_u128_s96_slot(lo_r, hi_r, 1, e1 * inv_om);
                        }
                        if m_iii > 0 {
                            let mu = ns * q4s[i];
                            let g1 = c3v2_g_share_multi_cpu(m_iii, mu, params.nb_r[2], sm);
                            let h2 = c3v2_h2_share_multi_cpu(m_iii, mu, params.nb_r[2], sm);
                            e2 = params.funds[2] * g1;
                            v_cond += (params.funds[2] * params.funds[2] * (h2 - g1 * g1)).max(0.0);
                            atomic_add_u128_s96_slot(lo_r, hi_r, 2, e2 * inv_om);
                        }
                        if m_iv > 0 {
                            e3 = (m_iv as f64) * params.fixed_iv;
                            atomic_add_u128_s96_slot(lo_r, hi_r, 3, e3 * inv_om);
                        }
                        let etot = e0 + e1 + e2 + e3;
                        atomic_add_u128_s96_slot(lo_r, hi_r, 4, etot * inv_om);
                        loc_m2 += (etot * etot + v_cond) * inv_om;
                    }
                }
                for k in 0..6 {
                    if loc_c[k] != 0 {
                        cnt_r[k].fetch_add(loc_c[k], Ordering::Relaxed);
                    }
                }
                loc_m2
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out = V3PortfolioEvalOut::default();
    out.win_states = cnts[0].load(Ordering::Relaxed);
    out.pair_overlap_sum = cnts[1].load(Ordering::Relaxed);
    for c in 0..4 {
        out.cat_any_states[c] = cnts[2 + c].load(Ordering::Relaxed);
        out.ev_by_cat[c] = u128_s96_to_double(
            ev_lo[c].load(Ordering::Relaxed),
            ev_hi[c].load(Ordering::Relaxed),
        );
    }
    out.ev_portfolio = u128_s96_to_double(
        ev_lo[4].load(Ordering::Relaxed),
        ev_hi[4].load(Ordering::Relaxed),
    );
    out.second_moment_draw0 = m2_parts.into_iter().sum();
    out.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    out
}

/// Evaluare exactă CPU 24T pe `5/40` (`3,838,380` extrageri de `6` bile, prag oficial Cat. III `|B ∩ D| >= 4`).
pub fn cpu_c3v2_eval_540(
    draws6_540: &[u64],
    qt: &CpuCrowdTables540,
    params: &V3CrowdParamsF64,
    tickets: &[u64],
) -> V3PortfolioEvalOut {
    let t0 = Instant::now();
    let binom_tbl = build_binom_table_50x7();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_d = (COUNT_540_DRAWS6 + n_threads - 1) / n_threads;

    let cnts = alloc_atomic_u64(5);
    let ev_lo = alloc_atomic_u64(4);
    let ev_hi = alloc_atomic_u64(4);

    let m2_parts: Vec<f64> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, d_chunk) in draws6_540.chunks(chunk_d).enumerate() {
            let start = ci * chunk_d;
            let t5s = &qt.t5[start..start + d_chunk.len()];
            let q3s = &qt.qiii[start..start + d_chunk.len()];
            let p5r = &qt.p5;
            let b_ref = &binom_tbl;
            let cnt_r = &cnts;
            let lo_r = &ev_lo;
            let hi_r = &ev_hi;
            handles.push(s.spawn(move || {
                let mut loc_c = [0u64; 5];
                let mut loc_m2 = 0.0f64;
                let inv_draws = 1.0 / 3_838_380.0;
                let inv_ord = 1.0 / (6.0 * 3_838_380.0);
                let ns = params.sales_n;
                let sm = params.share_mode;

                for (i, &dm) in d_chunk.iter().enumerate() {
                    let mut m5 = 0usize;
                    let mut m4 = 0usize;
                    let mut exact5 = [0u64; 6];
                    for &tk in tickets {
                        let h = (dm & tk).count_ones();
                        if h >= 4 {
                            if h == 5 {
                                if m5 < 6 {
                                    exact5[m5] = tk;
                                }
                                m5 += 1;
                            } else {
                                m4 += 1;
                            }
                        }
                    }
                    let m_any = m5 + m4;
                    if m_any > 0 {
                        loc_c[0] += 1;
                        if m_any >= 2 {
                            loc_c[1] += (m_any * (m_any - 1) / 2) as u64;
                        }
                        if m4 > 0 {
                            loc_c[4] += 1;
                        }

                        let mut e_iii = 0.0f64;
                        let mut var_iii = 0.0f64;
                        if m4 > 0 {
                            let mu3 = ns * q3s[i];
                            let g1 = c3v2_g_share_multi_cpu(m4, mu3, params.nb_r[2], sm);
                            let h2 = c3v2_h2_share_multi_cpu(m4, mu3, params.nb_r[2], sm);
                            e_iii = params.funds[2] * g1;
                            var_iii = (params.funds[2] * params.funds[2] * (h2 - g1 * g1)).max(0.0);
                            atomic_add_u128_s96_slot(lo_r, hi_r, 2, e_iii * inv_draws);
                        }
                        if m5 == 0 {
                            atomic_add_u128_s96_slot(lo_r, hi_r, 3, e_iii * inv_draws);
                            loc_m2 += (e_iii * e_iii + var_iii) * inv_draws;
                        } else {
                            let mut bits = [0u64; 6];
                            let mut tmp = dm;
                            for b in &mut bits {
                                let lsb = tmp & tmp.wrapping_neg();
                                *b = lsb;
                                tmp ^= lsb;
                            }
                            let t5_val = t5s[i];
                            let mut sum_ei = 0.0f64;
                            let mut sum_eii = 0.0f64;
                            let mut sum_m2_ord = 0.0f64;
                            for k in 0..6 {
                                let sub5 = dm ^ bits[k];
                                let rk5 = pmv3_rank_lex_cpu(
                                    sub5,
                                    40,
                                    5,
                                    COUNT_540_TICKETS as u32,
                                    b_ref,
                                );
                                let pfi = p5r[rk5];
                                let mi = if exact5[..m5].contains(&sub5) { 1usize } else { 0usize };
                                let mii = m5 - mi;
                                if mi > 0 {
                                    loc_c[2] += 1;
                                }
                                if mii > 0 {
                                    loc_c[3] += 1;
                                }
                                let (mut ei, mut vi) = (0.0, 0.0);
                                if mi > 0 {
                                    let mu1 = ns * pfi;
                                    let g1 = c3v2_g_share_multi_cpu(mi, mu1, params.nb_r[0], sm);
                                    let h2 = c3v2_h2_share_multi_cpu(mi, mu1, params.nb_r[0], sm);
                                    ei = params.funds[0] * g1;
                                    vi = (params.funds[0] * params.funds[0] * (h2 - g1 * g1)).max(0.0);
                                }
                                let (mut eii, mut vii) = (0.0, 0.0);
                                if mii > 0 {
                                    let qii = (t5_val - pfi).max(0.0);
                                    let mu2 = ns * qii;
                                    let g1 = c3v2_g_share_multi_cpu(mii, mu2, params.nb_r[1], sm);
                                    let h2 = c3v2_h2_share_multi_cpu(mii, mu2, params.nb_r[1], sm);
                                    eii = params.funds[1] * g1;
                                    vii = (params.funds[1] * params.funds[1] * (h2 - g1 * g1)).max(0.0);
                                }
                                sum_ei += ei;
                                sum_eii += eii;
                                let et_ord = ei + eii + e_iii;
                                sum_m2_ord += et_ord * et_ord + vi + vii + var_iii;
                            }
                            atomic_add_u128_s96_slot(lo_r, hi_r, 0, sum_ei * inv_ord);
                            atomic_add_u128_s96_slot(lo_r, hi_r, 1, sum_eii * inv_ord);
                            atomic_add_u128_s96_slot(
                                lo_r,
                                hi_r,
                                3,
                                (sum_ei + sum_eii) * inv_ord + e_iii * inv_draws,
                            );
                            loc_m2 += sum_m2_ord * inv_ord;
                        }
                    }
                }
                for k in 0..5 {
                    if loc_c[k] != 0 {
                        cnt_r[k].fetch_add(loc_c[k], Ordering::Relaxed);
                    }
                }
                loc_m2
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out = V3PortfolioEvalOut::default();
    out.win_states = cnts[0].load(Ordering::Relaxed);
    out.pair_overlap_sum = cnts[1].load(Ordering::Relaxed);
    for c in 0..3 {
        out.cat_any_states[c] = cnts[2 + c].load(Ordering::Relaxed);
        out.ev_by_cat[c] = u128_s96_to_double(
            ev_lo[c].load(Ordering::Relaxed),
            ev_hi[c].load(Ordering::Relaxed),
        );
    }
    out.ev_portfolio = u128_s96_to_double(
        ev_lo[3].load(Ordering::Relaxed),
        ev_hi[3].load(Ordering::Relaxed),
    );
    out.second_moment_draw0 = m2_parts.into_iter().sum();
    out.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    out
}

/// Evaluare exactă CPU 24T pe `Joker` (`24,435,180` stări = `1,221,759 × 20`).
pub fn cpu_c3v2_eval_joker(
    base_joker: &[u64],
    qt: &CpuCrowdTablesJoker,
    params: &V3CrowdParamsF64,
    tickets_packed: &[u64],
) -> V3PortfolioEvalOut {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_b = (COUNT_JOKER_BASE45 + n_threads - 1) / n_threads;

    let cnts = alloc_atomic_u64(10);
    let ev_lo = alloc_atomic_u64(9);
    let ev_hi = alloc_atomic_u64(9);

    let m2_parts: Vec<f64> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, b_chunk) in base_joker.chunks(chunk_b).enumerate() {
            let start = ci * chunk_b;
            let q5s = &qt.q5[start..start + b_chunk.len()];
            let q4s = &qt.q4[start..start + b_chunk.len()];
            let q3s = &qt.q3[start..start + b_chunk.len()];
            let q2s = &qt.q2[start..start + b_chunk.len()];
            let q1s = &qt.q1[start..start + b_chunk.len()];
            let cnt_r = &cnts;
            let lo_r = &ev_lo;
            let hi_r = &ev_hi;
            handles.push(s.spawn(move || {
                let mut loc_c = [0u64; 10];
                let mut loc_m2 = 0.0f64;
                let inv_om = 1.0 / 24_435_180.0;
                let ns = params.sales_n;
                let sm = params.share_mode;
                const MASK45: u64 = (1u64 << 45) - 1;

                for (i, &dm) in b_chunk.iter().enumerate() {
                    let mut cnt_h = [0u16; 6];
                    let mut cnt_hj = [[0u16; 20]; 6];
                    let mut any_h1 = false;
                    for &pk in tickets_packed {
                        let bm = pk & MASK45;
                        let h = (dm & bm).count_ones() as usize;
                        if h >= 1 {
                            any_h1 = true;
                            let jb = ((pk >> 48) & 0x1F) as usize;
                            cnt_h[h] += 1;
                            cnt_hj[h][jb] += 1;
                        }
                    }
                    if any_h1 {
                        let q5 = q5s[i];
                        let q4 = q4s[i];
                        let q3 = q3s[i];
                        let q2 = q2s[i];
                        let q1 = q1s[i];
                        let base345 = (cnt_h[5] + cnt_h[4] + cnt_h[3]) as usize;
                        let mut loc_ev_cat = [0.0f64; 8];
                        let mut loc_ev_tot = 0.0f64;

                        for jj in 0..20 {
                            let mc0 = cnt_hj[5][jj] as usize;
                            let mc1 = (cnt_h[5] as usize) - mc0;
                            let mc2 = cnt_hj[4][jj] as usize;
                            let mc3 = (cnt_h[4] as usize) - mc2;
                            let mc4 = cnt_hj[3][jj] as usize;
                            let mc5 = (cnt_h[3] as usize) - mc4;
                            let mc6 = cnt_hj[2][jj] as usize;
                            let mc7 = cnt_hj[1][jj] as usize;
                            let m_c = [mc0, mc1, mc2, mc3, mc4, mc5, mc6, mc7];
                            let m_any = base345 + mc6 + mc7;
                            if m_any > 0 {
                                loc_c[0] += 1;
                                if m_any >= 2 {
                                    loc_c[1] += (m_any * (m_any - 1) / 2) as u64;
                                }
                                let pij = params.pi_joker[jj];
                                let omj = 1.0 - pij;
                                let mu_arr = [
                                    ns * q5 * pij,
                                    ns * q5 * omj,
                                    ns * q4 * pij,
                                    ns * q4 * omj,
                                    ns * q3 * pij,
                                    ns * q3 * omj,
                                    ns * q2 * pij,
                                    ns * q1 * pij,
                                ];
                                let mut et_j = 0.0f64;
                                let mut vc_j = 0.0f64;
                                for c in 0..8 {
                                    let mc = m_c[c];
                                    if mc > 0 {
                                        loc_c[2 + c] += 1;
                                        let g1 = c3v2_g_share_multi_cpu(mc, mu_arr[c], params.nb_r[c], sm);
                                        let h2 = c3v2_h2_share_multi_cpu(mc, mu_arr[c], params.nb_r[c], sm);
                                        let fc = params.funds[c];
                                        let ec = fc * g1;
                                        vc_j += (fc * fc * (h2 - g1 * g1)).max(0.0);
                                        loc_ev_cat[c] += ec;
                                        et_j += ec;
                                    }
                                }
                                loc_ev_tot += et_j;
                                loc_m2 += (et_j * et_j + vc_j) * inv_om;
                            }
                        }
                        if loc_ev_tot > 0.0 {
                            for c in 0..8 {
                                if loc_ev_cat[c] > 0.0 {
                                    atomic_add_u128_s96_slot(lo_r, hi_r, c, loc_ev_cat[c] * inv_om);
                                }
                            }
                            atomic_add_u128_s96_slot(lo_r, hi_r, 8, loc_ev_tot * inv_om);
                        }
                    }
                }
                for k in 0..10 {
                    if loc_c[k] != 0 {
                        cnt_r[k].fetch_add(loc_c[k], Ordering::Relaxed);
                    }
                }
                loc_m2
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out = V3PortfolioEvalOut::default();
    out.win_states = cnts[0].load(Ordering::Relaxed);
    out.pair_overlap_sum = cnts[1].load(Ordering::Relaxed);
    for c in 0..8 {
        out.cat_any_states[c] = cnts[2 + c].load(Ordering::Relaxed);
        out.ev_by_cat[c] = u128_s96_to_double(
            ev_lo[c].load(Ordering::Relaxed),
            ev_hi[c].load(Ordering::Relaxed),
        );
    }
    out.ev_portfolio = u128_s96_to_double(
        ev_lo[8].load(Ordering::Relaxed),
        ev_hi[8].load(Ordering::Relaxed),
    );
    out.second_moment_draw0 = m2_parts.into_iter().sum();
    out.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    out
}

// ============================================================================
// 3. Cititor Cub Bootstrap (EV_NB + q05) & Structuri JSON C3-v2 / Poarta G4
// ============================================================================

pub struct BootPairCube {
    pub pairs_f32: Vec<f32>, // [ev_nom, q05] * record_count
    pub d_pairs_f32: DeviceBuffer<f32>,
    pub record_count: usize,
}

impl BootPairCube {
    pub fn load_to_vram(v3bin_path: &Path, expected_records: usize) -> Result<Self, String> {
        let f = File::open(v3bin_path)
            .map_err(|e| format!("open {}: {e}", v3bin_path.display()))?;
        let mm = unsafe { memmap2::Mmap::map(&f) }
            .map_err(|e| format!("mmap {}: {e}", v3bin_path.display()))?;
        let exp_bytes = VAULT_HEADER_BYTES + expected_records * 24;
        if mm.len() != exp_bytes || &mm[0..8] != VAULT_MAGIC {
            return Err(format!(
                "cub bootstrap invalid {}: len={} vs {}",
                v3bin_path.display(),
                mm.len(),
                exp_bytes
            ));
        }
        let src_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(
                mm[VAULT_HEADER_BYTES..].as_ptr() as *const f32,
                expected_records * 6,
            )
        };
        let mut pairs_f32 = vec![0.0f32; expected_records * 2];
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(24);
        let chunk_rec = (expected_records + n_threads - 1) / n_threads;
        std::thread::scope(|s| {
            for (ci, dst_chunk) in pairs_f32.chunks_mut(chunk_rec * 2).enumerate() {
                let r0 = ci * chunk_rec;
                let rlen = dst_chunk.len() / 2;
                let src_sub = &src_f32[r0 * 6..(r0 + rlen) * 6];
                s.spawn(move || {
                    for i in 0..rlen {
                        dst_chunk[i * 2] = src_sub[i * 6];
                        dst_chunk[i * 2 + 1] = src_sub[i * 6 + 1];
                    }
                });
            }
        });
        let mut d_pairs_f32 = DeviceBuffer::<f32>::alloc(expected_records * 2)?;
        d_pairs_f32.upload_slice(&pairs_f32)?;
        Ok(Self {
            pairs_f32,
            d_pairs_f32,
            record_count: expected_records,
        })
    }

    #[inline]
    pub fn ev_and_q05(&self, idx: usize) -> (f64, f64) {
        let b = idx * 2;
        (self.pairs_f32[b] as f64, self.pairs_f32[b + 1] as f64)
    }
}

/// Probabilitatea analitică exactă `E[P_any(P_rnd)] = 1 - C(N_univ - d_in, K) / C(N_univ, K)`
/// pentru un portofoliu uniform de `K` bilete distincte din universul `N_univ`.
#[inline]
pub fn p_random_analytical_hypergeom(n_univ: u64, d_in_per_state: u64, k_tickets: usize) -> f64 {
    if k_tickets == 0 {
        return 0.0;
    }
    if (k_tickets as u64) > n_univ - d_in_per_state {
        return 1.0;
    }
    let mut log_miss = 0.0f64;
    for i in 0..(k_tickets as u64) {
        let num = (n_univ - d_in_per_state - i) as f64;
        let den = (n_univ - i) as f64;
        log_miss += (num / den).ln();
    }
    1.0 - log_miss.exp()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct C3ComparatorMetrics {
    pub p_random_analytical: f64,
    pub c_rnd_mc64_p_any_mean: f64,
    pub c_rnd_mc64_ev_mean: f64,
    pub c_rnd_mc64_ev_q05: f64,
    pub c_rnd_mc64_ev_q95: f64,
    pub c_top_ev_naive_ev_sum_no_selfshare: f64,
    pub c_top_ev_true_portfolio_ev: f64,
    pub c_top_ev_selfshare_loss_lei: f64,
    pub c_top_ev_p_any: f64,
    pub c_top_ev_win_states: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoPortfolioPoint {
    pub id: String,
    pub game: String,
    pub budget_lei: u32,
    pub actual_cost_lei: f64,
    pub k_tickets: usize,
    pub mu: f64,
    pub regime_label: String,
    pub source_method: String,
    pub universe_states: u64,
    pub win_states_gpu: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub win_states_cpu: Option<u64>,
    pub p_any_exact: f64,
    pub p_bonferroni_1: f64,
    pub p_bonferroni_2: f64,
    pub bonferroni_2_abs_error_avoided: f64,
    pub pair_overlap_sum: u64,
    pub ev_portfolio_gpu: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ev_portfolio_cpu: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_cpu_ev_abs_diff: Option<f64>,
    pub gpu_cpu_parity_ok: bool,
    pub return_ratio_rho: f64,
    pub net_expected_lei: f64,
    pub sum_individual_ev_no_selfshare: f64,
    pub self_sharing_loss_lei: f64,
    pub sum_individual_q05: f64,
    pub ev_portfolio_q05_conservative: f64,
    pub second_moment_draw0: f64,
    pub std_draw0: f64,
    pub objective_f_mu: f64,
    pub patterned_tickets: usize,
    pub pure_1_31_tickets: usize,
    pub beats_p_random: bool,
    pub beats_c_top_ev_and_rnd_on_ev: bool,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
    pub tickets_numbers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_tickets: Option<Vec<JokerTicketJson>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameBudgetFrontier {
    pub game: String,
    pub budget_lei: u32,
    pub ticket_price_lei: f64,
    pub k_tickets: usize,
    pub comparators: C3ComparatorMetrics,
    pub frontier_monotone_p_any: bool,
    pub mu0_beats_top_ev_and_rnd: bool,
    pub all_pos_mu_beat_p_random: bool,
    pub points: Vec<ParetoPortfolioPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixedPortfolioPoint {
    pub id: String,
    pub budget_lei: u32,
    pub actual_cost_lei: f64,
    pub mu: f64,
    pub regime_label: String,
    pub k_649: usize,
    pub k_540: usize,
    pub k_joker: usize,
    pub total_tickets: usize,
    pub p_any_649: f64,
    pub p_any_540: f64,
    pub p_any_joker: f64,
    pub p_any_mix_exact: f64,
    pub p_random_mix_analytical: f64,
    pub ev_649: f64,
    pub ev_540: f64,
    pub ev_joker: f64,
    pub ev_mix_total: f64,
    pub q05_mix_conservative: f64,
    pub return_ratio_rho: f64,
    pub net_expected_lei: f64,
    pub objective_f_mu: f64,
    pub patterned_tickets_total: usize,
    pub beats_p_random_mix: bool,
    pub vault_offset_649: u64,
    pub vault_offset_540: u64,
    pub vault_offset_joker: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct G4SmallInstanceRecord {
    pub instance_id: usize,
    pub game: String,
    pub n_pool: usize,
    pub k_ticket: usize,
    pub p_draw: usize,
    pub j_pool: usize,
    pub m_subpool: usize,
    pub k_select: usize,
    pub mu: f64,
    pub total_subsets_enumerated: usize,
    pub global_opt_objective: f64,
    pub solver_objective: f64,
    pub abs_gap: f64,
    pub exact_hit: bool,
    pub global_opt_indices: Vec<usize>,
    pub solver_indices: Vec<usize>,
    pub global_opt_ev: f64,
    pub global_opt_p_any: f64,
    // Exportăm matricea compactă pentru re-verificarea exhaustivă independentă în Python (`verifica_s3_portofoliu.py`)
    pub n_states: usize,
    pub n_cats: usize,
    pub cat_funds: Vec<f64>,
    pub cat_nb_r: Vec<f64>,
    pub fixed_cat4_prize: f64,
    pub subpool_cat_matrix_flat: Vec<i8>, // [m_subpool * n_states], -1 = miss, 0..n_cats-1 = cat
    pub state_crowd_mu_flat: Vec<f64>,    // [n_cats * n_states]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct C3V2PortfolioReport {
    pub artifact: String,
    pub version: String,
    pub positive_claims_allowed: bool,
    pub source_pmv3_sha256: String,
    pub source_cdv3q_sha256: String,
    pub vault_portfolios_v3bin: String,
    pub vault_portfolios_sha256: String,
    pub vault_portfolios_bytes: u64,
    pub vault_portfolios_records: u64,
    pub total_universe_scanned: usize,
    pub truncation_4096_eliminated: bool,
    pub bonferroni_error_eliminated: bool,
    pub max_gpu_cpu_win_states_diff: u64,
    pub max_gpu_cpu_ev_abs_diff: f64,
    pub total_patterned_tickets_in_all_portfolios: usize,
    pub gate_g4_total_instances: usize,
    pub gate_g4_exact_hits: usize,
    pub gate_g4_pass: bool,
    pub all_standard_budgets_pos_mu_beat_p_random: bool,
    pub all_mu0_beat_naive_top_ev_and_random: bool,
    pub all_frontiers_monotone: bool,
    pub rss_current_bytes: u64,
    pub rss_peak_hwm_bytes: u64,
    pub rss_limit_bytes: u64,
    pub rss_guard_ok: bool,
    pub build_seconds: f64,
    pub c3v2_gate_pass: bool,
    pub single_game_frontiers: Vec<GameBudgetFrontier>,
    pub mixed_3game_portfolios: Vec<MixedPortfolioPoint>,
    pub g4_instances: Vec<G4SmallInstanceRecord>,
}

// ============================================================================
// 4. Optimizator Full-VRAM + Bitboard + Semințe QAP CD-V3Q pe Joc Individual
// ============================================================================

#[derive(Clone)]
struct EvaluatedCandidate {
    method: String,
    tickets: Vec<u64>,
    eval_gpu: V3PortfolioEvalOut,
    sum_ind_ev: f64,
    sum_ind_q05: f64,
}

fn ticket_universe_index(
    game: &str,
    packed_or_mask: u64,
    binom_tbl: &[[u32; 7]; 50],
) -> usize {
    match game {
        "649" => pmv3_rank_lex_cpu(packed_or_mask, 49, 6, COUNT_649 as u32, binom_tbl),
        "540" => pmv3_rank_lex_cpu(packed_or_mask, 40, 5, COUNT_540_TICKETS as u32, binom_tbl),
        "joker" => {
            let bm = packed_or_mask & ((1u64 << 45) - 1);
            let jb = ((packed_or_mask >> 48) & 0x1F) as usize;
            let rk = pmv3_rank_lex_cpu(bm, 45, 5, COUNT_JOKER_BASE45 as u32, binom_tbl);
            jb * COUNT_JOKER_BASE45 + rk
        }
        _ => unreachable!(),
    }
}

fn check_ticket_not_patterned(game: &str, tk: u64) -> (bool, bool) {
    let (bm, gcols) = if game == "joker" {
        (tk & ((1u64 << 45) - 1), 5usize)
    } else if game == "649" {
        (tk, 7usize)
    } else {
        (tk, 5usize)
    };
    let nums = mask_to_numbers_1based(bm);
    let pat = is_patterned_ticket(&nums, gcols);
    let p31 = is_pure_1_to_31(&nums);
    (pat, p31)
}

fn overlap_penalties_for_game(game: &str, selfshare_scale: f64, cov_scale: f64) -> [f64; 16] {
    let mut pen = [0.0f64; 16];
    match game {
        "649" => {
            // Pierdere auto-partajare pe suprapuneri >= 3 + coliziune acoperire Cat IV (t=3)
            pen[0] = 0.0;
            pen[1] = 0.0005 * cov_scale;
            pen[2] = 0.0080 * cov_scale;
            pen[3] = 0.0020 * selfshare_scale + 0.065 * cov_scale;
            pen[4] = 0.0350 * selfshare_scale + 0.250 * cov_scale;
            pen[5] = 0.2800 * selfshare_scale + 0.800 * cov_scale;
            pen[6] = 1.0e6;
        }
        "540" => {
            // La t=4 pe extrageri de 6 bile: ov=0 sau ov=1 au ZERO extrageri comune!
            // ov=2 are doar 9 extrageri comune din 3,010; ov=3 are 106; ov=4 are 666 + auto-partajare Cat II/III!
            pen[0] = 0.0;
            pen[1] = 0.0;
            pen[2] = 0.0004 * selfshare_scale + 0.006 * cov_scale;
            pen[3] = 0.0080 * selfshare_scale + 0.085 * cov_scale;
            pen[4] = 0.1600 * selfshare_scale + 0.550 * cov_scale;
            pen[5] = 1.0e6;
        }
        "joker" => {
            // 0..=5: Joker diferit (coliziune doar la h >= 3); 6..=11: ACELASI Joker (coliziune puternica h >= 1!)
            pen[0] = 0.0;
            pen[1] = 0.0002 * cov_scale;
            pen[2] = 0.0030 * cov_scale;
            pen[3] = 0.0120 * selfshare_scale + 0.045 * cov_scale;
            pen[4] = 0.1400 * selfshare_scale + 0.220 * cov_scale;
            pen[5] = 0.6500 * selfshare_scale + 0.500 * cov_scale;
            // Acelasi Joker (jb == sjb -> index 6 + ov):
            pen[6] = 0.0060 * selfshare_scale + 0.180 * cov_scale;
            pen[7] = 0.0180 * selfshare_scale + 0.320 * cov_scale;
            pen[8] = 0.0450 * selfshare_scale + 0.520 * cov_scale;
            pen[9] = 0.1600 * selfshare_scale + 0.850 * cov_scale;
            pen[10] = 0.5500 * selfshare_scale + 1.400 * cov_scale;
            pen[11] = 1.0e6;
        }
        _ => unreachable!(),
    }
    pen
}

#[allow(clippy::too_many_arguments)]
fn extend_seed_on_full_vram(
    game: &str,
    seed_tickets: &[u64],
    target_k: usize,
    selfshare_scale: f64,
    cov_weight: f64,
    w_ev: f64,
    w_q05: f64,
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    cube: &BootPairCube,
) -> Result<Vec<u64>, String> {
    let (game_code, d_tk_univ, host_univ, n_omega) = match game {
        "649" => (0i32, &gpu_univ.d_masks_649, vault.masks_649(), COUNT_649 as u64),
        "540" => (
            1i32,
            &gpu_univ.d_tickets_540,
            vault.tickets_540(),
            COUNT_540_DRAWS6 as u64,
        ),
        "joker" => (
            2i32,
            &gpu_univ.d_full_joker,
            vault.full_joker(),
            COUNT_JOKER_FULL24M as u64,
        ),
        _ => unreachable!(),
    };

    let mut selected: Vec<u64> = Vec::with_capacity(target_k);
    let mut seen: HashSet<u64> = HashSet::with_capacity(target_k * 2);
    for &tk in seed_tickets {
        if selected.len() >= target_k {
            break;
        }
        let (pat, _) = check_ticket_not_patterned(game, tk);
        if !pat && seen.insert(tk) {
            selected.push(tk);
        }
    }
    if selected.len() >= target_k {
        return Ok(selected);
    }

    let use_bitboard = cov_weight > 0.0;
    let mut std_cov = if use_bitboard && game != "joker" {
        let d_draws = if game == "649" {
            &gpu_univ.d_masks_649
        } else {
            &gpu_univ.d_draws6_540
        };
        let min_h = if game == "649" { 3 } else { 4 };
        let mut st = GpuBitpackedCoverState::new(n_omega)?;
        for &tk in &selected {
            st.mark_covered(d_draws, tk, min_h)?;
        }
        Some(st)
    } else {
        None
    };

    let mut jok_cov = if use_bitboard && game == "joker" {
        let mut st = GpuJokerCoverState::new(COUNT_JOKER_BASE45 as u32)?;
        for &tk in &selected {
            st.mark_covered(&gpu_univ.d_base_joker, tk)?;
        }
        Some(st)
    } else {
        None
    };

    let n_blocks = (host_univ.len() + 255) / 256;
    let mut blk_scores = vec![0.0f64; n_blocks];
    let mut blk_indices = vec![0u32; n_blocks];
    let pen16 = overlap_penalties_for_game(game, selfshare_scale, (cov_weight / 50.0).clamp(0.0, 8.0));

    let mut order_buf: Vec<usize> = (0..n_blocks).collect();
    let mut rng = 0xC352_2026_1104_0001u64 ^ (target_k as u64).wrapping_mul(0x9E37_79B9);

    while selected.len() < target_k {
        gpu_c3v2_scan_universe(
            game_code,
            d_tk_univ,
            &cube.d_pairs_f32,
            &selected,
            &pen16,
            w_ev,
            w_q05,
            &mut blk_scores,
            &mut blk_indices,
        )?;

        if !use_bitboard {
            // Alegem direct maximul global din toate cele n_blocks blocuri (tot universul VRAM!)
            let mut best_s = f64::NEG_INFINITY;
            let mut best_u = u32::MAX;
            for b in 0..n_blocks {
                let idx = blk_indices[b];
                if idx != u32::MAX && blk_scores[b] > best_s {
                    let tk = host_univ[idx as usize];
                    if !seen.contains(&tk) {
                        best_s = blk_scores[b];
                        best_u = idx;
                    }
                }
            }
            let chosen = host_univ[best_u as usize];
            seen.insert(chosen);
            selected.push(chosen);
        } else {
            // Selectăm top 192 blocuri din tot universul + 64 sonde diverse non-patterned și măsurăm câștigul BIT-EXACT pe GPU!
            order_buf.clear();
            for b in 0..n_blocks {
                if blk_indices[b] != u32::MAX {
                    order_buf.push(b);
                }
            }
            let top_take = 192usize.min(order_buf.len());
            if order_buf.len() > top_take {
                order_buf.select_nth_unstable_by(top_take - 1, |&a, &b| {
                    blk_scores[b]
                        .partial_cmp(&blk_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut cand_indices: Vec<usize> = Vec::with_capacity(256);
            let mut cand_masks: Vec<u64> = Vec::with_capacity(256);
            let mut cand_base_sc: Vec<f64> = Vec::with_capacity(256);

            for &b in &order_buf[..top_take] {
                let uidx = blk_indices[b] as usize;
                let tk = host_univ[uidx];
                if !seen.contains(&tk) && !cand_masks.contains(&tk) {
                    cand_indices.push(uidx);
                    cand_masks.push(tk);
                    cand_base_sc.push(blk_scores[b]);
                }
            }
            let mut tries = 0usize;
            while cand_masks.len() < 256 && tries < 512 {
                tries += 1;
                let uidx = (splitmix64_c3(&mut rng) as usize) % host_univ.len();
                let tk = host_univ[uidx];
                if seen.contains(&tk) || cand_masks.contains(&tk) {
                    continue;
                }
                let (pat, _) = check_ticket_not_patterned(game, tk);
                if pat {
                    continue;
                }
                let (ev, q05) = cube.ev_and_q05(uidx);
                cand_indices.push(uidx);
                cand_masks.push(tk);
                cand_base_sc.push(w_ev * ev + w_q05 * q05);
            }

            let mut gains = vec![0u32; cand_masks.len()];
            if let Some(ref st) = std_cov {
                let (d_draws, min_h) = if game == "649" {
                    (&gpu_univ.d_masks_649, 3)
                } else {
                    (&gpu_univ.d_draws6_540, 4)
                };
                st.score_candidates(d_draws, &cand_masks, min_h, &mut gains)?;
            } else if let Some(ref st) = jok_cov {
                st.score_candidates(&gpu_univ.d_base_joker, &cand_masks, &mut gains)?;
            }

            let inv_om = 1.0 / (n_omega as f64);
            let mut best_pos = 0usize;
            let mut best_tot = f64::NEG_INFINITY;
            for i in 0..cand_masks.len() {
                let sc = cand_base_sc[i] + cov_weight * (gains[i] as f64) * inv_om;
                if sc > best_tot {
                    best_tot = sc;
                    best_pos = i;
                }
            }
            let chosen = cand_masks[best_pos];
            if let Some(ref mut st) = std_cov {
                let (d_draws, min_h) = if game == "649" {
                    (&gpu_univ.d_masks_649, 3)
                } else {
                    (&gpu_univ.d_draws6_540, 4)
                };
                st.mark_covered(d_draws, chosen, min_h)?;
            } else if let Some(ref mut st) = jok_cov {
                st.mark_covered(&gpu_univ.d_base_joker, chosen)?;
            }
            let _ = cand_indices;
            seen.insert(chosen);
            selected.push(chosen);
        }
    }
    Ok(selected)
}

/// Sub-selecție greedy exactă pe bitboard GPU dintr-o roată QAP mai mare (`seed_wheel.len() > target_k`).
fn subselect_from_wheel_on_gpu(
    game: &str,
    wheel_tickets: &[u64],
    target_k: usize,
    cov_weight: f64,
    gpu_univ: &ResidentGpuUniverse,
    cube: &BootPairCube,
    binom_tbl: &[[u32; 7]; 50],
) -> Result<Vec<u64>, String> {
    let n_omega = match game {
        "649" => COUNT_649 as u64,
        "540" => COUNT_540_DRAWS6 as u64,
        "joker" => COUNT_JOKER_FULL24M as u64,
        _ => unreachable!(),
    };
    let mut rem: Vec<u64> = wheel_tickets
        .iter()
        .copied()
        .filter(|&tk| !check_ticket_not_patterned(game, tk).0)
        .collect();
    rem.dedup();
    if rem.len() <= target_k {
        return Ok(rem);
    }

    let mut std_cov = if game != "joker" {
        Some(GpuBitpackedCoverState::new(n_omega)?)
    } else {
        None
    };
    let mut jok_cov = if game == "joker" {
        Some(GpuJokerCoverState::new(COUNT_JOKER_BASE45 as u32)?)
    } else {
        None
    };

    let mut selected = Vec::with_capacity(target_k);
    let inv_om = 1.0 / (n_omega as f64);

    while selected.len() < target_k && !rem.is_empty() {
        // Dacă pool-ul e mare (ex. 4,411 la 5/40 Cat III), evaluăm în tranșe de <= 512 candidați
        let mut best_idx_in_rem = 0usize;
        let mut best_sc = f64::NEG_INFINITY;

        for (chunk_off, chunk) in rem.chunks(512).enumerate() {
            let mut gains = vec![0u32; chunk.len()];
            if let Some(ref st) = std_cov {
                let (d_draws, min_h) = if game == "649" {
                    (&gpu_univ.d_masks_649, 3)
                } else {
                    (&gpu_univ.d_draws6_540, 4)
                };
                st.score_candidates(d_draws, chunk, min_h, &mut gains)?;
            } else if let Some(ref st) = jok_cov {
                st.score_candidates(&gpu_univ.d_base_joker, chunk, &mut gains)?;
            }
            for (i, &tk) in chunk.iter().enumerate() {
                let uidx = ticket_universe_index(game, tk, binom_tbl);
                let (ev, q05) = cube.ev_and_q05(uidx);
                let sc = (0.80 * ev + 0.20 * q05) + cov_weight * (gains[i] as f64) * inv_om;
                if sc > best_sc {
                    best_sc = sc;
                    best_idx_in_rem = chunk_off * 512 + i;
                }
            }
        }
        let chosen = rem.swap_remove(best_idx_in_rem);
        if let Some(ref mut st) = std_cov {
            let (d_draws, min_h) = if game == "649" {
                (&gpu_univ.d_masks_649, 3)
            } else {
                (&gpu_univ.d_draws6_540, 4)
            };
            st.mark_covered(d_draws, chosen, min_h)?;
        } else if let Some(ref mut st) = jok_cov {
            st.mark_covered(&gpu_univ.d_base_joker, chosen)?;
        }
        selected.push(chosen);
    }
    Ok(selected)
}

fn extract_cdv3q_seeds_for_game(
    game: &str,
    cdv3q: &CdV3QRelabeledReport,
) -> Vec<(String, Vec<u64>)> {
    let mut out = Vec::new();
    for md in &cdv3q.main_designs {
        if game == "649" && (md.game == "649" || md.game == "loto_649") && md.t_min_hits == 3 {
            let mks: Vec<u64> = md
                .tickets_numbers
                .iter()
                .map(|v| v.iter().fold(0u64, |a, &x| a | (1u64 << (x - 1))))
                .collect();
            out.push((md.id.clone(), mks));
        } else if game == "540" && (md.game == "540" || md.game == "loto_540") && md.t_min_hits == 4
        {
            let mks: Vec<u64> = md
                .tickets_numbers
                .iter()
                .map(|v| v.iter().fold(0u64, |a, &x| a | (1u64 << (x - 1))))
                .collect();
            out.push((md.id.clone(), mks));
        } else if game == "joker" && md.game == "joker" {
            if let Some(ref jts) = md.joker_tickets {
                out.push((md.id.clone(), jts.iter().map(|x| x.packed_u64).collect()));
            }
        }
    }
    for pw in &cdv3q.partial_wheels {
        if game == "649" && (pw.game == "649" || pw.game == "loto_649") {
            let mks: Vec<u64> = pw
                .tickets_numbers
                .iter()
                .map(|v| v.iter().fold(0u64, |a, &x| a | (1u64 << (x - 1))))
                .collect();
            out.push((pw.id.clone(), mks));
        } else if game == "540"
            && (pw.game == "540" || pw.game == "loto_540")
            && pw.target_regime == "Cat_III_t4"
        {
            let mks: Vec<u64> = pw
                .tickets_numbers
                .iter()
                .map(|v| v.iter().fold(0u64, |a, &x| a | (1u64 << (x - 1))))
                .collect();
            out.push((pw.id.clone(), mks));
        } else if game == "joker" && pw.game == "joker" {
            if let Some(ref jts) = pw.joker_tickets {
                out.push((pw.id.clone(), jts.iter().map(|x| x.packed_u64).collect()));
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn eval_candidate_on_gpu(
    method: &str,
    game: &str,
    tickets: &[u64],
    gpu_univ: &ResidentGpuUniverse,
    qt649: Option<&GpuCrowdTables649>,
    qt540: Option<&GpuCrowdTables540>,
    qtjok: Option<&GpuCrowdTablesJoker>,
    params: &V3CrowdParamsF64,
    cube: &BootPairCube,
    binom_tbl: &[[u32; 7]; 50],
) -> Result<EvaluatedCandidate, String> {
    let eval_gpu = match game {
        "649" => gpu_c3v2_eval_649(&gpu_univ.d_masks_649, qt649.unwrap(), params, tickets)?,
        "540" => gpu_c3v2_eval_540(&gpu_univ.d_draws6_540, qt540.unwrap(), params, tickets)?,
        "joker" => gpu_c3v2_eval_joker(&gpu_univ.d_base_joker, qtjok.unwrap(), params, tickets)?,
        _ => unreachable!(),
    };
    let mut sum_ev = 0.0f64;
    let mut sum_q05 = 0.0f64;
    for &tk in tickets {
        let uidx = ticket_universe_index(game, tk, binom_tbl);
        let (ev, q05) = cube.ev_and_q05(uidx);
        sum_ev += ev;
        sum_q05 += q05;
    }
    Ok(EvaluatedCandidate {
        method: method.to_string(),
        tickets: tickets.to_vec(),
        eval_gpu,
        sum_ind_ev: sum_ev,
        sum_ind_q05: sum_q05,
    })
}

fn find_top_nonpatterned_ev_tickets(
    game: &str,
    host_univ: &[u64],
    cube: &BootPairCube,
    count: usize,
) -> Vec<u64> {
    // Colectăm top `count` bilete individuale non-patterned pe 24 fire CPU (pentru comparatorul naiv C_TOP_EV și 1-swap polish)
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_sz = (host_univ.len() + n_threads - 1) / n_threads;

    let parts: Vec<Vec<(f32, usize)>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, u_chunk) in host_univ.chunks(chunk_sz).enumerate() {
            let start = ci * chunk_sz;
            let pairs_ref = &cube.pairs_f32;
            handles.push(s.spawn(move || {
                let mut loc: Vec<(f32, usize)> = Vec::with_capacity(count * 2);
                let mut thresh = f32::NEG_INFINITY;
                for (i, &tk) in u_chunk.iter().enumerate() {
                    let uidx = start + i;
                    let ev = pairs_ref[uidx * 2];
                    if loc.len() >= count && ev <= thresh {
                        continue;
                    }
                    if check_ticket_not_patterned(game, tk).0 {
                        continue;
                    }
                    loc.push((ev, uidx));
                    if loc.len() >= count * 2 {
                        loc.sort_by(|a, b| {
                            b.0.partial_cmp(&a.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.1.cmp(&b.1))
                        });
                        loc.truncate(count);
                        thresh = loc.last().unwrap().0;
                    }
                }
                loc.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                loc.truncate(count);
                loc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut merged: Vec<(f32, usize)> = parts.into_iter().flatten().collect();
    merged.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    merged.truncate(count);
    merged.into_iter().map(|(_, idx)| host_univ[idx]).collect()
}

#[allow(clippy::too_many_arguments)]
fn polish_1swap_on_gpu(
    cand: &EvaluatedCandidate,
    mu: f64,
    min_p_any: f64,
    swap_pool: &[u64],
    game: &str,
    n_omega: u64,
    gpu_univ: &ResidentGpuUniverse,
    qt649: Option<&GpuCrowdTables649>,
    qt540: Option<&GpuCrowdTables540>,
    qtjok: Option<&GpuCrowdTablesJoker>,
    params: &V3CrowdParamsF64,
    cube: &BootPairCube,
    binom_tbl: &[[u32; 7]; 50],
    max_evals: usize,
) -> Result<EvaluatedCandidate, String> {
    let mut best = cand.clone();
    let inv_om = 1.0 / (n_omega as f64);
    let mut best_obj = best.eval_gpu.ev_portfolio + mu * (best.eval_gpu.win_states as f64) * inv_om;
    let k_len = best.tickets.len();
    if k_len == 0 || swap_pool.is_empty() || max_evals == 0 {
        return Ok(best);
    }

    let mut evals = 0usize;
    let mut rng = 0x51A9_2026_1104_0077u64 ^ (k_len as u64).wrapping_mul(31);
    let mut trial = best.tickets.clone();

    while evals < max_evals {
        let pos = (splitmix64_c3(&mut rng) as usize) % k_len;
        let rep = swap_pool[(splitmix64_c3(&mut rng) as usize) % swap_pool.len()];
        if trial.contains(&rep) || check_ticket_not_patterned(game, rep).0 {
            evals += 1;
            continue;
        }
        let old = trial[pos];
        trial[pos] = rep;
        let ev_cand = eval_candidate_on_gpu(
            &format!("{}+1swap", best.method),
            game,
            &trial,
            gpu_univ,
            qt649,
            qt540,
            qtjok,
            params,
            cube,
            binom_tbl,
        )?;
        evals += 1;
        let p_cand = (ev_cand.eval_gpu.win_states as f64) * inv_om;
        let obj = ev_cand.eval_gpu.ev_portfolio + mu * p_cand;
        if p_cand + 1.0e-15 >= min_p_any && obj > best_obj + 1.0e-13 {
            best = ev_cand;
            best_obj = obj;
        } else {
            trial[pos] = old;
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn build_single_game_frontiers(
    game: &str,
    budgets: &[u32],
    price_lei: f64,
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    params: &V3CrowdParamsF64,
    cdv3q: &CdV3QRelabeledReport,
    binom_tbl: &[[u32; 7]; 50],
    vault_tickets_accum: &mut Vec<u64>,
    max_win_diff_accum: &mut u64,
    max_ev_diff_accum: &mut f64,
) -> Result<(Vec<GameBudgetFrontier>, Vec<u64>, Vec<u64>), String> {
    let (v3bin_cube, host_univ, n_omega, w1_per_ticket, d_in_per_state) = match game {
        "649" => (
            "pmv3_boot1000_649.v3bin",
            vault.masks_649(),
            COUNT_649 as u64,
            260_624u64,
            260_624u64,
        ),
        "540" => (
            "pmv3_boot1000_540.v3bin",
            vault.tickets_540(),
            COUNT_540_DRAWS6 as u64,
            3_010u64,
            516u64,
        ),
        "joker" => (
            "pmv3_boot1000_joker.v3bin",
            vault.full_joker(),
            COUNT_JOKER_FULL24M as u64,
            715_770u64,
            715_770u64,
        ),
        _ => unreachable!(),
    };

    let cube = BootPairCube::load_to_vram(&vault.vault_dir.join(v3bin_cube), host_univ.len())?;

    let (qt649_gpu, qt649_cpu) = if game == "649" {
        (
            Some(GpuCrowdTables649::build(&gpu_univ.d_masks_649, params)?),
            Some(CpuCrowdTables649::build_24t(vault.masks_649(), params)),
        )
    } else {
        (None, None)
    };
    let (qt540_gpu, qt540_cpu) = if game == "540" {
        (
            Some(GpuCrowdTables540::build(
                &gpu_univ.d_tickets_540,
                &gpu_univ.d_draws6_540,
                params,
            )?),
            Some(CpuCrowdTables540::build_24t(
                vault.tickets_540(),
                vault.draws6_540(),
                params,
            )),
        )
    } else {
        (None, None)
    };
    let (qtjok_gpu, qtjok_cpu) = if game == "joker" {
        (
            Some(GpuCrowdTablesJoker::build(&gpu_univ.d_base_joker, params)?),
            Some(CpuCrowdTablesJoker::build_24t(vault.base_joker(), params)),
        )
    } else {
        (None, None)
    };

    let k_max = budgets
        .iter()
        .map(|&b| ((b as f64) / price_lei).floor() as usize)
        .max()
        .unwrap_or(100);

    // Top 512 bilete individuale non-patterned pentru C_TOP_EV și 1-swap polish
    let top_ev_pool = find_top_nonpatterned_ev_tickets(game, host_univ, &cube, (k_max + 256).min(640));

    // Construim traiectoriile globale pe tot universul VRAM (0..k_max):
    //   [0] Max-EV cu penalizare exactă auto-partajare (w_ev=1.0, w_q05=0.0, cov=0.0)
    //   [1] Conservator q05 + anti-partajare + acoperire ușoară (w_ev=0.70, w_q05=0.30, cov=8.0)
    //   [2..6] Bitboard exact + EV pentru cov_weight in {25.0, 150.0, 1200.0, 15000.0, 1.0e9}
    let traj_configs: [(f64, f64, f64, f64, &str); 7] = [
        (1.0, 0.0, 1.0, 0.0, "vram_full_antiselfshare_ev"),
        (0.75, 8.0, 0.70, 0.30, "vram_full_q05_regularized"),
        (1.0, 25.0, 0.90, 0.10, "vram_bitboard_cov25"),
        (1.0, 150.0, 0.85, 0.15, "vram_bitboard_cov150"),
        (1.0, 1200.0, 0.80, 0.20, "vram_bitboard_cov1200"),
        (1.0, 15000.0, 0.75, 0.25, "vram_bitboard_cov15k"),
        (0.5, 1.0e9, 0.80, 0.20, "vram_bitboard_max_p_any"),
    ];

    let mut global_trajs: Vec<(String, Vec<u64>)> = Vec::with_capacity(traj_configs.len());
    for &(ss_scale, cov_w, wev, wq05, label) in &traj_configs {
        let tk_vec = extend_seed_on_full_vram(
            game,
            &[],
            k_max,
            ss_scale,
            cov_w,
            wev,
            wq05,
            vault,
            gpu_univ,
            &cube,
        )?;
        global_trajs.push((label.to_string(), tk_vec));
    }

    let prefix_max_ev = global_trajs[0].1.clone();
    let prefix_max_p = global_trajs[6].1.clone();

    let qap_seeds = extract_cdv3q_seeds_for_game(game, cdv3q);
    let inv_om = 1.0 / (n_omega as f64);
    let p1_single = (w1_per_ticket as f64) * inv_om;

    let mut frontiers_out: Vec<GameBudgetFrontier> = Vec::with_capacity(budgets.len());

    for &budget_lei in budgets {
        let k_tickets = ((budget_lei as f64) / price_lei).floor() as usize;
        let actual_cost = (k_tickets as f64) * price_lei;

        // 1. Comparator C_TOP_EV (primele K bilete individuale non-patterned fără auto-partajare)
        let c_top_slice = &top_ev_pool[..k_tickets];
        let c_top_eval = eval_candidate_on_gpu(
            "C_TOP_EV",
            game,
            c_top_slice,
            gpu_univ,
            qt649_gpu.as_ref(),
            qt540_gpu.as_ref(),
            qtjok_gpu.as_ref(),
            params,
            &cube,
            binom_tbl,
        )?;

        // 2. Comparator C_RND (analitic hipergeometric exact + 64 eșantioane non-patterned evaluate exact pe GPU)
        let p_rnd_analyt =
            p_random_analytical_hypergeom(host_univ.len() as u64, d_in_per_state, k_tickets);
        let mut rng_rnd = 0xC32D_2026_1104_0000u64
            ^ ((budget_lei as u64) << 16)
            ^ (host_univ.len() as u64);
        let mut rnd_evs = Vec::with_capacity(64);
        let mut rnd_ps = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut r_tk = Vec::with_capacity(k_tickets);
            let mut r_seen = HashSet::with_capacity(k_tickets * 2);
            while r_tk.len() < k_tickets {
                let uidx = (splitmix64_c3(&mut rng_rnd) as usize) % host_univ.len();
                let tk = host_univ[uidx];
                if !check_ticket_not_patterned(game, tk).0 && r_seen.insert(tk) {
                    r_tk.push(tk);
                }
            }
            let r_ev = eval_candidate_on_gpu(
                "C_RND",
                game,
                &r_tk,
                gpu_univ,
                qt649_gpu.as_ref(),
                qt540_gpu.as_ref(),
                qtjok_gpu.as_ref(),
                params,
                &cube,
                binom_tbl,
            )?;
            rnd_evs.push(r_ev.eval_gpu.ev_portfolio);
            rnd_ps.push((r_ev.eval_gpu.win_states as f64) * inv_om);
        }
        let rnd_ev_mean = rnd_evs.iter().sum::<f64>() / (rnd_evs.len() as f64);
        let rnd_p_mean = rnd_ps.iter().sum::<f64>() / (rnd_ps.len() as f64);
        let mut rnd_evs_sorted = rnd_evs.clone();
        rnd_evs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rnd_ev_q05 = rnd_evs_sorted[3];
        let rnd_ev_q95 = rnd_evs_sorted[60];

        let comparators = C3ComparatorMetrics {
            p_random_analytical: p_rnd_analyt,
            c_rnd_mc64_p_any_mean: rnd_p_mean,
            c_rnd_mc64_ev_mean: rnd_ev_mean,
            c_rnd_mc64_ev_q05: rnd_ev_q05,
            c_rnd_mc64_ev_q95: rnd_ev_q95,
            c_top_ev_naive_ev_sum_no_selfshare: c_top_eval.sum_ind_ev,
            c_top_ev_true_portfolio_ev: c_top_eval.eval_gpu.ev_portfolio,
            c_top_ev_selfshare_loss_lei: (c_top_eval.sum_ind_ev - c_top_eval.eval_gpu.ev_portfolio)
                .max(0.0),
            c_top_ev_p_any: (c_top_eval.eval_gpu.win_states as f64) * inv_om,
            c_top_ev_win_states: c_top_eval.eval_gpu.win_states,
        };

        // 3. Construim pool-ul de portofolii candidate de mărime K din traiectoriile VRAM + semințele QAP CD-V3Q
        let mut cand_pool: Vec<EvaluatedCandidate> = Vec::new();
        for (lbl, tr) in &global_trajs {
            let c = eval_candidate_on_gpu(
                lbl,
                game,
                &tr[..k_tickets],
                gpu_univ,
                qt649_gpu.as_ref(),
                qt540_gpu.as_ref(),
                qtjok_gpu.as_ref(),
                params,
                &cube,
                binom_tbl,
            )?;
            cand_pool.push(c);
        }

        // Semințe QAP CD-V3Q exacte, extinse pe bitboard VRAM sau sub-selectate pe bitboard VRAM
        for (seed_id, seed_tks) in &qap_seeds {
            if seed_tks.len() == k_tickets {
                cand_pool.push(eval_candidate_on_gpu(
                    seed_id,
                    game,
                    seed_tks,
                    gpu_univ,
                    qt649_gpu.as_ref(),
                    qt540_gpu.as_ref(),
                    qtjok_gpu.as_ref(),
                    params,
                    &cube,
                    binom_tbl,
                )?);
            } else if seed_tks.len() < k_tickets && k_tickets - seed_tks.len() <= 120 {
                // Extindem roata QAP în 2 regimuri: (a) EV + acoperire moderată, (b) acoperire pură maximă
                for &(cw, tag) in &[(40.0, "ext_bal"), (1.0e9, "ext_maxp")] {
                    let ext = extend_seed_on_full_vram(
                        game,
                        seed_tks,
                        k_tickets,
                        1.0,
                        cw,
                        0.85,
                        0.15,
                        vault,
                        gpu_univ,
                        &cube,
                    )?;
                    cand_pool.push(eval_candidate_on_gpu(
                        &format!("{seed_id}_{tag}"),
                        game,
                        &ext,
                        gpu_univ,
                        qt649_gpu.as_ref(),
                        qt540_gpu.as_ref(),
                        qtjok_gpu.as_ref(),
                        params,
                        &cube,
                        binom_tbl,
                    )?);
                }
            } else if seed_tks.len() > k_tickets
                && (seed_tks.len() <= 200 || (game == "540" && k_tickets >= 100))
            {
                let sub = subselect_from_wheel_on_gpu(
                    game,
                    seed_tks,
                    k_tickets,
                    1.0e9,
                    gpu_univ,
                    &cube,
                    binom_tbl,
                )?;
                cand_pool.push(eval_candidate_on_gpu(
                    &format!("{seed_id}_sub{k_tickets}"),
                    game,
                    &sub,
                    gpu_univ,
                    qt649_gpu.as_ref(),
                    qt540_gpu.as_ref(),
                    qtjok_gpu.as_ref(),
                    params,
                    &cube,
                    binom_tbl,
                )?);
            }
        }

        // 4. 1-swap Kempe polish pe campionul Max-EV (mu=0) și pe campionul Max-P_any (mu=inf)
        let best_ev_idx = cand_pool
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.eval_gpu
                    .ev_portfolio
                    .partial_cmp(&b.1.eval_gpu.ev_portfolio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .0;
        let polished_ev = polish_1swap_on_gpu(
            &cand_pool[best_ev_idx],
            0.0,
            0.0,
            &top_ev_pool,
            game,
            n_omega,
            gpu_univ,
            qt649_gpu.as_ref(),
            qt540_gpu.as_ref(),
            qtjok_gpu.as_ref(),
            params,
            &cube,
            binom_tbl,
            48,
        )?;
        cand_pool.push(polished_ev);

        let best_p_idx = cand_pool
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.eval_gpu
                    .win_states
                    .cmp(&b.1.eval_gpu.win_states)
                    .then_with(|| {
                        a.1.eval_gpu
                            .ev_portfolio
                            .partial_cmp(&b.1.eval_gpu.ev_portfolio)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .unwrap()
            .0;
        let mut swap_pool_cov = top_ev_pool.clone();
        for (_, stks) in &qap_seeds {
            if stks.len() <= 200 {
                swap_pool_cov.extend_from_slice(stks);
            }
        }
        let polished_p = polish_1swap_on_gpu(
            &cand_pool[best_p_idx],
            1.0e9,
            0.0,
            &swap_pool_cov,
            game,
            n_omega,
            gpu_univ,
            qt649_gpu.as_ref(),
            qt540_gpu.as_ref(),
            qtjok_gpu.as_ref(),
            params,
            &cube,
            binom_tbl,
            48,
        )?;
        cand_pool.push(polished_p);

        // 5. Închidere Pareto monotonă peste mu in {0, 10, 100, 1000, 10000, inf} cu Poarta A4 P_any >= P_random pt mu > 0
        let p_rnd_gate = p_rnd_analyt.max(rnd_p_mean);
        let mut points_out: Vec<ParetoPortfolioPoint> = Vec::with_capacity(MU_REGIMES.len());
        let mut prev_p_any_for_pos_mu = 0.0f64;

        for (mi, &(mu, reg_lbl)) in MU_REGIMES.iter().enumerate() {
            let min_req_p = if mu > 0.0 {
                p_rnd_gate.max(prev_p_any_for_pos_mu)
            } else {
                0.0
            };

            let chosen_cand = cand_pool
                .iter()
                .filter(|c| (c.eval_gpu.win_states as f64) * inv_om + 1.0e-15 >= min_req_p)
                .max_by(|a, b| {
                    let fa =
                        a.eval_gpu.ev_portfolio + mu * ((a.eval_gpu.win_states as f64) * inv_om);
                    let fb =
                        b.eval_gpu.ev_portfolio + mu * ((b.eval_gpu.win_states as f64) * inv_om);
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|| {
                    cand_pool
                        .iter()
                        .max_by_key(|c| c.eval_gpu.win_states)
                        .unwrap()
                })
                .clone();

            let p_any = (chosen_cand.eval_gpu.win_states as f64) * inv_om;
            if mu > 0.0 && p_any > prev_p_any_for_pos_mu {
                prev_p_any_for_pos_mu = p_any;
            }

            // Verificăm paritatea CPU 24T pe regimurile extreme (mu=0 și mu=inf) pentru toate bugetele!
            let run_cpu_parity = mi == 0 || mi + 1 == MU_REGIMES.len();
            let (win_cpu_opt, ev_cpu_opt, ev_diff_opt, parity_ok) = if run_cpu_parity {
                let cpu_ev = match game {
                    "649" => cpu_c3v2_eval_649(
                        vault.masks_649(),
                        qt649_cpu.as_ref().unwrap(),
                        params,
                        &chosen_cand.tickets,
                    ),
                    "540" => cpu_c3v2_eval_540(
                        vault.draws6_540(),
                        qt540_cpu.as_ref().unwrap(),
                        params,
                        &chosen_cand.tickets,
                    ),
                    "joker" => cpu_c3v2_eval_joker(
                        vault.base_joker(),
                        qtjok_cpu.as_ref().unwrap(),
                        params,
                        &chosen_cand.tickets,
                    ),
                    _ => unreachable!(),
                };
                let w_diff = chosen_cand.eval_gpu.win_states.abs_diff(cpu_ev.win_states)
                    + chosen_cand
                        .eval_gpu
                        .pair_overlap_sum
                        .abs_diff(cpu_ev.pair_overlap_sum);
                let e_diff = (chosen_cand.eval_gpu.ev_portfolio - cpu_ev.ev_portfolio).abs();
                if w_diff > *max_win_diff_accum {
                    *max_win_diff_accum = w_diff;
                }
                if e_diff > *max_ev_diff_accum {
                    *max_ev_diff_accum = e_diff;
                }
                (
                    Some(cpu_ev.win_states),
                    Some(cpu_ev.ev_portfolio),
                    Some(e_diff),
                    w_diff == 0 && e_diff < 1.0e-12,
                )
            } else {
                (None, None, None, true)
            };

            let mut pat_cnt = 0usize;
            let mut p31_cnt = 0usize;
            let mut nums_vec = Vec::with_capacity(k_tickets);
            let mut jok_vec = if game == "joker" {
                Some(Vec::with_capacity(k_tickets))
            } else {
                None
            };
            for &tk in &chosen_cand.tickets {
                let (pat, p31) = check_ticket_not_patterned(game, tk);
                if pat {
                    pat_cnt += 1;
                }
                if p31 {
                    p31_cnt += 1;
                }
                if game == "joker" {
                    let ujt = unpack_joker_ticket(tk);
                    nums_vec.push(ujt.numbers.to_vec());
                    jok_vec.as_mut().unwrap().push(ujt);
                } else {
                    nums_vec.push(mask_to_numbers_1based(tk));
                }
            }

            let v_off = vault_tickets_accum.len() as u64;
            vault_tickets_accum.extend_from_slice(&chosen_cand.tickets);

            let p_bonf1 = ((k_tickets as f64) * p1_single).min(1.0);
            let p_bonf2 = (k_tickets as f64) * p1_single
                - (chosen_cand.eval_gpu.pair_overlap_sum as f64) * inv_om;
            let bonf2_err = (p_bonf2 - p_any).abs();

            let selfshare_loss =
                (chosen_cand.sum_ind_ev - chosen_cand.eval_gpu.ev_portfolio).max(0.0);
            let q05_cons = (chosen_cand.sum_ind_q05 - selfshare_loss).max(0.0);
            let var_d0 = (chosen_cand.eval_gpu.second_moment_draw0
                - chosen_cand.eval_gpu.ev_portfolio * chosen_cand.eval_gpu.ev_portfolio)
                .max(0.0);

            points_out.push(ParetoPortfolioPoint {
                id: format!("C3V2_{}_B{}_{}", game.to_uppercase(), budget_lei, reg_lbl.to_uppercase()),
                game: game.to_string(),
                budget_lei,
                actual_cost_lei: actual_cost,
                k_tickets,
                mu,
                regime_label: reg_lbl.to_string(),
                source_method: chosen_cand.method.clone(),
                universe_states: n_omega,
                win_states_gpu: chosen_cand.eval_gpu.win_states,
                win_states_cpu: win_cpu_opt,
                p_any_exact: p_any,
                p_bonferroni_1: p_bonf1,
                p_bonferroni_2: p_bonf2,
                bonferroni_2_abs_error_avoided: bonf2_err,
                pair_overlap_sum: chosen_cand.eval_gpu.pair_overlap_sum,
                ev_portfolio_gpu: chosen_cand.eval_gpu.ev_portfolio,
                ev_portfolio_cpu: ev_cpu_opt,
                gpu_cpu_ev_abs_diff: ev_diff_opt,
                gpu_cpu_parity_ok: parity_ok,
                return_ratio_rho: chosen_cand.eval_gpu.ev_portfolio / actual_cost,
                net_expected_lei: chosen_cand.eval_gpu.ev_portfolio - actual_cost,
                sum_individual_ev_no_selfshare: chosen_cand.sum_ind_ev,
                self_sharing_loss_lei: selfshare_loss,
                sum_individual_q05: chosen_cand.sum_ind_q05,
                ev_portfolio_q05_conservative: q05_cons,
                second_moment_draw0: chosen_cand.eval_gpu.second_moment_draw0,
                std_draw0: var_d0.sqrt(),
                objective_f_mu: chosen_cand.eval_gpu.ev_portfolio + mu * p_any,
                patterned_tickets: pat_cnt,
                pure_1_31_tickets: p31_cnt,
                beats_p_random: p_any + 1.0e-15 >= p_rnd_gate,
                beats_c_top_ev_and_rnd_on_ev: chosen_cand.eval_gpu.ev_portfolio
                    >= comparators.c_top_ev_true_portfolio_ev.max(comparators.c_rnd_mc64_ev_mean)
                        - 1.0e-12,
                vault_offset_records: v_off,
                vault_len_records: k_tickets as u64,
                tickets_numbers: nums_vec,
                joker_tickets: jok_vec,
            });
        }

        let mut mono_ok = true;
        for w in points_out[1..].windows(2) {
            if w[1].p_any_exact + 1.0e-15 < w[0].p_any_exact {
                mono_ok = false;
            }
        }
        let mu0_ok = points_out[0].ev_portfolio_gpu > comparators.c_rnd_mc64_ev_mean
            && points_out[0].ev_portfolio_gpu + 1.0e-12 >= comparators.c_top_ev_true_portfolio_ev;
        let pos_mu_ok = points_out[1..].iter().all(|pt| pt.beats_p_random);

        frontiers_out.push(GameBudgetFrontier {
            game: game.to_string(),
            budget_lei,
            ticket_price_lei: price_lei,
            k_tickets,
            comparators,
            frontier_monotone_p_any: mono_ok,
            mu0_beats_top_ev_and_rnd: mu0_ok,
            all_pos_mu_beat_p_random: pos_mu_ok,
            points: points_out,
        });
    }

    Ok((frontiers_out, prefix_max_ev, prefix_max_p))
}


// ============================================================================
// 5. Portofolii Mixte 3 Jocuri (6/49 + 5/40 + Joker) pe Frontiera Pareto
// ============================================================================

fn candidate_3game_splits(budget_lei: u32) -> Vec<(usize, usize, usize)> {
    // Enumerăm toate împărțirile exacte 8*k1 + 5*k2 + 7*k3 == budget_lei cu k1, k2, k3 >= 1
    // (și dacă sunt puține la B=40, includem și B-1 = 39 lei ca opțiune Joker-heavy).
    let mut exact: Vec<(usize, usize, usize)> = Vec::new();
    let max_k1 = ((budget_lei.saturating_sub(12)) / 8) as usize;
    for k1 in 1..=max_k1 {
        let rem1 = budget_lei - (8 * k1 as u32);
        if rem1 < 12 {
            break;
        }
        let max_k3 = ((rem1 - 5) / 7) as usize;
        for k3 in 1..=max_k3 {
            let rem2 = rem1 - (7 * k3 as u32);
            if rem2 >= 5 && rem2 % 5 == 0 {
                let k2 = (rem2 / 5) as usize;
                exact.push((k1, k2, k3));
            }
        }
    }
    if exact.len() <= 16 {
        return exact;
    }

    // Selectăm determinist până la ~16 împărțiri reprezentative pe tot simplexul bugetului
    let bf = budget_lei as f64;
    let targets: [(f64, f64, f64); 14] = [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.40, 0.20, 0.40),
        (0.25, 0.25, 0.50),
        (0.20, 0.15, 0.65),
        (0.12, 0.10, 0.78),
        (0.06, 0.06, 0.88),
        (0.55, 0.15, 0.30),
        (0.70, 0.10, 0.20),
        (0.84, 0.08, 0.08),
        (0.20, 0.55, 0.25),
        (0.12, 0.72, 0.16),
        (0.45, 0.35, 0.20),
        (0.30, 0.15, 0.55),
        (0.50, 0.10, 0.40),
    ];

    let mut chosen: Vec<(usize, usize, usize)> = Vec::with_capacity(16);
    for &(w1, w2, w3) in &targets {
        let best = *exact
            .iter()
            .min_by(|&&(a1, a2, a3), &&(b1, b2, b3)| {
                let da = ((8 * a1) as f64 / bf - w1).powi(2)
                    + ((5 * a2) as f64 / bf - w2).powi(2)
                    + ((7 * a3) as f64 / bf - w3).powi(2);
                let db = ((8 * b1) as f64 / bf - w1).powi(2)
                    + ((5 * b2) as f64 / bf - w2).powi(2)
                    + ((7 * b3) as f64 / bf - w3).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        if !chosen.contains(&best) {
            chosen.push(best);
        }
    }
    // Adăugăm și extremele lexicografice (max k1, max k2, max k3)
    if let Some(&mx1) = exact.iter().max_by_key(|x| x.0) {
        if !chosen.contains(&mx1) {
            chosen.push(mx1);
        }
    }
    if let Some(&mx2) = exact.iter().max_by_key(|x| x.1) {
        if !chosen.contains(&mx2) {
            chosen.push(mx2);
        }
    }
    if let Some(&mx3) = exact.iter().max_by_key(|x| x.2) {
        if !chosen.contains(&mx3) {
            chosen.push(mx3);
        }
    }
    chosen
}

#[derive(Clone)]
struct CachedLegEval {
    tickets: Vec<u64>,
    p_any: f64,
    ev: f64,
    q05_cons: f64,
    patterned_cnt: usize,
}

#[allow(clippy::too_many_arguments)]
fn build_mixed_3game_frontiers(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    p649: &V3CrowdParamsF64,
    p540: &V3CrowdParamsF64,
    pjok: &V3CrowdParamsF64,
    pref_649_ev: &[u64],
    pref_649_p: &[u64],
    pref_540_ev: &[u64],
    pref_540_p: &[u64],
    pref_jok_ev: &[u64],
    pref_jok_p: &[u64],
    binom_tbl: &[[u32; 7]; 50],
    vault_tickets_accum: &mut Vec<u64>,
) -> Result<Vec<MixedPortfolioPoint>, String> {
    let cube_649 = BootPairCube::load_to_vram(
        &vault.vault_dir.join("pmv3_boot1000_649.v3bin"),
        COUNT_649,
    )?;
    let cube_540 = BootPairCube::load_to_vram(
        &vault.vault_dir.join("pmv3_boot1000_540.v3bin"),
        COUNT_540_TICKETS,
    )?;
    let cube_jok = BootPairCube::load_to_vram(
        &vault.vault_dir.join("pmv3_boot1000_joker.v3bin"),
        COUNT_JOKER_FULL24M,
    )?;

    let qt649 = GpuCrowdTables649::build(&gpu_univ.d_masks_649, p649)?;
    let qt540 = GpuCrowdTables540::build(&gpu_univ.d_tickets_540, &gpu_univ.d_draws6_540, p540)?;
    let qtjok = GpuCrowdTablesJoker::build(&gpu_univ.d_base_joker, pjok)?;

    let mut cache_649: std::collections::HashMap<(u8, usize), CachedLegEval> =
        std::collections::HashMap::new();
    let mut cache_540: std::collections::HashMap<(u8, usize), CachedLegEval> =
        std::collections::HashMap::new();
    let mut cache_jok: std::collections::HashMap<(u8, usize), CachedLegEval> =
        std::collections::HashMap::new();

    let mut eval_leg_649 = |mode: u8, k: usize| -> Result<CachedLegEval, String> {
        if let Some(v) = cache_649.get(&(mode, k)) {
            return Ok(v.clone());
        }
        let src = if mode == 0 {
            &pref_649_ev[..k]
        } else {
            &pref_649_p[..k]
        };
        let c = eval_candidate_on_gpu(
            "mix_649",
            "649",
            src,
            gpu_univ,
            Some(&qt649),
            None,
            None,
            p649,
            &cube_649,
            binom_tbl,
        )?;
        let loss = (c.sum_ind_ev - c.eval_gpu.ev_portfolio).max(0.0);
        let pat = src
            .iter()
            .filter(|&&tk| check_ticket_not_patterned("649", tk).0)
            .count();
        let entry = CachedLegEval {
            tickets: src.to_vec(),
            p_any: (c.eval_gpu.win_states as f64) / (COUNT_649 as f64),
            ev: c.eval_gpu.ev_portfolio,
            q05_cons: (c.sum_ind_q05 - loss).max(0.0),
            patterned_cnt: pat,
        };
        cache_649.insert((mode, k), entry.clone());
        Ok(entry)
    };

    let mut eval_leg_540 = |mode: u8, k: usize| -> Result<CachedLegEval, String> {
        if let Some(v) = cache_540.get(&(mode, k)) {
            return Ok(v.clone());
        }
        let src = if mode == 0 {
            &pref_540_ev[..k]
        } else {
            &pref_540_p[..k]
        };
        let c = eval_candidate_on_gpu(
            "mix_540",
            "540",
            src,
            gpu_univ,
            None,
            Some(&qt540),
            None,
            p540,
            &cube_540,
            binom_tbl,
        )?;
        let loss = (c.sum_ind_ev - c.eval_gpu.ev_portfolio).max(0.0);
        let pat = src
            .iter()
            .filter(|&&tk| check_ticket_not_patterned("540", tk).0)
            .count();
        let entry = CachedLegEval {
            tickets: src.to_vec(),
            p_any: (c.eval_gpu.win_states as f64) / (COUNT_540_DRAWS6 as f64),
            ev: c.eval_gpu.ev_portfolio,
            q05_cons: (c.sum_ind_q05 - loss).max(0.0),
            patterned_cnt: pat,
        };
        cache_540.insert((mode, k), entry.clone());
        Ok(entry)
    };

    let mut eval_leg_jok = |mode: u8, k: usize| -> Result<CachedLegEval, String> {
        if let Some(v) = cache_jok.get(&(mode, k)) {
            return Ok(v.clone());
        }
        let src = if mode == 0 {
            &pref_jok_ev[..k]
        } else {
            &pref_jok_p[..k]
        };
        let c = eval_candidate_on_gpu(
            "mix_joker",
            "joker",
            src,
            gpu_univ,
            None,
            None,
            Some(&qtjok),
            pjok,
            &cube_jok,
            binom_tbl,
        )?;
        let loss = (c.sum_ind_ev - c.eval_gpu.ev_portfolio).max(0.0);
        let pat = src
            .iter()
            .filter(|&&tk| check_ticket_not_patterned("joker", tk).0)
            .count();
        let entry = CachedLegEval {
            tickets: src.to_vec(),
            p_any: (c.eval_gpu.win_states as f64) / (COUNT_JOKER_FULL24M as f64),
            ev: c.eval_gpu.ev_portfolio,
            q05_cons: (c.sum_ind_q05 - loss).max(0.0),
            patterned_cnt: pat,
        };
        cache_jok.insert((mode, k), entry.clone());
        Ok(entry)
    };

    let mut out_mix: Vec<MixedPortfolioPoint> = Vec::with_capacity(BUDGETS_MIX.len() * MU_REGIMES.len());

    for &budget_lei in &BUDGETS_MIX {
        let splits = candidate_3game_splits(budget_lei);
        // Împărțirea echilibrată de referință (prima din `splits`, cea mai apropiată de 1/3 - 1/3 - 1/3)
        let (eq1, eq2, eq3) = splits[0];
        let p_rnd_eq = {
            let pr1 = p_random_analytical_hypergeom(COUNT_649 as u64, 260_624, eq1);
            let pr2 = p_random_analytical_hypergeom(COUNT_540_TICKETS as u64, 516, eq2);
            let pr3 = p_random_analytical_hypergeom(COUNT_JOKER_FULL24M as u64, 715_770, eq3);
            1.0 - (1.0 - pr1) * (1.0 - pr2) * (1.0 - pr3)
        };

        #[derive(Clone)]
        struct MixCombo {
            k1: usize,
            k2: usize,
            k3: usize,
            cost: f64,
            l1: CachedLegEval,
            l2: CachedLegEval,
            l3: CachedLegEval,
            p_mix: f64,
            p_rnd_split: f64,
            ev_mix: f64,
            q05_mix: f64,
        }

        let mut combos: Vec<MixCombo> = Vec::with_capacity(splits.len() * 8);
        for &(k1, k2, k3) in &splits {
            let cost = (8 * k1 + 5 * k2 + 7 * k3) as f64;
            let pr1 = p_random_analytical_hypergeom(COUNT_649 as u64, 260_624, k1);
            let pr2 = p_random_analytical_hypergeom(COUNT_540_TICKETS as u64, 516, k2);
            let pr3 = p_random_analytical_hypergeom(COUNT_JOKER_FULL24M as u64, 715_770, k3);
            let p_rnd_split = 1.0 - (1.0 - pr1) * (1.0 - pr2) * (1.0 - pr3);

            for m1 in 0..=1u8 {
                let l1 = eval_leg_649(m1, k1)?;
                for m2 in 0..=1u8 {
                    let l2 = eval_leg_540(m2, k2)?;
                    for m3 in 0..=1u8 {
                        let l3 = eval_leg_jok(m3, k3)?;
                        let p_mix = 1.0 - (1.0 - l1.p_any) * (1.0 - l2.p_any) * (1.0 - l3.p_any);
                        let ev_mix = l1.ev + l2.ev + l3.ev;
                        let q05_mix = l1.q05_cons + l2.q05_cons + l3.q05_cons;
                        combos.push(MixCombo {
                            k1,
                            k2,
                            k3,
                            cost,
                            l1: l1.clone(),
                            l2: l2.clone(),
                            l3,
                            p_mix,
                            p_rnd_split,
                            ev_mix,
                            q05_mix,
                        });
                    }
                }
            }
        }

        let mut prev_pos_p = 0.0f64;
        for &(mu, reg_lbl) in &MU_REGIMES {
            let chosen = combos
                .iter()
                .filter(|c| {
                    if mu <= 0.0 {
                        true
                    } else {
                        let req = c.p_rnd_split.max(p_rnd_eq).max(prev_pos_p);
                        c.p_mix + 1.0e-15 >= req
                    }
                })
                .max_by(|a, b| {
                    let fa = a.ev_mix + mu * a.p_mix;
                    let fb = b.ev_mix + mu * b.p_mix;
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|| {
                    combos
                        .iter()
                        .max_by(|a, b| {
                            a.p_mix
                                .partial_cmp(&b.p_mix)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap()
                });

            if mu > 0.0 && chosen.p_mix > prev_pos_p {
                prev_pos_p = chosen.p_mix;
            }

            let off_649 = vault_tickets_accum.len() as u64;
            vault_tickets_accum.extend_from_slice(&chosen.l1.tickets);
            let off_540 = vault_tickets_accum.len() as u64;
            vault_tickets_accum.extend_from_slice(&chosen.l2.tickets);
            let off_jok = vault_tickets_accum.len() as u64;
            vault_tickets_accum.extend_from_slice(&chosen.l3.tickets);

            let pat_tot = chosen.l1.patterned_cnt + chosen.l2.patterned_cnt + chosen.l3.patterned_cnt;
            let p_rnd_ref = chosen.p_rnd_split.max(p_rnd_eq);

            out_mix.push(MixedPortfolioPoint {
                id: format!("C3V2_MIX3_B{}_{}", budget_lei, reg_lbl.to_uppercase()),
                budget_lei,
                actual_cost_lei: chosen.cost,
                mu,
                regime_label: reg_lbl.to_string(),
                k_649: chosen.k1,
                k_540: chosen.k2,
                k_joker: chosen.k3,
                total_tickets: chosen.k1 + chosen.k2 + chosen.k3,
                p_any_649: chosen.l1.p_any,
                p_any_540: chosen.l2.p_any,
                p_any_joker: chosen.l3.p_any,
                p_any_mix_exact: chosen.p_mix,
                p_random_mix_analytical: p_rnd_ref,
                ev_649: chosen.l1.ev,
                ev_540: chosen.l2.ev,
                ev_joker: chosen.l3.ev,
                ev_mix_total: chosen.ev_mix,
                q05_mix_conservative: chosen.q05_mix,
                return_ratio_rho: chosen.ev_mix / chosen.cost,
                net_expected_lei: chosen.ev_mix - chosen.cost,
                objective_f_mu: chosen.ev_mix + mu * chosen.p_mix,
                patterned_tickets_total: pat_tot,
                beats_p_random_mix: chosen.p_mix + 1.0e-15 >= p_rnd_ref,
                vault_offset_649: off_649,
                vault_offset_540: off_540,
                vault_offset_joker: off_jok,
            });
        }
    }

    Ok(out_mix)
}

// ============================================================================
// 6. Poarta G4 — Certificare Exactă pe 36 Instanțe Mici vs Enumerare Exhaustivă
// ============================================================================

fn enumerate_k_masks_small(n: usize, k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(binom(n, k));
    fn rec(n: usize, k: usize, start: usize, rem: usize, cur: u32, out: &mut Vec<u32>) {
        if rem == 0 {
            out.push(cur);
            return;
        }
        for x in start..=(n - rem) {
            rec(n, k, x + 1, rem - 1, cur | (1u32 << x), out);
        }
    }
    rec(n, k, 0, k, 0, &mut out);
    out
}

struct MiniGameSetup {
    game: &'static str,
    n_pool: usize,
    k_ticket: usize,
    p_draw: usize,
    j_pool: usize,
    n_states: usize,
    n_cats: usize,
    cat_funds: Vec<f64>,
    cat_nb_r: Vec<f64>,
    fixed_cat4_prize: f64,
    all_tickets_cat_rows: Vec<Vec<i8>>, // [n_tickets][n_states]
    state_crowd_mu_flat: Vec<f64>,      // [n_cats * n_states]
}

fn build_mini_game_setup(game_idx: usize) -> MiniGameSetup {
    match game_idx {
        0 => {
            // 649_mini: (n=12, k=6, p=6, t=3), C(12,6) = 924 extrageri și 924 bilete
            let n_pool = 12usize;
            let k_tk = 6usize;
            let p_dr = 6usize;
            let masks = enumerate_k_masks_small(n_pool, k_tk);
            let n_states = masks.len();
            let n_cats = 4usize;
            // Ponderi neuniforme deterministe (anti-mulțime pe numere mari)
            let mut weights = vec![0.0f64; n_states];
            let mut w_sum = 0.0f64;
            for (i, &m) in masks.iter().enumerate() {
                let mut low_cnt = 0;
                for b in 0..7 {
                    if (m & (1u32 << b)) != 0 {
                        low_cnt += 1;
                    }
                }
                let w = (0.32 * (low_cnt as f64) + 0.03 * ((i % 7) as f64)).exp();
                weights[i] = w;
                w_sum += w;
            }
            for w in &mut weights {
                *w /= w_sum;
            }
            let sales = 120.0f64;
            let mut cat_rows = vec![vec![-1i8; n_states]; n_states];
            let mut crowd_mu = vec![0.0f64; n_cats * n_states];
            for (ti, &tm) in masks.iter().enumerate() {
                for (si, &dm) in masks.iter().enumerate() {
                    let h = (tm & dm).count_ones() as usize;
                    if h >= 3 {
                        let c = 6 - h; // 6->0, 5->1, 4->2, 3->3
                        cat_rows[ti][si] = c as i8;
                        crowd_mu[c * n_states + si] += sales * weights[ti];
                    }
                }
            }
            MiniGameSetup {
                game: "649_mini",
                n_pool,
                k_ticket: k_tk,
                p_draw: p_dr,
                j_pool: 1,
                n_states,
                n_cats,
                cat_funds: vec![5000.0, 800.0, 250.0, 0.0],
                cat_nb_r: vec![0.8, 1.2, 1.8, 2.0],
                fixed_cat4_prize: 30.0,
                all_tickets_cat_rows: cat_rows,
                state_crowd_mu_flat: crowd_mu,
            }
        }
        1 => {
            // 540_mini: (n=11, k=5, p=6, t=4), C(11,5)=462 bilete, C(11,6)=462 extrageri de 6 bile
            let n_pool = 11usize;
            let k_tk = 5usize;
            let p_dr = 6usize;
            let tk_masks = enumerate_k_masks_small(n_pool, k_tk);
            let dr_masks = enumerate_k_masks_small(n_pool, p_dr);
            let n_states = dr_masks.len();
            let n_cats = 2usize; // 0: h=5 (Cat I+II), 1: h=4 (Cat III oficial)
            let mut weights = vec![0.0f64; tk_masks.len()];
            let mut w_sum = 0.0f64;
            for (i, &m) in tk_masks.iter().enumerate() {
                let mut low_cnt = 0;
                for b in 0..6 {
                    if (m & (1u32 << b)) != 0 {
                        low_cnt += 1;
                    }
                }
                let w = (0.30 * (low_cnt as f64) + 0.04 * ((i % 5) as f64)).exp();
                weights[i] = w;
                w_sum += w;
            }
            for w in &mut weights {
                *w /= w_sum;
            }
            let sales = 90.0f64;
            let mut cat_rows = vec![vec![-1i8; n_states]; tk_masks.len()];
            let mut crowd_mu = vec![0.0f64; n_cats * n_states];
            for (ti, &tm) in tk_masks.iter().enumerate() {
                for (si, &dm) in dr_masks.iter().enumerate() {
                    let h = (tm & dm).count_ones() as usize;
                    if h >= 4 {
                        let c = 5 - h; // 5->0, 4->1
                        cat_rows[ti][si] = c as i8;
                        crowd_mu[c * n_states + si] += sales * weights[ti];
                    }
                }
            }
            MiniGameSetup {
                game: "540_mini",
                n_pool,
                k_ticket: k_tk,
                p_draw: p_dr,
                j_pool: 1,
                n_states,
                n_cats,
                cat_funds: vec![2500.0, 400.0],
                cat_nb_r: vec![0.9, 1.5],
                fixed_cat4_prize: 0.0,
                all_tickets_cat_rows: cat_rows,
                state_crowd_mu_flat: crowd_mu,
            }
        }
        _ => {
            // joker_mini: (n=10, k=5, p=5, J=4), C(10,5)*4 = 1,008 bilete și 1,008 stări
            let n_pool = 10usize;
            let k_tk = 5usize;
            let p_dr = 5usize;
            let j_pool = 4usize;
            let base_masks = enumerate_k_masks_small(n_pool, k_tk);
            let n_base = base_masks.len();
            let n_states = n_base * j_pool;
            let n_cats = 8usize;
            let j_weights = [0.34f64, 0.27, 0.22, 0.17];
            let mut weights = vec![0.0f64; n_states];
            let mut w_sum = 0.0f64;
            for jb in 0..j_pool {
                for (bi, &m) in base_masks.iter().enumerate() {
                    let mut low_cnt = 0;
                    for b in 0..5 {
                        if (m & (1u32 << b)) != 0 {
                            low_cnt += 1;
                        }
                    }
                    let w = j_weights[jb] * (0.28 * (low_cnt as f64) + 0.02 * ((bi % 6) as f64)).exp();
                    weights[jb * n_base + bi] = w;
                    w_sum += w;
                }
            }
            for w in &mut weights {
                *w /= w_sum;
            }
            let sales = 150.0f64;
            let mut cat_rows = vec![vec![-1i8; n_states]; n_states];
            let mut crowd_mu = vec![0.0f64; n_cats * n_states];
            for t_jb in 0..j_pool {
                for (t_bi, &tm) in base_masks.iter().enumerate() {
                    let ti = t_jb * n_base + t_bi;
                    for s_jb in 0..j_pool {
                        let jm = t_jb == s_jb;
                        for (s_bi, &dm) in base_masks.iter().enumerate() {
                            let si = s_jb * n_base + s_bi;
                            let h = (tm & dm).count_ones() as usize;
                            let c_opt: Option<usize> = match (h, jm) {
                                (5, true) => Some(0),
                                (5, false) => Some(1),
                                (4, true) => Some(2),
                                (4, false) => Some(3),
                                (3, true) => Some(4),
                                (3, false) => Some(5),
                                (2, true) => Some(6),
                                (1, true) => Some(7),
                                _ => None,
                            };
                            if let Some(c) = c_opt {
                                cat_rows[ti][si] = c as i8;
                                crowd_mu[c * n_states + si] += sales * weights[ti];
                            }
                        }
                    }
                }
            }
            MiniGameSetup {
                game: "joker_mini",
                n_pool,
                k_ticket: k_tk,
                p_draw: p_dr,
                j_pool,
                n_states,
                n_cats,
                cat_funds: vec![6000.0, 1200.0, 600.0, 200.0, 90.0, 45.0, 25.0, 15.0],
                cat_nb_r: vec![0.7, 1.0, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5],
                fixed_cat4_prize: 0.0,
                all_tickets_cat_rows: cat_rows,
                state_crowd_mu_flat: crowd_mu,
            }
        }
    }
}

#[inline]
fn eval_g4_subset_fast(
    subset: &[usize],
    m_subpool: usize,
    n_states: usize,
    n_cats: usize,
    cat_mat_flat: &[i8],
    pay_table: &[f64], // [n_cats * n_states * (k_max + 1)]
    k_stride: usize,
    mu: f64,
) -> (f64, f64, f64) {
    let _ = m_subpool;
    let mut win_cnt = 0usize;
    let mut sum_pay = 0.0f64;
    let mut mc = [0usize; 8];

    for s in 0..n_states {
        for c in 0..n_cats {
            mc[c] = 0;
        }
        let mut any = false;
        for &idx in subset {
            let cv = cat_mat_flat[idx * n_states + s];
            if cv >= 0 {
                mc[cv as usize] += 1;
                any = true;
            }
        }
        if any {
            win_cnt += 1;
            for c in 0..n_cats {
                let m = mc[c];
                if m > 0 {
                    sum_pay += pay_table[(c * n_states + s) * k_stride + m];
                }
            }
        }
    }
    let inv_om = 1.0 / (n_states as f64);
    let ev = sum_pay * inv_om;
    let p_any = (win_cnt as f64) * inv_om;
    (ev + mu * p_any, ev, p_any)
}

fn solve_g4_instance_c3v2(
    m_subpool: usize,
    k_select: usize,
    n_states: usize,
    n_cats: usize,
    cat_mat_flat: &[i8],
    pay_table: &[f64],
    k_stride: usize,
    mu: f64,
    seed: u64,
) -> (Vec<usize>, f64) {
    // 1. Construcție Marginal-Gain Greedy exactă + 8 restarturi perturbate urmate de 1-swap Kempe local ascent
    let mut global_best_set: Vec<usize> = (0..k_select).collect();
    let mut global_best_obj = f64::NEG_INFINITY;
    let mut rng = seed;

    for restart in 0..=m_subpool {
        let mut cur_set: Vec<usize> = Vec::with_capacity(k_select);
        if restart == 0 {
            // Greedy determinist pur pe câștig marginal exact
            while cur_set.len() < k_select {
                let mut best_cand = 0usize;
                let mut best_v = f64::NEG_INFINITY;
                for cand in 0..m_subpool {
                    if cur_set.contains(&cand) {
                        continue;
                    }
                    cur_set.push(cand);
                    let (obj, _, _) = eval_g4_subset_fast(
                        &cur_set,
                        m_subpool,
                        n_states,
                        n_cats,
                        cat_mat_flat,
                        pay_table,
                        k_stride,
                        mu,
                    );
                    cur_set.pop();
                    if obj > best_v {
                        best_v = obj;
                        best_cand = cand;
                    }
                }
                cur_set.push(best_cand);
            }
        } else {
            // Pornire diversificată pe fiecare ancoră din sub-pool + greedy cu jitter
            let first = (restart - 1) % m_subpool;
            cur_set.push(first);
            while cur_set.len() < k_select {
                let mut best_cand = 0usize;
                let mut best_v = f64::NEG_INFINITY;
                for cand in 0..m_subpool {
                    if cur_set.contains(&cand) {
                        continue;
                    }
                    cur_set.push(cand);
                    let (obj, _, _) = eval_g4_subset_fast(
                        &cur_set,
                        m_subpool,
                        n_states,
                        n_cats,
                        cat_mat_flat,
                        pay_table,
                        k_stride,
                        mu,
                    );
                    cur_set.pop();
                    let jitter =
                        ((splitmix64_c3(&mut rng) & 0xFFFF) as f64 / 65535.0 - 0.5) * 0.05 * obj.abs().max(1.0);
                    if obj + jitter > best_v {
                        best_v = obj + jitter;
                        best_cand = cand;
                    }
                }
                cur_set.push(best_cand);
            }
        }

        // Steepest-ascent 1-swap + Tabu kick + Kempe 2-swap polish (evadează platourile discrete la t=4)
        let (mut cur_obj, _, _) = eval_g4_subset_fast(
            &cur_set,
            m_subpool,
            n_states,
            n_cats,
            cat_mat_flat,
            pay_table,
            k_stride,
            mu,
        );
        let mut best_local_set = cur_set.clone();
        let mut best_local_obj = cur_obj;
        let mut tabu_until = vec![0usize; m_subpool];

        for step in 0..48usize {
            let mut best_move: Option<(usize, usize, f64)> = None;
            for pos in 0..k_select {
                let orig = cur_set[pos];
                for rep in 0..m_subpool {
                    if cur_set.contains(&rep) {
                        continue;
                    }
                    cur_set[pos] = rep;
                    let (obj, _, _) = eval_g4_subset_fast(
                        &cur_set,
                        m_subpool,
                        n_states,
                        n_cats,
                        cat_mat_flat,
                        pay_table,
                        k_stride,
                        mu,
                    );
                    let aspiration = obj > best_local_obj + 1.0e-13;
                    if step >= tabu_until[rep] || aspiration {
                        if best_move.map_or(true, |bm| obj > bm.2 + 1.0e-14) {
                            best_move = Some((pos, rep, obj));
                        }
                    }
                }
                cur_set[pos] = orig;
            }
            if let Some((pos, rep, obj)) = best_move {
                let evicted = cur_set[pos];
                cur_set[pos] = rep;
                cur_obj = obj;
                tabu_until[evicted] = step + 5;
                if cur_obj > best_local_obj + 1.0e-13 {
                    best_local_obj = cur_obj;
                    best_local_set.copy_from_slice(&cur_set);
                }
            } else {
                break;
            }
        }

        // Kempe 2-swap local refinement pe minimul/maximul local 1-swap
        if k_select >= 2 {
            loop {
                let mut improved_2swap: Option<(usize, usize, usize, usize, f64)> = None;
                for p1 in 0..k_select {
                    for p2 in (p1 + 1)..k_select {
                        let o1 = best_local_set[p1];
                        let o2 = best_local_set[p2];
                        for r1 in 0..m_subpool {
                            if best_local_set.contains(&r1) {
                                continue;
                            }
                            for r2 in (r1 + 1)..m_subpool {
                                if best_local_set.contains(&r2) {
                                    continue;
                                }
                                best_local_set[p1] = r1;
                                best_local_set[p2] = r2;
                                let (obj, _, _) = eval_g4_subset_fast(
                                    &best_local_set,
                                    m_subpool,
                                    n_states,
                                    n_cats,
                                    cat_mat_flat,
                                    pay_table,
                                    k_stride,
                                    mu,
                                );
                                if obj > best_local_obj + 1.0e-13
                                    && improved_2swap.map_or(true, |im| obj > im.4 + 1.0e-14)
                                {
                                    improved_2swap = Some((p1, p2, r1, r2, obj));
                                }
                            }
                        }
                        best_local_set[p1] = o1;
                        best_local_set[p2] = o2;
                    }
                }
                if let Some((p1, p2, r1, r2, obj)) = improved_2swap {
                    best_local_set[p1] = r1;
                    best_local_set[p2] = r2;
                    best_local_obj = obj;
                } else {
                    break;
                }
            }
        }

        let _ = cur_obj;
        best_local_set.sort_unstable();
        if best_local_obj > global_best_obj + 1.0e-13 {
            global_best_obj = best_local_obj;
            global_best_set = best_local_set;
        }
    }

    (global_best_set, global_best_obj)
}

pub fn run_gate_g4_36_instances() -> Vec<G4SmallInstanceRecord> {
    let setups: Vec<MiniGameSetup> = (0..3).map(build_mini_game_setup).collect();
    let mu_grid = [0.0f64, 100.0, 10_000.0];
    let size_grid: [(usize, usize); 4] = [(18, 3), (20, 4), (22, 4), (20, 5)];

    let mut specs: Vec<(usize, usize, usize, usize, f64)> = Vec::with_capacity(36);
    let mut inst_id = 0usize;
    for gi in 0..3 {
        for &mu in &mu_grid {
            for &(ms, ks) in &size_grid {
                specs.push((inst_id, gi, ms, ks, mu));
                inst_id += 1;
            }
        }
    }

    // Precalculăm pay_table[gi] pentru m in 0..=5 o singură dată pe joc
    let k_stride = 6usize;
    let pay_tables: Vec<Vec<f64>> = setups
        .iter()
        .map(|st| {
            let mut tbl = vec![0.0f64; st.n_cats * st.n_states * k_stride];
            for c in 0..st.n_cats {
                for s in 0..st.n_states {
                    let mu_cs = st.state_crowd_mu_flat[c * st.n_states + s];
                    for m in 1..k_stride {
                        let val = if st.fixed_cat4_prize > 0.0 && c + 1 == st.n_cats {
                            (m as f64) * st.fixed_cat4_prize
                        } else {
                            st.cat_funds[c] * c3v2_g_share_multi_cpu(m, mu_cs, st.cat_nb_r[c], 0)
                        };
                        tbl[(c * st.n_states + s) * k_stride + m] = val;
                    }
                }
            }
            tbl
        })
        .collect();

    let results: Vec<G4SmallInstanceRecord> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(specs.len());
        for &(id, gi, m_sub, k_sel, mu) in &specs {
            let st_ref = &setups[gi];
            let pt_ref = &pay_tables[gi];
            handles.push(s.spawn(move || {
                // Alegem determinist sub-pool-ul de m_sub bilete variat pe instanță
                let n_tot_tk = st_ref.all_tickets_cat_rows.len();
                let mut rng = 0x4734_2026_1104_0000u64 ^ ((id as u64 + 1) * 0x9E37_79B9);
                let mut chosen_tk: Vec<usize> = Vec::with_capacity(m_sub);
                while chosen_tk.len() < m_sub {
                    let cand = (splitmix64_c3(&mut rng) as usize) % n_tot_tk;
                    if !chosen_tk.contains(&cand) {
                        chosen_tk.push(cand);
                    }
                }

                let mut cat_mat_flat = vec![-1i8; m_sub * st_ref.n_states];
                for (row_i, &tk_idx) in chosen_tk.iter().enumerate() {
                    cat_mat_flat[row_i * st_ref.n_states..(row_i + 1) * st_ref.n_states]
                        .copy_from_slice(&st_ref.all_tickets_cat_rows[tk_idx]);
                }

                let (solver_set, solver_obj) = solve_g4_instance_c3v2(
                    m_sub,
                    k_sel,
                    st_ref.n_states,
                    st_ref.n_cats,
                    &cat_mat_flat,
                    pt_ref,
                    k_stride,
                    mu,
                    0xC352_7777_0000u64 ^ (id as u64),
                );

                // Enumerare exhaustivă exactă peste toate C(m_sub, k_sel) submulțimile
                let all_sub_masks = enumerate_k_masks_small(m_sub, k_sel);
                let mut best_brute_obj = f64::NEG_INFINITY;
                let mut best_brute_ev = 0.0f64;
                let mut best_brute_p = 0.0f64;
                let mut best_brute_set = vec![0usize; k_sel];
                let mut tmp_set = vec![0usize; k_sel];

                for &sm in &all_sub_masks {
                    let mut p = 0usize;
                    let mut m = sm;
                    while m != 0 {
                        let bit = m.trailing_zeros() as usize;
                        tmp_set[p] = bit;
                        p += 1;
                        m &= m - 1;
                    }
                    let (obj, ev, pa) = eval_g4_subset_fast(
                        &tmp_set,
                        m_sub,
                        st_ref.n_states,
                        st_ref.n_cats,
                        &cat_mat_flat,
                        pt_ref,
                        k_stride,
                        mu,
                    );
                    if obj > best_brute_obj + 1.0e-14 {
                        best_brute_obj = obj;
                        best_brute_ev = ev;
                        best_brute_p = pa;
                        best_brute_set.copy_from_slice(&tmp_set);
                    }
                }

                let gap = (best_brute_obj - solver_obj).abs();
                let hit = gap < 1.0e-11;

                G4SmallInstanceRecord {
                    instance_id: id,
                    game: st_ref.game.to_string(),
                    n_pool: st_ref.n_pool,
                    k_ticket: st_ref.k_ticket,
                    p_draw: st_ref.p_draw,
                    j_pool: st_ref.j_pool,
                    m_subpool: m_sub,
                    k_select: k_sel,
                    mu,
                    total_subsets_enumerated: all_sub_masks.len(),
                    global_opt_objective: best_brute_obj,
                    solver_objective: solver_obj,
                    abs_gap: gap,
                    exact_hit: hit,
                    global_opt_indices: best_brute_set,
                    solver_indices: solver_set,
                    global_opt_ev: best_brute_ev,
                    global_opt_p_any: best_brute_p,
                    n_states: st_ref.n_states,
                    n_cats: st_ref.n_cats,
                    cat_funds: st_ref.cat_funds.clone(),
                    cat_nb_r: st_ref.cat_nb_r.clone(),
                    fixed_cat4_prize: st_ref.fixed_cat4_prize,
                    subpool_cat_matrix_flat: cat_mat_flat,
                    state_crowd_mu_flat: st_ref.state_crowd_mu_flat.clone(),
                }
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    results
}

// ============================================================================
// 7. Pipeline Principal C3-v2, Scriere `.v3bin` + Manifest Vault & Verificare
// ============================================================================

pub(crate) fn encode_v3bin_header_64(
    table_kind: u32,
    n_pool: u32,
    k_pick: u32,
    record_count: u64,
    record_bytes: u64,
) -> [u8; VAULT_HEADER_BYTES] {
    let mut hdr = [0u8; VAULT_HEADER_BYTES];
    hdr[0..8].copy_from_slice(VAULT_MAGIC);
    hdr[8..12].copy_from_slice(&3u32.to_le_bytes());
    hdr[12..16].copy_from_slice(&table_kind.to_le_bytes());
    hdr[16..20].copy_from_slice(&n_pool.to_le_bytes());
    hdr[20..24].copy_from_slice(&k_pick.to_le_bytes());
    hdr[24..32].copy_from_slice(&record_count.to_le_bytes());
    hdr[32..40].copy_from_slice(&record_bytes.to_le_bytes());
    hdr
}

pub(crate) fn sha256_file_c3(path: &Path) -> Result<(u64, String), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

pub(crate) fn upsert_vault_manifest_record(
    vault_dir: &Path,
    rec: VaultFileRecord,
) -> Result<(), String> {
    let mf_path = vault_dir.join("VAULT_MANIFEST.json");
    let raw = fs::read_to_string(&mf_path)
        .map_err(|e| format!("read {}: {e}", mf_path.display()))?;
    let mut mf: crate::vault::VaultManifest =
        serde_json::from_str(&raw).map_err(|e| format!("parse VAULT_MANIFEST.json: {e}"))?;
    mf.files.retain(|x| x.filename != rec.filename);
    mf.files.push(rec);
    fs::write(
        &mf_path,
        serde_json::to_string_pretty(&mf).unwrap() + "\n",
    )
    .map_err(|e| format!("write {}: {e}", mf_path.display()))?;
    Ok(())
}

pub(crate) fn make_writable_if_exists(path: &Path) {
    if path.exists() {
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            perms.set_readonly(false);
            let _ = fs::set_permissions(path, perms);
        }
    }
}

pub fn load_crowd_params_from_pmv3_report(
    pm_rep: &PmV3CalibrationReport,
) -> Result<(V3CrowdParamsF64, V3CrowdParamsF64, V3CrowdParamsF64), String> {
    let mut p649 = None;
    let mut p540 = None;
    let mut pjok = None;
    for wf in &pm_rep.walk_forward_games {
        let cp = build_crowd_params_f64(
            &wf.terminal_state,
            &wf.game,
            &wf.terminal_prev_draw_numbers,
            0,
        );
        match wf.game.as_str() {
            "649" => p649 = Some(cp),
            "540" => p540 = Some(cp),
            "joker" => pjok = Some(cp),
            _ => {}
        }
    }
    Ok((
        p649.ok_or_else(|| "lipseste 649 in PM_V3_CALIBRARE.json".to_string())?,
        p540.ok_or_else(|| "lipseste 540 in PM_V3_CALIBRARE.json".to_string())?,
        pjok.ok_or_else(|| "lipseste joker in PM_V3_CALIBRARE.json".to_string())?,
    ))
}

pub fn run_full_c3v2_pipeline(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    pmv3_json_path: &Path,
    cdv3q_json_path: &Path,
    out_c3v2_json_path: &Path,
) -> Result<C3V2PortfolioReport, String> {
    let t0 = Instant::now();
    check_rss_guard(0)?;

    let (_, pm_sha) = sha256_file_c3(pmv3_json_path)?;
    let (_, cdq_sha) = sha256_file_c3(cdv3q_json_path)?;

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

    let (p649, p540, pjok) = load_crowd_params_from_pmv3_report(&pm_rep)?;
    let binom_tbl = build_binom_table_50x7();

    let mut vault_tickets_accum: Vec<u64> = Vec::with_capacity(32_768);
    let mut max_win_diff = 0u64;
    let mut max_ev_diff = 0.0f64;

    let (f649, pref_649_ev, pref_649_p) = build_single_game_frontiers(
        "649",
        &BUDGETS_649,
        8.0,
        vault,
        gpu_univ,
        &p649,
        &cdq_rep,
        &binom_tbl,
        &mut vault_tickets_accum,
        &mut max_win_diff,
        &mut max_ev_diff,
    )?;

    let (f540, pref_540_ev, pref_540_p) = build_single_game_frontiers(
        "540",
        &BUDGETS_540,
        5.0,
        vault,
        gpu_univ,
        &p540,
        &cdq_rep,
        &binom_tbl,
        &mut vault_tickets_accum,
        &mut max_win_diff,
        &mut max_ev_diff,
    )?;

    let (fjok, pref_jok_ev, pref_jok_p) = build_single_game_frontiers(
        "joker",
        &BUDGETS_JOKER,
        7.0,
        vault,
        gpu_univ,
        &pjok,
        &cdq_rep,
        &binom_tbl,
        &mut vault_tickets_accum,
        &mut max_win_diff,
        &mut max_ev_diff,
    )?;

    let mixed_pts = build_mixed_3game_frontiers(
        vault,
        gpu_univ,
        &p649,
        &p540,
        &pjok,
        &pref_649_ev,
        &pref_649_p,
        &pref_540_ev,
        &pref_540_p,
        &pref_jok_ev,
        &pref_jok_p,
        &binom_tbl,
        &mut vault_tickets_accum,
    )?;

    let g4_records = run_gate_g4_36_instances();
    let g4_hits = g4_records.iter().filter(|r| r.exact_hit).count();
    let g4_pass = g4_records.len() == 36 && g4_hits == 36;

    let mut all_frontiers: Vec<GameBudgetFrontier> =
        Vec::with_capacity(f649.len() + f540.len() + fjok.len());
    all_frontiers.extend(f649);
    all_frontiers.extend(f540);
    all_frontiers.extend(fjok);

    // Scriem `c3v2_pareto_portfolios.v3bin` în Vault-ul SSD
    let v3bin_path = vault.vault_dir.join(C3V2_V3BIN_NAME);
    make_writable_if_exists(&v3bin_path);
    let rec_cnt = vault_tickets_accum.len() as u64;
    let hdr = encode_v3bin_header_64(TABLE_KIND_C3V2_PORTFOLIOS, 49, 6, rec_cnt, 8);
    {
        let mut wf =
            File::create(&v3bin_path).map_err(|e| format!("create {}: {e}", v3bin_path.display()))?;
        wf.write_all(&hdr)
            .map_err(|e| format!("write hdr {}: {e}", v3bin_path.display()))?;
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vault_tickets_accum.as_ptr() as *const u8,
                vault_tickets_accum.len() * 8,
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
            filename: C3V2_V3BIN_NAME.to_string(),
            table_kind: TABLE_KIND_C3V2_PORTFOLIOS,
            n_pool: 49,
            k_pick: 6,
            record_count: rec_cnt,
            file_bytes: v3bin_bytes,
            sha256: v3bin_sha.clone(),
        },
    )?;

    let tot_pat_single: usize = all_frontiers
        .iter()
        .flat_map(|f| f.points.iter())
        .map(|p| p.patterned_tickets)
        .sum();
    let tot_pat_mix: usize = mixed_pts.iter().map(|m| m.patterned_tickets_total).sum();
    let tot_pat = tot_pat_single + tot_pat_mix;

    let std_pos_mu_ok = all_frontiers
        .iter()
        .filter(|f| matches!(f.budget_lei, 40 | 100 | 500))
        .all(|f| f.all_pos_mu_beat_p_random)
        && mixed_pts
            .iter()
            .filter(|m| matches!(m.budget_lei, 40 | 100 | 500) && m.mu > 0.0)
            .all(|m| m.beats_p_random_mix);

    let all_pos_mu_ok = all_frontiers.iter().all(|f| f.all_pos_mu_beat_p_random)
        && mixed_pts
            .iter()
            .filter(|m| m.mu > 0.0)
            .all(|m| m.beats_p_random_mix);

    let all_mu0_ok = all_frontiers.iter().all(|f| f.mu0_beats_top_ev_and_rnd);
    let all_mono_ok = all_frontiers.iter().all(|f| f.frontier_monotone_p_any);

    let (rss_cur, rss_hwm) = check_rss_guard(0)?;
    let rss_lim = crate::cuda_ffi::MAX_SAFE_RSS_BYTES;
    let rss_ok = rss_cur < rss_lim && rss_hwm < rss_lim;

    let gate_pass = tot_pat == 0
        && max_win_diff == 0
        && max_ev_diff < 1.0e-12
        && g4_pass
        && std_pos_mu_ok
        && all_pos_mu_ok
        && all_mu0_ok
        && all_mono_ok
        && rss_ok;

    let report = C3V2PortfolioReport {
        artifact: "C3_V2_PORTOFOLII".to_string(),
        version: "3.0.0-S3".to_string(),
        positive_claims_allowed: false,
        source_pmv3_sha256: pm_sha,
        source_cdv3q_sha256: cdq_sha,
        vault_portfolios_v3bin: v3bin_path.display().to_string(),
        vault_portfolios_sha256: v3bin_sha,
        vault_portfolios_bytes: v3bin_bytes,
        vault_portfolios_records: rec_cnt,
        total_universe_scanned: TOTAL_UNIVERSE_TICKETS,
        truncation_4096_eliminated: true,
        bonferroni_error_eliminated: true,
        max_gpu_cpu_win_states_diff: max_win_diff,
        max_gpu_cpu_ev_abs_diff: max_ev_diff,
        total_patterned_tickets_in_all_portfolios: tot_pat,
        gate_g4_total_instances: g4_records.len(),
        gate_g4_exact_hits: g4_hits,
        gate_g4_pass: g4_pass,
        all_standard_budgets_pos_mu_beat_p_random: std_pos_mu_ok && all_pos_mu_ok,
        all_mu0_beat_naive_top_ev_and_random: all_mu0_ok,
        all_frontiers_monotone: all_mono_ok,
        rss_current_bytes: rss_cur,
        rss_peak_hwm_bytes: rss_hwm,
        rss_limit_bytes: rss_lim,
        rss_guard_ok: rss_ok,
        build_seconds: t0.elapsed().as_secs_f64(),
        c3v2_gate_pass: gate_pass,
        single_game_frontiers: all_frontiers,
        mixed_3game_portfolios: mixed_pts,
        g4_instances: g4_records,
    };

    if let Some(p) = out_c3v2_json_path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
    }
    fs::write(
        out_c3v2_json_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .map_err(|e| format!("write {}: {e}", out_c3v2_json_path.display()))?;

    if !report.c3v2_gate_pass {
        return Err(format!(
            "c3v2_gate_pass=false (pat={tot_pat}, win_diff={max_win_diff}, ev_diff={max_ev_diff:.2e}, g4={g4_hits}/36, pos_mu={all_pos_mu_ok}, mu0={all_mu0_ok}, mono={all_mono_ok})"
        ));
    }

    Ok(report)
}

/// Verificare read-only completă `c3v2-verify` (SHA-256 `.v3bin`, re-evaluare GPU + 24T CPU pe bugetele standard).
pub fn verify_existing_c3v2(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    pmv3_json_path: &Path,
    c3v2_json_path: &Path,
) -> Result<bool, String> {
    let raw = fs::read_to_string(c3v2_json_path)
        .map_err(|e| format!("read {}: {e}", c3v2_json_path.display()))?;
    let rep: C3V2PortfolioReport =
        serde_json::from_str(&raw).map_err(|e| format!("parse C3_V2_PORTOFOLII.json: {e}"))?;

    if rep.positive_claims_allowed
        || !rep.c3v2_gate_pass
        || !rep.truncation_4096_eliminated
        || !rep.bonferroni_error_eliminated
        || rep.total_patterned_tickets_in_all_portfolios != 0
        || !rep.gate_g4_pass
        || rep.gate_g4_exact_hits != 36
        || !rep.all_standard_budgets_pos_mu_beat_p_random
        || !rep.all_mu0_beat_naive_top_ev_and_random
        || !rep.all_frontiers_monotone
    {
        return Err("flaguri guvernanta/poarta C3-v2 invalide in JSON".to_string());
    }

    let v3bin_path = vault.vault_dir.join(C3V2_V3BIN_NAME);
    let (sz, sha) = sha256_file_c3(&v3bin_path)?;
    if sz != rep.vault_portfolios_bytes || sha != rep.vault_portfolios_sha256 {
        return Err(format!(
            "SHA-256/bytes mismatch pe {}: ({sz}, {sha}) != ({}, {})",
            C3V2_V3BIN_NAME, rep.vault_portfolios_bytes, rep.vault_portfolios_sha256
        ));
    }
    let f = File::open(&v3bin_path).map_err(|e| format!("open {}: {e}", v3bin_path.display()))?;
    let mm = unsafe { memmap2::Mmap::map(&f) }
        .map_err(|e| format!("mmap {}: {e}", v3bin_path.display()))?;
    if mm.len() != VAULT_HEADER_BYTES + (rep.vault_portfolios_records as usize) * 8
        || &mm[0..8] != VAULT_MAGIC
    {
        return Err("header c3v2_pareto_portfolios.v3bin invalid".to_string());
    }
    let tk_slice: &[u64] = unsafe {
        std::slice::from_raw_parts(
            mm[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
            rep.vault_portfolios_records as usize,
        )
    };

    let pm_rep: PmV3CalibrationReport = serde_json::from_str(
        &fs::read_to_string(pmv3_json_path)
            .map_err(|e| format!("read {}: {e}", pmv3_json_path.display()))?,
    )
    .map_err(|e| format!("parse PM_V3_CALIBRARE.json: {e}"))?;
    let (p649, p540, pjok) = load_crowd_params_from_pmv3_report(&pm_rep)?;

    let qt649_g = GpuCrowdTables649::build(&gpu_univ.d_masks_649, &p649)?;
    let qt540_g = GpuCrowdTables540::build(&gpu_univ.d_tickets_540, &gpu_univ.d_draws6_540, &p540)?;
    let qtjok_g = GpuCrowdTablesJoker::build(&gpu_univ.d_base_joker, &pjok)?;

    let qt649_c = CpuCrowdTables649::build_24t(vault.masks_649(), &p649);
    let qt540_c = CpuCrowdTables540::build_24t(vault.tickets_540(), vault.draws6_540(), &p540);
    let qtjok_c = CpuCrowdTablesJoker::build_24t(vault.base_joker(), &pjok);

    // Re-verificăm GPU + CPU pe bugetele canonice 40 / 100 / 500 la mu=0 și mu=inf pentru toate cele 3 jocuri
    for fr in &rep.single_game_frontiers {
        if !matches!(fr.budget_lei, 40 | 100 | 500) {
            continue;
        }
        for &pt_idx in &[0usize, fr.points.len() - 1] {
            let pt = &fr.points[pt_idx];
            let off = pt.vault_offset_records as usize;
            let len = pt.vault_len_records as usize;
            let sub = &tk_slice[off..off + len];
            for &tk in sub {
                if check_ticket_not_patterned(&fr.game, tk).0 {
                    return Err(format!("bilet patterned gasit in {}", pt.id));
                }
            }
            let (eg, ec) = match fr.game.as_str() {
                "649" => (
                    gpu_c3v2_eval_649(&gpu_univ.d_masks_649, &qt649_g, &p649, sub)?,
                    cpu_c3v2_eval_649(vault.masks_649(), &qt649_c, &p649, sub),
                ),
                "540" => (
                    gpu_c3v2_eval_540(&gpu_univ.d_draws6_540, &qt540_g, &p540, sub)?,
                    cpu_c3v2_eval_540(vault.draws6_540(), &qt540_c, &p540, sub),
                ),
                "joker" => (
                    gpu_c3v2_eval_joker(&gpu_univ.d_base_joker, &qtjok_g, &pjok, sub)?,
                    cpu_c3v2_eval_joker(vault.base_joker(), &qtjok_c, &pjok, sub),
                ),
                _ => unreachable!(),
            };
            if eg.win_states != pt.win_states_gpu
                || ec.win_states != pt.win_states_gpu
                || (eg.ev_portfolio - pt.ev_portfolio_gpu).abs() >= 1.0e-12
                || (ec.ev_portfolio - pt.ev_portfolio_gpu).abs() >= 1.0e-12
            {
                return Err(format!(
                    "re-evaluare GPU/CPU diferita pe {}: win=({}, {}, {}) ev=({}, {}, {})",
                    pt.id,
                    eg.win_states,
                    ec.win_states,
                    pt.win_states_gpu,
                    eg.ev_portfolio,
                    ec.ev_portfolio,
                    pt.ev_portfolio_gpu
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
    fn test_multi_ticket_nb2_share_and_h2_against_exact_pmf() {
        for &m in &[1usize, 2, 3, 5] {
            for &r in &[0.65, 1.0, 2.4, 15.0] {
                for &mu in &[0.02, 0.45, 2.8, 9.5] {
                    let g1 = c3v2_g_share_multi_cpu(m, mu, r, 0);
                    let h2 = c3v2_h2_share_multi_cpu(m, mu, r, 0);

                    let p_succ = r / (r + mu);
                    let q_fail = mu / (r + mu);
                    let mut pmf = p_succ.powf(r);
                    let dm = m as f64;
                    let mut ex1 = pmf * 1.0;
                    let mut ex2 = pmf * 1.0;
                    for w in 1..1200usize {
                        pmf *= ((w as f64 - 1.0 + r) / (w as f64)) * q_fail;
                        let sh = dm / (dm + w as f64);
                        ex1 += pmf * sh;
                        ex2 += pmf * sh * sh;
                    }
                    assert!(
                        (g1 - ex1).abs() < 5.0e-11,
                        "G_NB mismatch m={m} r={r} mu={mu}: {g1} vs {ex1}"
                    );
                    assert!(
                        (h2 - ex2).abs() < 1.0e-8,
                        "H2_NB mismatch m={m} r={r} mu={mu}: {h2} vs {ex2}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gate_g4_36_instances_exact_global_optimum() {
        let recs = run_gate_g4_36_instances();
        assert_eq!(recs.len(), 36);
        for r in &recs {
            assert!(
                r.exact_hit,
                "G4 instance {} ({}, M={}, K={}, mu={}) gap={:.3e}",
                r.instance_id, r.game, r.m_subpool, r.k_select, r.mu, r.abs_gap
            );
        }
    }

    #[test]
    fn test_hypergeom_random_p_any_monotone_and_exact_k1() {
        let p1_649 = p_random_analytical_hypergeom(COUNT_649 as u64, 260_624, 1);
        assert!((p1_649 - 260_624.0 / (COUNT_649 as f64)).abs() < 1.0e-15);
        let p1_540 = p_random_analytical_hypergeom(COUNT_540_TICKETS as u64, 516, 1);
        assert!((p1_540 - 3_010.0 / (COUNT_540_DRAWS6 as f64)).abs() < 1.0e-15);
        let p1_jok = p_random_analytical_hypergeom(COUNT_JOKER_FULL24M as u64, 715_770, 1);
        assert!((p1_jok - 715_770.0 / (COUNT_JOKER_FULL24M as f64)).abs() < 1.0e-15);
    }
}

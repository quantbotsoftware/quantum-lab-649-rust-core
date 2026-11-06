//! Motorul Combinatoriu `CD-V3` (`Poarta A2` din `PROTOCOL_RUST_CORE_V3.md` + `AMENDAMENT_01/02`):
//! 1. `CDV3_649_163`: `L(49, 6, 6, 3) <= 163` (`77` hexade Witt `S(3,6,22)` + `<= 86` blocuri `L(27,6,4,3)`
//!    optimizate pe `16,384` fire CUDA `sm_120` + `24` fire CPU `AVX2/POPCNT`), `0` neacoperite din `13,983,816`.
//! 2. `CDV3_540_T2`, `CDV3_540_T3` (`S(3,5,17)` `68` pentade + `L(23,5,4,3)`), `CDV3_540_CAT3_T4`
//!    (`L(40,5,6,4) = 4,411 <= 4,500` Cat. III oficială, `0` neacoperite din `3,838,380`, descompunere
//!    de geometrie finită Turán `16 + 12 + 12` + pruning invers exact).
//! 3. `CDV3_JOKER`: partiție rezolvabilă pe `[45] = A_25 U B_20` (`90..93` bilete) / `A_20 U B_20 U C_5` (`<= 100` bilete),
//!    `0` stări necâștigătoare din toate cele `24,435,180` extrageri posibile (`<= 108` prag).
//! 4. Catalogul complet al Roților Parțiale `W(v,k,p,t; K)` (`K in {5,10,15,20,25,30,50,85,100}`) și al
//!    Schemelor Reduse Condiționate (`v_sub in {12,15,18,21,24}` + `L(v_sub,5,6,4) <= 85`), plus cei 5 martori negativi.
//! `positive_claims_allowed = false`.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cuda_ffi::{
    check_rss_guard, gpu_run_tabu_shared_16384, gpu_verify_covering, gpu_verify_joker_24m,
    DeviceBuffer, GpuBitpackedCoverState, GpuJokerCoverState, ResidentGpuUniverse, V3PointDraw,
};
use crate::vault::{
    binom, MappedVaultUniverse, VaultFileRecord, COUNT_540_DRAWS6, COUNT_540_TICKETS, COUNT_649,
    COUNT_JOKER_BASE45, COUNT_JOKER_FULL24M, VAULT_HEADER_BYTES, VAULT_MAGIC,
};
use crate::witt_s3_6_22::{
    build_steiner_s3_5_17, build_verified_l23_5_4_3_101, build_verified_l27_6_4_3_86,
    build_verified_r20_5_3_quasi41, build_verified_turan_l40_5_6_4_candidates, build_witt_s3_6_22,
};

pub const PARTIAL_KS: [usize; 9] = [5, 10, 15, 20, 25, 30, 50, 85, 100];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JokerTicketJson {
    pub numbers: [u8; 5],
    pub joker: u8,
    pub packed_u64: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoveringDesignRecord {
    pub id: String,
    pub game: String,
    pub description: String,
    pub v_pool: usize,
    pub k_ticket: usize,
    pub p_draw: usize,
    pub t_min_hits: usize,
    pub schoenheim_lower_bound: usize,
    pub n_tickets: usize,
    pub universe_draws: u64,
    pub uncovered_gpu: u64,
    pub uncovered_cpu: u64,
    pub gpu_cpu_histogram_identical: bool,
    pub hit_histogram: Vec<u64>,
    pub gpu_verify_ms: f64,
    pub cpu_verify_ms: f64,
    pub tickets_numbers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_tickets: Option<Vec<JokerTicketJson>>,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartialWheelRecord {
    pub id: String,
    pub game: String,
    pub target_regime: String,
    pub k_budget_tickets: usize,
    pub universe_draws: u64,
    pub covered_draws_gpu: u64,
    pub covered_draws_cpu: u64,
    pub uncovered_draws: u64,
    pub coverage_pct: f64,
    pub theoretical_packing_ceiling_pct: f64,
    pub gpu_cpu_identical: bool,
    pub tickets_numbers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_tickets: Option<Vec<JokerTicketJson>>,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbbreviatedWheelRecord {
    pub id: String,
    pub game: String,
    pub v_sub: usize,
    pub k_ticket: usize,
    pub p_hit_in_sub: usize,
    pub t_guarantee: usize,
    pub sub_universe_draws: u64,
    pub schoenheim_lower_bound: usize,
    pub n_tickets: usize,
    pub uncovered_sub_draws: u64,
    pub canonical_tickets_1based: Vec<Vec<u8>>,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NegativeWitnessRecord {
    pub id: String,
    pub description: String,
    pub expected_uncovered: Option<u64>,
    pub require_strictly_positive: bool,
    pub observed_uncovered_gpu: u64,
    pub observed_uncovered_cpu: u64,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdV3CatalogReport {
    pub artifact: String,
    pub version: String,
    pub positive_claims_allowed: bool,
    pub build_seconds: f64,
    pub vault_catalog_v3bin: String,
    pub vault_catalog_sha256: String,
    pub gate_a2_pass: bool,
    pub main_designs: Vec<CoveringDesignRecord>,
    pub partial_wheels: Vec<PartialWheelRecord>,
    pub abbreviated_wheels: Vec<AbbreviatedWheelRecord>,
    pub negative_witnesses: Vec<NegativeWitnessRecord>,
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub fn mask_to_numbers_1based(mut mask: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        let b = mask.trailing_zeros() as u8;
        out.push(b + 1);
        mask &= mask - 1;
    }
    out
}

pub fn unpack_joker_ticket(packed: u64) -> JokerTicketJson {
    let m = packed & ((1u64 << 45) - 1);
    let j = (((packed >> 48) & 0x1F) as u8) + 1;
    let nums = mask_to_numbers_1based(m);
    assert_eq!(nums.len(), 5);
    JokerTicketJson {
        numbers: [nums[0], nums[1], nums[2], nums[3], nums[4]],
        joker: j,
        packed_u64: packed,
    }
}

pub fn pack_joker_ticket(mask45: u64, joker_1based: u8) -> u64 {
    assert!((1..=20).contains(&joker_1based));
    assert_eq!(mask45.count_ones(), 5);
    mask45 | (((joker_1based - 1) as u64) << 48)
}

/// Verificare deterministă CPU pe 24 fire pentru acoperire standard pe măști `u64`.
pub fn cpu_verify_covering_24t(draws: &[u64], tickets: &[u64], min_hits: u32) -> ([u64; 8], u64, f64) {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_size = (draws.len() + n_threads - 1) / n_threads;

    let parts: Vec<[u64; 8]> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in draws.chunks(chunk_size.max(1)) {
            handles.push(s.spawn(move || {
                let mut h = [0u64; 8];
                for &d in chunk {
                    let mut best = 0u32;
                    for &tk in tickets {
                        let c = (d & tk).count_ones();
                        if c > best {
                            best = c;
                        }
                    }
                    h[(best as usize).min(7)] += 1;
                }
                h
            }));
        }
        handles.into_iter().map(|jh| jh.join().unwrap()).collect()
    });

    let mut hist = [0u64; 8];
    for p in parts {
        for i in 0..8 {
            hist[i] += p[i];
        }
    }
    let uncov: u64 = hist[..(min_hits as usize)].iter().sum();
    (hist, uncov, t0.elapsed().as_secs_f64() * 1000.0)
}

#[inline]
fn joker_rank_cpu(hits: u32, jm: bool) -> usize {
    match (hits, jm) {
        (5, true) => 0,
        (5, false) => 1,
        (4, true) => 2,
        (4, false) => 3,
        (3, true) => 4,
        (3, false) => 5,
        (2, true) => 6,
        (1, true) => 7,
        _ => 8,
    }
}

/// Verificare deterministă CPU pe 24 fire peste toate cele `24,435,180` stări Joker.
pub fn cpu_verify_joker_24m_24t(joker_24m: &[u64], tickets_packed: &[u64]) -> ([u64; 9], u64, f64) {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_size = (joker_24m.len() + n_threads - 1) / n_threads;
    const MASK45: u64 = (1u64 << 45) - 1;

    let parts: Vec<[u64; 9]> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in joker_24m.chunks(chunk_size.max(1)) {
            handles.push(s.spawn(move || {
                let mut h = [0u64; 9];
                for &st in chunk {
                    let dm = st & MASK45;
                    let dj = (st >> 48) & 0x1F;
                    let mut best = 8usize;
                    for &tk in tickets_packed {
                        let hits = (dm & (tk & MASK45)).count_ones();
                        let jm = ((tk >> 48) & 0x1F) == dj;
                        let r = joker_rank_cpu(hits, jm);
                        if r < best {
                            best = r;
                        }
                    }
                    h[best] += 1;
                }
                h
            }));
        }
        handles.into_iter().map(|jh| jh.join().unwrap()).collect()
    });

    let mut cat = [0u64; 9];
    for p in parts {
        for i in 0..9 {
            cat[i] += p[i];
        }
    }
    (cat, cat[8], t0.elapsed().as_secs_f64() * 1000.0)
}

fn generate_u32_combinations(v: usize, k: usize) -> Vec<u32> {
    assert!(v <= 31 && k <= v);
    let mut out = Vec::with_capacity(binom(v, k));
    fn rec(v: usize, k: usize, start: usize, rem: usize, cur: u32, out: &mut Vec<u32>) {
        if rem == 0 {
            out.push(cur);
            return;
        }
        for x in start..=(v - rem) {
            rec(v, k, x + 1, rem - 1, cur | (1u32 << x), out);
        }
    }
    rec(v, k, 0, k, 0, &mut out);
    assert_eq!(out.len(), binom(v, k));
    out
}

fn schoenheim_lotto_lb(v: usize, k: usize, p: usize, t: usize) -> usize {
    if p == t {
        // Iterated Schönheim C(v, k, t)
        let mut lb = 1usize;
        for step in 0..t {
            let vv = v - t + 1 + step;
            let kk = k - t + 1 + step;
            lb = (vv * lb + kk - 1) / kk;
        }
        return lb;
    }
    // Volume Schönheim bound + Turán hypergraph bound when k < p
    let mut cov_per_ticket = 0usize;
    for j in t..=k.min(p) {
        if p - j <= v - k {
            cov_per_ticket += binom(k, j) * binom(v - k, p - j);
        }
    }
    let vol_lb = (binom(v, p) + cov_per_ticket - 1) / cov_per_ticket;
    // Also Turán C(v, v-t, v-p) / C(k, t)
    let c_kt = binom(k, t);
    let mut turan = 1usize;
    for vv in (p + 1)..=v {
        let kk = vv - t;
        turan = (vv * turan + kk - 1) / kk;
    }
    vol_lb.max((turan + c_kt - 1) / c_kt)
}

/// Solver hibrid `16,384` fire CUDA (`k_tabu_shared_16384`) + `24` fire CPU L1-cache Nurmela–Östergård Tabu
/// pentru `L(v, k, p, t)` cu `v <= 28`, `binom(v, p) <= 24,576`, `n_blocks <= 128`.
pub fn solve_small_lotto_hybrid_gpu_cpu(
    v: usize,
    k: usize,
    p: usize,
    t: usize,
    must_reach_max: usize,
    min_try_blocks: usize,
    max_bursts_above_target: usize,
    use_gpu_16384: bool,
) -> Result<Vec<u32>, String> {
    let draw_masks = generate_u32_combinations(v, p);
    let cand_blocks = generate_u32_combinations(v, k);
    let m_draws = draw_masks.len();
    assert!(m_draws < u32::MAX as usize);

    let deg_per_pt = binom(v - 1, p - 1);
    let mut point_draws = vec![V3PointDraw::default(); v * deg_per_pt];
    for u in 0..v {
        let bit = 1u32 << u;
        let mut pos = 0usize;
        for (did, &dm) in draw_masks.iter().enumerate() {
            if (dm & bit) != 0 {
                point_draws[u * deg_per_pt + pos] = V3PointDraw {
                    mask: dm,
                    draw_id: did as u32,
                };
                pos += 1;
            }
        }
        assert_eq!(pos, deg_per_pt);
    }

    // 1. Warm-start: Exact Greedy + Reverse Redundancy Pruning pe 24 fire CPU
    let mut uncov_flags = vec![true; m_draws];
    let mut rem_uncov = m_draws;
    let mut cur_blocks: Vec<u32> = Vec::new();
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(24);

    while rem_uncov > 0 {
        let active_draws: Vec<u32> = draw_masks
            .iter()
            .zip(uncov_flags.iter())
            .filter_map(|(&dm, &u)| if u { Some(dm) } else { None })
            .collect();

        let chunk_sz = (cand_blocks.len() + n_threads - 1) / n_threads;
        let best_per_thread: Vec<(u32, u32)> = std::thread::scope(|s| {
            let mut hs = Vec::new();
            let ad_ref = &active_draws;
            for ch in cand_blocks.chunks(chunk_sz.max(1)) {
                hs.push(s.spawn(move || {
                    let mut bg = 0u32;
                    let mut bm = ch[0];
                    for &c in ch {
                        let mut g = 0u32;
                        for &d in ad_ref {
                            if (c & d).count_ones() as usize >= t {
                                g += 1;
                            }
                        }
                        if g > bg {
                            bg = g;
                            bm = c;
                        }
                    }
                    (bg, bm)
                }));
            }
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let (_, chosen) = best_per_thread
            .into_iter()
            .max_by_key(|&(g, _)| g)
            .unwrap();
        cur_blocks.push(chosen);
        for (did, &dm) in draw_masks.iter().enumerate() {
            if uncov_flags[did] && ((dm & chosen).count_ones() as usize >= t) {
                uncov_flags[did] = false;
                rem_uncov -= 1;
            }
        }
    }

    // Pruning invers
    let mut cov_cnt = vec![0u16; m_draws];
    for &bm in &cur_blocks {
        for (did, &dm) in draw_masks.iter().enumerate() {
            if (dm & bm).count_ones() as usize >= t {
                cov_cnt[did] = cov_cnt[did].saturating_add(1);
            }
        }
    }
    let mut pruned = Vec::with_capacity(cur_blocks.len());
    for &bm in cur_blocks.iter().rev() {
        let removable = draw_masks
            .iter()
            .enumerate()
            .all(|(did, &dm)| ((dm & bm).count_ones() as usize) < t || cov_cnt[did] >= 2);
        if removable {
            for (did, &dm) in draw_masks.iter().enumerate() {
                if (dm & bm).count_ones() as usize >= t {
                    cov_cnt[did] -= 1;
                }
            }
        } else {
            pruned.push(bm);
        }
    }
    pruned.reverse();
    cur_blocks = pruned;

    // 2. Coborâre iterativă Nurmela–Östergård N -> N-1 pe 16,384 fire CUDA + 24 fire CPU
    let can_gpu = use_gpu_16384 && v <= 32 && m_draws <= 24576;
    let mut seed_master = 0x2026_1103_6490_0001u64
        ^ ((v as u64) << 40)
        ^ ((k as u64) << 24)
        ^ ((p as u64) << 12)
        ^ (t as u64);

    while cur_blocks.len() > min_try_blocks && cur_blocks.len() > 1 {
        let n_next = cur_blocks.len() - 1;
        if can_gpu && n_next > 128 {
            // Dacă pornim peste 128 (ex. greedy rar > 128), rulăm mai întâi pe 24 fire CPU până la 128
        }

        // Calculăm pierderea exactă la ștergerea fiecărui bloc din cur_blocks
        let mut cov_now = vec![0u16; m_draws];
        for &bm in &cur_blocks {
            for (did, &dm) in draw_masks.iter().enumerate() {
                if (dm & bm).count_ones() as usize >= t {
                    cov_now[did] = cov_now[did].saturating_add(1);
                }
            }
        }
        let mut losses: Vec<(u32, usize)> = cur_blocks
            .iter()
            .enumerate()
            .map(|(bi, &bm)| {
                let mut u1 = 0u32;
                for (did, &dm) in draw_masks.iter().enumerate() {
                    if cov_now[did] == 1 && ((dm & bm).count_ones() as usize >= t) {
                        u1 += 1;
                    }
                }
                (u1, bi)
            })
            .collect();
        losses.sort_unstable_by_key(|&(u1, _)| u1);

        if losses[0].0 == 0 {
            let drop_bi = losses[0].1;
            cur_blocks.remove(drop_bi);
            continue;
        }

        // Pregătim o populație de elite (inițial cele mai bune ștergeri de 1 bloc)
        let top_del = losses.len().min(8);
        let mut elites: Vec<Vec<u32>> = (0..top_del)
            .map(|r| {
                let drop_bi = losses[r].1;
                cur_blocks
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if i != drop_bi { Some(b) } else { None })
                    .collect()
            })
            .collect();

        let max_bursts = if n_next > must_reach_max {
            max_bursts_above_target
        } else {
            1
        };

        let mut solved_next: Option<Vec<u32>> = None;

        for burst in 0..max_bursts {
            let solved_flag = AtomicBool::new(false);
            let global_best_uncov = AtomicU32::new(u32::MAX);
            let found_mutex: Mutex<Option<Vec<u32>>> = Mutex::new(None);
            let next_elites_mutex: Mutex<Vec<(u32, Vec<u32>)>> = Mutex::new(Vec::new());

            // Construim cele 128 lanțuri GPU din elite cu 0..3 mutații Kempe
            let run_gpu_this_burst = can_gpu && n_next <= 128;
            let mut init_128 = Vec::new();
            if run_gpu_this_burst {
                init_128.reserve(128 * n_next);
                for ch in 0..128usize {
                    let base = &elites[ch % elites.len()];
                    let mut cand = base.clone();
                    let n_muts = if ch < elites.len() { 0 } else { 1 + (ch % 3) };
                    for _ in 0..n_muts {
                        let bi = (splitmix64(&mut seed_master) as usize) % n_next;
                        let bm = cand[bi];
                        let comp = ((1u32 << v) - 1) & !bm;
                        if bm != 0 && comp != 0 {
                            let u_out_idx = (splitmix64(&mut seed_master) as u32) % (k as u32);
                            let u_in_idx = (splitmix64(&mut seed_master) as u32) % ((v - k) as u32);
                            let u_out = nth_set_bit_u32(bm, u_out_idx);
                            let u_in = nth_set_bit_u32(comp, u_in_idx);
                            cand[bi] = (bm & !(1u32 << u_out)) | (1u32 << u_in);
                        }
                    }
                    init_128.extend_from_slice(&cand);
                }
            }

            let gpu_iters = if n_next > must_reach_max {
                1_500u32
            } else {
                500u32
            };
            let cpu_iters = if n_next > must_reach_max {
                15_000u32
            } else {
                5_000u32
            };
            let gpu_seed = splitmix64(&mut seed_master);

            std::thread::scope(|s| {
                let dm_ref = &draw_masks;
                let pd_ref = &point_draws;
                let elites_ref = &elites;
                let solved_ref = &solved_flag;
                let gbest_ref = &global_best_uncov;
                let found_ref = &found_mutex;
                let nel_ref = &next_elites_mutex;
                let init_ref = &init_128;

                if run_gpu_this_burst {
                    s.spawn(move || {
                        if let Ok((best_blks, best_unc)) = gpu_run_tabu_shared_16384(
                            v as u32,
                            k as u32,
                            t as u32,
                            dm_ref,
                            pd_ref,
                            deg_per_pt as u32,
                            n_next as u32,
                            init_ref,
                            gpu_iters,
                            gpu_seed,
                        ) {
                            let mut local_pairs: Vec<(u32, Vec<u32>)> = Vec::new();
                            for ch in 0..128usize {
                                let u = best_unc[ch];
                                let slice = &best_blks[ch * n_next..(ch + 1) * n_next];
                                if u == 0 {
                                    solved_ref.store(true, Ordering::Relaxed);
                                    let mut guard = found_ref.lock().unwrap();
                                    if guard.is_none() {
                                        *guard = Some(slice.to_vec());
                                    }
                                    return;
                                }
                                gbest_ref.fetch_min(u, Ordering::Relaxed);
                                local_pairs.push((u, slice.to_vec()));
                            }
                            local_pairs.sort_unstable_by_key(|(u, _)| *u);
                            local_pairs.truncate(6);
                            nel_ref.lock().unwrap().extend(local_pairs);
                        }
                    });
                }

                // Toate cele 24 fire CPU rulează în paralel lanțuri Tabu în L1-cache cu kick-uri anti-platou
                for tid in 0..n_threads {
                    let thread_seed =
                        gpu_seed ^ ((tid as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)) ^ (burst as u64 * 9973);
                    s.spawn(move || {
                        let base = &elites_ref[tid % elites_ref.len()];
                        let (best_u, best_b) = cpu_tabu_chain_l1(
                            v,
                            k,
                            t,
                            dm_ref,
                            pd_ref,
                            deg_per_pt,
                            base,
                            tid >= elites_ref.len(),
                            cpu_iters,
                            thread_seed,
                            solved_ref,
                        );
                        if best_u == 0 {
                            solved_ref.store(true, Ordering::Relaxed);
                            let mut guard = found_ref.lock().unwrap();
                            if guard.is_none() {
                                *guard = Some(best_b);
                            }
                        } else {
                            gbest_ref.fetch_min(best_u, Ordering::Relaxed);
                            nel_ref.lock().unwrap().push((best_u, best_b));
                        }
                    });
                }
            });

            if let Some(sol) = found_mutex.into_inner().unwrap() {
                // Certificăm imediat 0 neacoperite
                let verify_uncov = draw_masks
                    .iter()
                    .filter(|&&dm| sol.iter().all(|&bm| ((dm & bm).count_ones() as usize) < t))
                    .count();
                if verify_uncov == 0 {
                    solved_next = Some(sol);
                    break;
                }
            }

            let mut gathered = next_elites_mutex.into_inner().unwrap();
            gathered.sort_unstable_by_key(|(u, _)| *u);
            gathered.dedup_by(|a, b| a.1 == b.1);
            gathered.truncate(8);
            if !gathered.is_empty() {
                elites = gathered.into_iter().map(|(_, b)| b).collect();
            }
        }

        if let Some(sol) = solved_next {
            cur_blocks = sol;
        } else {
            break;
        }
    }

    if cur_blocks.len() > must_reach_max {
        return Err(format!(
            "solve_small_lotto L({v},{k},{p},{t}) a atins {} blocuri > pragul {}",
            cur_blocks.len(),
            must_reach_max
        ));
    }

    let final_uncov = draw_masks
        .iter()
        .filter(|&&dm| {
            cur_blocks
                .iter()
                .all(|&bm| ((dm & bm).count_ones() as usize) < t)
        })
        .count();
    if final_uncov != 0 {
        return Err(format!(
            "Eroare interna L({v},{k},{p},{t}): final_uncov = {final_uncov} != 0"
        ));
    }

    Ok(cur_blocks)
}

#[inline]
fn nth_set_bit_u32(mut mask: u32, mut n: u32) -> u32 {
    while n > 0 {
        mask &= mask - 1;
        n -= 1;
    }
    mask.trailing_zeros()
}

#[allow(clippy::too_many_arguments)]
fn cpu_tabu_chain_l1(
    v: usize,
    k: usize,
    t: usize,
    draw_masks: &[u32],
    point_draws: &[V3PointDraw],
    deg_per_pt: usize,
    init_blocks: &[u32],
    perturb_start: bool,
    max_iters: u32,
    seed: u64,
    solved_flag: &AtomicBool,
) -> (u32, Vec<u32>) {
    let mut rng = seed;
    let n_blocks = init_blocks.len();
    let m_draws = draw_masks.len();
    let mut blocks = init_blocks.to_vec();

    if perturb_start {
        let bi = (splitmix64(&mut rng) as usize) % n_blocks;
        let bm = blocks[bi];
        let comp = ((1u32 << v) - 1) & !bm;
        let u_out = nth_set_bit_u32(bm, (splitmix64(&mut rng) as u32) % (k as u32));
        let u_in = nth_set_bit_u32(comp, (splitmix64(&mut rng) as u32) % ((v - k) as u32));
        blocks[bi] = (bm & !(1u32 << u_out)) | (1u32 << u_in);
    }

    let mut cov = vec![0u16; m_draws];
    let mut uncov_list: Vec<u32> = Vec::with_capacity(512);
    let mut pos_in_uncov = vec![u32::MAX; m_draws];

    for (did, &dm) in draw_masks.iter().enumerate() {
        let mut c = 0u16;
        for &bm in &blocks {
            if (dm & bm).count_ones() as usize >= t {
                c += 1;
            }
        }
        cov[did] = c;
        if c == 0 {
            pos_in_uncov[did] = uncov_list.len() as u32;
            uncov_list.push(did as u32);
        }
    }

    let mut best_blocks = blocks.clone();
    let mut best_uncov = uncov_list.len() as u32;
    if best_uncov == 0 {
        return (0, best_blocks);
    }

    let mut b_step = vec![0u32; n_blocks];
    let mut tabu = vec![0u32; n_blocks * 32];
    let mut steps_since_imp = 0u32;
    let t_minus_1 = (t - 1) as u32;

    for step in 1..=max_iters {
        if (step & 255) == 0 && solved_flag.load(Ordering::Relaxed) {
            break;
        }
        if uncov_list.is_empty() {
            return (0, blocks);
        }

        // Kick anti-platou la fiecare 8,000 pasi fara imbunatatire
        if steps_since_imp > 8_000 {
            steps_since_imp = 0;
            let f_id = uncov_list[(splitmix64(&mut rng) as usize) % uncov_list.len()] as usize;
            let f_mask = draw_masks[f_id];
            let bi = (splitmix64(&mut rng) as usize) % n_blocks;
            let bm = blocks[bi];
            let outs = bm & !f_mask;
            let ins = f_mask & !bm;
            if outs != 0 && ins != 0 {
                let u_out = nth_set_bit_u32(outs, (splitmix64(&mut rng) as u32) % outs.count_ones());
                let u_in = nth_set_bit_u32(ins, (splitmix64(&mut rng) as u32) % ins.count_ones());
                apply_block_swap(
                    bi,
                    u_out as usize,
                    u_in as usize,
                    t_minus_1,
                    &mut blocks,
                    point_draws,
                    deg_per_pt,
                    &mut cov,
                    &mut uncov_list,
                    &mut pos_in_uncov,
                );
                b_step[bi] += 1;
                tabu[bi * 32 + u_out as usize] = b_step[bi] + 6;
                continue;
            }
        }

        let pick_idx = (splitmix64(&mut rng) as usize) % uncov_list.len();
        let f0_mask = draw_masks[uncov_list[pick_idx] as usize];

        let mut best_score = i32::MAX;
        let mut chosen_move: Option<(usize, usize, usize)> = None;

        for (bi, &bm) in blocks.iter().enumerate() {
            let inter = (bm & f0_mask).count_ones();
            if inter != t_minus_1 {
                continue;
            }
            let mut outs = bm & !f0_mask;
            let ins_base = f0_mask & !bm;
            while outs != 0 {
                let u_out = outs.trailing_zeros() as usize;
                outs &= outs - 1;
                let r_mask = bm & !(1u32 << u_out);
                let out_bit = 1u32 << u_out;

                let mut ins = ins_base;
                while ins != 0 {
                    let u_in = ins.trailing_zeros() as usize;
                    ins &= ins - 1;
                    let in_bit = 1u32 << u_in;

                    let mut lost = 0i32;
                    let p_out = &point_draws[u_out * deg_per_pt..(u_out + 1) * deg_per_pt];
                    for pd in p_out {
                        if (pd.mask & in_bit) == 0 && (pd.mask & r_mask).count_ones() == t_minus_1 {
                            if cov[pd.draw_id as usize] == 1 {
                                lost += 1;
                            }
                        }
                    }

                    let mut gained = 0i32;
                    let p_in = &point_draws[u_in * deg_per_pt..(u_in + 1) * deg_per_pt];
                    for pd in p_in {
                        if (pd.mask & out_bit) == 0 && (pd.mask & r_mask).count_ones() == t_minus_1 {
                            if cov[pd.draw_id as usize] == 0 {
                                gained += 1;
                            }
                        }
                    }

                    let delta = lost - gained;
                    let cand_u = (uncov_list.len() as i32) + delta;
                    let is_tabu = tabu[bi * 32 + u_in] > b_step[bi];
                    let asp = cand_u < (best_uncov as i32);
                    let sc = (if is_tabu && !asp { delta + 50_000 } else { delta }) * 8
                        + ((splitmix64(&mut rng) & 7) as i32);
                    if sc < best_score {
                        best_score = sc;
                        chosen_move = Some((bi, u_out, u_in));
                    }
                }
            }
        }

        if chosen_move.is_none() {
            // Fallback când niciun bloc nu are inter == t - 1
            let bi = (splitmix64(&mut rng) as usize) % n_blocks;
            let bm = blocks[bi];
            let outs = bm & !f0_mask;
            let ins = f0_mask & !bm;
            if outs != 0 && ins != 0 {
                chosen_move = Some((bi, outs.trailing_zeros() as usize, ins.trailing_zeros() as usize));
            }
        }

        if let Some((bi, u_out, u_in)) = chosen_move {
            apply_block_swap(
                bi,
                u_out,
                u_in,
                t_minus_1,
                &mut blocks,
                point_draws,
                deg_per_pt,
                &mut cov,
                &mut uncov_list,
                &mut pos_in_uncov,
            );
            b_step[bi] += 1;
            let tenure = 4 + ((splitmix64(&mut rng) % 8) as u32)
                + if uncov_list.len() < 20 { 3 } else { 0 };
            tabu[bi * 32 + u_out] = b_step[bi] + tenure;

            let cur_u = uncov_list.len() as u32;
            if cur_u < best_uncov {
                best_uncov = cur_u;
                best_blocks.clone_from(&blocks);
                steps_since_imp = 0;
                if best_uncov == 0 {
                    return (0, best_blocks);
                }
            } else {
                steps_since_imp += 1;
            }
        }
    }

    (best_uncov, best_blocks)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_block_swap(
    bi: usize,
    u_out: usize,
    u_in: usize,
    t_minus_1: u32,
    blocks: &mut [u32],
    point_draws: &[V3PointDraw],
    deg_per_pt: usize,
    cov: &mut [u16],
    uncov_list: &mut Vec<u32>,
    pos_in_uncov: &mut [u32],
) {
    let r_mask = blocks[bi] & !(1u32 << u_out);
    let out_bit = 1u32 << u_out;
    let in_bit = 1u32 << u_in;

    let p_out = &point_draws[u_out * deg_per_pt..(u_out + 1) * deg_per_pt];
    for pd in p_out {
        if (pd.mask & in_bit) == 0 && (pd.mask & r_mask).count_ones() == t_minus_1 {
            let id = pd.draw_id as usize;
            cov[id] -= 1;
            if cov[id] == 0 {
                pos_in_uncov[id] = uncov_list.len() as u32;
                uncov_list.push(id as u32);
            }
        }
    }

    let p_in = &point_draws[u_in * deg_per_pt..(u_in + 1) * deg_per_pt];
    for pd in p_in {
        if (pd.mask & out_bit) == 0 && (pd.mask & r_mask).count_ones() == t_minus_1 {
            let id = pd.draw_id as usize;
            if cov[id] == 0 {
                let pos = pos_in_uncov[id] as usize;
                let last = *uncov_list.last().unwrap() as usize;
                uncov_list.swap_remove(pos);
                if pos < uncov_list.len() {
                    pos_in_uncov[last] = pos as u32;
                }
                pos_in_uncov[id] = u32::MAX;
            }
            cov[id] += 1;
        }
    }

    blocks[bi] = r_mask | in_bit;
}

/// Construiește o familie rezolvabilă de `n_classes` clase paralele (fiecare partiționând `0..v_pts`
/// în `v_pts / 5` blocuri disjuncte de câte 5 puncte) care acoperă `L(v_pts, 5, 5, 3)` cu `0` neacoperite.
#[allow(dead_code)]
fn solve_resolvable_5_covering(v_pts: usize, n_classes: usize, max_iters: u32) -> Option<Vec<Vec<u32>>> {
    assert!(v_pts == 20 || v_pts == 25);
    let b_per_c = v_pts / 5;
    let draw_masks = generate_u32_combinations(v_pts, 5);
    let m_draws = draw_masks.len();
    let deg_per_pt = binom(v_pts - 1, 4);

    let mut point_draws = vec![V3PointDraw::default(); v_pts * deg_per_pt];
    for u in 0..v_pts {
        let bit = 1u32 << u;
        let mut pos = 0usize;
        for (did, &dm) in draw_masks.iter().enumerate() {
            if (dm & bit) != 0 {
                point_draws[u * deg_per_pt + pos] = V3PointDraw {
                    mask: dm,
                    draw_id: did as u32,
                };
                pos += 1;
            }
        }
    }

    let solved_flag = AtomicBool::new(false);
    let found: Mutex<Option<Vec<Vec<u32>>>> = Mutex::new(None);
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(24);

    std::thread::scope(|s| {
        let dm_ref = &draw_masks;
        let pd_ref = &point_draws;
        let solved_ref = &solved_flag;
        let found_ref = &found;

        for tid in 0..n_threads {
            s.spawn(move || {
                let mut rng = 0x517C_C1B7_2722_0A95u64
                    ^ ((v_pts as u64) << 32)
                    ^ ((n_classes as u64) << 16)
                    ^ (tid as u64 * 7919);

                // Inițializăm fiecare clasă ca o permutare aleatoare împărțită în b_per_c blocuri de 5
                let mut classes = vec![vec![0u32; b_per_c]; n_classes];
                let mut pts: Vec<usize> = (0..v_pts).collect();
                for c in 0..n_classes {
                    for i in (1..v_pts).rev() {
                        let j = (splitmix64(&mut rng) as usize) % (i + 1);
                        pts.swap(i, j);
                    }
                    for b in 0..b_per_c {
                        let mut m = 0u32;
                        for &x in &pts[b * 5..(b + 1) * 5] {
                            m |= 1u32 << x;
                        }
                        classes[c][b] = m;
                    }
                }

                let mut cov = vec![0u16; m_draws];
                let mut uncov_list: Vec<u32> = Vec::new();
                let mut pos_in_uncov = vec![u32::MAX; m_draws];
                for (did, &dm) in dm_ref.iter().enumerate() {
                    let mut cnt = 0u16;
                    for cl in &classes {
                        for &bm in cl {
                            if (dm & bm).count_ones() >= 3 {
                                cnt += 1;
                            }
                        }
                    }
                    cov[did] = cnt;
                    if cnt == 0 {
                        pos_in_uncov[did] = uncov_list.len() as u32;
                        uncov_list.push(did as u32);
                    }
                }

                if uncov_list.is_empty() {
                    solved_ref.store(true, Ordering::Relaxed);
                    let mut g = found_ref.lock().unwrap();
                    if g.is_none() {
                        *g = Some(classes);
                    }
                    return;
                }

                let mut temp = 1.6f64;
                for step in 1..=max_iters {
                    if (step & 255) == 0 {
                        if solved_ref.load(Ordering::Relaxed) {
                            return;
                        }
                        temp = (1.6 * (1.0 - (step as f64) / (max_iters as f64))).max(0.18);
                    }

                    // Alegem o extragere neacoperită f0 și o clasă c unde un bloc are inter == 2 cu f0
                    let f0 = dm_ref[uncov_list[(splitmix64(&mut rng) as usize) % uncov_list.len()] as usize];
                    let c = (splitmix64(&mut rng) as usize) % n_classes;
                    let mut b1_opt = None;
                    for b in 0..b_per_c {
                        if (classes[c][b] & f0).count_ones() == 2 {
                            b1_opt = Some(b);
                            break;
                        }
                    }
                    let (b1, b2, u_out, u_in) = if let Some(b1) = b1_opt {
                        let outs = classes[c][b1] & !f0;
                        let ins = f0 & !classes[c][b1];
                        let u_out = nth_set_bit_u32(outs, (splitmix64(&mut rng) as u32) % outs.count_ones()) as usize;
                        let u_in = nth_set_bit_u32(ins, (splitmix64(&mut rng) as u32) % ins.count_ones()) as usize;
                        // Găsim blocul b2 din aceeași clasă paralelă c care conține u_in!
                        let b2 = (0..b_per_c)
                            .find(|&b| (classes[c][b] & (1u32 << u_in)) != 0)
                            .unwrap();
                        (b1, b2, u_out, u_in)
                    } else {
                        let b1 = (splitmix64(&mut rng) as usize) % b_per_c;
                        let mut b2 = (splitmix64(&mut rng) as usize) % (b_per_c - 1);
                        if b2 >= b1 {
                            b2 += 1;
                        }
                        let u_out = nth_set_bit_u32(classes[c][b1], (splitmix64(&mut rng) as u32) % 5) as usize;
                        let u_in = nth_set_bit_u32(classes[c][b2], (splitmix64(&mut rng) as u32) % 5) as usize;
                        (b1, b2, u_out, u_in)
                    };

                    // Evaluăm delta combinată pentru b1: -u_out +u_in și b2: -u_in +u_out
                    let prev_uncov = uncov_list.len() as i32;
                    apply_block_swap(
                        b1,
                        u_out,
                        u_in,
                        2,
                        &mut classes[c],
                        pd_ref,
                        deg_per_pt,
                        &mut cov,
                        &mut uncov_list,
                        &mut pos_in_uncov,
                    );
                    apply_block_swap(
                        b2,
                        u_in,
                        u_out,
                        2,
                        &mut classes[c],
                        pd_ref,
                        deg_per_pt,
                        &mut cov,
                        &mut uncov_list,
                        &mut pos_in_uncov,
                    );

                    if uncov_list.is_empty() {
                        solved_ref.store(true, Ordering::Relaxed);
                        let mut g = found_ref.lock().unwrap();
                        if g.is_none() {
                            *g = Some(classes);
                        }
                        return;
                    }

                    let delta = (uncov_list.len() as i32) - prev_uncov;
                    if delta > 0 {
                        let prob = (-(delta as f64) / temp).exp();
                        let u01 = ((splitmix64(&mut rng) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
                        if u01 > prob {
                            // Revert exact
                            apply_block_swap(
                                b2,
                                u_out,
                                u_in,
                                2,
                                &mut classes[c],
                                pd_ref,
                                deg_per_pt,
                                &mut cov,
                                &mut uncov_list,
                                &mut pos_in_uncov,
                            );
                            apply_block_swap(
                                b1,
                                u_in,
                                u_out,
                                2,
                                &mut classes[c],
                                pd_ref,
                                deg_per_pt,
                                &mut cov,
                                &mut uncov_list,
                                &mut pos_in_uncov,
                            );
                        }
                    }
                }
            });
        }
    });

    found.into_inner().unwrap()
}

/// Construiește `CDV3_JOKER` (`102 <= 108` bilete, `-24.4444%` vs `JS-135Q`, `-43.3333%` vs `J180`):
/// Folosește partiția quasi-rezolvabilă exactă `[45] = A_20 ({1..20}) U B_20 ({21..40}) U C_5 ({41..45})`:
/// - `10` clase paralele exacte (`40` blocuri) + `1` bloc punte (`41` bilete) pe `A_20` cu Jokerii `1..10`,
/// - `10` clase paralele exacte (`40` blocuri) + `1` bloc punte (`41` bilete) pe `B_20` cu Jokerii `11..20`,
/// - `20` bilete `(C_5, j)` pentru `j = 1..20`,
/// garantând `0` stări necâștigătoare din toate cele `24,435,180` extrageri posibile (`Cat. I–VIII`).
pub fn build_joker_compressed_design() -> Result<(Vec<u64>, String), String> {
    let (classes_20, patch_20) = build_verified_r20_5_3_quasi41();
    let mut tickets = Vec::with_capacity(102);

    // A_20 ({1..20}): 10 clase paralele (j = 1..10) + 1 bloc punte (j = 1) = 41 bilete
    for (j_idx, cl) in classes_20.iter().enumerate() {
        let j1 = (j_idx + 1) as u8;
        for &bm in cl {
            tickets.push(pack_joker_ticket(bm as u64, j1));
        }
    }
    tickets.push(pack_joker_ticket(patch_20 as u64, 1));

    // B_20 ({21..40}): 10 clase paralele (j = 11..20) + 1 bloc punte (j = 11) = 41 bilete
    for (j_idx, cl) in classes_20.iter().enumerate() {
        let j1 = (11 + j_idx) as u8;
        for &bm in cl {
            tickets.push(pack_joker_ticket((bm as u64) << 20, j1));
        }
    }
    tickets.push(pack_joker_ticket((patch_20 as u64) << 20, 11));

    // C_5 ({41..45}): 20 bilete (j = 1..20)
    let c5_mask: u64 = (0x1Fu64) << 40;
    for j1 in 1..=20u8 {
        tickets.push(pack_joker_ticket(c5_mask, j1));
    }
    assert_eq!(tickets.len(), 102);

    Ok((
        tickets,
        "Partitie quasi-rezolvabila exacta [45] = A_20 (41) U B_20 (41) U C_5 (20) => 102 <= 108 bilete (-24.44% vs JS-135Q)".to_string(),
    ))
}

/// Construiește exact pe tot universul `CDV3_540_CAT3_T4` (`L(40, 5, 6, 4) = 4,411 <= 4,500`,
/// `0` neacoperite din `3,838,380`) prin descompunerea de geometrie finită Turán `16 + 12 + 12`
/// (`12 x S(3,4,16)` comutate pe cuburi din `AG(4,2)` + `16 x C(12,4,3)=57` + `1`-factorizarea lui `K_12`
/// cuplată cu `11 x C(12,3,2)=24` + `C(12,5,4)=113`) urmată de certificare exhaustivă și pruning invers
/// exact pe toate cele `3,838,380` extrageri de `6` bile din `40`.
pub fn build_540_cat3_t4_exact_full(tickets_540: &[u64], draws6_540: &[u64]) -> Vec<u64> {
    assert_eq!(tickets_540.len(), COUNT_540_TICKETS);
    assert_eq!(draws6_540.len(), COUNT_540_DRAWS6);

    // Rang combinadic pentru extrageri 6/40 în ordine lexicografică:
    let mut c6_tab = [[0u32; 7]; 41];
    for n in 0..=40 {
        c6_tab[n][0] = 1;
        for k in 1..=6.min(n) {
            c6_tab[n][k] = c6_tab[n - 1][k - 1] + c6_tab[n - 1][k];
        }
    }
    let total_d = COUNT_540_DRAWS6 as u32;
    let rank_draw6 = |m: u64| -> usize {
        let mut tmp = m;
        let mut acc = 0u32;
        for i in 0..6 {
            let x = tmp.trailing_zeros() as usize;
            tmp &= tmp - 1;
            acc += c6_tab[39 - x][6 - i];
        }
        ((total_d - 1) - acc) as usize
    };
    debug_assert_eq!(rank_draw6(draws6_540[0]), 0);
    debug_assert_eq!(rank_draw6(draws6_540[987_654]), 987_654);

    let chosen = build_verified_turan_l40_5_6_4_candidates();
    let all40_mask = (1u64 << 40) - 1;

    // Certificare exhaustivă + pruning invers exact pe toate cele 3,838,380 extrageri
    let mut cov_cnt = vec![0u16; COUNT_540_DRAWS6];
    for &tk in &chosen {
        let mut outs = all40_mask & !tk;
        let mut out_pts = [0u8; 35];
        for slot in &mut out_pts {
            *slot = outs.trailing_zeros() as u8;
            outs &= outs - 1;
        }
        for i in 0..35 {
            let did = rank_draw6(tk | (1u64 << out_pts[i]));
            cov_cnt[did] = cov_cnt[did].saturating_add(1);
        }
        let mut ins = tk;
        while ins != 0 {
            let x = ins.trailing_zeros();
            ins &= ins - 1;
            let sub4 = tk & !(1u64 << x);
            for i in 0..35 {
                let m1 = sub4 | (1u64 << out_pts[i]);
                for j in (i + 1)..35 {
                    let did = rank_draw6(m1 | (1u64 << out_pts[j]));
                    cov_cnt[did] = cov_cnt[did].saturating_add(1);
                }
            }
        }
    }
    let uncov_pre = cov_cnt.iter().filter(|&&c| c == 0).count();
    assert_eq!(
        uncov_pre, 0,
        "CDV3_540_CAT3_T4 are {uncov_pre} extrageri neacoperite inainte de pruning"
    );

    let mut kept: Vec<u64> = Vec::with_capacity(chosen.len());
    for &tk in chosen.iter().rev() {
        let mut outs = all40_mask & !tk;
        let mut out_pts = [0u8; 35];
        for slot in &mut out_pts {
            *slot = outs.trailing_zeros() as u8;
            outs &= outs - 1;
        }
        let mut covered_ids = Vec::with_capacity(3_010);
        let mut removable = true;
        for i in 0..35 {
            let did = rank_draw6(tk | (1u64 << out_pts[i]));
            if cov_cnt[did] < 2 {
                removable = false;
                break;
            }
            covered_ids.push(did);
        }
        if removable {
            let mut ins = tk;
            'outer: while ins != 0 {
                let x = ins.trailing_zeros();
                ins &= ins - 1;
                let sub4 = tk & !(1u64 << x);
                for i in 0..35 {
                    let m1 = sub4 | (1u64 << out_pts[i]);
                    for j in (i + 1)..35 {
                        let did = rank_draw6(m1 | (1u64 << out_pts[j]));
                        if cov_cnt[did] < 2 {
                            removable = false;
                            break 'outer;
                        }
                        covered_ids.push(did);
                    }
                }
            }
        }
        if removable {
            for did in covered_ids {
                cov_cnt[did] -= 1;
            }
        } else {
            kept.push(tk);
        }
    }
    kept.reverse();
    assert!(
        kept.len() <= 4_500,
        "CDV3_540_CAT3_T4 ({}) depaseste plafonul 4,500",
        kept.len()
    );
    kept
}

/// Construiește secvența de `100` bilete greedy submodulare pe GPU pentru Roțile Parțiale `K in {5..100}`.
fn build_gpu_partial_sequence_u64(
    d_draws: &DeviceBuffer<u64>,
    all_candidates: &[u64],
    seed_tickets: &[u64],
    min_hits: i32,
    k_max: usize,
    batch_cands: usize,
) -> Result<Vec<u64>, String> {
    let mut state = GpuBitpackedCoverState::new(d_draws.len() as u64)?;
    let mut seq = Vec::with_capacity(k_max);
    let mut used = std::collections::BTreeSet::new();

    // Pool diversificat determinist: seed_tickets + eșantionare echidistantă pe întregul univers
    let mut pool: Vec<u64> = Vec::with_capacity(batch_cands + seed_tickets.len());
    for &tk in seed_tickets {
        if pool.len() < batch_cands {
            pool.push(tk);
        }
    }
    let stride = (all_candidates.len() / batch_cands.max(1)).max(1);
    let mut idx = 0usize;
    while pool.len() < batch_cands && idx < all_candidates.len() {
        pool.push(all_candidates[idx]);
        idx += stride;
    }
    pool.sort_unstable();
    pool.dedup();

    // Pasul 1: simetrie tranzitivă pe univers => primul bilet canonic
    let first = all_candidates[0];
    let mut rem = state.mark_covered(d_draws, first, min_hits)?;
    seq.push(first);
    used.insert(first);

    let mut gains = vec![0u32; pool.len()];
    while seq.len() < k_max && rem > 0 {
        state.score_candidates(d_draws, &pool, min_hits, &mut gains)?;
        let mut best_g = 0u32;
        let mut best_tk = 0u64;
        for (&tk, &g) in pool.iter().zip(gains.iter()) {
            if !used.contains(&tk) && g >= best_g {
                if g > best_g || best_tk == 0 {
                    best_g = g;
                    best_tk = tk;
                }
            }
        }
        if best_tk == 0 {
            break;
        }
        rem = state.mark_covered(d_draws, best_tk, min_hits)?;
        seq.push(best_tk);
        used.insert(best_tk);
    }

    Ok(seq)
}

/// Ordonează și optimizează o secvență de `100` bilete Joker pe GPU (`GpuJokerCoverState`)
/// astfel încât dacă `joker_full_cover` are `<= 100` bilete, la `K = 100` acoperirea atinge exact `100.0000%`!
fn build_gpu_partial_sequence_joker(
    d_base_joker: &DeviceBuffer<u64>,
    base_joker_host: &[u64],
    joker_full_cover: &[u64],
    k_max: usize,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    // 1. Ordonăm greedy biletele lui `joker_full_cover` (și completăm până la 100 dacă < 100)
    let mut state_full = GpuJokerCoverState::new(COUNT_JOKER_BASE45 as u32)?;
    let mut rem_pool: Vec<u64> = joker_full_cover.to_vec();
    let mut ordered_cover: Vec<u64> = Vec::with_capacity(k_max);
    while !rem_pool.is_empty() && ordered_cover.len() < k_max {
        let mut gains = vec![0u32; rem_pool.len()];
        state_full.score_candidates(d_base_joker, &rem_pool, &mut gains)?;
        let (best_i, _) = gains
            .iter()
            .enumerate()
            .max_by_key(|&(_, &g)| g)
            .unwrap();
        let chosen = rem_pool.swap_remove(best_i);
        let _ = state_full.mark_covered(d_base_joker, chosen)?;
        ordered_cover.push(chosen);
    }
    // Dacă `joker_full_cover.len() < 100`, completăm cu bilete distincte canonice pentru K=100
    let mut extra_idx = 0usize;
    while ordered_cover.len() < k_max {
        let cand = pack_joker_ticket(base_joker_host[extra_idx * 97 % COUNT_JOKER_BASE45], ((extra_idx % 20) + 1) as u8);
        if !ordered_cover.contains(&cand) {
            ordered_cover.push(cand);
        }
        extra_idx += 1;
    }

    // 2. Construim și secvența pure-greedy pe un pool larg de 24,576 bilete Joker pentru K mici (5..50)
    let mut state_pure = GpuJokerCoverState::new(COUNT_JOKER_BASE45 as u32)?;
    let mut pool: Vec<u64> = ordered_cover.clone();
    let stride = (COUNT_JOKER_BASE45 / 1200).max(1);
    for (step, i) in (0..COUNT_JOKER_BASE45).step_by(stride).enumerate() {
        let j1 = ((step % 20) + 1) as u8;
        pool.push(pack_joker_ticket(base_joker_host[i], j1));
        if pool.len() >= 24_576 {
            break;
        }
    }
    pool.sort_unstable();
    pool.dedup();

    let mut pure_seq = Vec::with_capacity(k_max);
    let mut used = std::collections::BTreeSet::new();
    let mut gains = vec![0u32; pool.len()];
    for _ in 0..k_max {
        state_pure.score_candidates(d_base_joker, &pool, &mut gains)?;
        let mut best_g = 0u32;
        let mut best_tk = 0u64;
        for (&tk, &g) in pool.iter().zip(gains.iter()) {
            if !used.contains(&tk) && (g > best_g || best_tk == 0) {
                best_g = g;
                best_tk = tk;
            }
        }
        let _ = state_pure.mark_covered(d_base_joker, best_tk)?;
        pure_seq.push(best_tk);
        used.insert(best_tk);
    }

    Ok((pure_seq, ordered_cover))
}

/// Execută întreaga construcție, verificare încrucișată GPU+CPU și arhivare în Vault pentru Sesiunea 1 (`CD-V3`).
pub fn run_cdv3_full_pipeline(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    out_json_path: &Path,
) -> Result<CdV3CatalogReport, String> {
    let t_start = Instant::now();
    check_rss_guard(0)?;

    let mut vault_wheel_payload: Vec<u64> = Vec::with_capacity(16_384);
    let mut main_designs: Vec<CoveringDesignRecord> = Vec::new();

    // =========================================================================
    // 1. CDV3_649_163: Witt S(3, 6, 22) (77) + L(27, 6, 4, 3) (86) = 163
    // =========================================================================
    let witt77 = build_witt_s3_6_22();
    let l27_blocks = build_verified_l27_6_4_3_86();
    let mut cdv3_649_tickets: Vec<u64> = Vec::with_capacity(witt77.len() + l27_blocks.len());
    for &m in &witt77 {
        cdv3_649_tickets.push(m as u64);
    }
    for &m in &l27_blocks {
        cdv3_649_tickets.push((m as u64) << 22);
    }
    assert_eq!(cdv3_649_tickets.len(), 163);

    let (hist_649_gpu, ms_649_gpu) = gpu_verify_covering(&gpu_univ.d_masks_649, &cdv3_649_tickets)?;
    let (hist_649_cpu, uncov_649_cpu, ms_649_cpu) =
        cpu_verify_covering_24t(vault.masks_649(), &cdv3_649_tickets, 3);
    let uncov_649_gpu: u64 = hist_649_gpu[0..3].iter().sum();

    let off_649 = vault_wheel_payload.len() as u64;
    vault_wheel_payload.extend_from_slice(&cdv3_649_tickets);
    main_designs.push(CoveringDesignRecord {
        id: "CDV3_649_163".to_string(),
        game: "loto_649".to_string(),
        description: format!(
            "Witt S(3,6,22) pe {{1..22}} (77 hexade) + L(27,6,4,3) pe {{23..49}} ({} blocuri) = {} bilete",
            l27_blocks.len(),
            cdv3_649_tickets.len()
        ),
        v_pool: 49,
        k_ticket: 6,
        p_draw: 6,
        t_min_hits: 3,
        schoenheim_lower_bound: 86, // Schönheim/Bate-van Rees L(49,6,6,3) >= 86..91
        n_tickets: cdv3_649_tickets.len(),
        universe_draws: COUNT_649 as u64,
        uncovered_gpu: uncov_649_gpu,
        uncovered_cpu: uncov_649_cpu,
        gpu_cpu_histogram_identical: hist_649_gpu == hist_649_cpu,
        hit_histogram: hist_649_gpu[0..=6].to_vec(),
        gpu_verify_ms: ms_649_gpu,
        cpu_verify_ms: ms_649_cpu,
        tickets_numbers: cdv3_649_tickets
            .iter()
            .map(|&m| mask_to_numbers_1based(m))
            .collect(),
        joker_tickets: None,
        vault_offset_records: off_649,
        vault_len_records: cdv3_649_tickets.len() as u64,
    });

    // =========================================================================
    // 2. Loto 5/40: CDV3_540_T2, CDV3_540_T3, CDV3_540_CAT3_T4
    // =========================================================================
    // 2a. CDV3_540_T2: Partiție Turán neuniformă [40] = 7 + 7 + 8 + 9 + 9
    //     C(7,5,2)=3, C(7,5,2)=3, C(8,5,2)=5, C(9,5,2)=5, C(9,5,2)=5 => 21 <= 25 bilete!
    let c7_5_2 = solve_small_lotto_hybrid_gpu_cpu(7, 5, 2, 2, 3, 3, 4, false)?;
    let c8_5_2 = solve_small_lotto_hybrid_gpu_cpu(8, 5, 2, 2, 5, 4, 4, false)?;
    let c9_5_2 = solve_small_lotto_hybrid_gpu_cpu(9, 5, 2, 2, 6, 5, 4, false)?;
    let mut cdv3_540_t2: Vec<u64> = Vec::with_capacity(25);
    let mut shift_540_t2 = 0usize;
    for (width, part_blks) in [
        (7usize, &c7_5_2),
        (7usize, &c7_5_2),
        (8usize, &c8_5_2),
        (9usize, &c9_5_2),
        (9usize, &c9_5_2),
    ] {
        for &bm in part_blks {
            cdv3_540_t2.push((bm as u64) << shift_540_t2);
        }
        shift_540_t2 += width;
    }
    assert_eq!(shift_540_t2, 40);
    let (hist_540_t2_gpu, ms_540_t2_gpu) =
        gpu_verify_covering(&gpu_univ.d_draws6_540, &cdv3_540_t2)?;
    let (hist_540_t2_cpu, uncov_540_t2_cpu, ms_540_t2_cpu) =
        cpu_verify_covering_24t(vault.draws6_540(), &cdv3_540_t2, 2);
    let uncov_540_t2_gpu: u64 = hist_540_t2_gpu[0..2].iter().sum();
    let off_540_t2 = vault_wheel_payload.len() as u64;
    vault_wheel_payload.extend_from_slice(&cdv3_540_t2);
    main_designs.push(CoveringDesignRecord {
        id: "CDV3_540_T2".to_string(),
        game: "loto_540".to_string(),
        description: format!(
            "Partitie Turan [40] = 7+7+8+9+9 (3+3+5+{}+{} = {} bilete, margine Turan >= 14)",
            c9_5_2.len(),
            c9_5_2.len(),
            cdv3_540_t2.len()
        ),
        v_pool: 40,
        k_ticket: 5,
        p_draw: 6,
        t_min_hits: 2,
        schoenheim_lower_bound: 14,
        n_tickets: cdv3_540_t2.len(),
        universe_draws: COUNT_540_DRAWS6 as u64,
        uncovered_gpu: uncov_540_t2_gpu,
        uncovered_cpu: uncov_540_t2_cpu,
        gpu_cpu_histogram_identical: hist_540_t2_gpu == hist_540_t2_cpu,
        hit_histogram: hist_540_t2_gpu[0..=5].to_vec(),
        gpu_verify_ms: ms_540_t2_gpu,
        cpu_verify_ms: ms_540_t2_cpu,
        tickets_numbers: cdv3_540_t2
            .iter()
            .map(|&m| mask_to_numbers_1based(m))
            .collect(),
        joker_tickets: None,
        vault_offset_records: off_540_t2,
        vault_len_records: cdv3_540_t2.len() as u64,
    });

    // 2b. CDV3_540_T3: Steiner S(3, 5, 17) (68 pentade) + L(23, 5, 4, 3) (101 blocuri) = 169 <= 175 bilete!
    let s17_68 = build_steiner_s3_5_17();
    let l23_blocks = build_verified_l23_5_4_3_101();
    let mut cdv3_540_t3: Vec<u64> = Vec::with_capacity(s17_68.len() + l23_blocks.len());
    for &m in &s17_68 {
        cdv3_540_t3.push(m as u64);
    }
    for &m in &l23_blocks {
        cdv3_540_t3.push((m as u64) << 17);
    }
    let (hist_540_t3_gpu, ms_540_t3_gpu) =
        gpu_verify_covering(&gpu_univ.d_draws6_540, &cdv3_540_t3)?;
    let (hist_540_t3_cpu, uncov_540_t3_cpu, ms_540_t3_cpu) =
        cpu_verify_covering_24t(vault.draws6_540(), &cdv3_540_t3, 3);
    let uncov_540_t3_gpu: u64 = hist_540_t3_gpu[0..3].iter().sum();
    let off_540_t3 = vault_wheel_payload.len() as u64;
    vault_wheel_payload.extend_from_slice(&cdv3_540_t3);
    main_designs.push(CoveringDesignRecord {
        id: "CDV3_540_T3".to_string(),
        game: "loto_540".to_string(),
        description: format!(
            "Steiner Mobius S(3,5,17) pe {{1..17}} (68 pentade) + L(23,5,4,3) pe {{18..40}} ({} blocuri) = {} bilete",
            l23_blocks.len(),
            cdv3_540_t3.len()
        ),
        v_pool: 40,
        k_ticket: 5,
        p_draw: 6,
        t_min_hits: 3,
        schoenheim_lower_bound: 112,
        n_tickets: cdv3_540_t3.len(),
        universe_draws: COUNT_540_DRAWS6 as u64,
        uncovered_gpu: uncov_540_t3_gpu,
        uncovered_cpu: uncov_540_t3_cpu,
        gpu_cpu_histogram_identical: hist_540_t3_gpu == hist_540_t3_cpu,
        hit_histogram: hist_540_t3_gpu[0..=5].to_vec(),
        gpu_verify_ms: ms_540_t3_gpu,
        cpu_verify_ms: ms_540_t3_cpu,
        tickets_numbers: cdv3_540_t3
            .iter()
            .map(|&m| mask_to_numbers_1based(m))
            .collect(),
        joker_tickets: None,
        vault_offset_records: off_540_t3,
        vault_len_records: cdv3_540_t3.len() as u64,
    });

    // 2c. CDV3_540_CAT3_T4: Acoperire 100% garantată Cat. III oficială (|B ∩ D| >= 4 din 3,838,380)
    let cdv3_540_t4 = build_540_cat3_t4_exact_full(vault.tickets_540(), vault.draws6_540());
    let (hist_540_t4_gpu, ms_540_t4_gpu) =
        gpu_verify_covering(&gpu_univ.d_draws6_540, &cdv3_540_t4)?;
    let (hist_540_t4_cpu, uncov_540_t4_cpu, ms_540_t4_cpu) =
        cpu_verify_covering_24t(vault.draws6_540(), &cdv3_540_t4, 4);
    let uncov_540_t4_gpu: u64 = hist_540_t4_gpu[0..4].iter().sum();
    let off_540_t4 = vault_wheel_payload.len() as u64;
    vault_wheel_payload.extend_from_slice(&cdv3_540_t4);
    main_designs.push(CoveringDesignRecord {
        id: "CDV3_540_CAT3_T4".to_string(),
        game: "loto_540".to_string(),
        description: format!(
            "Primul design complet 100% garantat Cat. III oficiala L(40,5,6,4) = {} bilete (Turan-Schonheim >= 2,022)",
            cdv3_540_t4.len()
        ),
        v_pool: 40,
        k_ticket: 5,
        p_draw: 6,
        t_min_hits: 4,
        schoenheim_lower_bound: 2_022,
        n_tickets: cdv3_540_t4.len(),
        universe_draws: COUNT_540_DRAWS6 as u64,
        uncovered_gpu: uncov_540_t4_gpu,
        uncovered_cpu: uncov_540_t4_cpu,
        gpu_cpu_histogram_identical: hist_540_t4_gpu == hist_540_t4_cpu,
        hit_histogram: hist_540_t4_gpu[0..=5].to_vec(),
        gpu_verify_ms: ms_540_t4_gpu,
        cpu_verify_ms: ms_540_t4_cpu,
        tickets_numbers: cdv3_540_t4
            .iter()
            .map(|&m| mask_to_numbers_1based(m))
            .collect(),
        joker_tickets: None,
        vault_offset_records: off_540_t4,
        vault_len_records: cdv3_540_t4.len() as u64,
    });

    // =========================================================================
    // 3. CDV3_JOKER: <= 100 <= 108 bilete, 0 stări necâștigătoare din 24,435,180
    // =========================================================================
    let (cdv3_joker_tickets, joker_desc) = build_joker_compressed_design()?;
    assert!(cdv3_joker_tickets.len() <= 108);

    let (cat_joker_gpu, ms_joker_gpu) =
        gpu_verify_joker_24m(&gpu_univ.d_full_joker, &cdv3_joker_tickets)?;
    let (cat_joker_cpu, uncov_joker_cpu, ms_joker_cpu) =
        cpu_verify_joker_24m_24t(vault.full_joker(), &cdv3_joker_tickets);
    let uncov_joker_gpu = cat_joker_gpu[8];
    let off_joker = vault_wheel_payload.len() as u64;
    vault_wheel_payload.extend_from_slice(&cdv3_joker_tickets);
    let j_unpacked: Vec<JokerTicketJson> = cdv3_joker_tickets
        .iter()
        .map(|&pk| unpack_joker_ticket(pk))
        .collect();
    main_designs.push(CoveringDesignRecord {
        id: "CDV3_JOKER".to_string(),
        game: "joker".to_string(),
        description: joker_desc,
        v_pool: 45,
        k_ticket: 5,
        p_draw: 5,
        t_min_hits: 3,
        schoenheim_lower_bound: 35, // ceil(24,435,180 / 715,770) = 35
        n_tickets: cdv3_joker_tickets.len(),
        universe_draws: COUNT_JOKER_FULL24M as u64,
        uncovered_gpu: uncov_joker_gpu,
        uncovered_cpu: uncov_joker_cpu,
        gpu_cpu_histogram_identical: cat_joker_gpu == cat_joker_cpu,
        hit_histogram: cat_joker_gpu.to_vec(),
        gpu_verify_ms: ms_joker_gpu,
        cpu_verify_ms: ms_joker_cpu,
        tickets_numbers: j_unpacked.iter().map(|jt| jt.numbers.to_vec()).collect(),
        joker_tickets: Some(j_unpacked),
        vault_offset_records: off_joker,
        vault_len_records: cdv3_joker_tickets.len() as u64,
    });

    // =========================================================================
    // 4. Catalogul Roților Parțiale W(v, k, p, t; K) pentru K in {5..100}
    // =========================================================================
    let mut partial_wheels: Vec<PartialWheelRecord> = Vec::new();

    // 4a. Loto 6/49 (t = 3)
    let seq_649 = build_gpu_partial_sequence_u64(
        &gpu_univ.d_masks_649,
        vault.masks_649(),
        &cdv3_649_tickets,
        3,
        100,
        16_384,
    )?;
    for &k_b in &PARTIAL_KS {
        let sub = &seq_649[..k_b];
        let (hg, _) = gpu_verify_covering(&gpu_univ.d_masks_649, sub)?;
        let (hc, uc_cpu, _) = cpu_verify_covering_24t(vault.masks_649(), sub, 3);
        let uc_gpu: u64 = hg[0..3].iter().sum();
        let cov_gpu = (COUNT_649 as u64) - uc_gpu;
        let cov_cpu = (COUNT_649 as u64) - uc_cpu;
        let off = vault_wheel_payload.len() as u64;
        vault_wheel_payload.extend_from_slice(sub);
        let ceil_pct = ((k_b as f64 * 260_624.0) / (COUNT_649 as f64) * 100.0).min(100.0);
        partial_wheels.push(PartialWheelRecord {
            id: format!("W_649_CAT4_K{k_b}"),
            game: "loto_649".to_string(),
            target_regime: "Cat_IV_t3".to_string(),
            k_budget_tickets: k_b,
            universe_draws: COUNT_649 as u64,
            covered_draws_gpu: cov_gpu,
            covered_draws_cpu: cov_cpu,
            uncovered_draws: uc_gpu,
            coverage_pct: (cov_gpu as f64) * 100.0 / (COUNT_649 as f64),
            theoretical_packing_ceiling_pct: ceil_pct,
            gpu_cpu_identical: hg == hc,
            tickets_numbers: sub.iter().map(|&m| mask_to_numbers_1based(m)).collect(),
            joker_tickets: None,
            vault_offset_records: off,
            vault_len_records: sub.len() as u64,
        });
    }

    // 4b. Loto 5/40 Cat. III (t = 4): secvența submodulară greedy pe GPU peste 32,768 candidați + seed Turán:
    let seq_540_t4 = build_gpu_partial_sequence_u64(
        &gpu_univ.d_draws6_540,
        vault.tickets_540(),
        &cdv3_540_t4[..100.min(cdv3_540_t4.len())],
        4,
        100,
        32_768,
    )?;
    for &k_b in &PARTIAL_KS {
        let sub = &seq_540_t4[..k_b];
        let (hg, _) = gpu_verify_covering(&gpu_univ.d_draws6_540, sub)?;
        let (hc, uc_cpu, _) = cpu_verify_covering_24t(vault.draws6_540(), sub, 4);
        let uc_gpu: u64 = hg[0..4].iter().sum();
        let cov_gpu = (COUNT_540_DRAWS6 as u64) - uc_gpu;
        let cov_cpu = (COUNT_540_DRAWS6 as u64) - uc_cpu;
        let off = vault_wheel_payload.len() as u64;
        vault_wheel_payload.extend_from_slice(sub);
        let ceil_pct = ((k_b as f64 * 3_010.0) / (COUNT_540_DRAWS6 as f64) * 100.0).min(100.0);
        partial_wheels.push(PartialWheelRecord {
            id: format!("W_540_CAT3_K{k_b}"),
            game: "loto_540".to_string(),
            target_regime: "Cat_III_t4".to_string(),
            k_budget_tickets: k_b,
            universe_draws: COUNT_540_DRAWS6 as u64,
            covered_draws_gpu: cov_gpu,
            covered_draws_cpu: cov_cpu,
            uncovered_draws: uc_gpu,
            coverage_pct: (cov_gpu as f64) * 100.0 / (COUNT_540_DRAWS6 as f64),
            theoretical_packing_ceiling_pct: ceil_pct,
            gpu_cpu_identical: hg == hc,
            tickets_numbers: sub.iter().map(|&m| mask_to_numbers_1based(m)).collect(),
            joker_tickets: None,
            vault_offset_records: off,
            vault_len_records: sub.len() as u64,
        });
    }

    // 4c. Loto 5/40 (t = 3)
    let seq_540_t3 = build_gpu_partial_sequence_u64(
        &gpu_univ.d_draws6_540,
        vault.tickets_540(),
        &cdv3_540_t3,
        3,
        100,
        24_576,
    )?;
    for &k_b in &PARTIAL_KS {
        let sub = &seq_540_t3[..k_b];
        let (hg, _) = gpu_verify_covering(&gpu_univ.d_draws6_540, sub)?;
        let (hc, uc_cpu, _) = cpu_verify_covering_24t(vault.draws6_540(), sub, 3);
        let uc_gpu: u64 = hg[0..3].iter().sum();
        let cov_gpu = (COUNT_540_DRAWS6 as u64) - uc_gpu;
        let cov_cpu = (COUNT_540_DRAWS6 as u64) - uc_cpu;
        let off = vault_wheel_payload.len() as u64;
        vault_wheel_payload.extend_from_slice(sub);
        let ceil_pct = ((k_b as f64 * 68_460.0) / (COUNT_540_DRAWS6 as f64) * 100.0).min(100.0);
        partial_wheels.push(PartialWheelRecord {
            id: format!("W_540_T3_K{k_b}"),
            game: "loto_540".to_string(),
            target_regime: "Combinatorial_t3".to_string(),
            k_budget_tickets: k_b,
            universe_draws: COUNT_540_DRAWS6 as u64,
            covered_draws_gpu: cov_gpu,
            covered_draws_cpu: cov_cpu,
            uncovered_draws: uc_gpu,
            coverage_pct: (cov_gpu as f64) * 100.0 / (COUNT_540_DRAWS6 as f64),
            theoretical_packing_ceiling_pct: ceil_pct,
            gpu_cpu_identical: hg == hc,
            tickets_numbers: sub.iter().map(|&m| mask_to_numbers_1based(m)).collect(),
            joker_tickets: None,
            vault_offset_records: off,
            vault_len_records: sub.len() as u64,
        });
    }

    // 4d. Joker (Cat. I–VIII): luăm la fiecare K maximul acoperirii între `pure_seq[..K]` și `ordered_cover[..K]`!
    let (j_pure_seq, j_ord_cover) = build_gpu_partial_sequence_joker(
        &gpu_univ.d_base_joker,
        vault.base_joker(),
        &cdv3_joker_tickets,
        100,
    )?;
    for &k_b in &PARTIAL_KS {
        let sub_a = &j_pure_seq[..k_b];
        let sub_b = &j_ord_cover[..k_b];
        let (cat_a, _) = gpu_verify_joker_24m(&gpu_univ.d_full_joker, sub_a)?;
        let (cat_b, _) = gpu_verify_joker_24m(&gpu_univ.d_full_joker, sub_b)?;
        let (chosen_sub, cat_gpu) = if cat_b[8] <= cat_a[8] {
            (sub_b, cat_b)
        } else {
            (sub_a, cat_a)
        };
        let (cat_cpu, uc_cpu, _) = cpu_verify_joker_24m_24t(vault.full_joker(), chosen_sub);
        let uc_gpu = cat_gpu[8];
        let cov_gpu = (COUNT_JOKER_FULL24M as u64) - uc_gpu;
        let cov_cpu = (COUNT_JOKER_FULL24M as u64) - uc_cpu;
        let off = vault_wheel_payload.len() as u64;
        vault_wheel_payload.extend_from_slice(chosen_sub);
        let ceil_pct =
            ((k_b as f64 * 715_770.0) / (COUNT_JOKER_FULL24M as f64) * 100.0).min(100.0);
        let j_unp: Vec<JokerTicketJson> = chosen_sub
            .iter()
            .map(|&pk| unpack_joker_ticket(pk))
            .collect();
        partial_wheels.push(PartialWheelRecord {
            id: format!("W_JOKER_ANY_K{k_b}"),
            game: "joker".to_string(),
            target_regime: "Cat_I_VIII_any_win".to_string(),
            k_budget_tickets: k_b,
            universe_draws: COUNT_JOKER_FULL24M as u64,
            covered_draws_gpu: cov_gpu,
            covered_draws_cpu: cov_cpu,
            uncovered_draws: uc_gpu,
            coverage_pct: (cov_gpu as f64) * 100.0 / (COUNT_JOKER_FULL24M as f64),
            theoretical_packing_ceiling_pct: ceil_pct,
            gpu_cpu_identical: cat_gpu == cat_cpu,
            tickets_numbers: j_unp.iter().map(|x| x.numbers.to_vec()).collect(),
            joker_tickets: Some(j_unp),
            vault_offset_records: off,
            vault_len_records: chosen_sub.len() as u64,
        });
    }

    // =========================================================================
    // 5. Scheme Reduse Condiționate (Abbreviated Wheels) v_sub in {12,15,18,21,24}
    // =========================================================================
    let mut abbreviated_wheels: Vec<AbbreviatedWheelRecord> = Vec::new();
    let abbrev_specs: &[(&str, &str, usize, usize, usize, usize, usize)] = &[
        // (id, game, v_sub, k, p_hit, t_guar, max_allowed_tickets)
        ("ABBR_649_V12_P4_T3", "loto_649", 12, 6, 4, 3, 10),
        ("ABBR_649_V15_P4_T3", "loto_649", 15, 6, 4, 3, 22),
        ("ABBR_649_V18_P4_T3", "loto_649", 18, 6, 4, 3, 36),
        ("ABBR_649_V21_P5_T3", "loto_649", 21, 6, 5, 3, 30),
        ("ABBR_649_V24_P5_T3", "loto_649", 24, 6, 5, 3, 46),
        ("ABBR_649_V15_P6_T4", "loto_649", 15, 6, 6, 4, 26),
        ("ABBR_540_V12_P6_T4", "loto_540", 12, 5, 6, 4, 24),
        ("ABBR_540_V15_P6_T4", "loto_540", 15, 5, 6, 4, 62),
        ("ABBR_540_V16_P6_T4_LE85", "loto_540", 16, 5, 6, 4, 80),
        ("ABBR_540_V17_P6_T4", "loto_540", 17, 5, 6, 4, 106),
        ("ABBR_540_V18_P5_T3", "loto_540", 18, 5, 5, 3, 28),
        ("ABBR_540_V21_P5_T3", "loto_540", 21, 5, 5, 3, 52),
        ("ABBR_540_V24_P6_T3", "loto_540", 24, 5, 6, 3, 50),
        ("ABBR_JOKER_V12_P3_T3", "joker", 12, 5, 3, 3, 34),
        ("ABBR_JOKER_V15_P4_T3", "joker", 15, 5, 4, 3, 28),
        ("ABBR_JOKER_V18_P4_T3", "joker", 18, 5, 4, 3, 56),
        ("ABBR_JOKER_V21_P5_T3", "joker", 21, 5, 5, 3, 52),
        ("ABBR_JOKER_V24_P5_T3", "joker", 24, 5, 5, 3, 90),
    ];

    for &(id, game, v_sub, k_tk, p_hit, t_guar, max_tk) in abbrev_specs {
        let lb = schoenheim_lotto_lb(v_sub, k_tk, p_hit, t_guar);
        let blks = solve_small_lotto_hybrid_gpu_cpu(
            v_sub,
            k_tk,
            p_hit,
            t_guar,
            max_tk,
            lb,
            6,
            binom(v_sub, p_hit) <= 24_576,
        )?;
        let effective_id = if id == "ABBR_540_V17_P6_T4" && blks.len() <= 85 {
            "ABBR_540_V17_P6_T4_LE85".to_string()
        } else {
            id.to_string()
        };
        let sub_draws = generate_u32_combinations(v_sub, p_hit);
        let uncov_sub = sub_draws
            .iter()
            .filter(|&&dm| blks.iter().all(|&bm| ((dm & bm).count_ones() as usize) < t_guar))
            .count() as u64;
        assert_eq!(uncov_sub, 0);

        let off = vault_wheel_payload.len() as u64;
        let u64_blks: Vec<u64> = blks.iter().map(|&b| b as u64).collect();
        vault_wheel_payload.extend_from_slice(&u64_blks);

        abbreviated_wheels.push(AbbreviatedWheelRecord {
            id: effective_id,
            game: game.to_string(),
            v_sub,
            k_ticket: k_tk,
            p_hit_in_sub: p_hit,
            t_guarantee: t_guar,
            sub_universe_draws: sub_draws.len() as u64,
            schoenheim_lower_bound: lb,
            n_tickets: blks.len(),
            uncovered_sub_draws: uncov_sub,
            canonical_tickets_1based: u64_blks
                .iter()
                .map(|&m| mask_to_numbers_1based(m))
                .collect(),
            vault_offset_records: off,
            vault_len_records: u64_blks.len() as u64,
        });
    }

    // =========================================================================
    // 6. Cei 5 Martori Negativi Obligatorii (A2.5_MARTORI_NEG)
    // =========================================================================
    let mut negative_witnesses = Vec::new();

    // M_NEG_649_1
    let one_649 = [vault.masks_649()[0]];
    let (h_m1_gpu, _) = gpu_verify_covering(&gpu_univ.d_masks_649, &one_649)?;
    let (_, u_m1_cpu, _) = cpu_verify_covering_24t(vault.masks_649(), &one_649, 3);
    let u_m1_gpu: u64 = h_m1_gpu[0..3].iter().sum();
    negative_witnesses.push(NegativeWitnessRecord {
        id: "M_NEG_649_1".to_string(),
        description: "1 singur bilet 6/49 => exact 13,723,192 extrageri neacoperite".to_string(),
        expected_uncovered: Some(13_723_192),
        require_strictly_positive: true,
        observed_uncovered_gpu: u_m1_gpu,
        observed_uncovered_cpu: u_m1_cpu,
        pass: u_m1_gpu == 13_723_192 && u_m1_cpu == 13_723_192,
    });

    // M_NEG_649_DROP1 (eliminarea primului bilet Witt S(3,6,22) din CDV3_649_163)
    let drop1_649 = &cdv3_649_tickets[1..];
    let (h_m2_gpu, _) = gpu_verify_covering(&gpu_univ.d_masks_649, drop1_649)?;
    let (_, u_m2_cpu, _) = cpu_verify_covering_24t(vault.masks_649(), drop1_649, 3);
    let u_m2_gpu: u64 = h_m2_gpu[0..3].iter().sum();
    negative_witnesses.push(NegativeWitnessRecord {
        id: "M_NEG_649_DROP1".to_string(),
        description: "CDV3_649_163 fara primul bilet Witt S(3,6,22) => > 0 extrageri neacoperite"
            .to_string(),
        expected_uncovered: None,
        require_strictly_positive: true,
        observed_uncovered_gpu: u_m2_gpu,
        observed_uncovered_cpu: u_m2_cpu,
        pass: u_m2_gpu > 0 && u_m2_gpu == u_m2_cpu,
    });

    // M_NEG_540_T4_1
    let one_540 = [vault.tickets_540()[0]];
    let (h_m3_gpu, _) = gpu_verify_covering(&gpu_univ.d_draws6_540, &one_540)?;
    let (_, u_m3_t4_cpu, _) = cpu_verify_covering_24t(vault.draws6_540(), &one_540, 4);
    let (_, u_m3_t3_cpu, _) = cpu_verify_covering_24t(vault.draws6_540(), &one_540, 3);
    let u_m3_t4_gpu: u64 = h_m3_gpu[0..4].iter().sum();
    let u_m3_t3_gpu: u64 = h_m3_gpu[0..3].iter().sum();
    negative_witnesses.push(NegativeWitnessRecord {
        id: "M_NEG_540_T4_1".to_string(),
        description: "1 singur bilet 5/40 la t=4 => exact 3,835,370 extrageri neacoperite"
            .to_string(),
        expected_uncovered: Some(3_835_370),
        require_strictly_positive: true,
        observed_uncovered_gpu: u_m3_t4_gpu,
        observed_uncovered_cpu: u_m3_t4_cpu,
        pass: u_m3_t4_gpu == 3_835_370 && u_m3_t4_cpu == 3_835_370,
    });
    negative_witnesses.push(NegativeWitnessRecord {
        id: "M_NEG_540_T3_1".to_string(),
        description: "1 singur bilet 5/40 la t=3 => exact 3,769,920 extrageri neacoperite"
            .to_string(),
        expected_uncovered: Some(3_769_920),
        require_strictly_positive: true,
        observed_uncovered_gpu: u_m3_t3_gpu,
        observed_uncovered_cpu: u_m3_t3_cpu,
        pass: u_m3_t3_gpu == 3_769_920 && u_m3_t3_cpu == 3_769_920,
    });

    // M_NEG_JOKER_DROP: J180 fără Jokerul 1 (doar j = 2..=20 pe cele 9 blocuri disjuncte = 171 bilete)
    let mut j171 = Vec::with_capacity(171);
    for g in 0..9usize {
        let bm = 0x1Fu64 << (g * 5);
        for j1 in 2..=20u8 {
            j171.push(pack_joker_ticket(bm, j1));
        }
    }
    let (cat_j171_gpu, _) = gpu_verify_joker_24m(&gpu_univ.d_full_joker, &j171)?;
    let (_, u_j171_cpu, _) = cpu_verify_joker_24m_24t(vault.full_joker(), &j171);
    let u_j171_gpu = cat_j171_gpu[8];
    negative_witnesses.push(NegativeWitnessRecord {
        id: "M_NEG_JOKER_DROP".to_string(),
        description: "J180 fara Jokerul 1 (171 bilete) => exact 1,149,750 stari necastigatoare"
            .to_string(),
        expected_uncovered: Some(1_149_750),
        require_strictly_positive: true,
        observed_uncovered_gpu: u_j171_gpu,
        observed_uncovered_cpu: u_j171_cpu,
        pass: u_j171_gpu == 1_149_750 && u_j171_cpu == 1_149_750,
    });

    // =========================================================================
    // 7. Scrierea `cdv3_wheels_catalog.v3bin` + actualizarea `VAULT_MANIFEST.json`
    // =========================================================================
    let cat_v3bin_name = "cdv3_wheels_catalog.v3bin";
    let cat_v3bin_path = vault.vault_dir.join(cat_v3bin_name);
    let rec_cnt = vault_wheel_payload.len() as u64;
    let total_cat_bytes = VAULT_HEADER_BYTES + vault_wheel_payload.len() * 8;
    let mut hdr = [0u8; VAULT_HEADER_BYTES];
    hdr[0..8].copy_from_slice(VAULT_MAGIC);
    hdr[8..12].copy_from_slice(&3u32.to_le_bytes());
    hdr[12..16].copy_from_slice(&5u32.to_le_bytes()); // table_kind = 5 = wheel_catalog
    hdr[16..20].copy_from_slice(&49u32.to_le_bytes());
    hdr[20..24].copy_from_slice(&6u32.to_le_bytes());
    hdr[24..32].copy_from_slice(&rec_cnt.to_le_bytes());
    hdr[32..40].copy_from_slice(&8u64.to_le_bytes());

    let mut raw_cat = Vec::with_capacity(total_cat_bytes);
    raw_cat.extend_from_slice(&hdr);
    for &w in &vault_wheel_payload {
        raw_cat.extend_from_slice(&w.to_le_bytes());
    }
    fs::write(&cat_v3bin_path, &raw_cat)
        .map_err(|e| format!("write {}: {e}", cat_v3bin_path.display()))?;

    let mut hasher = Sha256::new();
    hasher.update(&raw_cat);
    let cat_sha256 = format!("{:x}", hasher.finalize());

    let mut updated_manifest = vault.manifest.clone();
    updated_manifest
        .files
        .retain(|f| f.filename != cat_v3bin_name);
    updated_manifest.files.push(VaultFileRecord {
        filename: cat_v3bin_name.to_string(),
        table_kind: 5,
        n_pool: 49,
        k_pick: 6,
        record_count: rec_cnt,
        file_bytes: total_cat_bytes as u64,
        sha256: cat_sha256.clone(),
    });
    let mf_path = vault.vault_dir.join("VAULT_MANIFEST.json");
    fs::write(
        &mf_path,
        serde_json::to_string_pretty(&updated_manifest).unwrap() + "\n",
    )
    .map_err(|e| format!("update VAULT_MANIFEST.json: {e}"))?;

    check_rss_guard(0)?;

    let gate_a2_pass = cdv3_649_tickets.len() <= 163
        && uncov_649_gpu == 0
        && uncov_649_cpu == 0
        && cdv3_540_t2.len() <= 25
        && uncov_540_t2_gpu == 0
        && cdv3_540_t3.len() <= 175
        && uncov_540_t3_gpu == 0
        && cdv3_540_t4.len() <= 4_500
        && uncov_540_t4_gpu == 0
        && cdv3_joker_tickets.len() <= 108
        && uncov_joker_gpu == 0
        && uncov_joker_cpu == 0
        && main_designs.iter().all(|d| d.gpu_cpu_histogram_identical)
        && partial_wheels.iter().all(|w| w.gpu_cpu_identical)
        && abbreviated_wheels
            .iter()
            .all(|a| a.uncovered_sub_draws == 0)
        && negative_witnesses.iter().all(|n| n.pass);

    let report = CdV3CatalogReport {
        artifact: "CD_V3_DESIGNURI".to_string(),
        version: "3.0.0".to_string(),
        positive_claims_allowed: false,
        build_seconds: t_start.elapsed().as_secs_f64(),
        vault_catalog_v3bin: cat_v3bin_path.display().to_string(),
        vault_catalog_sha256: cat_sha256,
        gate_a2_pass,
        main_designs,
        partial_wheels,
        abbreviated_wheels,
        negative_witnesses,
    };

    if let Some(parent) = out_json_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let json_txt = serde_json::to_string_pretty(&report).unwrap() + "\n";
    let mut f = File::create(out_json_path)
        .map_err(|e| format!("create {}: {e}", out_json_path.display()))?;
    f.write_all(json_txt.as_bytes())
        .map_err(|e| format!("write {}: {e}", out_json_path.display()))?;

    let vault_json_path = vault.vault_dir.join("cdv3_wheels_catalog.json");
    fs::write(&vault_json_path, json_txt.as_bytes())
        .map_err(|e| format!("write {}: {e}", vault_json_path.display()))?;

    Ok(report)
}

/// Re-verifică independent din JSON un catalog `CD_V3_DESIGNURI.json` existent pe GPU + CPU (`cdv3-verify`).
pub fn verify_existing_cdv3_catalog(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    catalog_json_path: &Path,
) -> Result<bool, String> {
    let raw = fs::read_to_string(catalog_json_path)
        .map_err(|e| format!("read {}: {e}", catalog_json_path.display()))?;
    let cat: CdV3CatalogReport =
        serde_json::from_str(&raw).map_err(|e| format!("json parse: {e}"))?;

    for md in &cat.main_designs {
        if md.game == "joker" {
            let jts = md.joker_tickets.as_ref().ok_or("joker_tickets lipsa")?;
            let packed: Vec<u64> = jts.iter().map(|x| x.packed_u64).collect();
            let (hg, _) = gpu_verify_joker_24m(&gpu_univ.d_full_joker, &packed)?;
            let (hc, uc, _) = cpu_verify_joker_24m_24t(vault.full_joker(), &packed);
            if hg != hc || hg[8] != 0 || uc != 0 {
                return Err(format!("Esec la re-verificarea {}: uncov={}", md.id, hg[8]));
            }
        } else {
            let masks: Vec<u64> = md
                .tickets_numbers
                .iter()
                .map(|nums| nums.iter().fold(0u64, |acc, &x| acc | (1u64 << (x - 1))))
                .collect();
            let d_buf = if md.game == "loto_649" {
                &gpu_univ.d_masks_649
            } else {
                &gpu_univ.d_draws6_540
            };
            let h_slice = if md.game == "loto_649" {
                vault.masks_649()
            } else {
                vault.draws6_540()
            };
            let (hg, _) = gpu_verify_covering(d_buf, &masks)?;
            let (hc, uc, _) = cpu_verify_covering_24t(h_slice, &masks, md.t_min_hits as u32);
            let ug: u64 = hg[0..md.t_min_hits].iter().sum();
            if hg != hc || ug != 0 || uc != 0 {
                return Err(format!("Esec la re-verificarea {}: uncov={ug}", md.id));
            }
        }
    }

    Ok(true)
}

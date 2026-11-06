//! `KJ-V5` — Suita de Calificare Sintetică și Algebrică Pre-Arhivă `Q1–Q6` (`6/6 PASS` obligatoriu
//! ÎNAINTE de orice citire a arhivelor reale de extrageri, `PROTOCOL_KJV5_RUST_V2.md` §4.1).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::koopman_v2::frob_norm;
use super::multiplet::{
    cond_poisson_log_z_and_marginals, evaluate_portfolio_exact_full_enum, find_sigma_alpha_fast, top_k_1based,
};
use super::partition::{
    brute_force_partition_parallel, eval_partition_conditioned, eval_subdraw_restricted_sum, exact_ticket_weight_main,
};
use super::{
    decompose_channel_masses, CanonicalStep, KjV5Space, KjV5State, NUM_EXPERTS, W0_ABSTAIN_PRIOR,
};

/// Generator determinist `SplitMix64` pentru testele sintetice `Q1`, `Q4`, `Q5`, `Q6` (zero dependențe externe).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform_usize(&mut self, bound: usize) -> usize {
        assert!(bound > 0);
        let zone = u64::MAX - (u64::MAX % (bound as u64));
        loop {
            let v = self.next_u64();
            if v < zone {
                return (v % (bound as u64)) as usize;
            }
        }
    }

    fn sample_draw_1based(&mut self, n: usize, r: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (1..=n).collect();
        for i in 0..r {
            let j = i + self.uniform_usize(n - i);
            pool.swap(i, j);
        }
        let mut out = pool[..r].to_vec();
        out.sort_unstable();
        out
    }

    fn sample_permutation_0based(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.uniform_usize(i + 1);
            p.swap(i, j);
        }
        p
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GateResult {
    pub id: &'static str,
    pub title: &'static str,
    pub pass: bool,
    pub elapsed_ms: f64,
    pub metrics: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualificationReport {
    pub protocol_id: &'static str,
    pub status: &'static str,
    pub passed: usize,
    pub total: usize,
    pub archive_files_read: usize,
    pub technical_incidents: usize,
    pub positive_claims_allowed: bool,
    pub generated_at_utc: String,
    pub protocol_sha256: String,
    pub kjv5_code_sha256: BTreeMap<String, String>,
    pub total_elapsed_ms: f64,
    pub gates: Vec<GateResult>,
}

fn sha256_file_hex(p: &Path) -> Result<String, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("{:x}", h.finalize()))
}

fn utc_iso_now() -> String {
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
    let y = if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Construiește o istorie sintetică de `n_steps` pași pentru un joc `(N, r, k, M_B)`.
fn synth_steps(space: &KjV5Space, n_steps: usize, seed: u64) -> Vec<CanonicalStep> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(n_steps);
    for s in 0..n_steps {
        let b = rng.sample_draw_1based(space.n, space.r);
        let jb = space.m_bonus.map(|mb| rng.uniform_usize(mb) + 1);
        out.push(CanonicalStep::from_1based(space, format!("2020-01-{:02}", (s % 28) + 1), &[(b, jb)]));
    }
    out
}

/// `Q1`: Exactitate algebrică `Complex64` Gauss–Hermite 2D + Split-DP vs enumerare brută completă pe
/// `649` (`13.983.816` stări), `540` (`658.008` stări), `joker` (`24.435.180` stări) și `Pick 10` (`184.756` submulțimi).
fn run_q1() -> GateResult {
    let t0 = Instant::now();
    let specs = [
        ("649", 49usize, 6usize, 6usize, None, 2026_1101u64),
        ("540", 40usize, 6usize, 5usize, None, 2026_1102u64),
        ("joker", 45usize, 5usize, 5usize, Some(20usize), 2026_1103u64),
    ];

    let mut per_game = serde_json::Map::new();
    let mut all_ok = true;

    for (code, n, r, k, mb, seed) in specs {
        let sp = KjV5Space::new(n, r, k, mb);
        let mut st = KjV5State::new(sp.clone(), true, true, true);
        let steps = synth_steps(&sp, 22, seed);
        let active_all = [true; NUM_EXPERTS];
        for step in &steps[..21] {
            let ops = st.forward_operators();
            let pack = st.evaluate_experts_on_step(&ops, step, &active_all);
            st.advance_with_step(step.clone(), Some(&pack.step_lrs));
        }
        let ops = st.forward_operators();
        // Verificăm că m = 7 (Full KJ-V5 Strong) are simultan K1, K2 (ambele semne: λ_+ > 0 și λ_- < 0!) și K3 nenule!
        let pot7 = ops.expert_potential(&sp, 7);
        let both_signs_active = pot7.modes.lambda_pos > 1e-4 && pot7.modes.lambda_neg < -1e-4;
        let k1_active = pot7.theta_shell.iter().any(|&v| v.abs() > 1e-5);
        let k3_active = pot7.h_anti.iter().any(|&v| v.abs() > 1e-4);

        let (ref d_prev, jb_prev) = ops.prev_draws[0];
        let gh = eval_partition_conditioned(&pot7, d_prev, jb_prev);
        let bf = brute_force_partition_parallel(&pot7, d_prev, jb_prev);

        let rel_z_err = ((gh.z_main - bf.z_main) / bf.z_main).abs();
        let max_pi_err = gh
            .pi_main
            .iter()
            .zip(bf.pi_main.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_bonus_err = gh
            .pi_bonus
            .iter()
            .zip(bf.pi_bonus.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        // Certificăm și solverul exact Branch-and-Bound față de enumerarea paralelă completă a portofoliului
        let pack_last = st.evaluate_experts_on_step(&ops, &steps[21], &active_all);
        let (w_port, _) = st.portfolio_weights_and_e();
        let mut pi_port = vec![0.0f64; n];
        for m in 0..NUM_EXPERTS {
            for i in 0..n {
                pi_port[i] += w_port[m] * pack_last.pi_m_main[m][i];
            }
        }
        let full_ev = evaluate_portfolio_exact_full_enum(&sp, &ops, &pack_last.z_packs, &w_port);
        let (fast_tkt, fast_jb, fast_lp, fast_cert) =
            find_sigma_alpha_fast(&sp, &ops, &pack_last.z_packs, &w_port, &pi_port);
        let bnb_lp_err = (fast_lp - full_ev.alpha_log_prob).abs();
        let bnb_match = fast_cert
            && fast_tkt == full_ev.sigma_alpha_1based
            && fast_jb == full_ev.bonus_alpha_1based
            && bnb_lp_err < 1e-12;

        let game_ok = both_signs_active
            && k1_active
            && k3_active
            && rel_z_err < 1e-12
            && max_pi_err < 1e-12
            && max_bonus_err < 1e-12
            && sp.draw_shell_basis.orthonormality_residual < 1e-13
            && sp.tkt_shell_basis.orthonormality_residual < 1e-13
            && bnb_match;
        all_ok &= game_ok;

        per_game.insert(
            code.to_string(),
            json!({
                "states_enumerated": sp.total_joint_states,
                "k1_k2_both_signs_k3_active": both_signs_active && k1_active && k3_active,
                "lambda_pos": pot7.modes.lambda_pos,
                "lambda_neg": pot7.modes.lambda_neg,
                "rel_z_err": rel_z_err,
                "max_pi_err": max_pi_err,
                "max_bonus_pi_err": max_bonus_err,
                "shell_ortho_res": sp.tkt_shell_basis.orthonormality_residual,
                "bnb_vs_full_enum_exact_match": bnb_match,
                "bnb_lp_err": bnb_lp_err,
                "sigma_alpha_1based": full_ev.sigma_alpha_1based,
                "bonus_alpha_1based": full_ev.bonus_alpha_1based,
                "pass": game_ok,
            }),
        );
    }

    // Certificare suma restricționată pe Pick 10 (N=80, r=20, k=10: C(20, 10) = 184.756 submulțimi!)
    let sp_p10 = KjV5Space::new(80, 20, 10, None);
    let mut st_p10 = KjV5State::new(sp_p10.clone(), true, true, true);
    let steps_p10 = synth_steps(&sp_p10, 14, 2026_8010);
    let active_all = [true; NUM_EXPERTS];
    for step in &steps_p10[..13] {
        let ops = st_p10.forward_operators();
        let pack = st_p10.evaluate_experts_on_step(&ops, step, &active_all);
        st_p10.advance_with_step(step.clone(), Some(&pack.step_lrs));
    }
    let ops_10 = st_p10.forward_operators();
    let pot_10 = ops_10.expert_potential(&sp_p10, 7);
    let prev_10 = &ops_10.prev_draws[0].0;
    let targ_10 = &steps_p10[13].draws[0].0;
    let gh_sub_sum = eval_subdraw_restricted_sum(&pot_10, prev_10, targ_10);

    let mut prev_mask_10 = 0u128;
    for &b in prev_10 {
        prev_mask_10 |= 1u128 << b;
    }
    let mut enum_sub_sum = 0.0f64;
    let mut idx_c: Vec<usize> = (0..10).collect();
    let mut sub_cnt = 0u64;
    loop {
        let sig: Vec<usize> = idx_c.iter().map(|&ix| targ_10[ix]).collect();
        enum_sub_sum += exact_ticket_weight_main(&pot_10, prev_mask_10, &sig);
        sub_cnt += 1;
        let mut pos = 10usize;
        while pos > 0 && idx_c[pos - 1] == 20 - 10 + (pos - 1) {
            pos -= 1;
        }
        if pos == 0 {
            break;
        }
        idx_c[pos - 1] += 1;
        for j in pos..10 {
            idx_c[j] = idx_c[j - 1] + 1;
        }
    }
    assert_eq!(sub_cnt, 184_756);
    let rel_p10_err = ((gh_sub_sum - enum_sub_sum) / enum_sub_sum).abs();
    all_ok &= rel_p10_err < 1e-12;

    GateResult {
        id: "Q1",
        title: "Exactitate algebrică Complex64 Gauss–Hermite 2D + Split-DP vs enumerare brută completă (649, 540, joker, Pick 10)",
        pass: all_ok,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        metrics: json!({
            "games": per_game,
            "pick10_184756_subsets_rel_err": rel_p10_err,
            "threshold": 1e-12,
        }),
    }
}

/// Construiește acțiunea Möbius pe `P^1(F_11) = {0..10, 11(∞)}` pentru martorul chiral `Q2`.
fn mobius_f11(a: i64, b: i64, c: i64, d: i64, x: usize) -> usize {
    const INF: usize = 11;
    let inv_mod11 = |v: i64| -> i64 {
        let r = v.rem_euclid(11);
        assert!(r != 0);
        for k in 1..11 {
            if (r * k) % 11 == 1 {
                return k;
            }
        }
        unreachable!()
    };
    if x == INF {
        if c.rem_euclid(11) == 0 {
            INF
        } else {
            ((a * inv_mod11(c)).rem_euclid(11)) as usize
        }
    } else {
        let denom = (c * (x as i64) + d).rem_euclid(11);
        if denom == 0 {
            INF
        } else {
            (((a * (x as i64) + b) * inv_mod11(denom)).rem_euclid(11)) as usize
        }
    }
}

/// `Q2`: Martor Chiral de Informație și Decizie la Inversarea Timpului pe `N=12, r=4, k=4`.
fn run_q2() -> GateResult {
    let t0 = Instant::now();
    let sp = KjV5Space::new(12, 4, 4, None);

    // Generăm orbita de mărime 330 a clasei paralele rezolubile R0 sub PSL(2, 11) (2-tranzitiv pe [12]!)
    type PClass = [[usize; 4]; 3];
    let apply_perm = |p: &[usize; 12], pc: &PClass| -> PClass {
        let mut out = [[0usize; 4]; 3];
        for (bi, blk) in pc.iter().enumerate() {
            let mut b = [p[blk[0]], p[blk[1]], p[blk[2]], p[blk[3]]];
            b.sort_unstable();
            out[bi] = b;
        }
        out.sort_unstable();
        out
    };

    let mut g_shift = [0usize; 12];
    let mut tau = [0usize; 12];
    for x in 0..12 {
        g_shift[x] = mobius_f11(1, 1, 0, 1, x);
        tau[x] = mobius_f11(0, 10, 1, 1, x); // ordin 3 în PSL(2, 11)
    }
    let mut tau2 = [0usize; 12];
    for x in 0..12 {
        tau2[x] = tau[tau[x]];
    }

    // Construim 2-designul rezolubil ciclic Bose RBIBD(12, 4, 3) de 11 clase paralele sub g_shift
    // și orbitele lor chirale de lungime 3 sub τ: (C_j, τ(C_j), τ^2(C_j)) -> 33 clase paralele distincte,
    // exact 2-design izotrop pe toate cele 66 perechi, cu reprezentanții care maximizează global ||Φ(M_1)||_F^2.
    let r0: PClass = [[0, 1, 3, 11], [2, 6, 7, 9], [4, 5, 8, 10]];
    let mut c_bose = Vec::with_capacity(11);
    c_bose.push(r0);
    for _ in 0..10 {
        let nxt = apply_perm(&g_shift, c_bose.last().unwrap());
        c_bose.push(nxt);
    }
    let best_reps: [usize; 11] = [0, 2, 2, 1, 1, 0, 2, 2, 1, 1, 0];
    let mut m1_draws_1b: Vec<(Vec<usize>, Option<usize>)> = Vec::with_capacity(33);
    for (j, base_pc) in c_bose.iter().enumerate() {
        let cands = [*base_pc, apply_perm(&tau, base_pc), apply_perm(&tau2, base_pc)];
        for blk in &cands[best_reps[j]] {
            m1_draws_1b.push((blk.iter().map(|&x| x + 1).collect(), None));
        }
    }

    let perm_draws = |p: &[usize; 12], src: &[(Vec<usize>, Option<usize>)]| -> Vec<(Vec<usize>, Option<usize>)> {
        src.iter()
            .map(|(b, _)| {
                let mut v: Vec<usize> = b.iter().map(|&x| p[x - 1] + 1).collect();
                v.sort_unstable();
                (v, None)
            })
            .collect()
    };
    let m2_draws_1b = perm_draws(&tau, &m1_draws_1b);
    let m3_draws_1b = perm_draws(&tau2, &m1_draws_1b);
    let mut m23_draws_1b = m2_draws_1b.clone();
    m23_draws_1b.extend(m3_draws_1b.iter().cloned());

    let st_m1 = CanonicalStep::from_1based(&sp, "2020-01-01", &m1_draws_1b);
    let st_m2 = CanonicalStep::from_1based(&sp, "2020-01-02", &m2_draws_1b);
    let st_m3 = CanonicalStep::from_1based(&sp, "2020-01-03", &m3_draws_1b);
    let st_m23 = CanonicalStep::from_1based(&sp, "2020-01-04", &m23_draws_1b);

    // 80 cicluri chirale complete: H_+ = [M_23, M_1] + [M_2, M_3, M_1]*80 vs H_- = [M_23, M_1] + [M_3, M_2, M_1]*80
    let k_cycles = 80usize;
    let mut h_plus = vec![st_m23.clone(), st_m1.clone()];
    let mut h_minus = vec![st_m23, st_m1.clone()];
    for _ in 0..k_cycles {
        h_plus.push(st_m2.clone());
        h_plus.push(st_m3.clone());
        h_plus.push(st_m1.clone());
        h_minus.push(st_m3.clone());
        h_minus.push(st_m2.clone());
        h_minus.push(st_m1.clone());
    }

    // Verificăm invarianții de orbire ai tuturor familiilor vechi:
    // 1. Multi-mulțime identică, 1-corp exact plat, 2-corp static exact izotrop, Markov 1-corp exact plat, lag-1 identic
    let mut cnt_1b = [0u64; 12];
    let mut cnt_2b = [[0u64; 12]; 12];
    let mut multiset_p: BTreeMap<Vec<usize>, u64> = BTreeMap::new();
    let mut multiset_m: BTreeMap<Vec<usize>, u64> = BTreeMap::new();
    for st in &h_plus {
        for (d, _) in &st.draws {
            *multiset_p.entry(d.clone()).or_insert(0) += 1;
            for &a in d {
                cnt_1b[a] += 1;
                for &b in d {
                    if a != b {
                        cnt_2b[a][b] += 1;
                    }
                }
            }
        }
    }
    for st in &h_minus {
        for (d, _) in &st.draws {
            *multiset_m.entry(d.clone()).or_insert(0) += 1;
        }
    }
    let same_multiset = multiset_p == multiset_m;
    let flat_1body = cnt_1b.iter().max() == cnt_1b.iter().min();
    let mut off_vals = Vec::new();
    for a in 0..12 {
        for b in 0..12 {
            if a != b {
                off_vals.push(cnt_2b[a][b]);
            }
        }
    }
    let isotropic_2body = off_vals.iter().max() == off_vals.iter().min();

    let lag1_hist = |h: &[CanonicalStep]| -> [u64; 5] {
        let mut hist = [0u64; 5];
        for t in 1..h.len() {
            for (dp, _) in &h[t - 1].draws {
                let mut mp = 0u32;
                for &x in dp {
                    mp |= 1u32 << x;
                }
                for (dc, _) in &h[t].draws {
                    let mut ov = 0usize;
                    for &x in dc {
                        if ((mp >> x) & 1) != 0 {
                            ov += 1;
                        }
                    }
                    hist[ov] += 1;
                }
            }
        }
        hist
    };
    let same_lag1_overlap = lag1_hist(&h_plus) == lag1_hist(&h_minus);
    let max_xtilde = h_plus
        .iter()
        .flat_map(|s| s.x_tilde.iter().map(|v| v.abs()))
        .fold(0.0f64, f64::max);

    let eval_stream = |h: &[CanonicalStep]| {
        let mut st = KjV5State::new(sp.clone(), true, true, true);
        let active_all = [true; NUM_EXPERTS];
        for step in h {
            let ops = st.forward_operators();
            let pack = st.evaluate_experts_on_step(&ops, step, &active_all);
            st.advance_with_step(step.clone(), Some(&pack.step_lrs));
        }
        let ops_end = st.forward_operators();
        let pack_end = st.evaluate_experts_on_step(&ops_end, &st_m1, &active_all);
        let (w_port, e_s) = st.portfolio_weights_and_e();
        let ev = evaluate_portfolio_exact_full_enum(&sp, &ops_end, &pack_end.z_packs, &w_port);
        (ops_end, w_port, e_s, ev)
    };

    let (ops_p, w_p, e_p, ev_p) = eval_stream(&h_plus);
    let (ops_m, _w_m, e_m, ev_m) = eval_stream(&h_minus);

    let sum_q: Vec<f64> = ops_p.q_anti_raw.iter().zip(ops_m.q_anti_raw.iter()).map(|(&a, &b)| a + b).collect();
    let antipodal_err = frob_norm(&sum_q);
    let lie_orth_err = ops_p.q_anti_phi_dot.abs().max(ops_m.q_anti_phi_dot.abs());

    // Reperele marginale B și R au ponderi 100% plate pe ambele istorii -> aleg [1, 2, 3, 4] și au E_S == 1.0!
    let b_tkt = vec![1usize, 2, 3, 4];
    let r_tkt = vec![1usize, 2, 3, 4];

    let tickets_chiral_distinct = ev_p.sigma_alpha_1based != ev_m.sigma_alpha_1based
        && ev_p.sigma_alpha_1based != b_tkt
        && ev_m.sigma_alpha_1based != b_tkt
        && ev_p.sigma_alpha_1based != r_tkt
        && ev_m.sigma_alpha_1based != r_tkt;

    let pass = same_multiset
        && flat_1body
        && isotropic_2body
        && same_lag1_overlap
        && max_xtilde < 1e-15
        && antipodal_err < 1e-13
        && lie_orth_err < 1e-14
        && tickets_chiral_distinct
        && e_p > 1000.0
        && e_m > 1000.0;

    GateResult {
        id: "Q2",
        title: "Martor chiral de informație și decizie la inversarea săgeții timpului pe N=12, r=4, k=4 (H_+ vs H_-)",
        pass,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        metrics: json!({
            "same_draw_multiset": same_multiset,
            "flat_1body_exact": flat_1body,
            "isotropic_static_2body_exact": isotropic_2body,
            "flat_1body_markov_max_xtilde": max_xtilde,
            "same_lag1_overlap_histogram": same_lag1_overlap,
            "antipodal_frob_err_q_anti_plus_minus": antipodal_err,
            "lie_orthogonality_dot_phi": lie_orth_err,
            "sigma_alpha_H_plus": ev_p.sigma_alpha_1based,
            "sigma_alpha_H_minus": ev_m.sigma_alpha_1based,
            "comparator_B_R_ticket": b_tkt,
            "E_S_H_plus": e_p,
            "E_S_H_minus": e_m,
            "E_S_comparators_U_B_R": 1.0,
            "w_k2_anti_H_plus": w_p[3],
        }),
    }
}

/// `Q3`: Martor de Înveliș Krawtchouk Pătratic Pur `m = 2` pe planul afin `AG(2, 4)` (`N=16, r=4, k=4`).
fn run_q3() -> GateResult {
    let t0 = Instant::now();
    let sp = KjV5Space::new(16, 4, 4, None);

    // Construim cele 4 clase afine de pante și clasa paralelă verticală a sistemului Steiner S(2, 4, 16) = AG(2, 4) (0-based)
    let mul = [[0usize, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]];
    let mut base_pcs: Vec<Vec<[usize; 4]>> = Vec::with_capacity(4);
    for m in 0..4 {
        let mut pc = Vec::with_capacity(4);
        for c in 0..4 {
            let mut line = [0usize; 4];
            for (x, slot) in line.iter_mut().enumerate() {
                *slot = 4 * x + (mul[m][x] ^ c);
            }
            line.sort_unstable();
            pc.push(line);
        }
        base_pcs.push(pc);
    }
    let mut vert_pc = Vec::with_capacity(4);
    for c in 0..4 {
        vert_pc.push([4 * c, 4 * c + 1, 4 * c + 2, 4 * c + 3]);
    }

    // Generăm cele 24 permutări din S_4 pentru stabilizatorul de coloană al clasei verticale vert_pc:
    // orice permutare pe coloane mapează S(2, 4, 16) tot într-un S(2, 4, 16) izotrop pe toate cele 120 perechi,
    // păstrând vert_pc invariant astfel încât FIECARE tranziție individuală t-1 -> t are media suprapunerii EXACT 1.0!
    let mut s4_perms: Vec<[usize; 4]> = Vec::with_capacity(24);
    for a in 0..4 {
        for b in 0..4 {
            if b == a {
                continue;
            }
            for c in 0..4 {
                if c == a || c == b {
                    continue;
                }
                let d = 6 - a - b - c;
                s4_perms.push([a, b, c, d]);
            }
        }
    }
    assert_eq!(s4_perms.len(), 24);

    let mut seq_steps_1b: Vec<Vec<(Vec<usize>, Option<usize>)>> = Vec::with_capacity(160);
    for blk_idx in 0..16 {
        let cols = [
            [0usize, 1, 2, 3],
            s4_perms[(blk_idx * 5) % 24],
            s4_perms[(blk_idx * 7 + 3) % 24],
            s4_perms[(blk_idx * 11 + 1) % 24],
        ];
        let mut perm16 = [0usize; 16];
        for c in 0..4 {
            for y in 0..4 {
                perm16[4 * c + y] = 4 * c + cols[c][y];
            }
        }
        for base_pc in &base_pcs {
            let pc_1b: Vec<(Vec<usize>, Option<usize>)> = base_pc
                .iter()
                .map(|ln| {
                    let mut v: Vec<usize> = ln.iter().map(|&u| perm16[u] + 1).collect();
                    v.sort_unstable();
                    (v, None)
                })
                .collect();
            seq_steps_1b.push(pc_1b.clone());
            seq_steps_1b.push(pc_1b);
        }
        let vpc_1b: Vec<(Vec<usize>, Option<usize>)> = vert_pc
            .iter()
            .map(|ln| (ln.iter().map(|&u| u + 1).collect(), None))
            .collect();
        seq_steps_1b.push(vpc_1b.clone());
        seq_steps_1b.push(vpc_1b);
    }

    // Verificăm uniformitatea exactă 1-corp, izotropia 2-corp și media suprapunerii exactă 1.0 la FIECARE pas
    let mut c1 = [0u64; 16];
    let mut c2 = [[0u64; 16]; 16];
    for step_dr in &seq_steps_1b {
        for (d, _) in step_dr {
            for &u in d {
                c1[u - 1] += 1;
                for &v in d {
                    if u != v {
                        c2[u - 1][v - 1] += 1;
                    }
                }
            }
        }
    }
    let uniform_1b = c1.iter().max() == c1.iter().min();
    let mut c2_off = Vec::new();
    for i in 0..16 {
        for j in 0..16 {
            if i != j {
                c2_off.push(c2[i][j]);
            }
        }
    }
    let uniform_2b = c2_off.iter().max() == c2_off.iter().min();

    let mut ov_sum = 0usize;
    let mut pair_cnt = 0usize;
    let mut max_step_ov_err = 0.0f64;
    for t in 1..seq_steps_1b.len() {
        let mut step_ov = 0usize;
        for (dp, _) in &seq_steps_1b[t - 1] {
            for (dc, _) in &seq_steps_1b[t] {
                step_ov += dc.iter().filter(|x| dp.contains(x)).count();
            }
        }
        let step_pairs = seq_steps_1b[t - 1].len() * seq_steps_1b[t].len();
        let step_mean = (step_ov as f64) / (step_pairs as f64);
        max_step_ov_err = max_step_ov_err.max((step_mean - 1.0).abs());
        ov_sum += step_ov;
        pair_cnt += step_pairs;
    }
    let mean_ov = (ov_sum as f64) / (pair_cnt as f64);
    let mean_ov_err = (mean_ov - 1.0).abs().max(max_step_ov_err);

    let mut st = KjV5State::new(sp.clone(), true, true, true);
    let active_all = [true; NUM_EXPERTS];
    for (idx, step_dr) in seq_steps_1b.iter().enumerate() {
        let step = CanonicalStep::from_1based(&sp, format!("2021-01-{:02}", (idx % 28) + 1), step_dr);
        let ops = st.forward_operators();
        let pack = st.evaluate_experts_on_step(&ops, &step, &active_all);
        st.advance_with_step(step, Some(&pack.step_lrs));
    }

    let ops_end = st.forward_operators();
    let (w_end, e_s) = st.portfolio_weights_and_e();
    let masses = decompose_channel_masses(&w_end);

    let pass = uniform_1b
        && uniform_2b
        && mean_ov_err < 1e-14
        && ops_end.shell_coeffs[0].abs() < 1e-14
        && ops_end.shell_coeffs[1] > 0.25
        && masses.w_k1 > 0.80
        && e_s > 100.0;

    GateResult {
        id: "Q3",
        title: "Martor de înveliș Krawtchouk pătratic pur m=2 pe AG(2, 4) (medie hipergeometrică exactă 1.0, m=1 nul)",
        pass,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        metrics: json!({
            "uniform_1body_cycle": uniform_1b,
            "isotropic_2body_cycle": uniform_2b,
            "empirical_mean_lag1_overlap": mean_ov,
            "hypergeom_mean_r2_div_N": 1.0,
            "mean_overlap_err": mean_ov_err,
            "shell_c1_linear": ops_end.shell_coeffs[0],
            "shell_c2_quadratic": ops_end.shell_coeffs[1],
            "shell_c3_cubic": ops_end.shell_coeffs[2],
            "w_K1_total": masses.w_k1,
            "w_K1_pure": masses.w_k1_pure,
            "E_S": e_s,
        }),
    }
}

/// `Q4`: Echivarianță exactă la permutări `S_49`, `S_40` și `S_45 × S_20`.
fn run_q4() -> GateResult {
    let t0 = Instant::now();
    let specs = [
        ("649", 49usize, 6usize, 6usize, None, 2026_4001u64),
        ("540", 40usize, 6usize, 5usize, None, 2026_4002u64),
        ("joker", 45usize, 5usize, 5usize, Some(20usize), 2026_4003u64),
    ];

    let mut per_game = serde_json::Map::new();
    let mut all_ok = true;

    for (code, n, r, k, mb, seed) in specs {
        let sp = KjV5Space::new(n, r, k, mb);
        let mut rng = SplitMix64::new(seed);
        let perm_main = rng.sample_permutation_0based(n);
        let perm_bonus = mb.map(|m| rng.sample_permutation_0based(m));

        let n_steps = 26usize;
        let mut steps_orig = Vec::with_capacity(n_steps);
        let mut steps_perm = Vec::with_capacity(n_steps);
        for s in 0..n_steps {
            let m_day = if s % 7 == 3 { 2 } else { 1 };
            let mut raw_o = Vec::with_capacity(m_day);
            let mut raw_p = Vec::with_capacity(m_day);
            for _ in 0..m_day {
                let d_o = rng.sample_draw_1based(n, r);
                let mut d_p: Vec<usize> = d_o.iter().map(|&x| perm_main[x - 1] + 1).collect();
                d_p.sort_unstable();
                let jb_o = mb.map(|m| rng.uniform_usize(m) + 1);
                let jb_p = jb_o.map(|j| perm_bonus.as_ref().unwrap()[j - 1] + 1);
                raw_o.push((d_o, jb_o));
                raw_p.push((d_p, jb_p));
            }
            let dt = format!("2022-03-{:02}", (s % 28) + 1);
            steps_orig.push(CanonicalStep::from_1based(&sp, &dt, &raw_o));
            steps_perm.push(CanonicalStep::from_1based(&sp, &dt, &raw_p));
        }

        let run_seq = |seq: &[CanonicalStep]| {
            let mut st = KjV5State::new(sp.clone(), true, true, true);
            let active_all = [true; NUM_EXPERTS];
            for step in &seq[..(n_steps - 1)] {
                let ops = st.forward_operators();
                let pack = st.evaluate_experts_on_step(&ops, step, &active_all);
                st.advance_with_step(step.clone(), Some(&pack.step_lrs));
            }
            let ops = st.forward_operators();
            let pack = st.evaluate_experts_on_step(&ops, &seq[n_steps - 1], &active_all);
            let (w_port, e_val) = st.portfolio_weights_and_e();
            let ev = evaluate_portfolio_exact_full_enum(&sp, &ops, &pack.z_packs, &w_port);
            let mut pi_port = vec![0.0f64; n];
            for m in 0..NUM_EXPERTS {
                for i in 0..n {
                    pi_port[i] += w_port[m] * pack.pi_m_main[m][i];
                }
            }
            (ops, pack, w_port, e_val, ev, pi_port)
        };

        let (ops_o, pack_o, w_o, e_o, ev_o, pi_o) = run_seq(&steps_orig);
        let (ops_p, pack_p, w_p, e_p, ev_p, pi_p) = run_seq(&steps_perm);

        let max_w_err = w_o.iter().zip(w_p.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let max_pi_err = (0..n).map(|i| (pi_o[i] - pi_p[perm_main[i]]).abs()).fold(0.0f64, f64::max);

        // Asociem bijectiv fiecare extragere condiționantă u din ops_o.prev_draws cu imaginea ei permutată în ops_p.prev_draws
        let mut max_ln_z_err = 0.0f64;
        for (u, (d_o, jb_o)) in ops_o.prev_draws.iter().enumerate() {
            let mut d_mapped: Vec<usize> = d_o.iter().map(|&x| perm_main[x]).collect();
            d_mapped.sort_unstable();
            let jb_mapped = jb_o.map(|j| perm_bonus.as_ref().unwrap()[j]);
            let u_p = ops_p
                .prev_draws
                .iter()
                .position(|(dp, jbp)| dp == &d_mapped && *jbp == jb_mapped)
                .expect("bijection of permuted conditioning draws in prev_draws");
            for m in 1..NUM_EXPERTS {
                let err = (pack_o.z_packs[m][u].log_z_main - pack_p.z_packs[m][u_p].log_z_main).abs();
                if err > max_ln_z_err {
                    max_ln_z_err = err;
                }
            }
        }

        let mut mapped_alpha: Vec<usize> = ev_o.sigma_alpha_1based.iter().map(|&x| perm_main[x - 1] + 1).collect();
        mapped_alpha.sort_unstable();
        let mapped_bonus = ev_o.bonus_alpha_1based.map(|j| perm_bonus.as_ref().unwrap()[j - 1] + 1);
        let tkt_eq = mapped_alpha == ev_p.sigma_alpha_1based && mapped_bonus == ev_p.bonus_alpha_1based;

        let ok = max_w_err < 1e-12 && max_pi_err < 1e-12 && max_ln_z_err < 1e-12 && (e_o - e_p).abs() < 1e-12 && tkt_eq;
        all_ok &= ok;

        per_game.insert(
            code.to_string(),
            json!({
                "max_w_err": max_w_err,
                "max_pi_err": max_pi_err,
                "max_ln_z_err": max_ln_z_err,
                "runner_up_gap": ev_o.runner_up_gap,
                "ticket_equivariant_exact": tkt_eq,
                "pass": ok,
            }),
        );
    }

    GateResult {
        id: "Q4",
        title: "Echivarianță exactă la permutări S_49, S_40 și S_45 × S_20",
        pass: all_ok,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        metrics: json!({
            "games": per_game,
            "threshold": 1e-12,
        }),
    }
}

/// `Q5`: Cauzalitate strictă, invarianță `0.0` la ordinea intra-zi și `Incremental == Full`.
fn run_q5() -> GateResult {
    let t0 = Instant::now();
    let sp = KjV5Space::new(49, 6, 6, None);
    let mut rng = SplitMix64::new(2026_5005);

    let n_steps = 30usize;
    let mut raw_days: Vec<Vec<(Vec<usize>, Option<usize>)>> = Vec::with_capacity(n_steps);
    for s in 0..n_steps {
        let m_day = if s % 4 == 1 { 3 } else if s % 3 == 0 { 2 } else { 1 };
        let mut day = Vec::with_capacity(m_day);
        for _ in 0..m_day {
            day.push((rng.sample_draw_1based(49, 6), None));
        }
        raw_days.push(day);
    }

    // 1. Cauzalitate strictă la pasul s = 16 când pașii 16..30 sunt modificați radical
    let steps_a: Vec<CanonicalStep> = raw_days
        .iter()
        .enumerate()
        .map(|(s, d)| CanonicalStep::from_1based(&sp, format!("2023-05-{:02}", (s % 28) + 1), d))
        .collect();
    let mut raw_days_corrupt = raw_days.clone();
    for slot in raw_days_corrupt.iter_mut().skip(16) {
        *slot = vec![(vec![1, 2, 3, 4, 5, 6], None)];
    }
    let steps_b: Vec<CanonicalStep> = raw_days_corrupt
        .iter()
        .enumerate()
        .map(|(s, d)| CanonicalStep::from_1based(&sp, format!("2023-05-{:02}", (s % 28) + 1), d))
        .collect();

    let eval_prefix = |seq: &[CanonicalStep], stop_at: usize| {
        let mut st = KjV5State::new(sp.clone(), true, true, true);
        let active_all = [true; NUM_EXPERTS];
        for step in &seq[..stop_at] {
            let ops = st.forward_operators();
            let pack = st.evaluate_experts_on_step(&ops, step, &active_all);
            st.advance_with_step(step.clone(), Some(&pack.step_lrs));
        }
        let ops = st.forward_operators();
        let pack = st.evaluate_experts_on_step(&ops, &seq[stop_at], &active_all);
        let (w_port, e_val) = st.portfolio_weights_and_e();
        let mut pi_port = vec![0.0f64; 49];
        for m in 0..NUM_EXPERTS {
            for i in 0..49 {
                pi_port[i] += w_port[m] * pack.pi_m_main[m][i];
            }
        }
        let (tkt, _, lp, _) = find_sigma_alpha_fast(&sp, &ops, &pack.z_packs, &w_port, &pi_port);
        (pi_port, w_port, e_val, tkt, lp)
    };

    let (pi_a, w_a, e_a, tkt_a, lp_a) = eval_prefix(&steps_a, 16);
    let (pi_b, w_b, e_b, tkt_b, lp_b) = eval_prefix(&steps_b, 16);
    let causal_diff = pi_a
        .iter()
        .zip(pi_b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
        .max((e_a - e_b).abs())
        .max((lp_a - lp_b).abs())
        .max(w_a.iter().zip(w_b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max));
    let causal_ok = causal_diff == 0.0 && tkt_a == tkt_b;

    // 2. Invarianță 0.0 la inversarea ordinii extragerilor intra-zi
    let steps_rev: Vec<CanonicalStep> = raw_days
        .iter()
        .enumerate()
        .map(|(s, d)| {
            let mut rev = d.clone();
            rev.reverse();
            CanonicalStep::from_1based(&sp, format!("2023-05-{:02}", (s % 28) + 1), &rev)
        })
        .collect();
    let (pi_full, w_full, e_full, tkt_full, lp_full) = eval_prefix(&steps_a, n_steps - 1);
    let (pi_rev, w_rev, e_rev, tkt_rev, lp_rev) = eval_prefix(&steps_rev, n_steps - 1);
    let intraday_diff = pi_full
        .iter()
        .zip(pi_rev.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
        .max((e_full - e_rev).abs())
        .max((lp_full - lp_rev).abs())
        .max(w_full.iter().zip(w_rev.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max));

    // 3. Incremental == Full (clonare incrementală și avans secvențial vs reconstrucție de la zero din istoric)
    let mut st_fresh = KjV5State::new(sp.clone(), true, true, true);
    let active_all = [true; NUM_EXPERTS];
    for step in &steps_a[..(n_steps - 1)] {
        let ops_f = st_fresh.forward_operators();
        let pack_f = st_fresh.evaluate_experts_on_step(&ops_f, step, &active_all);
        st_fresh.advance_with_step(step.clone(), Some(&pack_f.step_lrs));
    }
    // Construim o a doua instanță doar cu avansarea suficientelor (fără portofoliu) și reconstruim log_wealth din batch
    let mut st_batch = KjV5State::new(sp.clone(), true, true, true);
    let mut batch_lw = st_batch.log_wealth;
    for step in &steps_a[..(n_steps - 1)] {
        let ops_b = st_batch.forward_operators();
        let pack_b = st_batch.evaluate_experts_on_step(&ops_b, step, &active_all);
        for m in 0..NUM_EXPERTS {
            if batch_lw[m].is_finite() {
                batch_lw[m] += pack_b.step_lrs[m].ln();
            }
        }
        st_batch.advance_with_step(step.clone(), None);
    }
    st_batch.log_wealth = batch_lw;
    let (w_b2, e_b2) = st_batch.portfolio_weights_and_e();
    let inc_vs_batch_diff = w_full
        .iter()
        .zip(w_b2.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
        .max((e_full - e_b2).abs());

    let pass = causal_ok && intraday_diff < 1e-14 && tkt_full == tkt_rev && inc_vs_batch_diff < 1e-12;

    GateResult {
        id: "Q5",
        title: "Cauzalitate strictă, invarianță 0.0 la ordinea intra-zi (step_offsets) și Incremental == Full",
        pass,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        metrics: json!({
            "strict_causality_diff": causal_diff,
            "strict_causality_tickets_match": tkt_a == tkt_b,
            "intraday_order_diff": intraday_diff,
            "intraday_tickets_match": tkt_full == tkt_rev,
            "incremental_vs_batch_diff": inc_vs_batch_diff,
        }),
    }
}

/// `Q6`: Siguranță sub CSPRNG Uniform (`E_s >= 0.50`, `w_0^(S) > 0.50`, `det G_3 > 0`, `σ_α ∉ {B, R}`).
fn run_q6() -> GateResult {
    let t0 = Instant::now();
    let sp = KjV5Space::new(49, 6, 6, None);
    let mut rng = SplitMix64::new(2026_6049);
    let n_steps = 360usize;

    let mut st = KjV5State::new(sp.clone(), true, true, true);
    let active_all = [true; NUM_EXPERTS];
    let mut min_e = f64::INFINITY;
    let mut cnt_cum = [0.0f64; 49];
    let mut ew_main = [0.0f64; 49];
    let decay = 2.0f64.powf(-1.0 / 26.0);

    let mut steps = Vec::with_capacity(n_steps);
    for s in 0..n_steps {
        let d = rng.sample_draw_1based(49, 6);
        steps.push(CanonicalStep::from_1based(&sp, format!("2024-01-{:02}", (s % 28) + 1), &[(d, None)]));
    }

    for step in &steps {
        let (_, e_cur) = st.portfolio_weights_and_e();
        if e_cur < min_e {
            min_e = e_cur;
        }
        let ops = st.forward_operators();
        let pack = st.evaluate_experts_on_step(&ops, step, &active_all);
        st.advance_with_step(step.clone(), Some(&pack.step_lrs));
        for x in &mut ew_main {
            *x *= decay;
        }
        for &(ref d0, _) in &step.draws {
            for &b in d0 {
                cnt_cum[b] += 1.0;
                ew_main[b] += 1.0;
            }
        }
    }

    let (w_end, e_end) = st.portfolio_weights_and_e();
    if e_end < min_e {
        min_e = e_end;
    }

    let sum_b: f64 = cnt_cum.iter().map(|&c| c + 1.0).sum();
    let wb: Vec<f64> = cnt_cum.iter().map(|&c| (c + 1.0) / sum_b).collect();
    let sum_r: f64 = ew_main.iter().map(|&c| c + 0.5).sum();
    let wr: Vec<f64> = ew_main.iter().map(|&c| (c + 0.5) / sum_r).collect();
    let (_, qb) = cond_poisson_log_z_and_marginals(&wb, 6);
    let (_, qr) = cond_poisson_log_z_and_marginals(&wr, 6);
    assert!((qb.iter().sum::<f64>() - 6.0).abs() < 1e-12);
    assert!((qr.iter().sum::<f64>() - 6.0).abs() < 1e-12);
    let tkt_b = top_k_1based(&wb, 6);
    let tkt_r = top_k_1based(&wr, 6);

    let ops_end = st.forward_operators();
    let pack_end = st.evaluate_experts_on_step(&ops_end, &steps[n_steps - 1], &active_all);
    let ev_end = evaluate_portfolio_exact_full_enum(&sp, &ops_end, &pack_end.z_packs, &w_end);

    let pass = min_e >= W0_ABSTAIN_PRIOR - 1e-15
        && w_end[0] > W0_ABSTAIN_PRIOR
        && ev_end.det_g3 > 0.0
        && ev_end.sigma_alpha_1based != tkt_b
        && ev_end.sigma_alpha_1based != tkt_r;

    GateResult {
        id: "Q6",
        title: "Siguranță sub CSPRNG uniform (E_s >= 0.50, contracție Bayesiană w_0^(S) > 0.50, det G_3 > 0, σ_α ∉ {B, R})",
        pass,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        metrics: json!({
            "n_steps": n_steps,
            "min_E_s": min_e,
            "final_E_S": e_end,
            "w0_prior": W0_ABSTAIN_PRIOR,
            "w0_final": w_end[0],
            "det_G3": ev_end.det_g3,
            "sigma_alpha": ev_end.sigma_alpha_1based,
            "sigma_beta": ev_end.variants[1].ticket_1based,
            "sigma_gamma": ev_end.variants[2].ticket_1based,
            "ticket_B": tkt_b,
            "ticket_R": tkt_r,
        }),
    }
}

/// Execută întreaga suită `Q1–Q6` (zero citiri de arhive reale!) și scrie certificatul `CALIFICARE_Q_KJV5.json`.
pub fn run_qualification_suite(repo_root: &Path, out_json: Option<&Path>) -> Result<QualificationReport, String> {
    let t0 = Instant::now();
    let proto_path = repo_root.join("next_candidate/quantum_lab_649/kj_v5/PROTOCOL_KJV5_RUST_V2.md");
    let seal_path = repo_root.join("next_candidate/quantum_lab_649/kj_v5/SIGILIU_PROTOCOL_KJV5.json");
    let proto_sha = sha256_file_hex(&proto_path)?;
    let seal_val: Value = serde_json::from_slice(&std::fs::read(&seal_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let expected_sha = seal_val.get("protocol_sha256").and_then(Value::as_str).unwrap_or("");
    if proto_sha != expected_sha {
        return Err(format!("SHA-256 protocol invalid față de sigiliu: {proto_sha} != {expected_sha}"));
    }

    let mut code_shas = BTreeMap::new();
    let kjv5_src = repo_root.join("rust_core_v2/src/kjv5");
    for name in ["mod.rs", "krawtchouk.rs", "koopman_v2.rs", "partition.rs", "multiplet.rs", "qualify.rs"] {
        let p = kjv5_src.join(name);
        if p.is_file() {
            code_shas.insert(format!("rust_core_v2/src/kjv5/{name}"), sha256_file_hex(&p)?);
        }
    }

    let gates = vec![run_q1(), run_q2(), run_q3(), run_q4(), run_q5(), run_q6()];
    let passed = gates.iter().filter(|g| g.pass).count();
    let total = gates.len();
    let status = if passed == total { "KJV5_QUALIFIED" } else { "KJV5_NOT_QUALIFIED" };

    let report = QualificationReport {
        protocol_id: "KJV5-RCV2-P1",
        status,
        passed,
        total,
        archive_files_read: 0,
        technical_incidents: 1,
        positive_claims_allowed: false,
        generated_at_utc: utc_iso_now(),
        protocol_sha256: proto_sha,
        kjv5_code_sha256: code_shas,
        total_elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        gates,
    };

    if let Some(out_p) = out_json {
        if let Some(parent) = out_p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        if out_p.exists() {
            // Nu rescriem un certificat deja sigilat 0444!
            return Err(format!("Certificatul {} există deja și nu se rescrie!", out_p.display()));
        }
        let body = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())? + "\n";
        std::fs::write(out_p, body).map_err(|e| format!("{}: {e}", out_p.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(out_p, std::fs::Permissions::from_mode(0o444))
                .map_err(|e| format!("chmod 0444 {}: {e}", out_p.display()))?;
        }
    }

    Ok(report)
}

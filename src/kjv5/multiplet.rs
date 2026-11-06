//! `KJ-V5` — Solverul exact de stare fundamentală `σ_α` (enumerare paralelă completă pe 24 fire CPU +
//! Branch-and-Bound pe intervale certificat global), multipletul repulsiv Coulomb–Jastrow `(α, β, γ)`
//! (`det G_3 > 0`), nucleul determinist și reperele cauzale `U`, `B`, `R` (`PROTOCOL_KJV5_RUST_V2.md` §3.4, §4).

use serde::Serialize;

use super::krawtchouk::binom_f64;
use super::partition::ExpertPotential;
use super::{KjV5Space, KjV5StepOperators, NUM_EXPERTS};

/// O variantă din multipletul Coulomb–Jastrow (`α`, `β` sau `γ`) — stare statistică istorică, NU predicție.
#[derive(Debug, Clone, Serialize)]
pub struct MultipletVariant {
    pub label: &'static str,
    pub ticket_1based: Vec<usize>,
    pub bonus_1based: Option<usize>,
    pub log_prob: f64,
    pub log_lift_vs_uniform: f64,
}

/// Rezultatul complet al evaluării portofoliului în spațiul stărilor.
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioFullEval {
    pub sigma_alpha_1based: Vec<usize>,
    pub bonus_alpha_1based: Option<usize>,
    pub alpha_log_prob: f64,
    pub runner_up_gap: f64,
    pub variants: Vec<MultipletVariant>,
    pub gram_matrix_3x3: [[f64; 3]; 3],
    pub det_g3: f64,
    pub deterministic_core_1based: Vec<usize>,
    pub exact_global_certified: bool,
}

#[derive(Debug, Clone)]
struct Candidate {
    prob: f64,
    balls_0based: Vec<usize>,
    bonus_0based: Option<usize>,
}

fn lex_lt(a: &[usize], ab: Option<usize>, b: &[usize], bb: Option<usize>) -> bool {
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x != y {
            return x < y;
        }
    }
    ab < bb
}

fn better_cand(p_new: f64, b_new: &[usize], j_new: Option<usize>, cur: &Candidate) -> bool {
    let scale = cur.prob.abs().max(p_new.abs()).max(1e-300);
    if p_new > cur.prob + 1e-13 * scale {
        true
    } else if (p_new - cur.prob).abs() <= 1e-13 * scale {
        lex_lt(b_new, j_new, &cur.balls_0based, cur.bonus_0based)
    } else {
        false
    }
}

fn push_top_k(top: &mut Vec<Candidate>, cand: Candidate, cap: usize) {
    // Evită duplicatele exacte
    if top.iter().any(|c| c.balls_0based == cand.balls_0based && c.bonus_0based == cand.bonus_0based) {
        return;
    }
    if top.len() < cap {
        top.push(cand);
        if top.len() == cap {
            top.sort_by(|a, b| {
                if better_cand(a.prob, &a.balls_0based, a.bonus_0based, b) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });
        }
        return;
    }
    if let Some(worst) = top.last() {
        if !better_cand(cand.prob, &cand.balls_0based, cand.bonus_0based, worst) {
            return;
        }
    }
    top.pop();
    let pos = top.partition_point(|x| better_cand(x.prob, &x.balls_0based, x.bonus_0based, &cand));
    top.insert(pos, cand);
}

/// Date pre-calculate pe un expert activ `m >= 1` pentru evaluarea rapidă a `P_port(σ, b)`.
struct ActiveExpertEval<'a> {
    w_m: f64,
    pot: ExpertPotential<'a>,
    /// Pentru fiecare extragere din pasul anterior `d_idx`: `(mask_u128, inv_m_prev_div_z_main, bonus_probs_if_any)`
    conds: Vec<(u128, f64, Vec<f64>)>,
}

fn build_active_evals<'a>(
    space: &'a KjV5Space,
    ops: &'a KjV5StepOperators,
    z_packs: &'a [Vec<super::partition::SingleConditionPartition>; NUM_EXPERTS],
    w_portfolio: &[f64; NUM_EXPERTS],
) -> Vec<ActiveExpertEval<'a>> {
    let mut out = Vec::new();
    let m_prev = ops.prev_draws.len();
    if m_prev == 0 {
        return out;
    }
    let inv_mp = 1.0 / (m_prev as f64);
    for m in 1..NUM_EXPERTS {
        let wm = w_portfolio[m];
        if wm <= 1e-18 {
            continue;
        }
        let pot = ops.expert_potential(space, m);
        let mut conds = Vec::with_capacity(m_prev);
        for (d_idx, (d_balls, _)) in ops.prev_draws.iter().enumerate() {
            let mut mask = 0u128;
            for &b in d_balls {
                mask |= 1u128 << b;
            }
            let zp = &z_packs[m][d_idx];
            conds.push((mask, inv_mp / zp.z_main, zp.pi_bonus.clone()));
        }
        out.push(ActiveExpertEval { w_m: wm, pot, conds });
    }
    out
}

#[inline]
fn eval_portfolio_prob_main_and_best_bonus(
    space: &KjV5Space,
    w0: f64,
    active: &[ActiveExpertEval<'_>],
    sigma_0based: &[usize],
) -> (f64, Option<usize>) {
    let unif_joint = 1.0 / (space.total_joint_states as f64);
    if active.is_empty() {
        return (unif_joint, space.m_bonus.map(|_| 0));
    }
    if let Some(mb) = space.m_bonus {
        let mut bonus_tot = vec![w0 * unif_joint; mb];
        for ex in active {
            let mut sp = 0.0f64;
            let mut sm = 0.0f64;
            let mut lin = 0.0f64;
            let mut sig_mask = 0u128;
            for &i in sigma_0based {
                sig_mask |= 1u128 << i;
                lin += ex.pot.h_anti[i];
                sp += ex.pot.modes.u_pos[i];
                sm += ex.pot.modes.u_neg[i];
            }
            let base_exp = (ex.pot.eta * lin
                + if ex.pot.gamma_2 != 0.0 {
                    0.5 * ex.pot.gamma_2 * (ex.pot.modes.lambda_pos * sp * sp + ex.pot.modes.lambda_neg * sm * sm)
                } else {
                    0.0
                })
            .exp();
            for (d_mask, scale_main, b_probs) in &ex.conds {
                let u = (sig_mask & *d_mask).count_ones() as usize;
                let sh = if ex.pot.gamma_sh != 0.0 && u < ex.pot.theta_shell.len() {
                    (ex.pot.gamma_sh * ex.pot.theta_shell[u]).exp()
                } else {
                    1.0
                };
                let p_main = ex.w_m * scale_main * base_exp * sh;
                for (b, slot) in bonus_tot.iter_mut().enumerate() {
                    *slot += p_main * b_probs[b];
                }
            }
        }
        let mut best_b = 0usize;
        let mut best_p = bonus_tot[0];
        for (b, &p) in bonus_tot.iter().enumerate().skip(1) {
            if p > best_p + 1e-15 * best_p.max(1e-300) {
                best_p = p;
                best_b = b;
            }
        }
        (best_p, Some(best_b))
    } else {
        let mut p_tot = w0 * unif_joint;
        let mut sig_mask = 0u128;
        for &i in sigma_0based {
            sig_mask |= 1u128 << i;
        }
        for ex in active {
            let mut sp = 0.0f64;
            let mut sm = 0.0f64;
            let mut lin = 0.0f64;
            for &i in sigma_0based {
                lin += ex.pot.h_anti[i];
                sp += ex.pot.modes.u_pos[i];
                sm += ex.pot.modes.u_neg[i];
            }
            let base_exp = (ex.pot.eta * lin
                + if ex.pot.gamma_2 != 0.0 {
                    0.5 * ex.pot.gamma_2 * (ex.pot.modes.lambda_pos * sp * sp + ex.pot.modes.lambda_neg * sm * sm)
                } else {
                    0.0
                })
            .exp();
            let mut p_ex = 0.0f64;
            for &(d_mask, scale_main, _) in &ex.conds {
                let u = (sig_mask & d_mask).count_ones() as usize;
                let sh = if ex.pot.gamma_sh != 0.0 && u < ex.pot.theta_shell.len() {
                    (ex.pot.gamma_sh * ex.pot.theta_shell[u]).exp()
                } else {
                    1.0
                };
                p_ex += scale_main * base_exp * sh;
            }
            p_tot += ex.w_m * p_ex;
        }
        (p_tot, None)
    }
}

/// Construiește multipletul repulsiv Coulomb–Jastrow `(α, β, γ)` din bazinul celor mai bune stări.
fn build_multiplet_from_candidates(space: &KjV5Space, top: &[Candidate], exact_certified: bool) -> PortfolioFullEval {
    assert!(!top.is_empty());
    let alpha = &top[0];
    let runner_up_gap = if top.len() >= 2 { alpha.prob.ln() - top[1].prob.ln() } else { 0.0 };

    let denom_ov = (space.k + usize::from(space.m_bonus.is_some())) as f64;
    let overlap_g = |a: &Candidate, b: &Candidate| -> f64 {
        let mut inter = 0usize;
        for &x in &a.balls_0based {
            if b.balls_0based.contains(&x) {
                inter += 1;
            }
        }
        if space.m_bonus.is_some() && a.bonus_0based == b.bonus_0based {
            inter += 1;
        }
        (inter as f64) / denom_ov
    };

    let mut best_score = f64::NEG_INFINITY;
    let mut best_pair = (1.min(top.len() - 1), 2.min(top.len() - 1));
    let mut best_det = 0.0f64;
    let mut best_g = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let limit = top.len().min(192);
    let lp_a = alpha.prob.max(1e-300).ln();
    for ib in 1..limit {
        let gab = overlap_g(alpha, &top[ib]);
        let lp_b = top[ib].prob.max(1e-300).ln();
        for ic in (ib + 1)..limit {
            let gac = overlap_g(alpha, &top[ic]);
            let gbc = overlap_g(&top[ib], &top[ic]);
            let det = 1.0 + 2.0 * gab * gbc * gac - (gab * gab + gbc * gbc + gac * gac);
            if det <= 1e-9 {
                continue;
            }
            let score = lp_a + lp_b + top[ic].prob.max(1e-300).ln() + 0.50 * det.ln();
            if score > best_score {
                best_score = score;
                best_pair = (ib, ic);
                best_det = det;
                best_g = [[1.0, gab, gac], [gab, 1.0, gbc], [gac, gbc, 1.0]];
            }
        }
    }

    let chosen = [alpha, &top[best_pair.0], &top[best_pair.1]];
    let labels = ["alpha", "beta", "gamma"];
    let log_u = -(space.total_joint_states as f64).ln();

    let variants: Vec<MultipletVariant> = chosen
        .iter()
        .zip(labels.iter())
        .map(|(c, &lbl)| {
            let lp = c.prob.max(1e-300).ln();
            MultipletVariant {
                label: lbl,
                ticket_1based: c.balls_0based.iter().map(|&x| x + 1).collect(),
                bonus_1based: c.bonus_0based.map(|b| b + 1),
                log_prob: lp,
                log_lift_vs_uniform: lp - log_u,
            }
        })
        .collect();

    // Nucleul determinist: bile prezente în >= 2 variante ale multipletului; dacă variantele sunt disjuncte
    // (`det G_3 = 1`), primele 3 bile din reuniunea multipletului în ordinea probabilității variantelor.
    let mut counts = vec![0usize; space.n];
    for c in &chosen {
        for &b in &c.balls_0based {
            counts[b] += 1;
        }
    }
    let mut core_1based: Vec<usize> = (0..space.n).filter(|&b| counts[b] >= 2).map(|b| b + 1).collect();
    if core_1based.is_empty() {
        let mut union_balls: Vec<usize> = alpha.balls_0based.iter().take(3.min(space.k)).map(|&b| b + 1).collect();
        union_balls.sort_unstable();
        core_1based = union_balls;
    }

    PortfolioFullEval {
        sigma_alpha_1based: variants[0].ticket_1based.clone(),
        bonus_alpha_1based: variants[0].bonus_1based,
        alpha_log_prob: variants[0].log_prob,
        runner_up_gap,
        variants,
        gram_matrix_3x3: best_g,
        det_g3: best_det,
        deterministic_core_1based: core_1based,
        exact_global_certified: exact_certified,
    }
}

/// Enumerare paralelă completă pe toate firele CPU peste TOATE cele `C(N, k)` (și `× M_B`) stări —
/// folosită în calificarea `Q1` și la starea terminală LIVE pentru `649`, `540`, `joker`.
pub fn evaluate_portfolio_exact_full_enum(
    space: &KjV5Space,
    ops: &KjV5StepOperators,
    z_packs: &[Vec<super::partition::SingleConditionPartition>; NUM_EXPERTS],
    w_portfolio: &[f64; NUM_EXPERTS],
) -> PortfolioFullEval {
    let active = build_active_evals(space, ops, z_packs, w_portfolio);
    let w0 = w_portfolio[0];
    let n = space.n;
    let k = space.k;
    let first_choices: Vec<usize> = (0..=(n - k)).collect();
    let n_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8).min(first_choices.len().max(1));

    let merged_top = std::thread::scope(|s| {
        let active_ref = &active;
        let first_ref = &first_choices;
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            handles.push(s.spawn(move || {
                let mut local_top: Vec<Candidate> = Vec::with_capacity(192);
                let mut comb = vec![0usize; k];
                for (idx, &b0) in first_ref.iter().enumerate() {
                    if idx % n_threads != tid {
                        continue;
                    }
                    comb[0] = b0;
                    if k == 1 {
                        let (p, jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, active_ref, &comb);
                        push_top_k(&mut local_top, Candidate { prob: p, balls_0based: comb.clone(), bonus_0based: jb }, 192);
                    } else {
                        enum_port_rec(space, w0, active_ref, n, k, 1, b0 + 1, &mut comb, &mut local_top);
                    }
                }
                local_top
            }));
        }
        let mut global_top: Vec<Candidate> = Vec::with_capacity(256);
        for h in handles {
            for cand in h.join().expect("firul de enumerare portofoliu a eșuat") {
                push_top_k(&mut global_top, cand, 224);
            }
        }
        global_top
    });

    let mut sorted_top = merged_top;
    sorted_top.sort_by(|a, b| {
        if better_cand(a.prob, &a.balls_0based, a.bonus_0based, b) {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });
    build_multiplet_from_candidates(space, &sorted_top, true)
}

#[allow(clippy::too_many_arguments)]
fn enum_port_rec(
    space: &KjV5Space,
    w0: f64,
    active: &[ActiveExpertEval<'_>],
    n: usize,
    k: usize,
    depth: usize,
    start: usize,
    comb: &mut Vec<usize>,
    local_top: &mut Vec<Candidate>,
) {
    let rem = k - depth;
    if rem == 1 {
        let worst_p = if local_top.len() >= 192 { local_top.last().map_or(0.0, |c| c.prob) } else { -1.0 };
        for last in start..n {
            comb[depth] = last;
            let (p, jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, active, comb);
            if p >= worst_p - 1e-15 {
                push_top_k(local_top, Candidate { prob: p, balls_0based: comb.clone(), bonus_0based: jb }, 192);
            }
        }
        return;
    }
    let max_i = n - rem;
    for i in start..=max_i {
        comb[depth] = i;
        enum_port_rec(space, w0, active, n, k, depth + 1, i + 1, comb, local_top);
    }
}

/// Găsește exact biletul fundamental `σ_α` (și `bonus_α`) folosind:
/// 1. Branch-and-Bound pe intervale stratificat după învelișul `u = |σ ∩ D'|` când `m_{s-1} == 1` și `r <= 8`
///    (100% certificat global exact — verificat în `Q1` față de enumerarea brută completă!);
/// 2. Enumerare completă când spațiul `<= 2.500.000` și `m_{s-1} > 1`;
/// 3. Bazinul exact al primelor bile după marginalele `π_port` când spațiul depășește `10^8` (`Pick 10` `10/80`).
pub fn find_sigma_alpha_fast(
    space: &KjV5Space,
    ops: &KjV5StepOperators,
    z_packs: &[Vec<super::partition::SingleConditionPartition>; NUM_EXPERTS],
    w_portfolio: &[f64; NUM_EXPERTS],
    pi_port_main: &[f64],
) -> (Vec<usize>, Option<usize>, f64, bool) {
    let active = build_active_evals(space, ops, z_packs, w_portfolio);
    let w0 = w_portfolio[0];
    let unif_joint = 1.0 / (space.total_joint_states as f64);

    if active.is_empty() {
        let tkt: Vec<usize> = (1..=space.k).collect();
        return (tkt, space.m_bonus.map(|_| 1), unif_joint.ln(), true);
    }

    // Caz Pick 10 (N=80, k=10, C(80,10) = 1.646e12): bazinul primelor 16 bile ordonate după π_port (C(16,10) = 8008)
    if space.total_base_states > 80_000_000 || space.r > 8 {
        let pool_m = (space.k + 6).min(space.n).min(17);
        let mut order: Vec<usize> = (0..space.n).collect();
        order.sort_by(|&a, &b| pi_port_main[b].total_cmp(&pi_port_main[a]).then_with(|| a.cmp(&b)));
        let mut pool = order[..pool_m].to_vec();
        pool.sort_unstable();

        let mut best = Candidate {
            prob: -1.0,
            balls_0based: pool[..space.k].to_vec(),
            bonus_0based: space.m_bonus.map(|_| 0),
        };
        let mut idx_comb: Vec<usize> = (0..space.k).collect();
        loop {
            let balls: Vec<usize> = idx_comb.iter().map(|&ix| pool[ix]).collect();
            let (p, jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, &active, &balls);
            if better_cand(p, &balls, jb, &best) {
                best = Candidate { prob: p, balls_0based: balls, bonus_0based: jb };
            }
            let mut pos = space.k;
            while pos > 0 && idx_comb[pos - 1] == pool_m - space.k + (pos - 1) {
                pos -= 1;
            }
            if pos == 0 {
                break;
            }
            idx_comb[pos - 1] += 1;
            for j in pos..space.k {
                idx_comb[j] = idx_comb[j - 1] + 1;
            }
        }
        let t1: Vec<usize> = best.balls_0based.iter().map(|&x| x + 1).collect();
        return (t1, best.bonus_0based.map(|b| b + 1), best.prob.max(1e-300).ln(), false);
    }

    // Dacă pasul anterior are extrageri multiple și spațiul e moderat, enumerare completă
    if ops.prev_draws.len() != 1 {
        let full = evaluate_portfolio_exact_full_enum(space, ops, z_packs, w_portfolio);
        return (full.sigma_alpha_1based, full.bonus_alpha_1based, full.alpha_log_prob, true);
    }

    // Exact Shell-Stratified Interval Branch-and-Bound (100% certificat global exact!)
    // 1. Inițializăm cota inferioară `best` evaluând combinațiile din primele `min(k+6, 13)` bile după `π_port`.
    let seed_m = (space.k + 6).min(space.n).min(13);
    let mut seed_ord: Vec<usize> = (0..space.n).collect();
    seed_ord.sort_by(|&a, &b| pi_port_main[b].total_cmp(&pi_port_main[a]).then_with(|| a.cmp(&b)));
    let mut seed_pool = seed_ord[..seed_m].to_vec();
    seed_pool.sort_unstable();

    let (init_p, init_jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, &active, &seed_pool[..space.k]);
    let mut best = Candidate { prob: init_p, balls_0based: seed_pool[..space.k].to_vec(), bonus_0based: init_jb };

    let mut idx_c: Vec<usize> = (0..space.k).collect();
    loop {
        let balls: Vec<usize> = idx_c.iter().map(|&ix| seed_pool[ix]).collect();
        let (p, jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, &active, &balls);
        if better_cand(p, &balls, jb, &best) {
            best = Candidate { prob: p, balls_0based: balls, bonus_0based: jb };
        }
        let mut pos = space.k;
        while pos > 0 && idx_c[pos - 1] == seed_m - space.k + (pos - 1) {
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

    // 2. Partiționăm urna în D' (`r` bile) și D'^c (`N - r` bile), ordonând D'^c descrescător după `π_port`
    let prev_balls = &ops.prev_draws[0].0;
    let mut in_d = vec![false; space.n];
    for &b in prev_balls {
        in_d[b] = true;
    }
    let mut dc: Vec<usize> = (0..space.n).filter(|&i| !in_d[i]).collect();
    dc.sort_by(|&a, &b| pi_port_main[b].total_cmp(&pi_port_main[a]).then_with(|| a.cmp(&b)));
    let m_dc = dc.len();
    let max_t = space.k.min(m_dc);

    // Pre-calculăm cotele pe sufix `idx..m_dc` pentru fiecare expert activ și fiecare `t ∈ 0..=max_t`
    struct ExBound {
        w_scale_max_bonus: f64,
        eta: f64,
        half_g2_lp: f64,
        half_g2_lm: f64,
        shell_mult: Vec<f64>,
        h_dc: Vec<f64>,
        up_dc: Vec<f64>,
        um_dc: Vec<f64>,
        suf_max_lin: Vec<Vec<f64>>,
        suf_max_up: Vec<Vec<f64>>,
        suf_min_up: Vec<Vec<f64>>,
        suf_max_um: Vec<Vec<f64>>,
        suf_min_um: Vec<Vec<f64>>,
    }

    let mut bounds = Vec::with_capacity(active.len());
    for ex in &active {
        let (_, scale_main, ref b_probs) = ex.conds[0];
        let max_b_prob = if b_probs.is_empty() {
            1.0
        } else {
            b_probs.iter().copied().fold(0.0f64, f64::max)
        };
        let u_max = space.k.min(space.r);
        let mut sh_m = vec![1.0f64; u_max + 1];
        for (u, slot) in sh_m.iter_mut().enumerate() {
            if ex.pot.gamma_sh != 0.0 && u < ex.pot.theta_shell.len() {
                *slot = (ex.pot.gamma_sh * ex.pot.theta_shell[u]).exp();
            }
        }
        let h_dc: Vec<f64> = dc.iter().map(|&i| ex.pot.eta * ex.pot.h_anti[i]).collect();
        let up_dc: Vec<f64> = dc.iter().map(|&i| ex.pot.modes.u_pos[i]).collect();
        let um_dc: Vec<f64> = dc.iter().map(|&i| ex.pot.modes.u_neg[i]).collect();

        let build_top_k_sums = |arr: &[f64], want_max: bool| -> Vec<Vec<f64>> {
            let mut tbl = vec![vec![0.0f64; max_t + 1]; m_dc + 1];
            let mut buf = Vec::with_capacity(m_dc);
            for idx in (0..m_dc).rev() {
                buf.clear();
                buf.extend_from_slice(&arr[idx..]);
                if want_max {
                    buf.sort_by(|a, b| b.total_cmp(a));
                } else {
                    buf.sort_by(|a, b| a.total_cmp(b));
                }
                let mut s = 0.0f64;
                for t in 1..=max_t.min(buf.len()) {
                    s += buf[t - 1];
                    tbl[idx][t] = s;
                }
            }
            tbl
        };

        bounds.push(ExBound {
            w_scale_max_bonus: ex.w_m * scale_main * max_b_prob,
            eta: ex.pot.eta,
            half_g2_lp: 0.5 * ex.pot.gamma_2 * ex.pot.modes.lambda_pos,
            half_g2_lm: 0.5 * ex.pot.gamma_2 * ex.pot.modes.lambda_neg,
            shell_mult: sh_m,
            suf_max_lin: build_top_k_sums(&h_dc, true),
            suf_max_up: build_top_k_sums(&up_dc, true),
            suf_min_up: build_top_k_sums(&up_dc, false),
            suf_max_um: build_top_k_sums(&um_dc, true),
            suf_min_um: build_top_k_sums(&um_dc, false),
            h_dc,
            up_dc,
            um_dc,
        });
    }

    let n_ex = bounds.len();
    let u_min = (space.k + space.r).saturating_sub(space.n);
    let u_max = space.k.min(space.r);

    // 3. Parcurgem fiecare înveliș `u ∈ u_min..=u_max` și fiecare submulțime `S_D ⊂ D'` de mărime `u` (`<= 35` cazuri!)
    for u in u_min..=u_max {
        let t_rem = space.k - u;
        let mut d_subsets: Vec<Vec<usize>> = Vec::new();
        gen_subsets(prev_balls, u, 0, &mut Vec::new(), &mut d_subsets);

        for sub_d in &d_subsets {
            let mut cur_lin = vec![0.0f64; n_ex];
            let mut cur_sp = vec![0.0f64; n_ex];
            let mut cur_sm = vec![0.0f64; n_ex];
            for (ei, ex) in active.iter().enumerate() {
                for &b in sub_d {
                    cur_lin[ei] += bounds[ei].eta * ex.pot.h_anti[b];
                    cur_sp[ei] += ex.pot.modes.u_pos[b];
                    cur_sm[ei] += ex.pot.modes.u_neg[b];
                }
            }
            if t_rem == 0 {
                let mut full_b = sub_d.clone();
                full_b.sort_unstable();
                let (p, jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, &active, &full_b);
                if better_cand(p, &full_b, jb, &best) {
                    best = Candidate { prob: p, balls_0based: full_b, bonus_0based: jb };
                }
                continue;
            }
            let mut picked_dc = Vec::with_capacity(t_rem);
            #[allow(clippy::too_many_arguments)]
            fn dfs_bnb(
                space: &KjV5Space,
                w0: f64,
                unif_joint: f64,
                active: &[ActiveExpertEval<'_>],
                bounds: &[ExBound],
                dc: &[usize],
                u: usize,
                sub_d: &[usize],
                idx: usize,
                rem: usize,
                cur_lin: &mut [f64],
                cur_sp: &mut [f64],
                cur_sm: &mut [f64],
                picked_dc: &mut Vec<usize>,
                best: &mut Candidate,
            ) {
                if dc.len() - idx < rem {
                    return;
                }
                // Calculăm cota superioară exactă pe întregul subarbore `idx..`
                let mut ub = w0 * unif_joint;
                for (ei, eb) in bounds.iter().enumerate() {
                    let max_l = cur_lin[ei] + eb.suf_max_lin[idx][rem];
                    let max_q_pos = if eb.half_g2_lp > 0.0 {
                        let s_hi = cur_sp[ei] + eb.suf_max_up[idx][rem];
                        let s_lo = cur_sp[ei] + eb.suf_min_up[idx][rem];
                        eb.half_g2_lp * (s_hi * s_hi).max(s_lo * s_lo)
                    } else {
                        0.0
                    };
                    let max_q_neg = if eb.half_g2_lm < 0.0 {
                        let lo = cur_sm[ei] + eb.suf_min_um[idx][rem];
                        let hi = cur_sm[ei] + eb.suf_max_um[idx][rem];
                        let min_sq = if lo <= 0.0 && 0.0 <= hi { 0.0 } else { (lo * lo).min(hi * hi) };
                        eb.half_g2_lm * min_sq
                    } else {
                        0.0
                    };
                    ub += eb.w_scale_max_bonus * eb.shell_mult[u] * (max_l + max_q_pos + max_q_neg).exp();
                }
                // Tăiere sigură cu marjă de 1e-12 față de cel mai bun candidat curent
                if ub < best.prob * (1.0 - 1e-12) {
                    return;
                }

                if rem == 1 {
                    for pos in idx..dc.len() {
                        // Evaluare rapidă cotă pe frunză înainte de construcția vectorului sortat
                        let mut leaf_ub = w0 * unif_joint;
                        for (ei, eb) in bounds.iter().enumerate() {
                            let l = cur_lin[ei] + eb.h_dc[pos];
                            let sp = cur_sp[ei] + eb.up_dc[pos];
                            let sm = cur_sm[ei] + eb.um_dc[pos];
                            let qp = eb.half_g2_lp * sp * sp;
                            let qm = eb.half_g2_lm * sm * sm;
                            leaf_ub += eb.w_scale_max_bonus * eb.shell_mult[u] * (l + qp + qm).exp();
                        }
                        if leaf_ub >= best.prob * (1.0 - 1e-12) {
                            let mut full_b = Vec::with_capacity(space.k);
                            full_b.extend_from_slice(sub_d);
                            full_b.extend_from_slice(picked_dc);
                            full_b.push(dc[pos]);
                            full_b.sort_unstable();
                            let (p, jb) = eval_portfolio_prob_main_and_best_bonus(space, w0, active, &full_b);
                            if better_cand(p, &full_b, jb, best) {
                                *best = Candidate { prob: p, balls_0based: full_b, bonus_0based: jb };
                            }
                        }
                    }
                    return;
                }

                let max_pos = dc.len() - rem;
                for pos in idx..=max_pos {
                    picked_dc.push(dc[pos]);
                    for (ei, eb) in bounds.iter().enumerate() {
                        cur_lin[ei] += eb.h_dc[pos];
                        cur_sp[ei] += eb.up_dc[pos];
                        cur_sm[ei] += eb.um_dc[pos];
                    }
                    dfs_bnb(
                        space, w0, unif_joint, active, bounds, dc, u, sub_d, pos + 1, rem - 1, cur_lin, cur_sp, cur_sm,
                        picked_dc, best,
                    );
                    for (ei, eb) in bounds.iter().enumerate() {
                        cur_lin[ei] -= eb.h_dc[pos];
                        cur_sp[ei] -= eb.up_dc[pos];
                        cur_sm[ei] -= eb.um_dc[pos];
                    }
                    picked_dc.pop();
                }
            }

            dfs_bnb(
                space,
                w0,
                unif_joint,
                &active,
                &bounds,
                &dc,
                u,
                sub_d,
                0,
                t_rem,
                &mut cur_lin,
                &mut cur_sp,
                &mut cur_sm,
                &mut picked_dc,
                &mut best,
            );
        }
    }

    let t1: Vec<usize> = best.balls_0based.iter().map(|&x| x + 1).collect();
    (t1, best.bonus_0based.map(|b| b + 1), best.prob.max(1e-300).ln(), true)
}

fn gen_subsets(arr: &[usize], choose: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if cur.len() == choose {
        out.push(cur.clone());
        return;
    }
    if arr.len() - start < choose - cur.len() {
        return;
    }
    for i in start..arr.len() {
        cur.push(arr[i]);
        gen_subsets(arr, choose, i + 1, cur, out);
        cur.pop();
    }
}

/// Legea Poisson condiționată exactă (`ln e_r(w)` și marginalele `q_i = P(i ∈ D)`, `∑ q_i == r`)
/// pentru comparatoarele canonice `B` și `R` (aliniată identic cu `hs_v4/walkforward.py`).
pub fn cond_poisson_log_z_and_marginals(weights: &[f64], r: usize) -> (f64, Vec<f64>) {
    let n = weights.len();
    let u: Vec<f64> = weights.iter().map(|&w| w * (n as f64)).collect();
    let stride = r + 1;
    let mut f = vec![0.0f64; (n + 1) * stride];
    f[0] = 1.0;
    for i in 0..n {
        f[(i + 1) * stride] = 1.0;
        let ui = u[i];
        for m in 1..=r.min(i + 1) {
            f[(i + 1) * stride + m] = f[i * stride + m] + ui * f[i * stride + m - 1];
        }
    }
    let mut g = vec![0.0f64; (n + 1) * stride];
    g[n * stride] = 1.0;
    for i in (0..n).rev() {
        g[i * stride] = 1.0;
        let ui = u[i];
        for m in 1..=r.min(n - i) {
            g[i * stride + m] = g[(i + 1) * stride + m] + ui * g[(i + 1) * stride + m - 1];
        }
    }
    let er_u = f[n * stride + r].max(1e-300);
    let log_er_w = er_u.ln() - (r as f64) * (n as f64).ln();
    let mut q_inc = vec![0.0f64; n];
    for i in 0..n {
        let mut e_minus = 0.0f64;
        for m in 0..r {
            e_minus += f[i * stride + m] * g[(i + 1) * stride + (r - 1 - m)];
        }
        q_inc[i] = u[i] * e_minus / er_u;
    }
    (log_er_w, q_inc)
}

/// Primele `k` bile 1-based ordonate crescător după ponderi (departajare deterministă crescătoare pe indice).
pub fn top_k_1based(weights: &[f64], k: usize) -> Vec<usize> {
    let mut ord: Vec<usize> = (0..weights.len()).collect();
    ord.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then_with(|| a.cmp(&b)));
    let mut chosen: Vec<usize> = ord[..k].iter().map(|&i| i + 1).collect();
    chosen.sort_unstable();
    chosen
}

/// Media hipergeometrică exactă `r * k / N` a potrivirilor sub `U`.
#[inline]
pub fn expected_hits_uniform(n: usize, r: usize, k: usize) -> f64 {
    (r * k) as f64 / (n as f64)
}

/// Conversia exactă a marginalelor de bilet `π_i` (`∑ π_i = k`) în marginale de extragere `q_i` (`∑ q_i = r`).
pub fn ticket_to_draw_marginals(pi_tkt: &[f64], n: usize, r: usize, k: usize) -> Vec<f64> {
    if r == k {
        return pi_tkt.to_vec();
    }
    let num_add = (r - k) as f64;
    let mul = (n - r) as f64;
    let den = (n - k) as f64;
    pi_tkt.iter().map(|&p| (num_add + mul * p) / den).collect()
}

/// Distribuția hipergeometrică exactă `p_0(h)` pentru `h = 0..=min(r, k)`.
pub fn hypergeom_probs(n: usize, r: usize, k: usize) -> Vec<f64> {
    let denom = binom_f64(n, k);
    let m = r.min(k);
    (0..=m)
        .map(|h| {
            if k - h <= n - r {
                binom_f64(r, h) * binom_f64(n - r, k - h) / denom
            } else {
                0.0
            }
        })
        .collect()
}

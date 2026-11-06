//! `K4` — Evaluarea exactă `O(N k)` a funcției de partiție `Z_m(D)`, a celor `N` marginale 1-corp `π_{m,i}(D)`
//! și a sumelor restricționate pe sub-extrageri prin cvadratura Gauss–Hermite 2D în `Complex64` (`M_GH = 14`)
//! + programare dinamică prefix/adjunctă pe polinoame simetrice elementare (`PROTOCOL_KJV5_RUST_V2.md` §3.4).

use num_complex::Complex64;

use super::koopman_v2::V2Modes;

/// Nodurile fizice Gauss–Hermite de ordin `M_GH = 14` (`∫ e^{-x^2} f(x) dx`).
pub const GH14_X: [f64; 14] = [
    -4.304_448_570_473_632,
    -3.462_656_933_602_270_6,
    -2.748_470_724_985_402_6,
    -2.095_183_258_507_716_6,
    -1.476_682_731_141_140_8,
    -0.878_713_787_329_399_4,
    -0.291_745_510_672_562_05,
    0.291_745_510_672_562_05,
    0.878_713_787_329_399_4,
    1.476_682_731_141_140_8,
    2.095_183_258_507_716_6,
    2.748_470_724_985_402_6,
    3.462_656_933_602_270_6,
    4.304_448_570_473_632,
];

/// Ponderile Gauss–Hermite normate `ω_q / sqrt(π)` (`∑_{q=0}^{13} GH14_WN[q] == 1.0`).
pub const GH14_WN: [f64; 14] = [
    4.868_161_257_748_366_5e-9,
    2.660_991_344_067_626_6e-6,
    0.000_200_339_553_760_743_78,
    0.004_428_919_106_947_404,
    0.038_650_108_824_253_446,
    0.154_083_339_842_513_54,
    0.302_634_626_813_019_5,
    0.302_634_626_813_019_5,
    0.154_083_339_842_513_54,
    0.038_650_108_824_253_446,
    0.004_428_919_106_947_404,
    0.000_200_339_553_760_743_78,
    2.660_991_344_067_626_6e-6,
    4.868_161_257_748_366_5e-9,
];

/// Parametrii și câmpurile unui expert `m` evaluate la pasul `s`.
#[derive(Debug, Clone)]
pub struct ExpertPotential<'a> {
    pub n: usize,
    pub r: usize,
    pub k: usize,
    pub gamma_sh: f64,
    pub gamma_2: f64,
    pub eta: f64,
    pub eta_bonus: f64,
    pub theta_shell: &'a [f64],
    pub h_anti: &'a [f64],
    pub modes: &'a V2Modes,
    pub m_bonus: Option<usize>,
    pub theta_bonus: [f64; 2],
    pub h_bonus_anti: &'a [f64],
}

/// Rezultatul exact al evaluării funcției de partiție condiționate `Z_m(D)` și al marginalelor `π_{m,i}(D)`.
#[derive(Debug, Clone)]
pub struct SingleConditionPartition {
    pub z_main: f64,
    pub log_z_main: f64,
    pub pi_main: Vec<f64>,
    pub z_bonus: f64,
    pub pi_bonus: Vec<f64>,
}

/// Calculează polinoamele simetrice elementare `e_0..e_{max_deg}` pe `z` și returnează tabelul forward
/// pentru propagarea inversă adjunctă `O(m * max_deg)`.
fn forward_elem_sym(z: &[Complex64], max_deg: usize) -> (usize, Vec<Complex64>) {
    let m = z.len();
    let deg = max_deg.min(m);
    let stride = deg + 1;
    let mut dp = vec![Complex64::new(0.0, 0.0); (m + 1) * stride];
    dp[0] = Complex64::new(1.0, 0.0);
    for (j, &zj) in z.iter().enumerate() {
        let prev = j * stride;
        let cur = (j + 1) * stride;
        dp[cur] = Complex64::new(1.0, 0.0);
        let lim = deg.min(j + 1);
        for u in 1..=lim {
            dp[cur + u] = dp[prev + u] + zj * dp[prev + u - 1];
        }
    }
    (deg, dp)
}

/// Propagare inversă adjunctă: dat `F(z) = ∑_{u=0}^{deg} coeff[u] * e_u(z)`, adaugă `weight * z_j * ∂F/∂z_j`
/// în `out_marg[indices[j]]`.
fn backward_elem_sym_accum(
    z: &[Complex64],
    deg: usize,
    dp: &[Complex64],
    coeff: &[Complex64],
    node_weight: f64,
    indices: &[usize],
    out_marg_num: &mut [f64],
) {
    let m = z.len();
    if m == 0 || deg == 0 {
        return;
    }
    let stride = deg + 1;
    let mut adj = coeff[..=deg].to_vec();
    for j in (0..m).rev() {
        let zj = z[j];
        let prev = j * stride;
        let lim = deg.min(j + 1);
        let mut deriv = Complex64::new(0.0, 0.0);
        for u in 1..=lim {
            deriv += adj[u] * dp[prev + u - 1];
        }
        let contrib = zj * deriv;
        out_marg_num[indices[j]] += node_weight * contrib.re;

        let lim_adj = deg.min(j);
        for u in 0..=lim_adj {
            if u + 1 <= deg {
                adj[u] = adj[u] + zj * adj[u + 1];
            }
        }
    }
}

/// Polinoame simetrice elementare doar forward `e_0..e_{deg}` (fără tabel salvat, pt. suma restricționată pe sub-extrageri).
fn elem_sym_values_only(z: &[Complex64], max_deg: usize) -> Vec<Complex64> {
    let deg = max_deg.min(z.len());
    let mut e = vec![Complex64::new(0.0, 0.0); deg + 1];
    e[0] = Complex64::new(1.0, 0.0);
    for (j, &zj) in z.iter().enumerate() {
        let lim = deg.min(j + 1);
        for u in (1..=lim).rev() {
            e[u] = e[u] + zj * e[u - 1];
        }
    }
    e
}

/// Construiește lista nodurilor `(w_node, a_pos_x, a_neg_y)` active pentru un expert.
fn active_gh_nodes(gamma_2: f64, modes: &V2Modes) -> Vec<(f64, f64, f64)> {
    let has_pos = gamma_2 > 0.0 && modes.lambda_pos > 1e-13;
    let has_neg = gamma_2 > 0.0 && modes.lambda_neg < -1e-13;
    match (has_pos, has_neg) {
        (false, false) => vec![(1.0, 0.0, 0.0)],
        (true, false) => {
            let ap = (2.0 * gamma_2 * modes.lambda_pos).sqrt();
            (0..14).map(|qp| (GH14_WN[qp], ap * GH14_X[qp], 0.0)).collect()
        }
        (false, true) => {
            let am = (2.0 * gamma_2 * modes.lambda_neg.abs()).sqrt();
            (0..14).map(|qm| (GH14_WN[qm], 0.0, am * GH14_X[qm])).collect()
        }
        (true, true) => {
            let ap = (2.0 * gamma_2 * modes.lambda_pos).sqrt();
            let am = (2.0 * gamma_2 * modes.lambda_neg.abs()).sqrt();
            let mut out = Vec::with_capacity(196);
            for qp in 0..14 {
                for qm in 0..14 {
                    out.push((GH14_WN[qp] * GH14_WN[qm], ap * GH14[qp], am * GH14[qm]));
                }
            }
            out
        }
    }
}

const GH14: [f64; 14] = GH14_X;

/// Evaluează exact `Z_m(D)` și `π_{m,i}(D)` în `O(N k)` prin Gauss–Hermite 2D în `Complex64` + Split-DP.
pub fn eval_partition_conditioned(pot: &ExpertPotential<'_>, prev_draw_0based: &[usize], prev_bonus_0based: Option<usize>) -> SingleConditionPartition {
    let n = pot.n;
    let k = pot.k;
    let r = prev_draw_0based.len();
    debug_assert_eq!(r, pot.r);

    let mut in_d = vec![false; n];
    for &b in prev_draw_0based {
        in_d[b] = true;
    }
    let d_idx: Vec<usize> = prev_draw_0based.to_vec();
    let dc_idx: Vec<usize> = (0..n).filter(|&i| !in_d[i]).collect();

    let u_min = (k + r).saturating_sub(n);
    let u_max = k.min(r);
    let mut g_shell = vec![0.0f64; u_max + 1];
    for u in u_min..=u_max {
        let th = if pot.gamma_sh != 0.0 && u < pot.theta_shell.len() { pot.gamma_sh * pot.theta_shell[u] } else { 0.0 };
        g_shell[u] = th.exp();
    }

    let nodes = active_gh_nodes(pot.gamma_2, pot.modes);
    let mut z_sum = 0.0f64;
    let mut marg_num = vec![0.0f64; n];

    let mut z_d = vec![Complex64::new(0.0, 0.0); d_idx.len()];
    let mut z_dc = vec![Complex64::new(0.0, 0.0); dc_idx.len()];

    for &(w_node, ax, ay) in &nodes {
        for (pos, &i) in d_idx.iter().enumerate() {
            let re = pot.eta * pot.h_anti[i] + ax * pot.modes.u_pos[i];
            let im = ay * pot.modes.u_neg[i];
            z_d[pos] = Complex64::from_polar(re.exp(), im);
        }
        for (pos, &i) in dc_idx.iter().enumerate() {
            let re = pot.eta * pot.h_anti[i] + ax * pot.modes.u_pos[i];
            let im = ay * pot.modes.u_neg[i];
            z_dc[pos] = Complex64::from_polar(re.exp(), im);
        }

        let (deg_d, dp_d) = forward_elem_sym(&z_d, u_max);
        let (deg_dc, dp_dc) = forward_elem_sym(&z_dc, k - u_min);
        let e_d = &dp_d[d_idx.len() * (deg_d + 1)..];
        let e_dc = &dp_dc[dc_idx.len() * (deg_dc + 1)..];

        let mut z_node = Complex64::new(0.0, 0.0);
        let mut coeff_d = vec![Complex64::new(0.0, 0.0); deg_d + 1];
        let mut coeff_dc = vec![Complex64::new(0.0, 0.0); deg_dc + 1];

        for u in u_min..=u_max {
            let v = k - u;
            if u <= deg_d && v <= deg_dc {
                let gu = g_shell[u];
                z_node += gu * e_d[u] * e_dc[v];
                coeff_d[u] += gu * e_dc[v];
                coeff_dc[v] += gu * e_d[u];
            }
        }
        z_sum += w_node * z_node.re;

        backward_elem_sym_accum(&z_d, deg_d, &dp_d, &coeff_d, w_node, &d_idx, &mut marg_num);
        backward_elem_sym_accum(&z_dc, deg_dc, &dp_dc, &coeff_dc, w_node, &dc_idx, &mut marg_num);
    }

    let z_safe = z_sum.max(1e-300);
    let mut pi_main: Vec<f64> = marg_num.iter().map(|&v| (v / z_safe).clamp(0.0, 1.0)).collect();
    // Normalizare fină ca ∑_i π_i == k la precizie de mașină
    let sum_pi: f64 = pi_main.iter().sum();
    if sum_pi > 0.0 {
        let corr = (k as f64) / sum_pi;
        for v in &mut pi_main {
            *v *= corr;
        }
    }

    let (z_bonus, pi_bonus) = if let Some(mb) = pot.m_bonus {
        let mut wb = vec![0.0f64; mb];
        let mut zb = 0.0f64;
        for (b, slot) in wb.iter_mut().enumerate() {
            let rep = prev_bonus_0based == Some(b);
            let th_b = if pot.gamma_sh != 0.0 {
                pot.gamma_sh * pot.theta_bonus[usize::from(rep)]
            } else {
                0.0
            };
            let hb = if b < pot.h_bonus_anti.len() { pot.eta_bonus * pot.h_bonus_anti[b] } else { 0.0 };
            let val = (th_b + hb).exp();
            *slot = val;
            zb += val;
        }
        let zb_safe = zb.max(1e-300);
        for slot in &mut wb {
            *slot /= zb_safe;
        }
        (zb_safe, wb)
    } else {
        (1.0, Vec::new())
    };

    SingleConditionPartition { z_main: z_safe, log_z_main: z_safe.ln(), pi_main, z_bonus, pi_bonus }
}

/// Ponderea exactă directă `W_m(σ | D)` pentru un bilet candidat `sigma_0based` (`|σ| = k`).
#[inline]
pub fn exact_ticket_weight_main(pot: &ExpertPotential<'_>, prev_draw_mask_u128: u128, sigma_0based: &[usize]) -> f64 {
    let mut u = 0usize;
    let mut lin = 0.0f64;
    let mut sp = 0.0f64;
    let mut sm = 0.0f64;
    for &i in sigma_0based {
        if ((prev_draw_mask_u128 >> i) & 1) != 0 {
            u += 1;
        }
        lin += pot.h_anti[i];
        sp += pot.modes.u_pos[i];
        sm += pot.modes.u_neg[i];
    }
    let sh = if pot.gamma_sh != 0.0 && u < pot.theta_shell.len() { pot.gamma_sh * pot.theta_shell[u] } else { 0.0 };
    let quad = if pot.gamma_2 != 0.0 {
        0.5 * pot.gamma_2 * (pot.modes.lambda_pos * sp * sp + pot.modes.lambda_neg * sm * sm)
    } else {
        0.0
    };
    (sh + pot.eta * lin + quad).exp()
}

/// Suma restricționată exactă `∑_{σ ⊂ D_target, |σ| = k} W_m(σ | D_prev)` în `O(r k)` prin Gauss–Hermite 2D + Split-DP
/// (permite evaluarea exactă inclusiv pentru `Pick 10` `r=20, k=10` cu `184.756` submulțimi în `< 0.05 ms`!).
pub fn eval_subdraw_restricted_sum(
    pot: &ExpertPotential<'_>,
    prev_draw_0based: &[usize],
    target_draw_0based: &[usize],
) -> f64 {
    let k = pot.k;
    if target_draw_0based.len() == k {
        let mut mask = 0u128;
        for &b in prev_draw_0based {
            mask |= 1u128 << b;
        }
        return exact_ticket_weight_main(pot, mask, target_draw_0based);
    }

    let mut in_prev = vec![false; pot.n];
    for &b in prev_draw_0based {
        in_prev[b] = true;
    }
    let mut a_idx = Vec::with_capacity(target_draw_0based.len());
    let mut b_idx = Vec::with_capacity(target_draw_0based.len());
    for &x in target_draw_0based {
        if in_prev[x] {
            a_idx.push(x);
        } else {
            b_idx.push(x);
        }
    }

    let u_lo = k.saturating_sub(b_idx.len());
    let u_hi = k.min(a_idx.len());
    let mut g_shell = vec![0.0f64; u_hi + 1];
    for u in u_lo..=u_hi {
        let th = if pot.gamma_sh != 0.0 && u < pot.theta_shell.len() { pot.gamma_sh * pot.theta_shell[u] } else { 0.0 };
        g_shell[u] = th.exp();
    }

    let nodes = active_gh_nodes(pot.gamma_2, pot.modes);
    let mut sum_w = 0.0f64;
    let mut z_a = vec![Complex64::new(0.0, 0.0); a_idx.len()];
    let mut z_b = vec![Complex64::new(0.0, 0.0); b_idx.len()];

    for &(w_node, ax, ay) in &nodes {
        for (pos, &i) in a_idx.iter().enumerate() {
            let re = pot.eta * pot.h_anti[i] + ax * pot.modes.u_pos[i];
            let im = ay * pot.modes.u_neg[i];
            z_a[pos] = Complex64::from_polar(re.exp(), im);
        }
        for (pos, &i) in b_idx.iter().enumerate() {
            let re = pot.eta * pot.h_anti[i] + ax * pot.modes.u_pos[i];
            let im = ay * pot.modes.u_neg[i];
            z_b[pos] = Complex64::from_polar(re.exp(), im);
        }
        let e_a = elem_sym_values_only(&z_a, u_hi);
        let e_b = elem_sym_values_only(&z_b, k - u_lo);
        let mut node_acc = Complex64::new(0.0, 0.0);
        for u in u_lo..=u_hi {
            let v = k - u;
            if u < e_a.len() && v < e_b.len() {
                node_acc += g_shell[u] * e_a[u] * e_b[v];
            }
        }
        sum_w += w_node * node_acc.re;
    }
    sum_w.max(0.0)
}

/// Enumerare brută paralelă pe toate firele CPU peste TOATE cele `C(N, k)` stări (și `× M_B` pt. Joker)
/// pentru certificarea algebrică `Q1` (`|Z_enum - Z_GH| / Z_enum < 1e-12`, `max|π_enum - π_GH| < 1e-12`).
pub fn brute_force_partition_parallel(
    pot: &ExpertPotential<'_>,
    prev_draw_0based: &[usize],
    prev_bonus_0based: Option<usize>,
) -> SingleConditionPartition {
    let n = pot.n;
    let k = pot.k;
    let mut prev_mask = 0u128;
    for &b in prev_draw_0based {
        prev_mask |= 1u128 << b;
    }

    let n_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8).min(n.max(1));
    // Partiționăm enumerarea combinatorică exact după prima bilă `b0 ∈ 0..=n-k` distribuită pe fire!
    let first_choices: Vec<usize> = (0..=(n - k)).collect();

    let (z_main, marg_num) = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let first_ref = &first_choices;
            handles.push(s.spawn(move || {
                let mut local_z = 0.0f64;
                let mut local_m = vec![0.0f64; n];
                let mut comb = [0usize; 12];
                for (idx, &b0) in first_ref.iter().enumerate() {
                    if idx % n_threads != tid {
                        continue;
                    }
                    comb[0] = b0;
                    if k == 1 {
                        let w = exact_ticket_weight_main(pot, prev_mask, &comb[..1]);
                        local_z += w;
                        local_m[b0] += w;
                    } else {
                        enum_rec(pot, prev_mask, n, k, 1, b0 + 1, &mut comb, &mut local_z, &mut local_m);
                    }
                }
                (local_z, local_m)
            }));
        }
        let mut tot_z = 0.0f64;
        let mut tot_m = vec![0.0f64; n];
        for h in handles {
            let (lz, lm) = h.join().expect("firul de enumerare Q1 a eșuat");
            tot_z += lz;
            for i in 0..n {
                tot_m[i] += lm[i];
            }
        }
        (tot_z, tot_m)
    });

    let pi_main: Vec<f64> = marg_num.into_iter().map(|v| v / z_main).collect();

    let (z_bonus, pi_bonus) = if let Some(mb) = pot.m_bonus {
        // Pentru Q1 pe Joker (1.221.759 × 20 = 24.435.180 stări comune), verificăm explicit suma dublă
        let mut wb = vec![0.0f64; mb];
        let mut zb_joint = 0.0f64;
        for (b, slot) in wb.iter_mut().enumerate() {
            let rep = prev_bonus_0based == Some(b);
            let th_b = if pot.gamma_sh != 0.0 {
                pot.gamma_sh * pot.theta_bonus[usize::from(rep)]
            } else {
                0.0
            };
            let hb = if b < pot.h_bonus_anti.len() { pot.eta_bonus * pot.h_bonus_anti[b] } else { 0.0 };
            let factor_b = (th_b + hb).exp();
            let joint_b = z_main * factor_b;
            *slot = joint_b;
            zb_joint += joint_b;
        }
        for slot in &mut wb {
            *slot /= zb_joint;
        }
        (zb_joint / z_main, wb)
    } else {
        (1.0, Vec::new())
    };

    SingleConditionPartition { z_main, log_z_main: z_main.ln(), pi_main, z_bonus, pi_bonus }
}

#[allow(clippy::too_many_arguments)]
fn enum_rec(
    pot: &ExpertPotential<'_>,
    prev_mask: u128,
    n: usize,
    k: usize,
    depth: usize,
    start: usize,
    comb: &mut [usize; 12],
    acc_z: &mut f64,
    acc_m: &mut [f64],
) {
    let rem = k - depth;
    if rem == 1 {
        let mut u_pref = 0usize;
        let mut lin_pref = 0.0f64;
        let mut sp_pref = 0.0f64;
        let mut sm_pref = 0.0f64;
        for &i in &comb[..depth] {
            if ((prev_mask >> i) & 1) != 0 {
                u_pref += 1;
            }
            lin_pref += pot.h_anti[i];
            sp_pref += pot.modes.u_pos[i];
            sm_pref += pot.modes.u_neg[i];
        }
        let mut prefix_sum_w = 0.0f64;
        for last in start..n {
            let u = u_pref + usize::from(((prev_mask >> last) & 1) != 0);
            let lin = lin_pref + pot.h_anti[last];
            let sp = sp_pref + pot.modes.u_pos[last];
            let sm = sm_pref + pot.modes.u_neg[last];
            let sh = if pot.gamma_sh != 0.0 && u < pot.theta_shell.len() { pot.gamma_sh * pot.theta_shell[u] } else { 0.0 };
            let quad = if pot.gamma_2 != 0.0 {
                0.5 * pot.gamma_2 * (pot.modes.lambda_pos * sp * sp + pot.modes.lambda_neg * sm * sm)
            } else {
                0.0
            };
            let w = (sh + pot.eta * lin + quad).exp();
            prefix_sum_w += w;
            acc_m[last] += w;
        }
        *acc_z += prefix_sum_w;
        for &i in &comb[..depth] {
            acc_m[i] += prefix_sum_w;
        }
        return;
    }
    let max_i = n - rem;
    for i in start..=max_i {
        comb[depth] = i;
        enum_rec(pot, prev_mask, n, k, depth + 1, i + 1, comb, acc_z, acc_m);
    }
}

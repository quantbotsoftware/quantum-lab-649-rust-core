//! Motorul matematic nativ Rust `KJ-V5` — Transport Spectral Neechilibru Johnson–Krawtchouk (`K1`),
//! Operator Ireversibil Koopman `V_2 -> V_2` (`K2`), Curent Antisimetric Schnakenberg `V_1` (`K3`)
//! și Portofoliu Bayesian cu Masă Punctuală `w_0 = 0.50` (`K4`) (`PROTOCOL_KJV5_RUST_V2.md`).

pub mod koopman_v2;
pub mod krawtchouk;
pub mod multiplet;
pub mod partition;
pub mod qualify;

use serde::Serialize;

use koopman_v2::{
    frob_dot, jacobi_extremal_modes, normalize_centered_field, project_perp_phi, project_v2, scale_q_tilde,
    schnakenberg_field_from_corr, V2Modes, K3_LAG_WEIGHTS, LAMBDA_K2, LAMBDA_K3, TAU_K2, TAU_K3,
};
use krawtchouk::{
    binom_f64, compute_bonus_shell_potential, compute_shell_potential, BonusShellBasis, KrawtchoukShellBasis,
    MAX_SHELL_ORDER, TAU_SHELL,
};
use partition::{eval_partition_conditioned, eval_subdraw_restricted_sum, ExpertPotential, SingleConditionPartition};

pub const NUM_EXPERTS: usize = 8;
pub const W0_ABSTAIN_PRIOR: f64 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum V2Source {
    None,
    Anti,
    Dir,
    Mix,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExpertSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub base_prior: f64,
    pub gamma_sh: f64,
    pub gamma_2: f64,
    pub v2_source: V2Source,
    pub eta: f64,
    pub eta_bonus: f64,
}

/// Grila fixată preînregistrată de `M = 8` experți (`PROTOCOL_KJV5_RUST_V2.md` §3.4).
pub const EXPERT_GRID: [ExpertSpec; NUM_EXPERTS] = [
    ExpertSpec {
        id: "m0_uniform",
        label: "Abținere / Uniform",
        base_prior: 0.50,
        gamma_sh: 0.00,
        gamma_2: 0.00,
        v2_source: V2Source::None,
        eta: 0.00,
        eta_bonus: 0.00,
    },
    ExpertSpec {
        id: "m1_k1_mild",
        label: "K1 Shell Mild",
        base_prior: 0.08,
        gamma_sh: 0.50,
        gamma_2: 0.00,
        v2_source: V2Source::None,
        eta: 0.00,
        eta_bonus: 0.00,
    },
    ExpertSpec {
        id: "m2_k1_strong",
        label: "K1 Shell Strong",
        base_prior: 0.07,
        gamma_sh: 1.00,
        gamma_2: 0.00,
        v2_source: V2Source::None,
        eta: 0.00,
        eta_bonus: 0.00,
    },
    ExpertSpec {
        id: "m3_k2_anti",
        label: "K2 Koopman V2 Antisimetric",
        base_prior: 0.09,
        gamma_sh: 0.00,
        gamma_2: 0.70,
        v2_source: V2Source::Anti,
        eta: 0.00,
        eta_bonus: 0.00,
    },
    ExpertSpec {
        id: "m4_k2_dir",
        label: "K2 Koopman V2 Direcționat",
        base_prior: 0.08,
        gamma_sh: 0.00,
        gamma_2: 0.65,
        v2_source: V2Source::Dir,
        eta: 0.00,
        eta_bonus: 0.00,
    },
    ExpertSpec {
        id: "m5_k3_schnak",
        label: "K3 Schnakenberg V1",
        base_prior: 0.06,
        gamma_sh: 0.00,
        gamma_2: 0.00,
        v2_source: V2Source::None,
        eta: 0.50,
        eta_bonus: 0.50,
    },
    ExpertSpec {
        id: "m6_k1_k2_mild",
        label: "K1+K2 Cuplat Mild",
        base_prior: 0.06,
        gamma_sh: 0.55,
        gamma_2: 0.55,
        v2_source: V2Source::Mix,
        eta: 0.00,
        eta_bonus: 0.00,
    },
    ExpertSpec {
        id: "m7_full_kjv5",
        label: "Full KJ-V5 Strong",
        base_prior: 0.06,
        gamma_sh: 0.75,
        gamma_2: 0.75,
        v2_source: V2Source::Mix,
        eta: 0.45,
        eta_bonus: 0.45,
    },
];

/// Configurațiile de ablație din `PROTOCOL_KJV5_RUST_V2.md` §3.4 & §4.2.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AblationConfig {
    pub key: &'static str,
    pub label: &'static str,
    pub enable_k1: bool,
    pub enable_k2: bool,
    pub enable_k3: bool,
}

pub const ABLATION_CONFIGS: [AblationConfig; 4] = [
    AblationConfig { key: "K1", label: "K1 pur (Învelișuri Krawtchouk)", enable_k1: true, enable_k2: false, enable_k3: false },
    AblationConfig { key: "K2", label: "K2 pur (Koopman V2 -> V2)", enable_k1: false, enable_k2: true, enable_k3: false },
    AblationConfig { key: "K1_K2", label: "K1+K2 (Johnson + Koopman V2)", enable_k1: true, enable_k2: true, enable_k3: false },
    AblationConfig { key: "Full_KJV5", label: "Full KJ-V5 (K1+K2+K3)", enable_k1: true, enable_k2: true, enable_k3: true },
];

/// Calculează ponderile a priori pentru o configurație de ablație (`w_0^(0) = 0.50` fix).
pub fn compute_prior_weights(enable_k1: bool, enable_k2: bool, enable_k3: bool) -> [f64; NUM_EXPERTS] {
    let mut priors = [0.0f64; NUM_EXPERTS];
    priors[0] = W0_ABSTAIN_PRIOR;
    let mut sum_act = 0.0f64;
    for m in 1..NUM_EXPERTS {
        let sp = &EXPERT_GRID[m];
        let needs_k1 = sp.gamma_sh > 0.0;
        let needs_k2 = sp.gamma_2 > 0.0;
        let needs_k3 = sp.eta > 0.0;
        if (!needs_k1 || enable_k1) && (!needs_k2 || enable_k2) && (!needs_k3 || enable_k3) {
            priors[m] = sp.base_prior;
            sum_act += sp.base_prior;
        }
    }
    if sum_act > 0.0 {
        let scale = (1.0 - W0_ABSTAIN_PRIOR) / sum_act;
        for slot in priors.iter_mut().skip(1) {
            *slot *= scale;
        }
    } else {
        priors[0] = 1.0;
    }
    priors
}

/// Descompune vectorul celor 8 experți în masele canalelor `(w_0, w_K1, w_K2, w_K3)` cu sumă exactă `1.0`.
pub fn decompose_channel_masses(w: &[f64; NUM_EXPERTS]) -> ChannelMasses {
    ChannelMasses {
        w_0: w[0],
        w_k1: w[1] + w[2] + 0.5 * w[6] + (w[7] / 3.0),
        w_k2: w[3] + w[4] + 0.5 * w[6] + (w[7] / 3.0),
        w_k3: w[5] + (w[7] / 3.0),
        w_k1_pure: w[1] + w[2],
        w_k2_anti_pure: w[3],
        w_k2_dir_pure: w[4],
        w_k3_pure: w[5],
        w_coupled: w[6] + w[7],
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ChannelMasses {
    pub w_0: f64,
    pub w_k1: f64,
    pub w_k2: f64,
    pub w_k3: f64,
    pub w_k1_pure: f64,
    pub w_k2_anti_pure: f64,
    pub w_k2_dir_pure: f64,
    pub w_k3_pure: f64,
    pub w_coupled: f64,
}

/// Spațiul combinatorial și bazele Krawtchouk pre-construite pentru un joc `(N, r, k, M_B)`.
#[derive(Debug, Clone, Serialize)]
pub struct KjV5Space {
    pub n: usize,
    pub r: usize,
    pub k: usize,
    pub m_bonus: Option<usize>,
    pub total_base_states: u64,
    pub total_joint_states: u64,
    pub draw_shell_basis: KrawtchoukShellBasis,
    pub tkt_shell_basis: KrawtchoukShellBasis,
    pub bonus_shell_basis: Option<BonusShellBasis>,
}

impl KjV5Space {
    pub fn new(n: usize, r: usize, k: usize, m_bonus: Option<usize>) -> Self {
        let base = binom_f64(n, k).round() as u64;
        let joint = base.saturating_mul(m_bonus.unwrap_or(1) as u64);
        Self {
            n,
            r,
            k,
            m_bonus,
            total_base_states: base,
            total_joint_states: joint,
            draw_shell_basis: KrawtchoukShellBasis::new(n, r, r),
            tkt_shell_basis: KrawtchoukShellBasis::new(n, r, k),
            bonus_shell_basis: m_bonus.map(BonusShellBasis::new),
        }
    }
}

/// Un pas temporal canonic (o dată calendaristică): extragerile din aceeași zi sunt sortate canonic
/// lexicografic pentru invarianță exactă `0.0` la ordinea intra-zi (`Q5`).
#[derive(Debug, Clone)]
pub struct CanonicalStep {
    pub date: String,
    pub draws: Vec<(Vec<usize>, Option<usize>)>,
    pub x_tilde: Vec<f64>,
    pub y_tilde: Vec<f64>,
    pub pairs_per_draw: Vec<Vec<(usize, usize)>>,
    pub phi_v2: Vec<f64>,
    pub phi_norm2: f64,
}

impl CanonicalStep {
    /// Construiește pasul canonic din extrageri 1-based `(balls_1based, bonus_1based)`.
    pub fn from_1based(space: &KjV5Space, date: impl Into<String>, raw_draws: &[(Vec<usize>, Option<usize>)]) -> Self {
        assert!(!raw_draws.is_empty());
        let mut draws: Vec<(Vec<usize>, Option<usize>)> = raw_draws
            .iter()
            .map(|(b, jb)| {
                let mut b0: Vec<usize> = b.iter().map(|&x| x - 1).collect();
                b0.sort_unstable();
                assert_eq!(b0.len(), space.r);
                (b0, jb.map(|j| j - 1))
            })
            .collect();
        draws.sort();

        let m_s = draws.len() as f64;
        let inv_m = 1.0 / m_s;
        let base_p = (space.r as f64) / (space.n as f64);
        let mut x_tilde = vec![-base_p; space.n];
        let mut p_bar = vec![0.0f64; space.n * space.n];
        let mut pairs_per_draw = Vec::with_capacity(draws.len());

        for (b0, _) in &draws {
            for &i in b0 {
                x_tilde[i] += inv_m;
            }
            let mut prs = Vec::with_capacity(space.r * (space.r - 1) / 2);
            for a_pos in 0..b0.len() {
                let u = b0[a_pos];
                for &v in &b0[(a_pos + 1)..] {
                    prs.push((u, v));
                    p_bar[u * space.n + v] += inv_m;
                    p_bar[v * space.n + u] += inv_m;
                }
            }
            pairs_per_draw.push(prs);
        }

        let y_tilde = if let Some(mb) = space.m_bonus {
            let mut yt = vec![-1.0 / (mb as f64); mb];
            for &(_, jb) in &draws {
                if let Some(j0) = jb {
                    yt[j0] += inv_m;
                }
            }
            yt
        } else {
            Vec::new()
        };

        let phi_v2 = project_v2(&p_bar, space.n);
        let phi_norm2 = frob_dot(&phi_v2, &phi_v2);

        Self { date: date.into(), draws, x_tilde, y_tilde, pairs_per_draw, phi_v2, phi_norm2 }
    }

    /// Produsul Frobenius rapid `<P_bar_self, Mat>_F` pentru o matrice simetrică cu diagonală nulă `Mat`.
    #[inline]
    pub fn frob_dot_pbar(&self, sym_zero_diag_mat: &[f64], n: usize) -> f64 {
        let mut sum_upper = 0.0f64;
        for prs in &self.pairs_per_draw {
            for &(u, v) in prs {
                sum_upper += sym_zero_diag_mat[u * n + v];
            }
        }
        2.0 * sum_upper / (self.draws.len() as f64)
    }

    /// Adună `coeff * P_bar_self` într-o matrice simetrică `acc`.
    #[inline]
    pub fn add_scaled_pbar(&self, coeff: f64, acc: &mut [f64], n: usize) {
        if coeff == 0.0 {
            return;
        }
        let w = coeff / (self.draws.len() as f64);
        for prs in &self.pairs_per_draw {
            for &(u, v) in prs {
                acc[u * n + v] += w;
                acc[v * n + u] += w;
            }
        }
    }
}

/// Operatorii cauzali `K1 + K2 + K3` evaluați la începutul pasului `s` (folosind strict pașii `0..s-1`).
#[derive(Debug, Clone, Serialize)]
pub struct KjV5StepOperators {
    pub step_idx: usize,
    pub shell_coeffs: [f64; MAX_SHELL_ORDER],
    pub theta_shell: Vec<f64>,
    pub bonus_shell_coeff: f64,
    pub theta_bonus: [f64; 2],
    pub q_anti_raw_norm: f64,
    pub q_dir_raw_norm: f64,
    pub q_anti_phi_dot: f64,
    pub modes_zero: V2Modes,
    pub modes_anti: V2Modes,
    pub modes_dir: V2Modes,
    pub modes_mix: V2Modes,
    pub h_anti: Vec<f64>,
    pub h_bonus_anti: Vec<f64>,
    pub prev_draws: Vec<(Vec<usize>, Option<usize>)>,
    /// Salvăm și matricea brută `Q_anti` pentru certificarea algebrică `Q2` (`||Q_anti(H_+) + Q_anti(H_-)||_F < 1e-13`).
    #[serde(skip)]
    pub q_anti_raw: Vec<f64>,
}

impl KjV5StepOperators {
    pub fn expert_potential<'a>(&'a self, space: &'a KjV5Space, m: usize) -> ExpertPotential<'a> {
        let sp = &EXPERT_GRID[m];
        let modes = match sp.v2_source {
            V2Source::None => &self.modes_zero,
            V2Source::Anti => &self.modes_anti,
            V2Source::Dir => &self.modes_dir,
            V2Source::Mix => &self.modes_mix,
        };
        ExpertPotential {
            n: space.n,
            r: space.r,
            k: space.k,
            gamma_sh: sp.gamma_sh,
            gamma_2: sp.gamma_2,
            eta: sp.eta,
            eta_bonus: sp.eta_bonus,
            theta_shell: &self.theta_shell,
            h_anti: &self.h_anti,
            modes,
            m_bonus: space.m_bonus,
            theta_bonus: self.theta_bonus,
            h_bonus_anti: &self.h_bonus_anti,
        }
    }
}

/// Pachetul evaluării celor 8 experți la pasul `s`.
#[derive(Debug, Clone)]
pub struct StepExpertPack {
    pub z_packs: [Vec<SingleConditionPartition>; NUM_EXPERTS],
    pub pi_m_main: [[f64; 96]; NUM_EXPERTS],
    pub pi_m_bonus: [Vec<f64>; NUM_EXPERTS],
    /// `draw_lrs[u][m]` = raportul de verosimilitate `P_m(D_u) / P_U(D_u)` pe extragerea `u` din pasul `s`.
    pub draw_lrs: Vec<[f64; NUM_EXPERTS]>,
    /// `step_lrs[m] = (1 / m_s) ∑_u draw_lrs[u][m]` (`E_U[step_lrs[m]] == 1.0` exact).
    pub step_lrs: [f64; NUM_EXPERTS],
}

/// Starea cauzală incrementală a motorului `KJ-V5` (`Incremental == Full` la `< 1e-15`).
#[derive(Debug, Clone)]
pub struct KjV5State {
    pub space: KjV5Space,
    pub steps: Vec<CanonicalStep>,
    pub s_sh: [f64; MAX_SHELL_ORDER],
    pub s_sh_bonus: f64,
    pub n_sh: f64,
    pub c_to_main: Vec<f64>,
    pub c_to_bonus: Vec<f64>,
    pub c_xb: Vec<f64>,
    pub n_k3: f64,
    pub priors: [f64; NUM_EXPERTS],
    pub log_wealth: [f64; NUM_EXPERTS],
}

impl KjV5State {
    pub fn new(space: KjV5Space, enable_k1: bool, enable_k2: bool, enable_k3: bool) -> Self {
        let n = space.n;
        let mb = space.m_bonus.unwrap_or(0);
        let priors = compute_prior_weights(enable_k1, enable_k2, enable_k3);
        let mut log_wealth = [f64::NEG_INFINITY; NUM_EXPERTS];
        for m in 0..NUM_EXPERTS {
            if priors[m] > 0.0 {
                log_wealth[m] = priors[m].ln();
            }
        }
        Self {
            space,
            steps: Vec::new(),
            s_sh: [0.0; MAX_SHELL_ORDER],
            s_sh_bonus: 0.0,
            n_sh: 0.0,
            c_to_main: vec![0.0; n * n],
            c_to_bonus: vec![0.0; mb * mb],
            c_xb: vec![0.0; n * mb],
            n_k3: 0.0,
            priors,
            log_wealth,
        }
    }

    /// Ponderile curente ale portofoliului Bayesian `w_m^(s)` și valoarea martingală `E_s >= 0.50`.
    pub fn portfolio_weights_and_e(&self) -> ([f64; NUM_EXPERTS], f64) {
        let mut max_lw = f64::NEG_INFINITY;
        for &lw in &self.log_wealth {
            if lw.is_finite() && lw > max_lw {
                max_lw = lw;
            }
        }
        let mut w = [0.0f64; NUM_EXPERTS];
        let mut sum_rel = 0.0f64;
        for m in 0..NUM_EXPERTS {
            if self.log_wealth[m].is_finite() {
                let rel = (self.log_wealth[m] - max_lw).exp();
                w[m] = rel;
                sum_rel += rel;
            }
        }
        let e_val = sum_rel * max_lw.exp();
        for slot in &mut w {
            *slot /= sum_rel;
        }
        (w, e_val)
    }

    /// Construiește operatorii cauzali `K1`, `K2`, `K3` înainte de observarea pasului `s = self.steps.len()`.
    pub fn forward_operators(&self) -> KjV5StepOperators {
        let s = self.steps.len();
        let n = self.space.n;
        let mb = self.space.m_bonus.unwrap_or(0);

        let (shell_coeffs, theta_shell) = compute_shell_potential(&self.space.tkt_shell_basis, &self.s_sh, self.n_sh, s);
        let (bonus_shell_coeff, theta_bonus) = if let Some(ref bb) = self.space.bonus_shell_basis {
            compute_bonus_shell_potential(bb, self.s_sh_bonus, self.n_sh, s)
        } else {
            (0.0, [0.0, 0.0])
        };

        let modes_zero = V2Modes::zero(n);
        if s < 2 {
            return KjV5StepOperators {
                step_idx: s,
                shell_coeffs,
                theta_shell,
                bonus_shell_coeff,
                theta_bonus,
                q_anti_raw_norm: 0.0,
                q_dir_raw_norm: 0.0,
                q_anti_phi_dot: 0.0,
                modes_zero: modes_zero.clone(),
                modes_anti: modes_zero.clone(),
                modes_dir: modes_zero.clone(),
                modes_mix: modes_zero,
                h_anti: vec![0.0; n],
                h_bonus_anti: vec![0.0; mb],
                prev_draws: self.steps.last().map(|st| st.draws.clone()).unwrap_or_default(),
                q_anti_raw: vec![0.0; n * n],
            };
        }

        // K2: Transport Koopman V_2 -> V_2 pe t = 1 .. s-1 cu declanșator Φ_{s-1}
        let trig_step = &self.steps[s - 1];
        let phi_trig = &trig_step.phi_v2;
        let rho_k2 = (-1.0 / TAU_K2).exp();
        let mut z2_eff = LAMBDA_K2;
        let mut acc_dir = vec![0.0f64; n * n];
        let mut acc_anti = vec![0.0f64; n * n];

        let mut w_t = 1.0f64;
        for t in (1..s).rev() {
            let st_prev = &self.steps[t - 1];
            let st_cur = &self.steps[t];
            z2_eff += w_t * st_prev.phi_norm2;
            let k_fwd = st_prev.frob_dot_pbar(phi_trig, n);
            let k_rev = st_cur.frob_dot_pbar(phi_trig, n);
            st_cur.add_scaled_pbar(w_t * k_fwd, &mut acc_dir, n);
            st_cur.add_scaled_pbar(0.5 * w_t * k_fwd, &mut acc_anti, n);
            st_prev.add_scaled_pbar(-0.5 * w_t * k_rev, &mut acc_anti, n);
            w_t *= rho_k2;
        }
        let inv_z2 = 1.0 / z2_eff;
        for v in &mut acc_dir {
            *v *= inv_z2;
        }
        for v in &mut acc_anti {
            *v *= inv_z2;
        }

        let q_dir_raw = project_v2(&acc_dir, n);
        let q_anti_v2 = project_v2(&acc_anti, n);
        let q_anti_raw = project_perp_phi(&q_anti_v2, phi_trig);
        let q_anti_phi_dot = frob_dot(&q_anti_raw, phi_trig);
        let q_anti_raw_norm = koopman_v2::frob_norm(&q_anti_raw);
        let q_dir_raw_norm = koopman_v2::frob_norm(&q_dir_raw);

        let q_mix_raw: Vec<f64> = q_anti_raw.iter().zip(q_dir_raw.iter()).map(|(&a, &d)| 0.5 * (a + d)).collect();

        let modes_anti = jacobi_extremal_modes(&scale_q_tilde(&q_anti_raw), n);
        let modes_dir = jacobi_extremal_modes(&scale_q_tilde(&q_dir_raw), n);
        let modes_mix = jacobi_extremal_modes(&scale_q_tilde(&q_mix_raw), n);

        // K3: Curent Schnakenberg V_1 + cuplaj bipartit secundar
        let denom_k3 = LAMBDA_K3 + self.n_k3;
        let h_anti = schnakenberg_field_from_corr(&self.c_to_main, &trig_step.x_tilde, denom_k3, n);

        let h_bonus_anti = if mb >= 2 {
            let inv_d = 1.0 / denom_k3;
            let mut hb_raw = vec![0.0f64; mb];
            for a in 0..mb {
                let ya = trig_step.y_tilde[a];
                if ya != 0.0 {
                    for b in 0..mb {
                        let j_ab = 0.5 * (self.c_to_bonus[a * mb + b] - self.c_to_bonus[b * mb + a]) * inv_d;
                        hb_raw[b] += j_ab * ya;
                    }
                }
            }
            for i in 0..n {
                let xi = trig_step.x_tilde[i];
                if xi != 0.0 {
                    for b in 0..mb {
                        hb_raw[b] += 0.5 * self.c_xb[i * mb + b] * inv_d * xi;
                    }
                }
            }
            normalize_centered_field(&hb_raw)
        } else {
            Vec::new()
        };

        KjV5StepOperators {
            step_idx: s,
            shell_coeffs,
            theta_shell,
            bonus_shell_coeff,
            theta_bonus,
            q_anti_raw_norm,
            q_dir_raw_norm,
            q_anti_phi_dot,
            modes_zero,
            modes_anti,
            modes_dir,
            modes_mix,
            h_anti,
            h_bonus_anti,
            prev_draws: trig_step.draws.clone(),
            q_anti_raw,
        }
    }

    /// Evaluează toți experții din `active_mask` pe extragerile observate în pasul `target_step`.
    pub fn evaluate_experts_on_step(
        &self,
        ops: &KjV5StepOperators,
        target_step: &CanonicalStep,
        active_mask: &[bool; NUM_EXPERTS],
    ) -> StepExpertPack {
        let n = self.space.n;
        let r = self.space.r;
        let k = self.space.k;
        let mb_opt = self.space.m_bonus;
        let m_draws_target = target_step.draws.len();

        let mut z_packs: [Vec<SingleConditionPartition>; NUM_EXPERTS] = Default::default();
        let mut pi_m_main = [[0.0f64; 96]; NUM_EXPERTS];
        let mut pi_m_bonus: [Vec<f64>; NUM_EXPERTS] = Default::default();

        let unif_pi = (k as f64) / (n as f64);
        for i in 0..n {
            pi_m_main[0][i] = unif_pi;
        }
        if let Some(mb) = mb_opt {
            pi_m_bonus[0] = vec![1.0 / (mb as f64); mb];
        }

        if ops.prev_draws.is_empty() {
            let draw_lrs = vec![[1.0f64; NUM_EXPERTS]; m_draws_target];
            return StepExpertPack { z_packs, pi_m_main, pi_m_bonus, draw_lrs, step_lrs: [1.0; NUM_EXPERTS] };
        }

        let inv_m_prev = 1.0 / (ops.prev_draws.len() as f64);
        for m in 1..NUM_EXPERTS {
            if !active_mask[m] {
                for i in 0..n {
                    pi_m_main[m][i] = unif_pi;
                }
                if let Some(mb) = mb_opt {
                    pi_m_bonus[m] = vec![1.0 / (mb as f64); mb];
                }
                continue;
            }
            let pot = ops.expert_potential(&self.space, m);
            let mut cond_vec = Vec::with_capacity(ops.prev_draws.len());
            if let Some(mb) = mb_opt {
                pi_m_bonus[m] = vec![0.0; mb];
            }
            for (d_prev, jb_prev) in &ops.prev_draws {
                let cp = eval_partition_conditioned(&pot, d_prev, *jb_prev);
                for i in 0..n {
                    pi_m_main[m][i] += inv_m_prev * cp.pi_main[i];
                }
                if let Some(mb) = mb_opt {
                    for b in 0..mb {
                        pi_m_bonus[m][b] += inv_m_prev * cp.pi_bonus[b];
                    }
                }
                cond_vec.push(cp);
            }
            z_packs[m] = cond_vec;
        }

        let norm_comb = (self.space.total_base_states as f64) / binom_f64(r, k);
        let mut draw_lrs = vec![[1.0f64; NUM_EXPERTS]; m_draws_target];
        let mut step_lrs = [0.0f64; NUM_EXPERTS];
        step_lrs[0] = 1.0;

        for (u_idx, (d_targ, jb_targ)) in target_step.draws.iter().enumerate() {
            draw_lrs[u_idx][0] = 1.0;
            for m in 1..NUM_EXPERTS {
                if !active_mask[m] {
                    continue;
                }
                let pot = ops.expert_potential(&self.space, m);
                let mut mix_prob_ratio = 0.0f64;
                for (p_idx, (d_prev, _)) in ops.prev_draws.iter().enumerate() {
                    let zp = &z_packs[m][p_idx];
                    let sum_sub = eval_subdraw_restricted_sum(&pot, d_prev, d_targ);
                    let mut lr_cond = norm_comb * (sum_sub / zp.z_main);
                    if let (Some(mb), Some(j_t)) = (mb_opt, *jb_targ) {
                        lr_cond *= (mb as f64) * zp.pi_bonus[j_t];
                    }
                    mix_prob_ratio += inv_m_prev * lr_cond;
                }
                draw_lrs[u_idx][m] = mix_prob_ratio.max(1e-300);
                step_lrs[m] += draw_lrs[u_idx][m] / (m_draws_target as f64);
            }
        }
        for m in 1..NUM_EXPERTS {
            if !active_mask[m] {
                step_lrs[m] = 1.0;
            }
        }

        StepExpertPack { z_packs, pi_m_main, pi_m_bonus, draw_lrs, step_lrs }
    }

    /// Avansează starea internă (`K1`, `K3` și arhiva pașilor `K2`) cu un nou pas canonic `step`,
    /// actualizând opțional și averile portofoliului dacă `step_lrs_opt` este furnizat.
    pub fn advance_with_step(&mut self, step: CanonicalStep, step_lrs_opt: Option<&[f64; NUM_EXPERTS]>) {
        if let Some(lrs) = step_lrs_opt {
            for m in 0..NUM_EXPERTS {
                if self.log_wealth[m].is_finite() {
                    self.log_wealth[m] += lrs[m].max(1e-300).ln();
                }
            }
        }

        let s = self.steps.len();
        if s >= 1 {
            // Actualizare K1 (tranziția s-1 -> s)
            let rho_sh = (-1.0 / TAU_SHELL).exp();
            let prev = &self.steps[s - 1];
            let pair_cnt = (prev.draws.len() * step.draws.len()) as f64;
            let mut bar_psi = [0.0f64; MAX_SHELL_ORDER];
            let mut bar_psi_b = 0.0f64;

            for (dp, jbp) in &prev.draws {
                let mut mask_p = 0u128;
                for &b in dp {
                    mask_p |= 1u128 << b;
                }
                for (dc, jbc) in &step.draws {
                    let mut ov = 0usize;
                    for &b in dc {
                        if ((mask_p >> b) & 1) != 0 {
                            ov += 1;
                        }
                    }
                    for m in 1..=MAX_SHELL_ORDER {
                        bar_psi[m - 1] += self.space.draw_shell_basis.eval_order(m, ov) / pair_cnt;
                    }
                    if let (Some(ref bb), Some(jp), Some(jc)) = (&self.space.bonus_shell_basis, *jbp, *jbc) {
                        bar_psi_b += bb.eval(jp == jc) / pair_cnt;
                    }
                }
            }
            for m in 0..MAX_SHELL_ORDER {
                self.s_sh[m] = rho_sh * self.s_sh[m] + bar_psi[m];
            }
            self.s_sh_bonus = rho_sh * self.s_sh_bonus + bar_psi_b;
            self.n_sh = rho_sh * self.n_sh + 1.0;

            // Actualizare K3 (curent Schnakenberg pe lag-urile l = 1, 2, 3)
            let rho_k3 = (-1.0 / TAU_K3).exp();
            for v in &mut self.c_to_main {
                *v *= rho_k3;
            }
            for v in &mut self.c_to_bonus {
                *v *= rho_k3;
            }
            for v in &mut self.c_xb {
                *v *= rho_k3;
            }
            self.n_k3 = rho_k3 * self.n_k3 + 1.0;

            let n = self.space.n;
            let mb = self.space.m_bonus.unwrap_or(0);
            for (lag_idx, &w_lag) in K3_LAG_WEIGHTS.iter().enumerate() {
                let lag = lag_idx + 1;
                if s >= lag {
                    let st_lag = &self.steps[s - lag];
                    for i in 0..n {
                        let xi = w_lag * st_lag.x_tilde[i];
                        if xi != 0.0 {
                            let row = &mut self.c_to_main[i * n..(i + 1) * n];
                            for j in 0..n {
                                row[j] += xi * step.x_tilde[j];
                            }
                        }
                    }
                    if mb >= 2 {
                        for a in 0..mb {
                            let ya = w_lag * st_lag.y_tilde[a];
                            if ya != 0.0 {
                                let row = &mut self.c_to_bonus[a * mb..(a + 1) * mb];
                                for b in 0..mb {
                                    row[b] += ya * step.y_tilde[b];
                                }
                            }
                        }
                    }
                }
            }
            if mb >= 2 {
                let st_l1 = &self.steps[s - 1];
                for i in 0..n {
                    let xi = st_l1.x_tilde[i];
                    if xi != 0.0 {
                        let row = &mut self.c_xb[i * mb..(i + 1) * mb];
                        for b in 0..mb {
                            row[b] += xi * step.y_tilde[b];
                        }
                    }
                }
            }
        }
        self.steps.push(step);
    }
}

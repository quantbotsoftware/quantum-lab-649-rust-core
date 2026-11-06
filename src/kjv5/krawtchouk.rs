//! `K1` — Baza ortonormată hipergeometrică / Krawtchouk pe învelișurile schemei Johnson `J(N, k)`
//! și potențialul cauzal de înveliș `θ_u^(s)` (`PROTOCOL_KJV5_RUST_V2.md` §3.1).

use serde::Serialize;

pub const TAU_SHELL: f64 = 128.0;
pub const N0_SHELL: f64 = 24.0;
pub const AMP_SHELL: f64 = 0.55;
pub const MAX_SHELL_ORDER: usize = 3;

/// Log-binomial exact `ln C(n, k)` calculat prin suma logaritmilor `f64` fără overflow pentru `n <= 256`.
pub fn ln_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    if k == 0 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for i in 1..=k {
        acc += ((n + 1 - i) as f64).ln() - (i as f64).ln();
    }
    acc
}

/// Binomial exact ca `f64` (pentru `N <= 80, k <= 20`, încape exact sau în mantisa IEEE-754).
pub fn binom_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut res = 1.0f64;
    for i in 1..=k {
        res = res * ((n + 1 - i) as f64) / (i as f64);
    }
    res
}

/// Baza ortonormată Krawtchouk / Hahn discretă `ψ_0, ψ_1, ψ_2, ψ_3` pentru suprapunerea `u = |σ ∩ D|`
/// sub legea hipergeometrică uniformă `p_0(u; N, r, k)`.
#[derive(Debug, Clone, Serialize)]
pub struct KrawtchoukShellBasis {
    pub n: usize,
    pub r: usize,
    pub k: usize,
    pub u_min: usize,
    pub u_max: usize,
    pub mean_u: f64,
    pub p0: Vec<f64>,
    pub psi: [[f64; 32]; MAX_SHELL_ORDER + 1],
    pub active_orders: [bool; MAX_SHELL_ORDER],
    pub orthonormality_residual: f64,
}

impl KrawtchoukShellBasis {
    pub fn new(n: usize, r: usize, k: usize) -> Self {
        assert!(n >= 4 && r >= 1 && k >= 1 && r < n && k <= n && r <= 31 && k <= 31);
        let u_min = (k + r).saturating_sub(n);
        let u_max = k.min(r);
        let mean_u = (r * k) as f64 / (n as f64);

        let mut log_p = vec![f64::NEG_INFINITY; u_max + 1];
        let mut max_lp = f64::NEG_INFINITY;
        let ln_tot = ln_binom(n, k);
        for (u, lp_slot) in log_p.iter_mut().enumerate().take(u_max + 1).skip(u_min) {
            let lp = ln_binom(r, u) + ln_binom(n - r, k - u) - ln_tot;
            *lp_slot = lp;
            if lp > max_lp {
                max_lp = lp;
            }
        }
        let mut p0 = vec![0.0f64; u_max + 1];
        let mut sum_p = 0.0f64;
        for u in u_min..=u_max {
            let val = (log_p[u] - max_lp).exp();
            p0[u] = val;
            sum_p += val;
        }
        for val in p0.iter_mut().take(u_max + 1).skip(u_min) {
            *val /= sum_p;
        }

        let mut psi = [[0.0f64; 32]; MAX_SHELL_ORDER + 1];
        for val in psi[0].iter_mut().take(u_max + 1).skip(u_min) {
            *val = 1.0;
        }
        let mut active_orders = [false; MAX_SHELL_ORDER];

        // Gram–Schmidt în dublă trecere pe monomiile centrate (u - mean_u)^m, m = 1..=3.
        for m in 1..=MAX_SHELL_ORDER {
            let mut v = [0.0f64; 32];
            for (u, slot) in v.iter_mut().enumerate().take(u_max + 1).skip(u_min) {
                *slot = ((u as f64) - mean_u).powi(m as i32);
            }
            for _pass in 0..2 {
                for (j, psi_j) in psi.iter().enumerate().take(m) {
                    if j > 0 && !active_orders[j - 1] {
                        continue;
                    }
                    let mut dot = 0.0f64;
                    for u in u_min..=u_max {
                        dot += p0[u] * v[u] * psi_j[u];
                    }
                    for u in u_min..=u_max {
                        v[u] -= dot * psi_j[u];
                    }
                }
            }
            let mut norm2 = 0.0f64;
            for u in u_min..=u_max {
                norm2 += p0[u] * v[u] * v[u];
            }
            let norm = norm2.sqrt();
            if norm > 1e-12 {
                active_orders[m - 1] = true;
                for u in u_min..=u_max {
                    psi[m][u] = v[u] / norm;
                }
            }
        }

        // Reziduu de ortonormalitate max |<ψ_a, ψ_b> - δ_ab| pe ordinele active
        let mut max_res = 0.0f64;
        for a in 0..=MAX_SHELL_ORDER {
            if a > 0 && !active_orders[a - 1] {
                continue;
            }
            for b in 0..=MAX_SHELL_ORDER {
                if b > 0 && !active_orders[b - 1] {
                    continue;
                }
                let mut dot = 0.0f64;
                for u in u_min..=u_max {
                    dot += p0[u] * psi[a][u] * psi[b][u];
                }
                let want = if a == b { 1.0 } else { 0.0 };
                let err = (dot - want).abs();
                if err > max_res {
                    max_res = err;
                }
            }
        }

        Self { n, r, k, u_min, u_max, mean_u, p0, psi, active_orders, orthonormality_residual: max_res }
    }

    #[inline]
    pub fn eval_order(&self, m: usize, u: usize) -> f64 {
        debug_assert!(m >= 1 && m <= MAX_SHELL_ORDER && u <= self.u_max);
        self.psi[m][u]
    }
}

/// Baza ortonormată de ordin 1 pentru urna secundară (Bonus / Joker, `M_B >= 2`, `r_B = 1`).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BonusShellBasis {
    pub m_bonus: usize,
    pub psi_0: f64,
    pub psi_1: f64,
}

impl BonusShellBasis {
    pub fn new(m_bonus: usize) -> Self {
        assert!(m_bonus >= 2);
        let mf = (m_bonus - 1) as f64;
        Self { m_bonus, psi_0: -1.0 / mf.sqrt(), psi_1: mf.sqrt() }
    }

    #[inline]
    pub fn eval(&self, is_repeat: bool) -> f64 {
        if is_repeat {
            self.psi_1
        } else {
            self.psi_0
        }
    }
}

///Calculează coeficienții cauzali `c_m^(s)` și potențialul de înveliș `θ_u^(s)` la pasul `s`.
pub fn compute_shell_potential(
    tkt_basis: &KrawtchoukShellBasis,
    s_sh: &[f64; MAX_SHELL_ORDER],
    n_sh: f64,
    step_idx: usize,
) -> ([f64; MAX_SHELL_ORDER], Vec<f64>) {
    let mut coeffs = [0.0f64; MAX_SHELL_ORDER];
    let mut theta = vec![0.0f64; tkt_basis.u_max + 1];
    if step_idx < 2 || n_sh <= 0.0 {
        return (coeffs, theta);
    }
    let denom = n_sh + N0_SHELL;
    for m in 1..=MAX_SHELL_ORDER {
        if tkt_basis.active_orders[m - 1] {
            coeffs[m - 1] = AMP_SHELL * (s_sh[m - 1] / denom).tanh();
        }
    }
    for (u, slot) in theta.iter_mut().enumerate().take(tkt_basis.u_max + 1).skip(tkt_basis.u_min) {
        let mut acc = 0.0f64;
        for m in 1..=MAX_SHELL_ORDER {
            acc += coeffs[m - 1] * tkt_basis.psi[m][u];
        }
        *slot = acc;
    }
    (coeffs, theta)
}

/// Potențialul de înveliș pe bila secundară `[θ_B(0), θ_B(1)]`.
pub fn compute_bonus_shell_potential(
    bonus_basis: &BonusShellBasis,
    s_sh_bonus: f64,
    n_sh: f64,
    step_idx: usize,
) -> (f64, [f64; 2]) {
    if step_idx < 2 || n_sh <= 0.0 {
        return (0.0, [0.0, 0.0]);
    }
    let c_b = AMP_SHELL * (s_sh_bonus / (n_sh + N0_SHELL)).tanh();
    (c_b, [c_b * bonus_basis.psi_0, c_b * bonus_basis.psi_1])
}

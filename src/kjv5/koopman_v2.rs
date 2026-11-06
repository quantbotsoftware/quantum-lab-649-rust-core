//! `K2` & `K3` — Transportul Ireversibil Koopman `V_2 -> V_2` (`Q_dir`, `Q_anti`),
//! diagonalizarea simetrică Jacobi în `f64` cu protecție de interstițiu spectral (echivarianță `S_N`
//! și simetrie exactă `Q -> -Q`), și curentul antisimetric Schnakenberg în `V_1` (`PROTOCOL_KJV5_RUST_V2.md` §3.2–§3.3).

use serde::Serialize;

pub const TAU_K2: f64 = 160.0;
pub const LAMBDA_K2: f64 = 16.0;
pub const AMP_K2: f64 = 2.5;
pub const GAIN_K2: f64 = 4.0;
/// Pragul de nedegenerare a modului extrem (garantează echivarianța exactă sub `S_N` când există
/// simetrii reziduale pe pașii timpurii, păstrând totodată simetria exactă `Q -> -Q`).
pub const SPECTRAL_GAP_EPS: f64 = 1e-8;

pub const TAU_K3: f64 = 128.0;
pub const LAMBDA_K3: f64 = 16.0;
pub const K3_LAG_WEIGHTS: [f64; 3] = [0.55, 0.30, 0.15];

/// Produs scalar Frobenius `<A, B>_F = ∑_{i,j} A_{ij} B_{ij}`.
#[inline]
pub fn frob_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

#[inline]
pub fn frob_norm(a: &[f64]) -> f64 {
    frob_dot(a, a).sqrt()
}

/// Proiectorul ortogonal exact `Π_{V_2}(P)` pe subspațiul armonic ireductibil `V_2` (`S^{(N-2, 2)}`):
/// diagonală nulă și sume pe linii identic nule (`∑_j Q_{ij} = 0`), anihilând toate marginalele 1-corp.
pub fn project_v2(p: &[f64], n: usize) -> Vec<f64> {
    assert!(n >= 4 && p.len() == n * n);
    let mut row_sums = vec![0.0f64; n];
    let mut s_tot = 0.0f64;
    for i in 0..n {
        let row = &p[i * n..(i + 1) * n];
        let mut si = 0.0f64;
        for (j, &val) in row.iter().enumerate() {
            if i != j {
                si += val;
            }
        }
        row_sums[i] = si;
        s_tot += si;
    }
    let inv_nm2 = 1.0 / ((n - 2) as f64);
    let shift_all = s_tot / (((n - 1) * (n - 2)) as f64);
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        let si = row_sums[i];
        let out_row = &mut q[i * n..(i + 1) * n];
        let in_row = &p[i * n..(i + 1) * n];
        for j in 0..n {
            if i != j {
                out_row[j] = in_row[j] - (si + row_sums[j]) * inv_nm2 + shift_all;
            }
        }
    }
    q
}

/// Proiectorul ortogonal Frobenius `Π_{⊥ Φ}(Q) = Q - (<Q, Φ>_F / max(||Φ||_F^2, 1e-15)) Φ`.
pub fn project_perp_phi(q: &[f64], phi: &[f64]) -> Vec<f64> {
    debug_assert_eq!(q.len(), phi.len());
    let phi_n2 = frob_dot(phi, phi);
    if phi_n2 <= 1e-15 {
        return q.to_vec();
    }
    let alpha = frob_dot(q, phi) / phi_n2;
    q.iter().zip(phi.iter()).map(|(&qv, &pv)| qv - alpha * pv).collect()
}

/// Scalarea netedă `O(N)`-invariantă `Q_tilde = (2.5 * tanh(4 ||Q||_F) / max(||Q||_F, 1e-12)) Q`.
pub fn scale_q_tilde(q: &[f64]) -> Vec<f64> {
    let norm = frob_norm(q);
    if norm < 1e-12 {
        return vec![0.0f64; q.len()];
    }
    let factor = AMP_K2 * (GAIN_K2 * norm).tanh() / norm;
    q.iter().map(|&v| v * factor).collect()
}

/// Modurile spectrale extreme `(λ_+, u_+)` (`λ_+ >= 0`) și `(λ_-, u_-)` (`λ_- <= 0`) ale unei matrici `Q_tilde`.
#[derive(Debug, Clone, Serialize)]
pub struct V2Modes {
    pub lambda_pos: f64,
    pub u_pos: Vec<f64>,
    pub lambda_neg: f64,
    pub u_neg: Vec<f64>,
    pub frob_norm_tilde: f64,
    pub top_gap: f64,
    pub bottom_gap: f64,
}

impl V2Modes {
    pub fn zero(n: usize) -> Self {
        Self {
            lambda_pos: 0.0,
            u_pos: vec![0.0; n],
            lambda_neg: 0.0,
            u_neg: vec![0.0; n],
            frob_norm_tilde: 0.0,
            top_gap: 0.0,
            bottom_gap: 0.0,
        }
    }
}

/// Diagonalizare simetrică Jacobi ciclică cu prag în `f64`, urmată de extragerea modurilor extreme simple.
pub fn jacobi_extremal_modes(q_tilde: &[f64], n: usize) -> V2Modes {
    assert_eq!(q_tilde.len(), n * n);
    let fnorm = frob_norm(q_tilde);
    if fnorm < 1e-13 || n < 2 {
        return V2Modes::zero(n);
    }

    let mut a = q_tilde.to_vec();
    // Symmetrize exact input
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = avg;
            a[j * n + i] = avg;
        }
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let tol = 1e-15 * (fnorm + 1e-30);
    for _sweep in 0..50 {
        let mut off2 = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i * n + j];
                off2 += aij * aij;
            }
        }
        if (2.0 * off2).sqrt() <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-18 * fnorm {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        let new_rp = c * arp - s * arq;
                        let new_rq = s * arp + c * arq;
                        a[r * n + p] = new_rp;
                        a[p * n + r] = new_rp;
                        a[r * n + q] = new_rq;
                        a[q * n + r] = new_rq;
                    }
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]));

    let extract_col = |col: usize| -> Vec<f64> {
        let mut u: Vec<f64> = (0..n).map(|r| v[r * n + col]).collect();
        // Proiectăm orice reziduu numeric ~1e-16 pe 1_N (deoarece Q 1_N = 0) și renormăm unitar
        let mean = u.iter().sum::<f64>() / (n as f64);
        for x in &mut u {
            *x -= mean;
        }
        let nrm = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-15 {
            for x in &mut u {
                *x /= nrm;
            }
        }
        u
    };

    let c_max = order[n - 1];
    let c_max2 = order[n - 2];
    let eval_max = a[c_max * n + c_max];
    let eval_max2 = a[c_max2 * n + c_max2];
    let top_gap = eval_max - eval_max2;

    let (lambda_pos, u_pos) = if eval_max > 1e-12 && top_gap > SPECTRAL_GAP_EPS {
        (eval_max, extract_col(c_max))
    } else {
        (0.0, vec![0.0; n])
    };

    let c_min = order[0];
    let c_min2 = order[1];
    let eval_min = a[c_min * n + c_min];
    let eval_min2 = a[c_min2 * n + c_min2];
    let bottom_gap = eval_min2 - eval_min;

    let (lambda_neg, u_neg) = if eval_min < -1e-12 && bottom_gap > SPECTRAL_GAP_EPS {
        (eval_min, extract_col(c_min))
    } else {
        (0.0, vec![0.0; n])
    };

    V2Modes { lambda_pos, u_pos, lambda_neg, u_neg, frob_norm_tilde: fnorm, top_gap, bottom_gap }
}

/// Calculează câmpul antisimetric centrat și normat `h^(anti)` din tensorul de corelație direcționată `C_to`.
pub fn schnakenberg_field_from_corr(c_to: &[f64], x_prev: &[f64], denom: f64, n: usize) -> Vec<f64> {
    assert_eq!(c_to.len(), n * n);
    assert_eq!(x_prev.len(), n);
    if denom <= 0.0 {
        return vec![0.0; n];
    }
    // J_1 = 0.5 * (C_to - C_to^T) / denom; h_raw_j = ((J_1)^T x_prev)_j = ∑_i J_{1,ij} x_prev_i
    let inv_2d = 0.5 / denom;
    let mut h_raw = vec![0.0f64; n];
    for i in 0..n {
        let xi = x_prev[i];
        if xi == 0.0 {
            continue;
        }
        for j in 0..n {
            let j_ij = (c_to[i * n + j] - c_to[j * n + i]) * inv_2d;
            h_raw[j] += j_ij * xi;
        }
    }
    normalize_centered_field(&h_raw)
}

/// Centrează exact la sumă nulă și normează neted `h / max(||h||_2/sqrt(n), 1e-12) * tanh(||h||_2)`.
pub fn normalize_centered_field(h_raw: &[f64]) -> Vec<f64> {
    let n = h_raw.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = h_raw.iter().sum::<f64>() / (n as f64);
    let mut h_tilde: Vec<f64> = h_raw.iter().map(|&v| v - mean).collect();
    let norm2 = h_tilde.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm2 < 1e-14 {
        return vec![0.0; n];
    }
    let rms = (norm2 / (n as f64).sqrt()).max(1e-12);
    let scale = norm2.tanh() / rms;
    for v in &mut h_tilde {
        *v *= scale;
    }
    // Re-centrare finală la zero de mașină
    let mean2 = h_tilde.iter().sum::<f64>() / (n as f64);
    for v in &mut h_tilde {
        *v -= mean2;
    }
    h_tilde
}

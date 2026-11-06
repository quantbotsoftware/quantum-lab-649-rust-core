//! Densitățile legii Fisher (Poisson condiționat) a modelelor H1/H3 — protocol S19 Q5.
//!
//! Legea pe k-submulțimi: P_w(S) = Π_{i∈S} w_i / e_k(w), cu w = exp(ln ŵ). Mărimi (toate exacte ca definiție, calculate
//! în binary64, cu erorile măsurate):
//! - e_0..e_k(w): polinoamele simetrice elementare, ACEEAȘI ordine a operațiilor ca `ip_core.esp`;
//! - π_i = P(i ∈ S) = w_i e_{k−1}(w_{−i}) / e_k(w), prin recursia `emite.inclusion` (e_j(w_{−i}) = e_j − w_i e_{j−1}(w_{−i}));
//! - D = KL(P_w ‖ U) = Σ π_i ln w_i − ln e_k(w) + ln C(n,k) (ca `ip_core.info_D`);
//! - Π_ij = P(i, j ∈ S) = w_i w_j e_{k−2}(w_{−i,−j}) / e_k(w), Π_ii = π_i (matricea de includere de ordinul 2);
//! - ρ = Π / k: operator densitate (hermitian în `Complex64`, urma 1, semidefinit pozitiv, fiind E[XXᵀ]/k pentru
//!   vectorul indicator X ∈ {0,1}ⁿ al submulțimii extrase SUB LEGEA MODELULUI);
//! - q = softmax(ln ŵ) pentru urna cu o bilă (k = 1).
//!
//! ρ descrie legea MODELULUI, nu urna reală (sub urna uniformă reală toate biletele au aceeași șansă).
//! `positive_claims_allowed=false`.

use num_complex::Complex64;
use serde::Serialize;

use super::spectral::frft::jacobi_eigh;
use super::spectral::{neumaier_sum, HermitianMatrix, SpectralError};

/// e_0..e_k(w) (aceeași buclă ca `ip_core.esp`: pentru fiecare i, j = k..1: e_j += w_i e_{j−1}).
pub fn esp(w: &[f64], k: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; k + 1];
    e[0] = 1.0;
    for &wi in w {
        for j in (1..=k).rev() {
            e[j] += wi * e[j - 1];
        }
    }
    e
}

/// ln C(n, k) din coeficientul binomial exact (u128).
pub fn ln_comb(n: usize, k: usize) -> f64 {
    super::spectral::admissible::ln_binomial(n, k)
}

/// e_0..e_{k−1}(w_{−i}) prin recursia descendentă (ca `emite.inclusion`).
fn esp_minus(e: &[f64], wi: f64, k: usize) -> Vec<f64> {
    let mut em = vec![0.0f64; k];
    if k == 0 {
        return em;
    }
    em[0] = 1.0;
    for j in 1..k {
        em[j] = e[j] - wi * em[j - 1];
    }
    em
}

/// π_i (recursia `emite.inclusion`).
pub fn inclusion(logw: &[f64], k: usize) -> Vec<f64> {
    let w: Vec<f64> = logw.iter().map(|v| v.exp()).collect();
    let e = esp(&w, k);
    w.iter()
        .map(|&wi| {
            let em = esp_minus(&e, wi, k);
            wi * em[k - 1] / e[k]
        })
        .collect()
}

/// D = KL(P_w ‖ U) (ca `ip_core.info_D`).
pub fn info_d(logw: &[f64], k: usize) -> f64 {
    let n = logw.len();
    let w: Vec<f64> = logw.iter().map(|v| v.exp()).collect();
    let e = esp(&w, k);
    let pi = inclusion(logw, k);
    neumaier_sum(pi.iter().zip(logw).map(|(p, l)| p * l)) - e[k].ln() + ln_comb(n, k)
}

/// Π (row-major n×n): Π_ii = π_i, Π_ij = w_i w_j e_{k−2}(w_{−i,−j}) / e_k(w).
pub fn pair_inclusion(logw: &[f64], k: usize) -> Vec<f64> {
    let n = logw.len();
    let w: Vec<f64> = logw.iter().map(|v| v.exp()).collect();
    let e = esp(&w, k);
    let pi = inclusion(logw, k);
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = pi[i];
        if k < 2 {
            continue;
        }
        let emi = esp_minus(&e, w[i], k); // e_0..e_{k−1}(w_{−i})
        for j in 0..n {
            if j == i {
                continue;
            }
            // e_m(w_{−i,−j}) = e_m(w_{−i}) − w_j e_{m−1}(w_{−i,−j}), m = 0..k−2
            let mut prev = 1.0f64;
            for &e_m in emi.iter().skip(1).take(k - 2) {
                prev = e_m - w[j] * prev;
            }
            m[i * n + j] = w[i] * w[j] * prev / e[k];
        }
    }
    m
}

/// q = softmax(ln ŵ) (ca în `emite.model_block` pentru k = 1: exp(lw − max) / Σ).
pub fn softmax(logw: &[f64]) -> Vec<f64> {
    let mx = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = logw.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = ex.iter().sum();
    ex.iter().map(|v| v / s).collect()
}

/// Operatorul densitate ρ = Π / k și diagnosticele lui.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DensityReport {
    pub n: usize,
    pub k: usize,
    /// ρ (row-major n×n, partea reală; partea imaginară este 0 prin construcție).
    pub rho: Vec<f64>,
    pub trace: f64,
    /// Tr ρ² (1/n ≤ puritate ≤ 1).
    pub purity: f64,
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
    pub hermiticity_residual: f64,
    /// max_i |Σ_{j≠i} Π_ij − (k−1) π_i|.
    pub row_identity_residual: f64,
    /// |Σ π − k|.
    pub pi_sum_residual: f64,
}

/// ρ ca matrice hermitiană `Complex64`.
pub fn density_matrix(logw: &[f64], k: usize) -> Result<HermitianMatrix, SpectralError> {
    let n = logw.len();
    if k == 0 || k >= n {
        return Err(SpectralError::InvalidSpec { n, k });
    }
    let p = pair_inclusion(logw, k);
    let mut rho = HermitianMatrix::zeros(n);
    for i in 0..n {
        for j in i..n {
            rho.set(i, j, Complex64::new(p[i * n + j] / k as f64, 0.0));
        }
    }
    Ok(rho)
}

/// Raportul complet al lui ρ (valorile proprii prin Jacobi).
pub fn density_report(logw: &[f64], k: usize) -> Result<DensityReport, SpectralError> {
    let n = logw.len();
    let rho = density_matrix(logw, k)?;
    let pmat = pair_inclusion(logw, k);
    let pi = inclusion(logw, k);
    let real: Vec<f64> = (0..n * n).map(|idx| rho.get(idx / n, idx % n).re).collect();
    let eig = jacobi_eigh(&real, n)?;
    let row_res = (0..n)
        .map(|i| (neumaier_sum((0..n).filter(|&j| j != i).map(|j| pmat[i * n + j])) - (k as f64 - 1.0) * pi[i]).abs())
        .fold(0.0, f64::max);
    Ok(DensityReport {
        n,
        k,
        trace: rho.trace(),
        purity: neumaier_sum(real.iter().map(|v| v * v)),
        min_eigenvalue: eig.values.iter().copied().fold(f64::INFINITY, f64::min),
        max_eigenvalue: eig.values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        hermiticity_residual: rho.hermiticity_residual(),
        row_identity_residual: row_res,
        pi_sum_residual: (neumaier_sum(pi.iter().copied()) - k as f64).abs(),
        rho: real,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forța brută: toate k-submulțimile, P(S) = Π w / Σ.
    fn brute(logw: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, f64) {
        let n = logw.len();
        let w: Vec<f64> = logw.iter().map(|v| v.exp()).collect();
        let mut subsets = Vec::new();
        for m in 0u32..(1 << n) {
            if m.count_ones() as usize == k {
                let p: f64 = (0..n).filter(|i| m >> i & 1 == 1).map(|i| w[i]).product();
                subsets.push((m, p));
            }
        }
        let z: f64 = neumaier_sum(subsets.iter().map(|s| s.1));
        let mut pi = vec![0.0; n];
        let mut pm = vec![0.0; n * n];
        let mut d = 0.0;
        let lnc = (subsets.len() as f64).ln();
        for &(m, p) in &subsets {
            let pr = p / z;
            let s: f64 = (0..n).filter(|i| m >> i & 1 == 1).map(|i| logw[i]).sum();
            d += pr * (s - z.ln() + lnc);
            for i in 0..n {
                if m >> i & 1 == 1 {
                    pi[i] += pr;
                    for j in 0..n {
                        if m >> j & 1 == 1 {
                            pm[i * n + j] += pr;
                        }
                    }
                }
            }
        }
        (pi, pm, d)
    }

    #[test]
    fn esp_matches_brute_force() {
        let w = [0.7, 1.3, 0.9, 1.1, 2.0, 0.5];
        let e = esp(&w, 4);
        assert_eq!(e[0], 1.0);
        assert!((e[1] - w.iter().sum::<f64>()).abs() <= 1e-12);
        let mut e3 = 0.0;
        for a in 0..6 {
            for b in a + 1..6 {
                for c in b + 1..6 {
                    e3 += w[a] * w[b] * w[c];
                }
            }
        }
        assert!((e[3] - e3).abs() <= 1e-12);
        assert!((ln_comb(10, 3) - 120f64.ln()).abs() == 0.0);
    }

    #[test]
    fn densities_match_enumeration() {
        for (n, k) in [(6usize, 2usize), (8, 3), (10, 4), (9, 5), (7, 1)] {
            let lw: Vec<f64> = (0..n).map(|i| ((i * 7 % 5) as f64 - 2.0) * 0.13).collect();
            let (pi_b, pm_b, d_b) = brute(&lw, k);
            let pi = inclusion(&lw, k);
            let pm = pair_inclusion(&lw, k);
            assert!(pi.iter().zip(&pi_b).all(|(a, b)| (a - b).abs() <= 1e-12), "n={n} k={k}");
            if k >= 2 {
                assert!(pm.iter().zip(&pm_b).all(|(a, b)| (a - b).abs() <= 1e-12), "n={n} k={k}");
            }
            assert!((info_d(&lw, k) - d_b).abs() <= 1e-12, "n={n} k={k}");
            if k >= 2 {
                let r = density_report(&lw, k).unwrap();
                assert!((r.trace - 1.0).abs() <= 1e-12);
                assert!(r.min_eigenvalue >= -1e-12, "{}", r.min_eigenvalue);
                assert_eq!(r.hermiticity_residual, 0.0);
                assert!(r.row_identity_residual <= 1e-12 && r.pi_sum_residual <= 1e-12);
                assert!(r.purity <= 1.0 + 1e-12 && r.purity >= 1.0 / n as f64 - 1e-12);
            }
        }
    }

    #[test]
    fn uniform_law_closed_form() {
        let (n, k) = (49usize, 6usize);
        let lw = vec![0.0; n];
        let pi = inclusion(&lw, k);
        assert!(pi.iter().all(|p| (p - k as f64 / n as f64).abs() <= 1e-12));
        let pm = pair_inclusion(&lw, k);
        let off = (k * (k - 1)) as f64 / (n * (n - 1)) as f64;
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { k as f64 / n as f64 } else { off };
                assert!((pm[i * n + j] - want).abs() <= 1e-12);
            }
        }
        assert!(info_d(&lw, k).abs() <= 1e-12);
        let q = softmax(&[0.0; 20]);
        assert!(q.iter().all(|v| (v - 0.05).abs() <= 1e-15));
        assert!(density_matrix(&lw, 0).is_err() && density_matrix(&lw, 49).is_err());
    }
}

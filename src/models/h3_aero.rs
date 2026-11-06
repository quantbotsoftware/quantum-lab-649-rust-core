//! H3 „AD” — dinamica aerului: înclinare netedă pe poziția de încărcare (schelet; protocol S18 §3, R7).
//!
//! Port structural al `tu4/PROTOCOL_TU4_IP_H3_AD.md` / `ip_core.logw_h3` la momentul curent:
//! b_m(i) = √(2/n) cos(π m (i − ½)/n), m = 1..K, K = 4 (ortonormale, ortogonale pe 1);
//! z_i = √((n−1)/n)(c_i − Np)/√(N p(1−p)); X²_s = Σ_m (b_m·z)²; θ̂_s = Σ_m b_m (b_m·θ̂);
//! λ_s = max(0, 1 − (K−2)/X²_s); ln ŵ = λ_s θ̂_s.
//! Diagnostice `Complex64`: spectrul DFT unitar al lui z pe etichete și energia modurilor în R.
//! Premisa fizică (încărcare ordonată, amestec incomplet) NU este verificată pentru urnele reale.

use num_complex::Complex64;

use super::h1_mass::{james_stein_lambda, theta_hat};
use super::spectral::{label_dft, neumaier_sum, CountState, HermitianMatrix, SpectralError, UrnSpec};

/// Numărul de moduri cosinus fixat în protocolul sigilat H3.
pub const K_AD: usize = 4;

/// Baza cosinus discretă (DCT-II ortonormală fără modul constant): `basis[m-1][i-1] = b_m(i)`.
pub fn cosine_basis(n: usize, modes: usize) -> Result<Vec<Vec<f64>>, SpectralError> {
    if modes == 0 || modes >= n {
        return Err(SpectralError::InvalidModes { modes, n });
    }
    let nf = n as f64;
    let amp = (2.0 / nf).sqrt();
    Ok((1..=modes)
        .map(|m| (1..=n).map(|i| amp * (std::f64::consts::PI * m as f64 * (i as f64 - 0.5) / nf).cos()).collect())
        .collect())
}

/// max |⟨b_a, b_b⟩ − δ_ab| (măsura erorii de rotunjire a bazei).
pub fn basis_gram_residual(basis: &[Vec<f64>]) -> f64 {
    let mut worst = 0.0f64;
    for (a, ba) in basis.iter().enumerate() {
        for (b, bb) in basis.iter().enumerate() {
            let dot = neumaier_sum(ba.iter().zip(bb).map(|(x, y)| x * y));
            worst = worst.max((dot - if a == b { 1.0 } else { 0.0 }).abs());
        }
    }
    worst
}

/// max_m |Σ_i b_m(i)| (ortogonalitatea pe vectorul constant).
pub fn basis_constant_residual(basis: &[Vec<f64>]) -> f64 {
    basis.iter().map(|b| neumaier_sum(b.iter().copied()).abs()).fold(0.0, f64::max)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    neumaier_sum(a.iter().zip(b).map(|(x, y)| x * y))
}

/// Starea H3 evaluată pe numărările curente.
#[derive(Debug, Clone, PartialEq)]
pub struct H3Aero {
    pub spec: UrnSpec,
    pub draws: u64,
    pub modes: usize,
    pub basis: Vec<Vec<f64>>,
    pub z: Vec<f64>,
    /// b_m · z, m = 1..K.
    pub mode_scores: Vec<f64>,
    pub x2_s: f64,
    pub theta: Vec<f64>,
    pub theta_s: Vec<f64>,
    pub lambda_s: f64,
    pub log_w: Vec<f64>,
}

impl H3Aero {
    pub fn from_counts(state: &CountState, modes: usize) -> Result<H3Aero, SpectralError> {
        let theta = theta_hat(state)?;
        let spec = state.spec();
        let basis = cosine_basis(spec.n, modes)?;
        let (n, p, big_n) = (spec.n as f64, spec.p(), state.draws() as f64);
        let scale = ((n - 1.0) / n).sqrt();
        let sd = (big_n * p * (1.0 - p)).sqrt();
        let z: Vec<f64> = (0..spec.n).map(|i| scale * (state.count(i) as f64 - big_n * p) / sd).collect();
        let mode_scores: Vec<f64> = basis.iter().map(|b| dot(&z, b)).collect();
        let x2_s = neumaier_sum(mode_scores.iter().map(|s| s * s));
        let coef: Vec<f64> = basis.iter().map(|b| dot(&theta, b)).collect();
        let theta_s: Vec<f64> =
            (0..spec.n).map(|i| neumaier_sum(basis.iter().zip(&coef).map(|(b, c)| b[i] * c))).collect();
        let lambda_s = james_stein_lambda(x2_s, modes as f64 - 2.0);
        let log_w = theta_s.iter().map(|t| lambda_s * t).collect();
        Ok(H3Aero { spec, draws: state.draws(), modes, basis, z, mode_scores, x2_s, theta, theta_s, lambda_s, log_w })
    }

    /// Spectrul DFT unitar al lui z pe etichete (diagnostic `Complex64`).
    pub fn label_spectrum(&self) -> Vec<Complex64> {
        label_dft(&self.z)
    }

    /// b_mᵀ R b_m pentru fiecare mod (energia corelației pe perechi în subspațiul neted).
    pub fn mode_correlation(&self, r: &HermitianMatrix) -> Result<Vec<f64>, SpectralError> {
        self.basis.iter().map(|b| r.real_quadratic_form(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::h1_mass::H1Mass;
    use crate::models::spectral::tests::{full_design, synthetic};
    use crate::models::spectral::{pair_correlation, parseval_relative_residual};

    #[test]
    fn basis_is_orthonormal_and_orthogonal_to_constant() {
        for n in [20usize, 40, 45, 49] {
            let b = cosine_basis(n, K_AD).unwrap();
            assert_eq!((b.len(), b[0].len()), (K_AD, n));
            assert!(basis_gram_residual(&b) <= 1e-12, "n={n}");
            assert!(basis_constant_residual(&b) <= 1e-12, "n={n}");
        }
        assert!(cosine_basis(4, 0).is_err() && cosine_basis(4, 4).is_err());
    }

    #[test]
    fn uniform_design_gives_zero() {
        let h = H3Aero::from_counts(&full_design(6, 3), 2).unwrap();
        assert_eq!((h.x2_s, h.lambda_s), (0.0, 0.0));
        assert!(h.log_w.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn projection_identities_on_synthetic_draws() {
        let s = synthetic(49, 6, 600, 2026);
        let h3 = H3Aero::from_counts(&s, K_AD).unwrap();
        let h1 = H1Mass::from_counts(&s).unwrap();
        // Σ z² = X²(H1) și inegalitatea lui Bessel X²_s <= X²
        let ez = neumaier_sum(h3.z.iter().map(|v| v * v));
        assert!((ez - h1.x2).abs() <= 1e-12 * h1.x2.max(1.0));
        assert!(h3.x2_s <= h1.x2 * (1.0 + 1e-12));
        assert_eq!(h3.theta, h1.theta);
        // θ̂_s este o proiecție: idempotentă, iar reziduul este ortogonal pe fiecare mod
        let again: Vec<f64> =
            (0..49).map(|i| neumaier_sum(h3.basis.iter().map(|b| b[i] * dot(&h3.theta_s, b)))).collect();
        assert!(again.iter().zip(&h3.theta_s).all(|(a, b)| (a - b).abs() <= 1e-12));
        let resid: Vec<f64> = h3.theta.iter().zip(&h3.theta_s).map(|(a, b)| a - b).collect();
        assert!(h3.basis.iter().all(|b| dot(&resid, b).abs() <= 1e-12));
        assert!((0.0..=1.0).contains(&h3.lambda_s));
        // diagnostice Complex64
        let spec = h3.label_spectrum();
        assert!(parseval_relative_residual(&h3.z, &spec) <= 1e-12);
        assert!(spec[0].norm() <= 1e-9); // Σ z = 0 ⇒ componenta DC nulă
        let r = pair_correlation(&s).unwrap();
        let mc = h3.mode_correlation(&r).unwrap();
        assert!(mc.len() == K_AD && mc.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn smooth_tilt_is_kept_by_projection() {
        // Numărări construite cu o pantă netedă pe etichete (n = 20, k = 1): mod 1 dominant ⇒ λ_s > 0.
        let mut s = CountState::new(UrnSpec::new(20, 1).unwrap());
        for l in 1..=20u8 {
            for _ in 0..(60 - 2 * u32::from(l)) {
                s.push(&[l]).unwrap();
            }
        }
        let h = H3Aero::from_counts(&s, K_AD).unwrap();
        assert!(h.lambda_s > 0.9, "{}", h.lambda_s);
        assert!(h.mode_scores[0].abs() > h.mode_scores[1..].iter().map(|v| v.abs()).fold(0.0, f64::max));
    }
}

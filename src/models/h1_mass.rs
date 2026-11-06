//! H1 „MA” — asimetria de masă staționară (schelet; protocol S18 §3, decizia R7).
//!
//! Port structural al `tu4/PROTOCOL_TU4_IP_H1_MA.md` / `ip_core.logw_h1` la momentul curent (N2 = N):
//! θ̂_i = (n−1)/n · (c_i/N − p)/(p(1−p)); X² = (n−1)/n · Σ(c_i − Np)²/(N p(1−p));
//! λ = max(0, 1 − (n−3)/X²) (X² = 0 ⇒ λ = 0); ln ŵ = λ θ̂ (contracție James–Stein spre urna uniformă).
//! Matricea de corelație pe perechi (hermitiană, `Complex64`) însoțește modelul ca DIAGNOSTIC.
//! Fără bilete, emisii sau afirmații predictive (`positive_claims_allowed=false`).

use super::spectral::{neumaier_sum, pair_correlation, CountState, HermitianMatrix, SpectralError, UrnSpec};

/// Factorul de contracție James–Stein, cu λ = 0 când X² nu este pozitiv și finit.
pub(crate) fn james_stein_lambda(x2: f64, penalty: f64) -> f64 {
    if x2 > 0.0 && x2.is_finite() {
        (1.0 - penalty / x2).max(0.0)
    } else {
        0.0
    }
}

/// θ̂ comun H1/H3 (aceeași ordine a operațiilor ca în `ip_core.theta_x2`).
pub(crate) fn theta_hat(state: &CountState) -> Result<Vec<f64>, SpectralError> {
    if state.draws() == 0 {
        return Err(SpectralError::NoDraws);
    }
    let spec = state.spec();
    UrnSpec::new(spec.n, spec.k)?;
    let (n, p, big_n) = (spec.n as f64, spec.p(), state.draws() as f64);
    let shrink = (n - 1.0) / n;
    Ok((0..spec.n).map(|i| shrink * (state.count(i) as f64 / big_n - p) / (p * (1.0 - p))).collect())
}

/// Starea H1 evaluată pe numărările curente.
#[derive(Debug, Clone, PartialEq)]
pub struct H1Mass {
    pub spec: UrnSpec,
    pub draws: u64,
    pub theta: Vec<f64>,
    pub x2: f64,
    pub lambda: f64,
    pub log_w: Vec<f64>,
}

impl H1Mass {
    pub fn from_counts(state: &CountState) -> Result<H1Mass, SpectralError> {
        let theta = theta_hat(state)?;
        let spec = state.spec();
        let (n, p, big_n) = (spec.n as f64, spec.p(), state.draws() as f64);
        let dev2 = neumaier_sum((0..spec.n).map(|i| {
            let d = state.count(i) as f64 - big_n * p;
            d * d
        }));
        let x2 = (n - 1.0) / n * dev2 / (big_n * p * (1.0 - p));
        let lambda = james_stein_lambda(x2, n - 3.0);
        let log_w = theta.iter().map(|t| lambda * t).collect();
        Ok(H1Mass { spec, draws: state.draws(), theta, x2, lambda, log_w })
    }
}

/// Diagnosticul de corelație atașat H1.
#[derive(Debug, Clone, PartialEq)]
pub struct MassCorrelation {
    pub matrix: HermitianMatrix,
    pub hermiticity_residual: f64,
    pub trace: f64,
    /// θ̂ᵀ R θ̂ / θ̂ᵀ θ̂ (None dacă θ̂ = 0).
    pub theta_rayleigh: Option<f64>,
}

/// Construiește R pentru aceleași numărări ca modelul și coeficientul Rayleigh al lui θ̂.
pub fn mass_correlation(state: &CountState, model: &H1Mass) -> Result<MassCorrelation, SpectralError> {
    if state.spec() != model.spec || state.draws() != model.draws {
        return Err(SpectralError::DimensionMismatch { expected: model.spec.n, found: state.spec().n });
    }
    let matrix = pair_correlation(state)?;
    let norm2 = neumaier_sum(model.theta.iter().map(|t| t * t));
    let theta_rayleigh = if norm2 > 0.0 { Some(matrix.real_quadratic_form(&model.theta)? / norm2) } else { None };
    Ok(MassCorrelation {
        hermiticity_residual: matrix.hermiticity_residual(),
        trace: matrix.trace(),
        theta_rayleigh,
        matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::spectral::tests::{full_design, synthetic};

    #[test]
    fn uniform_design_gives_zero_shrinkage() {
        let s = full_design(6, 3);
        let h = H1Mass::from_counts(&s).unwrap();
        assert_eq!((h.x2, h.lambda), (0.0, 0.0));
        assert!(h.theta.iter().chain(&h.log_w).all(|v| *v == 0.0));
        let c = mass_correlation(&s, &h).unwrap();
        assert_eq!((c.hermiticity_residual, c.theta_rayleigh), (0.0, None));
    }

    #[test]
    fn closed_form_fixture() {
        // n = 5, k = 2, extrageri {1,2},{1,3},{1,4},{2,3}: N = 4, c = (3,2,2,1,0), p = 2/5.
        // θ̂ = (7/6, 1/3, 1/3, −1/2, −4/3); X² = 13/3; λ = 1 − 2/(13/3) = 7/13; ln ŵ = λ θ̂.
        let mut s = CountState::new(UrnSpec::new(5, 2).unwrap());
        for d in [[1, 2], [1, 3], [1, 4], [2, 3]] {
            s.push(&d).unwrap();
        }
        let h = H1Mass::from_counts(&s).unwrap();
        let theta = [7.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, -0.5, -4.0 / 3.0];
        assert!((h.x2 - 13.0 / 3.0).abs() <= 1e-12);
        assert!((h.lambda - 7.0 / 13.0).abs() <= 1e-12);
        for ((got, lw), want) in h.theta.iter().zip(&h.log_w).zip(theta) {
            assert!((got - want).abs() <= 1e-12);
            assert!((lw - 7.0 / 13.0 * want).abs() <= 1e-12);
        }
        assert!(neumaier_sum(h.theta.iter().copied()).abs() <= 1e-12); // Σ θ̂ = 0 (Σ c = kN)
    }

    #[test]
    fn small_signal_is_fully_shrunk() {
        // X² < n − 3 ⇒ λ = 0 (contracție completă spre uniform).
        let mut s = CountState::new(UrnSpec::new(10, 1).unwrap());
        for l in 1..=10u8 {
            s.push(&[l]).unwrap();
        }
        s.push(&[1]).unwrap();
        let h = H1Mass::from_counts(&s).unwrap();
        assert!(h.x2 > 0.0 && h.x2 < 7.0);
        assert_eq!(h.lambda, 0.0);
        assert!(h.log_w.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn errors_and_correlation_diagnostics() {
        assert_eq!(H1Mass::from_counts(&CountState::new(UrnSpec::new(49, 6).unwrap())), Err(SpectralError::NoDraws));
        let s = synthetic(49, 6, 400, 11);
        let h = H1Mass::from_counts(&s).unwrap();
        let c = mass_correlation(&s, &h).unwrap();
        assert_eq!(c.hermiticity_residual, 0.0);
        assert!(c.theta_rayleigh.unwrap().is_finite());
        let other = synthetic(49, 6, 401, 11);
        assert!(mass_correlation(&other, &h).is_err());
        assert!((0.0..=1.0).contains(&h.lambda));
    }
}

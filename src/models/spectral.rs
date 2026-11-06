//! Utilitare spectrale comune pe `num_complex::Complex64`.
//!
//! Onestitate numerică (decizia R6): `Complex64` = pereche IEEE-754 binary64. Erorile de rotunjire
//! sunt MĂRGINITE și MĂSURATE (reziduul hermitic, Parseval), nu eliminate. Numărările de bază sunt
//! întregi exacte (`u64`); sumele în virgulă mobilă folosesc compensarea Neumaier; unghiurile DFT se
//! reduc exact modulo n în aritmetică întreagă înainte de trigonometrie.

use std::fmt;

use num_complex::Complex64;

use crate::parser::{Extraction, Game};

// --- Adaos S19 (tu4/PROTOCOL_SUPERQUANTUM_S19.md, Q6/Q7): numai declarații de submodul și reexporturi. ---
pub mod admissible;
pub mod frft;

pub use admissible::{admissible, enumerate_all, AdmState, Admissible, Enumeration, ExactSum, LinearTable, Mask};
pub use frft::{dft_matrix, jacobi_eigh, Frft, SymEigen};

/// Erorile modulelor de model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectralError {
    InvalidSpec { n: usize, k: usize },
    NoDraws,
    LabelOutOfRange { label: u8, n: usize },
    WrongBallCount { expected: usize, found: usize },
    DuplicateLabel { label: u8 },
    InvalidModes { modes: usize, n: usize },
    DimensionMismatch { expected: usize, found: usize },
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpectralError::InvalidSpec { n, k } => {
                write!(f, "urnă invalidă n={n}, k={k} (cerut 4 <= n <= 255, 1 <= k < n)")
            }
            SpectralError::NoDraws => f.write_str("nicio extragere (N = 0)"),
            SpectralError::LabelOutOfRange { label, n } => write!(f, "eticheta {label} în afara 1..{n}"),
            SpectralError::WrongBallCount { expected, found } => write!(f, "{found} bile, așteptat {expected}"),
            SpectralError::DuplicateLabel { label } => write!(f, "eticheta {label} repetată"),
            SpectralError::InvalidModes { modes, n } => {
                write!(f, "K={modes} moduri invalid pentru n={n} (cerut 1 <= K < n)")
            }
            SpectralError::DimensionMismatch { expected, found } => {
                write!(f, "dimensiune {found}, așteptat {expected}")
            }
        }
    }
}

impl std::error::Error for SpectralError {}

/// Suma compensată Neumaier (Kahan–Babuška): eroarea nu crește liniar cu numărul de termeni.
pub fn neumaier_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Suma compensată pe componente pentru numere complexe.
pub fn neumaier_sum_complex<I: IntoIterator<Item = Complex64>>(values: I) -> Complex64 {
    let (mut re, mut ci_re, mut im, mut ci_im) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let step = |s: &mut f64, c: &mut f64, x: f64| {
        let t = *s + x;
        if s.abs() >= x.abs() {
            *c += (*s - t) + x;
        } else {
            *c += (x - t) + *s;
        }
        *s = t;
    };
    for z in values {
        step(&mut re, &mut ci_re, z.re);
        step(&mut im, &mut ci_im, z.im);
    }
    Complex64::new(re + ci_re, im + ci_im)
}

/// Urna: n etichete 1..n, k bile distincte pe extragere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UrnSpec {
    pub n: usize,
    pub k: usize,
}

impl UrnSpec {
    /// Domeniul în care formulele sigilate sunt definite (n − 3 > 0 pentru H1): 4 <= n <= 255, 1 <= k < n.
    pub fn new(n: usize, k: usize) -> Result<UrnSpec, SpectralError> {
        if !(4..=255).contains(&n) || k == 0 || k >= n {
            return Err(SpectralError::InvalidSpec { n, k });
        }
        Ok(UrnSpec { n, k })
    }

    /// Urna principală a jocului (49/6, 40/6, 45/5).
    pub fn main(game: Game) -> UrnSpec {
        UrnSpec { n: usize::from(game.main_max()), k: game.main_balls() }
    }

    /// Urna bilei suplimentare (Joker: 20/1).
    pub fn bonus(game: Game) -> Option<UrnSpec> {
        game.bonus_max().map(|m| UrnSpec { n: usize::from(m), k: 1 })
    }

    /// p = k/n.
    pub fn p(&self) -> f64 {
        self.k as f64 / self.n as f64
    }
}

/// Numărări exacte: N extrageri, matricea simetrică P (n×n) a aparițiilor pe perechi, cu P_ii = c_i.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountState {
    spec: UrnSpec,
    draws: u64,
    pairs: Vec<u64>,
}

impl CountState {
    pub fn new(spec: UrnSpec) -> CountState {
        CountState { spec, draws: 0, pairs: vec![0; spec.n * spec.n] }
    }

    /// Adaugă o extragere (etichete 1..n, exact k, distincte); altfel eroare fără modificarea stării.
    pub fn push(&mut self, labels: &[u8]) -> Result<(), SpectralError> {
        let n = self.spec.n;
        if labels.len() != self.spec.k {
            return Err(SpectralError::WrongBallCount { expected: self.spec.k, found: labels.len() });
        }
        let mut seen = [false; 256];
        for &l in labels {
            if l == 0 || usize::from(l) > n {
                return Err(SpectralError::LabelOutOfRange { label: l, n });
            }
            if std::mem::replace(&mut seen[usize::from(l)], true) {
                return Err(SpectralError::DuplicateLabel { label: l });
            }
        }
        for &a in labels {
            for &b in labels {
                self.pairs[(usize::from(a) - 1) * n + (usize::from(b) - 1)] += 1;
            }
        }
        self.draws += 1;
        Ok(())
    }

    /// Starea urnei principale din extragerile jocului `game` (celelalte jocuri sunt ignorate).
    pub fn from_main(game: Game, records: &[Extraction]) -> Result<CountState, SpectralError> {
        let mut s = CountState::new(UrnSpec::main(game));
        for r in records.iter().filter(|r| r.game == game) {
            s.push(r.numbers())?;
        }
        Ok(s)
    }

    /// Starea bilei suplimentare (numai Joker); `None` pentru jocurile fără bilă.
    pub fn from_bonus(game: Game, records: &[Extraction]) -> Result<Option<CountState>, SpectralError> {
        let Some(spec) = UrnSpec::bonus(game) else { return Ok(None) };
        let mut s = CountState::new(spec);
        for r in records.iter().filter(|r| r.game == game) {
            if let Some(b) = r.bonus {
                s.push(&[b])?;
            }
        }
        Ok(Some(s))
    }

    pub fn spec(&self) -> UrnSpec {
        self.spec
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// c_i pentru eticheta i (indice 0-based).
    pub fn count(&self, i: usize) -> u64 {
        self.pairs[i * self.spec.n + i]
    }

    pub fn counts(&self) -> Vec<u64> {
        (0..self.spec.n).map(|i| self.count(i)).collect()
    }

    /// P_ij (indici 0-based).
    pub fn pair(&self, i: usize, j: usize) -> u64 {
        self.pairs[i * self.spec.n + j]
    }
}

/// Matrice hermitiană densă n×n pe `Complex64`; hermiticitatea este impusă la scriere.
#[derive(Debug, Clone, PartialEq)]
pub struct HermitianMatrix {
    n: usize,
    data: Vec<Complex64>,
}

impl HermitianMatrix {
    pub fn zeros(n: usize) -> HermitianMatrix {
        HermitianMatrix { n, data: vec![Complex64::new(0.0, 0.0); n * n] }
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.data[i * self.n + j]
    }

    /// Scrie a_ij = v și a_ji = conj(v); pe diagonală se păstrează numai partea reală.
    pub fn set(&mut self, i: usize, j: usize, v: Complex64) {
        if i == j {
            self.data[i * self.n + i] = Complex64::new(v.re, 0.0);
        } else {
            self.data[i * self.n + j] = v;
            self.data[j * self.n + i] = v.conj();
        }
    }

    /// max_{i,j} |a_ij − conj(a_ji)| (0 exact prin construcție; verificat, nu presupus).
    pub fn hermiticity_residual(&self) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..self.n {
            for j in 0..self.n {
                worst = worst.max((self.get(i, j) - self.get(j, i).conj()).norm());
            }
        }
        worst
    }

    /// Urma (reală pentru o matrice hermitiană), sumă compensată.
    pub fn trace(&self) -> f64 {
        neumaier_sum((0..self.n).map(|i| self.get(i, i).re))
    }

    fn check_dim(&self, len: usize) -> Result<(), SpectralError> {
        if len != self.n {
            return Err(SpectralError::DimensionMismatch { expected: self.n, found: len });
        }
        Ok(())
    }

    /// y = A x.
    pub fn mat_vec(&self, x: &[Complex64]) -> Result<Vec<Complex64>, SpectralError> {
        self.check_dim(x.len())?;
        Ok((0..self.n).map(|i| neumaier_sum_complex((0..self.n).map(|j| self.get(i, j) * x[j]))).collect())
    }

    /// x† A x (real până la rotunjire pentru A hermitiană).
    pub fn quadratic_form(&self, x: &[Complex64]) -> Result<Complex64, SpectralError> {
        let ax = self.mat_vec(x)?;
        Ok(neumaier_sum_complex(x.iter().zip(&ax).map(|(xi, yi)| xi.conj() * yi)))
    }

    /// xᵀ A x pentru x real.
    pub fn real_quadratic_form(&self, x: &[f64]) -> Result<f64, SpectralError> {
        let xc: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        Ok(self.quadratic_form(&xc)?.re)
    }
}

/// Corelația normalizată pe perechi: R_ij = (P_ij/N − (c_i/N)(c_j/N)) / (p(1−p)), încorporată hermitian
/// în `Complex64` (parte imaginară 0). Sub urna uniformă fără întoarcere: E R_ii → 1, E R_ij → −1/(n−1).
pub fn pair_correlation(state: &CountState) -> Result<HermitianMatrix, SpectralError> {
    if state.draws() == 0 {
        return Err(SpectralError::NoDraws);
    }
    let spec = state.spec();
    let big_n = state.draws() as f64;
    let p = spec.p();
    let q = p * (1.0 - p);
    let mut m = HermitianMatrix::zeros(spec.n);
    for i in 0..spec.n {
        let fi = state.count(i) as f64 / big_n;
        for j in i..spec.n {
            let fj = state.count(j) as f64 / big_n;
            let pij = state.pair(i, j) as f64 / big_n;
            m.set(i, j, Complex64::new((pij - fi * fj) / q, 0.0));
        }
    }
    Ok(m)
}

/// DFT unitară pe etichete: Z_m = n^{-1/2} Σ_i z_i e^{−2π i m i / n} (i, m 0-based). Indicele
/// unghiului (m·i) mod n se reduce exact în întregi, apoi se folosește un tabel de rotații.
pub fn label_dft(z: &[f64]) -> Vec<Complex64> {
    let n = z.len();
    if n == 0 {
        return Vec::new();
    }
    let twiddle: Vec<Complex64> = (0..n)
        .map(|r| {
            let ang = -2.0 * std::f64::consts::PI * (r as f64) / (n as f64);
            Complex64::new(ang.cos(), ang.sin())
        })
        .collect();
    let scale = 1.0 / (n as f64).sqrt();
    (0..n)
        .map(|m| neumaier_sum_complex(z.iter().enumerate().map(|(i, &zi)| twiddle[(m * i) % n] * zi)) * scale)
        .collect()
}

/// |Σ z_i² − Σ |Z_m|²| / Σ z_i² (absolut dacă energia este 0) — măsura erorii de rotunjire a DFT.
pub fn parseval_relative_residual(z: &[f64], spectrum: &[Complex64]) -> f64 {
    let e_time = neumaier_sum(z.iter().map(|v| v * v));
    let e_freq = neumaier_sum(spectrum.iter().map(|c| c.norm_sqr()));
    if e_time == 0.0 {
        e_freq.abs()
    } else {
        (e_time - e_freq).abs() / e_time
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Toate cele C(n, k) combinații (urnă „perfect echilibrată”).
    pub(crate) fn full_design(n: usize, k: usize) -> CountState {
        fn rec(start: u8, n: u8, k: usize, cur: &mut Vec<u8>, s: &mut CountState) {
            if cur.len() == k {
                s.push(cur).unwrap();
                return;
            }
            for l in start..=n {
                cur.push(l);
                rec(l + 1, n, k, cur, s);
                cur.pop();
            }
        }
        let mut s = CountState::new(UrnSpec::new(n, k).unwrap());
        rec(1, n as u8, k, &mut Vec::new(), &mut s);
        s
    }

    /// Extrageri sintetice deterministe (LCG fix) — NU provin din arhivă.
    pub(crate) fn synthetic(n: usize, k: usize, draws: usize, seed: u64) -> CountState {
        let mut s = CountState::new(UrnSpec::new(n, k).unwrap());
        let mut x = seed;
        for _ in 0..draws {
            let mut d: Vec<u8> = Vec::with_capacity(k);
            while d.len() < k {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let l = ((x >> 33) % n as u64) as u8 + 1;
                if !d.contains(&l) {
                    d.push(l);
                }
            }
            s.push(&d).unwrap();
        }
        s
    }

    #[test]
    fn neumaier_beats_naive_on_cancellation() {
        let v = [1.0, 1e100, 1.0, -1e100];
        assert_eq!(neumaier_sum(v), 2.0);
        assert_eq!(neumaier_sum_complex(v.iter().map(|&x| Complex64::new(x, -x))), Complex64::new(2.0, -2.0));
    }

    #[test]
    fn count_state_validation_and_counts() {
        let mut s = CountState::new(UrnSpec::new(5, 2).unwrap());
        assert_eq!(s.push(&[1]), Err(SpectralError::WrongBallCount { expected: 2, found: 1 }));
        assert_eq!(s.push(&[1, 6]), Err(SpectralError::LabelOutOfRange { label: 6, n: 5 }));
        assert_eq!(s.push(&[0, 2]), Err(SpectralError::LabelOutOfRange { label: 0, n: 5 }));
        assert_eq!(s.push(&[3, 3]), Err(SpectralError::DuplicateLabel { label: 3 }));
        assert_eq!(s.draws(), 0);
        for d in [[1, 2], [1, 3], [1, 4], [2, 3]] {
            s.push(&d).unwrap();
        }
        assert_eq!((s.draws(), s.counts()), (4, vec![3, 2, 2, 1, 0]));
        assert_eq!((s.pair(0, 1), s.pair(1, 0), s.pair(1, 2), s.pair(0, 4)), (1, 1, 1, 0));
        assert!(UrnSpec::new(3, 1).is_err() && UrnSpec::new(10, 10).is_err() && UrnSpec::new(10, 0).is_err());
    }

    #[test]
    fn spec_from_games_and_bonus_state() {
        assert_eq!(UrnSpec::main(Game::Loto649), UrnSpec { n: 49, k: 6 });
        assert_eq!(UrnSpec::main(Game::Loto540), UrnSpec { n: 40, k: 6 });
        assert_eq!(UrnSpec::main(Game::Joker), UrnSpec { n: 45, k: 5 });
        assert_eq!(UrnSpec::bonus(Game::Joker), Some(UrnSpec { n: 20, k: 1 }));
        assert_eq!(UrnSpec::bonus(Game::Loto649), None);
        let text = concat!(
            r#"{"game":"joker","draw_date":"2000-01-02","numbers":[1,2,3,4,5],"joker":7}"#,
            "\n",
            r#"{"game":"joker","draw_date":"2000-01-09","numbers":[6,2,3,4,5],"joker":7}"#,
            "\n",
            r#"{"game":"649","draw_date":"2000-01-09","numbers":[6,2,3,4,5,1]}"#
        );
        let recs = crate::parser::parse_bytes(text.as_bytes(), None).records;
        let main = CountState::from_main(Game::Joker, &recs).unwrap();
        assert_eq!((main.draws(), main.count(1), main.count(0), main.count(5)), (2, 2, 1, 1));
        let bonus = CountState::from_bonus(Game::Joker, &recs).unwrap().unwrap();
        assert_eq!((bonus.draws(), bonus.count(6)), (2, 2));
        assert!(CountState::from_bonus(Game::Loto649, &recs).unwrap().is_none());
        assert_eq!(CountState::from_main(Game::Loto649, &recs).unwrap().draws(), 1);
    }

    #[test]
    fn pair_correlation_on_balanced_design_is_exact() {
        // n = 6, k = 3, toate cele 20 de combinații: R_ii = 1, R_ij = −1/(n−1) = −0.2.
        let s = full_design(6, 3);
        let r = pair_correlation(&s).unwrap();
        assert_eq!(r.hermiticity_residual(), 0.0);
        for i in 0..6 {
            for j in 0..6 {
                let want = if i == j { 1.0 } else { -0.2 };
                assert!((r.get(i, j).re - want).abs() <= 1e-12 && r.get(i, j).im == 0.0);
            }
        }
        assert!((r.trace() - 6.0).abs() <= 1e-12);
        // vectorul constant este în nucleu: 1ᵀ R 1 = n·1 + n(n−1)(−1/(n−1)) = 0
        assert!(r.real_quadratic_form(&[1.0; 6]).unwrap().abs() <= 1e-12);
        assert!(pair_correlation(&CountState::new(UrnSpec::new(6, 3).unwrap())).is_err());
        assert_eq!(r.real_quadratic_form(&[1.0; 5]), Err(SpectralError::DimensionMismatch { expected: 6, found: 5 }));
    }

    #[test]
    fn pair_correlation_trace_matches_definition() {
        let s = synthetic(49, 6, 500, 7);
        let r = pair_correlation(&s).unwrap();
        let (p, nn) = (6.0 / 49.0, 500.0);
        let want = neumaier_sum((0..49).map(|i| {
            let f = s.count(i) as f64 / nn;
            (f - f * f) / (p * (1.0 - p))
        }));
        assert_eq!(r.hermiticity_residual(), 0.0);
        assert!((r.trace() - want).abs() <= 1e-12 * want.abs().max(1.0));
    }

    #[test]
    fn hermitian_complex_quadratic_form_is_real() {
        let mut a = HermitianMatrix::zeros(3);
        a.set(0, 0, Complex64::new(2.0, 5.0)); // partea imaginară a diagonalei este eliminată
        a.set(0, 1, Complex64::new(1.0, 2.0));
        a.set(1, 2, Complex64::new(-0.5, 0.25));
        a.set(2, 2, Complex64::new(1.0, 0.0));
        assert_eq!(a.get(0, 0), Complex64::new(2.0, 0.0));
        assert_eq!(a.get(1, 0), Complex64::new(1.0, -2.0));
        assert_eq!(a.hermiticity_residual(), 0.0);
        let x = [Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0), Complex64::new(3.0, 0.5)];
        let q = a.quadratic_form(&x).unwrap();
        assert!(q.im.abs() <= 1e-12, "{q}");
    }

    #[test]
    fn dft_parseval_and_constant() {
        let z: Vec<f64> = (0..49).map(|i| ((i * 37 % 11) as f64 - 5.0) / 3.0).collect();
        let spec = label_dft(&z);
        assert!(parseval_relative_residual(&z, &spec) <= 1e-12);
        let c = label_dft(&[2.0; 8]);
        assert!((c[0].re - 2.0 * 8f64.sqrt()).abs() <= 1e-12);
        assert!(c[1..].iter().all(|v| v.norm() <= 1e-12));
        assert!(label_dft(&[]).is_empty());
        // spectrul unui semnal real este hermitic: Z_{n−m} = conj(Z_m)
        for m in 1..49 {
            assert!((spec[49 - m] - spec[m].conj()).norm() <= 1e-12);
        }
    }
}

//! Transformata Fourier fracționară discretă (FrFT) — construcția Candan–Kutay–Ozaktas (2000), protocol S19 Q6.
//!
//! Definiția este cea publicată în `tu4/PROTOCOL_TU4_S16.md` (C3/§2) și implementată de referință în
//! `tu4/s16_martori.py` (`s_matrix`, `p_matrix`, `frft_basis`, `frft_matrix`):
//! S[j,j] = 2cos(2πj/n) − 4, S[j, j±1 mod n] = 1; descompunerea pară/impară P; vectorii proprii ai blocurilor
//! sortați descrescător după valoarea proprie și intercalați pe ordinele Hermite r = 0, 1, …, n−2, n−1+[n par];
//! F_a = V diag(e^{−iπ a r/2}) Vᵀ. Vectorii proprii se obțin cu algoritmul Jacobi ciclic propriu ([`jacobi_eigh`]),
//! fără crate de algebră liniară.
//!
//! F_a este UNITARĂ: energia (deci X²) este invariantă, iar H1/H3 după dus-întorsul F_a⁻¹F_a sunt identice până la
//! rotunjire. FrFT NU adaugă informație și NU selectează bilete (S16, S19 Q8). Precedente: Namias (1980),
//! Ozaktas–Zalevsky–Kutay (2001), Candan–Kutay–Ozaktas (2000).

use num_complex::Complex64;

use super::{neumaier_sum, neumaier_sum_complex, SpectralError};

/// Descompunerea proprie a unei matrice reale simetrice: `values[j]` și coloana j din `vectors` (row-major n×n,
/// `vectors[i * n + j] = V_ij`). Ordinea este cea în care a convers algoritmul (nesortată).
#[derive(Debug, Clone, PartialEq)]
pub struct SymEigen {
    pub n: usize,
    pub values: Vec<f64>,
    pub vectors: Vec<f64>,
    pub sweeps: usize,
}

/// Algoritmul Jacobi ciclic pentru o matrice reală simetrică `a` (row-major n×n). Se oprește când suma pătratelor
/// elementelor din afara diagonalei scade sub 1e-32 × norma Frobenius² (sau la 100 de baleiaje), apoi mai face un
/// baleiaj de curățare. Rotațiile sunt cele stabile (t = sgn θ / (|θ| + √(θ² + 1))).
pub fn jacobi_eigh(a: &[f64], n: usize) -> Result<SymEigen, SpectralError> {
    if a.len() != n * n {
        return Err(SpectralError::DimensionMismatch { expected: n * n, found: a.len() });
    }
    let mut m = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let frob2 = neumaier_sum(m.iter().map(|x| x * x));
    let off2 = |m: &[f64]| {
        neumaier_sum(
            (0..n)
                .flat_map(|p| (0..n).filter(move |&q| q != p).map(move |q| (p, q)))
                .map(|(p, q)| m[p * n + q] * m[p * n + q]),
        )
    };
    let mut sweeps = 0usize;
    let mut cleanup_done = false;
    while sweeps < 100 {
        let converged = off2(&m) <= 1e-32 * frob2;
        if converged && cleanup_done {
            break;
        }
        if converged {
            cleanup_done = true;
        }
        sweeps += 1;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    Ok(SymEigen { n, values: (0..n).map(|i| m[i * n + i]).collect(), vectors: v, sweeps })
}

/// max_{i,j} |(A V − V Λ)_ij| — reziduul descompunerii (măsurat, nu presupus).
pub fn eigen_residual(a: &[f64], e: &SymEigen) -> f64 {
    let n = e.n;
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let av = neumaier_sum((0..n).map(|k| a[i * n + k] * e.vectors[k * n + j]));
            worst = worst.max((av - e.vectors[i * n + j] * e.values[j]).abs());
        }
    }
    worst
}

/// Matricea S (Dickinson–Steiglitz / Candan–Kutay–Ozaktas), row-major n×n.
pub fn s_matrix(n: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; n * n];
    for j in 0..n {
        s[j * n + j] = 2.0 * (2.0 * std::f64::consts::PI * j as f64 / n as f64).cos() - 4.0;
        s[j * n + (j + 1) % n] += 1.0;
        s[j * n + (j + n - 1) % n] += 1.0;
    }
    s
}

/// Matricea ortogonală simetrică P care separă subspațiul par (x_j = x_{n−j}) de cel impar (ca `p_matrix` S16).
pub fn p_matrix(n: usize) -> Vec<f64> {
    let (r, even) = (n / 2, n.is_multiple_of(2));
    let mut p = vec![0.0f64; n * n];
    p[0] = 1.0;
    let s = 1.0 / 2f64.sqrt();
    for i in 1..=(r - usize::from(even)) {
        p[i * n + i] = s;
        p[i * n + (n - i)] = s;
    }
    if even {
        p[r * n + r] = 1.0;
    }
    for i in (r + 1)..n {
        p[i * n + i] = -s;
        p[i * n + (n - i)] = s;
    }
    p
}

fn matmul_real(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = neumaier_sum((0..inner).map(|k| a[i * inner + k] * b[k * cols + j]));
        }
    }
    out
}

/// Planul FrFT pentru dimensiunea n: baza V (row-major n×n, coloanele = vectori Hermite–Gauss discreți) și ordinele r.
#[derive(Debug, Clone, PartialEq)]
pub struct Frft {
    n: usize,
    v: Vec<f64>,
    orders: Vec<f64>,
    eigen_residual: f64,
}

impl Frft {
    /// Construcția CKO pentru n ≥ 2 (pentru n < 4 blocul impar poate fi gol; tratat).
    pub fn new(n: usize) -> Result<Frft, SpectralError> {
        if !(2..=255).contains(&n) {
            return Err(SpectralError::InvalidSpec { n, k: 0 });
        }
        let s = s_matrix(n);
        let p = p_matrix(n);
        // CS = P S Pᵀ (P este simetrică: Pᵀ = P ca matrice, dar o transpunem explicit pentru claritate).
        let pt: Vec<f64> = (0..n * n).map(|idx| p[(idx % n) * n + idx / n]).collect();
        let cs = matmul_real(&matmul_real(&p, &s, n, n, n), &pt, n, n, n);
        let ne = n / 2 + 1;
        let no = n - ne;
        let block = |off: usize, len: usize| -> Vec<f64> {
            (0..len * len).map(|idx| cs[(off + idx / len) * n + off + idx % len]).collect()
        };
        let ev = jacobi_eigh(&block(0, ne), ne)?;
        let od = if no > 0 { Some(jacobi_eigh(&block(ne, no), no)?) } else { None };
        let mut resid = eigen_residual(&block(0, ne), &ev);
        if let Some(o) = &od {
            resid = resid.max(eigen_residual(&block(ne, no), o));
        }
        // sortare descrescătoare stabilă după valoarea proprie
        let sorted = |e: &SymEigen| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..e.n).collect();
            idx.sort_by(|&a, &b| e.values[b].total_cmp(&e.values[a]));
            idx
        };
        // ve = P[:, :ne] · vc, vo = P[:, ne:] · vs
        let lift = |e: &SymEigen, off: usize, col: usize| -> Vec<f64> {
            (0..n).map(|i| neumaier_sum((0..e.n).map(|k| p[i * n + off + k] * e.vectors[k * e.n + col]))).collect()
        };
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut orders = Vec::with_capacity(n);
        for (j, &c) in sorted(&ev).iter().enumerate() {
            cols.push(lift(&ev, 0, c));
            orders.push((2 * j) as f64);
        }
        if let Some(o) = &od {
            for (j, &c) in sorted(o).iter().enumerate() {
                cols.push(lift(o, ne, c));
                orders.push((2 * j + 1) as f64);
            }
        }
        let mut v = vec![0.0f64; n * n];
        for (j, col) in cols.iter().enumerate() {
            for i in 0..n {
                v[i * n + j] = col[i];
            }
        }
        Ok(Frft { n, v, orders, eigen_residual: resid })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Ordinele Hermite r atribuite coloanelor lui V.
    pub fn orders(&self) -> &[f64] {
        &self.orders
    }

    /// Reziduul maxim A V − V Λ al celor două blocuri (diagnostic al solverului Jacobi).
    pub fn eigen_residual(&self) -> f64 {
        self.eigen_residual
    }

    /// e^{−iπ a r/2}; când a·r este întreg, faza este exact una din {1, −i, −1, i} (sferturi de tură exacte).
    fn phase(a: f64, r: f64) -> Complex64 {
        let q = a * r;
        if q.fract() == 0.0 && q.abs() < 9.0e15 {
            return match (q as i64).rem_euclid(4) {
                0 => Complex64::new(1.0, 0.0),
                1 => Complex64::new(0.0, -1.0),
                2 => Complex64::new(-1.0, 0.0),
                _ => Complex64::new(0.0, 1.0),
            };
        }
        let ang = -std::f64::consts::FRAC_PI_2 * q.rem_euclid(4.0);
        Complex64::new(ang.cos(), ang.sin())
    }

    /// Matricea F_a (row-major n×n).
    pub fn matrix(&self, a: f64) -> Vec<Complex64> {
        let n = self.n;
        let ph: Vec<Complex64> = self.orders.iter().map(|&r| Self::phase(a, r)).collect();
        let mut f = vec![Complex64::new(0.0, 0.0); n * n];
        for j in 0..n {
            for k in 0..n {
                f[j * n + k] = neumaier_sum_complex((0..n).map(|r| ph[r] * (self.v[j * n + r] * self.v[k * n + r])));
            }
        }
        f
    }

    /// y = F_a x = V (φ ∘ (Vᵀ x)).
    pub fn apply(&self, a: f64, x: &[Complex64]) -> Result<Vec<Complex64>, SpectralError> {
        let n = self.n;
        if x.len() != n {
            return Err(SpectralError::DimensionMismatch { expected: n, found: x.len() });
        }
        let coef: Vec<Complex64> = (0..n)
            .map(|r| Self::phase(a, self.orders[r]) * neumaier_sum_complex((0..n).map(|i| x[i] * self.v[i * n + r])))
            .collect();
        Ok((0..n).map(|j| neumaier_sum_complex((0..n).map(|r| coef[r] * self.v[j * n + r]))).collect())
    }

    /// F_a⁻¹ y = F_aᴴ y = F_{−a} y (unitaritate).
    pub fn apply_inverse(&self, a: f64, y: &[Complex64]) -> Result<Vec<Complex64>, SpectralError> {
        self.apply(-a, y)
    }

    /// F_a pe un vector real.
    pub fn apply_real(&self, a: f64, x: &[f64]) -> Result<Vec<Complex64>, SpectralError> {
        let xc: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        self.apply(a, &xc)
    }
}

/// DFT unitară W[m,j] = e^{−2πi m j/n}/√n (indicele (m·j) mod n redus exact în întregi).
pub fn dft_matrix(n: usize) -> Vec<Complex64> {
    let scale = 1.0 / (n as f64).sqrt();
    let tw: Vec<Complex64> = (0..n)
        .map(|r| {
            let ang = -2.0 * std::f64::consts::PI * r as f64 / n as f64;
            Complex64::new(ang.cos(), ang.sin())
        })
        .collect();
    (0..n * n).map(|idx| tw[((idx / n) * (idx % n)) % n] * scale).collect()
}

/// Produsul complex A B (row-major n×n), sume compensate.
pub fn complex_matmul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    (0..n * n).map(|idx| neumaier_sum_complex((0..n).map(|k| a[(idx / n) * n + k] * b[k * n + idx % n]))).collect()
}

/// max |a − b| pe componente.
pub fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(if a.len() == b.len() { 0.0 } else { f64::INFINITY }, f64::max)
}

/// max |F − I|.
pub fn identity_residual(f: &[Complex64], n: usize) -> f64 {
    (0..n * n)
        .map(|idx| {
            (f[idx] - if idx / n == idx % n { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }).norm()
        })
        .fold(0.0, f64::max)
}

/// max |Fᴴ F − I| (unitaritatea).
pub fn unitarity_residual(f: &[Complex64], n: usize) -> f64 {
    let fh: Vec<Complex64> = (0..n * n).map(|idx| f[(idx % n) * n + idx / n].conj()).collect();
    identity_residual(&complex_matmul(&fh, f, n), n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalizes_symmetric_matrix() {
        let n: usize = 7;
        // a_ij = 1/(1 + |i − j|) + i·δ_ij (simetrică, valori proprii distincte)
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                1.0 / (1.0 + i.abs_diff(j) as f64) + if i == j { i as f64 } else { 0.0 }
            })
            .collect();
        let e = jacobi_eigh(&a, n).unwrap();
        assert!(eigen_residual(&a, &e) <= 1e-12, "{}", eigen_residual(&a, &e));
        // V ortogonală
        for i in 0..n {
            for j in 0..n {
                let d = neumaier_sum((0..n).map(|k| e.vectors[k * n + i] * e.vectors[k * n + j]));
                assert!((d - if i == j { 1.0 } else { 0.0 }).abs() <= 1e-13);
            }
        }
        // urma conservată
        let tr = neumaier_sum((0..n).map(|i| a[i * n + i]));
        assert!((neumaier_sum(e.values.iter().copied()) - tr).abs() <= 1e-12);
        assert!(jacobi_eigh(&a, 6).is_err());
    }

    #[test]
    fn frft_w0_w1_construction_and_unitarity() {
        for n in [20usize, 40, 45, 49] {
            let f = Frft::new(n).unwrap();
            assert!(f.eigen_residual() <= 1e-12, "n={n} {}", f.eigen_residual());
            assert!(identity_residual(&f.matrix(0.0), n) <= 1e-12, "n={n}");
            assert!(max_abs_diff(&f.matrix(1.0), &dft_matrix(n)) <= 1e-10, "n={n}");
            for (a, b) in [(0.3, 0.4), (0.5, 0.5), (0.25, 1.5)] {
                let lhs = complex_matmul(&f.matrix(a), &f.matrix(b), n);
                assert!(max_abs_diff(&lhs, &f.matrix(a + b)) <= 1e-12, "n={n} a={a} b={b}");
            }
            for a in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9] {
                assert!(unitarity_residual(&f.matrix(a), n) <= 1e-12, "n={n} a={a}");
            }
            let mut ord = f.orders().to_vec();
            ord.sort_by(f64::total_cmp);
            let mut want: Vec<f64> = (0..n - 1).map(|r| r as f64).collect();
            want.push((n - 1 + usize::from(n.is_multiple_of(2))) as f64);
            assert_eq!(ord, want, "n={n}");
        }
    }

    #[test]
    fn frft_energy_and_round_trip() {
        let n = 49;
        let f = Frft::new(n).unwrap();
        let z: Vec<f64> = (0..n).map(|i| ((5 * (i + 1)) % 49) as f64 - 24.0).collect();
        let e0 = neumaier_sum(z.iter().map(|v| v * v));
        for a in [0.1, 0.5, 1.0, 1.9] {
            let y = f.apply_real(a, &z).unwrap();
            let e1 = neumaier_sum(y.iter().map(|c| c.norm_sqr()));
            assert!((e1 - e0).abs() <= 1e-12 * e0);
            let back = f.apply_inverse(a, &y).unwrap();
            assert!(back.iter().zip(&z).all(|(b, &x)| (b.re - x).abs() <= 1e-12 && b.im.abs() <= 1e-12));
            // apply == matrix · x
            let m = f.matrix(a);
            let mx: Vec<Complex64> =
                (0..n).map(|j| neumaier_sum_complex((0..n).map(|k| m[j * n + k] * z[k]))).collect();
            assert!(max_abs_diff(&mx, &y) <= 1e-12);
        }
        assert!(f.apply_real(0.5, &z[..10]).is_err());
        assert!(Frft::new(1).is_err());
        // dimensiuni mici (bloc impar gol / minim)
        for n in [2usize, 3, 4, 5] {
            let f = Frft::new(n).unwrap();
            assert!(unitarity_residual(&f.matrix(0.37), n) <= 1e-12);
            assert!(max_abs_diff(&f.matrix(1.0), &dft_matrix(n)) <= 1e-10, "n={n}");
        }
    }
}

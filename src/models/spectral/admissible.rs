//! Filtrul de admisibilitate S17, nativ (protocol S19 Q7): biletele „pur liniare” filtrate la nivel de memorie.
//!
//! Definiția (fixată în `tu4/PROTOCOL_TU4_S17.md` C4, NESCHIMBATĂ): biletul T (b ≥ 3 numere distincte, sortate) este
//! **pur liniar** ⇔ este o progresie aritmetică cu pas constant d ≥ 1. Bila Joker nu intră în definiție.
//! Densitatea sub legea modelului: q(T) ∝ Π_{i∈T} w_i; scorul s(T) = Σ_{i∈T} ln w_i calculat EXACT; ordinea totală:
//! s descrescător, apoi tuplul sortat crescător. Biletul admisibil = primul bilet NE-liniar în această ordine;
//! `NU_PARIAZA` dacă max|ln w| = 0.
//!
//! Reprezentarea „la nivel de memorie”: biletul = mască `u64` (bitul i−1 ⇔ eticheta i); tabela tuturor măștilor
//! liniare pentru (n, b) este precalculată și sortată (căutare binară). Exactitatea ordinii: sumele de `f64` sunt
//! ținute ca expansiuni Shewchuk fără suprapunere (TwoSum este exact în rotunjirea la cel mai apropiat), iar două
//! sume se compară prin semnul expansiunii diferenței — fără nicio toleranță.
//!
//! Filtrul NU schimbă nicio probabilitate: sub urna uniformă toate biletele au aceeași șansă; nu este sfat de joc.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use serde::Serialize;

/// Biletul ca mască de biți: bitul i−1 setat ⇔ eticheta i aparține biletului (1 ≤ i ≤ 64).
pub type Mask = u64;

/// Masca etichetelor date (etichete 1..=64; altfel `None`).
pub fn mask_from_labels(labels: &[u8]) -> Option<Mask> {
    let mut m = 0u64;
    for &l in labels {
        if l == 0 || l > 64 {
            return None;
        }
        m |= 1u64 << (l - 1);
    }
    Some(m)
}

/// Etichetele măștii, crescător.
pub fn labels_from_mask(mut m: Mask) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.count_ones() as usize);
    while m != 0 {
        out.push(m.trailing_zeros() as u8 + 1);
        m &= m - 1;
    }
    out
}

/// Definiția S17 pe etichete sortate crescător, distincte, b ≥ 3.
pub fn is_linear_sorted(t: &[u8]) -> bool {
    if t.len() < 3 {
        return false;
    }
    let d = i16::from(t[1]) - i16::from(t[0]);
    d >= 1 && t.windows(2).all(|w| i16::from(w[1]) - i16::from(w[0]) == d)
}

/// Definiția S17 direct pe mască (b = popcount ≥ 3).
pub fn is_linear_mask(m: Mask) -> bool {
    if m.count_ones() < 3 {
        return false;
    }
    let low = m.trailing_zeros();
    let rest = m & (m - 1);
    let d = rest.trailing_zeros() - low;
    let mut ap = 0u64;
    let mut pos = low;
    for _ in 0..m.count_ones() {
        if pos >= 64 {
            return false;
        }
        ap |= 1u64 << pos;
        pos += d;
    }
    ap == m
}

/// Forma închisă a numărului de progresii: Σ_{d≥1} max(0, n − (b−1)d).
pub fn ap_count_closed(n: usize, b: usize) -> u64 {
    if b < 2 {
        return 0;
    }
    (1..n).map(|d| n.saturating_sub((b - 1) * d) as u64).sum()
}

/// Tabela precalculată, sortată, a tuturor măștilor pur liniare pentru (n, b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearTable {
    pub n: usize,
    pub b: usize,
    masks: Vec<Mask>,
}

impl LinearTable {
    pub fn new(n: usize, b: usize) -> LinearTable {
        let mut masks = Vec::new();
        if b >= 3 && n <= 64 {
            for start in 1..=n {
                for d in 1..n {
                    let last = start + (b - 1) * d;
                    if last > n {
                        break;
                    }
                    masks.push((0..b).fold(0u64, |m, j| m | 1u64 << (start + j * d - 1)));
                }
            }
        }
        masks.sort_unstable();
        masks.dedup();
        LinearTable { n, b, masks }
    }

    pub fn len(&self) -> usize {
        self.masks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.masks.is_empty()
    }

    /// Filtrul la nivel de memorie: căutare binară în tabela sortată.
    pub fn contains(&self, m: Mask) -> bool {
        self.masks.binary_search(&m).is_ok()
    }
}

// ------------------------------------------------------------------ sume exacte (expansiuni Shewchuk)

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Suma exactă a unor `f64` ca expansiune fără suprapunere (componente nenule, magnitudine crescătoare).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExactSum {
    comps: Vec<f64>,
}

impl ExactSum {
    pub fn new() -> ExactSum {
        ExactSum { comps: Vec::new() }
    }

    pub fn of(values: impl IntoIterator<Item = f64>) -> ExactSum {
        let mut s = ExactSum::new();
        for v in values {
            s.add(v);
        }
        s
    }

    /// GROW-EXPANSION (Shewchuk 1997) cu eliminarea zerourilor; exact pentru intrări finite fără depășire.
    pub fn add(&mut self, x: f64) {
        let mut q = x;
        let mut out = Vec::with_capacity(self.comps.len() + 1);
        for &e in &self.comps {
            let (s, err) = two_sum(q, e);
            if err != 0.0 {
                out.push(err);
            }
            q = s;
        }
        if q != 0.0 {
            out.push(q);
        }
        self.comps = out;
    }

    /// Semnul valorii exacte = semnul celei mai mari componente.
    pub fn signum(&self) -> Ordering {
        match self.comps.last() {
            None => Ordering::Equal,
            Some(v) if *v > 0.0 => Ordering::Greater,
            Some(_) => Ordering::Less,
        }
    }

    /// Comparația exactă a două sume.
    pub fn cmp_exact(&self, other: &ExactSum) -> Ordering {
        let mut d = self.clone();
        for &c in &other.comps {
            d.add(-c);
        }
        d.signum()
    }

    /// Diferența exactă self − other.
    pub fn minus(&self, other: &ExactSum) -> ExactSum {
        let mut d = self.clone();
        for &c in &other.comps {
            d.add(-c);
        }
        d
    }

    /// Aproximarea `f64` (suma componentelor de la mic la mare; eroare ≤ 1 ulp).
    pub fn approx(&self) -> f64 {
        self.comps.iter().sum()
    }

    pub fn components(&self) -> &[f64] {
        &self.comps
    }
}

// ------------------------------------------------------------------ ordinea totală și heap-ul S17

/// Etichetele 1..n ordonate după (−ln w, etichetă) — exact ca `np.argsort(-lw, kind="stable")`.
/// Comparația este NUMERICĂ (−0.0 == +0.0, ca `Fraction` în S17), apoi după etichetă (amendamentul S19-A01).
pub fn label_order(logw: &[f64]) -> Vec<u8> {
    let mut idx: Vec<u8> = (1..=logw.len() as u8).collect();
    idx.sort_by(|&a, &b| {
        let (wa, wb) = (logw[usize::from(a) - 1], logw[usize::from(b) - 1]);
        wb.partial_cmp(&wa).unwrap_or_else(|| wb.total_cmp(&wa)).then(a.cmp(&b))
    });
    idx
}

/// Biletul top (b etichete cu ln w maxim, departajare după etichetă), sortat — ca `ip_core.ticket`.
pub fn top_ticket(logw: &[f64], b: usize) -> Vec<u8> {
    let mut t: Vec<u8> = label_order(logw).into_iter().take(b).collect();
    t.sort_unstable();
    t
}

#[derive(Debug, Clone)]
struct Node {
    s: ExactSum,
    labels: Vec<u8>,
    mask: Mask,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    /// „Mai mare” = mai devreme în ordinea totală (BinaryHeap este max-heap).
    fn cmp(&self, other: &Self) -> Ordering {
        self.s.cmp_exact(&other.s).then_with(|| other.labels.cmp(&self.labels))
    }
}

fn node(labels: Vec<u8>, logw: &[f64]) -> Node {
    let s = ExactSum::of(labels.iter().map(|&l| logw[usize::from(l) - 1]));
    let mask = mask_from_labels(&labels).unwrap_or(0);
    Node { s, labels, mask }
}

/// Un bilet enumerat de heap: rangul (1 = top), etichetele, suma exactă și numărul de noduri la momentul extragerii.
#[derive(Debug, Clone)]
pub struct Ranked {
    pub rank: u64,
    pub labels: Vec<u8>,
    pub mask: Mask,
    pub s: ExactSum,
    pub nodes: u64,
}

/// Enumerarea best-first exactă (S17 `heap_order`): se oprește după biletul pentru care `stop` e adevărat sau după
/// `limit` bilete. `nodes` = numărul de bilete distincte introduse în heap înainte de expandarea biletului curent.
pub fn heap_order(logw: &[f64], b: usize, mut stop: impl FnMut(&Ranked) -> bool, limit: Option<u64>) -> Vec<Ranked> {
    let n = logw.len();
    let mut out = Vec::new();
    if b == 0 || b > n || n > 64 {
        return out;
    }
    let top = node(top_ticket(logw, b), logw);
    let mut seen: HashSet<Mask> = HashSet::from([top.mask]);
    let mut heap = BinaryHeap::from([top]);
    let (mut rank, mut nodes) = (0u64, 1u64);
    while let Some(t) = heap.pop() {
        rank += 1;
        let r = Ranked { rank, labels: t.labels.clone(), mask: t.mask, s: t.s.clone(), nodes };
        let halt = stop(&r) || limit.is_some_and(|l| rank >= l);
        out.push(r);
        if halt {
            break;
        }
        for &a in &t.labels {
            for c in 1..=n as u8 {
                if t.mask & (1u64 << (c - 1)) != 0 {
                    continue;
                }
                let m = (t.mask & !(1u64 << (a - 1))) | (1u64 << (c - 1));
                if seen.insert(m) {
                    nodes += 1;
                    heap.push(node(labels_from_mask(m), logw));
                }
            }
        }
    }
    out
}

/// Starea filtrului pentru un flux.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AdmState {
    #[serde(rename = "ADMISIBIL_IDENTIC")]
    Identic,
    #[serde(rename = "ADMISIBIL_INLOCUIT")]
    Inlocuit,
    #[serde(rename = "NU_PARIAZA")]
    NuPariaza,
}

impl AdmState {
    pub fn as_str(self) -> &'static str {
        match self {
            AdmState::Identic => "ADMISIBIL_IDENTIC",
            AdmState::Inlocuit => "ADMISIBIL_INLOCUIT",
            AdmState::NuPariaza => "NU_PARIAZA",
        }
    }
}

/// Rezultatul filtrului (câmpurile au aceleași nume ca în fișierul S17).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Admissible {
    pub stare: AdmState,
    pub top_ticket: Vec<u8>,
    pub top_liniar: bool,
    pub bilet_afisare: Option<Vec<u8>>,
    pub rang: Option<u64>,
    pub noduri_heap: Option<u64>,
    pub ln_q_u_top: Option<f64>,
    pub ln_q_u_admisibil: Option<f64>,
    pub q_adm_pe_q_top: Option<f64>,
}

/// ln e_b(w) (aceeași recursie ca `s17_admisibil.log_esp`).
pub fn log_esp(logw: &[f64], b: usize) -> f64 {
    let mut e = vec![0.0f64; b + 1];
    e[0] = 1.0;
    for &lw in logw {
        let wi = lw.exp();
        for j in (1..=b).rev() {
            e[j] += wi * e[j - 1];
        }
    }
    e[b].ln()
}

/// ln C(n, b) din coeficientul binomial exact (u128).
pub fn ln_binomial(n: usize, b: usize) -> f64 {
    if b > n {
        return f64::NEG_INFINITY;
    }
    let b = b.min(n - b);
    let mut c: u128 = 1;
    for i in 0..b {
        c = c * (n - i) as u128 / (i + 1) as u128;
    }
    (c as f64).ln()
}

/// Filtrul de admisibilitate complet pentru un flux cu `ln ŵ` dat și mărimea biletului b.
pub fn admissible(logw: &[f64], b: usize) -> Admissible {
    let top = top_ticket(logw, b);
    let top_liniar = is_linear_sorted(&top);
    if logw.iter().all(|v| v.abs() == 0.0) {
        return Admissible {
            stare: AdmState::NuPariaza,
            top_ticket: top,
            top_liniar,
            bilet_afisare: None,
            rang: None,
            noduri_heap: None,
            ln_q_u_top: None,
            ln_q_u_admisibil: None,
            q_adm_pe_q_top: None,
        };
    }
    let table = LinearTable::new(logw.len(), b);
    let seq = heap_order(logw, b, |r| !table.contains(r.mask), None);
    let first = &seq[0];
    let last = &seq[seq.len() - 1];
    let (le, lc) = (log_esp(logw, b), ln_binomial(logw.len(), b));
    Admissible {
        stare: if last.labels == first.labels { AdmState::Identic } else { AdmState::Inlocuit },
        top_ticket: first.labels.clone(),
        top_liniar,
        bilet_afisare: Some(last.labels.clone()),
        rang: Some(last.rank),
        noduri_heap: Some(last.nodes),
        ln_q_u_top: Some(first.s.approx() - le + lc),
        ln_q_u_admisibil: Some(last.s.approx() - le + lc),
        q_adm_pe_q_top: Some(last.s.minus(&first.s).approx().exp()),
    }
}

// ------------------------------------------------------------------ enumerarea completă (Gosper, multi-fir)

/// Următoarea mască cu același număr de biți (Gosper's hack).
#[inline]
pub fn gosper_next(x: Mask) -> Mask {
    let c = x & x.wrapping_neg();
    let r = x.wrapping_add(c);
    (((r ^ x) >> 2) / c) | r
}

/// Rezultatul verificării prin enumerare completă (analogul `enumerate_check` din S17, pe CPU).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Enumeration {
    pub n: usize,
    pub b: usize,
    pub n_combos: u64,
    pub n_linear: u64,
    pub n_linear_closed_form: u64,
    pub top: Vec<u8>,
    pub admissible: Vec<u8>,
    pub rank: u64,
    pub window_top: u64,
    pub window_adm: u64,
    pub window_rank: u64,
    pub threads: usize,
    pub seconds: f64,
}

const WINDOW: f64 = 1e-9;

/// Acumulatorul local al unui fir de enumerare (fuzionat la final, fără blocări pe mască).
#[derive(Debug)]
struct Acc {
    count: u64,
    linear: u64,
    max_all: f64,
    max_nl: f64,
    list_a: Vec<Mask>,
    list_b: Vec<Mask>,
}

impl Default for Acc {
    fn default() -> Acc {
        Acc {
            count: 0,
            linear: 0,
            max_all: f64::NEG_INFINITY,
            max_nl: f64::NEG_INFINITY,
            list_a: Vec::new(),
            list_b: Vec::new(),
        }
    }
}

impl Acc {
    fn merge(mut self, o: Acc) -> Acc {
        self.count += o.count;
        self.linear += o.linear;
        self.max_all = self.max_all.max(o.max_all);
        self.max_nl = self.max_nl.max(o.max_nl);
        self.list_a.extend(o.list_a);
        self.list_b.extend(o.list_b);
        self
    }
}

/// Enumeră toate cele C(n, b) măști, pe toate nucleele; numără biletele liniare (tabela) și găsește exact topul,
/// biletul admisibil și rangul lui (fereastră f64 de 1e-9, apoi ordonare exactă ca în S17).
pub fn enumerate_all(logw: &[f64], b: usize) -> Option<Enumeration> {
    let n = logw.len();
    if b < 3 || b > n || n > 63 {
        return None;
    }
    let t0 = std::time::Instant::now();
    let table = LinearTable::new(n, b);
    let threads = std::thread::available_parallelism().map_or(1, |v| v.get()).min(n - b + 1);
    // Partiționare după cea mai mică etichetă `lo` (0-based): restul b−1 biți dintre pozițiile > lo.
    let score = |m: Mask| -> f64 {
        let mut s = 0.0;
        let mut x = m;
        while x != 0 {
            s += logw[x.trailing_zeros() as usize];
            x &= x - 1;
        }
        s
    };
    let for_each = |f: &(dyn Fn(&mut Acc, Mask) + Sync)| -> Acc {
        let next = AtomicUsize::new(0);
        let parts: Vec<Acc> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut acc = Acc::default();
                        loop {
                            let lo = next.fetch_add(1, AtomicOrdering::Relaxed);
                            if lo + b > n {
                                break;
                            }
                            let free = n - lo - 1;
                            let k = b - 1;
                            let mut x: Mask = (1u64 << k) - 1;
                            let limit: Mask = 1u64 << free;
                            while x < limit {
                                f(&mut acc, (x << (lo + 1)) | (1u64 << lo));
                                x = gosper_next(x);
                            }
                        }
                        acc
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
        });
        parts.into_iter().fold(Acc::default(), Acc::merge)
    };
    // Trecerea 1: numărări și maximele f64.
    let p1 = for_each(&|acc, m| {
        let s = score(m);
        acc.count += 1;
        if table.contains(m) {
            acc.linear += 1;
        } else if s > acc.max_nl {
            acc.max_nl = s;
        }
        if s > acc.max_all {
            acc.max_all = s;
        }
    });
    let (n_combos, n_linear, smax_all, smax_nl) = (p1.count, p1.linear, p1.max_all, p1.max_nl);
    // Trecerea 2: candidații din fereastră.
    let p2 = for_each(&|acc, m| {
        let s = score(m);
        if s >= smax_all - WINDOW {
            acc.list_a.push(m);
        }
        if s >= smax_nl - WINDOW && !table.contains(m) {
            acc.list_b.push(m);
        }
    });
    let (c_all, c_nl) = (p2.list_a, p2.list_b);
    let best = |c: &[Mask]| c.iter().map(|&m| node(labels_from_mask(m), logw)).max();
    let top = best(&c_all)?;
    let adm = best(&c_nl)?;
    let sa = adm.s.approx();
    // Trecerea 3: rangul = #(s > sa + W) + #(fereastră cu cheie exactă înaintea admisibilului) + 1.
    let p3 = for_each(&|acc, m| {
        let s = score(m);
        if s > sa + WINDOW {
            acc.count += 1;
        } else if (s - sa).abs() <= WINDOW {
            acc.list_a.push(m);
        }
    });
    let (above, win) = (p3.count, p3.list_a);
    let before = win.iter().filter(|&&m| node(labels_from_mask(m), logw) > adm).count() as u64;
    Some(Enumeration {
        n,
        b,
        n_combos,
        n_linear,
        n_linear_closed_form: ap_count_closed(n, b),
        top: top.labels,
        admissible: adm.labels,
        rank: above + before + 1,
        window_top: c_all.len() as u64,
        window_adm: c_nl.len() as u64,
        window_rank: win.len() as u64,
        threads,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binom(n: usize, k: usize) -> u64 {
        (0..k).fold(1u64, |c, i| c * (n - i) as u64 / (i + 1) as u64)
    }

    /// Toate b-submulțimile lui 1..n, sortate după ordinea totală exactă (forța brută).
    fn brute_order(logw: &[f64], b: usize) -> Vec<Vec<u8>> {
        let n = logw.len();
        let mut all = Vec::new();
        let mut x: Mask = (1u64 << b) - 1;
        while x < (1u64 << n) {
            all.push(node(labels_from_mask(x), logw));
            x = gosper_next(x);
        }
        all.sort_by(|a, b| b.cmp(a));
        all.into_iter().map(|nd| nd.labels).collect()
    }

    #[test]
    fn masks_and_linear_definition() {
        assert_eq!(mask_from_labels(&[1, 3, 64]), Some(0b101 | 1u64 << 63));
        assert_eq!(mask_from_labels(&[0]), None);
        assert_eq!(labels_from_mask(0b1010_0001), vec![1, 6, 8]);
        for (t, want) in [
            (vec![1u8, 2, 3, 4, 5, 6], true),
            (vec![5, 10, 15, 20, 25, 30], true),
            (vec![1, 2, 3, 4, 5, 7], false),
            (vec![3, 5, 7], true),
            (vec![1, 2], false),
            (vec![2, 4, 8], false),
        ] {
            assert_eq!(is_linear_sorted(&t), want, "{t:?}");
            assert_eq!(is_linear_mask(mask_from_labels(&t).unwrap()), want, "{t:?}");
        }
    }

    #[test]
    fn a1_linear_counts_match_closed_form() {
        for (n, b) in [(12usize, 3usize), (12, 4), (20, 5), (30, 6), (16, 3)] {
            let table = LinearTable::new(n, b);
            let mut x: Mask = (1u64 << b) - 1;
            let (mut total, mut lin, mut lin_direct) = (0u64, 0u64, 0u64);
            while x < (1u64 << n) {
                total += 1;
                lin += u64::from(table.contains(x));
                lin_direct += u64::from(is_linear_mask(x));
                x = gosper_next(x);
            }
            assert_eq!(total, binom(n, b));
            assert_eq!((lin, lin_direct, table.len() as u64), (ap_count_closed(n, b), lin, lin));
        }
        assert_eq!((ap_count_closed(49, 6), ap_count_closed(40, 5), ap_count_closed(45, 5)), (216, 180, 231));
    }

    #[test]
    fn exact_sum_is_exact() {
        let a = ExactSum::of([1e16, 1.0, -1e16]);
        assert_eq!(a.approx(), 1.0);
        assert_eq!(a.signum(), Ordering::Greater);
        let x = ExactSum::of([0.1, 0.2]);
        let y = ExactSum::of([0.3]);
        // 0.1 + 0.2 ≠ 0.3 exact (în binar), deși diferența este sub 1 ulp
        assert_eq!(x.cmp_exact(&y), Ordering::Greater);
        assert_eq!(ExactSum::of([0.25, 0.5]).cmp_exact(&ExactSum::of([0.75])), Ordering::Equal);
        assert_eq!(ExactSum::of([1e-300, -1e300, 1e300]).signum(), Ordering::Greater);
        assert_eq!(ExactSum::new().signum(), Ordering::Equal);
    }

    #[test]
    fn a2_heap_order_matches_brute_force() {
        let mut state = 2026102817u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for n in 6usize..=12 {
            for b in 3usize..=5 {
                let random: Vec<f64> = (0..n).map(|_| rnd() * 0.2).collect();
                let repeated: Vec<f64> = (0..n).map(|i| [0.01, -0.02, 0.01, 0.0][i % 4]).collect();
                let constant = vec![0.0; n];
                for lw in [random, repeated, constant] {
                    let brute = brute_order(&lw, b);
                    let k = brute.len().min(40);
                    let heap = heap_order(&lw, b, |_| false, Some(k as u64));
                    let got: Vec<Vec<u8>> = heap.iter().map(|r| r.labels.clone()).collect();
                    assert_eq!(got, brute[..k].to_vec(), "n={n} b={b} lw={lw:?}");
                    let want_adm = brute.iter().find(|t| !is_linear_sorted(t)).cloned();
                    let a = admissible(&lw, b);
                    if lw.iter().all(|v| *v == 0.0) {
                        assert_eq!(a.stare, AdmState::NuPariaza);
                    } else {
                        assert_eq!(a.bilet_afisare, want_adm, "n={n} b={b}");
                        let pos = brute.iter().position(|t| Some(t) == want_adm.as_ref()).unwrap() as u64 + 1;
                        assert_eq!(a.rang, Some(pos));
                        let e = enumerate_all(&lw, b).unwrap();
                        assert_eq!(
                            (e.admissible.clone(), e.rank, e.top.clone()),
                            (want_adm.clone().unwrap(), pos, brute[0].clone())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn admissible_states() {
        // top liniar (1,2,3) înlocuit; top neliniar identic
        let lw = [0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3];
        let a = admissible(&lw, 3);
        assert_eq!((a.stare, a.top_ticket.clone(), a.top_liniar), (AdmState::Inlocuit, vec![1, 2, 3], true));
        assert_eq!(a.bilet_afisare, Some(vec![1, 2, 4]));
        assert_eq!(a.rang, Some(2));
        assert!(a.q_adm_pe_q_top.unwrap() < 1.0);
        let lw2 = [0.3, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0];
        assert_eq!(admissible(&lw2, 3).stare, AdmState::Inlocuit); // 1,3,5 este progresie (d = 2)
        let lw3 = [0.3, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0];
        let a3 = admissible(&lw3, 3);
        assert_eq!((a3.stare, a3.rang, a3.q_adm_pe_q_top), (AdmState::Identic, Some(1), Some(1.0)));
        assert_eq!(admissible(&[0.0; 9], 4).stare, AdmState::NuPariaza);
        assert!((ln_binomial(49, 6) - (13_983_816f64).ln()).abs() == 0.0);
    }

    #[test]
    fn signed_zeros_order_numerically() {
        // Amendamentul S19-A01: λ = 0 ⇒ ln ŵ = 0·θ̂ conține −0.0; ordinea trebuie să fie cea numerică (după etichetă).
        let lw = [-0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0];
        assert_eq!(label_order(&lw), (1..=8).collect::<Vec<u8>>());
        assert_eq!(top_ticket(&lw, 5), vec![1, 2, 3, 4, 5]);
        let a = admissible(&lw, 5);
        assert_eq!((a.stare, a.top_ticket.clone(), a.top_liniar), (AdmState::NuPariaza, vec![1, 2, 3, 4, 5], true));
        assert_eq!(top_ticket(&[0.1, -0.0, 0.1, 0.0], 2), vec![1, 3]);
    }
}

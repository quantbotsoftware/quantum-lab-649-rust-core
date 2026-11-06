//! Integritatea emisiilor afișate (protocol S20, decizia R3 și amendamentul S20-T01).
//!
//! - SHA-256 conform FIPS 180-4, implementat aici, fără dependențe noi. Martori: vectorii NIST și sha-urile din
//!   `tu4/tribunal/LANT.jsonl`.
//! - Validarea țintei (T1): `AAAA-LL-ZZ`, dată gregoriană validă, joi sau duminică (`WEEKDAYS = (3, 6)` din
//!   `tu4/emite.py`), prin aritmetică civilă exactă, fără fus orar.
//! - Lista țintelor (T2) și certificarea emisiei față de LANT (T3): octeții citiți o singură dată, hash-uiți, apoi
//!   parsați din același buffer.
//!
//! Nimic de aici nu scrie pe disc, nu emite și nu decontează.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

/// Jocurile tribunalului, în ordinea afișării.
pub const GAMES: [&str; 3] = ["649", "540", "joker"];

// ------------------------------------------------------------------ SHA-256 (FIPS 180-4)

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

const H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

/// Starea incrementală SHA-256.
#[derive(Debug, Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    used: usize,
    total_bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Sha256 {
        Sha256::new()
    }
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 { h: H0, buf: [0; 64], used: 0, total_bytes: 0 }
    }

    fn compress(h: &mut [u32; 8], block: &[u8]) {
        let mut w = [0u32; 64];
        for (t, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([block[4 * t], block[4 * t + 1], block[4 * t + 2], block[4 * t + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.used > 0 {
            let take = (64 - self.used).min(data.len());
            self.buf[self.used..self.used + take].copy_from_slice(&data[..take]);
            self.used += take;
            data = &data[take..];
            if self.used < 64 {
                return;
            }
            let block = self.buf;
            Sha256::compress(&mut self.h, &block);
            self.used = 0;
        }
        let mut chunks = data.chunks_exact(64);
        for block in &mut chunks {
            Sha256::compress(&mut self.h, block);
        }
        let rest = chunks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.used = rest.len();
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        let zeros = (64 + 56 - (self.used + 1) % 64) % 64;
        pad.extend(std::iter::repeat_n(0u8, zeros));
        pad.extend_from_slice(&bit_len.to_be_bytes());
        let total = self.total_bytes;
        self.update(&pad);
        debug_assert_eq!(self.used, 0);
        self.total_bytes = total;
        let mut out = [0u8; 32];
        for (i, v) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

/// SHA-256 al unui buffer.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finalize()
}

/// SHA-256 în hexazecimal cu litere mici (formatul din LANT și din `sha256sum`).
pub fn sha256_hex(data: &[u8]) -> String {
    sha256(data).iter().map(|b| format!("{b:02x}")).collect()
}

// ------------------------------------------------------------------ calendarul (T1)

/// Zilele de la 1970-01-01 pentru data civilă gregoriană (algoritmul lui H. Hinnant, exact pe i64).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inversa lui [`days_from_civil`].
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 }, m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        _ => 28,
    }
}

/// Ziua săptămânii ca în Python `date.weekday()`: luni = 0, …, duminică = 6 (1970-01-01 a fost joi = 3).
pub fn weekday(days: i64) -> u32 {
    (days + 3).rem_euclid(7) as u32
}

/// Zilele extragerilor (`WEEKDAYS = (3, 6)` din `tu4/emite.py`): joi și duminică.
pub const DRAW_WEEKDAYS: [u32; 2] = [3, 6];

/// O țintă validată (T1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub y: i64,
    pub m: u32,
    pub d: u32,
    pub days: i64,
}

impl Target {
    pub fn iso(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.y, self.m, self.d)
    }
}

/// Validarea strictă T1: exact `AAAA-LL-ZZ`, dată validă, joi sau duminică.
pub fn parse_target(s: &str) -> Result<Target, String> {
    let b = s.as_bytes();
    let shape_ok = b.len() == 10
        && b.iter().enumerate().all(|(i, &c)| if i == 4 || i == 7 { c == b'-' } else { c.is_ascii_digit() });
    if !shape_ok {
        return Err(format!("ținta „{s}” nu are forma AAAA-LL-ZZ"));
    }
    let num = |r: std::ops::Range<usize>| s[r].parse::<u32>().map_err(|e| format!("ținta „{s}”: {e}"));
    let (y, m, d) = (num(0..4)? as i64, num(5..7)?, num(8..10)?);
    if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
        return Err(format!("ținta „{s}” nu este o dată calendaristică validă"));
    }
    let days = days_from_civil(y, m, d);
    let wd = weekday(days);
    if !DRAW_WEEKDAYS.contains(&wd) {
        return Err(format!("ținta „{s}” cade în ziua {wd} (luni = 0); extragerile sunt joi (3) și duminică (6)"));
    }
    Ok(Target { y, m, d, days })
}

/// Prima zi de extragere strict după `t` (joi sau duminică).
pub fn next_draw_after(t: &Target) -> Target {
    let mut z = t.days + 1;
    while !DRAW_WEEKDAYS.contains(&weekday(z)) {
        z += 1;
    }
    let (y, m, d) = civil_from_days(z);
    Target { y, m, d, days: z }
}

// ------------------------------------------------------------------ LANT și certificarea (T2, T3)

/// O linie din `tu4/tribunal/LANT.jsonl` (câmpurile folosite).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LantEntry {
    pub i: u64,
    pub game: String,
    pub target_date: String,
    pub file: String,
    pub sha256: String,
    pub emitted_utc: String,
}

pub fn lant_path(root: &Path) -> PathBuf {
    root.join("tu4").join("tribunal").join("LANT.jsonl")
}

pub fn emission_file(root: &Path, game: &str, target: &str) -> PathBuf {
    root.join("tu4").join("tribunal").join("emisii").join(game).join(format!("{target}.json"))
}

/// Citește LANT. Orice linie neparsabilă sau fără câmpurile obligatorii este eroare (lanțul nu se interpretează parțial).
pub fn load_lant(root: &Path) -> Result<Vec<LantEntry>, String> {
    let p = lant_path(root);
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut out = Vec::new();
    for (no, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("LANT linia {}: {e}", no + 1))?;
        let field =
            |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| format!("LANT linia {}: lipsește „{k}”", no + 1));
        out.push(LantEntry {
            i: v["i"].as_u64().ok_or_else(|| format!("LANT linia {}: lipsește „i”", no + 1))?,
            game: field("game")?,
            target_date: field("target_date")?,
            file: field("file")?,
            sha256: field("sha256")?,
            emitted_utc: field("emitted_utc")?,
        });
    }
    Ok(out)
}

/// O emisie certificată (T3): conținutul parsat din aceiași octeți care au fost hash-uiți.
#[derive(Debug, Clone)]
pub struct CertifiedEmission {
    pub game: String,
    pub target: String,
    pub sha256: String,
    pub bytes: usize,
    pub lant_i: u64,
    pub doc: Value,
}

/// Certificarea T3 pentru (joc, țintă). `Ok(None)` ⇔ fișierul lipsește ȘI LANT nu are linie pentru (joc, țintă).
pub fn certify_emission(
    root: &Path,
    lant: &[LantEntry],
    game: &str,
    target: &Target,
) -> Result<Option<CertifiedEmission>, String> {
    let t = target.iso();
    let entries: Vec<&LantEntry> = lant.iter().filter(|e| e.game == game && e.target_date == t).collect();
    let path = emission_file(root, game, &t);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return if entries.is_empty() {
                Ok(None)
            } else {
                Err(format!("LANT are {} linii pentru {game}/{t}, dar {} lipsește", entries.len(), path.display()))
            };
        }
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let sha = sha256_hex(&bytes);
    let expected_file = format!("emisii/{game}/{t}.json");
    let Some(entry) = entries.iter().find(|e| e.sha256 == sha) else {
        return Err(format!("emisia {game}/{t} (sha {sha}) nu este certificată de LANT; afișarea este refuzată"));
    };
    if entry.file != expected_file {
        return Err(format!("LANT indică „{}” pentru {game}/{t}, așteptat „{expected_file}”", entry.file));
    }
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    for (k, want) in [("game", game), ("target_date", t.as_str())] {
        if let Some(have) = doc.get(k).and_then(Value::as_str) {
            if have != want {
                return Err(format!("emisia {game}/{t}: câmpul „{k}” = „{have}”, așteptat „{want}”"));
            }
        }
    }
    Ok(Some(CertifiedEmission {
        game: game.to_string(),
        target: t,
        sha256: sha,
        bytes: bytes.len(),
        lant_i: entry.i,
        doc,
    }))
}

/// Starea unei ținte în lista T2.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TargetStatus {
    pub tinta: String,
    /// `COMPLETA`, `PARTIALA` sau `EMISIE_ABSENTA`.
    pub stare: String,
    pub programata: bool,
    pub jocuri: Vec<Value>,
    pub motiv: Option<String>,
}

/// Lista T2: țintele din LANT (crescător) + ținta programată; ținta implicită = ultima COMPLETA.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TargetList {
    pub tinte: Vec<TargetStatus>,
    pub implicita: String,
    pub programata: String,
    pub regula: &'static str,
}

pub fn target_list(root: &Path) -> Result<TargetList, String> {
    let lant = load_lant(root)?;
    let mut dates: Vec<String> = lant.iter().map(|e| e.target_date.clone()).collect();
    dates.sort();
    dates.dedup();
    let mut tinte = Vec::new();
    let mut last: Option<Target> = None;
    for d in &dates {
        let t = parse_target(d)?;
        last = Some(t);
        let mut jocuri = Vec::new();
        let mut motive = Vec::new();
        for g in GAMES {
            match certify_emission(root, &lant, g, &t) {
                Ok(Some(c)) => jocuri.push(serde_json::json!({"joc": g, "certificata": true, "sha256": c.sha256})),
                Ok(None) => {
                    motive.push(format!("{g}: emisie absentă"));
                    jocuri.push(serde_json::json!({"joc": g, "certificata": false, "sha256": null}));
                }
                Err(e) => {
                    motive.push(format!("{g}: {e}"));
                    jocuri.push(serde_json::json!({"joc": g, "certificata": false, "sha256": null}));
                }
            }
        }
        tinte.push(TargetStatus {
            tinta: d.clone(),
            stare: if motive.is_empty() { "COMPLETA" } else { "PARTIALA" }.to_string(),
            programata: false,
            jocuri,
            motiv: (!motive.is_empty()).then(|| motive.join("; ")),
        });
    }
    let last = last.ok_or("LANT nu conține nicio țintă")?;
    let next = next_draw_after(&last);
    tinte.push(TargetStatus {
        tinta: next.iso(),
        stare: "EMISIE_ABSENTA".to_string(),
        programata: true,
        jocuri: GAMES.iter().map(|g| serde_json::json!({"joc": g, "certificata": false, "sha256": null})).collect(),
        motiv: Some("ținta programată: emisia nu există încă în LANT".to_string()),
    });
    let implicita = tinte
        .iter()
        .rev()
        .find(|t| t.stare == "COMPLETA")
        .map(|t| t.tinta.clone())
        .ok_or("nicio țintă COMPLETA în LANT")?;
    Ok(TargetList {
        tinte,
        implicita,
        programata: next.iso(),
        regula:
            "ținte = LANT.jsonl (certificare SHA-256 per emisie) + prima zi de extragere (joi/duminică) după ultima \
                 țintă din LANT; implicită = ultima țintă COMPLETA (amendamentul S20-T01)",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &[u8]) -> String {
        sha256_hex(s)
    }

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(hex(&vec![b'a'; 1_000_000]), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn sha256_incremental_equals_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        for split in [0, 1, 55, 56, 63, 64, 65, 127, 128, 500, 999, 1000] {
            let mut s = Sha256::new();
            s.update(&data[..split]);
            s.update(&data[split..]);
            assert_eq!(s.finalize(), sha256(&data), "split {split}");
        }
        // lungimi la granița blocului de umplutură
        for len in 50..70 {
            let mut s = Sha256::new();
            for b in &data[..len] {
                s.update(std::slice::from_ref(b));
            }
            assert_eq!(s.finalize(), sha256(&data[..len]));
        }
    }

    #[test]
    fn calendar_roundtrip_and_weekdays() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(weekday(0), 3);
        for z in -800_000..800_000i64 {
            if z % 997 == 0 || (-3000..3000).contains(&z) {
                let (y, m, d) = civil_from_days(z);
                assert_eq!(days_from_civil(y, m, d), z);
            }
        }
        // 2026-10-29 = joi, 2026-11-01 = duminică, 2026-10-30 = vineri
        assert_eq!(weekday(days_from_civil(2026, 10, 29)), 3);
        assert_eq!(weekday(days_from_civil(2026, 11, 1)), 6);
        assert_eq!(weekday(days_from_civil(2026, 10, 30)), 4);
        assert_eq!(next_draw_after(&parse_target("2026-10-29").unwrap()).iso(), "2026-11-01");
        assert_eq!(next_draw_after(&parse_target("2026-11-01").unwrap()).iso(), "2026-11-05");
        assert_eq!(next_draw_after(&parse_target("2026-12-31").unwrap()).iso(), "2027-01-03");
    }

    #[test]
    fn target_validation_refusals() {
        for bad in ["", "2026-11-1", "../x/2026", "2026/11/01", "2026-02-30", "2026-13-01", "2026-10-30", "２026-11-01"]
        {
            assert!(parse_target(bad).is_err(), "{bad}");
        }
        assert!(parse_target("2028-02-29").is_err()); // marți
        assert_eq!(parse_target("2026-10-29").unwrap().iso(), "2026-10-29");
        assert_eq!(parse_target("2026-11-01").unwrap().iso(), "2026-11-01");
    }

    struct TempRoot(PathBuf);

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_root(tag: &str) -> TempRoot {
        let p = std::env::temp_dir().join(format!("s20_integrity_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(p.join("tu4/tribunal/emisii")).unwrap();
        TempRoot(p)
    }

    fn write_emission(root: &Path, game: &str, target: &str, body: &str) -> String {
        let f = emission_file(root, game, target);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(&f, body).unwrap();
        sha256_hex(body.as_bytes())
    }

    fn lant_line(i: u64, game: &str, target: &str, sha: &str) -> String {
        serde_json::json!({"i": i, "game": game, "target_date": target, "file": format!("emisii/{game}/{target}.json"),
            "sha256": sha, "emitted_utc": "2026-10-27T00:00:00Z", "prev_line_sha256": "x"})
        .to_string()
    }

    #[test]
    fn certification_on_synthetic_roots() {
        let r = temp_root("cert");
        let root = r.0.as_path();
        let t = parse_target("2026-10-29").unwrap();
        let mut lines = Vec::new();
        for (i, g) in GAMES.iter().enumerate() {
            let body = format!("{{\"game\": \"{g}\", \"target_date\": \"2026-10-29\", \"x\": {i}}}");
            let sha = write_emission(root, g, "2026-10-29", &body);
            lines.push(lant_line(i as u64, g, "2026-10-29", &sha));
        }
        std::fs::write(lant_path(root), lines.join("\n") + "\n").unwrap();
        let lant = load_lant(root).unwrap();
        for g in GAMES {
            let c = certify_emission(root, &lant, g, &t).unwrap().unwrap();
            assert_eq!(c.game, g);
        }
        let tl = target_list(root).unwrap();
        assert_eq!(tl.implicita, "2026-10-29");
        assert_eq!(tl.programata, "2026-11-01");
        assert_eq!(tl.tinte.len(), 2);
        assert_eq!(tl.tinte[0].stare, "COMPLETA");
        assert_eq!(tl.tinte[1].stare, "EMISIE_ABSENTA");
        // ținta programată: fișier absent + fără linie LANT ⇒ None
        let t2 = parse_target("2026-11-01").unwrap();
        assert!(certify_emission(root, &lant, "649", &t2).unwrap().is_none());
        // fișier modificat ⇒ sha absent din LANT ⇒ refuz
        write_emission(root, "540", "2026-10-29", "{\"game\": \"540\", \"target_date\": \"2026-10-29\", \"x\": 99}");
        assert!(certify_emission(root, &lant, "540", &t).is_err());
        // singura țintă devine PARTIALA ⇒ nicio țintă COMPLETA ⇒ eroare (T2)
        assert!(target_list(root).unwrap_err().contains("COMPLETA"));
        // o țintă anterioară completă (duminică 2026-10-25) ⇒ implicita = 2026-10-25, 2026-10-29 = PARTIALA
        for (i, g) in GAMES.iter().enumerate() {
            let body = format!("{{\"game\": \"{g}\", \"target_date\": \"2026-10-25\"}}");
            let sha = write_emission(root, g, "2026-10-25", &body);
            lines.push(lant_line(10 + i as u64, g, "2026-10-25", &sha));
        }
        std::fs::write(lant_path(root), lines.join("\n") + "\n").unwrap();
        let tl = target_list(root).unwrap();
        let states: Vec<(&str, &str)> = tl.tinte.iter().map(|t| (t.tinta.as_str(), t.stare.as_str())).collect();
        assert_eq!(
            states,
            vec![("2026-10-25", "COMPLETA"), ("2026-10-29", "PARTIALA"), ("2026-11-01", "EMISIE_ABSENTA")]
        );
        assert_eq!(tl.implicita, "2026-10-25");
        assert!(tl.tinte[1].motiv.as_deref().unwrap().contains("540"));
        let lant = load_lant(root).unwrap();
        // emisie fără linie LANT, dar prezentă ⇒ refuz
        write_emission(root, "649", "2026-11-01", "{\"game\": \"649\", \"target_date\": \"2026-11-01\"}");
        assert!(certify_emission(root, &lant, "649", &t2).is_err());
    }

    #[test]
    fn certification_mismatch_and_missing_file() {
        let r = temp_root("mismatch");
        let root = r.0.as_path();
        let t = parse_target("2026-10-29").unwrap();
        // câmpul game din emisie ≠ jocul ⇒ refuz, chiar dacă sha-ul e în LANT
        let sha = write_emission(root, "649", "2026-10-29", "{\"game\": \"540\", \"target_date\": \"2026-10-29\"}");
        // linie LANT pentru joker fără fișier ⇒ lanț inconsistent
        let lines = [lant_line(0, "649", "2026-10-29", &sha), lant_line(1, "joker", "2026-10-29", "00")];
        std::fs::write(lant_path(root), lines.join("\n") + "\n").unwrap();
        let lant = load_lant(root).unwrap();
        assert!(certify_emission(root, &lant, "649", &t).unwrap_err().contains("câmpul"));
        assert!(certify_emission(root, &lant, "joker", &t).unwrap_err().contains("lipsește"));
        // LANT corupt ⇒ eroare
        std::fs::write(lant_path(root), "{nu e json}\n").unwrap();
        assert!(load_lant(root).is_err());
    }

    #[test]
    fn real_lant_hashes_match_files() {
        let root = crate::bridge::ql_root();
        if !lant_path(&root).exists() {
            return;
        }
        let lant = load_lant(&root).unwrap();
        assert!(lant.len() >= 3);
        for e in &lant {
            let b = std::fs::read(root.join("tu4/tribunal").join(&e.file)).unwrap();
            assert_eq!(sha256_hex(&b), e.sha256, "{}", e.file);
        }
        let tl = target_list(&root).unwrap();
        assert!(tl.tinte.iter().any(|t| t.tinta == "2026-10-29" && t.stare == "COMPLETA"));
    }
}

//! Parser nativ al arhivelor istorice `.jsonl` (6/49, 5/40, Joker) — protocol S18 §2.
//!
//! Fluxul I/O (decizia R3): fișierul se deschide și se inspectează ASINCRON prin `tokio::fs`;
//! descriptorul este apoi mapat read-only cu `memmap2::Mmap` într-un fir `spawn_blocking`
//! (maparea și accesul la pagini sunt operații sincrone, deci nu blochează runtime-ul async).
//! Liniile sunt delimitate direct pe octeți (`\n`, cu `\r` final eliminat), iar fiecare linie este
//! deserializată cu `serde_json::from_slice` în structuri care ÎMPRUMUTĂ șirurile din mapare
//! (fără copiere) și bilele printr-un vizitor fără alocare pe heap.
//!
//! Rezultatul este un vector de [`Extraction`] — valori `Copy` de dimensiune fixă, fără heap și
//! fără destructor (`needs_drop == false`); maparea se eliberează automat (RAII) la ieșirea din
//! funcție. Liniile invalide NU se repară și NU se ghicesc: se resping cu număr de linie și motiv.
//!
//! Scheme acceptate:
//! - (A) `tu3`/LIVE: `game`, `draw_date`, `numbers`, `joker`, `draw_key`;
//! - (B) `probe_date` 6/49: fără `game`, cu `numbers_sorted`, `draw_date`, `draw_key` (`649:...`).

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

/// Numărul maxim de bile principale într-o extragere (6/49 și 5/40 extrag câte 6; Joker 5 + bila).
pub const MAX_BALLS: usize = 6;

/// Jocurile arhivate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Game {
    #[serde(rename = "649")]
    Loto649,
    #[serde(rename = "540")]
    Loto540,
    #[serde(rename = "joker")]
    Joker,
}

impl Game {
    pub const ALL: [Game; 3] = [Game::Loto649, Game::Loto540, Game::Joker];

    /// Eticheta din arhivă (`"649"`, `"540"`, `"joker"`).
    pub fn from_tag(tag: &str) -> Option<Game> {
        match tag {
            "649" => Some(Game::Loto649),
            "540" => Some(Game::Loto540),
            "joker" => Some(Game::Joker),
            _ => None,
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Game::Loto649 => "649",
            Game::Loto540 => "540",
            Game::Joker => "joker",
        }
    }

    /// Numărul de bile principale extrase.
    pub fn main_balls(self) -> usize {
        match self {
            Game::Loto649 | Game::Loto540 => 6,
            Game::Joker => 5,
        }
    }

    /// Eticheta maximă a urnei principale.
    pub fn main_max(self) -> u8 {
        match self {
            Game::Loto649 => 49,
            Game::Loto540 => 40,
            Game::Joker => 45,
        }
    }

    /// Eticheta maximă a bilei suplimentare (numai Joker: 1..20).
    pub fn bonus_max(self) -> Option<u8> {
        match self {
            Game::Joker => Some(20),
            Game::Loto649 | Game::Loto540 => None,
        }
    }
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// Data extragerii (calendar gregorian, validată exact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DrawDate {
    pub year: u16,
    pub month: u8,
    pub day: u8,
}

fn is_leap(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

impl DrawDate {
    /// Parsează exact `AAAA-LL-ZZ` (10 octeți ASCII); orice altă formă întoarce `None`.
    pub fn parse(s: &[u8]) -> Option<DrawDate> {
        if s.len() != 10 || s[4] != b'-' || s[7] != b'-' {
            return None;
        }
        let digits = |r: &[u8]| -> Option<u32> {
            r.iter().try_fold(
                0u32,
                |acc, &b| {
                    if b.is_ascii_digit() {
                        Some(acc * 10 + u32::from(b - b'0'))
                    } else {
                        None
                    }
                },
            )
        };
        let (y, m, d) = (digits(&s[0..4])?, digits(&s[5..7])?, digits(&s[8..10])?);
        if y == 0 || !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
            return None;
        }
        Some(DrawDate { year: y as u16, month: m as u8, day: d as u8 })
    }
}

impl fmt::Display for DrawDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl Serialize for DrawDate {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// O extragere validată: valoare `Copy`, dimensiune fixă, fără heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extraction {
    pub game: Game,
    pub date: DrawDate,
    numbers: [u8; MAX_BALLS],
    len: u8,
    /// Bila Joker (1..20) — numai pentru `Game::Joker`.
    pub bonus: Option<u8>,
    /// Numărul liniei (1-based) în fișierul sursă.
    pub line: u64,
}

impl Extraction {
    /// Bilele principale, sortate crescător, distincte și în intervalul jocului.
    pub fn numbers(&self) -> &[u8] {
        &self.numbers[..usize::from(self.len)]
    }

    pub fn contains(&self, label: u8) -> bool {
        self.numbers().binary_search(&label).is_ok()
    }
}

impl Serialize for Extraction {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Extraction", 5)?;
        s.serialize_field("game", &self.game)?;
        s.serialize_field("date", &self.date)?;
        s.serialize_field("numbers", self.numbers())?;
        s.serialize_field("bonus", &self.bonus)?;
        s.serialize_field("line", &self.line)?;
        s.end()
    }
}

/// Motivul exact al respingerii unei linii.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RejectReason {
    Json { message: String },
    UnknownGame { tag: Option<String> },
    GameConflict { game_field: String, key_prefix: String },
    MissingDate,
    BadDate { value: String },
    MissingNumbers,
    NumbersConflict,
    WrongCount { expected: usize, found: usize },
    OutOfRange { value: u8, max: u8 },
    Duplicate { value: u8 },
    MissingBonus,
    UnexpectedBonus { value: u8 },
    BonusOutOfRange { value: u8, max: u8 },
}

/// O linie respinsă.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Reject {
    pub line: u64,
    pub reason: RejectReason,
}

/// Lista de bile deserializată fără heap (cel mult [`MAX_BALLS`] valori `u8`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Balls {
    v: [u8; MAX_BALLS],
    len: u8,
}

impl Balls {
    fn as_slice(&self) -> &[u8] {
        &self.v[..usize::from(self.len)]
    }

    fn sorted(mut self) -> Balls {
        let n = usize::from(self.len);
        self.v[..n].sort_unstable();
        self
    }
}

struct BallsVisitor;

impl<'de> Visitor<'de> for BallsVisitor {
    type Value = Balls;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "o listă de cel mult {MAX_BALLS} întregi 0..255")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Balls, A::Error> {
        let mut b = Balls::default();
        while let Some(x) = seq.next_element::<u64>()? {
            if usize::from(b.len) >= MAX_BALLS {
                return Err(de::Error::invalid_length(MAX_BALLS + 1, &self));
            }
            let x = u8::try_from(x).map_err(|_| de::Error::custom(format!("bila {x} depășește 255")))?;
            b.v[usize::from(b.len)] = x;
            b.len += 1;
        }
        Ok(b)
    }
}

impl<'de> Deserialize<'de> for Balls {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Balls, D::Error> {
        deserializer.deserialize_seq(BallsVisitor)
    }
}

/// Vederea împrumutată a unei linii: șirurile indică direct în mapare (fără copiere). Un șir cu
/// secvențe de evadare JSON nu poate fi împrumutat și duce la respingerea liniei (conservator).
#[derive(Deserialize)]
struct RawRecord<'a> {
    #[serde(borrow, default)]
    game: Option<&'a str>,
    #[serde(borrow, default)]
    draw_date: Option<&'a str>,
    #[serde(borrow, default)]
    draw_key: Option<&'a str>,
    #[serde(default)]
    numbers: Option<Balls>,
    #[serde(default)]
    numbers_sorted: Option<Balls>,
    #[serde(default)]
    joker: Option<u8>,
}

fn resolve_game(raw: &RawRecord<'_>, hint: Option<Game>) -> Result<Game, RejectReason> {
    let key_prefix = raw.draw_key.and_then(|k| k.split_once(':').map(|(p, _)| p));
    let unknown = |t: &str| RejectReason::UnknownGame { tag: Some(t.to_owned()) };
    match (raw.game, key_prefix) {
        (Some(g), Some(p)) => {
            let game = Game::from_tag(g).ok_or_else(|| unknown(g))?;
            if Game::from_tag(p) != Some(game) {
                return Err(RejectReason::GameConflict { game_field: g.to_owned(), key_prefix: p.to_owned() });
            }
            Ok(game)
        }
        (Some(g), None) => Game::from_tag(g).ok_or_else(|| unknown(g)),
        (None, Some(p)) => Game::from_tag(p).ok_or_else(|| unknown(p)),
        (None, None) => hint.ok_or(RejectReason::UnknownGame { tag: None }),
    }
}

/// Parsează și validează o linie nevidă. Întoarce extragerea și `draw_key` împrumutat din linie.
fn parse_line<'a>(
    line: &'a [u8],
    line_no: u64,
    hint: Option<Game>,
) -> Result<(Extraction, Option<&'a str>), RejectReason> {
    let raw: RawRecord<'a> = serde_json::from_slice(line).map_err(|e| RejectReason::Json { message: e.to_string() })?;
    let game = resolve_game(&raw, hint)?;

    let date_str = raw.draw_date.ok_or(RejectReason::MissingDate)?;
    let date =
        DrawDate::parse(date_str.as_bytes()).ok_or_else(|| RejectReason::BadDate { value: date_str.to_owned() })?;

    let balls = match (raw.numbers, raw.numbers_sorted) {
        (Some(a), Some(b)) => {
            let (a, b) = (a.sorted(), b.sorted());
            if a != b {
                return Err(RejectReason::NumbersConflict);
            }
            a
        }
        (Some(a), None) => a.sorted(),
        (None, Some(b)) => b.sorted(),
        (None, None) => return Err(RejectReason::MissingNumbers),
    };
    let expected = game.main_balls();
    if usize::from(balls.len) != expected {
        return Err(RejectReason::WrongCount { expected, found: usize::from(balls.len) });
    }
    let max = game.main_max();
    let s = balls.as_slice();
    if let Some(&value) = s.iter().find(|&&x| x == 0 || x > max) {
        return Err(RejectReason::OutOfRange { value, max });
    }
    if let Some(w) = s.windows(2).find(|w| w[0] == w[1]) {
        return Err(RejectReason::Duplicate { value: w[0] });
    }

    let bonus = match (game.bonus_max(), raw.joker) {
        (Some(bmax), Some(j)) if j == 0 || j > bmax => {
            return Err(RejectReason::BonusOutOfRange { value: j, max: bmax })
        }
        (Some(_), Some(j)) => Some(j),
        (Some(_), None) => return Err(RejectReason::MissingBonus),
        (None, Some(j)) => return Err(RejectReason::UnexpectedBonus { value: j }),
        (None, None) => None,
    };

    Ok((Extraction { game, date, numbers: balls.v, len: balls.len, bonus, line: line_no }, raw.draw_key))
}

/// Rezultatul parsării unui buffer de octeți.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParseOutcome {
    /// Liniile nevide examinate.
    pub lines: u64,
    pub records: Vec<Extraction>,
    pub rejects: Vec<Reject>,
    /// `draw_key` al ultimei extrageri acceptate (dacă linia îl conține).
    pub last_key: Option<String>,
}

/// Parsează un buffer `.jsonl` direct la nivel de octet (folosit pe mapare și în teste).
pub fn parse_bytes(bytes: &[u8], hint: Option<Game>) -> ParseOutcome {
    let mut out = ParseOutcome { records: Vec::with_capacity(bytes.len() / 160 + 1), ..ParseOutcome::default() };
    let mut last_key: Option<&str> = None;
    for (idx, raw_line) in bytes.split(|&b| b == b'\n').enumerate() {
        let line_no = idx as u64 + 1;
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        out.lines += 1;
        match parse_line(line, line_no, hint) {
            Ok((extraction, key)) => {
                out.records.push(extraction);
                last_key = key;
            }
            Err(reason) => out.rejects.push(Reject { line: line_no, reason }),
        }
    }
    out.last_key = last_key.map(str::to_owned);
    out
}

/// O arhivă încărcată de pe disc.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedArchive {
    pub path: PathBuf,
    pub bytes: u64,
    pub elapsed_micros: u128,
    pub lines: u64,
    pub records: Vec<Extraction>,
    pub rejects: Vec<Reject>,
    pub last_key: Option<String>,
}

/// Rezumatul de inventar (fără nicio statistică de model).
#[derive(Debug, Clone, Serialize)]
pub struct ArchiveSummary {
    pub path: String,
    pub bytes: u64,
    pub lines: u64,
    pub records: usize,
    pub rejects: usize,
    pub reject_samples: Vec<Reject>,
    pub per_game: BTreeMap<String, usize>,
    pub first_date: Option<DrawDate>,
    pub last_date: Option<DrawDate>,
    pub last_key: Option<String>,
    pub dates_non_decreasing: bool,
    pub elapsed_micros: u128,
    pub extraction_size_bytes: usize,
}

impl ParsedArchive {
    pub fn summary(&self) -> ArchiveSummary {
        let mut per_game = BTreeMap::new();
        for r in &self.records {
            *per_game.entry(r.game.tag().to_owned()).or_insert(0usize) += 1;
        }
        ArchiveSummary {
            path: self.path.display().to_string(),
            bytes: self.bytes,
            lines: self.lines,
            records: self.records.len(),
            rejects: self.rejects.len(),
            reject_samples: self.rejects.iter().take(10).cloned().collect(),
            per_game,
            first_date: self.records.first().map(|r| r.date),
            last_date: self.records.last().map(|r| r.date),
            last_key: self.last_key.clone(),
            dates_non_decreasing: self.records.windows(2).all(|w| w[0].date <= w[1].date),
            elapsed_micros: self.elapsed_micros,
            extraction_size_bytes: std::mem::size_of::<Extraction>(),
        }
    }
}

/// Erori de I/O (o linie invalidă NU este o eroare de I/O: devine [`Reject`]).
#[derive(Debug)]
pub enum ParserError {
    Io { path: PathBuf, source: std::io::Error },
    NotAFile { path: PathBuf },
    Join { path: PathBuf, message: String },
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::Io { path, source } => write!(f, "I/O {}: {source}", path.display()),
            ParserError::NotAFile { path } => write!(f, "nu este fișier obișnuit: {}", path.display()),
            ParserError::Join { path, message } => write!(f, "sarcina de parsare {}: {message}", path.display()),
        }
    }
}

impl std::error::Error for ParserError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParserError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn map_and_parse(
    path: PathBuf,
    file: std::fs::File,
    hint: Option<Game>,
    started: Instant,
) -> Result<ParsedArchive, ParserError> {
    let io = |source| ParserError::Io { path: path.clone(), source };
    let len = file.metadata().map_err(io)?.len();
    let (bytes, outcome) = if len == 0 {
        (0, ParseOutcome::default()) // o mapare de lungime 0 nu are sens; arhiva goală = zero linii
    } else {
        // SAFETY: mapare numai-citire; parserul nu scrie niciodată prin ea. Fișierele arhivei sunt
        // write-once (directoarele `versiuni/` sunt imuabile). Dacă un proces extern ar trunchia
        // fișierul în timpul parsării, accesul la pagini ar putea genera SIGBUS — risc documentat
        // în protocolul S18 (R3); de aceea parserul nu se rulează pe fișiere în curs de scriere.
        let map: Mmap = unsafe { MmapOptions::new().map(&file) }.map_err(io)?;
        #[cfg(unix)]
        {
            // Indiciu pentru kernel (citire secvențială); un eșec nu afectează corectitudinea.
            let _ = map.advise(memmap2::Advice::Sequential);
        }
        let outcome = parse_bytes(&map[..], hint);
        (map.len() as u64, outcome)
    }; // `map` se eliberează aici (munmap) — RAII
    Ok(ParsedArchive {
        path,
        bytes,
        elapsed_micros: started.elapsed().as_micros(),
        lines: outcome.lines,
        records: outcome.records,
        rejects: outcome.rejects,
        last_key: outcome.last_key,
    })
}

/// Încarcă o arhivă: deschidere/metadate asincron (`tokio::fs`), mapare + parsare în `spawn_blocking`.
/// `hint` se folosește NUMAI pentru liniile fără `game` și fără `draw_key`.
pub async fn load_archive(path: impl AsRef<Path>, hint: Option<Game>) -> Result<ParsedArchive, ParserError> {
    let started = Instant::now();
    let path = path.as_ref().to_path_buf();
    let file = tokio::fs::File::open(&path).await.map_err(|source| ParserError::Io { path: path.clone(), source })?;
    let meta = file.metadata().await.map_err(|source| ParserError::Io { path: path.clone(), source })?;
    if !meta.is_file() {
        return Err(ParserError::NotAFile { path });
    }
    let std_file = file.into_std().await;
    let task_path = path.clone();
    tokio::task::spawn_blocking(move || map_and_parse(task_path, std_file, hint, started))
        .await
        .map_err(|e| ParserError::Join { path, message: e.to_string() })?
}

/// Încarcă mai multe arhive concurent (`JoinSet`), păstrând ordinea căilor date.
pub async fn load_many(paths: Vec<PathBuf>, hint: Option<Game>) -> Vec<Result<ParsedArchive, ParserError>> {
    let mut set = tokio::task::JoinSet::new();
    for (i, p) in paths.iter().cloned().enumerate() {
        set.spawn(async move { (i, load_archive(p, hint).await) });
    }
    let mut slots: Vec<Option<Result<ParsedArchive, ParserError>>> = (0..paths.len()).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        if let Ok((i, result)) = joined {
            slots[i] = Some(result);
        }
    }
    slots
        .into_iter()
        .zip(paths)
        .map(|(slot, path)| {
            slot.unwrap_or_else(|| Err(ParserError::Join { path, message: "sarcină pierdută".to_owned() }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const L649: &str = r#"{"draw_date": "2026-10-25", "draw_key": "649:2026-10-25:10-27-28-29-33-39", "draw_kind": "regular", "game": "649", "has_source_disagreement": false, "joker": null, "numbers": [10, 27, 28, 29, 33, 39], "numbers_order_as_stored": [33, 10, 27, 39, 28, 29], "provenance": {"database_sha256": "ab", "run_id": "x"}, "source_count": 1, "verification": "official_only"}"#;
    const L540: &str = r#"{"draw_date": "2026-10-25", "draw_key": "540:2026-10-25:1-8-12-21-22-16", "game": "540", "joker": null, "numbers": [1, 8, 12, 16, 21, 22]}"#;
    const LJOK: &str = r#"{"draw_date": "2026-10-25", "draw_key": "joker:2026-10-25:1-10-16-20-39:J8", "game": "joker", "joker": 8, "numbers": [1, 10, 16, 20, 39]}"#;
    const LPROBE: &str = r#"{"ball_measurements": null, "draw_date": "2026-10-18", "draw_key": "649:2026-10-18:19-22-31-38-42-49", "numbers_as_stored": [38, 19, 42, 31, 22, 49], "numbers_sorted": [19, 22, 31, 38, 42, 49], "source_refs": [{"a": 1}]}"#;

    fn one(line: &str, hint: Option<Game>) -> Result<Extraction, RejectReason> {
        let o = parse_bytes(line.as_bytes(), hint);
        assert_eq!(o.lines, 1);
        match (o.records.first(), o.rejects.first()) {
            (Some(e), None) => Ok(*e),
            (None, Some(r)) => Err(r.reason.clone()),
            _ => panic!("exact un rezultat așteptat"),
        }
    }

    #[test]
    fn schema_a_three_games() {
        let e = one(L649, None).unwrap();
        assert_eq!((e.game, e.numbers(), e.bonus), (Game::Loto649, &[10, 27, 28, 29, 33, 39][..], None));
        assert_eq!(e.date.to_string(), "2026-10-25");
        let e = one(L540, None).unwrap();
        assert_eq!((e.game, e.numbers()), (Game::Loto540, &[1, 8, 12, 16, 21, 22][..]));
        let e = one(LJOK, None).unwrap();
        assert_eq!((e.game, e.numbers(), e.bonus), (Game::Joker, &[1, 10, 16, 20, 39][..], Some(8)));
        assert!(e.contains(16) && !e.contains(17));
    }

    #[test]
    fn schema_b_probe_649_by_key_prefix() {
        let e = one(LPROBE, None).unwrap();
        assert_eq!((e.game, e.numbers()), (Game::Loto649, &[19, 22, 31, 38, 42, 49][..]));
    }

    #[test]
    fn unsorted_numbers_are_sorted() {
        let e = one(r#"{"game":"649","draw_date":"2000-01-02","numbers":[49,1,7,3,22,9]}"#, None).unwrap();
        assert_eq!(e.numbers(), &[1, 3, 7, 9, 22, 49]);
    }

    #[test]
    fn rejections_are_exact() {
        use RejectReason as R;
        let cases: Vec<(&str, R)> = vec![
            (r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,1,7,3,22,9]}"#, R::Duplicate { value: 1 }),
            (
                r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,2,3,4,5,50]}"#,
                R::OutOfRange { value: 50, max: 49 },
            ),
            (r#"{"game":"649","draw_date":"2000-01-02","numbers":[0,2,3,4,5,6]}"#, R::OutOfRange { value: 0, max: 49 }),
            (
                r#"{"game":"540","draw_date":"2000-01-02","numbers":[1,2,3,4,5,41]}"#,
                R::OutOfRange { value: 41, max: 40 },
            ),
            (
                r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,2,3,4,5]}"#,
                R::WrongCount { expected: 6, found: 5 },
            ),
            (
                r#"{"game":"joker","draw_date":"2000-01-02","numbers":[1,2,3,4,5,6],"joker":3}"#,
                R::WrongCount { expected: 5, found: 6 },
            ),
            (
                r#"{"game":"joker","draw_date":"2000-01-02","numbers":[1,2,3,4,5],"joker":21}"#,
                R::BonusOutOfRange { value: 21, max: 20 },
            ),
            (
                r#"{"game":"joker","draw_date":"2000-01-02","numbers":[1,2,3,4,5],"joker":0}"#,
                R::BonusOutOfRange { value: 0, max: 20 },
            ),
            (r#"{"game":"joker","draw_date":"2000-01-02","numbers":[1,2,3,4,5],"joker":null}"#, R::MissingBonus),
            (
                r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,2,3,4,5,6],"joker":4}"#,
                R::UnexpectedBonus { value: 4 },
            ),
            (
                r#"{"game":"649","draw_date":"2026-13-01","numbers":[1,2,3,4,5,6]}"#,
                R::BadDate { value: "2026-13-01".into() },
            ),
            (
                r#"{"game":"649","draw_date":"2026-02-29","numbers":[1,2,3,4,5,6]}"#,
                R::BadDate { value: "2026-02-29".into() },
            ),
            (r#"{"game":"649","numbers":[1,2,3,4,5,6]}"#, R::MissingDate),
            (r#"{"game":"649","draw_date":"2000-01-02"}"#, R::MissingNumbers),
            (
                r#"{"game":"keno","draw_date":"2000-01-02","numbers":[1,2,3,4,5,6]}"#,
                R::UnknownGame { tag: Some("keno".into()) },
            ),
            (r#"{"draw_date":"2000-01-02","numbers":[1,2,3,4,5,6]}"#, R::UnknownGame { tag: None }),
            (
                r#"{"game":"540","draw_key":"649:2000-01-02:1-2-3-4-5-6","draw_date":"2000-01-02","numbers":[1,2,3,4,5,6]}"#,
                R::GameConflict { game_field: "540".into(), key_prefix: "649".into() },
            ),
            (
                r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,2,3,4,5,6],"numbers_sorted":[1,2,3,4,5,7]}"#,
                R::NumbersConflict,
            ),
        ];
        for (line, want) in cases {
            assert_eq!(one(line, None), Err(want), "{line}");
        }
        assert!(matches!(
            one(r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,2,3,4,5,6,7]}"#, None),
            Err(R::Json { .. })
        ));
        assert!(matches!(
            one(r#"{"game":"649","draw_date":"2000-01-02","numbers":[1,2,3,4,5,300]}"#, None),
            Err(R::Json { .. })
        ));
        assert!(matches!(one(r#"{"game":"649","draw_date": "#, None), Err(R::Json { .. })));
        assert!(matches!(one("nu este json", None), Err(R::Json { .. })));
    }

    #[test]
    fn leap_day_is_valid() {
        assert!(DrawDate::parse(b"2024-02-29").is_some());
        assert!(DrawDate::parse(b"1900-02-29").is_none());
        assert!(DrawDate::parse(b"2000-02-29").is_some());
        assert!(DrawDate::parse(b"2000-1-02").is_none());
    }

    #[test]
    fn hint_is_fallback_only() {
        let line = r#"{"draw_date":"2000-01-02","numbers":[1,2,3,4,5,6]}"#;
        assert_eq!(one(line, Some(Game::Loto540)).unwrap().game, Game::Loto540);
        // `game` explicit câștigă în fața indiciului
        assert_eq!(one(L649, Some(Game::Loto540)).unwrap().game, Game::Loto649);
    }

    #[test]
    fn empty_crlf_blank_and_no_trailing_newline() {
        assert_eq!(parse_bytes(b"", None), ParseOutcome::default());
        let text = format!("{L649}\r\n\r\n   \n{L540}\r\n{LJOK}");
        let o = parse_bytes(text.as_bytes(), None);
        assert_eq!((o.lines, o.records.len(), o.rejects.len()), (3, 3, 0));
        assert_eq!(o.records.iter().map(|r| r.line).collect::<Vec<_>>(), vec![1, 4, 5]);
        assert_eq!(o.last_key.as_deref(), Some("joker:2026-10-25:1-10-16-20-39:J8"));
    }

    #[test]
    fn reject_line_numbers_and_continuation() {
        let text = format!("{L649}\n{{bad\n{L540}\n");
        let o = parse_bytes(text.as_bytes(), None);
        assert_eq!((o.records.len(), o.rejects.len(), o.rejects[0].line), (2, 1, 2));
        assert_eq!(o.last_key.as_deref(), Some("540:2026-10-25:1-8-12-21-22-16"));
    }

    #[test]
    fn extraction_is_fixed_size_without_heap() {
        assert!(!std::mem::needs_drop::<Extraction>());
        assert!(std::mem::size_of::<Extraction>() <= 24);
    }

    fn temp_file(name: &str, content: &[u8]) -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("rsc_s18_{}_{nanos}_{name}", std::process::id()));
        std::fs::write(&p, content).unwrap();
        p
    }

    #[tokio::test]
    async fn async_mmap_load_and_many_preserve_order() {
        let a = temp_file("a.jsonl", format!("{L649}\n{LPROBE}\n").as_bytes());
        let b = temp_file("b.jsonl", format!("{LJOK}\n").as_bytes());
        let empty = temp_file("empty.jsonl", b"");
        let missing = std::env::temp_dir().join("rsc_s18_nu_exista.jsonl");
        let res = load_many(vec![a.clone(), b.clone(), empty.clone(), missing.clone()], None).await;
        let ra = res[0].as_ref().unwrap();
        assert_eq!((ra.records.len(), ra.rejects.len(), ra.path.clone()), (2, 0, a.clone()));
        let s = ra.summary();
        assert_eq!(s.per_game.get("649"), Some(&2));
        assert!(!s.dates_non_decreasing); // 2026-10-25 urmat de 2026-10-18 — raportat, nu reparat
        assert_eq!(res[1].as_ref().unwrap().records[0].bonus, Some(8));
        let re = res[2].as_ref().unwrap();
        assert_eq!((re.bytes, re.lines, re.records.len()), (0, 0, 0));
        assert!(matches!(res[3], Err(ParserError::Io { .. })));
        let dir = load_archive(std::env::temp_dir(), None).await;
        assert!(matches!(dir, Err(ParserError::NotAFile { .. })));
        for p in [a, b, empty] {
            std::fs::remove_file(p).unwrap();
        }
    }
}

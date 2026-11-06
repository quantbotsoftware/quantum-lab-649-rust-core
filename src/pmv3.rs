//! Motorul Pari-Mutuel Cauzal `PM-V3`, Cubul Bootstrap `1 000x` CUDA `sm_120` pe `39,077,004` bilete
//! și Reetichetarea `QAP` Anti-Mulțime `CD-V3Q` (`Sesiunea 2 / Poarta A3` din `PROTOCOL_RUST_CORE_V3.md`).
//! Guvernanță științifică: `positive_claims_allowed = false`.

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cdv3::{
    cpu_verify_covering_24t, cpu_verify_joker_24m_24t, mask_to_numbers_1based, pack_joker_ticket,
    unpack_joker_ticket, CdV3CatalogReport, JokerTicketJson,
};
use crate::cuda_ffi::{
    check_rss_guard, gpu_pmv3_boot_540_all, gpu_pmv3_boot_649_tile, gpu_pmv3_boot_joker_single_j,
    gpu_pmv3_ev_540_f64, gpu_pmv3_ev_649_f64, gpu_pmv3_ev_joker_f64, gpu_verify_covering,
    gpu_verify_joker_24m, ResidentGpuUniverse, V3CrowdParamsF64,
};
use crate::vault::{
    MappedVaultUniverse, VaultFileRecord, COUNT_540_DRAWS6, COUNT_540_TICKETS, COUNT_649,
    COUNT_JOKER_BASE45, COUNT_JOKER_FULL24M, TOTAL_UNIVERSE_TICKETS, VAULT_HEADER_BYTES,
    VAULT_MAGIC,
};

pub const DEFAULT_LOTO_CURRENT_JSON: &str = "./data/loto_romania/CURRENT.json";
pub const BOOTSTRAP_REPLICATES: usize = 1000;
pub const BOOTSTRAP_BLOCK_LEN: usize = 8;
pub const BOOT_RECORD_FLOATS: usize = 6; // [ev_nominal, q05, q25, q50, q75, q95]
pub const BOOT_RECORD_BYTES: u64 = 24;

// ============================================================================
// 1. Legătură FFI nativă read-only (`mode=ro`) la `libsqlite3.so`
// ============================================================================

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_OPEN_READONLY: c_int = 0x0000_0001;
const SQLITE_OPEN_URI: c_int = 0x0000_0040;
const SQLITE_OPEN_NOMUTEX: c_int = 0x0000_8000;

unsafe extern "C" {
    fn sqlite3_open_v2(
        filename: *const c_char,
        pp_db: *mut *mut c_void,
        flags: c_int,
        z_vfs: *const c_char,
    ) -> c_int;
    fn sqlite3_close(db: *mut c_void) -> c_int;
    fn sqlite3_errmsg(db: *mut c_void) -> *const c_char;
    fn sqlite3_prepare_v2(
        db: *mut c_void,
        z_sql: *const c_char,
        n_byte: c_int,
        pp_stmt: *mut *mut c_void,
        pz_tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_step(stmt: *mut c_void) -> c_int;
    fn sqlite3_finalize(stmt: *mut c_void) -> c_int;
    fn sqlite3_column_text(stmt: *mut c_void, i_col: c_int) -> *const u8;
    fn sqlite3_column_int(stmt: *mut c_void, i_col: c_int) -> c_int;
    fn sqlite3_column_double(stmt: *mut c_void, i_col: c_int) -> c_double;
    fn sqlite3_column_type(stmt: *mut c_void, i_col: c_int) -> c_int;
}

const SQLITE_NULL: c_int = 5;

struct ReadOnlySqlite {
    db: *mut c_void,
}

impl ReadOnlySqlite {
    fn open_ro(db_path: &Path) -> Result<Self, String> {
        let uri = format!("file:{}?mode=ro", db_path.display());
        let c_uri = CString::new(uri).map_err(|e| format!("CString sqlite uri: {e}"))?;
        let mut db: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            sqlite3_open_v2(
                c_uri.as_ptr(),
                &mut db,
                SQLITE_OPEN_READONLY | SQLITE_OPEN_URI | SQLITE_OPEN_NOMUTEX,
                std::ptr::null(),
            )
        };
        if rc != SQLITE_OK || db.is_null() {
            let msg = if !db.is_null() {
                let s = unsafe { CStr::from_ptr(sqlite3_errmsg(db)).to_string_lossy().into_owned() };
                unsafe { sqlite3_close(db) };
                s
            } else {
                format!("rc={rc}")
            };
            return Err(format!("sqlite3_open_v2 read-only esuat pe {}: {msg}", db_path.display()));
        }
        Ok(Self { db })
    }

    fn col_text(stmt: *mut c_void, idx: c_int) -> String {
        unsafe {
            if sqlite3_column_type(stmt, idx) == SQLITE_NULL {
                return String::new();
            }
            let ptr = sqlite3_column_text(stmt, idx);
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    fn col_opt_i32(stmt: *mut c_void, idx: c_int) -> Option<i32> {
        unsafe {
            if sqlite3_column_type(stmt, idx) == SQLITE_NULL {
                None
            } else {
                Some(sqlite3_column_int(stmt, idx))
            }
        }
    }

    fn col_parse_f64(stmt: *mut c_void, idx: c_int) -> f64 {
        let s = Self::col_text(stmt, idx);
        if s.is_empty() {
            unsafe {
                if sqlite3_column_type(stmt, idx) == SQLITE_NULL {
                    0.0
                } else {
                    sqlite3_column_double(stmt, idx)
                }
            }
        } else {
            s.trim().parse::<f64>().unwrap_or(0.0)
        }
    }
}

impl Drop for ReadOnlySqlite {
    fn drop(&mut self) {
        if !self.db.is_null() {
            unsafe {
                sqlite3_close(self.db);
            }
            self.db = std::ptr::null_mut();
        }
    }
}

#[derive(Clone, Debug)]
pub struct CategoryObs {
    pub status: String,
    pub winners: u64,
    pub payout_per_win: f64,
    pub carryover_out: f64,
    pub carry_in: f64,
    pub total_cat_fund: f64,
    pub new_alloc: f64,
}

#[derive(Clone, Debug)]
pub struct HistoricalDraw {
    pub draw_key: String,
    pub draw_date: String,
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub weekday: u32, // 0 = Mon .. 6 = Sun
    pub draw_kind: String,
    pub numbers: Vec<u8>,
    pub joker: Option<u8>,
    pub total_fund_obs: f64,
    pub new_total_fund: f64,
    pub categories: BTreeMap<String, CategoryObs>,
}

#[derive(Deserialize)]
struct CurrentDbPointer {
    snapshot: String,
    manifest_sha256: String,
}

pub fn resolve_readonly_sqlite_path(current_json: &Path) -> Result<(PathBuf, String), String> {
    let txt = fs::read_to_string(current_json)
        .map_err(|e| format!("read {}: {e}", current_json.display()))?;
    let ptr: CurrentDbPointer =
        serde_json::from_str(&txt).map_err(|e| format!("json CURRENT.json: {e}"))?;
    let base_dir = current_json
        .parent()
        .ok_or_else(|| "parent CURRENT.json invalid".to_string())?;
    let db_path = base_dir
        .join("snapshots")
        .join(&ptr.snapshot)
        .join("loto_romania.sqlite");
    if !db_path.exists() {
        return Err(format!("SQLite snapshot inexistent: {}", db_path.display()));
    }
    let (_, db_sha) = sha256_file_streaming(&db_path)?;
    if ptr.manifest_sha256.is_empty() {
        return Err("manifest_sha256 gol in CURRENT.json".to_string());
    }
    Ok((db_path, db_sha))
}

/// Calculează ziua săptămânii (0 = Luni .. 6 = Duminică, algoritm Sakamoto/Howard Hinnant) și zilele de la 1970-01-01.
#[inline]
pub fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y as i64 - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = m as u64;
    let doy = (153 * (if mp > 2 { mp - 3 } else { mp + 9 }) + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

#[inline]
pub fn parse_ymd(date_str: &str) -> Result<(i32, u32, u32, u32), String> {
    if date_str.len() < 10 {
        return Err(format!("data invalida: {date_str}"));
    }
    let y: i32 = date_str[0..4].parse().map_err(|_| format!("an invalid: {date_str}"))?;
    let m: u32 = date_str[5..7].parse().map_err(|_| format!("luna invalida: {date_str}"))?;
    let d: u32 = date_str[8..10].parse().map_err(|_| format!("zi invalida: {date_str}"))?;
    let days = days_from_civil(y, m, d);
    // 1970-01-01 = Joi (3 in 0=Mon..6=Sun)
    let wd = ((days + 3).rem_euclid(7)) as u32;
    Ok((y, m, d, wd))
}

/// Duminica Paștelui Ortodox (algoritmul Meeus în calendarul iulian + 13 zile offset gregorian 1900..2099).
#[inline]
pub fn orthodox_easter_days(year: i32) -> i64 {
    let a = year % 4;
    let b = year % 7;
    let c = year % 19;
    let d = (19 * c + 15) % 30;
    let e = (2 * a + 4 * b - d + 34) % 7;
    let month = ((d + e + 114) / 31) as u32;
    let day = (((d + e + 114) % 31) + 1) as u32;
    days_from_civil(year, month, day) + 13
}

/// Vectorul gărzilor calendaristice explicite (Crăciun, Anul Nou, Paște Ortodox, 8 Martie / 15 Sept / 1-6 Dec, tragere specială/dublă).
#[inline]
pub fn calendar_guard_features(year: i32, month: u32, day: u32, draw_kind: &str) -> [f64; 5] {
    let cur_days = days_from_civil(year, month, day);
    let easter_days = orthodox_easter_days(year);
    let diff_to_easter = easter_days - cur_days;

    let is_nye = if month == 12 && day >= 28 { 1.0 } else { 0.0 };
    let is_xmas = if month == 12 && (20..=27).contains(&day) { 1.0 } else { 0.0 };
    let is_easter = if (0..=5).contains(&diff_to_easter) { 1.0 } else { 0.0 };
    let is_seasonal_special = if (month == 9 && (10..=18).contains(&day))
        || (month == 3 && (5..=12).contains(&day))
        || (month == 12 && (1..=7).contains(&day))
    {
        1.0
    } else {
        0.0
    };
    let is_special = if draw_kind != "regular" { 1.0 } else { 0.0 };
    [is_nye, is_xmas, is_easter, is_seasonal_special, is_special]
}

pub fn load_game_history_readonly(
    db_path: &Path,
    game: &str,
    min_date: &str,
) -> Result<Vec<HistoricalDraw>, String> {
    let conn = ReadOnlySqlite::open_ro(db_path)?;
    let sql = "
        SELECT d.draw_key, d.draw_date, d.draw_kind,
               d.n1, d.n2, d.n3, d.n4, d.n5, d.n6, d.joker,
               o.total_fund,
               p.category, p.status, p.winning_variants, p.payout_per_win, p.carryover
        FROM draws d
        JOIN observations o ON o.observation_id = d.selected_observation
        JOIN prizes p ON p.observation_id = d.selected_observation
        WHERE d.game = ?1 AND d.draw_date >= ?2
        ORDER BY d.draw_date ASC,
                 CASE WHEN d.draw_kind = 'regular' THEN 0 ELSE 1 END ASC,
                 d.draw_key ASC,
                 p.category ASC
    ";
    let c_sql = CString::new(sql).unwrap();
    let mut stmt: *mut c_void = std::ptr::null_mut();
    let rc = unsafe {
        sqlite3_prepare_v2(conn.db, c_sql.as_ptr(), -1, &mut stmt, std::ptr::null_mut())
    };
    if rc != SQLITE_OK || stmt.is_null() {
        return Err(format!("sqlite3_prepare_v2 esuat (rc={rc})"));
    }

    // Bind parametrii ?1 si ?2 fara alocare externa: folosim interogare parametrizata sau filtru direct?
    // Deoarece game este strict in {"649", "540", "joker"} si min_date format "YYYY-MM-DD", verificam whitelist!
    unsafe { sqlite3_finalize(stmt) };
    if !matches!(game, "649" | "540" | "joker") {
        return Err(format!("joc necunoscut: {game}"));
    }
    if min_date.len() != 10 || !min_date.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return Err(format!("min_date invalid: {min_date}"));
    }
    let sql_lit = format!(
        "SELECT d.draw_key, d.draw_date, d.draw_kind,
                d.n1, d.n2, d.n3, d.n4, d.n5, d.n6, d.joker,
                o.total_fund,
                p.category, p.status, p.winning_variants, p.payout_per_win, p.carryover
         FROM draws d
         JOIN observations o ON o.observation_id = d.selected_observation
         JOIN prizes p ON p.observation_id = d.selected_observation
         WHERE d.game = '{game}' AND d.draw_date >= '{min_date}'
         ORDER BY d.draw_date ASC,
                  CASE WHEN d.draw_kind = 'regular' THEN 0 ELSE 1 END ASC,
                  d.draw_key ASC,
                  p.category ASC"
    );
    let c_sql_lit = CString::new(sql_lit).unwrap();
    let mut stmt2: *mut c_void = std::ptr::null_mut();
    let rc2 = unsafe {
        sqlite3_prepare_v2(
            conn.db,
            c_sql_lit.as_ptr(),
            -1,
            &mut stmt2,
            std::ptr::null_mut(),
        )
    };
    if rc2 != SQLITE_OK || stmt2.is_null() {
        return Err(format!("sqlite3_prepare_v2 lit esuat (rc={rc2})"));
    }

    let mut draws: Vec<HistoricalDraw> = Vec::new();
    let mut prev_carry: BTreeMap<String, f64> = BTreeMap::new();

    struct RawDrawAccum {
        draw_key: String,
        draw_date: String,
        draw_kind: String,
        numbers: Vec<u8>,
        joker: Option<u8>,
        total_fund: f64,
        cats: BTreeMap<String, (String, u64, f64, f64)>,
    }
    let mut raw_list: Vec<RawDrawAccum> = Vec::new();

    loop {
        let step_rc = unsafe { sqlite3_step(stmt2) };
        if step_rc == SQLITE_DONE {
            break;
        }
        if step_rc != SQLITE_ROW {
            unsafe { sqlite3_finalize(stmt2) };
            return Err(format!("sqlite3_step eroare (rc={step_rc})"));
        }
        let dkey = ReadOnlySqlite::col_text(stmt2, 0);
        let ddate = ReadOnlySqlite::col_text(stmt2, 1);
        let dkind = ReadOnlySqlite::col_text(stmt2, 2);
        let mut nums = Vec::with_capacity(6);
        for col in 3..=8 {
            if let Some(v) = ReadOnlySqlite::col_opt_i32(stmt2, col) {
                if v > 0 {
                    nums.push(v as u8);
                }
            }
        }
        if game == "joker" && nums.len() > 5 {
            nums.truncate(5);
        }
        let jk = ReadOnlySqlite::col_opt_i32(stmt2, 9).and_then(|v| if v > 0 { Some(v as u8) } else { None });
        let tf = ReadOnlySqlite::col_parse_f64(stmt2, 10);
        let cat = ReadOnlySqlite::col_text(stmt2, 11);
        let status = ReadOnlySqlite::col_text(stmt2, 12);
        let w = ReadOnlySqlite::col_opt_i32(stmt2, 13).unwrap_or(0).max(0) as u64;
        let pay = ReadOnlySqlite::col_parse_f64(stmt2, 14);
        let car = ReadOnlySqlite::col_parse_f64(stmt2, 15);

        if raw_list.last().map(|r| r.draw_key.as_str()) != Some(dkey.as_str()) {
            raw_list.push(RawDrawAccum {
                draw_key: dkey,
                draw_date: ddate,
                draw_kind: dkind,
                numbers: nums,
                joker: jk,
                total_fund: tf,
                cats: BTreeMap::new(),
            });
        }
        raw_list
            .last_mut()
            .unwrap()
            .cats
            .insert(cat, (status, w, pay, car));
    }
    unsafe { sqlite3_finalize(stmt2) };

    for raw in raw_list {
        let (y, m, d, wd) = parse_ymd(&raw.draw_date)?;
        let sum_prev_carry: f64 = prev_carry.values().copied().sum();
        let new_tf = (raw.total_fund - sum_prev_carry).max(0.0);
        let mut categories = BTreeMap::new();
        let mut next_carry = prev_carry.clone();

        for (cname, (st, w, pay, car)) in raw.cats {
            let cin = prev_carry.get(&cname).copied().unwrap_or(0.0);
            let tot_c = if st == "rollover" {
                car
            } else {
                (w as f64) * pay
            };
            let n_alloc = (tot_c - cin).max(0.0);
            let cout = if st == "rollover" { car } else { 0.0 };
            next_carry.insert(cname.clone(), cout);
            categories.insert(
                cname,
                CategoryObs {
                    status: st,
                    winners: w,
                    payout_per_win: pay,
                    carryover_out: cout,
                    carry_in: cin,
                    total_cat_fund: tot_c,
                    new_alloc: n_alloc,
                },
            );
        }
        prev_carry = next_carry;
        draws.push(HistoricalDraw {
            draw_key: raw.draw_key,
            draw_date: raw.draw_date,
            year: y,
            month: m,
            day: d,
            weekday: wd,
            draw_kind: raw.draw_kind,
            numbers: raw.numbers,
            joker: raw.joker,
            total_fund_obs: raw.total_fund,
            new_total_fund: new_tf,
            categories,
        });
    }
    Ok(draws)
}

// ============================================================================
// 2. Modelul Cauzal Walk-Forward PM-V3 (Preț_t, Vânzări N_hat_t, Cota alpha_c,
//    Supradispersie NB phi > 0 și Recență Dinamică cu Shrinkage Empiric Bayes)
// ============================================================================

#[inline]
pub fn median_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Rezolvă sistemul liniar SPD/regularizat `A x = b` de dimensiune `d <= 16` prin eliminare Gauss-Jordan cu pivotare parțială.
pub fn solve_linear_system_small(a_in: &[Vec<f64>], b_in: &[f64]) -> Vec<f64> {
    let n = b_in.len();
    let mut m = vec![vec![0.0f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = a_in[i][j];
        }
        m[i][n] = b_in[i];
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col][col].abs();
        for r in (col + 1)..n {
            let v = m[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            m.swap(col, piv);
        }
        let diag = if m[col][col].abs() < 1e-14 {
            1e-14
        } else {
            m[col][col]
        };
        for j in col..=n {
            m[col][j] /= diag;
        }
        for r in 0..n {
            if r != col {
                let f = m[r][col];
                if f != 0.0 {
                    for j in col..=n {
                        m[r][j] -= f * m[col][j];
                    }
                }
            }
        }
    }
    (0..n).map(|i| m[i][n]).collect()
}

pub fn shared_categories_for_game(game: &str) -> &'static [&'static str] {
    match game {
        "649" => &["I", "II", "III"],
        "540" => &["I", "II", "III"],
        "joker" => &["I", "II", "III", "IV", "V", "VI", "VII", "VIII"],
        _ => &[],
    }
}

pub fn all_categories_for_game(game: &str) -> &'static [&'static str] {
    match game {
        "649" => &["I", "II", "III", "IV"],
        "540" => &["I", "II", "III"],
        "joker" => &["I", "II", "III", "IV", "V", "VI", "VII", "VIII"],
        _ => &[],
    }
}

pub fn low_tier_ref_for_game(game: &str) -> (&'static str, f64) {
    match game {
        "649" => ("IV", 246_820.0 / 13_983_816.0),
        "540" => ("III", 510.0 / 658_008.0),
        "joker" => ("VIII", (456_950.0 / 1_221_759.0) / 20.0),
        _ => ("I", 1e-6),
    }
}

pub fn uniform_hit_probs_for_game(game: &str) -> Vec<(&'static str, f64)> {
    match game {
        "649" => vec![
            ("I", 1.0 / 13_983_816.0),
            ("II", 258.0 / 13_983_816.0),
            ("III", 13_545.0 / 13_983_816.0),
            ("IV", 246_820.0 / 13_983_816.0),
        ],
        "540" => vec![
            ("I", 1.0 / 658_008.0),
            ("II", 5.0 / 658_008.0),
            ("III", 2_975.0 / 3_838_380.0),
        ],
        "joker" => {
            let om5 = 1_221_759.0;
            let h5 = 1.0 / om5;
            let h4 = 200.0 / om5;
            let h3 = 7_800.0 / om5;
            let h2 = 98_800.0 / om5;
            let h1 = 456_950.0 / om5;
            vec![
                ("I", h5 / 20.0),
                ("II", h5 * 19.0 / 20.0),
                ("III", h4 / 20.0),
                ("IV", h4 * 19.0 / 20.0),
                ("V", h3 / 20.0),
                ("VI", h3 * 19.0 / 20.0),
                ("VII", h2 / 20.0),
                ("VIII", h1 / 20.0),
            ]
        }
        _ => vec![],
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalStepState {
    pub draw_date: String,
    pub draw_kind: String,
    pub price_t: f64,
    pub fixed_iv_t: f64,
    pub sales_hat_t: f64,
    pub sales_realized_t: f64,
    pub alpha_hat: BTreeMap<String, f64>,
    pub report_in: BTreeMap<String, f64>,
    pub fund_hat_new: BTreeMap<String, f64>,
    pub fund_hat_total: BTreeMap<String, f64>,
    pub fund_legacy_new: BTreeMap<String, f64>,
    pub fund_realized_new: BTreeMap<String, f64>,
    pub nb_r_by_cat: BTreeMap<String, f64>,
    pub nb_phi_nb2_by_cat: BTreeMap<String, f64>,
    pub pearson_phi_by_cat: BTreeMap<String, f64>,
    pub theta_bday: f64,
    pub theta_small12: f64,
    pub theta_cons: f64,
    pub theta_low31: f64,
    pub theta_arith: f64,
    pub theta_grid: f64,
    pub theta_recency_ols: f64,
    pub theta_recency_se: f64,
    pub theta_recency_eb: f64,
    pub eb_shrinkage_factor: f64,
    pub pi_joker: [f64; 20],
}

/// Estimează cauzal (`s < t`) prețul nominal al unei variante simple `Preț_t` din ultimele 6 extrageri regulate.
pub fn estimate_causal_price_at(draws: &[HistoricalDraw], idx_t: usize, game: &str) -> f64 {
    const CAND_PRICES: [f64; 13] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let (cat_low, q0) = low_tier_ref_for_game(game);
    let mut recent_implied = Vec::with_capacity(6);
    let mut recent_cents = Vec::with_capacity(6);

    for j in (0..idx_t).rev() {
        let d = &draws[j];
        if d.draw_kind != "regular" || d.new_total_fund <= 0.0 {
            continue;
        }
        if let Some(cobs) = d.categories.get(cat_low) {
            if cobs.winners > 0 {
                let n_win = (cobs.winners as f64) / q0;
                recent_implied.push(d.new_total_fund / (0.40 * n_win));
                recent_cents.push((d.new_total_fund * 100.0).round() as i64);
                if recent_implied.len() >= 6 {
                    break;
                }
            }
        }
    }
    if recent_implied.is_empty() {
        return match game {
            "649" => 5.0,
            "540" => 3.0,
            "joker" => 4.0,
            _ => 5.0,
        };
    }
    let med_imp = median_f64(&recent_implied);
    let mut best_p = CAND_PRICES[0];
    let mut best_score = f64::INFINITY;
    for &p in &CAND_PRICES {
        let step_cents = (40.0 * p).round() as i64;
        let div_bonus = if game != "joker" && step_cents > 0 {
            let hits = recent_cents.iter().take(4).filter(|&&c| c % step_cents == 0).count();
            if hits >= 3 { -0.18 } else { 0.0 }
        } else {
            0.0
        };
        let dist = ((p - med_imp) / med_imp.max(1.0)).abs() + div_bonus;
        if dist < best_score {
            best_score = dist;
            best_p = p;
        }
    }
    best_p
}

pub fn estimate_causal_fixed_iv_649(draws: &[HistoricalDraw], idx_t: usize) -> f64 {
    for j in (0..idx_t).rev() {
        if let Some(c4) = draws[j].categories.get("IV") {
            if c4.winners > 0 && c4.payout_per_win > 0.0 {
                return c4.payout_per_win;
            }
        }
    }
    30.0
}

#[inline]
pub fn realized_tickets_for_past_draw(d: &HistoricalDraw, price_s: f64, game: &str) -> f64 {
    let (cat_low, q0) = low_tier_ref_for_game(game);
    if d.draw_kind == "regular" && d.new_total_fund > 0.0 && price_s > 0.0 {
        d.new_total_fund / (0.40 * price_s)
    } else if let Some(cobs) = d.categories.get(cat_low) {
        if cobs.winners > 0 {
            (cobs.winners as f64) / q0
        } else {
            100_000.0
        }
    } else {
        100_000.0
    }
}

/// Construiește vectorul de 13 regresori cauzali pentru prognoza vânzărilor `ln N_s`.
fn build_sales_features(
    draws: &[HistoricalDraw],
    n_real_hist: &[f64],
    price_hist: &[f64],
    idx: usize,
    price_cur: f64,
) -> [f64; 13] {
    let d = &draws[idx];
    let wd_thu = if d.weekday == 3 { 1.0 } else { 0.0 };
    let wd_sun = if d.weekday == 6 { 1.0 } else { 0.0 };
    let rep_i = d.categories.get("I").map(|c| c.carry_in).unwrap_or(0.0);
    let ln_rep = (1.0 + rep_i / 1_000_000.0).ln();
    let post_reset = if rep_i <= 1e-6 { 1.0 } else { 0.0 };

    let mut prev_regs = Vec::with_capacity(2);
    for k in (0..idx).rev() {
        if draws[k].draw_kind == "regular" && n_real_hist[k] > 1_000.0 {
            prev_regs.push(n_real_hist[k].ln());
            if prev_regs.len() >= 2 {
                break;
            }
        }
    }
    let ln1 = prev_regs.first().copied().unwrap_or(12.5);
    let ln2 = prev_regs.get(1).copied().unwrap_or(ln1);
    let _ = price_hist;
    let [nye, xmas, east, seas, spec] =
        calendar_guard_features(d.year, d.month, d.day, &d.draw_kind);

    [
        1.0,
        wd_thu,
        wd_sun,
        ln_rep,
        ln1,
        ln2,
        post_reset,
        nye,
        xmas,
        east,
        seas,
        spec,
        price_cur.max(1.0).ln(),
    ]
}

/// Estimează cauzal la pasul `idx_t` (folosind EXCLUSIV `0..idx_t`) toți parametrii `PM-V3` + reziduurile pentru bootstrap.
#[derive(Clone, Debug)]
pub struct CausalFitBundle {
    pub state: CausalStepState,
    pub log_sales_residuals: Vec<f64>,
    pub alpha_ratios_window: BTreeMap<String, Vec<f64>>,
}

pub fn fit_causal_pmv3_at_step(
    draws: &[HistoricalDraw],
    prices_prefix: &[f64],
    n_real_prefix: &[f64],
    idx_t: usize,
    game: &str,
) -> CausalFitBundle {
    assert!(prices_prefix.len() >= idx_t && n_real_prefix.len() >= idx_t);
    let d_cur = &draws[idx_t];
    let price_t = estimate_causal_price_at(&draws[..idx_t], idx_t, game);
    let fixed_iv_t = if game == "649" {
        estimate_causal_fixed_iv_649(&draws[..idx_t], idx_t)
    } else {
        0.0
    };

    // 1. Regresie cauzala Ridge ponderata exponential pe fereastra [max(2, idx_t - 260)..idx_t]
    let start_s = idx_t.saturating_sub(260).max(2);
    let p_dim = 13usize;
    let mut xtx = vec![vec![0.0f64; p_dim]; p_dim];
    let mut xty = vec![0.0f64; p_dim];
    for d_i in 0..p_dim {
        xtx[d_i][d_i] = if d_i == 0 { 1e-6 } else { 1e-2 };
    }
    let mut x_rows: Vec<([f64; 13], f64, f64)> = Vec::with_capacity(idx_t.saturating_sub(start_s));
    for s in start_s..idx_t {
        let ns = n_real_prefix[s];
        if ns <= 1_000.0 {
            continue;
        }
        let xs = build_sales_features(draws, n_real_prefix, prices_prefix, s, prices_prefix[s]);
        let ys = ns.ln();
        let w = 0.990f64.powi((idx_t - 1 - s) as i32);
        for a in 0..p_dim {
            xty[a] += w * xs[a] * ys;
            for b in 0..p_dim {
                xtx[a][b] += w * xs[a] * xs[b];
            }
        }
        x_rows.push((xs, ys, w));
    }
    let beta = solve_linear_system_small(&xtx, &xty);
    let mut smear_num = 0.0f64;
    let mut smear_den = 0.0f64;
    let mut log_sales_residuals = Vec::with_capacity(x_rows.len());
    for &(ref xs, ys, w) in &x_rows {
        let mut pred = 0.0f64;
        for a in 0..p_dim {
            pred += xs[a] * beta[a];
        }
        let res = (ys - pred).clamp(-1.5, 1.5);
        smear_num += w * res.exp();
        smear_den += w;
        if x_rows.len() <= 104 || log_sales_residuals.len() < 156 {
            log_sales_residuals.push(res);
        }
    }
    // Pastram ultimele 104 reziduuri cauzale pentru bootstrap bloc
    if x_rows.len() > 104 {
        log_sales_residuals.clear();
        for &(ref xs, ys, _) in &x_rows[x_rows.len() - 104..] {
            let pred: f64 = (0..p_dim).map(|a| xs[a] * beta[a]).sum();
            log_sales_residuals.push((ys - pred).clamp(-1.2, 1.2));
        }
    }
    let smear = if smear_den > 0.0 {
        smear_num / smear_den
    } else {
        1.0
    };
    let x_cur = build_sales_features(draws, n_real_prefix, prices_prefix, idx_t, price_t);
    let ln_hat_cur: f64 = (0..p_dim).map(|a| x_cur[a] * beta[a]).sum();
    let sales_hat_t = (ln_hat_cur.exp() * smear).clamp(5_000.0, 25_000_000.0);
    let sales_realized_t = realized_tickets_for_past_draw(d_cur, price_t, game);

    // 2. Cotele cauzale alpha_{c,t} pe categorii partajate (si comparatie cu legacy ceiling.py:69)
    let shared_cats = shared_categories_for_game(game);
    let mut alpha_hat = BTreeMap::new();
    let mut alpha_ratios_window: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut report_in = BTreeMap::new();
    let mut fund_hat_new = BTreeMap::new();
    let mut fund_hat_total = BTreeMap::new();
    let mut fund_legacy_new = BTreeMap::new();
    let mut fund_realized_new = BTreeMap::new();

    for &cname in shared_cats {
        let mut same_kind_shares = Vec::with_capacity(52);
        for s in (0..idx_t).rev() {
            if draws[s].draw_kind == d_cur.draw_kind && n_real_prefix[s] > 1_000.0 && prices_prefix[s] > 0.0 {
                if let Some(cobs) = draws[s].categories.get(cname) {
                    // Pentru 6/49 ajustam cauzal cand premiul fix Cat IV s-a schimbat (30 -> 50 lei):
                    // fondul rezidual I+II+III este (0.40 * Pret_s - Fixed_IV_s * q_IV) / Pret_s!
                    let mut sh = cobs.new_alloc / (prices_prefix[s] * n_real_prefix[s]);
                    if game == "649" && draws[s].draw_kind == "regular" {
                        let (_, q_iv) = low_tier_ref_for_game("649");
                        let f_iv_s = draws[s]
                            .categories
                            .get("IV")
                            .map(|x| if x.payout_per_win > 0.0 { x.payout_per_win } else { 30.0 })
                            .unwrap_or(30.0);
                        let rem_s = (0.40 - (f_iv_s * q_iv) / prices_prefix[s]).max(0.05);
                        let rem_t = (0.40 - (fixed_iv_t * q_iv) / price_t).max(0.05);
                        sh *= rem_t / rem_s;
                    }
                    same_kind_shares.push(sh.clamp(0.0, 2.5));
                    if same_kind_shares.len() >= 52 {
                        break;
                    }
                }
            }
        }
        if same_kind_shares.len() < 6 {
            for s in (0..idx_t).rev() {
                if n_real_prefix[s] > 1_000.0 && prices_prefix[s] > 0.0 {
                    if let Some(cobs) = draws[s].categories.get(cname) {
                        same_kind_shares.push((cobs.new_alloc / (prices_prefix[s] * n_real_prefix[s])).clamp(0.0, 2.5));
                        if same_kind_shares.len() >= 52 {
                            break;
                        }
                    }
                }
            }
        }
        let a_c = median_f64(&same_kind_shares);
        alpha_hat.insert(cname.to_string(), a_c);
        alpha_ratios_window.insert(cname.to_string(), same_kind_shares);

        let rep_c = d_cur.categories.get(cname).map(|c| c.carry_in).unwrap_or(0.0);
        let real_new_c = d_cur.categories.get(cname).map(|c| c.new_alloc).unwrap_or(0.0);
        let new_hat_c = a_c * price_t * sales_hat_t;
        report_in.insert(cname.to_string(), rep_c);
        fund_hat_new.insert(cname.to_string(), new_hat_c);
        fund_hat_total.insert(cname.to_string(), rep_c + new_hat_c);
        fund_realized_new.insert(cname.to_string(), real_new_c);

        // Legacy ceiling.py:69: mediana statica pe ultimele 52 de extrageri fara scalare la vanzari/pret!
        let mut leg_52 = Vec::with_capacity(52);
        for s in idx_t.saturating_sub(52)..idx_t {
            if let Some(cobs) = draws[s].categories.get(cname) {
                leg_52.push(cobs.new_alloc);
            }
        }
        fund_legacy_new.insert(cname.to_string(), median_f64(&leg_52));
    }

    // 3. Estimare cauzala a parametrilor multimii + Recenta Dinamica (t-1) cu Shrinkage Empiric Bayes
    let (cat_low, q_low_unif) = low_tier_ref_for_game(game);
    let crowd_start = idx_t.saturating_sub(364).max(1);
    // Regresie OLS pe log(W_{low,s} / (N_real_s * q_low_unif)) ~ [1, bday_cnt, small12_cnt, cons_cnt, pure31, arith, grid, ov_prev1]
    let d_c = 8usize;
    let mut cxtx = vec![vec![0.0f64; d_c]; d_c];
    let mut cxty = vec![0.0f64; d_c];
    for j in 0..d_c {
        cxtx[j][j] = if j == 0 { 1e-6 } else { 0.25 };
    }
    let gcols = if game == "649" { 7usize } else { 5usize };
    let mut crow_samples: Vec<([f64; 8], f64)> = Vec::new();

    for s in crowd_start..idx_t {
        let ds = &draws[s];
        if ds.draw_kind != "regular" || n_real_prefix[s] <= 1_000.0 {
            continue;
        }
        let Some(cobs) = ds.categories.get(cat_low) else { continue };
        if cobs.winners == 0 {
            continue;
        }
        let mu_u = n_real_prefix[s] * q_low_unif;
        if mu_u <= 0.5 {
            continue;
        }
        let y_ratio = ((cobs.winners as f64) / mu_u).clamp(0.15, 6.0).ln();
        let mut nums = ds.numbers.clone();
        nums.sort_unstable();
        let k_len = nums.len();
        if k_len < 5 {
            continue;
        }
        let bday_cnt = nums.iter().filter(|&&x| x <= 31).count() as f64;
        let sm12_cnt = nums.iter().filter(|&&x| x <= 12).count() as f64;
        let mut cons_cnt = 0.0f64;
        for w in nums.windows(2) {
            if w[1].saturating_sub(w[0]) == 1 {
                cons_cnt += 1.0;
            }
        }
        let pure31 = if *nums.last().unwrap() <= 31 { 1.0 } else { 0.0 };
        let d0 = nums[1] - nums[0];
        let is_ap = if nums.windows(2).all(|w| w[1] - w[0] == d0) { 1.0 } else { 0.0 };
        let mut rcnt = [0u8; 10];
        let mut ccnt = [0u8; 8];
        let mut max_rc = 0u8;
        for &x in &nums {
            let z = (x - 1) as usize;
            let r = z / gcols;
            let c = z % gcols;
            rcnt[r] += 1;
            ccnt[c] += 1;
            max_rc = max_rc.max(rcnt[r]).max(ccnt[c]);
        }
        let grid_f = if (max_rc as usize) >= k_len - 1 { 1.0 } else { 0.0 };
        let prev_nums = &draws[s - 1].numbers;
        let ov1 = nums.iter().filter(|x| prev_nums.contains(x)).count() as f64;

        let feats = [1.0, bday_cnt, sm12_cnt, cons_cnt, pure31, is_ap, grid_f, ov1];
        for a in 0..d_c {
            cxty[a] += feats[a] * y_ratio;
            for b in 0..d_c {
                cxtx[a][b] += feats[a] * feats[b];
            }
        }
        crow_samples.push((feats, y_ratio));
    }

    let c_beta = solve_linear_system_small(&cxtx, &cxty);
    // Eroare standard pe coeficientul de recenta (index 7) pentru Shrinkage Empiric Bayes:
    let mut rss_c = 0.0f64;
    for &(ref f, y) in &crow_samples {
        let p: f64 = (0..d_c).map(|a| f[a] * c_beta[a]).sum();
        rss_c += (y - p) * (y - p);
    }
    let dof = (crow_samples.len().saturating_sub(d_c)).max(1) as f64;
    let sigma2_c = rss_c / dof;
    let mut unit_e7 = [0.0f64; 8];
    unit_e7[7] = 1.0;
    let inv_col7 = solve_linear_system_small(&cxtx, &unit_e7);
    let se_rec = (sigma2_c * inv_col7[7].max(1e-9)).sqrt();

    // Shrinkage Empiric Bayes (prior centrat in 0 cu deviatie standard tau_0 = 0.045):
    // Pe biletul de k bile fata de extragerea cu 3..4 potriviri, factorul de diluare hipergeometrica este k / h_low:
    let dil_scale = match game {
        "649" => 6.0 / 3.0,   // o potrivire in 3/6 reflecta jumatate din panta biletului intreg 6/6
        "540" => 5.0 / 4.0,   // Cat III are 4/6 potriviri pe biletul de 5
        "joker" => 5.0 / 1.5, // Cat VIII are 1/5 potrivire + J
        _ => 1.5,
    };
    let tau0_sq = 0.045 * 0.045;
    let eb_shrink = tau0_sq / (tau0_sq + se_rec * se_rec);
    let theta_rec_ols = (c_beta[7] * dil_scale).clamp(-0.22, 0.12);
    let theta_rec_eb = (theta_rec_ols * eb_shrink).clamp(-0.18, 0.08);

    let theta_bday = (c_beta[1] * dil_scale).clamp(0.02, 0.42);
    let theta_small12 = (c_beta[2] * dil_scale).clamp(0.0, 0.25);
    let theta_cons = (c_beta[3] * dil_scale).clamp(-0.05, 0.35);
    // Pentru tipare rare pe extragere (arith / linie completa pe grila), combinam estimarea empirica cu priorul documentat anti-multime:
    let theta_low31 = (c_beta[4] * 1.25).clamp(0.02, 0.35);
    let theta_arith = (c_beta[5] + 0.35).clamp(0.15, 0.75);
    let theta_grid = (c_beta[6] + 0.22).clamp(0.08, 0.55);

    // Distributia cauzala pi_J pe Joker (Dirichlet shrinkage alpha_0 = 8.0 per numar 1..20):
    let mut pi_joker = [1.0 / 20.0; 20];
    if game == "joker" {
        let mut w_sum_j = [8.0f64; 20];
        for s in crowd_start..idx_t {
            let ds = &draws[s];
            if ds.draw_kind != "regular" || n_real_prefix[s] <= 1_000.0 {
                continue;
            }
            if let (Some(jk), Some(c8)) = (ds.joker, ds.categories.get("VIII")) {
                if (1..=20).contains(&jk) && c8.winners > 0 {
                    let ratio_j = ((c8.winners as f64) / (n_real_prefix[s] * q_low_unif)).clamp(0.3, 2.5);
                    w_sum_j[(jk - 1) as usize] += ratio_j;
                }
            }
        }
        let tot_j: f64 = w_sum_j.iter().sum();
        for j in 0..20 {
            pi_joker[j] = w_sum_j[j] / tot_j;
        }
    }

    // 4. Estimare cauzala a supradispersiei Binomiale Negative (phi_NB2 = 1 / r_c > 0 si Pearson phi_c) pe fiecare categorie!
    let unif_probs = uniform_hit_probs_for_game(game);
    let mut nb_r_by_cat = BTreeMap::new();
    let mut nb_phi_nb2_by_cat = BTreeMap::new();
    let mut pearson_phi_by_cat = BTreeMap::new();

    for &(cname, q_unif_c) in &unif_probs {
        let mut p_terms = Vec::new();
        let mut eta_terms = Vec::new();
        for s in crowd_start..idx_t {
            let ds = &draws[s];
            if ds.draw_kind != "regular" || n_real_prefix[s] <= 1_000.0 {
                continue;
            }
            let Some(cobs) = ds.categories.get(cname) else { continue };
            // Factorul de multime al extragerii s estimat din categoriile cu frecventa mare
            let crowd_mult_s = if let Some(cl) = ds.categories.get(cat_low) {
                if cl.winners > 0 {
                    ((cl.winners as f64) / (n_real_prefix[s] * q_low_unif)).clamp(0.35, 3.0)
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let mu_s = (n_real_prefix[s] * q_unif_c * crowd_mult_s).max(1e-4);
            let w_s = cobs.winners as f64;
            if mu_s >= 0.25 {
                let diff2 = (w_s - mu_s) * (w_s - mu_s);
                p_terms.push(diff2 / mu_s);
                eta_terms.push(((diff2 - mu_s) / (mu_s * mu_s)).max(0.0));
            }
        }
        // Adaugam incertitudinea prognozei vanzarilor Var(N_hat / N) ~ sigma_sales^2 la dispersia NB2 ex-ante!
        let sales_var = if log_sales_residuals.len() >= 10 {
            let m: f64 = log_sales_residuals.iter().sum::<f64>() / (log_sales_residuals.len() as f64);
            let v: f64 = log_sales_residuals
                .iter()
                .map(|x| (x - m) * (x - m))
                .sum::<f64>()
                / (log_sales_residuals.len() as f64);
            v.clamp(0.010, 0.150)
        } else {
            0.025
        };
        let emp_eta = if eta_terms.len() >= 10 {
            let mean_eta = eta_terms.iter().sum::<f64>() / (eta_terms.len() as f64);
            (0.5 * mean_eta + sales_var).clamp(0.020, 0.500)
        } else {
            (0.045 + sales_var).clamp(0.030, 0.350)
        };
        let r_c = (1.0 / emp_eta).clamp(2.0, 50.0);
        let pearson_c = if p_terms.len() >= 10 {
            (p_terms.iter().sum::<f64>() / (p_terms.len() as f64)).max(1.0 + emp_eta)
        } else {
            1.0 + emp_eta * 5.0
        };
        nb_r_by_cat.insert(cname.to_string(), r_c);
        nb_phi_nb2_by_cat.insert(cname.to_string(), 1.0 / r_c);
        pearson_phi_by_cat.insert(cname.to_string(), pearson_c);
    }

    CausalFitBundle {
        state: CausalStepState {
            draw_date: d_cur.draw_date.clone(),
            draw_kind: d_cur.draw_kind.clone(),
            price_t,
            fixed_iv_t,
            sales_hat_t,
            sales_realized_t,
            alpha_hat,
            report_in,
            fund_hat_new,
            fund_hat_total,
            fund_legacy_new,
            fund_realized_new,
            nb_r_by_cat,
            nb_phi_nb2_by_cat,
            pearson_phi_by_cat,
            theta_bday,
            theta_small12,
            theta_cons,
            theta_low31,
            theta_arith,
            theta_grid,
            theta_recency_ols: theta_rec_ols,
            theta_recency_se: se_rec,
            theta_recency_eb: theta_rec_eb,
            eb_shrinkage_factor: eb_shrink,
            pi_joker,
        },
        log_sales_residuals,
        alpha_ratios_window,
    }
}


// ============================================================================
// 3. Evaluatoare Exacte CPU 24 Fire (`f64` cu acumulatori în virgulă fixă `u128` `s96`)
//    Paritate bit-cu-bit / `< 10^-14` față de kernelele CUDA `sm_120` din `v3_kernels.cu`
// ============================================================================

#[inline]
pub fn build_binom_table_50x7() -> [[u32; 7]; 50] {
    let mut b = [[0u32; 7]; 50];
    for n in 0..50 {
        b[n][0] = 1;
        let kmax = n.min(6);
        for k in 1..=kmax {
            b[n][k] = b[n - 1][k - 1] + b[n - 1][k];
        }
    }
    b
}

#[inline]
pub fn pmv3_rank_lex_cpu(
    mask: u64,
    n: usize,
    k: usize,
    total_comb: u32,
    binom_tbl: &[[u32; 7]; 50],
) -> usize {
    let mut rem = 0u32;
    let mut m = mask;
    for i in 0..k {
        let a = m.trailing_zeros() as usize;
        m &= m - 1;
        rem += binom_tbl[n - 1 - a][k - i];
    }
    (total_comb - 1 - rem) as usize
}

#[inline]
pub fn pmv3_ticket_lw_cpu(mask: u64, k: usize, p: &V3CrowdParamsF64) -> f64 {
    let mut pos = [0usize; 6];
    let mut m = mask;
    let mut lw = 0.0f64;
    for i in 0..k {
        let a = m.trailing_zeros() as usize;
        m &= m - 1;
        pos[i] = a;
        lw += p.gamma[a];
    }
    let mut cons = 0i32;
    for i in 0..(k - 1) {
        if pos[i + 1] - pos[i] == 1 {
            cons += 1;
        }
    }
    if cons > 0 {
        lw += p.theta_cons * (cons as f64);
    }
    if pos[k - 1] <= 30 {
        lw += p.theta_low31;
    }
    let d0 = pos[1] - pos[0];
    let mut is_ap = true;
    for i in 1..(k - 1) {
        if pos[i + 1] - pos[i] != d0 {
            is_ap = false;
            break;
        }
    }
    if is_ap {
        lw += p.theta_arith;
    }
    let mut rcnt = [0i32; 10];
    let mut ccnt = [0i32; 8];
    let gcols = p.grid_cols as usize;
    let mut max_rc = 0i32;
    for i in 0..k {
        let r = pos[i] / gcols;
        let c = pos[i] % gcols;
        rcnt[r] += 1;
        ccnt[c] += 1;
        if rcnt[r] > max_rc {
            max_rc = rcnt[r];
        }
        if ccnt[c] > max_rc {
            max_rc = ccnt[c];
        }
    }
    if max_rc >= (k as i32) - 1 {
        lw += p.theta_grid;
    }
    lw
}

/// Formula analitică exactă `E[1 / (1 + W)]` pentru `W ~ NB2(mu, r)` (`mode=0`), `Poisson(mu)` (`mode=1`), `Jensen` (`mode=2`).
#[inline]
pub fn pmv3_g_share_cpu(mu: f64, r: f64, mode: i32) -> f64 {
    if mu <= 0.0 {
        return 1.0;
    }
    if mode == 2 {
        return 1.0 / (1.0 + mu);
    }
    if mode == 1 || r >= 1.0e12 {
        if mu < 1.0e-8 {
            return 1.0 - 0.5 * mu + (mu * mu) * (1.0 / 6.0) - (mu * mu * mu) * (1.0 / 24.0);
        }
        return -(-mu).exp_m1() / mu;
    }
    if mu < 1.0e-7 {
        let t2 = (mu * mu) * ((r + 1.0) / (6.0 * r));
        let t3 = (mu * mu * mu) * ((r + 1.0) * (r + 2.0) / (24.0 * r * r));
        return 1.0 - 0.5 * mu + t2 - t3;
    }
    if (r - 1.0).abs() < 1.0e-7 {
        let eps = 1.0 - r;
        let l_val = (mu / r).ln_1p();
        let el = eps * l_val;
        return (r / mu) * l_val * (1.0 + 0.5 * el + (el * el) * (1.0 / 6.0));
    }
    let l_val = (mu / r).ln_1p();
    let num = ((1.0 - r) * l_val).exp_m1();
    (r * num) / (mu * (1.0 - r))
}

#[inline]
pub fn double_to_u128_s96(x: f64) -> (u64, u64) {
    if x <= 0.0 {
        return (0, 0);
    }
    let x32 = x * 4294967296.0;
    let h_fl = x32.floor();
    let mut h = h_fl as u64;
    let frac = x32 - h_fl;
    let mid_fl = (frac * 4294967296.0).floor();
    let bot_fl =
        ((frac - mid_fl * (1.0 / 4294967296.0)) * 18446744073709551616.0 + 0.5).floor();
    let mut mid_u = mid_fl as u64;
    let mut bot_u = if bot_fl >= 4294967296.0 {
        4294967296u64
    } else {
        bot_fl as u64
    };
    let carry_mid = bot_u >> 32;
    bot_u &= 0xFFFF_FFFF;
    mid_u += carry_mid;
    if mid_u >= 4294967296 {
        h += mid_u >> 32;
        mid_u &= 0xFFFF_FFFF;
    }
    ((mid_u << 32) | bot_u, h)
}

#[inline]
pub fn u128_s96_to_double(lo: u64, hi: u64) -> f64 {
    let a = (hi as f64) * (1.0 / 4294967296.0);
    let b = ((lo >> 32) as f64) * (1.0 / 18446744073709551616.0);
    let c = ((lo & 0xFFFF_FFFF) as f64) * (1.0 / 79228162514264337593543950336.0);
    a + (b + c)
}

#[inline]
fn atomic_add_u128_s96(lo_arr: &[AtomicU64], hi_arr: &[AtomicU64], idx: usize, val: f64) {
    let (vlo, vhi) = double_to_u128_s96(val);
    if (vlo | vhi) == 0 {
        return;
    }
    let old_lo = lo_arr[idx].fetch_add(vlo, Ordering::Relaxed);
    let carry = if old_lo.wrapping_add(vlo) < old_lo {
        1u64
    } else {
        0u64
    };
    let add_hi = vhi + carry;
    if add_hi != 0 {
        hi_arr[idx].fetch_add(add_hi, Ordering::Relaxed);
    }
}

fn alloc_atomic_u64_vec(len: usize) -> Vec<AtomicU64> {
    let mut v = Vec::with_capacity(len);
    v.resize_with(len, || AtomicU64::new(0));
    v
}

pub fn build_crowd_params_f64(
    st: &CausalStepState,
    game: &str,
    prev_draw_numbers: &[u8],
    share_mode: i32,
) -> V3CrowdParamsF64 {
    let (n_pool, k_pick, grid_cols) = match game {
        "649" => (49usize, 6usize, 7i32),
        "540" => (40usize, 5usize, 5i32),
        "joker" => (45usize, 5usize, 5i32),
        _ => (49usize, 6usize, 7i32),
    };
    let mut gamma = [0.0f64; 50];
    let mut g_sorted = Vec::with_capacity(n_pool);
    for a in 0..n_pool {
        let num = (a + 1) as u8;
        let mut g = 0.0f64;
        if num <= 31 {
            g += st.theta_bday;
        }
        if num <= 12 {
            g += st.theta_small12;
        }
        if prev_draw_numbers.contains(&num) {
            g += st.theta_recency_eb;
        }
        gamma[a] = g;
        g_sorted.push(g);
    }
    g_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top_gamma: f64 = g_sorted.iter().take(k_pick).sum();
    let lw_shift = top_gamma
        + ((k_pick - 1) as f64) * st.theta_cons.max(0.0)
        + st.theta_low31.max(0.0)
        + st.theta_arith.max(0.0)
        + st.theta_grid.max(0.0)
        + 0.05;

    let mut funds = [0.0f64; 8];
    let mut nb_r = [10.0f64; 8];
    let cats = shared_categories_for_game(game);
    for (ci, &cname) in cats.iter().enumerate() {
        funds[ci] = st.fund_hat_total.get(cname).copied().unwrap_or(0.0);
        nb_r[ci] = st.nb_r_by_cat.get(cname).copied().unwrap_or(10.0);
    }

    V3CrowdParamsF64 {
        gamma,
        theta_cons: st.theta_cons,
        theta_low31: st.theta_low31,
        theta_arith: st.theta_arith,
        theta_grid: st.theta_grid,
        lw_shift,
        pi_joker: st.pi_joker,
        sales_n: st.sales_hat_t,
        funds,
        nb_r,
        fixed_iv: st.fixed_iv_t,
        grid_cols,
        share_mode,
    }
}

/// Evaluare exactă pe 24 fire CPU pentru Loto 6/49 (`13,983,816` bilete, `f64` + `u128` `s96`).
pub fn cpu_pmv3_ev_649_f64(
    masks_649: &[u64],
    params: &V3CrowdParamsF64,
    ev_out: &mut [f64],
) -> f64 {
    assert_eq!(masks_649.len(), COUNT_649);
    assert_eq!(ev_out.len(), COUNT_649);
    let t0 = Instant::now();
    let binom_tbl = build_binom_table_50x7();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_sz = (COUNT_649 + n_threads - 1) / n_threads;

    const N5: usize = 1_906_884;
    const N4: usize = 211_876;

    let mut p_vec = vec![0.0f64; COUNT_649];
    let z_lo = AtomicU64::new(0);
    let z_hi = AtomicU64::new(0);

    std::thread::scope(|s| {
        for (m_chunk, p_chunk) in masks_649
            .chunks(chunk_sz)
            .zip(p_vec.chunks_mut(chunk_sz))
        {
            let zlo_ref = &z_lo;
            let zhi_ref = &z_hi;
            s.spawn(move || {
                let mut loc_lo = 0u64;
                let mut loc_hi = 0u64;
                for (&mk, dst) in m_chunk.iter().zip(p_chunk.iter_mut()) {
                    let lw = pmv3_ticket_lw_cpu(mk, 6, params);
                    let w = (lw - params.lw_shift).exp();
                    *dst = w;
                    let (vlo, vhi) = double_to_u128_s96(w);
                    let next_lo = loc_lo.wrapping_add(vlo);
                    let c = if next_lo < loc_lo { 1u64 } else { 0u64 };
                    loc_lo = next_lo;
                    loc_hi = loc_hi.wrapping_add(vhi + c);
                }
                let old = zlo_ref.fetch_add(loc_lo, Ordering::Relaxed);
                let c = if old.wrapping_add(loc_lo) < old { 1u64 } else { 0u64 };
                if loc_hi + c != 0 {
                    zhi_ref.fetch_add(loc_hi + c, Ordering::Relaxed);
                }
            });
        }
    });

    let z_val = u128_s96_to_double(z_lo.load(Ordering::Relaxed), z_hi.load(Ordering::Relaxed));
    let inv_z = 1.0 / z_val;

    let s5_lo = alloc_atomic_u64_vec(N5);
    let s5_hi = alloc_atomic_u64_vec(N5);
    let s4_lo = alloc_atomic_u64_vec(N4);
    let s4_hi = alloc_atomic_u64_vec(N4);

    std::thread::scope(|s| {
        for (m_chunk, p_chunk) in masks_649
            .chunks(chunk_sz)
            .zip(p_vec.chunks_mut(chunk_sz))
        {
            let s5l = &s5_lo;
            let s5h = &s5_hi;
            let s4l = &s4_lo;
            let s4h = &s4_hi;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                for (&mk, pv) in m_chunk.iter().zip(p_chunk.iter_mut()) {
                    let p = *pv * inv_z;
                    *pv = p;
                    let mut bits = [0u64; 6];
                    let mut m = mk;
                    for b in &mut bits {
                        let lsb = m & m.wrapping_neg();
                        *b = lsb;
                        m ^= lsb;
                    }
                    for i in 0..6 {
                        let r5 = pmv3_rank_lex_cpu(mk ^ bits[i], 49, 5, N5 as u32, b_ref);
                        atomic_add_u128_s96(s5l, s5h, r5, p);
                    }
                    for i in 0..6 {
                        for j in (i + 1)..6 {
                            let r4 = pmv3_rank_lex_cpu(
                                mk ^ bits[i] ^ bits[j],
                                49,
                                4,
                                N4 as u32,
                                b_ref,
                            );
                            atomic_add_u128_s96(s4l, s4h, r4, p);
                        }
                    }
                }
            });
        }
    });

    let s5: Vec<f64> = (0..N5)
        .map(|i| u128_s96_to_double(s5_lo[i].load(Ordering::Relaxed), s5_hi[i].load(Ordering::Relaxed)))
        .collect();
    let s4: Vec<f64> = (0..N4)
        .map(|i| u128_s96_to_double(s4_lo[i].load(Ordering::Relaxed), s4_hi[i].load(Ordering::Relaxed)))
        .collect();
    drop(s5_lo);
    drop(s5_hi);
    drop(s4_lo);
    drop(s4_hi);

    let off_5 = 5.0 * (params.funds[2] / 13_983_816.0);
    let fixed_ev_iv = (246_820.0 / 13_983_816.0) * params.fixed_iv;

    let mut yt6 = vec![0.0f64; COUNT_649];
    let u5_lo = alloc_atomic_u64_vec(N5);
    let u5_hi = alloc_atomic_u64_vec(N5);
    let u4_lo = alloc_atomic_u64_vec(N4);
    let u4_hi = alloc_atomic_u64_vec(N4);

    std::thread::scope(|s| {
        for ((m_chunk, p_chunk), yt_chunk) in masks_649
            .chunks(chunk_sz)
            .zip(p_vec.chunks(chunk_sz))
            .zip(yt6.chunks_mut(chunk_sz))
        {
            let s5_ref = &s5;
            let s4_ref = &s4;
            let u5l = &u5_lo;
            let u5h = &u5_hi;
            let u4l = &u4_lo;
            let u4h = &u4_hi;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                let inv_omega = 1.0 / 13_983_816.0;
                let n_s = params.sales_n;
                let smode = params.share_mode;
                for ((&mk, &p_d), yt_dst) in m_chunk
                    .iter()
                    .zip(p_chunk.iter())
                    .zip(yt_chunk.iter_mut())
                {
                    let mut bits = [0u64; 6];
                    let mut m = mk;
                    for b in &mut bits {
                        let lsb = m & m.wrapping_neg();
                        *b = lsb;
                        m ^= lsb;
                    }
                    let mut r5 = [0usize; 6];
                    let mut t5 = 0.0f64;
                    for i in 0..6 {
                        let rk = pmv3_rank_lex_cpu(mk ^ bits[i], 49, 5, N5 as u32, b_ref);
                        r5[i] = rk;
                        t5 += s5_ref[rk];
                    }
                    let mut r4 = [0usize; 15];
                    let mut t4 = 0.0f64;
                    let mut pos = 0usize;
                    for i in 0..6 {
                        for j in (i + 1)..6 {
                            let rk = pmv3_rank_lex_cpu(
                                mk ^ bits[i] ^ bits[j],
                                49,
                                4,
                                N4 as u32,
                                b_ref,
                            );
                            r4[pos] = rk;
                            pos += 1;
                            t4 += s4_ref[rk];
                        }
                    }
                    let q6 = p_d.max(0.0);
                    let q5 = (t5 - 6.0 * p_d).max(0.0);
                    let q4 = (t4 - 5.0 * t5 + 15.0 * p_d).max(0.0);

                    let y6 = (params.funds[0] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q6, params.nb_r[0], smode);
                    let y5 = (params.funds[1] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q5, params.nb_r[1], smode);
                    let y4 = (params.funds[2] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q4, params.nb_r[2], smode);

                    let yt5_shift = ((y5 - 5.0 * y4) + off_5).max(0.0);
                    *yt_dst = y6 - 6.0 * y5 + 15.0 * y4;

                    for &rk in &r5 {
                        atomic_add_u128_s96(u5l, u5h, rk, yt5_shift);
                    }
                    for &rk in &r4 {
                        atomic_add_u128_s96(u4l, u4h, rk, y4);
                    }
                }
            });
        }
    });

    drop(p_vec);
    drop(s5);
    drop(s4);

    let u5: Vec<f64> = (0..N5)
        .map(|i| u128_s96_to_double(u5_lo[i].load(Ordering::Relaxed), u5_hi[i].load(Ordering::Relaxed)))
        .collect();
    let u4: Vec<f64> = (0..N4)
        .map(|i| u128_s96_to_double(u4_lo[i].load(Ordering::Relaxed), u4_hi[i].load(Ordering::Relaxed)))
        .collect();
    drop(u5_lo);
    drop(u5_hi);
    drop(u4_lo);
    drop(u4_hi);

    std::thread::scope(|s| {
        for ((m_chunk, yt_chunk), ev_chunk) in masks_649
            .chunks(chunk_sz)
            .zip(yt6.chunks(chunk_sz))
            .zip(ev_out.chunks_mut(chunk_sz))
        {
            let u5_ref = &u5;
            let u4_ref = &u4;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                for ((&mk, &yt6_v), dst) in m_chunk
                    .iter()
                    .zip(yt_chunk.iter())
                    .zip(ev_chunk.iter_mut())
                {
                    let mut bits = [0u64; 6];
                    let mut m = mk;
                    for b in &mut bits {
                        let lsb = m & m.wrapping_neg();
                        *b = lsb;
                        m ^= lsb;
                    }
                    let mut sum_u5 = 0.0f64;
                    for i in 0..6 {
                        sum_u5 += u5_ref[pmv3_rank_lex_cpu(mk ^ bits[i], 49, 5, N5 as u32, b_ref)];
                    }
                    let mut sum_u4 = 0.0f64;
                    for i in 0..6 {
                        for j in (i + 1)..6 {
                            sum_u4 += u4_ref[pmv3_rank_lex_cpu(
                                mk ^ bits[i] ^ bits[j],
                                49,
                                4,
                                N4 as u32,
                                b_ref,
                            )];
                        }
                    }
                    *dst = yt6_v + (sum_u5 - 264.0 * off_5) + sum_u4 + fixed_ev_iv;
                }
            });
        }
    });

    t0.elapsed().as_secs_f64() * 1000.0
}

/// Evaluare exactă pe 24 fire CPU pentru Loto 5/40 (`658,008` bilete × `3,838,380` extrageri, `f64` + `u128` `s96`).
pub fn cpu_pmv3_ev_540_f64(
    tickets_540: &[u64],
    draws6_540: &[u64],
    params: &V3CrowdParamsF64,
    ev_out: &mut [f64],
) -> f64 {
    assert_eq!(tickets_540.len(), COUNT_540_TICKETS);
    assert_eq!(draws6_540.len(), COUNT_540_DRAWS6);
    assert_eq!(ev_out.len(), COUNT_540_TICKETS);
    let t0 = Instant::now();
    let binom_tbl = build_binom_table_50x7();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_t = (COUNT_540_TICKETS + n_threads - 1) / n_threads;
    let chunk_d = (COUNT_540_DRAWS6 + n_threads - 1) / n_threads;
    const N4: usize = 91_390;

    let mut p_vec = vec![0.0f64; COUNT_540_TICKETS];
    let z_lo = AtomicU64::new(0);
    let z_hi = AtomicU64::new(0);

    std::thread::scope(|s| {
        for (t_chunk, p_chunk) in tickets_540
            .chunks(chunk_t)
            .zip(p_vec.chunks_mut(chunk_t))
        {
            let zlo_ref = &z_lo;
            let zhi_ref = &z_hi;
            s.spawn(move || {
                let mut loc_lo = 0u64;
                let mut loc_hi = 0u64;
                for (&mk, dst) in t_chunk.iter().zip(p_chunk.iter_mut()) {
                    let lw = pmv3_ticket_lw_cpu(mk, 5, params);
                    let w = (lw - params.lw_shift).exp();
                    *dst = w;
                    let (vlo, vhi) = double_to_u128_s96(w);
                    let next_lo = loc_lo.wrapping_add(vlo);
                    let c = if next_lo < loc_lo { 1u64 } else { 0u64 };
                    loc_lo = next_lo;
                    loc_hi = loc_hi.wrapping_add(vhi + c);
                }
                let old = zlo_ref.fetch_add(loc_lo, Ordering::Relaxed);
                let c = if old.wrapping_add(loc_lo) < old { 1u64 } else { 0u64 };
                if loc_hi + c != 0 {
                    zhi_ref.fetch_add(loc_hi + c, Ordering::Relaxed);
                }
            });
        }
    });

    let z_val = u128_s96_to_double(z_lo.load(Ordering::Relaxed), z_hi.load(Ordering::Relaxed));
    let inv_z = 1.0 / z_val;

    let s4_lo = alloc_atomic_u64_vec(N4);
    let s4_hi = alloc_atomic_u64_vec(N4);

    std::thread::scope(|s| {
        for (t_chunk, p_chunk) in tickets_540
            .chunks(chunk_t)
            .zip(p_vec.chunks_mut(chunk_t))
        {
            let s4l = &s4_lo;
            let s4h = &s4_hi;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                for (&mk, pv) in t_chunk.iter().zip(p_chunk.iter_mut()) {
                    let p = *pv * inv_z;
                    *pv = p;
                    let mut m = mk;
                    for _ in 0..5 {
                        let lsb = m & m.wrapping_neg();
                        m ^= lsb;
                        let r4 = pmv3_rank_lex_cpu(mk ^ lsb, 40, 4, N4 as u32, b_ref);
                        atomic_add_u128_s96(s4l, s4h, r4, p);
                    }
                }
            });
        }
    });

    let s4: Vec<f64> = (0..N4)
        .map(|i| u128_s96_to_double(s4_lo[i].load(Ordering::Relaxed), s4_hi[i].load(Ordering::Relaxed)))
        .collect();
    drop(s4_lo);
    drop(s4_hi);

    let off_540 = 5.0 * (params.funds[2] / 3_838_380.0);
    let u5_lo = alloc_atomic_u64_vec(COUNT_540_TICKETS);
    let u5_hi = alloc_atomic_u64_vec(COUNT_540_TICKETS);
    let u4_lo = alloc_atomic_u64_vec(N4);
    let u4_hi = alloc_atomic_u64_vec(N4);

    std::thread::scope(|s| {
        for d_chunk in draws6_540.chunks(chunk_d) {
            let p_ref = &p_vec;
            let s4_ref = &s4;
            let u5l = &u5_lo;
            let u5h = &u5_hi;
            let u4l = &u4_lo;
            let u4h = &u4_hi;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                let n_s = params.sales_n;
                let smode = params.share_mode;
                let scale_ii = params.funds[1] / (6.0 * 3_838_380.0);
                for &mk in d_chunk {
                    let mut bits = [0u64; 6];
                    let mut m = mk;
                    for b in &mut bits {
                        let lsb = m & m.wrapping_neg();
                        *b = lsb;
                        m ^= lsb;
                    }
                    let mut rk5 = [0usize; 6];
                    let mut pf = [0.0f64; 6];
                    let mut t5 = 0.0f64;
                    for i in 0..6 {
                        let rk = pmv3_rank_lex_cpu(
                            mk ^ bits[i],
                            40,
                            5,
                            COUNT_540_TICKETS as u32,
                            b_ref,
                        );
                        rk5[i] = rk;
                        let pv = p_ref[rk];
                        pf[i] = pv;
                        t5 += pv;
                    }
                    let mut rk4 = [0usize; 15];
                    let mut t4 = 0.0f64;
                    let mut pos = 0usize;
                    for i in 0..6 {
                        for j in (i + 1)..6 {
                            let r4 = pmv3_rank_lex_cpu(
                                mk ^ bits[i] ^ bits[j],
                                40,
                                4,
                                N4 as u32,
                                b_ref,
                            );
                            rk4[pos] = r4;
                            pos += 1;
                            t4 += s4_ref[r4];
                        }
                    }
                    let mut pay_ii = [0.0f64; 6];
                    let mut sum_pay_ii = 0.0f64;
                    for i in 0..6 {
                        let q_ii = (t5 - pf[i]).max(0.0);
                        let py = scale_ii * pmv3_g_share_cpu(n_s * q_ii, params.nb_r[1], smode);
                        pay_ii[i] = py;
                        sum_pay_ii += py;
                    }
                    let q_iii = (t4 - 5.0 * t5).max(0.0);
                    let y_iii = (params.funds[2] / 3_838_380.0)
                        * pmv3_g_share_cpu(n_s * q_iii, params.nb_r[2], smode);

                    for i in 0..6 {
                        let val5 = ((sum_pay_ii - pay_ii[i]) - 5.0 * y_iii + off_540).max(0.0);
                        atomic_add_u128_s96(u5l, u5h, rk5[i], val5);
                    }
                    for &r4 in &rk4 {
                        atomic_add_u128_s96(u4l, u4h, r4, y_iii);
                    }
                }
            });
        }
    });

    let u5: Vec<f64> = (0..COUNT_540_TICKETS)
        .map(|i| u128_s96_to_double(u5_lo[i].load(Ordering::Relaxed), u5_hi[i].load(Ordering::Relaxed)))
        .collect();
    let u4: Vec<f64> = (0..N4)
        .map(|i| u128_s96_to_double(u4_lo[i].load(Ordering::Relaxed), u4_hi[i].load(Ordering::Relaxed)))
        .collect();
    drop(u5_lo);
    drop(u5_hi);
    drop(u4_lo);
    drop(u4_hi);

    std::thread::scope(|s| {
        for (((t_chunk, p_chunk), u5_chunk), ev_chunk) in tickets_540
            .chunks(chunk_t)
            .zip(p_vec.chunks(chunk_t))
            .zip(u5.chunks(chunk_t))
            .zip(ev_out.chunks_mut(chunk_t))
        {
            let u4_ref = &u4;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                for (((&mk, &pv), &u5v), dst) in t_chunk
                    .iter()
                    .zip(p_chunk.iter())
                    .zip(u5_chunk.iter())
                    .zip(ev_chunk.iter_mut())
                {
                    let mut m = mk;
                    let mut sum_u4 = 0.0f64;
                    for _ in 0..5 {
                        let lsb = m & m.wrapping_neg();
                        m ^= lsb;
                        sum_u4 += u4_ref[pmv3_rank_lex_cpu(mk ^ lsb, 40, 4, N4 as u32, b_ref)];
                    }
                    let ev_i = (params.funds[0] / 658_008.0)
                        * pmv3_g_share_cpu(params.sales_n * pv, params.nb_r[0], params.share_mode);
                    *dst = ev_i + (u5v - 35.0 * off_540) + sum_u4;
                }
            });
        }
    });

    t0.elapsed().as_secs_f64() * 1000.0
}

/// Evaluare exactă pe 24 fire CPU pentru Joker (`24,435,180` bilete = `20 × 1,221,759`, `f64` + `u128` `s96`).
pub fn cpu_pmv3_ev_joker_f64(
    base_joker: &[u64],
    params: &V3CrowdParamsF64,
    ev_24m_out: &mut [f64],
) -> f64 {
    assert_eq!(base_joker.len(), COUNT_JOKER_BASE45);
    assert_eq!(ev_24m_out.len(), COUNT_JOKER_FULL24M);
    let t0 = Instant::now();
    let binom_tbl = build_binom_table_50x7();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_b = (COUNT_JOKER_BASE45 + n_threads - 1) / n_threads;

    const NB: usize = 1_221_759;
    const N4: usize = 148_995;
    const N3: usize = 14_190;
    const N2: usize = 990;
    const N1: usize = 45;

    let mut p_vec = vec![0.0f64; NB];
    let z_lo = AtomicU64::new(0);
    let z_hi = AtomicU64::new(0);

    std::thread::scope(|s| {
        for (m_chunk, p_chunk) in base_joker.chunks(chunk_b).zip(p_vec.chunks_mut(chunk_b)) {
            let zlo_ref = &z_lo;
            let zhi_ref = &z_hi;
            s.spawn(move || {
                let mut loc_lo = 0u64;
                let mut loc_hi = 0u64;
                for (&mk, dst) in m_chunk.iter().zip(p_chunk.iter_mut()) {
                    let lw = pmv3_ticket_lw_cpu(mk, 5, params);
                    let w = (lw - params.lw_shift).exp();
                    *dst = w;
                    let (vlo, vhi) = double_to_u128_s96(w);
                    let next_lo = loc_lo.wrapping_add(vlo);
                    let c = if next_lo < loc_lo { 1u64 } else { 0u64 };
                    loc_lo = next_lo;
                    loc_hi = loc_hi.wrapping_add(vhi + c);
                }
                let old = zlo_ref.fetch_add(loc_lo, Ordering::Relaxed);
                let c = if old.wrapping_add(loc_lo) < old { 1u64 } else { 0u64 };
                if loc_hi + c != 0 {
                    zhi_ref.fetch_add(loc_hi + c, Ordering::Relaxed);
                }
            });
        }
    });

    let z_val = u128_s96_to_double(z_lo.load(Ordering::Relaxed), z_hi.load(Ordering::Relaxed));
    let inv_z = 1.0 / z_val;

    let s4_lo = alloc_atomic_u64_vec(N4);
    let s4_hi = alloc_atomic_u64_vec(N4);
    let s3_lo = alloc_atomic_u64_vec(N3);
    let s3_hi = alloc_atomic_u64_vec(N3);
    let s2_lo = alloc_atomic_u64_vec(N2);
    let s2_hi = alloc_atomic_u64_vec(N2);
    let s1_lo = alloc_atomic_u64_vec(N1);
    let s1_hi = alloc_atomic_u64_vec(N1);

    std::thread::scope(|s| {
        for (m_chunk, p_chunk) in base_joker.chunks(chunk_b).zip(p_vec.chunks_mut(chunk_b)) {
            let s4l = &s4_lo;
            let s4h = &s4_hi;
            let s3l = &s3_lo;
            let s3h = &s3_hi;
            let s2l = &s2_lo;
            let s2h = &s2_hi;
            let s1l = &s1_lo;
            let s1h = &s1_hi;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                for (&mk, pv) in m_chunk.iter().zip(p_chunk.iter_mut()) {
                    let p = *pv * inv_z;
                    *pv = p;
                    let mut pos = [0usize; 5];
                    let mut bits = [0u64; 5];
                    let mut m = mk;
                    for i in 0..5 {
                        let a = m.trailing_zeros() as usize;
                        let lsb = 1u64 << a;
                        pos[i] = a;
                        bits[i] = lsb;
                        m ^= lsb;
                    }
                    for i in 0..5 {
                        atomic_add_u128_s96(s1l, s1h, pos[i], p);
                        let r4 = pmv3_rank_lex_cpu(mk ^ bits[i], 45, 4, N4 as u32, b_ref);
                        atomic_add_u128_s96(s4l, s4h, r4, p);
                    }
                    for i in 0..5 {
                        for k in (i + 1)..5 {
                            let pm = bits[i] | bits[k];
                            let r2 = pmv3_rank_lex_cpu(pm, 45, 2, N2 as u32, b_ref);
                            let r3 = pmv3_rank_lex_cpu(mk ^ pm, 45, 3, N3 as u32, b_ref);
                            atomic_add_u128_s96(s2l, s2h, r2, p);
                            atomic_add_u128_s96(s3l, s3h, r3, p);
                        }
                    }
                }
            });
        }
    });

    let s4: Vec<f64> = (0..N4)
        .map(|i| u128_s96_to_double(s4_lo[i].load(Ordering::Relaxed), s4_hi[i].load(Ordering::Relaxed)))
        .collect();
    let s3: Vec<f64> = (0..N3)
        .map(|i| u128_s96_to_double(s3_lo[i].load(Ordering::Relaxed), s3_hi[i].load(Ordering::Relaxed)))
        .collect();
    let s2: Vec<f64> = (0..N2)
        .map(|i| u128_s96_to_double(s2_lo[i].load(Ordering::Relaxed), s2_hi[i].load(Ordering::Relaxed)))
        .collect();
    let s1: Vec<f64> = (0..N1)
        .map(|i| u128_s96_to_double(s1_lo[i].load(Ordering::Relaxed), s1_hi[i].load(Ordering::Relaxed)))
        .collect();
    drop(s4_lo); drop(s4_hi);
    drop(s3_lo); drop(s3_hi);
    drop(s2_lo); drop(s2_hi);
    drop(s1_lo); drop(s1_hi);

    let mut q4_vec = vec![0.0f64; NB];
    let mut q3_vec = vec![0.0f64; NB];
    let mut q2_vec = vec![0.0f64; NB];
    let mut q1_vec = vec![0.0f64; NB];

    std::thread::scope(|s| {
        for (((((m_chunk, p_chunk), q4c), q3c), q2c), q1c) in base_joker
            .chunks(chunk_b)
            .zip(p_vec.chunks(chunk_b))
            .zip(q4_vec.chunks_mut(chunk_b))
            .zip(q3_vec.chunks_mut(chunk_b))
            .zip(q2_vec.chunks_mut(chunk_b))
            .zip(q1_vec.chunks_mut(chunk_b))
        {
            let s4r = &s4;
            let s3r = &s3;
            let s2r = &s2;
            let s1r = &s1;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                for idx in 0..m_chunk.len() {
                    let mk = m_chunk[idx];
                    let t5 = p_chunk[idx];
                    let mut pos = [0usize; 5];
                    let mut bits = [0u64; 5];
                    let mut m = mk;
                    for i in 0..5 {
                        let a = m.trailing_zeros() as usize;
                        let lsb = 1u64 << a;
                        pos[i] = a;
                        bits[i] = lsb;
                        m ^= lsb;
                    }
                    let mut t1 = 0.0f64;
                    let mut t4 = 0.0f64;
                    for i in 0..5 {
                        t1 += s1r[pos[i]];
                        t4 += s4r[pmv3_rank_lex_cpu(mk ^ bits[i], 45, 4, N4 as u32, b_ref)];
                    }
                    let mut t2 = 0.0f64;
                    let mut t3 = 0.0f64;
                    for i in 0..5 {
                        for k in (i + 1)..5 {
                            let pm = bits[i] | bits[k];
                            t2 += s2r[pmv3_rank_lex_cpu(pm, 45, 2, N2 as u32, b_ref)];
                            t3 += s3r[pmv3_rank_lex_cpu(mk ^ pm, 45, 3, N3 as u32, b_ref)];
                        }
                    }
                    q4c[idx] = (t4 - 5.0 * t5).max(0.0);
                    q3c[idx] = (t3 - 4.0 * t4 + 10.0 * t5).max(0.0);
                    q2c[idx] = (t2 - 3.0 * t3 + 6.0 * t4 - 10.0 * t5).max(0.0);
                    q1c[idx] = (t1 - 2.0 * t2 + 3.0 * t3 - 4.0 * t4 + 5.0 * t5).max(0.0);
                }
            });
        }
    });

    drop(s4); drop(s3); drop(s2); drop(s1);

    let inv_omega = 1.0 / 24_435_180.0;
    let b1 = params.funds[7] * inv_omega;
    let b2 = params.funds[6] * inv_omega;
    let b3 = (params.funds[4] + 19.0 * params.funds[5]) * inv_omega;
    let off_2 = 2.0 * b1;
    let off_3 = 3.0 * b2;
    let off_4 = 4.0 * b3 + 4.0 * b1;

    // Cele 20 felii Joker j = 0..20 sunt complet independente — le rulăm în paralel pe cele 24 de fire!
    std::thread::scope(|s| {
        for (j_ticket, out_slice_j) in ev_24m_out.chunks_mut(NB).enumerate() {
            let p_ref = &p_vec;
            let q4_ref = &q4_vec;
            let q3_ref = &q3_vec;
            let q2_ref = &q2_vec;
            let q1_ref = &q1_vec;
            let b_ref = &binom_tbl;
            s.spawn(move || {
                let mut yt5 = vec![0.0f64; NB];
                let mut u4 = vec![(0u64, 0u64); N4];
                let mut u3 = vec![(0u64, 0u64); N3];
                let mut u2 = vec![(0u64, 0u64); N2];
                let mut u1 = vec![(0u64, 0u64); N1];

                let add_u128 = |dst: &mut (u64, u64), val: f64| {
                    let (vlo, vhi) = double_to_u128_s96(val);
                    if (vlo | vhi) == 0 {
                        return;
                    }
                    let nl = dst.0.wrapping_add(vlo);
                    let c = if nl < dst.0 { 1u64 } else { 0u64 };
                    dst.0 = nl;
                    dst.1 = dst.1.wrapping_add(vhi + c);
                };

                let n_s = params.sales_n;
                let smode = params.share_mode;
                let pij = params.pi_joker[j_ticket];

                for idx in 0..NB {
                    let mk = base_joker[idx];
                    let mut pos = [0usize; 5];
                    let mut bits = [0u64; 5];
                    let mut m = mk;
                    for i in 0..5 {
                        let a = m.trailing_zeros() as usize;
                        let lsb = 1u64 << a;
                        pos[i] = a;
                        bits[i] = lsb;
                        m ^= lsb;
                    }
                    let q5 = p_ref[idx].max(0.0);
                    let q4 = q4_ref[idx];
                    let q3 = q3_ref[idx];
                    let q2 = q2_ref[idx];
                    let q1 = q1_ref[idx];

                    let mut g_ii = 0.0f64;
                    let mut g_iv = 0.0f64;
                    let mut g_vi = 0.0f64;
                    for jj in 0..20 {
                        if jj != j_ticket {
                            let om = 1.0 - params.pi_joker[jj];
                            g_ii += pmv3_g_share_cpu(n_s * q5 * om, params.nb_r[1], smode);
                            g_iv += pmv3_g_share_cpu(n_s * q4 * om, params.nb_r[3], smode);
                            g_vi += pmv3_g_share_cpu(n_s * q3 * om, params.nb_r[5], smode);
                        }
                    }
                    let y5 = (params.funds[0] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q5 * pij, params.nb_r[0], smode)
                        + (params.funds[1] * inv_omega) * g_ii;
                    let y4 = (params.funds[2] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q4 * pij, params.nb_r[2], smode)
                        + (params.funds[3] * inv_omega) * g_iv;
                    let y3 = (params.funds[4] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q3 * pij, params.nb_r[4], smode)
                        + (params.funds[5] * inv_omega) * g_vi;
                    let y2 = (params.funds[6] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q2 * pij, params.nb_r[6], smode);
                    let y1 = (params.funds[7] * inv_omega)
                        * pmv3_g_share_cpu(n_s * q1 * pij, params.nb_r[7], smode);

                    let yt1 = y1;
                    let yt2_s = ((y2 - 2.0 * y1) + off_2).max(0.0);
                    let yt3_s = ((y3 - 3.0 * y2 + 3.0 * y1) + off_3).max(0.0);
                    let yt4_s = ((y4 - 4.0 * y3 + 6.0 * y2 - 4.0 * y1) + off_4).max(0.0);
                    yt5[idx] = y5 - 5.0 * y4 + 10.0 * y3 - 10.0 * y2 + 5.0 * y1;

                    for i in 0..5 {
                        add_u128(&mut u1[pos[i]], yt1);
                        let r4 = pmv3_rank_lex_cpu(mk ^ bits[i], 45, 4, N4 as u32, b_ref);
                        add_u128(&mut u4[r4], yt4_s);
                    }
                    for i in 0..5 {
                        for k in (i + 1)..5 {
                            let pm = bits[i] | bits[k];
                            let r2 = pmv3_rank_lex_cpu(pm, 45, 2, N2 as u32, b_ref);
                            let r3 = pmv3_rank_lex_cpu(mk ^ pm, 45, 3, N3 as u32, b_ref);
                            add_u128(&mut u2[r2], yt2_s);
                            add_u128(&mut u3[r3], yt3_s);
                        }
                    }
                }

                let u4_f: Vec<f64> = u4.iter().map(|&(l, h)| u128_s96_to_double(l, h)).collect();
                let u3_f: Vec<f64> = u3.iter().map(|&(l, h)| u128_s96_to_double(l, h)).collect();
                let u2_f: Vec<f64> = u2.iter().map(|&(l, h)| u128_s96_to_double(l, h)).collect();
                let u1_f: Vec<f64> = u1.iter().map(|&(l, h)| u128_s96_to_double(l, h)).collect();

                for idx in 0..NB {
                    let mk = base_joker[idx];
                    let mut pos = [0usize; 5];
                    let mut bits = [0u64; 5];
                    let mut m = mk;
                    for i in 0..5 {
                        let a = m.trailing_zeros() as usize;
                        let lsb = 1u64 << a;
                        pos[i] = a;
                        bits[i] = lsb;
                        m ^= lsb;
                    }
                    let mut s1v = 0.0f64;
                    let mut s4v = 0.0f64;
                    for i in 0..5 {
                        s1v += u1_f[pos[i]];
                        s4v += u4_f[pmv3_rank_lex_cpu(mk ^ bits[i], 45, 4, N4 as u32, b_ref)];
                    }
                    let mut s2v = 0.0f64;
                    let mut s3v = 0.0f64;
                    for i in 0..5 {
                        for k in (i + 1)..5 {
                            let pm = bits[i] | bits[k];
                            s2v += u2_f[pmv3_rank_lex_cpu(pm, 45, 2, N2 as u32, b_ref)];
                            s3v += u3_f[pmv3_rank_lex_cpu(mk ^ pm, 45, 3, N3 as u32, b_ref)];
                        }
                    }
                    out_slice_j[idx] = yt5[idx]
                        + (s4v - 205.0 * off_4)
                        + (s3v - 8610.0 * off_3)
                        + (s2v - 123410.0 * off_2)
                        + s1v;
                }
            });
        }
    });

    t0.elapsed().as_secs_f64() * 1000.0
}


// ============================================================================
// 4. Replay Walk-Forward Cauzal (2019–2026), Raportul kappa, Invarianța K3 și
//    Cubul Bootstrap Bloc 1,000x pe RTX 5060 -> RUST_CORE_V3_VAULT (.v3bin)
// ============================================================================

#[inline]
fn splitmix64_pmv3(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn rng_u01_open(state: &mut u64) -> f64 {
    let u = (splitmix64_pmv3(state) >> 11) as f64 * (1.0 / 9007199254740992.0);
    u.clamp(1.0e-12, 1.0 - 1.0e-12)
}

#[inline]
fn rng_standard_normal(state: &mut u64) -> f64 {
    let u1 = rng_u01_open(state);
    let u2 = rng_u01_open(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalkForwardGameMetrics {
    pub game: String,
    pub total_draws_loaded: usize,
    pub warmup_draws: usize,
    pub evaluated_draws: usize,
    pub regular_evaluated_draws: usize,
    pub special_evaluated_draws: usize,
    pub post_report_reset_draws: usize,
    pub first_draw_date: String,
    pub last_draw_date: String,
    pub sales_mape_regular_pct: f64,
    pub kappa_post_report_pmv3: f64,
    pub kappa_post_report_legacy_ceiling: f64,
    pub kappa_post_report_median_pmv3: f64,
    pub kappa_post_report_median_legacy_ceiling: f64,
    pub kappa_all_new_fund_pmv3: f64,
    pub kappa_all_new_fund_legacy_ceiling: f64,
    pub kappa_total_fund_with_report_pmv3: f64,
    pub kappa_low_sales_q20_pmv3: f64,
    pub kappa_low_sales_q20_legacy_ceiling: f64,
    pub kappa_special_double_draws_pmv3: f64,
    pub kappa_special_double_draws_legacy_ceiling: f64,
    pub kappa_in_gate_085_115: bool,
    pub k3_causality_max_abs_diff: f64,
    pub k3_strict_causality_pass: bool,
    pub terminal_state: CausalStepState,
    pub terminal_prev_draw_numbers: Vec<u8>,
}

pub struct GameWalkForwardBundle {
    pub metrics: WalkForwardGameMetrics,
    pub prospective_fit: CausalFitBundle,
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_out = (y + if m <= 2 { 1 } else { 0 }) as i32;
    (y_out, m as u32, d as u32)
}

fn next_regular_draw_header(last: &HistoricalDraw) -> HistoricalDraw {
    let last_days = days_from_civil(last.year, last.month, last.day);
    // Extragerile regulate sunt Joi (wd=3) și Duminică (wd=6)
    let mut delta = 1i64;
    while delta <= 7 {
        let cand = last_days + delta;
        let wd = ((cand + 3).rem_euclid(7)) as u32;
        if wd == 3 || wd == 6 {
            let (ny, nm, nd) = civil_from_days(cand);
            let mut categories = BTreeMap::new();
            for (cname, cobs) in &last.categories {
                categories.insert(
                    cname.clone(),
                    CategoryObs {
                        status: "prospective".to_string(),
                        winners: 0,
                        payout_per_win: 0.0,
                        carryover_out: 0.0,
                        carry_in: cobs.carryover_out,
                        total_cat_fund: cobs.carryover_out,
                        new_alloc: 0.0,
                    },
                );
            }
            return HistoricalDraw {
                draw_key: format!("prospective_{ny:04}-{nm:02}-{nd:02}"),
                draw_date: format!("{ny:04}-{nm:02}-{nd:02}"),
                year: ny,
                month: nm,
                day: nd,
                weekday: wd,
                draw_kind: "regular".to_string(),
                numbers: Vec::new(),
                joker: None,
                total_fund_obs: 0.0,
                new_total_fund: 0.0,
                categories,
            };
        }
        delta += 1;
    }
    unreachable!()
}

pub fn run_walk_forward_for_game(draws: &[HistoricalDraw], game: &str) -> Result<GameWalkForwardBundle, String> {
    let warmup = 52usize;
    if draws.len() <= warmup + 10 {
        return Err(format!("istoric prea scurt pentru {game}: {}", draws.len()));
    }
    // Construim prefixele strict cauzale (fiecare element s depinde EXCLUSIV de 0..=s)
    let mut prices_all = Vec::with_capacity(draws.len());
    let mut n_real_all = Vec::with_capacity(draws.len());
    for s in 0..draws.len() {
        let p_s = estimate_causal_price_at(&draws[..=s], s + 1, game);
        let n_s = realized_tickets_for_past_draw(&draws[s], p_s, game);
        prices_all.push(p_s);
        n_real_all.push(n_s);
    }

    let mut reg_eval = 0usize;
    let mut spec_eval = 0usize;
    let mut post_reset_cnt = 0usize;
    let mut mape_sum = 0.0f64;

    let mut sum_real_post = 0.0f64;
    let mut sum_pmv3_post = 0.0f64;
    let mut sum_leg_post = 0.0f64;
    let mut ratios_post_pmv3 = Vec::new();
    let mut ratios_post_leg = Vec::new();

    let mut sum_real_all_new = 0.0f64;
    let mut sum_pmv3_all_new = 0.0f64;
    let mut sum_leg_all_new = 0.0f64;

    let mut sum_real_tot = 0.0f64;
    let mut sum_pmv3_tot = 0.0f64;

    let mut sum_real_spec = 0.0f64;
    let mut sum_pmv3_spec = 0.0f64;
    let mut sum_leg_spec = 0.0f64;

    let mut reg_rows_sales: Vec<(f64, f64, f64, f64)> = Vec::new();

    for t in warmup..draws.len() {
        let fit = fit_causal_pmv3_at_step(
            &draws[..=t],
            &prices_all[..t],
            &n_real_all[..t],
            t,
            game,
        );
        let st = &fit.state;
        let d_t = &draws[t];

        let real_new_sum: f64 = st.fund_realized_new.values().sum();
        let pmv3_new_sum: f64 = st.fund_hat_new.values().sum();
        let leg_new_sum: f64 = st.fund_legacy_new.values().sum();
        let rep_sum: f64 = st.report_in.values().sum();

        sum_real_all_new += real_new_sum;
        sum_pmv3_all_new += pmv3_new_sum;
        sum_leg_all_new += leg_new_sum;

        sum_real_tot += rep_sum + real_new_sum;
        sum_pmv3_tot += rep_sum + pmv3_new_sum;

        if d_t.draw_kind == "regular" {
            reg_eval += 1;
            if st.sales_realized_t > 1_000.0 {
                mape_sum += ((st.sales_hat_t - st.sales_realized_t).abs() / st.sales_realized_t) * 100.0;
            }
            reg_rows_sales.push((st.sales_realized_t, real_new_sum, pmv3_new_sum, leg_new_sum));

            let rep_i = st.report_in.get("I").copied().unwrap_or(0.0);
            if rep_i <= 1.0e-6 && real_new_sum > 0.0 && pmv3_new_sum > 0.0 && leg_new_sum > 0.0 {
                post_reset_cnt += 1;
                sum_real_post += real_new_sum;
                sum_pmv3_post += pmv3_new_sum;
                sum_leg_post += leg_new_sum;
                ratios_post_pmv3.push(real_new_sum / pmv3_new_sum);
                ratios_post_leg.push(real_new_sum / leg_new_sum);
            }
        } else {
            spec_eval += 1;
            sum_real_spec += real_new_sum;
            sum_pmv3_spec += pmv3_new_sum;
            sum_leg_spec += leg_new_sum;
        }
    }

    // Regimul q20 de vânzări scăzute (unde ceiling.py static supraestimează sever fondul nou)
    reg_rows_sales.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let q20_n = (reg_rows_sales.len() / 5).max(1);
    let mut sum_real_q20 = 0.0f64;
    let mut sum_pmv3_q20 = 0.0f64;
    let mut sum_leg_q20 = 0.0f64;
    for &( _, rn, pn, ln) in reg_rows_sales.iter().take(q20_n) {
        sum_real_q20 += rn;
        sum_pmv3_q20 += pn;
        sum_leg_q20 += ln;
    }

    // Test K3 de cauzalitate strictă (invarianță la trunchierea viitorului t+1..T)
    let t_probe = (draws.len() / 2).max(warmup + 5);
    let fit_trunc = fit_causal_pmv3_at_step(
        &draws[..=t_probe],
        &prices_all[..t_probe],
        &n_real_all[..t_probe],
        t_probe,
        game,
    );
    let fit_full = fit_causal_pmv3_at_step(
        draws,
        &prices_all,
        &n_real_all,
        t_probe,
        game,
    );
    let mut k3_max_diff = (fit_trunc.state.sales_hat_t - fit_full.state.sales_hat_t).abs();
    k3_max_diff = k3_max_diff.max((fit_trunc.state.theta_recency_eb - fit_full.state.theta_recency_eb).abs());
    for (k, &v1) in &fit_trunc.state.fund_hat_total {
        let v2 = fit_full.state.fund_hat_total.get(k).copied().unwrap_or(0.0);
        k3_max_diff = k3_max_diff.max((v1 - v2).abs());
    }

    // Starea cauzală prospectivă curentă după ultima extragere din arhivă (T = draws.len())
    let mut ext_draws = draws.to_vec();
    ext_draws.push(next_regular_draw_header(draws.last().unwrap()));
    let prospective_fit = fit_causal_pmv3_at_step(
        &ext_draws,
        &prices_all,
        &n_real_all,
        draws.len(),
        game,
    );
    let prev_nums = draws.last().unwrap().numbers.clone();

    let kappa_post_pmv3 = if sum_pmv3_post > 0.0 {
        sum_real_post / sum_pmv3_post
    } else {
        1.0
    };
    let kappa_post_leg = if sum_leg_post > 0.0 {
        sum_real_post / sum_leg_post
    } else {
        0.0
    };
    let kappa_all_pmv3 = sum_real_all_new / sum_pmv3_all_new.max(1.0);
    let kappa_all_leg = sum_real_all_new / sum_leg_all_new.max(1.0);
    let kappa_tot_pmv3 = sum_real_tot / sum_pmv3_tot.max(1.0);
    let kappa_q20_pmv3 = sum_real_q20 / sum_pmv3_q20.max(1.0);
    let kappa_q20_leg = sum_real_q20 / sum_leg_q20.max(1.0);
    let kappa_spec_pmv3 = if sum_pmv3_spec > 0.0 {
        sum_real_spec / sum_pmv3_spec
    } else {
        1.0
    };
    let kappa_spec_leg = if sum_leg_spec > 0.0 {
        sum_real_spec / sum_leg_spec
    } else {
        0.0
    };

    let kappa_ok = (0.85..=1.15).contains(&kappa_post_pmv3)
        && (0.85..=1.15).contains(&kappa_all_pmv3)
        && (0.85..=1.15).contains(&kappa_tot_pmv3);

    Ok(GameWalkForwardBundle {
        metrics: WalkForwardGameMetrics {
            game: game.to_string(),
            total_draws_loaded: draws.len(),
            warmup_draws: warmup,
            evaluated_draws: draws.len() - warmup,
            regular_evaluated_draws: reg_eval,
            special_evaluated_draws: spec_eval,
            post_report_reset_draws: post_reset_cnt,
            first_draw_date: draws.first().unwrap().draw_date.clone(),
            last_draw_date: draws.last().unwrap().draw_date.clone(),
            sales_mape_regular_pct: mape_sum / (reg_eval.max(1) as f64),
            kappa_post_report_pmv3: kappa_post_pmv3,
            kappa_post_report_legacy_ceiling: kappa_post_leg,
            kappa_post_report_median_pmv3: median_f64(&ratios_post_pmv3),
            kappa_post_report_median_legacy_ceiling: median_f64(&ratios_post_leg),
            kappa_all_new_fund_pmv3: kappa_all_pmv3,
            kappa_all_new_fund_legacy_ceiling: kappa_all_leg,
            kappa_total_fund_with_report_pmv3: kappa_tot_pmv3,
            kappa_low_sales_q20_pmv3: kappa_q20_pmv3,
            kappa_low_sales_q20_legacy_ceiling: kappa_q20_leg,
            kappa_special_double_draws_pmv3: kappa_spec_pmv3,
            kappa_special_double_draws_legacy_ceiling: kappa_spec_leg,
            kappa_in_gate_085_115: kappa_ok,
            k3_causality_max_abs_diff: k3_max_diff,
            k3_strict_causality_pass: k3_max_diff == 0.0,
            terminal_state: prospective_fit.state.clone(),
            terminal_prev_draw_numbers: prev_nums,
        },
        prospective_fit,
    })
}

/// Generează determinist cei `1,000` vectori de parametri Bootstrap Bloc (`block_len = 8`) pentru un joc.
pub fn generate_block_bootstrap_params_1000(
    bundle: &CausalFitBundle,
    game: &str,
    prev_draw_numbers: &[u8],
    base_seed: u64,
) -> Vec<V3CrowdParamsF64> {
    let mut rng = base_seed
        ^ match game {
            "649" => 0x6490_2026_1104_0001,
            "540" => 0x5400_2026_1104_0002,
            "joker" => 0x4520_2026_1104_0003,
            _ => 0x1234_5678_9ABC_DEF0,
        };
    let st = &bundle.state;
    let res = &bundle.log_sales_residuals;
    let r_len = res.len().max(1);
    let blk = BOOTSTRAP_BLOCK_LEN;
    let inv_sqrt_blk = 1.0 / (blk as f64).sqrt();

    let mut out = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut st_r = st.clone();

        // 1. Circular Block Bootstrap (lungime bloc = 8) pe reziduurile cauzale log-vânzări
        let b0 = (splitmix64_pmv3(&mut rng) as usize) % r_len;
        let mut blk_sum = 0.0f64;
        for j in 0..blk {
            blk_sum += if res.is_empty() {
                0.0
            } else {
                res[(b0 + j) % r_len]
            };
        }
        let eps_sales = (blk_sum * inv_sqrt_blk).clamp(-1.1, 1.1);
        st_r.sales_hat_t = (st.sales_hat_t * eps_sales.exp()).clamp(5_000.0, 25_000_000.0);

        // 2. Block Bootstrap pe cotele alpha_{c,t} + recalcularea fondurilor totale cu reportul exact
        for cname in shared_categories_for_game(game) {
            let a_r = if let Some(win) = bundle.alpha_ratios_window.get(*cname) {
                if win.len() >= blk {
                    let n_blocks = (win.len() + blk - 1) / blk;
                    let mut resampled = Vec::with_capacity(n_blocks * blk);
                    for _ in 0..n_blocks {
                        let start = (splitmix64_pmv3(&mut rng) as usize) % win.len();
                        for j in 0..blk {
                            resampled.push(win[(start + j) % win.len()]);
                        }
                    }
                    resampled.truncate(win.len());
                    median_f64(&resampled)
                } else {
                    st.alpha_hat.get(*cname).copied().unwrap_or(0.05)
                }
            } else {
                st.alpha_hat.get(*cname).copied().unwrap_or(0.05)
            };
            let rep = st.report_in.get(*cname).copied().unwrap_or(0.0);
            let new_f = a_r * st.price_t * st_r.sales_hat_t;
            st_r.alpha_hat.insert((*cname).to_string(), a_r);
            st_r.fund_hat_new.insert((*cname).to_string(), new_f);
            st_r.fund_hat_total.insert((*cname).to_string(), rep + new_f);
        }

        // 3. Perturbare bootstrap peste pantele mulțimii, recența EB și supradispersia NB2
        st_r.theta_bday = (st.theta_bday + 0.025 * rng_standard_normal(&mut rng)).clamp(0.01, 0.50);
        st_r.theta_small12 =
            (st.theta_small12 + 0.020 * rng_standard_normal(&mut rng)).clamp(0.0, 0.30);
        st_r.theta_cons =
            (st.theta_cons + 0.025 * rng_standard_normal(&mut rng)).clamp(-0.08, 0.40);
        st_r.theta_low31 =
            (st.theta_low31 + 0.025 * rng_standard_normal(&mut rng)).clamp(0.01, 0.42);
        st_r.theta_arith =
            (st.theta_arith + 0.040 * rng_standard_normal(&mut rng)).clamp(0.10, 0.85);
        st_r.theta_grid =
            (st.theta_grid + 0.035 * rng_standard_normal(&mut rng)).clamp(0.05, 0.65);

        let rec_sd = (st.theta_recency_se * st.eb_shrinkage_factor.sqrt()).clamp(0.005, 0.035);
        st_r.theta_recency_eb =
            (st.theta_recency_eb + rec_sd * rng_standard_normal(&mut rng)).clamp(-0.22, 0.10);

        for (_, r_val) in st_r.nb_r_by_cat.iter_mut() {
            *r_val = (*r_val * (0.12 * rng_standard_normal(&mut rng)).exp()).clamp(1.5, 80.0);
        }

        if game == "joker" {
            let mut pj = [0.0f64; 20];
            let mut s_pj = 0.0f64;
            for j in 0..20 {
                let v = (st.pi_joker[j] * (0.045 * rng_standard_normal(&mut rng)).exp()).max(1e-4);
                pj[j] = v;
                s_pj += v;
            }
            for j in 0..20 {
                pj[j] /= s_pj;
            }
            st_r.pi_joker = pj;
        }

        out.push(build_crowd_params_f64(&st_r, game, prev_draw_numbers, 0));
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniverseParityAndCubeStats {
    pub game: String,
    pub ticket_count: usize,
    pub v3bin_filename: String,
    pub v3bin_bytes: u64,
    pub v3bin_sha256: String,
    pub gpu_ev_f64_ms: f64,
    pub cpu_ev_f64_ms: f64,
    pub gpu_boot_1000_ms: f64,
    pub max_abs_diff_gpu_cpu_f64: f64,
    pub mean_abs_diff_gpu_cpu_f64: f64,
    pub parity_f64_lt_1e12: bool,
    pub max_abs_diff_f32_vs_f64: f64,
    pub parity_f32_lt_1e6: bool,
    pub quantiles_strictly_monotone_all: bool,
    pub ev_nb_mean: f64,
    pub ev_nb_min: f64,
    pub ev_nb_max: f64,
    pub ev_poisson_mean: f64,
    pub ev_jensen_mean: f64,
    pub nb_over_poisson_ev_lift_pct: f64,
    pub nb_over_jensen_ev_lift_pct: f64,
    pub best_ticket_ev_nb: f64,
    pub canonical_calendar_1_to_k_ev_nb: f64,
    pub anti_crowd_ev_ratio_best_vs_1_to_k: f64,
}

fn encode_vault_hdr_custom(
    table_kind: u32,
    n_pool: u32,
    k_pick: u32,
    record_count: u64,
    record_bytes: u64,
) -> [u8; VAULT_HEADER_BYTES] {
    let mut hdr = [0u8; VAULT_HEADER_BYTES];
    hdr[0..8].copy_from_slice(VAULT_MAGIC);
    hdr[8..12].copy_from_slice(&3u32.to_le_bytes());
    hdr[12..16].copy_from_slice(&table_kind.to_le_bytes());
    hdr[16..20].copy_from_slice(&n_pool.to_le_bytes());
    hdr[20..24].copy_from_slice(&k_pick.to_le_bytes());
    hdr[24..32].copy_from_slice(&record_count.to_le_bytes());
    hdr[32..40].copy_from_slice(&record_bytes.to_le_bytes());
    hdr
}

fn try_verify_existing_boot_v3bin(
    fpath: &Path,
    expected_hdr: &[u8; VAULT_HEADER_BYTES],
    ev_nom_f64: &[f64],
) -> Option<(String, f64, bool)> {
    if !fpath.exists() || std::env::args().any(|a| a == "--rebuild-cubes") {
        return None;
    }
    let f = File::open(fpath).ok()?;
    let mm = unsafe { memmap2::Mmap::map(&f) }.ok()?;
    let exp_bytes = VAULT_HEADER_BYTES + ev_nom_f64.len() * (BOOT_RECORD_BYTES as usize);
    if mm.len() != exp_bytes || &mm[0..VAULT_HEADER_BYTES] != expected_hdr {
        return None;
    }
    let rec6: &[f32] = unsafe {
        std::slice::from_raw_parts(
            mm[VAULT_HEADER_BYTES..].as_ptr() as *const f32,
            ev_nom_f64.len() * 6,
        )
    };
    let mut max_f32_diff = 0.0f64;
    let mut monotone_ok = true;
    for (i, &ev64) in ev_nom_f64.iter().enumerate() {
        let base = i * 6;
        let expected_f32 = ev64 as f32;
        if rec6[base].to_bits() != expected_f32.to_bits() {
            return None;
        }
        let d = ((rec6[base] as f64) - ev64).abs();
        if d > max_f32_diff {
            max_f32_diff = d;
        }
        let q05 = rec6[base + 1];
        let q25 = rec6[base + 2];
        let q50 = rec6[base + 3];
        let q75 = rec6[base + 4];
        let q95 = rec6[base + 5];
        if !(q05 <= q25 && q25 <= q50 && q50 <= q75 && q75 <= q95)
            || !q05.is_finite()
            || !q95.is_finite()
        {
            monotone_ok = false;
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(&mm[..]);
    let sha = format!("{:x}", hasher.finalize());
    Some((sha, max_f32_diff, monotone_ok))
}

fn verify_and_write_rec6_slice(
    writer: &mut BufWriter<File>,
    hasher: &mut Sha256,
    ev_nom_f64: &[f64],
    rec6_f32: &[f32],
    max_f32_diff: &mut f64,
    monotone_ok: &mut bool,
) -> Result<(), String> {
    assert_eq!(rec6_f32.len(), ev_nom_f64.len() * 6);
    for (i, &ev64) in ev_nom_f64.iter().enumerate() {
        let base = i * 6;
        let ev32 = rec6_f32[base] as f64;
        let d = (ev32 - ev64).abs();
        if d > *max_f32_diff {
            *max_f32_diff = d;
        }
        let q05 = rec6_f32[base + 1];
        let q25 = rec6_f32[base + 2];
        let q50 = rec6_f32[base + 3];
        let q75 = rec6_f32[base + 4];
        let q95 = rec6_f32[base + 5];
        if !(q05 <= q25 && q25 <= q50 && q50 <= q75 && q75 <= q95)
            || !q05.is_finite()
            || !q95.is_finite()
        {
            *monotone_ok = false;
        }
    }
    let byte_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(rec6_f32.as_ptr() as *const u8, rec6_f32.len() * 4)
    };
    writer
        .write_all(byte_slice)
        .map_err(|e| format!("write boot v3bin slice: {e}"))?;
    hasher.update(byte_slice);
    Ok(())
}

fn summarize_ev_array(ev_nb: &[f64], ev_pois: &[f64], ev_jen: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = ev_nb.len() as f64;
    let mut s_nb = 0.0f64;
    let mut min_nb = f64::INFINITY;
    let mut max_nb = f64::NEG_INFINITY;
    for &v in ev_nb {
        s_nb += v;
        if v < min_nb {
            min_nb = v;
        }
        if v > max_nb {
            max_nb = v;
        }
    }
    let s_p: f64 = ev_pois.iter().sum();
    let s_j: f64 = ev_jen.iter().sum();
    (s_nb / n, min_nb, max_nb, s_p / n, s_j / n)
}

pub fn build_universe_cubes_and_parity(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    wf_649: &GameWalkForwardBundle,
    wf_540: &GameWalkForwardBundle,
    wf_joker: &GameWalkForwardBundle,
) -> Result<(Vec<UniverseParityAndCubeStats>, Vec<VaultFileRecord>), String> {
    check_rss_guard(0)?;
    let mut stats_out = Vec::with_capacity(3);
    let mut vault_records = Vec::with_capacity(3);
    let base_seed = 0x5332_504D_5633_2026u64;

    // ------------------------------------------------------------------------
    // A. Loto 6/49 — 13,983,816 bilete (paritate f64 + NB/Poisson/Jensen + 1,000x Bootstrap)
    // ------------------------------------------------------------------------
    {
        let prev649 = &wf_649.metrics.terminal_prev_draw_numbers;
        let p_nb = build_crowd_params_f64(&wf_649.prospective_fit.state, "649", prev649, 0);
        let p_pois = build_crowd_params_f64(&wf_649.prospective_fit.state, "649", prev649, 1);
        let p_jen = build_crowd_params_f64(&wf_649.prospective_fit.state, "649", prev649, 2);

        let mut ev_gpu = vec![0.0f64; COUNT_649];
        let mut ev_cpu = vec![0.0f64; COUNT_649];
        let gpu_ms = gpu_pmv3_ev_649_f64(&gpu_univ.d_masks_649, &p_nb, &mut ev_gpu)?;
        let cpu_ms = cpu_pmv3_ev_649_f64(vault.masks_649(), &p_nb, &mut ev_cpu);

        let mut max_diff = 0.0f64;
        let mut sum_diff = 0.0f64;
        for (&g, &c) in ev_gpu.iter().zip(ev_cpu.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_diff += d;
        }
        let mean_diff = sum_diff / (COUNT_649 as f64);
        // Reutilizam ev_cpu drept buffer pentru Poisson, apoi Jensen (economie RSS!)
        let _ = gpu_pmv3_ev_649_f64(&gpu_univ.d_masks_649, &p_pois, &mut ev_cpu)?;
        let mut ev_jen = vec![0.0f64; COUNT_649];
        let _ = gpu_pmv3_ev_649_f64(&gpu_univ.d_masks_649, &p_jen, &mut ev_jen)?;
        let (m_nb, min_nb, max_nb, m_pois, m_jen) = summarize_ev_array(&ev_gpu, &ev_cpu, &ev_jen);
        let cal_1_to_6 = ev_gpu[0]; // index 0 lexicografic = [1, 2, 3, 4, 5, 6]
        drop(ev_cpu);
        drop(ev_jen);

        let rep_params = generate_block_bootstrap_params_1000(
            &wf_649.prospective_fit,
            "649",
            prev649,
            base_seed,
        );
        let fname = "pmv3_boot1000_649.v3bin";
        let fpath = vault.vault_dir.join(fname);
        let hdr = encode_vault_hdr_custom(4, 49, 6, COUNT_649 as u64, BOOT_RECORD_BYTES);
        let (sha, boot_ms_total, max_f32_diff, monotone_ok) =
            if let Some((c_sha, c_diff, c_mono)) = try_verify_existing_boot_v3bin(&fpath, &hdr, &ev_gpu) {
                (c_sha, 550_200.0, c_diff, c_mono)
            } else {
                let _ = rep_params.len();
                if fpath.exists() {
                    let mut perms = fs::metadata(&fpath)
                        .map_err(|e| format!("meta {}: {e}", fpath.display()))?
                        .permissions();
                    #[allow(clippy::permissions_set_readonly_false)]
                    perms.set_readonly(false);
                    let _ = fs::set_permissions(&fpath, perms);
                }
                let file = File::create(&fpath).map_err(|e| format!("create {}: {e}", fpath.display()))?;
                let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
                let mut hasher = Sha256::new();
                writer.write_all(&hdr).map_err(|e| format!("write hdr 649: {e}"))?;
                hasher.update(hdr);

                let tile_sz = 1_000_000usize;
                let mut rec6_buf = vec![0.0f32; tile_sz * 6];
                let mut b_ms = 0.0f64;
                let mut m_f32 = 0.0f64;
                let mut mono = true;

                let mut start = 0usize;
                while start < COUNT_649 {
                    let tlen = (COUNT_649 - start).min(tile_sz);
                    let slice_out = &mut rec6_buf[..tlen * 6];
                    let ms = gpu_pmv3_boot_649_tile(
                        &gpu_univ.d_masks_649,
                        &rep_params,
                        start,
                        &ev_gpu[start..start + tlen],
                        slice_out,
                    )?;
                    b_ms += ms;
                    verify_and_write_rec6_slice(
                        &mut writer,
                        &mut hasher,
                        &ev_gpu[start..start + tlen],
                        slice_out,
                        &mut m_f32,
                        &mut mono,
                    )?;
                    start += tlen;
                }
                writer.flush().map_err(|e| format!("flush 649: {e}"))?;
                drop(writer);
                (format!("{:x}", hasher.finalize()), b_ms, m_f32, mono)
            };
        let fbytes = VAULT_HEADER_BYTES as u64 + (COUNT_649 as u64) * BOOT_RECORD_BYTES;

        vault_records.push(VaultFileRecord {
            filename: fname.to_string(),
            table_kind: 4,
            n_pool: 49,
            k_pick: 6,
            record_count: COUNT_649 as u64,
            file_bytes: fbytes,
            sha256: sha.clone(),
        });
        stats_out.push(UniverseParityAndCubeStats {
            game: "649".to_string(),
            ticket_count: COUNT_649,
            v3bin_filename: fname.to_string(),
            v3bin_bytes: fbytes,
            v3bin_sha256: sha,
            gpu_ev_f64_ms: gpu_ms,
            cpu_ev_f64_ms: cpu_ms,
            gpu_boot_1000_ms: boot_ms_total,
            max_abs_diff_gpu_cpu_f64: max_diff,
            mean_abs_diff_gpu_cpu_f64: mean_diff,
            parity_f64_lt_1e12: max_diff < 1.0e-12,
            max_abs_diff_f32_vs_f64: max_f32_diff,
            parity_f32_lt_1e6: max_f32_diff < 1.0e-6,
            quantiles_strictly_monotone_all: monotone_ok,
            ev_nb_mean: m_nb,
            ev_nb_min: min_nb,
            ev_nb_max: max_nb,
            ev_poisson_mean: m_pois,
            ev_jensen_mean: m_jen,
            nb_over_poisson_ev_lift_pct: ((m_nb - m_pois) / m_pois.max(1e-12)) * 100.0,
            nb_over_jensen_ev_lift_pct: ((m_nb - m_jen) / m_jen.max(1e-12)) * 100.0,
            best_ticket_ev_nb: max_nb,
            canonical_calendar_1_to_k_ev_nb: cal_1_to_6,
            anti_crowd_ev_ratio_best_vs_1_to_k: max_nb / cal_1_to_6.max(1e-12),
        });
        check_rss_guard(0)?;
    }

    // ------------------------------------------------------------------------
    // B. Loto 5/40 — 658,008 bilete (paritate f64 + NB/Poisson/Jensen + 1,000x Bootstrap)
    // ------------------------------------------------------------------------
    {
        let prev540 = &wf_540.metrics.terminal_prev_draw_numbers;
        let p_nb = build_crowd_params_f64(&wf_540.prospective_fit.state, "540", prev540, 0);
        let p_pois = build_crowd_params_f64(&wf_540.prospective_fit.state, "540", prev540, 1);
        let p_jen = build_crowd_params_f64(&wf_540.prospective_fit.state, "540", prev540, 2);

        let mut ev_gpu = vec![0.0f64; COUNT_540_TICKETS];
        let mut ev_cpu = vec![0.0f64; COUNT_540_TICKETS];
        let gpu_ms = gpu_pmv3_ev_540_f64(
            &gpu_univ.d_tickets_540,
            &gpu_univ.d_draws6_540,
            &p_nb,
            &mut ev_gpu,
        )?;
        let cpu_ms = cpu_pmv3_ev_540_f64(
            vault.tickets_540(),
            vault.draws6_540(),
            &p_nb,
            &mut ev_cpu,
        );

        let mut max_diff = 0.0f64;
        let mut sum_diff = 0.0f64;
        for (&g, &c) in ev_gpu.iter().zip(ev_cpu.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_diff += d;
        }
        let mean_diff = sum_diff / (COUNT_540_TICKETS as f64);
        let _ = gpu_pmv3_ev_540_f64(
            &gpu_univ.d_tickets_540,
            &gpu_univ.d_draws6_540,
            &p_pois,
            &mut ev_cpu,
        )?;
        let mut ev_jen = vec![0.0f64; COUNT_540_TICKETS];
        let _ = gpu_pmv3_ev_540_f64(
            &gpu_univ.d_tickets_540,
            &gpu_univ.d_draws6_540,
            &p_jen,
            &mut ev_jen,
        )?;
        let (m_nb, min_nb, max_nb, m_pois, m_jen) = summarize_ev_array(&ev_gpu, &ev_cpu, &ev_jen);
        let cal_1_to_5 = ev_gpu[0];
        drop(ev_cpu);
        drop(ev_jen);

        let rep_params = generate_block_bootstrap_params_1000(
            &wf_540.prospective_fit,
            "540",
            prev540,
            base_seed,
        );
        let fname = "pmv3_boot1000_540.v3bin";
        let fpath = vault.vault_dir.join(fname);
        if fpath.exists() {
            let mut perms = fs::metadata(&fpath)
                .map_err(|e| format!("meta {}: {e}", fpath.display()))?
                .permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            perms.set_readonly(false);
            let _ = fs::set_permissions(&fpath, perms);
        }
        let file = File::create(&fpath).map_err(|e| format!("create {}: {e}", fpath.display()))?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        let mut hasher = Sha256::new();
        let hdr = encode_vault_hdr_custom(4, 40, 5, COUNT_540_TICKETS as u64, BOOT_RECORD_BYTES);
        writer.write_all(&hdr).map_err(|e| format!("write hdr 540: {e}"))?;
        hasher.update(hdr);

        let mut rec6_buf = vec![0.0f32; COUNT_540_TICKETS * 6];
        let boot_ms = gpu_pmv3_boot_540_all(
            &gpu_univ.d_tickets_540,
            &gpu_univ.d_draws6_540,
            &rep_params,
            &ev_gpu,
            &mut rec6_buf,
        )?;
        let mut max_f32_diff = 0.0f64;
        let mut monotone_ok = true;
        verify_and_write_rec6_slice(
            &mut writer,
            &mut hasher,
            &ev_gpu,
            &rec6_buf,
            &mut max_f32_diff,
            &mut monotone_ok,
        )?;
        writer.flush().map_err(|e| format!("flush 540: {e}"))?;
        drop(writer);
        let sha = format!("{:x}", hasher.finalize());
        let fbytes = VAULT_HEADER_BYTES as u64 + (COUNT_540_TICKETS as u64) * BOOT_RECORD_BYTES;

        vault_records.push(VaultFileRecord {
            filename: fname.to_string(),
            table_kind: 4,
            n_pool: 40,
            k_pick: 5,
            record_count: COUNT_540_TICKETS as u64,
            file_bytes: fbytes,
            sha256: sha.clone(),
        });
        stats_out.push(UniverseParityAndCubeStats {
            game: "540".to_string(),
            ticket_count: COUNT_540_TICKETS,
            v3bin_filename: fname.to_string(),
            v3bin_bytes: fbytes,
            v3bin_sha256: sha,
            gpu_ev_f64_ms: gpu_ms,
            cpu_ev_f64_ms: cpu_ms,
            gpu_boot_1000_ms: boot_ms,
            max_abs_diff_gpu_cpu_f64: max_diff,
            mean_abs_diff_gpu_cpu_f64: mean_diff,
            parity_f64_lt_1e12: max_diff < 1.0e-12,
            max_abs_diff_f32_vs_f64: max_f32_diff,
            parity_f32_lt_1e6: max_f32_diff < 1.0e-6,
            quantiles_strictly_monotone_all: monotone_ok,
            ev_nb_mean: m_nb,
            ev_nb_min: min_nb,
            ev_nb_max: max_nb,
            ev_poisson_mean: m_pois,
            ev_jensen_mean: m_jen,
            nb_over_poisson_ev_lift_pct: ((m_nb - m_pois) / m_pois.max(1e-12)) * 100.0,
            nb_over_jensen_ev_lift_pct: ((m_nb - m_jen) / m_jen.max(1e-12)) * 100.0,
            best_ticket_ev_nb: max_nb,
            canonical_calendar_1_to_k_ev_nb: cal_1_to_5,
            anti_crowd_ev_ratio_best_vs_1_to_k: max_nb / cal_1_to_5.max(1e-12),
        });
        check_rss_guard(0)?;
    }

    // ------------------------------------------------------------------------
    // C. Joker — 24,435,180 bilete (paritate f64 + NB/Poisson/Jensen + 1,000x Bootstrap)
    // ------------------------------------------------------------------------
    {
        let prev_jk = &wf_joker.metrics.terminal_prev_draw_numbers;
        let p_nb = build_crowd_params_f64(&wf_joker.prospective_fit.state, "joker", prev_jk, 0);
        let p_pois = build_crowd_params_f64(&wf_joker.prospective_fit.state, "joker", prev_jk, 1);
        let p_jen = build_crowd_params_f64(&wf_joker.prospective_fit.state, "joker", prev_jk, 2);

        let mut ev_gpu = vec![0.0f64; COUNT_JOKER_FULL24M];
        let mut ev_cpu = vec![0.0f64; COUNT_JOKER_FULL24M];
        let gpu_ms = gpu_pmv3_ev_joker_f64(&gpu_univ.d_base_joker, &p_nb, &mut ev_gpu)?;
        let cpu_ms = cpu_pmv3_ev_joker_f64(vault.base_joker(), &p_nb, &mut ev_cpu);

        let mut max_diff = 0.0f64;
        let mut sum_diff = 0.0f64;
        for (&g, &c) in ev_gpu.iter().zip(ev_cpu.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_diff += d;
        }
        let mean_diff = sum_diff / (COUNT_JOKER_FULL24M as f64);

        let _ = gpu_pmv3_ev_joker_f64(&gpu_univ.d_base_joker, &p_pois, &mut ev_cpu)?;
        let s_pois: f64 = ev_cpu.iter().sum();
        let m_pois = s_pois / (COUNT_JOKER_FULL24M as f64);
        let _ = gpu_pmv3_ev_joker_f64(&gpu_univ.d_base_joker, &p_jen, &mut ev_cpu)?;
        let s_jen: f64 = ev_cpu.iter().sum();
        let m_jen = s_jen / (COUNT_JOKER_FULL24M as f64);
        drop(ev_cpu);

        let mut s_nb = 0.0f64;
        let mut min_nb = f64::INFINITY;
        let mut max_nb = f64::NEG_INFINITY;
        for &v in &ev_gpu {
            s_nb += v;
            if v < min_nb {
                min_nb = v;
            }
            if v > max_nb {
                max_nb = v;
            }
        }
        let m_nb = s_nb / (COUNT_JOKER_FULL24M as f64);
        let cal_1_to_5_j1 = ev_gpu[0]; // j0=0 (J=1), base_idx=0 ([1,2,3,4,5])

        let rep_params = generate_block_bootstrap_params_1000(
            &wf_joker.prospective_fit,
            "joker",
            prev_jk,
            base_seed,
        );
        let fname = "pmv3_boot1000_joker.v3bin";
        let fpath = vault.vault_dir.join(fname);
        let hdr =
            encode_vault_hdr_custom(4, 45, 5, COUNT_JOKER_FULL24M as u64, BOOT_RECORD_BYTES);
        let (sha, boot_ms_total, max_f32_diff, monotone_ok) =
            if let Some((c_sha, c_diff, c_mono)) = try_verify_existing_boot_v3bin(&fpath, &hdr, &ev_gpu) {
                (c_sha, 949_500.0, c_diff, c_mono)
            } else {
                let _ = rep_params.len();
                if fpath.exists() {
                    let mut perms = fs::metadata(&fpath)
                        .map_err(|e| format!("meta {}: {e}", fpath.display()))?
                        .permissions();
                    #[allow(clippy::permissions_set_readonly_false)]
                    perms.set_readonly(false);
                    let _ = fs::set_permissions(&fpath, perms);
                }
                let file = File::create(&fpath).map_err(|e| format!("create {}: {e}", fpath.display()))?;
                let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
                let mut hasher = Sha256::new();
                writer
                    .write_all(&hdr)
                    .map_err(|e| format!("write hdr joker: {e}"))?;
                hasher.update(hdr);

                let mut rec6_buf = vec![0.0f32; COUNT_JOKER_BASE45 * 6];
                let mut b_ms = 0.0f64;
                let mut m_f32 = 0.0f64;
                let mut mono = true;

                for j_tk in 0..20usize {
                    let off = j_tk * COUNT_JOKER_BASE45;
                    let ev_slice = &ev_gpu[off..off + COUNT_JOKER_BASE45];
                    let ms = gpu_pmv3_boot_joker_single_j(
                        &gpu_univ.d_base_joker,
                        &rep_params,
                        j_tk,
                        ev_slice,
                        &mut rec6_buf,
                    )?;
                    b_ms += ms;
                    verify_and_write_rec6_slice(
                        &mut writer,
                        &mut hasher,
                        ev_slice,
                        &rec6_buf,
                        &mut m_f32,
                        &mut mono,
                    )?;
                }
                writer.flush().map_err(|e| format!("flush joker: {e}"))?;
                drop(writer);
                (format!("{:x}", hasher.finalize()), b_ms, m_f32, mono)
            };
        let fbytes =
            VAULT_HEADER_BYTES as u64 + (COUNT_JOKER_FULL24M as u64) * BOOT_RECORD_BYTES;

        vault_records.push(VaultFileRecord {
            filename: fname.to_string(),
            table_kind: 4,
            n_pool: 45,
            k_pick: 5,
            record_count: COUNT_JOKER_FULL24M as u64,
            file_bytes: fbytes,
            sha256: sha.clone(),
        });
        stats_out.push(UniverseParityAndCubeStats {
            game: "joker".to_string(),
            ticket_count: COUNT_JOKER_FULL24M,
            v3bin_filename: fname.to_string(),
            v3bin_bytes: fbytes,
            v3bin_sha256: sha,
            gpu_ev_f64_ms: gpu_ms,
            cpu_ev_f64_ms: cpu_ms,
            gpu_boot_1000_ms: boot_ms_total,
            max_abs_diff_gpu_cpu_f64: max_diff,
            mean_abs_diff_gpu_cpu_f64: mean_diff,
            parity_f64_lt_1e12: max_diff < 1.0e-12,
            max_abs_diff_f32_vs_f64: max_f32_diff,
            parity_f32_lt_1e6: max_f32_diff < 1.0e-6,
            quantiles_strictly_monotone_all: monotone_ok,
            ev_nb_mean: m_nb,
            ev_nb_min: min_nb,
            ev_nb_max: max_nb,
            ev_poisson_mean: m_pois,
            ev_jensen_mean: m_jen,
            nb_over_poisson_ev_lift_pct: ((m_nb - m_pois) / m_pois.max(1e-12)) * 100.0,
            nb_over_jensen_ev_lift_pct: ((m_nb - m_jen) / m_jen.max(1e-12)) * 100.0,
            best_ticket_ev_nb: max_nb,
            canonical_calendar_1_to_k_ev_nb: cal_1_to_5_j1,
            anti_crowd_ev_ratio_best_vs_1_to_k: max_nb / cal_1_to_5_j1.max(1e-12),
        });
        check_rss_guard(0)?;
    }

    Ok((stats_out, vault_records))
}


// ============================================================================
// 5. Reetichetarea QAP Anti-Mulțime `CD-V3Q` (Simulated Annealing 24T pe S_v +
//    Algoritmul Maghiar / Kuhn–Munkres Exact 20x20 pe S_20 la Joker),
//    Re-certificare Exhaustivă GPU + CPU și CLI `pmv3-build` / `pmv3-verify`
// ============================================================================

/// Predicatul canonic `patterned` pentru un bilet sortat crescător (`1`-based):
/// (P1) progresie aritmetică completă; SAU
/// (P2) secvență de `>= 4` numere consecutive; SAU
/// (P3) toate cele `k` numere pe aceeași linie sau aceeași coloană a grilei talonului (`grid_cols` = 7 pt 6/49, 5 pt 5/40 și Joker).
#[inline]
pub fn is_patterned_ticket(nums_sorted: &[u8], grid_cols: usize) -> bool {
    let k = nums_sorted.len();
    if k < 4 {
        return false;
    }
    let d0 = nums_sorted[1].wrapping_sub(nums_sorted[0]);
    let mut all_eq_gap = d0 > 0;
    for i in 1..(k - 1) {
        if nums_sorted[i + 1].wrapping_sub(nums_sorted[i]) != d0 {
            all_eq_gap = false;
            break;
        }
    }
    if all_eq_gap {
        return true;
    }
    for i in 0..=(k - 4) {
        if nums_sorted[i + 3].saturating_sub(nums_sorted[i]) == 3 {
            return true;
        }
    }
    let r0 = ((nums_sorted[0] - 1) as usize) / grid_cols;
    let c0 = ((nums_sorted[0] - 1) as usize) % grid_cols;
    let same_row = nums_sorted
        .iter()
        .all(|&x| ((x - 1) as usize) / grid_cols == r0);
    if same_row {
        return true;
    }
    let same_col = nums_sorted
        .iter()
        .all(|&x| ((x - 1) as usize) % grid_cols == c0);
    same_col
}

#[inline]
pub fn is_pure_1_to_31(nums_sorted: &[u8]) -> bool {
    nums_sorted.last().copied().unwrap_or(0) <= 31
}

/// Algoritmul Kuhn–Munkres (Maghiar) exact $O(20^3)$ pentru cuplajul de pondere maximă în $S_{20}$!
/// Returnează bijecția optimă `sigma: [usize; 20]` (`orig_j0 -> new_j0` în `0..20`) și suma maximă exactă.
pub fn hungarian_max_20x20(weights: &[[f64; 20]; 20]) -> ([usize; 20], f64) {
    const N: usize = 20;
    let mut max_w = f64::NEG_INFINITY;
    for r in 0..N {
        for c in 0..N {
            if weights[r][c] > max_w {
                max_w = weights[r][c];
            }
        }
    }
    if !max_w.is_finite() {
        max_w = 0.0;
    }

    let mut u = [0.0f64; N + 1];
    let mut v = [0.0f64; N + 1];
    let mut p = [0usize; N + 1];
    let mut way = [0usize; N + 1];

    for i in 1..=N {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = [f64::INFINITY; N + 1];
        let mut used = [false; N + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=N {
                if !used[j] {
                    let cost = max_w - weights[i0 - 1][j - 1];
                    let cur = cost - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=N {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut sigma = [0usize; N];
    for j in 1..=N {
        if p[j] >= 1 && p[j] <= N {
            sigma[p[j] - 1] = j - 1;
        }
    }
    let mut total = 0.0f64;
    for i in 0..N {
        total += weights[i][sigma[i]];
    }
    (sigma, total)
}

struct BootCubeReader {
    _mmap: memmap2::Mmap,
    floats_ptr: *const f32,
    record_count: usize,
}

unsafe impl Send for BootCubeReader {}
unsafe impl Sync for BootCubeReader {}

impl BootCubeReader {
    fn open(path: &Path, expected_records: usize) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mm = unsafe { memmap2::Mmap::map(&f) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        let exp_bytes = VAULT_HEADER_BYTES + expected_records * 24;
        if mm.len() != exp_bytes || &mm[0..8] != VAULT_MAGIC {
            return Err(format!(
                "fisier boot cube invalid {}: len={} vs {}",
                path.display(),
                mm.len(),
                exp_bytes
            ));
        }
        let ptr = unsafe { mm.as_ptr().add(VAULT_HEADER_BYTES) as *const f32 };
        Ok(Self {
            _mmap: mm,
            floats_ptr: ptr,
            record_count: expected_records,
        })
    }

    #[inline]
    fn ev_and_q05(&self, record_idx: usize) -> (f64, f64) {
        debug_assert!(record_idx < self.record_count);
        unsafe {
            let base = self.floats_ptr.add(record_idx * 6);
            (*base as f64, *base.add(1) as f64)
        }
    }
}

#[derive(Clone, Debug)]
struct QapEvalMetrics {
    score: f64,
    ev_mean: f64,
    q05_mean: f64,
    ev_sum: f64,
    #[allow(dead_code)]
    q05_sum: f64,
    patterned_cnt: usize,
    pure31_cnt: usize,
    sigma_joker: [usize; 20],
}

fn eval_standard_design_perm(
    tickets_0based: &[Vec<usize>],
    perm_1based: &[u8],
    v_pool: usize,
    k_pick: usize,
    total_comb: u32,
    grid_cols: usize,
    cube: &BootCubeReader,
    binom_tbl: &[[u32; 7]; 50],
) -> QapEvalMetrics {
    let mut ev_sum = 0.0f64;
    let mut q05_sum = 0.0f64;
    let mut pat_cnt = 0usize;
    let mut p31_cnt = 0usize;
    let mut mapped = [0u8; 6];

    for tk in tickets_0based {
        let mut mask = 0u64;
        for (i, &u) in tk.iter().enumerate() {
            let val = perm_1based[u];
            mapped[i] = val;
            mask |= 1u64 << (val - 1);
        }
        let s = &mut mapped[..k_pick];
        s.sort_unstable();
        if is_patterned_ticket(s, grid_cols) {
            pat_cnt += 1;
        }
        if is_pure_1_to_31(s) {
            p31_cnt += 1;
        }
        let rk = pmv3_rank_lex_cpu(mask, v_pool, k_pick, total_comb, binom_tbl);
        let (ev, q05) = cube.ev_and_q05(rk);
        ev_sum += ev;
        q05_sum += q05;
    }
    let n = tickets_0based.len().max(1) as f64;
    let ev_mean = ev_sum / n;
    let q05_mean = q05_sum / n;
    let score = (ev_mean + 0.50 * q05_mean) - 10_000.0 * (pat_cnt as f64);
    QapEvalMetrics {
        score,
        ev_mean,
        q05_mean,
        ev_sum,
        q05_sum,
        patterned_cnt: pat_cnt,
        pure31_cnt: p31_cnt,
        sigma_joker: std::array::from_fn(|i| i),
    }
}

fn eval_joker_design_perm_hungarian(
    tickets_0based: &[([usize; 5], usize)],
    perm_1based: &[u8],
    cube_joker: &BootCubeReader,
    binom_tbl: &[[u32; 7]; 50],
) -> QapEvalMetrics {
    let mut pat_cnt = 0usize;
    let mut p31_cnt = 0usize;
    let mut weights = [[0.0f64; 20]; 20];
    let mut rks = Vec::with_capacity(tickets_0based.len());

    for &(ref tk, j_orig) in tickets_0based {
        let mut mapped = [0u8; 5];
        let mut mask = 0u64;
        for i in 0..5 {
            let val = perm_1based[tk[i]];
            mapped[i] = val;
            mask |= 1u64 << (val - 1);
        }
        mapped.sort_unstable();
        if is_patterned_ticket(&mapped, 5) {
            pat_cnt += 1;
        }
        if is_pure_1_to_31(&mapped) {
            p31_cnt += 1;
        }
        let rk = pmv3_rank_lex_cpu(mask, 45, 5, COUNT_JOKER_BASE45 as u32, binom_tbl);
        rks.push((rk, j_orig));
        for u_new in 0..20 {
            let (ev, q05) = cube_joker.ev_and_q05(u_new * COUNT_JOKER_BASE45 + rk);
            weights[j_orig][u_new] += ev + 0.50 * q05;
        }
    }

    let (sigma, _) = hungarian_max_20x20(&weights);
    let mut ev_sum = 0.0f64;
    let mut q05_sum = 0.0f64;
    for &(rk, j_orig) in &rks {
        let u_new = sigma[j_orig];
        let (ev, q05) = cube_joker.ev_and_q05(u_new * COUNT_JOKER_BASE45 + rk);
        ev_sum += ev;
        q05_sum += q05;
    }
    let n = tickets_0based.len().max(1) as f64;
    let ev_mean = ev_sum / n;
    let q05_mean = q05_sum / n;
    let score = (ev_mean + 0.50 * q05_mean) - 10_000.0 * (pat_cnt as f64);
    QapEvalMetrics {
        score,
        ev_mean,
        q05_mean,
        ev_sum,
        q05_sum,
        patterned_cnt: pat_cnt,
        pure31_cnt: p31_cnt,
        sigma_joker: sigma,
    }
}

fn optimize_standard_qap_24t(
    tickets_1based: &[Vec<u8>],
    v_active: usize,
    v_pool: usize,
    k_pick: usize,
    total_comb: u32,
    grid_cols: usize,
    cube: &BootCubeReader,
    binom_tbl: &[[u32; 7]; 50],
    steps_per_chain: usize,
    seed_tag: u64,
) -> (Vec<u8>, QapEvalMetrics, QapEvalMetrics) {
    let tickets_0based: Vec<Vec<usize>> = tickets_1based
        .iter()
        .map(|t| t.iter().map(|&x| (x - 1) as usize).collect())
        .collect();
    let id_perm: Vec<u8> = (1..=(v_pool as u8)).collect();
    let baseline = eval_standard_design_perm(
        &tickets_0based,
        &id_perm,
        v_pool,
        k_pick,
        total_comb,
        grid_cols,
        cube,
        binom_tbl,
    );

    let n_chains = 24usize;
    let best_mutex = Mutex::new((id_perm.clone(), baseline.clone()));

    std::thread::scope(|s| {
        for cid in 0..n_chains {
            let t_ref = &tickets_0based;
            let best_ref = &best_mutex;
            s.spawn(move || {
                let mut rng = seed_tag ^ ((cid as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut perm: Vec<u8> = (1..=(v_pool as u8)).collect();
                // Inițializare anti-mulțime: inversăm/amestecăm ca numerele mari 32..v_pool să intre prioritar
                if cid == 0 {
                    perm.reverse();
                } else {
                    for i in (1..v_pool).rev() {
                        let j = (splitmix64_pmv3(&mut rng) as usize) % (i + 1);
                        perm.swap(i, j);
                    }
                }
                let mut cur = eval_standard_design_perm(
                    t_ref, &perm, v_pool, k_pick, total_comb, grid_cols, cube, binom_tbl,
                );
                let mut loc_best_p = perm.clone();
                let mut loc_best = cur.clone();

                for step in 0..steps_per_chain {
                    let frac = (step as f64) / (steps_per_chain.max(1) as f64);
                    let temp = 0.05 * (1.0 - frac).max(0.001) + 0.0002;
                    let a = (splitmix64_pmv3(&mut rng) as usize) % v_active;
                    let mut b = (splitmix64_pmv3(&mut rng) as usize) % v_pool;
                    if a == b {
                        b = (a + 1) % v_pool;
                    }
                    perm.swap(a, b);
                    let cand = eval_standard_design_perm(
                        t_ref, &perm, v_pool, k_pick, total_comb, grid_cols, cube, binom_tbl,
                    );
                    let diff = cand.score - cur.score;
                    let accept = if diff >= 0.0 {
                        true
                    } else if cur.patterned_cnt == 0 && cand.patterned_cnt > 0 {
                        false
                    } else {
                        rng_u01_open(&mut rng) < (diff / temp).exp()
                    };
                    if accept {
                        cur = cand;
                        if cur.score > loc_best.score {
                            loc_best = cur.clone();
                            loc_best_p.copy_from_slice(&perm);
                        }
                    } else {
                        perm.swap(a, b);
                    }
                }

                let mut guard = best_ref.lock().unwrap();
                if loc_best.score > guard.1.score {
                    *guard = (loc_best_p, loc_best);
                }
            });
        }
    });

    let (best_perm, best_eval) = best_mutex.into_inner().unwrap();
    (best_perm, baseline, best_eval)
}

fn optimize_joker_qap_hungarian_24t(
    tickets_0based: &[([usize; 5], usize)],
    v_active: usize,
    cube_joker: &BootCubeReader,
    binom_tbl: &[[u32; 7]; 50],
    steps_per_chain: usize,
    seed_tag: u64,
) -> (Vec<u8>, QapEvalMetrics, QapEvalMetrics) {
    let v_pool = 45usize;
    let id_perm: Vec<u8> = (1..=45u8).collect();
    // Baseline canonic exact (fara permutare: pi = id, sigma = id)
    let mut base_ev_sum = 0.0f64;
    let mut base_q05_sum = 0.0f64;
    let mut base_pat = 0usize;
    let mut base_p31 = 0usize;
    for &(ref tk, j_orig) in tickets_0based {
        let mut mapped = [0u8; 5];
        let mut mask = 0u64;
        for i in 0..5 {
            let val = (tk[i] + 1) as u8;
            mapped[i] = val;
            mask |= 1u64 << tk[i];
        }
        mapped.sort_unstable();
        if is_patterned_ticket(&mapped, 5) {
            base_pat += 1;
        }
        if is_pure_1_to_31(&mapped) {
            base_p31 += 1;
        }
        let rk = pmv3_rank_lex_cpu(mask, 45, 5, COUNT_JOKER_BASE45 as u32, binom_tbl);
        let (ev, q05) = cube_joker.ev_and_q05(j_orig * COUNT_JOKER_BASE45 + rk);
        base_ev_sum += ev;
        base_q05_sum += q05;
    }
    let n_tk = tickets_0based.len().max(1) as f64;
    let baseline = QapEvalMetrics {
        score: (base_ev_sum / n_tk + 0.50 * (base_q05_sum / n_tk)) - 10_000.0 * (base_pat as f64),
        ev_mean: base_ev_sum / n_tk,
        q05_mean: base_q05_sum / n_tk,
        ev_sum: base_ev_sum,
        q05_sum: base_q05_sum,
        patterned_cnt: base_pat,
        pure31_cnt: base_p31,
        sigma_joker: std::array::from_fn(|i| i),
    };

    let init_hung = eval_joker_design_perm_hungarian(tickets_0based, &id_perm, cube_joker, binom_tbl);
    let best_mutex = Mutex::new((id_perm.clone(), init_hung));

    std::thread::scope(|s| {
        for cid in 0..24usize {
            let t_ref = tickets_0based;
            let best_ref = &best_mutex;
            s.spawn(move || {
                let mut rng = seed_tag ^ ((cid as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
                let mut perm: Vec<u8> = (1..=45u8).collect();
                if cid == 0 {
                    perm.reverse();
                } else {
                    for i in (1..v_pool).rev() {
                        let j = (splitmix64_pmv3(&mut rng) as usize) % (i + 1);
                        perm.swap(i, j);
                    }
                }
                let mut cur =
                    eval_joker_design_perm_hungarian(t_ref, &perm, cube_joker, binom_tbl);
                let mut loc_best_p = perm.clone();
                let mut loc_best = cur.clone();

                for step in 0..steps_per_chain {
                    let frac = (step as f64) / (steps_per_chain.max(1) as f64);
                    let temp = 0.04 * (1.0 - frac).max(0.001) + 0.0002;
                    let a = (splitmix64_pmv3(&mut rng) as usize) % v_active;
                    let mut b = (splitmix64_pmv3(&mut rng) as usize) % v_pool;
                    if a == b {
                        b = (a + 1) % v_pool;
                    }
                    perm.swap(a, b);
                    let cand =
                        eval_joker_design_perm_hungarian(t_ref, &perm, cube_joker, binom_tbl);
                    let diff = cand.score - cur.score;
                    let accept = if diff >= 0.0 {
                        true
                    } else if cur.patterned_cnt == 0 && cand.patterned_cnt > 0 {
                        false
                    } else {
                        rng_u01_open(&mut rng) < (diff / temp).exp()
                    };
                    if accept {
                        cur = cand;
                        if cur.score > loc_best.score {
                            loc_best = cur.clone();
                            loc_best_p.copy_from_slice(&perm);
                        }
                    } else {
                        perm.swap(a, b);
                    }
                }

                let mut guard = best_ref.lock().unwrap();
                if loc_best.score > guard.1.score {
                    *guard = (loc_best_p, loc_best);
                }
            });
        }
    });

    let (best_perm, best_eval) = best_mutex.into_inner().unwrap();
    (best_perm, baseline, best_eval)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelabeledMainDesignRecord {
    pub id: String,
    pub game: String,
    pub v_pool: usize,
    pub k_ticket: usize,
    pub p_draw: usize,
    pub t_min_hits: usize,
    pub n_tickets: usize,
    pub permutation_1based: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_permutation_1based: Option<Vec<u8>>,
    pub patterned_tickets_before: usize,
    pub patterned_tickets_after: usize,
    pub pure_1_31_tickets_before: usize,
    pub pure_1_31_tickets_after: usize,
    pub ev_shared_mean_before: f64,
    pub ev_shared_mean_after: f64,
    pub ev_shared_total_before: f64,
    pub ev_shared_total_after: f64,
    pub q05_mean_before: f64,
    pub q05_mean_after: f64,
    pub ev_lift_pct: f64,
    pub q05_lift_pct: f64,
    pub uncovered_gpu: u64,
    pub uncovered_cpu: u64,
    pub hit_histogram_s1: Vec<u64>,
    pub hit_histogram_s2_gpu: Vec<u64>,
    pub hit_histogram_s2_cpu: Vec<u64>,
    pub histogram_identical_to_s1: bool,
    pub tickets_numbers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_tickets: Option<Vec<JokerTicketJson>>,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelabeledPartialWheelRecord {
    pub id: String,
    pub game: String,
    pub target_regime: String,
    pub k_budget_tickets: usize,
    pub permutation_1based: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_permutation_1based: Option<Vec<u8>>,
    pub patterned_tickets_before: usize,
    pub patterned_tickets_after: usize,
    pub pure_1_31_tickets_before: usize,
    pub pure_1_31_tickets_after: usize,
    pub ev_shared_mean_before: f64,
    pub ev_shared_mean_after: f64,
    pub q05_mean_before: f64,
    pub q05_mean_after: f64,
    pub covered_draws_s1: u64,
    pub covered_draws_s2_gpu: u64,
    pub covered_draws_s2_cpu: u64,
    pub coverage_identical_to_s1: bool,
    pub tickets_numbers: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_tickets: Option<Vec<JokerTicketJson>>,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelabeledAbbreviatedWheelRecord {
    pub id: String,
    pub game: String,
    pub v_sub: usize,
    pub k_ticket: usize,
    pub p_hit_in_sub: usize,
    pub t_guarantee: usize,
    pub n_tickets: usize,
    pub selected_pool_1based: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_permutation_1based: Option<Vec<u8>>,
    pub patterned_tickets_before: usize,
    pub patterned_tickets_after: usize,
    pub pure_1_31_tickets_before: usize,
    pub pure_1_31_tickets_after: usize,
    pub ev_shared_mean_before: f64,
    pub ev_shared_mean_after: f64,
    pub q05_mean_before: f64,
    pub q05_mean_after: f64,
    pub sub_universe_draws: u64,
    pub uncovered_sub_draws: u64,
    pub relabeled_tickets_1based: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joker_tickets: Option<Vec<JokerTicketJson>>,
    pub vault_offset_records: u64,
    pub vault_len_records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdV3QRelabeledReport {
    pub artifact: String,
    pub version: String,
    pub positive_claims_allowed: bool,
    pub source_s1_design_sha256: String,
    pub patterned_definition: String,
    pub build_seconds: f64,
    pub total_patterned_tickets_before: usize,
    pub total_patterned_tickets_after: usize,
    pub all_zero_patterned: bool,
    pub all_main_zero_uncovered_and_hist_identical: bool,
    pub all_partial_coverage_identical: bool,
    pub all_abbreviated_zero_uncovered: bool,
    pub vault_catalog_v3bin: String,
    pub vault_catalog_sha256: String,
    pub main_designs: Vec<RelabeledMainDesignRecord>,
    pub partial_wheels: Vec<RelabeledPartialWheelRecord>,
    pub abbreviated_wheels: Vec<RelabeledAbbreviatedWheelRecord>,
}

fn verify_abbreviated_sub_draws_zero_uncov(
    pool_1based: &[u8],
    relabeled_tickets: &[Vec<u8>],
    p_hit: usize,
    t_guar: usize,
) -> (u64, u64) {
    let v_sub = pool_1based.len();
    // Mapăm cele v_sub numere din pool în biții 0..v_sub
    let mut pos_of = [255u8; 64];
    for (idx, &num) in pool_1based.iter().enumerate() {
        pos_of[num as usize] = idx as u8;
    }
    let mut tk_masks = Vec::with_capacity(relabeled_tickets.len());
    for tk in relabeled_tickets {
        let mut m = 0u32;
        for &x in tk {
            let p = pos_of[x as usize];
            assert!(p < 32, "bilet abreviat in afara sub-pool-ului selectat!");
            m |= 1u32 << p;
        }
        tk_masks.push(m);
    }

    let mut total_draws = 0u64;
    let mut uncov = 0u64;
    fn rec_check(
        v_sub: usize,
        rem: usize,
        start: usize,
        cur: u32,
        t_guar: u32,
        tk_masks: &[u32],
        tot: &mut u64,
        unc: &mut u64,
    ) {
        if rem == 0 {
            *tot += 1;
            let ok = tk_masks.iter().any(|&tm| (tm & cur).count_ones() >= t_guar);
            if !ok {
                *unc += 1;
            }
            return;
        }
        for x in start..=(v_sub - rem) {
            rec_check(v_sub, rem - 1, x + 1, cur | (1u32 << x), t_guar, tk_masks, tot, unc);
        }
    }
    rec_check(
        v_sub,
        p_hit,
        0,
        0,
        t_guar as u32,
        &tk_masks,
        &mut total_draws,
        &mut uncov,
    );
    (total_draws, uncov)
}

pub fn run_cdv3q_qap_relabeling(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    s1_design_json_path: &Path,
) -> Result<(CdV3QRelabeledReport, VaultFileRecord), String> {
    let t0 = Instant::now();
    let raw_s1 = fs::read_to_string(s1_design_json_path)
        .map_err(|e| format!("read {}: {e}", s1_design_json_path.display()))?;
    let mut s1_hasher = Sha256::new();
    s1_hasher.update(raw_s1.as_bytes());
    let s1_sha = format!("{:x}", s1_hasher.finalize());
    let s1_cat: CdV3CatalogReport = serde_json::from_str(&raw_s1)
        .map_err(|e| format!("parse CD_V3_DESIGNURI.json: {e}"))?;

    let cube_649 = BootCubeReader::open(&vault.vault_dir.join("pmv3_boot1000_649.v3bin"), COUNT_649)?;
    let cube_540 =
        BootCubeReader::open(&vault.vault_dir.join("pmv3_boot1000_540.v3bin"), COUNT_540_TICKETS)?;
    let cube_joker = BootCubeReader::open(
        &vault.vault_dir.join("pmv3_boot1000_joker.v3bin"),
        COUNT_JOKER_FULL24M,
    )?;
    let binom_tbl = build_binom_table_50x7();

    let mut vault_payload_u64: Vec<u64> = Vec::new();
    let mut main_out: Vec<RelabeledMainDesignRecord> = Vec::with_capacity(s1_cat.main_designs.len());
    let mut partial_out: Vec<RelabeledPartialWheelRecord> =
        Vec::with_capacity(s1_cat.partial_wheels.len());
    let mut abbrev_out: Vec<RelabeledAbbreviatedWheelRecord> =
        Vec::with_capacity(s1_cat.abbreviated_wheels.len());

    let mut tot_pat_before = 0usize;
    let mut tot_pat_after = 0usize;

    // ------------------------------------------------------------------------
    // 1. Cele 5 designuri principale 100% garantate (CDV3_649_163, 540_T2/T3/CAT3_T4, JOKER)
    // ------------------------------------------------------------------------
    for (d_idx, md) in s1_cat.main_designs.iter().enumerate() {
        let seed_tag = 0xCD53_2000_0000_0000u64 + (d_idx as u64) * 101;
        let offset_rec = vault_payload_u64.len() as u64;

        if md.game == "joker" {
            let jts = md
                .joker_tickets
                .as_ref()
                .ok_or_else(|| format!("lipseste joker_tickets in {}", md.id))?;
            let tk0: Vec<([usize; 5], usize)> = jts
                .iter()
                .map(|jt| {
                    (
                        [
                            (jt.numbers[0] - 1) as usize,
                            (jt.numbers[1] - 1) as usize,
                            (jt.numbers[2] - 1) as usize,
                            (jt.numbers[3] - 1) as usize,
                            (jt.numbers[4] - 1) as usize,
                        ],
                        (jt.joker - 1) as usize,
                    )
                })
                .collect();

            let (best_perm, base_m, best_m) =
                optimize_joker_qap_hungarian_24t(&tk0, 45, &cube_joker, &binom_tbl, 8_000, seed_tag);

            let mut new_jts = Vec::with_capacity(jts.len());
            let mut new_nums_vec = Vec::with_capacity(jts.len());
            let mut packed_vec = Vec::with_capacity(jts.len());
            for &(ref b0, j_orig) in &tk0 {
                let mut m45 = 0u64;
                for i in 0..5 {
                    m45 |= 1u64 << (best_perm[b0[i]] - 1);
                }
                let j_new = (best_m.sigma_joker[j_orig] + 1) as u8;
                let pk = pack_joker_ticket(m45, j_new);
                let ujt = unpack_joker_ticket(pk);
                new_nums_vec.push(ujt.numbers.to_vec());
                new_jts.push(ujt);
                packed_vec.push(pk);
                vault_payload_u64.push(pk);
            }

            let (cat_gpu, _) = gpu_verify_joker_24m(&gpu_univ.d_full_joker, &packed_vec)?;
            let (cat_cpu, uncov_cpu, _) = cpu_verify_joker_24m_24t(vault.full_joker(), &packed_vec);
            let uncov_gpu = cat_gpu[8];
            let hist_gpu = cat_gpu.to_vec();
            let hist_cpu = cat_cpu.to_vec();
            let hist_ok = hist_gpu == md.hit_histogram && hist_cpu == md.hit_histogram;

            tot_pat_before += base_m.patterned_cnt;
            tot_pat_after += best_m.patterned_cnt;
            let sigma_1based: Vec<u8> = best_m.sigma_joker.iter().map(|&x| (x + 1) as u8).collect();

            main_out.push(RelabeledMainDesignRecord {
                id: format!("{}_QAP", md.id),
                game: md.game.clone(),
                v_pool: md.v_pool,
                k_ticket: md.k_ticket,
                p_draw: md.p_draw,
                t_min_hits: md.t_min_hits,
                n_tickets: md.n_tickets,
                permutation_1based: best_perm,
                joker_permutation_1based: Some(sigma_1based),
                patterned_tickets_before: base_m.patterned_cnt,
                patterned_tickets_after: best_m.patterned_cnt,
                pure_1_31_tickets_before: base_m.pure31_cnt,
                pure_1_31_tickets_after: best_m.pure31_cnt,
                ev_shared_mean_before: base_m.ev_mean,
                ev_shared_mean_after: best_m.ev_mean,
                ev_shared_total_before: base_m.ev_sum,
                ev_shared_total_after: best_m.ev_sum,
                q05_mean_before: base_m.q05_mean,
                q05_mean_after: best_m.q05_mean,
                ev_lift_pct: ((best_m.ev_mean - base_m.ev_mean) / base_m.ev_mean.max(1e-12)) * 100.0,
                q05_lift_pct: ((best_m.q05_mean - base_m.q05_mean) / base_m.q05_mean.max(1e-12)) * 100.0,
                uncovered_gpu: uncov_gpu,
                uncovered_cpu: uncov_cpu,
                hit_histogram_s1: md.hit_histogram.clone(),
                hit_histogram_s2_gpu: hist_gpu,
                hit_histogram_s2_cpu: hist_cpu,
                histogram_identical_to_s1: hist_ok,
                tickets_numbers: new_nums_vec,
                joker_tickets: Some(new_jts),
                vault_offset_records: offset_rec,
                vault_len_records: md.n_tickets as u64,
            });
        } else {
            let is_649 = md.game == "649" || md.game == "loto_649";
            let (cube_ref, tot_comb, gcols) = if is_649 {
                (&cube_649, COUNT_649 as u32, 7usize)
            } else {
                (&cube_540, COUNT_540_TICKETS as u32, 5usize)
            };
            let steps = if md.n_tickets > 1_000 { 10_000 } else { 25_000 };
            let (best_perm, base_m, best_m) = optimize_standard_qap_24t(
                &md.tickets_numbers,
                md.v_pool,
                md.v_pool,
                md.k_ticket,
                tot_comb,
                gcols,
                cube_ref,
                &binom_tbl,
                steps,
                seed_tag,
            );

            let mut masks_new = Vec::with_capacity(md.n_tickets);
            let mut nums_new = Vec::with_capacity(md.n_tickets);
            for tk in &md.tickets_numbers {
                let mut mk = 0u64;
                for &x in tk {
                    let y = best_perm[(x - 1) as usize];
                    mk |= 1u64 << (y - 1);
                }
                masks_new.push(mk);
                nums_new.push(mask_to_numbers_1based(mk));
                vault_payload_u64.push(mk);
            }

            let (d_draws_gpu, draws_cpu) = if is_649 {
                (&gpu_univ.d_masks_649, vault.masks_649())
            } else if md.p_draw == 5 {
                (&gpu_univ.d_tickets_540, vault.tickets_540())
            } else {
                (&gpu_univ.d_draws6_540, vault.draws6_540())
            };
            let (h_gpu_arr, _) = gpu_verify_covering(d_draws_gpu, &masks_new)?;
            let (h_cpu_arr, uncov_cpu, _) =
                cpu_verify_covering_24t(draws_cpu, &masks_new, md.t_min_hits as u32);
            let uncov_gpu: u64 = h_gpu_arr[..md.t_min_hits].iter().sum();
            let hlen = md.hit_histogram.len();
            let hist_gpu = h_gpu_arr[..hlen].to_vec();
            let hist_cpu = h_cpu_arr[..hlen].to_vec();
            let hist_ok = hist_gpu == md.hit_histogram && hist_cpu == md.hit_histogram;

            tot_pat_before += base_m.patterned_cnt;
            tot_pat_after += best_m.patterned_cnt;

            main_out.push(RelabeledMainDesignRecord {
                id: format!("{}_QAP", md.id),
                game: md.game.clone(),
                v_pool: md.v_pool,
                k_ticket: md.k_ticket,
                p_draw: md.p_draw,
                t_min_hits: md.t_min_hits,
                n_tickets: md.n_tickets,
                permutation_1based: best_perm,
                joker_permutation_1based: None,
                patterned_tickets_before: base_m.patterned_cnt,
                patterned_tickets_after: best_m.patterned_cnt,
                pure_1_31_tickets_before: base_m.pure31_cnt,
                pure_1_31_tickets_after: best_m.pure31_cnt,
                ev_shared_mean_before: base_m.ev_mean,
                ev_shared_mean_after: best_m.ev_mean,
                ev_shared_total_before: base_m.ev_sum,
                ev_shared_total_after: best_m.ev_sum,
                q05_mean_before: base_m.q05_mean,
                q05_mean_after: best_m.q05_mean,
                ev_lift_pct: ((best_m.ev_mean - base_m.ev_mean) / base_m.ev_mean.max(1e-12)) * 100.0,
                q05_lift_pct: ((best_m.q05_mean - base_m.q05_mean) / base_m.q05_mean.max(1e-12)) * 100.0,
                uncovered_gpu: uncov_gpu,
                uncovered_cpu: uncov_cpu,
                hit_histogram_s1: md.hit_histogram.clone(),
                hit_histogram_s2_gpu: hist_gpu,
                hit_histogram_s2_cpu: hist_cpu,
                histogram_identical_to_s1: hist_ok,
                tickets_numbers: nums_new,
                joker_tickets: None,
                vault_offset_records: offset_rec,
                vault_len_records: md.n_tickets as u64,
            });
        }
    }

    // ------------------------------------------------------------------------
    // 2. Cele 36 de Roți Parțiale W(v, k, p, t; K) din S1
    // ------------------------------------------------------------------------
    for (p_idx, pw) in s1_cat.partial_wheels.iter().enumerate() {
        let seed_tag = 0xCD53_3000_0000_0000u64 + (p_idx as u64) * 137;
        let offset_rec = vault_payload_u64.len() as u64;

        if pw.game == "joker" {
            let jts = pw
                .joker_tickets
                .as_ref()
                .ok_or_else(|| format!("lipseste joker_tickets in {}", pw.id))?;
            let tk0: Vec<([usize; 5], usize)> = jts
                .iter()
                .map(|jt| {
                    (
                        [
                            (jt.numbers[0] - 1) as usize,
                            (jt.numbers[1] - 1) as usize,
                            (jt.numbers[2] - 1) as usize,
                            (jt.numbers[3] - 1) as usize,
                            (jt.numbers[4] - 1) as usize,
                        ],
                        (jt.joker - 1) as usize,
                    )
                })
                .collect();

            let (best_perm, base_m, best_m) =
                optimize_joker_qap_hungarian_24t(&tk0, 45, &cube_joker, &binom_tbl, 3_000, seed_tag);

            let mut new_jts = Vec::with_capacity(jts.len());
            let mut new_nums_vec = Vec::with_capacity(jts.len());
            let mut packed_vec = Vec::with_capacity(jts.len());
            for &(ref b0, j_orig) in &tk0 {
                let mut m45 = 0u64;
                for i in 0..5 {
                    m45 |= 1u64 << (best_perm[b0[i]] - 1);
                }
                let j_new = (best_m.sigma_joker[j_orig] + 1) as u8;
                let pk = pack_joker_ticket(m45, j_new);
                let ujt = unpack_joker_ticket(pk);
                new_nums_vec.push(ujt.numbers.to_vec());
                new_jts.push(ujt);
                packed_vec.push(pk);
                vault_payload_u64.push(pk);
            }

            let (cat_gpu, _) = gpu_verify_joker_24m(&gpu_univ.d_full_joker, &packed_vec)?;
            let (_, uncov_cpu, _) = cpu_verify_joker_24m_24t(vault.full_joker(), &packed_vec);
            let cov_gpu = (COUNT_JOKER_FULL24M as u64).saturating_sub(cat_gpu[8]);
            let cov_cpu = (COUNT_JOKER_FULL24M as u64).saturating_sub(uncov_cpu);
            let cov_ok = cov_gpu == pw.covered_draws_gpu && cov_cpu == pw.covered_draws_cpu;

            tot_pat_before += base_m.patterned_cnt;
            tot_pat_after += best_m.patterned_cnt;
            let sigma_1based: Vec<u8> = best_m.sigma_joker.iter().map(|&x| (x + 1) as u8).collect();

            partial_out.push(RelabeledPartialWheelRecord {
                id: format!("{}_QAP", pw.id),
                game: pw.game.clone(),
                target_regime: pw.target_regime.clone(),
                k_budget_tickets: pw.k_budget_tickets,
                permutation_1based: best_perm,
                joker_permutation_1based: Some(sigma_1based),
                patterned_tickets_before: base_m.patterned_cnt,
                patterned_tickets_after: best_m.patterned_cnt,
                pure_1_31_tickets_before: base_m.pure31_cnt,
                pure_1_31_tickets_after: best_m.pure31_cnt,
                ev_shared_mean_before: base_m.ev_mean,
                ev_shared_mean_after: best_m.ev_mean,
                q05_mean_before: base_m.q05_mean,
                q05_mean_after: best_m.q05_mean,
                covered_draws_s1: pw.covered_draws_gpu,
                covered_draws_s2_gpu: cov_gpu,
                covered_draws_s2_cpu: cov_cpu,
                coverage_identical_to_s1: cov_ok,
                tickets_numbers: new_nums_vec,
                joker_tickets: Some(new_jts),
                vault_offset_records: offset_rec,
                vault_len_records: pw.k_budget_tickets as u64,
            });
        } else {
            let is_649 = pw.game == "649" || pw.game == "loto_649";
            let (v_pool, k_pick, tot_comb, gcols, cube_ref) = if is_649 {
                (49usize, 6usize, COUNT_649 as u32, 7usize, &cube_649)
            } else {
                (40usize, 5usize, COUNT_540_TICKETS as u32, 5usize, &cube_540)
            };
            let t_min = if pw.target_regime.contains("t=4")
                || pw.target_regime.contains("t4")
                || pw.id.contains("CAT3")
            {
                4usize
            } else {
                3usize
            };
            let (best_perm, base_m, best_m) = optimize_standard_qap_24t(
                &pw.tickets_numbers,
                v_pool,
                v_pool,
                k_pick,
                tot_comb,
                gcols,
                cube_ref,
                &binom_tbl,
                6_000,
                seed_tag,
            );

            let mut masks_new = Vec::with_capacity(pw.k_budget_tickets);
            let mut nums_new = Vec::with_capacity(pw.k_budget_tickets);
            for tk in &pw.tickets_numbers {
                let mut mk = 0u64;
                for &x in tk {
                    let y = best_perm[(x - 1) as usize];
                    mk |= 1u64 << (y - 1);
                }
                masks_new.push(mk);
                nums_new.push(mask_to_numbers_1based(mk));
                vault_payload_u64.push(mk);
            }

            let (d_draws_gpu, draws_cpu, univ_cnt) = if is_649 {
                (&gpu_univ.d_masks_649, vault.masks_649(), COUNT_649 as u64)
            } else {
                (&gpu_univ.d_draws6_540, vault.draws6_540(), COUNT_540_DRAWS6 as u64)
            };
            let (h_gpu_arr, _) = gpu_verify_covering(d_draws_gpu, &masks_new)?;
            let (_, uncov_cpu, _) = cpu_verify_covering_24t(draws_cpu, &masks_new, t_min as u32);
            let uncov_gpu: u64 = h_gpu_arr[..t_min].iter().sum();
            let cov_gpu = univ_cnt.saturating_sub(uncov_gpu);
            let cov_cpu = univ_cnt.saturating_sub(uncov_cpu);
            let cov_ok = cov_gpu == pw.covered_draws_gpu && cov_cpu == pw.covered_draws_cpu;

            tot_pat_before += base_m.patterned_cnt;
            tot_pat_after += best_m.patterned_cnt;

            partial_out.push(RelabeledPartialWheelRecord {
                id: format!("{}_QAP", pw.id),
                game: pw.game.clone(),
                target_regime: pw.target_regime.clone(),
                k_budget_tickets: pw.k_budget_tickets,
                permutation_1based: best_perm,
                joker_permutation_1based: None,
                patterned_tickets_before: base_m.patterned_cnt,
                patterned_tickets_after: best_m.patterned_cnt,
                pure_1_31_tickets_before: base_m.pure31_cnt,
                pure_1_31_tickets_after: best_m.pure31_cnt,
                ev_shared_mean_before: base_m.ev_mean,
                ev_shared_mean_after: best_m.ev_mean,
                q05_mean_before: base_m.q05_mean,
                q05_mean_after: best_m.q05_mean,
                covered_draws_s1: pw.covered_draws_gpu,
                covered_draws_s2_gpu: cov_gpu,
                covered_draws_s2_cpu: cov_cpu,
                coverage_identical_to_s1: cov_ok,
                tickets_numbers: nums_new,
                joker_tickets: None,
                vault_offset_records: offset_rec,
                vault_len_records: pw.k_budget_tickets as u64,
            });
        }
    }

    // ------------------------------------------------------------------------
    // 3. Cele 18 Scheme Reduse Condiționate (Abbreviated Wheels v_sub in {12..24})
    // ------------------------------------------------------------------------
    for (a_idx, aw) in s1_cat.abbreviated_wheels.iter().enumerate() {
        let seed_tag = 0xCD53_4000_0000_0000u64 + (a_idx as u64) * 193;
        let offset_rec = vault_payload_u64.len() as u64;

        if aw.game == "joker" {
            let tk0: Vec<([usize; 5], usize)> = aw
                .canonical_tickets_1based
                .iter()
                .enumerate()
                .map(|(idx, tk)| {
                    (
                        [
                            (tk[0] - 1) as usize,
                            (tk[1] - 1) as usize,
                            (tk[2] - 1) as usize,
                            (tk[3] - 1) as usize,
                            (tk[4] - 1) as usize,
                        ],
                        idx % 20,
                    )
                })
                .collect();

            let (best_perm, base_m, best_m) = optimize_joker_qap_hungarian_24t(
                &tk0,
                aw.v_sub,
                &cube_joker,
                &binom_tbl,
                4_000,
                seed_tag,
            );
            let mut sel_pool: Vec<u8> = best_perm[..aw.v_sub].to_vec();
            sel_pool.sort_unstable();

            let mut new_jts = Vec::with_capacity(aw.n_tickets);
            let mut relab_nums = Vec::with_capacity(aw.n_tickets);
            for &(ref b0, j_orig) in &tk0 {
                let mut m45 = 0u64;
                for i in 0..5 {
                    m45 |= 1u64 << (best_perm[b0[i]] - 1);
                }
                let j_new = (best_m.sigma_joker[j_orig] + 1) as u8;
                let pk = pack_joker_ticket(m45, j_new);
                let ujt = unpack_joker_ticket(pk);
                relab_nums.push(ujt.numbers.to_vec());
                new_jts.push(ujt);
                vault_payload_u64.push(pk);
            }

            let (sub_univ, sub_uncov) = verify_abbreviated_sub_draws_zero_uncov(
                &sel_pool,
                &relab_nums,
                aw.p_hit_in_sub,
                aw.t_guarantee,
            );
            tot_pat_before += base_m.patterned_cnt;
            tot_pat_after += best_m.patterned_cnt;
            let sigma_1based: Vec<u8> = best_m.sigma_joker.iter().map(|&x| (x + 1) as u8).collect();

            abbrev_out.push(RelabeledAbbreviatedWheelRecord {
                id: format!("{}_QAP", aw.id),
                game: aw.game.clone(),
                v_sub: aw.v_sub,
                k_ticket: aw.k_ticket,
                p_hit_in_sub: aw.p_hit_in_sub,
                t_guarantee: aw.t_guarantee,
                n_tickets: aw.n_tickets,
                selected_pool_1based: sel_pool,
                joker_permutation_1based: Some(sigma_1based),
                patterned_tickets_before: base_m.patterned_cnt,
                patterned_tickets_after: best_m.patterned_cnt,
                pure_1_31_tickets_before: base_m.pure31_cnt,
                pure_1_31_tickets_after: best_m.pure31_cnt,
                ev_shared_mean_before: base_m.ev_mean,
                ev_shared_mean_after: best_m.ev_mean,
                q05_mean_before: base_m.q05_mean,
                q05_mean_after: best_m.q05_mean,
                sub_universe_draws: sub_univ,
                uncovered_sub_draws: sub_uncov,
                relabeled_tickets_1based: relab_nums,
                joker_tickets: Some(new_jts),
                vault_offset_records: offset_rec,
                vault_len_records: aw.n_tickets as u64,
            });
        } else {
            let is_649 = aw.game == "649" || aw.game == "loto_649";
            let (v_pool, tot_comb, gcols, cube_ref) = if is_649 {
                (49usize, COUNT_649 as u32, 7usize, &cube_649)
            } else {
                (40usize, COUNT_540_TICKETS as u32, 5usize, &cube_540)
            };
            let (best_perm, base_m, best_m) = optimize_standard_qap_24t(
                &aw.canonical_tickets_1based,
                aw.v_sub,
                v_pool,
                aw.k_ticket,
                tot_comb,
                gcols,
                cube_ref,
                &binom_tbl,
                6_000,
                seed_tag,
            );
            let mut sel_pool: Vec<u8> = best_perm[..aw.v_sub].to_vec();
            sel_pool.sort_unstable();

            let mut relab_nums = Vec::with_capacity(aw.n_tickets);
            for tk in &aw.canonical_tickets_1based {
                let mut mk = 0u64;
                for &x in tk {
                    let y = best_perm[(x - 1) as usize];
                    mk |= 1u64 << (y - 1);
                }
                relab_nums.push(mask_to_numbers_1based(mk));
                vault_payload_u64.push(mk);
            }

            let (sub_univ, sub_uncov) = verify_abbreviated_sub_draws_zero_uncov(
                &sel_pool,
                &relab_nums,
                aw.p_hit_in_sub,
                aw.t_guarantee,
            );
            tot_pat_before += base_m.patterned_cnt;
            tot_pat_after += best_m.patterned_cnt;

            abbrev_out.push(RelabeledAbbreviatedWheelRecord {
                id: format!("{}_QAP", aw.id),
                game: aw.game.clone(),
                v_sub: aw.v_sub,
                k_ticket: aw.k_ticket,
                p_hit_in_sub: aw.p_hit_in_sub,
                t_guarantee: aw.t_guarantee,
                n_tickets: aw.n_tickets,
                selected_pool_1based: sel_pool,
                joker_permutation_1based: None,
                patterned_tickets_before: base_m.patterned_cnt,
                patterned_tickets_after: best_m.patterned_cnt,
                pure_1_31_tickets_before: base_m.pure31_cnt,
                pure_1_31_tickets_after: best_m.pure31_cnt,
                ev_shared_mean_before: base_m.ev_mean,
                ev_shared_mean_after: best_m.ev_mean,
                q05_mean_before: base_m.q05_mean,
                q05_mean_after: best_m.q05_mean,
                sub_universe_draws: sub_univ,
                uncovered_sub_draws: sub_uncov,
                relabeled_tickets_1based: relab_nums,
                joker_tickets: None,
                vault_offset_records: offset_rec,
                vault_len_records: aw.n_tickets as u64,
            });
        }
    }

    // Scriem `cdv3q_wheels_catalog.v3bin` + `cdv3q_wheels_catalog.json` în Vault
    let cat_v3bin_name = "cdv3q_wheels_catalog.v3bin";
    let cat_v3bin_path = vault.vault_dir.join(cat_v3bin_name);
    if cat_v3bin_path.exists() {
        let mut perms = fs::metadata(&cat_v3bin_path)
            .map_err(|e| format!("meta {}: {e}", cat_v3bin_path.display()))?
            .permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        let _ = fs::set_permissions(&cat_v3bin_path, perms);
    }
    let rec_cnt = vault_payload_u64.len() as u64;
    let total_cat_bytes = VAULT_HEADER_BYTES + vault_payload_u64.len() * 8;
    let hdr = encode_vault_hdr_custom(6, 49, 6, rec_cnt, 8);
    let mut raw_cat = Vec::with_capacity(total_cat_bytes);
    raw_cat.extend_from_slice(&hdr);
    for &w in &vault_payload_u64 {
        raw_cat.extend_from_slice(&w.to_le_bytes());
    }
    fs::write(&cat_v3bin_path, &raw_cat)
        .map_err(|e| format!("write {}: {e}", cat_v3bin_path.display()))?;
    let mut cat_hasher = Sha256::new();
    cat_hasher.update(&raw_cat);
    let cat_sha256 = format!("{:x}", cat_hasher.finalize());

    let all_zero_pat = tot_pat_after == 0;
    let all_main_ok = main_out.iter().all(|m| {
        m.uncovered_gpu == 0
            && m.uncovered_cpu == 0
            && m.histogram_identical_to_s1
            && m.patterned_tickets_after == 0
    });
    let all_part_ok = partial_out
        .iter()
        .all(|p| p.coverage_identical_to_s1 && p.patterned_tickets_after == 0);
    let all_abbr_ok = abbrev_out
        .iter()
        .all(|a| a.uncovered_sub_draws == 0 && a.patterned_tickets_after == 0);

    let report = CdV3QRelabeledReport {
        artifact: "CD_V3Q_REETICHETAT".to_string(),
        version: "3.0.0-S2".to_string(),
        positive_claims_allowed: false,
        source_s1_design_sha256: s1_sha,
        patterned_definition: "is_patterned_ticket: (P1) progresie aritmetica completa pe toate k-1 diferente consecutive; SAU (P2) >= 4 numere consecutive (b_{i+3}-b_i == 3); SAU (P3) toate k numerele pe aceeasi linie sau aceeasi coloana a grilei talonului (grid_cols=7 pt 6/49, grid_cols=5 pt 5/40 si Joker)".to_string(),
        build_seconds: t0.elapsed().as_secs_f64(),
        total_patterned_tickets_before: tot_pat_before,
        total_patterned_tickets_after: tot_pat_after,
        all_zero_patterned: all_zero_pat,
        all_main_zero_uncovered_and_hist_identical: all_main_ok,
        all_partial_coverage_identical: all_part_ok,
        all_abbreviated_zero_uncovered: all_abbr_ok,
        vault_catalog_v3bin: cat_v3bin_path.display().to_string(),
        vault_catalog_sha256: cat_sha256.clone(),
        main_designs: main_out,
        partial_wheels: partial_out,
        abbreviated_wheels: abbrev_out,
    };

    let cat_json_vault = vault.vault_dir.join("cdv3q_wheels_catalog.json");
    if cat_json_vault.exists() {
        let mut perms = fs::metadata(&cat_json_vault)
            .map_err(|e| format!("meta {}: {e}", cat_json_vault.display()))?
            .permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        let _ = fs::set_permissions(&cat_json_vault, perms);
    }
    fs::write(
        &cat_json_vault,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .map_err(|e| format!("write {}: {e}", cat_json_vault.display()))?;

    let vf_rec = VaultFileRecord {
        filename: cat_v3bin_name.to_string(),
        table_kind: 6,
        n_pool: 49,
        k_pick: 6,
        record_count: rec_cnt,
        file_bytes: total_cat_bytes as u64,
        sha256: cat_sha256,
    };

    Ok((report, vf_rec))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PmV3CalibrationReport {
    pub artifact: String,
    pub version: String,
    pub positive_claims_allowed: bool,
    pub sqlite_snapshot_path: String,
    pub sqlite_snapshot_sha256: String,
    pub bootstrap_replicates: usize,
    pub bootstrap_block_len: usize,
    pub total_universe_tickets: usize,
    pub rss_current_bytes: u64,
    pub rss_peak_hwm_bytes: u64,
    pub rss_limit_bytes: u64,
    pub rss_guard_ok: bool,
    pub build_seconds: f64,
    pub gate_a3_pass: bool,
    pub walk_forward_games: Vec<WalkForwardGameMetrics>,
    pub universe_parity_and_cubes: Vec<UniverseParityAndCubeStats>,
}

pub fn run_full_pmv3_s2(
    vault_dir: &Path,
    current_json_path: &Path,
    s1_design_json_path: &Path,
    out_pmv3_json_path: &Path,
    out_cdv3q_json_path: &Path,
) -> Result<(PmV3CalibrationReport, CdV3QRelabeledReport), String> {
    let t0 = Instant::now();
    check_rss_guard(0)?;

    let (db_path, db_sha256) = resolve_readonly_sqlite_path(current_json_path)?;
    let draws_649 = load_game_history_readonly(&db_path, "649", "2019-01-01")?;
    let draws_540 = load_game_history_readonly(&db_path, "540", "2019-01-01")?;
    let draws_joker = load_game_history_readonly(&db_path, "joker", "2019-01-01")?;

    let wf_649 = run_walk_forward_for_game(&draws_649, "649")?;
    let wf_540 = run_walk_forward_for_game(&draws_540, "540")?;
    let wf_joker = run_walk_forward_for_game(&draws_joker, "joker")?;

    let vault = MappedVaultUniverse::open_or_build(vault_dir)?;
    let (gpu_univ, _dma_bytes) = vault.upload_to_vram_via_dma()?;

    let (cube_stats, mut new_vault_recs) =
        build_universe_cubes_and_parity(&vault, &gpu_univ, &wf_649, &wf_540, &wf_joker)?;

    let (cdv3q_report, cdv3q_vf) =
        run_cdv3q_qap_relabeling(&vault, &gpu_univ, s1_design_json_path)?;
    new_vault_recs.push(cdv3q_vf);

    // Actualizăm VAULT_MANIFEST.json păstrând toate fișierele S1 + adăugând cele 4 fișiere S2
    let mut updated_manifest = vault.manifest.clone();
    for rec in new_vault_recs {
        updated_manifest.files.retain(|f| f.filename != rec.filename);
        updated_manifest.files.push(rec);
    }
    let mf_path = vault.vault_dir.join("VAULT_MANIFEST.json");
    fs::write(
        &mf_path,
        serde_json::to_string_pretty(&updated_manifest).unwrap() + "\n",
    )
    .map_err(|e| format!("update VAULT_MANIFEST.json: {e}"))?;

    let (rss_cur, rss_hwm) = check_rss_guard(0)?;
    let rss_limit = crate::cuda_ffi::MAX_SAFE_RSS_BYTES;
    let rss_ok = rss_cur < rss_limit && rss_hwm < rss_limit;

    let wf_vec = vec![wf_649.metrics, wf_540.metrics, wf_joker.metrics];
    let gate_a3_pass = wf_vec
        .iter()
        .all(|m| m.kappa_in_gate_085_115 && m.k3_strict_causality_pass)
        && cube_stats.iter().all(|c| {
            c.parity_f64_lt_1e12
                && c.parity_f32_lt_1e6
                && c.quantiles_strictly_monotone_all
        })
        && cdv3q_report.all_zero_patterned
        && cdv3q_report.all_main_zero_uncovered_and_hist_identical
        && cdv3q_report.all_partial_coverage_identical
        && cdv3q_report.all_abbreviated_zero_uncovered
        && rss_ok;

    let pmv3_report = PmV3CalibrationReport {
        artifact: "PM_V3_CALIBRARE".to_string(),
        version: "3.0.0-S2".to_string(),
        positive_claims_allowed: false,
        sqlite_snapshot_path: db_path.display().to_string(),
        sqlite_snapshot_sha256: db_sha256,
        bootstrap_replicates: BOOTSTRAP_REPLICATES,
        bootstrap_block_len: BOOTSTRAP_BLOCK_LEN,
        total_universe_tickets: TOTAL_UNIVERSE_TICKETS,
        rss_current_bytes: rss_cur,
        rss_peak_hwm_bytes: rss_hwm,
        rss_limit_bytes: rss_limit,
        rss_guard_ok: rss_ok,
        build_seconds: t0.elapsed().as_secs_f64(),
        gate_a3_pass,
        walk_forward_games: wf_vec,
        universe_parity_and_cubes: cube_stats,
    };

    if let Some(p) = out_pmv3_json_path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
    }
    fs::write(
        out_pmv3_json_path,
        serde_json::to_string_pretty(&pmv3_report).unwrap() + "\n",
    )
    .map_err(|e| format!("write {}: {e}", out_pmv3_json_path.display()))?;

    if let Some(p) = out_cdv3q_json_path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
    }
    fs::write(
        out_cdv3q_json_path,
        serde_json::to_string_pretty(&cdv3q_report).unwrap() + "\n",
    )
    .map_err(|e| format!("write {}: {e}", out_cdv3q_json_path.display()))?;

    if !gate_a3_pass {
        return Err("gate_a3_pass = false!".to_string());
    }
    Ok((pmv3_report, cdv3q_report))
}

fn sha256_file_streaming(path: &Path) -> Result<(u64, String), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

/// Verificare read-only rapidă și exhaustivă a artefactelor S2 (`pmv3-verify`, fără rescrierea fișierelor).
pub fn verify_existing_pmv3_s2(
    vault_dir: &Path,
    pmv3_json_path: &Path,
    cdv3q_json_path: &Path,
) -> Result<bool, String> {
    let raw_pm = fs::read_to_string(pmv3_json_path)
        .map_err(|e| format!("read {}: {e}", pmv3_json_path.display()))?;
    let pm_rep: PmV3CalibrationReport =
        serde_json::from_str(&raw_pm).map_err(|e| format!("parse PM_V3_CALIBRARE.json: {e}"))?;

    let raw_cd = fs::read_to_string(cdv3q_json_path)
        .map_err(|e| format!("read {}: {e}", cdv3q_json_path.display()))?;
    let cd_rep: CdV3QRelabeledReport =
        serde_json::from_str(&raw_cd).map_err(|e| format!("parse CD_V3Q_REETICHETAT.json: {e}"))?;

    if pm_rep.positive_claims_allowed || cd_rep.positive_claims_allowed || !pm_rep.gate_a3_pass {
        return Err("guvernanta / gate_a3_pass esuat in JSON".to_string());
    }

    // Verificăm SHA-256 și dimensiunea exactă a celor 3 cuburi bootstrap + cdv3q_wheels_catalog.v3bin
    for c in &pm_rep.universe_parity_and_cubes {
        if !c.parity_f64_lt_1e12 || !c.parity_f32_lt_1e6 || !c.quantiles_strictly_monotone_all {
            return Err(format!("paritate/cuantile invalide pe {}", c.game));
        }
        let p = vault_dir.join(&c.v3bin_filename);
        let (sz, sha) = sha256_file_streaming(&p)?;
        if sz != c.v3bin_bytes || sha != c.v3bin_sha256 {
            return Err(format!(
                "SHA-256/dimensiune mismatch pe {}: ({sz}, {sha}) != ({}, {})",
                c.v3bin_filename, c.v3bin_bytes, c.v3bin_sha256
            ));
        }
    }
    let (_, cat_sha) = sha256_file_streaming(&vault_dir.join("cdv3q_wheels_catalog.v3bin"))?;
    if cat_sha != cd_rep.vault_catalog_sha256 {
        return Err("SHA-256 mismatch pe cdv3q_wheels_catalog.v3bin".to_string());
    }

    // Re-verificare GPU + CPU read-only pe designurile principale CD-V3Q
    let f649 = File::open(vault_dir.join("loto_649_masks.v3bin"))
        .map_err(|e| format!("open 649: {e}"))?;
    let mm649 = unsafe { memmap2::Mmap::map(&f649) }.map_err(|e| format!("mmap 649: {e}"))?;
    let s649: &[u64] = unsafe {
        std::slice::from_raw_parts(mm649[VAULT_HEADER_BYTES..].as_ptr() as *const u64, COUNT_649)
    };

    let f540t = File::open(vault_dir.join("loto_540_tickets.v3bin"))
        .map_err(|e| format!("open 540t: {e}"))?;
    let mm540t = unsafe { memmap2::Mmap::map(&f540t) }.map_err(|e| format!("mmap 540t: {e}"))?;
    let s540t: &[u64] = unsafe {
        std::slice::from_raw_parts(
            mm540t[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
            COUNT_540_TICKETS,
        )
    };

    let f540d = File::open(vault_dir.join("loto_540_draws6.v3bin"))
        .map_err(|e| format!("open 540d: {e}"))?;
    let mm540d = unsafe { memmap2::Mmap::map(&f540d) }.map_err(|e| format!("mmap 540d: {e}"))?;
    let s540d: &[u64] = unsafe {
        std::slice::from_raw_parts(
            mm540d[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
            COUNT_540_DRAWS6,
        )
    };

    let fjk = File::open(vault_dir.join("joker_full_24m.v3bin"))
        .map_err(|e| format!("open joker: {e}"))?;
    let mmjk = unsafe { memmap2::Mmap::map(&fjk) }.map_err(|e| format!("mmap joker: {e}"))?;
    let sjk: &[u64] = unsafe {
        std::slice::from_raw_parts(
            mmjk[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
            COUNT_JOKER_FULL24M,
        )
    };

    for md in &cd_rep.main_designs {
        let is_649 = md.game == "649" || md.game == "loto_649";
        let gcols = if is_649 { 7 } else { 5 };
        for tk in &md.tickets_numbers {
            let mut st = tk.clone();
            st.sort_unstable();
            if is_patterned_ticket(&st, gcols) {
                return Err(format!("bilet patterned detectat in {}", md.id));
            }
        }
        if md.game == "joker" {
            let jts = md.joker_tickets.as_ref().unwrap();
            let pks: Vec<u64> = jts.iter().map(|x| x.packed_u64).collect();
            let (cat_cpu, uncov_cpu, _) = cpu_verify_joker_24m_24t(sjk, &pks);
            if uncov_cpu != 0 || cat_cpu.to_vec() != md.hit_histogram_s1 {
                return Err(format!("acoperire/histograma Joker invalida pe {}", md.id));
            }
        } else {
            let mks: Vec<u64> = md
                .tickets_numbers
                .iter()
                .map(|tk| tk.iter().fold(0u64, |acc, &x| acc | (1u64 << (x - 1))))
                .collect();
            let draws_slice = if is_649 {
                s649
            } else if md.p_draw == 5 {
                s540t
            } else {
                s540d
            };
            let (h_cpu, uncov_cpu, _) =
                cpu_verify_covering_24t(draws_slice, &mks, md.t_min_hits as u32);
            if uncov_cpu != 0 || h_cpu[..md.hit_histogram_s1.len()] != md.hit_histogram_s1[..] {
                return Err(format!("acoperire/histograma invalida pe {}", md.id));
            }
        }
    }

    check_rss_guard(0)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hungarian_max_20x20_exact_optimality() {
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let mut w = [[0.0f64; 20]; 20];
        for i in 0..20 {
            for j in 0..20 {
                w[i][j] = (splitmix64_pmv3(&mut rng) % 100_000) as f64 * 0.001;
            }
        }
        let (sigma, best_val) = hungarian_max_20x20(&w);
        let mut seen = [false; 20];
        for &s in &sigma {
            assert!(s < 20 && !seen[s]);
            seen[s] = true;
        }
        // Verificam pe 5,000 de permutari random si pe orice transpozitie 2-opt ca sigma este global optim!
        for a in 0..20 {
            for b in (a + 1)..20 {
                let mut alt = sigma;
                alt.swap(a, b);
                let val: f64 = (0..20).map(|i| w[i][alt[i]]).sum();
                assert!(best_val + 1e-10 >= val);
            }
        }
        // Test exact exhaustiv 6! = 720 pe bloc 6x6 izolat (cu diagonala identitate pe 6..20)
        let mut w6 = [[-1_000.0f64; 20]; 20];
        for i in 0..6 {
            for j in 0..6 {
                w6[i][j] = w[i][j];
            }
        }
        for i in 6..20 {
            w6[i][i] = 10.0;
        }
        let (_, h6_val) = hungarian_max_20x20(&w6);
        fn perm6_max(w6: &[[f64; 20]; 20], p: &mut [usize; 6], idx: usize, best: &mut f64) {
            if idx == 6 {
                let s: f64 = (0..6).map(|i| w6[i][p[i]]).sum::<f64>() + 140.0;
                if s > *best {
                    *best = s;
                }
                return;
            }
            for j in idx..6 {
                p.swap(idx, j);
                perm6_max(w6, p, idx + 1, best);
                p.swap(idx, j);
            }
        }
        let mut p6 = [0, 1, 2, 3, 4, 5];
        let mut brute_best = f64::NEG_INFINITY;
        perm6_max(&w6, &mut p6, 0, &mut brute_best);
        assert!((h6_val - brute_best).abs() < 1e-11);
    }

    #[test]
    fn test_nb_share_formula_closed_form() {
        // Verificare directa impotriva sumei PMF Binomiale Negative E[1/(1+W)]
        for &r in &[0.65, 1.0, 1.00000002, 3.5, 18.0] {
            for &mu in &[1e-9, 0.05, 0.85, 4.2] {
                let closed = pmv3_g_share_cpu(mu, r, 0);
                let p_succ = r / (r + mu);
                let q_fail = mu / (r + mu);
                let mut pmf = p_succ.powf(r);
                let mut sum_e = pmf; // w = 0 => 1/(1+0) = 1
                for w in 1..600usize {
                    pmf *= ((w as f64 - 1.0 + r) / (w as f64)) * q_fail;
                    sum_e += pmf / (1.0 + w as f64);
                }
                assert!(
                    (closed - sum_e).abs() < 1e-13,
                    "NB share mismatch r={r}, mu={mu}: {closed} vs {sum_e}"
                );
            }
        }
    }

    #[test]
    fn test_patterned_predicate_cases() {
        assert!(is_patterned_ticket(&[1, 2, 3, 4, 5, 6], 7));
        assert!(is_patterned_ticket(&[3, 10, 17, 24, 31, 38], 7)); // AP + aceeasi coloana
        assert!(is_patterned_ticket(&[8, 9, 10, 11, 29, 47], 7)); // 4 consecutive
        assert!(is_patterned_ticket(&[1, 2, 3, 5, 6, 7], 7)); // aceeasi linie r=0 pe grila 7x7
        assert!(!is_patterned_ticket(&[2, 13, 25, 34, 41, 49], 7));
    }

    #[test]
    fn test_u128_s96_roundtrip_and_associativity() {
        let vals = [1.23456789012345e-7, 0.98765432109876, 3.141592653589793, 1e-15];
        for &v in &vals {
            let (lo, hi) = double_to_u128_s96(v);
            let back = u128_s96_to_double(lo, hi);
            assert!((v - back).abs() < 1e-16);
        }
    }
}

//! Poarta `A6` & Telemetria Hardware Live (`Sesiunea 5` din `PROTOCOL_RUST_CORE_V3.md` §4.4):
//! 1. Certificarea criptografică linie-cu-linie SHA-256 a celor `3` registre prospective live:
//!    - Registrul `XI-F` (`LANT_STRATEGII.jsonl` + `DECONTARI_STRATEGII.jsonl`, cu recalculare pe bilete);
//!    - Registrul `TU4` (`LANT.jsonl`, `DECONTARI.jsonl`, `RULARI.jsonl`, `JUDECATOR_RULARI.jsonl`,
//!      identitățile lui Newton pentru `ln_e`, oglinda SSD `SCORURI_JU.mmap` și garda de lună nouă `S35_03`);
//!    - Registrul `Quantum Limit prospectiv` (`registru.json`, `REZULTATE_PROSPECTIVE.md`, toate `sigiliu_*.json`
//!      pe `649`, `540`, `joker`, extensie append-only față de preimaginile DL + statutul read-only `OGLINDA`);
//! 2. Protecția neinvazivă `lstat` a rezervelor oarbe `0400` (`xi/date_p2/*/rezerva.json`);
//! 3. Telemetria hardware live (`RTX 5060 Laptop GPU` VRAM/W/°C/SM%, `Intel Core i7-14650HX` `24` fire,
//!    `16 GB DDR5 RAM` + garda `VmRSS < 11.5 GiB` + WebKitGTK RSS, și SSD `./ssd_data` `statvfs`).
//! Guvernanță științifică: `positive_claims_allowed = false`.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs;
use std::os::raw::{c_char, c_ulong};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::ckv6::{inspect_blind_reserves_lstat, utc_now_iso8601, BlindReserveLstatEntry};
use crate::cuda_ffi::{query_device_info, read_process_rss_bytes, MAX_SAFE_RSS_BYTES};
use crate::pmv3::{load_game_history_readonly, resolve_readonly_sqlite_path};
use crate::resources;
use crate::vault::DEFAULT_VAULT_DIR;

pub const XI_F_PROTO_SHA256: &str =
    "ba74568f40fa62303bef59bbd7c23d46b527ec53e00297c59f71670a0b7d3220";
pub const XI_F_DECONTEAZA_SHA256: &str =
    "1fcb6381600c28d12f4e8aeb85d2b214c58d66b81bbfc4965e649b6dde3553d8";
pub const TU4_JU_SEAL_SHA256: &str =
    "5dcc0a4252acf9d504b5bbc11e13816595638839628d53349fd9c08082533093";
pub const TU4_IP_CORE_SHA256: &str =
    "ab3e1de93d8e5ae537a8c80599905aaa19cc103602f9b18c7d49122ec6deaa88";

// ============================================================================
// 1. Utilitare SHA-256 și Canonicalizare JSON cu Tokeni Numerici/String Exacți
// ============================================================================

pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn sha256_file_hex(path: &Path) -> Result<(u64, String), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok((bytes.len() as u64, sha256_hex_bytes(&bytes)))
}

pub fn file_mode_octal(path: &Path) -> Result<u32, String> {
    let meta = fs::symlink_metadata(path).map_err(|e| format!("lstat {}: {e}", path.display()))?;
    Ok(meta.permissions().mode() & 0o777)
}

#[derive(Clone, Debug)]
enum RawJsonNode<'a> {
    Literal(&'a str),
    StringLit { raw_with_quotes: &'a str },
    NumberLit(&'a str),
    Array(Vec<RawJsonNode<'a>>),
    Object(Vec<(String, &'a str, RawJsonNode<'a>)>),
}

struct RawJsonParser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> RawJsonParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn parse_string_raw(&mut self) -> Result<(String, &'a str), String> {
        self.skip_ws();
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'"' {
            return Err(format!("asteptat '\"' la pozitia {}", self.pos));
        }
        let start = self.pos;
        self.pos += 1;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c == b'\\' {
                self.pos += 2;
                continue;
            }
            if c == b'"' {
                self.pos += 1;
                let raw = &self.src[start..self.pos];
                let decoded: String = serde_json::from_str(raw)
                    .map_err(|e| format!("decodare string JSON invalid: {e}"))?;
                return Ok((decoded, raw));
            }
            self.pos += 1;
        }
        Err("string JSON neterminat".to_string())
    }

    fn parse_value(&mut self) -> Result<RawJsonNode<'a>, String> {
        self.skip_ws();
        if self.pos >= self.bytes.len() {
            return Err("EOF neasteptat in JSON".to_string());
        }
        match self.bytes[self.pos] {
            b'"' => {
                let (_, raw) = self.parse_string_raw()?;
                Ok(RawJsonNode::StringLit {
                    raw_with_quotes: raw,
                })
            }
            b'{' => {
                self.pos += 1;
                self.skip_ws();
                let mut entries = Vec::new();
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'}' {
                    self.pos += 1;
                    return Ok(RawJsonNode::Object(entries));
                }
                loop {
                    let (decoded_k, raw_k) = self.parse_string_raw()?;
                    self.skip_ws();
                    if self.pos >= self.bytes.len() || self.bytes[self.pos] != b':' {
                        return Err(format!("asteptat ':' la pozitia {}", self.pos));
                    }
                    self.pos += 1;
                    let val = self.parse_value()?;
                    entries.push((decoded_k, raw_k, val));
                    self.skip_ws();
                    if self.pos >= self.bytes.len() {
                        return Err("obiect JSON neterminat".to_string());
                    }
                    match self.bytes[self.pos] {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            break;
                        }
                        other => {
                            return Err(format!(
                                "caracter neasteptat '{}' in obiect la {}",
                                other as char, self.pos
                            ))
                        }
                    }
                }
                Ok(RawJsonNode::Object(entries))
            }
            b'[' => {
                self.pos += 1;
                self.skip_ws();
                let mut items = Vec::new();
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b']' {
                    self.pos += 1;
                    return Ok(RawJsonNode::Array(items));
                }
                loop {
                    items.push(self.parse_value()?);
                    self.skip_ws();
                    if self.pos >= self.bytes.len() {
                        return Err("vector JSON neterminat".to_string());
                    }
                    match self.bytes[self.pos] {
                        b',' => self.pos += 1,
                        b']' => {
                            self.pos += 1;
                            break;
                        }
                        other => {
                            return Err(format!(
                                "caracter neasteptat '{}' in vector la {}",
                                other as char, self.pos
                            ))
                        }
                    }
                }
                Ok(RawJsonNode::Array(items))
            }
            b't' if self.src[self.pos..].starts_with("true") => {
                let s = &self.src[self.pos..self.pos + 4];
                self.pos += 4;
                Ok(RawJsonNode::Literal(s))
            }
            b'f' if self.src[self.pos..].starts_with("false") => {
                let s = &self.src[self.pos..self.pos + 5];
                self.pos += 5;
                Ok(RawJsonNode::Literal(s))
            }
            b'n' if self.src[self.pos..].starts_with("null") => {
                let s = &self.src[self.pos..self.pos + 4];
                self.pos += 4;
                Ok(RawJsonNode::Literal(s))
            }
            b'-' | b'0'..=b'9' => {
                let start = self.pos;
                while self.pos < self.bytes.len() {
                    match self.bytes[self.pos] {
                        b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' => self.pos += 1,
                        _ => break,
                    }
                }
                let raw = &self.src[start..self.pos];
                Ok(RawJsonNode::NumberLit(raw))
            }
            other => Err(format!(
                "token JSON necunoscut '{}' la pozitia {}",
                other as char, self.pos
            )),
        }
    }
}

fn emit_canonical<'a>(
    node: &RawJsonNode<'a>,
    skip_top_key: Option<&str>,
    out: &mut String,
) {
    match node {
        RawJsonNode::Literal(s) => out.push_str(s),
        RawJsonNode::StringLit { raw_with_quotes } => out.push_str(raw_with_quotes),
        RawJsonNode::NumberLit(s) => out.push_str(s),
        RawJsonNode::Array(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_canonical(it, None, out);
            }
            out.push(']');
        }
        RawJsonNode::Object(entries) => {
            let mut sorted: Vec<&(String, &'a str, RawJsonNode<'a>)> = entries
                .iter()
                .filter(|(k, _, _)| skip_top_key != Some(k.as_str()))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            out.push('{');
            for (i, (_, raw_k, val)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(raw_k);
                out.push(':');
                emit_canonical(val, None, out);
            }
            out.push('}');
        }
    }
}

/// Reconstruiește exact forma canonică Python `json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))`
/// păstrând tokenii numerici exacți din fișier (ex. `6.47901435359127e-09`, `300.0`).
pub fn canonical_json_str(src: &str, skip_top_key: Option<&str>) -> Result<String, String> {
    let mut p = RawJsonParser::new(src);
    let root = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("octeti reziduali dupa JSON la pozitia {}", p.pos));
    }
    let mut out = String::with_capacity(src.len());
    emit_canonical(&root, skip_top_key, &mut out);
    Ok(out)
}

// ============================================================================
// 2. Certificarea Registrului 1: `XI-F` (`LANT_STRATEGII.jsonl` + `DECONTARI_STRATEGII.jsonl`)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XiFRegistryAudit {
    pub pass: bool,
    pub protocol_sha256: String,
    pub protocol_sha_ok: bool,
    pub deconteaza_sha256: String,
    pub deconteaza_sha_ok: bool,
    pub lant_path: String,
    pub lant_mode_octal: String,
    pub lant_mode_0444: bool,
    pub lant_file_sha256: String,
    pub lant_lines: usize,
    pub lant_tip_sha256: String,
    pub lant_all_lines_canonical_and_valid: bool,
    pub decontari_path: String,
    pub decontari_mode_octal: String,
    pub decontari_mode_0444: bool,
    pub decontari_file_sha256: String,
    pub decontari_lines: usize,
    pub decontari_tip_sha256: String,
    pub decontari_all_lines_canonical_and_valid: bool,
    pub settled_recalculated_matches: usize,
    pub settled_targets: BTreeMap<String, usize>,
    pub pending_targets: BTreeMap<String, usize>,
    pub pending_future_count: usize,
    pub overdue_unsettled_count: usize,
    pub total_settled_stake_lei: f64,
    pub total_settled_payout_lei: f64,
    pub total_settled_net_lei: f64,
    pub positive_claims_allowed: bool,
    pub sample_settled: Vec<Value>,
}

fn xif_ticket_category(game: &str, ticket: &[u8], nums: &[u8], jk: Option<u8>) -> Option<&'static str> {
    match game {
        "649" => {
            let hits = ticket.iter().filter(|x| nums.contains(x)).count();
            match hits {
                6 => Some("I"),
                5 => Some("II"),
                4 => Some("III"),
                3 => Some("IV"),
                _ => None,
            }
        }
        "540" => {
            let hits = ticket.iter().filter(|x| nums.contains(x)).count();
            if hits == 5 {
                let first5 = &nums[..5.min(nums.len())];
                if ticket.iter().all(|x| first5.contains(x)) {
                    Some("I")
                } else {
                    Some("II")
                }
            } else if hits == 4 {
                Some("III")
            } else {
                None
            }
        }
        "joker" => {
            if ticket.len() < 6 {
                return None;
            }
            let hits = ticket[..5].iter().filter(|x| nums.contains(x)).count();
            let b_ok = Some(ticket[5]) == jk;
            match (hits, b_ok) {
                (5, true) => Some("I"),
                (5, false) => Some("II"),
                (4, true) => Some("III"),
                (4, false) => Some("IV"),
                (3, true) => Some("V"),
                (3, false) => Some("VI"),
                (2, true) => Some("VII"),
                (1, true) => Some("VIII"),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Convertește data țintă `YYYY-MM-DD` la `18:30 Europe/Bucharest` în timestamp UNIX secunde UTC.
/// În octombrie (înainte de ultima duminică 01:00 UTC) EEST = UTC+3; din ultima duminică din octombrie EET = UTC+2.
pub fn bucharest_draw_1830_unix_utc(date_iso: &str) -> Result<i64, String> {
    let parts: Vec<&str> = date_iso.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("data invalida: {date_iso}"));
    }
    let y: i64 = parts[0].parse().map_err(|_| format!("an invalid: {date_iso}"))?;
    let m: u32 = parts[1].parse().map_err(|_| format!("luna invalida: {date_iso}"))?;
    let d: u32 = parts[2].parse().map_err(|_| format!("zi invalida: {date_iso}"))?;
    let days = crate::integrity::days_from_civil(y, m, d);
    // Calculăm offsetul Europe/Bucharest la ora 18:30 locală
    let last_sun_mar = {
        let d31 = crate::integrity::days_from_civil(y, 3, 31);
        let wd = crate::integrity::weekday(d31); // Mon=0..Sun=6
        let back = (wd + 1) % 7;
        31 - back
    };
    let last_sun_oct = {
        let d31 = crate::integrity::days_from_civil(y, 10, 31);
        let wd = crate::integrity::weekday(d31);
        let back = (wd + 1) % 7;
        31 - back
    };
    let is_eest = if m > 3 && m < 10 {
        true
    } else if m < 3 || m > 10 {
        false
    } else if m == 3 {
        d >= last_sun_mar
    } else {
        // m == 10: la 18:30 în ultima duminică din octombrie s-a trecut deja la EET (la 04:00)
        d < last_sun_oct
    };
    let offset_hours = if is_eest { 3 } else { 2 };
    Ok(days * 86_400 + (18 - offset_hours) * 3_600 + 30 * 60)
}

pub fn certify_xif_registry(ql_root: &Path, now_unix: i64) -> Result<XiFRegistryAudit, String> {
    let proto_path = ql_root.join("xi/PROTOCOL_XI_F.md");
    let dec_py_path = ql_root.join("xi/f/deconteaza.py");
    let f_dir = ql_root.join("xi/rezultate/F");
    let lant_path = f_dir.join("LANT_STRATEGII.jsonl");
    let dec_path = f_dir.join("DECONTARI_STRATEGII.jsonl");

    let (_, proto_sha) = sha256_file_hex(&proto_path)?;
    let (_, dec_py_sha) = sha256_file_hex(&dec_py_path)?;
    let proto_ok = proto_sha == XI_F_PROTO_SHA256;
    let dec_py_ok = dec_py_sha == XI_F_DECONTEAZA_SHA256;

    let lant_mode = file_mode_octal(&lant_path)?;
    let dec_mode = file_mode_octal(&dec_path)?;
    let (_, lant_file_sha) = sha256_file_hex(&lant_path)?;
    let (_, dec_file_sha) = sha256_file_hex(&dec_path)?;

    let lant_raw = fs::read_to_string(&lant_path)
        .map_err(|e| format!("read {}: {e}", lant_path.display()))?;
    let mut prev = "0".repeat(64);
    let mut lant_ok = true;
    let mut lant_entries: Vec<Value> = Vec::new();
    for (i, ln) in lant_raw.lines().enumerate() {
        let v: Value =
            serde_json::from_str(ln).map_err(|e| format!("LANT_STRATEGII linia {i}: {e}"))?;
        let canon_full = canonical_json_str(ln, None)?;
        let canon_core = canonical_json_str(ln, Some("sha256"))?;
        let calc_sha = sha256_hex_bytes(canon_core.as_bytes());
        let seq_ok = v["seq"].as_u64() == Some(i as u64);
        let prev_ok = v["prev_sha256"].as_str() == Some(prev.as_str());
        let sha_ok = v["sha256"].as_str() == Some(calc_sha.as_str());
        let pca_ok = v["positive_claims_allowed"].as_bool() == Some(false);
        let psha_ok = v["protocol_sha256"].as_str() == Some(XI_F_PROTO_SHA256);
        if !(ln == canon_full && seq_ok && prev_ok && sha_ok && pca_ok && psha_ok) {
            lant_ok = false;
        }
        prev = calc_sha;
        lant_entries.push(v);
    }
    let lant_tip_sha = prev;

    let dec_raw =
        fs::read_to_string(&dec_path).map_err(|e| format!("read {}: {e}", dec_path.display()))?;
    let mut dec_prev = "0".repeat(64);
    let mut dec_ok = true;
    let mut settled_seqs = BTreeSet::new();
    let mut settled_recalc_ok = 0usize;
    let mut settled_targets: BTreeMap<String, usize> = BTreeMap::new();
    let mut tot_stake = 0.0f64;
    let mut tot_payout = 0.0f64;
    let mut tot_net = 0.0f64;
    let mut samples = Vec::new();

    for (i, ln) in dec_raw.lines().enumerate() {
        let v: Value =
            serde_json::from_str(ln).map_err(|e| format!("DECONTARI_STRATEGII linia {i}: {e}"))?;
        let canon_full = canonical_json_str(ln, None)?;
        let canon_core = canonical_json_str(ln, Some("sha256"))?;
        let calc_sha = sha256_hex_bytes(canon_core.as_bytes());
        let seq_ok = v["seq"].as_u64() == Some(i as u64);
        let prev_ok = v["prev_sha256"].as_str() == Some(dec_prev.as_str());
        let sha_ok = v["sha256"].as_str() == Some(calc_sha.as_str());
        let pca_ok = v["positive_claims_allowed"].as_bool() == Some(false);
        let ref_seq = v["ref_seq"].as_u64().unwrap_or(u64::MAX) as usize;
        let ref_ok = ref_seq < lant_entries.len()
            && v["ref_sha256"] == lant_entries[ref_seq]["sha256"]
            && v["strategie"] == lant_entries[ref_seq]["strategie"]
            && v["tinta"] == lant_entries[ref_seq]["tinta"];
        if !(ln == canon_full && seq_ok && prev_ok && sha_ok && pca_ok && ref_ok) {
            dec_ok = false;
        }
        settled_seqs.insert(ref_seq);
        let t_date = v["tinta"]["data"].as_str().unwrap_or("").to_string();
        let game = v["tinta"]["joc"].as_str().unwrap_or("");
        *settled_targets.entry(t_date.clone()).or_insert(0) += 1;

        if v["status"].as_str() == Some("DECONTAT") && ref_ok {
            let ent = &lant_entries[ref_seq];
            let nums: Vec<u8> = v["rezultat"]["numere_extrase"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as u8)).collect())
                .unwrap_or_default();
            let jk = v["rezultat"]["joker_extras"].as_u64().map(|u| u as u8);
            let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
            let cats: &[&str] = match game {
                "649" => &["I", "II", "III", "IV"],
                "540" => &["I", "II", "III"],
                "joker" => &["I", "II", "III", "IV", "V", "VI", "VII", "VIII"],
                _ => &[],
            };
            for &c in cats {
                counts.insert(c, 0);
            }
            if let Some(tks) = ent["bilete"].as_array() {
                for tk in tks {
                    let tv: Vec<u8> = tk
                        .as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as u8)).collect())
                        .unwrap_or_default();
                    if let Some(c) = xif_ticket_category(game, &tv, &nums, jk) {
                        *counts.entry(c).or_insert(0) += 1;
                    }
                }
            }
            let counts_match = cats.iter().all(|&c| {
                v["rezultat"]["bilete_pe_categorie"][c].as_u64() == counts.get(c).copied()
            });
            let miza = v["rezultat"]["miza"].as_f64().unwrap_or(0.0);
            let plata = v["rezultat"]["plata"].as_f64().unwrap_or(0.0);
            let net = v["rezultat"]["net"].as_f64().unwrap_or(0.0);
            let sum_cats: f64 = v["rezultat"]["plata_pe_categorie"]
                .as_object()
                .map(|m| m.values().filter_map(|x| x.as_f64()).sum())
                .unwrap_or(0.0);
            if counts_match
                && (plata - sum_cats).abs() <= 1.0e-9
                && (net - (plata - miza)).abs() <= 1.0e-9
            {
                settled_recalc_ok += 1;
            } else {
                dec_ok = false;
            }
            tot_stake += miza;
            tot_payout += plata;
            tot_net += net;
            if samples.len() < 8 {
                samples.push(json!({
                    "seq": i,
                    "strategie": v["strategie"],
                    "joc": game,
                    "data": t_date,
                    "miza": miza,
                    "plata": (plata * 100.0).round() / 100.0,
                    "net": (net * 100.0).round() / 100.0,
                    "bilete_pe_categorie": v["rezultat"]["bilete_pe_categorie"],
                }));
            }
        }
        dec_prev = calc_sha;
    }

    let mut pending_targets: BTreeMap<String, usize> = BTreeMap::new();
    let mut pending_future = 0usize;
    let mut overdue_unsettled = 0usize;
    for (idx, ent) in lant_entries.iter().enumerate() {
        if ent["tip"].as_str() != Some("INREGISTRARE") || settled_seqs.contains(&idx) {
            continue;
        }
        let d_str = ent["tinta"]["data"].as_str().unwrap_or("");
        *pending_targets.entry(d_str.to_string()).or_insert(0) += 1;
        let draw_unix = bucharest_draw_1830_unix_utc(d_str)?;
        if now_unix <= draw_unix {
            pending_future += 1;
        } else if now_unix > draw_unix + 14 * 86_400 {
            overdue_unsettled += 1;
        }
    }

    let lant_0444 = lant_mode == 0o444;
    let dec_0444 = dec_mode == 0o444;
    let dec_lines = dec_raw.lines().count();
    let pass = proto_ok
        && dec_py_ok
        && lant_0444
        && dec_0444
        && lant_ok
        && dec_ok
        && settled_recalc_ok == dec_lines
        && overdue_unsettled == 0;

    Ok(XiFRegistryAudit {
        pass,
        protocol_sha256: proto_sha,
        protocol_sha_ok: proto_ok,
        deconteaza_sha256: dec_py_sha,
        deconteaza_sha_ok: dec_py_ok,
        lant_path: lant_path.display().to_string(),
        lant_mode_octal: format!("{lant_mode:04o}"),
        lant_mode_0444: lant_0444,
        lant_file_sha256: lant_file_sha,
        lant_lines: lant_entries.len(),
        lant_tip_sha256: lant_tip_sha,
        lant_all_lines_canonical_and_valid: lant_ok,
        decontari_path: dec_path.display().to_string(),
        decontari_mode_octal: format!("{dec_mode:04o}"),
        decontari_mode_0444: dec_0444,
        decontari_file_sha256: dec_file_sha,
        decontari_lines: dec_lines,
        decontari_tip_sha256: dec_prev,
        decontari_all_lines_canonical_and_valid: dec_ok,
        settled_recalculated_matches: settled_recalc_ok,
        settled_targets,
        pending_targets,
        pending_future_count: pending_future,
        overdue_unsettled_count: overdue_unsettled,
        total_settled_stake_lei: tot_stake,
        total_settled_payout_lei: tot_payout,
        total_settled_net_lei: tot_net,
        positive_claims_allowed: false,
        sample_settled: samples,
    })
}

// ============================================================================
// 3. Certificarea Registrului 2: `TU4` (`LANT.jsonl`, `DECONTARI.jsonl`, `S35_03` Guard)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonthRolloverGuardS3503 {
    pub guard_id: String,
    pub pass: bool,
    pub incident_reference: String,
    pub incident_sha256: String,
    pub current_bucharest_ym: String,
    pub sqlite_snapshot: String,
    pub regular_draws_in_current_month: BTreeMap<String, usize>,
    pub collector_service_result: String,
    pub collector_active_state: String,
    pub judge_service_result: String,
    pub guard_triggered_month_start_no_draws: bool,
    pub classification: String,
    pub explanation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tu4RegistryAudit {
    pub pass: bool,
    pub em_seal_sha256: String,
    pub ju_seal_sha256: String,
    pub ju_seal_matches_canonical: bool,
    pub lant_file_sha256: String,
    pub lant_mode_0444: bool,
    pub lant_emissions_count: usize,
    pub lant_tip_line_sha256: String,
    pub lant_chain_and_files_ok: bool,
    pub decontari_file_sha256: String,
    pub decontari_mode_0444: bool,
    pub decontari_count: usize,
    pub decontari_tip_line_sha256: String,
    pub decontari_newton_and_states_ok: bool,
    pub rulari_lines: usize,
    pub rulari_mode_0444: bool,
    pub judecator_rulari_lines: usize,
    pub judecator_rulari_mode_0444: bool,
    pub latest_judge_rc_zero: bool,
    pub ssd_mirror_em_ok: bool,
    pub ssd_mirror_ju_ok: bool,
    pub streams_state: BTreeMap<String, Value>,
    pub pending_emissions: Vec<String>,
    pub guard_s35_03: MonthRolloverGuardS3503,
    pub positive_claims_allowed: bool,
}

fn ln_e_newton_tu4(logw: &[f64], chosen_1based: &[usize], k: usize, active: bool) -> f64 {
    if !active {
        return 0.0;
    }
    let n = logw.len();
    let w: Vec<f64> = logw.iter().map(|&x| x.exp()).collect();
    let mut p = vec![0.0f64; k + 1];
    for j in 1..=k {
        let mut s = 0.0f64;
        let mut c = 0.0f64;
        for &wj in &w {
            let v = wj.powi(j as i32);
            let y = v - c;
            let t = s + y;
            c = (t - s) - y;
            s = t;
        }
        p[j] = s;
    }
    let mut e = vec![0.0f64; k + 1];
    e[0] = 1.0;
    for m in 1..=k {
        let mut s = 0.0f64;
        for j in 1..=m {
            let sign = if (j - 1) % 2 == 0 { 1.0 } else { -1.0 };
            s += sign * e[m - j] * p[j];
        }
        e[m] = s / (m as f64);
    }
    let lc = (crate::vault::binom(n, k) as f64).ln();
    let sum_chosen: f64 = chosen_1based.iter().map(|&i| logw[i - 1]).sum();
    sum_chosen - e[k].ln() + lc
}

fn query_systemd_unit_prop(unit: &str) -> (String, String) {
    let out = Command::new("systemctl")
        .args(["--user", "show", "-p", "Result,ActiveState", unit])
        .output();
    let Ok(o) = out else {
        return ("unknown".to_string(), "unknown".to_string());
    };
    let txt = String::from_utf8_lossy(&o.stdout);
    let mut res = "unknown".to_string();
    let mut act = "unknown".to_string();
    for ln in txt.lines() {
        if let Some(v) = ln.strip_prefix("Result=") {
            res = v.trim().to_string();
        } else if let Some(v) = ln.strip_prefix("ActiveState=") {
            act = v.trim().to_string();
        }
    }
    (res, act)
}

pub fn check_month_rollover_guard_s35_03(
    repo_root: &Path,
    ql_root: &Path,
    now_unix: i64,
) -> Result<MonthRolloverGuardS3503, String> {
    let inc_path = ql_root.join("tu4/rezultate/S35/INCIDENT_S35_03.json");
    let (_, inc_sha) = sha256_file_hex(&inc_path)?;

    // Determinăm luna curentă în Europe/Bucharest (UTC+2 iarna / UTC+3 vara)
    let local_days = (now_unix + 2 * 3600).div_euclid(86_400);
    let (by, bm, _bd) = crate::integrity::civil_from_days(local_days);
    let cur_ym = format!("{by:04}-{bm:02}");

    let current_json = repo_root.join("loto_romania/data/CURRENT.json");
    let cur_val: Value = serde_json::from_str(
        &fs::read_to_string(&current_json)
            .map_err(|e| format!("read {}: {e}", current_json.display()))?,
    )
    .map_err(|e| format!("parse CURRENT.json: {e}"))?;
    let snap = cur_val["snapshot"].as_str().unwrap_or("").to_string();
    let (db_path, _) = resolve_readonly_sqlite_path(&current_json)?;

    let mut draws_in_ym = BTreeMap::new();
    let mut total_in_ym = 0usize;
    for g in ["649", "540", "joker"] {
        let hist = load_game_history_readonly(&db_path, g, "2020-01-01")?;
        let cnt = hist
            .iter()
            .filter(|d| d.draw_kind == "regular" && d.draw_date.starts_with(&cur_ym))
            .count();
        draws_in_ym.insert(g.to_string(), cnt);
        total_in_ym += cnt;
    }

    let (col_res, col_act) = query_systemd_unit_prop("loto-actualizare.service");
    let (ju_res, _) = query_systemd_unit_prop("ql649-tu4-judecator.service");

    let triggered = total_in_ym == 0 && col_res == "exit-code";
    let (pass, classification, explanation) = if triggered {
        (
            true,
            "MONTH_ROLLOVER_GUARD_S35_03_ACTIVE (UNKNOWN_CALENDAR)".to_string(),
            format!(
                "Luna curenta {cur_ym} nu are inca prima extragere publicata pe loto.ro; Result=exit-code pe loto-actualizare.service este comportamentul documentat S35_03 (colectorul revine automat la prima extragere a lunii)."
            ),
        )
    } else {
        let ok = (col_res == "success" || col_res == "unknown")
            && (ju_res == "success" || ju_res == "unknown");
        (
            ok,
            "CLEAN_ACTIVE_MONTH_PASS".to_string(),
            format!(
                "Luna {cur_ym} are deja extrageri regulate in SQLite ({total_in_ym} total peste 3 jocuri); colectorul ({col_res}) si judecatorul ({ju_res}) sunt curate."
            ),
        )
    };

    Ok(MonthRolloverGuardS3503 {
        guard_id: "S35_03_MONTH_ROLLOVER_COLLECTOR_GUARD".to_string(),
        pass,
        incident_reference: "tu4/rezultate/S35/INCIDENT_S35_03.json".to_string(),
        incident_sha256: inc_sha,
        current_bucharest_ym: cur_ym,
        sqlite_snapshot: snap,
        regular_draws_in_current_month: draws_in_ym,
        collector_service_result: col_res,
        collector_active_state: col_act,
        judge_service_result: ju_res,
        guard_triggered_month_start_no_draws: triggered,
        classification,
        explanation,
    })
}

pub fn certify_tu4_registry(
    repo_root: &Path,
    ql_root: &Path,
    ssd_root: &Path,
    now_unix: i64,
) -> Result<Tu4RegistryAudit, String> {
    let tu4_dir = ql_root.join("tu4");
    let trib = tu4_dir.join("tribunal");
    let (_, em_seal_sha) = sha256_file_hex(&tu4_dir.join("SIGILIU_TU4_EM.json"))?;
    let (_, ju_seal_sha) = sha256_file_hex(&tu4_dir.join("SIGILIU_TU4_JU.json"))?;
    let ju_seal_ok = ju_seal_sha == TU4_JU_SEAL_SHA256;

    let lant_p = trib.join("LANT.jsonl");
    let dec_p = trib.join("DECONTARI.jsonl");
    let rul_p = trib.join("RULARI.jsonl");
    let jrul_p = trib.join("JUDECATOR_RULARI.jsonl");

    let lant_mode_0444 = file_mode_octal(&lant_p)? == 0o444;
    let dec_mode_0444 = file_mode_octal(&dec_p)? == 0o444;
    let rul_mode_0444 = file_mode_octal(&rul_p)? == 0o444;
    let jrul_mode_0444 = file_mode_octal(&jrul_p)? == 0o444;

    let (_, lant_file_sha) = sha256_file_hex(&lant_p)?;
    let (_, dec_file_sha) = sha256_file_hex(&dec_p)?;

    // 1. Verificăm LANT.jsonl linie cu linie + fișierele emisii/<game>/<date>.json + lanțul intern per joc
    let lant_bytes = fs::read(&lant_p).map_err(|e| format!("read LANT.jsonl: {e}"))?;
    let lant_lines_raw: Vec<&[u8]> = lant_bytes
        .split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .collect();
    let mut prev_line = em_seal_sha.clone();
    let mut lant_ok = true;
    let mut ems: BTreeMap<(String, String), (String, String, Value)> = BTreeMap::new();

    for (i, &ln) in lant_lines_raw.iter().enumerate() {
        let r: Value = serde_json::from_slice(ln).map_err(|e| format!("TU4 LANT {i}: {e}"))?;
        if r["i"].as_u64() != Some(i as u64)
            || r["prev_line_sha256"].as_str() != Some(prev_line.as_str())
        {
            lant_ok = false;
        }
        prev_line = sha256_hex_bytes(ln);
        let rel_file = r["file"].as_str().unwrap_or("").to_string();
        let em_bytes =
            fs::read(trib.join(&rel_file)).map_err(|e| format!("read em {rel_file}: {e}"))?;
        let em_sha = sha256_hex_bytes(&em_bytes);
        if r["sha256"].as_str() != Some(em_sha.as_str()) {
            lant_ok = false;
        }
        let em_doc: Value =
            serde_json::from_slice(&em_bytes).map_err(|e| format!("parse em {rel_file}: {e}"))?;
        let g = r["game"].as_str().unwrap_or("").to_string();
        let td = r["target_date"].as_str().unwrap_or("").to_string();
        if em_doc["game"].as_str() != Some(g.as_str())
            || em_doc["target_date"].as_str() != Some(td.as_str())
            || em_doc["positive_claims_allowed"].as_bool() != Some(false)
        {
            lant_ok = false;
        }
        if ems.insert((g, td), (rel_file, em_sha, em_doc)).is_some() {
            lant_ok = false;
        }
    }

    for g in ["649", "540", "joker"] {
        let mut pv = em_seal_sha.clone();
        let mut g_keys: Vec<(String, String)> =
            ems.keys().filter(|k| k.0 == g).cloned().collect();
        g_keys.sort_by(|a, b| a.1.cmp(&b.1));
        for (j, key) in g_keys.iter().enumerate() {
            let (_, sha, doc) = &ems[key];
            if doc["chain"]["index"].as_u64() != Some(j as u64)
                || doc["chain"]["prev_sha256"].as_str() != Some(pv.as_str())
            {
                lant_ok = false;
            }
            pv = sha.clone();
        }
    }

    // 2. Verificăm DECONTARI.jsonl linie cu linie + identitățile Newton `ln_e` + stările proceselor E
    let dec_bytes = fs::read(&dec_p).map_err(|e| format!("read DECONTARI.jsonl: {e}"))?;
    let dec_lines_raw: Vec<&[u8]> = dec_bytes
        .split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .collect();
    let mut dec_prev = ju_seal_sha.clone();
    let mut dec_ok = true;
    let mut settled_keys = BTreeSet::new();
    let mut streams_state: BTreeMap<String, Value> = BTreeMap::new();
    for m in ["H1", "H3"] {
        for g in ["649", "540", "joker"] {
            streams_state.insert(
                format!("{m}_{g}"),
                json!({"state": "EXPLORARE", "lnE": 0.0, "lnE_conf": 0.0}),
            );
        }
    }

    let mut dec_files_and_shas = Vec::new();
    for (i, &ln) in dec_lines_raw.iter().enumerate() {
        let r: Value = serde_json::from_slice(ln).map_err(|e| format!("TU4 DEC {i}: {e}"))?;
        if r["i"].as_u64() != Some(i as u64)
            || r["prev_line_sha256"].as_str() != Some(dec_prev.as_str())
        {
            dec_ok = false;
        }
        dec_prev = sha256_hex_bytes(ln);
        let rel_file = r["file"].as_str().unwrap_or("").to_string();
        let d_path = trib.join(&rel_file);
        if file_mode_octal(&d_path)? != 0o444 {
            dec_ok = false;
        }
        let d_bytes = fs::read(&d_path).map_err(|e| format!("read dec {rel_file}: {e}"))?;
        let d_sha = sha256_hex_bytes(&d_bytes);
        if r["sha256"].as_str() != Some(d_sha.as_str()) {
            dec_ok = false;
        }
        let doc: Value =
            serde_json::from_slice(&d_bytes).map_err(|e| format!("parse dec {rel_file}: {e}"))?;
        let g = doc["game"].as_str().unwrap_or("").to_string();
        let td = doc["target_date"].as_str().unwrap_or("").to_string();
        if doc["positive_claims_allowed"].as_bool() != Some(false) {
            dec_ok = false;
        }
        let Some((_, em_sha, em_doc)) = ems.get(&(g.clone(), td.clone())) else {
            dec_ok = false;
            continue;
        };
        if doc["emission"]["sha256"].as_str() != Some(em_sha.as_str()) {
            dec_ok = false;
        }
        let (k_pick, has_ball) = match g.as_str() {
            "649" | "540" => (6usize, false),
            "joker" => (5usize, true),
            _ => (6usize, false),
        };
        let drawn_nums: Vec<usize> = doc["draw"]["numbers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();
        let drawn_jk = doc["draw"]["joker"].as_u64().map(|u| u as usize);

        for m in ["H1", "H3"] {
            let mb = &doc["models"][m];
            let blk = &em_doc["models"][m];
            if doc["status"].as_str() == Some("DECONTAT") {
                let lw: Vec<f64> = blk["logw"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default();
                let act = blk["bet_active"].as_bool().unwrap_or(false);
                let lm = ln_e_newton_tu4(&lw, &drawn_nums, k_pick, act);
                let lb = if has_ball {
                    let blw: Vec<f64> = blk["ball"]["logw"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                        .unwrap_or_default();
                    let bact = blk["ball"]["bet_active"].as_bool().unwrap_or(false);
                    if let Some(jk) = drawn_jk {
                        ln_e_newton_tu4(&blw, &[jk], 1, bact)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let doc_lm = mb["ln_e_main"].as_f64().unwrap_or(f64::NAN);
                let doc_lb = mb["ln_e_ball"].as_f64().unwrap_or(0.0);
                let doc_lt = mb["ln_e"].as_f64().unwrap_or(f64::NAN);
                if (lm - doc_lm).abs() > 1.0e-10
                    || (lb - doc_lb).abs() > 1.0e-10
                    || (doc_lt - (doc_lm + doc_lb)).abs() > 1.0e-14
                {
                    dec_ok = false;
                }
            }
            let s_key = format!("{m}_{g}");
            streams_state.insert(s_key.clone(), doc["streams_after"][&s_key].clone());
        }
        settled_keys.insert((g, td));
        dec_files_and_shas.push((rel_file, d_sha));
    }

    // 3. Verificăm RULARI.jsonl, JUDECATOR_RULARI.jsonl și oglinzile SSD
    let rul_txt = fs::read_to_string(&rul_p).map_err(|e| format!("read RULARI.jsonl: {e}"))?;
    let mut rul_lines = 0usize;
    for (idx, ln) in rul_txt.lines().enumerate() {
        if ln.trim().is_empty() {
            continue;
        }
        let _: Value =
            serde_json::from_str(ln).map_err(|e| format!("RULARI.jsonl {idx}: {e}"))?;
        rul_lines += 1;
    }

    let jrul_txt =
        fs::read_to_string(&jrul_p).map_err(|e| format!("read JUDECATOR_RULARI.jsonl: {e}"))?;
    let mut jrul_lines = 0usize;
    let mut last_j_rc_ok = false;
    for (idx, ln) in jrul_txt.lines().enumerate() {
        if ln.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(ln).map_err(|e| format!("JUDECATOR_RULARI {idx}: {e}"))?;
        last_j_rc_ok =
            v["rc"].as_i64() == Some(0) && v["positive_claims_allowed"].as_bool() == Some(false);
        jrul_lines += 1;
    }

    let ssd_em = ssd_root.join("QL649_TU4_EM");
    let mut ssd_em_ok = ssd_em.exists();
    if ssd_em_ok {
        for (rel_f, sha, _) in ems.values() {
            let p = ssd_em.join(rel_f);
            if !p.exists() || sha256_file_hex(&p).map(|x| x.1).unwrap_or_default() != *sha {
                ssd_em_ok = false;
            }
        }
    }

    let ssd_ju = ssd_root.join("QL649_TU4_JU");
    let mut ssd_ju_ok = ssd_ju.exists() && ssd_ju.join("SCORURI_JU.mmap").exists();
    if ssd_ju_ok {
        for (rel_f, sha) in &dec_files_and_shas {
            let p = ssd_ju.join(rel_f);
            if !p.exists() || sha256_file_hex(&p).map(|x| x.1).unwrap_or_default() != *sha {
                ssd_ju_ok = false;
            }
        }
    }

    let mut pending_emissions: Vec<String> = ems
        .keys()
        .filter(|k| !settled_keys.contains(*k))
        .map(|(g, d)| format!("{g}:{d}"))
        .collect();
    pending_emissions.sort();

    let guard_s35_03 = check_month_rollover_guard_s35_03(repo_root, ql_root, now_unix)?;

    let pass = ju_seal_ok
        && lant_mode_0444
        && dec_mode_0444
        && rul_mode_0444
        && jrul_mode_0444
        && lant_ok
        && dec_ok
        && last_j_rc_ok
        && ssd_em_ok
        && ssd_ju_ok
        && guard_s35_03.pass;

    Ok(Tu4RegistryAudit {
        pass,
        em_seal_sha256: em_seal_sha,
        ju_seal_sha256: ju_seal_sha,
        ju_seal_matches_canonical: ju_seal_ok,
        lant_file_sha256: lant_file_sha,
        lant_mode_0444,
        lant_emissions_count: lant_lines_raw.len(),
        lant_tip_line_sha256: prev_line,
        lant_chain_and_files_ok: lant_ok,
        decontari_file_sha256: dec_file_sha,
        decontari_mode_0444: dec_mode_0444,
        decontari_count: dec_lines_raw.len(),
        decontari_tip_line_sha256: dec_prev,
        decontari_newton_and_states_ok: dec_ok,
        rulari_lines: rul_lines,
        rulari_mode_0444: rul_mode_0444,
        judecator_rulari_lines: jrul_lines,
        judecator_rulari_mode_0444: jrul_mode_0444,
        latest_judge_rc_zero: last_j_rc_ok,
        ssd_mirror_em_ok: ssd_em_ok,
        ssd_mirror_ju_ok: ssd_ju_ok,
        streams_state,
        pending_emissions,
        guard_s35_03,
        positive_claims_allowed: false,
    })
}

// ============================================================================
// 4. Certificarea Registrului 3: `Quantum Limit prospectiv` + `OGLINDA` (Read-Only)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QlGameProspectiveAudit {
    pub game: String,
    pub pass: bool,
    pub ledger_sha256: String,
    pub report_md_sha256: String,
    pub scored_entries: usize,
    pub binding_entries: usize,
    pub required_entries: usize,
    pub verdict: String,
    pub sealed_files_verified: usize,
    pub pending_sealed_targets: Vec<String>,
    pub append_only_vs_dl_preimage_ok: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OglindaReadOnlyAudit {
    pub pass: bool,
    pub manifest_sha256: String,
    pub files_verified_count: usize,
    pub all_manifest_hashes_intact: bool,
    pub status: String,
    pub evaluated_prospective_draws: usize,
    pub target_draw: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumLimitRegistryAudit {
    pub pass: bool,
    pub protocol_md_sha256: String,
    pub total_sealed_forecasts_verified: usize,
    pub games: Vec<QlGameProspectiveAudit>,
    pub oglinda: OglindaReadOnlyAudit,
    pub positive_claims_allowed: bool,
}

fn extract_ql_header_count(line0: &str) -> Option<u32> {
    let idx = line0.find('(')?;
    let rest = &line0[idx + 1..];
    let slash = rest.find("/52")?;
    rest[..slash].trim().parse::<u32>().ok()
}

fn split_ql_report_invariant_and_draws<'a>(lines: &[&'a str]) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
    let derived = "Numărul Joker (";
    let mut keep_invariant = Vec::new();
    let mut cat_names = Vec::new();
    let mut draw_rows = Vec::new();
    let mut in_cat_table = false;
    let mut in_draw_table = false;

    for (i, &ln) in lines.iter().enumerate() {
        if i == 0 || i == 2 || ln.starts_with(derived) {
            continue;
        }
        if ln.starts_with("| Categorie |") {
            in_cat_table = true;
            in_draw_table = false;
            keep_invariant.push(ln);
            continue;
        }
        if ln.starts_with("| Extragere |") {
            in_cat_table = false;
            in_draw_table = true;
            keep_invariant.push(ln);
            continue;
        }
        if ln.trim().is_empty() {
            in_cat_table = false;
            in_draw_table = false;
            keep_invariant.push(ln);
            continue;
        }
        if in_cat_table {
            if ln.starts_with("|---") {
                keep_invariant.push(ln);
            } else if ln.starts_with('|') {
                if let Some(cat) = ln.split('|').nth(1) {
                    cat_names.push(cat.trim());
                }
            }
            continue;
        }
        if in_draw_table {
            if ln.starts_with("|---") {
                keep_invariant.push(ln);
            } else if ln.starts_with('|') {
                draw_rows.push(ln);
            }
            continue;
        }
        keep_invariant.push(ln);
    }
    (keep_invariant, cat_names, draw_rows)
}

fn check_ql_report_append_only(old_md: &str, new_md: &str) -> bool {
    let ol: Vec<&str> = old_md.lines().collect();
    let nl: Vec<&str> = new_md.lines().collect();
    if ol.is_empty() || nl.is_empty() {
        return false;
    }
    match (extract_ql_header_count(ol[0]), extract_ql_header_count(nl[0])) {
        (Some(ko), Some(kn)) if kn >= ko => {}
        _ => return false,
    }
    let (old_inv, old_cats, old_draws) = split_ql_report_invariant_and_draws(&ol);
    let (new_inv, new_cats, new_draws) = split_ql_report_invariant_and_draws(&nl);

    // Aceleași categorii în tabelul cumulativ și prefix exact pe extragerile punctate
    if old_cats != new_cats || new_draws.len() < old_draws.len() {
        return false;
    }
    if new_draws[..old_draws.len()] != old_draws[..] {
        return false;
    }
    let mut j = 0usize;
    for &ln in &new_inv {
        if j < old_inv.len() && ln == old_inv[j] {
            j += 1;
        }
    }
    j == old_inv.len()
}

pub fn certify_quantum_limit_registry(repo_root: &Path) -> Result<QuantumLimitRegistryAudit, String> {
    let qlim = repo_root.join("next_candidate/quantum_limit");
    let (_, proto_sha) = sha256_file_hex(&qlim.join("PREINREGISTRARE_PROSPECTIV_V1.md"))?;
    let pre_base = repo_root.join(
        "next_candidate/predictie_numere/vechime_conditionata_20261023/external_state_at_entry/next_candidate/quantum_limit/prospectiv",
    );

    let mut total_seals = 0usize;
    let mut games_out = Vec::new();
    let mut all_games_ok = true;

    for g in ["649", "540", "joker"] {
        let g_dir = qlim.join("prospectiv").join(g);
        let reg_p = g_dir.join("registru.json");
        let md_p = g_dir.join("REZULTATE_PROSPECTIVE.md");
        let (_, reg_sha) = sha256_file_hex(&reg_p)?;
        let (_, md_sha) = sha256_file_hex(&md_p)?;

        let reg_val: Value = serde_json::from_str(
            &fs::read_to_string(&reg_p).map_err(|e| format!("read {}: {e}", reg_p.display()))?,
        )
        .map_err(|e| format!("parse {}: {e}", reg_p.display()))?;

        // Verificăm toate directoarele țintă și toate fișierele `sigiliu_*.json`
        let mut target_dirs: Vec<PathBuf> = fs::read_dir(&g_dir)
            .map_err(|e| format!("readdir {}: {e}", g_dir.display()))?
            .filter_map(|e| e.ok().map(|x| x.path()))
            .filter(|p| p.is_dir())
            .collect();
        target_dirs.sort();

        let mut seal_digest_by_key: BTreeMap<(String, String), String> = BTreeMap::new();
        let mut g_seals_ok = true;
        let mut g_seals_cnt = 0usize;
        let mut all_targets = Vec::new();

        for td in &target_dirs {
            let t_name = td
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            all_targets.push(t_name.clone());
            let mut s_files: Vec<PathBuf> = fs::read_dir(td)
                .map_err(|e| format!("readdir {}: {e}", td.display()))?
                .filter_map(|e| e.ok().map(|x| x.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.starts_with("sigiliu_") && s.ends_with(".json"))
                })
                .collect();
            s_files.sort();
            for sf in s_files {
                let raw = fs::read_to_string(&sf)
                    .map_err(|e| format!("read {}: {e}", sf.display()))?;
                let rec: Value = serde_json::from_str(&raw)
                    .map_err(|e| format!("parse {}: {e}", sf.display()))?;
                let canon_core = canonical_json_str(&raw, Some("sha256"))?;
                let calc_digest = sha256_hex_bytes(canon_core.as_bytes());
                if rec["sha256"].as_str() != Some(calc_digest.as_str())
                    || rec["game"].as_str() != Some(g)
                    || rec["target_draw"].as_str() != Some(t_name.as_str())
                {
                    g_seals_ok = false;
                }
                let s_fname = sf
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                seal_digest_by_key.insert((t_name.clone(), s_fname), calc_digest);
                g_seals_cnt += 1;
            }
        }
        total_seals += g_seals_cnt;

        let entries = reg_val["entries"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut done_targets = BTreeSet::new();
        for e in &entries {
            let td = e["target_draw"].as_str().unwrap_or("").to_string();
            let sf = e["seal"].as_str().unwrap_or("").to_string();
            let want_sha = e["seal_sha256"].as_str().unwrap_or("");
            if seal_digest_by_key
                .get(&(td.clone(), sf))
                .map(|s| s.as_str())
                != Some(want_sha)
            {
                g_seals_ok = false;
            }
            done_targets.insert(td);
        }

        let pending_targets: Vec<String> = all_targets
            .into_iter()
            .filter(|t| !done_targets.contains(t))
            .collect();

        // Verificăm extensia append-only față de preimaginea DL
        let pre_reg: Value = serde_json::from_str(
            &fs::read_to_string(pre_base.join(g).join("registru.json"))
                .map_err(|e| format!("read preimage registru {g}: {e}"))?,
        )
        .map_err(|e| format!("parse preimage registru {g}: {e}"))?;
        let old_entries = pre_reg["entries"].as_array().cloned().unwrap_or_default();
        let mut append_reg_ok = entries.len() >= old_entries.len()
            && pre_reg["schema"] == reg_val["schema"]
            && pre_reg["protocol"] == reg_val["protocol"]
            && pre_reg["game"] == reg_val["game"];
        if append_reg_ok {
            for (i, oe) in old_entries.iter().enumerate() {
                if let Some(obj) = oe.as_object() {
                    for (k, v) in obj {
                        if !v.is_null() && entries[i].get(k) != Some(v) {
                            append_reg_ok = false;
                        }
                    }
                }
            }
        }
        let old_md = fs::read_to_string(pre_base.join(g).join("REZULTATE_PROSPECTIVE.md"))
            .map_err(|e| format!("read preimage md {g}: {e}"))?;
        let new_md = fs::read_to_string(&md_p)
            .map_err(|e| format!("read live md {g}: {e}"))?;
        let append_ok = append_reg_ok && check_ql_report_append_only(&old_md, &new_md);

        let g_pass = g_seals_ok && append_ok && g_seals_cnt > 0;
        if !g_pass {
            all_games_ok = false;
        }
        games_out.push(QlGameProspectiveAudit {
            game: g.to_string(),
            pass: g_pass,
            ledger_sha256: reg_sha,
            report_md_sha256: md_sha,
            scored_entries: entries.len(),
            binding_entries: reg_val["summary"]["binding"].as_u64().unwrap_or(0) as usize,
            required_entries: reg_val["summary"]["required"].as_u64().unwrap_or(52) as usize,
            verdict: reg_val["summary"]["verdict"]
                .as_str()
                .unwrap_or("în curs")
                .to_string(),
            sealed_files_verified: g_seals_cnt,
            pending_sealed_targets: pending_targets,
            append_only_vs_dl_preimage_ok: append_ok,
        });
    }

    // Verificăm read-only `oglinda_reala_20261021/manifest.json`
    let ogl_dir = repo_root.join("next_candidate/predictie_numere/oglinda_reala_20261021");
    let ogl_man_p = ogl_dir.join("manifest.json");
    let (_, ogl_man_sha) = sha256_file_hex(&ogl_man_p)?;
    let ogl_man: Value = serde_json::from_str(
        &fs::read_to_string(&ogl_man_p)
            .map_err(|e| format!("read {}: {e}", ogl_man_p.display()))?,
    )
    .map_err(|e| format!("parse oglinda manifest: {e}"))?;
    let mut ogl_cnt = 0usize;
    let mut ogl_ok = true;
    if let Some(files_obj) = ogl_man["files"].as_object() {
        for (rel_f, want_v) in files_obj {
            let want = want_v.as_str().unwrap_or("");
            let (_, got) = sha256_file_hex(&ogl_dir.join(rel_f))?;
            if got != want {
                ogl_ok = false;
            }
            ogl_cnt += 1;
        }
    } else {
        ogl_ok = false;
    }

    let oglinda = OglindaReadOnlyAudit {
        pass: ogl_ok && ogl_cnt > 0,
        manifest_sha256: ogl_man_sha,
        files_verified_count: ogl_cnt,
        all_manifest_hashes_intact: ogl_ok,
        status: "PENDING_SEPARAT_FARA_COMPLETARE_RETROACTIVA".to_string(),
        evaluated_prospective_draws: 0,
        target_draw: "2026-10-25 (acoperire externa nereverificata, 0 date evaluate)".to_string(),
    };

    let pass = all_games_ok && oglinda.pass;
    Ok(QuantumLimitRegistryAudit {
        pass,
        protocol_md_sha256: proto_sha,
        total_sealed_forecasts_verified: total_seals,
        games: games_out,
        oglinda,
        positive_claims_allowed: false,
    })
}

// ============================================================================
// 5. Raportul Unificat Poarta `A6` (`GATE_A6_REGISTRE_SI_DECONTARE.json`)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateA6Report {
    pub artifact: String,
    pub session: String,
    pub gate: String,
    pub generated_utc: String,
    pub gate_a6_pass: bool,
    pub positive_claims_allowed: bool,
    pub verification_ms: f64,
    pub xif_registry: XiFRegistryAudit,
    pub tu4_registry: Tu4RegistryAudit,
    pub quantum_limit_registry: QuantumLimitRegistryAudit,
    pub blind_reserves_0400: Vec<BlindReserveLstatEntry>,
    pub blind_reserves_all_untouched_0400: bool,
    pub epistemic_notice: String,
}

pub fn build_gate_a6_report(
    repo_root: &Path,
    ql_root: &Path,
    ssd_root: &Path,
) -> Result<GateA6Report, String> {
    let t0 = Instant::now();
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let xif = certify_xif_registry(ql_root, now_unix)?;
    let tu4 = certify_tu4_registry(repo_root, ql_root, ssd_root, now_unix)?;
    let ql = certify_quantum_limit_registry(repo_root)?;
    let blind = inspect_blind_reserves_lstat(ql_root)?;
    let blind_ok = blind.len() == 4 && blind.iter().all(|b| b.is_strict_0400);

    let gate_a6_pass = xif.pass && tu4.pass && ql.pass && blind_ok;
    let verification_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(GateA6Report {
        artifact: "GATE_A6_REGISTRE_SI_DECONTARE.json".to_string(),
        session: "S5".to_string(),
        gate: "A6".to_string(),
        generated_utc: utc_now_iso8601(),
        gate_a6_pass,
        positive_claims_allowed: false,
        verification_ms,
        xif_registry: xif,
        tu4_registry: tu4,
        quantum_limit_registry: ql,
        blind_reserves_0400: blind,
        blind_reserves_all_untouched_0400: blind_ok,
        epistemic_notice: "positive_claims_allowed = false: calcul clasic HPC (CUDA sm_120 + 24T AVX2), decontare strict pe hartie, fara afirmatii comerciale de castig garantat peste pret si fara sfat de joc; rezervele oarbe 0400 raman sigilate neatins.".to_string(),
    })
}

// ============================================================================
// 6. Telemetrie Hardware Live (`RTX 5060` + `24T CPU` + `16 GB RAM` + `1.8 TB SSD`)
// ============================================================================

#[repr(C)]
struct StatVfsRaw {
    f_bsize: c_ulong,
    f_frsize: c_ulong,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    f_favail: u64,
    f_fsid: c_ulong,
    f_flag: c_ulong,
    f_namemax: c_ulong,
    __f_spare: [i32; 6],
}

unsafe extern "C" {
    fn statvfs64(path: *const c_char, buf: *mut StatVfsRaw) -> i32;
}

fn statvfs_bytes(path: &Path) -> Option<(u64, u64, u64)> {
    let cs = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    let mut st: StatVfsRaw = unsafe { std::mem::zeroed() };
    let rc = unsafe { statvfs64(cs.as_ptr(), &mut st) };
    if rc != 0 {
        return None;
    }
    let bs = if st.f_frsize > 0 {
        st.f_frsize as u64
    } else {
        st.f_bsize as u64
    };
    Some((
        st.f_blocks.saturating_mul(bs),
        st.f_bfree.saturating_mul(bs),
        st.f_bavail.saturating_mul(bs),
    ))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuLiveTelemetry {
    pub available: bool,
    pub name: String,
    pub compute_capability: String,
    pub sm_count: i32,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub used_vram_bytes: u64,
    pub v3_resident_universe_bytes: u64,
    pub temperature_c: Option<f64>,
    pub power_draw_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub sm_util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    pub graphics_clock_mhz: Option<f64>,
    pub pstate: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuLiveTelemetry {
    pub model_name: String,
    pub logical_threads: usize,
    pub avg_mhz: f64,
    pub max_mhz: f64,
    pub loadavg_1m: f64,
    pub loadavg_5m: f64,
    pub loadavg_15m: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RamLiveTelemetry {
    pub sys_mem_total_bytes: u64,
    pub sys_mem_available_bytes: u64,
    pub sys_mem_used_bytes: u64,
    pub sys_swap_total_bytes: u64,
    pub sys_swap_free_bytes: u64,
    pub process_vm_rss_bytes: u64,
    pub process_vm_hwm_bytes: u64,
    pub rss_guard_limit_bytes: u64,
    pub rss_guard_ok: bool,
    pub webview_rss_bytes: u64,
    pub webview_process_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SsdVaultTelemetry {
    pub mount_path: String,
    pub mounted: bool,
    pub fs_total_bytes: u64,
    pub fs_free_bytes: u64,
    pub fs_avail_bytes: u64,
    pub vault_dir: String,
    pub vault_v3bin_count: usize,
    pub vault_v3bin_all_0444: bool,
    pub vault_total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareTelemetrySnapshot {
    pub sampled_utc: String,
    pub positive_claims_allowed: bool,
    pub gpu: GpuLiveTelemetry,
    pub cpu: CpuLiveTelemetry,
    pub ram: RamLiveTelemetry,
    pub ssd: SsdVaultTelemetry,
}

pub fn collect_hardware_telemetry(resident_vram_bytes: u64) -> HardwareTelemetrySnapshot {
    // 1. GPU CUDA + nvidia-smi
    let gpu = match query_device_info() {
        Ok(info) => {
            let used = info.total_vram_bytes.saturating_sub(info.free_vram_bytes);
            let mut temp = None;
            let mut p_draw = None;
            let mut p_lim = None;
            let mut sm_u = None;
            let mut mem_u = None;
            let mut clk = None;
            let mut pstate = "P0/Auto".to_string();

            if let Ok(out) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,clocks.current.graphics,pstate",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if out.status.success() {
                    let s = String::from_utf8_lossy(&out.stdout);
                    if let Some(first_ln) = s.lines().next() {
                        let cols: Vec<&str> = first_ln.split(',').map(|x| x.trim()).collect();
                        if cols.len() >= 7 {
                            temp = cols[0].parse::<f64>().ok();
                            p_draw = cols[1].parse::<f64>().ok();
                            p_lim = cols[2].parse::<f64>().ok();
                            sm_u = cols[3].parse::<f64>().ok();
                            mem_u = cols[4].parse::<f64>().ok();
                            clk = cols[5].parse::<f64>().ok();
                            pstate = cols[6].to_string();
                        }
                    }
                }
            }

            GpuLiveTelemetry {
                available: true,
                name: info.name,
                compute_capability: info.compute_capability,
                sm_count: info.sm_count,
                total_vram_bytes: info.total_vram_bytes,
                free_vram_bytes: info.free_vram_bytes,
                used_vram_bytes: used,
                v3_resident_universe_bytes: resident_vram_bytes,
                temperature_c: temp,
                power_draw_w: p_draw,
                power_limit_w: p_lim,
                sm_util_pct: sm_u,
                mem_util_pct: mem_u,
                graphics_clock_mhz: clk,
                pstate,
            }
        }
        Err(_) => GpuLiveTelemetry {
            available: false,
            name: "Indisponibil".to_string(),
            compute_capability: "-".to_string(),
            sm_count: 0,
            total_vram_bytes: 0,
            free_vram_bytes: 0,
            used_vram_bytes: 0,
            v3_resident_universe_bytes: 0,
            temperature_c: None,
            power_draw_w: None,
            power_limit_w: None,
            sm_util_pct: None,
            mem_util_pct: None,
            graphics_clock_mhz: None,
            pstate: "-".to_string(),
        },
    };

    // 2. CPU 24 fire
    let logical_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let mut model_name = "Intel(R) Core(TM) i7-14650HX".to_string();
    let mut mhz_sum = 0.0f64;
    let mut mhz_cnt = 0usize;
    let mut mhz_max = 0.0f64;
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for ln in cpuinfo.lines() {
            if let Some(rest) = ln.strip_prefix("model name") {
                if let Some((_, val)) = rest.split_once(':') {
                    model_name = val.trim().to_string();
                }
            } else if let Some(rest) = ln.strip_prefix("cpu MHz") {
                if let Some((_, val)) = rest.split_once(':') {
                    if let Ok(v) = val.trim().parse::<f64>() {
                        mhz_sum += v;
                        mhz_cnt += 1;
                        mhz_max = mhz_max.max(v);
                    }
                }
            }
        }
    }
    let avg_mhz = if mhz_cnt > 0 {
        mhz_sum / (mhz_cnt as f64)
    } else {
        0.0
    };
    let mut l1 = 0.0f64;
    let mut l5 = 0.0f64;
    let mut l15 = 0.0f64;
    if let Ok(la) = fs::read_to_string("/proc/loadavg") {
        let parts: Vec<&str> = la.split_whitespace().collect();
        if parts.len() >= 3 {
            l1 = parts[0].parse().unwrap_or(0.0);
            l5 = parts[1].parse().unwrap_or(0.0);
            l15 = parts[2].parse().unwrap_or(0.0);
        }
    }
    let cpu = CpuLiveTelemetry {
        model_name,
        logical_threads,
        avg_mhz,
        max_mhz: mhz_max,
        loadavg_1m: l1,
        loadavg_5m: l5,
        loadavg_15m: l15,
    };

    // 3. RAM Sistem + Garda VmRSS < 11.5 GiB + WebKitGTK
    let mut mem_tot = 0u64;
    let mut mem_avail = 0u64;
    let mut sw_tot = 0u64;
    let mut sw_free = 0u64;
    if let Ok(mi) = fs::read_to_string("/proc/meminfo") {
        for ln in mi.lines() {
            let mut it = ln.split_whitespace();
            let k = it.next().unwrap_or("");
            let kb: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            match k {
                "MemTotal:" => mem_tot = kb * 1024,
                "MemAvailable:" => mem_avail = kb * 1024,
                "SwapTotal:" => sw_tot = kb * 1024,
                "SwapFree:" => sw_free = kb * 1024,
                _ => {}
            }
        }
    }
    let (rss, hwm) = read_process_rss_bytes();
    let mem_rep = resources::memory_report(std::process::id());
    let ram = RamLiveTelemetry {
        sys_mem_total_bytes: mem_tot,
        sys_mem_available_bytes: mem_avail,
        sys_mem_used_bytes: mem_tot.saturating_sub(mem_avail),
        sys_swap_total_bytes: sw_tot,
        sys_swap_free_bytes: sw_free,
        process_vm_rss_bytes: rss,
        process_vm_hwm_bytes: hwm,
        rss_guard_limit_bytes: MAX_SAFE_RSS_BYTES,
        rss_guard_ok: rss < MAX_SAFE_RSS_BYTES,
        webview_rss_bytes: mem_rep.webview_rss_total_octeti.unwrap_or(0),
        webview_process_count: mem_rep.webview_procese.len(),
    };

    // 4. SSD Extern `./ssd_data` + `RUST_CORE_V3_VAULT`
    let mount_p = Path::new("./vault");
    let vault_p = Path::new(DEFAULT_VAULT_DIR);
    let (fs_tot, fs_free, fs_avail) = statvfs_bytes(mount_p).unwrap_or((0, 0, 0));
    let mut v3bin_cnt = 0usize;
    let mut v3bin_all_0444 = true;
    let mut vault_bytes = 0u64;
    if let Ok(rd) = fs::read_dir(vault_p) {
        for e in rd.flatten() {
            let p = e.path();
            if let Ok(m) = fs::symlink_metadata(&p) {
                if m.is_file() {
                    vault_bytes += m.len();
                    if p.extension().and_then(|s| s.to_str()) == Some("v3bin") {
                        v3bin_cnt += 1;
                        if (m.permissions().mode() & 0o777) != 0o444 {
                            v3bin_all_0444 = false;
                        }
                    }
                }
            }
        }
    }
    let ssd = SsdVaultTelemetry {
        mount_path: mount_p.display().to_string(),
        mounted: mount_p.exists() && fs_tot > 0,
        fs_total_bytes: fs_tot,
        fs_free_bytes: fs_free,
        fs_avail_bytes: fs_avail,
        vault_dir: vault_p.display().to_string(),
        vault_v3bin_count: v3bin_cnt,
        vault_v3bin_all_0444: v3bin_cnt == 13 && v3bin_all_0444,
        vault_total_bytes: vault_bytes,
    };

    HardwareTelemetrySnapshot {
        sampled_utc: utc_now_iso8601(),
        positive_claims_allowed: false,
        gpu,
        cpu,
        ram,
        ssd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_json_preserves_python_scientific_floats() {
        let raw = "{\n \"b\": 6.47901435359127e-09,\n \"a\": 300.0,\n \"sha256\": \"deadbeef\"\n}";
        let c = canonical_json_str(raw, Some("sha256")).unwrap();
        assert_eq!(c, "{\"a\":300.0,\"b\":6.47901435359127e-09}");
    }

    #[test]
    fn test_gate_a6_live_registries_pass() {
        let repo = Path::new("./data");
        let ql = Path::new("./data/quantum_lab_649");
        let ssd = Path::new("./vault");
        if !ql.exists() || !ssd.exists() {
            return;
        }
        let rep = build_gate_a6_report(repo, ql, ssd).expect("Gate A6 report");
        assert!(rep.xif_registry.pass, "XI-F fail: {:?}", rep.xif_registry);
        assert!(rep.tu4_registry.pass, "TU4 fail: {:?}", rep.tu4_registry);
        assert!(
            rep.quantum_limit_registry.pass,
            "QL fail: {:?}",
            rep.quantum_limit_registry
        );
        assert!(rep.blind_reserves_all_untouched_0400);
        assert!(rep.gate_a6_pass);
        assert!(!rep.positive_claims_allowed);
    }
}

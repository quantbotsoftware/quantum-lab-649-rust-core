//! Punte nativă Rust read-only către artefactele sigilate `HS-V4` (`.json` + `.npz` + ERA5 meteo + arhive internaționale)
//! și inspectorul 1-Click `Câte & Unde` pe arhivele istorice RO LIVE + Internaționale/RNG.
//!
//! Toate fișierele `HS-V4` sunt verificate SHA-256 față de `hs_v4/SIGILIU_HSV4_UI.json` înainte de citire,
//! iar fișierele `.npz` sunt decodificate nativ în Rust (`crate::npz`) și sub-eșantionate la `<= 140` puncte
//! astfel încât Webview-ul Tauri nu reține niciodată serii mari în memorie.

use std::collections::BTreeSet;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::kjv5::krawtchouk::binom_f64;
use crate::kjv5_walkforward::{load_all_intl_streams, load_ro_live_stream, LoadedStream, INTL_SPECS, RO_SPECS};
use crate::npz::{downsample_indices, load_npz_archive};
use crate::POSITIVE_CLAIMS_ALLOWED;

const MAX_CURVE_POINTS: usize = 140;
const MAX_MATCH_ROWS: usize = 250;

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn read_verified_json(ql_root: &Path, rel_path: &str, frozen_hashes: &Value) -> Result<(Value, String), String> {
    let full = ql_root.join(rel_path);
    let bytes = std::fs::read(&full).map_err(|e| format!("{}: {e}", full.display()))?;
    let got_sha = sha256_bytes(&bytes);
    if let Some(exp_sha) = frozen_hashes.get(rel_path).and_then(Value::as_str) {
        if got_sha != exp_sha {
            return Err(format!("Integritate SHA-256 eșuată pentru {rel_path}: găsit {got_sha}, așteptat {exp_sha}"));
        }
    } else {
        return Err(format!("Cheia '{rel_path}' lipsește din hs_v4/SIGILIU_HSV4_UI.json[frozen_hashes]"));
    }
    let val: Value = serde_json::from_slice(&bytes).map_err(|e| format!("JSON {rel_path}: {e}"))?;
    Ok((val, got_sha))
}

fn load_hsv4_seal(ql_root: &Path) -> Result<(Value, String), String> {
    let p = ql_root.join("hs_v4/SIGILIU_HSV4_UI.json");
    let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let sha = sha256_bytes(&bytes);
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("SIGILIU_HSV4_UI.json: {e}"))?;
    Ok((v, sha))
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// Extrage și sub-eșantionează curbele dintr-o arhivă `.npz` `HS-V4` verificată SHA-256.
pub fn extract_hsv4_npz_curves(
    npz_path: &Path,
    expected_sha: Option<&str>,
    max_points: usize,
) -> Result<Value, String> {
    let bytes = std::fs::read(npz_path).map_err(|e| format!("{}: {e}", npz_path.display()))?;
    let got_sha = sha256_bytes(&bytes);
    if let Some(exp) = expected_sha {
        if got_sha != exp {
            return Err(format!("SHA-256 invalid pentru {}: {got_sha} != {exp}", npz_path.display()));
        }
    }
    let arch = load_npz_archive(npz_path)?;
    let steps_arr =
        arch.get("eval_steps").ok_or_else(|| format!("eval_steps lipsă în {}", npz_path.display()))?.as_i64()?;
    let dates_arr =
        arch.get("eval_dates").ok_or_else(|| format!("eval_dates lipsă în {}", npz_path.display()))?.as_strings()?;
    let e_arr = arch.get("E_curve").ok_or_else(|| format!("E_curve lipsă în {}", npz_path.display()))?.as_f64()?;
    let log10_e_arr =
        arch.get("log10_E_curve").ok_or_else(|| format!("log10_E_curve lipsă în {}", npz_path.display()))?.as_f64()?;
    let w_port_entry =
        arch.get("w_portfolio_curve").ok_or_else(|| format!("w_portfolio_curve lipsă în {}", npz_path.display()))?;
    let w_port_flat = w_port_entry.as_f64()?;
    let n_cols = if w_port_entry.shape.len() == 2 { w_port_entry.shape[1] } else { 8 };

    let ls_hsv4 = arch.get("log_step_hsv4").map(|x| x.as_f64()).transpose()?;
    let ls_u = arch.get("log_step_u").map(|x| x.as_f64()).transpose()?;
    let ls_b = arch.get("log_step_b").map(|x| x.as_f64()).transpose()?;

    let t_len = steps_arr.len();
    let mut cum_u_all = vec![0.0f64; t_len];
    let mut cum_b_all = vec![0.0f64; t_len];
    if let (Some(lh), Some(lu), Some(lb)) = (ls_hsv4, ls_u, ls_b) {
        let mut su = 0.0;
        let mut sb = 0.0;
        for i in 0..t_len.min(lh.len()).min(lu.len()).min(lb.len()) {
            su += lh[i] - lu[i];
            sb += lh[i] - lb[i];
            cum_u_all[i] = su;
            cum_b_all[i] = sb;
        }
    }

    let idx = downsample_indices(t_len, max_points);
    let mut steps = Vec::with_capacity(idx.len());
    let mut dates = Vec::with_capacity(idx.len());
    let mut e_curve = Vec::with_capacity(idx.len());
    let mut log10_e = Vec::with_capacity(idx.len());
    let mut w_0 = Vec::with_capacity(idx.len());
    let mut w_o1 = Vec::with_capacity(idx.len());
    let mut w_o2 = Vec::with_capacity(idx.len());
    let mut w_o4 = Vec::with_capacity(idx.len());
    let mut cum_diff_u = Vec::with_capacity(idx.len());
    let mut cum_diff_b = Vec::with_capacity(idx.len());

    for &i in &idx {
        steps.push(steps_arr[i]);
        dates.push(dates_arr.get(i).cloned().unwrap_or_default());
        e_curve.push(round6(e_arr[i]));
        log10_e.push(round6(log10_e_arr[i]));
        let base = i * n_cols;
        if base + 8 <= w_port_flat.len() {
            let m0 = w_port_flat[base];
            let m1 = w_port_flat[base + 1];
            let m2 = w_port_flat[base + 2];
            let m3 = w_port_flat[base + 3];
            let m4 = w_port_flat[base + 4];
            let m5 = w_port_flat[base + 5];
            let m6 = w_port_flat[base + 6];
            let m7 = w_port_flat[base + 7];
            w_0.push(round6(m0));
            w_o1.push(round6(m1 + m2 + m3 + 0.5 * m5));
            w_o2.push(round6(m4 + 0.5 * m5));
            w_o4.push(round6(m6 + m7));
        } else if base + n_cols <= w_port_flat.len() && n_cols >= 2 {
            // Ex. O2_NegCtrl (2 coloane: m0, m_O2)
            w_0.push(round6(w_port_flat[base]));
            w_o1.push(0.0);
            w_o2.push(round6(w_port_flat[base + 1]));
            w_o4.push(0.0);
        }
        cum_diff_u.push(round6(cum_u_all[i]));
        cum_diff_b.push(round6(cum_b_all[i]));
    }

    Ok(json!({
        "npz_sha256": got_sha,
        "total_points": t_len,
        "sampled_points": idx.len(),
        "steps": steps,
        "dates": dates,
        "E_curve": e_curve,
        "log10_E_curve": log10_e,
        "w_0": w_0,
        "w_O1": w_o1,
        "w_O2": w_o2,
        "w_O4": w_o4,
        "cum_log_diff_vs_U": cum_diff_u,
        "cum_log_diff_vs_B": cum_diff_b,
    }))
}

#[derive(Debug, Deserialize)]
struct Era5Row {
    date: String,
    hour_local: String,
    source: String,
    temp_2m_c: f64,
    rh_2m_pct: f64,
    surface_pressure_hpa: f64,
    dp_24h_hpa: f64,
    rho_air_kg_m3: f64,
}

#[derive(Default, Clone, Copy)]
struct Welford1D {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford1D {
    fn z_before_and_push(&mut self, x: f64) -> f64 {
        let z = if self.n >= 30 {
            let var = self.m2 / ((self.n - 1) as f64);
            if var > 1e-18 {
                (x - self.mean) / var.sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / (self.n as f64);
        let d2 = x - self.mean;
        self.m2 += d * d2;
        z
    }
}

/// Încarcă starea meteorologică ERA5 București `18:00` verificată SHA-256 + scorurile Welford cauzale.
pub fn load_era5_meteo_snapshot(
    ql_root: &Path,
    frozen_hashes: &Value,
    target_date: Option<&str>,
) -> Result<Value, String> {
    let rel = "hs_v4/date_meteo/bucuresti_era5_18h.jsonl";
    let full = ql_root.join(rel);
    let bytes = std::fs::read(&full).map_err(|e| format!("{}: {e}", full.display()))?;
    let got_sha = sha256_bytes(&bytes);
    if let Some(exp) = frozen_hashes.get(rel).and_then(Value::as_str) {
        if got_sha != exp {
            return Err(format!("SHA-256 invalid pentru {rel}: {got_sha} != {exp}"));
        }
    } else {
        return Err(format!("{rel} lipsește din frozen_hashes"));
    }

    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for ln in text.lines() {
        let ln = ln.trim();
        if ln.is_empty() {
            continue;
        }
        let r: Era5Row = serde_json::from_str(ln).map_err(|e| format!("ERA5 JSONL: {e}"))?;
        rows.push(r);
    }
    if rows.is_empty() {
        return Err("Cache ERA5 gol!".into());
    }

    let mut w_t = Welford1D::default();
    let mut w_rh = Welford1D::default();
    let mut w_p = Welford1D::default();
    let mut w_dp = Welford1D::default();
    let mut w_rho = Welford1D::default();

    let want_date = target_date.unwrap_or(&rows.last().unwrap().date);
    let mut chosen_idx = rows.len() - 1;
    let mut z_chosen = [0.0f64; 5];
    let mut z_last = [0.0f64; 5];

    for (idx, r) in rows.iter().enumerate() {
        let zt = w_t.z_before_and_push(r.temp_2m_c);
        let zrh = w_rh.z_before_and_push(r.rh_2m_pct);
        let zp = w_p.z_before_and_push(r.surface_pressure_hpa);
        let zdp = w_dp.z_before_and_push(r.dp_24h_hpa);
        let zrho = w_rho.z_before_and_push(r.rho_air_kg_m3);
        if r.date == want_date {
            chosen_idx = idx;
            z_chosen = [zt, zrh, zp, zdp, zrho];
        }
        if idx + 1 == rows.len() {
            z_last = [zt, zrh, zp, zdp, zrho];
        }
    }
    if rows[chosen_idx].date != want_date {
        z_chosen = z_last;
    }

    let c = &rows[chosen_idx];
    let tail_start = rows.len().saturating_sub(120);
    let tail_rho: Vec<f64> = rows[tail_start..].iter().map(|r| round6(r.rho_air_kg_m3)).collect();
    let tail_dates: Vec<String> = rows[tail_start..].iter().map(|r| r.date.clone()).collect();

    Ok(json!({
        "cache_sha256": got_sha,
        "total_days": rows.len(),
        "first_date": rows.first().map(|r| &r.date),
        "last_cache_date": rows.last().map(|r| &r.date),
        "snapshot": {
            "date": c.date,
            "hour_local": c.hour_local,
            "source": c.source,
            "temp_2m_c": c.temp_2m_c,
            "rh_2m_pct": c.rh_2m_pct,
            "surface_pressure_hpa": c.surface_pressure_hpa,
            "dp_24h_hpa": c.dp_24h_hpa,
            "rho_air_kg_m3": c.rho_air_kg_m3,
            "z_welford": {
                "temp_2m": round6(z_chosen[0]),
                "rh_2m": round6(z_chosen[1]),
                "pressure": round6(z_chosen[2]),
                "dp_24h": round6(z_chosen[3]),
                "rho_air": round6(z_chosen[4]),
            }
        },
        "recent_120d_rho_air": {
            "dates": tail_dates,
            "rho_air_kg_m3": tail_rho,
            "mean_rho_air": round6(w_rho.mean),
        }
    }))
}

/// Vedere completă read-only pentru fila 2 (`⚛ HS-V4 · RO Live & ERA5`).
pub fn load_hsv4_ro_view(ql_root: &Path, game_code: &str, config_key: &str) -> Result<Value, String> {
    let (seal_doc, seal_sha) = load_hsv4_seal(ql_root)?;
    let fh = &seal_doc["frozen_hashes"];

    let (qual_doc, qual_sha) = read_verified_json(ql_root, "hs_v4/rezultate/CALIFICARE_Q.json", fh)?;
    let (_wf_doc, wf_sha) = read_verified_json(ql_root, "hs_v4/rezultate/WALKFORWARD_RO.json", fh)?;
    let (abl_doc, abl_sha) = read_verified_json(ql_root, "hs_v4/rezultate/ABLATIE_RO.json", fh)?;

    let game_blk =
        abl_doc["games"].get(game_code).ok_or_else(|| format!("Joc necunoscut în ABLATIE_RO.json: {game_code}"))?;
    let cfg_rep = game_blk["configs"]
        .get(config_key)
        .ok_or_else(|| format!("Configurație necunoscută '{config_key}' (O1, O1_O2, O1_O4, Full_HSV4)"))?;

    // Verificare sincronizare cu arhiva LIVE curentă
    let live_stream = load_ro_live_stream(ql_root, game_code)?;
    let sealed_last_date = cfg_rep["post_archive_state"]["last_observed_date"].as_str().unwrap_or("");
    let live_last_date = live_stream.steps.last().map(|s| s.date.as_str()).unwrap_or("");
    let live_sync = json!({
        "in_sync": sealed_last_date == live_last_date && cfg_rep["archive"]["n_draws"].as_u64() == Some(live_stream.n_draws_total as u64),
        "sealed_last_date": sealed_last_date,
        "live_last_date": live_last_date,
        "sealed_draws": cfg_rep["archive"]["n_draws"],
        "live_draws": live_stream.n_draws_total,
        "live_version_dir": live_stream.version_dir,
        "live_file_sha256": live_stream.file_sha256,
    });

    // Decodare nativă Rust a fișierului .npz corespunzător configurației selectate
    let npz_meta = &cfg_rep["trajectory_npz"];
    let npz_file = npz_meta["file"].as_str().ok_or("trajectory_npz.file lipsă în ABLATIE_RO.json")?;
    let npz_sha_exp = npz_meta["sha256"].as_str();
    let npz_curves =
        extract_hsv4_npz_curves(&ql_root.join("hs_v4/rezultate").join(npz_file), npz_sha_exp, MAX_CURVE_POINTS)?;

    let meteo = load_era5_meteo_snapshot(ql_root, fh, Some(sealed_last_date))?;

    Ok(json!({
        "protocol": "HSV4-RO-BRIDGE",
        "seals": {
            "sigiliu_hsv4_ui_sha256": seal_sha,
            "calificare_q_sha256": qual_sha,
            "walkforward_ro_sha256": wf_sha,
            "ablatie_ro_sha256": abl_sha,
            "verified_all_frozen_hashes": true,
        },
        "qualification": {
            "status": qual_doc["status"],
            "passed": qual_doc["passed"],
            "total": qual_doc["total"],
            "device": qual_doc["device"],
            "checks": qual_doc["checks"],
        },
        "live_sync": live_sync,
        "meteo_era5": meteo,
        "selected": cfg_rep,
        "npz_curves": npz_curves,
        "comparison_table": game_blk["comparison_table"],
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

/// Evaluează un bilet pe un flux încărcat (`LoadedStream`) și întoarce histograma potrivirilor + extragerile găsite.
fn scan_stream_for_ticket(
    stream: &LoadedStream,
    balls_sorted: &[usize],
    bonus_opt: Option<usize>,
    min_hits: usize,
    collect_rows: bool,
) -> (Value, Vec<Value>) {
    let m_sel = balls_sorted.len();
    let spec = stream.spec;
    let mut mask_u128 = 0u128;
    for &b in balls_sorted {
        mask_u128 |= 1u128 << (b - 1);
    }
    let valid_bonus = match (bonus_opt, spec.m_bonus) {
        (Some(jb), Some(mb)) if jb >= 1 && jb <= mb => Some(jb - 1),
        _ => None,
    };

    let mut hist_main = vec![0usize; m_sel + 1];
    let mut bonus_hits_total = 0usize;
    let mut joint_bonus_hist = vec![0usize; m_sel + 1];
    let mut total_hits_sum = 0usize;
    let mut max_hits_obs = 0usize;
    let mut last_3plus_date: Option<String> = None;
    let mut matched_rows = Vec::new();
    let mut total_matching_draws = 0usize;

    for st in &stream.steps {
        for (d, jb_opt) in &st.draws {
            let jb_opt = *jb_opt;
            let mut hm = 0usize;
            let mut matched_balls = Vec::new();
            for &idx0 in d {
                if (mask_u128 & (1u128 << idx0)) != 0 {
                    hm += 1;
                    if collect_rows {
                        matched_balls.push(idx0 + 1);
                    }
                }
            }
            hist_main[hm] += 1;
            total_hits_sum += hm;
            if hm > max_hits_obs {
                max_hits_obs = hm;
            }
            if hm >= 3 {
                last_3plus_date = Some(st.date.clone());
            }

            let hb = match (valid_bonus, jb_opt) {
                (Some(want_b0), Some(got_b0)) if want_b0 == got_b0 => 1usize,
                _ => 0usize,
            };
            if hb == 1 {
                bonus_hits_total += 1;
                joint_bonus_hist[hm] += 1;
            }

            if hm >= min_hits {
                total_matching_draws += 1;
                if collect_rows {
                    let mut draw_sorted: Vec<usize> = d.iter().map(|&x| x + 1).collect();
                    draw_sorted.sort_unstable();
                    matched_balls.sort_unstable();
                    let jb_1b = jb_opt.map(|x| x + 1);
                    let tier_txt = if valid_bonus.is_some() && hb == 1 {
                        format!("{hm}/{m_sel} + Bonus")
                    } else {
                        format!("{hm}/{m_sel}")
                    };
                    matched_rows.push(json!({
                        "stream_code": spec.code,
                        "stream_name": spec.name,
                        "date": st.date,
                        "draw_numbers": draw_sorted,
                        "draw_bonus": jb_1b,
                        "matched_balls": matched_balls,
                        "hits_main": hm,
                        "hit_bonus": hb == 1,
                        "tier": tier_txt,
                    }));
                }
            }
        }
    }

    // Ordonare descrescătoare după (hits_main, hit_bonus, date)
    matched_rows.sort_by(|a, b| {
        let ha = a["hits_main"].as_u64().unwrap_or(0);
        let hb = b["hits_main"].as_u64().unwrap_or(0);
        hb.cmp(&ha)
            .then_with(|| b["hit_bonus"].as_bool().cmp(&a["hit_bonus"].as_bool()))
            .then_with(|| b["date"].as_str().cmp(&a["date"].as_str()))
    });

    // Așteptarea hipergeometrică exactă sub extragere uniformă C(r, h) * C(N - r, m_sel - h) / C(N, m_sel)
    let denom = binom_f64(spec.n, m_sel);
    let n_draws_f = stream.n_draws_total as f64;
    let mut exp_hist = vec![0.0f64; m_sel + 1];
    for (h, slot) in exp_hist.iter_mut().enumerate().take(m_sel + 1) {
        if h <= spec.r && m_sel - h <= spec.n.saturating_sub(spec.r) && denom > 0.0 {
            *slot = n_draws_f * (binom_f64(spec.r, h) * binom_f64(spec.n - spec.r, m_sel - h) / denom);
        }
    }

    let mut hist_table = Vec::with_capacity(m_sel + 1);
    for h in 0..=m_sel {
        hist_table.push(json!({
            "hits": h,
            "label": format!("{h}/{m_sel}"),
            "observed": hist_main[h],
            "expected_U": round6(exp_hist[h]),
            "with_bonus": if valid_bonus.is_some() { Some(joint_bonus_hist[h]) } else { None },
        }));
    }

    let c3 = (3..=m_sel).map(|h| hist_main[h]).sum::<usize>();
    let e3 = (3..=m_sel).map(|h| exp_hist[h]).sum::<f64>();
    let c4 = (4..=m_sel).map(|h| hist_main[h]).sum::<usize>();
    let e4 = (4..=m_sel).map(|h| exp_hist[h]).sum::<f64>();
    let c5 = (5..=m_sel).map(|h| hist_main[h]).sum::<usize>();
    let e5 = (5..=m_sel).map(|h| exp_hist[h]).sum::<f64>();
    let c6 = (6..=m_sel).map(|h| hist_main[h]).sum::<usize>();
    let e6 = (6..=m_sel).map(|h| exp_hist[h]).sum::<f64>();

    let exp_mean_u = (m_sel * spec.r) as f64 / (spec.n as f64);
    let act_mean = if stream.n_draws_total > 0 { (total_hits_sum as f64) / n_draws_f } else { 0.0 };

    let summary = json!({
        "stream_code": spec.code,
        "stream_name": spec.name,
        "group": spec.group,
        "urn": format!("{}/{}", spec.r, spec.n),
        "m_bonus": spec.m_bonus,
        "n_draws": stream.n_draws_total,
        "first_date": stream.steps.first().map(|s| &s.date),
        "last_date": stream.steps.last().map(|s| &s.date),
        "actual_mean_hits": round6(act_mean),
        "expected_mean_hits_U": round6(exp_mean_u),
        "max_hits_observed": max_hits_obs,
        "last_3plus_date": last_3plus_date,
        "total_matching_draws_ge_min": total_matching_draws,
        "bonus_evaluated": valid_bonus.is_some(),
        "bonus_hits_observed": if valid_bonus.is_some() { Some(bonus_hits_total) } else { None },
        "bonus_hits_expected_U": valid_bonus.and(spec.m_bonus).map(|mb| round6(n_draws_f / (mb as f64))),
        "tiers": {
            "3+": { "observed": c3, "expected_U": round6(e3) },
            "4+": { "observed": c4, "expected_U": round6(e4) },
            "5+": { "observed": c5, "expected_U": round6(e5) },
            "6+": { "observed": c6, "expected_U": round6(e6) },
        },
        "histogram": hist_table,
    });

    (summary, matched_rows)
}

/// Inspector 1-Click `Câte & Unde`: scanează arhiva istorică RO LIVE selectată + opțional toate arhivele RO și
/// Internaționale/RNG compatibile (`max(balls) <= N`), returnând histograma exactă față de urna uniformă și lista
/// extragerilor istorice în care s-au nimerit cel puțin `min_hits` bile.
pub fn inspect_ticket_matches(
    ql_root: &Path,
    game_code: &str,
    balls: &[usize],
    bonus: Option<usize>,
    min_hits: usize,
    include_intl: bool,
) -> Result<Value, String> {
    if balls.is_empty() || balls.len() > 12 {
        return Err(format!("Biletul trebuie să conțină între 1 și 12 bile distincte (primit {}).", balls.len()));
    }
    let mut uniq = BTreeSet::new();
    for &b in balls {
        if b == 0 || b > 90 {
            return Err(format!("Bila {b} este în afara domeniului admis 1..=90."));
        }
        if !uniq.insert(b) {
            return Err(format!("Bila {b} apare duplicată în bilet!"));
        }
    }
    let balls_sorted: Vec<usize> = uniq.into_iter().collect();
    let max_ball = *balls_sorted.last().unwrap();
    let eff_min_hits = min_hits.clamp(1, balls_sorted.len());

    // Încărcăm arhivele RO LIVE
    let mut all_streams = Vec::new();
    for sp in RO_SPECS {
        all_streams.push(load_ro_live_stream(ql_root, sp.code)?);
    }
    if include_intl || INTL_SPECS.iter().any(|s| s.code == game_code) {
        let (_, intl_st) = load_all_intl_streams(ql_root)?;
        all_streams.extend(intl_st);
    }

    let primary = all_streams
        .iter()
        .find(|s| s.spec.code == game_code)
        .ok_or_else(|| format!("Flux necunoscut pentru Câte & Unde: {game_code}"))?;

    if max_ball > primary.spec.n {
        return Err(format!(
            "Bila maximă {max_ball} depășește urna fluxului {} (N = {})!",
            primary.spec.name, primary.spec.n
        ));
    }

    let (primary_summary, mut primary_rows) = scan_stream_for_ticket(primary, &balls_sorted, bonus, eff_min_hits, true);
    let primary_truncated = primary_rows.len() > MAX_MATCH_ROWS;
    if primary_truncated {
        primary_rows.truncate(MAX_MATCH_ROWS);
    }

    let mut cross_streams = Vec::new();
    let mut intl_top_matches = Vec::new();
    if include_intl {
        let high_thr = eff_min_hits.max(4).min(balls_sorted.len());
        for st in &all_streams {
            if max_ball > st.spec.n {
                continue;
            }
            let collect_hi = st.spec.code != game_code;
            let (sum_val, rows) = scan_stream_for_ticket(st, &balls_sorted, bonus, high_thr, collect_hi);
            cross_streams.push(json!({
                "stream_code": sum_val["stream_code"],
                "stream_name": sum_val["stream_name"],
                "group": sum_val["group"],
                "urn": sum_val["urn"],
                "n_draws": sum_val["n_draws"],
                "actual_mean_hits": sum_val["actual_mean_hits"],
                "expected_mean_hits_U": sum_val["expected_mean_hits_U"],
                "hits_3plus": sum_val["tiers"]["3+"]["observed"],
                "exp_3plus_U": sum_val["tiers"]["3+"]["expected_U"],
                "hits_4plus": sum_val["tiers"]["4+"]["observed"],
                "exp_4plus_U": sum_val["tiers"]["4+"]["expected_U"],
                "hits_5plus": sum_val["tiers"]["5+"]["observed"],
                "max_hits_observed": sum_val["max_hits_observed"],
                "last_3plus_date": sum_val["last_3plus_date"],
            }));
            if collect_hi {
                for r in rows.into_iter().take(8) {
                    intl_top_matches.push(r);
                }
            }
        }
        intl_top_matches.sort_by(|a, b| {
            let ha = a["hits_main"].as_u64().unwrap_or(0);
            let hb = b["hits_main"].as_u64().unwrap_or(0);
            hb.cmp(&ha).then_with(|| b["date"].as_str().cmp(&a["date"].as_str()))
        });
        intl_top_matches.truncate(60);
    }

    Ok(json!({
        "ticket_balls": balls_sorted,
        "ticket_bonus": bonus,
        "min_hits": eff_min_hits,
        "primary": primary_summary,
        "primary_matches": primary_rows,
        "primary_matches_truncated": primary_truncated,
        "cross_streams": cross_streams,
        "cross_high_matches": intl_top_matches,
        "epistemic_notice": "Audit istoric retrospectiv („Câte & Unde”): verifică de câte ori și la ce date o mulțime fixată de bile a intersectat extragerile trecute. Sub o urnă uniformă și independentă orice k-mulțime are exact aceeași distribuție hipergeometrică așteptată. positive_claims_allowed=false.",
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

/// Vedere comparativă `HS-V4` + `KJ-V5` pe cele 10 loterii internaționale mecanice & electronice RNG (fila 3).
pub fn load_intl_side_by_side(ql_root: &Path, stream_code: &str) -> Result<Value, String> {
    let (seal_doc, _) = load_hsv4_seal(ql_root)?;
    let fh = &seal_doc["frozen_hashes"];

    let (hsv4_intl, hsv4_intl_sha) = read_verified_json(ql_root, "hs_v4/rezultate/INTERNATIONAL_RNG_EVAL.json", fh)?;
    let (prov_intl, prov_intl_sha) = read_verified_json(ql_root, "hs_v4/date_intl/PROVENIENTA_INTL.json", fh)?;

    let kjv5_intl_p = ql_root.join("kj_v5/rezultate/INTERNATIONAL_RNG_KJV5.json");
    let kjv5_bytes = std::fs::read(&kjv5_intl_p).map_err(|e| format!("{}: {e}", kjv5_intl_p.display()))?;
    let kjv5_intl_sha = sha256_bytes(&kjv5_bytes);
    let kjv5_intl: Value = serde_json::from_slice(&kjv5_bytes).map_err(|e| e.to_string())?;

    let mut overview_rows = Vec::with_capacity(INTL_SPECS.len());
    let mut selected_hsv4 = Value::Null;
    let mut selected_kjv5 = Value::Null;
    let mut selected_hsv4_npz = Value::Null;
    let mut selected_o2_negctrl_npz = Value::Null;

    for sp in INTL_SPECS {
        let grp = sp.group;
        let h_entry = &hsv4_intl[grp][sp.code];
        let h_full = &h_entry["configs"]["Full_HSV4"];
        let h_wt = &h_full["windows"]["total"];

        let k_full = &kjv5_intl[grp][sp.code];
        let k_wt = &k_full["windows"]["total"];

        let h_ldiff_u = h_wt["log_score_HSV4"].as_f64().unwrap_or(0.0) - h_wt["log_score_U"].as_f64().unwrap_or(0.0);
        let k_ldiff_u = k_wt["log_score_KJV5"].as_f64().unwrap_or(0.0) - k_wt["log_score_U"].as_f64().unwrap_or(0.0);

        overview_rows.push(json!({
            "code": sp.code,
            "name": sp.name,
            "group": grp,
            "urn": format!("{}/{}", sp.r, sp.n),
            "bonus_m": sp.m_bonus,
            "evaluated_steps": k_full["archive"]["evaluated_steps"],
            "hsv4_gates": h_full["verdict_gates"],
            "kjv5_gates": k_full["verdict_gates"],
            "hsv4_hits": h_wt["hits_HSV4"],
            "kjv5_hits": k_wt["hits_KJV5"],
            "hits_B": k_wt["hits_B"],
            "hits_R": k_wt["hits_R"],
            "hsv4_excess_U": h_wt["excess_hits_vs_U"],
            "kjv5_excess_U": k_wt["excess_hits_vs_U"],
            "hsv4_log_diff_U": round6(h_ldiff_u),
            "kjv5_log_diff_U": round6(k_ldiff_u),
            "hsv4_E_S": h_full["martingale_trajectory"]["E_S_final"],
            "kjv5_E_S": k_full["martingale_trajectory"]["E_S_final"],
            "hsv4_min_E_s": h_full["martingale_trajectory"]["min_E_s"],
            "kjv5_min_E_s": k_full["martingale_trajectory"]["min_E_s"],
            "hsv4_w0_final": h_full["bayesian_masses"]["posterior_final"]["w_0"],
            "kjv5_w0_final": k_full["bayesian_masses"]["posterior_channels"]["w_0"],
        }));

        if sp.code == stream_code {
            selected_hsv4 = h_entry.clone();
            selected_kjv5 = k_full.clone();
            if let Some(npz_f) = h_full["trajectory_npz"]["file"].as_str() {
                let npz_sha = h_full["trajectory_npz"]["sha256"].as_str();
                selected_hsv4_npz =
                    extract_hsv4_npz_curves(&ql_root.join("hs_v4/rezultate").join(npz_f), npz_sha, MAX_CURVE_POINTS)?;
            }
            if let Some(o2_cfg) = h_entry["configs"].get("O2_NegCtrl") {
                if let Some(npz_f) = o2_cfg["trajectory_npz"]["file"].as_str() {
                    let npz_sha = o2_cfg["trajectory_npz"]["sha256"].as_str();
                    selected_o2_negctrl_npz = extract_hsv4_npz_curves(
                        &ql_root.join("hs_v4/rezultate").join(npz_f),
                        npz_sha,
                        MAX_CURVE_POINTS,
                    )?;
                }
            }
        }
    }

    if selected_kjv5.is_null() {
        return Err(format!("Flux internațional necunoscut: {stream_code}"));
    }

    Ok(json!({
        "selected_code": stream_code,
        "seals": {
            "hsv4_intl_sha256": hsv4_intl_sha,
            "kjv5_intl_sha256": kjv5_intl_sha,
            "provenienta_intl_sha256": prov_intl_sha,
        },
        "provenance_dataset": prov_intl["datasets"][stream_code],
        "overview_rows": overview_rows,
        "selected_hsv4": selected_hsv4,
        "selected_hsv4_npz": selected_hsv4_npz,
        "selected_o2_negctrl_npz": selected_o2_negctrl_npz,
        "selected_kjv5": selected_kjv5,
        "paired_skandinav": {
            "hsv4": hsv4_intl["paired_skandinav_comparison"],
            "kjv5": kjv5_intl["paired_skandinav_comparison"],
        },
        "o2_negative_control_summary": hsv4_intl["o2_negative_control_summary"],
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::ql_root;

    #[test]
    fn hsv4_ro_view_and_ticket_inspector_work() {
        let root = ql_root();
        if !root.join("hs_v4/SIGILIU_HSV4_UI.json").exists() {
            return;
        }
        let v = load_hsv4_ro_view(&root, "649", "Full_HSV4").expect("hsv4 649");
        assert_eq!(v["seals"]["verified_all_frozen_hashes"], true);
        assert_eq!(v["positive_claims_allowed"], false);
        assert!(v["npz_curves"]["sampled_points"].as_u64().unwrap() <= MAX_CURVE_POINTS as u64);

        let insp = inspect_ticket_matches(&root, "649", &[12, 15, 17, 19, 26, 32], None, 3, true).expect("inspect 649");
        assert_eq!(insp["positive_claims_allowed"], false);
        assert!(insp["cross_streams"].as_array().unwrap().len() >= 10);
    }
}

//! Punctul de intrare CLI & Observatorul Nativ Tauri v2 pentru `rust-singularity-core-v3` (`v3.0.0`):
//! - Comenzi `S1–S4`: `vault-init`, `a0-parity`, `cdv3-build`, `cdv3-verify`, `pmv3-build`, `pmv3-verify`,
//!   `c3v2-build`, `c3v2-verify`, `c4v2-build`, `c4v2-verify`, `s3-build`, `s3-verify`,
//!   `ckv6-qualify`, `ckv6-power`, `ckv6-seal-code`, `ckv6-run`, `ckv6-verify`, `s4-build`, `s4-verify`;
//! - Comenzi `v2` moștenite: `kjv5-qualify`, `kjv5-walkforward`, `inventory`, `parity`, `admissibility`, `memory`, `monitor`;
//! - Comenzi `S5` (`Porțile A6 + A7`): `a6-certify`, `s5-verify`, `gui [--smoke <CALE_JSON>]` (`12` file dark-mode WebKitGTK).
//! Guvernanță științifică: `positive_claims_allowed = false`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use rust_singularity_core_v3::c3v2::{run_full_c3v2_pipeline, verify_existing_c3v2};
use rust_singularity_core_v3::c4v2::{run_full_c4v2_pipeline, verify_existing_c4v2};
use rust_singularity_core_v3::cdv3::{run_cdv3_full_pipeline, verify_existing_cdv3_catalog};
use rust_singularity_core_v3::ckv6::{
    run_ckv6_archives_and_dev, run_ckv6_qualify_and_power, seal_ckv6_code_pre_archive,
    verify_existing_s4_ckv6, DEFAULT_CKV6_RS_PATH, DEFAULT_CODE_SEAL_PATH, DEFAULT_QL_ROOT,
};
use rust_singularity_core_v3::cuda_ffi::query_device_info;
use rust_singularity_core_v3::kjv5::qualify::run_qualification_suite;
use rust_singularity_core_v3::parser::{self, Game};
use rust_singularity_core_v3::pmv3::{
    run_full_pmv3_s2, verify_existing_pmv3_s2, DEFAULT_LOTO_CURRENT_JSON,
};
use rust_singularity_core_v3::registries::{build_gate_a6_report, collect_hardware_telemetry};
use rust_singularity_core_v3::v3_observatory::V3ObservatoryState;
use rust_singularity_core_v3::vault::{
    run_gate_a0_parity, MappedVaultUniverse, DEFAULT_VAULT_DIR,
};
use rust_singularity_core_v3::{
    bridge, hsv4_bridge, kjv5_walkforward, models, resources, telemetry, POSITIVE_CLAIMS_ALLOWED,
};
use serde_json::{json, Value};
use tauri::Manager;

const DEFAULT_REZ_DIR: &str =
    "./results";
const DEFAULT_SSD_ROOT: &str = "./vault";

// ============================================================================
// Stare Partajată Tauri v2 (Read-Only `mmap` + Univers Rezident VRAM `RTX 5060`)
// ============================================================================

struct V3SharedState {
    obs: Arc<V3ObservatoryState>,
    gpu_exec_lock: Arc<Mutex<()>>,
}

struct SmokeState(Mutex<Option<PathBuf>>);
struct TelemetryState(Mutex<Option<telemetry::Monitor>>);

#[tauri::command]
fn v3_status_snapshot(v3: tauri::State<'_, V3SharedState>) -> Value {
    let gpu = query_device_info().ok();
    json!({
        "version": "3.0.0",
        "positive_claims_allowed": false,
        "session_stage": "S5_FINAL_12_TAB_OBSERVATORY",
        "gpu": gpu,
        "vault_dir": DEFAULT_VAULT_DIR,
        "vram_resident_bytes": v3.obs.vram.total_vram_allocated_bytes,
    })
}

#[tauri::command]
fn cmd_v3_overview(v3: tauri::State<'_, V3SharedState>) -> Value {
    v3.obs.overview_snapshot()
}

#[tauri::command]
fn cmd_v3_mmap_ticket(
    v3: tauri::State<'_, V3SharedState>,
    joc: String,
    bile: Option<Vec<u8>>,
    joker: Option<u8>,
    rang: Option<usize>,
) -> Result<Value, String> {
    v3.obs.mmap_query_ticket(&joc, bile, joker, rang)
}

#[tauri::command]
fn cmd_v3_wheel_detail(
    v3: tauri::State<'_, V3SharedState>,
    qap_relabeled: Option<bool>,
    wheel_id: String,
) -> Result<Value, String> {
    v3.obs.wheel_detail(qap_relabeled.unwrap_or(true), &wheel_id)
}

#[tauri::command]
fn cmd_v3_c3_c4(
    v3: tauri::State<'_, V3SharedState>,
    joc: Option<String>,
    buget: Option<u32>,
) -> Value {
    v3.obs.c3_c4_explorer(joc.as_deref().unwrap_or("649"), buget)
}

#[tauri::command]
async fn cmd_v3_gpu_eval_portfolio(
    v3: tauri::State<'_, V3SharedState>,
    joc: String,
    bilete: Vec<Vec<u8>>,
    jokeri: Option<Vec<u8>>,
) -> Result<Value, String> {
    let obs = Arc::clone(&v3.obs);
    let lock = Arc::clone(&v3.gpu_exec_lock);
    tokio::task::spawn_blocking(move || {
        let _g = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        obs.gpu_eval_custom_portfolio(&joc, &bilete, jokeri.as_deref())
    })
    .await
    .map_err(|e| format!("fir gpu_eval_portfolio: {e}"))?
}

#[tauri::command]
fn cmd_v3_ckv6(
    v3: tauri::State<'_, V3SharedState>,
    stream_id: Option<String>,
    cell_idx: Option<usize>,
) -> Value {
    v3.obs.ckv6_explorer(stream_id.as_deref(), cell_idx)
}

#[tauri::command]
async fn cmd_v3_ckv6_live_sim(
    v3: tauri::State<'_, V3SharedState>,
    delta_bp: u32,
    l_regime: u32,
    n_reps: Option<usize>,
    seed: Option<u64>,
) -> Result<Value, String> {
    let obs = Arc::clone(&v3.obs);
    let lock = Arc::clone(&v3.gpu_exec_lock);
    let reps = n_reps.unwrap_or(16);
    let sd = seed.unwrap_or(20_261_105);
    tokio::task::spawn_blocking(move || {
        let _g = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        obs.ckv6_live_gpu_sim(delta_bp, l_regime, reps, sd)
    })
    .await
    .map_err(|e| format!("fir ckv6_live_sim: {e}"))?
}

#[tauri::command]
async fn cmd_v3_registre_live() -> Result<Value, String> {
    tokio::task::spawn_blocking(|| {
        let repo = bridge::repo_root();
        let ql = bridge::ql_root();
        let ssd = PathBuf::from(DEFAULT_SSD_ROOT);
        let rep = build_gate_a6_report(&repo, &ql, &ssd)?;
        serde_json::to_value(rep).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("fir registre_live: {e}"))?
}

#[tauri::command]
async fn cmd_v3_hw_telemetry(v3: tauri::State<'_, V3SharedState>) -> Result<Value, String> {
    let vram_bytes = v3.obs.vram.total_vram_allocated_bytes;
    tokio::task::spawn_blocking(move || {
        serde_json::to_value(collect_hardware_telemetry(vram_bytes)).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("fir hw_telemetry: {e}"))?
}

// ============================================================================
// Comenzi IPC Moștenite din `Rust Core v2` (Filele 4..12)
// ============================================================================

#[tauri::command]
fn cmd_info(smoke: tauri::State<'_, SmokeState>, v3: tauri::State<'_, V3SharedState>) -> Value {
    let mut v = bridge::info();
    v["smoke"] = Value::Bool(smoke.0.lock().map(|g| g.is_some()).unwrap_or(false));
    v["vram_resident_bytes"] = json!(v3.obs.vram.total_vram_allocated_bytes);
    v["mmap_open_ms"] = json!(v3.obs.mmap_open_ms);
    v
}

#[tauri::command]
async fn cmd_kjv5_ro(joc: String, config: Option<String>) -> Result<Value, String> {
    let cfg = config.unwrap_or_else(|| "Full_KJV5".to_string());
    tokio::task::spawn_blocking(move || kjv5_walkforward::load_kjv5_ro_view(&bridge::ql_root(), &joc, &cfg))
        .await
        .map_err(|e| format!("fir kjv5_ro: {e}"))?
}

#[tauri::command]
async fn cmd_hsv4_ro(joc: String, config: Option<String>) -> Result<Value, String> {
    let cfg = config.unwrap_or_else(|| "Full_HSV4".to_string());
    tokio::task::spawn_blocking(move || hsv4_bridge::load_hsv4_ro_view(&bridge::ql_root(), &joc, &cfg))
        .await
        .map_err(|e| format!("fir hsv4_ro: {e}"))?
}

#[tauri::command]
async fn cmd_cate_si_unde(
    joc: String,
    bile: Vec<usize>,
    bonus: Option<usize>,
    min_potriviri: Option<usize>,
    include_intl: Option<bool>,
) -> Result<Value, String> {
    let mh = min_potriviri.unwrap_or(3);
    let inc = include_intl.unwrap_or(true);
    tokio::task::spawn_blocking(move || {
        hsv4_bridge::inspect_ticket_matches(&bridge::ql_root(), &joc, &bile, bonus, mh, inc)
    })
    .await
    .map_err(|e| format!("fir cate_si_unde: {e}"))?
}

#[tauri::command]
async fn cmd_intl_comparatie(flux: String) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || hsv4_bridge::load_intl_side_by_side(&bridge::ql_root(), &flux))
        .await
        .map_err(|e| format!("fir intl_comparatie: {e}"))?
}

#[tauri::command]
async fn cmd_inventar() -> Result<Value, String> {
    bridge::inventory_live(&bridge::ql_root()).await
}

#[tauri::command]
async fn cmd_densitati(joc: String, model: String) -> Result<bridge::DensityView, String> {
    bridge::densities(&bridge::ql_root(), &joc, &model).await
}

#[tauri::command]
async fn cmd_frft(joc: String, a: f64) -> Result<bridge::FrftView, String> {
    bridge::frft_view(&bridge::ql_root(), &joc, a).await
}

#[tauri::command]
fn cmd_admisibilitate(tinta: Option<String>) -> Result<Value, String> {
    let root = bridge::ql_root();
    let t = match tinta {
        Some(t) => t,
        None => bridge::default_target(&root)?,
    };
    bridge::admissibility_view(&root, &t)
}

#[tauri::command]
fn cmd_tinte() -> Result<Value, String> {
    bridge::targets_view(&bridge::ql_root())
}

#[tauri::command]
fn cmd_memorie() -> Result<Value, String> {
    serde_json::to_value(resources::memory_report(std::process::id())).map_err(|e| e.to_string())
}

#[tauri::command]
fn cmd_paritate() -> Result<Value, String> {
    bridge::parity_latest(&bridge::ql_root())
}

#[tauri::command]
fn cmd_telemetrie(tel: tauri::State<'_, TelemetryState>) -> Result<Value, String> {
    let g = tel.0.lock().map_err(|e| e.to_string())?;
    let m = g.as_ref().ok_or("telemetria a fost deja oprită")?;
    serde_json::to_value(m.snapshot()).map_err(|e| e.to_string())
}

fn stop_telemetry(tel: &TelemetryState) -> Option<telemetry::Aggregate> {
    let mut g = tel.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    g.take().map(|mut m| m.stop())
}

#[tauri::command]
fn cmd_smoke_done(
    app: tauri::AppHandle,
    smoke: tauri::State<'_, SmokeState>,
    tel: tauri::State<'_, TelemetryState>,
    v3: tauri::State<'_, V3SharedState>,
    raport: Value,
) -> Result<(), String> {
    let path = smoke
        .0
        .lock()
        .map_err(|e| e.to_string())?
        .clone()
        .ok_or("nu sunt în modul --smoke")?;
    let mem = resources::memory_report(std::process::id());
    let hw = collect_hardware_telemetry(v3.obs.vram.total_vram_allocated_bytes);
    let telemetrie = stop_telemetry(&tel);

    let fe_ok = raport.get("toate_ok").and_then(|x| x.as_bool()).unwrap_or(false);
    let tabs_cnt = raport.get("tabs_count").and_then(|x| x.as_u64()).unwrap_or(0);
    let white_boxes = raport
        .get("dark_mode_audit")
        .and_then(|d| d.get("white_boxes_count"))
        .and_then(|x| x.as_u64())
        .unwrap_or(999);
    let toate_ok = fe_ok
        && tabs_cnt == 12
        && white_boxes == 0
        && hw.gpu.available
        && hw.ram.rss_guard_ok
        && hw.ssd.mounted
        && hw.ssd.vault_v3bin_all_0444;

    let doc = json!({
        "artifact": "GUI_SMOKE_V3.json",
        "tool": "rust-singularity-core-v3 gui --smoke",
        "version": env!("CARGO_PKG_VERSION"),
        "session": "S5",
        "gate": "A7",
        "run_utc": models::parity::utc_now_iso(),
        "toate_ok": toate_ok,
        "verdict": if toate_ok { "PASS" } else { "FAIL" },
        "tabs_count": tabs_cnt,
        "white_boxes_count": white_boxes,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
        "raport_frontend": raport,
        "telemetrie_hardware": hw,
        "memorie_webview": mem,
        "telemetrie": telemetrie,
    });
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
    app.exit(if toate_ok { 0 } else { 4 });
    Ok(())
}

fn launch_gui(rest: &[String]) -> ExitCode {
    let smoke = match rest {
        [] => None,
        [flag, path] if flag == "--smoke" => Some(PathBuf::from(path)),
        _ => {
            eprintln!("utilizare: rust-singularity-core-v3 gui [--smoke <raport.json>]");
            return ExitCode::from(2);
        }
    };
    if env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    let ql_root = bridge::ql_root();
    let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
    let obs = match V3ObservatoryState::open_readonly_with_vram(&ql_root, &vault_dir) {
        Ok(st) => Arc::new(st),
        Err(e) => {
            eprintln!("EROARE initializare V3ObservatoryState (mmap + VRAM): {e}");
            return ExitCode::from(6);
        }
    };

    if let Some(p) = smoke.clone() {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(120));
            if !p.exists() {
                let _ = fs::write(
                    &p,
                    "{\"toate_ok\": false, \"verdict\": \"UNKNOWN\", \"motiv\": \"timeout 120 s fără raport din frontend\"}\n",
                );
            }
            std::process::exit(3);
        });
    }

    let interval = if smoke.is_some() {
        telemetry::INTERVAL_MS_SMOKE
    } else {
        telemetry::INTERVAL_MS_DEFAULT
    };
    let monitor = telemetry::Monitor::start(telemetry::Config::ssd(std::process::id(), interval));

    let built = tauri::Builder::default()
        .manage(V3SharedState {
            obs,
            gpu_exec_lock: Arc::new(Mutex::new(())),
        })
        .manage(SmokeState(Mutex::new(smoke)))
        .manage(TelemetryState(Mutex::new(Some(monitor))))
        .invoke_handler(tauri::generate_handler![
            v3_status_snapshot,
            cmd_v3_overview,
            cmd_v3_mmap_ticket,
            cmd_v3_wheel_detail,
            cmd_v3_c3_c4,
            cmd_v3_gpu_eval_portfolio,
            cmd_v3_ckv6,
            cmd_v3_ckv6_live_sim,
            cmd_v3_registre_live,
            cmd_v3_hw_telemetry,
            cmd_info,
            cmd_kjv5_ro,
            cmd_hsv4_ro,
            cmd_cate_si_unde,
            cmd_intl_comparatie,
            cmd_inventar,
            cmd_densitati,
            cmd_frft,
            cmd_admisibilitate,
            cmd_tinte,
            cmd_memorie,
            cmd_telemetrie,
            cmd_paritate,
            cmd_smoke_done
        ])
        .build(tauri::generate_context!());

    match built {
        Ok(app) => {
            app.run(|handle, event| {
                if let tauri::RunEvent::Exit = event {
                    if let Some(tel) = handle.try_state::<TelemetryState>() {
                        let _ = stop_telemetry(&tel);
                    }
                }
            });
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("EROARE Tauri v2: {e}");
            ExitCode::from(6)
        }
    }
}

// ============================================================================
// Utilitare CLI moștenite din `v2` + Verificare `S5` (`a6-certify` & `s5-verify`)
// ============================================================================

fn tokio_rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn print_json(v: &Value) -> bool {
    match serde_json::to_string_pretty(v) {
        Ok(s) => {
            println!("{s}");
            true
        }
        Err(e) => {
            eprintln!("serializare JSON eșuată: {e}");
            false
        }
    }
}

fn run_s5_verify_cli() -> Result<bool, String> {
    let repo_root = bridge::repo_root();
    let ql_root = bridge::ql_root();
    let ssd_root = Path::new(DEFAULT_SSD_ROOT);
    let vault_dir = Path::new(DEFAULT_VAULT_DIR);

    let obs = V3ObservatoryState::open_readonly_with_vram(&ql_root, vault_dir)?;
    let ov = obs.overview_snapshot();
    if ov["vault_13_v3bin_all_0444_and_valid"].as_bool() != Some(true) {
        return Err("Cele 13 tabele .v3bin nu sunt toate 0444 si valide!".to_string());
    }

    // 1. Verificam interogari < 1 ms zero-copy mmap pe 649, 540, Joker
    let q649 = obs.mmap_query_ticket("649", Some(vec![2, 11, 19, 27, 38, 49]), None, None)?;
    let q540 = obs.mmap_query_ticket("540", Some(vec![3, 12, 21, 30, 39]), None, None)?;
    let qjok = obs.mmap_query_ticket("joker", Some(vec![4, 13, 22, 33, 44]), Some(17), None)?;
    for (lbl, q) in [("649", &q649), ("540", &q540), ("joker", &qjok)] {
        if q["under_1ms"].as_bool() != Some(true) || q["mask_verified_equal"].as_bool() != Some(true) {
            return Err(format!("mmap_query_ticket {lbl} invalid: {q}"));
        }
    }

    // 2. Verificam detalii roata CD-V3Q, explorator C3/C4 si explorator CK-V6
    let wh = obs.wheel_detail(true, "CDV3_649_163")?;
    if wh["meta"]["patterned_after"].as_u64() != Some(0) {
        return Err("CDV3_649_163 patterned_after != 0".to_string());
    }
    let c34 = obs.c3_c4_explorer("649", Some(40));
    if c34["single_game_frontiers"].as_array().map(|a| a.len()).unwrap_or(0) != 1 {
        return Err("c3_c4_explorer 649 B=40 lipsa!".to_string());
    }
    let ck_exp = obs.ckv6_explorer(Some("649"), Some(12));
    if ck_exp["streams_summary"].as_array().map(|a| a.len()).unwrap_or(0) != 13 {
        return Err("ckv6_explorer streams_summary != 13".to_string());
    }

    // 3. Verificam executie interactiva LIVE GPU RTX 5060 + 24T AVX2 pe toate cele 3 jocuri + CK-V6
    let ev649 = obs.gpu_eval_custom_portfolio(
        "649",
        &[vec![2, 9, 17, 26, 35, 48], vec![5, 14, 23, 31, 40, 49]],
        None,
    )?;
    let ev540 = obs.gpu_eval_custom_portfolio(
        "540",
        &[vec![3, 11, 20, 29, 38], vec![6, 15, 24, 33, 40]],
        None,
    )?;
    let evjok = obs.gpu_eval_custom_portfolio(
        "joker",
        &[vec![4, 12, 21, 34, 43], vec![7, 16, 25, 36, 45]],
        Some(&[7, 19]),
    )?;
    for (lbl, ev) in [("649", &ev649), ("540", &ev540), ("joker", &evjok)] {
        if ev["gpu_cpu_bit_exact"].as_bool() != Some(true) {
            return Err(format!("gpu_eval_custom_portfolio {lbl} paritate GPU!=CPU: {ev}"));
        }
    }
    let ck_sim = obs.ckv6_live_gpu_sim(2500, 80, 12, 20_261_105)?;
    let ck_diff = ck_sim["gpu_cpu_witness0_max_diff"].as_f64().unwrap_or(1.0);
    if ck_diff >= 1.0e-12 || ck_sim["shield_w0_050_respected"].as_bool() != Some(true) {
        return Err(format!("ckv6_live_gpu_sim paritate/scut invalid: {ck_sim}"));
    }

    // 4. Verificam Poarta A6 live + telemetria hardware live
    let a6 = build_gate_a6_report(&repo_root, &ql_root, ssd_root)?;
    if !a6.gate_a6_pass || !a6.blind_reserves_all_untouched_0400 || a6.positive_claims_allowed {
        return Err("Gate A6 live nu a trecut!".to_string());
    }
    let hw = collect_hardware_telemetry(obs.vram.total_vram_allocated_bytes);
    if !hw.gpu.available || !hw.ram.rss_guard_ok || !hw.ssd.mounted || !hw.ssd.vault_v3bin_all_0444 {
        return Err("Telemetria hardware live a raportat o eroare!".to_string());
    }

    // 5. Daca GUI_SMOKE_V3.json exista deja, verificam ca are toate_ok == true si white_boxes_count == 0
    let smoke_p = ql_root.join("v3_upgrade/rezultate/GUI_SMOKE_V3.json");
    if smoke_p.exists() {
        let raw = fs::read_to_string(&smoke_p).map_err(|e| format!("read GUI_SMOKE_V3.json: {e}"))?;
        let sv: Value = serde_json::from_str(&raw).map_err(|e| format!("parse GUI_SMOKE_V3.json: {e}"))?;
        if sv["toate_ok"].as_bool() != Some(true)
            || sv["tabs_count"].as_u64() != Some(12)
            || sv["white_boxes_count"].as_u64() != Some(0)
            || sv["positive_claims_allowed"].as_bool() != Some(false)
        {
            return Err("GUI_SMOKE_V3.json prezent dar nu indeplineste toate_ok/12_tabs/0_white_boxes!".to_string());
        }
    }

    Ok(true)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        return launch_gui(&[]);
    }
    let cmd = args[1].as_str();
    let rest = &args[2..];

    match cmd {
        "--version" | "-V" => {
            println!("rust-singularity-core-v3 3.0.0 (CUDA sm_120 + 24T AVX2, positive_claims_allowed=false)");
            ExitCode::SUCCESS
        }
        "gui" => launch_gui(rest),
        "a6-certify" => {
            let out_json = rest
                .first()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(DEFAULT_REZ_DIR).join("GATE_A6_REGISTRE_SI_DECONTARE.json")
                });
            let run = (|| -> Result<bool, String> {
                let repo = bridge::repo_root();
                let ql = bridge::ql_root();
                let ssd = PathBuf::from(DEFAULT_SSD_ROOT);
                let rep = build_gate_a6_report(&repo, &ql, &ssd)?;
                if let Some(p) = out_json.parent() {
                    fs::create_dir_all(p).map_err(|e| format!("mkdir: {e}"))?;
                }
                let txt = serde_json::to_string_pretty(&rep).map_err(|e| e.to_string())? + "\n";
                fs::write(&out_json, &txt).map_err(|e| format!("write {}: {e}", out_json.display()))?;
                println!(
                    "[S5/A6] gate_a6_pass={} | XI-F={} ({} strat / {} dec / {} pend) | TU4={} ({} em / {} dec / S35_03={}) | QL={} ({} sigilii + OGLINDA={}) | blind_0400={} | {:.1} ms",
                    rep.gate_a6_pass,
                    rep.xif_registry.pass,
                    rep.xif_registry.lant_lines,
                    rep.xif_registry.decontari_lines,
                    rep.xif_registry.pending_future_count,
                    rep.tu4_registry.pass,
                    rep.tu4_registry.lant_emissions_count,
                    rep.tu4_registry.decontari_count,
                    rep.tu4_registry.guard_s35_03.classification,
                    rep.quantum_limit_registry.pass,
                    rep.quantum_limit_registry.total_sealed_forecasts_verified,
                    rep.quantum_limit_registry.oglinda.all_manifest_hashes_intact,
                    rep.blind_reserves_all_untouched_0400,
                    rep.verification_ms
                );
                Ok(rep.gate_a6_pass)
            })();
            match run {
                Ok(true) => ExitCode::SUCCESS,
                Ok(false) => ExitCode::from(20),
                Err(e) => {
                    eprintln!("EROARE a6-certify: {e}");
                    ExitCode::from(20)
                }
            }
        }
        "s5-verify" => match run_s5_verify_cli() {
            Ok(ok) => {
                println!("S5_VERIFY_OK={ok}");
                if ok {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(21)
                }
            }
            Err(e) => {
                eprintln!("EROARE s5-verify: {e}");
                ExitCode::from(21)
            }
        },
        // ------------------------------------------------------------------ Comenzi CLI v2 portate
        "kjv5-qualify" => {
            let root = bridge::repo_root();
            let out_path = rest.first().map(PathBuf::from).unwrap_or_else(|| {
                root.join("next_candidate/quantum_lab_649/kj_v5/rezultate/CALIFICARE_Q_KJV5.json")
            });
            match run_qualification_suite(&root, Some(&out_path)) {
                Ok(rep) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rep).unwrap_or_else(|_| "{}".to_string())
                    );
                    if rep.passed == rep.total {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(e) => {
                    eprintln!("eroare kjv5-qualify: {e}");
                    ExitCode::from(2)
                }
            }
        }
        "kjv5-walkforward" => {
            let mut repo = bridge::repo_root();
            let mut it = rest.iter();
            while let Some(a) = it.next() {
                if a == "--repo" {
                    let Some(v) = it.next() else {
                        return ExitCode::from(2);
                    };
                    repo = PathBuf::from(v);
                } else {
                    return ExitCode::from(2);
                }
            }
            match kjv5_walkforward::run_full_walkforward_suite(&repo) {
                Ok(val) if print_json(&val) => ExitCode::SUCCESS,
                Ok(_) => ExitCode::from(1),
                Err(e) => {
                    eprintln!("eroare kjv5-walkforward: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "inventory" => tokio_rt().block_on(async {
            let mut it = rest.iter();
            let mut hint = None;
            let mut paths = Vec::new();
            while let Some(a) = it.next() {
                if a == "--game" {
                    match it.next().map(String::as_str).and_then(Game::from_tag) {
                        Some(g) => hint = Some(g),
                        None => return ExitCode::from(2),
                    }
                } else {
                    paths.push(PathBuf::from(a));
                }
            }
            if paths.is_empty() {
                return ExitCode::from(2);
            }
            let results = parser::load_many(paths.clone(), hint).await;
            let (mut archives, mut errors) = (Vec::new(), Vec::new());
            let (mut total_records, mut total_rejects) = (0usize, 0usize);
            for (path, result) in paths.iter().zip(results) {
                match result {
                    Ok(a) => {
                        total_records += a.records.len();
                        total_rejects += a.rejects.len();
                        archives.push(a.summary());
                    }
                    Err(e) => errors.push(
                        json!({"path": path.display().to_string(), "error": e.to_string()}),
                    ),
                }
            }
            let report = json!({
                "tool": "rust-singularity-core-v3",
                "version": env!("CARGO_PKG_VERSION"),
                "command": "inventory",
                "kind": "INVENTAR_DATE_NU_EXPERIMENT",
                "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
                "archives": archives,
                "errors": errors,
                "total_records": total_records,
                "total_rejects": total_rejects,
            });
            if print_json(&report) && errors.is_empty() && total_rejects == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }),
        "parity" => {
            let (mut reference, mut root, mut target, mut s17) = (
                None,
                bridge::ql_root(),
                bridge::S19_PARITY_TARGET.to_string(),
                None,
            );
            let mut it = rest.iter();
            while let Some(a) = it.next() {
                let v = it.next();
                match (a.as_str(), v) {
                    ("--ref", Some(v)) => reference = Some(PathBuf::from(v)),
                    ("--root", Some(v)) => root = PathBuf::from(v),
                    ("--target", Some(v)) => target = v.clone(),
                    ("--s17", Some(v)) => s17 = Some(PathBuf::from(v)),
                    _ => return ExitCode::from(2),
                }
            }
            let Some(reference) = reference else {
                return ExitCode::from(2);
            };
            let s17 = s17.unwrap_or_else(|| {
                root.join("tu4/rezultate/S17/BILETE_ADMISIBILE_20261028T001924Z.json")
            });
            match models::parity::run(&reference, &root, &target, &s17) {
                Ok(rep) if print_json(&rep) && rep["verdict_B2_B7"] == "PASS" => ExitCode::SUCCESS,
                Ok(_) => ExitCode::from(1),
                Err(e) => {
                    eprintln!("paritate: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "admissibility" => {
            let (mut root, mut target, mut enumerate) = (bridge::ql_root(), None::<String>, false);
            let mut it = rest.iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--enumerate" => enumerate = true,
                    "--root" | "--target" => {
                        let Some(v) = it.next() else {
                            return ExitCode::from(2);
                        };
                        if a == "--root" {
                            root = PathBuf::from(v);
                        } else {
                            target = Some(v.clone());
                        }
                    }
                    _ => return ExitCode::from(2),
                }
            }
            let run = (|| -> Result<(Value, bool), String> {
                let tinte = bridge::targets_view(&root)?;
                let target = match target {
                    Some(t) => t,
                    None => bridge::default_target(&root)?,
                };
                let view = bridge::admissibility_view(&root, &target)?;
                let mut checks = Vec::new();
                if enumerate {
                    for s in bridge::admissibility_streams(&root, &target)? {
                        if let Some(c) = bridge::enumeration_check(&s) {
                            checks.push(c);
                        }
                    }
                }
                let ok = checks.iter().all(|c| c["ok"] == true);
                let rep = json!({
                    "tool": "rust-singularity-core-v3 admissibility",
                    "version": env!("CARGO_PKG_VERSION"),
                    "protocol": "TU4-S20 (amendamentul S20-T01)",
                    "run_utc": models::parity::utc_now_iso(),
                    "tinte": tinte,
                    "tinta": target,
                    "vedere": view,
                    "enumerare_ceruta": enumerate,
                    "enumerare": checks,
                    "verdict": if ok { "CONSISTENT" } else { "INCONSISTENT" },
                    "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
                });
                Ok((rep, ok))
            })();
            match run {
                Ok((rep, ok)) if print_json(&rep) && ok => ExitCode::SUCCESS,
                Ok(_) => ExitCode::from(1),
                Err(e) => {
                    eprintln!("admisibilitate: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "memory" => {
            let pid = match rest {
                [] => std::process::id(),
                [flag, p] if flag == "--pid" => match p.parse::<u32>() {
                    Ok(v) => v,
                    Err(_) => return ExitCode::from(2),
                },
                _ => return ExitCode::from(2),
            };
            match serde_json::to_value(resources::memory_report(pid)) {
                Ok(v) if print_json(&v) => ExitCode::SUCCESS,
                _ => ExitCode::from(1),
            }
        }
        "monitor" => {
            let mut cfg = telemetry::Config::ssd(std::process::id(), telemetry::INTERVAL_MS_DEFAULT);
            let mut samples = None;
            let mut it = rest.iter();
            while let Some(a) = it.next() {
                if a == "--fara-verificare-montare" {
                    cfg.mount = None;
                    continue;
                }
                let Some(v) = it.next() else {
                    return ExitCode::from(2);
                };
                let num = v.parse::<u64>();
                match (a.as_str(), num) {
                    ("--pid", Ok(n)) if u32::try_from(n).is_ok() => cfg.pid = n as u32,
                    ("--interval-ms", Ok(n)) => cfg.interval_ms = n,
                    ("--samples", Ok(n)) => samples = Some(n),
                    ("--root", _) => cfg.root = PathBuf::from(v),
                    _ => return ExitCode::from(2),
                }
            }
            let Some(k) = samples else {
                return ExitCode::from(2);
            };
            cfg.max_samples = Some(k);
            if let Err(e) = cfg.validate() {
                eprintln!("configurare: {e}");
                return ExitCode::from(2);
            }
            let agg = telemetry::Monitor::start(cfg).wait();
            let ok = agg.stare == "COMPLET" && agg.rezumat.is_some();
            match serde_json::to_value(&agg) {
                Ok(v) if print_json(&v) && ok => ExitCode::SUCCESS,
                _ => ExitCode::from(1),
            }
        }
        // ------------------------------------------------------------------ Comenzi S1–S4 Intacte
        "vault-init" => {
            let vault_dir = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_VAULT_DIR));
            match MappedVaultUniverse::open_or_build(&vault_dir) {
                Ok(v) => {
                    println!("{}", serde_json::to_string_pretty(&v.manifest).unwrap());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("EROARE vault-init: {e}");
                    ExitCode::from(2)
                }
            }
        }
        "a0-parity" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let out_json = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_REZ_DIR).join("GATE_A0_PARITY.json"));
            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, dma_bytes) = vault.upload_to_vram_via_dma()?;
                let rep = run_gate_a0_parity(&vault, &gpu_univ, dma_bytes)?;
                if let Some(p) = out_json.parent() {
                    fs::create_dir_all(p).map_err(|e| format!("mkdir: {e}"))?;
                }
                let txt = serde_json::to_string_pretty(&rep).unwrap() + "\n";
                fs::write(&out_json, &txt).map_err(|e| format!("write: {e}"))?;
                println!("{txt}");
                if !rep.pass {
                    return Err("Poarta A0 NU a trecut!".to_string());
                }
                Ok(())
            })();
            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE a0-parity: {e}");
                    ExitCode::from(3)
                }
            }
        }
        "cdv3-build" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let a0_path = rez_dir.join("GATE_A0_PARITY.json");
            let cd_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| rez_dir.join("CD_V3_DESIGNURI.json"));

            let run = (|| -> Result<(), String> {
                fs::create_dir_all(&rez_dir).map_err(|e| format!("mkdir rez: {e}"))?;
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, dma_bytes) = vault.upload_to_vram_via_dma()?;
                let a0_rep = run_gate_a0_parity(&vault, &gpu_univ, dma_bytes)?;
                fs::write(
                    &a0_path,
                    serde_json::to_string_pretty(&a0_rep).unwrap() + "\n",
                )
                .map_err(|e| format!("write a0: {e}"))?;
                if !a0_rep.pass {
                    return Err("Poarta A0 a esuat!".to_string());
                }
                let cd_rep = run_cdv3_full_pipeline(&vault, &gpu_univ, &cd_path)?;
                if !cd_rep.gate_a2_pass {
                    return Err("Poarta A2 NU a trecut!".to_string());
                }
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE cdv3-build: {e}");
                    ExitCode::from(4)
                }
            }
        }
        "cdv3-verify" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let cd_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_REZ_DIR).join("CD_V3_DESIGNURI.json"));
            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let ok = verify_existing_cdv3_catalog(&vault, &gpu_univ, &cd_path)?;
                println!("CDV3_VERIFY_OK={ok}");
                Ok(())
            })();
            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE cdv3-verify: {e}");
                    ExitCode::from(5)
                }
            }
        }
        "pmv3-build" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let current_json = PathBuf::from(DEFAULT_LOTO_CURRENT_JSON);
            let s1_cd_path = rez_dir.join("CD_V3_DESIGNURI.json");
            let out_pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let out_cdq_path = rez_dir.join("CD_V3Q_REETICHETAT.json");

            let run = (|| -> Result<(), String> {
                let (pm_rep, _) = run_full_pmv3_s2(
                    &vault_dir,
                    &current_json,
                    &s1_cd_path,
                    &out_pm_path,
                    &out_cdq_path,
                )?;
                println!(
                    "[S2/A3] gate_a3_pass={} in {:.2}s.",
                    pm_rep.gate_a3_pass, pm_rep.build_seconds
                );
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE pmv3-build: {e}");
                    ExitCode::from(7)
                }
            }
        }
        "pmv3-verify" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let cdq_path = rez_dir.join("CD_V3Q_REETICHETAT.json");
            match verify_existing_pmv3_s2(&vault_dir, &pm_path, &cdq_path) {
                Ok(ok) => {
                    println!("PMV3_VERIFY_OK={ok}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("EROARE pmv3-verify: {e}");
                    ExitCode::from(8)
                }
            }
        }
        "c3v2-build" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let cdq_path = rez_dir.join("CD_V3Q_REETICHETAT.json");
            let out_c3_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| rez_dir.join("C3_V2_PORTOFOLII.json"));

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let rep =
                    run_full_c3v2_pipeline(&vault, &gpu_univ, &pm_path, &cdq_path, &out_c3_path)?;
                println!(
                    "[S3/C3-v2] c3v2_gate_pass={} | G4={}/{} | {:.2}s",
                    rep.c3v2_gate_pass,
                    rep.gate_g4_exact_hits,
                    rep.gate_g4_total_instances,
                    rep.build_seconds
                );
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE c3v2-build: {e}");
                    ExitCode::from(9)
                }
            }
        }
        "c3v2-verify" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let c3_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| rez_dir.join("C3_V2_PORTOFOLII.json"));

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let ok = verify_existing_c3v2(&vault, &gpu_univ, &pm_path, &c3_path)?;
                println!("C3V2_VERIFY_OK={ok}");
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE c3v2-verify: {e}");
                    ExitCode::from(10)
                }
            }
        }
        "c4v2-build" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let cdq_path = rez_dir.join("CD_V3Q_REETICHETAT.json");
            let c3_path = rez_dir.join("C3_V2_PORTOFOLII.json");
            let out_c4_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| rez_dir.join("C4_V2_SIMULARE_1M.json"));

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let rep = run_full_c4v2_pipeline(
                    &vault,
                    &gpu_univ,
                    &pm_path,
                    &cdq_path,
                    &c3_path,
                    &out_c4_path,
                )?;
                println!(
                    "[S3/C4-v2] c4v2_gate_pass={} | strategii={} | {:.2}s",
                    rep.c4v2_gate_pass,
                    rep.strategies.len(),
                    rep.build_seconds
                );
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE c4v2-build: {e}");
                    ExitCode::from(11)
                }
            }
        }
        "c4v2-verify" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let c4_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_REZ_DIR).join("C4_V2_SIMULARE_1M.json"));

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let ok = verify_existing_c4v2(&vault, &c4_path)?;
                println!("C4V2_VERIFY_OK={ok}");
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE c4v2-verify: {e}");
                    ExitCode::from(12)
                }
            }
        }
        "s3-build" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let cdq_path = rez_dir.join("CD_V3Q_REETICHETAT.json");
            let c3_path = rez_dir.join("C3_V2_PORTOFOLII.json");
            let c4_path = rez_dir.join("C4_V2_SIMULARE_1M.json");

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let _ = run_full_c3v2_pipeline(&vault, &gpu_univ, &pm_path, &cdq_path, &c3_path)?;
                let _ = run_full_c4v2_pipeline(
                    &vault, &gpu_univ, &pm_path, &cdq_path, &c3_path, &c4_path,
                )?;
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE s3-build: {e}");
                    ExitCode::from(13)
                }
            }
        }
        "s3-verify" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let pm_path = rez_dir.join("PM_V3_CALIBRARE.json");
            let c3_path = rez_dir.join("C3_V2_PORTOFOLII.json");
            let c4_path = rez_dir.join("C4_V2_SIMULARE_1M.json");

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let ok3 = verify_existing_c3v2(&vault, &gpu_univ, &pm_path, &c3_path)?;
                let ok4 = verify_existing_c4v2(&vault, &c4_path)?;
                println!("C3V2_VERIFY_OK={ok3}");
                println!("C4V2_VERIFY_OK={ok4}");
                println!("S3_VERIFY_OK={}", ok3 && ok4);
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE s3-verify: {e}");
                    ExitCode::from(14)
                }
            }
        }
        "ckv6-qualify" | "ckv6-power" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let out_cal_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| rez_dir.join("CK_V6_CALIFICARE_PUTERE.json"));

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let rep = run_ckv6_qualify_and_power(&vault, &gpu_univ, &out_cal_path)?;
                println!(
                    "[S4/A5] Q={}/6 PASS | archive_files_read={} | {:.2}s",
                    rep.q_passed_count, rep.archive_files_read, rep.total_seconds
                );
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE {cmd}: {e}");
                    ExitCode::from(15)
                }
            }
        }
        "ckv6-seal-code" => {
            let ql_root = PathBuf::from(DEFAULT_QL_ROOT);
            let rs_path = PathBuf::from(DEFAULT_CKV6_RS_PATH);
            let cal_path = PathBuf::from(DEFAULT_REZ_DIR).join("CK_V6_CALIFICARE_PUTERE.json");
            let seal_path = PathBuf::from(DEFAULT_CODE_SEAL_PATH);

            match seal_ckv6_code_pre_archive(&ql_root, &rs_path, &cal_path, &seal_path) {
                Ok(seal) => {
                    println!(
                        "[S4/A5] SIGILIU PRE-ARHIVA ACTIV: ckv6.rs mode={} sha256={}",
                        seal.ckv6_rs_mode_octal, seal.ckv6_rs_sha256
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("EROARE ckv6-seal-code: {e}");
                    ExitCode::from(16)
                }
            }
        }
        "ckv6-run" => {
            let ql_root = PathBuf::from(DEFAULT_QL_ROOT);
            let rs_path = PathBuf::from(DEFAULT_CKV6_RS_PATH);
            let seal_path = PathBuf::from(DEFAULT_CODE_SEAL_PATH);
            let out_arch = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(DEFAULT_REZ_DIR).join("CK_V6_ARHIVE_SI_DEV.json")
                });

            match run_ckv6_archives_and_dev(&ql_root, &rs_path, &seal_path, &out_arch) {
                Ok(rep) => {
                    println!(
                        "[S4/A5] gate_a5_pass={} | verdict={}",
                        rep.gate_a5_pass, rep.scientific_verdict
                    );
                    if !rep.gate_a5_pass {
                        ExitCode::from(17)
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("EROARE ckv6-run: {e}");
                    ExitCode::from(17)
                }
            }
        }
        "s4-build" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let ql_root = PathBuf::from(DEFAULT_QL_ROOT);
            let rs_path = PathBuf::from(DEFAULT_CKV6_RS_PATH);
            let cal_path = rez_dir.join("CK_V6_CALIFICARE_PUTERE.json");
            let seal_path = PathBuf::from(DEFAULT_CODE_SEAL_PATH);
            let arch_path = rez_dir.join("CK_V6_ARHIVE_SI_DEV.json");

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let _ = run_ckv6_qualify_and_power(&vault, &gpu_univ, &cal_path)?;
                let _ = seal_ckv6_code_pre_archive(&ql_root, &rs_path, &cal_path, &seal_path)?;
                let arch = run_ckv6_archives_and_dev(&ql_root, &rs_path, &seal_path, &arch_path)?;
                if !arch.gate_a5_pass {
                    return Err("Poarta A5 NU a trecut!".to_string());
                }
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE s4-build: {e}");
                    ExitCode::from(18)
                }
            }
        }
        "ckv6-verify" | "s4-verify" => {
            let vault_dir = PathBuf::from(DEFAULT_VAULT_DIR);
            let rez_dir = PathBuf::from(DEFAULT_REZ_DIR);
            let ql_root = PathBuf::from(DEFAULT_QL_ROOT);
            let rs_path = PathBuf::from(DEFAULT_CKV6_RS_PATH);
            let seal_path = PathBuf::from(DEFAULT_CODE_SEAL_PATH);
            let cal_path = rez_dir.join("CK_V6_CALIFICARE_PUTERE.json");
            let arch_path = rez_dir.join("CK_V6_ARHIVE_SI_DEV.json");

            let run = (|| -> Result<(), String> {
                let vault = MappedVaultUniverse::open_or_build(&vault_dir)?;
                let (gpu_univ, _) = vault.upload_to_vram_via_dma()?;
                let ok = verify_existing_s4_ckv6(
                    &vault, &gpu_univ, &ql_root, &rs_path, &seal_path, &cal_path, &arch_path,
                )?;
                println!("CKV6_VERIFY_OK={ok}");
                println!("S4_VERIFY_OK={ok}");
                Ok(())
            })();

            match run {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("EROARE {cmd}: {e}");
                    ExitCode::from(19)
                }
            }
        }
        other => {
            eprintln!(
                "Comanda necunoscuta '{other}'. Comenzi: --version | gui [--smoke <json>] | a6-certify | s5-verify | vault-init | a0-parity | cdv3-build | cdv3-verify | pmv3-build | pmv3-verify | c3v2-build | c3v2-verify | c4v2-build | c4v2-verify | s3-build | s3-verify | ckv6-qualify | ckv6-power | ckv6-seal-code | ckv6-run | ckv6-verify | s4-build | s4-verify | kjv5-qualify | kjv5-walkforward | inventory | parity | admissibility | memory | monitor"
            );
            ExitCode::from(1)
        }
    }
}

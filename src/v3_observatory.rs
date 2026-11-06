//! Motorul Observatorului Nativ `Rust Core v3` (`Sesiunea 5 / Poarta A7`):
//! - Cititor Zero-Copy `mmap` strict read-only (`0` scrieri, `< 1 ms` latență) peste toate cele `13` tabele `.v3bin`
//!   din `./vault/` (`1 293 402 964 B` = `1.205 GiB`, sigilate `0444`);
//! - Univers rezident simultan în `RTX 5060` VRAM (`39 077 004` bilete + `3 838 380` extrageri `5/40` + tabelele
//!   de partajare `C3-v2` pe `6/49`, `5/40`, `Joker`) pentru evaluare interactivă live pe GPU CUDA `sm_120` + `24T AVX2`;
//! - Inspecție interactivă a roților `CD-V3 / CD-V3Q`, cuburilor bootstrap `1 000x` `PM-V3`, frontierelor Pareto `C3-v2`,
//!   distribuțiilor `1 000 000` traiectorii `C4-v2` și regimurilor cauzale `CK-V6` (`ckv6_power_10k.v3bin` + simulare GPU live).
//! Guvernanță științifică: `positive_claims_allowed = false`.

use std::fs::{self, File};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::c3v2::{
    load_crowd_params_from_pmv3_report, p_random_analytical_hypergeom, C3V2PortfolioReport,
};
use crate::c4v2::C4V2SimulationReport;
use crate::cdv3::{
    cpu_verify_covering_24t, cpu_verify_joker_24m_24t, mask_to_numbers_1based, pack_joker_ticket,
    unpack_joker_ticket, CdV3CatalogReport,
};
use crate::ckv6::{
    eval_649_history_cpu, gen_synth_649_history_cpu, inspect_blind_reserves_lstat,
    CkV6ArchivesAndDevReport, CkV6QualificationAndPowerReport, DELTA_BP_GRID, L_REGIME_GRID,
    POWER_REPS_PER_CELL,
};
use crate::cuda_ffi::{
    check_rss_guard, gpu_c3v2_eval_540, gpu_c3v2_eval_649, gpu_c3v2_eval_joker,
    gpu_ckv6_649_batch_f64, DeviceBuffer, GpuCrowdTables540, GpuCrowdTables649,
    GpuCrowdTablesJoker, PinnedBuffer, ResidentGpuUniverse, V3CrowdParamsF64,
};
use crate::pmv3::{
    build_binom_table_50x7, is_patterned_ticket, is_pure_1_to_31, pmv3_rank_lex_cpu,
    pmv3_ticket_lw_cpu, CdV3QRelabeledReport, PmV3CalibrationReport,
};
use crate::vault::{
    VaultManifest, COUNT_540_DRAWS6, COUNT_540_TICKETS, COUNT_649, COUNT_JOKER_BASE45,
    COUNT_JOKER_FULL24M, TOTAL_UNIVERSE_TICKETS, VAULT_HEADER_BYTES, VAULT_MAGIC,
};

pub const ALL_13_V3BIN_FILES: [(&str, u32, usize, usize); 13] = [
    ("loto_649_masks.v3bin", 1, COUNT_649, 8),
    ("loto_540_tickets.v3bin", 1, COUNT_540_TICKETS, 8),
    ("loto_540_draws6.v3bin", 2, COUNT_540_DRAWS6, 8),
    ("joker_base45_masks.v3bin", 1, COUNT_JOKER_BASE45, 8),
    ("joker_full_24m.v3bin", 3, COUNT_JOKER_FULL24M, 8),
    ("cdv3_wheels_catalog.v3bin", 5, 6926, 8),
    ("pmv3_boot1000_649.v3bin", 4, COUNT_649, 24),
    ("pmv3_boot1000_540.v3bin", 4, COUNT_540_TICKETS, 24),
    ("pmv3_boot1000_joker.v3bin", 4, COUNT_JOKER_FULL24M, 24),
    ("cdv3q_wheels_catalog.v3bin", 6, 6926, 8),
    ("c3v2_pareto_portfolios.v3bin", 7, 13945, 8),
    ("c4v2_mc1m_summary.v3bin", 8, 90, 128),
    ("ckv6_power_10k.v3bin", 9, 10000, 64),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VaultMmapTableStatus {
    pub filename: String,
    pub table_kind: u32,
    pub record_count: u64,
    pub record_bytes: u64,
    pub file_bytes: u64,
    pub mode_octal: String,
    pub is_0444: bool,
    pub header_valid: bool,
}

fn open_verified_readonly_mmap(
    vault_dir: &Path,
    filename: &str,
    expected_kind: u32,
    expected_records: usize,
    expected_rec_bytes: usize,
) -> Result<(Mmap, VaultMmapTableStatus), String> {
    let p = vault_dir.join(filename);
    let meta = fs::symlink_metadata(&p).map_err(|e| format!("lstat {}: {e}", p.display()))?;
    let mode = meta.permissions().mode() & 0o777;
    let f = File::open(&p).map_err(|e| format!("open readonly {}: {e}", p.display()))?;
    let mm = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", p.display()))?;
    let want_len = VAULT_HEADER_BYTES + expected_records * expected_rec_bytes;
    if mm.len() != want_len {
        return Err(format!(
            "{} dimensiune invalida: {} != {}",
            filename,
            mm.len(),
            want_len
        ));
    }
    if &mm[0..8] != VAULT_MAGIC {
        return Err(format!("{filename}: magic QLV3VLT invalid"));
    }
    let ver = u32::from_le_bytes(mm[8..12].try_into().unwrap());
    let kind = u32::from_le_bytes(mm[12..16].try_into().unwrap());
    let rec_cnt = u64::from_le_bytes(mm[24..32].try_into().unwrap());
    let rec_b = u64::from_le_bytes(mm[32..40].try_into().unwrap());
    let hdr_ok = ver == 3
        && kind == expected_kind
        && rec_cnt == (expected_records as u64)
        && rec_b == (expected_rec_bytes as u64);
    if !hdr_ok {
        return Err(format!(
            "{filename}: header invalid ver={ver} kind={kind} cnt={rec_cnt} rec_b={rec_b}"
        ));
    }
    let mm_bytes = mm.len() as u64;
    Ok((
        mm,
        VaultMmapTableStatus {
            filename: filename.to_string(),
            table_kind: kind,
            record_count: rec_cnt,
            record_bytes: rec_b,
            file_bytes: mm_bytes,
            mode_octal: format!("{mode:04o}"),
            is_0444: mode == 0o444,
            header_valid: hdr_ok,
        },
    ))
}

pub struct ResidentVramEngine {
    pub gpu_univ: ResidentGpuUniverse,
    pub qt_649: GpuCrowdTables649,
    pub qt_540: GpuCrowdTables540,
    pub qt_joker: GpuCrowdTablesJoker,
    pub total_vram_allocated_bytes: u64,
    pub init_ms: f64,
}

unsafe impl Send for ResidentVramEngine {}
unsafe impl Sync for ResidentVramEngine {}

pub struct V3ObservatoryState {
    pub ql_root: PathBuf,
    pub vault_dir: PathBuf,
    pub mmap_open_ms: f64,
    pub vault_manifest: VaultManifest,
    pub table_statuses: Vec<VaultMmapTableStatus>,
    mmap_649: Mmap,
    mmap_540_tickets: Mmap,
    mmap_540_draws6: Mmap,
    mmap_joker_base45: Mmap,
    mmap_joker_full24m: Mmap,
    mmap_cdv3_cat: Mmap,
    mmap_boot_649: Mmap,
    mmap_boot_540: Mmap,
    mmap_boot_joker: Mmap,
    mmap_cdv3q_cat: Mmap,
    mmap_c3_ports: Mmap,
    mmap_c4_mc1m: Mmap,
    mmap_ckv6_power: Mmap,
    pub binom_tbl: [[u32; 7]; 50],
    pub params_649: V3CrowdParamsF64,
    pub params_540: V3CrowdParamsF64,
    pub params_joker: V3CrowdParamsF64,
    pub cdv3_report: CdV3CatalogReport,
    pub cdv3q_report: CdV3QRelabeledReport,
    pub pmv3_report: PmV3CalibrationReport,
    pub c3_report: C3V2PortfolioReport,
    pub c4_report: C4V2SimulationReport,
    pub ckv6_qual_report: CkV6QualificationAndPowerReport,
    pub ckv6_arch_report: CkV6ArchivesAndDevReport,
    pub vram: ResidentVramEngine,
}

impl V3ObservatoryState {
    pub fn open_readonly_with_vram(ql_root: &Path, vault_dir: &Path) -> Result<Self, String> {
        check_rss_guard(0)?;
        let t_mm = Instant::now();
        let mf_raw = fs::read_to_string(vault_dir.join("VAULT_MANIFEST.json"))
            .map_err(|e| format!("read VAULT_MANIFEST.json: {e}"))?;
        let vault_manifest: VaultManifest = serde_json::from_str(&mf_raw)
            .map_err(|e| format!("parse VAULT_MANIFEST.json: {e}"))?;

        let mut statuses = Vec::with_capacity(13);
        let (mmap_649, s0) =
            open_verified_readonly_mmap(vault_dir, "loto_649_masks.v3bin", 1, COUNT_649, 8)?;
        statuses.push(s0);
        let (mmap_540_tickets, s1) = open_verified_readonly_mmap(
            vault_dir,
            "loto_540_tickets.v3bin",
            1,
            COUNT_540_TICKETS,
            8,
        )?;
        statuses.push(s1);
        let (mmap_540_draws6, s2) = open_verified_readonly_mmap(
            vault_dir,
            "loto_540_draws6.v3bin",
            2,
            COUNT_540_DRAWS6,
            8,
        )?;
        statuses.push(s2);
        let (mmap_joker_base45, s3) = open_verified_readonly_mmap(
            vault_dir,
            "joker_base45_masks.v3bin",
            1,
            COUNT_JOKER_BASE45,
            8,
        )?;
        statuses.push(s3);
        let (mmap_joker_full24m, s4) = open_verified_readonly_mmap(
            vault_dir,
            "joker_full_24m.v3bin",
            3,
            COUNT_JOKER_FULL24M,
            8,
        )?;
        statuses.push(s4);
        let (mmap_cdv3_cat, s5) =
            open_verified_readonly_mmap(vault_dir, "cdv3_wheels_catalog.v3bin", 5, 6926, 8)?;
        statuses.push(s5);
        let (mmap_boot_649, s6) =
            open_verified_readonly_mmap(vault_dir, "pmv3_boot1000_649.v3bin", 4, COUNT_649, 24)?;
        statuses.push(s6);
        let (mmap_boot_540, s7) = open_verified_readonly_mmap(
            vault_dir,
            "pmv3_boot1000_540.v3bin",
            4,
            COUNT_540_TICKETS,
            24,
        )?;
        statuses.push(s7);
        let (mmap_boot_joker, s8) = open_verified_readonly_mmap(
            vault_dir,
            "pmv3_boot1000_joker.v3bin",
            4,
            COUNT_JOKER_FULL24M,
            24,
        )?;
        statuses.push(s8);
        let (mmap_cdv3q_cat, s9) =
            open_verified_readonly_mmap(vault_dir, "cdv3q_wheels_catalog.v3bin", 6, 6926, 8)?;
        statuses.push(s9);
        let (mmap_c3_ports, s10) =
            open_verified_readonly_mmap(vault_dir, "c3v2_pareto_portfolios.v3bin", 7, 13945, 8)?;
        statuses.push(s10);
        let (mmap_c4_mc1m, s11) =
            open_verified_readonly_mmap(vault_dir, "c4v2_mc1m_summary.v3bin", 8, 90, 128)?;
        statuses.push(s11);
        let (mmap_ckv6_power, s12) =
            open_verified_readonly_mmap(vault_dir, "ckv6_power_10k.v3bin", 9, 10000, 64)?;
        statuses.push(s12);

        // Pre-încălzim paginile mmap ale cuburilor bootstrap și cataloagelor (< 940 MiB total, VmRSS ~3.3 GiB << 11.5 GiB)
        // astfel încât orice interogare zero-copy mmap pe 6/49, 5/40 și Joker să răspundă sub 20 µs (< 0.02 ms << 1 ms).
        let mut warm_acc = 0u8;
        for mm in [
            &mmap_cdv3_cat,
            &mmap_boot_649,
            &mmap_boot_540,
            &mmap_boot_joker,
            &mmap_cdv3q_cat,
            &mmap_c3_ports,
            &mmap_c4_mc1m,
            &mmap_ckv6_power,
        ] {
            let bytes: &[u8] = &mm[..];
            let mut off = 0usize;
            while off < bytes.len() {
                warm_acc ^= bytes[off];
                off += 4096;
            }
            if !bytes.is_empty() {
                warm_acc ^= bytes[bytes.len() - 1];
            }
        }
        std::hint::black_box(warm_acc);
        let mmap_open_ms = t_mm.elapsed().as_secs_f64() * 1000.0;

        let rez_dir = ql_root.join("v3_upgrade/rezultate");
        let read_j = |name: &str| -> Result<String, String> {
            let p = rez_dir.join(name);
            fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
        };
        let cdv3_report: CdV3CatalogReport = serde_json::from_str(&read_j("CD_V3_DESIGNURI.json")?)
            .map_err(|e| format!("parse CD_V3_DESIGNURI.json: {e}"))?;
        let cdv3q_report: CdV3QRelabeledReport =
            serde_json::from_str(&read_j("CD_V3Q_REETICHETAT.json")?)
                .map_err(|e| format!("parse CD_V3Q_REETICHETAT.json: {e}"))?;
        let pmv3_report: PmV3CalibrationReport =
            serde_json::from_str(&read_j("PM_V3_CALIBRARE.json")?)
                .map_err(|e| format!("parse PM_V3_CALIBRARE.json: {e}"))?;
        let c3_report: C3V2PortfolioReport =
            serde_json::from_str(&read_j("C3_V2_PORTOFOLII.json")?)
                .map_err(|e| format!("parse C3_V2_PORTOFOLII.json: {e}"))?;
        let c4_report: C4V2SimulationReport =
            serde_json::from_str(&read_j("C4_V2_SIMULARE_1M.json")?)
                .map_err(|e| format!("parse C4_V2_SIMULARE_1M.json: {e}"))?;
        let ckv6_qual_report: CkV6QualificationAndPowerReport =
            serde_json::from_str(&read_j("CK_V6_CALIFICARE_PUTERE.json")?)
                .map_err(|e| format!("parse CK_V6_CALIFICARE_PUTERE.json: {e}"))?;
        let ckv6_arch_report: CkV6ArchivesAndDevReport =
            serde_json::from_str(&read_j("CK_V6_ARHIVE_SI_DEV.json")?)
                .map_err(|e| format!("parse CK_V6_ARHIVE_SI_DEV.json: {e}"))?;

        let (params_649, params_540, params_joker) =
            load_crowd_params_from_pmv3_report(&pmv3_report)?;
        let binom_tbl = build_binom_table_50x7();

        // Încărcăm universul complet de 39,077,004 bilete + tabelele C3-v2 în cei 8 GB VRAM ai RTX 5060
        let t_gpu = Instant::now();
        let masks_649: &[u64] = unsafe {
            std::slice::from_raw_parts(mmap_649[VAULT_HEADER_BYTES..].as_ptr() as *const u64, COUNT_649)
        };
        let tickets_540: &[u64] = unsafe {
            std::slice::from_raw_parts(
                mmap_540_tickets[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_540_TICKETS,
            )
        };
        let draws6_540: &[u64] = unsafe {
            std::slice::from_raw_parts(
                mmap_540_draws6[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_540_DRAWS6,
            )
        };
        let base_joker: &[u64] = unsafe {
            std::slice::from_raw_parts(
                mmap_joker_base45[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_BASE45,
            )
        };
        let full_joker: &[u64] = unsafe {
            std::slice::from_raw_parts(
                mmap_joker_full24m[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_FULL24M,
            )
        };

        let gpu_univ = {
            let mut dma_buf = PinnedBuffer::<u64>::new_zeroed(8 * 1024 * 1024)?;
            let t_up = Instant::now();
            let mut d_masks_649 = DeviceBuffer::<u64>::alloc(COUNT_649)?;
            let mut d_tickets_540 = DeviceBuffer::<u64>::alloc(COUNT_540_TICKETS)?;
            let mut d_draws6_540 = DeviceBuffer::<u64>::alloc(COUNT_540_DRAWS6)?;
            let mut d_base_joker = DeviceBuffer::<u64>::alloc(COUNT_JOKER_BASE45)?;
            let mut d_full_joker = DeviceBuffer::<u64>::alloc(COUNT_JOKER_FULL24M)?;

            d_masks_649.upload_via_dma_chunked(masks_649, &mut dma_buf)?;
            d_tickets_540.upload_via_dma_chunked(tickets_540, &mut dma_buf)?;
            d_draws6_540.upload_via_dma_chunked(draws6_540, &mut dma_buf)?;
            d_base_joker.upload_via_dma_chunked(base_joker, &mut dma_buf)?;
            d_full_joker.upload_via_dma_chunked(full_joker, &mut dma_buf)?;
            let upload_ms = t_up.elapsed().as_secs_f64() * 1000.0;
            ResidentGpuUniverse {
                d_masks_649,
                d_tickets_540,
                d_draws6_540,
                d_base_joker,
                d_full_joker,
                upload_ms,
            }
        };

        let qt_649 = GpuCrowdTables649::build(&gpu_univ.d_masks_649, &params_649)?;
        let qt_540 = GpuCrowdTables540::build(
            &gpu_univ.d_tickets_540,
            &gpu_univ.d_draws6_540,
            &params_540,
        )?;
        let qt_joker = GpuCrowdTablesJoker::build(&gpu_univ.d_base_joker, &params_joker)?;

        let crowd_vram_bytes = ((COUNT_649 * 3
            + COUNT_540_TICKETS
            + COUNT_540_DRAWS6 * 2
            + COUNT_JOKER_BASE45 * 5)
            * 8) as u64;
        let total_vram_allocated_bytes = gpu_univ.total_vram_bytes() + crowd_vram_bytes;
        let init_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
        check_rss_guard(0)?;

        Ok(Self {
            ql_root: ql_root.to_path_buf(),
            vault_dir: vault_dir.to_path_buf(),
            mmap_open_ms,
            vault_manifest,
            table_statuses: statuses,
            mmap_649,
            mmap_540_tickets,
            mmap_540_draws6,
            mmap_joker_base45,
            mmap_joker_full24m,
            mmap_cdv3_cat,
            mmap_boot_649,
            mmap_boot_540,
            mmap_boot_joker,
            mmap_cdv3q_cat,
            mmap_c3_ports,
            mmap_c4_mc1m,
            mmap_ckv6_power,
            binom_tbl,
            params_649,
            params_540,
            params_joker,
            cdv3_report,
            cdv3q_report,
            pmv3_report,
            c3_report,
            c4_report,
            ckv6_qual_report,
            ckv6_arch_report,
            vram: ResidentVramEngine {
                gpu_univ,
                qt_649,
                qt_540,
                qt_joker,
                total_vram_allocated_bytes,
                init_ms,
            },
        })
    }

    #[inline]
    pub fn masks_649(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_649[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_649,
            )
        }
    }

    #[inline]
    pub fn tickets_540(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_540_tickets[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_540_TICKETS,
            )
        }
    }

    #[inline]
    pub fn draws6_540(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_540_draws6[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_540_DRAWS6,
            )
        }
    }

    #[inline]
    pub fn full_joker(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_joker_full24m[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_FULL24M,
            )
        }
    }

    #[inline]
    pub fn boot6_649(&self) -> &[[f32; 6]] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_boot_649[VAULT_HEADER_BYTES..].as_ptr() as *const [f32; 6],
                COUNT_649,
            )
        }
    }

    #[inline]
    pub fn boot6_540(&self) -> &[[f32; 6]] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_boot_540[VAULT_HEADER_BYTES..].as_ptr() as *const [f32; 6],
                COUNT_540_TICKETS,
            )
        }
    }

    #[inline]
    pub fn boot6_joker(&self) -> &[[f32; 6]] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_boot_joker[VAULT_HEADER_BYTES..].as_ptr() as *const [f32; 6],
                COUNT_JOKER_FULL24M,
            )
        }
    }

    #[inline]
    pub fn ckv6_power_records(&self) -> &[[f64; 8]] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_ckv6_power[VAULT_HEADER_BYTES..].as_ptr() as *const [f64; 8],
                10_000,
            )
        }
    }

    #[inline]
    pub fn c3_portfolio_masks(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_c3_ports[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                13_945,
            )
        }
    }

    #[inline]
    pub fn cdv3_catalog_masks(&self, qap_relabeled: bool) -> &[u64] {
        let mm = if qap_relabeled {
            &self.mmap_cdv3q_cat
        } else {
            &self.mmap_cdv3_cat
        };
        unsafe {
            std::slice::from_raw_parts(mm[VAULT_HEADER_BYTES..].as_ptr() as *const u64, 6_926)
        }
    }

    #[inline]
    pub fn base_joker_masks(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_joker_base45[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_BASE45,
            )
        }
    }

    #[inline]
    pub fn c4_mc1m_records(&self) -> &[[f64; 16]] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_c4_mc1m[VAULT_HEADER_BYTES..].as_ptr() as *const [f64; 16],
                90,
            )
        }
    }

    /// Panoramă instantanee (< 1 ms) peste toate motoarele `S1–S4`, tabelele `.v3bin` și VRAM-ul rezident.
    pub fn overview_snapshot(&self) -> Value {
        let t0 = Instant::now();
        let all_0444 = self.table_statuses.iter().all(|s| s.is_0444 && s.header_valid);
        let total_v3bin_bytes: u64 = self.table_statuses.iter().map(|s| s.file_bytes).sum();

        let main_wheels: Vec<Value> = self
            .cdv3q_report
            .main_designs
            .iter()
            .map(|d| {
                json!({
                    "id": d.id,
                    "game": d.game,
                    "n_tickets": d.n_tickets,
                    "uncovered_gpu": d.uncovered_gpu,
                    "uncovered_cpu": d.uncovered_cpu,
                    "patterned_before": d.patterned_tickets_before,
                    "patterned_after": d.patterned_tickets_after,
                    "mean_ev_before": d.ev_shared_mean_before,
                    "mean_ev_after": d.ev_shared_mean_after,
                    "ev_gain_pct": d.ev_lift_pct,
                })
            })
            .collect();

        let partial_list: Vec<Value> = self
            .cdv3q_report
            .partial_wheels
            .iter()
            .map(|w| {
                let univ = match w.game.as_str() {
                    "649" => COUNT_649 as f64,
                    "540" => COUNT_540_DRAWS6 as f64,
                    _ => COUNT_JOKER_FULL24M as f64,
                };
                json!({
                    "id": w.id,
                    "game": w.game,
                    "target_regime": w.target_regime,
                    "k_budget_tickets": w.k_budget_tickets,
                    "coverage_pct": ((w.covered_draws_s2_gpu as f64) / univ) * 100.0,
                    "patterned_after": w.patterned_tickets_after,
                    "mean_ev_after": w.ev_shared_mean_after,
                })
            })
            .collect();

        let abbrev_list: Vec<Value> = self
            .cdv3q_report
            .abbreviated_wheels
            .iter()
            .map(|a| {
                json!({
                    "id": a.id,
                    "game": a.game,
                    "v_sub": a.v_sub,
                    "p_hit_in_sub": a.p_hit_in_sub,
                    "t_guarantee": a.t_guarantee,
                    "n_tickets": a.n_tickets,
                    "uncovered_sub_draws": a.uncovered_sub_draws,
                    "patterned_after": a.patterned_tickets_after,
                })
            })
            .collect();

        let pm_games: Vec<Value> = self
            .pmv3_report
            .walk_forward_games
            .iter()
            .map(|wf| {
                json!({
                    "game": wf.game,
                    "evaluated_draws": wf.evaluated_draws,
                    "sales_mape_regular_pct": wf.sales_mape_regular_pct,
                    "kappa_post_report_pmv3": wf.kappa_post_report_pmv3,
                    "kappa_post_report_legacy_ceiling": wf.kappa_post_report_legacy_ceiling,
                    "kappa_all_new_fund_pmv3": wf.kappa_all_new_fund_pmv3,
                    "k3_causality_max_abs_diff": wf.k3_causality_max_abs_diff,
                })
            })
            .collect();

        let pm_cubes: Vec<Value> = self
            .pmv3_report
            .universe_parity_and_cubes
            .iter()
            .map(|c| {
                json!({
                    "game": c.game,
                    "ticket_count": c.ticket_count,
                    "v3bin_filename": c.v3bin_filename,
                    "v3bin_bytes": c.v3bin_bytes,
                    "max_abs_diff_gpu_cpu_f64": c.max_abs_diff_gpu_cpu_f64,
                    "ev_nb_mean": c.ev_nb_mean,
                    "ev_poisson_mean": c.ev_poisson_mean,
                    "ev_jensen_mean": c.ev_jensen_mean,
                    "anti_crowd_ev_ratio_best_vs_1_to_k": c.anti_crowd_ev_ratio_best_vs_1_to_k,
                    "best_ticket_ev": c.best_ticket_ev_nb,
                    "canonical_calendar_ev": c.canonical_calendar_1_to_k_ev_nb,
                })
            })
            .collect();

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        json!({
            "version": "3.0.0",
            "positive_claims_allowed": false,
            "query_ms": elapsed_ms,
            "mmap_open_ms": self.mmap_open_ms,
            "vram_init_ms": self.vram.init_ms,
            "vram_resident_universe_tickets": TOTAL_UNIVERSE_TICKETS,
            "vram_resident_bytes": self.vram.total_vram_allocated_bytes,
            "vault_dir": self.vault_dir.display().to_string(),
            "vault_13_v3bin_all_0444_and_valid": all_0444 && self.table_statuses.len() == 13,
            "vault_13_v3bin_bytes": total_v3bin_bytes,
            "vault_tables": self.table_statuses,
            "main_wheels_cdv3q": main_wheels,
            "partial_wheels_cdv3q": partial_list,
            "abbreviated_wheels_cdv3q": abbrev_list,
            "pmv3_walk_forward": pm_games,
            "pmv3_cubes": pm_cubes,
            "technical_incidents_s4": 1,
            "blind_reserves_verdict": self.ckv6_arch_report.scientific_verdict,
        })
    }

    /// Interogare Zero-Copy `mmap` (`< 1 ms`, în practică `~5–25 µs`) a oricărui bilet din cele `39 077 004` variante!
    pub fn mmap_query_ticket(
        &self,
        game: &str,
        numbers_opt: Option<Vec<u8>>,
        joker_opt: Option<u8>,
        rank_opt: Option<usize>,
    ) -> Result<Value, String> {
        let t0 = Instant::now();
        match game {
            "649" => {
                let (rank, mask, nums) = if let Some(mut ns) = numbers_opt {
                    ns.sort_unstable();
                    ns.dedup();
                    if ns.len() != 6 || ns.iter().any(|&x| !(1..=49).contains(&x)) {
                        return Err("6/49 cere exact 6 numere distincte in 1..=49".to_string());
                    }
                    let mut m = 0u64;
                    for &x in &ns {
                        m |= 1u64 << (x - 1);
                    }
                    let rk = pmv3_rank_lex_cpu(m, 49, 6, COUNT_649 as u32, &self.binom_tbl);
                    (rk, m, ns)
                } else {
                    let rk = rank_opt.unwrap_or(0);
                    if rk >= COUNT_649 {
                        return Err(format!("rang 6/49 {rk} >= {COUNT_649}"));
                    }
                    let m = self.masks_649()[rk];
                    (rk, m, mask_to_numbers_1based(m))
                };
                let mmap_mask = self.masks_649()[rank];
                let q6 = self.boot6_649()[rank];
                let lw = pmv3_ticket_lw_cpu(mask, 6, &self.params_649);
                let pop_rel = lw.exp();
                let pat = is_patterned_ticket(&nums, 7);
                let bday = is_pure_1_to_31(&nums);
                let us = t0.elapsed().as_secs_f64() * 1_000_000.0;
                Ok(json!({
                    "game": "649",
                    "universe_size": COUNT_649,
                    "rank": rank,
                    "mask_hex": format!("0x{mmap_mask:013x}"),
                    "mask_verified_equal": mmap_mask == mask,
                    "numbers": nums,
                    "joker": null,
                    "price_lei": 8.0,
                    "ev_nominal": q6[0],
                    "boot_1000_quantiles": {
                        "q05": q6[1], "q25": q6[2], "q50": q6[3], "q75": q6[4], "q95": q6[5]
                    },
                    "crowd_log_weight": lw,
                    "crowd_relative_weight": pop_rel,
                    "is_patterned": pat,
                    "is_birthday_1_31": bday,
                    "query_us": us,
                    "query_ms": us / 1000.0,
                    "under_1ms": (us / 1000.0) < 1.0,
                    "positive_claims_allowed": false,
                }))
            }
            "540" => {
                let (rank, mask, nums) = if let Some(mut ns) = numbers_opt {
                    ns.sort_unstable();
                    ns.dedup();
                    if ns.len() != 5 || ns.iter().any(|&x| !(1..=40).contains(&x)) {
                        return Err("5/40 cere exact 5 numere distincte in 1..=40".to_string());
                    }
                    let mut m = 0u64;
                    for &x in &ns {
                        m |= 1u64 << (x - 1);
                    }
                    let rk =
                        pmv3_rank_lex_cpu(m, 40, 5, COUNT_540_TICKETS as u32, &self.binom_tbl);
                    (rk, m, ns)
                } else {
                    let rk = rank_opt.unwrap_or(0);
                    if rk >= COUNT_540_TICKETS {
                        return Err(format!("rang 5/40 {rk} >= {COUNT_540_TICKETS}"));
                    }
                    let m = self.tickets_540()[rk];
                    (rk, m, mask_to_numbers_1based(m))
                };
                let mmap_mask = self.tickets_540()[rank];
                let q6 = self.boot6_540()[rank];
                let lw = pmv3_ticket_lw_cpu(mask, 5, &self.params_540);
                let pat = is_patterned_ticket(&nums, 8);
                let bday = is_pure_1_to_31(&nums);
                let us = t0.elapsed().as_secs_f64() * 1_000_000.0;
                Ok(json!({
                    "game": "540",
                    "universe_size": COUNT_540_TICKETS,
                    "rank": rank,
                    "mask_hex": format!("0x{mmap_mask:010x}"),
                    "mask_verified_equal": mmap_mask == mask,
                    "numbers": nums,
                    "joker": null,
                    "price_lei": 5.0,
                    "ev_nominal": q6[0],
                    "boot_1000_quantiles": {
                        "q05": q6[1], "q25": q6[2], "q50": q6[3], "q75": q6[4], "q95": q6[5]
                    },
                    "crowd_log_weight": lw,
                    "crowd_relative_weight": lw.exp(),
                    "is_patterned": pat,
                    "is_birthday_1_31": bday,
                    "query_us": us,
                    "query_ms": us / 1000.0,
                    "under_1ms": (us / 1000.0) < 1.0,
                    "positive_claims_allowed": false,
                }))
            }
            "joker" => {
                let (rank, packed, nums, jk) = if let Some(mut ns) = numbers_opt {
                    ns.sort_unstable();
                    ns.dedup();
                    let j = joker_opt.unwrap_or(1);
                    if ns.len() != 5
                        || ns.iter().any(|&x| !(1..=45).contains(&x))
                        || !(1..=20).contains(&j)
                    {
                        return Err(
                            "Joker cere exact 5 numere distincte in 1..=45 si Joker in 1..=20"
                                .to_string(),
                        );
                    }
                    let mut m45 = 0u64;
                    for &x in &ns {
                        m45 |= 1u64 << (x - 1);
                    }
                    let base_rk =
                        pmv3_rank_lex_cpu(m45, 45, 5, COUNT_JOKER_BASE45 as u32, &self.binom_tbl);
                    let full_rk = ((j as usize) - 1) * COUNT_JOKER_BASE45 + base_rk;
                    let pk = pack_joker_ticket(m45, j);
                    (full_rk, pk, ns, j)
                } else {
                    let rk = rank_opt.unwrap_or(0);
                    if rk >= COUNT_JOKER_FULL24M {
                        return Err(format!("rang Joker {rk} >= {COUNT_JOKER_FULL24M}"));
                    }
                    let pk = self.full_joker()[rk];
                    let unp = unpack_joker_ticket(pk);
                    (rk, pk, unp.numbers.to_vec(), unp.joker)
                };
                let mmap_pk = self.full_joker()[rank];
                let base_rk = rank % COUNT_JOKER_BASE45;
                let mmap_base45 = self.base_joker_masks()[base_rk];
                let q6 = self.boot6_joker()[rank];
                let m45 = packed & ((1u64 << 45) - 1);
                let lw = pmv3_ticket_lw_cpu(m45, 5, &self.params_joker);
                let pat = is_patterned_ticket(&nums, 9);
                let bday = is_pure_1_to_31(&nums);
                let us = t0.elapsed().as_secs_f64() * 1_000_000.0;
                Ok(json!({
                    "game": "joker",
                    "universe_size": COUNT_JOKER_FULL24M,
                    "rank": rank,
                    "base45_rank": base_rk,
                    "mask_hex": format!("0x{mmap_pk:014x}"),
                    "mask_verified_equal": mmap_pk == packed && mmap_base45 == m45,
                    "numbers": nums,
                    "joker": jk,
                    "price_lei": 7.0,
                    "ev_nominal": q6[0],
                    "boot_1000_quantiles": {
                        "q05": q6[1], "q25": q6[2], "q50": q6[3], "q75": q6[4], "q95": q6[5]
                    },
                    "crowd_log_weight": lw,
                    "crowd_relative_weight": lw.exp(),
                    "is_patterned": pat,
                    "is_birthday_1_31": bday,
                    "query_us": us,
                    "query_ms": us / 1000.0,
                    "under_1ms": (us / 1000.0) < 1.0,
                    "positive_claims_allowed": false,
                }))
            }
            _ => Err(format!("joc necunoscut: {game}")),
        }
    }

    /// Detaliu roată `CD-V3Q` / `CD-V3` din `mmap` (`cdv3q_wheels_catalog.v3bin` + cuantile bootstrap pe biletele roții).
    pub fn wheel_detail(&self, qap_relabeled: bool, wheel_id: &str) -> Result<Value, String> {
        let t0 = Instant::now();
        let cat_masks = self.cdv3_catalog_masks(qap_relabeled);
        let norm_game = |g: &str| -> &'static str {
            match g {
                "649" | "loto_649" => "649",
                "540" | "loto_540" => "540",
                _ => "joker",
            }
        };

        let inspect_slice = |game: &str, off: usize, len: usize| -> Vec<Value> {
            let g_norm = norm_game(game);
            let max_preview = len.min(200);
            let mut out = Vec::with_capacity(max_preview);
            for idx in 0..max_preview {
                let raw_m = cat_masks[off + idx];
                match g_norm {
                    "649" => {
                        let rk = pmv3_rank_lex_cpu(raw_m, 49, 6, COUNT_649 as u32, &self.binom_tbl);
                        let q = self.boot6_649()[rk];
                        out.push(json!({
                            "i": idx + 1,
                            "numbers": mask_to_numbers_1based(raw_m),
                            "joker": null,
                            "ev_nominal": q[0],
                            "q05": q[1],
                            "q50": q[3],
                            "q95": q[5],
                        }));
                    }
                    "540" => {
                        let rk = pmv3_rank_lex_cpu(
                            raw_m,
                            40,
                            5,
                            COUNT_540_TICKETS as u32,
                            &self.binom_tbl,
                        );
                        let q = self.boot6_540()[rk];
                        out.push(json!({
                            "i": idx + 1,
                            "numbers": mask_to_numbers_1based(raw_m),
                            "joker": null,
                            "ev_nominal": q[0],
                            "q05": q[1],
                            "q50": q[3],
                            "q95": q[5],
                        }));
                    }
                    "joker" => {
                        let unp = unpack_joker_ticket(raw_m);
                        let m45 = raw_m & ((1u64 << 45) - 1);
                        let brk = pmv3_rank_lex_cpu(
                            m45,
                            45,
                            5,
                            COUNT_JOKER_BASE45 as u32,
                            &self.binom_tbl,
                        );
                        let rk = ((unp.joker as usize) - 1) * COUNT_JOKER_BASE45 + brk;
                        let q = self.boot6_joker()[rk];
                        out.push(json!({
                            "i": idx + 1,
                            "numbers": unp.numbers,
                            "joker": unp.joker,
                            "ev_nominal": q[0],
                            "q05": q[1],
                            "q50": q[3],
                            "q95": q[5],
                        }));
                    }
                    _ => {}
                }
            }
            out
        };

        let base_id = wheel_id.strip_suffix("_QAP").unwrap_or(wheel_id);
        let qap_id = format!("{base_id}_QAP");

        if qap_relabeled {
            if let Some(d) = self
                .cdv3q_report
                .main_designs
                .iter()
                .find(|x| x.id == wheel_id || x.id == qap_id)
            {
                let g_norm = norm_game(&d.game);
                let preview = inspect_slice(
                    g_norm,
                    d.vault_offset_records as usize,
                    d.vault_len_records as usize,
                );
                let univ_d = match g_norm {
                    "649" => COUNT_649 as u64,
                    "540" => COUNT_540_DRAWS6 as u64,
                    _ => COUNT_JOKER_FULL24M as u64,
                };
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return Ok(json!({
                    "kind": "main",
                    "qap_relabeled": true,
                    "query_ms": ms,
                    "meta": {
                        "id": d.id, "game": g_norm, "n_tickets": d.n_tickets,
                        "universe_draws": univ_d,
                        "uncovered_gpu": d.uncovered_gpu, "uncovered_cpu": d.uncovered_cpu,
                        "patterned_before": d.patterned_tickets_before, "patterned_after": d.patterned_tickets_after,
                        "mean_ev_before": d.ev_shared_mean_before, "mean_ev_after": d.ev_shared_mean_after,
                        "ev_gain_pct": d.ev_lift_pct,
                        "hit_histogram": d.hit_histogram_s2_gpu,
                    },
                    "tickets_preview": preview,
                    "positive_claims_allowed": false,
                }));
            }
            if let Some(w) = self
                .cdv3q_report
                .partial_wheels
                .iter()
                .find(|x| x.id == wheel_id || x.id == qap_id)
            {
                let g_norm = norm_game(&w.game);
                let preview = inspect_slice(
                    g_norm,
                    w.vault_offset_records as usize,
                    w.vault_len_records as usize,
                );
                let univ_d = match g_norm {
                    "649" => COUNT_649 as u64,
                    "540" => COUNT_540_DRAWS6 as u64,
                    _ => COUNT_JOKER_FULL24M as u64,
                };
                let uncov = univ_d.saturating_sub(w.covered_draws_s2_gpu);
                let cov_pct = ((w.covered_draws_s2_gpu as f64) / (univ_d as f64)) * 100.0;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return Ok(json!({
                    "kind": "partial",
                    "qap_relabeled": true,
                    "query_ms": ms,
                    "meta": {
                        "id": w.id, "game": g_norm, "target_regime": w.target_regime,
                        "k_budget_tickets": w.k_budget_tickets,
                        "universe_draws": univ_d,
                        "covered_draws_gpu": w.covered_draws_s2_gpu,
                        "uncovered_draws": uncov,
                        "coverage_pct": cov_pct,
                        "patterned_before": w.patterned_tickets_before, "patterned_after": w.patterned_tickets_after,
                        "mean_ev_before": w.ev_shared_mean_before, "mean_ev_after": w.ev_shared_mean_after,
                    },
                    "tickets_preview": preview,
                    "positive_claims_allowed": false,
                }));
            }
            if let Some(a) = self
                .cdv3q_report
                .abbreviated_wheels
                .iter()
                .find(|x| x.id == wheel_id || x.id == qap_id)
            {
                let g_norm = norm_game(&a.game);
                let preview = inspect_slice(
                    g_norm,
                    a.vault_offset_records as usize,
                    a.vault_len_records as usize,
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return Ok(json!({
                    "kind": "abbreviated",
                    "qap_relabeled": true,
                    "query_ms": ms,
                    "meta": {
                        "id": a.id, "game": g_norm, "v_sub": a.v_sub,
                        "p_hit_in_sub": a.p_hit_in_sub, "t_guarantee": a.t_guarantee,
                        "n_tickets": a.n_tickets, "sub_universe_draws": a.sub_universe_draws,
                        "uncovered_sub_draws": a.uncovered_sub_draws,
                        "selected_pool_numbers": a.selected_pool_1based,
                        "patterned_after": a.patterned_tickets_after,
                        "mean_ev_after": a.ev_shared_mean_after,
                    },
                    "tickets_preview": preview,
                    "positive_claims_allowed": false,
                }));
            }
        } else {
            if let Some(d) = self
                .cdv3_report
                .main_designs
                .iter()
                .find(|x| x.id == wheel_id || x.id == base_id)
            {
                let g_norm = norm_game(&d.game);
                let preview = inspect_slice(
                    g_norm,
                    d.vault_offset_records as usize,
                    d.vault_len_records as usize,
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return Ok(json!({
                    "kind": "main",
                    "qap_relabeled": false,
                    "query_ms": ms,
                    "meta": {
                        "id": d.id, "game": g_norm, "n_tickets": d.n_tickets,
                        "universe_draws": d.universe_draws,
                        "uncovered_gpu": d.uncovered_gpu, "uncovered_cpu": d.uncovered_cpu,
                        "hit_histogram": d.hit_histogram,
                    },
                    "tickets_preview": preview,
                    "positive_claims_allowed": false,
                }));
            }
            if let Some(w) = self
                .cdv3_report
                .partial_wheels
                .iter()
                .find(|x| x.id == wheel_id || x.id == base_id)
            {
                let g_norm = norm_game(&w.game);
                let preview = inspect_slice(
                    g_norm,
                    w.vault_offset_records as usize,
                    w.vault_len_records as usize,
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return Ok(json!({
                    "kind": "partial",
                    "qap_relabeled": false,
                    "query_ms": ms,
                    "meta": {
                        "id": w.id, "game": g_norm, "target_regime": w.target_regime,
                        "k_budget_tickets": w.k_budget_tickets,
                        "universe_draws": w.universe_draws,
                        "covered_draws_gpu": w.covered_draws_gpu,
                        "uncovered_draws": w.uncovered_draws,
                        "coverage_pct": w.coverage_pct,
                    },
                    "tickets_preview": preview,
                    "positive_claims_allowed": false,
                }));
            }
            if let Some(a) = self
                .cdv3_report
                .abbreviated_wheels
                .iter()
                .find(|x| x.id == wheel_id || x.id == base_id)
            {
                let g_norm = norm_game(&a.game);
                let preview = inspect_slice(
                    g_norm,
                    a.vault_offset_records as usize,
                    a.vault_len_records as usize,
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return Ok(json!({
                    "kind": "abbreviated",
                    "qap_relabeled": false,
                    "query_ms": ms,
                    "meta": {
                        "id": a.id, "game": g_norm, "v_sub": a.v_sub,
                        "p_hit_in_sub": a.p_hit_in_sub, "t_guarantee": a.t_guarantee,
                        "n_tickets": a.n_tickets, "sub_universe_draws": a.sub_universe_draws,
                        "uncovered_sub_draws": a.uncovered_sub_draws,
                        "selected_pool_numbers": (1..=(a.v_sub as u8)).collect::<Vec<u8>>(),
                    },
                    "tickets_preview": preview,
                    "positive_claims_allowed": false,
                }));
            }
        }

        Err(format!("roata inexistenta in catalog: {wheel_id}"))
    }

    /// Explorator Zero-Copy `mmap` (`< 1 ms`) al frontierelor Pareto `C3-v2` și al celor `30` strategii `C4-v2` (`1M` traiectorii).
    pub fn c3_c4_explorer(&self, game: &str, budget_opt: Option<u32>) -> Value {
        let t0 = Instant::now();
        let c3_masks = self.c3_portfolio_masks();

        let frontiers: Vec<Value> = self
            .c3_report
            .single_game_frontiers
            .iter()
            .filter(|f| game == "all" || f.game == game)
            .filter(|f| budget_opt.map(|b| f.budget_lei == b).unwrap_or(true))
            .map(|f| {
                let pts: Vec<Value> = f
                    .points
                    .iter()
                    .map(|pt| {
                        let off = pt.vault_offset_records as usize;
                        let len = (pt.vault_len_records as usize).min(25);
                        let mut sample_tks = Vec::with_capacity(len);
                        for i in 0..len {
                            let m = c3_masks[off + i];
                            if f.game == "joker" {
                                let u = unpack_joker_ticket(m);
                                sample_tks.push(json!({"numbers": u.numbers, "joker": u.joker}));
                            } else {
                                sample_tks.push(json!({"numbers": mask_to_numbers_1based(m)}));
                            }
                        }
                        json!({
                            "mu_label": pt.regime_label,
                            "mu_value": pt.mu,
                            "k_tickets": pt.k_tickets,
                            "stake_lei": pt.actual_cost_lei,
                            "ev_portfolio": pt.ev_portfolio_gpu,
                            "ev_per_lei": pt.return_ratio_rho,
                            "p_any_exact": pt.p_any_exact,
                            "p_random_analytical": f.comparators.p_random_analytical,
                            "bonferroni_sum": pt.p_bonferroni_1,
                            "bonferroni_excess": pt.bonferroni_2_abs_error_avoided,
                            "patterned_tickets": pt.patterned_tickets,
                            "gpu_cpu_ev_abs_diff": pt.gpu_cpu_ev_abs_diff,
                            "sample_tickets": sample_tks,
                        })
                    })
                    .collect();
                json!({
                    "game": f.game,
                    "budget_lei": f.budget_lei,
                    "k_tickets": f.k_tickets,
                    "p_random_analytical": f.comparators.p_random_analytical,
                    "pass_p_any_ge_random": f.all_pos_mu_beat_p_random,
                    "comparators": f.comparators,
                    "pareto_points": pts,
                })
            })
            .collect();

        let mixed: Vec<Value> = self
            .c3_report
            .mixed_3game_portfolios
            .iter()
            .filter(|m| budget_opt.map(|b| m.budget_lei == b).unwrap_or(true))
            .map(|m| {
                json!({
                    "budget_lei": m.budget_lei,
                    "mu_label": m.regime_label,
                    "k_649": m.k_649,
                    "k_540": m.k_540,
                    "k_joker": m.k_joker,
                    "stake_used_lei": m.actual_cost_lei,
                    "ev_total": m.ev_mix_total,
                    "p_any_combined_independent": m.p_any_mix_exact,
                    "p_random_equal_split": m.p_random_mix_analytical,
                })
            })
            .collect();

        let c4_strats: Vec<Value> = self
            .c4_report
            .strategies
            .iter()
            .filter(|s| {
                game == "all"
                    || game == "mix"
                    || s.games_label == game
                    || s.games_label.contains(game)
                    || s.games_label == "649+540+joker"
            })
            .map(|s| {
                json!({
                    "strategy_id": s.strategy_id,
                    "category_group": s.category_group,
                    "games": s.games_label,
                    "stake_per_draw_lei": s.stake_per_draw_lei,
                    "p_any_single_draw_mc": s.p_win_single_draw_sim,
                    "p_any_single_draw_exact": s.p_win_single_draw_exact,
                    "draw0_z_score": s.draw0_z_score_exact_se,
                    "horizons": s.horizons,
                })
            })
            .collect();

        let c4_mmap_recs = self.c4_mc1m_records();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        json!({
            "query_ms": ms,
            "under_1ms": ms < 1.0,
            "single_game_frontiers": frontiers,
            "mixed_frontiers": mixed,
            "c4_mc1m_strategies": c4_strats,
            "c4_mc1m_mmap_records_count": c4_mmap_recs.len(),
            "c4_total_gpu_sim_ms": self.c4_report.gpu_sim_649_ms + self.c4_report.gpu_sim_540_ms + self.c4_report.gpu_sim_joker_ms,
            "positive_claims_allowed": false,
        })
    }

    /// Evaluare interactivă LIVE pe `NVIDIA GeForce RTX 5060` (`sm_120`) + `24` fire CPU `AVX2`
    /// a oricărei liste de bilete proprii (`1..256` bilete) peste întregul univers din VRAM!
    pub fn gpu_eval_custom_portfolio(
        &self,
        game: &str,
        tickets_numbers: &[Vec<u8>],
        jokers_opt: Option<&[u8]>,
    ) -> Result<Value, String> {
        if tickets_numbers.is_empty() || tickets_numbers.len() > 256 {
            return Err("Introduceti intre 1 si 256 de bilete pentru evaluarea live pe GPU!".to_string());
        }
        check_rss_guard(0)?;

        match game {
            "649" => {
                let mut masks = Vec::with_capacity(tickets_numbers.len());
                let mut pat_cnt = 0usize;
                let mut ticket_rows = Vec::with_capacity(tickets_numbers.len());
                for (i, raw) in tickets_numbers.iter().enumerate() {
                    let mut ns = raw.clone();
                    ns.sort_unstable();
                    ns.dedup();
                    if ns.len() != 6 || ns.iter().any(|&x| !(1..=49).contains(&x)) {
                        return Err(format!(
                            "Biletul #{} invalid: 6/49 cere exact 6 numere distincte in 1..=49",
                            i + 1
                        ));
                    }
                    let mut m = 0u64;
                    for &x in &ns {
                        m |= 1u64 << (x - 1);
                    }
                    if is_patterned_ticket(&ns, 7) {
                        pat_cnt += 1;
                    }
                    let rk = pmv3_rank_lex_cpu(m, 49, 6, COUNT_649 as u32, &self.binom_tbl);
                    let q6 = self.boot6_649()[rk];
                    masks.push(m);
                    ticket_rows.push(json!({
                        "i": i + 1, "numbers": ns, "rank": rk,
                        "ev_nominal": q6[0], "q05": q6[1], "q50": q6[3], "q95": q6[5],
                    }));
                }

                let gpu_out = gpu_c3v2_eval_649(
                    &self.vram.gpu_univ.d_masks_649,
                    &self.vram.qt_649,
                    &self.params_649,
                    &masks,
                )?;
                let (cpu_hist, cpu_uncov, cpu_ms) =
                    cpu_verify_covering_24t(self.masks_649(), &masks, 3);
                let cpu_win = (COUNT_649 as u64).saturating_sub(cpu_uncov);
                let bit_exact_match = gpu_out.win_states == cpu_win;
                let p_any = (gpu_out.win_states as f64) / (COUNT_649 as f64);
                let p_rand = p_random_analytical_hypergeom(COUNT_649 as u64, 260_624, masks.len());
                let stake = (masks.len() as f64) * 8.0;

                Ok(json!({
                    "game": "649",
                    "k_tickets": masks.len(),
                    "stake_lei": stake,
                    "universe_states": COUNT_649,
                    "win_states_gpu": gpu_out.win_states,
                    "win_states_cpu_24t": cpu_win,
                    "uncovered_states": cpu_uncov,
                    "gpu_cpu_bit_exact": bit_exact_match,
                    "p_any_exact": p_any,
                    "p_random_analytical": p_rand,
                    "ev_portfolio_nb2": gpu_out.ev_portfolio,
                    "ev_per_lei": gpu_out.ev_portfolio / stake,
                    "ev_by_cat": {
                        "I": gpu_out.ev_by_cat[0],
                        "II": gpu_out.ev_by_cat[1],
                        "III": gpu_out.ev_by_cat[2],
                        "IV": gpu_out.ev_by_cat[3],
                    },
                    "hit_histogram_0_to_6": &cpu_hist[0..=6],
                    "patterned_tickets": pat_cnt,
                    "gpu_ms": gpu_out.elapsed_ms,
                    "cpu_24t_ms": cpu_ms,
                    "tickets": ticket_rows,
                    "positive_claims_allowed": false,
                }))
            }
            "540" => {
                let mut masks = Vec::with_capacity(tickets_numbers.len());
                let mut pat_cnt = 0usize;
                let mut ticket_rows = Vec::with_capacity(tickets_numbers.len());
                for (i, raw) in tickets_numbers.iter().enumerate() {
                    let mut ns = raw.clone();
                    ns.sort_unstable();
                    ns.dedup();
                    if ns.len() != 5 || ns.iter().any(|&x| !(1..=40).contains(&x)) {
                        return Err(format!(
                            "Biletul #{} invalid: 5/40 cere exact 5 numere distincte in 1..=40",
                            i + 1
                        ));
                    }
                    let mut m = 0u64;
                    for &x in &ns {
                        m |= 1u64 << (x - 1);
                    }
                    if is_patterned_ticket(&ns, 8) {
                        pat_cnt += 1;
                    }
                    let rk = pmv3_rank_lex_cpu(
                        m,
                        40,
                        5,
                        COUNT_540_TICKETS as u32,
                        &self.binom_tbl,
                    );
                    let q6 = self.boot6_540()[rk];
                    masks.push(m);
                    ticket_rows.push(json!({
                        "i": i + 1, "numbers": ns, "rank": rk,
                        "ev_nominal": q6[0], "q05": q6[1], "q50": q6[3], "q95": q6[5],
                    }));
                }

                let gpu_out = gpu_c3v2_eval_540(
                    &self.vram.gpu_univ.d_draws6_540,
                    &self.vram.qt_540,
                    &self.params_540,
                    &masks,
                )?;
                let (cpu_hist, cpu_uncov, cpu_ms) =
                    cpu_verify_covering_24t(self.draws6_540(), &masks, 4);
                let cpu_win = (COUNT_540_DRAWS6 as u64).saturating_sub(cpu_uncov);
                let bit_exact_match = gpu_out.win_states == cpu_win;
                let p_any = (gpu_out.win_states as f64) / (COUNT_540_DRAWS6 as f64);
                let p_rand =
                    p_random_analytical_hypergeom(COUNT_540_DRAWS6 as u64, 3_010, masks.len());
                let stake = (masks.len() as f64) * 5.0;

                Ok(json!({
                    "game": "540",
                    "k_tickets": masks.len(),
                    "stake_lei": stake,
                    "universe_states": COUNT_540_DRAWS6,
                    "win_states_gpu": gpu_out.win_states,
                    "win_states_cpu_24t": cpu_win,
                    "uncovered_states": cpu_uncov,
                    "gpu_cpu_bit_exact": bit_exact_match,
                    "p_any_exact": p_any,
                    "p_random_analytical": p_rand,
                    "ev_portfolio_nb2": gpu_out.ev_portfolio,
                    "ev_per_lei": gpu_out.ev_portfolio / stake,
                    "ev_by_cat": {
                        "I": gpu_out.ev_by_cat[0],
                        "II": gpu_out.ev_by_cat[1],
                        "III": gpu_out.ev_by_cat[2],
                    },
                    "hit_histogram_0_to_5": &cpu_hist[0..=5],
                    "patterned_tickets": pat_cnt,
                    "gpu_ms": gpu_out.elapsed_ms,
                    "cpu_24t_ms": cpu_ms,
                    "tickets": ticket_rows,
                    "positive_claims_allowed": false,
                }))
            }
            "joker" => {
                let mut packed = Vec::with_capacity(tickets_numbers.len());
                let mut pat_cnt = 0usize;
                let mut ticket_rows = Vec::with_capacity(tickets_numbers.len());
                for (i, raw) in tickets_numbers.iter().enumerate() {
                    let mut ns = raw.clone();
                    let jk = jokers_opt
                        .and_then(|js| js.get(i).copied())
                        .unwrap_or(((i % 20) as u8) + 1);
                    ns.sort_unstable();
                    ns.dedup();
                    if ns.len() != 5
                        || ns.iter().any(|&x| !(1..=45).contains(&x))
                        || !(1..=20).contains(&jk)
                    {
                        return Err(format!(
                            "Biletul #{} invalid: Joker cere 5 numere in 1..=45 si Joker in 1..=20",
                            i + 1
                        ));
                    }
                    let mut m45 = 0u64;
                    for &x in &ns {
                        m45 |= 1u64 << (x - 1);
                    }
                    if is_patterned_ticket(&ns, 9) {
                        pat_cnt += 1;
                    }
                    let pk = pack_joker_ticket(m45, jk);
                    let brk = pmv3_rank_lex_cpu(
                        m45,
                        45,
                        5,
                        COUNT_JOKER_BASE45 as u32,
                        &self.binom_tbl,
                    );
                    let rk = ((jk as usize) - 1) * COUNT_JOKER_BASE45 + brk;
                    let q6 = self.boot6_joker()[rk];
                    packed.push(pk);
                    ticket_rows.push(json!({
                        "i": i + 1, "numbers": ns, "joker": jk, "rank": rk,
                        "ev_nominal": q6[0], "q05": q6[1], "q50": q6[3], "q95": q6[5],
                    }));
                }

                let gpu_out = gpu_c3v2_eval_joker(
                    &self.vram.gpu_univ.d_base_joker,
                    &self.vram.qt_joker,
                    &self.params_joker,
                    &packed,
                )?;
                let (cpu_cats, cpu_uncov, cpu_ms) =
                    cpu_verify_joker_24m_24t(self.full_joker(), &packed);
                let cpu_win = (COUNT_JOKER_FULL24M as u64).saturating_sub(cpu_uncov);
                let bit_exact_match = gpu_out.win_states == cpu_win;
                let p_any = (gpu_out.win_states as f64) / (COUNT_JOKER_FULL24M as f64);
                let p_rand = p_random_analytical_hypergeom(
                    COUNT_JOKER_FULL24M as u64,
                    694_380,
                    packed.len(),
                );
                let stake = (packed.len() as f64) * 7.0;

                Ok(json!({
                    "game": "joker",
                    "k_tickets": packed.len(),
                    "stake_lei": stake,
                    "universe_states": COUNT_JOKER_FULL24M,
                    "win_states_gpu": gpu_out.win_states,
                    "win_states_cpu_24t": cpu_win,
                    "uncovered_states": cpu_uncov,
                    "gpu_cpu_bit_exact": bit_exact_match,
                    "p_any_exact": p_any,
                    "p_random_analytical": p_rand,
                    "ev_portfolio_nb2": gpu_out.ev_portfolio,
                    "ev_per_lei": gpu_out.ev_portfolio / stake,
                    "ev_by_cat": {
                        "I": gpu_out.ev_by_cat[0], "II": gpu_out.ev_by_cat[1],
                        "III": gpu_out.ev_by_cat[2], "IV": gpu_out.ev_by_cat[3],
                        "V": gpu_out.ev_by_cat[4], "VI": gpu_out.ev_by_cat[5],
                        "VII": gpu_out.ev_by_cat[6], "VIII": gpu_out.ev_by_cat[7],
                    },
                    "cat_states_0_to_8": cpu_cats,
                    "patterned_tickets": pat_cnt,
                    "gpu_ms": gpu_out.elapsed_ms,
                    "cpu_24t_ms": cpu_ms,
                    "tickets": ticket_rows,
                    "positive_claims_allowed": false,
                }))
            }
            _ => Err(format!("joc necunoscut: {game}")),
        }
    }

    /// Explorator Zero-Copy `mmap` (`< 1 ms`) al regimurilor cauzale `CK-V6` (`Q1–Q6`, suprafața `ckv6_power_10k.v3bin`,
    /// cele `13` fluxuri reale cu `45 206` pași și rezervele oarbe `0400`).
    pub fn ckv6_explorer(&self, stream_id_opt: Option<&str>, cell_idx_opt: Option<usize>) -> Value {
        let t0 = Instant::now();
        let all_streams: Vec<&crate::ckv6::StreamWalkForwardReport> = self
            .ckv6_arch_report
            .ro_exploratory_streams
            .iter()
            .chain(self.ckv6_arch_report.intl_non_reserved_controls.iter())
            .chain(self.ckv6_arch_report.xi_p2_dev_streams.iter())
            .collect();

        let sel_id = stream_id_opt.unwrap_or("649");
        let sel_stream = all_streams
            .iter()
            .copied()
            .find(|s| s.code == sel_id)
            .or_else(|| all_streams.first().copied());

        let c_idx = cell_idx_opt.unwrap_or(12).min(24);
        let p_recs = self.ckv6_power_records();
        let slice_400 = &p_recs[c_idx * (POWER_REPS_PER_CELL as usize)
            ..(c_idx + 1) * (POWER_REPS_PER_CELL as usize)];
        let sample_20: Vec<Value> = slice_400
            .iter()
            .take(20)
            .enumerate()
            .map(|(r, v)| {
                json!({
                    "rep": r,
                    "delta": v[0],
                    "l_regime": v[1],
                    "log_m_final": v[3],
                    "max_log_m": v[4],
                    "min_e": v[5],
                    "cp_err": v[6],
                    "log_m_global": v[7],
                })
            })
            .collect();

        let stream_list: Vec<Value> = all_streams
            .iter()
            .map(|s| {
                json!({
                    "stream_id": s.code,
                    "name": s.name,
                    "category": s.family,
                    "n_balls": s.n_balls,
                    "r_draw": s.r_draw,
                    "canonical_steps": s.n_canonical_steps,
                    "m_final": s.m_t_final,
                    "max_m": s.max_m_t_anytime,
                    "min_e": s.min_e_t,
                    "final_map_r": s.final_map_run_length,
                    "m_post_warmup_100": s.m_t_post_warmup100,
                })
            })
            .collect();

        let blind_live = inspect_blind_reserves_lstat(&self.ql_root).unwrap_or_default();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        json!({
            "query_ms": ms,
            "under_1ms": ms < 1.0,
            "q_tests": self.ckv6_qual_report.q_tests,
            "power_grid_25_cells": self.ckv6_qual_report.power_cells,
            "resolution_thresholds": self.ckv6_qual_report.resolution_thresholds,
            "selected_power_cell_idx": c_idx,
            "selected_power_cell_delta_bp": DELTA_BP_GRID[c_idx / 5],
            "selected_power_cell_l_regime": L_REGIME_GRID[c_idx % 5],
            "selected_power_cell_mmap_samples": sample_20,
            "streams_summary": stream_list,
            "selected_stream": sel_stream,
            "holm_family_p2_dev": self.ckv6_arch_report.holm_dev_table,
            "dev_survivors_holm_1500": self.ckv6_arch_report.dev_survivors_holm_1500,
            "blind_reserves_0400_live": blind_live,
            "verdict": self.ckv6_arch_report.scientific_verdict,
            "positive_claims_allowed": false,
        })
    }

    /// Simulator Interactiv LIVE pe `RTX 5060` (`gpu_ckv6_649_batch_f64`) + verificare paritate CPU pentru `CK-V6`!
    pub fn ckv6_live_gpu_sim(
        &self,
        delta_bp: u32,
        l_regime: u32,
        n_reps: usize,
        seed: u64,
    ) -> Result<Value, String> {
        let dbp = delta_bp.min(5_000);
        let lrg = l_regime.clamp(20, 400);
        let reps = n_reps.clamp(4, 64);
        let t_steps = 240usize;
        check_rss_guard(0)?;

        let mut flat_draws = vec![0u64; reps * t_steps];
        std::thread::scope(|s| {
            for (r, dst) in flat_draws.chunks_mut(t_steps).enumerate() {
                s.spawn(move || {
                    let h = gen_synth_649_history_cpu(seed, 99, r as u32, dbp, lrg, t_steps);
                    dst.copy_from_slice(&h);
                });
            }
        });

        let mcp = if dbp > 0 && (lrg as usize) < t_steps {
            lrg
        } else {
            0
        };
        let gpu_res = gpu_ckv6_649_batch_f64(&flat_draws, reps, t_steps, mcp, true)?;
        let cpu_0 = eval_649_history_cpu(&flat_draws[0..t_steps], mcp, true);

        let g0 = gpu_res.summary5[0];
        let gpu_cpu_diff = (cpu_0.log_m_final - g0[0])
            .abs()
            .max((cpu_0.max_log_m - g0[1]).abs())
            .max((cpu_0.min_e - g0[2]).abs());

        let ln1500 = 1500.0f64.ln();
        let hits_1500 = gpu_res.summary5.iter().filter(|s| s[0] >= ln1500).count();
        let min_e_all = gpu_res
            .summary5
            .iter()
            .map(|s| s[2])
            .fold(f64::INFINITY, f64::min);
        let mut sorted_logm: Vec<f64> = gpu_res.summary5.iter().map(|s| s[0]).collect();
        sorted_logm.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_logm = sorted_logm[reps / 2];

        let traj0_points: Vec<Value> = (0..t_steps)
            .map(|t| {
                json!({
                    "step": t + 1,
                    "e_step": gpu_res.step_e[t],
                    "log_m": gpu_res.step_log_m[t],
                    "map_r": gpu_res.step_map_r[t],
                })
            })
            .collect();

        Ok(json!({
            "delta_bp": dbp,
            "delta": (dbp as f64) / 10_000.0,
            "l_regime": lrg,
            "t_steps": t_steps,
            "n_reps": reps,
            "seed": seed,
            "power_m_ge_1500": (hits_1500 as f64) / (reps as f64),
            "median_log_m_final": med_logm,
            "min_e_observed": min_e_all,
            "shield_w0_050_respected": min_e_all >= 0.50,
            "gpu_ms": gpu_res.elapsed_ms,
            "gpu_cpu_witness0_max_diff": gpu_cpu_diff,
            "witness0_trajectory": traj0_points,
            "positive_claims_allowed": false,
        }))
    }
}

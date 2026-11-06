//! Vault Activ `mmap` pe `./vault/` (39,077,004 bilete + 3,838,380 extrageri 5/40)
//! + Poarta A0: Paritate exactă la bit între `RTX 5060 sm_120 (__popcll)` și `i7-14650HX 24T AVX2/POPCNT`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::{Mmap, MmapMut};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cuda_ffi::{
    check_rss_guard, gpu_joker_parity_24m, gpu_popcnt_parity, query_device_info,
    read_process_rss_bytes, CudaDeviceInfo, DeviceBuffer, PinnedBuffer, ResidentGpuUniverse,
    MAX_SAFE_RSS_BYTES,
};

pub const DEFAULT_VAULT_DIR: &str = "./vault";
pub const VAULT_MAGIC: &[u8; 8] = b"QLV3VLT\x01";
pub const VAULT_HEADER_BYTES: usize = 64;

pub const COUNT_649: usize = 13_983_816;
pub const COUNT_540_TICKETS: usize = 658_008;
pub const COUNT_540_DRAWS6: usize = 3_838_380;
pub const COUNT_JOKER_BASE45: usize = 1_221_759;
pub const COUNT_JOKER_FULL24M: usize = 24_435_180;
pub const TOTAL_UNIVERSE_TICKETS: usize = COUNT_649 + COUNT_540_TICKETS + COUNT_JOKER_FULL24M;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VaultFileRecord {
    pub filename: String,
    pub table_kind: u32,
    pub n_pool: u32,
    pub k_pick: u32,
    pub record_count: u64,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VaultManifest {
    pub version: u32,
    pub vault_dir: String,
    pub total_universe_tickets: u64,
    pub positive_claims_allowed: bool,
    pub cpu_threads_used: usize,
    pub generation_seconds: f64,
    pub files: Vec<VaultFileRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateA0Report {
    pub gate: String,
    pub pass: bool,
    pub positive_claims_allowed: bool,
    pub gpu: CudaDeviceInfo,
    pub cpu_threads: usize,
    pub vault_dir: String,
    pub total_resident_tickets: u64,
    pub resident_vram_bytes: u64,
    pub dma_pinned_chunk_bytes: u64,
    pub vram_upload_ms: f64,
    pub rss_current_bytes: u64,
    pub rss_peak_hwm_bytes: u64,
    pub rss_limit_bytes: u64,
    pub rss_guard_ok: bool,
    pub parity_649_zero_diff: bool,
    pub parity_540_tickets_zero_diff: bool,
    pub parity_540_draws6_zero_diff: bool,
    pub parity_joker_24m_zero_diff: bool,
    pub gpu_649_ms: f64,
    pub cpu_649_ms: f64,
    pub gpu_540_draws_ms: f64,
    pub cpu_540_draws_ms: f64,
    pub gpu_joker_24m_ms: f64,
    pub cpu_joker_24m_ms: f64,
    pub gpu_throughput_gops_per_sec: f64,
}

#[inline]
pub fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut res = 1usize;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

fn encode_header(
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

pub fn verify_header(
    bytes: &[u8],
    expected_kind: u32,
    expected_n: u32,
    expected_k: u32,
    expected_count: usize,
) -> Result<(), String> {
    if bytes.len() != VAULT_HEADER_BYTES + expected_count * 8 {
        return Err(format!(
            "dimensiune fisier v3bin invalida: {} != {}",
            bytes.len(),
            VAULT_HEADER_BYTES + expected_count * 8
        ));
    }
    if &bytes[0..8] != VAULT_MAGIC {
        return Err("magic QLV3VLT invalid".to_string());
    }
    let ver = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let kind = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let n_pool = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let k_pick = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let rec_cnt = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    if ver != 3
        || kind != expected_kind
        || n_pool != expected_n
        || k_pick != expected_k
        || rec_cnt != expected_count as u64
    {
        return Err(format!(
            "header v3bin neasteptat: ver={ver}, kind={kind}, n={n_pool}, k={k_pick}, cnt={rec_cnt}"
        ));
    }
    Ok(())
}

/// Generează în paralel pe toate cele 24 de fire toate k-submulțimile lui `1..=n` în ordine lexicografică.
/// Partajarea pe primul element `first = 0..(n-k+1)` garantează ordine lexicografică globală exactă!
pub fn generate_comb_masks_parallel_to_slice(n: usize, k: usize, out: &mut [u64]) {
    assert_eq!(out.len(), binom(n, k));
    assert!(k >= 1 && k <= 6 && n <= 49);

    // Calculăm offsetul exact pentru fiecare alegere a primului element `a0` în `0..=(n-k)`
    let mut tasks: Vec<(usize, &mut [u64])> = Vec::with_capacity(n - k + 1);
    let mut rem_slice = out;
    for a0 in 0..=(n - k) {
        let cnt = binom(n - 1 - a0, k - 1);
        let (head, tail) = rem_slice.split_at_mut(cnt);
        tasks.push((a0, head));
        rem_slice = tail;
    }
    assert!(rem_slice.is_empty());

    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let next_task = std::sync::atomic::AtomicUsize::new(0);
    let tasks_mutex: Vec<std::sync::Mutex<Option<(usize, &mut [u64])>>> =
        tasks.into_iter().map(|t| std::sync::Mutex::new(Some(t))).collect();

    std::thread::scope(|s| {
        for _ in 0..n_threads {
            let tasks_ref = &tasks_mutex;
            let idx_ref = &next_task;
            s.spawn(move || loop {
                let idx = idx_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx >= tasks_ref.len() {
                    break;
                }
                let Some((a0, dst)) = tasks_ref[idx].lock().unwrap().take() else {
                    continue;
                };
                fill_subcombinations(n, k, a0, dst);
            });
        }
    });
}

fn fill_subcombinations(n: usize, k: usize, a0: usize, dst: &mut [u64]) {
    let m0 = 1u64 << a0;
    let mut pos = 0usize;
    match k {
        5 => {
            for a1 in (a0 + 1)..=(n - 4) {
                let m1 = m0 | (1u64 << a1);
                for a2 in (a1 + 1)..=(n - 3) {
                    let m2 = m1 | (1u64 << a2);
                    for a3 in (a2 + 1)..=(n - 2) {
                        let m3 = m2 | (1u64 << a3);
                        for a4 in (a3 + 1)..=(n - 1) {
                            dst[pos] = m3 | (1u64 << a4);
                            pos += 1;
                        }
                    }
                }
            }
        }
        6 => {
            for a1 in (a0 + 1)..=(n - 5) {
                let m1 = m0 | (1u64 << a1);
                for a2 in (a1 + 1)..=(n - 4) {
                    let m2 = m1 | (1u64 << a2);
                    for a3 in (a2 + 1)..=(n - 3) {
                        let m3 = m2 | (1u64 << a3);
                        for a4 in (a3 + 1)..=(n - 2) {
                            let m4 = m3 | (1u64 << a4);
                            for a5 in (a4 + 1)..=(n - 1) {
                                dst[pos] = m4 | (1u64 << a5);
                                pos += 1;
                            }
                        }
                    }
                }
            }
        }
        _ => unreachable!("doar k=5 sau k=6 pentru tabelele principale"),
    }
    assert_eq!(pos, dst.len());
}

fn sha256_mmap_parallel(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn build_or_reuse_comb_v3bin(
    vault_dir: &Path,
    filename: &str,
    table_kind: u32,
    n_pool: usize,
    k_pick: usize,
    expected_count: usize,
) -> Result<VaultFileRecord, String> {
    let path = vault_dir.join(filename);
    let total_bytes = VAULT_HEADER_BYTES + expected_count * 8;

    let need_write = if path.exists() {
        let f = File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
        verify_header(&mmap, table_kind, n_pool as u32, k_pick as u32, expected_count).is_err()
    } else {
        true
    };

    if need_write {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        file.set_len(total_bytes as u64)
            .map_err(|e| format!("set_len {}: {e}", path.display()))?;

        let mut mmap = unsafe { MmapMut::map_mut(&file) }
            .map_err(|e| format!("mmap_mut {}: {e}", path.display()))?;
        let hdr = encode_header(
            table_kind,
            n_pool as u32,
            k_pick as u32,
            expected_count as u64,
            8,
        );
        mmap[0..VAULT_HEADER_BYTES].copy_from_slice(&hdr);

        let payload_u64: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(
                mmap[VAULT_HEADER_BYTES..].as_mut_ptr() as *mut u64,
                expected_count,
            )
        };
        generate_comb_masks_parallel_to_slice(n_pool, k_pick, payload_u64);
        mmap.flush()
            .map_err(|e| format!("flush {}: {e}", path.display()))?;
    }

    let f = File::open(&path).map_err(|e| format!("reopen {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("remmap {}: {e}", path.display()))?;
    verify_header(&mmap, table_kind, n_pool as u32, k_pick as u32, expected_count)?;
    let digest = sha256_mmap_parallel(&mmap);

    Ok(VaultFileRecord {
        filename: filename.to_string(),
        table_kind,
        n_pool: n_pool as u32,
        k_pick: k_pick as u32,
        record_count: expected_count as u64,
        file_bytes: total_bytes as u64,
        sha256: digest,
    })
}

fn build_or_reuse_joker_24m_v3bin(
    vault_dir: &Path,
    base45_slice: &[u64],
) -> Result<VaultFileRecord, String> {
    assert_eq!(base45_slice.len(), COUNT_JOKER_BASE45);
    let filename = "joker_full_24m.v3bin";
    let path = vault_dir.join(filename);
    let total_bytes = VAULT_HEADER_BYTES + COUNT_JOKER_FULL24M * 8;

    let need_write = if path.exists() {
        let f = File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
        verify_header(&mmap, 3, 45, 5, COUNT_JOKER_FULL24M).is_err()
    } else {
        true
    };

    if need_write {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        file.set_len(total_bytes as u64)
            .map_err(|e| format!("set_len {}: {e}", path.display()))?;

        let mut mmap = unsafe { MmapMut::map_mut(&file) }
            .map_err(|e| format!("mmap_mut {}: {e}", path.display()))?;
        let hdr = encode_header(3, 45, 5, COUNT_JOKER_FULL24M as u64, 8);
        mmap[0..VAULT_HEADER_BYTES].copy_from_slice(&hdr);

        let payload_u64: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(
                mmap[VAULT_HEADER_BYTES..].as_mut_ptr() as *mut u64,
                COUNT_JOKER_FULL24M,
            )
        };

        // Împărțim cele 20 valori Joker (fiecare bloc de 1,221,759 măști) în paralel pe 24 fire:
        // ordonare canonică: pentru j0 = 0..20, toate cele 1,221,759 măști principale în ordine lexicografică!
        std::thread::scope(|s| {
            for (j0, chunk) in payload_u64.chunks_mut(COUNT_JOKER_BASE45).enumerate() {
                s.spawn(move || {
                    let j_tag = (j0 as u64) << 48;
                    for (dst, &bm) in chunk.iter_mut().zip(base45_slice.iter()) {
                        *dst = bm | j_tag;
                    }
                });
            }
        });

        mmap.flush()
            .map_err(|e| format!("flush {}: {e}", path.display()))?;
    }

    let f = File::open(&path).map_err(|e| format!("reopen {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("remmap {}: {e}", path.display()))?;
    verify_header(&mmap, 3, 45, 5, COUNT_JOKER_FULL24M)?;
    let digest = sha256_mmap_parallel(&mmap);

    Ok(VaultFileRecord {
        filename: filename.to_string(),
        table_kind: 3,
        n_pool: 45,
        k_pick: 5,
        record_count: COUNT_JOKER_FULL24M as u64,
        file_bytes: total_bytes as u64,
        sha256: digest,
    })
}

pub struct MappedVaultUniverse {
    pub vault_dir: PathBuf,
    pub manifest: VaultManifest,
    mmap_649: Mmap,
    mmap_540_tickets: Mmap,
    mmap_540_draws6: Mmap,
    mmap_joker_base45: Mmap,
    mmap_joker_full24m: Mmap,
}

impl MappedVaultUniverse {
    pub fn open_or_build(vault_dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(vault_dir)
            .map_err(|e| format!("mkdir {}: {e}", vault_dir.display()))?;
        check_rss_guard(0)?;

        let t0 = Instant::now();
        let cpu_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(24);

        let rec_649 = build_or_reuse_comb_v3bin(
            vault_dir,
            "loto_649_masks.v3bin",
            1,
            49,
            6,
            COUNT_649,
        )?;
        let rec_540_t = build_or_reuse_comb_v3bin(
            vault_dir,
            "loto_540_tickets.v3bin",
            1,
            40,
            5,
            COUNT_540_TICKETS,
        )?;
        let rec_540_d = build_or_reuse_comb_v3bin(
            vault_dir,
            "loto_540_draws6.v3bin",
            2,
            40,
            6,
            COUNT_540_DRAWS6,
        )?;
        let rec_j_base = build_or_reuse_comb_v3bin(
            vault_dir,
            "joker_base45_masks.v3bin",
            1,
            45,
            5,
            COUNT_JOKER_BASE45,
        )?;

        let f_jb = File::open(vault_dir.join("joker_base45_masks.v3bin"))
            .map_err(|e| format!("open joker_base45: {e}"))?;
        let mmap_joker_base45 = unsafe { Mmap::map(&f_jb) }
            .map_err(|e| format!("mmap joker_base45: {e}"))?;
        let jb_slice: &[u64] = unsafe {
            std::slice::from_raw_parts(
                mmap_joker_base45[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_BASE45,
            )
        };
        let rec_j_full = build_or_reuse_joker_24m_v3bin(vault_dir, jb_slice)?;

        let mf_path = vault_dir.join("VAULT_MANIFEST.json");
        let base_names = [
            "loto_649_masks.v3bin",
            "loto_540_tickets.v3bin",
            "loto_540_draws6.v3bin",
            "joker_base45_masks.v3bin",
            "joker_full_24m.v3bin",
        ];
        let mut files = vec![rec_649, rec_540_t, rec_540_d, rec_j_base, rec_j_full];

        // Păstrăm înregistrările suplimentare din sesiunile S1–S4 (ex. cdv3_wheels_catalog.v3bin,
        // cuburi bootstrap PM-V3, frontiere Pareto C3-v2) dacă fișierele lor există în Vault!
        if let Ok(prev_raw) = fs::read_to_string(&mf_path) {
            if let Ok(prev_mf) = serde_json::from_str::<VaultManifest>(&prev_raw) {
                for extra in prev_mf.files {
                    if !base_names.contains(&extra.filename.as_str())
                        && vault_dir.join(&extra.filename).exists()
                        && !files.iter().any(|f| f.filename == extra.filename)
                    {
                        files.push(extra);
                    }
                }
            }
        }
        let cat_name = "cdv3_wheels_catalog.v3bin";
        let cat_path = vault_dir.join(cat_name);
        if cat_path.exists() && !files.iter().any(|f| f.filename == cat_name) {
            let f_cat =
                File::open(&cat_path).map_err(|e| format!("open {}: {e}", cat_path.display()))?;
            let mm_cat = unsafe { Mmap::map(&f_cat) }
                .map_err(|e| format!("mmap {}: {e}", cat_path.display()))?;
            if mm_cat.len() >= VAULT_HEADER_BYTES && &mm_cat[0..8] == VAULT_MAGIC {
                let table_kind = u32::from_le_bytes(mm_cat[12..16].try_into().unwrap());
                let n_pool = u32::from_le_bytes(mm_cat[16..20].try_into().unwrap());
                let k_pick = u32::from_le_bytes(mm_cat[20..24].try_into().unwrap());
                let record_count = u64::from_le_bytes(mm_cat[24..32].try_into().unwrap());
                files.push(VaultFileRecord {
                    filename: cat_name.to_string(),
                    table_kind,
                    n_pool,
                    k_pick,
                    record_count,
                    file_bytes: mm_cat.len() as u64,
                    sha256: sha256_mmap_parallel(&mm_cat),
                });
            }
        }

        let manifest = VaultManifest {
            version: 3,
            vault_dir: vault_dir.display().to_string(),
            total_universe_tickets: TOTAL_UNIVERSE_TICKETS as u64,
            positive_claims_allowed: false,
            cpu_threads_used: cpu_threads,
            generation_seconds: t0.elapsed().as_secs_f64(),
            files,
        };

        let mut mf = File::create(&mf_path)
            .map_err(|e| format!("create {}: {e}", mf_path.display()))?;
        mf.write_all(
            (serde_json::to_string_pretty(&manifest).unwrap() + "\n").as_bytes(),
        )
        .map_err(|e| format!("write VAULT_MANIFEST.json: {e}"))?;

        let open_mm = |name: &str| -> Result<Mmap, String> {
            let f = File::open(vault_dir.join(name))
                .map_err(|e| format!("open {name}: {e}"))?;
            unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {name}: {e}"))
        };

        let mmap_649 = open_mm("loto_649_masks.v3bin")?;
        let mmap_540_tickets = open_mm("loto_540_tickets.v3bin")?;
        let mmap_540_draws6 = open_mm("loto_540_draws6.v3bin")?;
        let mmap_joker_full24m = open_mm("joker_full_24m.v3bin")?;

        check_rss_guard(0)?;

        Ok(Self {
            vault_dir: vault_dir.to_path_buf(),
            manifest,
            mmap_649,
            mmap_540_tickets,
            mmap_540_draws6,
            mmap_joker_base45,
            mmap_joker_full24m,
        })
    }

    pub fn masks_649(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_649[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_649,
            )
        }
    }

    pub fn tickets_540(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_540_tickets[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_540_TICKETS,
            )
        }
    }

    pub fn draws6_540(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_540_draws6[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_540_DRAWS6,
            )
        }
    }

    pub fn base_joker(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_joker_base45[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_BASE45,
            )
        }
    }

    pub fn full_joker(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_joker_full24m[VAULT_HEADER_BYTES..].as_ptr() as *const u64,
                COUNT_JOKER_FULL24M,
            )
        }
    }

    /// Încarcă întregul univers în cei 8 GB VRAM ai RTX 5060 prin pool DMA page-locked (`cudaHostAlloc`, 64 MiB).
    pub fn upload_to_vram_via_dma(&self) -> Result<(ResidentGpuUniverse, u64), String> {
        let dma_elems = 8 * 1024 * 1024usize; // 64 MiB buffer DMA page-locked
        let mut dma_buf = PinnedBuffer::<u64>::new_zeroed(dma_elems)?;
        let dma_bytes = (dma_elems * 8) as u64;

        let t0 = Instant::now();
        let mut d_masks_649 = DeviceBuffer::<u64>::alloc(COUNT_649)?;
        let mut d_tickets_540 = DeviceBuffer::<u64>::alloc(COUNT_540_TICKETS)?;
        let mut d_draws6_540 = DeviceBuffer::<u64>::alloc(COUNT_540_DRAWS6)?;
        let mut d_base_joker = DeviceBuffer::<u64>::alloc(COUNT_JOKER_BASE45)?;
        let mut d_full_joker = DeviceBuffer::<u64>::alloc(COUNT_JOKER_FULL24M)?;

        d_masks_649.upload_via_dma_chunked(self.masks_649(), &mut dma_buf)?;
        d_tickets_540.upload_via_dma_chunked(self.tickets_540(), &mut dma_buf)?;
        d_draws6_540.upload_via_dma_chunked(self.draws6_540(), &mut dma_buf)?;
        d_base_joker.upload_via_dma_chunked(self.base_joker(), &mut dma_buf)?;
        d_full_joker.upload_via_dma_chunked(self.full_joker(), &mut dma_buf)?;

        let upload_ms = t0.elapsed().as_secs_f64() * 1000.0;
        check_rss_guard(0)?;

        Ok((
            ResidentGpuUniverse {
                d_masks_649,
                d_tickets_540,
                d_draws6_540,
                d_base_joker,
                d_full_joker,
                upload_ms,
            },
            dma_bytes,
        ))
    }
}

/// CPU 24T AVX2/POPCNT referință deterministă pentru histograma suprapunerilor.
pub fn cpu_popcnt_parity_24t(univ: &[u64], probes: &[u64]) -> (Vec<[u64; 8]>, f64) {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_size = (univ.len() + n_threads - 1) / n_threads;

    let partials: Vec<Vec<[u64; 8]>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in univ.chunks(chunk_size.max(1)) {
            handles.push(s.spawn(move || {
                let mut loc = vec![[0u64; 8]; probes.len()];
                for &m in chunk {
                    for (p, &pr) in probes.iter().enumerate() {
                        let h = ((m & pr).count_ones() as usize).min(7);
                        loc[p][h] += 1;
                    }
                }
                loc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out = vec![[0u64; 8]; probes.len()];
    for loc in partials {
        for p in 0..probes.len() {
            for h in 0..8 {
                out[p][h] += loc[p][h];
            }
        }
    }
    (out, t0.elapsed().as_secs_f64() * 1000.0)
}

pub fn cpu_joker_parity_24t(joker_24m: &[u64], probes: &[u64]) -> (Vec<[u64; 12]>, f64) {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);
    let chunk_size = (joker_24m.len() + n_threads - 1) / n_threads;
    const MASK45: u64 = (1u64 << 45) - 1;

    let partials: Vec<Vec<[u64; 12]>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in joker_24m.chunks(chunk_size.max(1)) {
            handles.push(s.spawn(move || {
                let mut loc = vec![[0u64; 12]; probes.len()];
                for &packed in chunk {
                    let m = packed & MASK45;
                    let j0 = (packed >> 48) & 0x1F;
                    for (p, &pr) in probes.iter().enumerate() {
                        let h = (m & (pr & MASK45)).count_ones() as usize;
                        let pj0 = (pr >> 48) & 0x1F;
                        let jm = usize::from(j0 == pj0);
                        loc[p][h * 2 + jm] += 1;
                    }
                }
                loc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out = vec![[0u64; 12]; probes.len()];
    for loc in partials {
        for p in 0..probes.len() {
            for b in 0..12 {
                out[p][b] += loc[p][b];
            }
        }
    }
    (out, t0.elapsed().as_secs_f64() * 1000.0)
}

pub fn run_gate_a0_parity(
    vault: &MappedVaultUniverse,
    gpu_univ: &ResidentGpuUniverse,
    dma_bytes: u64,
) -> Result<GateA0Report, String> {
    let gpu_info = query_device_info()?;
    let cpu_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(24);

    // 8 probe canonice pe fiecare joc (margini, mijloc, prime, pare/impare)
    let probes_649: [u64; 8] = [
        vault.masks_649()[0],
        vault.masks_649()[COUNT_649 / 3],
        vault.masks_649()[2 * COUNT_649 / 3],
        vault.masks_649()[COUNT_649 - 1],
        (1 << 1) | (1 << 2) | (1 << 11) | (1 << 18) | (1 << 33) | (1 << 39),
        (1 << 10) | (1 << 25) | (1 << 34) | (1 << 37) | (1 << 44) | (1 << 46),
        (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24) | (1 << 32) | (1 << 48),
        (1 << 5) | (1 << 13) | (1 << 21) | (1 << 29) | (1 << 37) | (1 << 45),
    ];

    let probes_540: [u64; 8] = [
        vault.tickets_540()[0],
        vault.tickets_540()[COUNT_540_TICKETS / 3],
        vault.tickets_540()[2 * COUNT_540_TICKETS / 3],
        vault.tickets_540()[COUNT_540_TICKETS - 1],
        (1 << 0) | (1 << 9) | (1 << 19) | (1 << 29) | (1 << 39),
        (1 << 2) | (1 << 11) | (1 << 17) | (1 << 26) | (1 << 35),
        (1 << 4) | (1 << 12) | (1 << 20) | (1 << 28) | (1 << 36),
        (1 << 6) | (1 << 14) | (1 << 22) | (1 << 30) | (1 << 38),
    ];

    let probes_joker: [u64; 8] = [
        vault.full_joker()[0],
        vault.full_joker()[COUNT_JOKER_FULL24M / 5],
        vault.full_joker()[2 * COUNT_JOKER_FULL24M / 5],
        vault.full_joker()[3 * COUNT_JOKER_FULL24M / 5],
        vault.full_joker()[4 * COUNT_JOKER_FULL24M / 5],
        vault.full_joker()[COUNT_JOKER_FULL24M - 1],
        ((1 << 0) | (1 << 10) | (1 << 20) | (1 << 30) | (1 << 44)) | (7u64 << 48),
        ((1 << 3) | (1 << 13) | (1 << 23) | (1 << 33) | (1 << 43)) | (19u64 << 48),
    ];

    let (gpu_649, gpu_649_ms) = gpu_popcnt_parity(&gpu_univ.d_masks_649, &probes_649)?;
    let (cpu_649, cpu_649_ms) = cpu_popcnt_parity_24t(vault.masks_649(), &probes_649);
    let parity_649 = gpu_649 == cpu_649;

    let (gpu_540_t, _) = gpu_popcnt_parity(&gpu_univ.d_tickets_540, &probes_540)?;
    let (cpu_540_t, _) = cpu_popcnt_parity_24t(vault.tickets_540(), &probes_540);
    let parity_540_t = gpu_540_t == cpu_540_t;

    let (gpu_540_d, gpu_540_d_ms) = gpu_popcnt_parity(&gpu_univ.d_draws6_540, &probes_540)?;
    let (cpu_540_d, cpu_540_d_ms) = cpu_popcnt_parity_24t(vault.draws6_540(), &probes_540);
    let parity_540_d = gpu_540_d == cpu_540_d;

    let (gpu_j, gpu_j_ms) = gpu_joker_parity_24m(&gpu_univ.d_full_joker, &probes_joker)?;
    let (cpu_j, cpu_j_ms) = cpu_joker_parity_24t(vault.full_joker(), &probes_joker);
    let parity_j = gpu_j == cpu_j;

    let (rss_cur, rss_hwm) = read_process_rss_bytes();
    let rss_ok = rss_cur < MAX_SAFE_RSS_BYTES && rss_hwm < MAX_SAFE_RSS_BYTES;

    let total_evals = 8.0 * ((COUNT_649 + COUNT_540_DRAWS6 + COUNT_JOKER_FULL24M) as f64);
    let total_gpu_sec = (gpu_649_ms + gpu_540_d_ms + gpu_j_ms) / 1000.0;
    let gops = if total_gpu_sec > 0.0 {
        (total_evals / 1e9) / total_gpu_sec
    } else {
        0.0
    };

    let pass = gpu_info.major == 12
        && gpu_info.minor == 0
        && gpu_univ.total_resident_tickets() == TOTAL_UNIVERSE_TICKETS as u64
        && parity_649
        && parity_540_t
        && parity_540_d
        && parity_j
        && rss_ok;

    Ok(GateA0Report {
        gate: "A0_PARITATE_SI_SATURATIE_HARDWARE".to_string(),
        pass,
        positive_claims_allowed: false,
        gpu: gpu_info,
        cpu_threads,
        vault_dir: vault.vault_dir.display().to_string(),
        total_resident_tickets: gpu_univ.total_resident_tickets(),
        resident_vram_bytes: gpu_univ.total_vram_bytes(),
        dma_pinned_chunk_bytes: dma_bytes,
        vram_upload_ms: gpu_univ.upload_ms,
        rss_current_bytes: rss_cur,
        rss_peak_hwm_bytes: rss_hwm,
        rss_limit_bytes: MAX_SAFE_RSS_BYTES,
        rss_guard_ok: rss_ok,
        parity_649_zero_diff: parity_649,
        parity_540_tickets_zero_diff: parity_540_t,
        parity_540_draws6_zero_diff: parity_540_d,
        parity_joker_24m_zero_diff: parity_j,
        gpu_649_ms,
        cpu_649_ms,
        gpu_540_draws_ms: gpu_540_d_ms,
        cpu_540_draws_ms: cpu_540_d_ms,
        gpu_joker_24m_ms: gpu_j_ms,
        cpu_joker_24m_ms: cpu_j_ms,
        gpu_throughput_gops_per_sec: gops,
    })
}

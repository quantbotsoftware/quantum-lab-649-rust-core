//! FFI nativ către `cuda/v3_kernels.cu` (`nvcc -O3 -arch=sm_120`) + Pool DMA Pinned (`cudaHostAlloc`)
//! + Gardă dură `VmRSS < 11.5 GiB` (`12_348_030_976` octeți).
//! `positive_claims_allowed = false`.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::slice;

use serde::{Deserialize, Serialize};

/// Plafon dur RSS din Protocolul S1 (`11.5 GiB` = `12_348_030_976` octeți).
pub const MAX_SAFE_RSS_BYTES: u64 = (115 * 1024 * 1024 * 1024) / 10;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RawCudaDeviceInfo {
    pub device_count: i32,
    pub major: i32,
    pub minor: i32,
    pub sm_count: i32,
    pub max_threads_per_sm: i32,
    pub l2_cache_bytes: i32,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub name: [u8; 128],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct V3PointDraw {
    pub mask: u32,
    pub draw_id: u32,
}

extern "C" {
    fn v3_cuda_get_device_info(out: *mut RawCudaDeviceInfo) -> i32;
    fn v3_cuda_host_alloc(ptr_out: *mut *mut c_void, bytes: u64) -> i32;
    fn v3_cuda_host_free(ptr: *mut c_void) -> i32;
    fn v3_cuda_device_alloc(d_ptr_out: *mut *mut c_void, bytes: u64) -> i32;
    fn v3_cuda_device_free(d_ptr: *mut c_void) -> i32;
    fn v3_cuda_memcpy_h2d(dst_dev: *mut c_void, src_host: *const c_void, bytes: u64) -> i32;
    fn v3_cuda_memcpy_d2h(dst_host: *mut c_void, src_dev: *const c_void, bytes: u64) -> i32;

    fn v3_cuda_popcnt_parity_u64(
        d_univ: *const u64,
        n_univ: u64,
        h_probes: *const u64,
        n_probes: u32,
        h_out_hist: *mut u64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_joker_parity_24m(
        d_joker_24m: *const u64,
        n_univ: u64,
        h_probes: *const u64,
        n_probes: u32,
        h_out_hist12: *mut u64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_verify_covering_u64(
        d_draws: *const u64,
        n_draws: u64,
        h_tickets: *const u64,
        n_tickets: u32,
        h_out_hist8: *mut u64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_verify_joker_24m(
        d_joker_24m: *const u64,
        n_states: u64,
        h_tickets_packed: *const u64,
        n_tickets: u32,
        h_out_cat9: *mut u64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_init_uncov_bits(d_uncov_bits: *mut u64, n_draws: u64) -> i32;

    fn v3_cuda_mark_covered_bits(
        d_draws: *const u64,
        n_draws: u64,
        d_uncov_bits: *mut u64,
        chosen_mask: u64,
        min_hits: i32,
        rem_out: *mut u64,
    ) -> i32;

    fn v3_cuda_greedy_scores_bitpacked(
        d_draws: *const u64,
        n_draws: u64,
        d_uncov_bits: *const u64,
        h_cands: *const u64,
        n_cands: u32,
        min_hits: i32,
        h_gains_out: *mut u32,
    ) -> i32;

    fn v3_cuda_joker_init_uncov20(d_uncov20: *mut u32, n_base: u32) -> i32;

    fn v3_cuda_joker_mark_covered20(
        d_base_draws: *const u64,
        n_base: u32,
        d_uncov20: *mut u32,
        chosen_packed: u64,
        rem_out: *mut u64,
    ) -> i32;

    fn v3_cuda_joker_greedy_scores20(
        d_base_draws: *const u64,
        n_base: u32,
        d_uncov20: *const u32,
        h_cands_packed: *const u64,
        n_cands: u32,
        h_gains_out: *mut u32,
    ) -> i32;

    fn v3_cuda_run_tabu_shared_16384(
        v_pts: u32,
        k_pick: u32,
        t_hit: u32,
        m_draws: u32,
        h_draw_masks: *const u32,
        h_point_draws: *const V3PointDraw,
        deg_per_pt: u32,
        n_blocks: u32,
        h_init_blocks_128: *const u32,
        max_iters: u32,
        base_seed: u64,
        h_best_blocks_128_out: *mut u32,
        h_best_uncov_128_out: *mut u32,
    ) -> i32;

    fn v3_cuda_solve_l27_86(
        h_warm_86: *const u32,
        has_warm: i32,
        total_steps: u32,
        base_seed: u64,
        h_best_86_out: *mut u32,
        best_uncov_out: *mut u32,
    ) -> i32;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub device_count: i32,
    pub compute_capability: String,
    pub major: i32,
    pub minor: i32,
    pub sm_count: i32,
    pub max_threads_per_sm: i32,
    pub l2_cache_bytes: i32,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

pub fn query_device_info() -> Result<CudaDeviceInfo, String> {
    let mut raw = RawCudaDeviceInfo {
        device_count: 0,
        major: 0,
        minor: 0,
        sm_count: 0,
        max_threads_per_sm: 0,
        l2_cache_bytes: 0,
        total_vram_bytes: 0,
        free_vram_bytes: 0,
        name: [0u8; 128],
    };
    let rc = unsafe { v3_cuda_get_device_info(&mut raw) };
    if rc != 0 || raw.device_count <= 0 {
        return Err(format!("CUDA indisponibil (rc={rc}, count={})", raw.device_count));
    }
    let name = unsafe { CStr::from_ptr(raw.name.as_ptr() as *const i8) }
        .to_string_lossy()
        .into_owned();
    Ok(CudaDeviceInfo {
        name,
        device_count: raw.device_count,
        compute_capability: format!("{}.{}", raw.major, raw.minor),
        major: raw.major,
        minor: raw.minor,
        sm_count: raw.sm_count,
        max_threads_per_sm: raw.max_threads_per_sm,
        l2_cache_bytes: raw.l2_cache_bytes,
        total_vram_bytes: raw.total_vram_bytes,
        free_vram_bytes: raw.free_vram_bytes,
    })
}

/// Citește `VmRSS` și `VmHWM` din `/proc/self/status` în octeți.
pub fn read_process_rss_bytes() -> (u64, u64) {
    let Ok(txt) = std::fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let mut rss_kb = 0u64;
    let mut hwm_kb = 0u64;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm_kb = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
        }
    }
    (rss_kb * 1024, hwm_kb * 1024)
}

pub fn check_rss_guard(extra_bytes: u64) -> Result<(u64, u64), String> {
    let (rss, hwm) = read_process_rss_bytes();
    if rss.saturating_add(extra_bytes) >= MAX_SAFE_RSS_BYTES {
        return Err(format!(
            "Garda RSS declansata: VmRSS={} + extra={} >= plafon {}",
            rss, extra_bytes, MAX_SAFE_RSS_BYTES
        ));
    }
    Ok((rss, hwm))
}

/// Buffer host DMA page-locked (`cudaHostAlloc`).
pub struct PinnedBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Copy + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    pub fn new_zeroed(len: usize) -> Result<Self, String> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        check_rss_guard(bytes)?;
        let mut raw_ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { v3_cuda_host_alloc(&mut raw_ptr, bytes.max(8)) };
        if rc != 0 || raw_ptr.is_null() {
            return Err(format!("cudaHostAlloc esuat pentru {bytes} octeti (rc={rc})"));
        }
        unsafe {
            std::ptr::write_bytes(raw_ptr as *mut u8, 0, bytes as usize);
        }
        check_rss_guard(0)?;
        Ok(Self {
            ptr: raw_ptr as *mut T,
            len,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = v3_cuda_host_free(self.ptr as *mut c_void);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Buffer rezident în cei 8 GB GDDR7 VRAM ai RTX 5060.
pub struct DeviceBuffer<T: Copy> {
    d_ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self, String> {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let mut raw_ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { v3_cuda_device_alloc(&mut raw_ptr, bytes.max(8)) };
        if rc != 0 || raw_ptr.is_null() {
            return Err(format!("cudaMalloc esuat pentru {bytes} octeti (rc={rc})"));
        }
        Ok(Self {
            d_ptr: raw_ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    pub fn upload_slice(&mut self, src: &[T]) -> Result<(), String> {
        if src.len() != self.len {
            return Err(format!("upload_slice lungime {} != {}", src.len(), self.len));
        }
        let bytes = (src.len() * std::mem::size_of::<T>()) as u64;
        let rc = unsafe {
            v3_cuda_memcpy_h2d(
                self.d_ptr as *mut c_void,
                src.as_ptr() as *const c_void,
                bytes,
            )
        };
        if rc != 0 {
            return Err(format!("cudaMemcpy H2D esuat (rc={rc})"));
        }
        Ok(())
    }

    pub fn download_slice(&self, dst: &mut [T]) -> Result<(), String> {
        if dst.len() != self.len {
            return Err(format!("download_slice lungime {} != {}", dst.len(), self.len));
        }
        let bytes = (dst.len() * std::mem::size_of::<T>()) as u64;
        let rc = unsafe {
            v3_cuda_memcpy_d2h(
                dst.as_mut_ptr() as *mut c_void,
                self.d_ptr as *const c_void,
                bytes,
            )
        };
        if rc != 0 {
            return Err(format!("cudaMemcpy D2H esuat (rc={rc})"));
        }
        Ok(())
    }

    pub fn upload_via_dma_chunked(&mut self, src: &[T], dma_buf: &mut PinnedBuffer<T>) -> Result<(), String> {
        if src.len() != self.len {
            return Err(format!("upload_via_dma lungime {} != {}", src.len(), self.len));
        }
        let chunk_elems = dma_buf.len().max(1);
        let elem_size = std::mem::size_of::<T>();
        let mut offset = 0usize;
        while offset < src.len() {
            let take = (src.len() - offset).min(chunk_elems);
            dma_buf.as_mut_slice()[..take].copy_from_slice(&src[offset..offset + take]);
            let dst_byte_ptr = unsafe { (self.d_ptr as *mut u8).add(offset * elem_size) };
            let rc = unsafe {
                v3_cuda_memcpy_h2d(
                    dst_byte_ptr as *mut c_void,
                    dma_buf.as_slice().as_ptr() as *const c_void,
                    (take * elem_size) as u64,
                )
            };
            if rc != 0 {
                return Err(format!("DMA cudaMemcpy H2D esuat la offset {offset} (rc={rc})"));
            }
            offset += take;
        }
        Ok(())
    }

    pub fn as_ptr(&self) -> *const T {
        self.d_ptr as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.d_ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.d_ptr.is_null() {
            unsafe {
                let _ = v3_cuda_device_free(self.d_ptr as *mut c_void);
            }
            self.d_ptr = std::ptr::null_mut();
        }
    }
}

/// Universul complet al celor 39,077,004 bilete + 3,838,380 extrageri 5/40 rezident simultan în VRAM.
pub struct ResidentGpuUniverse {
    pub d_masks_649: DeviceBuffer<u64>,
    pub d_tickets_540: DeviceBuffer<u64>,
    pub d_draws6_540: DeviceBuffer<u64>,
    pub d_base_joker: DeviceBuffer<u64>,
    pub d_full_joker: DeviceBuffer<u64>,
    pub upload_ms: f64,
}

impl ResidentGpuUniverse {
    pub fn total_resident_tickets(&self) -> u64 {
        (self.d_masks_649.len() + self.d_tickets_540.len() + self.d_full_joker.len()) as u64
    }

    pub fn total_vram_bytes(&self) -> u64 {
        ((self.d_masks_649.len()
            + self.d_tickets_540.len()
            + self.d_draws6_540.len()
            + self.d_base_joker.len()
            + self.d_full_joker.len())
            * 8) as u64
    }
}

pub fn gpu_popcnt_parity(
    d_univ: &DeviceBuffer<u64>,
    probes: &[u64],
) -> Result<(Vec<[u64; 8]>, f64), String> {
    if probes.is_empty() || probes.len() > 16 {
        return Err("numar probe invalid (1..=16)".to_string());
    }
    let mut flat = vec![0u64; probes.len() * 8];
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_popcnt_parity_u64(
            d_univ.as_ptr(),
            d_univ.len() as u64,
            probes.as_ptr(),
            probes.len() as u32,
            flat.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_popcnt_parity_u64 esuat (rc={rc})"));
    }
    let mut out = Vec::with_capacity(probes.len());
    for p in 0..probes.len() {
        let mut row = [0u64; 8];
        row.copy_from_slice(&flat[p * 8..(p + 1) * 8]);
        out.push(row);
    }
    Ok((out, ms))
}

pub fn gpu_joker_parity_24m(
    d_joker_24m: &DeviceBuffer<u64>,
    probes: &[u64],
) -> Result<(Vec<[u64; 12]>, f64), String> {
    if probes.is_empty() || probes.len() > 16 {
        return Err("numar probe Joker invalid (1..=16)".to_string());
    }
    let mut flat = vec![0u64; probes.len() * 12];
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_joker_parity_24m(
            d_joker_24m.as_ptr(),
            d_joker_24m.len() as u64,
            probes.as_ptr(),
            probes.len() as u32,
            flat.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_joker_parity_24m esuat (rc={rc})"));
    }
    let mut out = Vec::with_capacity(probes.len());
    for p in 0..probes.len() {
        let mut row = [0u64; 12];
        row.copy_from_slice(&flat[p * 12..(p + 1) * 12]);
        out.push(row);
    }
    Ok((out, ms))
}

pub fn gpu_verify_covering(
    d_draws: &DeviceBuffer<u64>,
    tickets: &[u64],
) -> Result<([u64; 8], f64), String> {
    let mut hist = [0u64; 8];
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_verify_covering_u64(
            d_draws.as_ptr(),
            d_draws.len() as u64,
            tickets.as_ptr(),
            tickets.len() as u32,
            hist.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_verify_covering_u64 esuat (rc={rc})"));
    }
    Ok((hist, ms))
}

pub fn gpu_verify_joker_24m(
    d_joker_24m: &DeviceBuffer<u64>,
    tickets_packed: &[u64],
) -> Result<([u64; 9], f64), String> {
    let mut cat = [0u64; 9];
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_verify_joker_24m(
            d_joker_24m.as_ptr(),
            d_joker_24m.len() as u64,
            tickets_packed.as_ptr(),
            tickets_packed.len() as u32,
            cat.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_verify_joker_24m esuat (rc={rc})"));
    }
    Ok((cat, ms))
}

pub struct GpuBitpackedCoverState {
    pub d_uncov_bits: DeviceBuffer<u64>,
    pub n_draws: u64,
}

impl GpuBitpackedCoverState {
    pub fn new(n_draws: u64) -> Result<Self, String> {
        let n_words = ((n_draws + 63) / 64) as usize;
        let mut d_uncov_bits = DeviceBuffer::<u64>::alloc(n_words)?;
        let rc = unsafe { v3_cuda_init_uncov_bits(d_uncov_bits.as_mut_ptr(), n_draws) };
        if rc != 0 {
            return Err(format!("v3_cuda_init_uncov_bits esuat (rc={rc})"));
        }
        Ok(Self {
            d_uncov_bits,
            n_draws,
        })
    }

    pub fn mark_covered(
        &mut self,
        d_draws: &DeviceBuffer<u64>,
        chosen_mask: u64,
        min_hits: i32,
    ) -> Result<u64, String> {
        let mut rem = 0u64;
        let rc = unsafe {
            v3_cuda_mark_covered_bits(
                d_draws.as_ptr(),
                self.n_draws,
                self.d_uncov_bits.as_mut_ptr(),
                chosen_mask,
                min_hits,
                &mut rem,
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_mark_covered_bits esuat (rc={rc})"));
        }
        Ok(rem)
    }

    pub fn score_candidates(
        &self,
        d_draws: &DeviceBuffer<u64>,
        cands: &[u64],
        min_hits: i32,
        gains_out: &mut [u32],
    ) -> Result<(), String> {
        if cands.len() != gains_out.len() {
            return Err("score_candidates lungime diferita".to_string());
        }
        let rc = unsafe {
            v3_cuda_greedy_scores_bitpacked(
                d_draws.as_ptr(),
                self.n_draws,
                self.d_uncov_bits.as_ptr(),
                cands.as_ptr(),
                cands.len() as u32,
                min_hits,
                gains_out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_greedy_scores_bitpacked esuat (rc={rc})"));
        }
        Ok(())
    }
}

pub struct GpuJokerCoverState {
    pub d_uncov20: DeviceBuffer<u32>,
    pub n_base: u32,
}

impl GpuJokerCoverState {
    pub fn new(n_base: u32) -> Result<Self, String> {
        let mut d_uncov20 = DeviceBuffer::<u32>::alloc(n_base as usize)?;
        let rc = unsafe { v3_cuda_joker_init_uncov20(d_uncov20.as_mut_ptr(), n_base) };
        if rc != 0 {
            return Err(format!("v3_cuda_joker_init_uncov20 esuat (rc={rc})"));
        }
        Ok(Self { d_uncov20, n_base })
    }

    pub fn mark_covered(
        &mut self,
        d_base: &DeviceBuffer<u64>,
        chosen_packed: u64,
    ) -> Result<u64, String> {
        let mut rem = 0u64;
        let rc = unsafe {
            v3_cuda_joker_mark_covered20(
                d_base.as_ptr(),
                self.n_base,
                self.d_uncov20.as_mut_ptr(),
                chosen_packed,
                &mut rem,
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_joker_mark_covered20 esuat (rc={rc})"));
        }
        Ok(rem)
    }

    pub fn score_candidates(
        &self,
        d_base: &DeviceBuffer<u64>,
        cands_packed: &[u64],
        gains_out: &mut [u32],
    ) -> Result<(), String> {
        let rc = unsafe {
            v3_cuda_joker_greedy_scores20(
                d_base.as_ptr(),
                self.n_base,
                self.d_uncov20.as_ptr(),
                cands_packed.as_ptr(),
                cands_packed.len() as u32,
                gains_out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_joker_greedy_scores20 esuat (rc={rc})"));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_run_tabu_shared_16384(
    v_pts: u32,
    k_pick: u32,
    t_hit: u32,
    draw_masks: &[u32],
    point_draws: &[V3PointDraw],
    deg_per_pt: u32,
    n_blocks: u32,
    init_blocks_128: &[u32],
    max_iters: u32,
    base_seed: u64,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let expected_init = 128usize * (n_blocks as usize);
    if init_blocks_128.len() != expected_init {
        return Err(format!(
            "init_blocks_128 len {} != {}",
            init_blocks_128.len(),
            expected_init
        ));
    }
    let mut best_blocks = vec![0u32; expected_init];
    let mut best_uncov = vec![0u32; 128];
    let rc = unsafe {
        v3_cuda_run_tabu_shared_16384(
            v_pts,
            k_pick,
            t_hit,
            draw_masks.len() as u32,
            draw_masks.as_ptr(),
            point_draws.as_ptr(),
            deg_per_pt,
            n_blocks,
            init_blocks_128.as_ptr(),
            max_iters,
            base_seed,
            best_blocks.as_mut_ptr(),
            best_uncov.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_run_tabu_shared_16384 esuat (rc={rc})"));
    }
    Ok((best_blocks, best_uncov))
}

#[allow(dead_code)]
pub fn gpu_solve_l27_ground_basin_86(
    warm_86: Option<&[u32; 86]>,
    total_steps: u32,
    base_seed: u64,
) -> Result<([u32; 86], u32), String> {
    let mut best_86 = [0u32; 86];
    let mut best_uncov = u32::MAX;
    let (warm_ptr, has_warm) = match warm_86 {
        Some(w) => (w.as_ptr(), 1i32),
        None => (std::ptr::null(), 0i32),
    };
    let rc = unsafe {
        v3_cuda_solve_l27_86(
            warm_ptr,
            has_warm,
            total_steps,
            base_seed,
            best_86.as_mut_ptr(),
            &mut best_uncov,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_solve_l27_86 esuat (rc={rc})"));
    }
    Ok((best_86, best_uncov))
}

// ============================================================================
// Sesiunea 2 (S2 / Poarta A3) — FFI CUDA sm_120 pentru PM-V3 & Bootstrap 1000x
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct V3CrowdParamsF64 {
    pub gamma: [f64; 50],
    pub theta_cons: f64,
    pub theta_low31: f64,
    pub theta_arith: f64,
    pub theta_grid: f64,
    pub lw_shift: f64,
    pub pi_joker: [f64; 20],
    pub sales_n: f64,
    pub funds: [f64; 8],
    pub nb_r: [f64; 8],
    pub fixed_iv: f64,
    pub grid_cols: i32,
    pub share_mode: i32, // 0 = Negative Binomial exact, 1 = Poisson, 2 = Jensen 1/(1+mu)
}

unsafe extern "C" {
    fn v3_cuda_pmv3_ev_649_f64(
        d_masks_649: *const u64,
        h_params: *const V3CrowdParamsF64,
        h_ev_out: *mut f64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_pmv3_ev_540_f64(
        d_tickets_540: *const u64,
        d_draws6_540: *const u64,
        h_params: *const V3CrowdParamsF64,
        h_ev_out: *mut f64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_pmv3_ev_joker_f64(
        d_base_joker: *const u64,
        h_params: *const V3CrowdParamsF64,
        h_ev_24m_out: *mut f64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_pmv3_boot_649_tile(
        d_masks_649: *const u64,
        h_rep_params: *const V3CrowdParamsF64,
        n_reps: u32,
        tile_start: u32,
        tile_len: u32,
        h_ev_nom_tile: *const f64,
        h_out_rec6: *mut f32,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_pmv3_boot_540_all(
        d_tickets_540: *const u64,
        d_draws6_540: *const u64,
        h_rep_params: *const V3CrowdParamsF64,
        n_reps: u32,
        h_ev_nom: *const f64,
        h_out_rec6: *mut f32,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_pmv3_boot_joker_single_j(
        d_base_joker: *const u64,
        h_rep_params: *const V3CrowdParamsF64,
        n_reps: u32,
        j_ticket: i32,
        h_ev_nom_j: *const f64,
        h_out_rec6: *mut f32,
        elapsed_ms_out: *mut f64,
    ) -> i32;
}

pub fn gpu_pmv3_ev_649_f64(
    d_masks_649: &DeviceBuffer<u64>,
    params: &V3CrowdParamsF64,
    ev_out: &mut [f64],
) -> Result<f64, String> {
    if ev_out.len() != 13_983_816 {
        return Err(format!("ev_out 649 len {} != 13983816", ev_out.len()));
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_pmv3_ev_649_f64(d_masks_649.as_ptr(), params, ev_out.as_mut_ptr(), &mut ms)
    };
    if rc != 0 {
        return Err(format!("v3_cuda_pmv3_ev_649_f64 esuat (rc={rc})"));
    }
    Ok(ms)
}

pub fn gpu_pmv3_ev_540_f64(
    d_tickets_540: &DeviceBuffer<u64>,
    d_draws6_540: &DeviceBuffer<u64>,
    params: &V3CrowdParamsF64,
    ev_out: &mut [f64],
) -> Result<f64, String> {
    if ev_out.len() != 658_008 {
        return Err(format!("ev_out 540 len {} != 658008", ev_out.len()));
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_pmv3_ev_540_f64(
            d_tickets_540.as_ptr(),
            d_draws6_540.as_ptr(),
            params,
            ev_out.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_pmv3_ev_540_f64 esuat (rc={rc})"));
    }
    Ok(ms)
}

pub fn gpu_pmv3_ev_joker_f64(
    d_base_joker: &DeviceBuffer<u64>,
    params: &V3CrowdParamsF64,
    ev_24m_out: &mut [f64],
) -> Result<f64, String> {
    if ev_24m_out.len() != 24_435_180 {
        return Err(format!("ev_24m_out len {} != 24435180", ev_24m_out.len()));
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_pmv3_ev_joker_f64(
            d_base_joker.as_ptr(),
            params,
            ev_24m_out.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_pmv3_ev_joker_f64 esuat (rc={rc})"));
    }
    Ok(ms)
}

pub fn gpu_pmv3_boot_649_tile(
    d_masks_649: &DeviceBuffer<u64>,
    rep_params: &[V3CrowdParamsF64],
    tile_start: usize,
    ev_nom_tile: &[f64],
    out_rec6: &mut [f32],
) -> Result<f64, String> {
    let tile_len = ev_nom_tile.len();
    if out_rec6.len() != tile_len * 6 {
        return Err("out_rec6 649 lungime invalida".to_string());
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_pmv3_boot_649_tile(
            d_masks_649.as_ptr(),
            rep_params.as_ptr(),
            rep_params.len() as u32,
            tile_start as u32,
            tile_len as u32,
            ev_nom_tile.as_ptr(),
            out_rec6.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_pmv3_boot_649_tile esuat (rc={rc})"));
    }
    Ok(ms)
}

pub fn gpu_pmv3_boot_540_all(
    d_tickets_540: &DeviceBuffer<u64>,
    d_draws6_540: &DeviceBuffer<u64>,
    rep_params: &[V3CrowdParamsF64],
    ev_nom: &[f64],
    out_rec6: &mut [f32],
) -> Result<f64, String> {
    if ev_nom.len() != 658_008 || out_rec6.len() != 658_008 * 6 {
        return Err("out_rec6 540 lungime invalida".to_string());
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_pmv3_boot_540_all(
            d_tickets_540.as_ptr(),
            d_draws6_540.as_ptr(),
            rep_params.as_ptr(),
            rep_params.len() as u32,
            ev_nom.as_ptr(),
            out_rec6.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_pmv3_boot_540_all esuat (rc={rc})"));
    }
    Ok(ms)
}

pub fn gpu_pmv3_boot_joker_single_j(
    d_base_joker: &DeviceBuffer<u64>,
    rep_params: &[V3CrowdParamsF64],
    j_ticket: usize,
    ev_nom_j: &[f64],
    out_rec6: &mut [f32],
) -> Result<f64, String> {
    if ev_nom_j.len() != 1_221_759 || out_rec6.len() != 1_221_759 * 6 {
        return Err("out_rec6 joker lungime invalida".to_string());
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_pmv3_boot_joker_single_j(
            d_base_joker.as_ptr(),
            rep_params.as_ptr(),
            rep_params.len() as u32,
            j_ticket as i32,
            ev_nom_j.as_ptr(),
            out_rec6.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_pmv3_boot_joker_single_j esuat (rc={rc})"));
    }
    Ok(ms)
}

// ============================================================================
// Sesiunea 3 (S3 / Poarta A4) — FFI CUDA sm_120 pentru C3-v2 & C4-v2 (1,000,000)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct V3PortfolioEvalOut {
    pub win_states: u64,
    pub pair_overlap_sum: u64,
    pub cat_any_states: [u64; 8],
    pub ev_portfolio: f64,
    pub ev_by_cat: [f64; 8],
    pub second_moment_draw0: f64,
    pub elapsed_ms: f64,
}

unsafe extern "C" {
    fn v3_cuda_c3v2_prep_649(
        d_masks_649: *const u64,
        h_params: *const V3CrowdParamsF64,
        d_q6_out: *mut f64,
        d_q5_out: *mut f64,
        d_q4_out: *mut f64,
    ) -> i32;

    fn v3_cuda_c3v2_prep_540(
        d_tickets_540: *const u64,
        d_draws6_540: *const u64,
        h_params: *const V3CrowdParamsF64,
        d_p5_out: *mut f64,
        d_t5_out: *mut f64,
        d_qiii_out: *mut f64,
    ) -> i32;

    fn v3_cuda_c3v2_prep_joker(
        d_base_joker: *const u64,
        h_params: *const V3CrowdParamsF64,
        d_q5_out: *mut f64,
        d_q4_out: *mut f64,
        d_q3_out: *mut f64,
        d_q2_out: *mut f64,
        d_q1_out: *mut f64,
    ) -> i32;

    fn v3_cuda_c3v2_eval_649(
        d_masks_649: *const u64,
        d_q6: *const f64,
        d_q5: *const f64,
        d_q4: *const f64,
        h_params: *const V3CrowdParamsF64,
        h_tickets: *const u64,
        n_tickets: u32,
        h_out: *mut V3PortfolioEvalOut,
    ) -> i32;

    fn v3_cuda_c3v2_eval_540(
        d_draws6_540: *const u64,
        d_p5: *const f64,
        d_t5: *const f64,
        d_qiii: *const f64,
        h_params: *const V3CrowdParamsF64,
        h_tickets: *const u64,
        n_tickets: u32,
        h_out: *mut V3PortfolioEvalOut,
    ) -> i32;

    fn v3_cuda_c3v2_eval_joker(
        d_base_joker: *const u64,
        d_q5: *const f64,
        d_q4: *const f64,
        d_q3: *const f64,
        d_q2: *const f64,
        d_q1: *const f64,
        h_params: *const V3CrowdParamsF64,
        h_tickets_packed: *const u64,
        n_tickets: u32,
        h_out: *mut V3PortfolioEvalOut,
    ) -> i32;

    fn v3_cuda_c3v2_scan_universe(
        game_code: i32,
        d_tickets: *const u64,
        d_ev_q05_pair: *const f32,
        n_univ: u32,
        h_selected: *const u64,
        n_selected: u32,
        h_penalties16: *const f64,
        w_ev: f64,
        w_q05: f64,
        h_block_scores_out: *mut f64,
        h_block_indices_out: *mut u32,
    ) -> i32;

    fn v3_cuda_c4v2_prep_hits(
        game_code: i32,
        d_draws_or_base: *const u64,
        h_all_tickets: *const u64,
        total_tickets: u32,
        h_port_offs: *const u32,
        h_port_lens: *const u32,
        n_ports: u32,
        d_hits_out: *mut std::ffi::c_void,
    ) -> i32;

    fn v3_cuda_c4v2_compute_rates(
        game_code: i32,
        d_draws6_540: *const u64,
        d_q0: *const f64,
        d_q1: *const f64,
        d_q2: *const f64,
        d_q3: *const f64,
        d_q4: *const f64,
        d_hits: *const std::ffi::c_void,
        h_params: *const V3CrowdParamsF64,
        h_xi16: *const f64,
        n_ports: u32,
        h_rates_out: *mut f64,
    ) -> i32;

    fn v3_cuda_c4v2_sim_chunk(
        game_code: i32,
        d_draws6_540: *const u64,
        d_q0: *const f64,
        d_q1: *const f64,
        d_q2: *const f64,
        d_q3: *const f64,
        d_q4: *const f64,
        d_hits: *const std::ffi::c_void,
        h_params: *const V3CrowdParamsF64,
        h_xi16: *const f64,
        h_rep0_8: *const f64,
        h_fnew_8: *const f64,
        n_ports: u32,
        chunk_start: u32,
        chunk_len: u32,
        seed: u64,
        h_out_pay4: *mut f64,
        h_out_flags: *mut u8,
        elapsed_ms_out: *mut f64,
    ) -> i32;
}

pub struct GpuCrowdTables649 {
    pub d_q6: DeviceBuffer<f64>,
    pub d_q5: DeviceBuffer<f64>,
    pub d_q4: DeviceBuffer<f64>,
}

impl GpuCrowdTables649 {
    pub fn build(
        d_masks_649: &DeviceBuffer<u64>,
        params: &V3CrowdParamsF64,
    ) -> Result<Self, String> {
        let mut d_q6 = DeviceBuffer::<f64>::alloc(13_983_816)?;
        let mut d_q5 = DeviceBuffer::<f64>::alloc(13_983_816)?;
        let mut d_q4 = DeviceBuffer::<f64>::alloc(13_983_816)?;
        let rc = unsafe {
            v3_cuda_c3v2_prep_649(
                d_masks_649.as_ptr(),
                params,
                d_q6.as_mut_ptr(),
                d_q5.as_mut_ptr(),
                d_q4.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_c3v2_prep_649 esuat (rc={rc})"));
        }
        Ok(Self { d_q6, d_q5, d_q4 })
    }
}

pub struct GpuCrowdTables540 {
    pub d_p5: DeviceBuffer<f64>,
    pub d_t5: DeviceBuffer<f64>,
    pub d_qiii: DeviceBuffer<f64>,
}

impl GpuCrowdTables540 {
    pub fn build(
        d_tickets_540: &DeviceBuffer<u64>,
        d_draws6_540: &DeviceBuffer<u64>,
        params: &V3CrowdParamsF64,
    ) -> Result<Self, String> {
        let mut d_p5 = DeviceBuffer::<f64>::alloc(658_008)?;
        let mut d_t5 = DeviceBuffer::<f64>::alloc(3_838_380)?;
        let mut d_qiii = DeviceBuffer::<f64>::alloc(3_838_380)?;
        let rc = unsafe {
            v3_cuda_c3v2_prep_540(
                d_tickets_540.as_ptr(),
                d_draws6_540.as_ptr(),
                params,
                d_p5.as_mut_ptr(),
                d_t5.as_mut_ptr(),
                d_qiii.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_c3v2_prep_540 esuat (rc={rc})"));
        }
        Ok(Self { d_p5, d_t5, d_qiii })
    }
}

pub struct GpuCrowdTablesJoker {
    pub d_q5: DeviceBuffer<f64>,
    pub d_q4: DeviceBuffer<f64>,
    pub d_q3: DeviceBuffer<f64>,
    pub d_q2: DeviceBuffer<f64>,
    pub d_q1: DeviceBuffer<f64>,
}

impl GpuCrowdTablesJoker {
    pub fn build(
        d_base_joker: &DeviceBuffer<u64>,
        params: &V3CrowdParamsF64,
    ) -> Result<Self, String> {
        let mut d_q5 = DeviceBuffer::<f64>::alloc(1_221_759)?;
        let mut d_q4 = DeviceBuffer::<f64>::alloc(1_221_759)?;
        let mut d_q3 = DeviceBuffer::<f64>::alloc(1_221_759)?;
        let mut d_q2 = DeviceBuffer::<f64>::alloc(1_221_759)?;
        let mut d_q1 = DeviceBuffer::<f64>::alloc(1_221_759)?;
        let rc = unsafe {
            v3_cuda_c3v2_prep_joker(
                d_base_joker.as_ptr(),
                params,
                d_q5.as_mut_ptr(),
                d_q4.as_mut_ptr(),
                d_q3.as_mut_ptr(),
                d_q2.as_mut_ptr(),
                d_q1.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("v3_cuda_c3v2_prep_joker esuat (rc={rc})"));
        }
        Ok(Self {
            d_q5,
            d_q4,
            d_q3,
            d_q2,
            d_q1,
        })
    }
}

pub fn gpu_c3v2_eval_649(
    d_masks_649: &DeviceBuffer<u64>,
    qt: &GpuCrowdTables649,
    params: &V3CrowdParamsF64,
    tickets: &[u64],
) -> Result<V3PortfolioEvalOut, String> {
    if tickets.is_empty() || tickets.len() > 512 {
        return Err(format!("gpu_c3v2_eval_649 len invalid: {}", tickets.len()));
    }
    let mut out = V3PortfolioEvalOut::default();
    let rc = unsafe {
        v3_cuda_c3v2_eval_649(
            d_masks_649.as_ptr(),
            qt.d_q6.as_ptr(),
            qt.d_q5.as_ptr(),
            qt.d_q4.as_ptr(),
            params,
            tickets.as_ptr(),
            tickets.len() as u32,
            &mut out,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c3v2_eval_649 esuat (rc={rc})"));
    }
    Ok(out)
}

pub fn gpu_c3v2_eval_540(
    d_draws6_540: &DeviceBuffer<u64>,
    qt: &GpuCrowdTables540,
    params: &V3CrowdParamsF64,
    tickets: &[u64],
) -> Result<V3PortfolioEvalOut, String> {
    if tickets.is_empty() || tickets.len() > 512 {
        return Err(format!("gpu_c3v2_eval_540 len invalid: {}", tickets.len()));
    }
    let mut out = V3PortfolioEvalOut::default();
    let rc = unsafe {
        v3_cuda_c3v2_eval_540(
            d_draws6_540.as_ptr(),
            qt.d_p5.as_ptr(),
            qt.d_t5.as_ptr(),
            qt.d_qiii.as_ptr(),
            params,
            tickets.as_ptr(),
            tickets.len() as u32,
            &mut out,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c3v2_eval_540 esuat (rc={rc})"));
    }
    Ok(out)
}

pub fn gpu_c3v2_eval_joker(
    d_base_joker: &DeviceBuffer<u64>,
    qt: &GpuCrowdTablesJoker,
    params: &V3CrowdParamsF64,
    tickets_packed: &[u64],
) -> Result<V3PortfolioEvalOut, String> {
    if tickets_packed.is_empty() || tickets_packed.len() > 512 {
        return Err(format!(
            "gpu_c3v2_eval_joker len invalid: {}",
            tickets_packed.len()
        ));
    }
    let mut out = V3PortfolioEvalOut::default();
    let rc = unsafe {
        v3_cuda_c3v2_eval_joker(
            d_base_joker.as_ptr(),
            qt.d_q5.as_ptr(),
            qt.d_q4.as_ptr(),
            qt.d_q3.as_ptr(),
            qt.d_q2.as_ptr(),
            qt.d_q1.as_ptr(),
            params,
            tickets_packed.as_ptr(),
            tickets_packed.len() as u32,
            &mut out,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c3v2_eval_joker esuat (rc={rc})"));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_c3v2_scan_universe(
    game_code: i32,
    d_tickets: &DeviceBuffer<u64>,
    d_ev_q05_pair: &DeviceBuffer<f32>,
    selected: &[u64],
    penalties16: &[f64; 16],
    w_ev: f64,
    w_q05: f64,
    block_scores_out: &mut [f64],
    block_indices_out: &mut [u32],
) -> Result<(), String> {
    let n_univ = d_tickets.len();
    let blocks = (n_univ + 255) / 256;
    if block_scores_out.len() != blocks || block_indices_out.len() != blocks {
        return Err(format!(
            "gpu_c3v2_scan_universe block buffer mismatch: {} vs {blocks}",
            block_scores_out.len()
        ));
    }
    let sel_ptr = if selected.is_empty() {
        std::ptr::null()
    } else {
        selected.as_ptr()
    };
    let rc = unsafe {
        v3_cuda_c3v2_scan_universe(
            game_code,
            d_tickets.as_ptr(),
            d_ev_q05_pair.as_ptr(),
            n_univ as u32,
            sel_ptr,
            selected.len() as u32,
            penalties16.as_ptr(),
            w_ev,
            w_q05,
            block_scores_out.as_mut_ptr(),
            block_indices_out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c3v2_scan_universe esuat (rc={rc})"));
    }
    Ok(())
}

pub fn gpu_c4v2_prep_hits_u32(
    game_code: i32,
    d_draws: &DeviceBuffer<u64>,
    all_tickets: &[u64],
    port_offs: &[u32],
    port_lens: &[u32],
) -> Result<DeviceBuffer<u32>, String> {
    let n_ports = port_offs.len();
    if n_ports == 0 || n_ports > 16 || port_lens.len() != n_ports {
        return Err("gpu_c4v2_prep_hits_u32 n_ports invalid".to_string());
    }
    let n_states = if game_code == 0 {
        13_983_816usize
    } else {
        3_838_380usize
    };
    let mut d_hits = DeviceBuffer::<u32>::alloc(n_states * n_ports)?;
    let tk_ptr = if all_tickets.is_empty() {
        std::ptr::null()
    } else {
        all_tickets.as_ptr()
    };
    let rc = unsafe {
        v3_cuda_c4v2_prep_hits(
            game_code,
            d_draws.as_ptr(),
            tk_ptr,
            all_tickets.len() as u32,
            port_offs.as_ptr(),
            port_lens.as_ptr(),
            n_ports as u32,
            d_hits.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c4v2_prep_hits u32 esuat (rc={rc})"));
    }
    Ok(d_hits)
}

pub fn gpu_c4v2_prep_hits_joker(
    d_base_joker: &DeviceBuffer<u64>,
    all_tickets: &[u64],
    port_offs: &[u32],
    port_lens: &[u32],
) -> Result<DeviceBuffer<u64>, String> {
    let n_ports = port_offs.len();
    if n_ports == 0 || n_ports > 16 || port_lens.len() != n_ports {
        return Err("gpu_c4v2_prep_hits_joker n_ports invalid".to_string());
    }
    let mut d_hits = DeviceBuffer::<u64>::alloc(24_435_180usize * n_ports)?;
    let tk_ptr = if all_tickets.is_empty() {
        std::ptr::null()
    } else {
        all_tickets.as_ptr()
    };
    let rc = unsafe {
        v3_cuda_c4v2_prep_hits(
            2,
            d_base_joker.as_ptr(),
            tk_ptr,
            all_tickets.len() as u32,
            port_offs.as_ptr(),
            port_lens.as_ptr(),
            n_ports as u32,
            d_hits.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c4v2_prep_hits joker esuat (rc={rc})"));
    }
    Ok(d_hits)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_c4v2_compute_rates_raw(
    game_code: i32,
    d_draws6_540: *const u64,
    d_q0: *const f64,
    d_q1: *const f64,
    d_q2: *const f64,
    d_q3: *const f64,
    d_q4: *const f64,
    d_hits: *const std::ffi::c_void,
    params: &V3CrowdParamsF64,
    xi16: &[f64; 16],
    n_ports: usize,
) -> Result<Vec<f64>, String> {
    let n_cats = match game_code {
        0 => 4usize,
        1 => 3usize,
        _ => 8usize,
    };
    let mut out = vec![0.0f64; n_ports * n_cats * 12];
    let rc = unsafe {
        v3_cuda_c4v2_compute_rates(
            game_code,
            d_draws6_540,
            d_q0,
            d_q1,
            d_q2,
            d_q3,
            d_q4,
            d_hits,
            params,
            xi16.as_ptr(),
            n_ports as u32,
            out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c4v2_compute_rates esuat (rc={rc})"));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_c4v2_sim_chunk_raw(
    game_code: i32,
    d_draws6_540: *const u64,
    d_q0: *const f64,
    d_q1: *const f64,
    d_q2: *const f64,
    d_q3: *const f64,
    d_q4: *const f64,
    d_hits: *const std::ffi::c_void,
    params: &V3CrowdParamsF64,
    xi16: &[f64; 16],
    rep0_8: &[f64; 8],
    fnew_8: &[f64; 8],
    n_ports: usize,
    chunk_start: u32,
    chunk_len: u32,
    seed: u64,
    out_pay4: &mut [f64],
    out_flags: &mut [u8],
) -> Result<f64, String> {
    let exp_pay = (chunk_len as usize) * n_ports * 4;
    let exp_flg = (chunk_len as usize) * n_ports;
    if out_pay4.len() != exp_pay || out_flags.len() != exp_flg {
        return Err("gpu_c4v2_sim_chunk_raw buffer len mismatch".to_string());
    }
    let mut ms = 0.0f64;
    let rc = unsafe {
        v3_cuda_c4v2_sim_chunk(
            game_code,
            d_draws6_540,
            d_q0,
            d_q1,
            d_q2,
            d_q3,
            d_q4,
            d_hits,
            params,
            xi16.as_ptr(),
            rep0_8.as_ptr(),
            fnew_8.as_ptr(),
            n_ports as u32,
            chunk_start,
            chunk_len,
            seed,
            out_pay4.as_mut_ptr(),
            out_flags.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_c4v2_sim_chunk esuat (rc={rc})"));
    }
    Ok(ms)
}

// ============================================================================
// Sesiunea 4 (S4 / Poarta A5) — FFI CUDA sm_120 pentru Motorul Cauzal CK-V6
// (BOCPD Adams-MacKay + Koopman V1 (+) V2 Shrunk + Scut Martingal w0 = 0.50)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct V3Ckv6StreamMeta {
    pub n_balls: u32,
    pub r_draw: u32,
    pub m_bonus: u32,
    pub n_steps: u32,
    pub step_off_base: u32,
    pub out_step_base: u32,
}

unsafe extern "C" {
    fn v3_cuda_ckv6_649_batch_f64(
        h_draws_batch: *const u64,
        n_traj: u32,
        t_steps: u32,
        l_regime_cp: u32,
        record_steps: u32,
        h_summary5_out: *mut f64,
        h_step_e_out: *mut f64,
        h_step_log_m_out: *mut f64,
        h_step_map_r_out: *mut i32,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_ckv6_power_10k_f64(
        h_delta_bp_cells: *const u32,
        h_l_reg_cells: *const u32,
        n_cells: u32,
        reps_per_cell: u32,
        t_steps: u32,
        seed: u64,
        h_records8_out: *mut f64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_ckv6_general_streams_f64(
        h_metas: *const V3Ckv6StreamMeta,
        n_streams: u32,
        h_step_draw_offs: *const u32,
        total_step_offs_len: u32,
        h_draws_lo: *const u64,
        h_draws_hi: *const u32,
        h_draws_bonus: *const u8,
        total_raw_draws: u32,
        total_out_steps: u32,
        h_out_step_e: *mut f64,
        h_out_step_log_m: *mut f64,
        h_out_step_map_r: *mut i32,
        h_out_step_cp_prob: *mut f64,
        h_out_step_log_m_glob: *mut f64,
        elapsed_ms_out: *mut f64,
    ) -> i32;

    fn v3_cuda_ckv6_universe_eval_f64(
        game_code: i32,
        d_univ_masks: *const u64,
        n_states: u32,
        n_balls: i32,
        r_draw: i32,
        h_h1_64: *const f64,
        h_q2_2401: *const f64,
        h_joker_tables_41420: *const f64,
        h_dm_table: *const f64,
        n_dm_terms: u32,
        kappa: f64,
        mean_e_out: *mut f64,
        min_e_out: *mut f64,
        max_e_out: *mut f64,
        elapsed_ms_out: *mut f64,
    ) -> i32;
}

pub struct GpuCkv6Batch649Result {
    pub summary5: Vec<[f64; 5]>,
    pub step_e: Vec<f64>,
    pub step_log_m: Vec<f64>,
    pub step_map_r: Vec<i32>,
    pub elapsed_ms: f64,
}

pub fn gpu_ckv6_649_batch_f64(
    draws_batch: &[u64],
    n_traj: usize,
    t_steps: usize,
    l_regime_cp: u32,
    record_steps: bool,
) -> Result<GpuCkv6Batch649Result, String> {
    if n_traj == 0 || t_steps == 0 || t_steps > 400 || draws_batch.len() != n_traj * t_steps {
        return Err(format!(
            "gpu_ckv6_649_batch_f64 dimensiuni invalide: n_traj={n_traj}, t_steps={t_steps}, len={}",
            draws_batch.len()
        ));
    }
    let extra_rss = ((n_traj * 5 * 8)
        + if record_steps {
            n_traj * t_steps * 20
        } else {
            0
        }) as u64;
    check_rss_guard(extra_rss)?;

    let mut summary_flat = vec![0.0f64; n_traj * 5];
    let mut step_e = if record_steps {
        vec![0.0f64; n_traj * t_steps]
    } else {
        Vec::new()
    };
    let mut step_log_m = if record_steps {
        vec![0.0f64; n_traj * t_steps]
    } else {
        Vec::new()
    };
    let mut step_map_r = if record_steps {
        vec![0i32; n_traj * t_steps]
    } else {
        Vec::new()
    };
    let mut ms = 0.0f64;

    let rc = unsafe {
        v3_cuda_ckv6_649_batch_f64(
            draws_batch.as_ptr(),
            n_traj as u32,
            t_steps as u32,
            l_regime_cp,
            u32::from(record_steps),
            summary_flat.as_mut_ptr(),
            if record_steps {
                step_e.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            if record_steps {
                step_log_m.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            if record_steps {
                step_map_r.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_ckv6_649_batch_f64 esuat (rc={rc})"));
    }

    let mut summary5 = Vec::with_capacity(n_traj);
    for chunk in summary_flat.chunks_exact(5) {
        summary5.push([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4]]);
    }
    Ok(GpuCkv6Batch649Result {
        summary5,
        step_e,
        step_log_m,
        step_map_r,
        elapsed_ms: ms,
    })
}

pub fn gpu_ckv6_power_10k_f64(
    delta_bp_cells: &[u32],
    l_reg_cells: &[u32],
    reps_per_cell: u32,
    t_steps: u32,
    seed: u64,
) -> Result<(Vec<[f64; 8]>, f64), String> {
    if delta_bp_cells.is_empty()
        || delta_bp_cells.len() != l_reg_cells.len()
        || reps_per_cell == 0
        || t_steps == 0
        || t_steps > 400
    {
        return Err("gpu_ckv6_power_10k_f64 parametri invalizi".to_string());
    }
    let total_hist = delta_bp_cells.len() * (reps_per_cell as usize);
    check_rss_guard((total_hist * 64) as u64)?;
    let mut rec_flat = vec![0.0f64; total_hist * 8];
    let mut ms = 0.0f64;

    let rc = unsafe {
        v3_cuda_ckv6_power_10k_f64(
            delta_bp_cells.as_ptr(),
            l_reg_cells.as_ptr(),
            delta_bp_cells.len() as u32,
            reps_per_cell,
            t_steps,
            seed,
            rec_flat.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_ckv6_power_10k_f64 esuat (rc={rc})"));
    }

    let mut out = Vec::with_capacity(total_hist);
    for c in rec_flat.chunks_exact(8) {
        out.push([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
    }
    Ok((out, ms))
}

pub struct GpuCkv6GeneralStreamsResult {
    pub step_e: Vec<f64>,
    pub step_log_m: Vec<f64>,
    pub step_map_r: Vec<i32>,
    pub step_cp_prob: Vec<f64>,
    pub step_log_m_glob: Vec<f64>,
    pub elapsed_ms: f64,
}

pub fn gpu_ckv6_general_streams_f64(
    metas: &[V3Ckv6StreamMeta],
    step_draw_offs: &[u32],
    draws_lo: &[u64],
    draws_hi: &[u32],
    draws_bonus: &[u8],
    total_out_steps: usize,
) -> Result<GpuCkv6GeneralStreamsResult, String> {
    if metas.is_empty()
        || draws_lo.is_empty()
        || draws_lo.len() != draws_hi.len()
        || draws_lo.len() != draws_bonus.len()
        || total_out_steps == 0
    {
        return Err("gpu_ckv6_general_streams_f64 argumente invalide".to_string());
    }
    check_rss_guard((total_out_steps * 36) as u64)?;

    let mut step_e = vec![0.0f64; total_out_steps];
    let mut step_log_m = vec![0.0f64; total_out_steps];
    let mut step_map_r = vec![0i32; total_out_steps];
    let mut step_cp_prob = vec![0.0f64; total_out_steps];
    let mut step_log_m_glob = vec![0.0f64; total_out_steps];
    let mut ms = 0.0f64;

    let rc = unsafe {
        v3_cuda_ckv6_general_streams_f64(
            metas.as_ptr(),
            metas.len() as u32,
            step_draw_offs.as_ptr(),
            step_draw_offs.len() as u32,
            draws_lo.as_ptr(),
            draws_hi.as_ptr(),
            draws_bonus.as_ptr(),
            draws_lo.len() as u32,
            total_out_steps as u32,
            step_e.as_mut_ptr(),
            step_log_m.as_mut_ptr(),
            step_map_r.as_mut_ptr(),
            step_cp_prob.as_mut_ptr(),
            step_log_m_glob.as_mut_ptr(),
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_ckv6_general_streams_f64 esuat (rc={rc})"));
    }

    Ok(GpuCkv6GeneralStreamsResult {
        step_e,
        step_log_m,
        step_map_r,
        step_cp_prob,
        step_log_m_glob,
        elapsed_ms: ms,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_ckv6_universe_eval_f64(
    game_code: i32,
    d_univ_masks: &DeviceBuffer<u64>,
    n_balls: usize,
    r_draw: usize,
    h1_64: &[f64; 64],
    q2_2401: &[f64; 2401],
    joker_tables_41420: Option<&[f64]>,
    dm_table_70: &[f64],
    n_dm_terms: usize,
    kappa: f64,
) -> Result<(f64, f64, f64, f64), String> {
    check_rss_guard(0)?;
    if dm_table_70.len() != n_dm_terms * 70 || n_dm_terms > 68 {
        return Err(format!(
            "dm_table_70 invalid: len={}, n_dm_terms={n_dm_terms}",
            dm_table_70.len()
        ));
    }
    let j_ptr = match (game_code, joker_tables_41420) {
        (2, Some(tbl)) => {
            if tbl.len() != 20 * 2071 {
                return Err(format!(
                    "joker_tables_41420 len invalid: {} != {}",
                    tbl.len(),
                    20 * 2071
                ));
            }
            tbl.as_ptr()
        }
        _ => std::ptr::null(),
    };
    let dm_ptr = if n_dm_terms > 0 {
        dm_table_70.as_ptr()
    } else {
        std::ptr::null()
    };
    let mut mean_e = 0.0f64;
    let mut min_e = 0.0f64;
    let mut max_e = 0.0f64;
    let mut ms = 0.0f64;

    let rc = unsafe {
        v3_cuda_ckv6_universe_eval_f64(
            game_code,
            d_univ_masks.as_ptr(),
            d_univ_masks.len() as u32,
            n_balls as i32,
            r_draw as i32,
            h1_64.as_ptr(),
            q2_2401.as_ptr(),
            j_ptr,
            dm_ptr,
            n_dm_terms as u32,
            kappa,
            &mut mean_e,
            &mut min_e,
            &mut max_e,
            &mut ms,
        )
    };
    if rc != 0 {
        return Err(format!("v3_cuda_ckv6_universe_eval_f64 esuat (rc={rc})"));
    }
    Ok((mean_e, min_e, max_e, ms))
}

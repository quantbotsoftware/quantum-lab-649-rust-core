//! `rust-singularity-core-v3` (`v3.0.0`) — Quantum Lab 649 · Rust Core v3
//! Hardware nativ: `NVIDIA GeForce RTX 5060 Laptop GPU` (`CUDA 12.9 sm_120`, `8 GB GDDR7`)
//! + `Intel Core i7-14650HX` (`24` fire `AVX2/FMA/BMI2/POPCNT`) + SSD `mmap` Vault (`./vault/`).
//! Guvernanță științifică: `positive_claims_allowed = false`.

pub const POSITIVE_CLAIMS_ALLOWED: bool = false;

pub mod bridge;
pub mod c3v2;
pub mod c4v2;
pub mod cdv3;
pub mod ckv6;
pub mod cuda_ffi;
pub mod hsv4_bridge;
pub mod integrity;
#[allow(
    clippy::int_plus_one,
    clippy::manual_clamp,
    clippy::manual_range_contains,
    clippy::needless_borrowed_reference,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
pub mod kjv5;
#[allow(
    clippy::needless_borrowed_reference,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
pub mod kjv5_walkforward;
pub mod models;
pub mod npz;
pub mod parser;
pub mod pmv3;
pub mod registries;
pub mod resources;
pub mod telemetry;
pub mod v3_observatory;
pub mod vault;
pub mod witt_s3_6_22;

#[cfg(test)]
mod tests {
    use crate::cuda_ffi::{check_rss_guard, query_device_info, PinnedBuffer};
    use crate::vault::{binom, generate_comb_masks_parallel_to_slice, TOTAL_UNIVERSE_TICKETS};

    #[test]
    fn test_total_universe_constant() {
        assert_eq!(binom(49, 6), 13_983_816);
        assert_eq!(binom(40, 5), 658_008);
        assert_eq!(binom(40, 6), 3_838_380);
        assert_eq!(binom(45, 5) * 20, 24_435_180);
        assert_eq!(TOTAL_UNIVERSE_TICKETS, 39_077_004);
    }

    #[test]
    fn test_parallel_lexicographic_generator() {
        let mut out = vec![0u64; binom(12, 5)];
        generate_comb_masks_parallel_to_slice(12, 5, &mut out);
        assert_eq!(out.len(), 792);
        assert_eq!(out[0], 0b11111);
        assert_eq!(out[791], 0b11111_0000000);
        for w in out.windows(2) {
            assert_ne!(w[0], w[1]);
            assert_eq!(w[0].count_ones(), 5);
        }
    }

    #[test]
    fn test_cuda_sm120_and_dma_rss_guard() {
        let info = query_device_info().expect("CUDA RTX 5060 trebuie sa fie activa");
        assert_eq!(info.major, 12);
        assert_eq!(info.minor, 0);
        assert!(info.total_vram_bytes > 7_500_000_000);

        let mut dma = PinnedBuffer::<u64>::new_zeroed(65_536).expect("cudaHostAlloc DMA");
        dma.as_mut_slice()[0] = 0xCAFE_BABE_6490_0003;
        assert_eq!(dma.as_slice()[0], 0xCAFE_BABE_6490_0003);

        let (rss, _) = check_rss_guard(0).expect("VmRSS sub plafonul de 11.5 GiB");
        assert!(rss > 0);
    }
}

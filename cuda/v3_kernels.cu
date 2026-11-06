// Quantum Lab 649 · Rust Core v3 — Native CUDA 12.9 Kernels (Blackwell sm_120, RTX 5060 8GB GDDR7)
// Guvernanta: positive_claims_allowed = false. IEEE-754 exact (--fmad=true, fara --use_fast_math).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" {

struct V3CudaDeviceInfo {
    int32_t device_count;
    int32_t major;
    int32_t minor;
    int32_t sm_count;
    int32_t max_threads_per_sm;
    int32_t l2_cache_bytes;
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    char name[128];
};

struct V3PointDraw {
    uint32_t mask;
    uint32_t draw_id;
};

int32_t v3_cuda_get_device_info(V3CudaDeviceInfo* out) {
    if (!out) return -1;
    memset(out, 0, sizeof(V3CudaDeviceInfo));
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        out->device_count = 0;
        return (int32_t)err;
    }
    out->device_count = count;
    err = cudaSetDevice(0);
    if (err != cudaSuccess) return (int32_t)err;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int32_t)err;

    size_t free_b = 0, total_b = 0;
    err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) return (int32_t)err;

    out->major = prop.major;
    out->minor = prop.minor;
    out->sm_count = prop.multiProcessorCount;
    out->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    out->l2_cache_bytes = prop.l2CacheSize;
    out->total_vram_bytes = (uint64_t)total_b;
    out->free_vram_bytes = (uint64_t)free_b;
    strncpy(out->name, prop.name, 127);
    out->name[127] = '\0';
    return 0;
}

int32_t v3_cuda_host_alloc(void** ptr_out, uint64_t bytes) {
    if (!ptr_out) return -1;
    *ptr_out = nullptr;
    cudaError_t err = cudaHostAlloc(ptr_out, (size_t)bytes, cudaHostAllocDefault);
    return (int32_t)err;
}

int32_t v3_cuda_host_free(void* ptr) {
    if (!ptr) return 0;
    cudaError_t err = cudaFreeHost(ptr);
    return (int32_t)err;
}

int32_t v3_cuda_device_alloc(void** d_ptr_out, uint64_t bytes) {
    if (!d_ptr_out) return -1;
    *d_ptr_out = nullptr;
    cudaError_t err = cudaMalloc(d_ptr_out, (size_t)bytes);
    return (int32_t)err;
}

int32_t v3_cuda_device_free(void* d_ptr) {
    if (!d_ptr) return 0;
    cudaError_t err = cudaFree(d_ptr);
    return (int32_t)err;
}

int32_t v3_cuda_memcpy_h2d(void* dst_dev, const void* src_host, uint64_t bytes) {
    cudaError_t err = cudaMemcpy(dst_dev, src_host, (size_t)bytes, cudaMemcpyHostToDevice);
    return (int32_t)err;
}

int32_t v3_cuda_memcpy_d2h(void* dst_host, const void* src_dev, uint64_t bytes) {
    cudaError_t err = cudaMemcpy(dst_host, src_dev, (size_t)bytes, cudaMemcpyDeviceToHost);
    return (int32_t)err;
}

int32_t v3_cuda_memset_u8(void* dst_dev, int32_t val, uint64_t bytes) {
    cudaError_t err = cudaMemset(dst_dev, val, (size_t)bytes);
    return (int32_t)err;
}

} // extern "C"

// ============================================================================
// 1. Poarta A0 — Paritate exacta pe bit (__popcll) peste intregul univers
// ============================================================================

__constant__ uint64_t c_probes[16];

__global__ void k_popcnt_hist_multi(
    const uint64_t* __restrict__ d_univ,
    uint64_t n_univ,
    uint32_t n_probes,
    unsigned long long* __restrict__ d_hist // n_probes * 8
) {
    __shared__ unsigned long long s_hist[16 * 8];
    for (uint32_t i = threadIdx.x; i < n_probes * 8; i += blockDim.x) {
        s_hist[i] = 0ULL;
    }
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t i = idx; i < n_univ; i += stride) {
        uint64_t m = d_univ[i];
        for (uint32_t p = 0; p < n_probes; ++p) {
            int hits = __popcll(m & c_probes[p]);
            if (hits > 7) hits = 7;
            atomicAdd(&s_hist[p * 8 + (uint32_t)hits], 1ULL);
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < n_probes * 8; i += blockDim.x) {
        if (s_hist[i] > 0ULL) {
            atomicAdd(&d_hist[i], s_hist[i]);
        }
    }
}

__global__ void k_joker_packed_parity(
    const uint64_t* __restrict__ d_joker_24m,
    uint64_t n_univ,
    uint32_t n_probes,
    unsigned long long* __restrict__ d_hist // n_probes * 12 (hits 0..5 x j_match 0..1)
) {
    __shared__ unsigned long long s_hist[16 * 12];
    for (uint32_t i = threadIdx.x; i < n_probes * 12; i += blockDim.x) {
        s_hist[i] = 0ULL;
    }
    __syncthreads();

    const uint64_t MASK45 = (1ULL << 45) - 1ULL;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t i = idx; i < n_univ; i += stride) {
        uint64_t packed = d_joker_24m[i];
        uint64_t m = packed & MASK45;
        uint32_t j0 = (uint32_t)((packed >> 48) & 0x1FULL);
        for (uint32_t p = 0; p < n_probes; ++p) {
            uint64_t pr = c_probes[p];
            int hits = __popcll(m & (pr & MASK45));
            uint32_t pj0 = (uint32_t)((pr >> 48) & 0x1FULL);
            uint32_t jm = (j0 == pj0) ? 1u : 0u;
            atomicAdd(&s_hist[p * 12 + (uint32_t)hits * 2u + jm], 1ULL);
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < n_probes * 12; i += blockDim.x) {
        if (s_hist[i] > 0ULL) {
            atomicAdd(&d_hist[i], s_hist[i]);
        }
    }
}

// ============================================================================
// 2. Poarta A2 — Verificare Exhaustiva 100% pe GPU (6/49, 5/40, Joker 24.43M)
// ============================================================================

__global__ void k_verify_covering(
    const uint64_t* __restrict__ d_draws,
    uint64_t n_draws,
    const uint64_t* __restrict__ d_tickets,
    uint32_t n_tickets,
    unsigned long long* __restrict__ d_hist8
) {
    __shared__ uint64_t s_tickets[256];
    __shared__ unsigned long long s_hist[8];
    if (threadIdx.x < 8) {
        s_hist[threadIdx.x] = 0ULL;
    }
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (idx < n_draws);
    uint64_t draw = active ? d_draws[idx] : 0ULL;
    int best = 0;

    for (uint32_t base = 0; base < n_tickets; base += 256) {
        uint32_t chunk = n_tickets - base;
        if (chunk > 256) chunk = 256;
        if (threadIdx.x < chunk) {
            s_tickets[threadIdx.x] = d_tickets[base + threadIdx.x];
        }
        __syncthreads();

        if (active) {
            #pragma unroll 8
            for (uint32_t k = 0; k < chunk; ++k) {
                int h = __popcll(draw & s_tickets[k]);
                if (h > best) best = h;
            }
        }
        __syncthreads();
    }

    if (active) {
        if (best > 7) best = 7;
        atomicAdd(&s_hist[best], 1ULL);
    }
    __syncthreads();

    if (threadIdx.x < 8 && s_hist[threadIdx.x] > 0ULL) {
        atomicAdd(&d_hist8[threadIdx.x], s_hist[threadIdx.x]);
    }
}

__device__ __forceinline__ int joker_rank(int hits, bool j_match) {
    if (hits == 5 && j_match) return 0; // Cat I
    if (hits == 5) return 1;            // Cat II
    if (hits == 4 && j_match) return 2; // Cat III
    if (hits == 4) return 3;            // Cat IV
    if (hits == 3 && j_match) return 4; // Cat V
    if (hits == 3) return 5;            // Cat VI
    if (hits == 2 && j_match) return 6; // Cat VII
    if (hits == 1 && j_match) return 7; // Cat VIII
    return 8;                           // Necastigatoare
}

__global__ void k_verify_joker_packed_24m(
    const uint64_t* __restrict__ d_joker_24m,
    uint64_t n_states,
    const uint64_t* __restrict__ d_tickets_packed,
    uint32_t n_tickets,
    unsigned long long* __restrict__ d_cat_hist9
) {
    __shared__ uint64_t s_tickets[256];
    __shared__ unsigned long long s_hist[9];
    if (threadIdx.x < 9) {
        s_hist[threadIdx.x] = 0ULL;
    }
    __syncthreads();

    const uint64_t MASK45 = (1ULL << 45) - 1ULL;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (idx < n_states);
    uint64_t st = active ? d_joker_24m[idx] : 0ULL;
    uint64_t d_mask = st & MASK45;
    uint32_t d_j0 = (uint32_t)((st >> 48) & 0x1FULL);
    int best_rank = 8;

    for (uint32_t base = 0; base < n_tickets; base += 256) {
        uint32_t chunk = n_tickets - base;
        if (chunk > 256) chunk = 256;
        if (threadIdx.x < chunk) {
            s_tickets[threadIdx.x] = d_tickets_packed[base + threadIdx.x];
        }
        __syncthreads();

        if (active) {
            for (uint32_t k = 0; k < chunk; ++k) {
                uint64_t tk = s_tickets[k];
                int h = __popcll(d_mask & (tk & MASK45));
                bool jm = (((uint32_t)((tk >> 48) & 0x1FULL)) == d_j0);
                int r = joker_rank(h, jm);
                if (r < best_rank) best_rank = r;
            }
        }
        __syncthreads();
    }

    if (active) {
        atomicAdd(&s_hist[best_rank], 1ULL);
    }
    __syncthreads();

    if (threadIdx.x < 9 && s_hist[threadIdx.x] > 0ULL) {
        atomicAdd(&d_cat_hist9[threadIdx.x], s_hist[threadIdx.x]);
    }
}

// ============================================================================
// 3. Lazy Submodular Greedy pe GPU cu Bitpacking in L2 Cache (Zero Skip)
// ============================================================================

__global__ void k_init_uncov_bits(uint64_t* d_uncov_bits, uint64_t n_draws) {
    uint64_t n_words = (n_draws + 63ULL) >> 6;
    uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w < n_words) {
        uint64_t rem = n_draws - (w << 6);
        d_uncov_bits[w] = (rem >= 64ULL) ? ~0ULL : ((1ULL << rem) - 1ULL);
    }
}

__global__ void k_mark_covered_bits(
    const uint64_t* __restrict__ d_draws,
    uint64_t n_draws,
    uint64_t* __restrict__ d_uncov_bits,
    uint64_t chosen_mask,
    int32_t min_hits,
    unsigned long long* __restrict__ d_rem_out
) {
    __shared__ unsigned long long s_rem;
    if (threadIdx.x == 0) s_rem = 0ULL;
    __syncthreads();

    uint64_t n_words = (n_draws + 63ULL) >> 6;
    uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w < n_words) {
        uint64_t bits = d_uncov_bits[w];
        if (bits != 0ULL) {
            uint64_t base = w << 6;
            uint64_t cur = bits;
            while (cur != 0ULL) {
                int b = __ffsll((long long)cur) - 1;
                uint64_t idx = base + (uint64_t)b;
                if (idx < n_draws && __popcll(d_draws[idx] & chosen_mask) >= min_hits) {
                    bits &= ~(1ULL << b);
                }
                cur &= (cur - 1ULL);
            }
            d_uncov_bits[w] = bits;
            if (bits != 0ULL) {
                atomicAdd(&s_rem, (unsigned long long)__popcll(bits));
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_rem > 0ULL) {
        atomicAdd(d_rem_out, s_rem);
    }
}

// Fiecare bloc de 256 fire evalueaza simultan 256 de bilete candidate,
// colectand in __shared__ doar extragerile ramase neacoperite (zero-skip pe u64!).
__global__ void k_greedy_scores_bitpacked(
    const uint64_t* __restrict__ d_draws,
    uint64_t n_draws,
    const uint64_t* __restrict__ d_uncov_bits,
    const uint64_t* __restrict__ d_cands,
    uint32_t n_cands,
    int32_t min_hits,
    uint32_t* __restrict__ d_gains
) {
    __shared__ uint64_t s_active_draws[256];
    __shared__ uint32_t s_count;

    uint32_t cid = blockIdx.x * 256u + threadIdx.x;
    bool has_cand = (cid < n_cands);
    uint64_t my_cand = has_cand ? d_cands[cid] : 0ULL;
    uint32_t my_gain = 0u;

    uint64_t n_words = (n_draws + 63ULL) >> 6;
    uint64_t w = 0ULL;

    while (w < n_words) {
        if (threadIdx.x == 0) s_count = 0u;
        __syncthreads();

        // Fiecare fir inspecteaza un cuvant din fereastra w .. w+256;
        // doar daca gaseste biti nenuli ii adauga cat timp s_count <= 192.
        // Pentru determinism exact si zero pierderi, firul 0 avanseaza w secvential pe cuvinte nenule.
        if (threadIdx.x == 0) {
            uint32_t cnt = 0u;
            while (w < n_words && cnt <= 192u) {
                uint64_t bits = d_uncov_bits[w];
                if (bits != 0ULL) {
                    uint64_t base = w << 6;
                    while (bits != 0ULL) {
                        int b = __ffsll((long long)bits) - 1;
                        s_active_draws[cnt++] = base + (uint64_t)b;
                        bits &= (bits - 1ULL);
                    }
                }
                w++;
            }
            s_count = cnt;
        }
        // Broadcast w prin shared
        __shared__ uint64_t s_next_w;
        if (threadIdx.x == 0) s_next_w = w;
        __syncthreads();
        w = s_next_w;

        uint32_t cnt = s_count;
        if (threadIdx.x < cnt) {
            uint64_t draw_idx = s_active_draws[threadIdx.x];
            s_active_draws[threadIdx.x] = d_draws[draw_idx];
        }
        __syncthreads();

        if (has_cand) {
            #pragma unroll 8
            for (uint32_t k = 0; k < cnt; ++k) {
                if (__popcll(my_cand & s_active_draws[k]) >= min_hits) {
                    my_gain++;
                }
            }
        }
        __syncthreads();
    }

    if (has_cand) {
        d_gains[cid] = my_gain;
    }
}

// ============================================================================
// 4. Joker Greedy pe GPU (20-bit mask per 5-draw in 1,221,759 elemente)
// ============================================================================

__global__ void k_joker_init_uncov20(uint32_t* d_uncov20, uint32_t n_base) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_base) {
        d_uncov20[i] = (1u << 20) - 1u;
    }
}

__global__ void k_joker_mark_covered20(
    const uint64_t* __restrict__ d_base_draws,
    uint32_t n_base,
    uint32_t* __restrict__ d_uncov20,
    uint64_t chosen_packed,
    unsigned long long* __restrict__ d_rem_out
) {
    __shared__ unsigned long long s_rem;
    if (threadIdx.x == 0) s_rem = 0ULL;
    __syncthreads();

    const uint64_t MASK45 = (1ULL << 45) - 1ULL;
    uint64_t t_mask = chosen_packed & MASK45;
    uint32_t t_j0 = (uint32_t)((chosen_packed >> 48) & 0x1FULL);

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_base) {
        uint32_t u = d_uncov20[i];
        if (u != 0u) {
            int h = __popcll(d_base_draws[i] & t_mask);
            if (h >= 3) {
                u = 0u;
            } else if (h >= 1) {
                u &= ~(1u << t_j0);
            }
            d_uncov20[i] = u;
            if (u != 0u) {
                atomicAdd(&s_rem, (unsigned long long)__popc(u));
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_rem > 0ULL) {
        atomicAdd(d_rem_out, s_rem);
    }
}

__global__ void k_joker_greedy_scores20(
    const uint64_t* __restrict__ d_base_draws,
    uint32_t n_base,
    const uint32_t* __restrict__ d_uncov20,
    const uint64_t* __restrict__ d_cands_packed,
    uint32_t n_cands,
    uint32_t* __restrict__ d_gains
) {
    __shared__ uint64_t s_masks[256];
    __shared__ uint32_t s_u20[256];

    const uint64_t MASK45 = (1ULL << 45) - 1ULL;
    uint32_t cid = blockIdx.x * 256u + threadIdx.x;
    bool has_cand = (cid < n_cands);
    uint64_t c_packed = has_cand ? d_cands_packed[cid] : 0ULL;
    uint64_t c_mask = c_packed & MASK45;
    uint32_t c_jbit = 1u << ((uint32_t)((c_packed >> 48) & 0x1FULL));
    uint32_t my_gain = 0u;

    for (uint32_t base = 0; base < n_base; base += 256u) {
        uint32_t chunk = n_base - base;
        if (chunk > 256u) chunk = 256u;
        if (threadIdx.x < chunk) {
            uint32_t u = d_uncov20[base + threadIdx.x];
            s_u20[threadIdx.x] = u;
            s_masks[threadIdx.x] = (u != 0u) ? d_base_draws[base + threadIdx.x] : 0ULL;
        }
        __syncthreads();

        if (has_cand) {
            #pragma unroll 8
            for (uint32_t k = 0; k < chunk; ++k) {
                uint32_t u = s_u20[k];
                if (u != 0u) {
                    int h = __popcll(c_mask & s_masks[k]);
                    if (h >= 3) {
                        my_gain += (uint32_t)__popc(u);
                    } else if (h >= 1 && (u & c_jbit) != 0u) {
                        my_gain += 1u;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (has_cand) {
        d_gains[cid] = my_gain;
    }
}

// ============================================================================
// 5. 16,384-Thread Shared-SRAM Nurmela-Ostergard Tabu Search Kernel
//    128 blocks x 128 threads = 16,384 fire active CUDA sm_120!
//    Suporta v <= 32, M_sub <= 24,576, n_blocks <= 128 (ex. L(27,6,4,3)=86,
//    L(20,5,5,3), roți abreviate v_sub = 12..24).
// ============================================================================

__device__ __forceinline__ uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct CandMove {
    uint16_t b_idx;
    uint8_t u_out;
    uint8_t u_in;
};

__global__ void k_tabu_shared_16384(
    uint32_t v_pts,
    uint32_t k_pick,
    uint32_t t_hit,
    uint32_t m_draws,
    const uint32_t* __restrict__ d_draw_masks,
    const V3PointDraw* __restrict__ d_point_draws, // v_pts * deg_per_pt
    uint32_t deg_per_pt,
    uint32_t n_blocks,
    const uint32_t* __restrict__ d_init_blocks,    // 128 * n_blocks
    uint32_t max_iters,
    uint64_t base_seed,
    uint32_t* __restrict__ d_best_blocks_out,      // 128 * n_blocks
    uint32_t* __restrict__ d_best_uncov_out,       // 128
    volatile int32_t* __restrict__ d_solved_flag
) {
    __shared__ uint8_t s_cov[24576];
    __shared__ uint32_t s_blocks[128];
    __shared__ uint32_t s_best_blocks[128];
    __shared__ uint32_t s_bstep[128];
    __shared__ uint32_t s_tabu[128 * 32]; // block * v_pts per-block move expiry
    __shared__ CandMove s_cands[256];
    __shared__ uint32_t s_uncov_list[512];
    __shared__ uint32_t s_n_cands;
    __shared__ uint32_t s_uncov_total;
    __shared__ uint32_t s_best_uncov;
    __shared__ uint32_t s_chosen_f0;
    __shared__ int32_t s_move_deltas[128];
    __shared__ uint32_t s_move_idx[128];
    __shared__ uint64_t s_rng;

    const uint32_t tid = threadIdx.x;
    const uint32_t chain = blockIdx.x;

    if (tid == 0) {
        s_rng = base_seed ^ ((uint64_t)chain * 0xD1B54A32D192ED03ULL);
        splitmix64_next(&s_rng);
    }
    for (uint32_t i = tid; i < n_blocks; i += 128u) {
        uint32_t bm = d_init_blocks[chain * n_blocks + i];
        s_blocks[i] = bm;
        s_best_blocks[i] = bm;
        s_bstep[i] = 0u;
    }
    for (uint32_t i = tid; i < n_blocks * 32u; i += 128u) {
        s_tabu[i] = 0u;
    }
    __syncthreads();

    // Calculam acoperirea initiala exacta in s_cov[0..m_draws]
    uint32_t local_uncov = 0u;
    for (uint32_t d = tid; d < m_draws; d += 128u) {
        uint32_t dm = d_draw_masks[d];
        uint32_t cnt = 0u;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            if ((uint32_t)__popc(dm & s_blocks[b]) >= t_hit) {
                if (cnt < 255u) cnt++;
            }
        }
        s_cov[d] = (uint8_t)cnt;
        if (cnt == 0u) local_uncov++;
    }
    if (tid == 0) s_uncov_total = 0u;
    __syncthreads();
    if (local_uncov > 0u) atomicAdd(&s_uncov_total, local_uncov);
    __syncthreads();

    if (tid == 0) {
        s_best_uncov = s_uncov_total;
        if (s_uncov_total == 0u) {
            atomicExch((int*)d_solved_flag, 1);
        }
    }
    __syncthreads();

    for (uint32_t step = 1; step <= max_iters; ++step) {
        if (*d_solved_flag != 0 || s_uncov_total == 0u) break;

        // Colectam pana la 512 extrageri neacoperite
        if (tid == 0) s_n_cands = 0u; // refolosim temporar drept contor uncov
        __syncthreads();

        for (uint32_t d = tid; d < m_draws; d += 128u) {
            if (s_cov[d] == 0u) {
                uint32_t pos = atomicAdd(&s_n_cands, 1u);
                if (pos < 512u) {
                    s_uncov_list[pos] = d;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t avail = s_n_cands < 512u ? s_n_cands : 512u;
            uint32_t pick = (avail > 0u) ? (uint32_t)(splitmix64_next(&s_rng) % avail) : 0u;
            uint32_t f0_id = (avail > 0u) ? s_uncov_list[pick] : 0u;
            s_chosen_f0 = d_draw_masks[f0_id];
            s_n_cands = 0u;
        }
        __syncthreads();

        uint32_t f0_mask = s_chosen_f0;
        // Fiecare fir b = tid inspecteaza blocul b pentru mutari care acopera f0_mask
        for (uint32_t b = tid; b < n_blocks; b += 128u) {
            uint32_t bm = s_blocks[b];
            uint32_t inter = (uint32_t)__popc(bm & f0_mask);
            if (inter == t_hit - 1u) {
                uint32_t outs = bm & (~f0_mask);
                uint32_t ins = f0_mask & (~bm);
                uint32_t o_tmp = outs;
                while (o_tmp != 0u) {
                    int u_out = __ffs((int)o_tmp) - 1;
                    uint32_t i_tmp = ins;
                    while (i_tmp != 0u) {
                        int u_in = __ffs((int)i_tmp) - 1;
                        uint32_t idx = atomicAdd(&s_n_cands, 1u);
                        if (idx < 256u) {
                            s_cands[idx].b_idx = (uint16_t)b;
                            s_cands[idx].u_out = (uint8_t)u_out;
                            s_cands[idx].u_in = (uint8_t)u_in;
                        }
                        i_tmp &= (i_tmp - 1u);
                    }
                    o_tmp &= (o_tmp - 1u);
                }
            }
        }
        __syncthreads();

        // Daca niciun bloc nu are inter == t_hit - 1, generam mutari fallback spre f0_mask
        if (s_n_cands == 0u && tid == 0) {
            uint32_t b = (uint32_t)(splitmix64_next(&s_rng) % n_blocks);
            uint32_t bm = s_blocks[b];
            uint32_t outs = bm & (~f0_mask);
            uint32_t ins = f0_mask & (~bm);
            if (outs != 0u && ins != 0u) {
                s_cands[0].b_idx = (uint16_t)b;
                s_cands[0].u_out = (uint8_t)(__ffs((int)outs) - 1);
                s_cands[0].u_in = (uint8_t)(__ffs((int)ins) - 1);
                s_n_cands = 1u;
            }
        }
        __syncthreads();

        uint32_t n_c = s_n_cands < 256u ? s_n_cands : 256u;
        if (n_c == 0u) continue;

        int32_t best_local_delta = 0x3FFFFFFF;
        uint32_t best_local_c = 0xFFFFFFFFu;
        uint64_t tie_rng = base_seed ^ ((uint64_t)chain * 1315423911ULL) ^ ((uint64_t)step * 2654435761ULL) ^ tid;

        for (uint32_t c = tid; c < n_c; c += 128u) {
            uint32_t b = s_cands[c].b_idx;
            uint32_t u_out = s_cands[c].u_out;
            uint32_t u_in = s_cands[c].u_in;
            uint32_t r_mask = s_blocks[b] & ~(1u << u_out);
            uint32_t in_bit = 1u << u_in;
            uint32_t out_bit = 1u << u_out;

            int32_t lost = 0;
            const V3PointDraw* p_out = &d_point_draws[u_out * deg_per_pt];
            for (uint32_t j = 0; j < deg_per_pt; ++j) {
                uint32_t dm = p_out[j].mask;
                if ((dm & in_bit) == 0u && (uint32_t)__popc(dm & r_mask) == t_hit - 1u) {
                    if (s_cov[p_out[j].draw_id] == 1u) lost++;
                }
            }

            int32_t gained = 0;
            const V3PointDraw* p_in = &d_point_draws[u_in * deg_per_pt];
            for (uint32_t j = 0; j < deg_per_pt; ++j) {
                uint32_t dm = p_in[j].mask;
                if ((dm & out_bit) == 0u && (uint32_t)__popc(dm & r_mask) == t_hit - 1u) {
                    if (s_cov[p_in[j].draw_id] == 0u) gained++;
                }
            }

            int32_t delta = lost - gained;
            int32_t cand_uncov = (int32_t)s_uncov_total + delta;
            bool is_tabu = (s_tabu[b * 32u + u_in] > s_bstep[b]);
            bool aspiration = (cand_uncov < (int32_t)s_best_uncov);

            int32_t score = (is_tabu && !aspiration) ? (delta + 100000) : delta;
            score = score * 8 + (int32_t)(splitmix64_next(&tie_rng) & 7ULL);
            if (score < best_local_delta) {
                best_local_delta = score;
                best_local_c = c;
            }
        }

        s_move_deltas[tid] = best_local_delta;
        s_move_idx[tid] = best_local_c;
        __syncthreads();

        // Reducere pe arbore in 128 fire
        for (uint32_t s = 64u; s > 0u; s >>= 1u) {
            if (tid < s) {
                if (s_move_deltas[tid + s] < s_move_deltas[tid]) {
                    s_move_deltas[tid] = s_move_deltas[tid + s];
                    s_move_idx[tid] = s_move_idx[tid + s];
                }
            }
            __syncthreads();
        }

        uint32_t win_c = s_move_idx[0];
        if (win_c >= n_c) continue;

        uint32_t b_sel = s_cands[win_c].b_idx;
        uint32_t u_out = s_cands[win_c].u_out;
        uint32_t u_in = s_cands[win_c].u_in;
        uint32_t r_mask = s_blocks[b_sel] & ~(1u << u_out);
        uint32_t in_bit = 1u << u_in;
        uint32_t out_bit = 1u << u_out;

        __shared__ int32_t s_exact_delta;
        if (tid == 0) s_exact_delta = 0;
        __syncthreads();

        // Actualizam in paralel s_cov pe listele lui u_out si u_in (disjuncte fiindca in_bit == 0 vs out_bit == 0)
        int32_t my_d = 0;
        const V3PointDraw* p_out = &d_point_draws[u_out * deg_per_pt];
        for (uint32_t j = tid; j < deg_per_pt; j += 128u) {
            uint32_t dm = p_out[j].mask;
            if ((dm & in_bit) == 0u && (uint32_t)__popc(dm & r_mask) == t_hit - 1u) {
                uint16_t id = p_out[j].draw_id;
                uint8_t cval = s_cov[id];
                if (cval == 1u) my_d += 1;
                if (cval > 0u) s_cov[id] = cval - 1u;
            }
        }
        const V3PointDraw* p_in = &d_point_draws[u_in * deg_per_pt];
        for (uint32_t j = tid; j < deg_per_pt; j += 128u) {
            uint32_t dm = p_in[j].mask;
            if ((dm & out_bit) == 0u && (uint32_t)__popc(dm & r_mask) == t_hit - 1u) {
                uint16_t id = p_in[j].draw_id;
                uint8_t cval = s_cov[id];
                if (cval == 0u) my_d -= 1;
                if (cval < 255u) s_cov[id] = cval + 1u;
            }
        }
        if (my_d != 0) atomicAdd(&s_exact_delta, my_d);
        __syncthreads();

        if (tid == 0) {
            s_blocks[b_sel] = r_mask | in_bit;
            uint32_t nb = s_bstep[b_sel] + 1u;
            s_bstep[b_sel] = nb;
            uint32_t tenure = 5u + (uint32_t)(splitmix64_next(&s_rng) % 9u) + (s_uncov_total < 25u ? 3u : 0u);
            s_tabu[b_sel * 32u + u_out] = nb + tenure;
            s_uncov_total = (uint32_t)((int32_t)s_uncov_total + s_exact_delta);
        }
        __syncthreads();

        if (s_uncov_total < s_best_uncov) {
            for (uint32_t i = tid; i < n_blocks; i += 128u) {
                s_best_blocks[i] = s_blocks[i];
            }
            if (tid == 0) {
                s_best_uncov = s_uncov_total;
                if (s_uncov_total == 0u) {
                    atomicExch((int*)d_solved_flag, 1);
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < n_blocks; i += 128u) {
        d_best_blocks_out[chain * n_blocks + i] = s_best_blocks[i];
    }
    if (tid == 0) {
        d_best_uncov_out[chain] = s_best_uncov;
    }
}

// ============================================================================
// 6. Dynamic-Portal Ground-Basin Simulated Annealing Kernel (RTX 5060 sm_120)
//    300 lanturi warp-paralele cu indexare combinadica in registre pentru L(27,6,4,3)=86
// ============================================================================

static const int V3_L27_WARPS_PER_BLK = 5;

__device__ __forceinline__ int v3_nth_bit_d(uint32_t m, uint32_t n) {
    while (n > 0u) {
        m &= (m - 1u);
        --n;
    }
    return __ffs((int)m) - 1;
}

__device__ __forceinline__ uint16_t v3_draw_id_colex_d(
    int x, int ri, int rj, int y, const uint16_t* s_c4
) {
    int a = x, d = y;
    if (a > d) {
        int t = a;
        a = d;
        d = t;
    }
    int p0 = (a < ri) ? a : ri;
    int p3 = (d > rj) ? d : rj;
    int m1 = (a < ri) ? ri : a;
    int m2 = (d > rj) ? rj : d;
    int p1 = (m1 < m2) ? m1 : m2;
    int p2 = (m1 < m2) ? m2 : m1;
    return s_c4[p0 * 5 + 1] + s_c4[p1 * 5 + 2] + s_c4[p2 * 5 + 3] + s_c4[p3 * 5 + 4];
}

struct V3DynamicSaSmem {
    uint16_t c4[28 * 5];
    uint8_t cov[V3_L27_WARPS_PER_BLK][17568];
    uint32_t blocks[V3_L27_WARPS_PER_BLK][96];
    uint32_t elite[V3_L27_WARPS_PER_BLK][96];
};

__global__ void __launch_bounds__(160, 1) k_l27_ground_basin_sa(
    const uint32_t* __restrict__ d_draw_mask,
    const uint32_t* __restrict__ d_warm_86,
    int32_t has_warm,
    uint32_t total_steps,
    uint64_t base_seed,
    uint32_t* __restrict__ d_out_blocks,
    uint32_t* __restrict__ d_out_uncov,
    int32_t* __restrict__ d_solved_flag
) {
    extern __shared__ char raw_smem[];
    V3DynamicSaSmem* smem = reinterpret_cast<V3DynamicSaSmem*>(raw_smem);

    if (threadIdx.x < 28u) {
        int n = (int)threadIdx.x;
        smem->c4[n * 5 + 0] = 1u;
        for (int k = 1; k <= 4; ++k) {
            uint32_t val = 0u;
            if (k <= n) {
                val = 1u;
                for (int i = 0; i < k; ++i) val = val * (uint32_t)(n - i) / (uint32_t)(i + 1);
            }
            smem->c4[n * 5 + k] = (uint16_t)val;
        }
    }
    __syncthreads();

    const int warp_in_blk = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int chain_id = (int)blockIdx.x * V3_L27_WARPS_PER_BLK + warp_in_blk;

    const uint16_t* s_c4 = smem->c4;
    uint8_t* s_cov = smem->cov[warp_in_blk];
    uint32_t* s_blks = smem->blocks[warp_in_blk];
    uint32_t* s_elite = smem->elite[warp_in_blk];

    const uint32_t pm0 = 0x1FFu, pm1 = 0x1FFu << 9, pm2 = 0x1FFu << 18;
    const uint32_t pa0 = pm0 | pm1, pa1 = pm1 | pm2, pa2 = pm2 | pm0;
    const uint32_t ALL27 = (1u << 27) - 1u;
    const int N_BLKS = 86;
    const int M_DRAWS = 17550;

    uint64_t rng = base_seed ^ ((uint64_t)(chain_id + 1) * 0x9E3779B97F4A7C15ULL);
    const bool is_warm = (has_warm != 0) && (chain_id < 100);
    const bool freeze_v2 = is_warm && (chain_id < 50);

    if (lane == 0) {
        if (is_warm) {
            for (int b = 0; b < N_BLKS; ++b) {
                uint32_t m = d_warm_86[b];
                s_blks[b] = m;
                s_elite[b] = m;
            }
        } else {
            int sub = chain_id % 3;
            int n_p01 = (sub == 0) ? 32 : ((sub == 1) ? 33 : 31);
            int n_p12 = 24;
            for (int b = 0; b < N_BLKS; ++b) {
                uint32_t allow = (b < 2) ? pm0
                    : ((b < 4) ? pm1
                    : ((b < 6) ? pm2
                    : ((b < 6 + n_p01) ? pa0
                    : ((b < 6 + n_p01 + n_p12) ? pa1 : pa2))));
                uint32_t n_al = (uint32_t)__popc(allow);
                uint32_t m = 0u;
                while (__popc(m) < 6) {
                    m |= (1u << v3_nth_bit_d(allow, (uint32_t)(splitmix64_next(&rng) % n_al)));
                }
                s_blks[b] = m;
                s_elite[b] = m;
            }
        }
    }
    __syncwarp();

    int local_uncov = 0;
    for (int id = lane; id < M_DRAWS; id += 32) {
        uint32_t dm = d_draw_mask[id];
        int c = 0;
        for (int b = 0; b < N_BLKS; ++b) {
            if (__popc(s_blks[b] & dm) >= 3) c++;
        }
        s_cov[id] = (uint8_t)c;
        if (c == 0) local_uncov++;
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_uncov += __shfl_down_sync(0xFFFFFFFFu, local_uncov, off);
    }
    int uncov_cnt = __shfl_sync(0xFFFFFFFFu, local_uncov, 0);
    int elite_u = uncov_cnt;
    if (uncov_cnt == 0 && lane == 0) {
        atomicExch((int*)d_solved_flag, 1);
    }

    const int pi_tab[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    const int pj_tab[10] = {1, 2, 3, 4, 2, 3, 4, 3, 4, 4};

    float t_start = is_warm ? (0.30f + 0.004f * (float)(chain_id % 100)) : 1.42f;
    float temp = t_start;
    const uint32_t cool_steps = (total_steps * 70u) / 100u + 1u;
    const uint32_t saw_period = (total_steps > cool_steps ? (total_steps - cool_steps) / 4u : 0u) + 1u;

    for (uint32_t step = 1u; step <= total_steps; ++step) {
        if (uncov_cnt == 0) break;
        if ((step & 1023u) == 0u) {
            if (*((volatile int32_t*)d_solved_flag) != 0) break;
            if (step <= cool_steps) {
                float frac = (float)step / (float)cool_steps;
                temp = t_start * powf(0.105f / t_start, frac);
            } else {
                uint32_t rem = step - cool_steps;
                float sub = (float)(rem % saw_period) / (float)saw_period;
                if ((rem % saw_period) < 1024u && uncov_cnt > elite_u + 14) {
                    for (int b = lane; b < N_BLKS; b += 32) s_blks[b] = s_elite[b];
                    __syncwarp();
                    int lu = 0;
                    for (int id = lane; id < M_DRAWS; id += 32) {
                        uint32_t dm = d_draw_mask[id];
                        int c = 0;
                        for (int b = 0; b < N_BLKS; ++b) {
                            if (__popc(s_blks[b] & dm) >= 3) c++;
                        }
                        s_cov[id] = (uint8_t)c;
                        if (c == 0) lu++;
                    }
                    for (int off = 16; off > 0; off >>= 1) lu += __shfl_down_sync(0xFFFFFFFFu, lu, off);
                    uncov_cnt = __shfl_sync(0xFFFFFFFFu, lu, 0);
                }
                temp = 0.22f * powf(0.055f / 0.22f, sub);
            }
        }

        uint32_t move_pack = 0xFFFFFFFFu;
        float u_rand = 0.0f;
        if (lane == 0) {
            int bi = (int)(splitmix64_next(&rng) % (uint64_t)N_BLKS);
            uint32_t bm = s_blks[bi];
            if (!(freeze_v2 && ((bm >> 18) != 0u))) {
                int uo = v3_nth_bit_d(bm, (uint32_t)(splitmix64_next(&rng) % 6ULL));
                uint32_t r_mask = bm & ~(1u << uo);
                uint32_t dom;
                if (freeze_v2) {
                    dom = pa0 & ~bm;
                } else {
                    int c0 = __popc(r_mask & pm0);
                    int c1 = __popc(r_mask & pm1);
                    int c2 = __popc(r_mask & pm2);
                    dom = (c0 == 5 || c1 == 5 || c2 == 5) ? (ALL27 & ~bm)
                        : (c2 == 0 ? (pa0 & ~bm) : (c0 == 0 ? (pa1 & ~bm) : (pa2 & ~bm)));
                }
                uint32_t n_dom = (uint32_t)__popc(dom);
                int ui = v3_nth_bit_d(dom, (uint32_t)(splitmix64_next(&rng) % n_dom));
                move_pack = (uint32_t)bi | ((uint32_t)uo << 8) | ((uint32_t)ui << 16);
                u_rand = (float)((splitmix64_next(&rng) >> 40) * (1.0 / 16777216.0));
            }
        }
        move_pack = __shfl_sync(0xFFFFFFFFu, move_pack, 0);
        if (move_pack == 0xFFFFFFFFu) continue;
        u_rand = __shfl_sync(0xFFFFFFFFu, u_rand, 0);

        int bi = (int)(move_pack & 0xFFu);
        int uo = (int)((move_pack >> 8) & 0xFFu);
        int ui = (int)((move_pack >> 16) & 0xFFu);

        uint32_t bm = s_blks[bi];
        uint32_t r_mask = bm & ~(1u << uo);
        int r[5];
        uint32_t tmp = r_mask;
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            r[i] = __ffs((int)tmp) - 1;
            tmp &= (tmp - 1u);
        }

        uint32_t ext_mask = ALL27 & ~(r_mask | (1u << uo) | (1u << ui));
        int y_pts[20];
        tmp = ext_mask;
        #pragma unroll
        for (int i = 0; i < 20; ++i) {
            y_pts[i] = __ffs((int)tmp) - 1;
            tmp &= (tmp - 1u);
        }

        int l_loc = 0, g_loc = 0;
        uint16_t id_out_reg[7], id_in_reg[7];
        int k_it = 0;
        for (int idx = lane; idx < 200; idx += 32, ++k_it) {
            int p_idx = idx / 20;
            int y = y_pts[idx - p_idx * 20];
            int ri = r[pi_tab[p_idx]], rj = r[pj_tab[p_idx]];
            uint16_t ido = v3_draw_id_colex_d(uo, ri, rj, y, s_c4);
            uint16_t idi = v3_draw_id_colex_d(ui, ri, rj, y, s_c4);
            id_out_reg[k_it] = ido;
            id_in_reg[k_it] = idi;
            if (s_cov[ido] == 1u) l_loc++;
            if (s_cov[idi] == 0u) g_loc++;
        }
        int delta = l_loc - g_loc;
        for (int off = 16; off > 0; off >>= 1) {
            delta += __shfl_down_sync(0xFFFFFFFFu, delta, off);
        }
        delta = __shfl_sync(0xFFFFFFFFu, delta, 0);

        if ((delta <= 0) || (u_rand < __expf(-(float)delta / temp))) {
            for (int i = 0; i < k_it; ++i) {
                s_cov[id_out_reg[i]]--;
                s_cov[id_in_reg[i]]++;
            }
            if (lane == 0) s_blks[bi] = r_mask | (1u << ui);
            __syncwarp();
            uncov_cnt += delta;
            if (uncov_cnt < elite_u || (is_warm && step == 1024u)) {
                elite_u = uncov_cnt;
                for (int b = lane; b < N_BLKS; b += 32) s_elite[b] = s_blks[b];
                __syncwarp();
            }
            if (uncov_cnt == 0) {
                for (int b = lane; b < N_BLKS; b += 32) s_elite[b] = s_blks[b];
                elite_u = 0;
                __syncwarp();
                if (lane == 0) atomicExch((int*)d_solved_flag, 1);
                break;
            }
        }
    }

    for (int b = lane; b < N_BLKS; b += 32) {
        d_out_blocks[chain_id * N_BLKS + b] = s_elite[b];
    }
    if (lane == 0) {
        d_out_uncov[chain_id] = (uint32_t)elite_u;
    }
}

// ============================================================================
// Host Wrappers Extern "C" apelate din Rust FFI
// ============================================================================

extern "C" {

int32_t v3_cuda_popcnt_parity_u64(
    const uint64_t* d_univ,
    uint64_t n_univ,
    const uint64_t* h_probes,
    uint32_t n_probes,
    uint64_t* h_out_hist,
    double* elapsed_ms_out
) {
    if (n_probes == 0 || n_probes > 16) return -2;
    cudaError_t err = cudaMemcpyToSymbol(c_probes, h_probes, n_probes * sizeof(uint64_t));
    if (err != cudaSuccess) return (int32_t)err;

    unsigned long long* d_hist = nullptr;
    size_t hist_bytes = (size_t)n_probes * 8 * sizeof(unsigned long long);
    err = cudaMalloc(&d_hist, hist_bytes);
    if (err != cudaSuccess) return (int32_t)err;
    cudaMemset(d_hist, 0, hist_bytes);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((n_univ + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    k_popcnt_hist_multi<<<blocks, threads>>>(d_univ, n_univ, n_probes, d_hist);

    cudaEventRecord(ev1);
    err = cudaEventSynchronize(ev1);
    if (err != cudaSuccess) {
        cudaFree(d_hist);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        return (int32_t)err;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    err = cudaMemcpy(h_out_hist, d_hist, hist_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_hist);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return (int32_t)err;
}

int32_t v3_cuda_joker_parity_24m(
    const uint64_t* d_joker_24m,
    uint64_t n_univ,
    const uint64_t* h_probes,
    uint32_t n_probes,
    uint64_t* h_out_hist12,
    double* elapsed_ms_out
) {
    if (n_probes == 0 || n_probes > 16) return -2;
    cudaError_t err = cudaMemcpyToSymbol(c_probes, h_probes, n_probes * sizeof(uint64_t));
    if (err != cudaSuccess) return (int32_t)err;

    unsigned long long* d_hist = nullptr;
    size_t hist_bytes = (size_t)n_probes * 12 * sizeof(unsigned long long);
    err = cudaMalloc(&d_hist, hist_bytes);
    if (err != cudaSuccess) return (int32_t)err;
    cudaMemset(d_hist, 0, hist_bytes);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    k_joker_packed_parity<<<4096, 256>>>(d_joker_24m, n_univ, n_probes, d_hist);

    cudaEventRecord(ev1);
    err = cudaEventSynchronize(ev1);
    if (err != cudaSuccess) {
        cudaFree(d_hist);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        return (int32_t)err;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    err = cudaMemcpy(h_out_hist12, d_hist, hist_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_hist);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return (int32_t)err;
}

int32_t v3_cuda_verify_covering_u64(
    const uint64_t* d_draws,
    uint64_t n_draws,
    const uint64_t* h_tickets,
    uint32_t n_tickets,
    uint64_t* h_out_hist8,
    double* elapsed_ms_out
) {
    if (n_tickets == 0) return -2;
    uint64_t* d_tickets = nullptr;
    unsigned long long* d_hist = nullptr;
    cudaError_t err = cudaMalloc(&d_tickets, (size_t)n_tickets * sizeof(uint64_t));
    if (err != cudaSuccess) return (int32_t)err;
    err = cudaMalloc(&d_hist, 8 * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        cudaFree(d_tickets);
        return (int32_t)err;
    }
    cudaMemcpy(d_tickets, h_tickets, (size_t)n_tickets * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 8 * sizeof(unsigned long long));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((n_draws + threads - 1) / threads);
    k_verify_covering<<<blocks, threads>>>(d_draws, n_draws, d_tickets, n_tickets, d_hist);

    cudaEventRecord(ev1);
    err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        err = cudaMemcpy(h_out_hist8, d_hist, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_tickets);
    cudaFree(d_hist);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return (int32_t)err;
}

int32_t v3_cuda_verify_joker_24m(
    const uint64_t* d_joker_24m,
    uint64_t n_states,
    const uint64_t* h_tickets_packed,
    uint32_t n_tickets,
    uint64_t* h_out_cat9,
    double* elapsed_ms_out
) {
    if (n_tickets == 0) return -2;
    uint64_t* d_tickets = nullptr;
    unsigned long long* d_hist = nullptr;
    cudaError_t err = cudaMalloc(&d_tickets, (size_t)n_tickets * sizeof(uint64_t));
    if (err != cudaSuccess) return (int32_t)err;
    err = cudaMalloc(&d_hist, 9 * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        cudaFree(d_tickets);
        return (int32_t)err;
    }
    cudaMemcpy(d_tickets, h_tickets_packed, (size_t)n_tickets * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 9 * sizeof(unsigned long long));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((n_states + threads - 1) / threads);
    k_verify_joker_packed_24m<<<blocks, threads>>>(d_joker_24m, n_states, d_tickets, n_tickets, d_hist);

    cudaEventRecord(ev1);
    err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        err = cudaMemcpy(h_out_cat9, d_hist, 9 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_tickets);
    cudaFree(d_hist);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return (int32_t)err;
}

int32_t v3_cuda_init_uncov_bits(uint64_t* d_uncov_bits, uint64_t n_draws) {
    uint64_t n_words = (n_draws + 63ULL) >> 6;
    uint32_t blocks = (uint32_t)((n_words + 255ULL) / 256ULL);
    k_init_uncov_bits<<<blocks, 256>>>(d_uncov_bits, n_draws);
    return (int32_t)cudaDeviceSynchronize();
}

int32_t v3_cuda_mark_covered_bits(
    const uint64_t* d_draws,
    uint64_t n_draws,
    uint64_t* d_uncov_bits,
    uint64_t chosen_mask,
    int32_t min_hits,
    uint64_t* rem_out
) {
    unsigned long long* d_rem = nullptr;
    cudaError_t err = cudaMalloc(&d_rem, sizeof(unsigned long long));
    if (err != cudaSuccess) return (int32_t)err;
    cudaMemset(d_rem, 0, sizeof(unsigned long long));

    uint64_t n_words = (n_draws + 63ULL) >> 6;
    uint32_t blocks = (uint32_t)((n_words + 255ULL) / 256ULL);
    k_mark_covered_bits<<<blocks, 256>>>(d_draws, n_draws, d_uncov_bits, chosen_mask, min_hits, d_rem);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess && rem_out) {
        unsigned long long h_rem = 0ULL;
        cudaMemcpy(&h_rem, d_rem, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        *rem_out = (uint64_t)h_rem;
    }
    cudaFree(d_rem);
    return (int32_t)err;
}

int32_t v3_cuda_greedy_scores_bitpacked(
    const uint64_t* d_draws,
    uint64_t n_draws,
    const uint64_t* d_uncov_bits,
    const uint64_t* h_cands,
    uint32_t n_cands,
    int32_t min_hits,
    uint32_t* h_gains_out
) {
    if (n_cands == 0) return 0;
    uint64_t* d_cands = nullptr;
    uint32_t* d_gains = nullptr;
    cudaError_t err = cudaMalloc(&d_cands, (size_t)n_cands * sizeof(uint64_t));
    if (err != cudaSuccess) return (int32_t)err;
    err = cudaMalloc(&d_gains, (size_t)n_cands * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(d_cands);
        return (int32_t)err;
    }
    cudaMemcpy(d_cands, h_cands, (size_t)n_cands * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint32_t blocks = (n_cands + 255u) / 256u;
    k_greedy_scores_bitpacked<<<blocks, 256>>>(d_draws, n_draws, d_uncov_bits, d_cands, n_cands, min_hits, d_gains);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(h_gains_out, d_gains, (size_t)n_cands * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_cands);
    cudaFree(d_gains);
    return (int32_t)err;
}

int32_t v3_cuda_joker_init_uncov20(uint32_t* d_uncov20, uint32_t n_base) {
    uint32_t blocks = (n_base + 255u) / 256u;
    k_joker_init_uncov20<<<blocks, 256>>>(d_uncov20, n_base);
    return (int32_t)cudaDeviceSynchronize();
}

int32_t v3_cuda_joker_mark_covered20(
    const uint64_t* d_base_draws,
    uint32_t n_base,
    uint32_t* d_uncov20,
    uint64_t chosen_packed,
    uint64_t* rem_out
) {
    unsigned long long* d_rem = nullptr;
    cudaError_t err = cudaMalloc(&d_rem, sizeof(unsigned long long));
    if (err != cudaSuccess) return (int32_t)err;
    cudaMemset(d_rem, 0, sizeof(unsigned long long));

    uint32_t blocks = (n_base + 255u) / 256u;
    k_joker_mark_covered20<<<blocks, 256>>>(d_base_draws, n_base, d_uncov20, chosen_packed, d_rem);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess && rem_out) {
        unsigned long long h_rem = 0ULL;
        cudaMemcpy(&h_rem, d_rem, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        *rem_out = (uint64_t)h_rem;
    }
    cudaFree(d_rem);
    return (int32_t)err;
}

int32_t v3_cuda_joker_greedy_scores20(
    const uint64_t* d_base_draws,
    uint32_t n_base,
    const uint32_t* d_uncov20,
    const uint64_t* h_cands_packed,
    uint32_t n_cands,
    uint32_t* h_gains_out
) {
    if (n_cands == 0) return 0;
    uint64_t* d_cands = nullptr;
    uint32_t* d_gains = nullptr;
    cudaError_t err = cudaMalloc(&d_cands, (size_t)n_cands * sizeof(uint64_t));
    if (err != cudaSuccess) return (int32_t)err;
    err = cudaMalloc(&d_gains, (size_t)n_cands * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(d_cands);
        return (int32_t)err;
    }
    cudaMemcpy(d_cands, h_cands_packed, (size_t)n_cands * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint32_t blocks = (n_cands + 255u) / 256u;
    k_joker_greedy_scores20<<<blocks, 256>>>(d_base_draws, n_base, d_uncov20, d_cands, n_cands, d_gains);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(h_gains_out, d_gains, (size_t)n_cands * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_cands);
    cudaFree(d_gains);
    return (int32_t)err;
}

int32_t v3_cuda_run_tabu_shared_16384(
    uint32_t v_pts,
    uint32_t k_pick,
    uint32_t t_hit,
    uint32_t m_draws,
    const uint32_t* h_draw_masks,
    const V3PointDraw* h_point_draws,
    uint32_t deg_per_pt,
    uint32_t n_blocks,
    const uint32_t* h_init_blocks_128,
    uint32_t max_iters,
    uint64_t base_seed,
    uint32_t* h_best_blocks_128_out,
    uint32_t* h_best_uncov_128_out
) {
    if (v_pts > 32 || m_draws > 24576 || n_blocks > 128) return -2;

    uint32_t* d_draw_masks = nullptr;
    V3PointDraw* d_point_draws = nullptr;
    uint32_t* d_init = nullptr;
    uint32_t* d_best = nullptr;
    uint32_t* d_uncov = nullptr;
    int32_t* d_solved = nullptr;

    size_t draw_b = (size_t)m_draws * sizeof(uint32_t);
    size_t pd_b = (size_t)v_pts * deg_per_pt * sizeof(V3PointDraw);
    size_t blk_b = 128ULL * n_blocks * sizeof(uint32_t);

    if (cudaMalloc(&d_draw_masks, draw_b) != cudaSuccess) return -3;
    if (cudaMalloc(&d_point_draws, pd_b) != cudaSuccess) { cudaFree(d_draw_masks); return -3; }
    if (cudaMalloc(&d_init, blk_b) != cudaSuccess) { cudaFree(d_draw_masks); cudaFree(d_point_draws); return -3; }
    if (cudaMalloc(&d_best, blk_b) != cudaSuccess) { cudaFree(d_draw_masks); cudaFree(d_point_draws); cudaFree(d_init); return -3; }
    if (cudaMalloc(&d_uncov, 128 * sizeof(uint32_t)) != cudaSuccess) { cudaFree(d_draw_masks); cudaFree(d_point_draws); cudaFree(d_init); cudaFree(d_best); return -3; }
    if (cudaMalloc(&d_solved, sizeof(int32_t)) != cudaSuccess) { cudaFree(d_draw_masks); cudaFree(d_point_draws); cudaFree(d_init); cudaFree(d_best); cudaFree(d_uncov); return -3; }

    cudaMemcpy(d_draw_masks, h_draw_masks, draw_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_point_draws, h_point_draws, pd_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_init, h_init_blocks_128, blk_b, cudaMemcpyHostToDevice);
    cudaMemset(d_solved, 0, sizeof(int32_t));

    // 128 blocuri x 128 fire = 16,384 fire CUDA active simultan pe RTX 5060 sm_120!
    k_tabu_shared_16384<<<128, 128>>>(
        v_pts, k_pick, t_hit, m_draws,
        d_draw_masks, d_point_draws, deg_per_pt,
        n_blocks, d_init, max_iters, base_seed,
        d_best, d_uncov, d_solved
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemcpy(h_best_blocks_128_out, d_best, blk_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_uncov_128_out, d_uncov, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_draw_masks);
    cudaFree(d_point_draws);
    cudaFree(d_init);
    cudaFree(d_best);
    cudaFree(d_uncov);
    cudaFree(d_solved);
    return (int32_t)err;
}

int32_t v3_cuda_solve_l27_86(
    const uint32_t* h_warm_86,
    int32_t has_warm,
    uint32_t total_steps,
    uint64_t base_seed,
    uint32_t* h_best_86_out,
    uint32_t* best_uncov_out
) {
    const int V = 27;
    const int M = 17550;
    const int N_BLKS = 86;
    const int GRID_BLKS = 60;
    const int N_CHAINS = GRID_BLKS * V3_L27_WARPS_PER_BLK;

    uint16_t c4[28][5];
    memset(c4, 0, sizeof(c4));
    for (int n = 0; n <= 27; ++n) {
        c4[n][0] = 1u;
        for (int k = 1; k <= 4 && k <= n; ++k) {
            c4[n][k] = (uint16_t)(c4[n - 1][k - 1] + c4[n - 1][k]);
        }
    }
    uint32_t h_draw_mask[17550];
    for (int d = 3; d < V; ++d) {
        for (int c = 2; c < d; ++c) {
            for (int b = 1; b < c; ++b) {
                for (int a = 0; a < b; ++a) {
                    uint16_t id = c4[a][1] + c4[b][2] + c4[c][3] + c4[d][4];
                    h_draw_mask[id] = (1u << a) | (1u << b) | (1u << c) | (1u << d);
                }
            }
        }
    }

    uint32_t *d_draw_mask = nullptr, *d_warm = nullptr, *d_out_blocks = nullptr, *d_out_uncov = nullptr;
    int32_t* d_solved = nullptr;
    if (cudaMalloc(&d_draw_mask, M * sizeof(uint32_t)) != cudaSuccess) return -3;
    if (cudaMalloc(&d_warm, N_BLKS * sizeof(uint32_t)) != cudaSuccess) { cudaFree(d_draw_mask); return -3; }
    if (cudaMalloc(&d_out_blocks, N_CHAINS * N_BLKS * sizeof(uint32_t)) != cudaSuccess) { cudaFree(d_draw_mask); cudaFree(d_warm); return -3; }
    if (cudaMalloc(&d_out_uncov, N_CHAINS * sizeof(uint32_t)) != cudaSuccess) { cudaFree(d_draw_mask); cudaFree(d_warm); cudaFree(d_out_blocks); return -3; }
    if (cudaMalloc(&d_solved, sizeof(int32_t)) != cudaSuccess) { cudaFree(d_draw_mask); cudaFree(d_warm); cudaFree(d_out_blocks); cudaFree(d_out_uncov); return -3; }

    cudaMemcpy(d_draw_mask, h_draw_mask, M * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (has_warm != 0 && h_warm_86 != nullptr) {
        cudaMemcpy(d_warm, h_warm_86, N_BLKS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_warm, 0, N_BLKS * sizeof(uint32_t));
    }
    cudaMemset(d_solved, 0, sizeof(int32_t));

    size_t smem_bytes = sizeof(V3DynamicSaSmem);
    cudaFuncSetAttribute(k_l27_ground_basin_sa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    k_l27_ground_basin_sa<<<GRID_BLKS, V3_L27_WARPS_PER_BLK * 32, smem_bytes>>>(
        d_draw_mask, d_warm, has_warm, total_steps, base_seed, d_out_blocks, d_out_uncov, d_solved
    );
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        uint32_t h_uncov[300];
        uint32_t h_blocks[300 * 86];
        cudaMemcpy(h_uncov, d_out_uncov, N_CHAINS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blocks, d_out_blocks, N_CHAINS * N_BLKS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        int best_c = 0;
        for (int c = 1; c < N_CHAINS; ++c) {
            if (h_uncov[c] < h_uncov[best_c]) best_c = c;
        }
        if (best_uncov_out) *best_uncov_out = h_uncov[best_c];
        if (h_best_86_out) {
            memcpy(h_best_86_out, &h_blocks[best_c * N_BLKS], N_BLKS * sizeof(uint32_t));
        }
    }

    cudaFree(d_draw_mask);
    cudaFree(d_warm);
    cudaFree(d_out_blocks);
    cudaFree(d_out_uncov);
    cudaFree(d_solved);
    return (int32_t)err;
}

} // extern "C"

// ============================================================================
// 5. Sesiunea 2 (S2 / Poarta A3) — Motorul Pari-Mutuel Cauzal PM-V3 &
//    Cubul Bootstrap 1000x peste toate cele 39,077,004 bilete pe RTX 5060 sm_120
// ============================================================================

struct V3CrowdParamsF64 {
    double gamma[50];
    double theta_cons;
    double theta_low31;
    double theta_arith;
    double theta_grid;
    double lw_shift;
    double pi_joker[20];
    double sales_n;
    double funds[8];
    double nb_r[8];
    double fixed_iv;
    int32_t grid_cols;
    int32_t share_mode; // 0 = Negative Binomial exact, 1 = Poisson, 2 = Jensen 1/(1+mu)
};

__constant__ uint32_t c_pmv3_binom[50][7];
__constant__ V3CrowdParamsF64 c_pmv3_params;

static bool g_pmv3_binom_ready = false;

static int32_t ensure_pmv3_binom_init() {
    if (g_pmv3_binom_ready) return 0;
    uint32_t h_b[50][7];
    memset(h_b, 0, sizeof(h_b));
    for (int n = 0; n < 50; ++n) {
        h_b[n][0] = 1U;
        for (int k = 1; k <= 6 && k <= n; ++k) {
            h_b[n][k] = h_b[n - 1][k - 1] + h_b[n - 1][k];
        }
    }
    cudaError_t err = cudaMemcpyToSymbol(c_pmv3_binom, h_b, sizeof(h_b));
    if (err != cudaSuccess) return (int32_t)err;
    g_pmv3_binom_ready = true;
    return 0;
}

__device__ __forceinline__ uint32_t pmv3_rank_lex(uint64_t mask, int n, int k, uint32_t total_comb) {
    uint32_t rem = 0U;
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (i >= k) break;
        int a = __ffsll((long long)m) - 1;
        m &= (m - 1ULL);
        rem += c_pmv3_binom[n - 1 - a][k - i];
    }
    return total_comb - 1U - rem;
}

__device__ __forceinline__ double pmv3_ticket_lw(uint64_t mask, int k, const V3CrowdParamsF64& p) {
    int pos[6];
    uint64_t m = mask;
    double lw = 0.0;
    for (int i = 0; i < k; ++i) {
        int a = __ffsll((long long)m) - 1;
        m &= (m - 1ULL);
        pos[i] = a;
        lw += p.gamma[a];
    }
    int cons = 0;
    for (int i = 0; i < k - 1; ++i) {
        if (pos[i + 1] - pos[i] == 1) cons += 1;
    }
    if (cons > 0) lw += p.theta_cons * (double)cons;
    if (pos[k - 1] <= 30) lw += p.theta_low31;

    int d0 = pos[1] - pos[0];
    bool is_ap = true;
    for (int i = 1; i < k - 1; ++i) {
        if (pos[i + 1] - pos[i] != d0) { is_ap = false; break; }
    }
    if (is_ap) lw += p.theta_arith;

    int rcnt[10] = {0,0,0,0,0,0,0,0,0,0};
    int ccnt[8]  = {0,0,0,0,0,0,0,0};
    int gcols = p.grid_cols;
    int max_rc = 0;
    for (int i = 0; i < k; ++i) {
        int r = pos[i] / gcols;
        int c = pos[i] % gcols;
        int vr = ++rcnt[r];
        int vc = ++ccnt[c];
        if (vr > max_rc) max_rc = vr;
        if (vc > max_rc) max_rc = vc;
    }
    if (max_rc >= k - 1) lw += p.theta_grid;
    return lw;
}

__device__ __forceinline__ double pmv3_g_share(double mu, double r, int32_t mode) {
    if (mu <= 0.0) return 1.0;
    if (mode == 2) {
        return 1.0 / (1.0 + mu);
    }
    if (mode == 1 || r >= 1.0e12) {
        if (mu < 1.0e-8) {
            return 1.0 - 0.5 * mu + (mu * mu) * (1.0 / 6.0) - (mu * mu * mu) * (1.0 / 24.0);
        }
        return -expm1(-mu) / mu;
    }
    if (mu < 1.0e-7) {
        double t2 = (mu * mu) * ((r + 1.0) / (6.0 * r));
        double t3 = (mu * mu * mu) * ((r + 1.0) * (r + 2.0) / (24.0 * r * r));
        return 1.0 - 0.5 * mu + t2 - t3;
    }
    if (fabs(r - 1.0) < 1.0e-7) {
        double eps = 1.0 - r;
        double l_val = log1p(mu / r);
        double el = eps * l_val;
        return (r / mu) * l_val * (1.0 + 0.5 * el + (el * el) * (1.0 / 6.0));
    }
    double l_val = log1p(mu / r);
    double num = expm1((1.0 - r) * l_val);
    return (r * num) / (mu * (1.0 - r));
}

// Acumulator exact pe 128 biti (rezolutie 2^-96 ~ 1.26e-29, 100% asociativ intre GPU si CPU!)
__device__ __forceinline__ void pmv3_double_to_u128_s96(
    double x,
    unsigned long long* lo_out,
    unsigned long long* hi_out
) {
    if (x <= 0.0) {
        *lo_out = 0ULL;
        *hi_out = 0ULL;
        return;
    }
    double x32 = ldexp(x, 32);
    double h_fl = floor(x32);
    unsigned long long h = (unsigned long long)h_fl;
    double frac = x32 - h_fl;
    double mid_fl = floor(ldexp(frac, 32));
    double bot_fl = floor(ldexp(frac - ldexp(mid_fl, -32), 64) + 0.5);
    unsigned long long mid_u = (unsigned long long)mid_fl;
    unsigned long long bot_u = (bot_fl >= 4294967296.0) ? 4294967296ULL : (unsigned long long)bot_fl;
    unsigned long long carry_mid = bot_u >> 32;
    bot_u &= 0xFFFFFFFFULL;
    mid_u += carry_mid;
    if (mid_u >= 4294967296ULL) {
        h += (mid_u >> 32);
        mid_u &= 0xFFFFFFFFULL;
    }
    *lo_out = (mid_u << 32) | bot_u;
    *hi_out = h;
}

__device__ __forceinline__ double pmv3_u128_s96_to_double(
    unsigned long long lo,
    unsigned long long hi
) {
    double a = ldexp((double)hi, -32);
    double b = ldexp((double)(lo >> 32), -64);
    double c = ldexp((double)(lo & 0xFFFFFFFFULL), -96);
    return a + (b + c);
}

__device__ __forceinline__ void pmv3_atomic_add_u128(
    unsigned long long* lo_arr,
    unsigned long long* hi_arr,
    uint32_t idx,
    double val
) {
    unsigned long long vlo, vhi;
    pmv3_double_to_u128_s96(val, &vlo, &vhi);
    if ((vlo | vhi) == 0ULL) return;
    unsigned long long old_lo = atomicAdd(&lo_arr[idx], vlo);
    unsigned long long carry = (old_lo + vlo < old_lo) ? 1ULL : 0ULL;
    unsigned long long add_hi = vhi + carry;
    if (add_hi != 0ULL) {
        atomicAdd(&hi_arr[idx], add_hi);
    }
}

__global__ void k_pmv3_sum_z_u128(
    const uint64_t* __restrict__ d_masks,
    uint32_t n_tickets,
    int k_pick,
    double* __restrict__ d_unnorm_w,
    unsigned long long* __restrict__ d_z_lo,
    unsigned long long* __restrict__ d_z_hi
) {
    __shared__ unsigned long long s_lo[8];
    __shared__ unsigned long long s_hi[8];
    if (threadIdx.x < 8) { s_lo[threadIdx.x] = 0ULL; s_hi[threadIdx.x] = 0ULL; }
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long vlo = 0ULL, vhi = 0ULL;
    if (idx < n_tickets) {
        double lw = pmv3_ticket_lw(d_masks[idx], k_pick, c_pmv3_params);
        double w = exp(lw - c_pmv3_params.lw_shift);
        d_unnorm_w[idx] = w;
        pmv3_double_to_u128_s96(w, &vlo, &vhi);
    }
    int lane = threadIdx.x & 7;
    if ((vlo | vhi) != 0ULL) {
        unsigned long long old_lo = atomicAdd(&s_lo[lane], vlo);
        unsigned long long c = (old_lo + vlo < old_lo) ? 1ULL : 0ULL;
        if (vhi + c) atomicAdd(&s_hi[lane], vhi + c);
    }
    __syncthreads();
    if (threadIdx.x < 8) {
        unsigned long long bl = s_lo[threadIdx.x];
        unsigned long long bh = s_hi[threadIdx.x];
        if ((bl | bh) != 0ULL) {
            unsigned long long old_lo = atomicAdd(d_z_lo, bl);
            unsigned long long c = (old_lo + bl < old_lo) ? 1ULL : 0ULL;
            if (bh + c) atomicAdd(d_z_hi, bh + c);
        }
    }
}

// ----------------------------------------------------------------------------
// 5.1 Loto 6/49 — Evaluare exacta f64 pe toate cele 13,983,816 bilete
// ----------------------------------------------------------------------------

__global__ void k_pmv3_649_fwd_u128(
    const uint64_t* __restrict__ d_masks,
    double* __restrict__ d_p,
    double inv_z,
    unsigned long long* __restrict__ d_s5_lo,
    unsigned long long* __restrict__ d_s5_hi,
    unsigned long long* __restrict__ d_s4_lo,
    unsigned long long* __restrict__ d_s4_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    double p = d_p[idx] * inv_z;
    d_p[idx] = p;
    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint32_t r5 = pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U);
        pmv3_atomic_add_u128(d_s5_lo, d_s5_hi, r5, p);
    }
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            uint32_t r4 = pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U);
            pmv3_atomic_add_u128(d_s4_lo, d_s4_hi, r4, p);
        }
    }
}

__global__ void k_pmv3_u128_to_f64(
    const unsigned long long* __restrict__ d_lo,
    const unsigned long long* __restrict__ d_hi,
    double* __restrict__ d_out,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = pmv3_u128_s96_to_double(d_lo[idx], d_hi[idx]);
    }
}

__global__ void k_pmv3_649_bwd_u128(
    const uint64_t* __restrict__ d_masks,
    const double* __restrict__ d_p,
    const double* __restrict__ d_s5,
    const double* __restrict__ d_s4,
    double off_5,
    double* __restrict__ d_yt6,
    unsigned long long* __restrict__ d_u5_lo,
    unsigned long long* __restrict__ d_u5_hi,
    unsigned long long* __restrict__ d_u4_lo,
    unsigned long long* __restrict__ d_u4_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    uint32_t r5[6];
    double t5 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        r5[i] = pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U);
        t5 += d_s5[r5[i]];
    }
    uint32_t r4[15];
    double t4 = 0.0;
    int pos = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            uint32_t rk = pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U);
            r4[pos++] = rk;
            t4 += d_s4[rk];
        }
    }
    double p_d = d_p[idx];
    double q6 = (p_d > 0.0) ? p_d : 0.0;
    double q5 = t5 - 6.0 * p_d;
    if (q5 < 0.0) q5 = 0.0;
    double q4 = t4 - 5.0 * t5 + 15.0 * p_d;
    if (q4 < 0.0) q4 = 0.0;

    const double inv_omega = 1.0 / 13983816.0;
    double n_s = c_pmv3_params.sales_n;
    int32_t smode = c_pmv3_params.share_mode;
    double y6 = (c_pmv3_params.funds[0] * inv_omega) * pmv3_g_share(n_s * q6, c_pmv3_params.nb_r[0], smode);
    double y5 = (c_pmv3_params.funds[1] * inv_omega) * pmv3_g_share(n_s * q5, c_pmv3_params.nb_r[1], smode);
    double y4 = (c_pmv3_params.funds[2] * inv_omega) * pmv3_g_share(n_s * q4, c_pmv3_params.nb_r[2], smode);

    double yt4 = y4;
    double yt5_shift = (y5 - 5.0 * y4) + off_5;
    if (yt5_shift < 0.0) yt5_shift = 0.0;
    d_yt6[idx] = y6 - 6.0 * y5 + 15.0 * y4;

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        pmv3_atomic_add_u128(d_u5_lo, d_u5_hi, r5[i], yt5_shift);
    }
    #pragma unroll
    for (int i = 0; i < 15; ++i) {
        pmv3_atomic_add_u128(d_u4_lo, d_u4_hi, r4[i], yt4);
    }
}

__global__ void k_pmv3_649_gather_f64(
    const uint64_t* __restrict__ d_masks,
    const double* __restrict__ d_yt6,
    const double* __restrict__ d_u5,
    const double* __restrict__ d_u4,
    double off_5,
    double fixed_ev_iv,
    double* __restrict__ d_ev_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    double sum_u5 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        sum_u5 += d_u5[pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U)];
    }
    double sum_u4 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            sum_u4 += d_u4[pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U)];
        }
    }
    d_ev_out[idx] = d_yt6[idx] + (sum_u5 - 264.0 * off_5) + sum_u4 + fixed_ev_iv;
}

// ----------------------------------------------------------------------------
// 5.2 Loto 5/40 — Evaluare exacta f64 pe 658,008 bilete x 3,838,380 extrageri
// ----------------------------------------------------------------------------

__global__ void k_pmv3_540_fwd_u128(
    const uint64_t* __restrict__ d_tickets,
    double* __restrict__ d_p,
    double inv_z,
    unsigned long long* __restrict__ d_s4_lo,
    unsigned long long* __restrict__ d_s4_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 658008U) return;
    double p = d_p[idx] * inv_z;
    d_p[idx] = p;
    uint64_t mask = d_tickets[idx];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        m ^= lsb;
        uint32_t r4 = pmv3_rank_lex(mask ^ lsb, 40, 4, 91390U);
        pmv3_atomic_add_u128(d_s4_lo, d_s4_hi, r4, p);
    }
}

__global__ void k_pmv3_540_bwd_u128(
    const uint64_t* __restrict__ d_draws6,
    const double* __restrict__ d_p,
    const double* __restrict__ d_s4,
    double off_540,
    unsigned long long* __restrict__ d_u5_lo,
    unsigned long long* __restrict__ d_u5_hi,
    unsigned long long* __restrict__ d_u4_lo,
    unsigned long long* __restrict__ d_u4_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3838380U) return;
    uint64_t mask = d_draws6[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    uint32_t rk5[6];
    double pf[6];
    double t5 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint32_t rk = pmv3_rank_lex(mask ^ bits[i], 40, 5, 658008U);
        rk5[i] = rk;
        double pv = d_p[rk];
        pf[i] = pv;
        t5 += pv;
    }
    uint32_t rk4[15];
    double t4 = 0.0;
    int pos = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            uint32_t r4 = pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 40, 4, 91390U);
            rk4[pos++] = r4;
            t4 += d_s4[r4];
        }
    }
    double n_s = c_pmv3_params.sales_n;
    int32_t smode = c_pmv3_params.share_mode;
    double scale_ii = c_pmv3_params.funds[1] / (6.0 * 3838380.0);
    double pay_ii[6];
    double sum_pay_ii = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        double q_ii = t5 - pf[i];
        if (q_ii < 0.0) q_ii = 0.0;
        double py = scale_ii * pmv3_g_share(n_s * q_ii, c_pmv3_params.nb_r[1], smode);
        pay_ii[i] = py;
        sum_pay_ii += py;
    }

    double q_iii = t4 - 5.0 * t5;
    if (q_iii < 0.0) q_iii = 0.0;
    double y_iii = (c_pmv3_params.funds[2] / 3838380.0) * pmv3_g_share(n_s * q_iii, c_pmv3_params.nb_r[2], smode);

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        double val5 = (sum_pay_ii - pay_ii[i]) - 5.0 * y_iii + off_540;
        if (val5 < 0.0) val5 = 0.0;
        pmv3_atomic_add_u128(d_u5_lo, d_u5_hi, rk5[i], val5);
    }
    #pragma unroll
    for (int i = 0; i < 15; ++i) {
        pmv3_atomic_add_u128(d_u4_lo, d_u4_hi, rk4[i], y_iii);
    }
}

__global__ void k_pmv3_540_gather_f64(
    const uint64_t* __restrict__ d_tickets,
    const double* __restrict__ d_p,
    const double* __restrict__ d_u5,
    const double* __restrict__ d_u4,
    double off_540,
    double* __restrict__ d_ev_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 658008U) return;
    uint64_t mask = d_tickets[idx];
    uint64_t m = mask;
    double sum_u4 = 0.0;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        m ^= lsb;
        sum_u4 += d_u4[pmv3_rank_lex(mask ^ lsb, 40, 4, 91390U)];
    }
    double ev_i = (c_pmv3_params.funds[0] / 658008.0)
        * pmv3_g_share(c_pmv3_params.sales_n * d_p[idx], c_pmv3_params.nb_r[0], c_pmv3_params.share_mode);
    d_ev_out[idx] = ev_i + (d_u5[idx] - 35.0 * off_540) + sum_u4;
}

// ----------------------------------------------------------------------------
// 5.3 Joker (5/45 + 1/20) — Evaluare exacta f64 pe toate cele 24,435,180 bilete
// ----------------------------------------------------------------------------

__global__ void k_pmv3_joker_fwd_u128(
    const uint64_t* __restrict__ d_base45,
    double* __restrict__ d_p,
    double inv_z,
    unsigned long long* __restrict__ d_s4_lo,
    unsigned long long* __restrict__ d_s4_hi,
    unsigned long long* __restrict__ d_s3_lo,
    unsigned long long* __restrict__ d_s3_hi,
    unsigned long long* __restrict__ d_s2_lo,
    unsigned long long* __restrict__ d_s2_hi,
    unsigned long long* __restrict__ d_s1_lo,
    unsigned long long* __restrict__ d_s1_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    double p = d_p[idx] * inv_z;
    d_p[idx] = p;
    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    // m=1 (5 submultimi)
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        pmv3_atomic_add_u128(d_s1_lo, d_s1_hi, (uint32_t)pos[i], p);
    }
    // m=4 (5 submultimi)
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint32_t r4 = pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U);
        pmv3_atomic_add_u128(d_s4_lo, d_s4_hi, r4, p);
    }
    // m=2 si m=3 (10 perechi / triplete complementare!)
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            uint64_t pair_m = bits[i] | bits[j];
            uint32_t r2 = pmv3_rank_lex(pair_m, 45, 2, 990U);
            uint32_t r3 = pmv3_rank_lex(mask ^ pair_m, 45, 3, 14190U);
            pmv3_atomic_add_u128(d_s2_lo, d_s2_hi, r2, p);
            pmv3_atomic_add_u128(d_s3_lo, d_s3_hi, r3, p);
        }
    }
}

__global__ void k_pmv3_joker_compute_q_f64(
    const uint64_t* __restrict__ d_base45,
    const double* __restrict__ d_p,
    const double* __restrict__ d_s4,
    const double* __restrict__ d_s3,
    const double* __restrict__ d_s2,
    const double* __restrict__ d_s1,
    double* __restrict__ d_q4,
    double* __restrict__ d_q3,
    double* __restrict__ d_q2,
    double* __restrict__ d_q1
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    double t5 = d_p[idx];
    double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        t1 += d_s1[pos[i]];
        t4 += d_s4[pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U)];
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            uint64_t pm = bits[i] | bits[j];
            t2 += d_s2[pmv3_rank_lex(pm, 45, 2, 990U)];
            t3 += d_s3[pmv3_rank_lex(mask ^ pm, 45, 3, 14190U)];
        }
    }
    double q4 = t4 - 5.0 * t5;
    double q3 = t3 - 4.0 * t4 + 10.0 * t5;
    double q2 = t2 - 3.0 * t3 + 6.0 * t4 - 10.0 * t5;
    double q1 = t1 - 2.0 * t2 + 3.0 * t3 - 4.0 * t4 + 5.0 * t5;
    d_q4[idx] = (q4 > 0.0) ? q4 : 0.0;
    d_q3[idx] = (q3 > 0.0) ? q3 : 0.0;
    d_q2[idx] = (q2 > 0.0) ? q2 : 0.0;
    d_q1[idx] = (q1 > 0.0) ? q1 : 0.0;
}

__global__ void k_pmv3_joker_bwd_single_j_u128(
    const uint64_t* __restrict__ d_base45,
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    const double* __restrict__ d_q3,
    const double* __restrict__ d_q2,
    const double* __restrict__ d_q1,
    int32_t j_ticket,
    double off_4,
    double off_3,
    double off_2,
    double* __restrict__ d_yt5,
    unsigned long long* __restrict__ d_u4_lo,
    unsigned long long* __restrict__ d_u4_hi,
    unsigned long long* __restrict__ d_u3_lo,
    unsigned long long* __restrict__ d_u3_hi,
    unsigned long long* __restrict__ d_u2_lo,
    unsigned long long* __restrict__ d_u2_hi,
    unsigned long long* __restrict__ d_u1_lo,
    unsigned long long* __restrict__ d_u1_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    const double inv_omega = 1.0 / 24435180.0;
    double n_s = c_pmv3_params.sales_n;
    int32_t smode = c_pmv3_params.share_mode;
    double pij = c_pmv3_params.pi_joker[j_ticket];

    double q5 = (d_q5[idx] > 0.0) ? d_q5[idx] : 0.0;
    double q4 = d_q4[idx];
    double q3 = d_q3[idx];
    double q2 = d_q2[idx];
    double q1 = d_q1[idx];

    double g_ii = 0.0, g_iv = 0.0, g_vi = 0.0;
    #pragma unroll
    for (int jj = 0; jj < 20; ++jj) {
        if (jj != j_ticket) {
            double om = 1.0 - c_pmv3_params.pi_joker[jj];
            g_ii += pmv3_g_share(n_s * q5 * om, c_pmv3_params.nb_r[1], smode);
            g_iv += pmv3_g_share(n_s * q4 * om, c_pmv3_params.nb_r[3], smode);
            g_vi += pmv3_g_share(n_s * q3 * om, c_pmv3_params.nb_r[5], smode);
        }
    }
    double y5 = (c_pmv3_params.funds[0] * inv_omega) * pmv3_g_share(n_s * q5 * pij, c_pmv3_params.nb_r[0], smode)
              + (c_pmv3_params.funds[1] * inv_omega) * g_ii;
    double y4 = (c_pmv3_params.funds[2] * inv_omega) * pmv3_g_share(n_s * q4 * pij, c_pmv3_params.nb_r[2], smode)
              + (c_pmv3_params.funds[3] * inv_omega) * g_iv;
    double y3 = (c_pmv3_params.funds[4] * inv_omega) * pmv3_g_share(n_s * q3 * pij, c_pmv3_params.nb_r[4], smode)
              + (c_pmv3_params.funds[5] * inv_omega) * g_vi;
    double y2 = (c_pmv3_params.funds[6] * inv_omega) * pmv3_g_share(n_s * q2 * pij, c_pmv3_params.nb_r[6], smode);
    double y1 = (c_pmv3_params.funds[7] * inv_omega) * pmv3_g_share(n_s * q1 * pij, c_pmv3_params.nb_r[7], smode);

    double yt1 = y1;
    double yt2_s = (y2 - 2.0 * y1) + off_2;
    double yt3_s = (y3 - 3.0 * y2 + 3.0 * y1) + off_3;
    double yt4_s = (y4 - 4.0 * y3 + 6.0 * y2 - 4.0 * y1) + off_4;
    if (yt2_s < 0.0) yt2_s = 0.0;
    if (yt3_s < 0.0) yt3_s = 0.0;
    if (yt4_s < 0.0) yt4_s = 0.0;
    d_yt5[idx] = y5 - 5.0 * y4 + 10.0 * y3 - 10.0 * y2 + 5.0 * y1;

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        pmv3_atomic_add_u128(d_u1_lo, d_u1_hi, (uint32_t)pos[i], yt1);
        pmv3_atomic_add_u128(d_u4_lo, d_u4_hi, pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U), yt4_s);
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int k = i + 1; k < 5; ++k) {
            uint64_t pm = bits[i] | bits[k];
            pmv3_atomic_add_u128(d_u2_lo, d_u2_hi, pmv3_rank_lex(pm, 45, 2, 990U), yt2_s);
            pmv3_atomic_add_u128(d_u3_lo, d_u3_hi, pmv3_rank_lex(mask ^ pm, 45, 3, 14190U), yt3_s);
        }
    }
}

__global__ void k_pmv3_joker_gather_single_j_f64(
    const uint64_t* __restrict__ d_base45,
    const double* __restrict__ d_yt5,
    const double* __restrict__ d_u4,
    const double* __restrict__ d_u3,
    const double* __restrict__ d_u2,
    const double* __restrict__ d_u1,
    double off_4,
    double off_3,
    double off_2,
    double* __restrict__ d_ev_j_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        s1 += d_u1[pos[i]];
        s4 += d_u4[pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U)];
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int k = i + 1; k < 5; ++k) {
            uint64_t pm = bits[i] | bits[k];
            s2 += d_u2[pmv3_rank_lex(pm, 45, 2, 990U)];
            s3 += d_u3[pmv3_rank_lex(mask ^ pm, 45, 3, 14190U)];
        }
    }
    d_ev_j_out[idx] = d_yt5[idx]
        + (s4 - 205.0 * off_4)
        + (s3 - 8610.0 * off_3)
        + (s2 - 123410.0 * off_2)
        + s1;
}

// ----------------------------------------------------------------------------
// 5.4 Kernele rapide u64-fixed-point + Quantile Ciura Shellsort pentru
//     Cubul Bootstrap 1000x pe RTX 5060 (f32 precizie < 1e-6, determinist 100%)
// ----------------------------------------------------------------------------

// Scaleaza probabilitatea p in [0, 1] cu 2^58 intr-un singur u64 (suma <= 2^58 < 2^64)
#define PMV3_SCALE_P58 288230376151711744.0
#define PMV3_INV_SCALE_P58 (1.0 / 288230376151711744.0)
// Scaleaza y_c(D) cu 2^36 intr-un singur u64 (rezolutie 1.45e-11, suma <= 2^61 < 2^64)
#define PMV3_SCALE_Y36 68719476736.0
#define PMV3_INV_SCALE_Y36 (1.0 / 68719476736.0)
// Scaleaza w ne-normalizat in (0, 1] cu 2^38 (suma peste 1.4e7 bilete <= 2^62 < 2^64)
#define PMV3_SCALE_W38 274877906944.0
#define PMV3_INV_SCALE_W38 (1.0 / 274877906944.0)

__global__ void k_boot_sum_z_u64(
    const uint64_t* __restrict__ d_masks,
    uint32_t n_tickets,
    int k_pick,
    float* __restrict__ d_w_f32,
    unsigned long long* __restrict__ d_z_u64
) {
    __shared__ unsigned long long s_z;
    if (threadIdx.x == 0) s_z = 0ULL;
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long vw = 0ULL;
    if (idx < n_tickets) {
        double lw = pmv3_ticket_lw(d_masks[idx], k_pick, c_pmv3_params);
        double w = exp(lw - c_pmv3_params.lw_shift);
        d_w_f32[idx] = (float)w;
        vw = (unsigned long long)(w * PMV3_SCALE_W38 + 0.5);
    }
    if (vw) atomicAdd(&s_z, vw);
    __syncthreads();
    if (threadIdx.x == 0 && s_z) atomicAdd(d_z_u64, s_z);
}

__global__ void k_boot_649_fwd_u64(
    const uint64_t* __restrict__ d_masks,
    float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_z_u64,
    unsigned long long* __restrict__ d_s5_u64,
    unsigned long long* __restrict__ d_s4_u64
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    double z_val = (double)(*d_z_u64) * PMV3_INV_SCALE_W38;
    double p = (double)d_p_f32[idx] / z_val;
    d_p_f32[idx] = (float)p;
    unsigned long long pu = (unsigned long long)(p * PMV3_SCALE_P58 + 0.5);
    if (pu == 0ULL) return;

    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        atomicAdd(&d_s5_u64[pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U)], pu);
    }
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            atomicAdd(&d_s4_u64[pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U)], pu);
        }
    }
}

__global__ void k_boot_649_bwd_u64(
    const uint64_t* __restrict__ d_masks,
    const float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_s5_u64,
    const unsigned long long* __restrict__ d_s4_u64,
    double off_5,
    uint32_t tile_start,
    uint32_t tile_len,
    float* __restrict__ d_yt6_tile,
    unsigned long long* __restrict__ d_u5_u64,
    unsigned long long* __restrict__ d_u4_u64
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    uint32_t r5[6];
    unsigned long long t5_u = 0ULL;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        r5[i] = pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U);
        t5_u += d_s5_u64[r5[i]];
    }
    uint32_t r4[15];
    unsigned long long t4_u = 0ULL;
    int pos = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            uint32_t rk = pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U);
            r4[pos++] = rk;
            t4_u += d_s4_u64[rk];
        }
    }
    double p_d = (double)d_p_f32[idx];
    double t5 = (double)t5_u * PMV3_INV_SCALE_P58;
    double t4 = (double)t4_u * PMV3_INV_SCALE_P58;
    double q6 = (p_d > 0.0) ? p_d : 0.0;
    double q5 = t5 - 6.0 * p_d;
    if (q5 < 0.0) q5 = 0.0;
    double q4 = t4 - 5.0 * t5 + 15.0 * p_d;
    if (q4 < 0.0) q4 = 0.0;

    const double inv_omega = 1.0 / 13983816.0;
    double n_s = c_pmv3_params.sales_n;
    int32_t smode = c_pmv3_params.share_mode;
    double y6 = (c_pmv3_params.funds[0] * inv_omega) * pmv3_g_share(n_s * q6, c_pmv3_params.nb_r[0], smode);
    double y5 = (c_pmv3_params.funds[1] * inv_omega) * pmv3_g_share(n_s * q5, c_pmv3_params.nb_r[1], smode);
    double y4 = (c_pmv3_params.funds[2] * inv_omega) * pmv3_g_share(n_s * q4, c_pmv3_params.nb_r[2], smode);

    double yt5_shift = (y5 - 5.0 * y4) + off_5;
    if (yt5_shift < 0.0) yt5_shift = 0.0;
    if (idx >= tile_start && idx < tile_start + tile_len) {
        d_yt6_tile[idx - tile_start] = (float)(y6 - 6.0 * y5 + 15.0 * y4);
    }
    unsigned long long u5_add = (unsigned long long)(yt5_shift * PMV3_SCALE_Y36 + 0.5);
    unsigned long long u4_add = (unsigned long long)(y4 * PMV3_SCALE_Y36 + 0.5);
    if (u5_add) {
        #pragma unroll
        for (int i = 0; i < 6; ++i) atomicAdd(&d_u5_u64[r5[i]], u5_add);
    }
    if (u4_add) {
        #pragma unroll
        for (int i = 0; i < 15; ++i) atomicAdd(&d_u4_u64[r4[i]], u4_add);
    }
}

__global__ void k_boot_649_gather_tile_f32(
    const uint64_t* __restrict__ d_masks,
    uint32_t tile_start,
    uint32_t tile_len,
    const float* __restrict__ d_yt6_tile,
    const unsigned long long* __restrict__ d_u5_u64,
    const unsigned long long* __restrict__ d_u4_u64,
    double off_5,
    double fixed_ev_iv,
    float* __restrict__ d_tile_row_out
) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tile_len) return;
    uint32_t idx = tile_start + t;
    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    unsigned long long sum_u5 = 0ULL;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        sum_u5 += d_u5_u64[pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U)];
    }
    unsigned long long sum_u4 = 0ULL;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            sum_u4 += d_u4_u64[pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U)];
        }
    }
    double ev = (double)d_yt6_tile[t]
        + ((double)sum_u5 * PMV3_INV_SCALE_Y36 - 264.0 * off_5)
        + ((double)sum_u4 * PMV3_INV_SCALE_Y36)
        + fixed_ev_iv;
    d_tile_row_out[t] = (float)ev;
}

// 5/40 bootstrap u64 kernels
__global__ void k_boot_540_fwd_u64(
    const uint64_t* __restrict__ d_tickets,
    float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_z_u64,
    unsigned long long* __restrict__ d_s4_u64
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 658008U) return;
    double z_val = (double)(*d_z_u64) * PMV3_INV_SCALE_W38;
    double p = (double)d_p_f32[idx] / z_val;
    d_p_f32[idx] = (float)p;
    unsigned long long pu = (unsigned long long)(p * PMV3_SCALE_P58 + 0.5);
    if (!pu) return;
    uint64_t mask = d_tickets[idx];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        m ^= lsb;
        atomicAdd(&d_s4_u64[pmv3_rank_lex(mask ^ lsb, 40, 4, 91390U)], pu);
    }
}

__global__ void k_boot_540_bwd_u64(
    const uint64_t* __restrict__ d_draws6,
    const float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_s4_u64,
    double off_540,
    unsigned long long* __restrict__ d_u5_u64,
    unsigned long long* __restrict__ d_u4_u64
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3838380U) return;
    uint64_t mask = d_draws6[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    uint32_t rk5[6];
    double pf[6];
    double t5 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint32_t rk = pmv3_rank_lex(mask ^ bits[i], 40, 5, 658008U);
        rk5[i] = rk;
        double pv = (double)d_p_f32[rk];
        pf[i] = pv;
        t5 += pv;
    }
    uint32_t rk4[15];
    unsigned long long t4_u = 0ULL;
    int pos = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            uint32_t r4 = pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 40, 4, 91390U);
            rk4[pos++] = r4;
            t4_u += d_s4_u64[r4];
        }
    }
    double t4 = (double)t4_u * PMV3_INV_SCALE_P58;
    double n_s = c_pmv3_params.sales_n;
    int32_t smode = c_pmv3_params.share_mode;
    double scale_ii = c_pmv3_params.funds[1] / (6.0 * 3838380.0);
    double pay_ii[6];
    double sum_pay_ii = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        double q_ii = t5 - pf[i];
        if (q_ii < 0.0) q_ii = 0.0;
        double py = scale_ii * pmv3_g_share(n_s * q_ii, c_pmv3_params.nb_r[1], smode);
        pay_ii[i] = py;
        sum_pay_ii += py;
    }
    double q_iii = t4 - 5.0 * t5;
    if (q_iii < 0.0) q_iii = 0.0;
    double y_iii = (c_pmv3_params.funds[2] / 3838380.0) * pmv3_g_share(n_s * q_iii, c_pmv3_params.nb_r[2], smode);

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        double val5 = (sum_pay_ii - pay_ii[i]) - 5.0 * y_iii + off_540;
        if (val5 < 0.0) val5 = 0.0;
        unsigned long long u5_add = (unsigned long long)(val5 * PMV3_SCALE_Y36 + 0.5);
        if (u5_add) atomicAdd(&d_u5_u64[rk5[i]], u5_add);
    }
    unsigned long long u4_add = (unsigned long long)(y_iii * PMV3_SCALE_Y36 + 0.5);
    if (u4_add) {
        #pragma unroll
        for (int i = 0; i < 15; ++i) atomicAdd(&d_u4_u64[rk4[i]], u4_add);
    }
}

__global__ void k_boot_540_gather_f32(
    const uint64_t* __restrict__ d_tickets,
    const float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_u5_u64,
    const unsigned long long* __restrict__ d_u4_u64,
    double off_540,
    float* __restrict__ d_row_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 658008U) return;
    uint64_t mask = d_tickets[idx];
    uint64_t m = mask;
    unsigned long long sum_u4 = 0ULL;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        m ^= lsb;
        sum_u4 += d_u4_u64[pmv3_rank_lex(mask ^ lsb, 40, 4, 91390U)];
    }
    double ev_i = (c_pmv3_params.funds[0] / 658008.0)
        * pmv3_g_share(c_pmv3_params.sales_n * (double)d_p_f32[idx], c_pmv3_params.nb_r[0], c_pmv3_params.share_mode);
    double ev = ev_i
        + ((double)d_u5_u64[idx] * PMV3_INV_SCALE_Y36 - 35.0 * off_540)
        + ((double)sum_u4 * PMV3_INV_SCALE_Y36);
    d_row_out[idx] = (float)ev;
}

// Joker bootstrap u64 kernels
__global__ void k_boot_joker_fwd_u64(
    const uint64_t* __restrict__ d_base45,
    float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_z_u64,
    unsigned long long* __restrict__ d_s4_u64,
    unsigned long long* __restrict__ d_s3_u64,
    unsigned long long* __restrict__ d_s2_u64,
    unsigned long long* __restrict__ d_s1_u64
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    double z_val = (double)(*d_z_u64) * PMV3_INV_SCALE_W38;
    double p = (double)d_p_f32[idx] / z_val;
    d_p_f32[idx] = (float)p;
    unsigned long long pu = (unsigned long long)(p * PMV3_SCALE_P58 + 0.5);
    if (!pu) return;

    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        atomicAdd(&d_s1_u64[pos[i]], pu);
        atomicAdd(&d_s4_u64[pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U)], pu);
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int k = i + 1; k < 5; ++k) {
            uint64_t pm = bits[i] | bits[k];
            atomicAdd(&d_s2_u64[pmv3_rank_lex(pm, 45, 2, 990U)], pu);
            atomicAdd(&d_s3_u64[pmv3_rank_lex(mask ^ pm, 45, 3, 14190U)], pu);
        }
    }
}

__global__ void k_boot_joker_bwd_single_j_u64(
    const uint64_t* __restrict__ d_base45,
    const float* __restrict__ d_p_f32,
    const unsigned long long* __restrict__ d_s4_u64,
    const unsigned long long* __restrict__ d_s3_u64,
    const unsigned long long* __restrict__ d_s2_u64,
    const unsigned long long* __restrict__ d_s1_u64,
    int32_t j_ticket,
    double off_4,
    double off_3,
    double off_2,
    float* __restrict__ d_yt5_f32,
    unsigned long long* __restrict__ d_u4_u64,
    unsigned long long* __restrict__ d_u3_u64,
    unsigned long long* __restrict__ d_u2_u64,
    unsigned long long* __restrict__ d_u1_u64
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    uint32_t r4[5];
    unsigned long long t1_u = 0ULL, t2_u = 0ULL, t3_u = 0ULL, t4_u = 0ULL;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        t1_u += d_s1_u64[pos[i]];
        uint32_t rk = pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U);
        r4[i] = rk;
        t4_u += d_s4_u64[rk];
    }
    uint32_t r2[10], r3[10];
    int p23 = 0;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int k = i + 1; k < 5; ++k) {
            uint64_t pm = bits[i] | bits[k];
            uint32_t rk2 = pmv3_rank_lex(pm, 45, 2, 990U);
            uint32_t rk3 = pmv3_rank_lex(mask ^ pm, 45, 3, 14190U);
            r2[p23] = rk2;
            r3[p23] = rk3;
            p23++;
            t2_u += d_s2_u64[rk2];
            t3_u += d_s3_u64[rk3];
        }
    }
    double t5 = (double)d_p_f32[idx];
    double t4 = (double)t4_u * PMV3_INV_SCALE_P58;
    double t3 = (double)t3_u * PMV3_INV_SCALE_P58;
    double t2 = (double)t2_u * PMV3_INV_SCALE_P58;
    double t1 = (double)t1_u * PMV3_INV_SCALE_P58;

    double q5 = (t5 > 0.0) ? t5 : 0.0;
    double q4 = t4 - 5.0 * t5; if (q4 < 0.0) q4 = 0.0;
    double q3 = t3 - 4.0 * t4 + 10.0 * t5; if (q3 < 0.0) q3 = 0.0;
    double q2 = t2 - 3.0 * t3 + 6.0 * t4 - 10.0 * t5; if (q2 < 0.0) q2 = 0.0;
    double q1 = t1 - 2.0 * t2 + 3.0 * t3 - 4.0 * t4 + 5.0 * t5; if (q1 < 0.0) q1 = 0.0;

    const double inv_omega = 1.0 / 24435180.0;
    double n_s = c_pmv3_params.sales_n;
    int32_t smode = c_pmv3_params.share_mode;
    double pij = c_pmv3_params.pi_joker[j_ticket];

    double g_ii = 0.0, g_iv = 0.0, g_vi = 0.0;
    #pragma unroll
    for (int jj = 0; jj < 20; ++jj) {
        if (jj != j_ticket) {
            double om = 1.0 - c_pmv3_params.pi_joker[jj];
            g_ii += pmv3_g_share(n_s * q5 * om, c_pmv3_params.nb_r[1], smode);
            g_iv += pmv3_g_share(n_s * q4 * om, c_pmv3_params.nb_r[3], smode);
            g_vi += pmv3_g_share(n_s * q3 * om, c_pmv3_params.nb_r[5], smode);
        }
    }
    double y5 = (c_pmv3_params.funds[0] * inv_omega) * pmv3_g_share(n_s * q5 * pij, c_pmv3_params.nb_r[0], smode)
              + (c_pmv3_params.funds[1] * inv_omega) * g_ii;
    double y4 = (c_pmv3_params.funds[2] * inv_omega) * pmv3_g_share(n_s * q4 * pij, c_pmv3_params.nb_r[2], smode)
              + (c_pmv3_params.funds[3] * inv_omega) * g_iv;
    double y3 = (c_pmv3_params.funds[4] * inv_omega) * pmv3_g_share(n_s * q3 * pij, c_pmv3_params.nb_r[4], smode)
              + (c_pmv3_params.funds[5] * inv_omega) * g_vi;
    double y2 = (c_pmv3_params.funds[6] * inv_omega) * pmv3_g_share(n_s * q2 * pij, c_pmv3_params.nb_r[6], smode);
    double y1 = (c_pmv3_params.funds[7] * inv_omega) * pmv3_g_share(n_s * q1 * pij, c_pmv3_params.nb_r[7], smode);

    double yt2_s = (y2 - 2.0 * y1) + off_2; if (yt2_s < 0.0) yt2_s = 0.0;
    double yt3_s = (y3 - 3.0 * y2 + 3.0 * y1) + off_3; if (yt3_s < 0.0) yt3_s = 0.0;
    double yt4_s = (y4 - 4.0 * y3 + 6.0 * y2 - 4.0 * y1) + off_4; if (yt4_s < 0.0) yt4_s = 0.0;
    d_yt5_f32[idx] = (float)(y5 - 5.0 * y4 + 10.0 * y3 - 10.0 * y2 + 5.0 * y1);

    unsigned long long u1_a = (unsigned long long)(y1 * PMV3_SCALE_Y36 + 0.5);
    unsigned long long u2_a = (unsigned long long)(yt2_s * PMV3_SCALE_Y36 + 0.5);
    unsigned long long u3_a = (unsigned long long)(yt3_s * PMV3_SCALE_Y36 + 0.5);
    unsigned long long u4_a = (unsigned long long)(yt4_s * PMV3_SCALE_Y36 + 0.5);

    if (u1_a) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) atomicAdd(&d_u1_u64[pos[i]], u1_a);
    }
    if (u4_a) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) atomicAdd(&d_u4_u64[r4[i]], u4_a);
    }
    if (u2_a || u3_a) {
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            if (u2_a) atomicAdd(&d_u2_u64[r2[i]], u2_a);
            if (u3_a) atomicAdd(&d_u3_u64[r3[i]], u3_a);
        }
    }
}

__global__ void k_boot_joker_gather_single_j_f32(
    const uint64_t* __restrict__ d_base45,
    const float* __restrict__ d_yt5_f32,
    const unsigned long long* __restrict__ d_u4_u64,
    const unsigned long long* __restrict__ d_u3_u64,
    const unsigned long long* __restrict__ d_u2_u64,
    const unsigned long long* __restrict__ d_u1_u64,
    double off_4,
    double off_3,
    double off_2,
    float* __restrict__ d_row_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    uint64_t mask = d_base45[idx];
    uint64_t bits[5];
    int pos[5];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int a = __ffsll((long long)m) - 1;
        uint64_t lsb = 1ULL << a;
        pos[i] = a;
        bits[i] = lsb;
        m ^= lsb;
    }
    unsigned long long s1 = 0ULL, s2 = 0ULL, s3 = 0ULL, s4 = 0ULL;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        s1 += d_u1_u64[pos[i]];
        s4 += d_u4_u64[pmv3_rank_lex(mask ^ bits[i], 45, 4, 148995U)];
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        for (int k = i + 1; k < 5; ++k) {
            uint64_t pm = bits[i] | bits[k];
            s2 += d_u2_u64[pmv3_rank_lex(pm, 45, 2, 990U)];
            s3 += d_u3_u64[pmv3_rank_lex(mask ^ pm, 45, 3, 14190U)];
        }
    }
    double ev = (double)d_yt5_f32[idx]
        + ((double)s4 * PMV3_INV_SCALE_Y36 - 205.0 * off_4)
        + ((double)s3 * PMV3_INV_SCALE_Y36 - 8610.0 * off_3)
        + ((double)s2 * PMV3_INV_SCALE_Y36 - 123410.0 * off_2)
        + ((double)s1 * PMV3_INV_SCALE_Y36);
    d_row_out[idx] = (float)ev;
}

// Extragere cuantile exacte [ev_nom, q05, q25, q50, q75, q95] din R=1000 replici bootstrap (Ciura Shellsort)
__global__ void k_extract_quantiles_1000_f32(
    const float* __restrict__ d_cube_1000, // [1000][tile_len] coalesced
    const double* __restrict__ d_ev_nom,   // [tile_len]
    uint32_t tile_len,
    uint32_t n_reps,
    float* __restrict__ d_out_rec6         // [tile_len * 6]
) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tile_len) return;
    float v[1000];
    for (uint32_t r = 0; r < n_reps; ++r) {
        v[r] = d_cube_1000[(size_t)r * (size_t)tile_len + (size_t)t];
    }
    const int gaps[8] = {701, 301, 132, 57, 23, 10, 4, 1};
    for (int gi = 0; gi < 8; ++gi) {
        int gap = gaps[gi];
        for (int i = gap; i < (int)n_reps; ++i) {
            float temp = v[i];
            int j = i;
            while (j >= gap && v[j - gap] > temp) {
                v[j] = v[j - gap];
                j -= gap;
            }
            v[j] = temp;
        }
    }
    // Interpolare liniara exacta numpy.quantile(..., [0.05, 0.25, 0.50, 0.75, 0.95], method='linear')
    const float probs[5] = {0.05f, 0.25f, 0.50f, 0.75f, 0.95f};
    float qs[5];
    float span = (float)(n_reps - 1U);
    #pragma unroll
    for (int qi = 0; qi < 5; ++qi) {
        float pos = probs[qi] * span;
        int lo = (int)pos;
        int hi = (lo + 1 < (int)n_reps) ? (lo + 1) : lo;
        float frac = pos - (float)lo;
        qs[qi] = v[lo] * (1.0f - frac) + v[hi] * frac;
    }
    // Asigurare monotonie IEEE-754 la limita de rotunjire f32
    for (int qi = 1; qi < 5; ++qi) {
        if (qs[qi] < qs[qi - 1]) qs[qi] = qs[qi - 1];
    }
    size_t base = (size_t)t * 6ULL;
    d_out_rec6[base + 0] = (float)d_ev_nom[t];
    d_out_rec6[base + 1] = qs[0];
    d_out_rec6[base + 2] = qs[1];
    d_out_rec6[base + 3] = qs[2];
    d_out_rec6[base + 4] = qs[3];
    d_out_rec6[base + 5] = qs[4];
}

extern "C" {

int32_t v3_cuda_pmv3_ev_649_f64(
    const uint64_t* d_masks_649,
    const V3CrowdParamsF64* h_params,
    double* h_ev_out,
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));

    const uint32_t N6 = 13983816U;
    const uint32_t N5 = 1906884U;
    const uint32_t N4 = 211876U;

    double *d_p = nullptr, *d_s5 = nullptr, *d_s4 = nullptr, *d_yt6 = nullptr, *d_u5 = nullptr, *d_u4 = nullptr, *d_ev = nullptr;
    unsigned long long *d_z_lo = nullptr, *d_z_hi = nullptr;
    unsigned long long *d_s5_lo = nullptr, *d_s5_hi = nullptr, *d_s4_lo = nullptr, *d_s4_hi = nullptr;
    unsigned long long *d_u5_lo = nullptr, *d_u5_hi = nullptr, *d_u4_lo = nullptr, *d_u4_hi = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_s5) cudaFree(d_s5);
        if (d_s4) cudaFree(d_s4);
        if (d_yt6) cudaFree(d_yt6);
        if (d_u5) cudaFree(d_u5);
        if (d_u4) cudaFree(d_u4);
        if (d_ev) cudaFree(d_ev);
        if (d_z_lo) cudaFree(d_z_lo);
        if (d_z_hi) cudaFree(d_z_hi);
        if (d_s5_lo) cudaFree(d_s5_lo);
        if (d_s5_hi) cudaFree(d_s5_hi);
        if (d_s4_lo) cudaFree(d_s4_lo);
        if (d_s4_hi) cudaFree(d_s4_hi);
        if (d_u5_lo) cudaFree(d_u5_lo);
        if (d_u5_hi) cudaFree(d_u5_hi);
        if (d_u4_lo) cudaFree(d_u4_lo);
        if (d_u4_hi) cudaFree(d_u4_hi);
    };

    if (cudaMalloc(&d_p, (size_t)N6 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s5, (size_t)N5 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_yt6, (size_t)N6 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u5, (size_t)N5 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_ev, (size_t)N6 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_z_lo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_z_hi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s5_lo, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s5_hi, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4_lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4_hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u5_lo, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u5_hi, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4_lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4_hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    cudaMemset(d_z_lo, 0, sizeof(unsigned long long));
    cudaMemset(d_z_hi, 0, sizeof(unsigned long long));
    cudaMemset(d_s5_lo, 0, (size_t)N5 * sizeof(unsigned long long));
    cudaMemset(d_s5_hi, 0, (size_t)N5 * sizeof(unsigned long long));
    cudaMemset(d_s4_lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s4_hi, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_u5_lo, 0, (size_t)N5 * sizeof(unsigned long long));
    cudaMemset(d_u5_hi, 0, (size_t)N5 * sizeof(unsigned long long));
    cudaMemset(d_u4_lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_u4_hi, 0, (size_t)N4 * sizeof(unsigned long long));

    uint32_t g6 = (N6 + 255U) / 256U;
    uint32_t g5 = (N5 + 255U) / 256U;
    uint32_t g4 = (N4 + 255U) / 256U;

    k_pmv3_sum_z_u128<<<g6, 256>>>(d_masks_649, N6, 6, d_p, d_z_lo, d_z_hi);
    unsigned long long hz_lo = 0ULL, hz_hi = 0ULL;
    cudaMemcpy(&hz_lo, d_z_lo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hz_hi, d_z_hi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    double z_val = ldexp((double)hz_hi, -32) + (ldexp((double)(hz_lo >> 32), -64) + ldexp((double)(hz_lo & 0xFFFFFFFFULL), -96));
    double inv_z = 1.0 / z_val;

    k_pmv3_649_fwd_u128<<<g6, 256>>>(d_masks_649, d_p, inv_z, d_s5_lo, d_s5_hi, d_s4_lo, d_s4_hi);
    k_pmv3_u128_to_f64<<<g5, 256>>>(d_s5_lo, d_s5_hi, d_s5, N5);
    k_pmv3_u128_to_f64<<<g4, 256>>>(d_s4_lo, d_s4_hi, d_s4, N4);

    double off_5 = 5.0 * (h_params->funds[2] / 13983816.0);
    double fixed_ev_iv = (246820.0 / 13983816.0) * h_params->fixed_iv;

    k_pmv3_649_bwd_u128<<<g6, 256>>>(d_masks_649, d_p, d_s5, d_s4, off_5, d_yt6, d_u5_lo, d_u5_hi, d_u4_lo, d_u4_hi);
    k_pmv3_u128_to_f64<<<g5, 256>>>(d_u5_lo, d_u5_hi, d_u5, N5);
    k_pmv3_u128_to_f64<<<g4, 256>>>(d_u4_lo, d_u4_hi, d_u4, N4);
    k_pmv3_649_gather_f64<<<g6, 256>>>(d_masks_649, d_yt6, d_u5, d_u4, off_5, fixed_ev_iv, d_ev);

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess && h_ev_out) {
        err = cudaMemcpy(h_ev_out, d_ev, (size_t)N6 * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_pmv3_ev_540_f64(
    const uint64_t* d_tickets_540,
    const uint64_t* d_draws6_540,
    const V3CrowdParamsF64* h_params,
    double* h_ev_out,
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));

    const uint32_t NT = 658008U;
    const uint32_t ND = 3838380U;
    const uint32_t N4 = 91390U;

    double *d_p = nullptr, *d_s4 = nullptr, *d_u5 = nullptr, *d_u4 = nullptr, *d_ev = nullptr;
    unsigned long long *d_z_lo = nullptr, *d_z_hi = nullptr;
    unsigned long long *d_s4_lo = nullptr, *d_s4_hi = nullptr;
    unsigned long long *d_u5_lo = nullptr, *d_u5_hi = nullptr, *d_u4_lo = nullptr, *d_u4_hi = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_s4) cudaFree(d_s4);
        if (d_u5) cudaFree(d_u5);
        if (d_u4) cudaFree(d_u4);
        if (d_ev) cudaFree(d_ev);
        if (d_z_lo) cudaFree(d_z_lo);
        if (d_z_hi) cudaFree(d_z_hi);
        if (d_s4_lo) cudaFree(d_s4_lo);
        if (d_s4_hi) cudaFree(d_s4_hi);
        if (d_u5_lo) cudaFree(d_u5_lo);
        if (d_u5_hi) cudaFree(d_u5_hi);
        if (d_u4_lo) cudaFree(d_u4_lo);
        if (d_u4_hi) cudaFree(d_u4_hi);
    };

    if (cudaMalloc(&d_p, (size_t)NT * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u5, (size_t)NT * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_ev, (size_t)NT * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_z_lo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_z_hi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4_lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4_hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u5_lo, (size_t)NT * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u5_hi, (size_t)NT * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4_lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4_hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    cudaMemset(d_z_lo, 0, sizeof(unsigned long long));
    cudaMemset(d_z_hi, 0, sizeof(unsigned long long));
    cudaMemset(d_s4_lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s4_hi, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_u5_lo, 0, (size_t)NT * sizeof(unsigned long long));
    cudaMemset(d_u5_hi, 0, (size_t)NT * sizeof(unsigned long long));
    cudaMemset(d_u4_lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_u4_hi, 0, (size_t)N4 * sizeof(unsigned long long));

    uint32_t gt = (NT + 255U) / 256U;
    uint32_t gd = (ND + 255U) / 256U;
    uint32_t g4 = (N4 + 255U) / 256U;

    k_pmv3_sum_z_u128<<<gt, 256>>>(d_tickets_540, NT, 5, d_p, d_z_lo, d_z_hi);
    unsigned long long hz_lo = 0ULL, hz_hi = 0ULL;
    cudaMemcpy(&hz_lo, d_z_lo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hz_hi, d_z_hi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    double z_val = ldexp((double)hz_hi, -32) + (ldexp((double)(hz_lo >> 32), -64) + ldexp((double)(hz_lo & 0xFFFFFFFFULL), -96));
    double inv_z = 1.0 / z_val;

    k_pmv3_540_fwd_u128<<<gt, 256>>>(d_tickets_540, d_p, inv_z, d_s4_lo, d_s4_hi);
    k_pmv3_u128_to_f64<<<g4, 256>>>(d_s4_lo, d_s4_hi, d_s4, N4);

    double off_540 = 5.0 * (h_params->funds[2] / 3838380.0);
    k_pmv3_540_bwd_u128<<<gd, 256>>>(d_draws6_540, d_p, d_s4, off_540, d_u5_lo, d_u5_hi, d_u4_lo, d_u4_hi);
    k_pmv3_u128_to_f64<<<gt, 256>>>(d_u5_lo, d_u5_hi, d_u5, NT);
    k_pmv3_u128_to_f64<<<g4, 256>>>(d_u4_lo, d_u4_hi, d_u4, N4);
    k_pmv3_540_gather_f64<<<gt, 256>>>(d_tickets_540, d_p, d_u5, d_u4, off_540, d_ev);

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess && h_ev_out) {
        err = cudaMemcpy(h_ev_out, d_ev, (size_t)NT * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_pmv3_ev_joker_f64(
    const uint64_t* d_base_joker,
    const V3CrowdParamsF64* h_params,
    double* h_ev_24m_out,
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));

    const uint32_t NB = 1221759U;
    const uint32_t N4 = 148995U;
    const uint32_t N3 = 14190U;
    const uint32_t N2 = 990U;
    const uint32_t N1 = 45U;

    double *d_p = nullptr, *d_s4 = nullptr, *d_s3 = nullptr, *d_s2 = nullptr, *d_s1 = nullptr;
    double *d_q4 = nullptr, *d_q3 = nullptr, *d_q2 = nullptr, *d_q1 = nullptr;
    double *d_yt5 = nullptr, *d_u4 = nullptr, *d_u3 = nullptr, *d_u2 = nullptr, *d_u1 = nullptr, *d_ev_j = nullptr;
    unsigned long long *d_z_lo = nullptr, *d_z_hi = nullptr;
    unsigned long long *d_s4_lo = nullptr, *d_s4_hi = nullptr, *d_s3_lo = nullptr, *d_s3_hi = nullptr;
    unsigned long long *d_s2_lo = nullptr, *d_s2_hi = nullptr, *d_s1_lo = nullptr, *d_s1_hi = nullptr;
    unsigned long long *d_u4_lo = nullptr, *d_u4_hi = nullptr, *d_u3_lo = nullptr, *d_u3_hi = nullptr;
    unsigned long long *d_u2_lo = nullptr, *d_u2_hi = nullptr, *d_u1_lo = nullptr, *d_u1_hi = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_s4) cudaFree(d_s4); if (d_s3) cudaFree(d_s3); if (d_s2) cudaFree(d_s2); if (d_s1) cudaFree(d_s1);
        if (d_q4) cudaFree(d_q4); if (d_q3) cudaFree(d_q3); if (d_q2) cudaFree(d_q2); if (d_q1) cudaFree(d_q1);
        if (d_yt5) cudaFree(d_yt5); if (d_u4) cudaFree(d_u4); if (d_u3) cudaFree(d_u3); if (d_u2) cudaFree(d_u2); if (d_u1) cudaFree(d_u1);
        if (d_ev_j) cudaFree(d_ev_j);
        if (d_z_lo) cudaFree(d_z_lo); if (d_z_hi) cudaFree(d_z_hi);
        if (d_s4_lo) cudaFree(d_s4_lo); if (d_s4_hi) cudaFree(d_s4_hi);
        if (d_s3_lo) cudaFree(d_s3_lo); if (d_s3_hi) cudaFree(d_s3_hi);
        if (d_s2_lo) cudaFree(d_s2_lo); if (d_s2_hi) cudaFree(d_s2_hi);
        if (d_s1_lo) cudaFree(d_s1_lo); if (d_s1_hi) cudaFree(d_s1_hi);
        if (d_u4_lo) cudaFree(d_u4_lo); if (d_u4_hi) cudaFree(d_u4_hi);
        if (d_u3_lo) cudaFree(d_u3_lo); if (d_u3_hi) cudaFree(d_u3_hi);
        if (d_u2_lo) cudaFree(d_u2_lo); if (d_u2_hi) cudaFree(d_u2_hi);
        if (d_u1_lo) cudaFree(d_u1_lo); if (d_u1_hi) cudaFree(d_u1_hi);
    };

    if (cudaMalloc(&d_p, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s3, (size_t)N3 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s2, (size_t)N2 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s1, (size_t)N1 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_q4, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_q3, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_q2, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_q1, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_yt5, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u3, (size_t)N3 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u2, (size_t)N2 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_u1, (size_t)N1 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_ev_j, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_z_lo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_z_hi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4_lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4_hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s3_lo, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s3_hi, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s2_lo, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s2_hi, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s1_lo, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s1_hi, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4_lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4_hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u3_lo, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u3_hi, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u2_lo, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u2_hi, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u1_lo, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u1_hi, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    cudaMemset(d_z_lo, 0, sizeof(unsigned long long));
    cudaMemset(d_z_hi, 0, sizeof(unsigned long long));
    cudaMemset(d_s4_lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s4_hi, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s3_lo, 0, (size_t)N3 * sizeof(unsigned long long));
    cudaMemset(d_s3_hi, 0, (size_t)N3 * sizeof(unsigned long long));
    cudaMemset(d_s2_lo, 0, (size_t)N2 * sizeof(unsigned long long));
    cudaMemset(d_s2_hi, 0, (size_t)N2 * sizeof(unsigned long long));
    cudaMemset(d_s1_lo, 0, (size_t)N1 * sizeof(unsigned long long));
    cudaMemset(d_s1_hi, 0, (size_t)N1 * sizeof(unsigned long long));

    uint32_t gb = (NB + 255U) / 256U;
    uint32_t g4 = (N4 + 255U) / 256U;
    uint32_t g3 = (N3 + 255U) / 256U;
    uint32_t g2 = (N2 + 255U) / 256U;

    k_pmv3_sum_z_u128<<<gb, 256>>>(d_base_joker, NB, 5, d_p, d_z_lo, d_z_hi);
    unsigned long long hz_lo = 0ULL, hz_hi = 0ULL;
    cudaMemcpy(&hz_lo, d_z_lo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hz_hi, d_z_hi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    double z_val = ldexp((double)hz_hi, -32) + (ldexp((double)(hz_lo >> 32), -64) + ldexp((double)(hz_lo & 0xFFFFFFFFULL), -96));
    double inv_z = 1.0 / z_val;

    k_pmv3_joker_fwd_u128<<<gb, 256>>>(d_base_joker, d_p, inv_z, d_s4_lo, d_s4_hi, d_s3_lo, d_s3_hi, d_s2_lo, d_s2_hi, d_s1_lo, d_s1_hi);
    k_pmv3_u128_to_f64<<<g4, 256>>>(d_s4_lo, d_s4_hi, d_s4, N4);
    k_pmv3_u128_to_f64<<<g3, 256>>>(d_s3_lo, d_s3_hi, d_s3, N3);
    k_pmv3_u128_to_f64<<<g2, 256>>>(d_s2_lo, d_s2_hi, d_s2, N2);
    k_pmv3_u128_to_f64<<<1, 64>>>(d_s1_lo, d_s1_hi, d_s1, N1);
    k_pmv3_joker_compute_q_f64<<<gb, 256>>>(d_base_joker, d_p, d_s4, d_s3, d_s2, d_s1, d_q4, d_q3, d_q2, d_q1);

    const double inv_omega = 1.0 / 24435180.0;
    double b1 = h_params->funds[7] * inv_omega;
    double b2 = h_params->funds[6] * inv_omega;
    double b3 = (h_params->funds[4] + 19.0 * h_params->funds[5]) * inv_omega;
    double off_2 = 2.0 * b1;
    double off_3 = 3.0 * b2;
    double off_4 = 4.0 * b3 + 4.0 * b1;

    cudaError_t err = cudaSuccess;
    for (int32_t j = 0; j < 20; ++j) {
        cudaMemset(d_u4_lo, 0, (size_t)N4 * sizeof(unsigned long long));
        cudaMemset(d_u4_hi, 0, (size_t)N4 * sizeof(unsigned long long));
        cudaMemset(d_u3_lo, 0, (size_t)N3 * sizeof(unsigned long long));
        cudaMemset(d_u3_hi, 0, (size_t)N3 * sizeof(unsigned long long));
        cudaMemset(d_u2_lo, 0, (size_t)N2 * sizeof(unsigned long long));
        cudaMemset(d_u2_hi, 0, (size_t)N2 * sizeof(unsigned long long));
        cudaMemset(d_u1_lo, 0, (size_t)N1 * sizeof(unsigned long long));
        cudaMemset(d_u1_hi, 0, (size_t)N1 * sizeof(unsigned long long));

        k_pmv3_joker_bwd_single_j_u128<<<gb, 256>>>(
            d_base_joker, d_p, d_q4, d_q3, d_q2, d_q1, j, off_4, off_3, off_2,
            d_yt5, d_u4_lo, d_u4_hi, d_u3_lo, d_u3_hi, d_u2_lo, d_u2_hi, d_u1_lo, d_u1_hi
        );
        k_pmv3_u128_to_f64<<<g4, 256>>>(d_u4_lo, d_u4_hi, d_u4, N4);
        k_pmv3_u128_to_f64<<<g3, 256>>>(d_u3_lo, d_u3_hi, d_u3, N3);
        k_pmv3_u128_to_f64<<<g2, 256>>>(d_u2_lo, d_u2_hi, d_u2, N2);
        k_pmv3_u128_to_f64<<<1, 64>>>(d_u1_lo, d_u1_hi, d_u1, N1);
        k_pmv3_joker_gather_single_j_f64<<<gb, 256>>>(
            d_base_joker, d_yt5, d_u4, d_u3, d_u2, d_u1, off_4, off_3, off_2, d_ev_j
        );
        if (h_ev_24m_out) {
            err = cudaMemcpy(
                h_ev_24m_out + (size_t)j * (size_t)NB,
                d_ev_j,
                (size_t)NB * sizeof(double),
                cudaMemcpyDeviceToHost
            );
            if (err != cudaSuccess) break;
        }
    }

    cudaEventRecord(ev1);
    cudaError_t sync_err = cudaEventSynchronize(ev1);
    if (err == cudaSuccess) err = sync_err;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    cleanup();
    return (int32_t)err;
}

// ----------------------------------------------------------------------------
// Entry points C pentru Cubul Bootstrap 1000x (streaming pe tile-uri in VRAM)
// ----------------------------------------------------------------------------

int32_t v3_cuda_pmv3_boot_649_tile(
    const uint64_t* d_masks_649,
    const V3CrowdParamsF64* h_rep_params, // [n_reps]
    uint32_t n_reps,
    uint32_t tile_start,
    uint32_t tile_len,
    const double* h_ev_nom_tile,          // [tile_len]
    float* h_out_rec6,                    // [tile_len * 6]
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    if (n_reps == 0 || n_reps > 1000 || tile_len == 0) return -2;

    const uint32_t N6 = 13983816U;
    const uint32_t N5 = 1906884U;
    const uint32_t N4 = 211876U;

    float *d_p = nullptr, *d_yt6_tile = nullptr, *d_cube = nullptr, *d_out6 = nullptr;
    double *d_ev_nom = nullptr;
    unsigned long long *d_z = nullptr, *d_s5 = nullptr, *d_s4 = nullptr, *d_u5 = nullptr, *d_u4 = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_yt6_tile) cudaFree(d_yt6_tile);
        if (d_cube) cudaFree(d_cube);
        if (d_out6) cudaFree(d_out6);
        if (d_ev_nom) cudaFree(d_ev_nom);
        if (d_z) cudaFree(d_z);
        if (d_s5) cudaFree(d_s5);
        if (d_s4) cudaFree(d_s4);
        if (d_u5) cudaFree(d_u5);
        if (d_u4) cudaFree(d_u4);
    };

    if (cudaMalloc(&d_p, (size_t)N6 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_yt6_tile, (size_t)tile_len * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_cube, (size_t)n_reps * (size_t)tile_len * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out6, (size_t)tile_len * 6ULL * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_ev_nom, (size_t)tile_len * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_z, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s5, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u5, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaMemcpy(d_ev_nom, h_ev_nom_tile, (size_t)tile_len * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t g6 = (N6 + 255U) / 256U;
    uint32_t gt = (tile_len + 255U) / 256U;

    for (uint32_t r = 0; r < n_reps; ++r) {
        const V3CrowdParamsF64& pr = h_rep_params[r];
        cudaMemcpyToSymbol(c_pmv3_params, &pr, sizeof(V3CrowdParamsF64));
        cudaMemset(d_z, 0, sizeof(unsigned long long));
        cudaMemset(d_s5, 0, (size_t)N5 * sizeof(unsigned long long));
        cudaMemset(d_s4, 0, (size_t)N4 * sizeof(unsigned long long));
        cudaMemset(d_u5, 0, (size_t)N5 * sizeof(unsigned long long));
        cudaMemset(d_u4, 0, (size_t)N4 * sizeof(unsigned long long));

        double off_5 = 5.0 * (pr.funds[2] / 13983816.0);
        double fixed_ev_iv = (246820.0 / 13983816.0) * pr.fixed_iv;

        k_boot_sum_z_u64<<<g6, 256>>>(d_masks_649, N6, 6, d_p, d_z);
        k_boot_649_fwd_u64<<<g6, 256>>>(d_masks_649, d_p, d_z, d_s5, d_s4);
        k_boot_649_bwd_u64<<<g6, 256>>>(d_masks_649, d_p, d_s5, d_s4, off_5, tile_start, tile_len, d_yt6_tile, d_u5, d_u4);
        k_boot_649_gather_tile_f32<<<gt, 256>>>(
            d_masks_649, tile_start, tile_len, d_yt6_tile, d_u5, d_u4, off_5, fixed_ev_iv,
            d_cube + (size_t)r * (size_t)tile_len
        );
    }

    uint32_t gq = (tile_len + 127U) / 128U;
    k_extract_quantiles_1000_f32<<<gq, 128>>>(d_cube, d_ev_nom, tile_len, n_reps, d_out6);

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess && h_out_rec6) {
        err = cudaMemcpy(h_out_rec6, d_out6, (size_t)tile_len * 6ULL * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_pmv3_boot_540_all(
    const uint64_t* d_tickets_540,
    const uint64_t* d_draws6_540,
    const V3CrowdParamsF64* h_rep_params, // [n_reps]
    uint32_t n_reps,
    const double* h_ev_nom,               // [658008]
    float* h_out_rec6,                    // [658008 * 6]
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    if (n_reps == 0 || n_reps > 1000) return -2;

    const uint32_t NT = 658008U;
    const uint32_t ND = 3838380U;
    const uint32_t N4 = 91390U;

    float *d_p = nullptr, *d_cube = nullptr, *d_out6 = nullptr;
    double *d_ev_nom = nullptr;
    unsigned long long *d_z = nullptr, *d_s4 = nullptr, *d_u5 = nullptr, *d_u4 = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_cube) cudaFree(d_cube);
        if (d_out6) cudaFree(d_out6);
        if (d_ev_nom) cudaFree(d_ev_nom);
        if (d_z) cudaFree(d_z);
        if (d_s4) cudaFree(d_s4);
        if (d_u5) cudaFree(d_u5);
        if (d_u4) cudaFree(d_u4);
    };

    if (cudaMalloc(&d_p, (size_t)NT * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_cube, (size_t)n_reps * (size_t)NT * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out6, (size_t)NT * 6ULL * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_ev_nom, (size_t)NT * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_z, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u5, (size_t)NT * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaMemcpy(d_ev_nom, h_ev_nom, (size_t)NT * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t gt = (NT + 255U) / 256U;
    uint32_t gd = (ND + 255U) / 256U;

    for (uint32_t r = 0; r < n_reps; ++r) {
        const V3CrowdParamsF64& pr = h_rep_params[r];
        cudaMemcpyToSymbol(c_pmv3_params, &pr, sizeof(V3CrowdParamsF64));
        cudaMemset(d_z, 0, sizeof(unsigned long long));
        cudaMemset(d_s4, 0, (size_t)N4 * sizeof(unsigned long long));
        cudaMemset(d_u5, 0, (size_t)NT * sizeof(unsigned long long));
        cudaMemset(d_u4, 0, (size_t)N4 * sizeof(unsigned long long));

        double off_540 = 5.0 * (pr.funds[2] / 3838380.0);
        k_boot_sum_z_u64<<<gt, 256>>>(d_tickets_540, NT, 5, d_p, d_z);
        k_boot_540_fwd_u64<<<gt, 256>>>(d_tickets_540, d_p, d_z, d_s4);
        k_boot_540_bwd_u64<<<gd, 256>>>(d_draws6_540, d_p, d_s4, off_540, d_u5, d_u4);
        k_boot_540_gather_f32<<<gt, 256>>>(d_tickets_540, d_p, d_u5, d_u4, off_540, d_cube + (size_t)r * (size_t)NT);
    }

    uint32_t gq = (NT + 127U) / 128U;
    k_extract_quantiles_1000_f32<<<gq, 128>>>(d_cube, d_ev_nom, NT, n_reps, d_out6);

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess && h_out_rec6) {
        err = cudaMemcpy(h_out_rec6, d_out6, (size_t)NT * 6ULL * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_pmv3_boot_joker_single_j(
    const uint64_t* d_base_joker,
    const V3CrowdParamsF64* h_rep_params, // [n_reps]
    uint32_t n_reps,
    int32_t j_ticket,                     // 0..19
    const double* h_ev_nom_j,             // [1221759]
    float* h_out_rec6,                    // [1221759 * 6]
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    if (n_reps == 0 || n_reps > 1000 || j_ticket < 0 || j_ticket >= 20) return -2;

    const uint32_t NB = 1221759U;
    const uint32_t N4 = 148995U;
    const uint32_t N3 = 14190U;
    const uint32_t N2 = 990U;
    const uint32_t N1 = 45U;

    float *d_p = nullptr, *d_yt5 = nullptr, *d_cube = nullptr, *d_out6 = nullptr;
    double *d_ev_nom = nullptr;
    unsigned long long *d_z = nullptr, *d_s4 = nullptr, *d_s3 = nullptr, *d_s2 = nullptr, *d_s1 = nullptr;
    unsigned long long *d_u4 = nullptr, *d_u3 = nullptr, *d_u2 = nullptr, *d_u1 = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_yt5) cudaFree(d_yt5);
        if (d_cube) cudaFree(d_cube);
        if (d_out6) cudaFree(d_out6);
        if (d_ev_nom) cudaFree(d_ev_nom);
        if (d_z) cudaFree(d_z);
        if (d_s4) cudaFree(d_s4); if (d_s3) cudaFree(d_s3); if (d_s2) cudaFree(d_s2); if (d_s1) cudaFree(d_s1);
        if (d_u4) cudaFree(d_u4); if (d_u3) cudaFree(d_u3); if (d_u2) cudaFree(d_u2); if (d_u1) cudaFree(d_u1);
    };

    if (cudaMalloc(&d_p, (size_t)NB * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_yt5, (size_t)NB * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_cube, (size_t)n_reps * (size_t)NB * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out6, (size_t)NB * 6ULL * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_ev_nom, (size_t)NB * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_z, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s3, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s2, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s1, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u4, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u3, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u2, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_u1, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaMemcpy(d_ev_nom, h_ev_nom_j, (size_t)NB * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t gb = (NB + 255U) / 256U;
    const double inv_omega = 1.0 / 24435180.0;

    for (uint32_t r = 0; r < n_reps; ++r) {
        const V3CrowdParamsF64& pr = h_rep_params[r];
        cudaMemcpyToSymbol(c_pmv3_params, &pr, sizeof(V3CrowdParamsF64));
        cudaMemset(d_z, 0, sizeof(unsigned long long));
        cudaMemset(d_s4, 0, (size_t)N4 * sizeof(unsigned long long));
        cudaMemset(d_s3, 0, (size_t)N3 * sizeof(unsigned long long));
        cudaMemset(d_s2, 0, (size_t)N2 * sizeof(unsigned long long));
        cudaMemset(d_s1, 0, (size_t)N1 * sizeof(unsigned long long));
        cudaMemset(d_u4, 0, (size_t)N4 * sizeof(unsigned long long));
        cudaMemset(d_u3, 0, (size_t)N3 * sizeof(unsigned long long));
        cudaMemset(d_u2, 0, (size_t)N2 * sizeof(unsigned long long));
        cudaMemset(d_u1, 0, (size_t)N1 * sizeof(unsigned long long));

        double b1 = pr.funds[7] * inv_omega;
        double b2 = pr.funds[6] * inv_omega;
        double b3 = (pr.funds[4] + 19.0 * pr.funds[5]) * inv_omega;
        double off_2 = 2.0 * b1;
        double off_3 = 3.0 * b2;
        double off_4 = 4.0 * b3 + 4.0 * b1;

        k_boot_sum_z_u64<<<gb, 256>>>(d_base_joker, NB, 5, d_p, d_z);
        k_boot_joker_fwd_u64<<<gb, 256>>>(d_base_joker, d_p, d_z, d_s4, d_s3, d_s2, d_s1);
        k_boot_joker_bwd_single_j_u64<<<gb, 256>>>(
            d_base_joker, d_p, d_s4, d_s3, d_s2, d_s1, j_ticket, off_4, off_3, off_2,
            d_yt5, d_u4, d_u3, d_u2, d_u1
        );
        k_boot_joker_gather_single_j_f32<<<gb, 256>>>(
            d_base_joker, d_yt5, d_u4, d_u3, d_u2, d_u1, off_4, off_3, off_2,
            d_cube + (size_t)r * (size_t)NB
        );
    }

    uint32_t gq = (NB + 127U) / 128U;
    k_extract_quantiles_1000_f32<<<gq, 128>>>(d_cube, d_ev_nom, NB, n_reps, d_out6);

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess && h_out_rec6) {
        err = cudaMemcpy(h_out_rec6, d_out6, (size_t)NB * 6ULL * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cleanup();
    return (int32_t)err;
}

} // extern "C"

// ============================================================================
// 6. Sesiunea 3 (S3 / Poarta A4) — Optimizator Exact pe Tot Universul C3-v2
//    (39,077,004 bilete rezidente in VRAM, 0 eroare Bonferroni, auto-partajare
//    NB2 exacta) & Simulator CUDA C4-v2 cu 1,000,000 Traiectorii Multi-An
// ============================================================================

__constant__ double c_gl32_x[32];
__constant__ double c_gl32_w[32];
__constant__ uint64_t c_c3v2_tickets[512];
__constant__ double c_c3v2_penalties[16];
__constant__ double c_c4v2_xi16[16];

static bool g_c3v2_gl32_ready = false;

static int32_t ensure_c3v2_gl32_init() {
    if (g_c3v2_gl32_ready) return 0;
    const int N = 32;
    double hx[32];
    double hw[32];
    const double pi_v = 3.14159265358979323846264338327950288;
    for (int i = 0; i < N / 2; ++i) {
        double z = cos(pi_v * ((double)i + 0.75) / ((double)N + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 12; ++it) {
            double p1 = 1.0;
            double p2 = 0.0;
            for (int j = 1; j <= N; ++j) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * (double)j - 1.0) * z * p2 - ((double)j - 1.0) * p3) / (double)j;
            }
            dp = (double)N * (z * p1 - p2) / (z * z - 1.0);
            z -= p1 / dp;
        }
        double w = 2.0 / ((1.0 - z * z) * dp * dp);
        // Transformam [-1, 1] -> [0, 1]
        hx[i] = 0.5 * (1.0 - z);
        hx[N - 1 - i] = 0.5 * (1.0 + z);
        hw[i] = 0.5 * w;
        hw[N - 1 - i] = 0.5 * w;
    }
    cudaError_t e1 = cudaMemcpyToSymbol(c_gl32_x, hx, sizeof(hx));
    if (e1 != cudaSuccess) return (int32_t)e1;
    cudaError_t e2 = cudaMemcpyToSymbol(c_gl32_w, hw, sizeof(hw));
    if (e2 != cudaSuccess) return (int32_t)e2;
    g_c3v2_gl32_ready = true;
    return 0;
}

// E[m / (m + W)] exact pentru W ~ NB2(mu, r), m >= 0 (cu cuadratura Gauss-Legendre 32p rational-scalata)
__device__ __forceinline__ double c3v2_g_share_multi(int m, double mu, double r, int32_t mode) {
    if (m <= 0) return 0.0;
    if (m == 1) return pmv3_g_share(mu, r, mode);
    if (mu <= 0.0) return 1.0;
    double dm = (double)m;
    if (mode == 2) return dm / (dm + mu);
    double eff_r = (mode == 1 || r >= 1.0e12) ? 1.0e12 : r;
    double A = dm + mu;
    double logA = log(A);
    double acc = 0.0;
    #pragma unroll
    for (int q = 0; q < 32; ++q) {
        double wq = c_gl32_x[q];
        double omw = 1.0 - wq;
        double dw = omw * A + wq;
        double lp = (dm - 1.0) * log(omw) + dm * logA - (dm + 1.0) * log(dw);
        if (mode == 1 || r >= 1.0e12) {
            lp -= (mu * wq) / dw;
        } else {
            lp -= eff_r * log1p((mu * wq) / (eff_r * dw));
        }
        acc += c_gl32_w[q] * exp(lp);
    }
    return dm * acc;
}

// Momentul de ordin 2 exact E[(m / (m + W))^2] pentru W ~ NB2(mu, r), m >= 1
__device__ __forceinline__ double c3v2_h2_share_multi(int m, double mu, double r, int32_t mode) {
    if (m <= 0) return 0.0;
    if (mu <= 0.0) return 1.0;
    double dm = (double)m;
    if (mode == 2) {
        double g = dm / (dm + mu);
        return g * g;
    }
    double eff_r = (mode == 1 || r >= 1.0e12) ? 1.0e12 : r;
    double A = dm + mu;
    double logA = log(A);
    double acc = 0.0;
    #pragma unroll
    for (int q = 0; q < 32; ++q) {
        double tq = c_gl32_x[q];
        double omw = tq * tq * tq;
        double wq = 1.0 - omw;
        double jac = 3.0 * tq * tq;
        double dw = omw * A + wq;
        double neg_log_1ms = log1p(wq / (omw * A));
        double lp = (dm - 1.0) * log(omw) + dm * logA - (dm + 1.0) * log(dw);
        if (mode == 1 || r >= 1.0e12) {
            lp -= (mu * wq) / dw;
        } else {
            lp -= eff_r * log1p((mu * wq) / (eff_r * dw));
        }
        acc += c_gl32_w[q] * jac * neg_log_1ms * exp(lp);
    }
    return dm * dm * acc;
}

__device__ __forceinline__ double c3v2_nb_pzero(double mu, double r, int32_t mode) {
    if (mu <= 0.0) return 1.0;
    if (mode == 1 || r >= 1.0e12) return exp(-mu);
    return exp(-r * log1p(mu / r));
}

// Predicat canonic is_patterned_ticket direct pe masca de biti u64 (identic 100% cu Rust/Python)
__device__ __forceinline__ bool c3v2_is_patterned_mask(uint64_t mask, int k, int gcols) {
    int pos[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (i >= k) break;
        int a = __ffsll((long long)m) - 1;
        m &= (m - 1ULL);
        pos[i] = a;
    }
    int d0 = pos[1] - pos[0];
    bool all_eq = (d0 > 0);
    for (int i = 1; i < k - 1; ++i) {
        if (pos[i + 1] - pos[i] != d0) { all_eq = false; break; }
    }
    if (all_eq) return true;
    for (int i = 0; i <= k - 4; ++i) {
        if (pos[i + 3] - pos[i] == 3) return true;
    }
    int r0 = pos[0] / gcols;
    int c0 = pos[0] % gcols;
    bool same_r = true, same_c = true;
    for (int i = 1; i < k; ++i) {
        if (pos[i] / gcols != r0) same_r = false;
        if (pos[i] % gcols != c0) same_c = false;
    }
    return same_r || same_c;
}

// ----------------------------------------------------------------------------
// 6.1 Precalcul tabele exacte f64 de probabilitati ale multimii pe stari (VRAM)
// ----------------------------------------------------------------------------

__global__ void k_c3v2_649_extract_q(
    const uint64_t* __restrict__ d_masks,
    const double* __restrict__ d_p,
    const double* __restrict__ d_s5,
    const double* __restrict__ d_s4,
    double* __restrict__ d_q6,
    double* __restrict__ d_q5,
    double* __restrict__ d_q4
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    uint64_t mask = d_masks[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    double t5 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        t5 += d_s5[pmv3_rank_lex(mask ^ bits[i], 49, 5, 1906884U)];
    }
    double t4 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            t4 += d_s4[pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 49, 4, 211876U)];
        }
    }
    double p_d = d_p[idx];
    double q6 = (p_d > 0.0) ? p_d : 0.0;
    double q5 = t5 - 6.0 * p_d;
    if (q5 < 0.0) q5 = 0.0;
    double q4 = t4 - 5.0 * t5 + 15.0 * p_d;
    if (q4 < 0.0) q4 = 0.0;
    d_q6[idx] = q6;
    d_q5[idx] = q5;
    d_q4[idx] = q4;
}

__global__ void k_c3v2_540_extract_q(
    const uint64_t* __restrict__ d_draws6,
    const double* __restrict__ d_p,
    const double* __restrict__ d_s4,
    double* __restrict__ d_t5_out,
    double* __restrict__ d_qiii_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3838380U) return;
    uint64_t mask = d_draws6[idx];
    uint64_t bits[6];
    uint64_t m = mask;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = m & (~m + 1ULL);
        bits[i] = lsb;
        m ^= lsb;
    }
    double t5 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint32_t rk = pmv3_rank_lex(mask ^ bits[i], 40, 5, 658008U);
        t5 += d_p[rk];
    }
    double t4 = 0.0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            uint32_t r4 = pmv3_rank_lex(mask ^ bits[i] ^ bits[j], 40, 4, 91390U);
            t4 += d_s4[r4];
        }
    }
    double q_iii = t4 - 5.0 * t5;
    if (q_iii < 0.0) q_iii = 0.0;
    d_t5_out[idx] = t5;
    d_qiii_out[idx] = q_iii;
}

// ----------------------------------------------------------------------------
// 6.2 Evaluatoare Exacte de Portofoliu pe Tot Universul (CUDA sm_120 u128_s96)
// ----------------------------------------------------------------------------

struct V3PortfolioEvalOut {
    uint64_t win_states;
    uint64_t pair_overlap_sum;
    uint64_t cat_any_states[8];
    double ev_portfolio;
    double ev_by_cat[8];
    double second_moment_draw0;
    double elapsed_ms;
};

__global__ void k_c3v2_eval_649(
    const uint64_t* __restrict__ d_masks,
    const double* __restrict__ d_q6,
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    uint32_t n_tickets,
    unsigned long long* __restrict__ d_cnts, // [0]=win, [1]=pair_ov, [2..6]=cat_any
    unsigned long long* __restrict__ d_ev_lo, // [0..4]=cat 0..3, [4]=total
    unsigned long long* __restrict__ d_ev_hi,
    double* __restrict__ d_m2_block
) {
    __shared__ unsigned long long s_cnt[6];
    __shared__ double s_m2[256];
    if (threadIdx.x < 6) s_cnt[threadIdx.x] = 0ULL;
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double m2_contrib = 0.0;

    if (idx < 13983816U) {
        uint64_t dm = d_masks[idx];
        int m_i = 0, m_ii = 0, m_iii = 0, m_iv = 0;
        for (uint32_t t = 0; t < n_tickets; ++t) {
            int h = __popcll(dm & c_c3v2_tickets[t]);
            if (h >= 3) {
                if (h == 6) ++m_i;
                else if (h == 5) ++m_ii;
                else if (h == 4) ++m_iii;
                else ++m_iv;
            }
        }
        int m_any = m_i + m_ii + m_iii + m_iv;
        if (m_any > 0) {
            atomicAdd(&s_cnt[0], 1ULL);
            if (m_any >= 2) {
                atomicAdd(&s_cnt[1], (unsigned long long)((uint64_t)m_any * (uint64_t)(m_any - 1) / 2ULL));
            }
            if (m_i > 0) atomicAdd(&s_cnt[2], 1ULL);
            if (m_ii > 0) atomicAdd(&s_cnt[3], 1ULL);
            if (m_iii > 0) atomicAdd(&s_cnt[4], 1ULL);
            if (m_iv > 0) atomicAdd(&s_cnt[5], 1ULL);

            const double inv_om = 1.0 / 13983816.0;
            const double ns = c_pmv3_params.sales_n;
            const int32_t sm = c_pmv3_params.share_mode;
            double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;
            double v_cond = 0.0;

            if (m_i > 0) {
                double mu = ns * d_q6[idx];
                double g1 = c3v2_g_share_multi(m_i, mu, c_pmv3_params.nb_r[0], sm);
                double h2 = c3v2_h2_share_multi(m_i, mu, c_pmv3_params.nb_r[0], sm);
                double f0 = c_pmv3_params.funds[0];
                e0 = f0 * g1;
                double var0 = f0 * f0 * (h2 - g1 * g1);
                if (var0 > 0.0) v_cond += var0;
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 0U, e0 * inv_om);
            }
            if (m_ii > 0) {
                double mu = ns * d_q5[idx];
                double g1 = c3v2_g_share_multi(m_ii, mu, c_pmv3_params.nb_r[1], sm);
                double h2 = c3v2_h2_share_multi(m_ii, mu, c_pmv3_params.nb_r[1], sm);
                double f1 = c_pmv3_params.funds[1];
                e1 = f1 * g1;
                double var1 = f1 * f1 * (h2 - g1 * g1);
                if (var1 > 0.0) v_cond += var1;
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 1U, e1 * inv_om);
            }
            if (m_iii > 0) {
                double mu = ns * d_q4[idx];
                double g1 = c3v2_g_share_multi(m_iii, mu, c_pmv3_params.nb_r[2], sm);
                double h2 = c3v2_h2_share_multi(m_iii, mu, c_pmv3_params.nb_r[2], sm);
                double f2 = c_pmv3_params.funds[2];
                e2 = f2 * g1;
                double var2 = f2 * f2 * (h2 - g1 * g1);
                if (var2 > 0.0) v_cond += var2;
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 2U, e2 * inv_om);
            }
            if (m_iv > 0) {
                e3 = (double)m_iv * c_pmv3_params.fixed_iv;
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 3U, e3 * inv_om);
            }
            double etot = e0 + e1 + e2 + e3;
            pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 4U, etot * inv_om);
            m2_contrib = (etot * etot + v_cond) * inv_om;
        }
    }

    s_m2[threadIdx.x] = m2_contrib;
    __syncthreads();
    if (threadIdx.x < 6 && s_cnt[threadIdx.x] != 0ULL) {
        atomicAdd(&d_cnts[threadIdx.x], s_cnt[threadIdx.x]);
    }
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_m2[threadIdx.x] += s_m2[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_m2_block[blockIdx.x] = s_m2[0];
    }
}

__global__ void k_c3v2_eval_540(
    const uint64_t* __restrict__ d_draws6,
    const double* __restrict__ d_p5,
    const double* __restrict__ d_t5,
    const double* __restrict__ d_qiii,
    uint32_t n_tickets,
    unsigned long long* __restrict__ d_cnts, // [0]=win_draws6, [1]=pair_ov, [2]=any_I(ord), [3]=any_II(ord), [4]=any_III(draws6)
    unsigned long long* __restrict__ d_ev_lo, // [0..3]=cat 0..2, [3]=total
    unsigned long long* __restrict__ d_ev_hi,
    double* __restrict__ d_m2_block
) {
    __shared__ unsigned long long s_cnt[5];
    __shared__ double s_m2[256];
    if (threadIdx.x < 5) s_cnt[threadIdx.x] = 0ULL;
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double m2_contrib = 0.0;

    if (idx < 3838380U) {
        uint64_t dm = d_draws6[idx];
        int m5 = 0, m4 = 0;
        uint64_t exact5_masks[6];
        for (uint32_t t = 0; t < n_tickets; ++t) {
            uint64_t tk = c_c3v2_tickets[t];
            int h = __popcll(dm & tk);
            if (h >= 4) {
                if (h == 5) {
                    if (m5 < 6) exact5_masks[m5] = tk;
                    ++m5;
                } else {
                    ++m4;
                }
            }
        }
        int m_any = m5 + m4;
        if (m_any > 0) {
            atomicAdd(&s_cnt[0], 1ULL);
            if (m_any >= 2) {
                atomicAdd(&s_cnt[1], (unsigned long long)((uint64_t)m_any * (uint64_t)(m_any - 1) / 2ULL));
            }
            if (m4 > 0) atomicAdd(&s_cnt[4], 1ULL);

            uint64_t bits[6];
            uint64_t tmp = dm;
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                uint64_t lsb = tmp & (~tmp + 1ULL);
                bits[i] = lsb;
                tmp ^= lsb;
            }

            const double inv_draws = 1.0 / 3838380.0;
            const double inv_ord = 1.0 / (6.0 * 3838380.0);
            const double ns = c_pmv3_params.sales_n;
            const int32_t sm = c_pmv3_params.share_mode;

            double e_iii = 0.0, var_iii = 0.0;
            if (m4 > 0) {
                double mu3 = ns * d_qiii[idx];
                double g1 = c3v2_g_share_multi(m4, mu3, c_pmv3_params.nb_r[2], sm);
                double h2 = c3v2_h2_share_multi(m4, mu3, c_pmv3_params.nb_r[2], sm);
                double f2 = c_pmv3_params.funds[2];
                e_iii = f2 * g1;
                var_iii = f2 * f2 * (h2 - g1 * g1);
                if (var_iii < 0.0) var_iii = 0.0;
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 2U, e_iii * inv_draws);
            }

            if (m5 == 0) {
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 3U, e_iii * inv_draws);
                m2_contrib = (e_iii * e_iii + var_iii) * inv_draws;
            } else {
                double t5_val = d_t5[idx];
                double sum_ei = 0.0, sum_eii = 0.0, sum_m2_ord = 0.0;
                uint64_t cnt_ord_i = 0ULL, cnt_ord_ii = 0ULL;

                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    uint64_t sub5 = dm ^ bits[i];
                    uint32_t rk5 = pmv3_rank_lex(sub5, 40, 5, 658008U);
                    double pfi = d_p5[rk5];
                    int mi = 0;
                    for (int z = 0; z < m5; ++z) {
                        if (exact5_masks[z] == sub5) { mi = 1; break; }
                    }
                    int mii = m5 - mi;
                    if (mi > 0) ++cnt_ord_i;
                    if (mii > 0) ++cnt_ord_ii;

                    double ei = 0.0, vi = 0.0;
                    if (mi > 0) {
                        double mu1 = ns * pfi;
                        double g1 = c3v2_g_share_multi(mi, mu1, c_pmv3_params.nb_r[0], sm);
                        double h2 = c3v2_h2_share_multi(mi, mu1, c_pmv3_params.nb_r[0], sm);
                        double f0 = c_pmv3_params.funds[0];
                        ei = f0 * g1;
                        vi = f0 * f0 * (h2 - g1 * g1);
                        if (vi < 0.0) vi = 0.0;
                    }
                    double eii = 0.0, vii = 0.0;
                    if (mii > 0) {
                        double qii = t5_val - pfi;
                        if (qii < 0.0) qii = 0.0;
                        double mu2 = ns * qii;
                        double g1 = c3v2_g_share_multi(mii, mu2, c_pmv3_params.nb_r[1], sm);
                        double h2 = c3v2_h2_share_multi(mii, mu2, c_pmv3_params.nb_r[1], sm);
                        double f1 = c_pmv3_params.funds[1];
                        eii = f1 * g1;
                        vii = f1 * f1 * (h2 - g1 * g1);
                        if (vii < 0.0) vii = 0.0;
                    }
                    sum_ei += ei;
                    sum_eii += eii;
                    double et_ord = ei + eii + e_iii;
                    sum_m2_ord += et_ord * et_ord + vi + vii + var_iii;
                }
                if (cnt_ord_i > 0ULL) atomicAdd(&s_cnt[2], cnt_ord_i);
                if (cnt_ord_ii > 0ULL) atomicAdd(&s_cnt[3], cnt_ord_ii);

                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 0U, sum_ei * inv_ord);
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 1U, sum_eii * inv_ord);
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 3U, (sum_ei + sum_eii) * inv_ord + e_iii * inv_draws);
                m2_contrib = sum_m2_ord * inv_ord;
            }
        }
    }

    s_m2[threadIdx.x] = m2_contrib;
    __syncthreads();
    if (threadIdx.x < 5 && s_cnt[threadIdx.x] != 0ULL) {
        atomicAdd(&d_cnts[threadIdx.x], s_cnt[threadIdx.x]);
    }
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_m2[threadIdx.x] += s_m2[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_m2_block[blockIdx.x] = s_m2[0];
    }
}

__global__ void k_c3v2_eval_joker(
    const uint64_t* __restrict__ d_base45,
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    const double* __restrict__ d_q3,
    const double* __restrict__ d_q2,
    const double* __restrict__ d_q1,
    uint32_t n_tickets,
    unsigned long long* __restrict__ d_cnts, // [0]=win_24m, [1]=pair_ov_24m, [2..10]=cat_any 0..7
    unsigned long long* __restrict__ d_ev_lo, // [0..8]=cat 0..7, [8]=total
    unsigned long long* __restrict__ d_ev_hi,
    double* __restrict__ d_m2_block
) {
    __shared__ unsigned long long s_cnt[10];
    __shared__ double s_m2[256];
    if (threadIdx.x < 10) s_cnt[threadIdx.x] = 0ULL;
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double m2_contrib = 0.0;

    if (idx < 1221759U) {
        uint64_t dm = d_base45[idx];
        uint16_t cnt_h[6] = {0, 0, 0, 0, 0, 0};
        uint16_t cnt_hj[6][20];
        #pragma unroll
        for (int h = 1; h <= 5; ++h) {
            #pragma unroll
            for (int j = 0; j < 20; ++j) cnt_hj[h][j] = 0;
        }

        bool any_h1 = false;
        for (uint32_t t = 0; t < n_tickets; ++t) {
            uint64_t pk = c_c3v2_tickets[t];
            uint64_t bm = pk & 0x1FFFFFFFFFFFULL;
            int h = __popcll(dm & bm);
            if (h >= 1) {
                any_h1 = true;
                int jb = (int)((pk >> 48) & 0x1FULL);
                ++cnt_h[h];
                ++cnt_hj[h][jb];
            }
        }

        if (any_h1) {
            const double inv_om = 1.0 / 24435180.0;
            const double ns = c_pmv3_params.sales_n;
            const int32_t sm = c_pmv3_params.share_mode;
            const double q5 = d_q5[idx];
            const double q4 = d_q4[idx];
            const double q3 = d_q3[idx];
            const double q2 = d_q2[idx];
            const double q1 = d_q1[idx];

            uint64_t loc_win = 0ULL, loc_ov = 0ULL;
            uint64_t loc_cat_cnt[8] = {0,0,0,0,0,0,0,0};
            double loc_ev_cat[8] = {0,0,0,0,0,0,0,0};
            double loc_ev_tot = 0.0;

            int base345 = (int)cnt_h[5] + (int)cnt_h[4] + (int)cnt_h[3];
            for (int jj = 0; jj < 20; ++jj) {
                int m_c[8];
                m_c[0] = (int)cnt_hj[5][jj];
                m_c[1] = (int)cnt_h[5] - m_c[0];
                m_c[2] = (int)cnt_hj[4][jj];
                m_c[3] = (int)cnt_h[4] - m_c[2];
                m_c[4] = (int)cnt_hj[3][jj];
                m_c[5] = (int)cnt_h[3] - m_c[4];
                m_c[6] = (int)cnt_hj[2][jj];
                m_c[7] = (int)cnt_hj[1][jj];

                int m_any = base345 + m_c[6] + m_c[7];
                if (m_any > 0) {
                    ++loc_win;
                    if (m_any >= 2) {
                        loc_ov += (uint64_t)m_any * (uint64_t)(m_any - 1) / 2ULL;
                    }
                    double pij = c_pmv3_params.pi_joker[jj];
                    double omj = 1.0 - pij;
                    double mu_arr[8] = {
                        ns * q5 * pij, ns * q5 * omj,
                        ns * q4 * pij, ns * q4 * omj,
                        ns * q3 * pij, ns * q3 * omj,
                        ns * q2 * pij, ns * q1 * pij
                    };
                    double et_j = 0.0, vc_j = 0.0;
                    #pragma unroll
                    for (int ci = 0; ci < 8; ++ci) {
                        int mc = m_c[ci];
                        if (mc > 0) {
                            ++loc_cat_cnt[ci];
                            double g1 = c3v2_g_share_multi(mc, mu_arr[ci], c_pmv3_params.nb_r[ci], sm);
                            double h2 = c3v2_h2_share_multi(mc, mu_arr[ci], c_pmv3_params.nb_r[ci], sm);
                            double fc = c_pmv3_params.funds[ci];
                            double ec = fc * g1;
                            double vc = fc * fc * (h2 - g1 * g1);
                            if (vc > 0.0) vc_j += vc;
                            loc_ev_cat[ci] += ec;
                            et_j += ec;
                        }
                    }
                    loc_ev_tot += et_j;
                    m2_contrib += (et_j * et_j + vc_j) * inv_om;
                }
            }

            if (loc_win > 0ULL) {
                atomicAdd(&s_cnt[0], loc_win);
                if (loc_ov > 0ULL) atomicAdd(&s_cnt[1], loc_ov);
                #pragma unroll
                for (int ci = 0; ci < 8; ++ci) {
                    if (loc_cat_cnt[ci] > 0ULL) atomicAdd(&s_cnt[2 + ci], loc_cat_cnt[ci]);
                    if (loc_ev_cat[ci] > 0.0) {
                        pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, (uint32_t)ci, loc_ev_cat[ci] * inv_om);
                    }
                }
                pmv3_atomic_add_u128(d_ev_lo, d_ev_hi, 8U, loc_ev_tot * inv_om);
            }
        }
    }

    s_m2[threadIdx.x] = m2_contrib;
    __syncthreads();
    if (threadIdx.x < 10 && s_cnt[threadIdx.x] != 0ULL) {
        atomicAdd(&d_cnts[threadIdx.x], s_cnt[threadIdx.x]);
    }
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_m2[threadIdx.x] += s_m2[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_m2_block[blockIdx.x] = s_m2[0];
    }
}

// ----------------------------------------------------------------------------
// 6.3 Scanare Marginala pe TOT Universul de 39,077,004 Bilete in VRAM
//     (fara niciun plafon 4,096; filtrare stricta 0 bilete patterned!)
// ----------------------------------------------------------------------------

__global__ void k_c3v2_scan_universe_std(
    const uint64_t* __restrict__ d_tickets,
    const float* __restrict__ d_ev_q05_pair, // 2 floats per ticket: [ev_nom, q05]
    uint32_t n_univ,
    int k_pick,
    int gcols,
    uint32_t n_selected,
    double w_ev,
    double w_q05,
    double* __restrict__ d_block_best_score,
    uint32_t* __restrict__ d_block_best_idx
) {
    __shared__ double s_sc[256];
    __shared__ uint32_t s_id[256];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double best_s = -1.0e30;
    uint32_t best_i = 0xFFFFFFFFU;

    if (idx < n_univ) {
        uint64_t mk = d_tickets[idx];
        if (!c3v2_is_patterned_mask(mk, k_pick, gcols)) {
            bool dup = false;
            double pen = 0.0;
            for (uint32_t s = 0; s < n_selected; ++s) {
                uint64_t smk = c_c3v2_tickets[s];
                if (mk == smk) { dup = true; break; }
                int ov = __popcll(mk & smk);
                pen += c_c3v2_penalties[ov];
            }
            if (!dup) {
                double ev = (double)d_ev_q05_pair[(size_t)idx * 2ULL];
                double q05 = (double)d_ev_q05_pair[(size_t)idx * 2ULL + 1ULL];
                best_s = w_ev * ev + w_q05 * q05 - pen;
                best_i = idx;
            }
        }
    }

    s_sc[threadIdx.x] = best_s;
    s_id[threadIdx.x] = best_i;
    __syncthreads();

    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            double o_s = s_sc[threadIdx.x + step];
            uint32_t o_i = s_id[threadIdx.x + step];
            if (o_s > s_sc[threadIdx.x] || (o_s == s_sc[threadIdx.x] && o_i < s_id[threadIdx.x])) {
                s_sc[threadIdx.x] = o_s;
                s_id[threadIdx.x] = o_i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_block_best_score[blockIdx.x] = s_sc[0];
        d_block_best_idx[blockIdx.x] = s_id[0];
    }
}

__global__ void k_c3v2_scan_universe_joker(
    const uint64_t* __restrict__ d_full_joker, // 24,435,180 packed u64
    const float* __restrict__ d_ev_q05_pair,   // 2 floats per ticket
    uint32_t n_univ,
    uint32_t n_selected,
    double w_ev,
    double w_q05,
    double* __restrict__ d_block_best_score,
    uint32_t* __restrict__ d_block_best_idx
) {
    __shared__ double s_sc[256];
    __shared__ uint32_t s_id[256];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double best_s = -1.0e30;
    uint32_t best_i = 0xFFFFFFFFU;

    if (idx < n_univ) {
        uint64_t pk = d_full_joker[idx];
        uint64_t bm = pk & 0x1FFFFFFFFFFFULL;
        if (!c3v2_is_patterned_mask(bm, 5, 5)) {
            int jb = (int)((pk >> 48) & 0x1FULL);
            bool dup = false;
            double pen = 0.0;
            for (uint32_t s = 0; s < n_selected; ++s) {
                uint64_t spk = c_c3v2_tickets[s];
                if (pk == spk) { dup = true; break; }
                uint64_t sbm = spk & 0x1FFFFFFFFFFFULL;
                int sjb = (int)((spk >> 48) & 0x1FULL);
                int ov = __popcll(bm & sbm);
                int pidx = (jb == sjb) ? (6 + ov) : ov;
                pen += c_c3v2_penalties[pidx];
            }
            if (!dup) {
                double ev = (double)d_ev_q05_pair[(size_t)idx * 2ULL];
                double q05 = (double)d_ev_q05_pair[(size_t)idx * 2ULL + 1ULL];
                best_s = w_ev * ev + w_q05 * q05 - pen;
                best_i = idx;
            }
        }
    }

    s_sc[threadIdx.x] = best_s;
    s_id[threadIdx.x] = best_i;
    __syncthreads();

    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            double o_s = s_sc[threadIdx.x + step];
            uint32_t o_i = s_id[threadIdx.x + step];
            if (o_s > s_sc[threadIdx.x] || (o_s == s_sc[threadIdx.x] && o_i < s_id[threadIdx.x])) {
                s_sc[threadIdx.x] = o_s;
                s_id[threadIdx.x] = o_i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_block_best_score[blockIdx.x] = s_sc[0];
        d_block_best_idx[blockIdx.x] = s_id[0];
    }
}

// ----------------------------------------------------------------------------
// 6.4 Simulatorul Masiv Paralel C4-v2 pe RTX 5060 sm_120:
//     PRNG Counter-Based Philox-4x32-10 fara coliziuni de stare +
//     Esantionare NB2 Exacta (CDF Inversa pt mu <= 15 + Marsaglia-Tsang Gamma +
//     Knuth/PTRS Hormann Poisson pt mu > 15) + Reporturi Dinamice pe 1,040 Pasi
// ----------------------------------------------------------------------------

struct PhiloxStream {
    uint32_t k0, k1;
    uint32_t c0, c1, c2, c3;
    uint32_t out[4];
    int pos;
};

__device__ __forceinline__ void philox4x32_10_eval(
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t k0, uint32_t k1,
    uint32_t res[4]
) {
    const uint32_t M0 = 0xD2511F53U;
    const uint32_t M1 = 0xCD9E8D57U;
    const uint32_t W0 = 0x9E3779B9U;
    const uint32_t W1 = 0xBB67AE85U;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        uint32_t hi0 = __umulhi(M0, c0);
        uint32_t lo0 = M0 * c0;
        uint32_t hi1 = __umulhi(M1, c2);
        uint32_t lo1 = M1 * c2;
        c0 = hi1 ^ c1 ^ k0;
        c2 = hi0 ^ c3 ^ k1;
        c1 = lo1;
        c3 = lo0;
        k0 += W0;
        k1 += W1;
    }
    res[0] = c0; res[1] = c1; res[2] = c2; res[3] = c3;
}

__device__ __forceinline__ void philox_init(
    PhiloxStream* st,
    uint64_t seed,
    uint32_t game_tag,
    uint32_t traj_id,
    uint32_t draw_step
) {
    st->k0 = (uint32_t)(seed & 0xFFFFFFFFULL) ^ (game_tag * 0x85EBCA6BU);
    st->k1 = (uint32_t)(seed >> 32) ^ (game_tag * 0xC2B2AE35U);
    st->c0 = traj_id;
    st->c1 = draw_step;
    st->c2 = 0U;
    st->c3 = 0xA4093822U ^ game_tag;
    philox4x32_10_eval(st->c0, st->c1, st->c2, st->c3, st->k0, st->k1, st->out);
    st->pos = 0;
}

__device__ __forceinline__ uint32_t philox_next_u32(PhiloxStream* st) {
    if (st->pos >= 4) {
        st->c2 += 1U;
        philox4x32_10_eval(st->c0, st->c1, st->c2, st->c3, st->k0, st->k1, st->out);
        st->pos = 0;
    }
    return st->out[st->pos++];
}

__device__ __forceinline__ uint64_t philox_next_u64(PhiloxStream* st) {
    uint64_t lo = (uint64_t)philox_next_u32(st);
    uint64_t hi = (uint64_t)philox_next_u32(st);
    return (hi << 32) | lo;
}

// Uniform deschis in (0, 1) cu rezolutie 53-bit IEEE-754
__device__ __forceinline__ double philox_u01_open(PhiloxStream* st) {
    uint64_t u53 = (philox_next_u64(st) >> 11);
    return ((double)u53 + 0.5) * (1.0 / 9007199254740992.0);
}

// Intreg uniform fara bias in [0, bound)
__device__ __forceinline__ uint32_t philox_uniform_u32(PhiloxStream* st, uint32_t bound) {
    uint64_t m = (uint64_t)philox_next_u32(st) * (uint64_t)bound;
    uint32_t l = (uint32_t)m;
    if (l < bound) {
        uint32_t thresh = (0U - bound) % bound;
        while (l < thresh) {
            m = (uint64_t)philox_next_u32(st) * (uint64_t)bound;
            l = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

// Marsaglia-Tsang exact Gamma(r, 1) pentru r >= 1.5
__device__ __forceinline__ double philox_gamma_mt(PhiloxStream* st, double r) {
    const double d = r - (1.0 / 3.0);
    const double c = rsqrt(9.0 * d);
    const double two_pi = 6.283185307179586476925286766559;
    for (;;) {
        double u1 = philox_u01_open(st);
        double u2 = philox_u01_open(st);
        double z = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
        double v = 1.0 + c * z;
        if (v <= 0.0) continue;
        v = v * v * v;
        double u = philox_u01_open(st);
        double z2 = z * z;
        if (u < 1.0 - 0.0331 * z2 * z2) return d * v;
        if (log(u) < 0.5 * z2 + d * (1.0 - v + log(v))) return d * v;
    }
}

// Esantionator Poisson(lam) 100% exact: Knuth pt lam < 48, Hormann PTRS (1993) pt lam >= 48
__device__ __forceinline__ uint64_t philox_poisson_exact(PhiloxStream* st, double lam) {
    if (lam <= 0.0) return 0ULL;
    if (lam < 48.0) {
        double L = exp(-lam);
        uint64_t k = 0ULL;
        double p = 1.0;
        do {
            ++k;
            p *= philox_u01_open(st);
        } while (p > L);
        return k - 1ULL;
    }
    double slam = sqrt(lam);
    double loglam = log(lam);
    double b = 0.931 + 2.53 * slam;
    double a = -0.059 + 0.02483 * b;
    double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
        double U = philox_u01_open(st) - 0.5;
        double V = philox_u01_open(st);
        double us = 0.5 - fabs(U);
        long long k = (long long)floor((2.0 * a / us + b) * U + lam + 0.43);
        if (k < 0) continue;
        if (us >= 0.07 && V <= vr) return (uint64_t)k;
        if (us < 0.013 && V > us) continue;
        double lhs = log(V * inv_alpha / (a / (us * us) + b));
        double rhs = -lam + (double)k * loglam - lgamma((double)k + 1.0);
        if (lhs <= rhs) return (uint64_t)k;
    }
}

// Esantioneaza W ~ NB2(mu, r) conditionat de u0 >= p0 (adica W >= 1):
// Pentru mu <= 15.0 foloseste continuarea exacta a aceluiasi u0 pe CDF-ul NB2;
// Pentru mu > 15.0 foloseste rejectie exacta pe mixtura Gamma-Poisson!
__device__ __forceinline__ uint64_t philox_nb2_given_pos(
    PhiloxStream* st,
    double mu,
    double r,
    double p0,
    double u0
) {
    if (mu <= 15.0) {
        double q = mu / (r + mu);
        double pmf = p0;
        double cdf = p0;
        for (uint64_t w = 1ULL; w <= 512ULL; ++w) {
            pmf *= (((double)(w - 1ULL) + r) / (double)w) * q;
            cdf += pmf;
            if (u0 <= cdf || pmf < 1.0e-18) return w;
        }
        return 512ULL;
    }
    for (;;) {
        double g = philox_gamma_mt(st, r);
        uint64_t w = philox_poisson_exact(st, g * (mu / r));
        if (w >= 1ULL) return w;
    }
}

// Precalcul tabele compacte de nimeriri per stare in VRAM pentru C4-v2 (pana la P <= 16 portofolii/joc)
__global__ void k_c4v2_prep_hits_649(
    const uint64_t* __restrict__ d_masks,
    const uint64_t* __restrict__ d_port_tickets,
    const uint32_t* __restrict__ d_port_offs,
    const uint32_t* __restrict__ d_port_lens,
    uint32_t n_ports,
    uint32_t* __restrict__ d_hits_out // [13983816 * n_ports]
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    uint64_t dm = d_masks[idx];
    size_t base_out = (size_t)idx * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        uint32_t off = d_port_offs[p];
        uint32_t len = d_port_lens[p];
        uint32_t m_i = 0, m_ii = 0, m_iii = 0, m_iv = 0;
        for (uint32_t t = 0; t < len; ++t) {
            int h = __popcll(dm & d_port_tickets[off + t]);
            if (h >= 3) {
                if (h == 6) ++m_i;
                else if (h == 5) ++m_ii;
                else if (h == 4) ++m_iii;
                else ++m_iv;
            }
        }
        d_hits_out[base_out + p] = (m_i & 0xFFU) | ((m_ii & 0xFFU) << 8) | ((m_iii & 0xFFU) << 16) | ((m_iv & 0xFFU) << 24);
    }
}

__global__ void k_c4v2_prep_hits_540(
    const uint64_t* __restrict__ d_draws6,
    const uint64_t* __restrict__ d_port_tickets,
    const uint32_t* __restrict__ d_port_offs,
    const uint32_t* __restrict__ d_port_lens,
    uint32_t n_ports,
    uint32_t* __restrict__ d_hits_out // [3838380 * n_ports]: bits 0..5 = sub5 mask, 8..15 = m5, 16..31 = m4
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3838380U) return;
    uint64_t dm = d_draws6[idx];
    uint64_t sub5[6];
    uint64_t tmp = dm;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = tmp & (~tmp + 1ULL);
        sub5[i] = dm ^ lsb;
        tmp ^= lsb;
    }
    size_t base_out = (size_t)idx * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        uint32_t off = d_port_offs[p];
        uint32_t len = d_port_lens[p];
        uint32_t mask6 = 0U, m5 = 0U, m4 = 0U;
        for (uint32_t t = 0; t < len; ++t) {
            uint64_t tk = d_port_tickets[off + t];
            int h = __popcll(dm & tk);
            if (h >= 4) {
                if (h == 5) {
                    ++m5;
                    #pragma unroll
                    for (int i = 0; i < 6; ++i) {
                        if (tk == sub5[i]) mask6 |= (1U << i);
                    }
                } else {
                    ++m4;
                }
            }
        }
        d_hits_out[base_out + p] = mask6 | ((m5 & 0xFFU) << 8) | ((m4 & 0xFFFFU) << 16);
    }
}

__global__ void k_c4v2_prep_hits_joker(
    const uint64_t* __restrict__ d_base45,
    const uint64_t* __restrict__ d_port_tickets,
    const uint32_t* __restrict__ d_port_offs,
    const uint32_t* __restrict__ d_port_lens,
    uint32_t n_ports,
    uint64_t* __restrict__ d_hits24m_out // [24435180 * n_ports], 8 bytes = m_c[0..7] ca uint8_t
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    uint64_t dm = d_base45[idx];

    for (uint32_t p = 0; p < n_ports; ++p) {
        uint32_t off = d_port_offs[p];
        uint32_t len = d_port_lens[p];
        uint8_t cnt_h[6] = {0,0,0,0,0,0};
        uint8_t cnt_hj[6][20];
        #pragma unroll
        for (int h = 1; h <= 5; ++h) {
            #pragma unroll
            for (int j = 0; j < 20; ++j) cnt_hj[h][j] = 0;
        }
        for (uint32_t t = 0; t < len; ++t) {
            uint64_t pk = d_port_tickets[off + t];
            uint64_t bm = pk & 0x1FFFFFFFFFFFULL;
            int h = __popcll(dm & bm);
            if (h >= 1) {
                int jb = (int)((pk >> 48) & 0x1FULL);
                ++cnt_h[h];
                ++cnt_hj[h][jb];
            }
        }
        for (int jj = 0; jj < 20; ++jj) {
            uint64_t m0 = (uint64_t)cnt_hj[5][jj];
            uint64_t m1 = (uint64_t)(cnt_h[5] - cnt_hj[5][jj]);
            uint64_t m2 = (uint64_t)cnt_hj[4][jj];
            uint64_t m3 = (uint64_t)(cnt_h[4] - cnt_hj[4][jj]);
            uint64_t m4 = (uint64_t)cnt_hj[3][jj];
            uint64_t m5 = (uint64_t)(cnt_h[3] - cnt_hj[3][jj]);
            uint64_t m6 = (uint64_t)cnt_hj[2][jj];
            uint64_t m7 = (uint64_t)cnt_hj[1][jj];
            uint64_t packed = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
                            | (m4 << 32) | (m5 << 40) | (m6 << 48) | (m7 << 56);
            size_t st_idx = (size_t)jj * 1221759ULL + (size_t)idx;
            d_hits24m_out[st_idx * (size_t)n_ports + p] = packed;
        }
    }
}

// Calculul ratelor analitice exacte de ordinul 1 si 2 (u128_s96, 12 sloturi/categorie) pentru lantul Markov dinamic cu reporturi:
// [0]  psi0       = E[G_NB(m_c, N*q_c, r_c)] (s=0)
// [1]  rho0       = E[1(m_c==0) * P(W_c==0 | N*q_c)] (s=0)
// [2]  psi_bar    = (1/16) sum_l E[G_NB(m_c, N*xi_l*q_c, r_c)] (s>=1)
// [3]  eta_unit   = (1/16) sum_l xi_l * E[G_NB(m_c, N*xi_l*q_c, r_c)] (s>=1)
// [4]  rho_bar    = (1/16) sum_l E[1(m_c==0) * P(W_c==0 | N*xi_l*q_c)] (s>=1)
// [5]  beta_unit  = (1/16) sum_l xi_l * E[1(m_c==0) * P(W_c==0 | N*xi_l*q_c)] (s>=1)
// [6]  h2_0       = E[H2_NB(m_c, N*q_c, r_c)] (s=0)
// [7]  h2_bar0    = (1/16) sum_l E[H2_NB(m_c, N*xi_l*q_c, r_c)] (s>=1)
// [8]  h2_bar1    = (1/16) sum_l xi_l * E[H2_NB(m_c, N*xi_l*q_c, r_c)] (s>=1)
// [9]  h2_bar2    = (1/16) sum_l xi_l^2 * E[H2_NB(m_c, N*xi_l*q_c, r_c)] (s>=1)
// [10] gamma_unit = (1/16) sum_l xi_l^2 * E[1(m_c==0) * P(W_c==0 | N*xi_l*q_c)] (s>=1)
__device__ __forceinline__ void c4v2_accum_cat_rates_u128(
    int mc,
    double qc,
    double ns,
    double rc,
    double inv_weight,
    uint32_t base_slot,
    unsigned long long* d_lo,
    unsigned long long* d_hi
) {
    if (mc > 0) {
        double g0 = c3v2_g_share_multi(mc, ns * qc, rc, 0);
        double h0 = c3v2_h2_share_multi(mc, ns * qc, rc, 0);
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 0U, g0 * inv_weight);
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 6U, h0 * inv_weight);
        double s_psi = 0.0, s_eta = 0.0, s_h20 = 0.0, s_h21 = 0.0, s_h22 = 0.0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            double xl = c_c4v2_xi16[l];
            double gl = c3v2_g_share_multi(mc, ns * xl * qc, rc, 0);
            double hl = c3v2_h2_share_multi(mc, ns * xl * qc, rc, 0);
            s_psi += gl;
            s_eta += xl * gl;
            s_h20 += hl;
            s_h21 += xl * hl;
            s_h22 += (xl * xl) * hl;
        }
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 2U, (s_psi * 0.0625) * inv_weight);
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 3U, (s_eta * 0.0625) * inv_weight);
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 7U, (s_h20 * 0.0625) * inv_weight);
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 8U, (s_h21 * 0.0625) * inv_weight);
        pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 9U, (s_h22 * 0.0625) * inv_weight);
    } else {
        double p0 = c3v2_nb_pzero(ns * qc, rc, 0);
        if (p0 > 1.0e-18) {
            pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 1U, p0 * inv_weight);
        }
        double s_rho = 0.0, s_beta = 0.0, s_gam = 0.0;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            double xl = c_c4v2_xi16[l];
            double pl = c3v2_nb_pzero(ns * xl * qc, rc, 0);
            s_rho += pl;
            s_beta += xl * pl;
            s_gam += (xl * xl) * pl;
        }
        if (s_rho > 1.0e-18) {
            pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 4U, (s_rho * 0.0625) * inv_weight);
            pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 5U, (s_beta * 0.0625) * inv_weight);
            pmv3_atomic_add_u128(d_lo, d_hi, base_slot + 10U, (s_gam * 0.0625) * inv_weight);
        }
    }
}

__global__ void k_c4v2_rates_649(
    const double* __restrict__ d_q6,
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    const uint32_t* __restrict__ d_hits,
    uint32_t n_ports,
    unsigned long long* __restrict__ d_lo, // n_ports * 4 * 12 ([0..3]=cat I..III, [3][0]=fixed_iv mean, [3][6]=fixed_iv m2)
    unsigned long long* __restrict__ d_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 13983816U) return;
    const double inv_om = 1.0 / 13983816.0;
    const double ns = c_pmv3_params.sales_n;
    const double q_arr[3] = { d_q6[idx], d_q5[idx], d_q4[idx] };
    size_t b = (size_t)idx * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        uint32_t h = d_hits[b + p];
        int mc[4] = { (int)(h & 0xFFU), (int)((h >> 8) & 0xFFU), (int)((h >> 16) & 0xFFU), (int)(h >> 24) };
        for (int c = 0; c < 3; ++c) {
            c4v2_accum_cat_rates_u128(mc[c], q_arr[c], ns, c_pmv3_params.nb_r[c], inv_om, (p * 4U + (uint32_t)c) * 12U, d_lo, d_hi);
        }
        if (mc[3] > 0) {
            double v4 = (double)mc[3] * c_pmv3_params.fixed_iv;
            pmv3_atomic_add_u128(d_lo, d_hi, (p * 4U + 3U) * 12U + 0U, v4 * inv_om);
            pmv3_atomic_add_u128(d_lo, d_hi, (p * 4U + 3U) * 12U + 6U, (v4 * v4) * inv_om);
        }
    }
}

__global__ void k_c4v2_rates_540(
    const uint64_t* __restrict__ d_draws6,
    const double* __restrict__ d_p5,
    const double* __restrict__ d_t5,
    const double* __restrict__ d_qiii,
    const uint32_t* __restrict__ d_hits,
    uint32_t n_ports,
    unsigned long long* __restrict__ d_lo, // n_ports * 3 * 12
    unsigned long long* __restrict__ d_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3838380U) return;
    uint64_t dm = d_draws6[idx];
    uint64_t tmp = dm;
    double pf[6];
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        uint64_t lsb = tmp & (~tmp + 1ULL);
        uint32_t rk5 = pmv3_rank_lex(dm ^ lsb, 40, 5, 658008U);
        pf[i] = d_p5[rk5];
        tmp ^= lsb;
    }
    const double t5 = d_t5[idx];
    const double q3 = d_qiii[idx];
    const double inv_draws = 1.0 / 3838380.0;
    const double inv_ord = 1.0 / (6.0 * 3838380.0);
    const double ns = c_pmv3_params.sales_n;

    size_t b = (size_t)idx * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        uint32_t h = d_hits[b + p];
        uint32_t mask6 = h & 0x3FU;
        int m5 = (int)((h >> 8) & 0xFFU);
        int m4 = (int)(h >> 16);
        for (int i = 0; i < 6; ++i) {
            int mi = (int)((mask6 >> i) & 1U);
            int mii = m5 - mi;
            double qii = t5 - pf[i];
            if (qii < 0.0) qii = 0.0;
            c4v2_accum_cat_rates_u128(mi, pf[i], ns, c_pmv3_params.nb_r[0], inv_ord, (p * 3U + 0U) * 12U, d_lo, d_hi);
            c4v2_accum_cat_rates_u128(mii, qii, ns, c_pmv3_params.nb_r[1], inv_ord, (p * 3U + 1U) * 12U, d_lo, d_hi);
        }
        c4v2_accum_cat_rates_u128(m4, q3, ns, c_pmv3_params.nb_r[2], inv_draws, (p * 3U + 2U) * 12U, d_lo, d_hi);
    }
}

__global__ void k_c4v2_rates_joker(
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    const double* __restrict__ d_q3,
    const double* __restrict__ d_q2,
    const double* __restrict__ d_q1,
    const uint64_t* __restrict__ d_hits24m,
    uint32_t n_ports,
    unsigned long long* __restrict__ d_lo, // n_ports * 8 * 12
    unsigned long long* __restrict__ d_hi
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1221759U) return;
    const double inv_om = 1.0 / 24435180.0;
    const double ns = c_pmv3_params.sales_n;
    const double q5 = d_q5[idx], q4 = d_q4[idx], q3 = d_q3[idx], q2 = d_q2[idx], q1 = d_q1[idx];

    for (int jj = 0; jj < 20; ++jj) {
        double pij = c_pmv3_params.pi_joker[jj];
        double omj = 1.0 - pij;
        double q_arr[8] = {
            q5 * pij, q5 * omj, q4 * pij, q4 * omj,
            q3 * pij, q3 * omj, q2 * pij, q1 * pij
        };
        size_t st_idx = (size_t)jj * 1221759ULL + (size_t)idx;
        size_t b = st_idx * (size_t)n_ports;
        for (uint32_t p = 0; p < n_ports; ++p) {
            uint64_t pk = d_hits24m[b + p];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int mc = (int)((pk >> (c * 8)) & 0xFFULL);
                c4v2_accum_cat_rates_u128(mc, q_arr[c], ns, c_pmv3_params.nb_r[c], inv_om, (p * 8U + (uint32_t)c) * 12U, d_lo, d_hi);
            }
        }
    }
}

// Kernele principale de simulare Monte Carlo (1,040 extrageri per traiectorie)
__global__ void k_c4v2_sim_chunk_649(
    const double* __restrict__ d_q6,
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    const uint32_t* __restrict__ d_hits,
    uint32_t n_ports,
    uint32_t chunk_start,
    uint32_t chunk_len,
    uint64_t seed,
    double rep0_i, double rep0_ii, double rep0_iii,
    double fnew_i, double fnew_ii, double fnew_iii,
    double* __restrict__ d_out_pay4,   // [chunk_len * n_ports * 4]: [draw0, 1y, 5y, 10y]
    uint8_t* __restrict__ d_out_flags  // [chunk_len * n_ports]: bit0=win_d0, bit1=big1y, bit2=big5y, bit3=big10y
) {
    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= chunk_len) return;
    uint32_t traj_id = chunk_start + gtid;

    double R[16][3];
    double cum[16];
    double p_d0[16], p_1y[16], p_5y[16];
    uint8_t flg[16];

    for (uint32_t p = 0; p < n_ports; ++p) {
        R[p][0] = rep0_i;
        R[p][1] = rep0_ii;
        R[p][2] = rep0_iii;
        cum[p] = 0.0;
        p_d0[p] = 0.0; p_1y[p] = 0.0; p_5y[p] = 0.0;
        flg[p] = 0U;
    }

    const double ns_base = c_pmv3_params.sales_n;
    const double fnew[3] = { fnew_i, fnew_ii, fnew_iii };
    const double nbr[3] = { c_pmv3_params.nb_r[0], c_pmv3_params.nb_r[1], c_pmv3_params.nb_r[2] };
    const double fix_iv = c_pmv3_params.fixed_iv;

    for (uint32_t s = 0U; s < 1040U; ++s) {
        PhiloxStream rng;
        philox_init(&rng, seed, 649U, traj_id, s);

        uint32_t omega = philox_uniform_u32(&rng, 13983816U);
        double xi = (s == 0U) ? 1.0 : c_c4v2_xi16[philox_next_u32(&rng) & 15U];
        double ns = ns_base * xi;
        double q_arr[3] = { d_q6[omega], d_q5[omega], d_q4[omega] };

        uint32_t h_arr[16];
        uint32_t any_cat_m[3] = {0U, 0U, 0U};
        size_t hb = (size_t)omega * (size_t)n_ports;
        for (uint32_t p = 0; p < n_ports; ++p) {
            uint32_t hv = d_hits[hb + p];
            h_arr[p] = hv;
            any_cat_m[0] |= (hv & 0xFFU);
            any_cat_m[1] |= ((hv >> 8) & 0xFFU);
            any_cat_m[2] |= ((hv >> 16) & 0xFFU);
        }

        bool w_zero[3];
        uint64_t w_val[3] = {0ULL, 0ULL, 0ULL};
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            double mu = ns * q_arr[c];
            double p0 = c3v2_nb_pzero(mu, nbr[c], 0);
            double u0 = philox_u01_open(&rng);
            if (u0 < p0) {
                w_zero[c] = true;
                w_val[c] = 0ULL;
            } else {
                w_zero[c] = false;
                if (any_cat_m[c] != 0U) {
                    w_val[c] = philox_nb2_given_pos(&rng, mu, nbr[c], p0, u0);
                }
            }
        }

        for (uint32_t p = 0; p < n_ports; ++p) {
            uint32_t hv = h_arr[p];
            int mc[3] = { (int)(hv & 0xFFU), (int)((hv >> 8) & 0xFFU), (int)((hv >> 16) & 0xFFU) };
            int m_iv = (int)(hv >> 24);
            double step_win = 0.0;
            bool step_big = false;

            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                double A = R[p][c] + fnew[c] * xi;
                if (mc[c] > 0) {
                    double pay_tot = A * ((double)mc[c] / ((double)mc[c] + (double)w_val[c]));
                    double pay_one = A / ((double)mc[c] + (double)w_val[c]);
                    if (pay_one >= 1000.0) step_big = true;
                    step_win += pay_tot;
                    R[p][c] = 0.0;
                } else {
                    R[p][c] = w_zero[c] ? A : 0.0;
                }
            }
            if (m_iv > 0) {
                step_win += (double)m_iv * fix_iv;
            }
            if (s == 0U) {
                p_d0[p] = step_win;
                if (hv != 0U) flg[p] |= 1U;
            }
            if (step_big) {
                if (s < 104U) flg[p] |= 2U;
                if (s < 520U) flg[p] |= 4U;
                flg[p] |= 8U;
            }
            cum[p] += step_win;
            if (s == 103U) p_1y[p] = cum[p];
            else if (s == 519U) p_5y[p] = cum[p];
        }
    }

    size_t out_base = (size_t)gtid * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        size_t o4 = (out_base + p) * 4ULL;
        d_out_pay4[o4 + 0ULL] = p_d0[p];
        d_out_pay4[o4 + 1ULL] = p_1y[p];
        d_out_pay4[o4 + 2ULL] = p_5y[p];
        d_out_pay4[o4 + 3ULL] = cum[p];
        d_out_flags[out_base + p] = flg[p];
    }
}

__global__ void k_c4v2_sim_chunk_540(
    const uint64_t* __restrict__ d_draws6,
    const double* __restrict__ d_p5,
    const double* __restrict__ d_t5,
    const double* __restrict__ d_qiii,
    const uint32_t* __restrict__ d_hits,
    uint32_t n_ports,
    uint32_t chunk_start,
    uint32_t chunk_len,
    uint64_t seed,
    double rep0_i, double rep0_ii, double rep0_iii,
    double fnew_i, double fnew_ii, double fnew_iii,
    double* __restrict__ d_out_pay4,
    uint8_t* __restrict__ d_out_flags
) {
    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= chunk_len) return;
    uint32_t traj_id = chunk_start + gtid;

    double R[16][3];
    double cum[16];
    double p_d0[16], p_1y[16], p_5y[16];
    uint8_t flg[16];

    for (uint32_t p = 0; p < n_ports; ++p) {
        R[p][0] = rep0_i; R[p][1] = rep0_ii; R[p][2] = rep0_iii;
        cum[p] = 0.0;
        p_d0[p] = 0.0; p_1y[p] = 0.0; p_5y[p] = 0.0;
        flg[p] = 0U;
    }

    const double ns_base = c_pmv3_params.sales_n;
    const double fnew[3] = { fnew_i, fnew_ii, fnew_iii };
    const double nbr[3] = { c_pmv3_params.nb_r[0], c_pmv3_params.nb_r[1], c_pmv3_params.nb_r[2] };

    for (uint32_t s = 0U; s < 1040U; ++s) {
        PhiloxStream rng;
        philox_init(&rng, seed, 540U, traj_id, s);

        uint32_t ord_state = philox_uniform_u32(&rng, 23030280U); // 6 * 3,838,380
        uint32_t d_idx = ord_state / 6U;
        uint32_t ball6_pos = ord_state - d_idx * 6U;
        double xi = (s == 0U) ? 1.0 : c_c4v2_xi16[philox_next_u32(&rng) & 15U];
        double ns = ns_base * xi;

        uint64_t dm = d_draws6[d_idx];
        uint64_t tmp = dm;
        uint64_t lsb_i = 0ULL;
        for (uint32_t k = 0U; k <= ball6_pos; ++k) {
            lsb_i = tmp & (~tmp + 1ULL);
            tmp ^= lsb_i;
        }
        uint32_t rk5 = pmv3_rank_lex(dm ^ lsb_i, 40, 5, 658008U);
        double pfi = d_p5[rk5];
        double qii = d_t5[d_idx] - pfi;
        if (qii < 0.0) qii = 0.0;
        double q_arr[3] = { pfi, qii, d_qiii[d_idx] };

        int mc_mat[16][3];
        uint32_t any_cat_m[3] = {0U, 0U, 0U};
        size_t hb = (size_t)d_idx * (size_t)n_ports;
        for (uint32_t p = 0; p < n_ports; ++p) {
            uint32_t hv = d_hits[hb + p];
            int mi = (int)((hv >> ball6_pos) & 1U);
            int m5 = (int)((hv >> 8) & 0xFFU);
            int mii = m5 - mi;
            int m4 = (int)(hv >> 16);
            mc_mat[p][0] = mi;
            mc_mat[p][1] = mii;
            mc_mat[p][2] = m4;
            any_cat_m[0] |= (uint32_t)mi;
            any_cat_m[1] |= (uint32_t)mii;
            any_cat_m[2] |= (uint32_t)m4;
        }

        bool w_zero[3];
        uint64_t w_val[3] = {0ULL, 0ULL, 0ULL};
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            double mu = ns * q_arr[c];
            double p0 = c3v2_nb_pzero(mu, nbr[c], 0);
            double u0 = philox_u01_open(&rng);
            if (u0 < p0) {
                w_zero[c] = true;
                w_val[c] = 0ULL;
            } else {
                w_zero[c] = false;
                if (any_cat_m[c] != 0U) {
                    w_val[c] = philox_nb2_given_pos(&rng, mu, nbr[c], p0, u0);
                }
            }
        }

        for (uint32_t p = 0; p < n_ports; ++p) {
            double step_win = 0.0;
            bool step_big = false;
            int m_tot = mc_mat[p][0] + mc_mat[p][1] + mc_mat[p][2];

            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                double A = R[p][c] + fnew[c] * xi;
                int mc = mc_mat[p][c];
                if (mc > 0) {
                    double pay_tot = A * ((double)mc / ((double)mc + (double)w_val[c]));
                    double pay_one = A / ((double)mc + (double)w_val[c]);
                    if (pay_one >= 1000.0) step_big = true;
                    step_win += pay_tot;
                    R[p][c] = 0.0;
                } else {
                    R[p][c] = w_zero[c] ? A : 0.0;
                }
            }
            if (s == 0U) {
                p_d0[p] = step_win;
                if (m_tot > 0) flg[p] |= 1U;
            }
            if (step_big) {
                if (s < 104U) flg[p] |= 2U;
                if (s < 520U) flg[p] |= 4U;
                flg[p] |= 8U;
            }
            cum[p] += step_win;
            if (s == 103U) p_1y[p] = cum[p];
            else if (s == 519U) p_5y[p] = cum[p];
        }
    }

    size_t out_base = (size_t)gtid * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        size_t o4 = (out_base + p) * 4ULL;
        d_out_pay4[o4 + 0ULL] = p_d0[p];
        d_out_pay4[o4 + 1ULL] = p_1y[p];
        d_out_pay4[o4 + 2ULL] = p_5y[p];
        d_out_pay4[o4 + 3ULL] = cum[p];
        d_out_flags[out_base + p] = flg[p];
    }
}

__global__ void k_c4v2_sim_chunk_joker(
    const double* __restrict__ d_q5,
    const double* __restrict__ d_q4,
    const double* __restrict__ d_q3,
    const double* __restrict__ d_q2,
    const double* __restrict__ d_q1,
    const uint64_t* __restrict__ d_hits24m,
    uint32_t n_ports,
    uint32_t chunk_start,
    uint32_t chunk_len,
    uint64_t seed,
    const double* __restrict__ d_rep0_8,
    const double* __restrict__ d_fnew_8,
    double* __restrict__ d_out_pay4,
    uint8_t* __restrict__ d_out_flags
) {
    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= chunk_len) return;
    uint32_t traj_id = chunk_start + gtid;

    double R[16][8]; // n_ports <= 16 per batch pe Joker
    double cum[16];
    double p_d0[16], p_1y[16], p_5y[16];
    uint8_t flg[16];
    double fnew[8], nbr[8];

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        fnew[c] = d_fnew_8[c];
        nbr[c] = c_pmv3_params.nb_r[c];
    }
    for (uint32_t p = 0; p < n_ports; ++p) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) R[p][c] = d_rep0_8[c];
        cum[p] = 0.0;
        p_d0[p] = 0.0; p_1y[p] = 0.0; p_5y[p] = 0.0;
        flg[p] = 0U;
    }

    const double ns_base = c_pmv3_params.sales_n;

    for (uint32_t s = 0U; s < 1040U; ++s) {
        PhiloxStream rng;
        philox_init(&rng, seed, 4520U, traj_id, s);

        uint32_t st_idx = philox_uniform_u32(&rng, 24435180U);
        uint32_t jj = st_idx / 1221759U;
        uint32_t b_idx = st_idx - jj * 1221759U;
        double xi = (s == 0U) ? 1.0 : c_c4v2_xi16[philox_next_u32(&rng) & 15U];
        double ns = ns_base * xi;

        double pij = c_pmv3_params.pi_joker[jj];
        double omj = 1.0 - pij;
        double q5 = d_q5[b_idx], q4 = d_q4[b_idx], q3 = d_q3[b_idx], q2 = d_q2[b_idx], q1 = d_q1[b_idx];
        double q_arr[8] = {
            q5 * pij, q5 * omj, q4 * pij, q4 * omj,
            q3 * pij, q3 * omj, q2 * pij, q1 * pij
        };

        uint64_t pk_arr[16];
        uint64_t or_pk = 0ULL;
        size_t hb = (size_t)st_idx * (size_t)n_ports;
        for (uint32_t p = 0; p < n_ports; ++p) {
            uint64_t pk = d_hits24m[hb + p];
            pk_arr[p] = pk;
            or_pk |= pk;
        }

        bool w_zero[8];
        uint64_t w_val[8] = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            double mu = ns * q_arr[c];
            double p0 = c3v2_nb_pzero(mu, nbr[c], 0);
            double u0 = philox_u01_open(&rng);
            if (u0 < p0) {
                w_zero[c] = true;
                w_val[c] = 0ULL;
            } else {
                w_zero[c] = false;
                if (((or_pk >> (c * 8)) & 0xFFULL) != 0ULL) {
                    w_val[c] = philox_nb2_given_pos(&rng, mu, nbr[c], p0, u0);
                }
            }
        }

        for (uint32_t p = 0; p < n_ports; ++p) {
            uint64_t pk = pk_arr[p];
            double step_win = 0.0;
            bool step_big = false;

            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                double A = R[p][c] + fnew[c] * xi;
                int mc = (int)((pk >> (c * 8)) & 0xFFULL);
                if (mc > 0) {
                    double pay_tot = A * ((double)mc / ((double)mc + (double)w_val[c]));
                    double pay_one = A / ((double)mc + (double)w_val[c]);
                    if (pay_one >= 1000.0) step_big = true;
                    step_win += pay_tot;
                    R[p][c] = 0.0;
                } else {
                    R[p][c] = w_zero[c] ? A : 0.0;
                }
            }
            if (s == 0U) {
                p_d0[p] = step_win;
                if (pk != 0ULL) flg[p] |= 1U;
            }
            if (step_big) {
                if (s < 104U) flg[p] |= 2U;
                if (s < 520U) flg[p] |= 4U;
                flg[p] |= 8U;
            }
            cum[p] += step_win;
            if (s == 103U) p_1y[p] = cum[p];
            else if (s == 519U) p_5y[p] = cum[p];
        }
    }

    size_t out_base = (size_t)gtid * (size_t)n_ports;
    for (uint32_t p = 0; p < n_ports; ++p) {
        size_t o4 = (out_base + p) * 4ULL;
        d_out_pay4[o4 + 0ULL] = p_d0[p];
        d_out_pay4[o4 + 1ULL] = p_1y[p];
        d_out_pay4[o4 + 2ULL] = p_5y[p];
        d_out_pay4[o4 + 3ULL] = cum[p];
        d_out_flags[out_base + p] = flg[p];
    }
}

extern "C" {

int32_t v3_cuda_c3v2_prep_649(
    const uint64_t* d_masks_649,
    const V3CrowdParamsF64* h_params,
    double* d_q6_out,
    double* d_q5_out,
    double* d_q4_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init();
    if (rc != 0) return rc;

    const uint32_t N = 13983816U;
    const uint32_t N5 = 1906884U;
    const uint32_t N4 = 211876U;

    cudaError_t err = cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    if (err != cudaSuccess) return (int32_t)err;

    double *d_p = nullptr, *d_s5 = nullptr, *d_s4 = nullptr;
    unsigned long long *d_zlo = nullptr, *d_zhi = nullptr;
    unsigned long long *d_s5lo = nullptr, *d_s5hi = nullptr, *d_s4lo = nullptr, *d_s4hi = nullptr;

    auto cleanup = [&]() {
        if (d_p) cudaFree(d_p);
        if (d_s5) cudaFree(d_s5);
        if (d_s4) cudaFree(d_s4);
        if (d_zlo) cudaFree(d_zlo);
        if (d_zhi) cudaFree(d_zhi);
        if (d_s5lo) cudaFree(d_s5lo);
        if (d_s5hi) cudaFree(d_s5hi);
        if (d_s4lo) cudaFree(d_s4lo);
        if (d_s4hi) cudaFree(d_s4hi);
    };

    if (cudaMalloc(&d_p, (size_t)N * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s5, (size_t)N5 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_zlo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_zhi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s5lo, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s5hi, (size_t)N5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaMemset(d_zlo, 0, sizeof(unsigned long long));
    cudaMemset(d_zhi, 0, sizeof(unsigned long long));
    cudaMemset(d_s5lo, 0, (size_t)N5 * sizeof(unsigned long long));
    cudaMemset(d_s5hi, 0, (size_t)N5 * sizeof(unsigned long long));
    cudaMemset(d_s4lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s4hi, 0, (size_t)N4 * sizeof(unsigned long long));

    uint32_t gN = (N + 255U) / 256U;
    k_pmv3_sum_z_u128<<<gN, 256>>>(d_masks_649, N, 6, d_p, d_zlo, d_zhi);
    unsigned long long h_zlo = 0ULL, h_zhi = 0ULL;
    cudaMemcpy(&h_zlo, d_zlo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_zhi, d_zhi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    double z_val = ldexp((double)h_zhi, -32) + ldexp((double)(h_zlo >> 32), -64) + ldexp((double)(h_zlo & 0xFFFFFFFFULL), -96);

    k_pmv3_649_fwd_u128<<<gN, 256>>>(d_masks_649, d_p, 1.0 / z_val, d_s5lo, d_s5hi, d_s4lo, d_s4hi);
    k_pmv3_u128_to_f64<<<(N5 + 255U) / 256U, 256>>>(d_s5lo, d_s5hi, d_s5, N5);
    k_pmv3_u128_to_f64<<<(N4 + 255U) / 256U, 256>>>(d_s4lo, d_s4hi, d_s4, N4);
    k_c3v2_649_extract_q<<<gN, 256>>>(d_masks_649, d_p, d_s5, d_s4, d_q6_out, d_q5_out, d_q4_out);

    err = cudaDeviceSynchronize();
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_c3v2_prep_540(
    const uint64_t* d_tickets_540,
    const uint64_t* d_draws6_540,
    const V3CrowdParamsF64* h_params,
    double* d_p5_out,
    double* d_t5_out,
    double* d_qiii_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init();
    if (rc != 0) return rc;

    const uint32_t NT = 658008U;
    const uint32_t ND = 3838380U;
    const uint32_t N4 = 91390U;

    cudaError_t err = cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    if (err != cudaSuccess) return (int32_t)err;

    double* d_s4 = nullptr;
    unsigned long long *d_zlo = nullptr, *d_zhi = nullptr, *d_s4lo = nullptr, *d_s4hi = nullptr;

    auto cleanup = [&]() {
        if (d_s4) cudaFree(d_s4);
        if (d_zlo) cudaFree(d_zlo);
        if (d_zhi) cudaFree(d_zhi);
        if (d_s4lo) cudaFree(d_s4lo);
        if (d_s4hi) cudaFree(d_s4hi);
    };

    if (cudaMalloc(&d_s4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_zlo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_zhi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4lo, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4hi, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaMemset(d_zlo, 0, sizeof(unsigned long long));
    cudaMemset(d_zhi, 0, sizeof(unsigned long long));
    cudaMemset(d_s4lo, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s4hi, 0, (size_t)N4 * sizeof(unsigned long long));

    uint32_t gT = (NT + 255U) / 256U;
    uint32_t gD = (ND + 255U) / 256U;
    k_pmv3_sum_z_u128<<<gT, 256>>>(d_tickets_540, NT, 5, d_p5_out, d_zlo, d_zhi);
    unsigned long long h_zlo = 0ULL, h_zhi = 0ULL;
    cudaMemcpy(&h_zlo, d_zlo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_zhi, d_zhi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    double z_val = ldexp((double)h_zhi, -32) + ldexp((double)(h_zlo >> 32), -64) + ldexp((double)(h_zlo & 0xFFFFFFFFULL), -96);

    k_pmv3_540_fwd_u128<<<gT, 256>>>(d_tickets_540, d_p5_out, 1.0 / z_val, d_s4lo, d_s4hi);
    k_pmv3_u128_to_f64<<<(N4 + 255U) / 256U, 256>>>(d_s4lo, d_s4hi, d_s4, N4);
    k_c3v2_540_extract_q<<<gD, 256>>>(d_draws6_540, d_p5_out, d_s4, d_t5_out, d_qiii_out);

    err = cudaDeviceSynchronize();
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_c3v2_prep_joker(
    const uint64_t* d_base_joker,
    const V3CrowdParamsF64* h_params,
    double* d_q5_out,
    double* d_q4_out,
    double* d_q3_out,
    double* d_q2_out,
    double* d_q1_out
) {
    int32_t rc = ensure_pmv3_binom_init();
    if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init();
    if (rc != 0) return rc;

    const uint32_t NB = 1221759U;
    const uint32_t N4 = 148995U;
    const uint32_t N3 = 14190U;
    const uint32_t N2 = 990U;
    const uint32_t N1 = 45U;

    cudaError_t err = cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    if (err != cudaSuccess) return (int32_t)err;

    double *d_s4 = nullptr, *d_s3 = nullptr, *d_s2 = nullptr, *d_s1 = nullptr;
    unsigned long long *d_zlo = nullptr, *d_zhi = nullptr;
    unsigned long long *d_s4l = nullptr, *d_s4h = nullptr, *d_s3l = nullptr, *d_s3h = nullptr;
    unsigned long long *d_s2l = nullptr, *d_s2h = nullptr, *d_s1l = nullptr, *d_s1h = nullptr;

    auto cleanup = [&]() {
        if (d_s4) cudaFree(d_s4); if (d_s3) cudaFree(d_s3); if (d_s2) cudaFree(d_s2); if (d_s1) cudaFree(d_s1);
        if (d_zlo) cudaFree(d_zlo); if (d_zhi) cudaFree(d_zhi);
        if (d_s4l) cudaFree(d_s4l); if (d_s4h) cudaFree(d_s4h);
        if (d_s3l) cudaFree(d_s3l); if (d_s3h) cudaFree(d_s3h);
        if (d_s2l) cudaFree(d_s2l); if (d_s2h) cudaFree(d_s2h);
        if (d_s1l) cudaFree(d_s1l); if (d_s1h) cudaFree(d_s1h);
    };

    if (cudaMalloc(&d_s4, (size_t)N4 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s3, (size_t)N3 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s2, (size_t)N2 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_s1, (size_t)N1 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_zlo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_zhi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4l, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s4h, (size_t)N4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s3l, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s3h, (size_t)N3 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s2l, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s2h, (size_t)N2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s1l, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_s1h, (size_t)N1 * sizeof(unsigned long long)) != cudaSuccess) {
        cleanup();
        return -3;
    }

    cudaMemset(d_zlo, 0, sizeof(unsigned long long)); cudaMemset(d_zhi, 0, sizeof(unsigned long long));
    cudaMemset(d_s4l, 0, (size_t)N4 * sizeof(unsigned long long)); cudaMemset(d_s4h, 0, (size_t)N4 * sizeof(unsigned long long));
    cudaMemset(d_s3l, 0, (size_t)N3 * sizeof(unsigned long long)); cudaMemset(d_s3h, 0, (size_t)N3 * sizeof(unsigned long long));
    cudaMemset(d_s2l, 0, (size_t)N2 * sizeof(unsigned long long)); cudaMemset(d_s2h, 0, (size_t)N2 * sizeof(unsigned long long));
    cudaMemset(d_s1l, 0, (size_t)N1 * sizeof(unsigned long long)); cudaMemset(d_s1h, 0, (size_t)N1 * sizeof(unsigned long long));

    uint32_t gB = (NB + 255U) / 256U;
    k_pmv3_sum_z_u128<<<gB, 256>>>(d_base_joker, NB, 5, d_q5_out, d_zlo, d_zhi);
    unsigned long long h_zlo = 0ULL, h_zhi = 0ULL;
    cudaMemcpy(&h_zlo, d_zlo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_zhi, d_zhi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    double z_val = ldexp((double)h_zhi, -32) + ldexp((double)(h_zlo >> 32), -64) + ldexp((double)(h_zlo & 0xFFFFFFFFULL), -96);

    k_pmv3_joker_fwd_u128<<<gB, 256>>>(
        d_base_joker, d_q5_out, 1.0 / z_val,
        d_s4l, d_s4h, d_s3l, d_s3h, d_s2l, d_s2h, d_s1l, d_s1h
    );
    k_pmv3_u128_to_f64<<<(N4 + 255U) / 256U, 256>>>(d_s4l, d_s4h, d_s4, N4);
    k_pmv3_u128_to_f64<<<(N3 + 255U) / 256U, 256>>>(d_s3l, d_s3h, d_s3, N3);
    k_pmv3_u128_to_f64<<<(N2 + 255U) / 256U, 256>>>(d_s2l, d_s2h, d_s2, N2);
    k_pmv3_u128_to_f64<<<1, 64>>>(d_s1l, d_s1h, d_s1, N1);
    k_pmv3_joker_compute_q_f64<<<gB, 256>>>(
        d_base_joker, d_q5_out, d_s4, d_s3, d_s2, d_s1,
        d_q4_out, d_q3_out, d_q2_out, d_q1_out
    );

    err = cudaDeviceSynchronize();
    cleanup();
    return (int32_t)err;
}

int32_t v3_cuda_c3v2_eval_649(
    const uint64_t* d_masks_649,
    const double* d_q6,
    const double* d_q5,
    const double* d_q4,
    const V3CrowdParamsF64* h_params,
    const uint64_t* h_tickets,
    uint32_t n_tickets,
    V3PortfolioEvalOut* h_out
) {
    if (n_tickets == 0 || n_tickets > 512) return -2;
    int32_t rc = ensure_pmv3_binom_init(); if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init(); if (rc != 0) return rc;

    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    cudaMemcpyToSymbol(c_c3v2_tickets, h_tickets, (size_t)n_tickets * sizeof(uint64_t));

    const uint32_t N = 13983816U;
    const uint32_t gN = (N + 255U) / 256U;
    unsigned long long *d_cnts = nullptr, *d_lo = nullptr, *d_hi = nullptr;
    double* d_m2 = nullptr;
    if (cudaMalloc(&d_cnts, 6 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_lo, 5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_hi, 5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_m2, (size_t)gN * sizeof(double)) != cudaSuccess) {
        if (d_cnts) cudaFree(d_cnts); if (d_lo) cudaFree(d_lo); if (d_hi) cudaFree(d_hi); if (d_m2) cudaFree(d_m2);
        return -3;
    }
    cudaMemset(d_cnts, 0, 6 * sizeof(unsigned long long));
    cudaMemset(d_lo, 0, 5 * sizeof(unsigned long long));
    cudaMemset(d_hi, 0, 5 * sizeof(unsigned long long));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_c3v2_eval_649<<<gN, 256>>>(d_masks_649, d_q6, d_q5, d_q4, n_tickets, d_cnts, d_lo, d_hi, d_m2);
    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    if (err == cudaSuccess && h_out) {
        memset(h_out, 0, sizeof(V3PortfolioEvalOut));
        unsigned long long hc[6] = {0}, hlo[5] = {0}, hhi[5] = {0};
        cudaMemcpy(hc, d_cnts, sizeof(hc), cudaMemcpyDeviceToHost);
        cudaMemcpy(hlo, d_lo, sizeof(hlo), cudaMemcpyDeviceToHost);
        cudaMemcpy(hhi, d_hi, sizeof(hhi), cudaMemcpyDeviceToHost);
        double* hm2 = (double*)malloc((size_t)gN * sizeof(double));
        cudaMemcpy(hm2, d_m2, (size_t)gN * sizeof(double), cudaMemcpyDeviceToHost);
        double m2_sum = 0.0;
        for (uint32_t i = 0; i < gN; ++i) m2_sum += hm2[i];
        free(hm2);

        h_out->win_states = hc[0];
        h_out->pair_overlap_sum = hc[1];
        for (int c = 0; c < 4; ++c) {
            h_out->cat_any_states[c] = hc[2 + c];
            h_out->ev_by_cat[c] = ldexp((double)hhi[c], -32) + ldexp((double)(hlo[c] >> 32), -64) + ldexp((double)(hlo[c] & 0xFFFFFFFFULL), -96);
        }
        h_out->ev_portfolio = ldexp((double)hhi[4], -32) + ldexp((double)(hlo[4] >> 32), -64) + ldexp((double)(hlo[4] & 0xFFFFFFFFULL), -96);
        h_out->second_moment_draw0 = m2_sum;
        h_out->elapsed_ms = (double)ms;
    }
    cudaFree(d_cnts); cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_m2);
    return (int32_t)err;
}

int32_t v3_cuda_c3v2_eval_540(
    const uint64_t* d_draws6_540,
    const double* d_p5,
    const double* d_t5,
    const double* d_qiii,
    const V3CrowdParamsF64* h_params,
    const uint64_t* h_tickets,
    uint32_t n_tickets,
    V3PortfolioEvalOut* h_out
) {
    if (n_tickets == 0 || n_tickets > 512) return -2;
    int32_t rc = ensure_pmv3_binom_init(); if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init(); if (rc != 0) return rc;

    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    cudaMemcpyToSymbol(c_c3v2_tickets, h_tickets, (size_t)n_tickets * sizeof(uint64_t));

    const uint32_t ND = 3838380U;
    const uint32_t gD = (ND + 255U) / 256U;
    unsigned long long *d_cnts = nullptr, *d_lo = nullptr, *d_hi = nullptr;
    double* d_m2 = nullptr;
    if (cudaMalloc(&d_cnts, 5 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_lo, 4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_hi, 4 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_m2, (size_t)gD * sizeof(double)) != cudaSuccess) {
        if (d_cnts) cudaFree(d_cnts); if (d_lo) cudaFree(d_lo); if (d_hi) cudaFree(d_hi); if (d_m2) cudaFree(d_m2);
        return -3;
    }
    cudaMemset(d_cnts, 0, 5 * sizeof(unsigned long long));
    cudaMemset(d_lo, 0, 4 * sizeof(unsigned long long));
    cudaMemset(d_hi, 0, 4 * sizeof(unsigned long long));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_c3v2_eval_540<<<gD, 256>>>(d_draws6_540, d_p5, d_t5, d_qiii, n_tickets, d_cnts, d_lo, d_hi, d_m2);
    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    if (err == cudaSuccess && h_out) {
        memset(h_out, 0, sizeof(V3PortfolioEvalOut));
        unsigned long long hc[5] = {0}, hlo[4] = {0}, hhi[4] = {0};
        cudaMemcpy(hc, d_cnts, sizeof(hc), cudaMemcpyDeviceToHost);
        cudaMemcpy(hlo, d_lo, sizeof(hlo), cudaMemcpyDeviceToHost);
        cudaMemcpy(hhi, d_hi, sizeof(hhi), cudaMemcpyDeviceToHost);
        double* hm2 = (double*)malloc((size_t)gD * sizeof(double));
        cudaMemcpy(hm2, d_m2, (size_t)gD * sizeof(double), cudaMemcpyDeviceToHost);
        double m2_sum = 0.0;
        for (uint32_t i = 0; i < gD; ++i) m2_sum += hm2[i];
        free(hm2);

        h_out->win_states = hc[0];
        h_out->pair_overlap_sum = hc[1];
        for (int c = 0; c < 3; ++c) {
            h_out->cat_any_states[c] = hc[2 + c];
            h_out->ev_by_cat[c] = ldexp((double)hhi[c], -32) + ldexp((double)(hlo[c] >> 32), -64) + ldexp((double)(hlo[c] & 0xFFFFFFFFULL), -96);
        }
        h_out->ev_portfolio = ldexp((double)hhi[3], -32) + ldexp((double)(hlo[3] >> 32), -64) + ldexp((double)(hlo[3] & 0xFFFFFFFFULL), -96);
        h_out->second_moment_draw0 = m2_sum;
        h_out->elapsed_ms = (double)ms;
    }
    cudaFree(d_cnts); cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_m2);
    return (int32_t)err;
}

int32_t v3_cuda_c3v2_eval_joker(
    const uint64_t* d_base_joker,
    const double* d_q5,
    const double* d_q4,
    const double* d_q3,
    const double* d_q2,
    const double* d_q1,
    const V3CrowdParamsF64* h_params,
    const uint64_t* h_tickets_packed,
    uint32_t n_tickets,
    V3PortfolioEvalOut* h_out
) {
    if (n_tickets == 0 || n_tickets > 512) return -2;
    int32_t rc = ensure_pmv3_binom_init(); if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init(); if (rc != 0) return rc;

    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    cudaMemcpyToSymbol(c_c3v2_tickets, h_tickets_packed, (size_t)n_tickets * sizeof(uint64_t));

    const uint32_t NB = 1221759U;
    const uint32_t gB = (NB + 255U) / 256U;
    unsigned long long *d_cnts = nullptr, *d_lo = nullptr, *d_hi = nullptr;
    double* d_m2 = nullptr;
    if (cudaMalloc(&d_cnts, 10 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_lo, 9 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_hi, 9 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_m2, (size_t)gB * sizeof(double)) != cudaSuccess) {
        if (d_cnts) cudaFree(d_cnts); if (d_lo) cudaFree(d_lo); if (d_hi) cudaFree(d_hi); if (d_m2) cudaFree(d_m2);
        return -3;
    }
    cudaMemset(d_cnts, 0, 10 * sizeof(unsigned long long));
    cudaMemset(d_lo, 0, 9 * sizeof(unsigned long long));
    cudaMemset(d_hi, 0, 9 * sizeof(unsigned long long));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_c3v2_eval_joker<<<gB, 256>>>(d_base_joker, d_q5, d_q4, d_q3, d_q2, d_q1, n_tickets, d_cnts, d_lo, d_hi, d_m2);
    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    if (err == cudaSuccess && h_out) {
        memset(h_out, 0, sizeof(V3PortfolioEvalOut));
        unsigned long long hc[10] = {0}, hlo[9] = {0}, hhi[9] = {0};
        cudaMemcpy(hc, d_cnts, sizeof(hc), cudaMemcpyDeviceToHost);
        cudaMemcpy(hlo, d_lo, sizeof(hlo), cudaMemcpyDeviceToHost);
        cudaMemcpy(hhi, d_hi, sizeof(hhi), cudaMemcpyDeviceToHost);
        double* hm2 = (double*)malloc((size_t)gB * sizeof(double));
        cudaMemcpy(hm2, d_m2, (size_t)gB * sizeof(double), cudaMemcpyDeviceToHost);
        double m2_sum = 0.0;
        for (uint32_t i = 0; i < gB; ++i) m2_sum += hm2[i];
        free(hm2);

        h_out->win_states = hc[0];
        h_out->pair_overlap_sum = hc[1];
        for (int c = 0; c < 8; ++c) {
            h_out->cat_any_states[c] = hc[2 + c];
            h_out->ev_by_cat[c] = ldexp((double)hhi[c], -32) + ldexp((double)(hlo[c] >> 32), -64) + ldexp((double)(hlo[c] & 0xFFFFFFFFULL), -96);
        }
        h_out->ev_portfolio = ldexp((double)hhi[8], -32) + ldexp((double)(hlo[8] >> 32), -64) + ldexp((double)(hlo[8] & 0xFFFFFFFFULL), -96);
        h_out->second_moment_draw0 = m2_sum;
        h_out->elapsed_ms = (double)ms;
    }
    cudaFree(d_cnts); cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_m2);
    return (int32_t)err;
}

int32_t v3_cuda_c3v2_scan_universe(
    int32_t game_code, // 0 = 649, 1 = 540, 2 = joker
    const uint64_t* d_tickets,
    const float* d_ev_q05_pair,
    uint32_t n_univ,
    const uint64_t* h_selected,
    uint32_t n_selected,
    const double* h_penalties16,
    double w_ev,
    double w_q05,
    double* h_block_scores_out,
    uint32_t* h_block_indices_out
) {
    if (n_selected > 512) return -2;
    if (n_selected > 0) {
        cudaMemcpyToSymbol(c_c3v2_tickets, h_selected, (size_t)n_selected * sizeof(uint64_t));
    }
    cudaMemcpyToSymbol(c_c3v2_penalties, h_penalties16, 16 * sizeof(double));

    uint32_t blocks = (n_univ + 255U) / 256U;
    double* d_bs = nullptr;
    uint32_t* d_bi = nullptr;
    if (cudaMalloc(&d_bs, (size_t)blocks * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_bi, (size_t)blocks * sizeof(uint32_t)) != cudaSuccess) {
        if (d_bs) cudaFree(d_bs); if (d_bi) cudaFree(d_bi);
        return -3;
    }

    if (game_code == 0) {
        k_c3v2_scan_universe_std<<<blocks, 256>>>(d_tickets, d_ev_q05_pair, n_univ, 6, 7, n_selected, w_ev, w_q05, d_bs, d_bi);
    } else if (game_code == 1) {
        k_c3v2_scan_universe_std<<<blocks, 256>>>(d_tickets, d_ev_q05_pair, n_univ, 5, 5, n_selected, w_ev, w_q05, d_bs, d_bi);
    } else {
        k_c3v2_scan_universe_joker<<<blocks, 256>>>(d_tickets, d_ev_q05_pair, n_univ, n_selected, w_ev, w_q05, d_bs, d_bi);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemcpy(h_block_scores_out, d_bs, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_block_indices_out, d_bi, (size_t)blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_bs);
    cudaFree(d_bi);
    return (int32_t)err;
}

int32_t v3_cuda_c4v2_prep_hits(
    int32_t game_code, // 0=649, 1=540, 2=joker
    const uint64_t* d_draws_or_base,
    const uint64_t* h_all_tickets,
    uint32_t total_tickets,
    const uint32_t* h_port_offs,
    const uint32_t* h_port_lens,
    uint32_t n_ports,
    void* d_hits_out
) {
    uint64_t* d_tk = nullptr;
    uint32_t *d_offs = nullptr, *d_lens = nullptr;
    if (cudaMalloc(&d_tk, (size_t)(total_tickets > 0 ? total_tickets : 1) * sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_offs, (size_t)n_ports * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&d_lens, (size_t)n_ports * sizeof(uint32_t)) != cudaSuccess) {
        if (d_tk) cudaFree(d_tk); if (d_offs) cudaFree(d_offs); if (d_lens) cudaFree(d_lens);
        return -3;
    }
    if (total_tickets > 0) {
        cudaMemcpy(d_tk, h_all_tickets, (size_t)total_tickets * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_offs, h_port_offs, (size_t)n_ports * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lens, h_port_lens, (size_t)n_ports * sizeof(uint32_t), cudaMemcpyHostToDevice);

    if (game_code == 0) {
        k_c4v2_prep_hits_649<<<(13983816U + 255U) / 256U, 256>>>(d_draws_or_base, d_tk, d_offs, d_lens, n_ports, (uint32_t*)d_hits_out);
    } else if (game_code == 1) {
        k_c4v2_prep_hits_540<<<(3838380U + 255U) / 256U, 256>>>(d_draws_or_base, d_tk, d_offs, d_lens, n_ports, (uint32_t*)d_hits_out);
    } else {
        k_c4v2_prep_hits_joker<<<(1221759U + 255U) / 256U, 256>>>(d_draws_or_base, d_tk, d_offs, d_lens, n_ports, (uint64_t*)d_hits_out);
    }
    cudaError_t err = cudaDeviceSynchronize();
    cudaFree(d_tk); cudaFree(d_offs); cudaFree(d_lens);
    return (int32_t)err;
}

int32_t v3_cuda_c4v2_compute_rates(
    int32_t game_code,
    const uint64_t* d_draws6_540,
    const double* d_q0,
    const double* d_q1,
    const double* d_q2,
    const double* d_q3,
    const double* d_q4,
    const void* d_hits,
    const V3CrowdParamsF64* h_params,
    const double* h_xi16,
    uint32_t n_ports,
    double* h_rates_out // n_ports * n_cats * 6
) {
    int32_t rc = ensure_pmv3_binom_init(); if (rc != 0) return rc;
    rc = ensure_c3v2_gl32_init(); if (rc != 0) return rc;

    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    cudaMemcpyToSymbol(c_c4v2_xi16, h_xi16, 16 * sizeof(double));

    uint32_t n_cats = (game_code == 0) ? 4U : ((game_code == 1) ? 3U : 8U);
    uint32_t total_slots = n_ports * n_cats * 12U;
    unsigned long long *d_lo = nullptr, *d_hi = nullptr;
    if (cudaMalloc(&d_lo, (size_t)total_slots * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_hi, (size_t)total_slots * sizeof(unsigned long long)) != cudaSuccess) {
        if (d_lo) cudaFree(d_lo); if (d_hi) cudaFree(d_hi);
        return -3;
    }
    cudaMemset(d_lo, 0, (size_t)total_slots * sizeof(unsigned long long));
    cudaMemset(d_hi, 0, (size_t)total_slots * sizeof(unsigned long long));

    if (game_code == 0) {
        k_c4v2_rates_649<<<(13983816U + 255U) / 256U, 256>>>(d_q0, d_q1, d_q2, (const uint32_t*)d_hits, n_ports, d_lo, d_hi);
    } else if (game_code == 1) {
        k_c4v2_rates_540<<<(3838380U + 255U) / 256U, 256>>>(d_draws6_540, d_q0, d_q1, d_q2, (const uint32_t*)d_hits, n_ports, d_lo, d_hi);
    } else {
        k_c4v2_rates_joker<<<(1221759U + 255U) / 256U, 256>>>(d_q0, d_q1, d_q2, d_q3, d_q4, (const uint64_t*)d_hits, n_ports, d_lo, d_hi);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        unsigned long long* hlo = (unsigned long long*)malloc((size_t)total_slots * sizeof(unsigned long long));
        unsigned long long* hhi = (unsigned long long*)malloc((size_t)total_slots * sizeof(unsigned long long));
        cudaMemcpy(hlo, d_lo, (size_t)total_slots * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(hhi, d_hi, (size_t)total_slots * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < total_slots; ++i) {
            h_rates_out[i] = ldexp((double)hhi[i], -32) + ldexp((double)(hlo[i] >> 32), -64) + ldexp((double)(hlo[i] & 0xFFFFFFFFULL), -96);
        }
        free(hlo); free(hhi);
    }
    cudaFree(d_lo); cudaFree(d_hi);
    return (int32_t)err;
}

int32_t v3_cuda_c4v2_sim_chunk(
    int32_t game_code,
    const uint64_t* d_draws6_540,
    const double* d_q0,
    const double* d_q1,
    const double* d_q2,
    const double* d_q3,
    const double* d_q4,
    const void* d_hits,
    const V3CrowdParamsF64* h_params,
    const double* h_xi16,
    const double* h_rep0_8,
    const double* h_fnew_8,
    uint32_t n_ports,
    uint32_t chunk_start,
    uint32_t chunk_len,
    uint64_t seed,
    double* h_out_pay4,
    uint8_t* h_out_flags,
    double* elapsed_ms_out
) {
    int32_t rc = ensure_pmv3_binom_init(); if (rc != 0) return rc;
    cudaMemcpyToSymbol(c_pmv3_params, h_params, sizeof(V3CrowdParamsF64));
    cudaMemcpyToSymbol(c_c4v2_xi16, h_xi16, 16 * sizeof(double));

    size_t pay_b = (size_t)chunk_len * (size_t)n_ports * 4ULL * sizeof(double);
    size_t flg_b = (size_t)chunk_len * (size_t)n_ports * sizeof(uint8_t);
    double *d_pay = nullptr, *d_rep8 = nullptr, *d_fn8 = nullptr;
    uint8_t* d_flg = nullptr;

    if (cudaMalloc(&d_pay, pay_b) != cudaSuccess ||
        cudaMalloc(&d_flg, flg_b) != cudaSuccess ||
        cudaMalloc(&d_rep8, 8 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_fn8, 8 * sizeof(double)) != cudaSuccess) {
        if (d_pay) cudaFree(d_pay); if (d_flg) cudaFree(d_flg);
        if (d_rep8) cudaFree(d_rep8); if (d_fn8) cudaFree(d_fn8);
        return -3;
    }
    cudaMemcpy(d_rep8, h_rep0_8, 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fn8, h_fnew_8, 8 * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t blocks = (chunk_len + 255U) / 256U;
    if (game_code == 0) {
        k_c4v2_sim_chunk_649<<<blocks, 256>>>(
            d_q0, d_q1, d_q2, (const uint32_t*)d_hits, n_ports, chunk_start, chunk_len, seed,
            h_rep0_8[0], h_rep0_8[1], h_rep0_8[2], h_fnew_8[0], h_fnew_8[1], h_fnew_8[2],
            d_pay, d_flg
        );
    } else if (game_code == 1) {
        k_c4v2_sim_chunk_540<<<blocks, 256>>>(
            d_draws6_540, d_q0, d_q1, d_q2, (const uint32_t*)d_hits, n_ports, chunk_start, chunk_len, seed,
            h_rep0_8[0], h_rep0_8[1], h_rep0_8[2], h_fnew_8[0], h_fnew_8[1], h_fnew_8[2],
            d_pay, d_flg
        );
    } else {
        k_c4v2_sim_chunk_joker<<<blocks, 256>>>(
            d_q0, d_q1, d_q2, d_q3, d_q4, (const uint64_t*)d_hits, n_ports, chunk_start, chunk_len, seed,
            d_rep8, d_fn8, d_pay, d_flg
        );
    }

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        cudaMemcpy(h_out_pay4, d_pay, pay_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_flags, d_flg, flg_b, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_pay); cudaFree(d_flg); cudaFree(d_rep8); cudaFree(d_fn8);
    return (int32_t)err;
}

} // extern "C"

// ============================================================================
// 8. SESIUNEA 4 (S4 / Poarta A5) — Motorul Cauzal de Regimuri Fizice CK-V6
//    (BOCPD Adams-MacKay + Koopman V1 (+) V2 Shrunk + Scut Martingal w0 = 0.50)
//    pe NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell sm_120, IEEE-754 f64)
// ============================================================================

#define CKV6_L_MAX 512
#define CKV6_LAMBDA_0 0.01
#define CKV6_LAMBDA_1 0.004
#define CKV6_W0_SHIELD 0.50
#define CKV6_BETA_DM 0.75
#define CKV6_BETA_V2 0.25
#define CKV6_ALPHA_DM 1.5
#define CKV6_THETA_CAP 0.85
#define CKV6_RHO_MAX 0.90
#define CKV6_TAU_STAT_1 4.0
#define CKV6_TAU_KOOP_1 4.0
#define CKV6_TAU_STAT_2 8.0
#define CKV6_TAU_KOOP_2 8.0
#define CKV6_TAU_BON_STAT 4.0
#define CKV6_TAU_BON_KOOP 4.0

__device__ __forceinline__ int ckv6_pair_index(int u, int v, int n) {
    // 0 <= u < v < n -> u * n - u * (u + 1) / 2 + (v - u - 1)
    return u * n - ((u * (u + 1)) >> 1) + (v - u - 1);
}

__device__ __forceinline__ double ckv6_damp_factor(double b_tot) {
    if (b_tot <= 1.0e-15) return 1.0;
    return (CKV6_RHO_MAX * tanh(b_tot / CKV6_RHO_MAX)) / b_tot;
}

__device__ __host__ __forceinline__ double ckv6_binom_f64(int n, int r) {
    unsigned long long c = 1ULL;
    for (int k = 1; k <= r; ++k) {
        unsigned long long num = (unsigned long long)(n - r + k);
        unsigned long long den = (unsigned long long)k;
        unsigned long long a = num, b = den;
        while (b != 0ULL) {
            unsigned long long t = a % b;
            a = b;
            b = t;
        }
        c = (c / (den / a)) * (num / a);
    }
    return (double)c;
}

__device__ __forceinline__ double ckv6_ball_weight_f64(double theta, double th_cap) {
    double z = th_cap * tanh(theta / th_cap);
    return exp(z);
}

__device__ __forceinline__ void ckv6_kahan_add(double* sum, double* comp, double val) {
    double y = __dsub_rn(val, *comp);
    double t = __dadd_rn(*sum, y);
    *comp = __dsub_rn(__dsub_rn(t, *sum), y);
    *sum = t;
}

// Calculeaza exact polinomul simetric elementar e_{r}(w_0, ..., w_{N-1}) pe toate cele 32 lane-uri
// ale unui warp (100% termeni pozitivi -> precizie < 1e-14, zero shared memory, 5 pasi fluture)
__device__ __forceinline__ double ckv6_warp_elem_sym_f64(
    double w0, bool has0,
    double w1, bool has1,
    double w2, bool has2,
    int r_draw
) {
    double poly[21];
    #pragma unroll
    for (int k = 0; k <= 20; ++k) poly[k] = 0.0;
    poly[0] = 1.0;
    if (has0) {
        poly[1] = w0;
    }
    if (has1) {
        poly[2] = __dmul_rn(poly[1], w1);
        poly[1] = __dadd_rn(poly[1], w1);
    }
    if (has2) {
        poly[3] = __dmul_rn(poly[2], w2);
        poly[2] = __dadd_rn(poly[2], __dmul_rn(poly[1], w2));
        poly[1] = __dadd_rn(poly[1], w2);
    }
    #pragma unroll
    for (int step = 0; step < 5; ++step) {
        int off = 1 << step;
        int max_d_in = (3 << step);
        if (max_d_in > r_draw) max_d_in = r_draw;
        int max_d_out = (6 << step);
        if (max_d_out > r_draw) max_d_out = r_draw;
        double partner[21];
        for (int k = 0; k <= max_d_in; ++k) {
            partner[k] = __shfl_xor_sync(0xFFFFFFFFU, poly[k], off);
        }
        double next_p[21];
        for (int k = 0; k <= max_d_out; ++k) {
            int i_min = (k - max_d_in > 0) ? (k - max_d_in) : 0;
            int i_max = (k < max_d_in) ? k : max_d_in;
            double acc = 0.0;
            for (int i = i_min; i <= i_max; ++i) {
                acc = __dadd_rn(acc, __dmul_rn(poly[i], partner[k - i]));
            }
            next_p[k] = acc;
        }
        for (int k = 0; k <= max_d_out; ++k) poly[k] = next_p[k];
    }
    return poly[r_draw];
}

// Evaluare rapida in L1 __shared__ memory (1 warp = 32 fire per istorie) pentru Loto 6/49 (N=49, r=6, T <= 400)
__device__ void d_ckv6_eval_649_warp(
    uint32_t t_steps,
    const uint64_t* s_draws,
    double* s_R0,
    double* s_R1,
    double* s_pi,
    uint16_t* s_P,
    int32_t* s_U2,
    uint16_t* s_C1,
    int32_t* s_U1_v1,
    int32_t* s_U1_v2,
    double* d_step_e_out,        // optional [t_steps]
    double* d_step_log_m_out,    // optional [t_steps]
    int32_t* d_step_map_r_out,   // optional [t_steps]
    double* log_m_final_out,
    double* max_log_m_out,
    double* min_e_out,
    double* log_m_glob_out,
    uint32_t l_regime_cp,        // 0 daca nu masuram eroare CP
    double* mean_cp_err_out
) {
    const int lane = threadIdx.x & 31;
    // Constante exacte J(49, 6)
    const double d1 = 48.0;
    const double nu1 = 258.0 / 49.0;
    const double d2 = 1127.0;
    const double nu2 = 9030.0 / 376.0;
    const double sqrt_nu2 = sqrt(nu2);
    const int32_t k1_int_tbl[7] = { -36, 13, 62, 111, 160, 209, 258 };
    const int32_t k2_int_tbl[7] = { 150, -250, 102, 1206, 3062, 5670, 9030 };

    __shared__ double sh_scalars[2];
    __shared__ double s_w49[49];
    if (lane == 0) {
        s_R0[0] = 1.0;
        s_R1[0] = 1.0;
    }
    __syncwarp();

    double log_m = 0.0;
    double max_log_m = 0.0;
    double min_e = 1.0e9;
    double lw0 = 0.0;
    double lw1 = 0.0;
    double log_m_glob = 0.0;
    double cp_err_sum = 0.0;
    int cp_err_cnt = 0;

    for (uint32_t t = 0; t < t_steps; ++t) {
        // Curatam acumulatoarele sufixului pe toate cele 32 lane-uri
        for (int idx = lane; idx < 1176; idx += 32) {
            s_P[idx] = 0U;
            s_U2[idx] = 0;
        }
        for (int idx = lane; idx < 49; idx += 32) {
            s_C1[idx] = 0U;
            s_U1_v1[idx] = 0;
            s_U1_v2[idx] = 0;
        }
        if (lane == 0) {
            s_pi[0] = 1.0;
        }
        __syncwarp();

        uint64_t mask_dt = s_draws[t];
        uint64_t mask_dtm1 = (t > 0U) ? s_draws[t - 1U] : 0ULL;
        int b_dt[6];
        {
            uint64_t tmp_dt = mask_dt;
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                int bi = __ffsll((long long)tmp_dt) - 1;
                b_dt[i] = bi;
                tmp_dt ^= (1ULL << bi);
            }
        }

        long long sum_c1_sq = 0LL;
        long long sum_p2 = 0LL;
        long long sum_u1_v1_sq = 0LL;
        long long sum_c1_u1_v1 = 0LL;
        long long sum_u1_v2_sq = 0LL;
        long long sum_c1_u1_v2 = 0LL;
        long long sum_u2_sq = 0LL;
        long long sum_p_u2 = 0LL;
        long long w1_sum = 0LL;
        long long w1_sq_sum = 0LL;
        long long w2_sum = 0LL;
        long long w2_sq_sum = 0LL;
        long long dot_dt_x1 = 0LL;
        long long dot_dt_z1 = 0LL;
        long long dot_dt_x2 = 0LL;
        long long dot_dt_z2 = 0LL;

        int b_prev[6] = {0, 0, 0, 0, 0, 0};
        int p_prev[15];
        int m_dt_prev = 0;

        for (uint32_t ell = 1U; ell <= t; ++ell) {
            uint32_t s_idx = t - ell;
            uint64_t mask_ds = s_draws[s_idx];

            // Extragem cele 6 bile (0..48) si cele 15 perechi ale lui D_s in registri pe toate lane-urile
            int b_cur[6];
            uint64_t tmp = mask_ds;
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                int bi = __ffsll((long long)tmp) - 1;
                b_cur[i] = bi;
                tmp ^= (1ULL << bi);
            }
            int p_cur[15];
            int p_pos = 0;
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                for (int j = i + 1; j < 6; ++j) {
                    p_cur[p_pos++] = ckv6_pair_index(b_cur[i], b_cur[j], 49);
                }
            }

            // 1. Adaugam D_s in C1 si P (toate cele 6 bile sunt distincte, toate cele 15 perechi sunt distincte!)
            long long dc1_sq = 0LL, dc1_u1v1 = 0LL, dc1_u1v2 = 0LL;
            if (lane < 6) {
                int bi = b_cur[lane];
                long long old_c = (long long)s_C1[bi];
                s_C1[bi] = (uint16_t)(old_c + 1LL);
                dc1_sq = 2LL * old_c + 1LL;
                dc1_u1v1 = (long long)s_U1_v1[bi];
                dc1_u1v2 = (long long)s_U1_v2[bi];
            }
            long long dp2 = 0LL, dp_u2 = 0LL;
            if (lane < 15) {
                int pi_idx = p_cur[lane];
                long long old_p = (long long)s_P[pi_idx];
                s_P[pi_idx] = (uint16_t)(old_p + 1LL);
                dp2 = 2LL * old_p + 1LL;
                dp_u2 = (long long)s_U2[pi_idx];
            }
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                dc1_sq += __shfl_down_sync(0xFFFFFFFFU, dc1_sq, off);
                dc1_u1v1 += __shfl_down_sync(0xFFFFFFFFU, dc1_u1v1, off);
                dc1_u1v2 += __shfl_down_sync(0xFFFFFFFFU, dc1_u1v2, off);
                dp2 += __shfl_down_sync(0xFFFFFFFFU, dp2, off);
                dp_u2 += __shfl_down_sync(0xFFFFFFFFU, dp_u2, off);
            }
            __syncwarp();

            int m_dt_cur = __popcll(mask_dt & mask_ds);
            if (lane == 0) {
                sum_c1_sq += dc1_sq;
                sum_c1_u1_v1 += dc1_u1v1;
                sum_c1_u1_v2 += dc1_u1v2;
                sum_p2 += dp2;
                sum_p_u2 += dp_u2;
                dot_dt_x1 += (long long)k1_int_tbl[m_dt_cur];
                dot_dt_x2 += (long long)k2_int_tbl[m_dt_cur];
            }

            // 2. Daca ell >= 2, tranzitia (D_s -> D_{s+1}) intra in operatorul Koopman (D_{s+1} = b_prev / p_prev!)
            if (ell >= 2U) {
                int m_s_tm1 = __popcll(mask_ds & mask_dtm1);
                long long w1_i = (long long)k1_int_tbl[m_s_tm1];
                long long w2_i = (long long)k2_int_tbl[m_s_tm1];

                long long du1_v1_sq = 0LL, dc1_u1v1_add = 0LL;
                long long du1_v2_sq = 0LL, dc1_u1v2_add = 0LL;
                if (lane < 6) {
                    int bj = b_prev[lane];
                    long long old_u1 = (long long)s_U1_v1[bj];
                    s_U1_v1[bj] = (int32_t)(old_u1 + w1_i);
                    du1_v1_sq = 2LL * w1_i * old_u1 + w1_i * w1_i;
                    dc1_u1v1_add = w1_i * (long long)s_C1[bj];

                    long long old_u1_2 = (long long)s_U1_v2[bj];
                    s_U1_v2[bj] = (int32_t)(old_u1_2 + w2_i);
                    du1_v2_sq = 2LL * w2_i * old_u1_2 + w2_i * w2_i;
                    dc1_u1v2_add = w2_i * (long long)s_C1[bj];
                }
                long long du2_sq = 0LL, dp_u2_add = 0LL;
                if (lane < 15) {
                    int pj = p_prev[lane];
                    long long old_u2 = (long long)s_U2[pj];
                    s_U2[pj] = (int32_t)(old_u2 + w2_i);
                    du2_sq = 2LL * w2_i * old_u2 + w2_i * w2_i;
                    dp_u2_add = w2_i * (long long)s_P[pj];
                }
                #pragma unroll
                for (int off = 16; off >= 1; off >>= 1) {
                    du1_v1_sq += __shfl_down_sync(0xFFFFFFFFU, du1_v1_sq, off);
                    dc1_u1v1_add += __shfl_down_sync(0xFFFFFFFFU, dc1_u1v1_add, off);
                    du1_v2_sq += __shfl_down_sync(0xFFFFFFFFU, du1_v2_sq, off);
                    dc1_u1v2_add += __shfl_down_sync(0xFFFFFFFFU, dc1_u1v2_add, off);
                    du2_sq += __shfl_down_sync(0xFFFFFFFFU, du2_sq, off);
                    dp_u2_add += __shfl_down_sync(0xFFFFFFFFU, dp_u2_add, off);
                }
                __syncwarp();

                if (lane == 0) {
                    sum_u1_v1_sq += du1_v1_sq;
                    sum_c1_u1_v1 += dc1_u1v1_add;
                    sum_u1_v2_sq += du1_v2_sq;
                    sum_c1_u1_v2 += dc1_u1v2_add;
                    sum_u2_sq += du2_sq;
                    sum_p_u2 += dp_u2_add;
                    w1_sum += w1_i;
                    w1_sq_sum += w1_i * w1_i;
                    w2_sum += w2_i;
                    w2_sq_sum += w2_i * w2_i;
                    dot_dt_z1 += w1_i * (long long)k1_int_tbl[m_dt_prev];
                    dot_dt_z2 += w2_i * (long long)k2_int_tbl[m_dt_prev];
                }
            }

            // Calculam parametrii canalului 1 (DM + Koopman V1) si canalul 2 (V2 + Koopman V2) pe lane 0,
            // apoi toate cele 32 lane-uri evalueaza exact e_6(w) prin reducerea fluture fara pierdere de precizie!
            double eta_stat1 = 0.0;
            double scale_koop1 = 0.0;
            double w1_mean_shift = 0.0;
            double pi_v2 = 1.0;
            double d_ell = (double)ell;
            if (lane == 0) {
                double lr = 6.0 * d_ell;
                double e_x1 = __dsub_rn((double)sum_c1_sq, (lr * lr) / 49.0);
                if (e_x1 < 0.0) e_x1 = 0.0;
                double v_stat1 = __dmul_rn(d_ell, nu1);
                if (e_x1 > v_stat1) {
                    eta_stat1 = __dmul_rn(__dsub_rn(1.0, v_stat1 / e_x1), d_ell / __dadd_rn(d_ell, CKV6_TAU_STAT_1));
                }

                if (ell >= 2U) {
                    double m_tr = (double)(ell - 1U);
                    double w1_real = (double)w1_sum / 49.0;
                    w1_mean_shift = __dmul_rn(6.0, w1_real) / 49.0;
                    double u1_raw_sq = (double)sum_u1_v1_sq / (49.0 * 49.0);
                    double mw1 = __dmul_rn(6.0, w1_real);
                    double e_z1 = __dsub_rn(u1_raw_sq, __dmul_rn(mw1, mw1) / 49.0);
                    if (e_z1 < 0.0) e_z1 = 0.0;
                    double v_koop1 = __dmul_rn(nu1, (double)w1_sq_sum / (49.0 * 49.0));
                    if (e_z1 > v_koop1) {
                        double g_koop1 = __dmul_rn(
                            __dsub_rn(1.0, v_koop1 / e_z1),
                            d1 / __dmul_rn(nu1, __dadd_rn(m_tr, CKV6_TAU_KOOP_1))
                        );
                        scale_koop1 = __dmul_rn(d1 / nu1, g_koop1);
                    }
                }

                // --- V2 ---
                double e_x2 = __dsub_rn(
                    __dsub_rn(2.0 * (double)sum_p2, (30.0 * d_ell) * (30.0 * d_ell) / (49.0 * 48.0)),
                    __dmul_rn(50.0 / 47.0, e_x1)
                );
                if (e_x2 < 0.0) e_x2 = 0.0;
                double v_stat2 = __dmul_rn(d_ell, nu2);
                double g_stat2 = (e_x2 > v_stat2)
                    ? (__dsub_rn(1.0, v_stat2 / e_x2) / __dadd_rn(d_ell, CKV6_TAU_STAT_2))
                    : 0.0;

                double g_koop2 = 0.0;
                double e_z2 = 0.0;
                double cross_xz2 = 0.0;
                if (ell >= 2U) {
                    double m_tr = (double)(ell - 1U);
                    const double inv_s2 = 1.0 / 376.0;
                    const double inv_s2_sq = __dmul_rn(inv_s2, inv_s2);
                    double w2_real = __dmul_rn((double)w2_sum, inv_s2);
                    double mw2_6 = __dmul_rn(6.0, w2_real);
                    double mw2_30 = __dmul_rn(30.0, w2_real);
                    double u1v2_centered_sq = __dsub_rn(
                        __dmul_rn((double)sum_u1_v2_sq, inv_s2_sq),
                        __dmul_rn(mw2_6, mw2_6) / 49.0
                    );
                    e_z2 = __dsub_rn(
                        __dsub_rn(
                            __dmul_rn(2.0, __dmul_rn((double)sum_u2_sq, inv_s2_sq)),
                            __dmul_rn(mw2_30, mw2_30) / (49.0 * 48.0)
                        ),
                        __dmul_rn(50.0 / 47.0, u1v2_centered_sq)
                    );
                    if (e_z2 < 0.0) e_z2 = 0.0;
                    double v_koop2 = __dmul_rn(nu2, __dmul_rn((double)w2_sq_sum, inv_s2_sq));
                    if (e_z2 > v_koop2) {
                        g_koop2 = __dmul_rn(
                            __dsub_rn(1.0, v_koop2 / e_z2),
                            d2 / __dmul_rn(nu2, __dadd_rn(m_tr, CKV6_TAU_KOOP_2))
                        );
                    }
                    double cross_c1_u1v2 = __dsub_rn(
                        __dmul_rn((double)sum_c1_u1_v2, inv_s2),
                        __dmul_rn(__dmul_rn(lr, 6.0), w2_real) / 49.0
                    );
                    cross_xz2 = __dsub_rn(
                        __dsub_rn(
                            __dmul_rn(2.0, __dmul_rn((double)sum_p_u2, inv_s2)),
                            __dmul_rn(900.0 * d_ell, w2_real) / (49.0 * 48.0)
                        ),
                        __dmul_rn(50.0 / 47.0, cross_c1_u1v2)
                    );
                }
                double m2_norm2 = __dadd_rn(
                    __dadd_rn(
                        __dmul_rn(__dmul_rn(g_stat2, g_stat2), e_x2),
                        __dmul_rn(__dmul_rn(g_koop2, g_koop2), e_z2)
                    ),
                    __dmul_rn(__dmul_rn(2.0 * g_stat2, g_koop2), cross_xz2)
                );
                if (m2_norm2 < 0.0) m2_norm2 = 0.0;
                double b2 = __dmul_rn(d2 / sqrt_nu2, sqrt(m2_norm2));
                double s2_raw = __dmul_rn(
                    d2 / nu2,
                    __dadd_rn(
                        __dmul_rn(g_stat2, (double)dot_dt_x2 / 376.0),
                        __dmul_rn(g_koop2, (double)dot_dt_z2 / (376.0 * 376.0))
                    )
                );
                double c_v2 = ckv6_damp_factor(b2);
                pi_v2 = __dadd_rn(1.0, __dmul_rn(c_v2, s2_raw));
                if (pi_v2 < 0.05) pi_v2 = 0.05;
            }

            eta_stat1 = __shfl_sync(0xFFFFFFFFU, eta_stat1, 0);
            scale_koop1 = __shfl_sync(0xFFFFFFFFU, scale_koop1, 0);
            w1_mean_shift = __shfl_sync(0xFFFFFFFFU, w1_mean_shift, 0);

            double lam1 = 1.0;
            if (eta_stat1 > 0.0 || scale_koop1 > 0.0) {
                double base_c0 = __dadd_rn(__dmul_rn(d_ell, 6.0 / 49.0), CKV6_ALPHA_DM);
                int b0 = lane;
                double z1_0 = __dsub_rn((double)s_U1_v1[b0] / 49.0, w1_mean_shift);
                double th0 = __dadd_rn(
                    __dmul_rn(eta_stat1, log(__dadd_rn((double)s_C1[b0], CKV6_ALPHA_DM) / base_c0)),
                    __dmul_rn(scale_koop1, z1_0)
                );
                double w0 = ckv6_ball_weight_f64(th0, CKV6_THETA_CAP);
                s_w49[b0] = w0;

                bool has1 = (lane + 32 < 49);
                double w1 = 1.0;
                if (has1) {
                    int b1 = lane + 32;
                    double z1_1 = __dsub_rn((double)s_U1_v1[b1] / 49.0, w1_mean_shift);
                    double th1 = __dadd_rn(
                        __dmul_rn(eta_stat1, log(__dadd_rn((double)s_C1[b1], CKV6_ALPHA_DM) / base_c0)),
                        __dmul_rn(scale_koop1, z1_1)
                    );
                    w1 = ckv6_ball_weight_f64(th1, CKV6_THETA_CAP);
                    s_w49[b1] = w1;
                }
                __syncwarp();
                double er = ckv6_warp_elem_sym_f64(w0, true, w1, has1, 1.0, false, 6);
                if (lane == 0) {
                    double prod_dt = __dmul_rn(
                        __dmul_rn(
                            __dmul_rn(
                                __dmul_rn(__dmul_rn(s_w49[b_dt[0]], s_w49[b_dt[1]]), s_w49[b_dt[2]]),
                                s_w49[b_dt[3]]
                            ),
                            s_w49[b_dt[4]]
                        ),
                        s_w49[b_dt[5]]
                    );
                    lam1 = __dmul_rn(13983816.0, prod_dt) / er;
                }
            }
            if (lane == 0) {
                s_pi[ell] = __dadd_rn(__dmul_rn(CKV6_BETA_DM, lam1), __dmul_rn(CKV6_BETA_V2, pi_v2));
            }

            #pragma unroll
            for (int i = 0; i < 6; ++i) b_prev[i] = b_cur[i];
            #pragma unroll
            for (int i = 0; i < 15; ++i) p_prev[i] = p_cur[i];
            m_dt_prev = m_dt_cur;
            __syncwarp();
        }

        // Actualizare Adams-MacKay BOCPD + Scut Martingal w0 = 0.50 pe lane 0 (ordine determinista 0..t)
        if (lane == 0) {
            double L0 = 0.0;
            double L1 = 0.0;
            for (uint32_t ell = 0U; ell <= t; ++ell) {
                double pv = s_pi[ell];
                L0 = __dadd_rn(L0, __dmul_rn(s_R0[ell], pv));
                L1 = __dadd_rn(L1, __dmul_rn(s_R1[ell], pv));
            }
            double lw_max = (lw0 >= lw1) ? lw0 : lw1;
            double a0 = exp(__dsub_rn(lw0, lw_max));
            double a1 = exp(__dsub_rn(lw1, lw_max));
            double eta0 = a0 / __dadd_rn(a0, a1);
            double eta1 = a1 / __dadd_rn(a0, a1);

            double l_bocpd = __dadd_rn(__dmul_rn(eta0, L0), __dmul_rn(eta1, L1));
            double kappa = 1.0;
            if (log_m < 0.0) {
                double m_prev = exp(log_m);
                kappa = __dsub_rn(m_prev, 0.50) / __dmul_rn(0.50, m_prev);
                if (kappa < 0.0) kappa = 0.0;
                if (kappa > 1.0) kappa = 1.0;
            }
            double e_step = __dadd_rn(1.0, __dmul_rn(__dmul_rn(0.50, kappa), __dsub_rn(l_bocpd, 1.0)));
            if (e_step < 0.50) e_step = 0.50;
            log_m = __dadd_rn(log_m, log(e_step));
            if (log_m > max_log_m) max_log_m = log_m;
            if (e_step < min_e) min_e = e_step;

            lw0 = __dadd_rn(lw0, log(L0));
            lw1 = __dadd_rn(lw1, log(L1));

            // Comparator global fara resetare (lambda_H = 0, ell = t)
            double l_glob = (t == 0U) ? 1.0 : s_pi[t];
            double kappa_glob = 1.0;
            if (log_m_glob < 0.0) {
                double mg = exp(log_m_glob);
                kappa_glob = __dsub_rn(mg, 0.50) / __dmul_rn(0.50, mg);
                if (kappa_glob < 0.0) kappa_glob = 0.0;
                if (kappa_glob > 1.0) kappa_glob = 1.0;
            }
            double e_glob = __dadd_rn(1.0, __dmul_rn(__dmul_rn(0.50, kappa_glob), __dsub_rn(l_glob, 1.0)));
            if (e_glob < 0.50) e_glob = 0.50;
            log_m_glob = __dadd_rn(log_m_glob, log(e_glob));

            // Posterior filtrat pentru MAP run-length r_t dupa observarea D_t
            double lw_max_post = (lw0 >= lw1) ? lw0 : lw1;
            double ap0 = exp(lw0 - lw_max_post);
            double ap1 = exp(lw1 - lw_max_post);
            double ep0 = ap0 / (ap0 + ap1);
            double ep1 = ap1 / (ap0 + ap1);

            int best_ell = 0;
            double best_post = -1.0;
            for (uint32_t ell = 0U; ell <= t; ++ell) {
                double pv = s_pi[ell];
                double p_ell = ep0 * (s_R0[ell] * pv / L0) + ep1 * (s_R1[ell] * pv / L1);
                if (p_ell > best_post) {
                    best_post = p_ell;
                    best_ell = (int)ell;
                }
            }

            if (d_step_e_out) d_step_e_out[t] = e_step;
            if (d_step_log_m_out) d_step_log_m_out[t] = log_m;
            if (d_step_map_r_out) d_step_map_r_out[t] = best_ell;

            // Daca masuram localizarea punctelor de schimbare periodice tau_c = c * l_regime_cp
            if (l_regime_cp > 0U && l_regime_cp < t_steps) {
                uint32_t probe_off = (l_regime_cp > 30U) ? 30U : (l_regime_cp - 1U);
                if (t >= l_regime_cp && ((t % l_regime_cp) == probe_off)) {
                    int true_cp = (int)(t - probe_off);
                    int hat_cp = (int)t - best_ell;
                    cp_err_sum += fabs((double)(hat_cp - true_cp));
                    cp_err_cnt += 1;
                }
            }

            sh_scalars[0] = L0;
            sh_scalars[1] = L1;
        }
        __syncwarp();

        // Tranzitie Adams-MacKay t -> t+1 (in-place descrescator pe lane 0, apoi vizibil pe tot warpul)
        if (lane == 0) {
            double inv_L0 = (1.0 - CKV6_LAMBDA_0) / sh_scalars[0];
            double inv_L1 = (1.0 - CKV6_LAMBDA_1) / sh_scalars[1];
            for (int ell = (int)t; ell >= 0; --ell) {
                double pv = s_pi[ell];
                s_R0[ell + 1] = s_R0[ell] * pv * inv_L0;
                s_R1[ell + 1] = s_R1[ell] * pv * inv_L1;
            }
            s_R0[0] = CKV6_LAMBDA_0;
            s_R1[0] = CKV6_LAMBDA_1;
        }
        __syncwarp();
    }

    if (lane == 0) {
        *log_m_final_out = log_m;
        *max_log_m_out = max_log_m;
        *min_e_out = min_e;
        *log_m_glob_out = log_m_glob;
        *mean_cp_err_out = (cp_err_cnt > 0) ? (cp_err_sum / (double)cp_err_cnt) : 0.0;
    }
}

__global__ void k_ckv6_649_batch_f64(
    const uint64_t* __restrict__ d_draws_batch, // [n_traj * t_steps]
    uint32_t n_traj,
    uint32_t t_steps,
    uint32_t l_regime_cp,
    uint32_t record_steps,
    double* __restrict__ d_summary4_out,        // [n_traj * 5]: log_m, max_log_m, min_e, log_m_glob, cp_err
    double* __restrict__ d_step_e_out,          // optional [n_traj * t_steps]
    double* __restrict__ d_step_log_m_out,      // optional [n_traj * t_steps]
    int32_t* __restrict__ d_step_map_r_out      // optional [n_traj * t_steps]
) {
    uint32_t traj = blockIdx.x;
    if (traj >= n_traj) return;

    __shared__ uint64_t s_draws[400];
    __shared__ double s_R0[402];
    __shared__ double s_R1[402];
    __shared__ double s_pi[402];
    __shared__ uint16_t s_P[1176];
    __shared__ int32_t s_U2[1176];
    __shared__ uint16_t s_C1[49];
    __shared__ int32_t s_U1_v1[49];
    __shared__ int32_t s_U1_v2[49];

    const int lane = threadIdx.x & 31;
    const uint64_t* src = d_draws_batch + (size_t)traj * (size_t)t_steps;
    for (uint32_t s = lane; s < t_steps; s += 32U) {
        s_draws[s] = src[s];
    }
    __syncwarp();

    double log_m = 0.0, max_lm = 0.0, min_e = 1.0, log_mg = 0.0, cp_err = 0.0;
    size_t step_off = (size_t)traj * (size_t)t_steps;
    d_ckv6_eval_649_warp(
        t_steps,
        s_draws,
        s_R0,
        s_R1,
        s_pi,
        s_P,
        s_U2,
        s_C1,
        s_U1_v1,
        s_U1_v2,
        record_steps ? (d_step_e_out + step_off) : nullptr,
        record_steps ? (d_step_log_m_out + step_off) : nullptr,
        record_steps ? (d_step_map_r_out + step_off) : nullptr,
        &log_m,
        &max_lm,
        &min_e,
        &log_mg,
        l_regime_cp,
        &cp_err
    );
    if (lane == 0) {
        size_t o5 = (size_t)traj * 5ULL;
        d_summary4_out[o5 + 0ULL] = log_m;
        d_summary4_out[o5 + 1ULL] = max_lm;
        d_summary4_out[o5 + 2ULL] = min_e;
        d_summary4_out[o5 + 3ULL] = log_mg;
        d_summary4_out[o5 + 4ULL] = cp_err;
    }
}

// Generator determinist 100% pe intregi (Philox-4x32-10) pentru istoriile de putere CK-V6 (N=49, r=6, T=400)
__device__ void d_ckv6_gen_synth_649_history(
    uint64_t seed,
    uint32_t cell_id,
    uint32_t rep_id,
    uint32_t delta_bp,     // 0, 1500, 2500, 3500, 5000
    uint32_t l_regime,     // 40, 80, 120, 200, 400
    uint32_t t_steps,
    uint64_t* s_draws_out
) {
    uint64_t active_mask = 0ULL;
    uint64_t prev_active_mask = 0ULL;
    uint32_t cur_reg = 0xFFFFFFFFU;
    uint64_t prev_draw = 0ULL;

    for (uint32_t t = 0U; t < t_steps; ++t) {
        uint32_t reg = t / l_regime;
        if (reg != cur_reg) {
            cur_reg = reg;
            PhiloxStream rng_reg;
            philox_init(&rng_reg, seed, 64900U + cell_id, rep_id, 10000U + reg);
            // Alegem 16 bile active disjuncte fata de regimul imediat anterior (schimbare fizica de set!)
            uint8_t pool[49];
            uint32_t plen = 0U;
            for (uint32_t b = 0U; b < 49U; ++b) {
                if (((prev_active_mask >> b) & 1ULL) == 0ULL) {
                    pool[plen++] = (uint8_t)b;
                }
            }
            uint64_t new_active = 0ULL;
            for (uint32_t k = 0U; k < 16U; ++k) {
                uint32_t pick = k + philox_uniform_u32(&rng_reg, plen - k);
                uint8_t chosen = pool[pick];
                pool[pick] = pool[k];
                new_active |= (1ULL << chosen);
            }
            prev_active_mask = active_mask;
            active_mask = new_active;
        }

        PhiloxStream rng_step;
        philox_init(&rng_step, seed, 74900U + cell_id, rep_id, t);

        uint32_t w[49];
        uint32_t total_w = 0U;
        for (uint32_t b = 0U; b < 49U; ++b) {
            uint32_t wb = 100000U;
            if (delta_bp > 0U) {
                uint64_t bit = 1ULL << b;
                if (active_mask & bit) {
                    wb = 100000U + 22U * delta_bp;
                    if ((prev_draw & bit) != 0ULL) {
                        wb += 12U * delta_bp;
                    }
                } else {
                    wb = 100000U - 10U * delta_bp;
                }
            }
            w[b] = wb;
            total_w += wb;
        }

        uint64_t dmask = 0ULL;
        for (int pick = 0; pick < 6; ++pick) {
            uint32_t target = philox_uniform_u32(&rng_step, total_w);
            uint32_t acc = 0U;
            int chosen = 0;
            for (int b = 0; b < 49; ++b) {
                uint32_t wb = w[b];
                if (wb > 0U) {
                    if (target < acc + wb) {
                        chosen = b;
                        break;
                    }
                    acc += wb;
                }
            }
            dmask |= (1ULL << chosen);
            total_w -= w[chosen];
            w[chosen] = 0U;
        }
        s_draws_out[t] = dmask;
        prev_draw = dmask;
    }
}

__global__ void k_ckv6_power_10k_f64(
    const uint32_t* __restrict__ d_delta_bp_cells, // [n_cells]
    const uint32_t* __restrict__ d_l_reg_cells,    // [n_cells]
    uint32_t n_cells,
    uint32_t reps_per_cell,
    uint32_t t_steps,
    uint64_t seed,
    double* __restrict__ d_records8_out            // [n_cells * reps_per_cell * 8]
) {
    uint32_t g_idx = blockIdx.x;
    uint32_t total_hist = n_cells * reps_per_cell;
    if (g_idx >= total_hist) return;

    uint32_t cell_id = g_idx / reps_per_cell;
    uint32_t rep_id = g_idx - cell_id * reps_per_cell;
    uint32_t delta_bp = d_delta_bp_cells[cell_id];
    uint32_t l_reg = d_l_reg_cells[cell_id];

    __shared__ uint64_t s_draws[400];
    __shared__ double s_R0[402];
    __shared__ double s_R1[402];
    __shared__ double s_pi[402];
    __shared__ uint16_t s_P[1176];
    __shared__ int32_t s_U2[1176];
    __shared__ uint16_t s_C1[49];
    __shared__ int32_t s_U1_v1[49];
    __shared__ int32_t s_U1_v2[49];

    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        d_ckv6_gen_synth_649_history(seed, cell_id, rep_id, delta_bp, l_reg, t_steps, s_draws);
    }
    __syncwarp();

    double log_m = 0.0, max_lm = 0.0, min_e = 1.0, log_mg = 0.0, cp_err = 0.0;
    uint32_t measure_cp = (delta_bp > 0U && l_reg < t_steps) ? l_reg : 0U;
    d_ckv6_eval_649_warp(
        t_steps,
        s_draws,
        s_R0,
        s_R1,
        s_pi,
        s_P,
        s_U2,
        s_C1,
        s_U1_v1,
        s_U1_v2,
        nullptr,
        nullptr,
        nullptr,
        &log_m,
        &max_lm,
        &min_e,
        &log_mg,
        measure_cp,
        &cp_err
    );

    if (lane == 0) {
        size_t o8 = (size_t)g_idx * 8ULL;
        d_records8_out[o8 + 0ULL] = (double)delta_bp / 10000.0;
        d_records8_out[o8 + 1ULL] = (double)l_reg;
        d_records8_out[o8 + 2ULL] = (double)rep_id;
        d_records8_out[o8 + 3ULL] = log_m;
        d_records8_out[o8 + 4ULL] = max_lm;
        d_records8_out[o8 + 5ULL] = min_e;
        d_records8_out[o8 + 6ULL] = cp_err;
        d_records8_out[o8 + 7ULL] = log_mg;
    }
}

// Evaluare generala CK-V6 pe CUDA sm_120 pentru ORICE joc (N <= 90, r <= 20, M_bonus <= 32, pasi canonici m_s >= 1)
struct V3Ckv6StreamMeta {
    uint32_t n_balls;
    uint32_t r_draw;
    uint32_t m_bonus;
    uint32_t n_steps;
    uint32_t step_off_base; // offset in d_step_draw_offs (lungime n_steps + 1)
    uint32_t out_step_base; // offset in vectorii de iesire per pas
};

struct V3Ckv6WorkSlice {
    double P[4005];
    double U2[4005];
    double C1[90];
    double U1_v1[90];
    double U1_v2[90];
    double C_bon[32];
    double U_bon[32];
    double R0[CKV6_L_MAX + 2];
    double R1[CKV6_L_MAX + 2];
    double pi_buf[CKV6_L_MAX + 2];
};

__device__ __host__ __forceinline__ long long ckv6_k2_num(int m, int n, int r) {
    long long lm = (long long)m;
    long long lrm = (long long)(r - m);
    long long lnr = (long long)(n - r);
    long long lr = (long long)r;
    return lm * (lm - 1LL) * lnr * (lnr - 1LL)
         - 2LL * lm * lrm * (lr - 1LL) * (lnr - 1LL)
         + lrm * (lrm - 1LL) * lr * (lr - 1LL);
}

__device__ __host__ __forceinline__ long long ckv6_k2_gcd_div(int n, int r) {
    unsigned long long g = (unsigned long long)((n - 1) * (n - 2));
    for (int m = 0; m <= r; ++m) {
        long long v = ckv6_k2_num(m, n, r);
        unsigned long long uv = (v < 0LL) ? (unsigned long long)(-v) : (unsigned long long)v;
        while (uv != 0ULL) {
            unsigned long long t = g % uv;
            g = uv;
            uv = t;
        }
    }
    return (long long)((g > 0ULL) ? g : 1ULL);
}

__device__ __forceinline__ double ckv6_k2_val(int m, int n, int r, double denom2) {
    return (double)ckv6_k2_num(m, n, r) / denom2;
}

__global__ void k_ckv6_general_streams_f64(
    const V3Ckv6StreamMeta* __restrict__ d_metas,
    uint32_t n_streams,
    const uint32_t* __restrict__ d_step_draw_offs,
    const uint64_t* __restrict__ d_draws_lo,
    const uint32_t* __restrict__ d_draws_hi,
    const uint8_t* __restrict__ d_draws_bonus,
    V3Ckv6WorkSlice* __restrict__ d_work,
    double* __restrict__ d_out_step_e,
    double* __restrict__ d_out_step_log_m,
    int32_t* __restrict__ d_out_step_map_r,
    double* __restrict__ d_out_step_cp_prob,
    double* __restrict__ d_out_step_log_m_glob
) {
    uint32_t sid = blockIdx.x;
    if (sid >= n_streams) return;
    const int lane = threadIdx.x & 31;

    __shared__ double s_w96[96];
    __shared__ int dt_balls_sh[8][20];

    V3Ckv6StreamMeta meta = d_metas[sid];
    int N = (int)meta.n_balls;
    int r = (int)meta.r_draw;
    int MB = (int)meta.m_bonus;
    uint32_t T = meta.n_steps;
    int n_pairs = (N * (N - 1)) / 2;
    bool has_v2 = (N >= 4) && (r >= 2) && (r <= N - 2);
    bool has_bon = (MB >= 2);

    double dN = (double)N;
    double dr = (double)r;
    double th_cap = (r > 6) ? (CKV6_THETA_CAP * (6.0 / dr)) : CKV6_THETA_CAP;
    double comb_nr = ckv6_binom_f64(N, r);
    double d1 = (double)(N - 1);
    double p1 = dr / dN;
    double nu1 = (double)(r * (N - r)) / dN;

    long long k2_g = has_v2 ? ckv6_k2_gcd_div(N, r) : 1LL;
    double d2 = has_v2 ? ((double)(N * (N - 3)) * 0.5) : 1.0;
    double denom2 = has_v2 ? (double)((N - 1) * (N - 2)) : 1.0;
    double s2_scale = has_v2 ? (double)(((long long)((N - 1) * (N - 2))) / k2_g) : 1.0;
    double inv_s2 = 1.0 / s2_scale;
    double inv_s2_sq = inv_s2 * inv_s2;
    double nu2 = has_v2 ? (((double)(r * (r - 1)) * (double)((N - r) * (N - r - 1))) / denom2) : 1.0;
    double sqrt_nu2 = sqrt(nu2);
    double v1_proj_coef = has_v2 ? ((2.0 * (double)((r - 1) * (r - 1))) / (double)(N - 2)) : 0.0;

    double dMB = has_bon ? (double)MB : 1.0;
    double d1_b = has_bon ? (double)(MB - 1) : 1.0;
    double nu1_b = has_bon ? ((double)(MB - 1) / dMB) : 1.0;
    double sqrt_nu1_b = sqrt(nu1_b);
    double inv_mb = has_bon ? (1.0 / dMB) : 0.0;
    double inv_mb_sq = inv_mb * inv_mb;

    V3Ckv6WorkSlice* ws = &d_work[sid];
    if (lane == 0) {
        ws->R0[0] = 1.0;
        ws->R1[0] = 1.0;
    }
    __syncwarp();

    double log_m = 0.0;
    double lw0 = 0.0;
    double lw1 = 0.0;
    double log_m_glob = 0.0;

    const uint32_t* s_offs = d_step_draw_offs + meta.step_off_base;

    for (uint32_t t = 0U; t < T; ++t) {
        uint32_t l_act = (t < (uint32_t)CKV6_L_MAX) ? t : (uint32_t)CKV6_L_MAX;

        for (int i = lane; i < n_pairs; i += 32) {
            ws->P[i] = 0.0;
            ws->U2[i] = 0.0;
        }
        for (int i = lane; i < N; i += 32) {
            ws->C1[i] = 0.0;
            ws->U1_v1[i] = 0.0;
            ws->U1_v2[i] = 0.0;
        }
        if (has_bon) {
            for (int i = lane; i < MB; i += 32) {
                ws->C_bon[i] = 0.0;
                ws->U_bon[i] = 0.0;
            }
        }
        uint32_t dt_beg = s_offs[t];
        uint32_t dt_end = s_offs[t + 1U];
        uint32_t m_dt = dt_end - dt_beg;
        uint32_t m_dt_cap = (m_dt <= 8U) ? m_dt : 8U;
        double inv_m_dt = 1.0 / (double)m_dt;

        if (lane == 0) {
            ws->pi_buf[0] = 1.0;
            for (uint32_t a = 0U; a < m_dt_cap; ++a) {
                uint64_t a_lo = d_draws_lo[dt_beg + a];
                uint32_t a_hi = d_draws_hi[dt_beg + a];
                int nb_a = 0;
                while (a_lo != 0ULL) {
                    int b = __ffsll((long long)a_lo) - 1;
                    dt_balls_sh[a][nb_a++] = b;
                    a_lo ^= (1ULL << b);
                }
                while (a_hi != 0U) {
                    int b = __ffs((int)a_hi) - 1;
                    dt_balls_sh[a][nb_a++] = 64 + b;
                    a_hi ^= (1U << b);
                }
            }
        }
        __syncwarp();

        uint32_t dtm1_beg = (t > 0U) ? s_offs[t - 1U] : 0U;
        uint32_t dtm1_end = (t > 0U) ? s_offs[t] : 0U;
        double inv_m_dtm1 = (t > 0U) ? (1.0 / (double)(dtm1_end - dtm1_beg)) : 0.0;

        double n_eff_sum = 0.0;
        double sum_c1_sq = 0.0, comp_c1_sq = 0.0;
        double sum_p2 = 0.0, comp_p2 = 0.0;
        double sum_u1_v1_sq = 0.0, comp_u1_v1_sq = 0.0;
        double sum_u1_v2_sq = 0.0, comp_u1_v2_sq = 0.0;
        double sum_c1_u1_v2 = 0.0, comp_c1_u1_v2 = 0.0;
        double sum_u2_sq = 0.0, comp_u2_sq = 0.0;
        double sum_p_u2 = 0.0, comp_p_u2 = 0.0;
        double w1_sum = 0.0;
        double w1_sq_eff_sum = 0.0;
        double w2_sum = 0.0;
        double w2_sq_eff_sum = 0.0;

        double dot_a_x2[8] = {0.0}, dot_a_z2[8] = {0.0};
        double dot_a_xb[8] = {0.0}, dot_a_zb[8] = {0.0};
        double prev_a_k2[8] = {0.0}, prev_a_kb[8] = {0.0};
        double cur_a_k2[8] = {0.0}, cur_a_kb[8] = {0.0};

        double sum_cb_sq = 0.0, comp_cb_sq = 0.0;
        double sum_ub_sq = 0.0, comp_ub_sq = 0.0;
        double sum_cb_ub = 0.0, comp_cb_ub = 0.0;
        double wb_sum = 0.0;
        double wb_sq_eff_sum = 0.0;

        uint32_t prev_beg = 0U, prev_end = 0U;
        double prev_inv_m = 1.0;

        for (uint32_t ell = 1U; ell <= l_act; ++ell) {
            uint32_t s_idx = t - ell;
            uint32_t ds_beg = s_offs[s_idx];
            uint32_t ds_end = s_offs[s_idx + 1U];
            double inv_m_ds = 1.0 / (double)(ds_end - ds_beg);
            double d_ell = (double)ell;

            double eta_stat1 = 0.0;
            double scale_koop1 = 0.0;
            double w1_mean_shift = 0.0;
            double g_stat2 = 0.0, g_koop2 = 0.0, c_v2 = 1.0;
            double g_stat_b = 0.0, g_koop_b = 0.0, c_bon = 1.0;

            if (lane == 0) {
                n_eff_sum += inv_m_ds;

                // 1. Adaugam pasul s_idx in C1, P, C_bon (cu acumulare Kahan f64)
                for (uint32_t d_ptr = ds_beg; d_ptr < ds_end; ++d_ptr) {
                    uint64_t lo = d_draws_lo[d_ptr];
                    uint32_t hi = d_draws_hi[d_ptr];
                    int balls[20];
                    int nb = 0;
                    uint64_t t_lo = lo;
                    while (t_lo != 0ULL) {
                        int b = __ffsll((long long)t_lo) - 1;
                        balls[nb++] = b;
                        t_lo ^= (1ULL << b);
                    }
                    uint32_t t_hi = hi;
                    while (t_hi != 0U) {
                        int b = __ffs((int)t_hi) - 1;
                        balls[nb++] = 64 + b;
                        t_hi ^= (1U << b);
                    }
                    for (int i = 0; i < nb; ++i) {
                        int bi = balls[i];
                        double old_c = ws->C1[bi];
                        ws->C1[bi] = __dadd_rn(old_c, inv_m_ds);
                        ckv6_kahan_add(&sum_c1_sq, &comp_c1_sq, __dmul_rn(__dadd_rn(2.0 * old_c, inv_m_ds), inv_m_ds));
                        ckv6_kahan_add(&sum_c1_u1_v2, &comp_c1_u1_v2, __dmul_rn(inv_m_ds, ws->U1_v2[bi]));
                    }
                    if (has_v2) {
                        for (int i = 0; i < nb; ++i) {
                            for (int j = i + 1; j < nb; ++j) {
                                int pidx = ckv6_pair_index(balls[i], balls[j], N);
                                double old_p = ws->P[pidx];
                                ws->P[pidx] = __dadd_rn(old_p, inv_m_ds);
                                ckv6_kahan_add(&sum_p2, &comp_p2, __dmul_rn(__dadd_rn(2.0 * old_p, inv_m_ds), inv_m_ds));
                                ckv6_kahan_add(&sum_p_u2, &comp_p_u2, __dmul_rn(inv_m_ds, ws->U2[pidx]));
                            }
                        }
                    }
                    if (has_bon) {
                        int jb = (int)d_draws_bonus[d_ptr];
                        double old_cb = ws->C_bon[jb];
                        ws->C_bon[jb] = __dadd_rn(old_cb, inv_m_ds);
                        ckv6_kahan_add(&sum_cb_sq, &comp_cb_sq, __dmul_rn(__dadd_rn(2.0 * old_cb, inv_m_ds), inv_m_ds));
                        ckv6_kahan_add(&sum_cb_ub, &comp_cb_ub, __dmul_rn(inv_m_ds, ws->U_bon[jb]));
                    }
                }

                // Produse scalare scalate exact pe V2 (x s2_scale) si Bonus (x dMB) pentru fiecare extragere a din pasul t
                for (uint32_t a = 0U; a < m_dt_cap; ++a) {
                    uint64_t a_lo = d_draws_lo[dt_beg + a];
                    uint32_t a_hi = d_draws_hi[dt_beg + a];
                    int a_jb = has_bon ? (int)d_draws_bonus[dt_beg + a] : 0;
                    double sk2 = 0.0, skb = 0.0;
                    for (uint32_t d_ptr = ds_beg; d_ptr < ds_end; ++d_ptr) {
                        if (has_v2) {
                            int m_int = __popcll(a_lo & d_draws_lo[d_ptr]) + __popc(a_hi & d_draws_hi[d_ptr]);
                            sk2 = __dadd_rn(sk2, (double)(ckv6_k2_num(m_int, N, r) / k2_g));
                        }
                        if (has_bon) {
                            int b_jb = (int)d_draws_bonus[d_ptr];
                            skb = __dadd_rn(skb, ((a_jb == b_jb) ? dMB : 0.0) - 1.0);
                        }
                    }
                    cur_a_k2[a] = __dmul_rn(sk2, inv_m_ds);
                    cur_a_kb[a] = __dmul_rn(skb, inv_m_ds);
                    dot_a_x2[a] = __dadd_rn(dot_a_x2[a], cur_a_k2[a]);
                    dot_a_xb[a] = __dadd_rn(dot_a_xb[a], cur_a_kb[a]);
                }

                // 2. Daca ell >= 2, tranzitia (s_idx -> s_idx + 1) intra in Koopman V1 (+) V2 cu scalare intreaga exacta
                if (ell >= 2U) {
                    double w1 = 0.0, w2 = 0.0, wb = 0.0;
                    for (uint32_t p1_ptr = ds_beg; p1_ptr < ds_end; ++p1_ptr) {
                        uint64_t s_lo = d_draws_lo[p1_ptr];
                        uint32_t s_hi = d_draws_hi[p1_ptr];
                        int s_jb = has_bon ? (int)d_draws_bonus[p1_ptr] : 0;
                        for (uint32_t p2_ptr = dtm1_beg; p2_ptr < dtm1_end; ++p2_ptr) {
                            int m_int = __popcll(s_lo & d_draws_lo[p2_ptr]) + __popc(s_hi & d_draws_hi[p2_ptr]);
                            w1 = __dadd_rn(w1, (double)(N * m_int - r * r));
                            if (has_v2) {
                                w2 = __dadd_rn(w2, (double)(ckv6_k2_num(m_int, N, r) / k2_g));
                            }
                            if (has_bon) {
                                int t_jb = (int)d_draws_bonus[p2_ptr];
                                wb = __dadd_rn(wb, ((s_jb == t_jb) ? dMB : 0.0) - 1.0);
                            }
                        }
                    }
                    double norm_pair = __dmul_rn(inv_m_ds, inv_m_dtm1);
                    w1 = __dmul_rn(w1, norm_pair);
                    w2 = __dmul_rn(w2, norm_pair);
                    wb = __dmul_rn(wb, norm_pair);

                    w1_sum = __dadd_rn(w1_sum, w1);
                    w1_sq_eff_sum = __dadd_rn(w1_sq_eff_sum, __dmul_rn(__dmul_rn(w1, w1), prev_inv_m));
                    w2_sum = __dadd_rn(w2_sum, w2);
                    w2_sq_eff_sum = __dadd_rn(w2_sq_eff_sum, __dmul_rn(__dmul_rn(w2, w2), prev_inv_m));
                    wb_sum = __dadd_rn(wb_sum, wb);
                    wb_sq_eff_sum = __dadd_rn(wb_sq_eff_sum, __dmul_rn(__dmul_rn(wb, wb), prev_inv_m));

                    double a1_step = __dmul_rn(w1, prev_inv_m);
                    double a2_step = __dmul_rn(w2, prev_inv_m);
                    double ab_step = __dmul_rn(wb, prev_inv_m);

                    for (uint32_t d_ptr = prev_beg; d_ptr < prev_end; ++d_ptr) {
                        uint64_t lo = d_draws_lo[d_ptr];
                        uint32_t hi = d_draws_hi[d_ptr];
                        int balls[20];
                        int nb = 0;
                        uint64_t t_lo = lo;
                        while (t_lo != 0ULL) {
                            int b = __ffsll((long long)t_lo) - 1;
                            balls[nb++] = b;
                            t_lo ^= (1ULL << b);
                        }
                        uint32_t t_hi = hi;
                        while (t_hi != 0U) {
                            int b = __ffs((int)t_hi) - 1;
                            balls[nb++] = 64 + b;
                            t_hi ^= (1U << b);
                        }
                        for (int i = 0; i < nb; ++i) {
                            int bj = balls[i];
                            double old_u1 = ws->U1_v1[bj];
                            ws->U1_v1[bj] = __dadd_rn(old_u1, a1_step);
                            ckv6_kahan_add(&sum_u1_v1_sq, &comp_u1_v1_sq, __dmul_rn(__dadd_rn(2.0 * old_u1, a1_step), a1_step));

                            if (has_v2) {
                                double old_u12 = ws->U1_v2[bj];
                                ws->U1_v2[bj] = __dadd_rn(old_u12, a2_step);
                                ckv6_kahan_add(&sum_u1_v2_sq, &comp_u1_v2_sq, __dmul_rn(__dadd_rn(2.0 * old_u12, a2_step), a2_step));
                                ckv6_kahan_add(&sum_c1_u1_v2, &comp_c1_u1_v2, __dmul_rn(a2_step, ws->C1[bj]));
                            }
                        }
                        if (has_v2) {
                            for (int i = 0; i < nb; ++i) {
                                for (int j = i + 1; j < nb; ++j) {
                                    int pj = ckv6_pair_index(balls[i], balls[j], N);
                                    double old_u2 = ws->U2[pj];
                                    ws->U2[pj] = __dadd_rn(old_u2, a2_step);
                                    ckv6_kahan_add(&sum_u2_sq, &comp_u2_sq, __dmul_rn(__dadd_rn(2.0 * old_u2, a2_step), a2_step));
                                    ckv6_kahan_add(&sum_p_u2, &comp_p_u2, __dmul_rn(a2_step, ws->P[pj]));
                                }
                            }
                        }
                        if (has_bon) {
                            int jb = (int)d_draws_bonus[d_ptr];
                            double old_ub = ws->U_bon[jb];
                            ws->U_bon[jb] = __dadd_rn(old_ub, ab_step);
                            ckv6_kahan_add(&sum_ub_sq, &comp_ub_sq, __dmul_rn(__dadd_rn(2.0 * old_ub, ab_step), ab_step));
                            ckv6_kahan_add(&sum_cb_ub, &comp_cb_ub, __dmul_rn(ab_step, ws->C_bon[jb]));
                        }
                    }

                    for (uint32_t a = 0U; a < m_dt_cap; ++a) {
                        dot_a_z2[a] = __dadd_rn(dot_a_z2[a], __dmul_rn(w2, prev_a_k2[a]));
                        dot_a_zb[a] = __dadd_rn(dot_a_zb[a], __dmul_rn(wb, prev_a_kb[a]));
                    }
                }

                // 3. Evaluam coeficientii Ledoit-Wolf pe Canalul 1 (DM + Koopman V1), Canalul 2 (V2) si Bonus
                double lr = __dmul_rn(dr, d_ell);
                double e_x1 = __dsub_rn(__dadd_rn(sum_c1_sq, comp_c1_sq), __dmul_rn(lr, lr) / dN);
                if (e_x1 < 0.0) e_x1 = 0.0;
                double v_stat1 = __dmul_rn(n_eff_sum, nu1);
                if (e_x1 > v_stat1) {
                    eta_stat1 = __dmul_rn(__dsub_rn(1.0, v_stat1 / e_x1), d_ell / __dadd_rn(d_ell, CKV6_TAU_STAT_1));
                }

                if (ell >= 2U) {
                    double m_tr = (double)(ell - 1U);
                    double w1_real = w1_sum / dN;
                    w1_mean_shift = __dmul_rn(dr, w1_real) / dN;
                    double u1_raw_sq = __dadd_rn(sum_u1_v1_sq, comp_u1_v1_sq) / __dmul_rn(dN, dN);
                    double mw1 = __dmul_rn(dr, w1_real);
                    double e_z1 = __dsub_rn(u1_raw_sq, __dmul_rn(mw1, mw1) / dN);
                    if (e_z1 < 0.0) e_z1 = 0.0;
                    double v_koop1 = __dmul_rn(nu1, w1_sq_eff_sum / __dmul_rn(dN, dN));
                    if (e_z1 > v_koop1) {
                        double g_koop1 = __dmul_rn(
                            __dsub_rn(1.0, v_koop1 / e_z1),
                            d1 / __dmul_rn(nu1, __dadd_rn(m_tr, CKV6_TAU_KOOP_1))
                        );
                        scale_koop1 = __dmul_rn(d1 / nu1, g_koop1);
                    }
                }

                if (has_v2) {
                    double rr1 = (double)(r * (r - 1));
                    double n_n1 = (double)(N * (N - 1));
                    double l_rr1 = __dmul_rn(d_ell, rr1);
                    double e_x2 = __dsub_rn(
                        __dsub_rn(2.0 * __dadd_rn(sum_p2, comp_p2), __dmul_rn(l_rr1, l_rr1) / n_n1),
                        __dmul_rn(v1_proj_coef, e_x1)
                    );
                    if (e_x2 < 0.0) e_x2 = 0.0;
                    double v_stat2 = __dmul_rn(n_eff_sum, nu2);
                    if (e_x2 > v_stat2) {
                        g_stat2 = __dsub_rn(1.0, v_stat2 / e_x2) / __dadd_rn(d_ell, CKV6_TAU_STAT_2);
                    }
                    double e_z2 = 0.0, cross_xz2 = 0.0;
                    if (ell >= 2U) {
                        double m_tr = (double)(ell - 1U);
                        double w2_real = __dmul_rn(w2_sum, inv_s2);
                        double mw2_r = __dmul_rn(dr, w2_real);
                        double mw2_rr1 = __dmul_rn(rr1, w2_real);
                        double u1v2_c_sq = __dsub_rn(
                            __dmul_rn(__dadd_rn(sum_u1_v2_sq, comp_u1_v2_sq), inv_s2_sq),
                            __dmul_rn(mw2_r, mw2_r) / dN
                        );
                        e_z2 = __dsub_rn(
                            __dsub_rn(
                                2.0 * __dmul_rn(__dadd_rn(sum_u2_sq, comp_u2_sq), inv_s2_sq),
                                __dmul_rn(mw2_rr1, mw2_rr1) / n_n1
                            ),
                            __dmul_rn(v1_proj_coef, u1v2_c_sq)
                        );
                        if (e_z2 < 0.0) e_z2 = 0.0;
                        double v_koop2 = __dmul_rn(nu2, __dmul_rn(w2_sq_eff_sum, inv_s2_sq));
                        if (e_z2 > v_koop2) {
                            g_koop2 = __dmul_rn(
                                __dsub_rn(1.0, v_koop2 / e_z2),
                                d2 / __dmul_rn(nu2, __dadd_rn(m_tr, CKV6_TAU_KOOP_2))
                            );
                        }
                        double cross_c1_u12 = __dsub_rn(
                            __dmul_rn(__dadd_rn(sum_c1_u1_v2, comp_c1_u1_v2), inv_s2),
                            __dmul_rn(__dmul_rn(lr, dr), w2_real) / dN
                        );
                        cross_xz2 = __dsub_rn(
                            __dsub_rn(
                                2.0 * __dmul_rn(__dadd_rn(sum_p_u2, comp_p_u2), inv_s2),
                                __dmul_rn(__dmul_rn(__dmul_rn(d_ell, w2_real), rr1), rr1) / n_n1
                            ),
                            __dmul_rn(v1_proj_coef, cross_c1_u12)
                        );
                    }
                    double m2_norm2 = __dadd_rn(
                        __dadd_rn(
                            __dmul_rn(__dmul_rn(g_stat2, g_stat2), e_x2),
                            __dmul_rn(__dmul_rn(g_koop2, g_koop2), e_z2)
                        ),
                        __dmul_rn(__dmul_rn(2.0 * g_stat2, g_koop2), cross_xz2)
                    );
                    if (m2_norm2 < 0.0) m2_norm2 = 0.0;
                    double b2 = __dmul_rn(d2 / sqrt_nu2, sqrt(m2_norm2));
                    c_v2 = ckv6_damp_factor(b2);
                }

                if (has_bon) {
                    double e_xb = __dsub_rn(__dadd_rn(sum_cb_sq, comp_cb_sq), __dmul_rn(__dmul_rn(d_ell, d_ell), inv_mb));
                    if (e_xb < 0.0) e_xb = 0.0;
                    double v_stat_b = __dmul_rn(n_eff_sum, nu1_b);
                    if (e_xb > v_stat_b) {
                        g_stat_b = __dsub_rn(1.0, v_stat_b / e_xb) / __dadd_rn(d_ell, CKV6_TAU_BON_STAT);
                    }
                    double e_zb = 0.0, cross_xzb = 0.0;
                    if (ell >= 2U) {
                        double m_tr = (double)(ell - 1U);
                        double wb_real = __dmul_rn(wb_sum, inv_mb);
                        e_zb = __dsub_rn(
                            __dmul_rn(__dadd_rn(sum_ub_sq, comp_ub_sq), inv_mb_sq),
                            __dmul_rn(__dmul_rn(wb_real, wb_real), inv_mb)
                        );
                        if (e_zb < 0.0) e_zb = 0.0;
                        double v_koop_b = __dmul_rn(nu1_b, __dmul_rn(wb_sq_eff_sum, inv_mb_sq));
                        if (e_zb > v_koop_b) {
                            g_koop_b = __dmul_rn(
                                __dsub_rn(1.0, v_koop_b / e_zb),
                                d1_b / __dmul_rn(nu1_b, __dadd_rn(m_tr, CKV6_TAU_BON_KOOP))
                            );
                        }
                        cross_xzb = __dsub_rn(
                            __dmul_rn(__dadd_rn(sum_cb_ub, comp_cb_ub), inv_mb),
                            __dmul_rn(__dmul_rn(d_ell, wb_real), inv_mb)
                        );
                    }
                    double mb_norm2 = __dadd_rn(
                        __dadd_rn(
                            __dmul_rn(__dmul_rn(g_stat_b, g_stat_b), e_xb),
                            __dmul_rn(__dmul_rn(g_koop_b, g_koop_b), e_zb)
                        ),
                        __dmul_rn(__dmul_rn(2.0 * g_stat_b, g_koop_b), cross_xzb)
                    );
                    if (mb_norm2 < 0.0) mb_norm2 = 0.0;
                    double bb = __dmul_rn(d1_b / sqrt_nu1_b, sqrt(mb_norm2));
                    c_bon = ckv6_damp_factor(bb);
                }
            }
            __syncwarp();

            eta_stat1 = __shfl_sync(0xFFFFFFFFU, eta_stat1, 0);
            scale_koop1 = __shfl_sync(0xFFFFFFFFU, scale_koop1, 0);
            w1_mean_shift = __shfl_sync(0xFFFFFFFFU, w1_mean_shift, 0);

            bool dm_active = (eta_stat1 > 0.0 || scale_koop1 > 0.0);
            double er = comb_nr;
            if (dm_active) {
                double base_c0 = __dadd_rn(__dmul_rn(d_ell, p1), CKV6_ALPHA_DM);
                bool has0 = (lane < N);
                double w0_b = 1.0;
                if (has0) {
                    double z1_0 = __dsub_rn(ws->U1_v1[lane] / dN, w1_mean_shift);
                    double th0 = __dadd_rn(
                        __dmul_rn(eta_stat1, log(__dadd_rn(ws->C1[lane], CKV6_ALPHA_DM) / base_c0)),
                        __dmul_rn(scale_koop1, z1_0)
                    );
                    w0_b = ckv6_ball_weight_f64(th0, th_cap);
                    s_w96[lane] = w0_b;
                }
                bool has1 = (lane + 32 < N);
                double w1_b = 1.0;
                if (has1) {
                    int b1_idx = lane + 32;
                    double z1_1 = __dsub_rn(ws->U1_v1[b1_idx] / dN, w1_mean_shift);
                    double th1 = __dadd_rn(
                        __dmul_rn(eta_stat1, log(__dadd_rn(ws->C1[b1_idx], CKV6_ALPHA_DM) / base_c0)),
                        __dmul_rn(scale_koop1, z1_1)
                    );
                    w1_b = ckv6_ball_weight_f64(th1, th_cap);
                    s_w96[b1_idx] = w1_b;
                }
                bool has2 = (lane + 64 < N);
                double w2_b = 1.0;
                if (has2) {
                    int b2_idx = lane + 64;
                    double z1_2 = __dsub_rn(ws->U1_v1[b2_idx] / dN, w1_mean_shift);
                    double th2 = __dadd_rn(
                        __dmul_rn(eta_stat1, log(__dadd_rn(ws->C1[b2_idx], CKV6_ALPHA_DM) / base_c0)),
                        __dmul_rn(scale_koop1, z1_2)
                    );
                    w2_b = ckv6_ball_weight_f64(th2, th_cap);
                    s_w96[b2_idx] = w2_b;
                }
                __syncwarp();
                er = ckv6_warp_elem_sym_f64(w0_b, has0, w1_b, has1, w2_b, has2, r);
            }

            if (lane == 0) {
                double pi_step_sum = 0.0;
                for (uint32_t a = 0U; a < m_dt_cap; ++a) {
                    double lam1_a = 1.0;
                    if (dm_active) {
                        double prod_a = 1.0;
                        for (int i = 0; i < r; ++i) {
                            prod_a = __dmul_rn(prod_a, s_w96[dt_balls_sh[a][i]]);
                        }
                        lam1_a = __dmul_rn(comb_nr, prod_a) / er;
                    }
                    double lam2_a = 1.0;
                    if (has_v2) {
                        double s2 = __dmul_rn(
                            d2 / nu2,
                            __dadd_rn(
                                __dmul_rn(g_stat2, dot_a_x2[a] / s2_scale),
                                __dmul_rn(g_koop2, dot_a_z2[a] / __dmul_rn(s2_scale, s2_scale))
                            )
                        );
                        lam2_a = __dadd_rn(1.0, __dmul_rn(c_v2, s2));
                        if (lam2_a < 0.05) lam2_a = 0.05;
                    }
                    double pi_m = __dadd_rn(__dmul_rn(CKV6_BETA_DM, lam1_a), __dmul_rn(CKV6_BETA_V2, lam2_a));
                    double pi_b = 1.0;
                    if (has_bon) {
                        double sb = __dmul_rn(
                            d1_b / nu1_b,
                            __dadd_rn(
                                __dmul_rn(g_stat_b, __dmul_rn(dot_a_xb[a], inv_mb)),
                                __dmul_rn(g_koop_b, __dmul_rn(dot_a_zb[a], inv_mb_sq))
                            )
                        );
                        pi_b = __dadd_rn(1.0, __dmul_rn(c_bon, sb));
                        if (pi_b < 0.05) pi_b = 0.05;
                    }
                    pi_step_sum = __dadd_rn(pi_step_sum, __dmul_rn(pi_m, pi_b));
                }
                ws->pi_buf[ell] = __dmul_rn(pi_step_sum, inv_m_dt);

                prev_beg = ds_beg;
                prev_end = ds_end;
                prev_inv_m = inv_m_ds;
                for (uint32_t a = 0U; a < m_dt_cap; ++a) {
                    prev_a_k2[a] = cur_a_k2[a];
                    prev_a_kb[a] = cur_a_kb[a];
                }
            }
            __syncwarp();
        }

        if (lane == 0) {
            // Actualizare Adams-MacKay cu cap l_act <= CKV6_L_MAX (512)
            double L0 = 0.0, L1 = 0.0;
            for (uint32_t ell = 0U; ell <= l_act; ++ell) {
                double pv = ws->pi_buf[ell];
                L0 = __dadd_rn(L0, __dmul_rn(ws->R0[ell], pv));
                L1 = __dadd_rn(L1, __dmul_rn(ws->R1[ell], pv));
            }
            double lw_max = (lw0 >= lw1) ? lw0 : lw1;
            double a0 = exp(__dsub_rn(lw0, lw_max));
            double a1 = exp(__dsub_rn(lw1, lw_max));
            double eta0 = a0 / __dadd_rn(a0, a1);
            double eta1 = a1 / __dadd_rn(a0, a1);

            double l_bocpd = __dadd_rn(__dmul_rn(eta0, L0), __dmul_rn(eta1, L1));
            double kappa = 1.0;
            if (log_m < 0.0) {
                double m_prev = exp(log_m);
                kappa = __dsub_rn(m_prev, 0.50) / __dmul_rn(0.50, m_prev);
                if (kappa < 0.0) kappa = 0.0;
                if (kappa > 1.0) kappa = 1.0;
            }
            double e_step = __dadd_rn(1.0, __dmul_rn(__dmul_rn(0.50, kappa), __dsub_rn(l_bocpd, 1.0)));
            if (e_step < 0.50) e_step = 0.50;
            log_m = __dadd_rn(log_m, log(e_step));
            lw0 = __dadd_rn(lw0, log(L0));
            lw1 = __dadd_rn(lw1, log(L1));

            double l_glob = (l_act == 0U) ? 1.0 : ws->pi_buf[l_act];
            double kappa_glob = 1.0;
            if (log_m_glob < 0.0) {
                double mg = exp(log_m_glob);
                kappa_glob = __dsub_rn(mg, 0.50) / __dmul_rn(0.50, mg);
                if (kappa_glob < 0.0) kappa_glob = 0.0;
                if (kappa_glob > 1.0) kappa_glob = 1.0;
            }
            double e_glob = __dadd_rn(1.0, __dmul_rn(__dmul_rn(0.50, kappa_glob), __dsub_rn(l_glob, 1.0)));
            if (e_glob < 0.50) e_glob = 0.50;
            log_m_glob = __dadd_rn(log_m_glob, log(e_glob));

            double lw_max_p = (lw0 >= lw1) ? lw0 : lw1;
            double ap0 = exp(lw0 - lw_max_p);
            double ap1 = exp(lw1 - lw_max_p);
            double ep0 = ap0 / (ap0 + ap1);
            double ep1 = ap1 / (ap0 + ap1);

            int best_ell = 0;
            double best_post = -1.0;
            double cp_prob_short = 0.0; // P(r_t <= 5 | D_{1:t})
            for (uint32_t ell = 0U; ell <= l_act; ++ell) {
                double pv = ws->pi_buf[ell];
                double p_ell = ep0 * (ws->R0[ell] * pv / L0) + ep1 * (ws->R1[ell] * pv / L1);
                if (ell <= 5U) cp_prob_short += p_ell;
                if (p_ell > best_post) {
                    best_post = p_ell;
                    best_ell = (int)ell;
                }
            }

            uint32_t out_idx = meta.out_step_base + t;
            d_out_step_e[out_idx] = e_step;
            d_out_step_log_m[out_idx] = log_m;
            d_out_step_map_r[out_idx] = best_ell;
            d_out_step_cp_prob[out_idx] = cp_prob_short;
            d_out_step_log_m_glob[out_idx] = log_m_glob;

            // Tranzitie Adams-MacKay cu absorbtia supravietuitorului la CKV6_L_MAX (conservare exacta masa = 1.0)
            double inv_L0 = (1.0 - CKV6_LAMBDA_0) / L0;
            double inv_L1 = (1.0 - CKV6_LAMBDA_1) / L1;
            if (l_act == (uint32_t)CKV6_L_MAX) {
                double surv0 = ws->R0[CKV6_L_MAX] * ws->pi_buf[CKV6_L_MAX] * inv_L0;
                double surv1 = ws->R1[CKV6_L_MAX] * ws->pi_buf[CKV6_L_MAX] * inv_L1;
                for (int ell = CKV6_L_MAX - 1; ell >= 0; --ell) {
                    double pv = ws->pi_buf[ell];
                    ws->R0[ell + 1] = ws->R0[ell] * pv * inv_L0;
                    ws->R1[ell + 1] = ws->R1[ell] * pv * inv_L1;
                }
                ws->R0[CKV6_L_MAX] += surv0;
                ws->R1[CKV6_L_MAX] += surv1;
            } else {
                for (int ell = (int)l_act; ell >= 0; --ell) {
                    double pv = ws->pi_buf[ell];
                    ws->R0[ell + 1] = ws->R0[ell] * pv * inv_L0;
                    ws->R1[ell + 1] = ws->R1[ell] * pv * inv_L1;
                }
            }
            ws->R0[0] = CKV6_LAMBDA_0;
            ws->R1[0] = CKV6_LAMBDA_1;
        }
        __syncwarp();
    }
}

// Evaluare exhaustiva a campului predictiv CK-V6 pe intregul univers rezident in VRAM
// (13,983,816 la 6/49, 3,838,380 la 5/40 draws6, 24,435,180 la Joker):
// Verifica exact 1/|Omega| * sum_{omega} E_t(omega) == 1.0 si min_{omega} E_t(omega) >= 0.50!
__constant__ double c_ckv6_h1[64];
__constant__ double c_ckv6_q2[2401]; // 49 * 49

__device__ __forceinline__ void ckv6_atomic_min_f64(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    while (__longlong_as_double((long long)old) > val) {
        assumed = old;
        old = atomicCAS(addr_ull, assumed, (unsigned long long)__double_as_longlong(val));
        if (old == assumed) break;
    }
}

__device__ __forceinline__ void ckv6_atomic_max_f64(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    while (__longlong_as_double((long long)old) < val) {
        assumed = old;
        old = atomicCAS(addr_ull, assumed, (unsigned long long)__double_as_longlong(val));
        if (old == assumed) break;
    }
}

#define CKV6_MAX_UNIV_DM_TERMS 68
#define CKV6_DM_ROW_STRIDE 70

__global__ void k_ckv6_universe_eval_f64(
    int32_t game_code, // 0 = 649 (13.98M), 1 = 540_draws6 (3.84M), 2 = joker_24m (24.43M)
    const uint64_t* __restrict__ d_univ_masks,
    const double* __restrict__ d_joker_tables, // pt game_code == 2: 20 felii x 2071 f64 (b0 + pad[45] + q2[45*45])
    const double* __restrict__ d_dm_table,     // n_dm_terms x 70 f64 (u[0..49], alpha_main, alpha_jok[0..20])
    uint32_t n_dm_terms,
    uint32_t n_states,
    int32_t n_balls,
    int32_t r_draw,
    double kappa,
    unsigned long long* __restrict__ d_sum_lo,
    unsigned long long* __restrict__ d_sum_hi,
    double* __restrict__ d_min_e,
    double* __restrict__ d_max_e
) {
    __shared__ double s_dm_tbl[CKV6_MAX_UNIV_DM_TERMS * CKV6_DM_ROW_STRIDE];
    __shared__ double s_min;
    __shared__ double s_max;

    uint32_t tot_dm = n_dm_terms * (uint32_t)CKV6_DM_ROW_STRIDE;
    for (uint32_t i = threadIdx.x; i < tot_dm; i += blockDim.x) {
        s_dm_tbl[i] = d_dm_table[i];
    }
    if (threadIdx.x == 0) {
        s_min = 1.0e9;
        s_max = -1.0e9;
    }
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double loc_e = 1.0;
    bool active = (idx < n_states);

    if (active) {
        uint64_t raw = d_univ_masks[idx];
        uint64_t m = (game_code == 2) ? (raw & ((1ULL << 45) - 1ULL)) : raw;
        int balls[6];
        for (int i = 0; i < r_draw; ++i) {
            int bi = __ffsll((long long)m) - 1;
            balls[i] = bi;
            m ^= (1ULL << bi);
        }
        double pi_tot = 1.0;
        if (game_code == 2 && d_joker_tables != nullptr) {
            int jb = (int)((raw >> 48) & 0x1FULL);
            const double* slice = d_joker_tables + (size_t)jb * 2071ULL;
            double b0 = slice[0];
            const double* q2 = slice + 46;
            double s2 = 0.0;
            for (int i = 0; i < r_draw; ++i) {
                int row_off = balls[i] * 45;
                for (int j = i + 1; j < r_draw; ++j) {
                    s2 += q2[row_off + balls[j]];
                }
            }
            double s_dm = 0.0;
            int b0_i = balls[0], b1_i = balls[1], b2_i = balls[2], b3_i = balls[3], b4_i = balls[4];
            for (uint32_t k = 0U; k < n_dm_terms; ++k) {
                const double* row = s_dm_tbl + k * CKV6_DM_ROW_STRIDE;
                double prod_u = row[b0_i] * row[b1_i] * row[b2_i] * row[b3_i] * row[b4_i];
                s_dm += row[50 + jb] * prod_u;
            }
            pi_tot = b0 + s_dm + s2;
        } else {
            double b0 = c_ckv6_h1[0];
            double s2 = 0.0;
            for (int i = 0; i < r_draw; ++i) {
                int row_off = balls[i] * n_balls;
                for (int j = i + 1; j < r_draw; ++j) {
                    s2 += c_ckv6_q2[row_off + balls[j]];
                }
            }
            double s_dm = 0.0;
            if (r_draw == 6) {
                int b0_i = balls[0], b1_i = balls[1], b2_i = balls[2], b3_i = balls[3], b4_i = balls[4], b5_i = balls[5];
                for (uint32_t k = 0U; k < n_dm_terms; ++k) {
                    const double* row = s_dm_tbl + k * CKV6_DM_ROW_STRIDE;
                    double prod_u = row[b0_i] * row[b1_i] * row[b2_i] * row[b3_i] * row[b4_i] * row[b5_i];
                    s_dm += row[49] * prod_u;
                }
            } else {
                for (uint32_t k = 0U; k < n_dm_terms; ++k) {
                    const double* row = s_dm_tbl + k * CKV6_DM_ROW_STRIDE;
                    double prod_u = 1.0;
                    for (int i = 0; i < r_draw; ++i) prod_u *= row[balls[i]];
                    s_dm += row[49] * prod_u;
                }
            }
            pi_tot = b0 + s_dm + s2;
        }
        loc_e = 1.0 + 0.50 * kappa * (pi_tot - 1.0);
        double scaled_mean_term = loc_e / (double)n_states;
        pmv3_atomic_add_u128(d_sum_lo, d_sum_hi, 0U, scaled_mean_term);
        ckv6_atomic_min_f64(&s_min, loc_e);
        ckv6_atomic_max_f64(&s_max, loc_e);
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_min < 1.0e8) {
        ckv6_atomic_min_f64(d_min_e, s_min);
        ckv6_atomic_max_f64(d_max_e, s_max);
    }
}

extern "C" {

int32_t v3_cuda_ckv6_649_batch_f64(
    const uint64_t* h_draws_batch,
    uint32_t n_traj,
    uint32_t t_steps,
    uint32_t l_regime_cp,
    uint32_t record_steps,
    double* h_summary5_out,
    double* h_step_e_out,
    double* h_step_log_m_out,
    int32_t* h_step_map_r_out,
    double* elapsed_ms_out
) {
    if (n_traj == 0U || t_steps == 0U || t_steps > 400U) return -1;
    size_t draws_b = (size_t)n_traj * (size_t)t_steps * sizeof(uint64_t);
    size_t sum5_b = (size_t)n_traj * 5ULL * sizeof(double);
    size_t step_f_b = (size_t)n_traj * (size_t)t_steps * sizeof(double);
    size_t step_i_b = (size_t)n_traj * (size_t)t_steps * sizeof(int32_t);

    uint64_t* d_draws = nullptr;
    double *d_sum5 = nullptr, *d_se = nullptr, *d_slm = nullptr;
    int32_t* d_smr = nullptr;

    if (cudaMalloc(&d_draws, draws_b) != cudaSuccess ||
        cudaMalloc(&d_sum5, sum5_b) != cudaSuccess) {
        if (d_draws) cudaFree(d_draws);
        if (d_sum5) cudaFree(d_sum5);
        return -3;
    }
    if (record_steps) {
        if (cudaMalloc(&d_se, step_f_b) != cudaSuccess ||
            cudaMalloc(&d_slm, step_f_b) != cudaSuccess ||
            cudaMalloc(&d_smr, step_i_b) != cudaSuccess) {
            cudaFree(d_draws); cudaFree(d_sum5);
            if (d_se) cudaFree(d_se);
            if (d_slm) cudaFree(d_slm);
            if (d_smr) cudaFree(d_smr);
            return -3;
        }
    }
    cudaMemcpy(d_draws, h_draws_batch, draws_b, cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    k_ckv6_649_batch_f64<<<n_traj, 32>>>(
        d_draws, n_traj, t_steps, l_regime_cp, record_steps, d_sum5, d_se, d_slm, d_smr
    );

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        cudaMemcpy(h_summary5_out, d_sum5, sum5_b, cudaMemcpyDeviceToHost);
        if (record_steps) {
            if (h_step_e_out) cudaMemcpy(h_step_e_out, d_se, step_f_b, cudaMemcpyDeviceToHost);
            if (h_step_log_m_out) cudaMemcpy(h_step_log_m_out, d_slm, step_f_b, cudaMemcpyDeviceToHost);
            if (h_step_map_r_out) cudaMemcpy(h_step_map_r_out, d_smr, step_i_b, cudaMemcpyDeviceToHost);
        }
    }
    cudaFree(d_draws); cudaFree(d_sum5);
    if (d_se) cudaFree(d_se);
    if (d_slm) cudaFree(d_slm);
    if (d_smr) cudaFree(d_smr);
    return (int32_t)err;
}

int32_t v3_cuda_ckv6_power_10k_f64(
    const uint32_t* h_delta_bp_cells,
    const uint32_t* h_l_reg_cells,
    uint32_t n_cells,
    uint32_t reps_per_cell,
    uint32_t t_steps,
    uint64_t seed,
    double* h_records8_out,
    double* elapsed_ms_out
) {
    uint32_t total_hist = n_cells * reps_per_cell;
    if (total_hist == 0U || t_steps == 0U || t_steps > 400U) return -1;

    uint32_t *d_dbp = nullptr, *d_lrg = nullptr;
    double* d_rec8 = nullptr;
    size_t cell_b = (size_t)n_cells * sizeof(uint32_t);
    size_t rec8_b = (size_t)total_hist * 8ULL * sizeof(double);

    if (cudaMalloc(&d_dbp, cell_b) != cudaSuccess ||
        cudaMalloc(&d_lrg, cell_b) != cudaSuccess ||
        cudaMalloc(&d_rec8, rec8_b) != cudaSuccess) {
        if (d_dbp) cudaFree(d_dbp);
        if (d_lrg) cudaFree(d_lrg);
        if (d_rec8) cudaFree(d_rec8);
        return -3;
    }
    cudaMemcpy(d_dbp, h_delta_bp_cells, cell_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lrg, h_l_reg_cells, cell_b, cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    k_ckv6_power_10k_f64<<<total_hist, 32>>>(
        d_dbp, d_lrg, n_cells, reps_per_cell, t_steps, seed, d_rec8
    );

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        cudaMemcpy(h_records8_out, d_rec8, rec8_b, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_dbp); cudaFree(d_lrg); cudaFree(d_rec8);
    return (int32_t)err;
}

int32_t v3_cuda_ckv6_general_streams_f64(
    const V3Ckv6StreamMeta* h_metas,
    uint32_t n_streams,
    const uint32_t* h_step_draw_offs,
    uint32_t total_step_offs_len,
    const uint64_t* h_draws_lo,
    const uint32_t* h_draws_hi,
    const uint8_t* h_draws_bonus,
    uint32_t total_raw_draws,
    uint32_t total_out_steps,
    double* h_out_step_e,
    double* h_out_step_log_m,
    int32_t* h_out_step_map_r,
    double* h_out_step_cp_prob,
    double* h_out_step_log_m_glob,
    double* elapsed_ms_out
) {
    if (n_streams == 0U || total_raw_draws == 0U || total_out_steps == 0U) return -1;

    V3Ckv6StreamMeta* d_metas = nullptr;
    uint32_t *d_offs = nullptr, *d_hi = nullptr;
    uint64_t* d_lo = nullptr;
    uint8_t* d_bon = nullptr;
    V3Ckv6WorkSlice* d_work = nullptr;
    double *d_se = nullptr, *d_slm = nullptr, *d_scp = nullptr, *d_slmg = nullptr;
    int32_t* d_smr = nullptr;

    size_t meta_b = (size_t)n_streams * sizeof(V3Ckv6StreamMeta);
    size_t offs_b = (size_t)total_step_offs_len * sizeof(uint32_t);
    size_t lo_b = (size_t)total_raw_draws * sizeof(uint64_t);
    size_t hi_b = (size_t)total_raw_draws * sizeof(uint32_t);
    size_t bon_b = (size_t)total_raw_draws * sizeof(uint8_t);
    size_t work_b = (size_t)n_streams * sizeof(V3Ckv6WorkSlice);
    size_t out_f_b = (size_t)total_out_steps * sizeof(double);
    size_t out_i_b = (size_t)total_out_steps * sizeof(int32_t);

    if (cudaMalloc(&d_metas, meta_b) != cudaSuccess ||
        cudaMalloc(&d_offs, offs_b) != cudaSuccess ||
        cudaMalloc(&d_lo, lo_b) != cudaSuccess ||
        cudaMalloc(&d_hi, hi_b) != cudaSuccess ||
        cudaMalloc(&d_bon, bon_b) != cudaSuccess ||
        cudaMalloc(&d_work, work_b) != cudaSuccess ||
        cudaMalloc(&d_se, out_f_b) != cudaSuccess ||
        cudaMalloc(&d_slm, out_f_b) != cudaSuccess ||
        cudaMalloc(&d_smr, out_i_b) != cudaSuccess ||
        cudaMalloc(&d_scp, out_f_b) != cudaSuccess ||
        cudaMalloc(&d_slmg, out_f_b) != cudaSuccess) {
        if (d_metas) cudaFree(d_metas); if (d_offs) cudaFree(d_offs);
        if (d_lo) cudaFree(d_lo); if (d_hi) cudaFree(d_hi);
        if (d_bon) cudaFree(d_bon); if (d_work) cudaFree(d_work);
        if (d_se) cudaFree(d_se); if (d_slm) cudaFree(d_slm);
        if (d_smr) cudaFree(d_smr); if (d_scp) cudaFree(d_scp);
        if (d_slmg) cudaFree(d_slmg);
        return -3;
    }

    cudaMemcpy(d_metas, h_metas, meta_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offs, h_step_draw_offs, offs_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lo, h_draws_lo, lo_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hi, h_draws_hi, hi_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bon, h_draws_bonus, bon_b, cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    k_ckv6_general_streams_f64<<<n_streams, 32>>>(
        d_metas, n_streams, d_offs, d_lo, d_hi, d_bon, d_work, d_se, d_slm, d_smr, d_scp, d_slmg
    );

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        cudaMemcpy(h_out_step_e, d_se, out_f_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_step_log_m, d_slm, out_f_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_step_map_r, d_smr, out_i_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_step_cp_prob, d_scp, out_f_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_step_log_m_glob, d_slmg, out_f_b, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_metas); cudaFree(d_offs); cudaFree(d_lo); cudaFree(d_hi);
    cudaFree(d_bon); cudaFree(d_work); cudaFree(d_se); cudaFree(d_slm);
    cudaFree(d_smr); cudaFree(d_scp); cudaFree(d_slmg);
    return (int32_t)err;
}

int32_t v3_cuda_ckv6_universe_eval_f64(
    int32_t game_code,
    const uint64_t* d_univ_masks,
    uint32_t n_states,
    int32_t n_balls,
    int32_t r_draw,
    const double* h_h1_64,
    const double* h_q2_2401,
    const double* h_joker_tables_41420,
    const double* h_dm_table,
    uint32_t n_dm_terms,
    double kappa,
    double* mean_e_out,
    double* min_e_out,
    double* max_e_out,
    double* elapsed_ms_out
) {
    if (n_dm_terms > (uint32_t)CKV6_MAX_UNIV_DM_TERMS) return -2;
    cudaMemcpyToSymbol(c_ckv6_h1, h_h1_64, 64 * sizeof(double));
    cudaMemcpyToSymbol(c_ckv6_q2, h_q2_2401, 2401 * sizeof(double));

    double* d_jok_tbl = nullptr;
    if (game_code == 2 && h_joker_tables_41420 != nullptr) {
        size_t j_bytes = 20ULL * 2071ULL * sizeof(double);
        if (cudaMalloc(&d_jok_tbl, j_bytes) != cudaSuccess) {
            return -3;
        }
        cudaMemcpy(d_jok_tbl, h_joker_tables_41420, j_bytes, cudaMemcpyHostToDevice);
    }

    double* d_dm_tbl = nullptr;
    if (n_dm_terms > 0U && h_dm_table != nullptr) {
        size_t dm_bytes = (size_t)n_dm_terms * (size_t)CKV6_DM_ROW_STRIDE * sizeof(double);
        if (cudaMalloc(&d_dm_tbl, dm_bytes) != cudaSuccess) {
            if (d_jok_tbl) cudaFree(d_jok_tbl);
            return -3;
        }
        cudaMemcpy(d_dm_tbl, h_dm_table, dm_bytes, cudaMemcpyHostToDevice);
    }

    unsigned long long *d_lo = nullptr, *d_hi = nullptr;
    double *d_min = nullptr, *d_max = nullptr;
    if (cudaMalloc(&d_lo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_hi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&d_min, sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_max, sizeof(double)) != cudaSuccess) {
        if (d_jok_tbl) cudaFree(d_jok_tbl);
        if (d_dm_tbl) cudaFree(d_dm_tbl);
        if (d_lo) cudaFree(d_lo); if (d_hi) cudaFree(d_hi);
        if (d_min) cudaFree(d_min); if (d_max) cudaFree(d_max);
        return -3;
    }
    cudaMemset(d_lo, 0, sizeof(unsigned long long));
    cudaMemset(d_hi, 0, sizeof(unsigned long long));
    double init_min = 1.0e9, init_max = -1.0e9;
    cudaMemcpy(d_min, &init_min, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &init_max, sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    uint32_t blocks = (n_states + 255U) / 256U;
    k_ckv6_universe_eval_f64<<<blocks, 256>>>(
        game_code, d_univ_masks, d_jok_tbl, d_dm_tbl, n_dm_terms, n_states, n_balls, r_draw, kappa, d_lo, d_hi, d_min, d_max
    );

    cudaEventRecord(ev1);
    cudaError_t err = cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    if (elapsed_ms_out) *elapsed_ms_out = (double)ms;

    if (err == cudaSuccess) {
        unsigned long long hlo = 0ULL, hhi = 0ULL;
        cudaMemcpy(&hlo, d_lo, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&hhi, d_hi, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        double a = ldexp((double)hhi, -32);
        double b = ldexp((double)(hlo >> 32), -64);
        double c = ldexp((double)(hlo & 0xFFFFFFFFULL), -96);
        *mean_e_out = a + (b + c);
        cudaMemcpy(min_e_out, d_min, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(max_e_out, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (d_jok_tbl) cudaFree(d_jok_tbl);
    if (d_dm_tbl) cudaFree(d_dm_tbl);
    cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_min); cudaFree(d_max);
    return (int32_t)err;
}

} // extern "C"

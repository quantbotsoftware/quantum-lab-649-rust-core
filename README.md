# Quantum Lab 649 · Rust Core v3 — Native Tauri v2 + CUDA + AVX2 Combinatorial Observatory

![License: MIT](https://img.shields.io/badge/license-MIT-10b981) ![Stack: Rust%20%2B%20Tauri%20v2%20%2B%20CUDA](https://img.shields.io/badge/stack-Rust%20%7C%20Tauri%20v2%20%7C%20CUDA-3b82f6) ![Claims: None](https://img.shields.io/badge/positive__claims__allowed-false-ef4444) [![Donations: Good Samaritans](https://img.shields.io/badge/donations-good%20samaritans-ec4899)](https://ko-fi.com/quantbotsoftware)

Native 12-tab Dark-Mode desktop combinatorial observatory built in **Rust (2021 edition) + Tauri v2 + CUDA C++17 (`v3_kernels.cu`) + AVX2/POPCNT**, featuring zero-copy `mmap` over a **39.07M-ticket binary universe (`1.3 GB` Vault)**, Johnson association scheme / Krawtchouk / Koopman spectral decompositions (`KJ-V5`), covering designs (`CD-V3`), block-bootstrap engines (`PM-V3`), Philox-4x32-10 bit-identical CPU/GPU parity (`CK-V6`), and bounded-memory procfs telemetry.

> ⚠️ **STRICTLY AS-IS · CLASSICAL HPC ONLY · DOES NOT PREDICT LOTTERY NUMBERS (`positive_claims_allowed = false`):**
> All computations are 100% classical combinatorial/statistical research algorithms. Official lottery draws are independent random events where every ticket has identical probability. **This software does NOT predict winning numbers and guarantees ZERO winnings.** Provided with **no warranty, no liability, and no technical support**. See [DISCLAIMER.md](DISCLAIMER.md).

## 📘 User Manual (EN / RO)

- **English:** [QUANTUM_RUST_CORE_V3_USER_MANUAL.md](QUANTUM_RUST_CORE_V3_USER_MANUAL.md) · online: [bot-tradingsoftware.com/docs/QUANTUM_RUST_CORE_V3_USER_MANUAL.html](https://bot-tradingsoftware.com/docs/QUANTUM_RUST_CORE_V3_USER_MANUAL.html)
- **Română:** [QUANTUM_RUST_CORE_V3_USER_MANUAL_RO.md](QUANTUM_RUST_CORE_V3_USER_MANUAL_RO.md) · online: [bot-tradingsoftware.com/docs/QUANTUM_RUST_CORE_V3_USER_MANUAL_RO.html](https://bot-tradingsoftware.com/docs/QUANTUM_RUST_CORE_V3_USER_MANUAL_RO.html)

Build prerequisites and `CUDA_ARCH` table, the 1.3 GB Vault (download or rebuild), linking the Quantum Lab 649 data archive (`QL649_ROOT`), all 12 tabs and their controls, the CLI sub-commands, environment variables and troubleshooting.

## 🛠️ Build & Run (Linux + NVIDIA CUDA 12.x)

1. **System Prerequisites (Ubuntu/Debian):**
   ```bash
   sudo apt-get install -y build-essential pkg-config libssl-dev libsqlite3-dev \
     libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
   ```
   Requires Rust 1.78+ and NVIDIA CUDA Toolkit 12.x (`nvcc`).

2. **Compile (adjust `CUDA_ARCH` to match your GPU, e.g. `sm_86` RTX 30xx, `sm_89` RTX 40xx, `sm_120` RTX 50xx):**
   ```bash
   CUDA_ARCH=sm_89 cargo build --release
   ```

3. **Optional — Download the Precomputed `1.3 GB` Binary Vault (`RUST_CORE_V3_VAULT` — CERN Zenodo DOI `10.5281/zenodo.23992530`):**
   All 13 `.v3bin` tables (`13,983,816` 6/49 masks, `24,435,180` Joker masks, `3,838,380` 5/40 draws, bootstrap tables & covering catalogs) are permanently archived on **CERN Zenodo** ([https://doi.org/10.5281/zenodo.23992530](https://doi.org/10.5281/zenodo.23992530)) and verified against [`vault/VAULT_MANIFEST.json`](vault/VAULT_MANIFEST.json):
   ```bash
   ./scripts/download_vault.sh
   ```

4. **Launch GUI & Install Linux `.desktop` Shortcut:**
   ```bash
   ./launch_rust_core_v3.sh
   ./desktop/install-desktop-shortcut.sh
   ```

## 🖼️ UI Reference Screenshots (Native 12-Tab Observatory Confirmation)

> *Illustrative engineering confirmation only (`positive_claims_allowed = false`) — showing the compiled Rust + Tauri v2 + CUDA + `mmap` desktop observatory; does NOT predict lottery numbers.*

- 🌐 **Full Interactive UI Gallery:** [https://bot-tradingsoftware.com/#screenshots](https://bot-tradingsoftware.com/#screenshots)

![Rust Core v3 Zero-Copy Vault & CD-V3Q Wheels](https://bot-tradingsoftware.com/img/screenshots/rust-core-v3-1-vault-wheels.png)

![Rust Core v3 Exact Pareto Frontier & Live GPU == 24T CPU Parity](https://bot-tradingsoftware.com/img/screenshots/rust-core-v3-2-pareto-gpu-eval.png)

![Rust Core v3 Gate A6 SHA-256 Registries & CK-V6 Causal Engine](https://bot-tradingsoftware.com/img/screenshots/rust-core-v3-3-ckv6-registries.png)

## ❤️ Voluntary Donations ("Good Samaritans")
Released 100% free & open-source (MIT). If the Rust + Tauri v2 + CUDA FFI + `mmap` architecture helps your own projects:
- 🌐 **Open-Source Hub & Donations:** [https://bot-tradingsoftware.com/#donate](https://bot-tradingsoftware.com/#donate)

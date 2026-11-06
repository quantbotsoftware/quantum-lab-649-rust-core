# Quantum Lab 649 · Rust Core v3 — Native Tauri v2 + CUDA + AVX2 Combinatorial Observatory

[![CI](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/actions/workflows/ci.yml/badge.svg)](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23992530.svg)](https://doi.org/10.5281/zenodo.23992530) [![Hub](https://img.shields.io/badge/hub-bot--tradingsoftware.com-0a7)](https://bot-tradingsoftware.com/)

![License: MIT](https://img.shields.io/badge/license-MIT-10b981) ![Stack: Rust%20%2B%20Tauri%20v2%20%2B%20CUDA](https://img.shields.io/badge/stack-Rust%20%7C%20Tauri%20v2%20%7C%20CUDA-3b82f6) ![Claims: None](https://img.shields.io/badge/positive__claims__allowed-false-ef4444) [![Donations: Good Samaritans](https://img.shields.io/badge/donations-good%20samaritans-ec4899)](https://ko-fi.com/quantbotsoftware)

Native 12-tab Dark-Mode desktop combinatorial observatory built in **Rust (2021 edition) + Tauri v2 + CUDA C++17 (`v3_kernels.cu`) + AVX2/POPCNT**, featuring zero-copy `mmap` over a **39.07M-ticket binary universe (`1.3 GB` Vault)**, Johnson association scheme / Krawtchouk / Koopman spectral decompositions (`KJ-V5`), covering designs (`CD-V3`), block-bootstrap engines (`PM-V3`), Philox-4x32-10 bit-identical CPU/GPU parity (`CK-V6`), and bounded-memory procfs telemetry.

> ⚠️ **STRICTLY AS-IS · CLASSICAL HPC ONLY · DOES NOT PREDICT LOTTERY NUMBERS (`positive_claims_allowed = false`):**
> All computations are 100% classical combinatorial/statistical research algorithms. Official lottery draws are independent random events where every ticket has identical probability. **This software does NOT predict winning numbers and guarantees ZERO winnings.** Provided with **no warranty, no liability, and no technical support**. See [DISCLAIMER.md](DISCLAIMER.md).

## 🧱 For data engineers: the `.v3bin` format & the 1.3 GB Vault as a dataset

If you do not care about lotteries at all, this repository is still useful as a **reproducible data-engineering artefact**: 13 flat, little-endian, `mmap`-able tables with a 64-byte header (magic `QLV3VLT\x01`), 39,077,004 exhaustively enumerated k-subset bitmasks in lexicographic (= combinatorial-number-system) order, 1,000-replicate bootstrap statistics per ticket, SHA-256 pinned in `VAULT_MANIFEST.json`, bit-exact GPU ⇄ CPU parity harness (Gate A0), and a DOI on CERN Zenodo. Every table opens from Python in one line with `numpy.memmap` — no application needed.

- 📐 **Format spec, Python/Rust readers, measured Gate A0 numbers:** [V3BIN_FORMAT_SPEC.md](V3BIN_FORMAT_SPEC.md) · 🇷🇴 [V3BIN_FORMAT_SPEC_RO.md](V3BIN_FORMAT_SPEC_RO.md)
- ✅ **CI:** the CUDA kernels are compiled with `nvcc` on every push and 80+ CPU-side unit tests run on GitHub Actions (`.github/workflows/ci.yml`); GPU-parity tests are listed explicitly as skipped there and run locally.

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


## ✅ Continuous Integration

Every push to `main` installs the NVIDIA CUDA 12.8 toolkit on a GPU-less runner, **compiles `cuda/v3_kernels.cu` with `nvcc`, links against the driver stub and runs the 81 CPU-side unit tests** (v3bin parser, SHA-256 registries, integrity gates, telemetry, resources, PM-V3 CPU paths). The 4 GPU-parity tests and 1 test needing private reserve files are skipped by name in the workflow and run locally on real hardware (see `V3BIN_FORMAT_SPEC.md` §5 for measured results).

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) · runs: [https://github.com/quantbotsoftware/quantum-lab-649-rust-core/actions](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/actions). No secrets, no broker or network calls, no telemetry — fork it and the same checks run for you.

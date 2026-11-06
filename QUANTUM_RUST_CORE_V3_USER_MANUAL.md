# Quantum Lab 649 · Rust Core v3 — User Manual (English)

> **Scope notice.** Rust Core v3 is a native desktop *HPC research observatory* (Rust + Tauri v2 + CUDA C++17 + AVX2) over the complete combinatorial universes of Loto 6/49, 5/40 and Joker (39.07 million tickets in a zero-copy `mmap` Vault). It computes covering designs, exact Pareto frontiers, bootstrap calibrations, spectral decompositions and bit-exact GPU/CPU parity checks. It **does not predict lottery numbers, offers no advantage and guarantees nothing** — `positive_claims_allowed = false` is hard-wired into the window title, every screen and every exported file. Published AS-IS under the MIT License, without warranty or technical support. Read [DISCLAIMER.md](DISCLAIMER.md) first.

This manual explains what the archive contains, how to build and launch the application on Linux, how to obtain or rebuild the 1.3 GB Vault, what each of the twelve tabs and its controls do, which parts depend on the companion *Quantum Lab 649* (Python) archive, and how to diagnose problems.

---

## 1. What is in the box

| Item | Description |
|---|---|
| `src/` | Rust crate `rust-singularity-core-v3`: Tauri v2 GUI, Vault mapper (`vault.rs`), CK-V6 causal engine, KJ-V5 kernels, HS-V4 bridge, integrity/registry checks, procfs telemetry, CLI sub-commands (`main.rs`). |
| `cuda/v3_kernels.cu` | CUDA C++17 kernels (population counts with `__popcll`, Philox-4x32-10 streams, portfolio evaluators) compiled by `build.rs` through `nvcc` and linked statically. |
| `ui/` | The 12-tab dark-mode front-end (`index.html`, `app.js`, CSS) served inside the Tauri WebKitGTK webview. |
| `vault/VAULT_MANIFEST.json` | Names, byte sizes and SHA-256 of the 13 `.v3bin` tables (1,291.8 MB in total). The tables themselves are **not** in the zip (§4). |
| `scripts/download_vault.sh` | Downloads and SHA-256-verifies the 13 tables from CERN Zenodo (DOI 10.5281/zenodo.23992530), resolving the latest record automatically. |
| `launch_rust_core_v3.sh` | Builds on first use and launches the GUI; `--stop` closes it. |
| `desktop/install-desktop-shortcut.sh` | Installs the Linux `quantum-lab-649-rust-core.desktop` entry. |
| `Cargo.toml`, `Cargo.lock`, `build.rs`, `tauri.conf.json`, `capabilities/`, `icons/` | Build system and Tauri configuration. |
| `check-updates.sh`, `VERSION.json`, `CHANGELOG.md`, `LICENSE`, `DISCLAIMER.md`, `DONATE.md` | Release metadata and manual update check. |

---

## 2. Requirements

| Component | Requirement |
|---|---|
| Operating system | Linux x86-64 (tested on Ubuntu/Pop!_OS 22.04+). Windows/macOS are not supported: the build links CUDA and uses Linux procfs/`statvfs`. |
| GPU | **NVIDIA GPU is mandatory** (the binary links `libcudart`; GPU tabs upload up to 24 M states to VRAM). ≥ 6 GB VRAM recommended. |
| CUDA | CUDA Toolkit 12.x with `nvcc` on `PATH` (or `CUDA_HOME` / `NVCC` set) and a driver matching the toolkit. |
| Rust | 1.78 or newer (`rustup`), 2021 edition. |
| System libraries | `build-essential pkg-config libssl-dev libsqlite3-dev libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev` |
| CPU | x86-64 with AVX2 + FMA + POPCNT (any Intel/AMD since 2013). The CPU evaluators use every logical thread (24 on the reference machine). |
| RAM | 16 GB recommended. A resident-memory guard refuses work when the process would exceed **11.5 GiB** (`VmRSS`). |
| Disk | ~1.3 GB for the Vault, ~3–5 GB for the Rust/CUDA build artefacts in `target/`. |
| Companion data | The KJ-V5, HS-V4, International and registry tabs read files from the **Quantum Lab 649 (Python) archive** — see §5. |

---

## 3. Build and first launch

### 3.1 Download and verify

1. Download `quantum-lab-649-rust-core-opensource.zip` from [bot-tradingsoftware.com/#downloads](https://bot-tradingsoftware.com/#downloads) or the GitHub Release of [quantbotsoftware/quantum-lab-649-rust-core](https://github.com/quantbotsoftware/quantum-lab-649-rust-core).
2. `sha256sum -c SHA256SUMS.txt --ignore-missing`
3. Extract to a path without spaces, e.g. `~/apps/quantum-lab-649-rust-core/`.

### 3.2 Install prerequisites (Ubuntu/Debian example)

```bash
sudo apt-get install -y build-essential pkg-config libssl-dev libsqlite3-dev \
  libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh     # Rust toolchain
nvcc --version                                                      # CUDA 12.x must be present
```

### 3.3 Compile

Pick the compute capability of your GPU (`nvidia-smi --query-gpu=compute_cap --format=csv`):

| GPU family | `CUDA_ARCH` |
|---|---|
| RTX 20xx (Turing) | `sm_75` |
| RTX 30xx (Ampere) | `sm_86` |
| RTX 40xx (Ada) | `sm_89` |
| RTX 50xx (Blackwell) | `sm_120` |

```bash
cd ~/apps/quantum-lab-649-rust-core
CUDA_ARCH=sm_89 cargo build --release        # 5–15 minutes on first build
./target/release/rust-singularity-core-v3 --version
```

Build-time variables: `CUDA_ARCH` (default `sm_120`), `CUDA_HOME` (default `/usr/local/cuda`), `NVCC` (explicit compiler path). If the build cannot find `nvcc` it stops with a clear error — CUDA is not optional.

### 3.4 Launch

```bash
./launch_rust_core_v3.sh                     # builds if needed, then starts the GUI
./launch_rust_core_v3.sh --stop              # closes a running instance
./desktop/install-desktop-shortcut.sh        # optional desktop icon
```

The launcher `cd`s into the archive folder (all default paths are relative to it) and exports `WEBKIT_DISABLE_DMABUF_RENDERER=1`, which avoids blank windows on some NVIDIA/WebKitGTK combinations. A smoke-test mode exists for CI: `rust-singularity-core-v3 gui --smoke report.json` opens the window, runs the self-checks and exits.

---

## 4. The Vault (13 binary tables, 1.3 GB)

The Vault is the mmap-able universe the GPU tabs work on: every 6/49 mask (13,983,816), every Joker ticket (24,435,180), every 5/40 ticket (658,008) and 6-ball draw (3,838,380), the 1,000× bootstrap tables (PM-V3), the covering-wheel catalogues (CD-V3 / CD-V3Q), the exact Pareto portfolios (C3-v2), the 1 M-trajectory Monte-Carlo summary (C4-v2) and the CK-V6 10 k power table. Files are opened read-only (`0444`), checked for the `QLV3VLT` magic header, size and record count.

You have two ways to get it:

**Option A — download the sealed archive (recommended, identical bytes to the published research):**

```bash
./scripts/download_vault.sh
```

Downloads the 13 files from the latest Zenodo version into `./vault/`, verifying each SHA-256 against `vault/VAULT_MANIFEST.json`; already-valid files are skipped, so an interrupted download can simply be re-run. Variables: `QL649_VAULT_DIR` (target folder, default `./vault`), `QL649_VAULT_MIRROR_URL` (alternative mirror base URL). A browser alternative is the record page [doi.org/10.5281/zenodo.23992530](https://doi.org/10.5281/zenodo.23992530).

**Option B — rebuild the combinatorial tables locally (no download):**

```bash
./target/release/rust-singularity-core-v3 vault-init ./vault
```

`vault-init` enumerates the five combinatorial universes with all CPU threads and writes them in the same `.v3bin` format (≈ 350 MB, minutes to tens of minutes depending on the CPU). The derived research tables are then produced by the GPU pipelines, in this order: `a0-parity` → `cdv3-build` → `pmv3-build` → `c3v2-build` → `c4v2-build` (each has a matching `*-verify`). These runs take from minutes to hours on the GPU and write their JSON reports next to the Vault. Locally rebuilt files are mathematically equivalent but, for the bootstrap tables, not byte-identical to the Zenodo set unless the same seeds are used.

Without a Vault the GUI still opens; the Vault, Pareto and evaluator panels show *vault missing* errors instead of data.

---

## 5. Linking the Quantum Lab 649 (Python) archive

Several tabs read sealed archives, protocols and result files from the Python project (LIVE draw snapshots, ERA5 series, international feeds, HS-V4/KJ-V5 seals, TU4 tribunal). The binary looks for them under:

| Variable | Default (relative to the archive folder) | Expected content |
|---|---|---|
| `QL649_ROOT` | `./data/quantum_lab_649` | The extracted **quantum-lab-649-opensource.zip** (folders `tu3/`, `hs_v4/`, `kj_v5/`, `tu4/`, …). |
| `LOTO_RO_ROOT` | `./data` | A parent folder that contains `next_candidate/quantum_lab_649` → the same Python archive (used by the KJ-V5 CLI). |

Simplest setup — extract both archives side by side and create two symlinks:

```bash
cd ~/apps/quantum-lab-649-rust-core
mkdir -p data/next_candidate
ln -s ~/apps/quantum-lab-649 data/quantum_lab_649
ln -s ~/apps/quantum-lab-649 data/next_candidate/quantum_lab_649
```

Or export `QL649_ROOT=~/apps/quantum-lab-649` before launching. What then works:

| Tab | Needs Vault | Needs QL649 archive | Needs research outputs not shipped (`*/rezultate/`) |
|---|---|---|---|
| Vault · CD-V3Q · C3/C4 | yes | no | no |
| CK-V6 & 3 Live Registries | partly (`ckv6_power_10k`) | yes (`tu3/date/live`, `tu4/tribunal`) | registries report missing chain files as *lipsă* |
| Hardware Live | reads Vault files if present | no | no |
| KJ-V5 · Rust Nativ | no | yes | mostly (`kj_v5/rezultate/*.json`) — regenerate with `kjv5-qualify` / `kjv5-walkforward` |
| HS-V4 · RO Live & ERA5 | no | yes (`hs_v4/date_*`, `SIGILIU_HSV4_UI.json`) | `.npz` trajectories in `hs_v4/rezultate` (generated by the Python HS-V4 engine) |
| International & RNG | no | yes | KJ-V5 international JSON |
| Densities · FrFT · Admissibility · Parity · Inventory · Memory | no | yes for data-driven panels | partly |

Panels whose inputs are absent print an explicit Romanian error (`lipsă` = missing, `încă nu există` = does not exist yet) instead of inventing values. The missing research outputs are generated by the Python project's own scripts or can be requested as archived copies (see §9).

---

## 6. The twelve tabs

The interface language is Romanian (the protocols were sealed in Romanian); the controls are translated below. Numbers in the headings refer to the reference machine (RTX 5060 Laptop, i7-14650HX, 16 GB); your hardware is detected live.

### 6.1 🚀 Vault 39.07M · CD-V3Q · C3/C4
- **⚡ Inspector Instantaneu Zero-Copy** — `Joc` (game) selector, `Numere` (six numbers, space-separated), `Joker` (1–20), `Rang lex` (lexicographic rank). **Interoghează Bilet** looks a ticket up by its numbers; **Interoghează Rang** by its rank; both return the record in microseconds (the 8.4 μs figure on the card is measured live). **+ Adaugă în Eval GPU** appends the ticket to the custom portfolio of the evaluator; **🔍 Câte & Unde** sends it to the historical inspector in the HS-V4 tab.
- **Distribuția EV Partajat NB2** and **Calibrare Parimutuelă PM-V3** — bootstrap quantiles `[q05, q25, q50, q75, q95]` of the shared expected value under a negative-binomial crowding model (1,000 resamples, censored at 0).
- **🎡 Catalogul Roților de Acoperire** — `Catalog` = `CD-V3Q` (QAP-relabelled, anti-crowding) or `CD-V3` (canonical designs); `Roată` picks one of the 59 wheels. **⚡ Evaluează Roata Live** evaluates it on the GPU and the 24-thread CPU side by side; **📋 Fereastră Detalii Roată** opens a modal with every ticket of the wheel.
- **Cele 5 Designuri Integrale Principale** — the five full designs with zero uncovered states out of 42.25 M.
- **📈 Frontiere Pareto Exacte C3-v2** — `Joc` (6/49, 5/40, Joker or *Mix 3 Jocuri*), `Buget [lei]` (40 … 1,304 lei); the canvas draws the exact frontier (0 Bonferroni error). **⚡ Trimite μ=0.5 în Eval GPU** loads the μ = 0.5 portfolio into the evaluator.
- **🔥 Evaluator Interactiv LIVE** — `Joc evaluat` and a text area *Portofoliu personalizat* (one ticket per line; for Joker add the Joker number last). **🚀 Evaluează Live GPU + 24T CPU** runs both evaluators and shows the bit-exact parity between them (Philox-4x32-10 streams).

### 6.2 🛡️ CK-V6 & 3 Registre Live
- **📜 Poarta A6** — the three prospective registries (XI-F paper strategies, TU4 tribunal with the `S35_03` month-boundary guard, Quantum-Limit forecasts + `OGLINDA` mirror). **🔄 Reverifică Live Lanțurile SHA-256** recomputes every chain line by line. **🔒 Inspecție neinvazivă `lstat`** proves the blind reserves (`0400`) were never read (0 bytes).
- **🧬 Motorul Cauzal de Regimuri Fizice CK-V6** — `Flux analizat` (13 feeds, 45,206 steps) and `Celulă`; BOCPD (Adams–MacKay) + Koopman V₁⊕V₂ diagnostics, the pre-archive qualification Q1–Q6 and the Holm–Bonferroni family table.
- **🗺️ Suprafața de Putere 5×5** from `ckv6_power_10k.v3bin`.
- **⚡ Simulator Interactiv LIVE** — `Deplasare regim δ` (0 … 5,000 bp), `Schimbare L_regim` (step 40/80/120/200), `Replici GPU` (8/16/32 trajectories); **▶ Simulează** runs the Philox simulation on the GPU.

### 6.3 🖥️ Hardware Live
GPU (name, architecture, VRAM used/total), CPU (threads, AVX2/FMA/POPCNT flags), RAM with the active `VmRSS < 11.5 GiB` guard and the WebKitGTK helper processes, SSD free space (`statvfs`) and the table of the 13 `.v3bin` files with size, permissions, magic and mapping state. **🔄 Măsoară Acum** refreshes; the checkbox *Reîmprospătare automată (2 s)* keeps it live.

### 6.4 ⚡ KJ-V5 · Rust Nativ
`Joc RO` and `Configurație ablație` (`Full KJ-V5`, `K1 + K2`, `K1 pur` Krawtchouk envelopes, `K2 pur` Koopman). Shows the post-archive orthogonal multiplet (α, β, γ) and deterministic core, the Q1–Q6 and G1–G6 gates, the martingale trajectory E_s with its exact ≥ 0.50 guarantee, the Bayesian masses w₀/w_K1/w_K2/w_K3, π_i versus k/N, the RO ablation study, causal windows and hit categories versus the hypergeometric expectation. All computed natively in Rust on 24 threads from the LIVE archive.

### 6.5 ⚛ HS-V4 · RO Live & ERA5
`Joc RO`, `Configurație ablație` (`Full HS-V4`, `O1 + O2`, `O1 + O4`, `O1 pur`). Panels: Coulomb–Jastrow multiplet sealed by SHA-256, ERA5 Bucharest 18:00 telemetry with Welford causal scores, the `.npz` trajectory decoded natively (E_s ≥ 0.50), Bayesian portfolio masses, π_i and ablation, gates G1–G6 and hit categories.
**🔍 Inspector Istoric 1-Click „Câte & Unde”** — `Arhivă primară` (RO 6/49, 5/40, Joker or one of the international/RNG feeds), `Bile` (your numbers), `Bonus` (optional), `Min. potriviri` (≥ 2 … ≥ 5 balls), checkbox *include international*. **Verifică Câte & Unde** counts, after the fact, how often that ticket would have matched ≥ k balls across the whole archive and draws the exact histogram versus the uniform urn U. **Încarcă α KJ-V5 / Încarcă α HS-V4** paste the current α configuration into the fields. This is a descriptive audit — it says nothing about future draws.

### 6.6 🌍 Internațional & RNG
Comparative synthesis HS-V4 vs KJ-V5 over ten international feeds (five mechanical: FR Loto, HU Hatoslottó, HU Ötöslottó, US Powerball, HU Skandináv manual 7/35; five electronic RNG: FDJ EuroDreams, HU Skandináv machine 7/35, NY Take 5, NY Cash4Life, NY Pick 10), the paired natural experiment HU 7/35 (1,312 identical days, hand vs machine), the ERA5 negative control on pure RNG draws, compared martingale trajectories and the selected feed's post-archive multiplet (`Flux internațional detaliat`). **🔍 Verifică α KJ-V5 în Câte & Unde** hands the feed to the inspector.

### 6.7 Densități π · ρ
`Joc` and `Model` (`H1`/`H3`): π_i inclusion probabilities under the model law, the density operator ρ = Π/k (trace 1, PSD) as a deviation heat-map against the uniform urn, the unitary DFT spectrum |Z_m| and the scalar magnitudes.

### 6.8 FrFT
`Joc` and the slider `Ordin a` (0 … 2, step 0.01): fractional Fourier transform |F_a z| (Candan–Kutay–Ozaktas) of the label signal; a = 1 is the ordinary DFT.

### 6.9 Filtru admisibilitate
The native S17 admissibility filter (u64 masks, exact Shewchuk summation) applied to sealed emissions; `Țintă` lists the available targets.

### 6.10 Paritate & Porți v3
Synthesis of the acceptance gates of sessions S1–S5 (A0–A6), the S19 parity against the frozen Python `ip_core.py` (tolerance |Δ| ≤ 1e-12) and the independent Python ↔ Rust verification of the KJ-V5 engine.

### 6.11 Inventar LIVE
Inventory of the LIVE archives (async `tokio::fs` + `memmap2`): versions, manifests, SHA-256 and sizes.

### 6.12 Memorie & Proces
Window memory from procfs (`VmRSS`, `VmHWM`, `Pss`) against the 30 MB webview budget, continuous telemetry of the whole process tree and the live peak chart. **Remăsoară** refreshes.

---

## 7. Command-line interface

All sub-commands are run from the archive folder: `./target/release/rust-singularity-core-v3 <command> [args]`.

| Command | Purpose |
|---|---|
| *(none)* / `gui` | Start the desktop window. `gui --smoke <report.json>` = headless self-test. |
| `--version` | Print version, CUDA architecture and the compliance flag. |
| `vault-init [dir]` | Build or verify the combinatorial `.v3bin` tables (§4 option B). |
| `a0-parity [out.json]` | Gate A0: bit-exact GPU ↔ CPU parity over the whole universe. |
| `cdv3-build` / `cdv3-verify` | Build / verify the covering-wheel catalogues. |
| `pmv3-build` / `pmv3-verify` | Build / verify the 1,000× bootstrap tables. |
| `c3v2-build` / `c3v2-verify` | Build / verify the exact Pareto portfolios. |
| `c4v2-build` | Build the 1 M-trajectory Monte-Carlo summary. |
| `kjv5-qualify [out.json]` | Algebraic pre-archive qualification Q1–Q6 of KJ-V5 (reads nothing historical). |
| `kjv5-walkforward [--repo <dir>] [--game 649/540/joker]` | KJ-V5 causal walk-forward (writes `kj_v5/rezultate/*.json` in the QL649 archive). |
| `a6-certify [out.json]` | Gate A6: certify the three prospective registries line by line. |
| `s5-verify` | Session-5 acceptance verification. |
| `parity [--ref <file>] [--root <dir>] [--target <date>] [--s17 <file>]` | S19 parity against the frozen Python reference. |
| `admissibility [--enumerate] [--root <dir>] [--target <date>]` | Run the S17 admissibility filter. |
| `inventory` | Print the LIVE archive inventory as JSON. |
| `memory` / `monitor` | One-shot or continuous procfs telemetry. |

Exit codes: `0` success / all gates passed, `1` a gate failed, `2` error (missing input, I/O, CUDA).

---

## 8. Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `CUDA_ARCH`, `CUDA_HOME`, `NVCC` | `build.rs` | GPU architecture and CUDA toolkit location at compile time. |
| `QL649_ROOT` | GUI + CLI | Path of the Quantum Lab 649 Python archive (§5). |
| `LOTO_RO_ROOT` | KJ-V5 CLI | Parent folder containing `next_candidate/quantum_lab_649`. |
| `QL649_VAULT_DIR` | `download_vault.sh` | Where to store the downloaded Vault (the GUI expects `./vault`). |
| `QL649_VAULT_MIRROR_URL` | `download_vault.sh` | Alternative mirror base URL. |
| `WEBKIT_DISABLE_DMABUF_RENDERER=1` | launcher | Works around blank/black WebKitGTK windows on NVIDIA. |

---

## 9. Files not included and how to obtain them

| Item | Why it is not in the zip | How to get it |
|---|---|---|
| The 13 `.v3bin` Vault tables (1.3 GB) | size | `./scripts/download_vault.sh` (CERN Zenodo) or rebuild (§4). |
| `data/` (Python archive) | separate release | Download `quantum-lab-649-opensource.zip` and link it (§5). |
| `*/rezultate/` research outputs (KJ-V5, HS-V4, TU4, XI JSON/NPZ) | run artefacts of the sealed research sessions | Regenerate with the CLI (`kjv5-*`) and the Python engines, or request archived copies via a [GitHub documentation request](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/issues/new?template=documentation_request.md) or [contact@bot-tradingsoftware.com](mailto:contact@bot-tradingsoftware.com) — shared strictly AS-IS when available. |
| `./ssd_data` external SSD root | machine-specific | Any folder; only free space and the Vault location are reported. |

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nvcc: not found` or `CUDA_HOME` error during `cargo build` | Install CUDA Toolkit 12.x and export `CUDA_HOME=/usr/local/cuda` (and `PATH=$CUDA_HOME/bin:$PATH`). |
| `no kernel image is available for execution on the device` | `CUDA_ARCH` does not match your GPU. Rebuild with the right value (§3.3): `cargo clean && CUDA_ARCH=sm_86 cargo build --release`. |
| Window opens black or empty | Make sure you start through `launch_rust_core_v3.sh` (sets `WEBKIT_DISABLE_DMABUF_RENDERER=1`); update the NVIDIA driver; try `WEBKIT_DISABLE_COMPOSITING_MODE=1`. |
| `dimensiune fisier v3bin invalida` / `header v3bin neasteptat` | A Vault file is truncated or foreign. Delete it and re-run `download_vault.sh` (it re-fetches only invalid files). |
| SHA-256 mismatch reported by `download_vault.sh` | Corrupted download or wrong mirror; delete the file and retry. The manifest hashes are the ground truth. |
| Panels say `lipsă`, `încă nu există`, `CURRENT_LIVE fără version_dir` | `QL649_ROOT` is not set / linked, or the result files are not shipped (§5, §9). |
| `RSS guard` error | The process would exceed 11.5 GiB resident memory. Close other GPU/RAM-heavy programs or run the CLI builders one at a time. |
| GUI tab shows `vault missing` | Download or build the Vault (§4) into `./vault` relative to the archive folder. |
| Build is slow / `target/` is huge | Normal for Tauri + CUDA; a release build is 5–15 min and 3–5 GB. `cargo clean` frees the space. |

---

## 11. Legal and scope reminder

- MIT License: free to use, modify and redistribute, keeping the copyright and license notice.
- `positive_claims_allowed = false`: no screen, file or command presents a ticket as more likely to win. Covering designs and Pareto portfolios optimise *coverage and cost under a model*, never the odds of the draw — every ticket has the same probability.
- No gambling, financial or legal advice; no technical support. For *existing* documents only: [GitHub documentation request](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/issues/new?template=documentation_request.md) · [contact@bot-tradingsoftware.com](mailto:contact@bot-tradingsoftware.com).
- Permanent archive, Vault and citation: CERN Zenodo DOI [10.5281/zenodo.23992530](https://doi.org/10.5281/zenodo.23992530). Vault details: [VAULT_DATASET_GUIDE](https://bot-tradingsoftware.com/docs/VAULT_DATASET_GUIDE.html).

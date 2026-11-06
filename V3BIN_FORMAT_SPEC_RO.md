# `.v3bin` — Format binar zero-copy & Vault-ul combinatoric de 1,3 GB (specificație de inginerie a datelor)

> **Public:** ingineri de date, dezvoltatori Rust/CUDA/NumPy.
> **Pe scurt:** 13 tabele plate, little-endian, `mmap`-abile, cu header de 64 de octeți: 39.077.004 măști de k-submulțimi enumerate exhaustiv, statistici bootstrap, acoperiri (wheels) și portofolii Pareto — reproductibile din sursă, fixate prin SHA-256, identice la bit între GPU și CPU, arhivate cu DOI. Fiecare tabel se citește din Python într-o linie cu `numpy.memmap`; aplicația de loterie care le-a produs este irelevantă pentru folosirea lor ca set de date.
>
> Arhivă permanentă: [https://doi.org/10.5281/zenodo.23992530](https://doi.org/10.5281/zenodo.23992530) (CERN Zenodo, concept DOI → ultima versiune). Sursă: [github.com/quantbotsoftware/quantum-lab-649-rust-core](https://github.com/quantbotsoftware/quantum-lab-649-rust-core).

## 1. De ce un format propriu

Motorul trebuie să parcurgă **întregul univers de bilete** al mai multor jocuri (C(49,6) = 13.983.816 · C(40,5) = 658.008 · C(40,6) = 3.838.380 · C(45,5)×20 = 24.435.180) la fiecare evaluare, atât pe calea CPU AVX2/POPCNT cu 24 de fire, cât și într-un kernel CUDA, fără deserializare, alocare sau copiere. Cerințele care au exclus JSON/CSV/Parquet/Arrow pe calea critică:

| Cerință | Decizie de proiectare |
|---|---|
| Zero-copy la deschidere, acces aleator O(1) | Înregistrări de dimensiune fixă, header de 64 B, `memmap2::Mmap` (Rust) / `numpy.memmap` (Python) |
| Aceiași octeți consumați de CPU și GPU | Înregistrările sunt `u64` / `f32` little-endian simple, urcate în VRAM ca atare prin DMA cu memorie pinned (blocuri de 64 MiB) |
| Integritate verificabilă, fără corupere silențioasă | Header-ul conține kind/n/k/count/dimensiune înregistrare; lungimea fișierului trebuie să fie exact `64 + count × record_bytes`; SHA-256 al fiecărui fișier este fixat în `VAULT_MANIFEST.json` |
| Reproductibil | Fiecare tabel este regenerat determinist de `rust-singularity-core-v3 vault-init` (enumerare lexicografică, partiționată după primul element, astfel încât ordinea globală rămâne exactă și în paralel) |
| Imuabil după generare | Fișierele se scriu o singură dată și rămân `0444`; observatorul raportează `is_0444` + `header_valid` pentru fiecare tabel, deci orice modificare devine vizibilă |

## 2. Header (64 de octeți, little-endian)

| Offset | Dim. | Tip | Câmp | Valoare / semnificație |
|---|---|---|---|---|
| 0 | 8 | bytes | `magic` | `51 4C 56 33 56 4C 54 01` = `"QLV3VLT\x01"` |
| 8 | 4 | u32 | `version` | `3` |
| 12 | 4 | u32 | `table_kind` | vezi §3 |
| 16 | 4 | u32 | `n_pool` | mărimea urnei (49, 40, 45 …) |
| 20 | 4 | u32 | `k_pick` | numere per bilet / extragere (5, 6 …) |
| 24 | 8 | u64 | `record_count` | numărul de înregistrări |
| 32 | 8 | u64 | `record_bytes` | octeți per înregistrare (8, 24, 64, 128) |
| 40 | 24 | bytes | rezervat | toți zero |

Invariant: `dimensiune_fișier == 64 + record_count × record_bytes`. Cititorii trebuie să respingă orice fișier care încalcă magic-ul, versiunea sau dimensiunea (implementarea de referință o face: `vault::verify_header`).

## 3. Tipuri de tabele & cele 13 fișiere

| # | Fișier | kind | n | k | înregistrări | B/înreg. | Conținut |
|---|---|---|---|---|---|---|---|
| 1 | `loto_649_masks.v3bin` | 1 | 49 | 6 | 13.983.816 | 8 | fiecare 6-submulțime din {1..49} ca mască `u64`, lexicografic |
| 2 | `loto_540_tickets.v3bin` | 1 | 40 | 5 | 658.008 | 8 | fiecare 5-submulțime din {1..40} |
| 3 | `loto_540_draws6.v3bin` | 2 | 40 | 6 | 3.838.380 | 8 | fiecare 6-submulțime din {1..40} (spațiul extragerilor 5/40) |
| 4 | `joker_base45_masks.v3bin` | 1 | 45 | 5 | 1.221.759 | 8 | fiecare 5-submulțime din {1..45} |
| 5 | `joker_full_24m.v3bin` | 3 | 45 | 5 | 24.435.180 | 8 | 5-submulțime pe biții 0..44 **+ bila bonus (1..20) − 1 pe biții 48..52** |
| 6 | `cdv3_wheels_catalog.v3bin` | 5 | — | — | 6.926 | 8 | catalog de acoperiri („wheels”), CD-V3 |
| 7 | `cdv3q_wheels_catalog.v3bin` | 6 | — | — | 6.926 | 8 | varianta CD-V3Q |
| 8 | `pmv3_boot1000_649.v3bin` | 4 | 49 | 6 | 13.983.816 | 24 | per bilet `6 × f32`: `[ev, q05, q25, q50, q75, q95]` dintr-un bootstrap cu 1.000 de replici |
| 9 | `pmv3_boot1000_540.v3bin` | 4 | 40 | 5 | 658.008 | 24 | idem, 5/40 |
| 10 | `pmv3_boot1000_joker.v3bin` | 4 | 45 | 5 | 24.435.180 | 24 | idem, Joker (586 MB — cel mai mare tabel) |
| 11 | `c3v2_pareto_portfolios.v3bin` | 7 | — | — | 13.945 | 8 | portofolii de pe frontiera Pareto exactă, C3-V2 |
| 12 | `c4v2_mc1m_summary.v3bin` | 8 | 49 | 6 | 90 | 128 | rânduri sumar Monte-Carlo (1M), C4-V2 |
| 13 | `ckv6_power_10k.v3bin` | 9 | — | — | 10.000 | 64 | traiectorii de putere cauzală, CK-V6 |

Total: **1.291,8 MB**, 39.077.004 înregistrări-bilet în tipurile 1–3. Dimensiunile exacte și SHA-256 pentru fiecare fișier: [`vault/VAULT_MANIFEST.json`](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/blob/main/vault/VAULT_MANIFEST.json).

**Codificarea măștilor (tipurile 1–3):** bitul `i−1` setat ⇔ numărul `i` este în bilet. Popcount-ul fiecărei înregistrări este `k`. Enumerarea este lexicografică, ceea ce face fișierul *și* un hash perfect: rangul unui bilet este indexul său (sistemul numeric combinatoric), deci „bilet → rând” este aritmetică O(k), fără căutare.

## 4. Citirea tabelelor fără aplicație

### Python / NumPy (zero-copy)

```python
import numpy as np, struct
def open_v3bin(path):
    hdr = open(path, "rb").read(64)
    magic, ver, kind, n, k, count, rec = struct.unpack_from("<8sIIIIQQ", hdr, 0)
    assert magic == b"QLV3VLT\x01" and ver == 3
    dtype = {8: "<u8", 24: ("<f4", 6), 64: ("<u8", 8), 128: ("<u8", 16)}[rec]
    return np.memmap(path, dtype=dtype, mode="r", offset=64, shape=(count,)), (kind, n, k)

masks, (kind, n, k) = open_v3bin("vault/loto_649_masks.v3bin")
balls = [i + 1 for i in range(n) if (int(masks[0]) >> i) & 1]      # → [1, 2, 3, 4, 5, 6]
ev, (_, _, _) = open_v3bin("vault/pmv3_boot1000_649.v3bin")        # formă (13983816, 6) float32
```

Măsurat pe laptopul de referință (NVMe): deschidere `np.memmap` 0,2 ms; verificarea popcount pe toate cele 13,98 M măști 0,55 s în NumPy pur.

### Rust (implementarea de referință)

`memmap2::MmapOptions::new().map(&file)` → `verify_header(&mmap, kind, n, k, count)` → reinterpretarea `&mmap[64..]` ca `&[u64]`. Vezi `src/vault.rs` (`MappedVaultUniverse::open_or_build`) și `src/v3_observatory.rs` (`ALL_13_V3BIN_FILES`).

### Verificarea integrității

```bash
cd vault && sha256sum -c <(python3 -c "import json;[print(f['sha256'],' ',f['filename']) for f in json.load(open('VAULT_MANIFEST.json'))['files']]")
```

## 5. Poarta A0 — paritate GPU ⇄ CPU exactă la bit, măsurată

`rust-singularity-core-v3 a0-parity` mapează Vault-ul, urcă prin DMA cele 39 M de măști în VRAM (353 MB rezidente), rulează kernel-urile popcount/potrivire pe GPU și calea AVX2/POPCNT pe toate firele CPU, apoi compară rezultatele bit cu bit. Rulare reală, 2026-11-06, Intel i7-14650HX (24 fire) + RTX 5060 Laptop (sm_120, 26 SM, 8 GB):

| Tabel | GPU (ms) | CPU 24T (ms) | Diferență zero |
|---|---|---|---|
| Măști 6/49 (13,98 M) | 7,66 | 8,99 | ✅ |
| 5/40 draws6 (3,84 M) | 2,20 | 1,86 | ✅ |
| Joker complet (24,44 M) | 14,53 | 12,96 | ✅ |

Încărcare VRAM 45 ms pentru 353 MB prin blocuri pinned de 64 MiB · RSS proces 616 MB (limită gardă 11,5 GiB) · debit GPU 13,9 G op/s.

**Citire onestă:** pentru această sarcină popcount limitată de memorie, un CPU modern cu 24 de fire și POPCNT *egalează* un GPU de laptop — calea GPU există pentru kernel-urile mai grele PM-V3 / CK-V6, iar scopul Porții A0 este **paritatea** (aceiași octeți dau aceleași răspunsuri pe ambele dispozitive), nu o pretenție de accelerare. `positive_claims_allowed` este fixat pe `false` în fiecare raport scris de motor.

## 6. Reproducerea Vault-ului din sursă

```bash
./scripts/download_vault.sh          # 1,3 GB de pe Zenodo (ultima versiune rezolvată la rulare), verificat SHA-256
# sau regenerează tabelele local (enumerare lexicografică paralelă; tabelele bootstrap PM-V3 sunt partea lentă):
cargo build --release && ./target/release/rust-singularity-core-v3 vault-init
```

Cerințe de build și suprascrierea arhitecturii CUDA: vezi [`QUANTUM_RUST_CORE_V3_USER_MANUAL_RO.md`](QUANTUM_RUST_CORE_V3_USER_MANUAL_RO.md).

## 7. Ce *nu* este

Tabelele enumeră spații combinatorice și stochează statistici calculate din extrageri istorice. **Nu prezic — și matematic nu pot prezice — extragerile viitoare.** Sunt publicate ca artefact de inginerie reproductibil (format, lanț de integritate, harnașament de paritate GPU/CPU, build HPC) sub MIT, strict AS-IS.

# Quantum Lab 649 · Rust Core v3 — Manual de utilizare (Română)

> **Notă privind domeniul.** Rust Core v3 este un *observator de cercetare HPC* nativ pentru desktop (Rust + Tauri v2 + CUDA C++17 + AVX2) peste universurile combinatorice complete ale jocurilor Loto 6/49, 5/40 și Joker (39,07 milioane de bilete într-un Vault `mmap` zero-copy). Calculează designuri de acoperire, frontiere Pareto exacte, calibrări bootstrap, descompuneri spectrale și verificări de paritate bit-exactă GPU/CPU. **Nu prezice numere la loterie, nu oferă niciun avantaj și nu garantează nimic** — `positive_claims_allowed = false` este încorporat în titlul ferestrei, în fiecare ecran și în fiecare fișier exportat. Publicat AS-IS sub licența MIT, fără garanție și fără suport tehnic. Citește mai întâi [DISCLAIMER.md](DISCLAIMER.md).

Acest manual explică ce conține arhiva, cum compilezi și pornești aplicația pe Linux, cum obții sau reconstruiești Vault-ul de 1,3 GB, ce face fiecare dintre cele douăsprezece tab-uri și controalele lor, ce părți depind de arhiva-pereche *Quantum Lab 649* (Python) și cum diagnostichezi problemele.

---

## 1. Ce conține arhiva

| Element | Descriere |
|---|---|
| `src/` | Crate-ul Rust `rust-singularity-core-v3`: interfața Tauri v2, maparea Vault-ului (`vault.rs`), motorul cauzal CK-V6, nucleele KJ-V5, puntea HS-V4, verificările de integritate/registre, telemetria procfs, subcomenzile CLI (`main.rs`). |
| `cuda/v3_kernels.cu` | Nuclee CUDA C++17 (numărare de biți cu `__popcll`, fluxuri Philox-4x32-10, evaluatoare de portofoliu) compilate de `build.rs` prin `nvcc` și legate static. |
| `ui/` | Interfața cu 12 tab-uri în mod întunecat (`index.html`, `app.js`, CSS) servită în webview-ul WebKitGTK al Tauri. |
| `vault/VAULT_MANIFEST.json` | Numele, dimensiunile în octeți și SHA-256 ale celor 13 tabele `.v3bin` (1.291,8 MB în total). Tabelele în sine **nu** sunt în zip (§4). |
| `scripts/download_vault.sh` | Descarcă și verifică SHA-256 cele 13 tabele de pe CERN Zenodo (DOI 10.5281/zenodo.23992530), rezolvând automat cea mai nouă versiune. |
| `launch_rust_core_v3.sh` | Compilează la prima utilizare și pornește interfața; `--stop` o închide. |
| `desktop/install-desktop-shortcut.sh` | Instalează intrarea Linux `quantum-lab-649-rust-core.desktop`. |
| `Cargo.toml`, `Cargo.lock`, `build.rs`, `tauri.conf.json`, `capabilities/`, `icons/` | Sistemul de build și configurația Tauri. |
| `check-updates.sh`, `VERSION.json`, `CHANGELOG.md`, `LICENSE`, `DISCLAIMER.md`, `DONATE.md` | Metadatele versiunii și verificarea manuală a actualizărilor. |

---

## 2. Cerințe

| Componentă | Cerință |
|---|---|
| Sistem de operare | Linux x86-64 (testat pe Ubuntu/Pop!_OS 22.04+). Windows/macOS nu sunt suportate: build-ul leagă CUDA și folosește procfs/`statvfs` din Linux. |
| GPU | **GPU NVIDIA obligatoriu** (binarul leagă `libcudart`; tab-urile GPU încarcă până la 24 M de stări în VRAM). Recomandat ≥ 6 GB VRAM. |
| CUDA | CUDA Toolkit 12.x cu `nvcc` în `PATH` (sau `CUDA_HOME` / `NVCC` setate) și un driver compatibil cu toolkit-ul. |
| Rust | 1.78 sau mai nou (`rustup`), ediția 2021. |
| Biblioteci de sistem | `build-essential pkg-config libssl-dev libsqlite3-dev libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev` |
| CPU | x86-64 cu AVX2 + FMA + POPCNT (orice Intel/AMD din 2013 încoace). Evaluatoarele CPU folosesc toate firele logice (24 pe mașina de referință). |
| RAM | 16 GB recomandat. O gardă de memorie rezidentă refuză lucrul când procesul ar depăși **11,5 GiB** (`VmRSS`). |
| Disc | ~1,3 GB pentru Vault, ~3–5 GB pentru artefactele de build Rust/CUDA din `target/`. |
| Date-pereche | Tab-urile KJ-V5, HS-V4, Internațional și registrele citesc fișiere din **arhiva Quantum Lab 649 (Python)** — vezi §5. |

---

## 3. Compilare și prima pornire

### 3.1 Descărcare și verificare

1. Descarcă `quantum-lab-649-rust-core-opensource.zip` de pe [bot-tradingsoftware.com/#downloads](https://bot-tradingsoftware.com/#downloads) sau din GitHub Release [quantbotsoftware/quantum-lab-649-rust-core](https://github.com/quantbotsoftware/quantum-lab-649-rust-core).
2. `sha256sum -c SHA256SUMS.txt --ignore-missing`
3. Dezarhivează într-o cale fără spații, de ex. `~/apps/quantum-lab-649-rust-core/`.

### 3.2 Instalarea cerințelor (exemplu Ubuntu/Debian)

```bash
sudo apt-get install -y build-essential pkg-config libssl-dev libsqlite3-dev \
  libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh     # toolchain-ul Rust
nvcc --version                                                      # CUDA 12.x trebuie să fie prezent
```

### 3.3 Compilare

Alege capabilitatea de calcul a GPU-ului tău (`nvidia-smi --query-gpu=compute_cap --format=csv`):

| Familie GPU | `CUDA_ARCH` |
|---|---|
| RTX 20xx (Turing) | `sm_75` |
| RTX 30xx (Ampere) | `sm_86` |
| RTX 40xx (Ada) | `sm_89` |
| RTX 50xx (Blackwell) | `sm_120` |

```bash
cd ~/apps/quantum-lab-649-rust-core
CUDA_ARCH=sm_89 cargo build --release        # 5–15 minute la prima compilare
./target/release/rust-singularity-core-v3 --version
```

Variabile la compilare: `CUDA_ARCH` (implicit `sm_120`), `CUDA_HOME` (implicit `/usr/local/cuda`), `NVCC` (calea explicită a compilatorului). Dacă build-ul nu găsește `nvcc`, se oprește cu o eroare clară — CUDA nu este opțional.

### 3.4 Pornire

```bash
./launch_rust_core_v3.sh                     # compilează dacă e nevoie, apoi pornește interfața
./launch_rust_core_v3.sh --stop              # închide o instanță pornită
./desktop/install-desktop-shortcut.sh        # pictogramă de desktop opțională
```

Lansatorul face `cd` în folderul arhivei (toate căile implicite sunt relative la el) și exportă `WEBKIT_DISABLE_DMABUF_RENDERER=1`, care evită ferestrele goale pe unele combinații NVIDIA/WebKitGTK. Există și un mod de test pentru CI: `rust-singularity-core-v3 gui --smoke raport.json` deschide fereastra, rulează autoverificările și iese.

---

## 4. Vault-ul (13 tabele binare, 1,3 GB)

Vault-ul este universul mapabil în memorie pe care lucrează tab-urile GPU: fiecare mască 6/49 (13.983.816), fiecare bilet Joker (24.435.180), fiecare bilet 5/40 (658.008) și fiecare extragere de 6 bile (3.838.380), tabelele bootstrap 1.000× (PM-V3), cataloagele de roți de acoperire (CD-V3 / CD-V3Q), portofoliile Pareto exacte (C3-v2), sinteza Monte-Carlo pe 1 M de traiectorii (C4-v2) și tabelul de putere CK-V6 10 k. Fișierele se deschid numai-citire (`0444`) și sunt verificate după antetul magic `QLV3VLT`, dimensiune și numărul de înregistrări.

Ai două căi de a-l obține:

**Opțiunea A — descarcă arhiva sigilată (recomandat, octeți identici cu cercetarea publicată):**

```bash
./scripts/download_vault.sh
```

Descarcă cele 13 fișiere din cea mai nouă versiune Zenodo în `./vault/`, verificând fiecare SHA-256 față de `vault/VAULT_MANIFEST.json`; fișierele deja valide sunt sărite, deci o descărcare întreruptă se reia pur și simplu rulând din nou comanda. Variabile: `QL649_VAULT_DIR` (folderul țintă, implicit `./vault`), `QL649_VAULT_MIRROR_URL` (URL de bază al unei oglinzi alternative). Alternativa din browser este pagina înregistrării [doi.org/10.5281/zenodo.23992530](https://doi.org/10.5281/zenodo.23992530).

**Opțiunea B — reconstruiește local tabelele combinatorice (fără descărcare):**

```bash
./target/release/rust-singularity-core-v3 vault-init ./vault
```

`vault-init` enumeră cele cinci universuri combinatorice pe toate firele CPU și le scrie în același format `.v3bin` (≈ 350 MB, de la minute la zeci de minute în funcție de procesor). Tabelele de cercetare derivate sunt apoi produse de pipeline-urile GPU, în această ordine: `a0-parity` → `cdv3-build` → `pmv3-build` → `c3v2-build` → `c4v2-build` (fiecare are și un `*-verify`). Aceste rulări durează de la minute la ore pe GPU și scriu rapoartele JSON lângă Vault. Fișierele reconstruite local sunt echivalente matematic, dar, pentru tabelele bootstrap, nu sunt identice octet cu octet cu setul Zenodo decât dacă se folosesc aceleași semințe.

Fără Vault interfața pornește totuși; panourile Vault, Pareto și evaluator afișează erori de tip *vault lipsă* în loc de date.

---

## 5. Legarea arhivei Quantum Lab 649 (Python)

Mai multe tab-uri citesc arhive sigilate, protocoale și fișiere de rezultate din proiectul Python (instantaneele LIVE ale extragerilor, seriile ERA5, fluxurile internaționale, sigiliile HS-V4/KJ-V5, tribunalul TU4). Binarul le caută în:

| Variabilă | Implicit (relativ la folderul arhivei) | Conținut așteptat |
|---|---|---|
| `QL649_ROOT` | `./data/quantum_lab_649` | Arhiva dezarhivată **quantum-lab-649-opensource.zip** (folderele `tu3/`, `hs_v4/`, `kj_v5/`, `tu4/`, …). |
| `LOTO_RO_ROOT` | `./data` | Un folder-părinte care conține `next_candidate/quantum_lab_649` → aceeași arhivă Python (folosit de CLI-ul KJ-V5). |

Configurația cea mai simplă — dezarhivezi ambele arhive una lângă alta și creezi două legături simbolice:

```bash
cd ~/apps/quantum-lab-649-rust-core
mkdir -p data/next_candidate
ln -s ~/apps/quantum-lab-649 data/quantum_lab_649
ln -s ~/apps/quantum-lab-649 data/next_candidate/quantum_lab_649
```

Sau exportă `QL649_ROOT=~/apps/quantum-lab-649` înainte de pornire. Ce funcționează apoi:

| Tab | Are nevoie de Vault | Are nevoie de arhiva QL649 | Are nevoie de rezultate de cercetare nelivrate (`*/rezultate/`) |
|---|---|---|---|
| Vault · CD-V3Q · C3/C4 | da | nu | nu |
| CK-V6 & 3 Registre Live | parțial (`ckv6_power_10k`) | da (`tu3/date/live`, `tu4/tribunal`) | registrele raportează fișierele de lanț absente ca *lipsă* |
| Hardware Live | citește fișierele Vault dacă există | nu | nu |
| KJ-V5 · Rust Nativ | nu | da | în mare parte (`kj_v5/rezultate/*.json`) — se regenerează cu `kjv5-qualify` / `kjv5-walkforward` |
| HS-V4 · RO Live & ERA5 | nu | da (`hs_v4/date_*`, `SIGILIU_HSV4_UI.json`) | traiectoriile `.npz` din `hs_v4/rezultate` (generate de motorul HS-V4 din Python) |
| Internațional & RNG | nu | da | JSON-ul internațional KJ-V5 |
| Densități · FrFT · Admisibilitate · Paritate · Inventar · Memorie | nu | da, pentru panourile bazate pe date | parțial |

Panourile ale căror intrări lipsesc afișează o eroare explicită (`lipsă`, `încă nu există`) în loc să inventeze valori. Rezultatele de cercetare lipsă se generează cu scripturile proprii ale proiectului Python sau se pot cere drept copii arhivate (vezi §9).

---

## 6. Cele douăsprezece tab-uri

Cifrele din titluri se referă la mașina de referință (RTX 5060 Laptop, i7-14650HX, 16 GB); hardware-ul tău este detectat în direct.

### 6.1 🚀 Vault 39.07M · CD-V3Q · C3/C4
- **⚡ Inspector Instantaneu Zero-Copy** — selectorul `Joc`, `Numere` (șase numere separate prin spațiu), `Joker` (1–20), `Rang lex` (rangul lexicografic). **Interoghează Bilet** caută un bilet după numerele lui; **Interoghează Rang** după rang; ambele returnează înregistrarea în microsecunde (valoarea de 8,4 μs de pe card este măsurată în direct). **+ Adaugă în Eval GPU** adaugă biletul în portofoliul personalizat al evaluatorului; **🔍 Câte & Unde** îl trimite inspectorului istoric din tab-ul HS-V4.
- **Distribuția EV Partajat NB2** și **Calibrare Parimutuelă PM-V3** — cuantilele bootstrap `[q05, q25, q50, q75, q95]` ale valorii așteptate partajate sub un model de aglomerare binomial-negativ (1.000 de reeșantionări, cenzurare la 0).
- **🎡 Catalogul Roților de Acoperire** — `Catalog` = `CD-V3Q` (reetichetat QAP, anti-aglomerare) sau `CD-V3` (designuri canonice); `Roată` alege una dintre cele 59 de roți. **⚡ Evaluează Roata Live** o evaluează pe GPU și pe CPU-ul cu 24 de fire, în paralel; **📋 Fereastră Detalii Roată** deschide o fereastră cu fiecare bilet al roții.
- **Cele 5 Designuri Integrale Principale** — cele cinci designuri complete cu zero stări neacoperite din 42,25 M.
- **📈 Frontiere Pareto Exacte C3-v2** — `Joc` (6/49, 5/40, Joker sau *Mix 3 Jocuri*), `Buget [lei]` (40 … 1.304 lei); graficul desenează frontiera exactă (0 eroare Bonferroni). **⚡ Trimite μ=0.5 în Eval GPU** încarcă portofoliul μ = 0,5 în evaluator.
- **🔥 Evaluator Interactiv LIVE** — `Joc evaluat` și o zonă de text *Portofoliu personalizat* (un bilet pe linie; pentru Joker adaugă numărul Joker la final). **🚀 Evaluează Live GPU + 24T CPU** rulează ambele evaluatoare și arată paritatea bit-exactă dintre ele (fluxuri Philox-4x32-10).

### 6.2 🛡️ CK-V6 & 3 Registre Live
- **📜 Poarta A6** — cele trei registre prospective (strategiile pe hârtie XI-F, tribunalul TU4 cu garda `S35_03` la trecerea de lună, prognozele Quantum-Limit + oglinda `OGLINDA`). **🔄 Reverifică Live Lanțurile SHA-256** recalculează fiecare lanț linie cu linie. **🔒 Inspecție neinvazivă `lstat`** dovedește că rezervele oarbe (`0400`) nu au fost citite niciodată (0 octeți).
- **🧬 Motorul Cauzal de Regimuri Fizice CK-V6** — `Flux analizat` (13 fluxuri, 45.206 pași) și `Celulă`; diagnostice BOCPD (Adams–MacKay) + Koopman V₁⊕V₂, calificarea pre-arhivă Q1–Q6 și tabelul familiei Holm–Bonferroni.
- **🗺️ Suprafața de Putere 5×5** din `ckv6_power_10k.v3bin`.
- **⚡ Simulator Interactiv LIVE** — `Deplasare regim δ` (0 … 5.000 bp), `Schimbare L_regim` (pasul 40/80/120/200), `Replici GPU` (8/16/32 traiectorii); **▶ Simulează** rulează simularea Philox pe GPU.

### 6.3 🖥️ Hardware Live
GPU (nume, arhitectură, VRAM folosit/total), CPU (fire, indicatorii AVX2/FMA/POPCNT), RAM cu garda activă `VmRSS < 11.5 GiB` și procesele ajutătoare WebKitGTK, spațiul liber pe SSD (`statvfs`) și tabelul celor 13 fișiere `.v3bin` cu dimensiune, permisiuni, antet magic și starea mapării. **🔄 Măsoară Acum** reîmprospătează; bifa *Reîmprospătare automată (2 s)* îl menține în direct.

### 6.4 ⚡ KJ-V5 · Rust Nativ
`Joc RO` și `Configurație ablație` (`Full KJ-V5`, `K1 + K2`, `K1 pur` învelișuri Krawtchouk, `K2 pur` Koopman). Afișează multipletul ortogonal post-arhivă (α, β, γ) și nucleul determinist, porțile Q1–Q6 și G1–G6, traiectoria martingală E_s cu garanția exactă ≥ 0,50, masele bayesiene w₀/w_K1/w_K2/w_K3, π_i față de k/N, studiul de ablație RO, ferestrele cauzale și categoriile de potriviri față de așteptarea hipergeometrică. Totul calculat nativ în Rust pe 24 de fire, din arhiva LIVE.

### 6.5 ⚛ HS-V4 · RO Live & ERA5
`Joc RO`, `Configurație ablație` (`Full HS-V4`, `O1 + O2`, `O1 + O4`, `O1 pur`). Panouri: multipletul Coulomb–Jastrow sigilat SHA-256, telemetria ERA5 București 18:00 cu scoruri Welford cauzale, traiectoria `.npz` decodificată nativ (E_s ≥ 0,50), masele portofoliului bayesian, π_i și ablația, porțile G1–G6 și categoriile de potriviri.
**🔍 Inspector Istoric 1-Click „Câte & Unde”** — `Arhivă primară` (RO 6/49, 5/40, Joker sau unul dintre fluxurile internaționale/RNG), `Bile` (numerele tale), `Bonus` (opțional), `Min. potriviri` (≥ 2 … ≥ 5 bile), bifa *include internațional*. **Verifică Câte & Unde** numără, după fapt, cât de des acel bilet ar fi potrivit ≥ k bile pe întreaga arhivă și desenează histograma exactă față de urna uniformă U. **Încarcă α KJ-V5 / Încarcă α HS-V4** copiază configurația α curentă în câmpuri. Este un audit descriptiv — nu spune nimic despre extragerile viitoare.

### 6.6 🌍 Internațional & RNG
Sinteza comparativă HS-V4 vs KJ-V5 pe zece fluxuri internaționale (cinci mecanice: FR Loto, HU Hatoslottó, HU Ötöslottó, US Powerball, HU Skandináv manual 7/35; cinci RNG electronice: FDJ EuroDreams, HU Skandináv mașină 7/35, NY Take 5, NY Cash4Life, NY Pick 10), experimentul natural pereche HU 7/35 (1.312 zile identice, manual vs mașină), controlul negativ ERA5 pe extrageri pur RNG, traiectoriile martingale comparate și multipletul post-arhivă al fluxului selectat (`Flux internațional detaliat`). **🔍 Verifică α KJ-V5 în Câte & Unde** predă fluxul inspectorului.

### 6.7 Densități π · ρ
`Joc` și `Model` (`H1`/`H3`): probabilitățile de includere π_i sub legea modelului, operatorul densitate ρ = Π/k (urma 1, PSD) ca hartă termică a abaterii față de urna uniformă, spectrul DFT unitar |Z_m| și mărimile scalare.

### 6.8 FrFT
`Joc` și glisorul `Ordin a` (0 … 2, pas 0,01): transformata Fourier fracționară |F_a z| (Candan–Kutay–Ozaktas) a semnalului de etichete; a = 1 este DFT-ul obișnuit.

### 6.9 Filtru admisibilitate
Filtrul nativ de admisibilitate S17 (măști u64, sumare exactă Shewchuk) aplicat emisiilor sigilate; `Țintă` listează țintele disponibile.

### 6.10 Paritate & Porți v3
Sinteza porților de acceptare ale sesiunilor S1–S5 (A0–A6), paritatea S19 față de `ip_core.py` din Python, înghețat (toleranță |Δ| ≤ 1e-12), și verificarea independentă Python ↔ Rust a motorului KJ-V5.

### 6.11 Inventar LIVE
Inventarul arhivelor LIVE (`tokio::fs` asincron + `memmap2`): versiuni, manifeste, SHA-256 și dimensiuni.

### 6.12 Memorie & Proces
Memoria ferestrei din procfs (`VmRSS`, `VmHWM`, `Pss`) față de bugetul de 30 MB al webview-ului, telemetria continuă a întregului arbore de procese și graficul vârfurilor în direct. **Remăsoară** reîmprospătează.

---

## 7. Interfața în linie de comandă

Toate subcomenzile se rulează din folderul arhivei: `./target/release/rust-singularity-core-v3 <comandă> [argumente]`.

| Comandă | Scop |
|---|---|
| *(nimic)* / `gui` | Pornește fereastra desktop. `gui --smoke <raport.json>` = autotest fără interacțiune. |
| `--version` | Tipărește versiunea, arhitectura CUDA și indicatorul de conformitate. |
| `vault-init [dir]` | Construiește sau verifică tabelele combinatorice `.v3bin` (§4, opțiunea B). |
| `a0-parity [out.json]` | Poarta A0: paritate bit-exactă GPU ↔ CPU pe întregul univers. |
| `cdv3-build` / `cdv3-verify` | Construiește / verifică cataloagele de roți de acoperire. |
| `pmv3-build` / `pmv3-verify` | Construiește / verifică tabelele bootstrap 1.000×. |
| `c3v2-build` / `c3v2-verify` | Construiește / verifică portofoliile Pareto exacte. |
| `c4v2-build` | Construiește sinteza Monte-Carlo pe 1 M de traiectorii. |
| `kjv5-qualify [out.json]` | Calificarea algebrică pre-arhivă Q1–Q6 a KJ-V5 (nu citește nimic istoric). |
| `kjv5-walkforward [--repo <dir>] [--game 649/540/joker]` | Walk-forward cauzal KJ-V5 (scrie `kj_v5/rezultate/*.json` în arhiva QL649). |
| `a6-certify [out.json]` | Poarta A6: certifică linie cu linie cele trei registre prospective. |
| `s5-verify` | Verificarea de acceptare a sesiunii 5. |
| `parity [--ref <fișier>] [--root <dir>] [--target <dată>] [--s17 <fișier>]` | Paritatea S19 față de referința Python înghețată. |
| `admissibility [--enumerate] [--root <dir>] [--target <dată>]` | Rulează filtrul de admisibilitate S17. |
| `inventory` | Tipărește inventarul arhivelor LIVE ca JSON. |
| `memory` / `monitor` | Telemetrie procfs o singură dată sau continuă. |

Coduri de ieșire: `0` succes / toate porțile trecute, `1` o poartă a eșuat, `2` eroare (intrare lipsă, I/O, CUDA).

---

## 8. Variabile de mediu

| Variabilă | Folosită de | Semnificație |
|---|---|---|
| `CUDA_ARCH`, `CUDA_HOME`, `NVCC` | `build.rs` | Arhitectura GPU și locația toolkit-ului CUDA la compilare. |
| `QL649_ROOT` | GUI + CLI | Calea arhivei Python Quantum Lab 649 (§5). |
| `LOTO_RO_ROOT` | CLI KJ-V5 | Folderul-părinte care conține `next_candidate/quantum_lab_649`. |
| `QL649_VAULT_DIR` | `download_vault.sh` | Unde se salvează Vault-ul descărcat (interfața așteaptă `./vault`). |
| `QL649_VAULT_MIRROR_URL` | `download_vault.sh` | URL de bază al unei oglinzi alternative. |
| `WEBKIT_DISABLE_DMABUF_RENDERER=1` | lansator | Ocolește ferestrele WebKitGTK goale/negre pe NVIDIA. |

---

## 9. Fișiere neincluse și cum se obțin

| Element | De ce nu este în zip | Cum se obține |
|---|---|---|
| Cele 13 tabele Vault `.v3bin` (1,3 GB) | dimensiune | `./scripts/download_vault.sh` (CERN Zenodo) sau reconstruire (§4). |
| `data/` (arhiva Python) | versiune separată | Descarcă `quantum-lab-649-opensource.zip` și leag-o (§5). |
| Rezultatele de cercetare `*/rezultate/` (JSON/NPZ KJ-V5, HS-V4, TU4, XI) | artefacte de rulare ale sesiunilor de cercetare sigilate | Se regenerează cu CLI-ul (`kjv5-*`) și motoarele Python sau se cer copiile arhivate printr-o [cerere de documentație pe GitHub](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/issues/new?template=documentation_request.md) sau la [contact@bot-tradingsoftware.com](mailto:contact@bot-tradingsoftware.com) — oferite strict AS-IS, dacă există. |
| Rădăcina SSD extern `./ssd_data` | specifică mașinii | Orice folder; se raportează doar spațiul liber și locația Vault-ului. |

---

## 10. Depanare

| Simptom | Cauză / rezolvare |
|---|---|
| `nvcc: not found` sau eroare `CUDA_HOME` la `cargo build` | Instalează CUDA Toolkit 12.x și exportă `CUDA_HOME=/usr/local/cuda` (și `PATH=$CUDA_HOME/bin:$PATH`). |
| `no kernel image is available for execution on the device` | `CUDA_ARCH` nu corespunde GPU-ului. Recompilează cu valoarea corectă (§3.3): `cargo clean && CUDA_ARCH=sm_86 cargo build --release`. |
| Fereastra se deschide neagră sau goală | Pornește prin `launch_rust_core_v3.sh` (setează `WEBKIT_DISABLE_DMABUF_RENDERER=1`); actualizează driverul NVIDIA; încearcă `WEBKIT_DISABLE_COMPOSITING_MODE=1`. |
| `dimensiune fisier v3bin invalida` / `header v3bin neasteptat` | Un fișier Vault este trunchiat sau străin. Șterge-l și rulează din nou `download_vault.sh` (reia doar fișierele invalide). |
| `download_vault.sh` raportează nepotrivire SHA-256 | Descărcare coruptă sau oglindă greșită; șterge fișierul și reîncearcă. Hash-urile din manifest sunt referința. |
| Panourile afișează `lipsă`, `încă nu există`, `CURRENT_LIVE fără version_dir` | `QL649_ROOT` nu este setat / legat sau fișierele de rezultate nu sunt livrate (§5, §9). |
| Eroare `RSS guard` | Procesul ar depăși 11,5 GiB memorie rezidentă. Închide alte programe mari consumatoare de GPU/RAM sau rulează constructorii CLI unul câte unul. |
| Tab-ul arată `vault lipsă` | Descarcă sau construiește Vault-ul (§4) în `./vault`, relativ la folderul arhivei. |
| Build lent / `target/` uriaș | Normal pentru Tauri + CUDA; un build release durează 5–15 min și ocupă 3–5 GB. `cargo clean` eliberează spațiul. |

---

## 11. Memento legal și de domeniu

- Licența MIT: liber de folosit, modificat și redistribuit, păstrând notița de copyright și licența.
- `positive_claims_allowed = false`: niciun ecran, fișier sau comandă nu prezintă un bilet ca fiind mai probabil câștigător. Designurile de acoperire și portofoliile Pareto optimizează *acoperirea și costul sub un model*, niciodată șansele extragerii — fiecare bilet are aceeași probabilitate.
- Fără sfaturi de joc, financiare sau juridice; fără suport tehnic. Doar pentru documente *existente*: [cerere de documentație pe GitHub](https://github.com/quantbotsoftware/quantum-lab-649-rust-core/issues/new?template=documentation_request.md) · [contact@bot-tradingsoftware.com](mailto:contact@bot-tradingsoftware.com).
- Arhivă permanentă, Vault și citare: CERN Zenodo DOI [10.5281/zenodo.23992530](https://doi.org/10.5281/zenodo.23992530). Detalii despre Vault: [VAULT_DATASET_GUIDE](https://bot-tradingsoftware.com/docs/VAULT_DATASET_GUIDE.html).

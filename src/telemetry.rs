//! Monitorizarea asincronă continuă a memoriei totale (RSS-vârf), cu tampon de telemetrie pe SSD-ul extern
//! (protocol `tu4/PROTOCOL_QUANTUM_S21.md`, sigilat în `tu4/SIGILIU_TU4_S21.json` înaintea acestui cod).
//!
//! - Un fir dedicat (`ql649-telemetrie`) eșantionează periodic arborele de procese al pid-ului monitorizat (procesul
//!   + toți descendenții, inclusiv `WebKit*`) prin procfs, cu aceleași funcții ca măsurarea S20 (`resources`).
//! - Fiecare eșantion se scrie IMEDIAT ca o linie JSONL pe SSD (`./telemetry/`),
//!   în segmente de cel mult 16 MiB și cel mult 1 GiB pe sesiune; nimic nu se șterge automat.
//! - În memorie rămân numai agregatele de dimensiune fixă ([`Aggregate`]); seria nu se păstrează nici în procesul
//!   principal, nici în Webview (care primește doar agregatul, fără tablouri).
//! - Dacă SSD-ul nu este montat sau nu se poate scrie: starea `INDISPONIBIL`, niciun fir, nicio alternativă.
//!
//! Limite declarate: `rss_total_varf` este un vârf EȘANTIONAT (vârfurile mai scurte decât intervalul pot scăpa);
//! Σ VmHWM este marginea complementară; VmRSS numără paginile partajate de mai multe ori. Mecanismul NU reduce
//! memoria WebKitGTK; garantează doar că monitorizarea nu adaugă memorie care crește în timp.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::integrity::sha256_hex;
use crate::models::parity::utc_now_iso;
use crate::resources::{descendants, is_webview, read_proc, verdict, ProcMem, WEBVIEW_LIMIT_BYTES};
use crate::POSITIVE_CLAIMS_ALLOWED;

/// Rădăcina tamponului de telemetrie pe SSD-ul extern pentru Rust Core v3.
pub const DEFAULT_ROOT: &str = "./telemetry";
/// Punctul de montare cerut pentru rădăcina implicită.
pub const DEFAULT_MOUNT: &str = "./vault";
/// Intervalul în fereastra normală (Q6).
pub const INTERVAL_MS_DEFAULT: u64 = 1000;
/// Intervalul în modul `--smoke` (Q6).
pub const INTERVAL_MS_SMOKE: u64 = 100;
/// Domeniul permis al intervalului: 1…60 000 ms.
pub const INTERVAL_MS_MAX: u64 = 60_000;
/// Plafonul unui segment JSONL (Q5): 16 MiB.
pub const SEGMENT_MAX_BYTES: u64 = 16 * 1024 * 1024;
/// Plafonul unei sesiuni (Q5): 1 GiB.
pub const SESSION_MAX_BYTES: u64 = 1024 * 1024 * 1024;
/// Numărul maxim de eșantioane cerut prin CLI (Q9).
pub const MAX_SAMPLES_LIMIT: u64 = 10_000_000;
/// Pasul maxim de așteptare între verificările cererii de oprire (Q6).
const SLEEP_STEP: Duration = Duration::from_millis(50);

/// Configurația unei sesiuni de telemetrie.
#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub root: PathBuf,
    /// `Some(m)`: `m` trebuie să fie punct de montare, iar `root` trebuie să fie sub `m`.
    pub mount: Option<PathBuf>,
    pub pid: u32,
    pub interval_ms: u64,
    /// `Some(k)`: sesiunea se încheie `COMPLET` după `k` eșantioane scrise.
    pub max_samples: Option<u64>,
    pub segment_max_bytes: u64,
    pub session_max_bytes: u64,
    pub proc_root: PathBuf,
}

impl Config {
    /// Configurația de producție: rădăcina pe SSD, verificarea montării activă, `/proc` real.
    pub fn ssd(pid: u32, interval_ms: u64) -> Config {
        Config {
            root: PathBuf::from(DEFAULT_ROOT),
            mount: Some(PathBuf::from(DEFAULT_MOUNT)),
            pid,
            interval_ms,
            max_samples: None,
            segment_max_bytes: SEGMENT_MAX_BYTES,
            session_max_bytes: SESSION_MAX_BYTES,
            proc_root: PathBuf::from("/proc"),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(1..=INTERVAL_MS_MAX).contains(&self.interval_ms) {
            return Err(format!("interval {} ms în afara domeniului 1…{INTERVAL_MS_MAX} ms", self.interval_ms));
        }
        if let Some(k) = self.max_samples {
            if !(1..=MAX_SAMPLES_LIMIT).contains(&k) {
                return Err(format!("numărul de eșantioane {k} în afara domeniului 1…{MAX_SAMPLES_LIMIT}"));
            }
        }
        if self.segment_max_bytes == 0 || self.session_max_bytes == 0 {
            return Err("plafoanele segmentului și ale sesiunii trebuie să fie > 0".into());
        }
        Ok(())
    }
}

/// Un eșantion (protocol S21 §2). Câmpurile serializate formează o linie JSONL.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Sample {
    pub seq: u64,
    pub utc: String,
    pub t_ms: u64,
    pub pid: u32,
    pub n_procese: u32,
    pub citiri_esuate: u32,
    pub rss_total: u64,
    pub hwm_total: u64,
    pub pss_total: Option<u64>,
    pub rss_principal: Option<u64>,
    pub hwm_principal: Option<u64>,
    pub webview_rss: u64,
    pub webview_hwm: u64,
    pub n_webview: u32,
    /// Verdictul S20 §4 pe procesele WebKit ale eșantionului (numai pentru agregat, nu se scrie în JSONL).
    #[serde(skip)]
    pub verdict_webview: &'static str,
}

/// Un eșantion al arborelui {pid} ∪ descendenți(pid), dintr-o singură trecere prin `proc_root`.
pub fn take_sample(proc_root: &Path, pid: u32, seq: u64, t_ms: u64) -> Sample {
    let mut s = Sample { seq, utc: utc_now_iso(), t_ms, pid, ..Sample::default() };
    let mut pids = vec![pid];
    if let Ok(d) = descendants(proc_root, pid) {
        pids.extend(d);
    }
    let mut pss = Some(0u64);
    let mut webview: Vec<ProcMem> = Vec::new();
    let mut webview_incomplete = false;
    for p in pids {
        let Some(m) = read_proc(proc_root, p) else {
            s.citiri_esuate += 1;
            continue;
        };
        let wv = is_webview(m.name.as_deref());
        let (Some(r), Some(h)) = (m.rss_bytes, m.hwm_bytes) else {
            s.citiri_esuate += 1;
            webview_incomplete |= wv;
            continue;
        };
        s.n_procese += 1;
        s.rss_total += r;
        s.hwm_total += h;
        pss = pss.zip(m.pss_bytes).map(|(a, b)| a + b);
        if p == pid {
            s.rss_principal = Some(r);
            s.hwm_principal = Some(h);
        }
        if wv {
            s.n_webview += 1;
            s.webview_rss += r;
            s.webview_hwm += h;
            webview.push(m);
        }
    }
    s.pss_total = if s.n_procese > 0 { pss } else { None };
    s.verdict_webview = if webview_incomplete { "UNKNOWN" } else { verdict(&webview, WEBVIEW_LIMIT_BYTES) };
    s
}

/// Agregatul de dimensiune fixă al sesiunii (fără tablouri; protocol S21 §2, M5).
#[derive(Debug, Clone, Serialize)]
pub struct Aggregate {
    pub stare: &'static str,
    pub motiv: Option<String>,
    pub director: Option<String>,
    pub rezumat: Option<String>,
    pub pid: u32,
    pub interval_ms: u64,
    pub esantioane: u64,
    pub rss_total_varf: Option<u64>,
    pub seq_varf: Option<u64>,
    pub utc_varf: Option<String>,
    pub webview_rss_varf: Option<u64>,
    pub rss_principal_varf: Option<u64>,
    pub rss_total_ultim: Option<u64>,
    pub hwm_total_ultim: Option<u64>,
    pub webview_rss_ultim: Option<u64>,
    pub webview_hwm_ultim: Option<u64>,
    pub n_webview_ultim: Option<u32>,
    pub n_procese_ultim: Option<u32>,
    pub citiri_esuate_ultim: Option<u32>,
    pub verdict_webview_30mb_ultim: &'static str,
    pub utc_ultim: Option<String>,
    pub octeti_scrisi: u64,
    pub segmente: u32,
    pub pornit_utc: String,
    pub oprit_utc: Option<String>,
    pub metoda: &'static str,
    pub nota: &'static str,
    pub positive_claims_allowed: bool,
}

impl Aggregate {
    fn new(cfg: &Config) -> Aggregate {
        Aggregate {
            stare: "ACTIV",
            motiv: None,
            director: None,
            rezumat: None,
            pid: cfg.pid,
            interval_ms: cfg.interval_ms,
            esantioane: 0,
            rss_total_varf: None,
            seq_varf: None,
            utc_varf: None,
            webview_rss_varf: None,
            rss_principal_varf: None,
            rss_total_ultim: None,
            hwm_total_ultim: None,
            webview_rss_ultim: None,
            webview_hwm_ultim: None,
            n_webview_ultim: None,
            n_procese_ultim: None,
            citiri_esuate_ultim: None,
            verdict_webview_30mb_ultim: "UNKNOWN",
            utc_ultim: None,
            octeti_scrisi: 0,
            segmente: 0,
            pornit_utc: utc_now_iso(),
            oprit_utc: None,
            metoda: "fir dedicat; procfs /proc/<pid>/status (VmRSS, VmHWM) + smaps_rollup (Pss) pe {pid} ∪ \
                     descendenți; o linie JSONL per eșantion scrisă imediat pe SSD; agregate O(1) în memorie",
            nota: "RSS-vârf eșantionat (vârfurile mai scurte decât intervalul pot scăpa); Σ VmHWM = margine \
                   complementară; VmRSS numără paginile partajate de mai multe ori. Tamponul pe SSD NU reduce memoria \
                   WebKitGTK (protocol S21 §0, §3)",
            positive_claims_allowed: POSITIVE_CLAIMS_ALLOWED,
        }
    }

    fn unavailable(cfg: &Config, motiv: String) -> Aggregate {
        let mut a = Aggregate::new(cfg);
        a.stare = "INDISPONIBIL";
        a.motiv = Some(motiv);
        a.oprit_utc = Some(a.pornit_utc.clone());
        a
    }

    /// Absoarbe un eșantion scris (`bytes` = lungimea liniei).
    pub fn absorb(&mut self, s: &Sample, bytes: u64) {
        self.esantioane += 1;
        self.octeti_scrisi += bytes;
        if self.rss_total_varf.is_none_or(|v| s.rss_total > v) {
            self.rss_total_varf = Some(s.rss_total);
            self.seq_varf = Some(s.seq);
            self.utc_varf = Some(s.utc.clone());
        }
        self.webview_rss_varf = Some(self.webview_rss_varf.map_or(s.webview_rss, |v| v.max(s.webview_rss)));
        if let Some(r) = s.rss_principal {
            self.rss_principal_varf = Some(self.rss_principal_varf.map_or(r, |v| v.max(r)));
        }
        self.rss_total_ultim = Some(s.rss_total);
        self.hwm_total_ultim = Some(s.hwm_total);
        self.webview_rss_ultim = Some(s.webview_rss);
        self.webview_hwm_ultim = Some(s.webview_hwm);
        self.n_webview_ultim = Some(s.n_webview);
        self.n_procese_ultim = Some(s.n_procese);
        self.citiri_esuate_ultim = Some(s.citiri_esuate);
        self.verdict_webview_30mb_ultim = s.verdict_webview;
        self.utc_ultim = Some(s.utc.clone());
    }
}

/// Informațiile unui segment, calculate la oprire (citire unică a fișierului închis).
#[derive(Debug, Clone, Serialize)]
pub struct SegmentInfo {
    pub fisier: String,
    pub octeti: u64,
    pub linii: u64,
    pub sha256: String,
}

pub fn segment_name(index: u32) -> String {
    format!("telemetrie_{index:03}.jsonl")
}

enum WriteOutcome {
    Written,
    Cap,
}

/// Scriitorul de segmente: un fișier deschis în mod append, fără tampon propriu care să acumuleze.
struct SegmentWriter {
    dir: PathBuf,
    file: Option<File>,
    segments: u32,
    seg_bytes: u64,
    total: u64,
    seg_max: u64,
    total_max: u64,
}

impl SegmentWriter {
    fn write_line(&mut self, line: &[u8]) -> Result<WriteOutcome, String> {
        let len = line.len() as u64;
        if self.total + len > self.total_max {
            return Ok(WriteOutcome::Cap);
        }
        if self.file.is_none() || (self.seg_bytes > 0 && self.seg_bytes + len > self.seg_max) {
            if let Some(f) = self.file.take() {
                f.sync_all().map_err(|e| format!("sync segment: {e}"))?;
            }
            let path = self.dir.join(segment_name(self.segments));
            let f = OpenOptions::new()
                .create_new(true)
                .append(true)
                .open(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            self.file = Some(f);
            self.segments += 1;
            self.seg_bytes = 0;
        }
        let f = self.file.as_mut().ok_or("segment absent")?;
        f.write_all(line).map_err(|e| format!("scriere segment: {e}"))?;
        self.seg_bytes += len;
        self.total += len;
        Ok(WriteOutcome::Written)
    }

    fn finish(&mut self) -> Result<(), String> {
        if let Some(f) = self.file.take() {
            f.sync_all().map_err(|e| format!("sync segment: {e}"))?;
        }
        Ok(())
    }
}

/// Verificările dinaintea pornirii (Q3): configurația, montarea, rădăcina; apoi directorul sesiunii.
fn prepare(cfg: &Config) -> Result<PathBuf, String> {
    cfg.validate()?;
    if let Some(m) = &cfg.mount {
        let parent = m.parent().ok_or_else(|| format!("{}: fără director părinte", m.display()))?;
        let dm = std::fs::metadata(m).map_err(|e| format!("{}: {e}", m.display()))?.dev();
        let dp = std::fs::metadata(parent).map_err(|e| format!("{}: {e}", parent.display()))?.dev();
        if dm == dp {
            return Err(format!("{} nu este punct de montare (SSD-ul extern nu e montat)", m.display()));
        }
        if !cfg.root.starts_with(m) {
            return Err(format!("{} nu este sub punctul de montare {}", cfg.root.display(), m.display()));
        }
    }
    std::fs::create_dir_all(&cfg.root).map_err(|e| format!("{}: {e}", cfg.root.display()))?;
    let stamp: String = utc_now_iso().chars().filter(|c| *c != '-' && *c != ':').collect();
    for k in 0..1000u32 {
        let name = if k == 0 { format!("{stamp}_pid{}", cfg.pid) } else { format!("{stamp}_pid{}_{k}", cfg.pid) };
        let dir = cfg.root.join(name);
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        }
    }
    Err("nu s-a putut crea un director de sesiune unic".into())
}

/// Segmentele sesiunii, recitite o singură dată după închidere (octeți, linii, SHA-256).
pub fn segment_infos(dir: &Path, segments: u32) -> Result<Vec<SegmentInfo>, String> {
    (0..segments)
        .map(|i| {
            let path = dir.join(segment_name(i));
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(SegmentInfo {
                fisier: segment_name(i),
                octeti: bytes.len() as u64,
                linii: bytes.iter().filter(|b| **b == b'\n').count() as u64,
                sha256: sha256_hex(&bytes),
            })
        })
        .collect()
}

fn write_summary(dir: &Path, cfg: &Config, agg: &Aggregate) -> Result<PathBuf, String> {
    let segs = segment_infos(dir, agg.segmente)?;
    let doc = serde_json::json!({
        "protocol": "TU4-S21 (tu4/PROTOCOL_QUANTUM_S21.md)",
        "agregat": agg,
        "segmente": segs,
        "configuratie": cfg,
        "positive_claims_allowed": POSITIVE_CLAIMS_ALLOWED,
    });
    let path = dir.join("rezumat.json");
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

fn lock(shared: &Mutex<Aggregate>) -> std::sync::MutexGuard<'_, Aggregate> {
    shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn run_loop(cfg: Config, dir: PathBuf, stop: Arc<AtomicBool>, shared: Arc<Mutex<Aggregate>>) {
    let mut writer = SegmentWriter {
        dir: dir.clone(),
        file: None,
        segments: 0,
        seg_bytes: 0,
        total: 0,
        seg_max: cfg.segment_max_bytes,
        total_max: cfg.session_max_bytes,
    };
    let interval = Duration::from_millis(cfg.interval_ms);
    let t0 = Instant::now();
    let mut buf: Vec<u8> = Vec::with_capacity(512);
    let mut seq = 0u64;
    let (mut stare, mut motiv): (&'static str, Option<String>) = ("OPRIT", None);
    'outer: loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let it = Instant::now();
        let s = take_sample(&cfg.proc_root, cfg.pid, seq + 1, t0.elapsed().as_millis() as u64);
        buf.clear();
        if let Err(e) = serde_json::to_writer(&mut buf, &s) {
            (stare, motiv) = ("EROARE_SCRIERE", Some(format!("serializare: {e}")));
            break;
        }
        buf.push(b'\n');
        match writer.write_line(&buf) {
            Ok(WriteOutcome::Written) => {
                seq += 1;
                let mut g = lock(&shared);
                g.absorb(&s, buf.len() as u64);
                g.segmente = writer.segments;
            }
            Ok(WriteOutcome::Cap) => {
                (stare, motiv) = ("PLAFON_ATINS", Some(format!("plafonul sesiunii {} B", cfg.session_max_bytes)));
                break;
            }
            Err(e) => {
                (stare, motiv) = ("EROARE_SCRIERE", Some(e));
                break;
            }
        }
        if cfg.max_samples == Some(seq) {
            stare = "COMPLET";
            break;
        }
        let deadline = it + interval;
        loop {
            if stop.load(Ordering::SeqCst) {
                break 'outer;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep(SLEEP_STEP.min(deadline - now));
        }
    }
    if let Err(e) = writer.finish() {
        (stare, motiv) = ("EROARE_SCRIERE", Some(e));
    }
    let mut snapshot = {
        let mut g = lock(&shared);
        g.stare = stare;
        g.motiv = motiv;
        g.segmente = writer.segments;
        g.oprit_utc = Some(utc_now_iso());
        g.clone()
    };
    match write_summary(&dir, &cfg, &snapshot) {
        Ok(p) => snapshot.rezumat = Some(p.display().to_string()),
        Err(e) => {
            snapshot.stare = "EROARE_SCRIERE";
            snapshot.motiv = Some(format!("rezumat: {e}"));
        }
    }
    *lock(&shared) = snapshot;
}

/// O sesiune de telemetrie. Firul pornește în [`Monitor::start`]; agregatul se citește cu [`Monitor::snapshot`].
pub struct Monitor {
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<Aggregate>>,
    handle: Option<JoinHandle<()>>,
}

impl Monitor {
    /// Pornește sesiunea. Nu eșuează: o configurație invalidă sau un SSD indisponibil dau starea `INDISPONIBIL`, fără
    /// fir și fără fișiere.
    pub fn start(cfg: Config) -> Monitor {
        let stop = Arc::new(AtomicBool::new(false));
        let dir = match prepare(&cfg) {
            Ok(d) => d,
            Err(e) => {
                let shared = Arc::new(Mutex::new(Aggregate::unavailable(&cfg, e)));
                return Monitor { stop, shared, handle: None };
            }
        };
        let mut agg = Aggregate::new(&cfg);
        agg.director = Some(dir.display().to_string());
        let shared = Arc::new(Mutex::new(agg));
        let (s2, sh2) = (Arc::clone(&stop), Arc::clone(&shared));
        let spawned =
            std::thread::Builder::new().name("ql649-telemetrie".into()).spawn(move || run_loop(cfg, dir, s2, sh2));
        match spawned {
            Ok(h) => Monitor { stop, shared, handle: Some(h) },
            Err(e) => {
                let mut g = lock(&shared);
                g.stare = "INDISPONIBIL";
                g.motiv = Some(format!("firul de telemetrie nu a pornit: {e}"));
                drop(g);
                Monitor { stop, shared, handle: None }
            }
        }
    }

    /// Copia agregatului curent (lacăt scurt; firul UI nu așteaptă eșantionarea).
    pub fn snapshot(&self) -> Aggregate {
        lock(&self.shared).clone()
    }

    pub fn is_running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Cere oprirea, așteaptă firul (care scrie `rezumat.json`) și întoarce agregatul final.
    pub fn stop(&mut self) -> Aggregate {
        self.stop.store(true, Ordering::SeqCst);
        self.wait()
    }

    /// Așteaptă încheierea firului fără a cere oprirea (sesiuni cu `max_samples`).
    pub fn wait(&mut self) -> Aggregate {
        if let Some(h) = self.handle.take() {
            if h.join().is_err() {
                let mut g = lock(&self.shared);
                g.stare = "EROARE_SCRIERE";
                g.motiv = Some("firul de telemetrie s-a oprit anormal".into());
            }
        }
        self.snapshot()
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("s21_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn mk(root: &Path, pid: u32, ppid: u32, name: &str, rss_kb: Option<u64>, hwm_kb: Option<u64>, pss_kb: Option<u64>) {
        let d = root.join(pid.to_string());
        std::fs::create_dir_all(&d).unwrap();
        let mut s = format!("Name:\t{name}\nPPid:\t{ppid}\n");
        if let Some(h) = hwm_kb {
            s += &format!("VmHWM:\t{h} kB\n");
        }
        if let Some(r) = rss_kb {
            s += &format!("VmRSS:\t{r} kB\n");
        }
        std::fs::write(d.join("status"), s).unwrap();
        if let Some(p) = pss_kb {
            std::fs::write(d.join("smaps_rollup"), format!("Rss:\t1 kB\nPss:\t{p} kB\n")).unwrap();
        }
    }

    fn synthetic_proc(tag: &str) -> PathBuf {
        let root = tmp(tag);
        mk(&root, 100, 1, "rust-singularit", Some(50_000), Some(60_000), Some(40_000));
        mk(&root, 101, 100, "WebKitNetworkPr", Some(9_000), Some(10_000), Some(5_000));
        mk(&root, 102, 100, "bwrap", Some(1_000), Some(1_000), Some(500));
        mk(&root, 103, 102, "WebKitWebProces", Some(20_000), Some(25_000), Some(15_000));
        mk(&root, 104, 100, "kthread-like", None, None, None);
        mk(&root, 200, 1, "WebKitWebProces", Some(1), Some(1), Some(1)); // alt arbore: nu se numără
        root
    }

    fn test_cfg(proc_root: &Path, root: &Path, pid: u32, interval_ms: u64, max: Option<u64>) -> Config {
        Config {
            root: root.to_path_buf(),
            mount: None,
            pid,
            interval_ms,
            max_samples: max,
            segment_max_bytes: SEGMENT_MAX_BYTES,
            session_max_bytes: SESSION_MAX_BYTES,
            proc_root: proc_root.to_path_buf(),
        }
    }

    fn read_lines(dir: &Path, segments: u32) -> Vec<Value> {
        (0..segments)
            .flat_map(|i| {
                let text = std::fs::read_to_string(dir.join(segment_name(i))).unwrap();
                assert!(text.is_empty() || text.ends_with('\n'), "segment {i} nu se termină cu linie completă");
                text.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn m1a_synthetic_sample_sums() {
        let root = synthetic_proc("m1a");
        let s = take_sample(&root, 100, 7, 42);
        assert_eq!((s.seq, s.t_ms, s.pid), (7, 42, 100));
        assert_eq!(s.n_procese, 4);
        assert_eq!(s.citiri_esuate, 1); // 104 fără VmRSS/VmHWM
        assert_eq!(s.rss_total, (50_000 + 9_000 + 1_000 + 20_000) * 1024);
        assert_eq!(s.hwm_total, (60_000 + 10_000 + 1_000 + 25_000) * 1024);
        assert_eq!(s.pss_total, Some((40_000 + 5_000 + 500 + 15_000) * 1024));
        assert_eq!((s.rss_principal, s.hwm_principal), (Some(50_000 * 1024), Some(60_000 * 1024)));
        assert_eq!((s.webview_rss, s.webview_hwm, s.n_webview), (29_000 * 1024, 35_000 * 1024, 2));
        assert_eq!(s.verdict_webview, "FAIL"); // 35 840 000 B > 30 000 000 B
        std::fs::remove_file(root.join("102/smaps_rollup")).unwrap();
        assert_eq!(take_sample(&root, 100, 1, 0).pss_total, None);
        let line = serde_json::to_value(&s).unwrap();
        assert!(line.get("verdict_webview").is_none(), "verdictul nu intră în JSONL");
        let gone = take_sample(&root, 999, 1, 0);
        assert_eq!((gone.n_procese, gone.citiri_esuate, gone.rss_principal), (0, 1, None));
        assert_eq!(gone.verdict_webview, "UNKNOWN");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn m1b_absorb_and_file_consistency() {
        let cfg = test_cfg(Path::new("/proc"), Path::new("/tmp"), 1, 1, None);
        let mut a = Aggregate::new(&cfg);
        let mk_s = |seq, rss, wv, pr| Sample {
            seq,
            utc: format!("u{seq}"),
            rss_total: rss,
            webview_rss: wv,
            rss_principal: pr,
            verdict_webview: "PASS",
            ..Sample::default()
        };
        for (seq, rss, wv, pr) in [(1, 10, 5, Some(3)), (2, 30, 1, None), (3, 30, 9, Some(7)), (4, 20, 2, Some(1))] {
            a.absorb(&mk_s(seq, rss, wv, pr), 100);
        }
        assert_eq!((a.esantioane, a.octeti_scrisi), (4, 400));
        assert_eq!((a.rss_total_varf, a.seq_varf, a.utc_varf.as_deref()), (Some(30), Some(2), Some("u2")));
        assert_eq!((a.webview_rss_varf, a.rss_principal_varf, a.rss_total_ultim), (Some(9), Some(7), Some(20)));

        let proc_root = synthetic_proc("m1b_proc");
        let out = tmp("m1b_out");
        let mut m = Monitor::start(test_cfg(&proc_root, &out, 100, 1, Some(25)));
        let agg = m.wait();
        assert_eq!(agg.stare, "COMPLET", "{:?}", agg.motiv);
        assert_eq!(agg.esantioane, 25);
        let dir = PathBuf::from(agg.director.clone().unwrap());
        let lines = read_lines(&dir, agg.segmente);
        assert_eq!(lines.len(), 25);
        for (i, l) in lines.iter().enumerate() {
            assert_eq!(l["seq"].as_u64(), Some(i as u64 + 1));
        }
        let max = lines.iter().map(|l| l["rss_total"].as_u64().unwrap()).max();
        assert_eq!(max, agg.rss_total_varf);
        let bytes: u64 = (0..agg.segmente).map(|i| std::fs::metadata(dir.join(segment_name(i))).unwrap().len()).sum();
        assert_eq!(bytes, agg.octeti_scrisi);
        assert!(dir.join("rezumat.json").exists());
        std::fs::remove_dir_all(&proc_root).unwrap();
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn m1c_rotation_and_session_cap() {
        let proc_root = synthetic_proc("m1c_proc");
        let out = tmp("m1c_out");
        let one = {
            let mut v = serde_json::to_vec(&take_sample(&proc_root, 100, 1, 0)).unwrap();
            v.push(b'\n');
            v.len() as u64
        };
        let mut cfg = test_cfg(&proc_root, &out, 100, 1, Some(1000));
        cfg.segment_max_bytes = one * 2 + one / 2; // 2 linii per segment
        cfg.session_max_bytes = one * 7 + one / 2; // ~7 linii pe sesiune
        let agg = Monitor::start(cfg.clone()).wait();
        assert_eq!(agg.stare, "PLAFON_ATINS");
        assert!(agg.octeti_scrisi <= cfg.session_max_bytes);
        assert!((6..=8).contains(&agg.esantioane), "{}", agg.esantioane);
        let dir = PathBuf::from(agg.director.clone().unwrap());
        let segs = segment_infos(&dir, agg.segmente).unwrap();
        assert!(segs.len() >= 3);
        assert!(segs.iter().all(|s| s.octeti <= cfg.segment_max_bytes && s.linii >= 1));
        assert_eq!(segs.iter().map(|s| s.linii).sum::<u64>(), agg.esantioane);
        assert_eq!(read_lines(&dir, agg.segmente).len() as u64, agg.esantioane);
        let files = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, segs.len() + 1, "segmentele + rezumat.json, nimic șters");
        std::fs::remove_dir_all(&proc_root).unwrap();
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn m1d_not_a_mount_point_is_unavailable() {
        let base = tmp("m1d");
        let fake_mount = base.join("nu_e_montare");
        std::fs::create_dir_all(&fake_mount).unwrap();
        let mut cfg = test_cfg(Path::new("/proc"), &fake_mount.join("telemetrie"), std::process::id(), 10, None);
        cfg.mount = Some(fake_mount.clone());
        let mut m = Monitor::start(cfg.clone());
        assert!(!m.is_running());
        let a = m.snapshot();
        assert_eq!(a.stare, "INDISPONIBIL");
        assert!(a.motiv.as_deref().unwrap().contains("punct de montare"), "{:?}", a.motiv);
        assert!(!fake_mount.join("telemetrie").exists(), "niciun fișier creat");
        assert_eq!(m.stop().stare, "INDISPONIBIL");
        cfg.root = PathBuf::from("/altundeva/telemetrie");
        cfg.mount = Some(PathBuf::from("/"));
        let outside = Monitor::start(cfg).snapshot();
        assert_eq!(outside.stare, "INDISPONIBIL");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn m1e_prompt_stop_and_summary_hashes() {
        let proc_root = synthetic_proc("m1e_proc");
        let out = tmp("m1e_out");
        let mut m = Monitor::start(test_cfg(&proc_root, &out, 100, INTERVAL_MS_MAX, None));
        let t = Instant::now();
        while m.snapshot().esantioane < 1 && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(m.is_running());
        let t = Instant::now();
        let agg = m.stop();
        assert!(t.elapsed() <= Duration::from_secs(2), "oprire în {:?}", t.elapsed());
        assert_eq!((agg.stare, agg.esantioane), ("OPRIT", 1));
        let rez: Value =
            serde_json::from_str(&std::fs::read_to_string(agg.rezumat.as_ref().unwrap()).unwrap()).unwrap();
        let dir = PathBuf::from(agg.director.unwrap());
        let segs = rez["segmente"].as_array().unwrap();
        assert_eq!(segs.len(), 1);
        for s in segs {
            let bytes = std::fs::read(dir.join(s["fisier"].as_str().unwrap())).unwrap();
            assert_eq!(s["sha256"].as_str().unwrap(), sha256_hex(&bytes));
            assert_eq!(s["octeti"].as_u64().unwrap(), bytes.len() as u64);
        }
        assert_eq!(rez["agregat"]["stare"], "OPRIT");
        assert_eq!(rez["positive_claims_allowed"], false);
        std::fs::remove_dir_all(&proc_root).unwrap();
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn m1f_real_child_is_counted() {
        if !Path::new("/proc/self/status").exists() {
            return;
        }
        let mut child = std::process::Command::new("/bin/sleep").arg("5").spawn().unwrap();
        let s = take_sample(Path::new("/proc"), std::process::id(), 1, 0);
        let _ = child.kill();
        let _ = child.wait();
        assert!(s.n_procese >= 2, "{s:?}");
        assert!(s.rss_principal.unwrap() > 0 && s.rss_total >= s.rss_principal.unwrap());
        assert_eq!(s.n_webview, 0);
    }

    #[test]
    fn m1g_interval_domain() {
        let mut cfg = Config::ssd(1, 0);
        assert!(cfg.validate().is_err());
        cfg.interval_ms = INTERVAL_MS_MAX + 1;
        assert!(cfg.validate().is_err());
        cfg.interval_ms = 1;
        assert!(cfg.validate().is_ok());
        cfg.interval_ms = INTERVAL_MS_MAX;
        assert!(cfg.validate().is_ok());
        cfg.max_samples = Some(0);
        assert!(cfg.validate().is_err());
        cfg.max_samples = Some(MAX_SAMPLES_LIMIT + 1);
        assert!(cfg.validate().is_err());
        let a = Monitor::start(Config { interval_ms: 0, ..Config::ssd(1, 1) }).snapshot();
        assert_eq!(a.stare, "INDISPONIBIL");
        assert!(a.motiv.as_deref().unwrap().contains("interval"));
        let v = serde_json::to_value(&a).unwrap();
        assert!(v.as_object().unwrap().values().all(|x| !x.is_array()), "agregatul nu are tablouri (M5)");
    }
}

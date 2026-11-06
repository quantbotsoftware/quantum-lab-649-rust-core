//! Măsurarea memoriei rezidente a ferestrei Tauri (protocol S20 §4, decizia R5).
//!
//! Sursa este exclusiv procfs-ul Linux:
//! - `/proc/<pid>/status`: `Name`, `PPid`, `VmRSS`, `VmHWM`, în kB = 1024 B;
//! - opțional `/proc/<pid>/smaps_rollup`: `Pss`.
//!
//! Se citesc procesul principal și toți descendenții lui (arborele din `PPid`). Procesele Webview sunt descendenții al
//! căror `Name` începe cu `WebKit` (WebKitGTK). Limita este L = 30 MB = 30 000 000 B (SI).
//!
//! Verdictul:
//! - UNKNOWN dacă procfs lipsește, dacă nu există niciun proces WebKit sau dacă VmRSS/VmHWM nu se pot citi;
//! - PASS ⇔ Σ VmHWM(WebKit) ≤ L;
//! - altfel FAIL.
//!
//! S20 măsoară, nu optimizează. VmRSS include paginile partajate ale bibliotecilor mapate.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use serde::Serialize;

use crate::POSITIVE_CLAIMS_ALLOWED;

/// Limita cerută pentru Webview (Mesajul 64): 30 MB în sens SI.
pub const WEBVIEW_LIMIT_BYTES: u64 = 30_000_000;

/// Memoria unui proces, din procfs (`None` = câmp absent sau ilizibil).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcMem {
    pub pid: u32,
    pub ppid: Option<u32>,
    pub name: Option<String>,
    pub rss_bytes: Option<u64>,
    pub hwm_bytes: Option<u64>,
    pub pss_bytes: Option<u64>,
}

/// Câmpurile extrase din textul `/proc/<pid>/status`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusFields {
    pub name: Option<String>,
    pub ppid: Option<u32>,
    pub rss_bytes: Option<u64>,
    pub hwm_bytes: Option<u64>,
}

/// `"   1234 kB"` → 1234 × 1024 B. Orice altă unitate sau text ⇒ `None`.
pub fn parse_kb(value: &str) -> Option<u64> {
    let mut it = value.split_whitespace();
    let n: u64 = it.next()?.parse().ok()?;
    match (it.next(), it.next()) {
        (Some("kB"), None) => n.checked_mul(1024),
        _ => None,
    }
}

pub fn parse_status(text: &str) -> StatusFields {
    let mut f = StatusFields::default();
    for line in text.lines() {
        let Some((k, v)) = line.split_once(':') else { continue };
        match k {
            "Name" => f.name = Some(v.trim().to_string()),
            "PPid" => f.ppid = v.trim().parse().ok(),
            "VmRSS" => f.rss_bytes = parse_kb(v),
            "VmHWM" => f.hwm_bytes = parse_kb(v),
            _ => {}
        }
    }
    f
}

/// `Pss:` din `smaps_rollup` (primul rând cu cheia exactă `Pss`).
pub fn parse_pss(text: &str) -> Option<u64> {
    text.lines().find_map(|l| l.split_once(':').filter(|(k, _)| *k == "Pss").and_then(|(_, v)| parse_kb(v)))
}

/// Citește un proces; `None` dacă `status` nu se poate citi (proces dispărut sau acces refuzat).
pub fn read_proc(proc_root: &Path, pid: u32) -> Option<ProcMem> {
    let dir = proc_root.join(pid.to_string());
    let st = parse_status(&std::fs::read_to_string(dir.join("status")).ok()?);
    let pss = std::fs::read_to_string(dir.join("smaps_rollup")).ok().as_deref().and_then(parse_pss);
    Some(ProcMem {
        pid,
        ppid: st.ppid,
        name: st.name,
        rss_bytes: st.rss_bytes,
        hwm_bytes: st.hwm_bytes,
        pss_bytes: pss,
    })
}

/// Toți descendenții lui `root_pid` (BFS pe relația PPid, ordonați crescător după pid).
pub fn descendants(proc_root: &Path, root_pid: u32) -> Result<Vec<u32>, String> {
    let rd = std::fs::read_dir(proc_root).map_err(|e| format!("{}: {e}", proc_root.display()))?;
    let mut children: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for e in rd.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
        let Ok(text) = std::fs::read_to_string(e.path().join("status")) else { continue };
        if let Some(pp) = parse_status(&text).ppid {
            children.entry(pp).or_default().push(pid);
        }
    }
    let mut out = Vec::new();
    let mut queue = VecDeque::from([root_pid]);
    while let Some(p) = queue.pop_front() {
        for &c in children.get(&p).map(Vec::as_slice).unwrap_or(&[]) {
            if c != root_pid && !out.contains(&c) {
                out.push(c);
                queue.push_back(c);
            }
        }
    }
    out.sort_unstable();
    Ok(out)
}

pub fn is_webview(name: Option<&str>) -> bool {
    name.is_some_and(|n| n.starts_with("WebKit"))
}

/// Verdictul §4 pe lista proceselor Webview.
pub fn verdict(webview: &[ProcMem], limit: u64) -> &'static str {
    if webview.is_empty() || webview.iter().any(|p| p.rss_bytes.is_none() || p.hwm_bytes.is_none()) {
        return "UNKNOWN";
    }
    let hwm: u64 = webview.iter().filter_map(|p| p.hwm_bytes).sum();
    if hwm <= limit {
        "PASS"
    } else {
        "FAIL"
    }
}

fn sum_opt(ps: &[ProcMem], f: impl Fn(&ProcMem) -> Option<u64>) -> Option<u64> {
    ps.iter().map(f).try_fold(0u64, |acc, v| v.map(|x| acc + x))
}

/// Raportul complet al memoriei pentru arborele unui proces.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryReport {
    pub verdict: &'static str,
    pub limita_octeti: u64,
    pub limita_text: &'static str,
    pub pid_principal: u32,
    pub principal: Option<ProcMem>,
    pub webview_procese: Vec<ProcMem>,
    pub webview_rss_total_octeti: Option<u64>,
    pub webview_hwm_total_octeti: Option<u64>,
    pub webview_pss_total_octeti: Option<u64>,
    pub alte_descendente: Vec<ProcMem>,
    pub masurat_utc: String,
    pub motiv: Option<String>,
    pub metoda: &'static str,
    pub nota: &'static str,
    pub positive_claims_allowed: bool,
}

pub fn memory_report_at(proc_root: &Path, pid: u32, limit: u64) -> MemoryReport {
    let utc = crate::models::parity::utc_now_iso();
    let principal = read_proc(proc_root, pid);
    let (desc, motiv) = match descendants(proc_root, pid) {
        Ok(d) => (d, None),
        Err(e) => (Vec::new(), Some(format!("procfs indisponibil: {e}"))),
    };
    let mut webview = Vec::new();
    let mut alte = Vec::new();
    for d in desc {
        if let Some(p) = read_proc(proc_root, d) {
            if is_webview(p.name.as_deref()) {
                webview.push(p);
            } else {
                alte.push(p);
            }
        }
    }
    let v = if motiv.is_some() { "UNKNOWN" } else { verdict(&webview, limit) };
    let motiv = motiv.or_else(|| match v {
        "UNKNOWN" if webview.is_empty() => Some("niciun proces WebKit printre descendenți".to_string()),
        "UNKNOWN" => Some("VmRSS/VmHWM ilizibil pentru un proces WebKit".to_string()),
        _ => None,
    });
    MemoryReport {
        verdict: v,
        limita_octeti: limit,
        limita_text: "30 MB = 30 000 000 B (SI); verdict pe Σ VmHWM al proceselor WebKit*",
        pid_principal: pid,
        principal,
        webview_rss_total_octeti: sum_opt(&webview, |p| p.rss_bytes),
        webview_hwm_total_octeti: sum_opt(&webview, |p| p.hwm_bytes),
        webview_pss_total_octeti: sum_opt(&webview, |p| p.pss_bytes),
        webview_procese: webview,
        alte_descendente: alte,
        masurat_utc: utc,
        motiv,
        metoda: "procfs: /proc/<pid>/status (VmRSS, VmHWM; kB = 1024 B), /proc/<pid>/smaps_rollup (Pss, opțional); \
                 arborele de descendenți din PPid",
        nota: "măsurare, nu optimizare (protocol S20 §4); VmRSS include paginile partajate ale bibliotecilor \
               WebKitGTK/GTK",
        positive_claims_allowed: POSITIVE_CLAIMS_ALLOWED,
    }
}

/// Raportul pentru procesul `pid` din `/proc`, cu limita S20.
pub fn memory_report(pid: u32) -> MemoryReport {
    memory_report_at(Path::new("/proc"), pid, WEBVIEW_LIMIT_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATUS: &str = "Name:\tWebKitWebProces\nUmask:\t0022\nState:\tS (sleeping)\nPid:\t4242\nPPid:\t4200\n\
VmPeak:\t 2000000 kB\nVmHWM:\t  150000 kB\nVmRSS:\t  120000 kB\nThreads:\t12\n";

    fn pm(name: &str, rss: Option<u64>, hwm: Option<u64>) -> ProcMem {
        ProcMem { pid: 1, ppid: Some(0), name: Some(name.into()), rss_bytes: rss, hwm_bytes: hwm, pss_bytes: None }
    }

    #[test]
    fn parse_status_fields() {
        let f = parse_status(STATUS);
        assert_eq!(f.name.as_deref(), Some("WebKitWebProces"));
        assert_eq!(f.ppid, Some(4200));
        assert_eq!(f.rss_bytes, Some(120_000 * 1024));
        assert_eq!(f.hwm_bytes, Some(150_000 * 1024));
        let g = parse_status("Name:\tkthreadd\nPPid:\t0\n");
        assert_eq!((g.rss_bytes, g.hwm_bytes), (None, None));
        assert_eq!(parse_kb(" 12 kB"), Some(12 * 1024));
        assert_eq!(parse_kb(" 12 MB"), None);
        assert_eq!(parse_kb(" x kB"), None);
        assert_eq!(parse_kb(" 12 kB extra"), None);
        assert_eq!(parse_pss("Rss:  10 kB\nPss:   7 kB\nPss_Anon:  3 kB\n"), Some(7 * 1024));
        assert_eq!(parse_pss("Rss:  10 kB\n"), None);
    }

    #[test]
    fn verdict_boundaries() {
        let l = WEBVIEW_LIMIT_BYTES;
        assert_eq!(verdict(&[], l), "UNKNOWN");
        assert_eq!(verdict(&[pm("WebKitWebProces", Some(1), None)], l), "UNKNOWN");
        assert_eq!(verdict(&[pm("WebKitWebProces", Some(10), Some(l))], l), "PASS");
        assert_eq!(verdict(&[pm("WebKitWebProces", Some(10), Some(l + 1))], l), "FAIL");
        assert_eq!(
            verdict(
                &[pm("WebKitWebProces", Some(10), Some(l / 2)), pm("WebKitNetworkPr", Some(10), Some(l / 2 + 1))],
                l
            ),
            "FAIL"
        );
        assert!(is_webview(Some("WebKitNetworkPr")) && !is_webview(Some("rust-singularit")) && !is_webview(None));
        assert_eq!(sum_opt(&[pm("a", Some(1), Some(2)), pm("b", Some(3), Some(4))], |p| p.hwm_bytes), Some(6));
        assert_eq!(sum_opt(&[pm("a", Some(1), Some(2)), pm("b", None, Some(4))], |p| p.rss_bytes), None);
    }

    #[test]
    fn synthetic_proc_tree() {
        let root = std::env::temp_dir().join(format!("s20_proc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mk = |pid: u32, ppid: u32, name: &str, rss_kb: u64, hwm_kb: u64| {
            let d = root.join(pid.to_string());
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(
                d.join("status"),
                format!("Name:\t{name}\nPPid:\t{ppid}\nVmHWM:\t{hwm_kb} kB\nVmRSS:\t{rss_kb} kB\n"),
            )
            .unwrap();
        };
        mk(100, 1, "rust-singularit", 50_000, 60_000);
        mk(101, 100, "WebKitNetworkPr", 9_000, 10_000);
        mk(102, 100, "bwrap", 1_000, 1_000);
        mk(103, 102, "WebKitWebProces", 20_000, 25_000);
        mk(200, 1, "WebKitWebProces", 1, 1); // alt arbore: nu se numără
        std::fs::create_dir_all(root.join("self")).unwrap();
        assert_eq!(descendants(&root, 100).unwrap(), vec![101, 102, 103]);
        let r = memory_report_at(&root, 100, WEBVIEW_LIMIT_BYTES);
        assert_eq!(r.webview_procese.iter().map(|p| p.pid).collect::<Vec<_>>(), vec![101, 103]);
        assert_eq!(r.webview_hwm_total_octeti, Some(35_000 * 1024));
        assert_eq!(r.webview_rss_total_octeti, Some(29_000 * 1024));
        assert_eq!(r.webview_pss_total_octeti, None);
        assert_eq!(r.verdict, "FAIL"); // 35 840 000 B > 30 000 000 B
        let r2 = memory_report_at(&root, 100, 35_000 * 1024);
        assert_eq!(r2.verdict, "PASS");
        assert_eq!(r.alte_descendente.len(), 1);
        let none = memory_report_at(&root, 200, WEBVIEW_LIMIT_BYTES);
        assert_eq!(none.verdict, "UNKNOWN");
        assert!(none.motiv.is_some());
        let missing = memory_report_at(&root.join("nu-exista"), 100, WEBVIEW_LIMIT_BYTES);
        assert_eq!(missing.verdict, "UNKNOWN");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn real_child_process_is_found() {
        if !Path::new("/proc/self/status").exists() {
            return;
        }
        let mut child = std::process::Command::new("/bin/sleep").arg("5").spawn().unwrap();
        let me = std::process::id();
        let found = descendants(Path::new("/proc"), me).unwrap();
        let ok = found.contains(&child.id());
        let p = read_proc(Path::new("/proc"), child.id());
        let _ = child.kill();
        let _ = child.wait();
        assert!(ok, "copilul {} nu a fost găsit în {found:?}", child.id());
        assert_eq!(p.unwrap().name.as_deref(), Some("sleep"));
        let r = memory_report(me);
        let pr = r.principal.unwrap();
        assert!(pr.rss_bytes.unwrap() > 0 && pr.hwm_bytes.unwrap() >= pr.rss_bytes.unwrap());
        assert_eq!(r.verdict, "UNKNOWN"); // testul nu are Webview
        assert!(!r.positive_claims_allowed);
    }
}

//! Cititor pur Rust (zero dependențe externe suplimentare, peste `flate2`) pentru arhivele binare
//! NumPy `.npz` (ZIP Stored + Deflate) și tablourile `.npy` v1.0 / v2.0 / v3.0 (`<f8`, `<i8`, `<U*`).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use flate2::read::DeflateDecoder;

#[derive(Debug, Clone)]
pub enum NpyData {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Str(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct NpyArray {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn as_f64(&self) -> Result<&[f64], String> {
        match &self.data {
            NpyData::F64(v) => Ok(v.as_slice()),
            _ => Err(format!("așteptat float64 (<f8), găsit {}", self.descr)),
        }
    }

    pub fn as_i64(&self) -> Result<&[i64], String> {
        match &self.data {
            NpyData::I64(v) => Ok(v.as_slice()),
            _ => Err(format!("așteptat int64 (<i8), găsit {}", self.descr)),
        }
    }

    pub fn as_strings(&self) -> Result<&[String], String> {
        match &self.data {
            NpyData::Str(v) => Ok(v.as_slice()),
            _ => Err(format!("așteptat șiruri (<U*), găsit {}", self.descr)),
        }
    }
}

fn read_u16_le(buf: &[u8], off: usize) -> Result<u16, String> {
    let slice = buf.get(off..off + 2).ok_or("ZIP trunchiat (u16)")?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32, String> {
    let slice = buf.get(off..off + 4).ok_or("ZIP trunchiat (u32)")?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Extrage valoarea asociată unei chei dintr-un dicționar literal Python din antetul `.npy`.
fn extract_py_dict_field<'a>(hdr: &'a str, key: &str) -> Option<&'a str> {
    for quote in ['\'', '"'] {
        let pat = format!("{quote}{key}{quote}");
        if let Some(pos) = hdr.find(&pat) {
            let after_key = &hdr[pos + pat.len()..];
            let colon = after_key.find(':')?;
            return Some(after_key[colon + 1..].trim_start());
        }
    }
    None
}

fn parse_py_quoted_str(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let q = *b.first()?;
    if q != b'\'' && q != b'"' {
        return None;
    }
    let end = s[1..].find(q as char)?;
    Some(s[1..1 + end].to_string())
}

fn parse_py_shape(s: &str) -> Result<Vec<usize>, String> {
    let open = s.find('(').ok_or("antet .npy fără '(' la shape")?;
    let close = s[open + 1..].find(')').ok_or("antet .npy fără ')' la shape")?;
    let inner = s[open + 1..open + 1 + close].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let d = t.parse::<usize>().map_err(|e| format!("dimensiune invalidă în shape '{t}': {e}"))?;
        dims.push(d);
    }
    Ok(dims)
}

/// Parsează un fișier binar `.npy` din memorie.
pub fn parse_npy_bytes(bytes: &[u8]) -> Result<NpyArray, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("semnătură .npy invalidă".into());
    }
    let major = bytes[6];
    let (hdr_len, data_off) = match major {
        1 => {
            let hl = read_u16_le(bytes, 8)? as usize;
            (hl, 10 + hl)
        }
        2 | 3 => {
            let hl = read_u32_le(bytes, 8)? as usize;
            (hl, 12 + hl)
        }
        other => return Err(format!("versiune majoră .npy nesuportată: {other}")),
    };
    if data_off > bytes.len() {
        return Err("antet .npy depășește lungimea fișierului".into());
    }
    let hdr_str = std::str::from_utf8(&bytes[data_off - hdr_len..data_off])
        .map_err(|e| format!("antet .npy UTF-8 invalid: {e}"))?;

    let descr = extract_py_dict_field(hdr_str, "descr")
        .and_then(parse_py_quoted_str)
        .ok_or_else(|| format!("câmp descr lipsă în .npy: {hdr_str}"))?;
    if let Some(fo) = extract_py_dict_field(hdr_str, "fortran_order") {
        if fo.starts_with("True") {
            return Err("tablourile fortran_order=True nu sunt suportate".into());
        }
    }
    let shape_raw =
        extract_py_dict_field(hdr_str, "shape").ok_or_else(|| format!("câmp shape lipsă în .npy: {hdr_str}"))?;
    let shape = parse_py_shape(shape_raw)?;
    let count: usize = shape.iter().copied().product::<usize>();
    let payload = &bytes[data_off..];

    let data = if descr == "<f8" || descr == "=f8" {
        let need = count.checked_mul(8).ok_or("overflow dimensiune <f8")?;
        if payload.len() < need {
            return Err(format!("payload .npy <f8 trunchiat: {} < {need}", payload.len()));
        }
        let mut out = Vec::with_capacity(count);
        for chunk in payload[..need].chunks_exact(8) {
            out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        NpyData::F64(out)
    } else if descr == "<i8" || descr == "=i8" {
        let need = count.checked_mul(8).ok_or("overflow dimensiune <i8")?;
        if payload.len() < need {
            return Err(format!("payload .npy <i8 trunchiat: {} < {need}", payload.len()));
        }
        let mut out = Vec::with_capacity(count);
        for chunk in payload[..need].chunks_exact(8) {
            out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
        }
        NpyData::I64(out)
    } else if let Some(rest) = descr.strip_prefix("<U") {
        let n_chars = rest.parse::<usize>().map_err(|e| format!("lungime <U invalidă '{descr}': {e}"))?;
        let stride = n_chars.checked_mul(4).ok_or("overflow stride <U")?;
        let need = count.checked_mul(stride).ok_or("overflow dimensiune <U")?;
        if payload.len() < need {
            return Err(format!("payload .npy {descr} trunchiat: {} < {need}", payload.len()));
        }
        let mut out = Vec::with_capacity(count);
        if stride == 0 {
            out.resize(count, String::new());
        } else {
            for item in payload[..need].chunks_exact(stride) {
                let mut s = String::with_capacity(n_chars);
                for ch_bytes in item.chunks_exact(4) {
                    let cp = u32::from_le_bytes(ch_bytes.try_into().unwrap());
                    if cp == 0 {
                        continue;
                    }
                    if let Some(c) = char::from_u32(cp) {
                        s.push(c);
                    }
                }
                // Eliminăm doar caracterele NUL finale (deja ignorate mai sus)
                out.push(s);
            }
        }
        NpyData::Str(out)
    } else {
        return Err(format!("tip de date .npy nesuportat: {descr}"));
    };

    Ok(NpyArray { descr, shape, data })
}

/// Încarcă toate tablourile dintr-o arhivă `.npz` (ZIP Central Directory, fără citiri oarbe secvențiale cu data-descriptor).
pub fn load_npz_archive(path: &Path) -> Result<BTreeMap<String, NpyArray>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 22 {
        return Err(format!("{}: arhivă .npz prea scurtă", path.display()));
    }

    // Căutăm End of Central Directory (PK\x05\x06) de la coadă
    let min_eocd_off = bytes.len().saturating_sub(65_557);
    let mut eocd_pos = None;
    for pos in (min_eocd_off..=bytes.len() - 22).rev() {
        if &bytes[pos..pos + 4] == b"PK\x05\x06" {
            eocd_pos = Some(pos);
            break;
        }
    }
    let eocd = eocd_pos.ok_or_else(|| format!("{}: EOCD ZIP negăsit", path.display()))?;
    let total_entries = read_u16_le(&bytes, eocd + 10)? as usize;
    let cd_offset = read_u32_le(&bytes, eocd + 16)? as usize;

    let mut cur = cd_offset;
    let mut out = BTreeMap::new();

    for _ in 0..total_entries {
        if cur + 46 > bytes.len() || &bytes[cur..cur + 4] != b"PK\x01\x02" {
            return Err(format!("{}: antet Central Directory ZIP invalid la {cur}", path.display()));
        }
        let method = read_u16_le(&bytes, cur + 10)?;
        let comp_size = read_u32_le(&bytes, cur + 20)? as usize;
        let uncomp_size = read_u32_le(&bytes, cur + 24)? as usize;
        let name_len = read_u16_le(&bytes, cur + 28)? as usize;
        let extra_len = read_u16_le(&bytes, cur + 30)? as usize;
        let comment_len = read_u16_le(&bytes, cur + 32)? as usize;
        let local_off = read_u32_le(&bytes, cur + 42)? as usize;

        let name_end = cur + 46 + name_len;
        if name_end > bytes.len() {
            return Err("nume intrare ZIP trunchiat".into());
        }
        let raw_name =
            std::str::from_utf8(&bytes[cur + 46..name_end]).map_err(|e| format!("nume ZIP UTF-8 invalid: {e}"))?;
        cur = name_end + extra_len + comment_len;

        if local_off + 30 > bytes.len() || &bytes[local_off..local_off + 4] != b"PK\x03\x04" {
            return Err(format!("{}: Local File Header ZIP invalid la {local_off}", path.display()));
        }
        let l_name_len = read_u16_le(&bytes, local_off + 26)? as usize;
        let l_extra_len = read_u16_le(&bytes, local_off + 28)? as usize;
        let data_start = local_off + 30 + l_name_len + l_extra_len;
        let data_end = data_start.checked_add(comp_size).ok_or("overflow ZIP data_end")?;
        if data_end > bytes.len() {
            return Err("date comprimate ZIP trunchiate".into());
        }
        let comp_slice = &bytes[data_start..data_end];

        let raw_npy = match method {
            0 => comp_slice.to_vec(),
            8 => {
                let mut dec = DeflateDecoder::new(comp_slice);
                let mut buf = Vec::with_capacity(uncomp_size);
                dec.read_to_end(&mut buf).map_err(|e| format!("eroare Deflate în {}: {e}", path.display()))?;
                buf
            }
            other => return Err(format!("metodă compresie ZIP nesuportată ({other}) în {}", path.display())),
        };

        let key = raw_name.strip_suffix(".npy").unwrap_or(raw_name).to_string();
        let arr = parse_npy_bytes(&raw_npy).map_err(|e| format!("{}[{key}]: {e}", path.display()))?;
        out.insert(key, arr);
    }

    Ok(out)
}

/// Eșantionează uniform o serie de lungime `T` la cel mult `max_pts` puncte (păstrând întotdeauna capetele `0` și `T-1`).
pub fn downsample_indices(len: usize, max_pts: usize) -> Vec<usize> {
    if len == 0 || max_pts == 0 {
        return Vec::new();
    }
    if len <= max_pts || max_pts == 1 {
        return (0..len.min(max_pts)).collect();
    }
    let mut idxs = Vec::with_capacity(max_pts);
    for k in 0..max_pts {
        let pos = k * (len - 1) / (max_pts - 1);
        if idxs.last().copied() != Some(pos) {
            idxs.push(pos);
        }
    }
    idxs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn reads_hsv4_npz_archive_cleanly() {
        let p = PathBuf::from(
            "./data/quantum_lab_649/hs_v4/rezultate/ro_649_full_hsv4_walkforward.npz",
        );
        if !p.exists() {
            return;
        }
        let npz = load_npz_archive(&p).expect("load_npz_archive");
        let e_curve = npz.get("E_curve").expect("E_curve").as_f64().expect("f64");
        let dates = npz.get("eval_dates").expect("eval_dates").as_strings().expect("strings");
        let w_curve = npz.get("w_portfolio_curve").expect("w_portfolio_curve");
        assert_eq!(e_curve.len(), dates.len());
        assert_eq!(w_curve.shape, vec![e_curve.len(), 8]);
        assert!(e_curve.iter().all(|&v| v >= 0.50 - 1e-12));
        assert_eq!(dates[0].len(), 10);
    }
}

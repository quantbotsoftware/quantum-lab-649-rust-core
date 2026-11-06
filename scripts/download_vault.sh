#!/usr/bin/env bash
# Downloads and SHA-256 verifies the 1.3 GB RUST_CORE_V3_VAULT (.v3bin catalogs)
# directly from the official CERN Zenodo archival record (DOI: 10.5281/zenodo.23992530).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAULT_DIR="${QL649_VAULT_DIR:-$REPO_DIR/vault}"
MANIFEST="$REPO_DIR/vault/VAULT_MANIFEST.json"
ZENODO_CONCEPT_ID="23992530"
ZENODO_FALLBACK_ID="23994914"

# Resolve the LATEST Zenodo version of the Vault at runtime (concept DOI 10.5281/zenodo.23992530);
# fall back to the record id this package was built against if the API is unreachable.
resolve_latest_record() {
  python3 - "$ZENODO_CONCEPT_ID" "$ZENODO_FALLBACK_ID" <<'PYR'
import json, sys, urllib.request
concept, fallback = sys.argv[1], sys.argv[2]
try:
    url = f"https://zenodo.org/api/records?q=conceptrecid:{concept}&sort=mostrecent&size=1&all_versions=true"
    with urllib.request.urlopen(url, timeout=20) as r:
        hits = json.loads(r.read().decode("utf-8"))["hits"]["hits"]
    print(hits[0]["id"] if hits else fallback)
except Exception:
    print(fallback)
PYR
}

if [ -n "${QL649_VAULT_MIRROR_URL:-}" ]; then
  MIRROR_BASE="$QL649_VAULT_MIRROR_URL"
else
  MIRROR_BASE="https://zenodo.org/records/$(resolve_latest_record)/files"
fi
echo "Vault mirror: $MIRROR_BASE"

mkdir -p "$VAULT_DIR"
python3 - "$MANIFEST" "$VAULT_DIR" "$MIRROR_BASE" <<'PY'
import hashlib, json, pathlib, shutil, sys, urllib.request

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
vault_dir = pathlib.Path(sys.argv[2])
base = sys.argv[3].rstrip("/")

files = manifest.get("files", [])
print(f"Checking {len(files)} Vault files in {vault_dir} (CERN Zenodo DOI: 10.5281/zenodo.23992530) ...")
for rec in files:
    fn = rec["filename"]
    expected = rec["sha256"]
    dest = vault_dir / fn
    if dest.exists() and dest.stat().st_size == rec["file_bytes"]:
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if h == expected:
            print(f"  ✔ {fn} already verified")
            continue
    url = f"{base}/{fn}"
    print(f"  ↓ Downloading {fn} ({rec['file_bytes']/1e6:.1f} MB) from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 QuantBotSoftware-VaultDownloader/1.0"})
    with urllib.request.urlopen(req, timeout=1800) as resp, open(dest, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1024 * 1024)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if h != expected:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"  ✖ SHA-256 mismatch for {fn}: got {h}, expected {expected}")
    print(f"  ✔ {fn} verified (SHA-256 OK)")
PY

#!/usr/bin/env bash
# Downloads and SHA-256 verifies the 1.3 GB RUST_CORE_V3_VAULT (.v3bin catalogs)
# from the configured mirror (Zenodo / Hugging Face / GitHub Releases).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAULT_DIR="${QL649_VAULT_DIR:-$REPO_DIR/vault}"
MANIFEST="$REPO_DIR/vault/VAULT_MANIFEST.json"
MIRROR_BASE="${QL649_VAULT_MIRROR_URL:-https://huggingface.co/datasets/bot-tradingsoftware/quantum-lab-649-vault-v3/resolve/main}"

mkdir -p "$VAULT_DIR"
python3 - "$MANIFEST" "$VAULT_DIR" "$MIRROR_BASE" <<'PY'
import hashlib, json, pathlib, sys, urllib.request

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
vault_dir = pathlib.Path(sys.argv[2])
base = sys.argv[3].rstrip("/")

files = manifest.get("files", [])
print(f"Checking {len(files)} Vault files in {vault_dir} ...")
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
    urllib.request.urlretrieve(url, dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if h != expected:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"  ✖ SHA-256 mismatch for {fn}: got {h}, expected {expected}")
    print(f"  ✔ {fn} verified (SHA-256 OK)")
PY

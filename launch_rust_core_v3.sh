#!/usr/bin/env bash
# Portable launcher for Quantum Lab 649 · Rust Core v3 (Tauri v2 + CUDA + 24T AVX2).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
BIN="$REPO_DIR/target/release/rust-singularity-core-v3"

if [ "${1:-}" = "--stop" ]; then
  pkill -f "rust-singularity-core-v3 gui" 2>/dev/null || true
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "[rust-core-v3] Compiling release binary..."
  cargo build --release
fi

export WEBKIT_DISABLE_DMABUF_RENDERER=1
exec "$BIN" gui

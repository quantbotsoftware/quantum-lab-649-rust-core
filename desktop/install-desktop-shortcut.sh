#!/usr/bin/env bash
# Generates and installs the Linux .desktop shortcut using the current repo location.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_DIR/desktop/quantum-lab-649-rust-core.desktop.in"
OUT_APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$OUT_APPS"
sed "s|@REPO_DIR@|$REPO_DIR|g" "$TEMPLATE" > "$OUT_APPS/quantum-lab-649-rust-core.desktop"
chmod +x "$OUT_APPS/quantum-lab-649-rust-core.desktop"
if [ -d "$HOME/Desktop" ]; then
  cp "$OUT_APPS/quantum-lab-649-rust-core.desktop" "$HOME/Desktop/quantum-lab-649-rust-core.desktop"
  chmod +x "$HOME/Desktop/quantum-lab-649-rust-core.desktop"
fi
echo "✔ Installed quantum-lab-649-rust-core.desktop pointing to $REPO_DIR"

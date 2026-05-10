#!/usr/bin/env bash
# Build the ICML 2026 paper from REPORT.tex + references.bib.
# Requires Tectonic (https://tectonic-typesetting.github.io/).
# If `tectonic` is not on PATH, this script downloads a portable binary into
# ~/.local/bin and uses that.

set -e
cd "$(dirname "$0")"

# Ensure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Install tectonic if missing
if ! command -v tectonic >/dev/null 2>&1; then
    echo "tectonic not found, installing portable binary into ~/.local/bin"
    mkdir -p ~/.local/bin
    cd /tmp
    curl -sL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz \
        | tar xz
    mv tectonic ~/.local/bin/
    cd "$OLDPWD"
fi

echo "Compiling REPORT.tex with tectonic..."
tectonic REPORT.tex

echo
echo "Output: $(pwd)/REPORT.pdf"
ls -lh REPORT.pdf

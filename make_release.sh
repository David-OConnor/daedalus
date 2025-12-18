#!/usr/bin/env bash
set -euo pipefail

version=0_2_3
outdir=target/release
exe="$outdir/dchemforma"
readme="README.md"
setup="install_scripts/setup_linux_desktop.sh"
icon="resources/icon.png"
cufft="/usr/local/cuda/lib64/libcufft.so.12"

export PATH=/usr/local/cuda/bin:$PATH

chmod +x "$setup"

cargo build --release
zip -j -r "chemforma_${version}_linux.zip" "$exe" "$readme" "$setup" "$icon" "$cufft"

cargo build --release --no-default-features
zip -j -r "chemforma_${version}_linux_nocuda.zip" "$exe" "$readme" "$setup" "$icon"

#!/usr/bin/env bash
set -euo pipefail

version=0_2_1
outdir=target/release
exe="$outdir/daedalus"
gemmi="/mnt/c/Users/the_a/Program Files/gemmi"
readme="README.md"
setup="install_scripts/setup_linux_desktop.sh"
icon="resources/icon.png"
cufft="/usr/local/cuda/lib64/libcufft.so.12"

export PATH=/usr/local/cuda/bin:$PATH

chmod +x "$setup"

cargo build --release
zip -j -r "daedalus_${version}_linux.zip" "$exe" "$gemmi" "$readme" "$setup" "$icon" "$cufft"

cargo build --release --no-default-features
zip -j -r "daedalus_${version}_linux_nocuda.zip" "$exe" "$gemmi" "$readme" "$setup" "$icon"

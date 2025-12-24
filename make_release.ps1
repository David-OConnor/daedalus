$ErrorActionPreference = "Stop"

$version = "0_2_4"
$outDir = "target\release"
$exe    = Join-Path $outDir "molchanica.exe"
$gemmi  = "C:\Program Files\gemmi"
$readme = "README.md"
$cufft = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cufft64_12.dll"

cargo build --release
$zip1 = "molchanica_${version}_win.zip"
if (Test-Path $zip1) { Remove-Item $zip1 -Force }
Compress-Archive -LiteralPath $exe, $gemmi, $readme, $cufft -DestinationPath $zip1 -Force

cargo build --release --no-default-features
$zip2 = "molchanica_${version}_win_nocuda.zip"
if (Test-Path $zip2) { Remove-Item $zip2 -Force }
Compress-Archive -LiteralPath $exe, $gemmi, $readme -DestinationPath $zip2 -Force
# Computes SHA256 checksums for every .whl in wheelhouse/, writes hashes.sha256
# at the project root. Called from build_wheelhouse.bat.

$files = Get-ChildItem -Path "wheelhouse" -Filter "*.whl"

if ($files.Count -eq 0) {
    Write-Host "No .whl files found in wheelhouse/ - nothing to hash."
    exit 1
}

$lines = foreach ($f in $files) {
    $hash = (Get-FileHash -Path $f.FullName -Algorithm SHA256).Hash
    "$hash  $($f.Name)"
}

$lines | Out-File -Encoding ascii "hashes.sha256"
Write-Host "Wrote hashes.sha256 with $($files.Count) entries"
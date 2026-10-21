# Verifies every .whl in wheelhouse/ against hashes.sha256, catching any
# corruption from the Drive/email transfer before pip tries to use a broken
# wheel. Called from install.bat. Exit 0 = OK or skipped, exit 1 = failed.

if (-not (Test-Path "hashes.sha256")) {
    Write-Host "hashes.sha256 not found - skipping integrity check"
    Write-Host "(re-run scripts\build_wheelhouse.bat at home to generate it)"
    exit 0
}

$ok = $true

Get-Content "hashes.sha256" | ForEach-Object {
    $parts = $_ -split '  ', 2
    if ($parts.Count -lt 2) { return }
    $expectedHash = $parts[0]
    $fileName = $parts[1]
    $filePath = Join-Path "wheelhouse" $fileName

    if (Test-Path $filePath) {
        $actualHash = (Get-FileHash -Path $filePath -Algorithm SHA256).Hash
        if ($actualHash -ne $expectedHash) {
            Write-Host "MISMATCH: $fileName"
            $ok = $false
        }
    } else {
        Write-Host "MISSING: $fileName"
        $ok = $false
    }
}

if (-not $ok) {
    Write-Host "Integrity check FAILED - re-transfer the wheelhouse"
    exit 1
} else {
    Write-Host "Integrity check passed"
    exit 0
}
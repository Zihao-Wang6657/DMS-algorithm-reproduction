[CmdletBinding()]
param(
    [string]$CondaEnv = "android_world"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$repoRoot\src;$repoRoot\third_party\android_world"

Push-Location $repoRoot
try {
    Write-Host "[1/2] Running unit and integration tests"
    & conda run -n $CondaEnv python -m pytest tests -q
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed with exit code $LASTEXITCODE"
    }

    Write-Host "[2/2] Validating AndroidWorld task datasets"
    & conda run -n $CondaEnv python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        throw "dataset validation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "quick_checks_complete=1"

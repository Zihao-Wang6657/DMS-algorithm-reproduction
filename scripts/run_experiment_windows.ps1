[CmdletBinding()]
param(
    [ValidateSet("baseline_a_zero_shot", "baseline_b_static_memory", "dms_hierarchical_memory")]
    [string]$Method = "baseline_a_zero_shot",
    [string]$Dataset = "datasets/smoke_open_settings.yaml",
    [int]$Rounds = 1,
    [string]$RunDir,
    [string]$CondaEnv = "android_world",
    [string]$BaseUrl = "http://127.0.0.1:8000/v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$datasetPath = if ([System.IO.Path]::IsPathRooted($Dataset)) {
    (Resolve-Path $Dataset).Path
} else {
    (Resolve-Path (Join-Path $repoRoot $Dataset)).Path
}
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"

if (-not (Test-Path -LiteralPath $adb)) {
    throw "ADB is missing: $adb"
}
$devices = @(& $adb devices)
if (-not ($devices -match '^emulator-5554\s+device$')) {
    throw "AndroidWorld emulator-5554 is not ready. Run scripts/start_local_androidworld.ps1 first."
}

try {
    $null = Invoke-RestMethod -Uri "$BaseUrl/models" -Method Get -TimeoutSec 30
}
catch {
    throw "Remote model API is unavailable at $BaseUrl. Start the server and SSH tunnel first. $($_.Exception.Message)"
}

$env:PYTHONPATH = "$repoRoot\src;$repoRoot\third_party\android_world"
$env:DMS_MODEL_BASE_URL = $BaseUrl
$arguments = @(
    "run", "-n", $CondaEnv, "--no-capture-output", "python", "-m", "dms.runner",
    "--method", $Method,
    "--config", "$repoRoot\configs\eval_baselines.yaml",
    "--model-config", "$repoRoot\configs\model_qwen25vl_7b_remote.yaml",
    "--runtime-config", "$repoRoot\configs\runtime_windows.yaml",
    "--dataset", $datasetPath,
    "--rounds", "$Rounds"
)
if ($RunDir) {
    $arguments += @("--run-dir", $RunDir)
}

Write-Host "method=$Method"
Write-Host "dataset=$datasetPath"
Write-Host "rounds=$Rounds"
& conda @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed with exit code $LASTEXITCODE"
}

[CmdletBinding()]
param(
    [string]$RunRoot = "runs/formal_main_20apps_5rounds_v1_20260718",
    [string]$Dataset = "datasets/formal_main_20apps_v1.yaml",
    [string]$Manifest = "protocols/formal_main_v1/protocol_manifest.json",
    [string]$Config = "configs/eval_baselines.yaml",
    [string]$ModelConfig = "configs/model_qwen25vl_7b_remote.yaml",
    [string]$RuntimeConfig = "configs/runtime_windows.yaml",
    [string]$CondaEnv = "android_world",
    [string]$BaseUrl = "http://127.0.0.1:8000/v1",
    [int]$MaxProcessRestarts = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    return Join-Path $repoRoot $Value
}

$resolvedRunRoot = Resolve-RepoPath $RunRoot
$resolvedDataset = Resolve-RepoPath $Dataset
$resolvedManifest = Resolve-RepoPath $Manifest
$resolvedConfig = Resolve-RepoPath $Config
$resolvedModelConfig = Resolve-RepoPath $ModelConfig
$resolvedRuntimeConfig = Resolve-RepoPath $RuntimeConfig
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$statusPath = Join-Path $resolvedRunRoot "formal_orchestrator_status.json"
$identityPath = Join-Path $resolvedRunRoot "FORMAL_RUN_ROOT.json"

if (Test-Path $resolvedRunRoot) {
    $existing = @(Get-ChildItem -LiteralPath $resolvedRunRoot -Force)
    if ($existing.Count -gt 0 -and -not (Test-Path $identityPath)) {
        throw "Refusing to use a non-empty directory without FORMAL_RUN_ROOT.json: $resolvedRunRoot"
    }
}
New-Item -ItemType Directory -Force -Path $resolvedRunRoot | Out-Null

$manifestPayload = Get-Content -LiteralPath $resolvedManifest -Raw | ConvertFrom-Json
if ($manifestPayload.protocol_id -ne "formal_main_20apps_v1") {
    throw "Unexpected formal protocol id: $($manifestPayload.protocol_id)"
}
if (-not (Test-Path $identityPath)) {
    @{
        protocol_id = $manifestPayload.protocol_id
        manifest = $resolvedManifest
        created_at = (Get-Date).ToString("o")
        pilot_data_included = $false
    } | ConvertTo-Json | Set-Content -LiteralPath $identityPath -Encoding UTF8
}

function Write-FormalStatus {
    param([string]$State, [string]$Method, [int]$Attempt, [string]$Message)
    @{
        timestamp = (Get-Date).ToString("o")
        state = $State
        method = $Method
        process_attempt = $Attempt
        message = $Message
        run_root = $resolvedRunRoot
        protocol_id = $manifestPayload.protocol_id
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Assert-FormalServicesReady {
    $devices = @(& $adb devices)
    if (-not ($devices -match '^emulator-5554\s+device$')) {
        throw "AndroidWorld emulator-5554 is unavailable."
    }
    $boot = (& $adb -s emulator-5554 shell getprop sys.boot_completed).Trim()
    if ($boot -ne "1") {
        throw "AndroidWorld emulator has not finished booting."
    }
    $null = Invoke-RestMethod -Uri "$BaseUrl/models" -Method Get -TimeoutSec 30
}

$keepAwakeScript = Join-Path $PSScriptRoot "keep_awake_while_process.ps1"
Start-Process -FilePath powershell.exe -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $keepAwakeScript,
    "-OrchestratorPid", "$PID"
) -WindowStyle Hidden | Out-Null

$methods = @(
    @{ Key = "baseline_a"; RunnerName = "baseline_a_zero_shot" },
    @{ Key = "baseline_b"; RunnerName = "baseline_b_static_memory" },
    @{ Key = "dms"; RunnerName = "dms_hierarchical_memory" }
)

$env:PYTHONPATH = "$repoRoot\src;$repoRoot\third_party\android_world"
$env:DMS_MODEL_BASE_URL = $BaseUrl
$env:DMS_STRICT_INFRA_PROTOCOL = "1"

foreach ($method in $methods) {
    $methodDir = Join-Path $resolvedRunRoot $method.Key
    New-Item -ItemType Directory -Force -Path $methodDir | Out-Null
    $logPath = Join-Path $resolvedRunRoot ("{0}.log" -f $method.Key)
    $completed = $false
    for ($attempt = 1; $attempt -le $MaxProcessRestarts; $attempt++) {
        Write-FormalStatus -State "checking_services" -Method $method.Key -Attempt $attempt -Message "Checking frozen formal runtime."
        Assert-FormalServicesReady
        Write-FormalStatus -State "running" -Method $method.Key -Attempt $attempt -Message "Formal runner is active."
        & conda run -n $CondaEnv --no-capture-output python -m dms.formal_runner `
            --method $method.RunnerName `
            --config $resolvedConfig `
            --model-config $resolvedModelConfig `
            --runtime-config $resolvedRuntimeConfig `
            --dataset $resolvedDataset `
            --manifest $resolvedManifest `
            --run-dir $methodDir `
            --rounds 5 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -eq 0) {
            $completed = $true
            break
        }
        Write-FormalStatus -State "process_restart_pending" -Method $method.Key -Attempt $attempt -Message "Process exited unexpectedly; durable task transaction will enforce the one-retry rule."
        Start-Sleep -Seconds 10
    }
    if (-not $completed) {
        Write-FormalStatus -State "stopped_needs_attention" -Method $method.Key -Attempt $MaxProcessRestarts -Message "Process restart limit reached; no protocol changes were made."
        throw "Formal method $($method.Key) needs user attention."
    }
}

Write-FormalStatus -State "analyzing" -Method "all" -Attempt 1 -Message "Validating 300 scored records and generating the requested figures."
& conda run -n $CondaEnv --no-capture-output python (Join-Path $PSScriptRoot "analyze_main_experiment.py") `
    --run-root $resolvedRunRoot `
    --dataset $resolvedDataset `
    --rounds 5 `
    --tasks-per-round 20
if ($LASTEXITCODE -ne 0) {
    Write-FormalStatus -State "analysis_failed" -Method "all" -Attempt 1 -Message "Formal attempts completed, but strict analysis validation failed."
    throw "Formal analysis failed with exit code $LASTEXITCODE"
}

Write-FormalStatus -State "complete" -Method "all" -Attempt 1 -Message "All 300 formal scored records and four figures are complete."

[CmdletBinding()]
param(
    [string]$RunRoot = "runs/formal_recovery_bd_20apps_5rounds_v2_20260719",
    [string]$Dataset = "datasets/formal_main_20apps_v1.yaml",
    [string]$Manifest = "protocols/formal_recovery_bd_v2/protocol_manifest.json",
    [string]$Config = "configs/eval_baselines.yaml",
    [string]$ModelConfig = "configs/model_qwen25vl_7b_remote.yaml",
    [string]$RuntimeConfig = "configs/runtime_windows.yaml",
    [string]$CondaEnv = "android_world",
    [string]$BaseUrl = "http://127.0.0.1:8000/v1"
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
$statusPath = Join-Path $resolvedRunRoot "recovery_orchestrator_status.json"
$identityPath = Join-Path $resolvedRunRoot "FORMAL_RECOVERY_RUN_ROOT.json"

$manifestPayload = Get-Content -LiteralPath $resolvedManifest -Raw | ConvertFrom-Json
if ($manifestPayload.protocol_id -ne "formal_recovery_bd_20apps_v2") {
    throw "Unexpected recovery protocol id: $($manifestPayload.protocol_id)"
}
if ([System.IO.Path]::GetFullPath($manifestPayload.formal_run_root) -ne [System.IO.Path]::GetFullPath($resolvedRunRoot)) {
    throw "RunRoot differs from the frozen recovery manifest."
}

if (Test-Path -LiteralPath $resolvedRunRoot) {
    $existing = @(Get-ChildItem -LiteralPath $resolvedRunRoot -Force)
    if ($existing.Count -gt 0 -and -not (Test-Path -LiteralPath $identityPath)) {
        throw "Refusing non-empty recovery RunRoot without identity file: $resolvedRunRoot"
    }
}
New-Item -ItemType Directory -Force -Path $resolvedRunRoot | Out-Null
if (-not (Test-Path -LiteralPath $identityPath)) {
    @{
        protocol_id = $manifestPayload.protocol_id
        created_at = (Get-Date).ToString("o")
        baseline_a_reused = $true
        baseline_a_run_root = $manifestPayload.cross_run_design.baseline_a_run_root
        pilot_data_included = $false
    } | ConvertTo-Json | Set-Content -LiteralPath $identityPath -Encoding UTF8
}

function Write-RecoveryStatus {
    param([string]$State, [string]$Method, [int]$Round, [string]$Message)
    @{
        timestamp = (Get-Date).ToString("o")
        state = $State
        method = $Method
        round = $Round
        message = $Message
        run_root = $resolvedRunRoot
        protocol_id = $manifestPayload.protocol_id
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Assert-ServicesReady {
    $devices = @(& $adb devices)
    if (-not ($devices -match '^emulator-5554\s+device$')) {
        throw "AndroidWorld emulator-5554 is unavailable."
    }
    if ((& $adb -s emulator-5554 shell getprop sys.boot_completed).Trim() -ne "1") {
        throw "AndroidWorld emulator has not finished booting."
    }
    $models = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 30
    $served = @($models.data | ForEach-Object { $_.id })
    if ($served -notcontains $manifestPayload.model_config.model.name) {
        throw "Frozen model is not available from the local API tunnel."
    }
}

function Restart-FrozenEmulator {
    $devices = @(& $adb devices)
    if ($devices -match '^emulator-5554\s+') {
        & $adb -s emulator-5554 emu kill | Out-Null
        $deadline = (Get-Date).AddSeconds(60)
        do {
            Start-Sleep -Seconds 2
            $stillPresent = @(& $adb devices) -match '^emulator-5554\s+'
        } while ($stillPresent -and (Get-Date) -lt $deadline)
        if ($stillPresent) { throw "emulator-5554 did not stop cleanly." }
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot "start_local_androidworld.ps1") `
        -AvdName "AndroidWorldAvd" -ConsolePort 5554 -GrpcPort 8554 `
        -BootTimeoutSeconds 240
    if ($LASTEXITCODE -ne 0) { throw "AndroidWorld emulator restart failed." }
    Assert-ServicesReady
}

$keepAwakeScript = Join-Path $PSScriptRoot "keep_awake_while_process.ps1"
Start-Process -FilePath powershell.exe -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $keepAwakeScript,
    "-OrchestratorPid", "$PID"
) -WindowStyle Hidden | Out-Null

$env:PYTHONPATH = "$repoRoot\src;$repoRoot\third_party\android_world"
$env:DMS_MODEL_BASE_URL = $BaseUrl
$env:DMS_STRICT_INFRA_PROTOCOL = "1"

$methods = @(
    @{ Key = "baseline_b"; Runner = "baseline_b_static_memory" },
    @{ Key = "dms"; Runner = "dms_hierarchical_memory" }
)

foreach ($method in $methods) {
    $methodDir = Join-Path $resolvedRunRoot $method.Key
    New-Item -ItemType Directory -Force -Path $methodDir | Out-Null
    for ($roundLimit = 1; $roundLimit -le 5; $roundLimit++) {
        $resultsPath = Join-Path $methodDir "task_results.jsonl"
        $completedCount = if (Test-Path -LiteralPath $resultsPath) {
            @(Get-Content -LiteralPath $resultsPath | Where-Object { $_.Trim() }).Count
        } else { 0 }
        $targetCount = $roundLimit * 20
        if ($completedCount -ge $targetCount) { continue }

        Write-RecoveryStatus -State "restarting_emulator" -Method $method.Key `
            -Round $roundLimit -Message "Cold restarting the frozen emulator before this round."
        Restart-FrozenEmulator
        Write-RecoveryStatus -State "running" -Method $method.Key `
            -Round $roundLimit -Message "Formal recovery runner is active."

        $stdout = Join-Path $resolvedRunRoot ("{0}_r{1:00}.stdout.log" -f $method.Key, $roundLimit)
        $stderr = Join-Path $resolvedRunRoot ("{0}_r{1:00}.stderr.log" -f $method.Key, $roundLimit)
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & conda run -n $CondaEnv --no-capture-output python -m dms.formal_runner `
            --method $method.Runner `
            --config $resolvedConfig `
            --model-config $resolvedModelConfig `
            --runtime-config $resolvedRuntimeConfig `
            --dataset $resolvedDataset `
            --manifest $resolvedManifest `
            --run-dir $methodDir `
            --rounds 5 `
            --round-limit $roundLimit 1>> $stdout 2>> $stderr
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
        if ($exitCode -ne 0) {
            Write-RecoveryStatus -State "stopped_needs_attention" -Method $method.Key `
                -Round $roundLimit -Message "Round process exited unexpectedly; no protocol change was made."
            throw "Recovery method $($method.Key) round $roundLimit exited with $exitCode."
        }

        $completedCount = @(Get-Content -LiteralPath $resultsPath | Where-Object { $_.Trim() }).Count
        if ($completedCount -ne $targetCount) {
            throw "Expected $targetCount completed records after round $roundLimit, got $completedCount."
        }
    }
}

Write-RecoveryStatus -State "complete" -Method "all" -Round 5 `
    -Message "Baseline B and DMS each contain 100 recovered formal records."

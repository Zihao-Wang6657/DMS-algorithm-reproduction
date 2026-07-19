[CmdletBinding()]
param(
    [string]$RunRoot = "runs/formal_mini_5tasks_balanced_v1_20260719",
    [string]$Dataset = "datasets/formal_mini_5tasks_balanced_v1.yaml",
    [string]$Manifest = "protocols/formal_mini_5tasks_balanced_v1/protocol_manifest.json",
    [string]$Config = "configs/eval_baselines_mini_optimized.yaml",
    [string]$ModelConfig = "configs/model_qwen25vl_7b_remote_optimized.yaml",
    [string]$RuntimeConfig = "configs/runtime_windows.yaml",
    [string]$GoldenDirectory = "device_images/AndroidWorldAvd_clean_v3",
    [string]$AppSnapshotDirectory = "device_images/AndroidWorldAppSnapshots_clean_v3",
    [string]$DeviceImageManifest = "device_images/AndroidWorld_device_state_v3_manifest.json",
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

$resolvedRunRoot = [System.IO.Path]::GetFullPath((Resolve-RepoPath $RunRoot))
$expectedRunsRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "runs"))
if (-not $resolvedRunRoot.StartsWith($expectedRunsRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RunRoot must remain inside $expectedRunsRoot"
}
$resolvedDataset = Resolve-RepoPath $Dataset
$resolvedManifest = Resolve-RepoPath $Manifest
$resolvedConfig = Resolve-RepoPath $Config
$resolvedModelConfig = Resolve-RepoPath $ModelConfig
$resolvedRuntimeConfig = Resolve-RepoPath $RuntimeConfig
$resolvedGolden = Resolve-RepoPath $GoldenDirectory
$resolvedAppSnapshots = Resolve-RepoPath $AppSnapshotDirectory
$resolvedDeviceImageManifest = Resolve-RepoPath $DeviceImageManifest
$adb = Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe"
$statusPath = Join-Path $resolvedRunRoot "formal_orchestrator_status.json"
$identityPath = Join-Path $resolvedRunRoot "FORMAL_RUN_ROOT.json"

$manifestPayload = Get-Content -LiteralPath $resolvedManifest -Raw | ConvertFrom-Json
if ($manifestPayload.protocol_id -ne "formal_mini_5tasks_balanced_v1") {
    throw "Unexpected protocol id: $($manifestPayload.protocol_id)"
}
if ([System.IO.Path]::GetFullPath($manifestPayload.formal_run_root) -ne $resolvedRunRoot) {
    throw "RunRoot differs from the frozen manifest."
}
if ((Test-Path -LiteralPath (Join-Path $resolvedRunRoot "ABANDONED_INCOMPLETE.md")) -or
    (Test-Path -LiteralPath (Join-Path $resolvedRunRoot "resume_forbidden"))) {
    throw "Refusing an abandoned or resume-forbidden RunRoot: $resolvedRunRoot"
}
$taskCount = [int]$manifestPayload.tasks_per_round
$roundCount = [int]$manifestPayload.rounds

& conda run -n $CondaEnv --no-capture-output python `
    (Join-Path $PSScriptRoot "device_image_manifest.py") verify `
    --manifest $resolvedDeviceImageManifest
if ($LASTEXITCODE -ne 0) { throw "Golden device image integrity verification failed." }

if (Test-Path -LiteralPath $resolvedRunRoot) {
    $existing = @(Get-ChildItem -LiteralPath $resolvedRunRoot -Force)
    if ($existing.Count -gt 0 -and -not (Test-Path -LiteralPath $identityPath)) {
        throw "Refusing non-empty RunRoot without formal identity: $resolvedRunRoot"
    }
}
New-Item -ItemType Directory -Force -Path $resolvedRunRoot | Out-Null
if (-not (Test-Path -LiteralPath $identityPath)) {
    @{
        protocol_id = $manifestPayload.protocol_id
        created_at = (Get-Date).ToString("o")
        pilot_data_included = $false
        device_state_stored_outside_run_root = $true
        host_memory_starts_empty = $true
    } | ConvertTo-Json | Set-Content -LiteralPath $identityPath -Encoding UTF8
}

function Write-FormalStatus {
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

function Get-MethodStateFingerprint {
    param([string]$MethodDirectory)
    if (-not (Test-Path -LiteralPath $MethodDirectory)) { return "EMPTY" }
    $records = @(Get-ChildItem -LiteralPath $MethodDirectory -Recurse -Force -File |
        Sort-Object FullName | ForEach-Object {
            $relative = $_.FullName.Substring($MethodDirectory.Length)
            "$relative|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)"
        })
    $text = $records -join "`n"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes)) `
            -replace "-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Assert-ServicesReady {
    $devices = @(& $adb devices)
    if (-not ($devices -match "^emulator-5554\s+device$")) {
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

function Restore-RoundDevice {
    param([string]$MethodDirectory)
    $before = Get-MethodStateFingerprint $MethodDirectory
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot "restore_androidworld_golden.ps1") `
        -GoldenDirectory $resolvedGolden `
        -AppSnapshotDirectory $resolvedAppSnapshots
    if ($LASTEXITCODE -ne 0) { throw "Golden AVD restore failed." }
    $after = Get-MethodStateFingerprint $MethodDirectory
    if ($before -ne $after) {
        throw "Host experiment state changed during AVD-only restore."
    }
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
    @{ Key = "baseline_a"; Runner = "baseline_a_zero_shot" },
    @{ Key = "baseline_b"; Runner = "baseline_b_static_memory" },
    @{ Key = "dms"; Runner = "dms_hierarchical_memory" }
)

foreach ($method in $methods) {
    $methodDir = Join-Path $resolvedRunRoot $method.Key
    New-Item -ItemType Directory -Force -Path $methodDir | Out-Null
    for ($roundLimit = 1; $roundLimit -le $roundCount; $roundLimit++) {
        $resultsPath = Join-Path $methodDir "task_results.jsonl"
        $completedCount = if (Test-Path -LiteralPath $resultsPath) {
            @(Get-Content -LiteralPath $resultsPath | Where-Object { $_.Trim() }).Count
        } else { 0 }
        $targetCount = $roundLimit * $taskCount
        if ($completedCount -ge $targetCount) { continue }

        if ($roundLimit -eq 1 -and $completedCount -eq 0) {
            $unexpected = @(Get-ChildItem -LiteralPath $methodDir -Force)
            $activeTransaction = Test-Path -LiteralPath `
                (Join-Path $methodDir "active_task_transaction.json")
            if ($unexpected.Count -gt 0 -and -not $activeTransaction) {
                throw "$($method.Key) must start from an empty host experiment state."
            }
        }

        Write-FormalStatus -State "restoring_golden_device" -Method $method.Key `
            -Round $roundLimit -Message "Restoring clean AVD without touching host experiment state."
        Restore-RoundDevice -MethodDirectory $methodDir
        Write-FormalStatus -State "running" -Method $method.Key `
            -Round $roundLimit -Message "Formal device-separated runner is active."

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
            --rounds $roundCount `
            --round-limit $roundLimit 1>> $stdout 2>> $stderr
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
        if ($exitCode -ne 0) {
            Write-FormalStatus -State "stopped_needs_attention" -Method $method.Key `
                -Round $roundLimit -Message "Round process exited unexpectedly; frozen protocol was not changed."
            throw "Method $($method.Key) round $roundLimit exited with $exitCode."
        }

        $completedCount = @(Get-Content -LiteralPath $resultsPath | Where-Object { $_.Trim() }).Count
        if ($completedCount -ne $targetCount) {
            throw "Expected $targetCount completed records after round $roundLimit, got $completedCount."
        }
    }
}

Write-FormalStatus -State "complete" -Method "all" -Round $roundCount `
    -Message "Baseline A, Baseline B and DMS each contain $($taskCount * $roundCount) device-separated records."

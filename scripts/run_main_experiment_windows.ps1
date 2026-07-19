[CmdletBinding()]
param(
    [string]$RunRoot = "runs/main_20apps_5rounds_20260718",
    [string]$Dataset = "datasets/mini_benchmark_20apps.yaml",
    [int]$Rounds = 5,
    [string]$CondaEnv = "android_world",
    [string]$BaseUrl = "http://127.0.0.1:8000/v1",
    [int]$MaxProcessRetries = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedRunRoot = if ([System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot
} else {
    Join-Path $repoRoot $RunRoot
}
$resolvedDataset = if ([System.IO.Path]::IsPathRooted($Dataset)) {
    $Dataset
} else {
    Join-Path $repoRoot $Dataset
}
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$statusPath = Join-Path $resolvedRunRoot "orchestrator_status.json"

New-Item -ItemType Directory -Force -Path $resolvedRunRoot | Out-Null

function Write-OrchestratorStatus {
    param([string]$State, [string]$Method, [int]$Attempt, [string]$Message)
    @{
        timestamp = (Get-Date).ToString("o")
        state = $State
        method = $Method
        attempt = $Attempt
        message = $Message
        run_root = $resolvedRunRoot
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Assert-ServicesReady {
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

$methods = @(
    @{ Key = "baseline_a"; RunnerName = "baseline_a_zero_shot" },
    @{ Key = "baseline_b"; RunnerName = "baseline_b_static_memory" },
    @{ Key = "dms"; RunnerName = "dms_hierarchical_memory" }
)

foreach ($method in $methods) {
    $methodDir = Join-Path $resolvedRunRoot $method.Key
    New-Item -ItemType Directory -Force -Path $methodDir | Out-Null
    $logPath = Join-Path $resolvedRunRoot ("{0}.log" -f $method.Key)
    $completed = $false
    for ($attempt = 1; $attempt -le $MaxProcessRetries; $attempt++) {
        Write-OrchestratorStatus -State "checking_services" -Method $method.Key -Attempt $attempt -Message "Checking emulator and model API."
        Assert-ServicesReady
        Write-OrchestratorStatus -State "running" -Method $method.Key -Attempt $attempt -Message "Experiment process is active."
        & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_experiment_windows.ps1") `
            -Method $method.RunnerName `
            -Dataset $resolvedDataset `
            -Rounds $Rounds `
            -RunDir $methodDir `
            -CondaEnv $CondaEnv `
            -BaseUrl $BaseUrl 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -eq 0) {
            $completed = $true
            break
        }
        Write-OrchestratorStatus -State "retrying" -Method $method.Key -Attempt $attempt -Message "Runner exited nonzero; the next attempt resumes from task_results.jsonl."
        Start-Sleep -Seconds 10
    }
    if (-not $completed) {
        Write-OrchestratorStatus -State "failed" -Method $method.Key -Attempt $MaxProcessRetries -Message "Retry limit reached."
        throw "Method $($method.Key) did not complete after $MaxProcessRetries process attempts."
    }
}

Write-OrchestratorStatus -State "analyzing" -Method "all" -Attempt 1 -Message "Generating strict metrics and four requested figures."
$env:PYTHONPATH = "$repoRoot\src;$repoRoot\third_party\android_world"
& conda run -n $CondaEnv --no-capture-output python (Join-Path $PSScriptRoot "analyze_main_experiment.py") `
    --run-root $resolvedRunRoot `
    --dataset $resolvedDataset `
    --rounds $Rounds `
    --tasks-per-round 20
if ($LASTEXITCODE -ne 0) {
    Write-OrchestratorStatus -State "analysis_failed" -Method "all" -Attempt 1 -Message "Experiment completed but analysis failed."
    throw "Main experiment analysis failed with exit code $LASTEXITCODE"
}

Write-OrchestratorStatus -State "complete" -Method "all" -Attempt 1 -Message "All 300 attempts and requested figures are complete."

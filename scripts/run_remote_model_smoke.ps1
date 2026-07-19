[CmdletBinding()]
param(
    [string]$CondaEnv = "android_world",
    [string]$BaseUrl = "http://127.0.0.1:8000/v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$repoRoot\src;$repoRoot\third_party\android_world"
$env:DMS_MODEL_BASE_URL = $BaseUrl

Write-Host "Checking the forwarded vLLM endpoint"
$models = Invoke-RestMethod -Uri "$BaseUrl/models" -Method Get -TimeoutSec 30
$modelIds = @($models.data | ForEach-Object { $_.id })
Write-Host "remote_models=$($modelIds -join ',')"

& conda run -n $CondaEnv --no-capture-output python `
    "$repoRoot\scripts\qwen_vl_smoke.py" `
    --runtime-config "$repoRoot\configs\runtime_windows.yaml" `
    --model-config "$repoRoot\configs\model_qwen25vl_7b_remote.yaml"
if ($LASTEXITCODE -ne 0) {
    throw "Remote Qwen smoke test failed with exit code $LASTEXITCODE"
}

[CmdletBinding()]
param(
    [string]$CondaEnv = "android_world"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
& conda run -n $CondaEnv --no-capture-output python `
    "$repoRoot\scripts\download_local_embedder.py"
if ($LASTEXITCODE -ne 0) {
    throw "Embedding model download failed with exit code $LASTEXITCODE"
}

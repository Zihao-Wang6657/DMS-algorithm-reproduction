[CmdletBinding()]
param(
    [string]$ServerHost = "connect.nmb1.seetacloud.com",
    [int]$SshPort = 42258,
    [string]$User = "root",
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000
)

$ErrorActionPreference = "Stop"
Write-Host "Opening an encrypted local forward to the AutoDL model service."
Write-Host "Keep this window open while an experiment is running."
Write-Host "Enter the AutoDL password when SSH prompts for it."

$sshArgs = @(
    "-p", "$SshPort",
    "-N",
    "-o", "PubkeyAuthentication=no",
    "-o", "PreferredAuthentications=password",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}",
    "${User}@${ServerHost}"
)
& ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH model tunnel exited with code $LASTEXITCODE"
}

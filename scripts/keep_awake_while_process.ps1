[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$OrchestratorPid
)

$ErrorActionPreference = "Stop"
Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class DmsPowerState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$esContinuous = [uint32]2147483648L
$esSystemRequired = [uint32]0x00000001
try {
    while (Get-Process -Id $OrchestratorPid -ErrorAction SilentlyContinue) {
        [void][DmsPowerState]::SetThreadExecutionState($esContinuous -bor $esSystemRequired)
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][DmsPowerState]::SetThreadExecutionState($esContinuous)
}

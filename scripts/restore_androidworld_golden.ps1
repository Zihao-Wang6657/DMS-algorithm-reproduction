[CmdletBinding()]
param(
    [string]$GoldenDirectory = "device_images/AndroidWorldAvd_clean_v3",
    [string]$AppSnapshotDirectory = "device_images/AndroidWorldAppSnapshots_clean_v3",
    [string]$AvdIni = "$env:USERPROFILE/.android/avd/AndroidWorldAvd.ini",
    [int]$ConsolePort = 5554,
    [int]$GrpcPort = 8554,
    [int]$BootTimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$adb = Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe"

function Resolve-RepoPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    return Join-Path $repoRoot $Value
}

$golden = (Resolve-Path -LiteralPath (Resolve-RepoPath $GoldenDirectory)).Path
$appSnapshots = (Resolve-Path -LiteralPath (
    Resolve-RepoPath $AppSnapshotDirectory
)).Path
$goldenRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "device_images"))
$goldenFull = [System.IO.Path]::GetFullPath($golden)
if (-not $goldenFull.StartsWith($goldenRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Golden image must remain inside $goldenRoot; supplied=$goldenFull"
}
if (-not (Test-Path -LiteralPath (Join-Path $goldenFull "config.ini"))) {
    throw "Golden AVD is incomplete: $goldenFull"
}
$snapshotFull = [System.IO.Path]::GetFullPath($appSnapshots)
if (-not $snapshotFull.StartsWith($goldenRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "App snapshots must remain inside $goldenRoot; supplied=$snapshotFull"
}
$hostSnapshotNames = @(Get-ChildItem -LiteralPath $snapshotFull -Directory)
if ($hostSnapshotNames.Count -ne 24) {
    throw "Host snapshot asset must contain exactly 24 apps; found $($hostSnapshotNames.Count)."
}

$resolvedIni = (Resolve-Path -LiteralPath $AvdIni).Path
$pathLine = Get-Content -LiteralPath $resolvedIni | Where-Object { $_ -like "path=*" }
if (@($pathLine).Count -ne 1) { throw "AVD ini must contain exactly one path entry." }
$liveAvd = [System.IO.Path]::GetFullPath(($pathLine -replace "^path=", ""))
$expectedLive = [System.IO.Path]::GetFullPath(
    (Join-Path $env:USERPROFILE ".android/avd/AndroidWorld.avd")
)
if ($liveAvd -ne $expectedLive) {
    throw "Refusing unexpected live AVD target: $liveAvd"
}

$activeRunner = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "dms\.formal_runner"
})
if ($activeRunner.Count -gt 0) {
    throw "Refusing device restore while a formal_runner process is active."
}

$devices = @(& $adb devices)
if ($devices -match "^emulator-$ConsolePort\s+") {
    & $adb -s "emulator-$ConsolePort" emu kill | Out-Null
}
$deadline = (Get-Date).AddSeconds(90)
do {
    Start-Sleep -Seconds 2
    $devicePresent = @(& $adb devices) -match "^emulator-$ConsolePort\s+"
    $qemuPresent = @(Get-Process -Name "qemu-system-x86_64" -ErrorAction SilentlyContinue).Count -gt 0
} while (($devicePresent -or $qemuPresent) -and (Get-Date) -lt $deadline)
if ($devicePresent -or $qemuPresent) {
    throw "AndroidWorld emulator did not stop cleanly before golden restore."
}

# This is the only destructive filesystem operation in the restore workflow.
# The exact target was resolved from the named AndroidWorldAvd ini and checked
# against the single expected live AVD directory above.
if (Test-Path -LiteralPath $liveAvd) {
    Remove-Item -LiteralPath $liveAvd -Recurse -Force
}
New-Item -ItemType Directory -Path $liveAvd -Force | Out-Null
& robocopy $goldenFull $liveAvd /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP
$copyCode = $LASTEXITCODE
if ($copyCode -gt 7) { throw "Golden AVD copy failed with robocopy code $copyCode." }

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    (Join-Path $PSScriptRoot "start_local_androidworld.ps1") `
    -AvdName "AndroidWorldAvd" -ConsolePort $ConsolePort -GrpcPort $GrpcPort `
    -BootTimeoutSeconds $BootTimeoutSeconds
if ($LASTEXITCODE -ne 0) { throw "Restored AndroidWorld emulator failed to start." }

$requiredPackages = @(
    "ca.zgrs.clipper",
    "code.name.monkey.retromusic",
    "com.arduia.expense",
    "com.dimowner.audiorecorder",
    "com.example.androidworld",
    "com.flauschcode.broccoli",
    "com.google.androidenv.miniwob",
    "com.simplemobiletools.calendar.pro",
    "com.simplemobiletools.draw.pro",
    "com.simplemobiletools.gallery.pro",
    "com.simplemobiletools.smsmessenger",
    "de.dennisguse.opentracks",
    "net.cozic.joplin",
    "net.gsantner.markor",
    "net.osmand",
    "org.tasks",
    "org.videolan.vlc"
)
$installed = @(& $adb -s "emulator-$ConsolePort" shell pm list packages) |
    ForEach-Object { $_ -replace "^package:", "" }
$missing = @($requiredPackages | Where-Object { $installed -notcontains $_ })
if ($missing.Count -gt 0) {
    throw "Restored AVD is missing required packages: $($missing -join ', ')"
}

& $adb -s "emulator-$ConsolePort" root | Out-Null
& $adb -s "emulator-$ConsolePort" wait-for-device
& $adb -s "emulator-$ConsolePort" shell mkdir -p /data/data/android_world/snapshots
& $adb -s "emulator-$ConsolePort" push "$snapshotFull/." `
    /data/data/android_world/snapshots/ | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to restore official AndroidWorld app snapshots." }
foreach ($snapshotDirectory in $hostSnapshotNames) {
    & $adb -s "emulator-$ConsolePort" shell mkdir -p `
        "/data/data/android_world/snapshots/$($snapshotDirectory.Name)"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restore snapshot directory $($snapshotDirectory.Name)."
    }
}
$snapshotNames = @(& $adb -s "emulator-$ConsolePort" shell `
    ls -1 /data/data/android_world/snapshots 2>$null | Where-Object { $_.Trim() })
if ($snapshotNames.Count -lt 24) {
    throw "Restored AVD contains only $($snapshotNames.Count) AndroidWorld app snapshots."
}

[pscustomobject]@{
    golden_restore_ready = $true
    golden_directory = $goldenFull
    live_avd = $liveAvd
    package_count = $installed.Count
    androidworld_snapshot_count = $snapshotNames.Count
    device = "emulator-$ConsolePort"
    boot_completed = (& $adb -s "emulator-$ConsolePort" shell getprop sys.boot_completed).Trim()
} | ConvertTo-Json

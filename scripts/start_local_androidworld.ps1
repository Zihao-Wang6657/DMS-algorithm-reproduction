[CmdletBinding()]
param(
    [string]$AvdName = "AndroidWorldAvd",
    [int]$ConsolePort = 5554,
    [int]$GrpcPort = 8554,
    [int]$BootTimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
$emulator = Join-Path $env:LOCALAPPDATA "Android\Sdk\emulator\emulator.exe"
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$serial = "emulator-$ConsolePort"

foreach ($path in @($emulator, $adb)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required Android tool is missing: $path"
    }
}

Write-Host "Checking Android Emulator hardware acceleration"
& $emulator -accel-check
if ($LASTEXITCODE -ne 0) {
    throw "Android Emulator hardware acceleration is unavailable."
}

$availableAvds = @(& $emulator -list-avds)
if ($availableAvds -notcontains $AvdName) {
    throw "AVD '$AvdName' does not exist. Available AVDs: $($availableAvds -join ', ')"
}

& $adb start-server | Out-Null
$deviceLines = @(& $adb devices)
$alreadyRunning = $deviceLines -match "^$([regex]::Escape($serial))\s+device$"

if (-not $alreadyRunning) {
    Write-Host "Starting $AvdName on $serial with gRPC port $GrpcPort"
    $arguments = @(
        "-avd", $AvdName,
        "-no-snapshot",
        "-no-boot-anim",
        "-no-metrics",
        "-port", "$ConsolePort",
        "-grpc", "$GrpcPort",
        "-gpu", "auto"
    )
    $process = Start-Process -FilePath $emulator -ArgumentList $arguments -PassThru
    Write-Host "emulator_pid=$($process.Id)"
}
else {
    Write-Host "$serial is already running; reusing it."
}

& $adb -s $serial wait-for-device
$deadline = (Get-Date).AddSeconds($BootTimeoutSeconds)
$bootCompleted = ""
while ((Get-Date) -lt $deadline) {
    $bootCompleted = (& $adb -s $serial shell getprop sys.boot_completed 2>$null).Trim()
    if ($bootCompleted -eq "1") {
        break
    }
    Start-Sleep -Seconds 2
}
if ($bootCompleted -ne "1") {
    throw "$serial did not finish booting within $BootTimeoutSeconds seconds."
}

$tcp = [System.Net.Sockets.TcpClient]::new()
try {
    $connect = $tcp.ConnectAsync("127.0.0.1", $GrpcPort)
    if (-not $connect.Wait(5000) -or -not $tcp.Connected) {
        throw "Android Emulator gRPC port $GrpcPort is not reachable."
    }
}
finally {
    $tcp.Dispose()
}

Write-Host "androidworld_emulator_ready=1"
Write-Host "serial=$serial"
Write-Host "grpc_port=$GrpcPort"

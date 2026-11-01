param(
    [ValidateSet('Camera', 'Fixture')][string]$Mode = 'Camera',
    [ValidateSet('USBJTAG', 'UART')][string]$Transport = 'USBJTAG',
    [string]$Bundle = 'releases/v0.1.0',
    [string]$BuildDir = '',
    [string]$IdfPath = $env:IDF_PATH
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $IdfPath) { throw 'Run ESP-IDF 5.4.4 export.ps1 first, or supply -IdfPath.' }
$versionConfig = Get-Content (Join-Path $IdfPath 'tools/cmake/version.cmake') -Raw
if ($versionConfig -notmatch 'set\(IDF_VERSION_MAJOR 5\)' -or
    $versionConfig -notmatch 'set\(IDF_VERSION_MINOR 4\)' -or
    $versionConfig -notmatch 'set\(IDF_VERSION_PATCH 4\)') {
    throw 'This project uses ESP-IDF v5.4.4.'
}
if (-not $env:IDF_PYTHON_ENV_PATH) { . (Join-Path $IdfPath 'export.ps1') }
$idfPython = Join-Path $env:IDF_PYTHON_ENV_PATH 'Scripts/python.exe'
$appPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $appPython)) { $appPython = 'python' }
if (-not $BuildDir) { $BuildDir = "build-$($Mode.ToLower())" }
$buildPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $BuildDir))
New-Item -ItemType Directory -Force -Path $buildPath | Out-Null
$fragment = Join-Path $buildPath 'mode.defaults'
if ($Mode -eq 'Camera') {
    & $appPython (Join-Path $PSScriptRoot 'prepare_camera_firmware.py') --bundle (Join-Path $projectRoot $Bundle) `
        --destination (Join-Path $projectRoot 'firmware_example/main') --quality (Join-Path $projectRoot 'configs/quality.yaml')
    if ($LASTEXITCODE) { throw 'Model preparation failed.' }
    $transportKey = if ($Transport -eq 'USBJTAG') { 'CONFIG_ATV_TRANSPORT_USB_JTAG=y' } else { 'CONFIG_ATV_TRANSPORT_UART=y' }
    @('CONFIG_ATV_CAMERA_MODE=y', $transportKey) | Set-Content -LiteralPath $fragment -Encoding ascii
} else {
    '# CONFIG_ATV_CAMERA_MODE is not set' | Set-Content -LiteralPath $fragment -Encoding ascii
}
$defaults = (Join-Path $projectRoot 'firmware_example/sdkconfig.defaults') + ';' + $fragment
$sdkconfig = Join-Path $buildPath 'sdkconfig'
# Mode selections also update an existing build directory. Other hardware settings remain user configured.
if (Test-Path -LiteralPath $sdkconfig) {
    $currentConfig = Get-Content -LiteralPath $sdkconfig | Where-Object { $_ -notmatch '^#? ?CONFIG_ATV_(CAMERA_MODE|TRANSPORT_)' }
    @($currentConfig) + (Get-Content -LiteralPath $fragment) | Set-Content -LiteralPath $sdkconfig -Encoding ascii
}
& $idfPython (Join-Path $IdfPath 'tools/idf.py') -C (Join-Path $projectRoot 'firmware_example') -B $buildPath `
    -D "SDKCONFIG=$sdkconfig" -D "SDKCONFIG_DEFAULTS=$defaults" -D IDF_TARGET=esp32s3 build size
if ($LASTEXITCODE) { throw 'Firmware build failed.' }
Write-Host "Build: $buildPath"

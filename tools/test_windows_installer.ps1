param(
    [Parameter(Mandatory=$true)][string]$Installer,
    [Parameter(Mandatory=$true)][string]$ReleasePython,
    [string]$WorkArea = 'build/release/installer-check'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$Installer = (Resolve-Path -LiteralPath $Installer).Path
$ReleasePython = (Resolve-Path -LiteralPath $ReleasePython).Path
$workspace = [IO.Path]::GetFullPath((Join-Path $root $WorkArea))
if (-not $workspace.StartsWith($root + '\') -or (Test-Path -LiteralPath $workspace)) {
    throw 'Use a new installer-test directory inside this checkout.'
}
$registration = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{E09BF0B1-78BE-4A87-960F-542D9563271B}_is1'
if (Test-Path $registration) { throw 'An existing MultiSolid installation is registered. Do not replace it during this test.' }
New-Item -ItemType Directory -Path $workspace | Out-Null
$installDir = Join-Path $workspace 'Installed application'
$marker = Join-Path $workspace 'user-project-marker.txt'
'User files must survive reinstall and uninstall.' | Set-Content -LiteralPath $marker
$expected = (Get-FileHash -LiteralPath $marker).Hash
$completed = $false
try {
    foreach ($pass in 1, 2) {
        $log = Join-Path $workspace "install-$pass.log"
        $arguments = @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/CURRENTUSER', '/NOICONS',
                       "/DIR=`"$installDir`"", "/LOG=`"$log`"")
        $process = Start-Process -FilePath $Installer -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "Installer failed: $($process.ExitCode)" }
        if ((Get-FileHash -LiteralPath $marker).Hash -ne $expected) { throw 'Reinstall modified user data.' }
    }
    $apps = @(Get-ChildItem -LiteralPath (Join-Path $installDir 'app') -Filter MultiSolid.exe -Recurse)
    if ($apps.Count -ne 1) { throw 'Expected one installed application.' }
    & $ReleasePython tools/check_windows_release.py --bundle $apps[0].DirectoryName --output (Join-Path $workspace 'acceptance')
    if ($LASTEXITCODE -ne 0) { throw 'Installed application checks failed.' }
    $completed = $true
} finally {
    $uninstaller = Join-Path $installDir 'unins000.exe'
    if (Test-Path -LiteralPath $uninstaller) {
        $log = Join-Path $workspace 'uninstall.log'
        $process = Start-Process -FilePath $uninstaller -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/LOG=`"$log`"") -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "Uninstaller failed: $($process.ExitCode)" }
    }
}
if (-not $completed -or (Test-Path $registration) -or (Get-FileHash -LiteralPath $marker).Hash -ne $expected) {
    throw 'Installer acceptance or user-data preservation check failed.'
}
$result = @{passed=$true; installer_sha256=(Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLower();
            checks=@('per-user silent install', 'same-version reinstall', 'installed executable acceptance', 'uninstall', 'external user file preserved')}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $workspace 'validation.json') -Encoding UTF8
Write-Output (Join-Path $workspace 'validation.json')

param(
    [Parameter(Mandatory=$true)][string]$ReleasePython,
    [Parameter(Mandatory=$true)][string]$ISCC,
    [string]$Output = 'dist/desktop',
    [string]$Validation = 'build/release/acceptance',
    [string]$Artifacts = 'dist/releases'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$ReleasePython = (Resolve-Path -LiteralPath $ReleasePython).Path
$ISCC = (Resolve-Path -LiteralPath $ISCC).Path
$env:PYINSTALLER_CONFIG_DIR = Join-Path $root 'build/release/pyinstaller-cache'
$env:PYTHONIOENCODING = 'utf-8'
function Run-Python {
    & $ReleasePython @args
    if ($LASTEXITCODE -ne 0) { throw "Release step failed with exit code $LASTEXITCODE" }
}
# Dependencies and both vendor bundles must first be prepared as documented in
# desktop/WINDOWS_PACKAGING.md. Application wheels are always rebuilt here.
Run-Python tools/build_app_wheels.py
$wheels = @('multisolid_cl', 'multisolid_cl_ui') | ForEach-Object {
    $wheelMatches = @(Get-ChildItem -LiteralPath 'build/release/wheels' -Filter "$_-*.whl")
    if ($wheelMatches.Count -ne 1) { throw "Expected exactly one current $_ wheel; use a clean release wheelhouse." }
    $wheelMatches[0].FullName
}
Run-Python -m pip install --no-index --no-deps --force-reinstall @wheels
Run-Python tools/freeze_desktop.py --output $Output
$bundle = Join-Path $Output 'MultiSolid'
Run-Python tools/check_windows_release.py --bundle $bundle --output $Validation
Run-Python tools/package_windows.py --bundle $bundle --output $Artifacts --iscc $ISCC --validation (Join-Path $Validation 'validation.json')

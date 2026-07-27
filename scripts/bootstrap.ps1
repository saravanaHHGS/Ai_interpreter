# ---------------------------------------------------------------------------
# One-command environment setup for AI Interpreter.
#
#   .\scripts\bootstrap.ps1              development install (tests + linters)
#   .\scripts\bootstrap.ps1 -Locked      exact pinned versions, reproducible
#   .\scripts\bootstrap.ps1 -Recreate    delete and rebuild the virtual env
#
# Deliberately does NOT run "pip install --upgrade pip". On Windows, pip
# upgrading itself while running can fail with "WinError 32: file in use" and
# leave the virtual environment with a broken pip. The bundled pip is fine.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [switch]$Locked,
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv'
$python = Join-Path $venvPath 'Scripts\python.exe'

Write-Host "AI Interpreter - environment setup" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot"

# -- 1. Verify Python 3.12+ -------------------------------------------------
$launcher = Get-Command 'py' -ErrorAction SilentlyContinue
if (-not $launcher) {
    Write-Host 'The Python launcher "py" was not found.' -ForegroundColor Red
    Write-Host 'Install Python 3.12 or newer from https://www.python.org/downloads/' -ForegroundColor Yellow
    Write-Host 'and tick "Add python.exe to PATH" during installation.' -ForegroundColor Yellow
    exit 1
}

$versions = & py -0p
if ($versions -notmatch '3\.(1[2-9]|[2-9]\d)') {
    Write-Host 'Python 3.12 or newer is required. Installed versions:' -ForegroundColor Red
    Write-Host $versions
    exit 1
}

# -- 2. Create the virtual environment --------------------------------------
if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host 'Removing the existing virtual environment...' -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $python)) {
    Write-Host 'Creating the virtual environment in .venv ...' -ForegroundColor Cyan
    & py -3.12 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host 'Virtual environment already exists.'
}

# -- 3. Install build backend -----------------------------------------------
Write-Host 'Installing build tools...' -ForegroundColor Cyan
& $python -m pip install setuptools wheel --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# -- 4. Install the project -------------------------------------------------
if ($Locked) {
    Write-Host 'Installing exact pinned versions from requirements.lock.txt ...' -ForegroundColor Cyan
    & $python -m pip install -r (Join-Path $projectRoot 'requirements.lock.txt') --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m pip install -e $projectRoot --no-deps --disable-pip-version-check
} else {
    Write-Host 'Installing the project with development dependencies...' -ForegroundColor Cyan
    & $python -m pip install -r (Join-Path $projectRoot 'requirements-dev.txt') --disable-pip-version-check
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# -- 5. Create the secrets file ---------------------------------------------
$envFile = Join-Path $projectRoot '.env'
$envExample = Join-Path $projectRoot '.env.example'
if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host 'Created .env from .env.example (all values empty - that is correct).' -ForegroundColor Green
}

# -- 6. Verify --------------------------------------------------------------
Write-Host "`nRunning the environment check...`n" -ForegroundColor Cyan
& $python -m ai_interpreter --check
$checkResult = $LASTEXITCODE

Write-Host ''
if ($checkResult -eq 0) {
    Write-Host 'Setup complete.' -ForegroundColor Green
    Write-Host 'Next: .\run.ps1 --print-config' -ForegroundColor Green
} else {
    Write-Host 'Setup finished, but the environment check reported problems above.' -ForegroundColor Yellow
}
exit $checkResult

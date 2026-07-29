# ---------------------------------------------------------------------------
# Build the distributable archive.
#
#   .\scripts\package.ps1            -> dist\ai-interpreter-<version>.zip
#
# The distribution is deliberately a SOURCE archive, not a frozen
# executable: Smart App Control (enforced on the reference machine, and
# increasingly common on managed Windows 11) blocks unsigned binaries, and
# a PyInstaller bootloader without a code-signing certificate is exactly
# that. A source install runs the signed python.exe from python.org and
# installs signed wheels from PyPI - every binary that executes carries a
# trusted signature. See docs\deployment.md.
#
# Excluded on purpose: the virtual environment (machine-specific), models
# (downloaded pinned on first use), recordings and logs (private), .env
# (secrets), git history and caches.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

# -- 1. Version from the single source of truth ------------------------------
$version = & $python -c "import ai_interpreter; print(ai_interpreter.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Packaging AI Interpreter $version" -ForegroundColor Cyan

# -- 2. Quality gates must pass before anything ships ------------------------
Write-Host 'Running the fast test suite...' -ForegroundColor Cyan
& $python -m pytest (Join-Path $projectRoot 'tests') -q
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Tests failed - nothing was packaged.' -ForegroundColor Red
    exit 1
}

# -- 3. Stage exactly what ships ---------------------------------------------
$distDir = Join-Path $projectRoot 'dist'
$stageDir = Join-Path $distDir "ai-interpreter-$version"
$zipPath = Join-Path $distDir "ai-interpreter-$version.zip"

if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
New-Item -ItemType Directory -Force $stageDir | Out-Null

$include = @(
    'src', 'config', 'docs', 'scripts', 'tests',
    'run.ps1', 'pyproject.toml',
    'requirements.txt', 'requirements-dev.txt', 'requirements.lock.txt',
    'README.md', 'LICENSE', 'THIRD-PARTY-NOTICES.md', '.env.example',
    '.gitignore'
)
foreach ($item in $include) {
    $source = Join-Path $projectRoot $item
    if (-not (Test-Path $source)) {
        Write-Host "Missing expected file: $item" -ForegroundColor Red
        exit 1
    }
    Copy-Item -Recurse -Force $source (Join-Path $stageDir $item)
}

# Strip caches and build artefacts that Copy-Item dragged along.
Get-ChildItem $stageDir -Recurse -Directory |
    Where-Object { $_.Name -eq '__pycache__' -or $_.Name -like '*.egg-info' } |
    Remove-Item -Recurse -Force

# -- 4. Zip and verify --------------------------------------------------------
Compress-Archive -Path "$stageDir\*" -DestinationPath $zipPath
Remove-Item -Recurse -Force $stageDir

$size = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ''
Write-Host "Built $zipPath ($size MB)" -ForegroundColor Green
Write-Host 'Install on another machine: unzip, then .\scripts\bootstrap.ps1 -Locked' -ForegroundColor Green

# ---------------------------------------------------------------------------
# Runs every quality gate the project enforces, in the order that gives the
# fastest useful feedback:
#
#   1. ruff format --check   formatting        (instant)
#   2. ruff check            lint              (instant)
#   3. mypy                  static types      (seconds)
#   4. pytest                behaviour         (seconds)
#
# Run this before every commit.
#
#   .\scripts\quality.ps1            check only
#   .\scripts\quality.ps1 -Fix       auto-fix formatting and safe lint issues
#   .\scripts\quality.ps1 -Coverage  include a coverage report
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [switch]$Fix,
    [switch]$Coverage
)

$ErrorActionPreference = 'Continue'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host 'Virtual environment not found. Run .\scripts\bootstrap.ps1 first.' -ForegroundColor Red
    exit 1
}

Push-Location $projectRoot
$failures = @()

if ($Fix) {
    Write-Host '--- ruff format ---' -ForegroundColor Cyan
    & $python -m ruff format .
    Write-Host '--- ruff check --fix ---' -ForegroundColor Cyan
    & $python -m ruff check . --fix
}

Write-Host "`n--- ruff format --check ---" -ForegroundColor Cyan
& $python -m ruff format --check .
if ($LASTEXITCODE -ne 0) { $failures += 'formatting' }

Write-Host "`n--- ruff check ---" -ForegroundColor Cyan
& $python -m ruff check .
if ($LASTEXITCODE -ne 0) { $failures += 'lint' }

Write-Host "`n--- mypy ---" -ForegroundColor Cyan
& $python -m mypy
if ($LASTEXITCODE -ne 0) { $failures += 'types' }

Write-Host "`n--- pytest ---" -ForegroundColor Cyan
if ($Coverage) {
    & $python -m pytest --cov --cov-report=term-missing
} else {
    & $python -m pytest -q
}
if ($LASTEXITCODE -ne 0) { $failures += 'tests' }

Pop-Location

Write-Host ''
if ($failures.Count -eq 0) {
    Write-Host 'All quality gates passed.' -ForegroundColor Green
    exit 0
}

Write-Host ("Failed gates: " + ($failures -join ', ')) -ForegroundColor Red
if (-not $Fix) {
    Write-Host 'Try: .\scripts\quality.ps1 -Fix' -ForegroundColor Yellow
}
exit 1

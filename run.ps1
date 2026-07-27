# ---------------------------------------------------------------------------
# AI Interpreter launcher.
#
#   .\run.ps1                 show available commands
#   .\run.ps1 --check         run the environment doctor
#   .\run.ps1 --print-config  show the effective configuration
#
# Calls the virtual environment's interpreter directly instead of activating
# it. Activation changes the state of your shell session; calling the
# interpreter does not, so this script cannot leave your terminal in a
# different state than it found it.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host 'Virtual environment not found.' -ForegroundColor Red
    Write-Host 'Create it first by running:' -ForegroundColor Yellow
    Write-Host '    .\scripts\bootstrap.ps1' -ForegroundColor Yellow
    exit 1
}

& $python -m ai_interpreter @Arguments
exit $LASTEXITCODE

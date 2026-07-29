# ---------------------------------------------------------------------------
# Create Start Menu and Desktop shortcuts for the desktop interface.
#
#   .\scripts\install-shortcut.ps1            both shortcuts
#   .\scripts\install-shortcut.ps1 -Remove    delete them again
#
# The shortcuts launch run.ps1 --ui through the user's own PowerShell -
# no new executable is introduced, which is what keeps Smart App Control
# happy. Per-user locations only; nothing touches HKLM or Program Files.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $projectRoot 'run.ps1'
$name = 'AI Interpreter.lnk'
$targets = @(
    (Join-Path ([Environment]::GetFolderPath('Programs')) $name),
    (Join-Path ([Environment]::GetFolderPath('Desktop')) $name)
)

if ($Remove) {
    foreach ($path in $targets) {
        if (Test-Path $path) {
            Remove-Item -Force $path
            Write-Host "Removed $path"
        }
    }
    exit 0
}

$shell = New-Object -ComObject WScript.Shell
foreach ($path in $targets) {
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = 'powershell.exe'
    $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$runScript`" --ui"
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.Description = 'AI Interpreter - real-time Tamil/English speech translation'
    $shortcut.Save()
    Write-Host "Created $path" -ForegroundColor Green
}
Write-Host 'The shortcut opens the desktop interface directly.' -ForegroundColor Green

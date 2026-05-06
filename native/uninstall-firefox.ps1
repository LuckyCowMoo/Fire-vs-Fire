# Requires Windows PowerShell 5.1 or PowerShell 7+
# Removes the Firefox Native Messaging registry registration.
# Usage:
#   pwsh -File ./uninstall-firefox.ps1
#   powershell -ExecutionPolicy Bypass -File .\uninstall-firefox.ps1

param(
  [string]$HostName
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $scriptRoot 'manifest.firefox.json'

if (-not $HostName -and (Test-Path $manifestPath)) {
  try {
    $manifest = Get-Content -Raw -Path $manifestPath | ConvertFrom-Json
    if ($manifest.name) {
      $HostName = $manifest.name
    }
  } catch {
    # Ignore malformed manifest; fall back to default.
  }
}

if (-not $HostName) {
  $HostName = 'com.aidetector.classifier'
}

$targets = @(
  @{ Path = "HKCU:\Software\Mozilla\NativeMessagingHosts\$HostName"; Reg = "HKCU\Software\Mozilla\NativeMessagingHosts\$HostName" },
  @{ Path = "HKLM:\Software\Mozilla\NativeMessagingHosts\$HostName"; Reg = "HKLM\Software\Mozilla\NativeMessagingHosts\$HostName" }
)

foreach ($target in $targets) {
  if (Test-Path $target.Path) {
    try {
      reg delete $target.Reg /f | Out-Null
      Write-Host "Removed: $($target.Reg)" -ForegroundColor Green
    } catch {
      Write-Host "Failed to remove: $($target.Reg)" -ForegroundColor Yellow
      Write-Host $_.Exception.Message -ForegroundColor Yellow
    }
  } else {
    Write-Host "Not found: $($target.Reg)" -ForegroundColor DarkGray
  }
}

Write-Host "Done. Verify with: reg query HKCU\Software\Mozilla\NativeMessagingHosts\$HostName"
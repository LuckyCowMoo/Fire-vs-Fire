# Uninstall script for Firefox native host messaging
# Removes registry entries created by install-firefox.ps1

$ErrorActionPreference = "Stop"

# Get the host name from manifest
$manifestPath = Join-Path $PSScriptRoot "manifest.firefox.json"
if (-not (Test-Path $manifestPath)) {
    Write-Host "Error: manifest.firefox.json not found"
    exit 1
}

$manifest = Get-Content $manifestPath | ConvertFrom-Json
$hostName = $manifest.name
Write-Host "Uninstalling native host: $hostName"

# Registry paths
$hkcuPath = "HKCU:\Software\Mozilla\NativeMessagingHosts\$hostName"
$hklmPath = "HKLM:\Software\Mozilla\NativeMessagingHosts\$hostName"

# Remove registry entries
if (Test-Path $hkcuPath) {
    Remove-Item -Path $hkcuPath -Force -ErrorAction SilentlyContinue
    Write-Host "Removed HKCU registry entry"
}

if (Test-Path $hklmPath) {
    # Requires admin
    if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")) {
        Write-Host "Warning: HKLM entry requires admin. Run with elevated privileges to remove system-wide registry entry."
    }
    else {
        Remove-Item -Path $hklmPath -Force -ErrorAction SilentlyContinue
        Write-Host "Removed HKLM registry entry"
    }
}

Write-Host "Uninstall complete"

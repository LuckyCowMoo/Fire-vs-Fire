# Requires Windows PowerShell 5.1 or PowerShell 7+
# Registers the Firefox Native Messaging host and fixes manifest paths
# Usage (include the whole line stupid):
#   pwsh -File ./install-firefox.ps1
#   powershell -ExecutionPolicy Bypass -File .\install-firefox.ps1

param(
  [string]$HostName,
  [string]$ExtensionId,
  [string]$PythonPath
)

$ErrorActionPreference = 'Stop'

function Resolve-PythonPath {
  param([string]$ExplicitPath)
  if ($ExplicitPath -and (Test-Path $ExplicitPath)) { return (Resolve-Path $ExplicitPath).Path }
  # Try Python launcher (prefer 3.11)
  try {
    $pyExe = (Get-Command py -ErrorAction Stop).Source
    $resolved = & $pyExe -3.11 -c "import sys; print(sys.executable)" | Select-Object -First 1
    if ($resolved) { return $resolved }
    $resolved = & $pyExe -3 -c "import sys; print(sys.executable)" | Select-Object -First 1
    if ($resolved) { return $resolved }
  }
  catch {}
  # Try python.exe on PATH
  try {
    $cmd = Get-Command python -ErrorAction Stop
    return $cmd.Source
  }
  catch {}
  throw 'Python not found. Provide -PythonPath or install Python 3.'
}

function Resolve-ExtensionId {
  param([string]$ManifestPath, [string]$ExplicitId)
  if ($ExplicitId) { return $ExplicitId }
  if (!(Test-Path $ManifestPath)) { return $null }
  try {
    $extManifest = Get-Content -Raw -Path $ManifestPath | ConvertFrom-Json
    $id = $extManifest.browser_specific_settings.gecko.id
    if ($id) { return $id }
  }
  catch {}
  return $null
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $scriptRoot 'manifest.firefox.json'
$echoHostPath = Join-Path $scriptRoot 'echo_host_V2.py'
$runHostPath = Join-Path $scriptRoot 'run_host.bat'
$repoRoot = Split-Path -Parent $scriptRoot
$extensionManifestPath = Join-Path $repoRoot 'manifest.json'
$requirementsPath = Join-Path $repoRoot 'requirements.txt'

if (!(Test-Path $manifestPath)) { throw "Manifest not found: $manifestPath" }
if (!(Test-Path $echoHostPath)) { throw "Host script not found: $echoHostPath" }
if (!(Test-Path $runHostPath)) { throw "Host runner not found: $runHostPath" }
if (!(Test-Path $requirementsPath)) { throw "Requirements file not found: $requirementsPath" }

$pythonExe = Resolve-PythonPath -ExplicitPath $PythonPath
$resolvedExtensionId = Resolve-ExtensionId -ManifestPath $extensionManifestPath -ExplicitId $ExtensionId
if (-not $resolvedExtensionId) {
  throw "Extension ID not found. Pass -ExtensionId or set browser_specific_settings.gecko.id in $extensionManifestPath"
}
Write-Host "Using Python: $pythonExe"
Write-Host "Host script: $echoHostPath"
Write-Host "Manifest: $manifestPath"

# Install Python dependencies
Write-Host "`n=== Installing Python dependencies ==="
Write-Host "Installing from: $requirementsPath"
& $pythonExe -m pip install --default-timeout=1000 -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
  Write-Host "Warning: pip install exited with code $LASTEXITCODE. Some dependencies may not have installed." -ForegroundColor Yellow
} else {
  Write-Host "Dependencies installed successfully." -ForegroundColor Green
}

# Update manifest JSON
$manifest = Get-Content -Raw -Path $manifestPath | ConvertFrom-Json
if (-not $HostName) {
  $HostName = if ($manifest.name) { $manifest.name } else { 'com.aidetector.classifier' }
}
$manifest.name = $HostName
$manifest.allowed_extensions = @($resolvedExtensionId)
$manifest.path = $runHostPath
if ($manifest.PSObject.Properties.Name -contains 'arguments') {
  $manifest.PSObject.Properties.Remove('arguments')
}
$manifest.type = 'stdio'
$manifest.description = if ($manifest.description) { $manifest.description } else { 'Local AI classifier bridge (Firefox)' }
$json = $manifest | ConvertTo-Json -Depth 5
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($manifestPath, $json, $utf8NoBom)

# Register in HKCU for Firefox
$regKey = "HKCU\Software\Mozilla\NativeMessagingHosts\$HostName"
$escapedManifest = $manifestPath -replace '\\', '\\\\'
Write-Host "Registering: $regKey -> $manifestPath"
# Use reg.exe to set default value
reg add $regKey /ve /d "$manifestPath" /f | Out-Null

Write-Host "Done. Verify with: reg query $regKey"
Write-Host "Restart Firefox, load the extension, and test native messaging."
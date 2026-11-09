# Requires PowerShell 5.1 or PowerShell 7+
# Packages Fire vs Fire into release-ready zip archives for GitHub Releases

param(
    [string]$Version = "0.2.0"
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$distDir = Join-Path $repoRoot "dist"

if (Test-Path $distDir) {
    Remove-Item -Recurse -Force $distDir
}
New-Item -ItemType Directory -Force -Path $distDir | Out-Null

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Packaging Fire vs Fire v$Version for Release" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Package Browser Extension
# ---------------------------------------------------------------------------
Write-Host "`n[1/2] Creating Browser Extension Package..." -ForegroundColor Yellow
$extTemp = Join-Path $distDir "temp_extension"
New-Item -ItemType Directory -Force -Path $extTemp | Out-Null

$extFiles = @(
    "manifest.json",
    "package.json",
    "LICENSE",
    "Fire Vs Fire green.svg",
    "Fire Vs Fire red.svg"
)
foreach ($f in $extFiles) {
    $srcPath = Join-Path $repoRoot $f
    if (Test-Path $srcPath) { Copy-Item -Path $srcPath -Destination $extTemp }
}

$extDirs = @("background", "content", "options", "shared", "icons", "docs")
foreach ($d in $extDirs) {
    $srcPath = Join-Path $repoRoot $d
    if (Test-Path $srcPath) { Copy-Item -Path $srcPath -Destination $extTemp -Recurse }
}

$extZip = Join-Path $distDir "FireVsFire-BrowserExtension.zip"
$extZipVersioned = Join-Path $distDir "FireVsFire-BrowserExtension-v$Version.zip"
Compress-Archive -Path "$extTemp\*" -DestinationPath $extZip -Force
Copy-Item -Path $extZip -Destination $extZipVersioned -Force
Remove-Item -Recurse -Force $extTemp
Write-Host "  -> Created: $extZip" -ForegroundColor Green
Write-Host "  -> Created: $extZipVersioned" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Package Backend Native Server
# ---------------------------------------------------------------------------
Write-Host "`n[2/2] Creating Backend Native Server Package..." -ForegroundColor Yellow
$serverTemp = Join-Path $distDir "temp_server"
New-Item -ItemType Directory -Force -Path $serverTemp | Out-Null

Copy-Item -Path (Join-Path $repoRoot "Install-FireVsFire-NativeHost.bat") -Destination $serverTemp
Copy-Item -Path (Join-Path $repoRoot "requirements.txt") -Destination $serverTemp
Copy-Item -Path (Join-Path $repoRoot "LICENSE") -Destination $serverTemp

# Copy native folder excluding heavy weight files (*.pt, *.pth) and caches
$nativeSrc = Join-Path $repoRoot "native"
$nativeDst = Join-Path $serverTemp "native"
New-Item -ItemType Directory -Force -Path $nativeDst | Out-Null

Get-ChildItem -Path $nativeSrc -Recurse | Where-Object {
    $_.FullName -notmatch '\\__pycache__' -and
    $_.Extension -notin @('.pt', '.pth', '.onnx', '.log')
} | ForEach-Object {
    $relPath = $_.FullName.Substring($nativeSrc.Length + 1)
    $targetPath = Join-Path $nativeDst $relPath
    if ($_.PSIsContainer) {
        if (!(Test-Path $targetPath)) { New-Item -ItemType Directory -Force -Path $targetPath | Out-Null }
    } else {
        Copy-Item -Path $_.FullName -Destination $targetPath -Force
    }
}

$serverZip = Join-Path $distDir "FireVsFire-NativeHost.zip"
$serverZipVersioned = Join-Path $distDir "FireVsFire-NativeHost-v$Version.zip"
Compress-Archive -Path "$serverTemp\*" -DestinationPath $serverZip -Force
Copy-Item -Path $serverZip -Destination $serverZipVersioned -Force
Remove-Item -Recurse -Force $serverTemp
Write-Host "  -> Created: $serverZip" -ForegroundColor Green
Write-Host "  -> Created: $serverZipVersioned" -ForegroundColor Green

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "  Packaging Complete! Output in ./dist/" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "When publishing to GitHub Releases:"
Write-Host "  1. Attach: $extZip"
Write-Host "  2. Attach: $serverZip"
Write-Host "  3. Attach: native/classifiers/fvf_heimdall_02_best.pt"
Write-Host "  4. Attach: native/classifiers/fvf_heimdall_03_best.pt"

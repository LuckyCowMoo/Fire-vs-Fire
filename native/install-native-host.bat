@echo off
title Fire vs Fire - Native Host Installer
color 0A
echo ============================================================================
echo   FIRE VS FIRE - LOCAL AI NATIVE HOST INSTALLER
echo   Hardware-Accelerated Local Inference Bridge
echo ============================================================================
echo.
echo Setting up Python virtual environment, GPU acceleration, and browser bridge...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-firefox.ps1" -ExtensionId "template@example.com"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Setup encountered an issue. Please ensure Python 3.10+ is installed.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo ============================================================================
echo   [SUCCESS] Fire vs Fire Native Host is installed and registered!
echo   You can now return to your browser onboarding tab.
echo ============================================================================
echo.
pause

@echo off
title Fire vs Fire - Native Host Uninstaller
color 0E
echo ============================================================================
echo   FIRE VS FIRE - NATIVE HOST UNINSTALLER
echo ============================================================================
echo.
echo Removing Native Messaging bridge registrations for Firefox, Chrome, and Edge...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0native\uninstall-firefox.ps1"
echo.
echo ============================================================================
echo   [SUCCESS] Fire vs Fire Native Host registrations have been removed.
echo ============================================================================
echo.
pause

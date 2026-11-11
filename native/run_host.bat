@echo off
setlocal
set SCRIPT_DIR=%~dp0

:: Use project .venv Python so classifier deps are available
"%SCRIPT_DIR%..\.venv\Scripts\python.exe" -u -B "%SCRIPT_DIR%echo_host_V2.py"
exit /b %errorlevel%

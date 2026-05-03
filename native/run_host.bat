@echo off
setlocal
set SCRIPT_DIR=%~dp0

:: REM Use x64 Python 3.11 (required for torch) - this was the systme python lol
:: "C:\Users\lukea\AppData\Local\Programs\Python\Python311\python.exe" -u -B "%SCRIPT_DIR%echo_host_V2.py"
:: Use project .venv_sys Python so classifier deps are available
"%SCRIPT_DIR%..\.venv_sys\Scripts\python.exe" -u -B "%SCRIPT_DIR%echo_host_V2.py"
exit /b %errorlevel%

@echo off
REM Humanizer - Windows launcher (no install needed)
REM Requires Python 3.9+ from https://www.python.org/downloads/
cd /d "%~dp0"
python app_windows.py
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found. Install Python 3.9+ and tick "Add python.exe to PATH".
    pause
)

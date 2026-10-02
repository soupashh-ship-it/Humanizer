@echo off
REM Build Humanizer-Neural.exe - the SAME app with the neural models baked in
REM (torch + transformers + sentence-transformers). ~450 MB.
REM
REM The model WEIGHTS (T5-small, MiniLM) still download once on first neural
REM run and are then cached. Use this build only if you want the Neural engine
REM toggle to work without running pip install -r requirements_optional.txt.
cd /d "%~dp0"
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name Humanizer-Neural --clean ^
  app_windows.py
echo.
echo Done. Your app is at: dist\Humanizer-Neural.exe
pause

@echo off
REM Build Humanizer.exe - the lean, fully offline app (~20 MB).
REM Run once:  pip install pyinstaller
REM
REM torch / transformers / sentence-transformers are EXCLUDED on purpose:
REM they are imported lazily, only when the Neural engine toggle is used,
REM and keeping them out makes the exe small and instant to start.
REM Use build_exe_neural.bat if you want the neural engine baked in.
REM
REM Attach document reads PDFs and Word files. Install pypdf / python-docx
REM BEFORE building if you want that in the exe - they are bundled only when
REM they are importable. Export PDF always works: the writer is stdlib only.
cd /d "%~dp0"
pip install pyinstaller pypdf python-docx
pyinstaller --noconfirm --onefile --windowed --name Humanizer --clean ^
  --exclude-module torch ^
  --exclude-module transformers ^
  --exclude-module sentence_transformers ^
  --exclude-module torchvision ^
  app_windows.py
echo.
echo Done. Your app is at: dist\Humanizer.exe
echo The Neural engine toggle needs:  pip install -r requirements_optional.txt
echo Everything else works offline with no install at all.
pause

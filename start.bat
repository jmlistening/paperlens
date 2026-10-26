@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PAPERLENS_PROJECT=%~dp0"
if not defined PAPERLENS_PYTHON set "PAPERLENS_PYTHON=python"
set "PAPERLENS_PORT=8501"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist "%PAPERLENS_PYTHON%" (
  where "%PAPERLENS_PYTHON%" >nul 2>nul
  if errorlevel 1 (
    echo [PaperLens] Python interpreter not found: %PAPERLENS_PYTHON%
    echo Set PAPERLENS_PYTHON to your project Python, or edit start.bat.
    pause
    exit /b 1
  )
)
if not exist "%PAPERLENS_PROJECT%app.py" (
  echo [PaperLens] app.py not found under %PAPERLENS_PROJECT%
  pause
  exit /b 1
)

echo [PaperLens] Starting Streamlit at http://127.0.0.1:%PAPERLENS_PORT%/
echo [PaperLens] Close this window to stop the server.
start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%PAPERLENS_PORT%/'"

"%PAPERLENS_PYTHON%" -m streamlit run "%PAPERLENS_PROJECT%app.py" --server.headless true --server.address 127.0.0.1 --server.port %PAPERLENS_PORT%

endlocal

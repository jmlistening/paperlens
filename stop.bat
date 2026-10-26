@echo off
setlocal EnableExtensions

if not defined PAPERLENS_PORT set "PAPERLENS_PORT=8501"

echo [PaperLens] Stopping the Streamlit server on 127.0.0.1:%PAPERLENS_PORT% ...

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$port = %PAPERLENS_PORT%; $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); if ($listeners.Count -eq 0) { Write-Host ('[PaperLens] No server is listening on port ' + $port + '.'); exit 0 }; $stopped = 0; foreach ($procId in $listeners) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $procId) -ErrorAction SilentlyContinue; if ($null -eq $proc) { Write-Host ('[PaperLens] PID ' + $procId + ' is already gone.'); continue }; $cmdLine = [string]$proc.CommandLine; if (($cmdLine -match 'streamlit') -and ($cmdLine -match 'app.py')) { Stop-Process -Id $procId -Force; Write-Host ('[PaperLens] Stopped PID ' + $procId + ' (' + $proc.Name + ').'); $stopped += 1 } else { Write-Host ('[PaperLens] Port ' + $port + ' is held by PID ' + $procId + ' (' + $proc.Name + '), not a PaperLens Streamlit process; skipped.') } }; if ($stopped -eq 0) { Write-Host '[PaperLens] Nothing was stopped. If this is PaperLens, close its black window or end python.exe in Task Manager.'; exit 1 }; exit 0"

if errorlevel 1 (
  echo [PaperLens] Could not stop the PaperLens server.
  if not defined PAPERLENS_NO_HOLD pause
  exit /b 1
)

echo [PaperLens] Done. The scheduled daily push task was not changed.
if not defined PAPERLENS_NO_HOLD timeout /t 2 /nobreak >nul 2>&1
exit /b 0

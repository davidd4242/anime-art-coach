@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0runtime\python.exe" (
  echo Extract the entire Windows portable ZIP before running this launcher.
  pause
  exit /b 1
)
if "%~1"=="" goto picker
"%~dp0runtime\python.exe" -B -X utf8 "%~dp0windows\launch.py" --mode codex "%~f1"
goto done
:picker
"%~dp0runtime\python.exe" -B -X utf8 "%~dp0windows\launch.py" --mode codex
:done
set "status=%errorlevel%"
pause
exit /b %status%

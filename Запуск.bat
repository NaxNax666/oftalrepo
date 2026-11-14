@echo off
setlocal
cd /d "%~dp0"
title Nomogram LASIK

if exist ".venv\installed.ok" goto run

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo  Python 3.11 or newer is not installed.
  echo  1. Download it from https://www.python.org/downloads/
  echo  2. During installation check "Add python.exe to PATH"
  echo  3. Run this file again.
  echo.
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)

echo.
echo  First start: installing components (1-3 minutes, internet required)...
echo.
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto fail
echo ok> ".venv\installed.ok"

:run
".venv\Scripts\python.exe" -m nomogram serve
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo  Error. See the messages above.
pause
exit /b 1

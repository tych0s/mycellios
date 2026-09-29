@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows-usb-node-install.ps1" -PackageDirectory "%~dp0Node"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo.
  echo Instalacion incompleta. Codigo: %RESULT%
  pause
)
exit /b %RESULT%

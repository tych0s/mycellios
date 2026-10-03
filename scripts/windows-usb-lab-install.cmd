@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows-usb-lab-install.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (echo Instalacion verificada.) else (echo La instalacion continuara automaticamente. Consulta C:\ProgramData\MycelliosUsb para ver el resultado.)
exit /b %RESULT%

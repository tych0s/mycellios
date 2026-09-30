@echo off
setlocal
title Instalar Mycellios
if not exist "%~dp0MYCELLIOS\windows-usb-lab-install.ps1" (
  echo Falta la carpeta MYCELLIOS del pendrive.
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0MYCELLIOS\windows-usb-lab-install.ps1"
exit /b %ERRORLEVEL%

#requires -Version 5.1
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot '..\scripts\windows-usb-lab-install.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Installer does not parse'}
$route=$ast.Find({param($item) $item -is [Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq 'Set-LabCoordinatorRoute'},$true)
if(-not $route){throw 'Missing coordinator route helper'}
Invoke-Expression $route.Extent.Text
function Set-Service { throw 'Controller route unexpectedly changes a service' }
function Start-Service { throw 'Controller route unexpectedly starts a service' }
$originalWindowsDirectory=$env:WINDIR
try {
  # An incorrect guard must fail without ever invoking the real netsh, even
  # when this regression is run with administrator privileges.
  $env:WINDIR=Join-Path $env:TEMP ('mycellios-no-netsh-'+[Guid]::NewGuid())
  if(Test-Path -LiteralPath $env:WINDIR){throw 'Unexpected test directory'}
  Set-LabCoordinatorRoute '100.93.34.56' '100.93.34.56'
  Write-Output 'PASS: controller installation preserves its own loopback server without invoking netsh or service changes.'
} finally { $env:WINDIR=$originalWindowsDirectory }

#requires -Version 5.1
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$source=Join-Path $PSScriptRoot '..\scripts\windows-usb-lab-install.ps1'
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($source,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Installer does not parse'}
$writer=$ast.Find({param($item) $item -is [Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq 'Write-LabNodeEnrollment'},$true)
if(-not $writer){throw 'Missing enrollment writer'}
Invoke-Expression $writer.Extent.Text
$path=Join-Path $env:TEMP ('mycellios-enrollment-encoding-'+[Guid]::NewGuid()+'.json')
try {
  Write-LabNodeEnrollment $path @{schema='mycellios-node-enrollment-bundle/1';coordinatorUrl='https://coordinator.example/';
    enrollmentId='4b5e61db-9e21-4268-b1a3-50dd0e818660';enrollmentToken=('t'*43);nonce=('n'*32);expiresAt='2030-08-10T12:10:00.000Z'}
  $bytes=[IO.File]::ReadAllBytes($path)
  if($bytes.Length -lt 3 -or ($bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191)){throw 'Enrollment has an unexpected UTF-8 BOM'}
  & node.exe -e "const v=JSON.parse(require('node:fs').readFileSync(process.argv[1],'utf8'));if(v.schema!=='mycellios-node-enrollment-bundle/1'||v.enrollmentToken.length!==43)process.exit(1)" $path
  if($LASTEXITCODE -ne 0){throw 'Native Node JSON.parse rejected the actual PowerShell output'}
  Write-Output 'PASS: production PowerShell enrollment writer produces native-readable UTF-8 without BOM.'
} finally {
  Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
}

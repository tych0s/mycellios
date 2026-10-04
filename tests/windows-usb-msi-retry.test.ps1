#requires -Version 5.1
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Assert($Condition,[string]$Message){if(-not $Condition){throw $Message}}
$source=[IO.File]::ReadAllText((Join-Path $PSScriptRoot '..\scripts\windows-usb-node-install.ps1'))
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseInput($source,[ref]$tokens,[ref]$errors)
Assert ($errors.Count -eq 0) 'Installer must parse'
foreach($name in @('Assert-RegularFile','Save-MsiProgress')){
  $functionAst=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true)
  Invoke-Expression $functionAst.Extent.Text
}
$entry=[scriptblock]::Create($source.Substring($source.IndexOf('$rebootPath =')))
$root=Join-Path $env:TEMP ('mycellios-msi-retry-'+[Guid]::NewGuid())
$previousProgramFiles=$env:ProgramFiles
$previousProgramData=$env:ProgramData
$script:registered=$false; $script:msiRuns=0; $script:bootstrapRuns=0; $script:resumed=$false
$script:product='{634461E0-8451-43A2-9A6E-54141B71F0B4}'
function Read-MsiProperty {param($Path,$Name) $script:product}
function Get-InstalledNode {if($script:registered){[pscustomobject]@{PSChildName=$script:product}}}
function Test-Package {param([switch]$ResumeMsiOnly) $script:resumed=[bool]$ResumeMsiOnly; @{Mode='new';Msi=(Join-Path $PackageDirectory 'mycellios-node-0.2.82-windows-x64.msi');Config=(Join-Path $root 'node.json');Pairing=(Join-Path $root 'fresh.mycellios-enrollment')}}
function Start-Process {param($FilePath,$ArgumentList,[switch]$Wait,[switch]$PassThru,$WindowStyle) Assert ($FilePath -eq 'msiexec.exe') 'Only MSI is mocked'; $script:registered=$true; $script:msiRuns++; [pscustomobject]@{ExitCode=0}}
function powershell.exe { $script:bootstrapRuns++; $global:LASTEXITCODE=1 }
try {
  $env:ProgramFiles=Join-Path $root 'programs'
  $PackageDirectory=Join-Path $root 'Node'; $ReturnRebootCode=$false; $PreflightOnly=$false
  $null=New-Item -ItemType Directory -Path $PackageDirectory,(Join-Path $env:ProgramFiles 'Mycellios') -Force
  [IO.File]::WriteAllText((Join-Path $PackageDirectory 'mycellios-node-0.2.82-windows-x64.msi'),'fixture bytes, not a real MSI')
  [IO.File]::WriteAllText((Join-Path $env:ProgramFiles 'Mycellios\install.ps1'),'# bootstrap fixture')
  foreach($attempt in @(1,2)){
    $rejected=$false
    try{ & $entry }catch{Assert ($_.Exception.Message -eq 'El servicio Mycellios no pudo emparejarse.') 'Expected only the simulated pairing outage'; $rejected=$true}
    Assert $rejected 'Pairing outage must not be classified Ready'
    Assert ($script:msiRuns -eq 1) 'Retry must not replay a registered successful MSI'
    Assert ($script:bootstrapRuns -eq $attempt) 'Retry must reach pairing again'
    $checkpoint=Get-Content (Join-Path $PackageDirectory 'msi-reboot.json') -Raw | ConvertFrom-Json
    Assert ($checkpoint.mode -eq 'new' -and $checkpoint.productCode -eq $script:product) 'Persist fresh MSI identity before pairing'
  }
  Assert $script:resumed 'Second attempt uses MSI-only resume'
  # Exercise the production package selection after native bootstrap began.
  $packageAst=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-Package'},$true)
  Invoke-Expression ($packageAst.Extent.Text.Replace('function Test-Package(', 'function Test-RealPackage('))
  function Read-MsiProperty {param($Path,$Name) switch($Name){'ProductName'{'Mycellios Node'} 'Manufacturer'{'Mycellios'} 'ProductVersion'{'0.2.82'} default{$script:product}}}
  function Get-Service {param($Name,$ErrorAction) $null}
  $env:ProgramData=Join-Path $root 'data'
  $configDirectory=Join-Path $env:ProgramData 'Mycellios\Configuration'
  $stateDirectory=Join-Path $env:ProgramData 'Mycellios\State'
  $null=New-Item -ItemType Directory -Path $configDirectory,$stateDirectory -Force
  [IO.File]::WriteAllText((Join-Path $configDirectory 'node.json'),'{}')
  $fixtureMsi=Join-Path $PackageDirectory 'mycellios-node-0.2.82-windows-x64.msi'
  [IO.File]::WriteAllText(($fixtureMsi+'.sha256'),((Get-FileHash $fixtureMsi).Hash+'  '+[IO.Path]::GetFileName($fixtureMsi)))
  $freshSource=Join-Path $PackageDirectory 'fresh.mycellios-enrollment'
  [IO.File]::WriteAllText($freshSource,'pairing fixture validated by native resume')
  [IO.File]::WriteAllText((Join-Path $stateDirectory 'bootstrap-progress.json'),'{}')
  $selection=Test-RealPackage -ResumeMsiOnly
  Assert ($selection.Mode -eq 'resume' -and $selection.Pairing -eq $freshSource) 'Incomplete bootstrap must use native resume and fresh pairing'
  Remove-Item -LiteralPath (Join-Path $stateDirectory 'bootstrap-progress.json')
  $selection=Test-RealPackage -ResumeMsiOnly
  Assert ($selection.Mode -eq 'resume' -and -not $selection.Pairing) 'Consumed pairing must not be replayed when finishing GPU setup'
  [IO.File]::WriteAllText((Join-Path $PackageDirectory 'mycellios-node-0.2.82-windows-x64.msi'),'tampered fixture')
  $rejected=$false
  try{ & $entry }catch{Assert ($_.Exception.Message -eq 'El MSI registrado no corresponde al reinicio pendiente.') 'Reject altered MSI before bootstrap'; $rejected=$true}
  Assert ($rejected -and $script:bootstrapRuns -eq 2 -and $script:msiRuns -eq 1) 'Changed bytes cannot resume or install'
  Write-Output 'PASS: fresh MSI success survives pairing failure, retries skip MSI, changed bytes rejected.'
}finally{
  $env:ProgramFiles=$previousProgramFiles
  $env:ProgramData=$previousProgramData
  $absolute=[IO.Path]::GetFullPath($root)
  if(-not $absolute.StartsWith([IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Unsafe test cleanup'}
  Remove-Item -LiteralPath $absolute -Recurse -Force
}

#requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot '..\scripts\windows-usb-lab-stage.ps1')
function Assert($Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Get-CimInstance { [pscustomobject]@{LastBootUpTime=$script:testBoot} }
function New-ScheduledTaskAction { param($Execute,$Argument) $script:testAction=$Argument; @{} }
function New-ScheduledTaskTrigger { param([switch]$AtStartup,[switch]$Once,$At,$RepetitionInterval) @{} }
function New-ScheduledTaskSettingsSet { param([switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,$ExecutionTimeLimit,$MultipleInstances) @{} }
function New-ScheduledTaskPrincipal { param($UserId,$LogonType,$RunLevel) Assert ($UserId -eq 'SYSTEM') 'Resume must run without Windows login'; @{} }
function Register-ScheduledTask { param($TaskName,$Action,$Trigger,$Settings,$Principal,[switch]$Force) $script:registered++; @{} }
function Unregister-ScheduledTask { param($TaskName,$Confirm,$ErrorAction) $script:removed++ }
function shutdown.exe { $script:shutdowns++; $global:LASTEXITCODE=0 }
$testRoot = Join-Path $env:TEMP ('mycellios-usb-resume-' + [Guid]::NewGuid())
$previousProgramData=$env:ProgramData; $previousProgramFiles=$env:ProgramFiles
try {
  $env:ProgramData=Join-Path $testRoot 'data'; $env:ProgramFiles=Join-Path $testRoot 'programs'
  $source=Join-Path $testRoot 'USB with spaces'; $state=Join-Path $env:ProgramData 'MycelliosUsb'
  $null=New-Item -ItemType Directory -Path (Join-Path $source 'Tickets'),$state -Force
  [IO.File]::WriteAllText((Join-Path $source 'windows-usb-lab-install.ps1'), '# fixture')
  $hash=(Get-FileHash (Join-Path $source 'windows-usb-lab-install.ps1')).Hash
  [IO.File]::WriteAllText((Join-Path $source 'SHA256SUMS.txt'), "$hash  windows-usb-lab-install.ps1")
  foreach ($index in @(1,2)) {
    [IO.File]::WriteAllText((Join-Path $source "Tickets\$index.json"), (@{authKey='test-only';claim='test-only';expiresAt=[DateTimeOffset]::UtcNow.AddDays(1).ToString('o')} | ConvertTo-Json))
  }
  $script:testBoot=[datetime]'2026-09-30T08:00:00Z'; $script:registered=0; $script:shutdowns=0; $script:removed=0
  $profile=Join-Path $testRoot 'User with spaces'
  $install=Initialize-LabInstallation $source $state $profile
  Assert (@(Get-ChildItem (Join-Path $source 'Tickets')).Count -eq 1) 'Allocate exactly one ticket'
  Assert (-not (Test-Path (Join-Path $install.Package 'Tickets'))) 'Do not copy ticket pool to target'
  Assert (Test-Path (Join-Path $install.Package 'windows-usb-lab-install.ps1')) 'Stage full installer locally'
  Assert ($script:testAction.Contains('"'+$profile+'" -Resume')) 'Quote original profile for startup'
  Request-LabInstallationReboot $install
  $waiting=Initialize-LabInstallation $install.Package $state $profile
  Assert $waiting.WaitingForReboot 'Do not rerun installation before requested reboot'
  Assert ($script:registered -eq 1 -and $script:shutdowns -eq 1) 'No duplicate restart requests'
  # The USB is unavailable now. Resume exclusively from local disk after a new boot.
  $script:testBoot=$script:testBoot.AddMinutes(3)
  $resumed=Initialize-LabInstallation $install.Package $state $profile
  Assert (-not $resumed.WaitingForReboot -and $resumed.Journal.attempts -eq 2) 'Resume on next boot'
  Assert (@(Get-ChildItem (Join-Path $source 'Tickets')).Count -eq 1) 'No additional tickets consumed by resume'
  $resumed.Journal.reboots=2
  $rejected=$false
  try { Request-LabInstallationReboot $resumed } catch { $rejected=$true }
  Assert ($rejected -and $script:shutdowns -eq 1) 'Bound automatic reboot count'
  $resumed.Journal.attempts=72
  [IO.File]::WriteAllText($resumed.Path, ($resumed.Journal | ConvertTo-Json))
  $rejected=$false
  try { Initialize-LabInstallation $install.Package $state $profile | Out-Null } catch { $rejected=$true }
  Assert ($rejected -and $script:removed -eq 1) 'Stop the startup task after bounded installation retries'
  $node=Join-Path $source 'Node'; $null=New-Item -ItemType Directory -Path $node -Force
  [IO.File]::WriteAllText((Join-Path $node 'old.msi'), 'old fixture')
  $oldHash=(Get-FileHash (Join-Path $node 'old.msi')).Hash
  [IO.File]::WriteAllText((Join-Path $source 'SHA256SUMS.txt'), "$hash  windows-usb-lab-install.ps1`n$oldHash  Node\old.msi")
  $upgrade=Initialize-LabInstallation $source $state $profile
  [IO.File]::WriteAllText((Join-Path $upgrade.Package 'operator-note.txt'), 'preserve unrelated local file')
  [IO.File]::WriteAllText((Join-Path $node 'new.msi'), 'new fixture')
  $newHash=(Get-FileHash (Join-Path $node 'new.msi')).Hash
  [IO.File]::WriteAllText((Join-Path $source 'SHA256SUMS.txt'), "$hash  windows-usb-lab-install.ps1`n$newHash  Node\new.msi")
  $upgrade=Initialize-LabInstallation $source $state $profile
  Assert (-not (Test-Path (Join-Path $upgrade.Package 'Node\old.msi'))) 'Remove obsolete MSI listed in previous manifest'
  Assert (Test-Path (Join-Path $upgrade.Package 'Node\new.msi')) 'Stage replacement MSI'
  Assert (Test-Path (Join-Path $upgrade.Package 'operator-note.txt')) 'Preserve files not owned by previous manifest'
  Assert (@(Get-ChildItem (Join-Path $source 'Tickets')).Count -eq 1) 'Upgrade retains assigned ticket'
  Write-Output 'PASS: single-ticket staging, quoted startup, offline-USB resume, reboot and retry limits, obsolete MSI upgrade.'
} finally {
  $env:ProgramData=$previousProgramData; $env:ProgramFiles=$previousProgramFiles
  $resolved=[IO.Path]::GetFullPath($testRoot)
  if (-not $resolved.StartsWith([IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe test cleanup path' }
  Remove-Item -LiteralPath $resolved -Recurse -Force
}

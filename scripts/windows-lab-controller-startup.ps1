#requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$Workspace)
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  $arguments = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}" -Workspace "{1}"' -f $PSCommandPath, $Workspace
  $child = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
  exit $child.ExitCode
}
$root = Join-Path $Workspace 'runtime\native-node-lab'
$launcher = Join-Path $root 'start-lab-coordinator.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw 'No se encuentra el coordinador privado verificado.' }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $launcher)
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
$taskPrincipal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$null = Register-ScheduledTask -TaskName 'MycelliosLabCoordinatorStartup' -Action $action -Trigger $trigger -Settings $settings -Principal $taskPrincipal -Force
Start-ScheduledTask -TaskName 'MycelliosLabCoordinatorStartup'
Get-ScheduledTask -TaskName 'MycelliosLabCoordinatorStartup' | Select-Object TaskName, State
$receipt = @{task='MycelliosLabCoordinatorStartup';principal='SYSTEM';trigger='AtStartup';workspace=$Workspace;configuredAt=[DateTimeOffset]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $root 'controller-startup-receipt.json'), ($receipt | ConvertTo-Json), [Text.Encoding]::UTF8)

#requires -Version 5.1
param([switch]$PreflightOnly, [string]$UserProfile = $env:USERPROFILE, [switch]$Resume)
function Write-LabNodeEnrollment([string]$Path, $Bundle) {
  # Native JSON.parse consumes UTF-8 bytes directly; do not add a Windows BOM.
  [IO.File]::WriteAllText($Path, ($Bundle | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}
function Set-LabCoordinatorRoute([string]$ControllerIp, [string]$TailIp) {
  # The controller already serves this loopback port. A proxy to itself would
  # compete for its listener on the next coordinator restart.
  if ($ControllerIp -eq $TailIp) { return }
  $proxy = Join-Path $env:WINDIR 'System32\netsh.exe'
  $null = & $proxy interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=18790 2>&1
  $null = & $proxy interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=18790 connectaddress=$ControllerIp connectport=18100
  if ($LASTEXITCODE -ne 0) { throw 'No se pudo establecer la ruta privada del coordinador.' }
  Set-Service iphlpsvc -StartupType Automatic
  Start-Service iphlpsvc
}
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = $PSScriptRoot
$settings = Get-Content -LiteralPath (Join-Path $root 'settings.json') -Raw | ConvertFrom-Json
if ($settings.schema -ne 'mycellios-private-usb/1') { throw 'Configuracion USB no valida.' }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'Este paquete Mycellios requiere Windows x64.' }
foreach ($entry in (Get-Content -LiteralPath (Join-Path $root 'SHA256SUMS.txt'))) {
  if ($entry -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw 'Manifest de archivos no valido.' }
  $digest = $Matches[1]; $relative = $Matches[2]
  $path = [IO.Path]::GetFullPath((Join-Path $root $relative))
  if (-not $path.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Ruta fuera del paquete.' }
  $item = Get-Item -LiteralPath $path
  if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
      (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $digest) { throw "Archivo modificado: $relative" }
}
if ($PreflightOnly) { Write-Output 'Archivos del paquete y arquitectura verificados; no se ha instalado nada.'; exit 0 }
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  $arguments = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}" -UserProfile "{1}"' -f $PSCommandPath, $UserProfile
  $child = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru
  exit $child.ExitCode
}
$stateRoot = Join-Path $env:ProgramData 'MycelliosUsb'
$null = New-Item -ItemType Directory -Path $stateRoot -Force
& icacls.exe $stateRoot /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'No se pudo proteger el estado local.' }
$lock = New-Object Threading.Mutex($false, 'Global\MycelliosUsbInstallation')
try { $locked = $lock.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
if (-not $locked) { $lock.Dispose(); Write-Output 'La instalacion ya esta en curso.'; exit 0 }
$log = Join-Path $stateRoot ('install-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $log | Out-Null
try {
  . (Join-Path $root 'windows-usb-lab-stage.ps1')
  $installation = Initialize-LabInstallation $root $stateRoot $UserProfile
  if ($installation.WaitingForReboot) { Write-Output 'Esperando el reinicio programado; continuara automaticamente.'; exit 0 }
  if ($Resume -and $installation.Journal.completed) { exit 0 }
  $root = $installation.Package
  # Verify the copied bytes before executing any staged helper.
  foreach ($entry in (Get-Content -LiteralPath (Join-Path $root 'SHA256SUMS.txt'))) {
    if ((Get-FileHash -LiteralPath (Join-Path $root $entry.Substring(66)) -Algorithm SHA256).Hash -ine $entry.Substring(0,64)) { throw 'La copia local no coincide con el pendrive.' }
  }
  Write-Output 'Paquete guardado en el PC. No necesitas abrir otro archivo; se reintentara automaticamente si falla la conexion.'
  $tail = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
  if (-not (Test-Path -LiteralPath $tail)) {
    $tailMsi = Join-Path $root 'Installers\tailscale-windows-amd64.msi'
    $signature = Get-AuthenticodeSignature -LiteralPath $tailMsi
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Tailscale') { throw 'El MSI Tailscale no tiene firma oficial valida.' }
    $process = Start-Process msiexec.exe -ArgumentList @('/i', ('"{0}"' -f $tailMsi), '/qn', '/norestart') -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) { throw 'Fallo al instalar Tailscale.' }
    if ($process.ExitCode -eq 3010) { $errorReboot = New-Object InvalidOperationException('Tailscale requiere reinicio.'); $errorReboot.Data['MycelliosReboot']=$true; throw $errorReboot }
  }
  Set-Service Tailscale -StartupType Automatic
  Start-Service Tailscale
  $status = $null
  $tailDeadline = [DateTimeOffset]::UtcNow.AddSeconds(30)
  while (-not $status -and [DateTimeOffset]::UtcNow -lt $tailDeadline) {
    try { $status = (& $tail status --json 2>$null) | ConvertFrom-Json } catch { $status = $null }
    if (-not $status) { Start-Sleep -Seconds 2 }
  }
  if (-not $status) { throw 'Tailscale no responde todavia; el instalador lo reintentara automaticamente.' }
  $ticketPath = Join-Path $stateRoot 'installation-ticket.json'
  $ticket = if (Test-Path -LiteralPath $ticketPath) { Get-Content -LiteralPath $ticketPath -Raw | ConvertFrom-Json } else { $null }
  $configuredNode = Test-Path -LiteralPath (Join-Path $env:ProgramData 'Mycellios\Configuration\node.json')
  $pendingNodePairing = Test-Path -LiteralPath (Join-Path $env:ProgramData 'Mycellios\Configuration\enrollment.json')
  $partialBootstrap = Test-Path -LiteralPath (Join-Path $env:ProgramData 'Mycellios\State\bootstrap-progress.json')
  if (($status.BackendState -ne 'Running' -or -not $configuredNode -or $pendingNodePairing -or $partialBootstrap) -and -not $ticket) {
    throw 'Falta el alta asignada a este ordenador; no se pueden usar las de otro PC.'
  }
  if ($status.BackendState -eq 'Running') {
    if ($status.CurrentTailnet.Name -ne $settings.tailnet) { throw 'El ordenador esta conectado a otra VPN.' }
    & $tail set --unattended=true | Out-Null
  } else {
    if ($status.BackendState -eq 'NeedsMachineAuth') { throw 'La VPN necesita aprobacion central; no se sustituira la identidad.' }
    if ($ticket.authKey -notmatch '^tskey-auth-[A-Za-z0-9-]+$') { throw 'Alta VPN no valida.' }
    # Suppress CLI output, including errors that could echo credentials.
    $null = & $tail up --auth-key=$($ticket.authKey) --unattended=true 2>&1
  }
  if ($LASTEXITCODE -ne 0) { throw 'Tailscale no pudo activar el acceso desatendido.' }
  $status = (& $tail status --json) | ConvertFrom-Json
  $prefs = (& $tail debug prefs) | ConvertFrom-Json
  if ($status.BackendState -ne 'Running' -or $status.CurrentTailnet.Name -ne $settings.tailnet -or -not $prefs.ForceDaemon) { throw 'VPN desatendida no confirmada.' }
  if ($ticket) { $ticket.authKey = ''; [IO.File]::WriteAllText($ticketPath, ($ticket | ConvertTo-Json), [Text.Encoding]::UTF8) }
  $tailIp = @($status.Self.TailscaleIPs | Where-Object { $_ -match '^100\.' })[0]
  . (Join-Path $root 'windows-lab-access.ps1')
  $account = Install-LabAccess (Join-Path $root 'controller.pub') $settings.controllerIp $tailIp $stateRoot
  Write-Output "VPN y SSH preparados: $account@$tailIp"
  $payload = $root
  $nodeDirectory = Join-Path $payload 'Node'
  $null = New-Item -ItemType Directory -Path $nodeDirectory -Force
  Set-LabCoordinatorRoute $settings.controllerIp $tailIp
  $nodeExecutable = Join-Path $env:ProgramFiles 'Mycellios\bin\node.exe'
  $jarvisRuntime = Join-Path $UserProfile '.jarvis\bin\jarvisai-local-node.cjs'
  $recovery = @{ userProfile = $UserProfile; nodeExecutable = $nodeExecutable; jarvisRuntime = $jarvisRuntime }
  [IO.File]::WriteAllText((Join-Path $stateRoot 'recovery.json'), ($recovery | ConvertTo-Json), [Text.Encoding]::UTF8)
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}" -StateRoot "{1}"' -f (Join-Path $payload 'windows-lab-recovery.ps1'), $stateRoot)
  $trigger = New-ScheduledTaskTrigger -AtStartup
  $taskSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
  $taskPrincipal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
  $null = Register-ScheduledTask -TaskName 'MycelliosUsbRecovery' -Action $action -Trigger $trigger -Settings $taskSettings -Principal $taskPrincipal -Force
  if (-not $configuredNode -or $pendingNodePairing -or $partialBootstrap) {
    $machineId = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Cryptography').MachineGuid
    try {
      $bundle = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:18790/lab/usb/enrollment" -TimeoutSec 30 `
        -ContentType 'application/json' -Body (@{machineId=$machineId}|ConvertTo-Json -Compress) `
        -Headers @{'x-mycellios-install-claim'=$ticket.claim}
      Write-LabNodeEnrollment (Join-Path $nodeDirectory 'usb.mycellios-enrollment') $bundle
    } catch {
      $failure = $_
      $retained = $null
      try { $retained = $failure.ErrorDetails.Message | ConvertFrom-Json } catch { }
      if (-not $pendingNodePairing -or -not $retained -or $retained.error -ne 'node_already_enrolled_retain_identity') { throw $failure }
      # Let the native signer prove the consumed enrollment with its retained key.
      $staleSource = Join-Path $nodeDirectory 'usb.mycellios-enrollment'
      if (Test-Path -LiteralPath $staleSource) { Remove-Item -LiteralPath $staleSource -Force }
    }
  }
  Stop-ScheduledTask -TaskName 'MycelliosUsbRecovery' -ErrorAction SilentlyContinue
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $payload 'windows-usb-node-install.ps1') -PackageDirectory $nodeDirectory -AllowLoopbackHttpForLab -ReturnRebootCode
  if ($LASTEXITCODE -eq 3010) { $errorReboot = New-Object InvalidOperationException('Mycellios requiere reinicio.'); $errorReboot.Data['MycelliosReboot']=$true; throw $errorReboot }
  if ($LASTEXITCODE -ne 0) { throw 'Mycellios no confirmo su instalacion. SSH sigue disponible para repararlo remotamente.' }
  $jarvisConfig = Join-Path $UserProfile '.jarvis\local-node.json'
  if (-not (Test-Path -LiteralPath $jarvisConfig)) {
    $env:USERPROFILE = $UserProfile
    $env:APPDATA = Join-Path $UserProfile 'AppData\Roaming'
    $env:LOCALAPPDATA = Join-Path $UserProfile 'AppData\Local'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'JARVIS\Install-JarvisRuntime.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Jarvis necesita aprobacion o reparacion. La VPN y SSH estan disponibles.' }
  }
  $health = Get-Content (Join-Path $env:ProgramData 'Mycellios\Configuration\state\health.json') -Raw | ConvertFrom-Json
  $receipt = @{schema='mycellios-private-usb-receipt/1';computer=$env:COMPUTERNAME;sshAccount=$account;
    tailIp=$tailIp;userProfile=$UserProfile;mycelliosState=$health.state;installedAt=[DateTimeOffset]::UtcNow.ToString('o');log=$log;
    sshHostPublicKey=([IO.File]::ReadAllText((Join-Path $env:ProgramData 'ssh\ssh_host_ed25519_key.pub'))).Trim()}
  [IO.File]::WriteAllText((Join-Path $stateRoot 'receipt.json'), ($receipt|ConvertTo-Json), [Text.Encoding]::UTF8)
  $receipts = Join-Path $root 'Receipts'; $null = New-Item -ItemType Directory -Path $receipts -Force
  Copy-Item -LiteralPath (Join-Path $stateRoot 'receipt.json') -Destination (Join-Path $receipts ($env:COMPUTERNAME + '.json')) -Force
  $installation.Journal.completed = $true
  [IO.File]::WriteAllText($installation.Path, ($installation.Journal | ConvertTo-Json), [Text.Encoding]::UTF8)
  Unregister-ScheduledTask -TaskName 'MycelliosUsbInstallResume' -Confirm:$false -ErrorAction SilentlyContinue
  Write-Output 'LISTO: VPN desatendida, SSH administrativo, recuperacion al arrancar y Mycellios Ready.'
  Write-Output 'Jarvis controla la pantalla cuando hay una sesion de Windows iniciada. SSH funciona antes de iniciar sesion.'
} catch {
  if ($_.Exception.Data['MycelliosReboot']) { Request-LabInstallationReboot $installation; exit 3010 }
  Write-Output ('Instalacion pendiente: ' + $_.Exception.Message)
  Write-Output "Log: $log. El reintento es automatico; no necesitas abrir otros archivos."
  exit 1
} finally {
  if (Get-ScheduledTask -TaskName 'MycelliosUsbRecovery' -ErrorAction SilentlyContinue) {
    Start-ScheduledTask -TaskName 'MycelliosUsbRecovery'
  }
  Stop-Transcript | Out-Null
  if ($locked) { $lock.ReleaseMutex() }
  $lock.Dispose()
}

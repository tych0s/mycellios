#requires -Version 5.1
param(
  [string]$PackageDirectory = (Join-Path $PSScriptRoot 'Node'),
  [switch]$PreflightOnly,
  [switch]$AllowLoopbackHttpForLab
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-RegularFile([string]$Path, [string]$Label) {
  $item = Get-Item -LiteralPath $Path -ErrorAction Stop
  if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw "$Label debe ser un archivo normal."
  }
  return $item
}

function Read-MsiProperty([string]$Path, [string]$Name) {
  $installer = New-Object -ComObject WindowsInstaller.Installer
  $database = $installer.OpenDatabase($Path, 0)
  $view = $database.OpenView("SELECT ``Value`` FROM ``Property`` WHERE ``Property`` = '$Name'")
  $null = $view.Execute()
  $record = $view.Fetch()
  if (-not $record) { throw "El MSI no contiene la propiedad $Name." }
  return [string]$record.StringData(1)
}

function Get-InstalledNode {
  $entries = @(Get-ItemProperty 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue |
    Where-Object { $_.PSObject.Properties['DisplayName'] -and $_.PSObject.Properties['Publisher'] -and
      $_.DisplayName -eq 'Mycellios Node' -and $_.Publisher -eq 'Mycellios' })
  if ($entries.Count -gt 1) { throw 'Hay mas de un MSI Mycellios Node registrado.' }
  if ($entries.Count -eq 0) { return $null }
  if ([string]$entries[0].PSChildName -notmatch '^\{[0-9A-Fa-f-]{36}\}$') {
    throw 'El MSI Mycellios Node registrado no tiene un ProductCode valido.'
  }
  return $entries[0]
}

function Test-RetainedNode([string]$ConfigPath, $Service, $Installed) {
  if (-not $Installed -or -not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw 'Hay una instalacion parcial: faltan el MSI o la configuracion. Requiere diagnostico antes de continuar.'
  }
  $dataRoot = Join-Path $env:ProgramData 'Mycellios'
  $identityPath = Join-Path $dataRoot 'Identity\node.json'
  $manifestPath = Join-Path $dataRoot 'State\installation.json'
  $workerPath = Join-Path $dataRoot 'Configuration\worker.json'
  foreach ($path in @($ConfigPath, $identityPath, $manifestPath, $workerPath)) {
    $null = Assert-RegularFile $path 'Dato protegido del nodo'
  }
  if (Test-Path -LiteralPath (Join-Path $dataRoot 'Configuration\enrollment.json')) {
    throw 'El nodo tiene un emparejamiento pendiente; no se puede actualizar automaticamente.'
  }
  $config = [IO.File]::ReadAllText($ConfigPath) | ConvertFrom-Json -ErrorAction Stop
  $identity = [IO.File]::ReadAllText($identityPath) | ConvertFrom-Json -ErrorAction Stop
  $manifest = [IO.File]::ReadAllText($manifestPath) | ConvertFrom-Json -ErrorAction Stop
  if ($config.schema -ne 'mycellios-node-configuration/1' -or
      [string]$config.nodeId -notmatch '^node-[0-9a-f]{32}$' -or
      $identity.schema -ne 'mycellios-node-identity/1' -or
      $identity.identityId -cne $config.nodeId -or
      $identity.provider -ne 'windows-dpapi-local-machine' -or
      [string]$identity.secretReference -notmatch '^[A-Za-z0-9._:-]{1,200}$' -or
      $manifest.schema -ne 'mycellios-node-installation/1' -or
      $manifest.platform -ne 'win32' -or
      $manifest.serviceName -ne 'MycelliosNode' -or
      $manifest.configPath -cne $ConfigPath -or
      $manifest.identityPath -cne $identityPath -or
      $manifest.installRoot -cne (Join-Path $env:ProgramFiles 'Mycellios')) {
    throw 'La identidad o las rutas del nodo instalado no son coherentes.'
  }
  $installRoot = Join-Path $env:ProgramFiles 'Mycellios'
  $expectedPaths = @{
    nodeExecutable = Join-Path $installRoot 'bin\node.exe'
    helperEntrypoint = Join-Path $installRoot 'app\node\uninstall-main.js'
    serviceDefinitionPath = Join-Path $dataRoot 'Service\MycelliosNode.xml'
    cachePath = Join-Path $dataRoot 'Cache'
    logsPath = Join-Path $dataRoot 'Logs'
    statePath = Join-Path $dataRoot 'State'
    receiptPath = Join-Path $dataRoot 'Receipts\uninstall.json'
  }
  foreach ($entry in $expectedPaths.GetEnumerator()) {
    if ([string]$manifest.($entry.Key) -cne [string]$entry.Value) {
      throw "La ruta retenida $($entry.Key) no corresponde a esta instalacion."
    }
  }
  if ($config.coordinator.identityPath -cne $identityPath -or
      $config.worker.configPath -cne $workerPath -or
      $config.runtime.pythonExecutable -cne (Join-Path $installRoot 'runtime\python.exe') -or
      $config.runtime.pythonPath -cne (Join-Path $installRoot 'python') -or
      $config.runtime.cachePath -cne $expectedPaths.cachePath -or
      $config.uninstall.manifestPath -cne $manifestPath) {
    throw 'La configuracion retenida apunta a rutas distintas del MSI instalado.'
  }
  $secretPath = Join-Path (Join-Path $dataRoot 'protected-identity') "$($identity.secretReference).dpapi"
  $null = Assert-RegularFile $secretPath 'Identidad cifrada'
  if ($Service) {
    $registeredService = Get-CimInstance Win32_Service -Filter "Name='MycelliosNode'"
    if ($registeredService.StartName -ne 'NT AUTHORITY\LocalService' -or
        $registeredService.PathName.Trim('"') -cne (Join-Path $dataRoot 'Service\MycelliosNode.exe')) {
      throw 'La cuenta o el ejecutable del servicio no coincide con la instalacion protegida.'
    }
  }
  return @{ NodeId = [string]$config.nodeId; Service = $Service }
}

function Test-Package {
  if (-not [Environment]::Is64BitOperatingSystem) { throw 'El nodo requiere Windows x64.' }
  if (-not (Test-Path -LiteralPath $PackageDirectory -PathType Container)) {
    throw 'Falta la carpeta Node del pendrive.'
  }
  $msis = @(Get-ChildItem -LiteralPath $PackageDirectory -File -Filter 'mycellios-node-*-windows-x64.msi')
  if ($msis.Count -ne 1) { throw 'La carpeta Node debe contener exactamente un MSI Windows x64.' }
  $msi = Assert-RegularFile $msis[0].FullName 'MSI'
  $checksum = Assert-RegularFile "$($msi.FullName).sha256" 'Checksum'
  $checksumText = [IO.File]::ReadAllText($checksum.FullName)
  $match = [regex]::Match($checksumText, '^(?<hash>[0-9a-fA-F]{64})  (?<name>[^\r\n\\/]+\.msi)\r?\n?$')
  if (-not $match.Success -or $match.Groups['name'].Value -cne $msi.Name) {
    throw 'El checksum no corresponde al MSI indicado.'
  }
  $actualHash = (Get-FileHash -LiteralPath $msi.FullName -Algorithm SHA256).Hash
  if ($actualHash -ine $match.Groups['hash'].Value) { throw 'El MSI no coincide con su SHA-256.' }

  if ((Read-MsiProperty $msi.FullName 'ProductName') -ne 'Mycellios Node' -or
      (Read-MsiProperty $msi.FullName 'Manufacturer') -ne 'Mycellios') {
    throw 'El MSI no es un instalador Mycellios Node.'
  }
  $productCode = Read-MsiProperty $msi.FullName 'ProductCode'
  $incomingVersion = [version](Read-MsiProperty $msi.FullName 'ProductVersion')
  $config = Join-Path $env:ProgramData 'Mycellios\Configuration\node.json'
  $service = Get-Service -Name MycelliosNode -ErrorAction SilentlyContinue
  $installed = Get-InstalledNode
  if ($service -or $installed -or (Test-Path -LiteralPath $config)) {
    $retained = Test-RetainedNode $config $service $installed
    if ($productCode -ine [string]$installed.PSChildName -and
        $incomingVersion -le [version]$installed.DisplayVersion) {
      throw 'Para actualizar este nodo se necesita un MSI con version superior a la instalada.'
    }
    return @{ Msi = $msi.FullName; Config = $config; Mode = 'existing'; NodeId = $retained.NodeId;
      ProductCode = $productCode; InstalledCode = [string]$installed.PSChildName; Service = $service }
  }

  $pairings = @(Get-ChildItem -LiteralPath $PackageDirectory -File -Filter '*.mycellios-enrollment')
  if ($pairings.Count -ne 1) { throw 'La carpeta Node debe contener exactamente un emparejamiento vigente.' }
  $pairing = Assert-RegularFile $pairings[0].FullName 'Emparejamiento'
  if ($pairing.Length -gt 4096) { throw 'El archivo de emparejamiento es demasiado grande.' }
  try { $bundle = [IO.File]::ReadAllText($pairing.FullName) | ConvertFrom-Json -ErrorAction Stop }
  catch { throw 'El archivo de emparejamiento no es JSON valido.' }
  $expectedFields = @('schema', 'coordinatorUrl', 'enrollmentId', 'enrollmentToken', 'nonce', 'expiresAt')
  $actualFields = @($bundle.PSObject.Properties.Name)
  if ($actualFields.Count -ne $expectedFields.Count -or
      @($actualFields | Where-Object { $_ -notin $expectedFields }).Count -ne 0) {
    throw 'El archivo de emparejamiento tiene campos inesperados o incompletos.'
  }
  if ($bundle.schema -ne 'mycellios-node-enrollment-bundle/1' -or
      [string]$bundle.enrollmentToken -notmatch '^[A-Za-z0-9_-]{32,256}$' -or
      [string]$bundle.enrollmentId -notmatch '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' -or
      [string]$bundle.nonce -notmatch '^[A-Za-z0-9_-]{22,128}$') {
    throw 'El archivo de emparejamiento no tiene el formato esperado.'
  }
  $coordinator = $null
  $validUri = [Uri]::TryCreate([string]$bundle.coordinatorUrl, [UriKind]::Absolute, [ref]$coordinator)
  $labLoopback = $validUri -and $AllowLoopbackHttpForLab -and
    $coordinator.Scheme -eq 'http' -and
    $coordinator.Host -in @('localhost', '127.0.0.1', '::1', '[::1]')
  if (-not $validUri -or
      ($coordinator.Scheme -ne 'https' -and -not $labLoopback) -or $coordinator.UserInfo -ne '' -or
      $coordinator.AbsolutePath -ne '/' -or $coordinator.Query -ne '' -or $coordinator.Fragment -ne '') {
    throw 'El coordinador del emparejamiento debe usar HTTPS.'
  }
  $expires = [DateTimeOffset]::MinValue
  if (-not [DateTimeOffset]::TryParse([string]$bundle.expiresAt, [ref]$expires) -or
      $expires -le [DateTimeOffset]::UtcNow.AddMinutes(10)) {
    throw 'El emparejamiento caduca pronto: crea uno nuevo antes de instalar.'
  }
  return @{ Msi = $msi.FullName; Pairing = $pairing.FullName; Config = $config; Mode = 'new' }
}

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $PSCommandPath),
    '-PackageDirectory', ('"{0}"' -f $PackageDirectory))
  if ($PreflightOnly) { $arguments += '-PreflightOnly' }
  if ($AllowLoopbackHttpForLab) { $arguments += '-AllowLoopbackHttpForLab' }
  $child = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru
  exit $child.ExitCode
}

$package = Test-Package
if ($PreflightOnly) {
  if ($package.Mode -eq 'existing') {
    Write-Output "Preflight listo: nodo $($package.NodeId), MSI y SHA-256 verificados; no se ha modificado nada."
  } else {
    Write-Output 'Preflight listo: MSI, SHA-256 y emparejamiento vigentes; no se ha instalado nada.'
  }
  exit 0
}
$sameProduct = $package.Mode -eq 'existing' -and $package.ProductCode -ieq $package.InstalledCode
if ($sameProduct -and $package.Service -and $package.Service.Status -eq 'Running') {
  $currentHealthPath = Join-Path (Split-Path -Parent $package.Config) 'state\health.json'
  if (Test-Path -LiteralPath $currentHealthPath -PathType Leaf) {
    try { $currentHealth = [IO.File]::ReadAllText($currentHealthPath) | ConvertFrom-Json -ErrorAction Stop }
    catch { $currentHealth = $null }
    if ($currentHealth -and $currentHealth.state -eq 'ready' -and
        (Get-Process -Id $currentHealth.pid -ErrorAction SilentlyContinue)) {
      Write-Output "Mycellios ya esta instalado y Ready: $($package.NodeId)."
      exit 0
    }
  }
  throw 'El servicio ya existe pero no esta Ready. Revisa los diagnosticos antes de actualizar.'
}
if (-not $sameProduct) {
  $msi = Start-Process msiexec.exe -ArgumentList @('/i', ('"{0}"' -f $package.Msi), '/qn', '/norestart') -Wait -PassThru -WindowStyle Hidden
  if ($msi.ExitCode -eq 3010) {
    throw 'El MSI requiere reinicio. Reinicia y vuelve a ejecutar el mismo archivo del pendrive para recuperar el nodo.'
  }
  if ($msi.ExitCode -ne 0) { throw "La instalacion MSI fallo con codigo $($msi.ExitCode)." }
  if ($package.Mode -eq 'existing') {
    $updated = Get-InstalledNode
    if (-not $updated -or [string]$updated.PSChildName -ine $package.ProductCode) {
      throw 'Windows Installer no registro la nueva version del nodo.'
    }
  }
}
if ($package.Mode -eq 'existing') {
  if ($package.Service -and $sameProduct) {
    Start-Service -Name MycelliosNode -ErrorAction Stop
  } else {
    $restorer = Join-Path $env:ProgramFiles 'Mycellios\app\node\restore-main.js'
    $node = Join-Path $env:ProgramFiles 'Mycellios\bin\node.exe'
    $null = Assert-RegularFile $restorer 'Restaurador del servicio'
    $null = Assert-RegularFile $node 'Runtime Node'
    & $node $restorer
    if ($LASTEXITCODE -ne 0) { throw 'El nodo no pudo recuperar su identidad o su servicio.' }
  }
} else {
  $installedScript = Join-Path $env:ProgramFiles 'Mycellios\install.ps1'
  if (-not (Test-Path -LiteralPath $installedScript -PathType Leaf)) { throw 'El MSI no dejo el instalador de emparejamiento.' }
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installedScript -Enrollment $package.Pairing
  if ($LASTEXITCODE -ne 0) { throw 'El servicio Mycellios no pudo emparejarse.' }
}
$service = Get-Service -Name MycelliosNode -ErrorAction Stop
$service.WaitForStatus('Running', [TimeSpan]::FromSeconds(30))
if (-not (Test-Path -LiteralPath $package.Config -PathType Leaf)) { throw 'No se encontro la configuracion protegida del nodo.' }
$configurationDirectory = Split-Path -Parent $package.Config
$healthPath = Join-Path $configurationDirectory 'state\health.json'
$diagnosticsPath = Join-Path $configurationDirectory 'state\diagnostics.json'
$enrollmentPath = Join-Path $configurationDirectory 'enrollment.json'
$deadline = [DateTimeOffset]::UtcNow.AddMinutes(2)
while ([DateTimeOffset]::UtcNow -lt $deadline) {
  $service.Refresh()
  if ($service.Status -ne 'Running') { throw 'El servicio Mycellios dejo de ejecutarse antes de estar listo.' }
  if (Test-Path -LiteralPath $healthPath -PathType Leaf) {
    try { $health = [IO.File]::ReadAllText($healthPath) | ConvertFrom-Json -ErrorAction Stop }
    catch { $health = $null }
    if ($health -and $health.state -eq 'failed') {
      throw "El nodo fallo al iniciar: $($health.error)"
    }
    if ($health -and $health.state -eq 'ready' -and
        -not (Test-Path -LiteralPath $enrollmentPath -PathType Leaf)) {
      if (Test-Path -LiteralPath $diagnosticsPath -PathType Leaf) {
        try { $diagnostics = [IO.File]::ReadAllText($diagnosticsPath) | ConvertFrom-Json -ErrorAction Stop }
        catch { $diagnostics = $null }
        if ($diagnostics -and $diagnostics.runtime.backend -in @('cpu', 'cuda', 'rocm') -and
            (Get-Process -Id $health.pid -ErrorAction SilentlyContinue)) {
          if ($package.Mode -eq 'existing') {
            Write-Output "Mycellios actualizado y Ready: $($package.NodeId). Backend verificado: $($diagnostics.runtime.backend)."
          } else {
            Write-Output "Mycellios instalado, emparejado y Ready. Backend verificado: $($diagnostics.runtime.backend)."
          }
          exit 0
        }
      }
    }
  }
  Start-Sleep -Seconds 2
}
throw 'El servicio arranco, pero no confirmo Ready y el emparejamiento en dos minutos. Revisa los diagnosticos locales.'

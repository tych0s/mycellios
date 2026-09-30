#requires -Version 5.1
param(
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [Parameter(Mandatory=$true)][string]$MsiPath,
  [Parameter(Mandatory=$true)][string]$ExpectedMsiHash,
  [Parameter(Mandatory=$true)][string]$JarvisPackageDirectory,
  [Parameter(Mandatory=$true)][string]$ClaimsPath,
  [Parameter(Mandatory=$true)][string]$PublicKeyPath,
  [string]$CredentialPath = (Join-Path $env:LOCALAPPDATA 'MycelliosLab\tailscale-oauth.dpapi.json'),
  [string]$ControllerIp = '100.93.34.56',
  [string]$Tailnet = 'nodecodex.io',
  [ValidateRange(1,50)][int]$EnrollmentCount = 10,
  [switch]$WithoutNewKeys,
  [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ((Get-FileHash -LiteralPath $MsiPath -Algorithm SHA256).Hash -ine $ExpectedMsiHash) { throw 'MSI Mycellios diferente del candidato verificado.' }
$root = [IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $root) -and -not $Resume) { throw 'Usa un directorio nuevo o Resume para conservar las altas anteriores.' }
$null = New-Item -ItemType Directory -Path $root -Force
foreach ($name in @('Node','Installers','Tickets','Receipts','JARVIS')) { $null = New-Item -ItemType Directory -Path (Join-Path $root $name) -Force }
foreach ($name in @('windows-usb-lab-install.ps1','windows-usb-lab-install.cmd','windows-usb-lab-stage.ps1','windows-lab-access.ps1','windows-lab-recovery.ps1','windows-usb-node-install.ps1')) {
  Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $root -Force
}
Copy-Item -LiteralPath $PublicKeyPath -Destination (Join-Path $root 'controller.pub') -Force
$msiFileName = [IO.Path]::GetFileName($MsiPath)
if ($msiFileName -notmatch '^mycellios-node-([0-9]+\.[0-9]+\.[0-9]+)-windows-x64(?:-[a-z0-9-]+)?\.msi$') { throw 'Nombre de MSI Mycellios Windows x64 no valido.' }
$msiName = 'mycellios-node-' + $Matches[1] + '-windows-x64.msi'
$otherMsis = @(Get-ChildItem -LiteralPath (Join-Path $root 'Node') -Filter '*.msi' -File | Where-Object Name -ne $msiName)
if ($otherMsis.Count -gt 0) { throw 'Prepara la nueva version en otro directorio; Node debe contener un solo MSI.' }
Copy-Item -LiteralPath $MsiPath -Destination (Join-Path $root "Node\$msiName") -Force
[IO.File]::WriteAllText((Join-Path $root "Node\$msiName.sha256"), "$ExpectedMsiHash  $msiName`n", [Text.Encoding]::ASCII)
Get-ChildItem -LiteralPath $JarvisPackageDirectory | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $root 'JARVIS') -Recurse -Force }
# Repair elevation quoting in the retained Jarvis bootstrap for paths containing spaces.
$jarvisInstaller = Join-Path $root 'JARVIS\Install-JarvisRuntime.ps1'
$source = [IO.File]::ReadAllText($jarvisInstaller)
$source = $source.Replace('"-File", $PSCommandPath, "-LogName", $LogName', '"-File", (''"{0}"'' -f $PSCommandPath), "-LogName", $LogName')
[IO.File]::WriteAllText($jarvisInstaller, $source, [Text.UTF8Encoding]::new($false))
$jarvisManifest = Join-Path $root 'JARVIS\SHA256SUMS.txt'
$oldManifest = [IO.File]::ReadAllText($jarvisManifest)
$jarvisHash = (Get-FileHash -LiteralPath $jarvisInstaller -Algorithm SHA256).Hash
$oldManifest = [regex]::Replace($oldManifest, '(?m)^[A-Fa-f0-9]{64}(  \*Install-JarvisRuntime\.ps1)', $jarvisHash + '$1')
[IO.File]::WriteAllText($jarvisManifest, $oldManifest, [Text.Encoding]::ASCII)
$versions = Invoke-RestMethod 'https://pkgs.tailscale.com/stable/?mode=json'
$tailscaleMsi = Join-Path $root 'Installers\tailscale-windows-amd64.msi'
if (-not (Test-Path -LiteralPath $tailscaleMsi)) {
  Invoke-WebRequest -UseBasicParsing -Uri ('https://pkgs.tailscale.com/stable/' + $versions.MSIs.amd64) -OutFile $tailscaleMsi
}
$signature = Get-AuthenticodeSignature -LiteralPath $tailscaleMsi
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Tailscale') { throw 'Firma Tailscale no valida.' }
[IO.File]::WriteAllText((Join-Path $root 'settings.json'), (@{schema='mycellios-private-usb/1';controllerIp=$ControllerIp;tailnet=$Tailnet;tailscaleVersion=$versions.MSIsVersion}|ConvertTo-Json), [Text.Encoding]::UTF8)
$claims = @()
if (Test-Path -LiteralPath $ClaimsPath) {
  $parsedClaims = Get-Content -LiteralPath $ClaimsPath -Raw | ConvertFrom-Json
  foreach ($entry in $parsedClaims) { $claims += $entry }
}
$existingTickets = @(Get-ChildItem -LiteralPath (Join-Path $root 'Tickets') -Filter '*.json' -File)
foreach ($file in $existingTickets) {
  $existing = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
  $hasher = [Security.Cryptography.SHA256]::Create()
  try { $digest = -join ($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($existing.claim)) | ForEach-Object { $_.ToString('x2') }) }
  finally { $hasher.Dispose() }
  if (-not ($claims | Where-Object tokenHash -eq $digest)) {
    $claims += @{tokenHash=$digest;expiresAt=([DateTimeOffset]::Parse($existing.expiresAt)).ToUnixTimeMilliseconds()}
  }
}
if ($existingTickets.Count -gt 0) { [IO.File]::WriteAllText($ClaimsPath, (ConvertTo-Json -InputObject $claims -Depth 6), [Text.Encoding]::UTF8) }
if (-not $WithoutNewKeys) {
  $credential = Get-Content -LiteralPath $CredentialPath -Raw | ConvertFrom-Json
  $secure = ConvertTo-SecureString $credential.encryptedSecret
  $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try { $secret = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
  finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
  try {
    $oauth = Invoke-RestMethod -Method Post -Uri 'https://api.tailscale.com/api/v2/oauth/token' -Body @{
      grant_type='client_credentials';client_id=$credential.clientId;client_secret=$secret;scope='auth_keys'
    }
    $secret = $null
    for ($index=$existingTickets.Count + 1; $index -le $EnrollmentCount; $index++) {
      $request = @{expirySeconds=7776000;description='Mycellios private USB single-PC enrollment';
        capabilities=@{devices=@{create=@{reusable=$false;ephemeral=$false;preauthorized=$true;tags=@($credential.tag)}}}} | ConvertTo-Json -Depth 8
      $key = Invoke-RestMethod -Method Post -Uri 'https://api.tailscale.com/api/v2/tailnet/-/keys' `
        -Headers @{Authorization=('Bearer ' + $oauth.access_token)} -ContentType 'application/json' -Body $request
      $bytes = New-Object byte[] 32; $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
      try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
      $claim = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
      $hasher = [Security.Cryptography.SHA256]::Create()
      try { $digest = -join ($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($claim)) | ForEach-Object { $_.ToString('x2') }) }
      finally { $hasher.Dispose() }
      $expires = [DateTimeOffset]::UtcNow.AddDays(90)
      $ticket = @{authKey=$key.key;claim=$claim;expiresAt=$expires.ToString('o')}
      [IO.File]::WriteAllText((Join-Path $root ('Tickets\{0:d3}.json' -f $index)), ($ticket|ConvertTo-Json), [Text.Encoding]::UTF8)
      $claims += @{tokenHash=$digest;expiresAt=$expires.ToUnixTimeMilliseconds()}
      $ticket = $null; $key = $null; $claim = $null
      # Persist after every issuance; an interrupted preparation never loses issued claims.
      [IO.File]::WriteAllText($ClaimsPath, (ConvertTo-Json -InputObject $claims -Depth 6), [Text.Encoding]::UTF8)
    }
  } finally { $secret=$null; $oauth=$null; $secure.Dispose() }
}
[IO.File]::WriteAllText((Join-Path $root 'LEEME.txt'), @'
MYCELLIOS - INSTALACION PRIVADA EN WINDOWS X64

1. Conecta este pendrive a un ordenador tuyo con Internet y Windows x64.
2. Abre INSTALAR-MYCELLIOS.cmd en la raiz del pendrive y acepta el permiso de administrador.
3. Espera a LISTO. La primera vinculacion Jarvis se aprueba desde tu PC de control;
   no requiere iniciar sesion en Jarvis en el ordenador nuevo.
4. No abras otros instaladores. Si Windows necesita reiniciar, se anuncia con
   60 segundos de margen y la instalacion continua sola desde el disco del PC.
   El resultado se guarda en C:\ProgramData\MycelliosUsb\receipt.json.
   VPN, SSH y Mycellios arrancan automaticamente.

PC-DANI debe estar conectado para emitir el emparejamiento inicial de Mycellios.
Cada alta sirve para un solo PC. Las altas NO usadas caducan a los 90 dias.
La reinstalacion conserva la identidad de un equipo ya configurado.
Los recibos estan en Receipts y C:\ProgramData\MycelliosUsb\receipt.json.
La pantalla Jarvis requiere una sesion Windows; SSH administra antes del login.
No hay acceso con el equipo apagado, sin Internet o sin arrancar Windows.
Este MSI Mycellios es un candidato privado sin firma publica.
'@, [Text.Encoding]::UTF8)
$manifest = @(Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object {
  $_.FullName -ne (Join-Path $root 'SHA256SUMS.txt') -and $_.FullName -notlike (Join-Path $root 'Tickets\*') -and $_.FullName -notlike (Join-Path $root 'Receipts\*')
} | Sort-Object FullName | ForEach-Object {
  '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.FullName.Substring($root.Length + 1)
})
[IO.File]::WriteAllLines((Join-Path $root 'SHA256SUMS.txt'), [string[]]$manifest, [Text.Encoding]::ASCII)
Write-Output "Paquete preparado: $root. Altas nuevas: $(if($WithoutNewKeys){0}else{$EnrollmentCount}). No se ha copiado al pendrive."

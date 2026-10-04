#requires -Version 5.1
function Install-LabAccess([string]$PublicKeyPath, [string]$ControllerIp, [string]$TailIp, [string]$StateRoot) {
  $key = ([IO.File]::ReadAllText($PublicKeyPath)).Trim()
  if ($key -notmatch '^ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]+)?$') { throw 'Clave publica no valida.' }
  foreach ($ip in @($ControllerIp, $TailIp)) {
    $parsed = [Net.IPAddress]::None
    if (-not [Net.IPAddress]::TryParse($ip, [ref]$parsed) -or $ip -notmatch '^100\.') {
      throw 'La conexion administrativa requiere direcciones privadas Tailscale.'
    }
  }
  $account = 'mycellios-control'
  $description = 'Mycellios USB controller key-only administration'
  $user = Get-LocalUser -Name $account -ErrorAction SilentlyContinue
  if ($user -and $user.Description -ne $description) { throw 'La cuenta mycellios-control ya existe y no pertenece a este instalador.' }
  if (-not $user) {
    $bytes = New-Object byte[] 48
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $password = ConvertTo-SecureString ([Convert]::ToBase64String($bytes) + '!aA1') -AsPlainText -Force
    $user = New-LocalUser -Name $account -Password $password -PasswordNeverExpires -AccountNeverExpires -Description $description
    $password = $null; [Array]::Clear($bytes, 0, $bytes.Length)
  }
  Enable-LocalUser -Name $account
  $admins = Get-LocalGroup -SID 'S-1-5-32-544'
  if (-not (Get-LocalGroupMember -Group $admins.Name | Where-Object SID -eq $user.SID)) {
    Add-LocalGroupMember -Group $admins.Name -Member $user.Name
  }
  if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
    $result = Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
    if ($result.RestartNeeded) {
      $errorReboot = New-Object InvalidOperationException('OpenSSH requiere reinicio.')
      $errorReboot.Data['MycelliosReboot'] = $true
      throw $errorReboot
    }
  }
  $sshRoot = Join-Path $env:ProgramData 'ssh'
  $null = New-Item -ItemType Directory -Path $sshRoot -Force
  $keysPath = Join-Path $sshRoot 'administrators_authorized_keys'
  $keys = if (Test-Path -LiteralPath $keysPath) { @(Get-Content -LiteralPath $keysPath) } else { @() }
  if ($keys -notcontains $key) { [IO.File]::WriteAllLines($keysPath, [string[]]@($keys + $key), [Text.Encoding]::ASCII) }
  $acl = New-Object Security.AccessControl.FileSecurity
  $acl.SetAccessRuleProtection($true, $false)
  $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
  foreach ($sid in @('S-1-5-18','S-1-5-32-544')) {
    $rule = New-Object Security.AccessControl.FileSystemAccessRule(
      (New-Object Security.Principal.SecurityIdentifier($sid)),
      [Security.AccessControl.FileSystemRights]::FullControl,
      [Security.AccessControl.AccessControlType]::Allow)
    $null = $acl.AddAccessRule($rule)
  }
  [IO.File]::SetAccessControl($keysPath, $acl)
  $configPath = Join-Path $sshRoot 'sshd_config'
  $original = if (Test-Path -LiteralPath $configPath) { [IO.File]::ReadAllText($configPath) } else { '' }
  $begin = '# BEGIN Mycellios USB access'
  $end = '# END Mycellios USB access'
  $original = [regex]::Replace($original, '(?ms)^# BEGIN Mycellios USB access\r?\n.*?^# END Mycellios USB access\r?\n?', '')
  # Preserve existing administrator AllowUsers entries while admitting our managed account.
  $original = [regex]::Replace($original, '(?m)^(AllowUsers\s+)([^\r\n]+)', {
    param($match)
    $names = @($match.Groups[2].Value.Trim() -split '\s+')
    if ($names -notcontains $account) { $names += $account }
    return $match.Groups[1].Value + ($names -join ' ')
  })
  $managedLines = @($begin, 'PasswordAuthentication no', 'KbdInteractiveAuthentication no', 'PubkeyAuthentication yes')
  if ($original -notmatch '(?im)^\s*Subsystem\s+sftp\s+') {
    $managedLines += ('Subsystem sftp "{0}"' -f ((Join-Path $env:WINDIR 'System32\OpenSSH\sftp-server.exe').Replace('\','/')))
  }
  $managed = @($managedLines + @($end, '')) -join "`r`n"
  $config = $managed + $original
  if ($config -notmatch '(?im)^\s*AuthorizedKeysFile\s+__PROGRAMDATA__/ssh/administrators_authorized_keys') {
    $config += "`r`nMatch Group administrators`r`n    AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys`r`n"
  }
  $candidate = Join-Path $sshRoot 'sshd_config.mycellios-candidate'
  [IO.File]::WriteAllText($candidate, $config, [Text.Encoding]::ASCII)
  $sshd = Join-Path $env:WINDIR 'System32\OpenSSH\sshd.exe'
  & (Join-Path $env:WINDIR 'System32\OpenSSH\ssh-keygen.exe') -A
  if ($LASTEXITCODE -ne 0) { throw 'No se pudieron crear las claves del servidor.' }
  & $sshd -t -f $candidate
  if ($LASTEXITCODE -ne 0) { throw 'OpenSSH rechazo la configuracion propuesta.' }
  if (-not (Test-Path -LiteralPath (Join-Path $StateRoot 'sshd_config.before-usb'))) {
    if (Test-Path -LiteralPath $configPath) { Copy-Item -LiteralPath $configPath -Destination (Join-Path $StateRoot 'sshd_config.before-usb') }
  }
  Copy-Item -LiteralPath $candidate -Destination $configPath -Force
  $rule = 'Mycellios-USB-Controller-SSH'
  if (Get-NetFirewallRule -Name $rule -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name $rule -Enabled True -RemoteAddress $ControllerIp -LocalAddress $TailIp | Out-Null
  } else {
    $null = New-NetFirewallRule -Name $rule -DisplayName 'Mycellios USB SSH from controller via Tailscale' -Profile Any `
      -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22 -RemoteAddress $ControllerIp -LocalAddress $TailIp
  }
  if (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue) {
    Disable-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' | Out-Null
  }
  Set-Service sshd -StartupType Automatic
  Restart-Service sshd -ErrorAction Stop
  foreach ($name in @('Tailscale', 'sshd')) {
    & sc.exe failure $name reset= 86400 actions= restart/5000/restart/15000/restart/30000 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "No se pudo configurar recuperacion de $name." }
    & sc.exe failureflag $name 1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "No se pudo activar recuperacion de $name." }
  }
  return $account
}

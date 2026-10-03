#requires -Version 5.1
# Called only by the elevated installer, after the USB manifest was verified.
function Initialize-LabInstallation([string]$SourceRoot, [string]$StateRoot, [string]$UserProfile) {
  $package = Join-Path $StateRoot 'Package'
  $journalPath = Join-Path $StateRoot 'resume.json'
  $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
  $journal = if (Test-Path -LiteralPath $journalPath) {
    Get-Content -LiteralPath $journalPath -Raw | ConvertFrom-Json
  } else { [pscustomobject]@{userProfile=$UserProfile;attempts=0;reboots=0;pendingBoot='';completed=$false} }
  if ($journal.userProfile -ine $UserProfile) { throw 'La instalacion pendiente pertenece a otro perfil Windows.' }
  if ($journal.pendingBoot -eq $boot) { return @{Package=$package;WaitingForReboot=$true;Journal=$journal;Path=$journalPath} }
  $journal.pendingBoot = ''
  if ($SourceRoot.TrimEnd('\') -ine $package.TrimEnd('\')) {
    $null = New-Item -ItemType Directory -Path $package -Force
    $newManifest = @(Get-Content -LiteralPath (Join-Path $SourceRoot 'SHA256SUMS.txt'))
    $newPaths = @($newManifest | ForEach-Object { $_.Substring(66) })
    $previousManifest = Join-Path $package 'SHA256SUMS.txt'
    if (Test-Path -LiteralPath $previousManifest) {
      foreach ($line in (Get-Content -LiteralPath $previousManifest)) {
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw 'Manifest local anterior no valido.' }
        $relative = $Matches[2]
        if ($newPaths -contains $relative) { continue }
        $obsolete = [IO.Path]::GetFullPath((Join-Path $package $relative))
        if (-not $obsolete.StartsWith($package.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Archivo anterior fuera del paquete local.' }
        if (-not (Test-Path -LiteralPath $obsolete)) { continue }
        $item = Get-Item -LiteralPath $obsolete
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Archivo anterior no regular.' }
        for ($parent = Split-Path -Parent $obsolete; $parent -ine $package; $parent = Split-Path -Parent $parent) {
          if ((Get-Item -LiteralPath $parent).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Directorio anterior no regular.' }
        }
        Remove-Item -LiteralPath $obsolete -Force
      }
    }
    foreach ($line in $newManifest) {
      $relative = $line.Substring(66)
      $destination = Join-Path $package $relative
      $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force
      Copy-Item -LiteralPath (Join-Path $SourceRoot $relative) -Destination $destination -Force
    }
    Copy-Item -LiteralPath (Join-Path $SourceRoot 'SHA256SUMS.txt') -Destination $package -Force
    # Never copy the pool of unused tickets onto a target computer.
    $ticketPath = Join-Path $StateRoot 'installation-ticket.json'
    $needsTicket = -not (Test-Path -LiteralPath (Join-Path $env:ProgramData 'Mycellios\Configuration\node.json'))
    $tail = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
    if (Test-Path -LiteralPath $tail) {
      try { $tailStatus = (& $tail status --json 2>$null) | ConvertFrom-Json; $needsTicket = $needsTicket -or $tailStatus.BackendState -ne 'Running' }
      catch { $needsTicket = $true }
    } else { $needsTicket = $true }
    if ($needsTicket -and -not (Test-Path -LiteralPath $ticketPath)) {
      $selected = $false
      foreach ($candidate in @(Get-ChildItem -LiteralPath (Join-Path $SourceRoot 'Tickets') -Filter '*.json' -File | Sort-Object Name)) {
        $ticket = Get-Content -LiteralPath $candidate.FullName -Raw | ConvertFrom-Json
        if ([DateTimeOffset]::Parse($ticket.expiresAt) -le [DateTimeOffset]::UtcNow) { continue }
        [IO.File]::WriteAllText($ticketPath, ($ticket | ConvertTo-Json), [Text.Encoding]::UTF8)
        Remove-Item -LiteralPath $candidate.FullName -Force
        $selected = $true
        break
      }
      if (-not $selected) { throw 'No quedan altas vigentes en este pendrive.' }
    }
    $journal.attempts = 0; $journal.reboots = 0; $journal.completed = $false
  }
  if ($journal.completed) { return @{Package=$package;WaitingForReboot=$false;Journal=$journal;Path=$journalPath} }
  if ($journal.attempts -ge 72) {
    Unregister-ScheduledTask -TaskName 'MycelliosUsbInstallResume' -Confirm:$false -ErrorAction SilentlyContinue
    throw 'Se han agotado los reintentos automaticos. Revisa el log local mediante SSH.'
  }
  $journal.attempts++
  [IO.File]::WriteAllText($journalPath, ($journal | ConvertTo-Json), [Text.Encoding]::UTF8)
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}" -UserProfile "{1}" -Resume' -f (Join-Path $package 'windows-usb-lab-install.ps1'), $UserProfile)
  $startup = New-ScheduledTaskTrigger -AtStartup
  $retry = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(10) -RepetitionInterval (New-TimeSpan -Minutes 10)
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
  $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
  $null = Register-ScheduledTask -TaskName 'MycelliosUsbInstallResume' -Action $action -Trigger @($startup,$retry) -Settings $settings -Principal $principal -Force
  return @{Package=$package;WaitingForReboot=$false;Journal=$journal;Path=$journalPath}
}

function Request-LabInstallationReboot($Installation) {
  if ($Installation.Journal.reboots -ge 2) { throw 'Windows sigue solicitando reinicio despues de dos intentos; se requiere diagnostico remoto.' }
  $Installation.Journal.reboots++
  $Installation.Journal.pendingBoot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
  [IO.File]::WriteAllText($Installation.Path, ($Installation.Journal | ConvertTo-Json), [Text.Encoding]::UTF8)
  Write-Output 'Windows necesita reiniciar. Guarda tu trabajo: reinicio en 60 segundos; la instalacion continuara sola.'
  & shutdown.exe /r /t 60 /d p:4:2 /c 'Mycellios: completar instalacion y continuar automaticamente.' | Out-Null
  if ($LASTEXITCODE -ne 0) { throw 'No se pudo programar el reinicio automatico.' }
}

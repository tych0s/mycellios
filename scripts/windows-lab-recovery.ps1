#requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$StateRoot)
$ErrorActionPreference = 'Stop'
$settings = Get-Content -LiteralPath (Join-Path $StateRoot 'recovery.json') -Raw | ConvertFrom-Json
$env:JARVIS_LOCAL_NODE_CONFIG = Join-Path $settings.userProfile '.jarvis\local-node.json'
$env:JARVIS_LOCAL_NODE_SUPERVISED = '1'
$env:JARVIS_DESKTOP_MCP_URL = ''
$worker = $null
$mutex = New-Object Threading.Mutex($false, 'Global\MycelliosUsbRecovery')
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
  while ($true) {
    foreach ($name in @('Tailscale', 'sshd', 'MycelliosNode')) {
      $service = Get-Service -Name $name -ErrorAction SilentlyContinue
      if ($service -and $service.Status -eq 'Stopped') {
        try { Start-Service -Name $name } catch { }
      }
    }
    # Session zero provides administrative jobs; the user tray owns an interactive desktop.
    $interactive = @(Get-Process explorer -ErrorAction SilentlyContinue | Where-Object SessionId -gt 0).Count -gt 0
    if ($interactive -and $worker -and -not $worker.HasExited) {
      Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue
      $worker = $null
    }
    if (-not $interactive -and (Test-Path -LiteralPath $env:JARVIS_LOCAL_NODE_CONFIG) -and
        (Test-Path -LiteralPath $settings.jarvisRuntime) -and (Test-Path -LiteralPath $settings.nodeExecutable)) {
      $running = @(Get-CimInstance Win32_Process -Filter "Name='node.exe'" |
        Where-Object { $_.CommandLine -match '\blocal-node\s+run\b' })
      if ($running.Count -eq 0) {
        $arguments = @(('"{0}"' -f $settings.jarvisRuntime), 'local-node', 'run',
          '--desktop-runtime', 'legacy-mcp', '--no-clipboard', '--no-activity')
        $worker = Start-Process -FilePath $settings.nodeExecutable -ArgumentList $arguments -PassThru -WindowStyle Hidden `
          -RedirectStandardOutput (Join-Path $StateRoot 'jarvis-headless.stdout.log') `
          -RedirectStandardError (Join-Path $StateRoot 'jarvis-headless.stderr.log')
      }
    }
    Start-Sleep -Seconds 10
  }
} finally {
  if ($worker -and -not $worker.HasExited) { Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue }
  $mutex.ReleaseMutex(); $mutex.Dispose()
}

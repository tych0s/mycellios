param(
  [int]$CoordinatorPort = 8787
)

$ErrorActionPreference = "Stop"

$workspacePath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimePath = Join-Path $workspacePath ".codex-runtime\local-federation"

function Stop-RecordedProcess {
  param(
    [string]$PidFile,
    [string]$ExpectedCommand
  )

  if (-not (Test-Path -LiteralPath $PidFile)) { return }
  $processId = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
  if (-not $process) {
    Remove-Item -LiteralPath $PidFile -Force
    return
  }
  if (-not $process.CommandLine.Contains($ExpectedCommand, [StringComparison]::OrdinalIgnoreCase)) {
    throw "PID $processId no longer belongs to the expected Mycellios local process."
  }
  $descendants = @(Get-ProcessDescendants -ParentId $processId)
  foreach ($descendantId in $descendants) {
    Stop-Process -Id $descendantId -Force -ErrorAction SilentlyContinue
  }
  Stop-Process -Id $processId -Force
  Remove-Item -LiteralPath $PidFile -Force
}

function Get-ProcessDescendants {
  param([int]$ParentId)

  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentId" -ErrorAction SilentlyContinue
  foreach ($child in $children) {
    Get-ProcessDescendants -ParentId ([int]$child.ProcessId)
    [int]$child.ProcessId
  }
}

try {
  Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:$CoordinatorPort/public/v1/admin/federation/emergency-stop" `
    -ContentType "application/json" `
    -Body '{"confirm":true}' `
    -TimeoutSec 15 |
    Out-Null
  Start-Sleep -Milliseconds 500
} catch {
  Write-Warning "The coordinator did not answer the local federation shutdown request."
}

Stop-RecordedProcess `
  -PidFile (Join-Path $runtimePath "ui.pid") `
  -ExpectedCommand "vite.landing.config.ts"
Stop-RecordedProcess `
  -PidFile (Join-Path $runtimePath "coordinator.pid") `
  -ExpectedCommand "src/coordinator/main.ts"

Write-Output "Mycellios local federation stopped."

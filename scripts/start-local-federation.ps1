param(
  [int]$CoordinatorPort = 8787,
  [int]$UiPort = 4174
)

$ErrorActionPreference = "Stop"

$workspacePath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimePath = Join-Path $workspacePath ".codex-runtime\local-federation"
$nodePath = (Get-Command node -ErrorAction Stop).Source
$tsxCliPath = Join-Path $workspacePath "node_modules\tsx\dist\cli.mjs"
$viteCliPath = Join-Path $workspacePath "node_modules\vite\bin\vite.js"
$externalRuntimeAPath = (Get-Command external-runtime-a -ErrorAction SilentlyContinue)?.Source

New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null

function Test-LocalPort {
  param([int]$Port)

  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $connection = $client.ConnectAsync("127.0.0.1", $Port)
    return $connection.Wait(250) -and $client.Connected
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

function Wait-LocalEndpoint {
  param(
    [string]$Uri,
    [int]$Seconds
  )

  $deadline = (Get-Date).AddSeconds($Seconds)
  do {
    Start-Sleep -Milliseconds 500
    try {
      return Invoke-RestMethod -Uri $Uri -TimeoutSec 2
    } catch {
      if ((Get-Date) -ge $deadline) { throw }
    }
  } while ($true)
}

if (-not (Test-Path -LiteralPath $tsxCliPath)) {
  throw "Missing local dependencies. Run npm install before starting federation."
}
if (-not (Test-Path -LiteralPath $viteCliPath)) {
  throw "Missing local Vite dependency. Run npm install before starting federation."
}
if (Test-LocalPort -Port $CoordinatorPort) {
  throw "Port $CoordinatorPort is already in use. Stop the existing coordinator first."
}
if (Test-LocalPort -Port $UiPort) {
  throw "Port $UiPort is already in use. Stop the existing local UI first."
}

$env:MYCELLIOS_FEDERATION_ENABLED = "1"
$env:GPU_MESH_DB = Join-Path $runtimePath "gpu-mesh.db"
$env:GPU_MESH_HOST = "127.0.0.1"
$env:GPU_MESH_PORT = [string]$CoordinatorPort
if ($externalRuntimeAPath) {
  $env:MYCELLIOS_EXTERNAL_RUNTIME_A_EXECUTABLE = $externalRuntimeAPath
}

$coordinator = Start-Process `
  -FilePath $nodePath `
  -ArgumentList @("""$tsxCliPath""", "src/coordinator/main.ts") `
  -WorkingDirectory $workspacePath `
  -RedirectStandardOutput (Join-Path $runtimePath "coordinator.out.log") `
  -RedirectStandardError (Join-Path $runtimePath "coordinator.err.log") `
  -WindowStyle Hidden `
  -PassThru
$coordinator.Id | Set-Content -LiteralPath (Join-Path $runtimePath "coordinator.pid")

try {
  $snapshot = Wait-LocalEndpoint `
    -Uri "http://127.0.0.1:$CoordinatorPort/public/v1/snapshot" `
    -Seconds 30

  $headers = @{ "Content-Type" = "application/json" }
  Invoke-RestMethod `
    -Method Put `
    -Uri "http://127.0.0.1:$CoordinatorPort/public/v1/admin/federation/settings" `
    -Headers $headers `
    -Body '{"enabled":true,"dailyBudgetUsd":0,"monthlyBudgetUsd":0,"autoscalingEnabled":false,"maxRentals":4}' |
    Out-Null

  foreach ($networkId in @("external-runtime-a", "ai-horde", "peer-runtime")) {
    Invoke-RestMethod `
      -Method Put `
      -Uri "http://127.0.0.1:$CoordinatorPort/public/v1/admin/federation/networks/$networkId" `
      -Headers $headers `
      -Body '{"enabled":true,"dailyBudgetUsd":0,"monthlyBudgetUsd":0}' |
      Out-Null
  }

  $env:MYCELLIOS_UI_API_TARGET = "http://127.0.0.1:$CoordinatorPort"
  $ui = Start-Process `
    -FilePath $nodePath `
    -ArgumentList @("""$viteCliPath""", "--config", "vite.landing.config.ts", "--port", [string]$UiPort) `
    -WorkingDirectory $workspacePath `
    -RedirectStandardOutput (Join-Path $runtimePath "ui.out.log") `
    -RedirectStandardError (Join-Path $runtimePath "ui.err.log") `
    -WindowStyle Hidden `
    -PassThru
  $ui.Id | Set-Content -LiteralPath (Join-Path $runtimePath "ui.pid")

  Wait-LocalEndpoint -Uri "http://127.0.0.1:$UiPort/network?view=networks" -Seconds 30 |
    Out-Null

  Write-Output "Mycellios local federation is ready."
  Write-Output "Panel: http://127.0.0.1:$UiPort/network?view=networks"
  Write-Output "Coordinator: http://127.0.0.1:$CoordinatorPort"
  Write-Output "Runtime logs: $runtimePath"
  Write-Output "external runtime A client: $(if ($externalRuntimeAPath) { $externalRuntimeAPath } else { "not installed; card remains unavailable" })"
  Write-Output "Paid APIs and rentals remain blocked with zero budget."
} catch {
  & (Join-Path $PSScriptRoot "stop-local-federation.ps1") -CoordinatorPort $CoordinatorPort
  throw
}

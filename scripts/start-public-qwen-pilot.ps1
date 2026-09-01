[CmdletBinding()]
param(
    [string] $CoordinatorUrl = 'https://mycellios.nodecodex.io'
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeDirectory = Join-Path $workspace 'runtime\public-qwen-pilot'
$statePath = Join-Path $runtimeDirectory 'state.json'
$python = & (Join-Path $PSScriptRoot 'resolve-distribution-python.ps1') -WorkspacePath $workspace
$node = (Get-Command node.exe -ErrorAction Stop).Source
$workerConfig = Join-Path $workspace 'config\worker.public-qwen-pilot.json'
$revision = 'c1899de289a04d12100db370d81485cdf75e47ca'
$model = 'Qwen/Qwen3-0.6B'
$publicModel = 'qwen3-0.6b-gdlp2-pilot'

if (Test-Path -LiteralPath $statePath) {
    $existing = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $live = @($existing.serverPid, $existing.workerPid) | Where-Object {
        $_ -and (Get-Process -Id $_ -ErrorAction SilentlyContinue)
    }
    if ($live.Count -gt 0) {
        throw "The public Qwen pilot is already running. Inspect $statePath or stop it first."
    }
}

New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$serverStdout = Join-Path $runtimeDirectory "server-$stamp.stdout.log"
$serverStderr = Join-Path $runtimeDirectory "server-$stamp.stderr.log"
$workerStdout = Join-Path $runtimeDirectory "worker-$stamp.stdout.log"
$workerStderr = Join-Path $runtimeDirectory "worker-$stamp.stderr.log"

function Wait-JsonEndpoint {
    param(
        [Parameter(Mandatory)] [string] $Uri,
        [Parameter(Mandatory)] [scriptblock] $Accept,
        [int] $TimeoutSeconds = 180,
        [System.Diagnostics.Process] $Process
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if ($Process -and $Process.HasExited) {
            throw "Process $($Process.Id) exited with code $($Process.ExitCode) while waiting for $Uri"
        }
        try {
            $body = Invoke-RestMethod -Uri $Uri -TimeoutSec 5
            if (& $Accept $body) { return $body }
        } catch {}
        Start-Sleep -Milliseconds 500
    } until ((Get-Date) -ge $deadline)
    throw "Timed out waiting for $Uri"
}

function Stop-PilotProcessTree {
    param(
        [System.Diagnostics.Process] $RootProcess,
        [Parameter(Mandatory)] [string] $ExpectedCommand
    )
    if (-not $RootProcess -or $RootProcess.HasExited) { return }
    $all = @(Get-CimInstance Win32_Process)
    $root = $all | Where-Object ProcessId -eq $RootProcess.Id | Select-Object -First 1
    if (-not $root -or $root.CommandLine -notlike "*$ExpectedCommand*") { return }
    $targets = [System.Collections.Generic.List[int]]::new()
    function Add-Descendants([int] $ParentId) {
        foreach ($child in $all | Where-Object ParentProcessId -eq $ParentId) {
            Add-Descendants $child.ProcessId
            $targets.Add([int]$child.ProcessId)
        }
    }
    Add-Descendants $RootProcess.Id
    $targets.Add($RootProcess.Id)
    foreach ($targetId in $targets) {
        Stop-Process -Id $targetId -Force -ErrorAction SilentlyContinue
    }
}

$previousHfHome = $env:HF_HOME
$previousPythonPath = $env:PYTHONPATH
$previousTokenizerParallelism = $env:TOKENIZERS_PARALLELISM
$server = $null
$worker = $null
try {
    $env:HF_HOME = Join-Path $workspace 'runtime\hf-cache'
    $env:PYTHONPATH = Join-Path $workspace 'python'
    $env:TOKENIZERS_PARALLELISM = 'false'
    $server = Start-Process -FilePath $python -ArgumentList @(
        '-m', 'distributed_runtime.server',
        '--model', $model,
        '--revision', $revision,
        '--public-model-name', $publicModel,
        '--host', '127.0.0.1',
        '--port', '8082',
        '--stages', '2',
        '--boundaries', '0,14,28',
        '--codec', 'fp16',
        '--threads-per-stage', '1',
        '--max-active-sequences', '1',
        '--max-batch-size', '1'
    ) -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $serverStdout -RedirectStandardError $serverStderr

    $health = Wait-JsonEndpoint -Uri 'http://127.0.0.1:8082/health' `
        -Accept { param($body) $body.status -eq 'ready' -and $body.stages -eq 2 } `
        -TimeoutSeconds 300 -Process $server

    $canaryBody = @{
        model = $publicModel
        messages = @(@{
            role = 'user'
            content = 'Responde solo OK si esta inferencia funciona. /no_think'
        })
        temperature = 0
        max_tokens = 16
        stream = $false
    } | ConvertTo-Json -Depth 6
    $canary = Invoke-RestMethod -Uri 'http://127.0.0.1:8082/v1/chat/completions' `
        -Method Post -ContentType 'application/json' -Body $canaryBody -TimeoutSec 180
    if ([string]::IsNullOrWhiteSpace($canary.choices[0].message.content)) {
        throw 'The distributed Qwen canary returned an empty response.'
    }

    $env:GPU_MESH_COORDINATOR = $CoordinatorUrl
    $env:GPU_MESH_WORKER_CONFIG = $workerConfig
    $worker = Start-Process -FilePath $node -ArgumentList @(
        '--import', 'tsx', 'src/worker/main.ts'
    ) -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $workerStdout -RedirectStandardError $workerStderr

    Wait-JsonEndpoint -Uri "$($CoordinatorUrl.TrimEnd('/'))/public/v1/snapshot" `
        -Accept { param($body) $body.models.id -contains $publicModel } `
        -TimeoutSeconds 45 -Process $worker | Out-Null

    [ordered]@{
        schema = 'mycellios-public-pilot-state/1'
        startedAt = (Get-Date).ToUniversalTime().ToString('o')
        coordinatorUrl = $CoordinatorUrl
        model = $publicModel
        revision = $revision
        boundaries = @(0, 14, 28)
        serverPid = $server.Id
        workerPid = $worker.Id
        serverStdout = $serverStdout
        serverStderr = $serverStderr
        workerStdout = $workerStdout
        workerStderr = $workerStderr
        canaryText = $canary.choices[0].message.content
        pipelineSnapshotIdentity = $health.pipeline_snapshot_identity
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding utf8

    Write-Host "Public distributed pilot is online: $publicModel"
    Write-Host "Stages: [0,14) -> [14,28) on this host"
    Write-Host "State: $statePath"
} catch {
    Stop-PilotProcessTree -RootProcess $worker -ExpectedCommand 'src/worker/main.ts'
    Stop-PilotProcessTree -RootProcess $server -ExpectedCommand 'distributed_runtime.server'
    throw
} finally {
    $env:HF_HOME = $previousHfHome
    $env:PYTHONPATH = $previousPythonPath
    $env:TOKENIZERS_PARALLELISM = $previousTokenizerParallelism
    Remove-Item Env:GPU_MESH_COORDINATOR -ErrorAction SilentlyContinue
    Remove-Item Env:GPU_MESH_WORKER_CONFIG -ErrorAction SilentlyContinue
}

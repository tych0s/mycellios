[CmdletBinding()]
param(
    [ValidateRange(1, 6)]
    [int] $Turns = 1
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$server = Join-Path $workspace 'runtime\llama-b10068-vulkan\llama-server.exe'
$model = Join-Path $workspace 'runtime\models\Qwen3-0.6B-Q8_0.gguf'
$workerConfig = Join-Path $workspace 'config\worker.local-qwen.example.json'
$database = Join-Path $workspace 'data\real-demo.db'
$node = (Get-Command node.exe -ErrorAction Stop).Source
$logDirectory = Join-Path $workspace 'runtime\logs'
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$demoStartedAt = Get-Date
$llamaStdoutLog = Join-Path $logDirectory "llama-server-$runStamp.stdout.log"
$llamaStderrLog = Join-Path $logDirectory "llama-server-$runStamp.stderr.log"
$coordinatorStdoutLog = Join-Path $logDirectory "coordinator-$runStamp.stdout.log"
$coordinatorStderrLog = Join-Path $logDirectory "coordinator-$runStamp.stderr.log"
$workerStdoutLog = Join-Path $logDirectory "worker-$runStamp.stdout.log"
$workerStderrLog = Join-Path $logDirectory "worker-$runStamp.stderr.log"

function Start-CapturedProcess {
    param(
        [Parameter(Mandatory)] [string] $FilePath,
        [Parameter(Mandatory)] [string[]] $ProcessArguments,
        [Parameter(Mandatory)] [string] $WorkingDirectory,
        [Parameter(Mandatory)] [string] $StdoutLog,
        [Parameter(Mandatory)] [string] $StderrLog
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    if ($null -ne $startInfo.ArgumentList) {
        foreach ($argument in $ProcessArguments) {
            [void] $startInfo.ArgumentList.Add($argument)
        }
    }
    else {
        # Windows PowerShell 5.1 has no ArgumentList API. These controlled
        # arguments contain no embedded quotes, so explicit quoting preserves
        # paths with spaces without invoking a shell.
        $startInfo.Arguments = ($ProcessArguments | ForEach-Object {
            '"' + $_.Replace('"', '\"') + '"'
        }) -join ' '
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    [void] $process.Start()

    [pscustomobject]@{
        Process = $process
        StdoutTask = $process.StandardOutput.ReadToEndAsync()
        StderrTask = $process.StandardError.ReadToEndAsync()
        StdoutLog = $StdoutLog
        StderrLog = $StderrLog
        Saved = $false
    }
}

function Save-CapturedProcessLogs {
    param([Parameter(Mandatory)] $CapturedProcess)

    if ($CapturedProcess.Saved) { return }
    if (-not $CapturedProcess.Process.HasExited) {
        throw 'Cannot save captured process logs while the process is still running.'
    }

    $stdout = $CapturedProcess.StdoutTask.GetAwaiter().GetResult()
    $stderr = $CapturedProcess.StderrTask.GetAwaiter().GetResult()
    [System.IO.File]::WriteAllText($CapturedProcess.StdoutLog, $stdout)
    [System.IO.File]::WriteAllText($CapturedProcess.StderrLog, $stderr)
    $CapturedProcess.Saved = $true
}

function Show-LlamaDiagnostics {
    param([Parameter(Mandatory)] $CapturedProcess)

    Write-Host "llama-server exit code: $($CapturedProcess.Process.ExitCode)" -ForegroundColor Yellow
    Write-Host "stdout: $($CapturedProcess.StdoutLog)" -ForegroundColor Yellow
    Get-Content -LiteralPath $CapturedProcess.StdoutLog -Tail 80 -ErrorAction SilentlyContinue
    Write-Host "stderr: $($CapturedProcess.StderrLog)" -ForegroundColor Yellow
    Get-Content -LiteralPath $CapturedProcess.StderrLog -Tail 80 -ErrorAction SilentlyContinue
}

function Wait-Http {
    param(
        [Parameter(Mandatory)] [string] $Uri,
        [int] $TimeoutSeconds = 30,
        [System.Diagnostics.Process] $Process
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if ($Process -and $Process.HasExited) {
            throw "Process $($Process.Id) exited with code $($Process.ExitCode) while waiting for $Uri"
        }
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        }
        catch {}
        Start-Sleep -Milliseconds 300
    } until ((Get-Date) -ge $deadline)
    throw "Timed out waiting for $Uri"
}

if (-not (Test-Path -LiteralPath $server) -or -not (Test-Path -LiteralPath $model)) {
    & (Join-Path $PSScriptRoot 'setup-local-externalggufruntime.ps1')
}

$started = @()
$llamaCapture = $null
try {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $llamaCapture = Start-CapturedProcess -FilePath $server -ProcessArguments @(
        '--model', $model,
        '--host', '127.0.0.1',
        '--port', '8080',
        '--ctx-size', '8192',
        '--n-gpu-layers', '99',
        '--jinja'
    ) -WorkingDirectory $workspace -StdoutLog $llamaStdoutLog -StderrLog $llamaStderrLog
    $llama = $llamaCapture.Process
    $started += $llama
    try {
        Wait-Http -Uri 'http://127.0.0.1:8080/health' -TimeoutSeconds 180 -Process $llama
    }
    catch {
        if (-not $llama.HasExited) {
            Stop-Process -Id $llama.Id -Force -ErrorAction SilentlyContinue
            [void] $llama.WaitForExit(10000)
        }
        Save-CapturedProcessLogs -CapturedProcess $llamaCapture
        Show-LlamaDiagnostics -CapturedProcess $llamaCapture
        throw
    }

    $coordinatorEnv = @{
        GPU_MESH_HOST = '127.0.0.1'
        GPU_MESH_PORT = '8787'
        GPU_MESH_DB = $database
    }
    foreach ($entry in $coordinatorEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    $coordinator = Start-Process -FilePath $node -ArgumentList @(
        '--import', 'tsx', 'src/coordinator/main.ts'
    ) `
        -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $coordinatorStdoutLog `
        -RedirectStandardError $coordinatorStderrLog
    $started += $coordinator
    Wait-Http -Uri 'http://127.0.0.1:8787/health' -TimeoutSeconds 30

    $env:GPU_MESH_COORDINATOR = 'http://127.0.0.1:8787'
    $env:GPU_MESH_WORKER_CONFIG = $workerConfig
    $worker = Start-Process -FilePath $node -ArgumentList @(
        '--import', 'tsx', 'src/worker/main.ts'
    ) `
        -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $workerStdoutLog `
        -RedirectStandardError $workerStderrLog
    $started += $worker

    $deadline = (Get-Date).AddSeconds(30)
    $connectedWorker = $null
    do {
        Start-Sleep -Milliseconds 300
        $workers = Invoke-RestMethod 'http://127.0.0.1:8787/internal/v1/workers'
        $connectedWorker = $workers.data | Where-Object {
            $_.status -eq 'online' -and $_.connected -eq $true
        } | Select-Object -First 1
    } until ($connectedWorker -or (Get-Date) -ge $deadline)
    if (-not $connectedWorker) { throw 'The real Qwen worker did not connect in time.' }

    $questions = @(
        'Answer in one sentence: what does this test demonstrate? /no_think',
        'In one sentence: what is the benefit of splitting a model by layers? /no_think',
        'In one sentence: why does session affinity help? /no_think',
        'In one sentence: what can a laptop contribute if it offers only 4 GB? /no_think',
        'In one sentence: why are fewer hops better for chat? /no_think',
        'In one sentence: which benchmark should we run next? /no_think'
    )
    $conversation = [System.Collections.ArrayList]::new()
    $measurements = @()
    $sessionId = "real-local-demo-$runStamp"
    for ($turn = 1; $turn -le $Turns; $turn++) {
        [void] $conversation.Add(@{ role = 'user'; content = $questions[$turn - 1] })
        $chatHeaders = @{
            'Idempotency-Key' = "real-demo-$runStamp-turn-$turn"
        }
        $chat = @{
            model = 'qwen3-0.6b-q8'
            messages = @($conversation)
            max_tokens = 80
            stream = $false
            session_id = $sessionId
        } | ConvertTo-Json -Depth 10
        $requestStartedAt = Get-Date
        $result = Invoke-RestMethod 'http://127.0.0.1:8787/v1/chat/completions' -Method Post `
            -Headers $chatHeaders -ContentType 'application/json' -Body $chat
        $requestSeconds = [Math]::Max(0.001, ((Get-Date) - $requestStartedAt).TotalSeconds)
        $answer = $result.choices[0].message.content
        [void] $conversation.Add(@{ role = 'assistant'; content = $answer })
        $measurements += [pscustomobject]@{
            Turn = $turn
            Model = $result.model
            Answer = $answer
            PromptTokens = $result.usage.prompt_tokens
            OutputTokens = $result.usage.completion_tokens
            WorkerTtftMs = $result.x_network.ttft_ms
            WorkerActiveMs = $result.x_network.active_ms
            Session = $result.x_network.session_id
            AffinityHit = $result.x_network.affinity_hit
            RequestSeconds = [Math]::Round($requestSeconds, 3)
            EstimatedTokensPerSecond = [Math]::Round(
                $result.usage.completion_tokens / $requestSeconds,
                2
            )
        }
    }

    if ($Turns -eq 1) {
        $measurements[0] | Select-Object *, @{
            Name = 'TotalDemoSeconds'
            Expression = { [Math]::Round(((Get-Date) - $demoStartedAt).TotalSeconds, 3) }
        } | Format-List
    }
    else {
        $measurements | Select-Object Turn, PromptTokens, OutputTokens, RequestSeconds,
            EstimatedTokensPerSecond, WorkerTtftMs, WorkerActiveMs, AffinityHit |
            Format-Table -AutoSize
        foreach ($measurement in $measurements) {
            Write-Host "Turno $($measurement.Turn): $($measurement.Answer)"
        }
        Write-Host "Tiempo total demo: $([Math]::Round(((Get-Date) - $demoStartedAt).TotalSeconds, 3)) s"
    }
}
finally {
    for ($index = $started.Count - 1; $index -ge 0; $index--) {
        $process = $started[$index]
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    if ($llamaCapture) {
        [void] $llamaCapture.Process.WaitForExit(10000)
        Save-CapturedProcessLogs -CapturedProcess $llamaCapture
    }
}

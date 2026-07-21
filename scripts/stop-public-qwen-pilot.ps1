[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$statePath = Join-Path $workspace 'runtime\public-qwen-pilot\state.json'
if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Host 'The public Qwen pilot has no state file.'
    exit 0
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
function Stop-VerifiedProcessTree {
    param(
        [int] $RootProcessId,
        [Parameter(Mandatory)] [string] $ExpectedCommand
    )
    if (-not $RootProcessId) { return }
    $all = @(Get-CimInstance Win32_Process)
    $root = $all | Where-Object ProcessId -eq $RootProcessId | Select-Object -First 1
    if (-not $root) { return }
    if ($root.CommandLine -notlike "*$ExpectedCommand*") {
        throw "Refusing to stop PID $RootProcessId because it is not the recorded pilot process."
    }
    $targets = [System.Collections.Generic.List[int]]::new()
    function Add-Descendants([int] $ParentId) {
        foreach ($child in $all | Where-Object ParentProcessId -eq $ParentId) {
            Add-Descendants $child.ProcessId
            $targets.Add([int]$child.ProcessId)
        }
    }
    Add-Descendants $RootProcessId
    $targets.Add($RootProcessId)
    foreach ($targetId in $targets) {
        Stop-Process -Id $targetId -Force -ErrorAction SilentlyContinue
    }
}

Stop-VerifiedProcessTree -RootProcessId $state.workerPid -ExpectedCommand 'src/worker/main.ts'
Stop-VerifiedProcessTree -RootProcessId $state.serverPid -ExpectedCommand 'distributed_runtime.server'
Write-Host 'The public Qwen pilot was stopped.'

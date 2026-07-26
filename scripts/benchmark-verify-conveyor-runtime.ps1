param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenchmarkArgs
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = & (Join-Path $PSScriptRoot "resolve-distribution-python.ps1") -WorkspacePath $workspacePath

$cachePath = Join-Path $workspacePath "runtime\hf-cache"
New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
$env:HF_HOME = $cachePath
$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:TOKENIZERS_PARALLELISM = "false"

Push-Location $workspacePath
$benchmarkExitCode = 1
try {
    & $venvPython -m distributed_runtime.verify_conveyor_runtime_benchmark @BenchmarkArgs
    $benchmarkExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $benchmarkExitCode

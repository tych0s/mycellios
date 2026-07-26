param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StageArgs
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = & (Join-Path $PSScriptRoot "resolve-distribution-python.ps1") -WorkspacePath $workspacePath

$env:HF_HOME = Join-Path $workspacePath "runtime\hf-cache"
$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:TOKENIZERS_PARALLELISM = "false"

Push-Location $workspacePath
$stageExitCode = 1
try {
    & $venvPython -m distributed_runtime.stage_cli @StageArgs
    $stageExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $stageExitCode

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StageArgs
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $workspacePath "runtime\distribution-venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Distribution runtime is missing. Run scripts\setup-distribution-runtime.ps1 first."
}

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

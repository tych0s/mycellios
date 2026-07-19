param(
    [switch]$SkipModelIntegration
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
if ($SkipModelIntegration) {
    Remove-Item Env:RUN_DISTRIBUTED_MODEL_TESTS -ErrorAction SilentlyContinue
} else {
    $env:RUN_DISTRIBUTED_MODEL_TESTS = "1"
}

Push-Location $workspacePath
$testExitCode = 1
try {
    & $venvPython -m unittest discover -s python\tests -v
    $testExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExitCode

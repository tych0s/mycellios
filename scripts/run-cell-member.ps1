param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MemberArgs
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $workspacePath "runtime\distribution-venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Distribution runtime is missing. Run scripts\setup-distribution-runtime.ps1 first."
}

$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:TOKENIZERS_PARALLELISM = "false"

Push-Location $workspacePath
$memberExitCode = 1
try {
    & $venvPython -m distributed_runtime.cell_member_cli @MemberArgs
    $memberExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $memberExitCode

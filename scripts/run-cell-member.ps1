param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MemberArgs
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = & (Join-Path $PSScriptRoot "resolve-distribution-python.ps1") -WorkspacePath $workspacePath

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

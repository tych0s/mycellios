param(
    [string]$PythonExe = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ReferenceArgs
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $selectedPython = & (Join-Path $PSScriptRoot "resolve-distribution-python.ps1") -WorkspacePath $workspacePath
} elseif (Test-Path -LiteralPath $PythonExe -PathType Leaf) {
    $selectedPython = [System.IO.Path]::GetFullPath($PythonExe)
} else {
    $pythonCommand = Get-Command -Name $PythonExe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $pythonCommand) {
        throw "Python runtime was not found: $PythonExe"
    }
    $selectedPython = $pythonCommand.Source
}

$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:HF_HOME = Join-Path $workspacePath "runtime\hf-cache"
$env:TOKENIZERS_PARALLELISM = "false"

Push-Location $workspacePath
try {
    & $selectedPython -m distributed_runtime.canary_reference @ReferenceArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

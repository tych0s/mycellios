param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,
    [string]$PythonExe = "python",
    [string]$Revision = "",
    [string]$LayerPrefix = "",
    [string]$JsonOut = ""
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$env:PYTHONPATH = Join-Path $workspacePath "python"

$arguments = @("-m", "distributed_runtime.profile", $ModelName)
if (-not [string]::IsNullOrWhiteSpace($Revision)) {
    $arguments += @("--revision", $Revision)
}
if (-not [string]::IsNullOrWhiteSpace($LayerPrefix)) {
    $arguments += @("--layer-prefix", $LayerPrefix)
}
if (-not [string]::IsNullOrWhiteSpace($JsonOut)) {
    $arguments += @("--json-out", $JsonOut)
}

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Model profile compilation failed with exit code $LASTEXITCODE."
}

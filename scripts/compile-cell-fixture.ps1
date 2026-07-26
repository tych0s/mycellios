param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [Parameter(Mandatory = $true)]
    [int]$LayerStart,
    [Parameter(Mandatory = $true)]
    [int]$LayerEnd,
    [Parameter(Mandatory = $true)]
    [int]$WorldSize,
    [string]$Revision = "",
    [ValidateSet("float32", "float16", "bfloat16")]
    [string]$OutputDtype = "float32",
    [double[]]$RankWeight = @()
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = & (Join-Path $PSScriptRoot "resolve-distribution-python.ps1") -WorkspacePath $workspacePath
$cachePath = Join-Path $workspacePath "runtime\hf-cache"
New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
$env:HF_HOME = $cachePath
$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:TOKENIZERS_PARALLELISM = "false"

$arguments = @(
    "-m", "distributed_runtime.cell_fixture_compiler",
    $ModelName,
    $Destination,
    "--layer-start", $LayerStart,
    "--layer-end", $LayerEnd,
    "--world-size", $WorldSize,
    "--output-dtype", $OutputDtype
)
if (-not [string]::IsNullOrWhiteSpace($Revision)) {
    $arguments += @("--revision", $Revision)
}
if ($RankWeight.Count -ne 0 -and $RankWeight.Count -ne $WorldSize) {
    throw "RankWeight must be omitted or contain exactly WorldSize values."
}
foreach ($weight in $RankWeight) {
    if ($weight -le 0 -or [double]::IsNaN($weight) -or [double]::IsInfinity($weight)) {
        throw "Every RankWeight must be finite and positive."
    }
    $arguments += @("--rank-weight", $weight)
}

& $venvPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Cell fixture compilation failed with exit code $LASTEXITCODE."
}

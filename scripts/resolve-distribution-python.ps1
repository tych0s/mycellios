[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspacePath
)

$workspace = [System.IO.Path]::GetFullPath($WorkspacePath)
$runtimeRoot = Join-Path (Join-Path $workspace "runtime") "distribution-venv"
$candidates = @(
    (Join-Path $runtimeRoot "python.exe"),
    (Join-Path $runtimeRoot "Scripts\python.exe"),
    (Join-Path $runtimeRoot "bin\python3"),
    (Join-Path $runtimeRoot "bin\python")
)

foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        Write-Output ([System.IO.Path]::GetFullPath($candidate))
        return
    }
}

throw "Distribution runtime is missing. Run npm run desktop:runtime first."

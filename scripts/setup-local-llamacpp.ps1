[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
if ($env:MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH -ne '1') {
    throw 'Archived external-runtime research harness. Set MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH=1 only for an isolated research run.'
}

$workspace = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $workspace 'runtime'
$llamaRoot = Join-Path $runtimeRoot 'llama-b10068-vulkan'
$modelRoot = Join-Path $runtimeRoot 'models'
$archive = Join-Path $runtimeRoot 'llama-b10068-bin-win-vulkan-x64.zip'
$model = Join-Path $modelRoot 'Qwen3-0.6B-Q8_0.gguf'

$llamaUrl = 'https://github.com/ggml-org/llama.cpp/releases/download/b10068/llama-b10068-bin-win-vulkan-x64.zip'
$llamaSha256 = '4f3e6fd215fdf22d2fd6232a5501f9e791a93d9193db4faf59e391eff90f6169'
$modelUrl = 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true'
$modelSha256 = '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'

New-Item -ItemType Directory -Force -Path $runtimeRoot, $modelRoot | Out-Null

function Assert-FileHash {
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [string] $Expected
    )
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "SHA256 mismatch for $Path. Expected $Expected, got $actual"
    }
}

if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host 'Downloading pinned llama.cpp Vulkan build (about 33 MB)...'
    Invoke-WebRequest -Uri $llamaUrl -OutFile $archive
}
Assert-FileHash -Path $archive -Expected $llamaSha256

$server = Join-Path $llamaRoot 'llama-server.exe'
if (-not (Test-Path -LiteralPath $server)) {
    New-Item -ItemType Directory -Force -Path $llamaRoot | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $llamaRoot -Force
}
if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server.exe was not found after extracting $archive"
}

if (-not (Test-Path -LiteralPath $model)) {
    Write-Host 'Downloading official Qwen3 0.6B Q8 GGUF (about 639 MB)...'
    Invoke-WebRequest -Uri $modelUrl -OutFile $model
}
Assert-FileHash -Path $model -Expected $modelSha256

Write-Host 'Local inference runtime is ready.'
Write-Host "Server: $server"
Write-Host "Model:  $model"

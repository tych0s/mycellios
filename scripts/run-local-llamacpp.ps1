[CmdletBinding()]
param(
    [int] $Port = 8080,
    [int] $Context = 8192
)

$ErrorActionPreference = 'Stop'
if ($env:MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH -ne '1') {
    throw 'Archived external-runtime research harness. Set MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH=1 only for an isolated research run.'
}
$workspace = Split-Path -Parent $PSScriptRoot
$server = Join-Path $workspace 'runtime\llama-b10068-vulkan\llama-server.exe'
$model = Join-Path $workspace 'runtime\models\Qwen3-0.6B-Q8_0.gguf'

if (-not (Test-Path -LiteralPath $server) -or -not (Test-Path -LiteralPath $model)) {
    throw 'Runtime missing. Run scripts\setup-local-llamacpp.ps1 first.'
}

& $server `
    --model $model `
    --host 127.0.0.1 `
    --port $Port `
    --ctx-size $Context `
    --n-gpu-layers 99 `
    --jinja

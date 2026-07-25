param(
    [string]$PythonExe = ""
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
        throw "GPU Python runtime was not found: $PythonExe"
    }
    $selectedPython = $pythonCommand.Source
}

$env:HF_HOME = Join-Path $workspacePath "runtime\hf-cache"
$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:TOKENIZERS_PARALLELISM = "false"
$env:RUN_DISTRIBUTED_GPU_TESTS = "1"

$preflight = @'
import json
import sys

import torch
import torch.distributed as distributed

evidence = {
    "torch": torch.__version__,
    "cuda_or_rocm_available": bool(torch.cuda.is_available()),
    "visible_gpu_count": int(torch.cuda.device_count()),
    "nccl_or_rccl_available": bool(
        distributed.is_available() and distributed.is_nccl_available()
    ),
    "cuda_runtime": torch.version.cuda,
    "rocm_runtime": getattr(torch.version, "hip", None),
}
if (
    not evidence["cuda_or_rocm_available"]
    or evidence["visible_gpu_count"] < 2
    or not evidence["nccl_or_rccl_available"]
):
    print(
        "Physical GPU cell preflight failed: this gate requires a CUDA/ROCm "
        "PyTorch runtime, NCCL/RCCL, and at least two visible GPUs on one host; "
        f"observed {json.dumps(evidence, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)
print(
    "Physical GPU cell preflight passed: " + json.dumps(evidence, sort_keys=True),
    flush=True,
)
'@

$preflight | & $selectedPython -
if ($LASTEXITCODE -ne 0) {
    throw "Physical GPU cell preflight failed with exit code $LASTEXITCODE. No GPU test was executed."
}

Push-Location $workspacePath
$testExitCode = 1
try {
    & $selectedPython -m unittest discover -s python\tests -p test_cell_gpu.py -v
    $testExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExitCode

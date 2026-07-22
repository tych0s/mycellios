param(
    [string]$PythonExe = "python",
    [string]$VenvPath = "",
    [string]$ModelName = "HuggingFaceTB/SmolLM2-135M-Instruct",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = Join-Path $workspacePath "runtime\distribution-venv"
} else {
    $VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
}
$requirementsPath = Join-Path $workspacePath "python\requirements-distribution.txt"
$cachePath = Join-Path $workspacePath "runtime\hf-cache"

$pythonVersion = & $PythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Python interpreter (exit code $LASTEXITCODE)."
}
if (-not $pythonVersion.StartsWith("3.12.")) {
    throw "The portable Windows runtime requires CPython 3.12; found $pythonVersion."
}

Write-Host "Creating distribution runtime at $VenvPath"
& $PythonExe -m venv $VenvPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the virtual environment (exit code $LASTEXITCODE)."
}
$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment Python was not created at $venvPython"
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip (exit code $LASTEXITCODE)."
}
& $venvPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the CPU PyTorch wheel (exit code $LASTEXITCODE)."
}
& $venvPython -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the distribution requirements (exit code $LASTEXITCODE)."
}
& $venvPython -c "import accelerate, sys, torch, transformers; assert sys.version_info[:2] == (3, 12); assert accelerate.__version__ == '1.14.0'; assert transformers.__version__ == '5.14.1'; assert torch.__version__ == '2.13.0+cpu'; assert torch.version.cuda is None; assert getattr(torch.version, 'hip', None) is None; assert not torch.cuda.is_available(); print('HF CPU bootstrap runtime:', 'python=' + '.'.join(map(str, sys.version_info[:3])), 'accelerate=' + accelerate.__version__, 'transformers=' + transformers.__version__, 'torch=' + torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Could not validate the pinned HF native TP/EP runtime (exit code $LASTEXITCODE)."
}

New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
$env:HF_HOME = $cachePath
$env:PYTHONPATH = Join-Path $workspacePath "python"
$env:TOKENIZERS_PARALLELISM = "false"

if (-not $SkipModelDownload) {
    Write-Host "Downloading safetensors and validating metadata for $ModelName in $cachePath"
    $env:GPU_DISTRIBUTION_MODEL_NAME = $ModelName
    & $venvPython -c "import os; from transformers import AutoConfig; from distributed_runtime.model import load_tokenizer, resolve_model_snapshot; name=os.environ['GPU_DISTRIBUTION_MODEL_NAME']; snapshot=resolve_model_snapshot(name); AutoConfig.from_pretrained(snapshot); load_tokenizer(snapshot); print('Selective model snapshot ready:', snapshot)"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download or validate $ModelName (exit code $LASTEXITCODE)."
    }
}

Write-Host "Distribution runtime ready. Run scripts\benchmark-distributed.ps1 next."

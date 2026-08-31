"""Low-latency layer-pipeline runtime used by the distribution laboratory."""

# A small set of control-plane helpers (for example activation-sketch
# verification) is deliberately Torch-free. Keep the package importable on a
# verifier host without the runtime wheel set; tensor modules still import
# Torch themselves and fail normally when that capability is actually used.
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
else:
    from .rocm_compat import install_windows_rocm_transformers_compat

    # This must run before any submodule imports Transformers. It is a no-op on
    # CPU, NVIDIA CUDA, Linux ROCm, and ROCm builds that include c10d.
    install_windows_rocm_transformers_compat(torch)

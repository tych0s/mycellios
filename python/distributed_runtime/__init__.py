"""Low-latency layer-pipeline runtime used by the distribution laboratory."""

import torch

from .rocm_compat import install_windows_rocm_transformers_compat


# This must run before any submodule imports Transformers. It is a no-op on
# CPU, NVIDIA CUDA, Linux ROCm, and ROCm builds that include c10d.
install_windows_rocm_transformers_compat(torch)

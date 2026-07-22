"""Compatibility guards for AMD's official Windows ROCm PyTorch build.

ROCm for Windows intentionally ships without ``torch.distributed`` support.
PyTorch 2.9.1 still exposes enough of the distributed surface for
Transformers 5.x to try importing FSDP, which then fails on the absent
``torch._C._distributed_c10d`` extension.  Mycellios uses its own relay and
stage protocol and never uses FSDP, so expose an explicit no-FSDP adapter
before Transformers is imported.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, MutableMapping


_TRANSFORMERS_FSDP_MODULE = "transformers.distributed.fsdp"
_TRANSFORMERS_SHARDING_MODULE = "transformers.distributed.sharding_utils"


def install_windows_rocm_transformers_compat(
    torch_module: Any,
    *,
    distributed_c10d_available: bool | None = None,
    module_registry: MutableMapping[str, Any] | None = None,
) -> bool:
    """Install a fail-closed FSDP stub only for ROCm builds missing c10d.

    Returns ``True`` when the adapter was installed. CUDA and CPU builds, plus
    ROCm builds that contain the real c10d extension, are left untouched.
    """

    if not getattr(getattr(torch_module, "version", None), "hip", None):
        return False
    if distributed_c10d_available is None:
        try:
            importlib.import_module("torch._C._distributed_c10d")
        except (ImportError, ModuleNotFoundError):
            distributed_c10d_available = False
        else:
            distributed_c10d_available = True
    if distributed_c10d_available:
        return False

    registry = sys.modules if module_registry is None else module_registry
    torch_distributed = getattr(torch_module, "distributed", None)
    if torch_distributed is None and module_registry is None:
        torch_distributed = importlib.import_module("torch.distributed")

    def unavailable(*_args: Any, **_kwargs: Any) -> bool:
        return False

    # The AMD wheel incorrectly advertises distributed availability through
    # ``_c10d_init`` even though its compiled ``_distributed_c10d`` module is
    # absent. Transformers uses this flag to decide whether to import DTensor.
    # Correct the capability report for this process before Transformers loads.
    if torch_distributed is not None:
        torch_distributed.is_available = unavailable

    if (
        _TRANSFORMERS_FSDP_MODULE in registry
        and _TRANSFORMERS_SHARDING_MODULE in registry
    ):
        return False

    adapter = types.ModuleType(_TRANSFORMERS_FSDP_MODULE)

    def verify_fsdp_plan(_module_names: Any, fsdp_plan: Any) -> None:
        if fsdp_plan:
            raise RuntimeError(
                "FSDP is unavailable in the official AMD ROCm runtime for Windows"
            )

    def unsupported(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "FSDP is unavailable in the official AMD ROCm runtime for Windows"
        )

    adapter.is_fsdp_enabled = unavailable
    adapter.is_fsdp_managed_module = unavailable
    adapter.verify_fsdp_plan = verify_fsdp_plan
    adapter.get_fsdp_ckpt_kwargs = lambda: {}
    adapter.update_fsdp_plugin_peft = unsupported
    adapter.apply_fsdp2 = unsupported
    adapter.__all__ = [
        "apply_fsdp2",
        "get_fsdp_ckpt_kwargs",
        "is_fsdp_enabled",
        "is_fsdp_managed_module",
        "update_fsdp_plugin_peft",
        "verify_fsdp_plan",
    ]
    registry[_TRANSFORMERS_FSDP_MODULE] = adapter

    sharding = types.ModuleType(_TRANSFORMERS_SHARDING_MODULE)

    class DtensorShardOperation:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            unsupported()

    sharding.DtensorShardOperation = DtensorShardOperation
    sharding._dtensor_from_local_like = unsupported
    sharding.__all__ = ["DtensorShardOperation", "_dtensor_from_local_like"]
    registry[_TRANSFORMERS_SHARDING_MODULE] = sharding
    return True

from __future__ import annotations

from collections.abc import Sequence


_EXTERNAL_BACKEND_ARGUMENT_PREFIXES = (
    ("--native_stage", "native_stage"),
    ("--external-gguf-runtime", "external GGUF runtime"),
    ("--local-model-runtime", "local-model-runtime"),
    ("--model-serving-runtime", "model-serving-runtime"),
)


def reject_external_backend_arguments(argv: Sequence[str]) -> None:
    """Keep canonical Mycellios processes native even with legacy argv."""

    for argument in argv:
        lowered = argument.lower()
        for prefix, backend in _EXTERNAL_BACKEND_ARGUMENT_PREFIXES:
            if lowered.startswith(prefix):
                raise ValueError(
                    "mycellios_native_runtime_forbids_external_backend:"
                    f"{backend}"
                )


__all__ = ["reject_external_backend_arguments"]

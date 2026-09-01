from __future__ import annotations

from collections.abc import Sequence


_EXTERNAL_BACKEND_ARGUMENT_PREFIXES = (
    ("--nakshatra", "nakshatra"),
    ("--llama-cpp", "llama.cpp"),
    ("--ollama", "ollama"),
    ("--vllm", "vllm"),
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

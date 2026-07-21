from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any
import uuid

import torch


SCHEMA = "gdlp-physical-probe/1"
NONCE_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{16,128}$")


def _sha256(domain: bytes, value: bytes) -> str:
    return "sha256:" + hashlib.sha256(domain + b"\0" + value).hexdigest()


def _machine_material() -> tuple[str, bytes]:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
            if isinstance(value, str) and value.strip():
                return "windows-machine-guid", value.strip().encode("utf-8")
        except (ImportError, OSError):
            pass
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return "linux-machine-id", value.encode("utf-8")
    fallback = f"{platform.node()}|{uuid.getnode():012x}|{platform.machine()}"
    return "host-mac-fallback", fallback.encode("utf-8")


def _version_text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _nccl_version() -> str | None:
    try:
        value = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if isinstance(value, tuple):
        return ".".join(str(item) for item in value)
    return str(value)


def _device_uuid(properties: Any) -> str | None:
    value = getattr(properties, "uuid", None)
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def collect_physical_probe(nonce: str) -> dict[str, Any]:
    if NONCE_PATTERN.fullmatch(nonce) is None:
        raise ValueError("nonce must be 16-128 URL-safe identity characters")
    machine_source, machine_material = _machine_material()
    host_fingerprint = _sha256(b"gdlp-host-fingerprint-v1", machine_material)
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        raw_uuid = _device_uuid(properties)
        try:
            free_bytes, runtime_total_bytes = torch.cuda.mem_get_info(index)
        except (AttributeError, RuntimeError, TypeError):
            free_bytes, runtime_total_bytes = 0, int(properties.total_memory)
        identity_material = (
            f"{host_fingerprint}|{index}|{properties.name}|"
            f"{int(properties.total_memory)}|{raw_uuid or 'no-uuid'}"
        ).encode("utf-8")
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "totalMemoryBytes": int(properties.total_memory),
                "freeMemoryBytes": int(free_bytes),
                "runtimeTotalMemoryBytes": int(runtime_total_bytes),
                "capability": [
                    int(getattr(properties, "major", 0)),
                    int(getattr(properties, "minor", 0)),
                ],
                "uuidSha256": (
                    _sha256(b"gdlp-gpu-uuid-v1", raw_uuid.encode("utf-8"))
                    if raw_uuid is not None
                    else None
                ),
                "fingerprintSha256": _sha256(
                    b"gdlp-gpu-fingerprint-v1", identity_material
                ),
            }
        )
    distributed_available = bool(torch.distributed.is_available())
    nccl_available = bool(
        distributed_available and torch.distributed.is_nccl_available()
    )
    return {
        "schema": SCHEMA,
        "nonce": nonce,
        "host": {
            "fingerprintSha256": host_fingerprint,
            "fingerprintSource": machine_source,
            "platform": platform.system().lower(),
            "architecture": platform.machine().lower(),
            "kernelRelease": platform.release(),
            "pythonVersion": platform.python_version(),
        },
        "runtime": {
            "torchVersion": str(torch.__version__),
            "cudaVersion": _version_text(getattr(torch.version, "cuda", None)),
            "rocmVersion": _version_text(getattr(torch.version, "hip", None)),
            "cudaApiAvailable": cuda_available,
            "distributedAvailable": distributed_available,
            "ncclAvailable": nccl_available,
            "ncclVersion": _nccl_version() if nccl_available else None,
        },
        "devices": devices,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a fixed, read-only GPU/NCCL attestation for a campaign nonce."
    )
    parser.add_argument("--nonce", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            collect_physical_probe(args.nonce),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

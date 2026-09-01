from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .llama_cpp import (
    CommandResult,
    GGUFArtifact,
    LlamaBenchmarkEvidence,
    LlamaCppDevice,
    LlamaCppRuntimeProbe,
    benchmark_llama_cpp,
    build_whole_model_deployment,
    inspect_gguf,
    parse_llama_cpp_deployment,
    probe_llama_cpp,
)


LLAMA_CPP_RPC_WORKER_SCHEMA = "gdlp-llama-cpp-rpc-worker/1"
LLAMA_CPP_RPC_CELL_SCHEMA = "gdlp-llama-cpp-rpc-cell/1"
LLAMA_CPP_RPC_TOPOLOGY_SCHEMA = "gdlp-llama-cpp-rpc-topology/2"
RPC_GRAPH_SCOPE = "whole-model-graph-split"
RPC_NETWORK_SCOPE = "private-lan-or-loopback"
RPC_TOPOLOGY_EVIDENCE_SCOPE = "point-in-time-live-observation"
RPC_TOPOLOGY_EXECUTION_VALIDATION = "reprobe-runtime-and-selected-device-identity"
_PROCESS_OUTPUT_TAIL_CHARS = 64 * 1024

_REQUIRED_SERVER_FLAGS = (
    "--cache",
    "--device",
    "--help",
    "--host",
    "--port",
    "--threads",
)
_DEVICE_RE = re.compile(
    r"(?m)^\s*(?P<id>[A-Za-z][A-Za-z0-9_-]*[0-9]+):\s*"
    r"(?P<name>.+?)\s*\((?P<total>[0-9]+) MiB,\s*"
    r"(?P<free>[0-9]+) MiB free\)\s*$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CommandRunner = Callable[[Sequence[str], Path, float], CommandResult]
PopenFactory = Callable[..., Any]


@dataclass(frozen=True)
class LlamaCppRpcRuntimeProbe:
    """Content-sealed stock llama.cpp runtime with its RPC components.

    The build is still a whole-model llama.cpp executor.  Presence of the RPC
    backend means that llama.cpp can place graph tensors on remote devices; it
    never promotes the stock build to GDLP's partial Stage Executor ABI.
    """

    runtime: LlamaCppRuntimeProbe
    rpc_runtime_id: str
    server_file_name: str
    server_sha256: str
    library_file_name: str
    library_sha256: str
    server_help_sha256: str
    server_flags: tuple[str, ...]

    @property
    def engine_version(self) -> str:
        return self.runtime.engine_version

    def to_document(self) -> dict[str, Any]:
        return {
            "rpcRuntimeId": self.rpc_runtime_id,
            "llamaRuntimeId": self.runtime.runtime_id,
            "engineVersion": self.engine_version,
            "buildNumber": self.runtime.build_number,
            "buildCommit": self.runtime.build_commit,
            "compiler": self.runtime.compiler,
            "binarySetSha256": self.runtime.binary_set_sha256,
            "rpcServer": {
                "fileName": self.server_file_name,
                "sha256": self.server_sha256,
            },
            "rpcLibrary": {
                "fileName": self.library_file_name,
                "sha256": self.library_sha256,
            },
            "serverHelpSha256": self.server_help_sha256,
            "serverFlags": list(self.server_flags),
            "capabilities": {
                "wholeModelGraphSplit": True,
                "nativePartialStageAbi": False,
                "partialLayerStage": False,
                "networkScope": RPC_NETWORK_SCOPE,
            },
        }


@dataclass(frozen=True)
class LlamaCppRpcWorkerSpec:
    bind_host: str
    advertise_host: str
    port: int
    devices: tuple[str, ...]
    threads: int = 1
    cache: bool = False

    def __post_init__(self) -> None:
        _lan_ip(self.bind_host, "RPC bind host")
        _lan_ip(self.advertise_host, "RPC advertise host")
        _port(self.port, "RPC worker port")
        if (
            not isinstance(self.devices, tuple)
            or not self.devices
            or len(set(self.devices)) != len(self.devices)
            or any(not isinstance(value, str) or not value.strip() for value in self.devices)
        ):
            raise ValueError("RPC worker devices must be a non-empty unique tuple")
        if not isinstance(self.threads, int) or isinstance(self.threads, bool) or self.threads < 1:
            raise ValueError("RPC worker threads must be a positive integer")
        if not isinstance(self.cache, bool):
            raise ValueError("RPC worker cache flag must be boolean")

    @property
    def endpoint(self) -> str:
        return _endpoint(self.advertise_host, self.port)


@dataclass(frozen=True)
class LlamaCppRpcWorkerManifest:
    document_json: str

    @property
    def worker_id(self) -> str:
        return str(self.to_document()["workerId"])

    @property
    def endpoint(self) -> str:
        return str(self.to_document()["network"]["endpoint"])

    @property
    def runtime_id(self) -> str:
        return str(self.to_document()["runtime"]["rpcRuntimeId"])

    def to_document(self) -> dict[str, Any]:
        return json.loads(self.document_json)


@dataclass(frozen=True)
class LlamaCppRpcTopologyProbe:
    probe_id: str
    runtime_id: str
    endpoints: tuple[str, ...]
    devices: tuple[LlamaCppDevice, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": LLAMA_CPP_RPC_TOPOLOGY_SCHEMA,
            "probeId": self.probe_id,
            "runtimeId": self.runtime_id,
            "endpoints": list(self.endpoints),
            "devices": [device.to_document() for device in self.devices],
            "observedVia": "llama-bench --rpc --list-devices",
            "evidence": {
                "scope": RPC_TOPOLOGY_EVIDENCE_SCOPE,
                "freeMemoryMiB": "admission-snapshot-only",
                "executionValidation": RPC_TOPOLOGY_EXECUTION_VALIDATION,
            },
        }


@dataclass(frozen=True)
class LlamaCppRpcCellManifest:
    document_json: str

    @property
    def cell_id(self) -> str:
        return str(self.to_document()["cellId"])

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(self.to_document()["client"]["rpcEndpoints"])

    @property
    def rpc_devices(self) -> tuple[str, ...]:
        return tuple(self.to_document()["client"]["rpcDevices"])

    def to_document(self) -> dict[str, Any]:
        return json.loads(self.document_json)


class _ProcessOutputDrain:
    """Continuously consume one child pipe while retaining only a bounded tail."""

    def __init__(self, stream: Any, *, limit_chars: int) -> None:
        self.stream = stream
        self.limit_chars = limit_chars
        self._tail = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                chunk = self.stream.read(4096)
                if not chunk:
                    return
                if not isinstance(chunk, str):
                    chunk = bytes(chunk).decode("utf-8", errors="replace")
                with self._lock:
                    self._tail = (self._tail + chunk)[-self.limit_chars :]
        except (OSError, ValueError):
            # Closing a pipe during process teardown may interrupt the reader.
            return

    @property
    def tail(self) -> str:
        with self._lock:
            return self._tail

    def close(self, *, timeout_seconds: float) -> None:
        self._thread.join(timeout=timeout_seconds)
        try:
            self.stream.close()
        except (OSError, ValueError):
            pass
        self._thread.join(timeout=min(timeout_seconds, 0.25))


@dataclass
class LaunchedLlamaCppRpcProcess:
    command: tuple[str, ...]
    process: Any
    _stdout_drain: _ProcessOutputDrain | None = field(init=False, default=None, repr=False)
    _stderr_drain: _ProcessOutputDrain | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        stdout = getattr(self.process, "stdout", None)
        stderr = getattr(self.process, "stderr", None)
        if stdout is not None:
            self._stdout_drain = _ProcessOutputDrain(
                stdout, limit_chars=_PROCESS_OUTPUT_TAIL_CHARS
            )
        if stderr is not None:
            self._stderr_drain = _ProcessOutputDrain(
                stderr, limit_chars=_PROCESS_OUTPUT_TAIL_CHARS
            )

    @property
    def stdout_tail(self) -> str:
        return "" if self._stdout_drain is None else self._stdout_drain.tail

    @property
    def stderr_tail(self) -> str:
        return "" if self._stderr_drain is None else self._stderr_drain.tail

    def _close_output_handles(self, timeout_seconds: float) -> None:
        for drain in (self._stdout_drain, self._stderr_drain):
            if drain is not None:
                drain.close(timeout_seconds=timeout_seconds)

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        timeout = _positive_float(timeout_seconds, "RPC process close timeout")
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=timeout)
        finally:
            # This also runs for a process that had already exited.  Leaving
            # Popen's pipe handles open otherwise leaks descriptors and emits
            # ResourceWarning during repeated worker churn.
            self._close_output_handles(timeout)


def probe_llama_cpp_rpc(
    runtime_directory: str | os.PathLike[str],
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
) -> LlamaCppRpcRuntimeProbe:
    """Inspect the client, RPC worker, dynamic backend and device inventory."""

    root = Path(runtime_directory).resolve()
    command_runner = runner or _default_command_runner
    runtime = probe_llama_cpp(
        root,
        runner=command_runner,
        timeout_seconds=timeout_seconds,
    )
    if not runtime.rpc_backend_present:
        raise ValueError("llama.cpp runtime did not expose the RPC backend")
    server = _required_executable(root, "ggml-rpc-server")
    library = _required_rpc_library(root)
    result = _run_checked(
        command_runner,
        (str(server), "--help"),
        root,
        timeout_seconds,
    )
    help_text = _normalize_text(result.stdout + "\n" + result.stderr)
    missing = [flag for flag in _REQUIRED_SERVER_FLAGS if flag not in help_text]
    if missing:
        raise ValueError(
            "llama.cpp RPC server help omits required flags: " + ", ".join(missing)
        )
    server_sha256 = _sha256_file(server)
    library_sha256 = _sha256_file(library)
    server_help_sha256 = hashlib.sha256(help_text.encode("utf-8")).hexdigest()
    body = {
        "llamaRuntimeId": runtime.runtime_id,
        "engineVersion": runtime.engine_version,
        "rpcServer": {"fileName": server.name, "sha256": server_sha256},
        "rpcLibrary": {"fileName": library.name, "sha256": library_sha256},
        "serverHelpSha256": server_help_sha256,
        "serverFlags": list(_REQUIRED_SERVER_FLAGS),
        "graphScope": RPC_GRAPH_SCOPE,
        "nativePartialStageAbi": False,
    }
    return LlamaCppRpcRuntimeProbe(
        runtime=runtime,
        rpc_runtime_id=hashlib.sha256(_canonical_json(body)).hexdigest()[:32],
        server_file_name=server.name,
        server_sha256=server_sha256,
        library_file_name=library.name,
        library_sha256=library_sha256,
        server_help_sha256=server_help_sha256,
        server_flags=_REQUIRED_SERVER_FLAGS,
    )


def build_rpc_worker_manifest(
    runtime: LlamaCppRpcRuntimeProbe,
    spec: LlamaCppRpcWorkerSpec,
) -> LlamaCppRpcWorkerManifest:
    available = {device.identifier: device for device in runtime.runtime.devices}
    unknown = [identifier for identifier in spec.devices if identifier not in available]
    if unknown:
        raise ValueError(
            "RPC worker requested devices absent from the runtime probe: "
            + ", ".join(unknown)
        )
    selected = [_static_device_document(available[value]) for value in spec.devices]
    arguments = _worker_arguments(spec)
    without_id = {
        "schema": LLAMA_CPP_RPC_WORKER_SCHEMA,
        "runtime": runtime.to_document(),
        "network": {
            "scope": RPC_NETWORK_SCOPE,
            "bindHost": str(ipaddress.ip_address(spec.bind_host)),
            "advertiseHost": str(ipaddress.ip_address(spec.advertise_host)),
            "port": spec.port,
            "endpoint": spec.endpoint,
            "publicAddressAccepted": False,
        },
        "execution": {
            "role": "llama.cpp-rpc-tensor-worker",
            "graphScope": RPC_GRAPH_SCOPE,
            "nativePartialStageAbi": False,
            "partialLayerStage": False,
            "devices": selected,
            "threads": spec.threads,
            "cache": spec.cache,
        },
        "launch": {
            "binaryFile": runtime.server_file_name,
            "arguments": list(arguments),
            "shell": False,
        },
    }
    document = {
        **without_id,
        "workerId": hashlib.sha256(_canonical_json(without_id)).hexdigest()[:32],
    }
    return parse_rpc_worker_manifest(document)


def parse_rpc_worker_manifest(value: object) -> LlamaCppRpcWorkerManifest:
    document = _exact_mapping(
        value,
        ("schema", "workerId", "runtime", "network", "execution", "launch"),
        "llama.cpp RPC worker manifest",
    )
    if document.get("schema") != LLAMA_CPP_RPC_WORKER_SCHEMA:
        raise ValueError("unsupported llama.cpp RPC worker schema")
    worker_id = _digest(document.get("workerId"), 32, "RPC worker id")
    expected_id = hashlib.sha256(
        _canonical_json(_without(document, "workerId"))
    ).hexdigest()[:32]
    if worker_id != expected_id:
        raise ValueError("RPC worker identity does not match its manifest")
    runtime = _parse_rpc_runtime_document(document.get("runtime"))
    network = _exact_mapping(
        document.get("network"),
        (
            "scope",
            "bindHost",
            "advertiseHost",
            "port",
            "endpoint",
            "publicAddressAccepted",
        ),
        "RPC worker network",
    )
    if network.get("scope") != RPC_NETWORK_SCOPE:
        raise ValueError("RPC worker must be restricted to a private LAN or loopback")
    bind_host = _lan_ip(network.get("bindHost"), "RPC bind host")
    advertise_host = _lan_ip(network.get("advertiseHost"), "RPC advertise host")
    port = _port(network.get("port"), "RPC worker port")
    if network.get("endpoint") != _endpoint(advertise_host, port):
        raise ValueError("RPC worker endpoint does not match its advertised address")
    if network.get("publicAddressAccepted") is not False:
        raise ValueError("RPC worker cannot accept public addressing")
    execution = _exact_mapping(
        document.get("execution"),
        (
            "role",
            "graphScope",
            "nativePartialStageAbi",
            "partialLayerStage",
            "devices",
            "threads",
            "cache",
        ),
        "RPC worker execution",
    )
    if execution.get("role") != "llama.cpp-rpc-tensor-worker":
        raise ValueError("RPC worker role is unsupported")
    _validate_graph_boundary(execution)
    devices = _parse_static_devices(execution.get("devices"))
    threads = _positive_int(execution.get("threads"), "RPC worker threads")
    cache = execution.get("cache")
    if not isinstance(cache, bool):
        raise ValueError("RPC worker cache flag must be boolean")
    spec = LlamaCppRpcWorkerSpec(
        bind_host=bind_host,
        advertise_host=advertise_host,
        port=port,
        devices=tuple(device["id"] for device in devices),
        threads=threads,
        cache=cache,
    )
    launch = _exact_mapping(
        document.get("launch"),
        ("binaryFile", "arguments", "shell"),
        "RPC worker launch",
    )
    if launch.get("binaryFile") != runtime["serverFileName"]:
        raise ValueError("RPC worker launch binary does not match its runtime")
    if launch.get("shell") is not False:
        raise ValueError("RPC worker launch must disable the command shell")
    if _string_tuple(launch.get("arguments"), "RPC worker arguments") != _worker_arguments(spec):
        raise ValueError("RPC worker argv does not match its sealed configuration")
    return LlamaCppRpcWorkerManifest(_canonical_json(document).decode("utf-8"))


def launch_rpc_worker(
    runtime_directory: str | os.PathLike[str],
    manifest_value: LlamaCppRpcWorkerManifest | Mapping[str, Any],
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
    popen_factory: PopenFactory | None = None,
) -> LaunchedLlamaCppRpcProcess:
    """Re-probe the local runtime and launch exactly one sealed RPC worker."""

    manifest = (
        manifest_value
        if isinstance(manifest_value, LlamaCppRpcWorkerManifest)
        else parse_rpc_worker_manifest(manifest_value)
    )
    root = Path(runtime_directory).resolve()
    actual = probe_llama_cpp_rpc(
        root,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    document = manifest.to_document()
    expected_runtime = _parse_rpc_runtime_document(document["runtime"])
    _match_rpc_runtime(actual, expected_runtime)
    expected_devices = {
        value["id"]: value for value in _parse_static_devices(document["execution"]["devices"])
    }
    available = {value.identifier: value for value in actual.runtime.devices}
    for identifier, expected in expected_devices.items():
        current = available.get(identifier)
        if current is None:
            raise RuntimeError(f"sealed RPC device is no longer available: {identifier}")
        if _static_device_document(current) != expected:
            raise RuntimeError(f"sealed RPC device changed since inspection: {identifier}")
    binary = _required_named_file(root, actual.server_file_name)
    arguments = tuple(str(value) for value in document["launch"]["arguments"])
    command = (str(binary), *arguments)
    factory = popen_factory or subprocess.Popen
    process = factory(
        list(command),
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return LaunchedLlamaCppRpcProcess(command=command, process=process)


def wait_rpc_worker_ready(
    launched: LaunchedLlamaCppRpcProcess,
    endpoint: str,
    *,
    timeout_seconds: float = 20.0,
) -> None:
    host, port = _parse_endpoint(endpoint)
    deadline = time.monotonic() + _positive_float(timeout_seconds, "RPC startup timeout")
    while time.monotonic() < deadline:
        returncode = launched.process.poll()
        if returncode is not None:
            # Let the background drain consume the final EOF before reporting
            # the bounded diagnostic tail.
            launched._close_output_handles(min(timeout_seconds, 0.5))
            stderr = launched.stderr_tail[-2000:]
            raise RuntimeError(
                f"llama.cpp RPC worker exited with code {returncode}: {stderr.strip()}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"llama.cpp RPC worker was not ready at {endpoint}")


def probe_rpc_topology(
    runtime_directory: str | os.PathLike[str],
    runtime: LlamaCppRpcRuntimeProbe,
    endpoints: Sequence[str],
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
) -> LlamaCppRpcTopologyProbe:
    normalized_endpoints = _endpoint_tuple(endpoints)
    root = Path(runtime_directory).resolve()
    bench = _required_executable(root, "llama-bench")
    result = _run_checked(
        runner or _default_command_runner,
        (
            str(bench),
            "--rpc",
            ",".join(normalized_endpoints),
            "--list-devices",
        ),
        root,
        timeout_seconds,
    )
    devices = _parse_rpc_devices(result.stdout, result.stderr)
    body = {
        "runtimeId": runtime.rpc_runtime_id,
        "endpoints": list(normalized_endpoints),
        "devices": [value.to_document() for value in devices],
        "observedVia": "llama-bench --rpc --list-devices",
        "evidence": {
            "scope": RPC_TOPOLOGY_EVIDENCE_SCOPE,
            "freeMemoryMiB": "admission-snapshot-only",
            "executionValidation": RPC_TOPOLOGY_EXECUTION_VALIDATION,
        },
    }
    return LlamaCppRpcTopologyProbe(
        probe_id=hashlib.sha256(_canonical_json(body)).hexdigest()[:32],
        runtime_id=runtime.rpc_runtime_id,
        endpoints=normalized_endpoints,
        devices=devices,
    )


def parse_rpc_topology_probe(value: object) -> LlamaCppRpcTopologyProbe:
    document = _exact_mapping(
        value,
        (
            "schema",
            "probeId",
            "runtimeId",
            "endpoints",
            "devices",
            "observedVia",
            "evidence",
        ),
        "llama.cpp RPC topology probe",
    )
    if document.get("schema") != LLAMA_CPP_RPC_TOPOLOGY_SCHEMA:
        raise ValueError("unsupported llama.cpp RPC topology schema")
    if document.get("observedVia") != "llama-bench --rpc --list-devices":
        raise ValueError("RPC topology observation method is unsupported")
    probe_id = _digest(document.get("probeId"), 32, "RPC topology id")
    runtime_id = _digest(document.get("runtimeId"), 32, "RPC topology runtime id")
    endpoints = _endpoint_tuple(document.get("endpoints"))
    devices = _parse_device_documents(document.get("devices"), rpc_only=True)
    evidence = _exact_mapping(
        document.get("evidence"),
        ("scope", "freeMemoryMiB", "executionValidation"),
        "RPC topology evidence semantics",
    )
    if evidence.get("scope") != RPC_TOPOLOGY_EVIDENCE_SCOPE:
        raise ValueError("RPC topology evidence must be a point-in-time observation")
    if evidence.get("freeMemoryMiB") != "admission-snapshot-only":
        raise ValueError("RPC topology free-memory evidence must remain a snapshot")
    if evidence.get("executionValidation") != RPC_TOPOLOGY_EXECUTION_VALIDATION:
        raise ValueError("RPC topology must require execution-time revalidation")
    body = {
        "runtimeId": runtime_id,
        "endpoints": list(endpoints),
        "devices": [value.to_document() for value in devices],
        "observedVia": document.get("observedVia"),
        "evidence": dict(evidence),
    }
    expected = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    if probe_id != expected:
        raise ValueError("RPC topology identity does not match its evidence")
    return LlamaCppRpcTopologyProbe(probe_id, runtime_id, endpoints, devices)


def certify_gguf_support(
    runtime_directory: str | os.PathLike[str],
    runtime: LlamaCppRpcRuntimeProbe,
    artifact: GGUFArtifact,
    artifact_path: str | os.PathLike[str],
    *,
    prompt_tokens: int = 1,
    generation_tokens: int = 1,
    device: str | None = None,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 180.0,
) -> LlamaBenchmarkEvidence:
    """Execute the GGUF before it can be admitted to an RPC cell manifest.

    This is intentionally stronger than an architecture-name allow-list.  A
    successful stock llama.cpp benchmark demonstrates that this exact build,
    model digest, quantization and architecture can be loaded and executed.
    """

    return benchmark_llama_cpp(
        runtime_directory,
        runtime.runtime,
        artifact,
        artifact_path,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        repetitions=1,
        gpu_layers=-1,
        device=device,
        no_warmup=True,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )


def build_rpc_cell_manifest(
    client_runtime: LlamaCppRpcRuntimeProbe,
    artifact: GGUFArtifact,
    support_evidence: LlamaBenchmarkEvidence,
    workers: Sequence[LlamaCppRpcWorkerManifest | Mapping[str, Any]],
    topology: LlamaCppRpcTopologyProbe | Mapping[str, Any],
    *,
    split_mode: str = "layer",
    gpu_layers: int = -1,
    rpc_devices: Sequence[str] | None = None,
) -> LlamaCppRpcCellManifest:
    """Seal one stock llama.cpp whole-model graph split over LAN RPC workers."""

    if artifact.storage_layout != "single-file":
        raise ValueError("RPC execution requires one complete GGUF model artifact")
    whole_model = build_whole_model_deployment(
        client_runtime.runtime,
        artifact,
        benchmark=support_evidence,
    )
    parsed_workers = tuple(
        item if isinstance(item, LlamaCppRpcWorkerManifest) else parse_rpc_worker_manifest(item)
        for item in workers
    )
    if not parsed_workers:
        raise ValueError("RPC cell requires at least one worker")
    endpoints = tuple(item.endpoint for item in parsed_workers)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("RPC cell worker endpoints must be unique")
    for worker in parsed_workers:
        worker_runtime = _parse_rpc_runtime_document(worker.to_document()["runtime"])
        if (
            worker_runtime["buildNumber"] != client_runtime.runtime.build_number
            or worker_runtime["buildCommit"] != client_runtime.runtime.build_commit
        ):
            raise ValueError("RPC workers and client must use the exact same llama.cpp build")
    topology_value = (
        topology
        if isinstance(topology, LlamaCppRpcTopologyProbe)
        else parse_rpc_topology_probe(topology)
    )
    if topology_value.runtime_id != client_runtime.rpc_runtime_id:
        raise ValueError("RPC topology belongs to another client runtime")
    if topology_value.endpoints != endpoints:
        raise ValueError("RPC topology endpoints do not match the ordered workers")
    selected_rpc_devices = (
        tuple(value.identifier for value in topology_value.devices)
        if rpc_devices is None
        else _string_tuple(rpc_devices, "RPC client devices")
    )
    available_rpc_devices = {value.identifier for value in topology_value.devices}
    if not selected_rpc_devices or not set(selected_rpc_devices).issubset(available_rpc_devices):
        raise ValueError("RPC client requested a device absent from the live topology probe")
    if split_mode not in ("layer", "row", "tensor"):
        raise ValueError("RPC split mode must be layer, row or tensor")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError("RPC gpu layers must be -1 or a non-negative integer")
    without_id = {
        "schema": LLAMA_CPP_RPC_CELL_SCHEMA,
        "clientRuntime": client_runtime.to_document(),
        "modelDeployment": whole_model.to_document(),
        "workers": [item.to_document() for item in parsed_workers],
        "topology": topology_value.to_document(),
        "execution": {
            "mode": "llama.cpp-rpc-whole-model",
            "graphScope": RPC_GRAPH_SCOPE,
            "splitMode": split_mode,
            "gpuLayers": gpu_layers,
            "nativePartialStageAbi": False,
            "partialLayerStage": False,
            "stageExecutor": None,
        },
        "client": {
            "rpcEndpoints": list(endpoints),
            "rpcDevices": list(selected_rpc_devices),
        },
    }
    document = {
        **without_id,
        "cellId": hashlib.sha256(_canonical_json(without_id)).hexdigest()[:32],
    }
    return parse_rpc_cell_manifest(document)


def parse_rpc_cell_manifest(value: object) -> LlamaCppRpcCellManifest:
    document = _exact_mapping(
        value,
        (
            "schema",
            "cellId",
            "clientRuntime",
            "modelDeployment",
            "workers",
            "topology",
            "execution",
            "client",
        ),
        "llama.cpp RPC cell manifest",
    )
    if document.get("schema") != LLAMA_CPP_RPC_CELL_SCHEMA:
        raise ValueError("unsupported llama.cpp RPC cell schema")
    cell_id = _digest(document.get("cellId"), 32, "RPC cell id")
    expected = hashlib.sha256(_canonical_json(_without(document, "cellId"))).hexdigest()[:32]
    if cell_id != expected:
        raise ValueError("RPC cell identity does not match its manifest")
    client_runtime = _parse_rpc_runtime_document(document.get("clientRuntime"))
    model_deployment = parse_llama_cpp_deployment(document.get("modelDeployment"))
    if model_deployment.mode != "whole-model" or model_deployment.stage_executor is not None:
        raise ValueError("RPC cell requires a whole-model llama.cpp deployment")
    worker_values = document.get("workers")
    if not isinstance(worker_values, Sequence) or isinstance(
        worker_values, (str, bytes, bytearray)
    ) or not worker_values:
        raise ValueError("RPC cell workers must be a non-empty list")
    workers = tuple(parse_rpc_worker_manifest(value) for value in worker_values)
    endpoints = tuple(value.endpoint for value in workers)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("RPC cell worker endpoints must be unique")
    for worker in workers:
        worker_runtime = _parse_rpc_runtime_document(worker.to_document()["runtime"])
        if (
            worker_runtime["buildNumber"] != client_runtime["buildNumber"]
            or worker_runtime["buildCommit"] != client_runtime["buildCommit"]
        ):
            raise ValueError("RPC worker build does not match the client build")
    topology = parse_rpc_topology_probe(document.get("topology"))
    if topology.runtime_id != client_runtime["rpcRuntimeId"] or topology.endpoints != endpoints:
        raise ValueError("RPC topology does not match the client runtime and workers")
    execution = _exact_mapping(
        document.get("execution"),
        (
            "mode",
            "graphScope",
            "splitMode",
            "gpuLayers",
            "nativePartialStageAbi",
            "partialLayerStage",
            "stageExecutor",
        ),
        "RPC cell execution",
    )
    if execution.get("mode") != "llama.cpp-rpc-whole-model":
        raise ValueError("RPC cell mode is unsupported")
    _validate_graph_boundary(execution)
    if execution.get("stageExecutor") is not None:
        raise ValueError("stock llama.cpp RPC cannot advertise a Stage Executor")
    if execution.get("splitMode") not in ("layer", "row", "tensor"):
        raise ValueError("RPC cell split mode is unsupported")
    gpu_layers = execution.get("gpuLayers")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError("RPC cell gpuLayers is invalid")
    client = _exact_mapping(
        document.get("client"),
        ("rpcEndpoints", "rpcDevices"),
        "RPC cell client",
    )
    client_endpoints = _endpoint_tuple(client.get("rpcEndpoints"))
    if client_endpoints != endpoints:
        raise ValueError("RPC client endpoints do not match the workers")
    rpc_devices = _string_tuple(client.get("rpcDevices"), "RPC client devices")
    available = {value.identifier for value in topology.devices}
    if not set(rpc_devices).issubset(available):
        raise ValueError("RPC client devices are absent from topology evidence")
    return LlamaCppRpcCellManifest(_canonical_json(document).decode("utf-8"))


def build_rpc_benchmark_argv(
    runtime_directory: str | os.PathLike[str],
    manifest_value: LlamaCppRpcCellManifest | Mapping[str, Any],
    model_path: str | os.PathLike[str],
    *,
    prompt_tokens: int = 32,
    generation_tokens: int = 8,
    repetitions: int = 1,
    no_warmup: bool = True,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
) -> tuple[str, ...]:
    manifest, root, model = _validate_client_inputs(
        runtime_directory,
        manifest_value,
        model_path,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    prompt_tokens = _positive_int(prompt_tokens, "RPC benchmark prompt tokens")
    generation_tokens = _positive_int(
        generation_tokens, "RPC benchmark generation tokens"
    )
    repetitions = _positive_int(repetitions, "RPC benchmark repetitions")
    document = manifest.to_document()
    execution = document["execution"]
    client = document["client"]
    command = [
        str(_required_executable(root, "llama-bench")),
        "-m",
        str(model),
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-r",
        str(repetitions),
        "-ngl",
        str(execution["gpuLayers"]),
        "-rpc",
        ",".join(client["rpcEndpoints"]),
        "-dev",
        "/".join(client["rpcDevices"]),
        "-sm",
        execution["splitMode"],
        "-o",
        "json",
    ]
    if no_warmup:
        command.append("--no-warmup")
    return tuple(command)


def build_rpc_completion_argv(
    runtime_directory: str | os.PathLike[str],
    manifest_value: LlamaCppRpcCellManifest | Mapping[str, Any],
    model_path: str | os.PathLike[str],
    *,
    prompt: str,
    predict: int = 8,
    seed: int = 1,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
) -> tuple[str, ...]:
    manifest, root, model = _validate_client_inputs(
        runtime_directory,
        manifest_value,
        model_path,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("RPC completion prompt cannot be empty")
    predict = _positive_int(predict, "RPC completion token count")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("RPC completion seed must be a non-negative integer")
    document = manifest.to_document()
    execution = document["execution"]
    client = document["client"]
    return (
        str(_required_executable(root, "llama-cli")),
        "-m",
        str(model),
        "--rpc",
        ",".join(client["rpcEndpoints"]),
        "--device",
        ",".join(client["rpcDevices"]),
        "--split-mode",
        execution["splitMode"],
        "--n-gpu-layers",
        str(execution["gpuLayers"]),
        "--prompt",
        prompt,
        "--predict",
        str(predict),
        "--seed",
        str(seed),
        "--temp",
        "0",
        "--no-display-prompt",
        "--simple-io",
        "--no-conversation",
        "--no-warmup",
    )


def launch_rpc_client(
    arguments: Sequence[str],
    *,
    cwd: str | os.PathLike[str],
    manifest_value: LlamaCppRpcCellManifest | Mapping[str, Any],
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
    popen_factory: PopenFactory | None = None,
) -> LaunchedLlamaCppRpcProcess:
    if isinstance(arguments, (str, bytes, bytearray)) or not arguments:
        raise ValueError("RPC client command must be a non-empty argument vector")
    command = tuple(str(value) for value in arguments)
    if any(not value for value in command):
        raise ValueError("RPC client argv cannot contain empty arguments")
    root = Path(cwd).resolve()
    manifest = (
        manifest_value
        if isinstance(manifest_value, LlamaCppRpcCellManifest)
        else parse_rpc_cell_manifest(manifest_value)
    )
    _validate_rpc_client_command(command, root, manifest)
    _reprobe_client_execution_state(
        root,
        manifest,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    factory = popen_factory or subprocess.Popen
    process = factory(
        list(command),
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return LaunchedLlamaCppRpcProcess(command, process)


def _validate_client_inputs(
    runtime_directory: str | os.PathLike[str],
    manifest_value: LlamaCppRpcCellManifest | Mapping[str, Any],
    model_path: str | os.PathLike[str],
    *,
    runner: CommandRunner | None,
    timeout_seconds: float,
) -> tuple[LlamaCppRpcCellManifest, Path, Path]:
    manifest = (
        manifest_value
        if isinstance(manifest_value, LlamaCppRpcCellManifest)
        else parse_rpc_cell_manifest(manifest_value)
    )
    root = Path(runtime_directory).resolve()
    model = Path(model_path).resolve()
    artifact = inspect_gguf(model)
    expected = manifest.to_document()["modelDeployment"]["artifact"]["identity"]
    if artifact.identity != expected:
        raise ValueError("RPC client GGUF does not match the sealed model deployment")
    _reprobe_client_execution_state(
        root,
        manifest,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    return manifest, root, model


def _reprobe_client_execution_state(
    root: Path,
    manifest: LlamaCppRpcCellManifest,
    *,
    runner: CommandRunner | None,
    timeout_seconds: float,
) -> None:
    """Fail closed if sealed binaries or selected RPC devices changed.

    The topology manifest is admission evidence captured at one point in time.
    Its free-memory value is deliberately not a lease.  Immediately before an
    execution command is returned or launched, this function re-hashes the
    entire local llama.cpp binary set plus the RPC server/library, then performs
    a fresh endpoint probe and matches stable identity for every selected RPC
    device.  Current free memory may legitimately differ from the snapshot.
    """

    document = manifest.to_document()
    expected_runtime = _parse_rpc_runtime_document(document["clientRuntime"])
    actual_runtime = probe_llama_cpp_rpc(
        root,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    _match_rpc_runtime(actual_runtime, expected_runtime, sealed_role="client")

    sealed_topology = parse_rpc_topology_probe(document["topology"])
    live_topology = probe_rpc_topology(
        root,
        actual_runtime,
        document["client"]["rpcEndpoints"],
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    _match_rpc_topology_for_execution(
        live_topology,
        sealed_topology,
        tuple(document["client"]["rpcDevices"]),
    )


def _match_rpc_topology_for_execution(
    actual: LlamaCppRpcTopologyProbe,
    expected: LlamaCppRpcTopologyProbe,
    selected_devices: tuple[str, ...],
) -> None:
    if actual.runtime_id != expected.runtime_id:
        raise RuntimeError("live RPC topology belongs to another client runtime")
    if actual.endpoints != expected.endpoints:
        raise RuntimeError("live RPC topology endpoints differ from sealed evidence")
    expected_devices = {value.identifier: value for value in expected.devices}
    actual_devices = {value.identifier: value for value in actual.devices}
    for identifier in selected_devices:
        expected_device = expected_devices.get(identifier)
        actual_device = actual_devices.get(identifier)
        if expected_device is None or actual_device is None:
            raise RuntimeError(
                f"sealed RPC device is absent from the live topology: {identifier}"
            )
        if _rpc_device_execution_identity(actual_device) != _rpc_device_execution_identity(
            expected_device
        ):
            raise RuntimeError(
                f"RPC device identity changed since topology admission: {identifier}"
            )


def _rpc_device_execution_identity(device: LlamaCppDevice) -> tuple[object, ...]:
    return (
        device.identifier,
        device.backend,
        device.name,
        device.total_memory_mib,
        device.properties,
    )


def _validate_rpc_client_command(
    command: tuple[str, ...],
    root: Path,
    manifest: LlamaCppRpcCellManifest,
) -> None:
    binary = Path(command[0]).resolve()
    allowed = {
        _required_executable(root, "llama-bench"),
        _required_executable(root, "llama-cli"),
    }
    if binary not in allowed or binary.parent != root:
        raise ValueError("RPC client command must use a sealed runtime executable")

    document = manifest.to_document()
    client = document["client"]
    execution = document["execution"]
    benchmark = binary.name.lower().startswith("llama-bench")
    canonical_values = _validate_canonical_rpc_client_argv(command, benchmark=benchmark)
    endpoint_flag = "-rpc" if benchmark else "--rpc"
    device_flag = "-dev" if benchmark else "--device"
    split_flag = "-sm" if benchmark else "--split-mode"
    gpu_layers_flag = "-ngl" if benchmark else "--n-gpu-layers"
    expected_devices = ("/" if benchmark else ",").join(client["rpcDevices"])
    required = {
        endpoint_flag: ",".join(client["rpcEndpoints"]),
        device_flag: expected_devices,
        split_flag: execution["splitMode"],
        gpu_layers_flag: str(execution["gpuLayers"]),
    }
    for canonical, expected in required.items():
        actual = canonical_values[canonical]
        if actual != expected:
            raise ValueError(
                f"RPC client command {canonical} differs from the sealed cell"
            )

    model_value = canonical_values["-m"]
    model_path = Path(model_value)
    artifact = inspect_gguf(
        (model_path if model_path.is_absolute() else root / model_path).resolve()
    )
    expected_artifact = document["modelDeployment"]["artifact"]["identity"]
    if artifact.identity != expected_artifact:
        raise ValueError("RPC client command model differs from the sealed deployment")


def _validate_canonical_rpc_client_argv(
    command: tuple[str, ...],
    *,
    benchmark: bool,
) -> dict[str, str]:
    """Accept only the exact argument layout emitted by this module's builders.

    llama.cpp has multiple aliases for several placement options (including
    ``-ngl``, ``--gpu-layers`` and ``--n-gpu-layers``), and parsers may accept
    ``--flag=value``.  An appended alias can override a previously validated
    value.  Exact layout validation rejects every duplicate, alternate alias,
    inline spelling and unrelated option instead of maintaining a brittle
    deny-list that could lag a future llama.cpp release.
    """

    if benchmark:
        value_flags = (
            (1, "-m"),
            (3, "-p"),
            (5, "-n"),
            (7, "-r"),
            (9, "-ngl"),
            (11, "-rpc"),
            (13, "-dev"),
            (15, "-sm"),
            (17, "-o"),
        )
        if len(command) not in (19, 20):
            raise ValueError("RPC client command is not canonical sealed benchmark argv")
        if len(command) == 20 and command[19] != "--no-warmup":
            raise ValueError("RPC client command is not canonical sealed benchmark argv")
    else:
        value_flags = (
            (1, "-m"),
            (3, "--rpc"),
            (5, "--device"),
            (7, "--split-mode"),
            (9, "--n-gpu-layers"),
            (11, "--prompt"),
            (13, "--predict"),
            (15, "--seed"),
            (17, "--temp"),
        )
        trailing = (
            "--no-display-prompt",
            "--simple-io",
            "--no-conversation",
            "--no-warmup",
        )
        if len(command) != 23 or command[19:] != trailing:
            raise ValueError("RPC client command is not canonical sealed completion argv")

    values: dict[str, str] = {}
    for index, flag in value_flags:
        if command[index] != flag:
            raise ValueError(
                f"RPC client command uses non-canonical sealed option at {flag}"
            )
        value = command[index + 1]
        if not value:
            raise ValueError(f"RPC client command {flag} has no value")
        values[flag] = value
    return values


def _worker_arguments(spec: LlamaCppRpcWorkerSpec) -> tuple[str, ...]:
    arguments = [
        "--host",
        str(ipaddress.ip_address(spec.bind_host)),
        "--port",
        str(spec.port),
        "--device",
        ",".join(spec.devices),
        "--threads",
        str(spec.threads),
    ]
    if spec.cache:
        arguments.append("--cache")
    return tuple(arguments)


def _parse_rpc_runtime_document(value: object) -> dict[str, Any]:
    document = _exact_mapping(
        value,
        (
            "rpcRuntimeId",
            "llamaRuntimeId",
            "engineVersion",
            "buildNumber",
            "buildCommit",
            "compiler",
            "binarySetSha256",
            "rpcServer",
            "rpcLibrary",
            "serverHelpSha256",
            "serverFlags",
            "capabilities",
        ),
        "llama.cpp RPC runtime",
    )
    rpc_runtime_id = _digest(document.get("rpcRuntimeId"), 32, "RPC runtime id")
    llama_runtime_id = _digest(document.get("llamaRuntimeId"), 32, "llama runtime id")
    build_number = _positive_int(document.get("buildNumber"), "llama.cpp build number")
    build_commit = _hex_text(document.get("buildCommit"), "llama.cpp build commit")
    if document.get("engineVersion") != f"b{build_number}-{build_commit}":
        raise ValueError("RPC runtime engine version does not match its build")
    compiler = _nonempty_string(document.get("compiler"), "llama.cpp compiler")
    binary_set_sha256 = _digest(
        document.get("binarySetSha256"), 64, "llama.cpp binary-set SHA-256"
    )
    server = _artifact_file(document.get("rpcServer"), "RPC server")
    library = _artifact_file(document.get("rpcLibrary"), "RPC library")
    help_sha256 = _digest(document.get("serverHelpSha256"), 64, "RPC help SHA-256")
    flags = _string_tuple(document.get("serverFlags"), "RPC server flags")
    if flags != _REQUIRED_SERVER_FLAGS:
        raise ValueError("RPC runtime server flags do not match the supported contract")
    capabilities = _exact_mapping(
        document.get("capabilities"),
        (
            "wholeModelGraphSplit",
            "nativePartialStageAbi",
            "partialLayerStage",
            "networkScope",
        ),
        "RPC runtime capabilities",
    )
    if capabilities.get("wholeModelGraphSplit") is not True:
        raise ValueError("RPC runtime must advertise whole-model graph splitting")
    if capabilities.get("nativePartialStageAbi") is not False:
        raise ValueError("stock llama.cpp RPC has no native partial Stage ABI")
    if capabilities.get("partialLayerStage") is not False:
        raise ValueError("stock llama.cpp RPC is not a partial layer stage")
    if capabilities.get("networkScope") != RPC_NETWORK_SCOPE:
        raise ValueError("RPC runtime network scope is unsupported")
    body = {
        "llamaRuntimeId": llama_runtime_id,
        "engineVersion": document.get("engineVersion"),
        "rpcServer": server,
        "rpcLibrary": library,
        "serverHelpSha256": help_sha256,
        "serverFlags": list(flags),
        "graphScope": RPC_GRAPH_SCOPE,
        "nativePartialStageAbi": False,
    }
    expected = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    if rpc_runtime_id != expected:
        raise ValueError("RPC runtime identity does not match its binaries")
    return {
        "rpcRuntimeId": rpc_runtime_id,
        "llamaRuntimeId": llama_runtime_id,
        "engineVersion": document.get("engineVersion"),
        "buildNumber": build_number,
        "buildCommit": build_commit,
        "compiler": compiler,
        "binarySetSha256": binary_set_sha256,
        "serverFileName": server["fileName"],
        "serverSha256": server["sha256"],
        "libraryFileName": library["fileName"],
        "librarySha256": library["sha256"],
        "serverHelpSha256": help_sha256,
    }


def _match_rpc_runtime(
    actual: LlamaCppRpcRuntimeProbe,
    expected: Mapping[str, Any],
    *,
    sealed_role: str = "worker",
) -> None:
    comparisons = {
        "rpcRuntimeId": actual.rpc_runtime_id,
        "llamaRuntimeId": actual.runtime.runtime_id,
        "engineVersion": actual.engine_version,
        "buildNumber": actual.runtime.build_number,
        "buildCommit": actual.runtime.build_commit,
        "compiler": actual.runtime.compiler,
        "binarySetSha256": actual.runtime.binary_set_sha256,
        "serverFileName": actual.server_file_name,
        "serverSha256": actual.server_sha256,
        "libraryFileName": actual.library_file_name,
        "librarySha256": actual.library_sha256,
        "serverHelpSha256": actual.server_help_sha256,
    }
    mismatch = [name for name, current in comparisons.items() if expected[name] != current]
    if mismatch:
        raise RuntimeError(
            f"local llama.cpp RPC runtime differs from the sealed {sealed_role}: "
            + ", ".join(mismatch)
        )


def _validate_graph_boundary(value: Mapping[str, Any]) -> None:
    if value.get("graphScope") != RPC_GRAPH_SCOPE:
        raise ValueError("llama.cpp RPC must remain a whole-model graph split")
    if value.get("nativePartialStageAbi") is not False:
        raise ValueError("stock llama.cpp RPC has no native partial Stage ABI")
    if value.get("partialLayerStage") is not False:
        raise ValueError("stock llama.cpp RPC is not a partial layer stage")


def _parse_rpc_devices(stdout: str, stderr: str) -> tuple[LlamaCppDevice, ...]:
    text = stdout + "\n" + stderr
    devices: list[LlamaCppDevice] = []
    seen: set[str] = set()
    for match in _DEVICE_RE.finditer(text):
        identifier = match.group("id")
        if not identifier.upper().startswith("RPC") or identifier in seen:
            continue
        total = int(match.group("total"))
        free = int(match.group("free"))
        if total < 1 or free < 0 or free > total:
            raise ValueError("RPC device memory report is invalid")
        seen.add(identifier)
        devices.append(
            LlamaCppDevice(
                identifier=identifier,
                backend="rpc",
                name=match.group("name").strip(),
                total_memory_mib=total,
                free_memory_mib=free,
            )
        )
    if not devices:
        raise RuntimeError("llama.cpp client reported no RPC devices")
    return tuple(devices)


def _parse_device_documents(value: object, *, rpc_only: bool) -> tuple[LlamaCppDevice, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("RPC topology devices must be a list")
    result: list[LlamaCppDevice] = []
    identifiers: set[str] = set()
    for item in value:
        document = _exact_mapping(
            item,
            ("id", "backend", "name", "totalMemoryMiB", "freeMemoryMiB", "properties"),
            "RPC topology device",
        )
        identifier = _nonempty_string(document.get("id"), "RPC topology device id")
        if rpc_only and not identifier.upper().startswith("RPC"):
            raise ValueError("RPC topology contains a non-RPC device")
        if identifier in identifiers:
            raise ValueError("RPC topology device ids must be unique")
        identifiers.add(identifier)
        backend = _nonempty_string(document.get("backend"), "RPC topology backend")
        if rpc_only and backend != "rpc":
            raise ValueError("RPC topology device backend must be rpc")
        total = _positive_int(document.get("totalMemoryMiB"), "RPC device total memory")
        free = _nonnegative_int(document.get("freeMemoryMiB"), "RPC device free memory")
        if free > total:
            raise ValueError("RPC device free memory exceeds total memory")
        properties_value = _mapping(document.get("properties"), "RPC device properties")
        properties: list[tuple[str, bool | int | str]] = []
        for key, property_value in properties_value.items():
            key_value = _nonempty_string(key, "RPC device property key")
            if not isinstance(property_value, (bool, int, str)):
                raise ValueError("RPC device property has an unsupported value")
            properties.append((key_value, property_value))
        result.append(
            LlamaCppDevice(
                identifier,
                backend,
                _nonempty_string(document.get("name"), "RPC topology device name"),
                total,
                free,
                tuple(sorted(properties)),
            )
        )
    if not result:
        raise ValueError("RPC topology must contain at least one device")
    return tuple(result)


def _static_device_document(device: LlamaCppDevice) -> dict[str, Any]:
    return {
        "id": device.identifier,
        "backend": device.backend,
        "name": device.name,
        "totalMemoryMiB": device.total_memory_mib,
        "properties": {key: value for key, value in device.properties},
    }


def _parse_static_devices(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("RPC worker devices must be a list")
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for item in value:
        document = _exact_mapping(
            item,
            ("id", "backend", "name", "totalMemoryMiB", "properties"),
            "RPC worker device",
        )
        identifier = _nonempty_string(document.get("id"), "RPC worker device id")
        if identifier in identifiers:
            raise ValueError("RPC worker device ids must be unique")
        identifiers.add(identifier)
        properties = _mapping(document.get("properties"), "RPC worker device properties")
        normalized_properties: dict[str, bool | int | str] = {}
        for key, property_value in properties.items():
            key_value = _nonempty_string(key, "RPC worker device property key")
            if not isinstance(property_value, (bool, int, str)):
                raise ValueError("RPC worker device property is unsupported")
            normalized_properties[key_value] = property_value
        result.append(
            {
                "id": identifier,
                "backend": _nonempty_string(document.get("backend"), "RPC worker backend"),
                "name": _nonempty_string(document.get("name"), "RPC worker device name"),
                "totalMemoryMiB": _positive_int(
                    document.get("totalMemoryMiB"), "RPC worker total memory"
                ),
                "properties": normalized_properties,
            }
        )
    if not result:
        raise ValueError("RPC worker requires at least one device")
    return tuple(result)


def _endpoint_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("RPC endpoints must be a list")
    endpoints: list[str] = []
    for item in value:
        host, port = _parse_endpoint(item)
        endpoints.append(_endpoint(host, port))
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("RPC endpoints must be a non-empty unique list")
    return tuple(endpoints)


def _parse_endpoint(value: object) -> tuple[str, int]:
    text = _nonempty_string(value, "RPC endpoint")
    if text.startswith("["):
        closing = text.find("]")
        if closing < 1 or closing + 1 >= len(text) or text[closing + 1] != ":":
            raise ValueError("RPC IPv6 endpoint must use [address]:port")
        host = text[1:closing]
        port_text = text[closing + 2 :]
    else:
        if text.count(":") != 1:
            raise ValueError("RPC endpoint must use address:port")
        host, port_text = text.rsplit(":", 1)
    if not port_text.isdigit():
        raise ValueError("RPC endpoint port must be numeric")
    return _lan_ip(host, "RPC endpoint host"), _port(int(port_text), "RPC endpoint port")


def _endpoint(host: str, port: int) -> str:
    address = ipaddress.ip_address(host)
    rendered = f"[{address}]" if address.version == 6 else str(address)
    return f"{rendered}:{port}"


def _lan_ip(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    try:
        address = ipaddress.ip_address(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a literal private-LAN or loopback IP") from error
    if address.is_unspecified or address.is_multicast:
        raise ValueError(f"{name} cannot be wildcard or multicast")
    if address.version == 4:
        allowed = address.is_loopback or address.is_link_local or any(
            address in network
            for network in (
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("172.16.0.0/12"),
                ipaddress.ip_network("192.168.0.0/16"),
            )
        )
    else:
        allowed = (
            address.is_loopback
            or address.is_link_local
            or address in ipaddress.ip_network("fc00::/7")
        )
    if not allowed:
        raise ValueError(f"{name} must remain on loopback or a private LAN")
    return str(address)


def _artifact_file(value: object, name: str) -> dict[str, str]:
    document = _exact_mapping(value, ("fileName", "sha256"), name)
    filename = _nonempty_string(document.get("fileName"), f"{name} filename")
    if Path(filename).name != filename:
        raise ValueError(f"{name} filename must not contain a path")
    return {
        "fileName": filename,
        "sha256": _digest(document.get("sha256"), 64, f"{name} SHA-256"),
    }


def _required_rpc_library(root: Path) -> Path:
    for name in ("ggml-rpc.dll", "libggml-rpc.so", "libggml-rpc.dylib"):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("missing llama.cpp RPC dynamic library")


def _required_executable(root: Path, stem: str) -> Path:
    for candidate in (root / f"{stem}.exe", root / stem):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing llama.cpp executable: {stem}")


def _required_named_file(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("runtime binary filename is not portable")
    path = (root / name).resolve()
    if not path.is_file() or path.parent != root:
        raise FileNotFoundError(path)
    return path


def _default_command_runner(
    arguments: Sequence[str], cwd: Path, timeout_seconds: float
) -> CommandResult:
    completed = subprocess.run(
        [str(value) for value in arguments],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _run_checked(
    runner: CommandRunner,
    arguments: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
) -> CommandResult:
    timeout = _positive_float(timeout_seconds, "command timeout")
    try:
        result = runner(tuple(arguments), cwd, timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"llama.cpp command failed to run: {arguments[0]}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(
            f"llama.cpp command exited with code {result.returncode}: {detail}"
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalize_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").split("\n")).strip()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_mapping(value: object, fields: Sequence[str], name: str) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != set(fields):
        raise ValueError(f"{name} has unknown or missing fields")
    return result


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a string list")
    result = tuple(_nonempty_string(item, name) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be a non-empty unique string list")
    return result


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _port(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def _digest(value: object, length: int, name: str) -> str:
    text = _nonempty_string(value, name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return text


def _hex_text(value: object, name: str) -> str:
    text = _nonempty_string(value, name).lower()
    if any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be hexadecimal")
    return text


__all__ = [
    "LLAMA_CPP_RPC_CELL_SCHEMA",
    "LLAMA_CPP_RPC_TOPOLOGY_SCHEMA",
    "LLAMA_CPP_RPC_WORKER_SCHEMA",
    "RPC_GRAPH_SCOPE",
    "RPC_NETWORK_SCOPE",
    "RPC_TOPOLOGY_EVIDENCE_SCOPE",
    "RPC_TOPOLOGY_EXECUTION_VALIDATION",
    "LaunchedLlamaCppRpcProcess",
    "LlamaCppRpcCellManifest",
    "LlamaCppRpcRuntimeProbe",
    "LlamaCppRpcTopologyProbe",
    "LlamaCppRpcWorkerManifest",
    "LlamaCppRpcWorkerSpec",
    "build_rpc_benchmark_argv",
    "build_rpc_cell_manifest",
    "build_rpc_completion_argv",
    "build_rpc_worker_manifest",
    "certify_gguf_support",
    "launch_rpc_client",
    "launch_rpc_worker",
    "parse_rpc_cell_manifest",
    "parse_rpc_topology_probe",
    "parse_rpc_worker_manifest",
    "probe_llama_cpp_rpc",
    "probe_rpc_topology",
    "wait_rpc_worker_ready",
]

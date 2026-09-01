from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import re
import struct
import subprocess
import sys
import threading
from typing import BinaryIO

import numpy as np
import torch

from .executor_abi import build_stage_executor_manifest
from .model import StageModelSpec
from .nakshatra_package import (
    NAKSHATRA_COMMIT,
    NAKSHATRA_LLAMA_CPP_COMMIT,
    NakshatraStagePackage,
    load_nakshatra_stage_package,
)


CMD_EMBD_DECODE = 2
CMD_INFO = 3
CMD_KV_TRUNCATE = 4

FLAG_KEEP_KV = 0x1
FLAG_ALL_LOGITS = 0x2

RESULT_HIDDEN = 0
RESULT_TOKEN = 1
RESULT_TOKENS = 2

_REQUEST_HEADER = struct.Struct("<IIIII")
_RESPONSE_HEADER = struct.Struct("<II")
_INFO_PAYLOAD = struct.Struct("<6i")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_VERSION_SHA = re.compile(r"(?m)^\s*sha\s+([0-9a-f]{40})\s*$")


class NakshatraDaemonError(RuntimeError):
    pass


@dataclass(frozen=True)
class NakshatraDaemonInfo:
    layer_start: int
    layer_end: int
    hidden_size: int
    has_token_embeddings: bool
    has_lm_head: bool
    vocab_size: int


@dataclass(frozen=True)
class NakshatraStageRuntimeSpec:
    package: str
    daemon_command: tuple[str, ...]
    context_tokens: int
    threads: int = 0
    gpu_layers: int = 0
    compute_api: str = "cpu"
    startup_timeout_seconds: float = 120.0
    call_timeout_seconds: float = 120.0
    close_timeout_seconds: float = 5.0
    expected_pipeline_id: int | None = None
    expected_package_id: str | None = None
    expected_manifest_sha256: str | None = None
    stderr_lines: int = 200

    def __post_init__(self) -> None:
        if not isinstance(self.package, str) or not self.package.strip():
            raise ValueError("Nakshatra package cannot be empty")
        if (
            not isinstance(self.daemon_command, tuple)
            or not self.daemon_command
            or any(not isinstance(value, str) or not value for value in self.daemon_command)
        ):
            raise ValueError("daemon_command must be a non-empty argv tuple")
        for name, value in (
            ("context_tokens", self.context_tokens),
            ("stderr_lines", self.stderr_lines),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("threads", self.threads), ("gpu_layers", self.gpu_layers)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.compute_api not in ("cpu", "cuda", "rocm", "metal", "vulkan"):
            raise ValueError("compute_api must be cpu, cuda, rocm, metal or vulkan")
        if self.expected_pipeline_id is not None:
            _request_id(self.expected_pipeline_id)
        for name, value in (
            ("startup_timeout_seconds", self.startup_timeout_seconds),
            ("call_timeout_seconds", self.call_timeout_seconds),
            ("close_timeout_seconds", self.close_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _DaemonResponse:
    status: int
    payload: bytes


@dataclass(frozen=True)
class _ReaderFailure:
    error: BaseException


class NakshatraStdioClient:
    """Long-lived client for the pinned Nakshatra little-endian daemon ABI."""

    def __init__(
        self,
        runtime: NakshatraStageRuntimeSpec,
        package: NakshatraStagePackage,
    ) -> None:
        max_decode_bytes = 4 + runtime.context_tokens * package.hidden_size * 4
        if max_decode_bytes > (1 << 31):
            raise ValueError("Nakshatra maximum response would exceed 2 GiB")
        self._runtime = runtime
        self._max_response_bytes = max(max_decode_bytes, _INFO_PAYLOAD.size)
        self._lock = threading.Lock()
        self._responses: queue.Queue[_DaemonResponse | _ReaderFailure] = queue.Queue()
        self._stderr = deque(maxlen=runtime.stderr_lines)
        self._closed = False
        self._broken: BaseException | None = None
        self.binary_commit = _probe_daemon_version(
            runtime.daemon_command,
            timeout_seconds=min(runtime.startup_timeout_seconds, 30.0),
        )
        command = [
            *runtime.daemon_command,
            str(package.artifact_path),
            package.mode,
            str(runtime.context_tokens),
            str(runtime.threads),
            str(runtime.gpu_layers),
        ]
        self.command = tuple(command)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            shell=False,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self._terminate_process()
            raise RuntimeError("failed to create Nakshatra daemon pipes")
        self._stdout_thread = threading.Thread(
            target=self._read_responses,
            name="gdlp-nakshatra-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="gdlp-nakshatra-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self.live_info = self.info(timeout_seconds=runtime.startup_timeout_seconds)
        except BaseException:
            self.close()
            raise

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr)

    def info(self, *, timeout_seconds: float | None = None) -> NakshatraDaemonInfo:
        payload = self._call(
            CMD_INFO,
            n_tokens=0,
            start_pos=0,
            flags=0,
            payload=b"",
            timeout_seconds=timeout_seconds,
        )
        if len(payload) != _INFO_PAYLOAD.size:
            raise NakshatraDaemonError(
                f"INFO returned {len(payload)} bytes, expected {_INFO_PAYLOAD.size}"
            )
        layer_start, layer_end, hidden_size, has_embd, has_lm, vocab_size = (
            _INFO_PAYLOAD.unpack(payload)
        )
        if hidden_size < 1 or vocab_size < 1 or layer_start < 0 or layer_end < layer_start:
            raise NakshatraDaemonError("INFO returned invalid model dimensions")
        if has_embd not in (0, 1) or has_lm not in (0, 1):
            raise NakshatraDaemonError("INFO returned invalid boolean flags")
        return NakshatraDaemonInfo(
            layer_start=layer_start,
            layer_end=layer_end,
            hidden_size=hidden_size,
            has_token_embeddings=bool(has_embd),
            has_lm_head=bool(has_lm),
            vocab_size=vocab_size,
        )

    def decode_embeddings(
        self,
        payload: bytes,
        *,
        n_tokens: int,
        start_pos: int,
        keep_kv: bool,
        all_logits: bool,
    ) -> tuple[int, bytes]:
        expected = n_tokens * self.live_info.hidden_size * 4
        if len(payload) != expected:
            raise ValueError(
                f"embedding payload has {len(payload)} bytes, expected {expected}"
            )
        flags = (FLAG_KEEP_KV if keep_kv else 0) | (
            FLAG_ALL_LOGITS if all_logits else 0
        )
        response = self._call(
            CMD_EMBD_DECODE,
            n_tokens=n_tokens,
            start_pos=start_pos,
            flags=flags,
            payload=payload,
        )
        if len(response) < 4:
            raise NakshatraDaemonError("decode response omits result_type")
        result_type = _U32.unpack_from(response)[0]
        if result_type not in (RESULT_HIDDEN, RESULT_TOKEN, RESULT_TOKENS):
            raise NakshatraDaemonError(f"unsupported Nakshatra result_type {result_type}")
        return result_type, response[4:]

    def truncate(self, token_count: int) -> None:
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        response = self._call(
            CMD_KV_TRUNCATE,
            n_tokens=0,
            start_pos=0,
            flags=0,
            payload=_U32.pack(token_count),
        )
        if response:
            raise NakshatraDaemonError("KV_TRUNCATE returned an unexpected payload")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stdin = self.process.stdin
            if stdin is not None:
                try:
                    stdin.close()
                except OSError:
                    pass
        try:
            self.process.wait(timeout=self._runtime.close_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._terminate_process()
        self._stdout_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _call(
        self,
        command: int,
        *,
        n_tokens: int,
        start_pos: int,
        flags: int,
        payload: bytes,
        timeout_seconds: float | None = None,
    ) -> bytes:
        for name, value in (
            ("command", command),
            ("n_tokens", n_tokens),
            ("start_pos", start_pos),
            ("flags", flags),
            ("payload length", len(payload)),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"{name} must fit an unsigned 32-bit integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("Nakshatra daemon client is closed")
            if self._broken is not None:
                raise NakshatraDaemonError("Nakshatra daemon client is broken") from self._broken
            if self.process.poll() is not None:
                raise NakshatraDaemonError(
                    f"Nakshatra daemon exited with code {self.process.returncode}: "
                    f"{self._stderr_summary()}"
                )
            stdin = self.process.stdin
            if stdin is None:
                raise NakshatraDaemonError("Nakshatra daemon stdin is unavailable")
            try:
                stdin.write(
                    _REQUEST_HEADER.pack(
                        command, n_tokens, start_pos, flags, len(payload)
                    )
                    + payload
                )
                stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self._broken = error
                raise NakshatraDaemonError(
                    f"failed to write to Nakshatra daemon: {self._stderr_summary()}"
                ) from error
            timeout = (
                self._runtime.call_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
            try:
                response = self._responses.get(timeout=timeout)
            except queue.Empty as error:
                timeout_error = TimeoutError(
                    f"Nakshatra daemon command {command} timed out after {timeout:.3f}s"
                )
                self._broken = timeout_error
                self._terminate_process()
                raise timeout_error from error
            if isinstance(response, _ReaderFailure):
                self._broken = response.error
                raise NakshatraDaemonError(
                    f"Nakshatra daemon response stream failed: {self._stderr_summary()}"
                ) from response.error
            if response.status != 0:
                names = {1: "decode error", 2: "wire format error", 3: "unsupported architecture"}
                raise NakshatraDaemonError(
                    f"Nakshatra daemon returned status {response.status} "
                    f"({names.get(response.status, 'unknown error')})"
                )
            return response.payload

    def _read_responses(self) -> None:
        stdout = self.process.stdout
        if stdout is None:
            self._responses.put(_ReaderFailure(RuntimeError("stdout is unavailable")))
            return
        try:
            while True:
                header = _read_exact(stdout, _RESPONSE_HEADER.size)
                status, payload_bytes = _RESPONSE_HEADER.unpack(header)
                if payload_bytes > self._max_response_bytes:
                    raise NakshatraDaemonError(
                        f"daemon response length {payload_bytes} exceeds sealed maximum "
                        f"{self._max_response_bytes}"
                    )
                payload = _read_exact(stdout, payload_bytes)
                self._responses.put(_DaemonResponse(status, payload))
        except BaseException as error:
            if not self._closed:
                self._responses.put(_ReaderFailure(error))

    def _drain_stderr(self) -> None:
        stderr = self.process.stderr
        if stderr is None:
            return
        try:
            for line in iter(stderr.readline, b""):
                self._stderr.append(line.decode("utf-8", errors="replace").rstrip())
        except OSError:
            pass

    def _stderr_summary(self) -> str:
        return " | ".join(tuple(self._stderr)[-5:]) or "no daemon stderr"

    def _terminate_process(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)


class NakshatraStageRunner:
    """StageRunnerContract adapter around one persistent Nakshatra daemon.

    The pinned daemon has one KV sequence and serializes all stdio calls.  This
    runner therefore rejects overlapping requests instead of pretending that
    request ids map to independent caches. Token batches within that one
    request (prefill and speculative verify) remain supported up to ``n_ctx``.
    """

    max_active_requests = 1

    def __init__(self, spec: StageModelSpec, runtime: NakshatraStageRuntimeSpec) -> None:
        package = load_nakshatra_stage_package(
            runtime.package,
            expected_package_id=runtime.expected_package_id,
            expected_manifest_sha256=runtime.expected_manifest_sha256,
        )
        if (
            runtime.expected_pipeline_id is not None
            and runtime.expected_pipeline_id != package.pipeline_id
        ):
            raise ValueError(
                "Nakshatra package pipeline id does not match the launch contract"
            )
        _validate_stage_matches_package(spec, package)
        if runtime.context_tokens > package.max_context_tokens:
            raise ValueError(
                f"context {runtime.context_tokens} exceeds sealed package limit "
                f"{package.max_context_tokens}"
            )
        self.spec = spec
        self.runtime = runtime
        self.package = package
        self.pipeline_id = package.pipeline_id
        self.hidden_size = package.hidden_size
        self.parameter_bytes = package.artifact_bytes
        self.loader = "nakshatra-sub-gguf-stdio"
        device_kinds = ("cpu",) if runtime.gpu_layers == 0 else ("gpu", "cpu")
        self.executor_manifest = build_stage_executor_manifest(
            engine="nakshatra-llama.cpp",
            engine_version=(
                f"nakshatra-{NAKSHATRA_COMMIT}@llama.cpp-"
                f"{NAKSHATRA_LLAMA_CPP_COMMIT}"
            ),
            adapter="nakshatra-stdio-little-endian",
            model_identity=package.model_identity,
            model_source=package.model_source,
            model_revision=package.model_revision,
            artifact_format="sub-gguf-nakshatra",
            layer_start=spec.layer_start,
            layer_end=spec.layer_end,
            total_layers=spec.total_layers,
            hidden_size=package.hidden_size,
            activation_dtype="float32",
            activation_codecs=(
                "fp32",
                "fp16",
                "int8",
                "int8-grouped",
                "int8-hadamard",
            ),
            kv_format="llama.cpp-seq0",
            max_batch_size=1,
            max_context_tokens=runtime.context_tokens,
            device_kinds=device_kinds,
            compute_apis=("llama.cpp", runtime.compute_api),
            weight_dtypes=(package.weight_type,),
            features=(
                "layer-range",
                "rank-local-kv",
                "rollback",
                "persistent-process",
                "sub-gguf",
                "greedy-verify",
                "single-kv-sequence",
                "llama-only",
                # ``executor_id`` is the recovery compatibility boundary and
                # hashes every feature.  The semantic fields above are not
                # sufficient for Nakshatra: two sub-GGUFs can advertise the
                # same model/range/dimensions while containing different
                # weights.  Seal the verified package and its exact bytes so
                # such a standby cannot impersonate the failed executor.
                f"nakshatra-package-id:{package.package_id}",
                f"nakshatra-manifest-sha256:{package.manifest_sha256}",
                f"nakshatra-artifact-sha256:{package.artifact_sha256}",
                f"nakshatra-model-content-sha256:{package.model_content_sha256}",
            ),
        )
        self._active_request: int | None = None
        self._tokens_seen = 0
        self._closed = False
        self._poisoned: BaseException | None = None
        self.client = NakshatraStdioClient(runtime, package)
        live = self.client.live_info
        # At the pinned revision INFO hardcodes the range/endpoint flags. The
        # sealed package is authoritative for those fields; dimensions are live
        # values and can be checked without relying on that upstream limitation.
        if live.hidden_size != package.hidden_size:
            self.close()
            raise ValueError(
                f"daemon hidden size {live.hidden_size} != package {package.hidden_size}"
            )
        if live.vocab_size != package.vocab_size:
            self.close()
            raise ValueError(
                f"daemon vocabulary {live.vocab_size} != package {package.vocab_size}"
            )

    def begin(self, request_id: int) -> None:
        self._require_usable()
        _request_id(request_id)
        if self._active_request is not None:
            raise RuntimeError(
                "the pinned Nakshatra daemon supports one active request/KV sequence"
            )
        self._active_request = request_id
        self._tokens_seen = 0

    def end(self, request_id: int) -> None:
        self._require_active(request_id)
        self._active_request = None
        self._tokens_seen = 0

    def truncate(self, request_id: int, token_count: int) -> None:
        self._require_active(request_id)
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        if not 0 <= token_count <= self._tokens_seen:
            raise ValueError(
                f"cannot truncate request {request_id} from {self._tokens_seen} "
                f"to {token_count} tokens"
            )
        try:
            self.client.truncate(token_count)
        except BaseException as error:
            self._poisoned = error
            raise
        self._tokens_seen = token_count

    def sequence_length(self, request_id: int) -> int:
        self._require_active(request_id)
        return self._tokens_seen

    @torch.inference_mode()
    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        self._require_active(request_id)
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] < 1
            or hidden.shape[2] != self.hidden_size
        ):
            raise ValueError(
                f"hidden state must have shape [1, tokens, {self.hidden_size}]"
            )
        if not hidden.is_floating_point():
            raise TypeError("hidden state must be floating point")
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        n_tokens = int(hidden.shape[1])
        if self._tokens_seen + n_tokens > self.runtime.context_tokens:
            raise ValueError("Nakshatra request exceeds the configured context")
        input_array = (
            hidden.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )
        try:
            result_type, payload = self.client.decode_embeddings(
                input_array.tobytes(order="C"),
                n_tokens=n_tokens,
                start_pos=self._tokens_seen,
                keep_kv=self._tokens_seen > 0,
                all_logits=token_mode == "all",
            )
            output, token = self._decode_result(
                result_type,
                payload,
                input_array,
                n_tokens=n_tokens,
                token_mode=token_mode,
            )
        except BaseException as error:
            # A failed/malformed response can arrive after llama_decode changed
            # seq0. Refuse to continue with an unknowable cache state.
            self._poisoned = error
            raise
        self._tokens_seen += n_tokens
        return output, token

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._active_request = None
        self._tokens_seen = 0
        client = getattr(self, "client", None)
        if client is not None:
            client.close()

    def _decode_result(
        self,
        result_type: int,
        payload: bytes,
        input_array: np.ndarray,
        *,
        n_tokens: int,
        token_mode: str,
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        if not self.spec.last:
            if result_type != RESULT_HIDDEN:
                raise NakshatraDaemonError(
                    "non-final Nakshatra stage did not return hidden states"
                )
            expected = n_tokens * self.hidden_size * 4
            if len(payload) != expected:
                raise NakshatraDaemonError(
                    f"hidden result has {len(payload)} bytes, expected {expected}"
                )
            array = np.frombuffer(payload, dtype="<f4").astype(
                np.float32, copy=True
            ).reshape(1, n_tokens, self.hidden_size)
            return torch.from_numpy(array), None

        # Nakshatra's final daemon returns logits-derived token ids, not its
        # final hidden activation. The physical GDLP last-stage path never
        # forwards that tensor, so a shape-correct copy of the input satisfies
        # the shared runner tuple without claiming it is a model output.
        placeholder = torch.from_numpy(input_array.astype(np.float32, copy=True))
        if token_mode == "all":
            if result_type != RESULT_TOKENS or len(payload) != n_tokens * 4:
                raise NakshatraDaemonError("verify result is not one token per position")
            tokens = struct.unpack(f"<{n_tokens}i", payload)
            return placeholder, tuple(int(token) for token in tokens)
        if result_type != RESULT_TOKEN or len(payload) != _I32.size:
            raise NakshatraDaemonError("final stage did not return one token")
        token = _I32.unpack(payload)[0]
        return placeholder, None if token_mode == "none" else int(token)

    def _require_active(self, request_id: int) -> None:
        self._require_usable()
        _request_id(request_id)
        if self._active_request != request_id:
            raise ValueError(f"request {request_id} has not received BEGIN")

    def _require_usable(self) -> None:
        if self._closed:
            raise RuntimeError("Nakshatra stage runner is closed")
        if self._poisoned is not None:
            raise RuntimeError("Nakshatra stage runner has an uncertain KV state") from self._poisoned


def _validate_stage_matches_package(
    spec: StageModelSpec, package: NakshatraStagePackage
) -> None:
    if spec.first:
        raise ValueError(
            "Nakshatra child-stage adapter cannot execute layer_start=0; "
            "the current root path requires token-ID/embedding support"
        )
    if (
        spec.model_name != package.model_source
        or spec.revision != package.model_revision
        or spec.layer_start != package.layer_start
        or spec.layer_end != package.layer_end
        or spec.total_layers != package.total_layers
    ):
        raise ValueError("StageModelSpec does not match the sealed Nakshatra package")


def _probe_daemon_version(
    command: tuple[str, ...], *, timeout_seconds: float
) -> str:
    try:
        completed = subprocess.run(
            [*command, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("failed to probe Nakshatra daemon version") from error
    output = completed.stdout + "\n" + completed.stderr
    match = _VERSION_SHA.search(output)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(
            "Nakshatra daemon lacks the required compile-time commit stamp"
        )
    commit = match.group(1)
    if commit != NAKSHATRA_COMMIT:
        raise RuntimeError(
            f"Nakshatra daemon commit {commit} != required {NAKSHATRA_COMMIT}"
        )
    return commit


def _request_id(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= (1 << 64) - 1
    ):
        raise ValueError("request_id must be an unsigned 64-bit integer")
    return value


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        chunk = stream.read(size - offset)
        if not chunk:
            raise EOFError(
                f"Nakshatra daemon closed stdout after {offset}/{size} bytes"
            )
        view[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return bytes(chunks)


__all__ = [
    "CMD_EMBD_DECODE",
    "CMD_INFO",
    "CMD_KV_TRUNCATE",
    "FLAG_ALL_LOGITS",
    "FLAG_KEEP_KV",
    "NakshatraDaemonError",
    "NakshatraDaemonInfo",
    "NakshatraStageRunner",
    "NakshatraStageRuntimeSpec",
    "NakshatraStdioClient",
    "RESULT_HIDDEN",
    "RESULT_TOKEN",
    "RESULT_TOKENS",
]

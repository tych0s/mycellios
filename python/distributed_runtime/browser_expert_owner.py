"""Bridge exact MoE expert execution to a foreground mycellios browser worker.

The browser owns only stateless SwiGLU experts.  The authoritative router,
attention state, KV cache, and weighted reduction remain in the existing
runtime.  Weight identity is proven by SHA-256 and a numerical canary before
the coordinator advertises browser residency.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F

from .ram_expert_cache import ExpertKey
from .resident_expert_mesh import (
    ExpertResidentSlotUnavailableError,
    OwnerExpertBatchItem,
    OwnerExpertBatchResult,
)


class BrowserExpertOwnerError(RuntimeError):
    """Coordinator or browser rejected an exact expert operation."""


class BrowserExpertOwner:
    """Synchronous ``ExpertOwner`` backed by any eligible mobile PWA worker."""

    supports_exact_input_coalescing = False
    coalesced_row_index_bytes_per_assignment = 4

    def __init__(
        self,
        node_id: str,
        coordinator_url: str,
        *,
        internal_token: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id cannot be empty")
        if not isinstance(coordinator_url, str) or not coordinator_url.strip():
            raise ValueError("coordinator_url cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.node_id = node_id.strip()
        self.coordinator_url = coordinator_url.rstrip("/")
        self._internal_token = _load_internal_token(internal_token)
        self.timeout_seconds = float(timeout_seconds)
        self._artifacts: dict[tuple[ExpertKey, str], str] = {}

    def attach_registered_expert(
        self,
        key: ExpertKey,
        content_id: str,
        artifact_id: str,
    ) -> None:
        """Attach an artifact previously published to this coordinator."""

        if not isinstance(key, ExpertKey):
            raise TypeError("key must be ExpertKey")
        if not isinstance(content_id, str) or not content_id.strip():
            raise ValueError("content_id cannot be empty")
        if (
            not isinstance(artifact_id, str)
            or len(artifact_id) != 64
            or any(character not in "0123456789abcdef" for character in artifact_id)
        ):
            raise ValueError("artifact_id must be a lowercase SHA-256 hex digest")
        self._artifacts[(key, content_id)] = artifact_id

    def publish_swiglu_expert(
        self,
        *,
        key: ExpertKey,
        content_id: str,
        model_id: str,
        model_digest: str,
        gate_projection: torch.Tensor,
        up_projection: torch.Tensor,
        down_projection: torch.Tensor,
    ) -> str:
        """Upload one exact float32 expert and register its identity contract."""

        if not isinstance(key, ExpertKey):
            raise TypeError("key must be ExpertKey")
        gate = _float32_cpu(gate_projection, "gate_projection")
        up = _float32_cpu(up_projection, "up_projection")
        down = _float32_cpu(down_projection, "down_projection")
        if gate.ndim != 2 or up.shape != gate.shape:
            raise ValueError("gate and up projections must share [intermediate, hidden]")
        intermediate_size, hidden_size = map(int, gate.shape)
        if tuple(down.shape) != (hidden_size, intermediate_size):
            raise ValueError("down projection must have shape [hidden, intermediate]")
        packed = torch.cat((gate.reshape(-1), up.reshape(-1), down.reshape(-1)))
        weights = packed.numpy().tobytes(order="C")
        weights_hash = hashlib.sha256(weights).hexdigest()
        self._request(
            "PUT",
            f"/internal/v1/mobile/experts/weights/{weights_hash}",
            weights,
            content_type="application/octet-stream",
        )
        canary = torch.linspace(-0.75, 0.75, hidden_size, dtype=torch.float32).reshape(1, -1)
        expected = F.linear(F.silu(F.linear(canary, gate)) * F.linear(canary, up), down)
        registered = self._request_json(
            "POST",
            "/internal/v1/mobile/experts/register",
            {
                "modelId": model_id,
                "modelDigest": model_digest,
                "layer": key.layer,
                "expert": key.expert,
                "contentId": content_id,
                "weightsHash": weights_hash,
                "hiddenSize": hidden_size,
                "intermediateSize": intermediate_size,
                "dtype": "float32",
                "activation": "silu",
                "canaryInputBase64": _tensor_base64(canary),
                "canaryOutputBase64": _tensor_base64(expected),
            },
        )
        artifact_id = registered.get("artifactId")
        if not isinstance(artifact_id, str) or len(artifact_id) != 64:
            raise BrowserExpertOwnerError("coordinator returned an invalid artifact identity")
        self._artifacts[(key, content_id)] = artifact_id
        return artifact_id

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        return (key, content_id) in self._artifacts

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        artifact_id = self._artifacts.get((key, content_id))
        if artifact_id is None:
            return False
        try:
            result = self._request_json(
                "POST",
                "/internal/v1/mobile/experts/prepare",
                {"artifactId": artifact_id},
            )
        except BrowserExpertOwnerError:
            return False
        return result.get("resident") is True

    def execute_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        results: list[OwnerExpertBatchResult] = []
        for item in items:
            artifact_id = self._artifacts.get((item.key, item.content_id))
            if artifact_id is None:
                raise BrowserExpertOwnerError(
                    f"browser owner lacks exact content for {item.key}"
                )
            activations = _float32_cpu(item.activations, "activations")
            if activations.ndim != 2:
                raise ValueError("expert activations must have shape [rows, hidden]")
            try:
                response = self._request_json(
                    "POST",
                    "/internal/v1/mobile/experts/execute",
                    {
                        "artifactId": artifact_id,
                        "rows": int(activations.shape[0]),
                        "hiddenSize": int(activations.shape[1]),
                        "activationsBase64": _tensor_base64(activations),
                    },
                )
            except BrowserExpertOwnerError as exc:
                raise ExpertResidentSlotUnavailableError(item.key, str(exc)) from exc
            output = _tensor_from_base64(
                response.get("outputBase64"),
                rows=int(activations.shape[0]),
                columns=int(activations.shape[1]),
            ).to(device=item.activations.device, dtype=item.activations.dtype)
            results.append(OwnerExpertBatchResult(item.key, output))
        return tuple(results)

    def _request_json(self, method: str, path: str, payload: dict[str, object]) -> dict[str, object]:
        raw = self._request(
            method,
            path,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            content_type="application/json",
        )
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BrowserExpertOwnerError("coordinator returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise BrowserExpertOwnerError("coordinator returned a non-object response")
        return decoded

    def _request(
        self,
        method: str,
        path: str,
        body: bytes,
        *,
        content_type: str,
    ) -> bytes:
        headers = {"content-type": content_type, "accept": "application/json"}
        if path.startswith("/internal/"):
            headers["authorization"] = f"Bearer {self._internal_token}"
        request = Request(
            f"{self.coordinator_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BrowserExpertOwnerError(
                f"coordinator rejected browser expert request (HTTP {exc.code}): {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BrowserExpertOwnerError(f"browser expert coordinator unavailable: {exc}") from exc


def _load_internal_token(explicit: str | None) -> str:
    token = explicit.strip() if explicit is not None else ""
    if not token:
        token = os.environ.get("MYCELLIOS_INTERNAL_TOKEN", "").strip()
    if not token:
        secret_path = os.environ.get("MYCELLIOS_INTERNAL_TOKEN_FILE", "").strip()
        if secret_path:
            try:
                token = Path(secret_path).expanduser().read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise BrowserExpertOwnerError(
                    "could not read MYCELLIOS_INTERNAL_TOKEN_FILE"
                ) from exc
    if not token:
        raise BrowserExpertOwnerError(
            "browser expert control requires internal_token, "
            "MYCELLIOS_INTERNAL_TOKEN, or MYCELLIOS_INTERNAL_TOKEN_FILE"
        )
    if len(token) > 4096 or any(
        character.isspace() or ord(character) < 0x21 or ord(character) > 0x7E
        for character in token
    ):
        raise BrowserExpertOwnerError("the internal token is not a valid HTTP bearer token")
    return token


def _float32_cpu(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() < 1:
        raise ValueError(f"{name} must be a non-empty tensor")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating dtype")
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _tensor_base64(tensor: torch.Tensor) -> str:
    return base64.b64encode(_float32_cpu(tensor, "tensor").numpy().tobytes(order="C")).decode("ascii")


def _tensor_from_base64(value: object, *, rows: int, columns: int) -> torch.Tensor:
    if not isinstance(value, str):
        raise BrowserExpertOwnerError("browser returned no output tensor")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise BrowserExpertOwnerError("browser returned invalid base64") from exc
    expected = rows * columns * 4
    if len(raw) != expected:
        raise BrowserExpertOwnerError("browser returned an invalid output shape")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(rows, columns).clone()

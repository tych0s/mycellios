from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from distributed_runtime.browser_expert_owner import (
    BrowserExpertOwner,
    BrowserExpertOwnerError,
)
from distributed_runtime.ram_expert_cache import ExpertKey, ExpertRecord
from distributed_runtime.resident_expert_mesh import (
    AuthoritativeRouting,
    MeshLinkProfile,
    MeshNodeProfile,
    ResidentExpertMesh,
    ResidentExpertReplica,
)


class _CoordinatorHandler(BaseHTTPRequestHandler):
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    artifact_id = "a" * 64
    execution_calls = 0
    requests: list[tuple[str, str | None]] = []

    def do_PUT(self) -> None:  # noqa: N802
        type(self).requests.append((self.path, self.headers.get("authorization")))
        self._read_body()
        self._json(201, {"stored": True})

    def do_POST(self) -> None:  # noqa: N802
        type(self).requests.append((self.path, self.headers.get("authorization")))
        body = json.loads(self._read_body().decode("utf-8"))
        if self.path == "/public/test":
            self._json(200, {"public": True})
            return
        if self.path.endswith("/register"):
            self._json(201, {"artifactId": self.artifact_id})
            return
        if self.path.endswith("/prepare"):
            self._json(200, {"artifactId": body["artifactId"], "resident": True})
            return
        if self.path.endswith("/execute"):
            raw = base64.b64decode(body["activationsBase64"])
            hidden = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
                body["rows"], body["hiddenSize"]
            )
            output = F.linear(
                F.silu(F.linear(hidden, self.gate)) * F.linear(hidden, self.up),
                self.down,
            )
            type(self).execution_calls += 1
            self._json(
                200,
                {
                    "outputBase64": base64.b64encode(
                        output.contiguous().numpy().tobytes()
                    ).decode("ascii")
                },
            )
            return
        self._json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("content-length", "0")))

    def _json(self, status: int, value: object) -> None:
        encoded = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class BrowserExpertOwnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        _CoordinatorHandler.gate = torch.randn((4, 3), dtype=torch.float32) * 0.2
        _CoordinatorHandler.up = torch.randn((4, 3), dtype=torch.float32) * 0.2
        _CoordinatorHandler.down = torch.randn((3, 4), dtype=torch.float32) * 0.2
        _CoordinatorHandler.execution_calls = 0
        _CoordinatorHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CoordinatorHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_resident_mesh_output_depends_on_browser_swiglu_result(self) -> None:
        key = ExpertKey(0, 0)
        content_id = "sha256:tiny-expert"
        owner = BrowserExpertOwner(
            "mobile-browser",
            f"http://127.0.0.1:{self.server.server_port}",
            internal_token="expert-admin",
        )
        owner.publish_swiglu_expert(
            key=key,
            content_id=content_id,
            model_id="tiny-moe",
            model_digest="sha256:tiny-model",
            gate_projection=_CoordinatorHandler.gate,
            up_projection=_CoordinatorHandler.up,
            down_projection=_CoordinatorHandler.down,
        )
        record = ExpertRecord(key, 144, content_id)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=(record,),
            nodes=(
                MeshNodeProfile(
                    "root", 1024, 0, 100.0,
                    expert_workspace_bytes_per_token=64,
                ),
                MeshNodeProfile(
                    "mobile-browser", 1024, 0, 0.01,
                    expert_workspace_bytes_per_token=64,
                ),
            ),
            links=(MeshLinkProfile("root", "mobile-browser", 0.01, 10_000),),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(ResidentExpertReplica(key, "mobile-browser", content_id),),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        hidden = torch.tensor([[0.25, -0.5, 1.0], [-0.2, 0.4, 0.7]])
        routing = AuthoritativeRouting(
            torch.zeros((2, 1), dtype=torch.long),
            torch.ones((2, 1), dtype=torch.float32),
        )
        try:
            actual, plan = mesh.execute_layer(
                hidden, 0, routing, {"mobile-browser": owner}
            )
        finally:
            mesh.close()
        expected = F.linear(
            F.silu(F.linear(hidden, _CoordinatorHandler.gate))
            * F.linear(hidden, _CoordinatorHandler.up),
            _CoordinatorHandler.down,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(_CoordinatorHandler.execution_calls, 1)
        self.assertEqual({dispatch.path for dispatch in plan.dispatches}, {"remote-resident"})
        self.assertGreaterEqual(len(_CoordinatorHandler.requests), 3)
        self.assertTrue(all(
            authorization == "Bearer expert-admin"
            for path, authorization in _CoordinatorHandler.requests
            if path.startswith("/internal/")
        ))

    def test_loads_internal_token_from_environment_or_secret_file(self) -> None:
        coordinator_url = f"http://127.0.0.1:{self.server.server_port}"
        key = ExpertKey(0, 0)
        artifact_id = "a" * 64

        with patch.dict(
            os.environ,
            {"MYCELLIOS_INTERNAL_TOKEN": "environment-secret"},
            clear=False,
        ):
            os.environ.pop("MYCELLIOS_INTERNAL_TOKEN_FILE", None)
            owner = BrowserExpertOwner("environment-owner", coordinator_url)
            owner.attach_registered_expert(key, "sha256:environment", artifact_id)
            self.assertTrue(owner.is_expert_resident(key, "sha256:environment"))

        with tempfile.TemporaryDirectory() as directory:
            secret_file = Path(directory, "internal-token")
            secret_file.write_text("file-secret\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"MYCELLIOS_INTERNAL_TOKEN_FILE": str(secret_file)},
                clear=False,
            ):
                os.environ.pop("MYCELLIOS_INTERNAL_TOKEN", None)
                owner = BrowserExpertOwner("file-owner", coordinator_url)
                owner.attach_registered_expert(key, "sha256:file", artifact_id)
                self.assertTrue(owner.is_expert_resident(key, "sha256:file"))

        self.assertIn(
            ("/internal/v1/mobile/experts/prepare", "Bearer environment-secret"),
            _CoordinatorHandler.requests,
        )
        self.assertIn(
            ("/internal/v1/mobile/experts/prepare", "Bearer file-secret"),
            _CoordinatorHandler.requests,
        )

    def test_fails_closed_without_an_internal_token(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MYCELLIOS_INTERNAL_TOKEN", None)
            os.environ.pop("MYCELLIOS_INTERNAL_TOKEN_FILE", None)
            with self.assertRaisesRegex(BrowserExpertOwnerError, "requires internal_token"):
                BrowserExpertOwner(
                    "missing-token",
                    f"http://127.0.0.1:{self.server.server_port}",
                )

    def test_does_not_send_internal_token_to_public_routes(self) -> None:
        owner = BrowserExpertOwner(
            "public-route-check",
            f"http://127.0.0.1:{self.server.server_port}",
            internal_token="must-not-leak",
        )
        owner._request(  # noqa: SLF001 - explicit security boundary regression test
            "POST",
            "/public/test",
            b"{}",
            content_type="application/json",
        )
        self.assertIn(("/public/test", None), _CoordinatorHandler.requests)


if __name__ == "__main__":
    unittest.main()

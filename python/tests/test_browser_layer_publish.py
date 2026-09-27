"""The publisher accepts a graph whose names and state differ from the RoPE decoder."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import onnx
    from onnx import TensorProto, helper
except ModuleNotFoundError:
    onnx = None
    TensorProto = None
    helper = None


def _encoded(values: list[float]) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def test_stateless_non_qwen_graph_uses_the_generic_manifest():
    script = Path(__file__).resolve().parents[2] / "scripts" / "publish-browser-layer.py"
    spec = importlib.util.spec_from_file_location("publish_browser_layer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = helper.make_graph(
        [helper.make_node("Add", ["activation", "bias"], ["result"])], "stateless-stage",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "tokens", 2])
         for name in ("activation", "bias")],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, "tokens", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stage.onnx"
        onnx.save(model, path)
        manifest = {
            "schema": "mycellios-browser-layer/2",
            "hidden": {"inputName": "activation", "outputName": "result", "width": 2},
            "auxiliary": [{"name": "bias", "shape": [1, "tokens", 2]}],
            "state": [],
            "canary": {"tokens": 1, "hiddenBase64": _encoded([1, 2]),
                       "auxiliaryBase64": {"bias": _encoded([3, 4])},
                       "expectedBase64": _encoded([4, 6])},
        }
        module._verify_graph(path, manifest)
        manifest["canary"]["expectedBase64"] = _encoded([4, 7])
        try:
            module._verify_graph(path, manifest)
        except ValueError as error:
            assert "canary" in str(error)
        else:
            raise AssertionError("publisher accepted a wrong numerical canary")


@unittest.skipUnless(onnx is not None, "onnx is an optional publisher dependency")
class BrowserLayerPublisherTests(unittest.TestCase):
    def test_stateless_generic_onnx_manifest(self):
        test_stateless_non_qwen_graph_uses_the_generic_manifest()

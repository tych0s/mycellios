from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import load_file

from distributed_runtime.native_gguf import (
    GgufTensor,
    NativeGgufError,
    build_native_gguf_stage,
    dequantize_gguf_tensor,
    materialize_native_gguf_stage,
    parse_gguf,
    verify_native_gguf_stage,
)


class NativeGgufTests(unittest.TestCase):
    def test_builds_exact_first_and_last_layer_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)

            first = build_native_gguf_stage(
                source,
                root / "first",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision="a" * 40,
            )
            last = build_native_gguf_stage(
                source,
                root / "last",
                config_source=config,
                layer_start=1,
                layer_end=2,
                model_source="hf://fixture/model",
                model_revision="a" * 40,
            )

            self.assertIn("token_embd.weight", first.tensor_names)
            self.assertTrue(any(name.startswith("blk.0.") for name in first.tensor_names))
            self.assertFalse(any(name.startswith("blk.1.") for name in first.tensor_names))
            self.assertNotIn("output.weight", first.tensor_names)

            self.assertIn("output.weight", last.tensor_names)
            self.assertIn("output_norm.weight", last.tensor_names)
            self.assertTrue(any(name.startswith("blk.1.") for name in last.tensor_names))
            self.assertFalse(any(name.startswith("blk.0.") for name in last.tensor_names))
            self.assertNotIn("token_embd.weight", last.tensor_names)
            self.assertNotEqual(first.package_id, last.package_id)

    def test_materializes_only_mapped_stage_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            fixture = _write_fixture_gguf(source)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=1,
                layer_end=2,
                model_source="hf://fixture/model",
                model_revision=None,
            )

            materialized = materialize_native_gguf_stage(
                package.root, root / "materialized"
            )
            tensors = load_file(materialized / "model.safetensors")
            self.assertEqual(
                set(tensors),
                {
                    "model.layers.1.input_layernorm.weight",
                    "model.layers.1.self_attn.q_proj.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                },
            )
            self.assertTrue(
                torch.equal(
                    tensors["model.layers.1.self_attn.q_proj.weight"],
                    torch.from_numpy(fixture["blk.1.attn_q.weight"]).reshape(4, 4),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["lm_head.weight"],
                    torch.from_numpy(fixture["output.weight"]).reshape(8, 4),
                )
            )

    def test_manifest_and_payload_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            stage_file = package.root / "native-stage.gguf"
            payload = bytearray(stage_file.read_bytes())
            payload[-1] ^= 0x01
            stage_file.write_bytes(payload)
            with self.assertRaisesRegex(NativeGgufError, "digest differs"):
                verify_native_gguf_stage(package.root)

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            destination = root / "stage"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("owned", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_native_gguf_stage(
                    source,
                    destination,
                    config_source=config,
                    layer_start=0,
                    layer_end=1,
                    model_source="hf://fixture/model",
                    model_revision=None,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "owned")

    def test_raw_k_quant_is_preserved_but_execution_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=1, hidden=4)
            tensors = _basic_tensors(layers=1)
            tensors["blk.0.attn_q.weight"] = (12, (256,), bytes(144))
            _write_gguf(source, layers=1, hidden=4, tensors=tensors)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            self.assertIn("blk.0.attn_q.weight", package.tensor_names)
            with self.assertRaisesRegex(NativeGgufError, "unsupported executable"):
                materialize_native_gguf_stage(package.root, root / "materialized")

    def test_overlapping_tensor_ranges_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.gguf"
            tensors = _basic_tensors(layers=1)
            _write_gguf(path, layers=1, hidden=4, tensors=tensors, overlap=True)
            with self.assertRaisesRegex(NativeGgufError, "overlap"):
                parse_gguf(path)

    def test_q4_q5_q8_native_dequantizers_match_constructed_values(self) -> None:
        cases = (
            (2, _q4_0_block(), np.arange(32, dtype=np.float32) % 16 - 8),
            (3, _q4_1_block(), (np.arange(32, dtype=np.float32) % 16) * 0.5 + 2),
            (6, _q5_0_block(), np.arange(32, dtype=np.float32) - 16),
            (7, _q5_1_block(), np.arange(32, dtype=np.float32) * 0.25 + 1),
            (8, _q8_0_block(), (np.arange(32, dtype=np.float32) - 16) * 0.5),
        )
        block_sizes = {2: 18, 3: 20, 6: 22, 7: 24, 8: 34}
        for ggml_type, raw, expected in cases:
            with self.subTest(ggml_type=ggml_type):
                tensor = GgufTensor(
                    name="fixture",
                    dimensions=(32,),
                    ggml_type=ggml_type,
                    relative_offset=0,
                    size_bytes=block_sizes[ggml_type],
                    data_offset=0,
                )
                actual = dequantize_gguf_tensor(tensor, raw)
                self.assertTrue(
                    torch.allclose(actual, torch.from_numpy(expected), atol=1e-3)
                )


def _write_config(root: Path, *, layers: int, hidden: int) -> Path:
    path = root / f"config-{layers}.json"
    path.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": layers,
                "hidden_size": hidden,
                "intermediate_size": hidden * 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 8,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10_000,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fixture_gguf(path: Path) -> dict[str, np.ndarray]:
    values = {
        "token_embd.weight": np.arange(32, dtype=np.float32),
        "blk.0.attn_norm.weight": np.arange(4, dtype=np.float32) + 100,
        "blk.0.attn_q.weight": np.arange(16, dtype=np.float32) + 200,
        "blk.1.attn_norm.weight": np.arange(4, dtype=np.float32) + 300,
        "blk.1.attn_q.weight": np.arange(16, dtype=np.float32) + 400,
        "output_norm.weight": np.arange(4, dtype=np.float32) + 500,
        "output.weight": np.arange(32, dtype=np.float32) + 600,
    }
    tensors = {
        "token_embd.weight": (0, (4, 8), values["token_embd.weight"].tobytes()),
        "blk.0.attn_norm.weight": (
            0,
            (4,),
            values["blk.0.attn_norm.weight"].tobytes(),
        ),
        "blk.0.attn_q.weight": (
            0,
            (4, 4),
            values["blk.0.attn_q.weight"].tobytes(),
        ),
        "blk.1.attn_norm.weight": (
            0,
            (4,),
            values["blk.1.attn_norm.weight"].tobytes(),
        ),
        "blk.1.attn_q.weight": (
            0,
            (4, 4),
            values["blk.1.attn_q.weight"].tobytes(),
        ),
        "output_norm.weight": (
            0,
            (4,),
            values["output_norm.weight"].tobytes(),
        ),
        "output.weight": (0, (4, 8), values["output.weight"].tobytes()),
    }
    _write_gguf(path, layers=2, hidden=4, tensors=tensors)
    return values


def _basic_tensors(*, layers: int) -> dict[str, tuple[int, tuple[int, ...], bytes]]:
    tensors: dict[str, tuple[int, tuple[int, ...], bytes]] = {
        "token_embd.weight": (0, (4, 8), bytes(4 * 8 * 4)),
        "output_norm.weight": (0, (4,), bytes(4 * 4)),
        "output.weight": (0, (4, 8), bytes(4 * 8 * 4)),
    }
    for layer in range(layers):
        tensors[f"blk.{layer}.attn_norm.weight"] = (0, (4,), bytes(4 * 4))
        tensors[f"blk.{layer}.attn_q.weight"] = (0, (4, 4), bytes(4 * 4 * 4))
    return tensors


def _write_gguf(
    path: Path,
    *,
    layers: int,
    hidden: int,
    tensors: dict[str, tuple[int, tuple[int, ...], bytes]],
    overlap: bool = False,
) -> None:
    metadata = (
        ("general.architecture", 8, "llama"),
        ("general.alignment", 4, 32),
        ("llama.block_count", 4, layers),
        ("llama.embedding_length", 4, hidden),
    )
    offsets: list[int] = []
    cursor = 0
    for _name, (_ggml_type, _dimensions, payload) in tensors.items():
        cursor = (cursor + 31) // 32 * 32
        offsets.append(0 if overlap else cursor)
        cursor += len(payload)
    with path.open("wb") as output:
        output.write(b"GGUF")
        output.write(struct.pack("<IQQ", 3, len(tensors), len(metadata)))
        for key, value_type, value in metadata:
            _string(output, key)
            output.write(struct.pack("<I", value_type))
            if value_type == 8:
                _string(output, value)
            else:
                output.write(struct.pack("<I", value))
        for (name, (ggml_type, dimensions, _payload)), offset in zip(
            tensors.items(), offsets, strict=True
        ):
            _string(output, name)
            output.write(struct.pack("<I", len(dimensions)))
            for dimension in dimensions:
                output.write(struct.pack("<Q", dimension))
            output.write(struct.pack("<IQ", ggml_type, offset))
        output.write(bytes((-output.tell()) % 32))
        data_start = output.tell()
        for (_name, (_ggml_type, _dimensions, payload)), offset in zip(
            tensors.items(), offsets, strict=True
        ):
            desired = data_start + offset
            if output.tell() < desired:
                output.write(bytes(desired - output.tell()))
            output.write(payload)


def _string(output, value: str) -> None:
    encoded = value.encode("utf-8")
    output.write(struct.pack("<Q", len(encoded)))
    output.write(encoded)


def _pack_nibbles(values: np.ndarray) -> bytes:
    return bytes(
        int(values[index] & 0x0F) | (int(values[index + 16] & 0x0F) << 4)
        for index in range(16)
    )


def _q4_0_block() -> bytes:
    values = np.arange(32, dtype=np.int16) % 16 - 8
    encoded = (values + 8).astype(np.uint8)
    return struct.pack("<e", 1.0) + _pack_nibbles(encoded)


def _q4_1_block() -> bytes:
    values = np.arange(32, dtype=np.uint8) % 16
    return struct.pack("<ee", 0.5, 2.0) + _pack_nibbles(values)


def _q5_payload(values: np.ndarray) -> tuple[bytes, bytes]:
    unsigned = values.astype(np.uint8)
    qh = 0
    for index, value in enumerate(unsigned):
        qh |= int((value >> 4) & 1) << index
    packed = _pack_nibbles(unsigned)
    return struct.pack("<I", qh), packed


def _q5_0_block() -> bytes:
    high, packed = _q5_payload(np.arange(32, dtype=np.uint8))
    return struct.pack("<e", 1.0) + high + packed


def _q5_1_block() -> bytes:
    high, packed = _q5_payload(np.arange(32, dtype=np.uint8))
    return struct.pack("<ee", 0.25, 1.0) + high + packed


def _q8_0_block() -> bytes:
    values = np.arange(32, dtype=np.int16) - 16
    return struct.pack("<e", 0.5) + values.astype(np.int8).tobytes()


if __name__ == "__main__":
    unittest.main()

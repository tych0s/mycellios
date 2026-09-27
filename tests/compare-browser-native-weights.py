"""Compare a pinned MLC F16 checkpoint with a local F16 GGUF checkpoint.

Requires numpy and gguf-py (for example, llama.cpp/gguf-py). The MLC index and
all eight shards must be downloaded from one exact model revision first.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--mlc-index", type=Path, required=True)
    parser.add_argument("--mlc-shards", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, help="Directory containing the gguf package")
    args = parser.parse_args()
    if args.gguf_py:
        sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader

    index = json.loads(args.mlc_index.read_text(encoding="utf-8"))
    records = {}
    shard_bytes = 0
    for shard in index["records"]:
        data = (args.mlc_shards / shard["dataPath"]).read_bytes()
        assert len(data) == shard["nbytes"], shard["dataPath"]
        assert hashlib.md5(data).hexdigest() == shard["md5sum"], shard["dataPath"]
        shard_bytes += len(data)
        for record in shard["records"]:
            assert record["dtype"] == "float16", record["name"]
            assert record["nbytes"] == np.prod(record["shape"]) * 2, record["name"]
            offset = record["byteOffset"]
            records[record["name"]] = memoryview(data)[offset : offset + record["nbytes"]]

    reader = GGUFReader(str(args.gguf))
    tensors = {tensor.name: tensor.data for tensor in reader.tensors}
    head_count = int(reader.get_field("llama.attention.head_count").parts[-1][0])
    kv_head_count = int(reader.get_field("llama.attention.head_count_kv").parts[-1][0])
    block_count = int(reader.get_field("llama.block_count").parts[-1][0])
    embedding_size = int(reader.get_field("llama.embedding_length").parts[-1][0])

    checked_bytes = 0
    components = 0

    def gguf_bytes(name: str) -> bytes:
        return tensors[name].tobytes()

    def compare(name: str, observed, expected: bytes) -> None:
        nonlocal checked_bytes, components
        if observed != expected:
            raise AssertionError(f"weight mismatch: {name}")
        checked_bytes += len(observed)
        components += 1

    def compare_norm(mlc: str, gguf: str) -> None:
        compare(mlc, records[mlc], tensors[gguf].astype("<f2").tobytes())

    def permute_qk(raw, heads: int) -> bytes:
        weight = np.frombuffer(raw, "<f2").reshape(-1, embedding_size)
        return (
            weight.reshape(heads, 2, weight.shape[0] // heads // 2, embedding_size)
            .swapaxes(1, 2)
            .reshape(-1, embedding_size)
            .tobytes()
        )

    compare("model.embed_tokens.weight", records["model.embed_tokens.weight"], gguf_bytes("token_embd.weight"))
    for layer in range(block_count):
        m = f"model.layers.{layer}."
        g = f"blk.{layer}."
        compare_norm(m + "input_layernorm.weight", g + "attn_norm.weight")
        compare(m + "mlp.down_proj.weight", records[m + "mlp.down_proj.weight"], gguf_bytes(g + "ffn_down.weight"))
        compare(
            m + "mlp.gate_up_proj.weight",
            records[m + "mlp.gate_up_proj.weight"],
            gguf_bytes(g + "ffn_gate.weight") + gguf_bytes(g + "ffn_up.weight"),
        )
        compare_norm(m + "post_attention_layernorm.weight", g + "ffn_norm.weight")
        qkv = records[m + "self_attn.qkv_proj.weight"]
        q_bytes = head_count * (embedding_size // head_count) * embedding_size * 2
        k_bytes = kv_head_count * (embedding_size // head_count) * embedding_size * 2
        compare(m + "self_attn.qkv_proj.Q", permute_qk(qkv[:q_bytes], head_count), gguf_bytes(g + "attn_q.weight"))
        compare(m + "self_attn.qkv_proj.K", permute_qk(qkv[q_bytes : q_bytes + k_bytes], kv_head_count), gguf_bytes(g + "attn_k.weight"))
        compare(m + "self_attn.qkv_proj.V", qkv[q_bytes + k_bytes :], gguf_bytes(g + "attn_v.weight"))
        compare(m + "self_attn.o_proj.weight", records[m + "self_attn.o_proj.weight"], gguf_bytes(g + "attn_output.weight"))
    compare_norm("model.norm.weight", "output_norm.weight")

    assert checked_bytes == shard_bytes, (checked_bytes, shard_bytes)
    assert len(records) == 2 + 6 * block_count, len(records)
    print(f"PASS: {len(index['records'])} MD5-verified shards; {len(records)} MLC records; {components} comparisons; {checked_bytes} checked bytes")


if __name__ == "__main__":
    main()

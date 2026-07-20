from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from distributed_runtime.hf_moe_extractor import (
    HF_MOE_EXTRACTOR_SCHEMA,
    UnsupportedHfMoeExpertLayoutError,
    extract_hf_moe_experts,
    hf_moe_extractor_registry_document,
)
from distributed_runtime.ram_expert_cache import (
    ExpertKey,
    PredictiveCacheConfig,
)
from distributed_runtime.torch_ram_expert_store import TorchRamExpertStore


_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class _Router(nn.Module):
    def __init__(self, experts: int, hidden: int, top_k: int, *, glm: bool) -> None:
        super().__init__()
        self.num_experts = experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.zeros(experts, hidden))
        if glm:
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(experts),
            )


class _GroupedExperts(nn.Module):
    def __init__(self, experts: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.num_experts = experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        values = torch.arange(
            experts * intermediate * 2 * hidden,
            dtype=torch.float32,
        ).reshape(experts, intermediate * 2, hidden)
        self.gate_up_proj = nn.Parameter(values)
        self.down_proj = nn.Parameter(
            torch.arange(
                experts * hidden * intermediate,
                dtype=torch.float32,
            ).reshape(experts, hidden, intermediate)
        )


class _CanonicalMlp(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _SparseMlp(nn.Module):
    def __init__(
        self,
        *,
        experts: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        glm: bool,
        layout: str,
        shared: nn.Module | None = None,
        expert_bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate = _Router(experts, hidden, top_k, glm=glm)
        if layout == "grouped":
            self.experts = _GroupedExperts(experts, hidden, intermediate)
        elif layout == "list":
            self.experts = nn.ModuleList(
                [
                    _CanonicalMlp(hidden, intermediate, bias=expert_bias)
                    for _ in range(experts)
                ]
            )
        else:
            raise AssertionError(f"unknown synthetic layout {layout}")
        if glm:
            self.shared_experts = shared


class _Layer(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp


class _Decoder(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _CausalModel(nn.Module):
    def __init__(self, config: SimpleNamespace, layers: list[nn.Module]) -> None:
        super().__init__()
        self.config = config
        self.model = _Decoder(layers)


def _config(
    family: str,
    *,
    layers: int,
    experts: int = 2,
    hidden: int = 4,
    moe_intermediate: int = 3,
    dense_prefix: int = 0,
    shared: int = 0,
) -> SimpleNamespace:
    common = dict(
        num_hidden_layers=layers,
        hidden_size=hidden,
        intermediate_size=6,
        moe_intermediate_size=moe_intermediate,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_experts_per_tok=1,
        num_local_experts=experts,
        layer_types=["full_attention"] * layers,
        use_sliding_window=False,
        base_model_ep_plan=dict(_EP_PLAN),
    )
    if family == "qwen":
        return SimpleNamespace(
            **common,
            model_type="qwen3_moe",
            architectures=["Qwen3MoeForCausalLM"],
            num_experts=experts,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
    if family == "glm":
        return SimpleNamespace(
            **common,
            model_type="glm4_moe",
            architectures=["Glm4MoeForCausalLM"],
            n_routed_experts=experts,
            n_shared_experts=shared,
            first_k_dense_replace=dense_prefix,
        )
    raise AssertionError(f"unknown synthetic family {family}")


def _sparse_layer(
    *,
    family: str,
    layout: str = "grouped",
    shared: nn.Module | None = None,
    expert_bias: bool = False,
) -> _Layer:
    return _Layer(
        _SparseMlp(
            experts=2,
            hidden=4,
            intermediate=3,
            top_k=1,
            glm=family == "glm",
            layout=layout,
            shared=shared,
            expert_bias=expert_bias,
        )
    )


class HfMoeExtractorTests(unittest.TestCase):
    def test_qwen_grouped_tensors_become_store_compatible_expert_views(self) -> None:
        model = _CausalModel(
            _config("qwen", layers=2),
            [_sparse_layer(family="qwen"), _sparse_layer(family="qwen")],
        )
        extracted = extract_hf_moe_experts(
            model,
            model_identity="sha256:qwen-fixture",
            global_layer_start=5,
        )

        self.assertEqual(extracted.adapter_id, "transformers-qwen3-moe-v1")
        self.assertEqual(extracted.storage_layout, "grouped-3d-swiglu")
        self.assertEqual(extracted.sparse_layers, (5, 6))
        self.assertEqual(extracted.dense_layers, ())
        self.assertEqual(len(extracted.inventory.records), 4)
        key = ExpertKey(5, 1)
        tensors = extracted.expert_tensors[key]
        self.assertEqual(
            tuple(tensors),
            ("gate_proj.weight", "up_proj.weight", "down_proj.weight"),
        )
        self.assertEqual(tuple(tensors["gate_proj.weight"].shape), (3, 4))
        grouped = model.model.layers[0].mlp.experts.gate_up_proj
        self.assertEqual(
            tensors["gate_proj.weight"].untyped_storage().data_ptr(),
            grouped.untyped_storage().data_ptr(),
            "extraction should expose a view, not clone grouped source weights",
        )
        self.assertEqual(extracted.inventory.record(key).byte_size, 144)
        self.assertEqual(extracted.shared_experts, ())

        cache_config = PredictiveCacheConfig(
            capacity_bytes=1_152,
            prefetch_reserve_bytes=288,
            pcie_bandwidth_gbytes_per_second=8.0,
        )
        store = TorchRamExpertStore(
            extracted.inventory,
            extracted.expert_tensors,
            cache_config,
            device="cpu",
        )
        self.assertEqual(store.snapshot().ram_bytes, extracted.routed_expert_bytes)

    def test_glm_dense_prefix_is_skipped_and_tied_shared_expert_is_stored_once(self) -> None:
        shared = _CanonicalMlp(4, 3)
        model = _CausalModel(
            _config("glm", layers=3, dense_prefix=1, shared=1),
            [
                _Layer(_CanonicalMlp(4, 6)),
                _sparse_layer(family="glm", shared=shared),
                _sparse_layer(family="glm", shared=shared),
            ],
        )
        extracted = extract_hf_moe_experts(
            model,
            model_identity="sha256:glm-fixture",
            global_layer_start=10,
        )

        self.assertEqual(extracted.adapter_id, "transformers-glm4-moe-v1")
        self.assertEqual(extracted.dense_layers, (10,))
        self.assertEqual(extracted.sparse_layers, (11, 12))
        self.assertEqual({record.key.layer for record in extracted.inventory.records}, {11, 12})
        self.assertEqual(len(extracted.shared_experts), 1)
        resident = extracted.shared_experts[0]
        self.assertEqual(resident.layers, (11, 12))
        self.assertEqual(resident.byte_size, 144)
        self.assertEqual(extracted.shared_expert_bytes, 144)
        self.assertEqual(
            extracted.shared_expert_content_by_layer[11],
            extracted.shared_expert_content_by_layer[12],
        )
        for tensors in extracted.expert_tensors.values():
            self.assertTrue(all("shared" not in name for name in tensors))

    def test_unpacked_module_list_layout_is_canonicalized_without_guessing(self) -> None:
        model = _CausalModel(
            _config("qwen", layers=1),
            [_sparse_layer(family="qwen", layout="list")],
        )
        extracted = extract_hf_moe_experts(
            model,
            model_identity="sha256:qwen-list-fixture",
        )
        self.assertEqual(extracted.storage_layout, "unpacked-module-list-swiglu")
        self.assertEqual(len(extracted.expert_tensors), 2)
        source = model.model.layers[0].mlp.experts[1].down_proj.weight
        self.assertIs(
            extracted.expert_tensors[ExpertKey(0, 1)]["down_proj.weight"],
            source,
        )

    def test_unknown_architecture_and_partial_ep_shards_fail_closed(self) -> None:
        model = _CausalModel(
            _config("qwen", layers=1),
            [_sparse_layer(family="qwen")],
        )
        model.config.architectures = ["PretendQwenMoeForCausalLM"]
        with self.assertRaisesRegex(
            UnsupportedHfMoeExpertLayoutError,
            "uncertified",
        ):
            extract_hf_moe_experts(model, model_identity="sha256:unsupported")

        model.config.architectures = ["Qwen3MoeForCausalLM"]
        model.config.num_local_experts = 1
        with self.assertRaisesRegex(
            UnsupportedHfMoeExpertLayoutError,
            "rank-local",
        ):
            extract_hf_moe_experts(model, model_identity="sha256:partial")

    def test_unexpected_bias_and_mixed_storage_layouts_fail_closed(self) -> None:
        biased = _CausalModel(
            _config("qwen", layers=1),
            [_sparse_layer(family="qwen", layout="list", expert_bias=True)],
        )
        with self.assertRaisesRegex(UnsupportedHfMoeExpertLayoutError, "bias-free"):
            extract_hf_moe_experts(biased, model_identity="sha256:biased")

        mixed = _CausalModel(
            _config("qwen", layers=2),
            [
                _sparse_layer(family="qwen", layout="grouped"),
                _sparse_layer(family="qwen", layout="list"),
            ],
        )
        with self.assertRaisesRegex(UnsupportedHfMoeExpertLayoutError, "mixing"):
            extract_hf_moe_experts(mixed, model_identity="sha256:mixed")

    def test_unexpected_router_parameters_and_persistent_buffers_fail_closed(self) -> None:
        for family in ("qwen", "glm"):
            for extra_kind in ("parameter", "persistent_buffer"):
                with self.subTest(family=family, extra_kind=extra_kind):
                    model = _CausalModel(
                        _config(
                            family,
                            layers=1,
                            shared=1 if family == "glm" else 0,
                        ),
                        [
                            _sparse_layer(
                                family=family,
                                shared=_CanonicalMlp(4, 3)
                                if family == "glm"
                                else None,
                            )
                        ],
                    )
                    router = model.model.layers[0].mlp.gate
                    if extra_kind == "parameter":
                        router.unexpected_scale = nn.Parameter(torch.ones(()))
                    else:
                        router.register_buffer(
                            "unexpected_running_value",
                            torch.ones(()),
                            persistent=True,
                        )

                    with self.assertRaisesRegex(
                        UnsupportedHfMoeExpertLayoutError,
                        "unexpected router parameters or persistent buffers",
                    ):
                        extract_hf_moe_experts(
                            model,
                            model_identity=f"sha256:{family}-{extra_kind}",
                        )

    def test_dense_only_glm_stage_has_no_false_expert_inventory(self) -> None:
        model = _CausalModel(
            _config("glm", layers=1, dense_prefix=1, shared=1),
            [_Layer(_CanonicalMlp(4, 6))],
        )
        with self.assertRaisesRegex(
            UnsupportedHfMoeExpertLayoutError,
            "no extractable sparse",
        ):
            extract_hf_moe_experts(model, model_identity="sha256:dense")

    def test_registry_names_only_the_two_certified_families(self) -> None:
        document = hf_moe_extractor_registry_document()
        self.assertEqual(document["schema"], HF_MOE_EXTRACTOR_SCHEMA)
        self.assertEqual(
            [item["architecture"] for item in document["architectures"]],
            ["Qwen3MoeForCausalLM", "Glm4MoeForCausalLM"],
        )
        self.assertTrue(
            all(not item["partialExpertShards"] for item in document["architectures"])
        )


if __name__ == "__main__":
    unittest.main()

"""Deterministic planner-only benchmark for prefix-closed tree work.

This deliberately measures token-step duplication, not model latency.  A
physical implementation still needs packed/frontier execution so the smaller
amount of compute does not turn into one WAN round trip per trie node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from .prefix_closed_schedule import plan_prefix_closed_schedule


SCHEMA = "gdlp-prefix-closed-planner-benchmark/1"


def _scenarios() -> dict[str, tuple[tuple[int, ...], ...]]:
    shared_trunk = tuple(
        (100, 101, 102, 103, 104, 105, 200 + branch, 300 + branch)
        for branch in range(8)
    )
    hierarchical = tuple(
        (400, 401, *(500 + bit_index * 2 + bit for bit_index, bit in enumerate(bits)))
        for value in range(16)
        for bits in (tuple((value >> shift) & 1 for shift in (3, 2, 1, 0)),)
    )
    disjoint = tuple(
        tuple(1_000 + branch * 16 + offset for offset in range(8))
        for branch in range(8)
    )
    clustered = (
        (50, 51, 52, 60, 70),
        (50, 51, 52, 60, 71),
        (50, 51, 52, 61, 72),
        (50, 51, 53, 62),
        (50, 51, 53, 63),
        (50, 54, 64),
    )
    return {
        "shared-trunk-8x8": shared_trunk,
        "hierarchical-binary-16x6": hierarchical,
        "disjoint-8x8": disjoint,
        "clustered-variable-depth": clustered,
        "single-leaf-control": ((9, 8, 7, 6),),
    }


def build_prefix_schedule_benchmark() -> dict[str, Any]:
    rows = []
    for name, paths in _scenarios().items():
        schedule = plan_prefix_closed_schedule(paths, shared_root_tokens=1)
        cost = schedule.cost
        rows.append(
            {
                "scenario": name,
                "leafCount": cost.leaf_count,
                "maximumDepth": cost.maximum_depth,
                "flatLeafTokenSteps": cost.flat_leaf_token_steps,
                "uniquePrefixTokenSteps": cost.unique_prefix_token_steps,
                "savedTokenSteps": cost.saved_token_steps,
                "computeReductionPercent": round(
                    100 * cost.compute_reduction_ratio, 6
                ),
                "tokenStepUpperBoundSpeedup": round(
                    cost.flat_leaf_token_steps / cost.unique_prefix_token_steps,
                    6,
                ),
                "flatForks": cost.flat_leaf_fork_steps,
                "prefixClosedForks": cost.prefix_closed_fork_steps,
                "divergences": cost.divergence_count,
                "lanes": cost.prefix_closed_lane_count,
            }
        )
    canonical_rows = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": SCHEMA,
        "evidenceClass": "deterministic-cpu-planner-only",
        "sharedRootTokens": 1,
        "interpretation": (
            "Token-step work only; not wall-clock speed, GPU throughput, WAN RTT, "
            "model parity, or physical two-host evidence."
        ),
        "rowsSha256": hashlib.sha256(canonical_rows).hexdigest(),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_prefix_schedule_benchmark(),
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "build_prefix_schedule_benchmark", "main"]

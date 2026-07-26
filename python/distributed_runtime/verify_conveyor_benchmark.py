"""Deterministic A/B harness for the native VERIFY conveyor.

This harness deliberately does **not** execute a model, GPU, socket or physical
network.  It compares the W=1 lockstep control with bounded W>1 VERIFY
conveyors in a deterministic event simulator.  RTT, verification service time
and draft accuracy are inputs, not measurements.

Every arm is checked against the same causal synthetic greedy oracle.  A case
is successful only when every committed token sequence has the oracle's exact
SHA-256 hash and both configured high-water limits are respected.  Modeled
latencies must never be reported as physical tokens/s.

The event model mirrors the native bridge contract: the first VERIFY carries
an already-visible seed, each continuation starts with the predecessor's
supposed bonus, and the predecessor's final target validates that first draft.
After rejection, new-generation dispatch is blocked until all condemned
in-flight waves have been drained.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import time
from typing import Any, Iterable, Sequence


SCHEMA = "mycellios-native-verify-conveyor-ab/1"
EVIDENCE_CLASS = "SIMULACION_DETERMINISTA_RTT_Y_COMPUTO_EMULADOS"
CONTROL_ARM = "A_LOCKSTEP_W1"
CONVEYOR_ARM = "B_VERIFY_CONVEYOR"
DEFAULT_SEED = 0x4D5943454C4C494F
DEFAULT_INFLIGHT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class _Wave:
    wave_id: int
    generation: int
    start: int
    proposal: tuple[int, ...]
    raw_targets: tuple[int, ...]
    verify_seed_tokens: int
    predecessor_wave_id: int | None
    dispatch_ms: float
    ready_ms: float
    reserved_bytes: int

    @property
    def token_count(self) -> int:
        return len(self.proposal)

    @property
    def input_token_count(self) -> int:
        return self.token_count + self.verify_seed_tokens

    def target_tokens(self, bridge: int | None) -> tuple[int, ...]:
        if self.verify_seed_tokens == 1:
            if bridge is not None or self.predecessor_wave_id is not None:
                raise AssertionError("seeded VERIFY wave cannot consume a bridge")
            targets = self.raw_targets
        elif self.verify_seed_tokens == 0:
            if bridge is None or self.predecessor_wave_id is None:
                raise AssertionError("continuation VERIFY wave lost its bonus bridge")
            targets = (bridge, *self.raw_targets)
        else:
            raise AssertionError("VERIFY seed count must be zero or one")
        if len(targets) != self.token_count + 1:
            raise AssertionError("VERIFY target vector must be drafts plus bonus")
        return targets


@dataclass(frozen=True)
class ConveyorEvent:
    kind: str
    time_ms: float
    wave_id: int
    generation: int


@dataclass(frozen=True)
class ConveyorSample:
    """One exact synthetic decode and its modeled scheduling evidence."""

    token_ids: tuple[int, ...]
    emission_ms: tuple[float, ...]
    finish_ms: float
    drain_complete_ms: float
    stats: dict[str, int | float | bool]
    events: tuple[ConveyorEvent, ...]

    @property
    def ttft_ms(self) -> float:
        return self.emission_ms[0]

    @property
    def tpot_ms(self) -> tuple[float, ...]:
        return tuple(
            right - left for left, right in zip(self.emission_ms, self.emission_ms[1:])
        )


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def _seed(raw: str) -> int:
    try:
        value = int(raw, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a decimal or 0x-prefixed integer"
        ) from error
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise argparse.ArgumentTypeError("must be in [0, 2^64-1]")
    return value


def parse_alphas(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("alphas must contain numbers") from error
    if (
        not values
        or any(not math.isfinite(value) or value < 0 or value > 1 for value in values)
        or tuple(sorted(set(values))) != values
    ):
        raise argparse.ArgumentTypeError(
            "alphas must be unique, increasing and between 0 and 1"
        )
    return values


def parse_rtts_ms(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("rtts-ms must contain numbers") from error
    if (
        not values
        or any(not math.isfinite(value) or value < 0 for value in values)
        or tuple(sorted(set(values))) != values
    ):
        raise argparse.ArgumentTypeError(
            "rtts-ms must be unique, increasing and non-negative"
        )
    return values


def parse_positive_int_sweep(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("sweep must contain integers") from error
    if (
        not values
        or any(value < 1 for value in values)
        or tuple(sorted(set(values))) != values
    ):
        raise argparse.ArgumentTypeError(
            "sweep must contain unique, increasing positive integers"
        )
    return values


def parse_windows(raw: str) -> tuple[int, ...]:
    values = parse_positive_int_sweep(raw)
    if values[0] != 1 or values[-1] > 64:
        raise argparse.ArgumentTypeError(
            "windows must start with the W=1 control and cannot exceed 64"
        )
    return values


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    materialized = sorted(float(value) for value in values)
    if not materialized:
        raise ValueError("cannot describe an empty sample")

    def percentile(fraction: float) -> float:
        if len(materialized) == 1:
            return materialized[0]
        position = (len(materialized) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return materialized[lower]
        weight = position - lower
        return materialized[lower] * (1.0 - weight) + materialized[upper] * weight

    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": materialized[0],
        "max": materialized[-1],
    }


def synthetic_greedy_oracle(
    *,
    seed: int,
    sample_index: int,
    token_count: int,
    vocabulary_size: int = 32_000,
) -> tuple[int, ...]:
    """Build a deterministic causal token sequence without model inference.

    The digest at each position is the unique maximum of an implicit synthetic
    score vector.  Feeding the selected token into the next digest makes the
    oracle causal, while keeping the harness dependency-free and reproducible.
    """

    if token_count < 1:
        raise ValueError("token_count must be positive")
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least 2")
    if sample_index < 0:
        raise ValueError("sample_index cannot be negative")
    state = hashlib.sha256(
        b"mycellios-greedy-oracle-v1" + struct.pack("<QI", seed, sample_index)
    ).digest()
    tokens: list[int] = []
    for position in range(token_count):
        scores = hashlib.sha256(
            state + struct.pack("<I", position) + b"implicit-argmax"
        ).digest()
        token = int.from_bytes(scores[:8], "little") % vocabulary_size
        tokens.append(token)
        state = hashlib.sha256(state + struct.pack("<I", token)).digest()
    return tuple(tokens)


def token_sequence_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(token_ids)))
    for token in token_ids:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("token ids must be integers")
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError("token ids must fit in uint32")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def _draft_matches(
    *,
    seed: int,
    sample_index: int,
    generation: int,
    position: int,
    alpha: float,
) -> bool:
    if alpha <= 0:
        return False
    if alpha >= 1:
        return True
    digest = hashlib.sha256(
        b"mycellios-controlled-draft-v1"
        + struct.pack("<QIII", seed, sample_index, generation, position)
    ).digest()
    draw = int.from_bytes(digest[:8], "little")
    threshold = int(alpha * (1 << 64))
    return draw < threshold


def _wrong_draft_token(
    *,
    seed: int,
    sample_index: int,
    generation: int,
    position: int,
    exact_token: int,
    vocabulary_size: int,
) -> int:
    digest = hashlib.sha256(
        b"mycellios-wrong-draft-v1"
        + struct.pack("<QIII", seed, sample_index, generation, position)
    ).digest()
    offset = 1 + int.from_bytes(digest[:8], "little") % (vocabulary_size - 1)
    return (exact_token + offset) % vocabulary_size


def _build_wave(
    *,
    wave_id: int,
    generation: int,
    start: int,
    draft_count: int,
    verify_seed_tokens: int,
    predecessor_wave_id: int | None,
    oracle_tokens: Sequence[int],
    seed: int,
    sample_index: int,
    alpha: float,
    vocabulary_size: int,
    dispatch_ms: float,
    verify_ms_per_token: float,
    rtt_ms: float,
    bytes_per_token: int,
) -> _Wave:
    if draft_count < 1 or start + draft_count >= len(oracle_tokens):
        raise ValueError("VERIFY wave must reserve one exact bonus target")
    exact_drafts = tuple(
        int(token) for token in oracle_tokens[start : start + draft_count]
    )
    proposal = tuple(
        exact_token
        if _draft_matches(
            seed=seed,
            sample_index=sample_index,
            generation=generation,
            position=start + offset,
            alpha=alpha,
        )
        else _wrong_draft_token(
            seed=seed,
            sample_index=sample_index,
            generation=generation,
            position=start + offset,
            exact_token=exact_token,
            vocabulary_size=vocabulary_size,
        )
        for offset, exact_token in enumerate(exact_drafts)
    )
    all_targets = tuple(
        int(token) for token in oracle_tokens[start : start + draft_count + 1]
    )
    if verify_seed_tokens == 1:
        if predecessor_wave_id is not None:
            raise ValueError("first VERIFY wave cannot have a predecessor")
        raw_targets = all_targets
    elif verify_seed_tokens == 0:
        if predecessor_wave_id is None:
            raise ValueError("continuation VERIFY wave needs a predecessor")
        # The predecessor's final exact target is the missing first target.
        # This wave returns predictions after each provisional draft, ending
        # with its own bonus target.
        raw_targets = all_targets[1:]
    else:
        raise ValueError("verify_seed_tokens must be zero or one")
    input_token_count = len(proposal) + verify_seed_tokens
    service_ms = input_token_count * verify_ms_per_token
    return _Wave(
        wave_id=wave_id,
        generation=generation,
        start=start,
        proposal=proposal,
        raw_targets=raw_targets,
        verify_seed_tokens=verify_seed_tokens,
        predecessor_wave_id=predecessor_wave_id,
        dispatch_ms=dispatch_ms,
        ready_ms=dispatch_ms + rtt_ms + service_ms,
        reserved_bytes=input_token_count * bytes_per_token,
    )


def simulate_verify_conveyor(
    *,
    alpha: float,
    rtt_ms: float,
    k: int,
    window: int,
    output_tokens: int,
    verify_ms_per_token: float,
    bytes_per_token: int,
    inflight_bytes: int,
    seed: int,
    sample_index: int,
    vocabulary_size: int = 32_000,
) -> ConveyorSample:
    """Run one exact event simulation of drain-and-truncate VERIFY semantics."""

    if not math.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha must be finite and between 0 and 1")
    if not math.isfinite(rtt_ms) or rtt_ms < 0:
        raise ValueError("rtt_ms must be finite and non-negative")
    if (
        isinstance(k, bool)
        or isinstance(window, bool)
        or isinstance(output_tokens, bool)
        or min(k, window, output_tokens) < 1
    ):
        raise ValueError("k, window and output_tokens must be positive integers")
    if output_tokens < 2:
        raise ValueError("output_tokens must be at least 2 to calculate TPOT")
    if window > 64:
        raise ValueError("window cannot exceed 64")
    if (
        not math.isfinite(verify_ms_per_token)
        or verify_ms_per_token <= 0
        or isinstance(bytes_per_token, bool)
        or bytes_per_token < 1
        or isinstance(inflight_bytes, bool)
        or inflight_bytes < (min(k, output_tokens - 1) + 1) * bytes_per_token
    ):
        raise ValueError(
            "service time must be positive and the byte cap must fit one "
            "seeded VERIFY wave"
        )
    if sample_index < 0:
        raise ValueError("sample_index cannot be negative")
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must fit in uint64")

    oracle_tokens = synthetic_greedy_oracle(
        seed=seed,
        sample_index=sample_index,
        token_count=output_tokens,
        vocabulary_size=vocabulary_size,
    )
    committed: list[int] = []
    emission_ms: list[float] = []
    inflight: deque[_Wave] = deque()
    inflight_reserved_bytes = 0
    next_speculative_position = 0
    next_dispatch_ms = 0.0
    now_ms = 0.0
    generation = 0
    next_wave_id = 0
    draining = False
    bridge_targets: dict[int, int] = {}
    events: list[ConveyorEvent] = []

    counters = {
        "waves_dispatched": 0,
        "first_seeded_verify_waves": 0,
        "continuation_verify_waves": 0,
        "classic_tail_waves": 0,
        "waves_committed": 0,
        "waves_fully_accepted": 0,
        "waves_rejected": 0,
        "waves_condemned_drained": 0,
        "restart_generations": 0,
        "drain_barriers_started": 0,
        "drain_barriers_completed": 0,
        "max_condemned_waves_per_drain": 0,
        "draft_tokens_verified": 0,
        "verify_input_positions_processed": 0,
        "draft_tokens_compared_on_committed_path": 0,
        "accepted_draft_tokens": 0,
        "discarded_draft_tokens": 0,
        "correction_tokens_committed": 0,
        "bonus_tokens_committed": 0,
        "bonus_bridges_published": 0,
        "bonus_bridges_consumed": 0,
        "max_inflight_waves": 0,
        "max_inflight_bytes": 0,
    }

    while len(committed) < output_tokens or inflight:
        candidate: _Wave | None = None
        candidate_dispatch_ms = max(now_ms, next_dispatch_ms)
        available_draft_positions = output_tokens - next_speculative_position - 1
        if (
            not draining
            and len(committed) < output_tokens
            and available_draft_positions > 0
            and len(inflight) < window
        ):
            current_generation_waves = tuple(
                wave for wave in inflight if wave.generation == generation
            )
            verify_seed_tokens = 0 if current_generation_waves else 1
            predecessor_wave_id = (
                current_generation_waves[-1].wave_id
                if current_generation_waves
                else None
            )
            candidate = _build_wave(
                wave_id=next_wave_id,
                generation=generation,
                start=next_speculative_position,
                draft_count=min(k, available_draft_positions),
                verify_seed_tokens=verify_seed_tokens,
                predecessor_wave_id=predecessor_wave_id,
                oracle_tokens=oracle_tokens,
                seed=seed,
                sample_index=sample_index,
                alpha=alpha,
                vocabulary_size=vocabulary_size,
                dispatch_ms=candidate_dispatch_ms,
                verify_ms_per_token=verify_ms_per_token,
                rtt_ms=rtt_ms,
                bytes_per_token=bytes_per_token,
            )
            if inflight_reserved_bytes + candidate.reserved_bytes > inflight_bytes:
                candidate = None

        next_result_ms = inflight[0].ready_ms if inflight else math.inf
        if candidate is not None and (
            not inflight or candidate.dispatch_ms < next_result_ms
        ):
            inflight.append(candidate)
            inflight_reserved_bytes += candidate.reserved_bytes
            next_speculative_position += candidate.token_count
            next_dispatch_ms = (
                candidate.dispatch_ms
                + candidate.input_token_count * verify_ms_per_token
            )
            next_wave_id += 1
            counters["waves_dispatched"] += 1
            if candidate.verify_seed_tokens == 1:
                counters["first_seeded_verify_waves"] += 1
            else:
                counters["continuation_verify_waves"] += 1
            counters["max_inflight_waves"] = max(
                counters["max_inflight_waves"], len(inflight)
            )
            counters["max_inflight_bytes"] = max(
                counters["max_inflight_bytes"], inflight_reserved_bytes
            )
            events.append(
                ConveyorEvent(
                    "dispatch_verify",
                    candidate.dispatch_ms,
                    candidate.wave_id,
                    candidate.generation,
                )
            )
            continue

        if not inflight:
            if (
                not draining
                and len(committed) == output_tokens - 1
                and next_speculative_position == len(committed)
            ):
                # The native runtime falls back to one classic greedy step
                # because a VERIFY proposal always needs at least one draft
                # plus one bonus target.
                dispatch_ms = max(now_ms, next_dispatch_ms)
                events.append(
                    ConveyorEvent(
                        "dispatch_classic",
                        dispatch_ms,
                        -1,
                        generation,
                    )
                )
                now_ms = dispatch_ms + rtt_ms + verify_ms_per_token
                committed.append(oracle_tokens[len(committed)])
                emission_ms.append(now_ms)
                next_speculative_position = len(committed)
                next_dispatch_ms = dispatch_ms + verify_ms_per_token
                counters["classic_tail_waves"] += 1
                events.append(
                    ConveyorEvent(
                        "resolve_classic",
                        now_ms,
                        -1,
                        generation,
                    )
                )
                continue
            raise RuntimeError("conveyor cannot dispatch or drain a wave")

        wave = inflight.popleft()
        inflight_reserved_bytes -= wave.reserved_bytes
        now_ms = wave.ready_ms
        counters["draft_tokens_verified"] += wave.token_count
        counters["verify_input_positions_processed"] += wave.input_token_count

        if wave.generation != generation or wave.start != len(committed):
            bridge_targets.pop(wave.wave_id, None)
            counters["waves_condemned_drained"] += 1
            counters["discarded_draft_tokens"] += wave.token_count
            events.append(
                ConveyorEvent(
                    "drain_condemned",
                    now_ms,
                    wave.wave_id,
                    wave.generation,
                )
            )
            if draining and not inflight:
                draining = False
                bridge_targets.clear()
                next_dispatch_ms = max(next_dispatch_ms, now_ms)
                counters["drain_barriers_completed"] += 1
                events.append(
                    ConveyorEvent(
                        "drain_complete",
                        now_ms,
                        wave.wave_id,
                        generation,
                    )
                )
            continue

        bridge = (
            bridge_targets.pop(wave.wave_id, None)
            if wave.verify_seed_tokens == 0
            else None
        )
        if wave.verify_seed_tokens == 0:
            counters["bonus_bridges_consumed"] += 1
        targets = wave.target_tokens(bridge)
        accepted = wave.token_count
        for index, (draft, exact) in enumerate(zip(wave.proposal, targets)):
            if draft != exact:
                accepted = index
                break
        successor = (
            inflight[0]
            if inflight
            and inflight[0].generation == generation
            and inflight[0].predecessor_wave_id == wave.wave_id
            else None
        )
        counters["waves_committed"] += 1
        counters["accepted_draft_tokens"] += accepted
        if accepted == wave.token_count:
            bonus = targets[-1]
            if successor is not None:
                resolved = wave.proposal
                bridge_targets[successor.wave_id] = bonus
                counters["bonus_bridges_published"] += 1
            else:
                resolved = (*wave.proposal, bonus)
                counters["bonus_tokens_committed"] += 1
            counters["waves_fully_accepted"] += 1
            counters["draft_tokens_compared_on_committed_path"] += wave.token_count
            events.append(
                ConveyorEvent(
                    "resolve_full",
                    now_ms,
                    wave.wave_id,
                    wave.generation,
                )
            )
        else:
            resolved = (*wave.proposal[:accepted], targets[accepted])
            counters["waves_rejected"] += 1
            counters["correction_tokens_committed"] += 1
            counters["discarded_draft_tokens"] += wave.token_count - accepted
            counters["draft_tokens_compared_on_committed_path"] += accepted + 1
            events.append(
                ConveyorEvent(
                    "resolve_reject",
                    now_ms,
                    wave.wave_id,
                    wave.generation,
                )
            )
            generation += 1
            bridge_targets.clear()
            if len(committed) + len(resolved) < output_tokens:
                counters["restart_generations"] += 1
            if inflight:
                draining = True
                counters["drain_barriers_started"] += 1
                counters["max_condemned_waves_per_drain"] = max(
                    counters["max_condemned_waves_per_drain"],
                    len(inflight),
                )
                events.append(
                    ConveyorEvent(
                        "drain_start",
                        now_ms,
                        wave.wave_id,
                        generation,
                    )
                )

        expected = oracle_tokens[len(committed) : len(committed) + len(resolved)]
        if tuple(resolved) != tuple(expected):
            raise AssertionError("VERIFY resolution diverged from the greedy oracle")
        committed.extend(resolved)
        emission_ms.extend(now_ms for _ in resolved)
        if accepted != wave.token_count or successor is None:
            next_speculative_position = len(committed)
            next_dispatch_ms = max(next_dispatch_ms, now_ms)

    if tuple(committed) != oracle_tokens:
        raise AssertionError("committed sequence diverged from the greedy oracle")
    if inflight_reserved_bytes != 0:
        raise AssertionError("inflight byte credits leaked")
    if draining or bridge_targets:
        raise AssertionError("conveyor ended with unresolved drain or bridge state")

    resource_limits_exact = (
        counters["max_inflight_waves"] <= window
        and counters["max_inflight_bytes"] <= inflight_bytes
    )
    counters.update(
        {
            "configured_window": window,
            "configured_inflight_bytes": inflight_bytes,
            "resource_limits_exact": resource_limits_exact,
            "final_generation": generation,
        }
    )
    return ConveyorSample(
        token_ids=tuple(committed),
        emission_ms=tuple(emission_ms),
        finish_ms=emission_ms[-1],
        drain_complete_ms=now_ms,
        stats=counters,
        events=tuple(events),
    )


def _sum_stat(samples: Sequence[ConveyorSample], name: str) -> int:
    return sum(int(sample.stats[name]) for sample in samples)


def _drain_semantics_exact(sample: ConveyorSample) -> bool:
    if int(sample.stats["drain_barriers_started"]) != int(
        sample.stats["drain_barriers_completed"]
    ):
        return False
    for index, event in enumerate(sample.events):
        if event.kind != "drain_start":
            continue
        completion_index = next(
            (
                candidate
                for candidate in range(index + 1, len(sample.events))
                if sample.events[candidate].kind == "drain_complete"
                and sample.events[candidate].generation == event.generation
            ),
            None,
        )
        if completion_index is None:
            return False
        if any(
            candidate.kind in {"dispatch_verify", "dispatch_classic"}
            and candidate.generation == event.generation
            for candidate in sample.events[index + 1 : completion_index]
        ):
            return False
    return True


def _bridge_semantics_exact(sample: ConveyorSample) -> bool:
    return int(sample.stats["bonus_bridges_published"]) == int(
        sample.stats["bonus_bridges_consumed"]
    )


def _native_semantics_exact(sample: ConveyorSample) -> bool:
    return _drain_semantics_exact(sample) and _bridge_semantics_exact(sample)


def _aggregate_arm(
    *,
    samples: Sequence[ConveyorSample],
    oracle_hashes: Sequence[str],
    alpha: float,
    window: int,
    inflight_bytes: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot aggregate zero samples")
    output_hashes = [token_sequence_sha256(sample.token_ids) for sample in samples]
    parity_exact = list(oracle_hashes) == output_hashes
    accepted = _sum_stat(samples, "accepted_draft_tokens")
    compared = _sum_stat(samples, "draft_tokens_compared_on_committed_path")
    verified = _sum_stat(samples, "draft_tokens_verified")
    verify_positions = _sum_stat(samples, "verify_input_positions_processed")
    discarded = _sum_stat(samples, "discarded_draft_tokens")
    committed = sum(len(sample.token_ids) for sample in samples)
    max_waves = max(int(sample.stats["max_inflight_waves"]) for sample in samples)
    max_bytes = max(int(sample.stats["max_inflight_bytes"]) for sample in samples)
    resource_exact = all(
        bool(sample.stats["resource_limits_exact"]) for sample in samples
    )
    drain_exact = all(_drain_semantics_exact(sample) for sample in samples)
    bridge_exact = all(_bridge_semantics_exact(sample) for sample in samples)
    semantics_exact = drain_exact and bridge_exact
    return {
        "arm": CONTROL_ARM if window == 1 else CONVEYOR_ARM,
        "window": window,
        "success": parity_exact and resource_exact and semantics_exact,
        "parity": {
            "exact": parity_exact,
            "hash_algorithm": "sha256(length_u64_le + token_ids_u32_le)",
            "sequences_checked": len(samples),
            "oracle_sha256_by_sample": list(oracle_hashes),
            "output_sha256_by_sample": output_hashes,
        },
        "draft_quality": {
            "target_alpha_per_draft_token": alpha,
            "accepted_draft_tokens": accepted,
            "compared_draft_tokens_on_committed_path": compared,
            "observed_alpha_on_committed_path": (
                accepted / compared if compared else 0.0
            ),
        },
        "work": {
            "waves_dispatched": _sum_stat(samples, "waves_dispatched"),
            "first_seeded_verify_waves": _sum_stat(
                samples, "first_seeded_verify_waves"
            ),
            "continuation_verify_waves": _sum_stat(
                samples, "continuation_verify_waves"
            ),
            "classic_tail_waves": _sum_stat(samples, "classic_tail_waves"),
            "waves_committed": _sum_stat(samples, "waves_committed"),
            "waves_fully_accepted": _sum_stat(samples, "waves_fully_accepted"),
            "waves_rejected": _sum_stat(samples, "waves_rejected"),
            "waves_condemned_drained": _sum_stat(samples, "waves_condemned_drained"),
            "restart_generations": _sum_stat(samples, "restart_generations"),
            "drain_barriers_started": _sum_stat(samples, "drain_barriers_started"),
            "drain_barriers_completed": _sum_stat(samples, "drain_barriers_completed"),
            "max_condemned_waves_per_drain": max(
                int(sample.stats["max_condemned_waves_per_drain"]) for sample in samples
            ),
            "draft_tokens_verified": verified,
            "verify_input_positions_processed": verify_positions,
            "discarded_draft_tokens": discarded,
            "discarded_work_fraction": discarded / verified if verified else 0.0,
            "verification_amplification_vs_committed_tokens": (
                verify_positions / committed if committed else 0.0
            ),
        },
        "bonus_bridge": {
            "exact": bridge_exact,
            "bonus_tokens_committed": _sum_stat(samples, "bonus_tokens_committed"),
            "bridges_published": _sum_stat(samples, "bonus_bridges_published"),
            "bridges_consumed": _sum_stat(samples, "bonus_bridges_consumed"),
            "semantic": (
                "The first wave includes one already-visible seed. A successor's "
                "first draft is the predecessor's supposed bonus; the "
                "predecessor's final exact target is consumed as its bridge."
            ),
        },
        "drain_barrier": {
            "exact": drain_exact,
            "semantic": (
                "After rejection, no new-generation dispatch occurs until all "
                "condemned in-flight waves have returned and been discarded."
            ),
        },
        "modeled_latency": {
            "ttft_ms": _describe(sample.ttft_ms for sample in samples),
            "time_to_last_token_ms": _describe(sample.finish_ms for sample in samples),
            "drain_complete_ms": _describe(
                sample.drain_complete_ms for sample in samples
            ),
            "tpot_ms": _describe(
                interval for sample in samples for interval in sample.tpot_ms
            ),
            "warning": (
                "Modeled milliseconds from deterministic events; not wall-clock "
                "latency and not physical tokens/s."
            ),
        },
        "high_water": {
            "configured_wave_limit": window,
            "observed_max_inflight_waves": max_waves,
            "configured_inflight_bytes": inflight_bytes,
            "observed_max_inflight_bytes": max_bytes,
            "exact": resource_exact,
        },
    }


def build_benchmark_report(
    *,
    alphas: Sequence[float] = (0.6, 0.8, 0.9, 0.95, 0.99, 1.0),
    rtts_ms: Sequence[float] = (0.0, 25.0, 50.0, 100.0, 200.0),
    draft_lengths: Sequence[int] = (1, 2, 4, 8),
    windows: Sequence[int] = (1, 2, 4, 8, 16),
    output_tokens: int = 128,
    samples: int = 7,
    verify_ms_per_token: float = 1.0,
    bytes_per_token: int = 256 * 1024,
    inflight_bytes: int = DEFAULT_INFLIGHT_BYTES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    normalized_alphas = tuple(float(value) for value in alphas)
    normalized_rtts = tuple(float(value) for value in rtts_ms)
    normalized_k = tuple(int(value) for value in draft_lengths)
    normalized_windows = tuple(int(value) for value in windows)
    if (
        not normalized_alphas
        or any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in normalized_alphas
        )
        or tuple(sorted(set(normalized_alphas))) != normalized_alphas
    ):
        raise ValueError("alphas must be unique, increasing and in [0, 1]")
    if (
        not normalized_rtts
        or any(not math.isfinite(value) or value < 0 for value in normalized_rtts)
        or tuple(sorted(set(normalized_rtts))) != normalized_rtts
    ):
        raise ValueError("rtts_ms must be unique, increasing and non-negative")
    if (
        not normalized_k
        or any(value < 1 for value in normalized_k)
        or tuple(sorted(set(normalized_k))) != normalized_k
    ):
        raise ValueError("draft_lengths must be unique increasing positive integers")
    if (
        not normalized_windows
        or normalized_windows[0] != 1
        or any(value < 1 or value > 64 for value in normalized_windows)
        or tuple(sorted(set(normalized_windows))) != normalized_windows
    ):
        raise ValueError("windows must be unique, increasing, start at 1 and be <= 64")
    if output_tokens < 2 or samples < 1:
        raise ValueError("output_tokens must be >= 2 and samples must be positive")
    if not math.isfinite(verify_ms_per_token) or verify_ms_per_token <= 0:
        raise ValueError("verify_ms_per_token must be finite and positive")
    if bytes_per_token < 1:
        raise ValueError("bytes_per_token must be positive")
    largest_seeded_wave = min(max(normalized_k), output_tokens - 1) + 1
    if inflight_bytes < largest_seeded_wave * bytes_per_token:
        raise ValueError("inflight_bytes must fit the largest seeded VERIFY wave")
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must fit in uint64")

    oracle_by_sample = [
        synthetic_greedy_oracle(
            seed=seed,
            sample_index=sample_index,
            token_count=output_tokens,
        )
        for sample_index in range(samples)
    ]
    oracle_hashes = [token_sequence_sha256(token_ids) for token_ids in oracle_by_sample]
    cases: list[dict[str, Any]] = []
    for alpha in normalized_alphas:
        for rtt_ms in normalized_rtts:
            for k in normalized_k:
                arms: list[dict[str, Any]] = []
                for window in normalized_windows:
                    sample_runs = [
                        simulate_verify_conveyor(
                            alpha=alpha,
                            rtt_ms=rtt_ms,
                            k=k,
                            window=window,
                            output_tokens=output_tokens,
                            verify_ms_per_token=verify_ms_per_token,
                            bytes_per_token=bytes_per_token,
                            inflight_bytes=inflight_bytes,
                            seed=seed,
                            sample_index=sample_index,
                        )
                        for sample_index in range(samples)
                    ]
                    arms.append(
                        _aggregate_arm(
                            samples=sample_runs,
                            oracle_hashes=oracle_hashes,
                            alpha=alpha,
                            window=window,
                            inflight_bytes=inflight_bytes,
                        )
                    )
                control = arms[0]
                control_completion_p50 = float(
                    control["modeled_latency"]["time_to_last_token_ms"]["p50"]
                )
                control_tpot_p95 = float(control["modeled_latency"]["tpot_ms"]["p95"])
                for arm in arms:
                    completion_p50 = float(
                        arm["modeled_latency"]["time_to_last_token_ms"]["p50"]
                    )
                    tpot_p95 = float(arm["modeled_latency"]["tpot_ms"]["p95"])
                    arm["comparison_vs_w1"] = {
                        "modeled_completion_speedup": (
                            control_completion_p50 / completion_p50
                        ),
                        "modeled_tpot_p95_delta_ms": tpot_p95 - control_tpot_p95,
                        "warning": (
                            "A scheduler-model comparison only; it cannot be "
                            "translated into physical tok/s."
                        ),
                    }
                cases.append(
                    {
                        "alpha": alpha,
                        "rtt_ms_emulated": rtt_ms,
                        "draft_length_k": k,
                        "success": all(arm["success"] for arm in arms),
                        "arms": arms,
                    }
                )

    return {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "success": all(case["success"] for case in cases),
        "claim_boundary": {
            "physical_hosts": 0,
            "physical_gpu": False,
            "model_inference_executed": False,
            "socket_or_loopback_io_executed": False,
            "wan_measured": False,
            "rtt_is_an_input_not_a_measurement": True,
            "verify_service_time_is_an_input_not_a_measurement": True,
            "physical_tokens_per_second_claimed": False,
            "statement": (
                "This artifact validates native conveyor scheduling, exact "
                "drain-and-truncate parity and resource accounting only."
            ),
        },
        "oracle": {
            "type": "causal_synthetic_greedy_argmax_surrogate",
            "real_model_logits": False,
            "hash_algorithm": "sha256(length_u64_le + token_ids_u32_le)",
            "parity_is_mandatory": True,
        },
        "native_verify_semantics": {
            "first_wave": (
                "One already-visible seed followed by k provisional drafts; "
                "a full W=1 result can commit k drafts plus one bonus."
            ),
            "continuation_wave": (
                "Starts with the predecessor's supposed bonus as its first "
                "draft. The predecessor's final exact target is the bridge "
                "used to verify that position."
            ),
            "rejection": (
                "Commit accepted prefix plus correction, condemn descendants, "
                "drain every tombstone, then dispatch the new generation."
            ),
        },
        "configuration": {
            "alphas": list(normalized_alphas),
            "rtts_ms_emulated": list(normalized_rtts),
            "draft_lengths_k": list(normalized_k),
            "windows": list(normalized_windows),
            "output_tokens_per_sample": output_tokens,
            "samples_per_arm": samples,
            "verify_ms_per_token_emulated": verify_ms_per_token,
            "bytes_per_token_reserved": bytes_per_token,
            "inflight_bytes_per_request": inflight_bytes,
            "seed": seed,
            "seed_hex": f"0x{seed:016x}",
        },
        "reproduction": {
            "working_directory": "GPU Distribuida-native repository root",
            "powershell": [
                (
                    "$python = & 'scripts/resolve-distribution-python.ps1' "
                    "-WorkspacePath (Get-Location).Path"
                ),
                "$env:PYTHONPATH='python'",
                (
                    "& $python -m distributed_runtime.verify_conveyor_benchmark "
                    "--json-out benchmarks/verify-conveyor-ab.json"
                ),
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cases": cases,
        "limits": [
            "No model, GPU, socket, NIC, multi-host route or physical WAN is exercised.",
            "Alpha is injected per draft token with a deterministic hash draw.",
            "RTT and verification service time are deterministic inputs.",
            "TPOT includes zero-ms gaps when one VERIFY result commits several tokens.",
            "A rejection blocks restart dispatch until every condemned wave is drained.",
            "The output must not be presented as measured or physical tokens/s.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A/B the exact native VERIFY conveyor against W=1 with deterministic "
            "RTT, alpha, k and W sweeps."
        )
    )
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=parse_alphas("0.6,0.8,0.9,0.95,0.99,1.0"),
    )
    parser.add_argument(
        "--rtts-ms",
        type=parse_rtts_ms,
        default=parse_rtts_ms("0,25,50,100,200"),
    )
    parser.add_argument(
        "--draft-lengths",
        type=parse_positive_int_sweep,
        default=parse_positive_int_sweep("1,2,4,8"),
    )
    parser.add_argument(
        "--windows",
        type=parse_windows,
        default=parse_windows("1,2,4,8,16"),
    )
    parser.add_argument("--output-tokens", type=_positive_int, default=128)
    parser.add_argument("--samples", type=_positive_int, default=7)
    parser.add_argument(
        "--verify-ms-per-token",
        type=_positive_float,
        default=1.0,
    )
    parser.add_argument(
        "--bytes-per-token",
        type=_positive_int,
        default=256 * 1024,
    )
    parser.add_argument(
        "--inflight-bytes",
        type=_positive_int,
        default=DEFAULT_INFLIGHT_BYTES,
    )
    parser.add_argument("--seed", type=_seed, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    if args.output_tokens < 2:
        parser.error("output-tokens must be at least 2 to calculate TPOT")
    largest_seeded_wave = min(max(args.draft_lengths), args.output_tokens - 1) + 1
    if args.inflight_bytes < largest_seeded_wave * args.bytes_per_token:
        parser.error("inflight-bytes must fit the largest seeded VERIFY wave")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_benchmark_report(
        alphas=args.alphas,
        rtts_ms=args.rtts_ms,
        draft_lengths=args.draft_lengths,
        windows=args.windows,
        output_tokens=args.output_tokens,
        samples=args.samples,
        verify_ms_per_token=args.verify_ms_per_token,
        bytes_per_token=args.bytes_per_token,
        inflight_bytes=args.inflight_bytes,
        seed=args.seed,
    )
    return {
        **result,
        "generated_at_unix_seconds": time.time(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact_json else 2,
        sort_keys=True,
    )
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

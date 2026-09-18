# Roadmap

Mycellios advances through evidence gates, not calendar promises. A gate closes
only when its required artifact is reproducible from a named source revision.
The exact capability boundary remains in
[`STATUS_AND_EVIDENCE.md`](STATUS_AND_EVIDENCE.md).

## Now — prove the native route

**Goal:** generate through two distinct physical computers using the native
contiguous pipeline.

Required outcome:

- the same pinned build and model identity on both machines;
- non-empty layer ranges assigned to distinct machine identities;
- a real generation canary with parity, TTFT, TPOT, throughput and memory;
- explicit transport mode and directed-link observations;
- verified process cleanup after success and failure.

Until this is committed, work should remove a blocker from the
[`two-host quickstart`](TWO_HOST_QUICKSTART.md), improve its diagnostics or make
its evidence more trustworthy. New product surfaces are out of scope.

## Current candidate — existing interface and runtime reliability

The local public-quality candidate prepares independently implemented changes
to model availability and error states, stream completion/cancellation, account
session isolation, worker/activation lifecycle, Windows runtime discovery and
authenticated local checkpoint control. Queued checkpoint restores must not
execute after their caller has cancelled or timed out. Existing stored
idempotency keys must retain same-account retry protection during an upgrade.
Release preparation must use the deployment policies actually shipped in the
public repository, and installed adapter metadata must resolve independently
of a caller's working directory.
Review these edits against the existing
contracts; record their acceptance and validation only in
[`STATUS_AND_EVIDENCE.md`](STATUS_AND_EVIDENCE.md). They do not close the physical
gate above.

The reference comparison was refreshed on 2026-09-18. c0mpute and shard were
unchanged from the previous review; AntSeed advanced to the revision below.
Source provenance and existing adaptations are recorded in
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

| Reference snapshot | Pattern to apply independently | Acceptance target |
| --- | --- | --- |
| [c0mpute `fcd4690`](https://github.com/leyten/c0mpute/blob/fcd4690fb6b19af51e801069ce6d35ae93b6db7a/app/earn/page.tsx) | Distinct preparation, waiting and working states, with availability for the selected model. | Check loading, empty, disconnected and ready states at desktop/narrow widths; preserve the draft on failure and make the next action clear. |
| [shard `fcf7280`](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/tests/test_wire_hardening.py) | Bounded transport failures and reproducible execution evidence. | Reject malformed/oversized frames before allocation; verify stream termination, cancellation and cleanup without treating truncated output as success. |
| [AntSeed `aef3462`](https://github.com/AntSeed/antseed/blob/aef3462a52d45b4d6734e230f86c0ecd01cb1be5/apps/desktop/src/renderer/ui/hooks/tee-auto-verification.ts) | Bind status evidence to its session/generation and retain retry backoff across discovery refreshes. | A stale success cannot revive a disconnected or reassigned route; newer failures invalidate readiness without discarding useful user input. |

Acceptance rules for the existing native delivery path:

- Run the staged package in a clean process without relying on a developer
  checkout or package manager; require every runtime/helper dependency to be
  present and imports to leave unrelated services stopped. AntSeed's
  [bundling fix](https://github.com/AntSeed/antseed/blob/aef3462a52d45b4d6734e230f86c0ecd01cb1be5/apps/desktop/scripts/prepare-dist.mjs)
  and separate [standalone entry](https://github.com/AntSeed/antseed/blob/aef3462a52d45b4d6734e230f86c0ecd01cb1be5/apps/ants/src/bin.ts)
  motivate this check. Mycellios outcomes and reproduced defects belong in the
  status and evidence document, independently of the external comparison.
- Keep completion driven by the protocol. Do not infer it from answer wording
  or synthesize tool calls to keep a request alive.
- Keep peer-provided names separate from application-owned readiness or trust
  indicators; preserve readable Unicode and keyboard navigation.

Speculative decoding, topology tuning and cache/prefetch changes remain
performance hypotheses until the physical gate identifies a bottleneck and a
same-model, same-topology comparison preserves parity while measuring
TTFT/TPOT, throughput, memory and transferred bytes. Third-party benchmark
numbers are not Mycellios evidence. This candidate adds no payment, blockchain
or TEE feature track.

## Next — reliability under failure

**Entry condition:** the two-host native route is reproducible.

- Repeat the route enough times to report distributions rather than one sample.
- Intentionally lose a remote stage and prove drain, replacement or recovery.
- Make placement react to measured memory and directed-link observations.
- Reduce operator steps that caused failed or irreproducible campaigns.

## Then — measured performance

**Entry condition:** ordinary failure and cleanup behavior are reproducible.

- Optimize the measured TTFT/TPOT bottleneck instead of projected bottlenecks.
- Demonstrate a model that needs the combined memory of multiple machines.
- Validate supported GPU execution shapes on real, distinct hosts.
- Publish repeatable LAN/WAN profiles with hardware and software identities.

## Later — productization

**Entry condition:** the native runtime has a defensible reliability and
performance envelope.

- Stabilize public protocols and publish a compatibility policy.
- Extract independently owned packages only where release boundaries require it.
- Add signed releases, support windows and operator upgrade/rollback guidance.
- Expand contribution surfaces only when the runtime can safely absorb them.

## Engineering health track

Repository quality progresses alongside the physical gates:

1. keep contracts acyclic and every source area explicitly owned;
2. prevent known oversized modules from growing;
3. extract new behavior from `coordinator/server.ts`, `landing/src/Panel.tsx`
   and `python/distributed_runtime/engine.py` behind tested interfaces;
4. move to `apps/` and `packages/` only after those interfaces are real and the
   two-host gate protects behavior during the migration.

This track must not delay a safe physical run merely to improve folder names.

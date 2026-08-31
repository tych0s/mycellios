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

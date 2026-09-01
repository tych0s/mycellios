# Community, physical-gate and performance hardening

## Problem

Mycellios has a broad native runtime and strong automated coverage, but its
community entry path, web performance policy and two-host operational path are
not yet one coherent, reproducible product journey. Organizational work must
make the physical two-computer gate easier to run and verify, not replace it.

## Actors

- Evaluator: understands status and runs safe local validation.
- Contributor: configures a supported toolchain and selects bounded work.
- Two-host operator: diagnoses and runs a physical native campaign.
- Maintainer: reviews architecture, evidence, performance and releases.

## Desired behavior

1. The active diff, spec and verification claims describe the same candidate.
2. Web budgets cover every JavaScript chunk and the actual eager/high-priority
   critical path, with a deliberate source-map publication policy.
3. A non-mutating two-host preflight reports stable, redacted diagnostics in
   human and JSON formats and rejects false physical evidence.
4. Linux/macOS contributors receive actionable capability diagnostics rather
   than raw PowerShell failures.
5. Refactors are incremental, characterized and reduce recorded debt.
6. README/development/CI provide deterministic paths for each actor.
7. Runtime performance claims remain blocked until immutable physical evidence
   exists for two distinct computers.

## Constraints

- npm and one `package-lock.json`; no workspaces or top-level migration yet.
- Contracts live in `src/contracts`; Python data plane remains under
  `python/distributed_runtime`; orchestration remains under `src/distribution`.
- No new marketplace, payment, plugin ecosystem or product surface.
- No competitor code, dependency or public comparison. Research observations
  stay in this excluded spec and are translated only into neutral decisions.
- Preserve unrelated worktree changes. No commit, push, release or deployment.
- Physical tests unavailable in this environment are `unavailable` or skipped.

## Reference research (internal only)

- AntSeed: clear contributor quickstart and explicit apps/packages/plugin map;
  useful onboarding pattern, but not a reason to migrate this repository now.
- Mesh-LLM: strong install/release/CI surface; useful automation pattern, but its
  workflow count would be excessive for Mycellios today.
- Shard: evidence receipts and status boundaries; useful proof discipline only.
- c0mpute: architectural separation is observable, but missing licensing and
  weaker community governance make it unsuitable as an implementation source.
- Shard revision `fcf728096948c7686bcf0897e9acb75d1abda1d5` remains
  Apache-2.0 and its receipt verifier rejects zero-work attestations before
  coverage settlement. Mycellios already has stronger typed envelopes, pinned
  Ed25519 keys, complete range coverage and activation-root chaining; only the
  zero-work/final-failure invariant is worth adapting now.
- c0mpute revision `f78b8857749d0896fd0f4b5577099746b7610cdb`
  declares its license as TBD, so none of its code is eligible for reuse.

## Acceptance criteria

- Closed bundle policy, negative fixtures and non-public production sourcemaps.
- Two-host preflight and physical receipt contract reject loopback, duplicate
  identity, empty ranges, build/model drift and incomplete cleanup.
- Platform capabilities appear in `npm run doctor -- --json`.
- At least one bounded UI extraction remains within bundle budgets; coordinator
  extraction proceeds only if characterized and low risk.
- Architecture debts have owner/target metadata and Python import policy is
  explicit and only-decreasing.
- Contributor journeys, Node pin, editor defaults and CI concurrency are clear.
- Required static/type/build gates pass; hardware-only gates remain unavailable.
- A repository-wide defect pass covers tracked source and configuration through
  compiler, architecture, policy, test and build gates; every reproducible
  project defect found is fixed with a regression test where practical.
- Environment or hardware limitations are separated from code defects and no
  physical or rendered claim is inferred from an unavailable gate.
- External open-source comparison results in no opaque dependency or copied
  architecture. Any adapted invariant has a compatible license, pinned source
  revision, explicit attribution and an adversarial regression test.
- A final execution receipt cannot settle a failed or zero-work stage.
- The licensed-feature inventory distinguishes existing equivalents from real
  gaps; importing code is never used as a substitute for that comparison.
- A stage activation can be challenged with an unpredictable commit-first
  projection seed and judged without requiring Torch in the control plane.
- Malformed, mismatched-seed, mismatched-shape and numerically divergent
  activation sketches fail closed.
- Python stage producers and the TypeScript control plane share one versioned
  activation-sketch wire contract and deterministic cross-language vectors.
- Physical evidence may bind a validated activation-integrity verdict, but its
  absence does not become a passing proof and no enforcement claim is made.

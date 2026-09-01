# Plan

## Delivery order

### 0. Reconcile

Inventory the dirty tree, exclude generated output and unrelated local work,
and align this spec/checklist with implementation evidence. Establish the
strongest baseline available in the current Node/Python environment.

### 1. Honest web measurement

Extend the Vite-manifest budget into a closed chunk policy. Model route-critical
bytes from static imports plus HTML high-priority preloads. Budget heavyweight
shared/optional chunks explicitly. Disable publicly served source maps while
retaining debuggability through deliberate CI artifacts later.

### 2. Two-host golden path

Reuse the existing auto-distribute schema, launch/campaign contracts and signed
physical evidence rather than inventing a second campaign format. Add a pure
preflight contract and CLI before any side effect. Bind distinct nodes,
non-loopback endpoints, exact build/model, non-empty layer ranges, configured
authentication, runtime capability and cleanup requirements. Keep output
redacted and error codes stable.

### 3. Portability

Extend the doctor with platform and command capabilities. Add portable Node
dispatch only around the highest-value physical setup/preflight entry points;
Windows-only actions fail with actionable status. Do not emulate hardware.

### 4. Incremental refactor

Characterize and extract one additional cohesive Panel unit, then re-run visual
and bundle QA. A coordinator extraction is optional and proceeds only if it is
pure and lifecycle-neutral. Python engine work is deferred until physical data.

### 5. Community path

Reorganize existing canonical docs by actor and command family, add editor/Node
pins, verify templates and labels rather than promising them, and describe a
small set of GPU-free tasks. Adopt formatting only through a narrow check that
does not mass-rewrite existing code.

### 6. CI

Add PR-safe concurrency and separate early static signals from expensive test
and build work while retaining protected check names. Validate workflows with
actionlint. Coverage is informative only after a stable baseline.

### 7. Runtime performance

Document the benchmark taxonomy now. Defer optimization until the first valid
two-host receipt identifies a bottleneck; then make one reversible A/B change.

### 8. Repository-wide defect pass

Inventory every tracked source/configuration area and run the broadest safe
static, type, policy, test, build and dependency checks. Triage each failure as
a reproducible project defect, an intentional physical skip, or an environment
limitation. Fix reproducible defects narrowly, add regression coverage, then
repeat the affected gate and the integrated suite. Do not mass-reformat, move
top-level areas or touch unrelated worktree changes during this audit.

### 9. Licensed external comparison

Compare current upstream source revisions against the existing Mycellios
contracts rather than their README claims alone. Reject unlicensed c0mpute code
and avoid importing another engine. Adapt only Shard's Apache-2.0 fail-closed
zero-work receipt invariant into the existing typed Mycellios receipt chain,
also rejecting a terminal receipt containing a failed stage. Preserve source
revision and copyright in `THIRD_PARTY.md`; add focused adversarial tests and
run the complete contract and repository gates.

## Security and privacy

- Preflight never prints tokens, secret values or unnecessary local paths.
- Physical identity is a stable public node identifier plus build binding, not
  a credential or raw machine account name.
- Receipt validation fails closed and cannot elevate loopback evidence.
- Browser QA uses the managed project preview, never localhost as evidence.

## Decisions

- Current repository topology and npm remain unchanged.
- Raw bytes are deterministic budgets; critical-path bytes are computed from
  manifest dependencies and explicit high-priority HTML preloads.
- Production source maps are hidden from static output unless a later release
  pipeline deliberately uploads them to a private artifact store.
- Existing signed physical-gate evidence is extended/reused, not duplicated.
- Node 24/Python 3.12 CI is authoritative for native packaging and offline
  runtime tests. This worker runs Node 22; a managed Python 3.12 interpreter is
  available, but the reviewed runtime wheel set is not installed locally.

## Verification

Focused negative policy tests; structure/docs/architecture/component gates;
all typechecks; builds and closed bundle checks; offline Python when available;
npm audit; actionlint; managed rendered desktop/mobile QA. Physical campaigns
remain unavailable until two distinct machines are supplied.

## Environment-limited evidence

- Managed desktop and mobile browser QA was attempted twice through the project
  preview and returned HTTP 502 from the preview host after Cloudflare succeeded.
- Local actionlint is unavailable because neither the binary nor Docker exists;
  the pinned `GitHub Actions syntax` public CI job performs that validation.
- The standalone offline Python command reaches Python 3.12 but cannot import
  the locked runtime dependencies. The full JavaScript suite and build pass;
offline Python remains owned by the dependency-installing public CI job.

The optional Graphify report is absent and the repository exposes no npm
Graphify command. Repository architecture remains governed by the checked-in
TypeScript/Python import analyzers; npm-only policy prevents invoking the
skill's generic pnpm fallback.

## Defect-pass outcome

The canonical check exposed a portability defect in native Python AST analysis:
it ignored the explicit `python3.12` command and uv-managed interpreters even
though the developer doctor and project policy support Python 3.12. The resolver
now tries an explicit override, platform-native 3.12 commands and a cached,
absolute uv-managed interpreter. Doctor reports the same managed capability.
Focused regression tests, the complete 1,836-test inventory, build, syntax
parsers and source/configuration policies pass without a temporary PATH shim.

## External comparison decision

Mycellios already implements the reusable engine-level ideas visible upstream:
authenticated framed transport, ordered replay fencing, contiguous stage
artifacts, speculative execution, recovery, per-stage signed receipts, complete
coverage and chained activation roots. Importing another engine would duplicate
contracts and weaken provenance. The bounded accepted change is settlement
hardening: completed/recovered observations must attest positive frames and
bytes, and a final receipt must contain no failed observation.

### Licensed capability completion

The source-level matrix against Shard revision
`fcf728096948c7686bcf0897e9acb75d1abda1d5` treats measured-link topology
planning, upload-sensitive transfer costs, stage-role constraints,
content-addressed resumable artifacts, speculation, recovery,
timeouts/watchdogs and signed coverage receipts as existing Mycellios
capabilities. Re-importing them would create duplicate authorities.

The remaining reusable security primitive is Shard's Apache-2.0 commit-first
activation sketch. Adapt it into the Python data plane: fresh seed,
domain-separated SHA-256 commitment, CPU-derived sample indexes for
cross-device consistency, bounded projection, cosine similarity and relative
norm checks. Comparison remains Torch-free and fails closed. This delivers a
verification seam, not a physical enforcement claim; settlement wiring waits
for a two-host route and a trusted recompute role.

c0mpute revision `f78b8857749d0896fd0f4b5577099746b7610cdb` remains
ineligible for reuse while unlicensed. Shard revision, source file, copyright
and modification status remain explicit in `THIRD_PARTY.md`.

Implementation outcome: `activation_integrity.py` is a shipped native Python
entrypoint and remains importable on a Torch-free verifier. Six focused Python
tests cover commitments, deterministic unique indexes, bounds, tolerated drift,
wrong activations and malformed/mismatched sketches. Native packaging now
contains 52 allowlisted files. The integrated JavaScript suite remains green at
1,821 passed and 16 physical skips; build, bundle policies, typechecks,
architecture, documentation, formatting and production audit also pass.

### Activation-integrity control-plane seam

Mirror the versioned sketch envelope and fail-closed comparison in
`src/contracts`, using the same domain separation, seed commitment, numerical
bounds and thresholds as the shipped Python entrypoint. Pin cross-language
vectors so either side changing its commitment or index derivation cannot
silently fork the protocol. Extend physical-gate evidence with an optional,
fully validated activation-integrity verdict whose challenge, suspect/trusted
stage bindings and metrics are included under the existing Ed25519 signature.

The optional field records real evidence when a trusted recompute exists; it is
not synthesized by the campaign and is not required for today's gate. A later
schema version may require it only after the two-host producer and trusted role
are operational. Reject the alternative of spawning Python from receipt
verification: signature checking and import must remain deterministic and
self-contained in the TypeScript control plane.

Implementation outcome: the TypeScript contract pins the Python commitment and
projection-index vectors, independently recomputes every verdict and rejects a
forged metric before a physical receipt can be signed. Existing receipts remain
valid because activation-integrity evidence is optional and absence has no
positive meaning. The integrated suite is green at 1,825 passed and 16 physical
skips; six focused Python tests, the complete build, bundle budgets, all
typechecks, architecture gates and the production audit also pass.

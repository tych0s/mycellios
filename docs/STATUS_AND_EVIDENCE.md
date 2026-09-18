# Status and evidence

Updated: 2026-09-18 (public-quality validation).

## Implemented and connected

- model profiling, certified adapters and contiguous range planning;
- deterministic runtime manifests and launch descriptions;
- local and authenticated HTTP launch agents;
- staged startup, readiness, canary, registration and cleanup;
- streaming API and persistent runtime transport;
- direct-channel selection with bounded relay fallback;
- recovery orchestration and execution traces;
- commit-first activation-integrity sketches with a Torch-free control-plane
  verifier and optional binding inside signed physical receipts;
- physical campaign, evidence import and gate-report tooling.

## Verified in automated or single-host runs

- selective Llama and Qwen-family pipelines;
- real subprocesses and sockets on one host;
- CPU pipeline parity and greedy kill-and-resume recovery;
- local direct/relay transport behavior and rejection paths;
- tensor/expert cell contracts and supported CPU fixtures;
- coordinator, worker, activation, security and packaging policies.

The normal test suite validates software behavior. Opt-in physical tests are
skipped when their required model, GPU or host environment is unavailable.

The local development gate enrolls signed workers in its temporary coordinator
database. Its two workers are logical processes on one computer. Cleanup
deadlines remain active until completion and cleanup failures return nonzero.

Activation checkpoint capture/restore retains an owner-only Unix socket on
POSIX. The local candidate adds authenticated loopback control on Windows,
using a signed endpoint descriptor and a per-launch secret. Focused Windows
tests exercised a real TypeScript/Python capture/restore round trip. This
establishes the local control path, not remote recovery after losing a machine.
Windows configuration writes flush the payload before atomic rename;
directory fsync is available only on POSIX.

## Public-quality validation

These changes were validated locally against public main `0e3347f`. Publishing
the source does not establish deployment or physical-host evidence. The
comparison and improvement plan are in
[`ROADMAP.md`](ROADMAP.md), with exact external revisions and source provenance
in [`THIRD_PARTY.md`](../THIRD_PARTY.md).

The changes prevent cross-account session access and retain retry protection
for new and pre-update requests. They cancel work while waiting for capacity or
reading an SSE stream, reject late worker starts and prevent activation during
shutdown. The interface retains drafts and partial
answers, distinguishes intentional stop from failure, preserves selected models
when availability changes, and keeps coordinator connectivity separate from
inference capacity. Studio draft persistence, publication versioning, hot reload
and narrow-screen layout have regression coverage.

The strict Python suite passes: 1079 tests run, 1061 passed, 18 skipped, no
failures or errors. Rank entrypoints no longer import the dense model loader
solely for type annotations. Checkpoint requests still queued after timeout,
disconnect or shutdown cannot mutate stage state later. The CPU collective
fixture keeps its rendezvous port bound throughout process startup, removing
the observed Windows address-in-use race without relaxing parity or timeouts.
An additional Unix-only transport case is skipped on Windows; the original
platform/model prerequisite skips remain separate from passing evidence.
An additional focused checkpoint run passes 16 cases with one Unix-only skip.
It reproduces and fixes slow fragmented input monopolizing the serial control
listener: one 30-second deadline starts at accept and includes framing, queueing
and response. Both unauthenticated slow headers and authenticated slow payloads
expire, after which the same server accepts a legitimate capture.

The complete JavaScript suite passes: 274 test files and 1872 tests passed, with
6 files and 20 tests skipped for their declared prerequisites. The final run
includes the launcher and legacy-idempotency changes. Subsequent package
corrections pass six release-policy tests and ten registry/catalog tests.

Release staging exposed requirements for two optional service files absent from
the public checkout. It now requires the two repository-owned native policies,
and tests copy the real deployment files instead of inventing empty fixtures.
The storage policy now uses the node-update directory variable consumed by the
runtime. Installed adapter metadata resolves from the package, independently
of an external importer's entrypoint or working directory. Optional Content Hub
and Supabase configuration remain available through their existing settings.
No deployed configuration or stored data was migrated during these checks.

The final production build and sealed coordinator staging pass. Fresh child
processes resolve only staged dependencies, serve the staged landing/mobile
assets, expose loopback health with the matching build identity, and exit with
their listeners closed. Both the server library and actual coordinator CLI
entrypoint are exercised. On Windows the CLI check invokes its installed
SIGTERM handler; it does not claim a physical operating-system signal test.

The native development gate completed real CPU inference through two signed
logical workers and two stages on this computer, with matching canary/routed
output and drained resources. Browser checks covered home, chat and Studio at
desktop/narrow widths, including light/dark cancellation, partial errors,
unavailable capacity and draft persistence against an isolated API fixture.
Neither the fixture nor the native local run is evidence of two physical hosts.
The complete npm dependency audit reported zero known vulnerabilities.

## Not yet demonstrated by committed evidence

- one inference route spanning two physical computers;
- a model that requires the combined memory of several hosts;
- GPU tensor or expert parallelism across machines;
- remote recovery after physically losing a stage;
- P50/P95 latency and sustained throughput on residential LAN/WAN;
- a reproducible comparison showing that Mycellios outperforms another system.
- activation-integrity enforcement using a distinct trusted recompute stage on
  a physical route (the contract exists, but no such receipt is committed).

## Evidence policy

A physical claim is accepted only when the artifact records at least:

- commit/build and model artifact identities;
- distinct machine identities and assigned layer ranges;
- transport mode and directed link measurements;
- exact canary/parity result;
- TTFT, TPOT, tokens per second and peak memory;
- process cleanup, plus the outcome of an intentional failure when recovery is
  claimed.

Existing files in `benchmarks/` are retained local/loopback evidence. They are
useful regression baselines, but none proves a two-computer route.

## Benchmark taxonomy

Physical reports separate control overhead, process startup, model loading,
activation transfer, TTFT, TPOT, sustained tokens/s, peak memory, cleanup and
recovery. Every result is bound to an exact source/build, operating systems,
hardware and drivers, model revision, assigned ranges, topology and transport.
An optimization comparison uses the same model and topology on both sides and
must retain output parity; loopback measurements cannot establish a multi-host
performance claim.

## Next gate

Run the native contiguous pipeline on two computers using
[`TWO_HOST_QUICKSTART.md`](TWO_HOST_QUICKSTART.md). Do not add a new engine or a
new product surface before that run identifies a concrete missing capability.

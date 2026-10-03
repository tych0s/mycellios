# Development

## Prerequisites

- Node.js 24 or newer
- npm 10 (the exact version is pinned in `package.json`)
- Python 3.12 and the runtime dependencies needed by the selected executor

## Install and verify

```bash
npm run doctor
npm ci
npm run check
npm run build
```

If another npm major is installed, `npx --yes npm@10.9.8 run doctor` runs the
pinned tool without changing the global installation. Use the same prefix for
the other npm commands.

Use focused commands while iterating:

```bash
npm test -- --run tests/<area>.test.ts
python -u scripts/run-python-tests.py
npm run verify:architecture
npm run verify:component-boundaries
npm run verify:bundle
```

`npm run doctor -- --json` provides machine-readable diagnostics for setup
scripts and support requests. Python 3.12 is required for full runtime work; a
missing Python interpreter is reported as a partial control-plane setup rather
than a successful full-runtime setup. Discovery checks `MYCELLIOS_PYTHON`, the
prepared runtime (including Windows `Scripts/python.exe`), and system/managed
interpreters. An incompatible Python earlier on PATH does not hide a prepared
Python 3.12 runtime.

Physical tests are opt-in and must not be presented as passed when skipped:

```bash
npm run test:physical-launcher
npm run test:physical-external-cell
npm run test:physical-recovery
npm run test:physical-gpu-cell
```

The Python runner fails on every failure or import error, including without
`--strict`; that flag remains accepted for existing callers. Skips are reported
separately. The small FP32 CPU cell fixtures compare each coordinate against its
token's activation scale with a fixed `32 * float32.eps` budget. Independent
FP64 reference measurements showed reduction-order roundoff in both dense and
sharded paths. This fixture budget is not a GPU accuracy claim; exact cache,
fork, promotion and alias checks remain in place, with separate corruption
rejection tests for the numerical helper.

The two cache/fork regression fixtures use fan-in-scaled projection matrices.
Unit-variance random matrices amplified rounding across repeated decode steps
enough to exceed the budget on a hosted CI CPU. Scaling the fixture inputs keeps
the same seeds, dimensions, state transitions and fixed acceptance budget; it
does not change runtime arithmetic or certify arbitrary ill-conditioned weights.

On Windows, point tools at the prepared runtime when Python is not installed
system-wide:

```powershell
$env:MYCELLIOS_PYTHON = (Resolve-Path runtime/distribution-venv/python.exe).Path
npm test -- --maxWorkers=1
& $env:MYCELLIOS_PYTHON -u scripts/run-python-tests.py --strict
npm run dev:e2e:distributed -- --runtime runtime/distribution-venv --timeout 120
```

The development gate needs its selected model in the Hugging Face cache. A
local validation used `HMellor/tiny-random-LlamaForCausalLM` at revision
`9408c553e5c189a7dcdc5a5dbd2feb476b061759`. Prepare the snapshot in the gate's
cache before an offline run. Random tiny-model output validates routing and
cleanup, not answer quality or physical multi-host performance.

Set `MYCELLIOS_DEV_E2E_HF_HOME` to an explicitly prepared Hugging Face cache when
validating an isolated checkout. Relative paths resolve from that checkout.
Copied snapshot symlinks must still resolve to blobs inside the same model
repository, or be materialized as regular files. Cross-repository links are
rejected by the checkpoint containment guard. `MYCELLIOS_DEV_E2E_DEBUG=1` enables
coordinator diagnostics for investigating preparation and worker disconnects.
Keep native runs, builds and the full test suite sequential on memory-constrained
machines; the example's `--maxWorkers=1` limits test concurrency.

## Native release asset assembly

The native CI workflow uploads one artifact per target. Download those three
artifacts into a private directory with the names
`mycellios-node-linux-x64`, `mycellios-node-macos-arm64` and
`mycellios-node-windows-x64`. GitHub strips their common `build/` prefix, so
each artifact directory must directly contain `node-package/`, `evidence/`
and its package file. The upload step must use `include-hidden-files: true`:
the staged manifest includes dotfiles and, on macOS, NumPy's `.dylibs` runtime
libraries. A download missing them must fail layout verification. Then assemble
one candidate set:

```bash
npm run node:release:assemble -- \
  --artifacts=build/ci-artifacts \
  --output=build/native-release-assets \
  --version=0.2.80 \
  --revision=<exact-CI-commit-SHA> \
  --source-id=sha256:<exact-source-ID> \
  --published-at=2026-09-29T00:00:00.000Z
```

The assembler verifies the three staged layouts, artifact/evidence hashes,
source identity and CI run/attempt. It copies the packages into a new directory
with `mycellios-node-latest.json` and `sha256sums.txt`. It refuses a mixed CI
run or changed package bytes. The output is a **candidate**, even if the
evidence says `signatureState: signed`: that field alone does not verify a
cryptographic signature. Verify platform signatures and the release identity
independently before preparing a public transaction or uploading any package.
The `Native node build` workflow also performs this assembly after its three
platform jobs and uploads a candidate manifest; it does not publish packages.

## Browser contribution pilot

### Browser registration recovery check

Build the browser bundle, then run the isolated fixture:

```bash
npm run mobile:build
npx tsx tests/serve-browser-recovery-integration.ts
```

It binds only `127.0.0.1:18910`, uses an in-memory coordinator and serves
`/browser/`. Its `/lab/` routes are test controls and are never attached to the
production coordinator. Start one browser worker and record `/lab/state`.
POST JSON `{}` to `/lab/drop` closes the test transport and expires its
disconnected registration. The browser should renew its signed registration
automatically, retain the same client identity and pass GPU/CPU admission again.

Use `/lab/hold` before a drop to hold the next admission challenge. Pause the
browser while it is recovering, then `/lab/release`; it must stay paused with
no registered worker. `/lab/deny` rejects subsequent challenges with HTTP 401;
the browser must stop instead of bypassing admission. In a fresh fixture, start
two tabs sharing the same origin: the older session must pause when superseded.
`/lab/remove` must pause the current worker without an automatic re-registration
loop. Use `/lab/shutdown` and close the test tabs after capturing results.
These are browser lifecycle tests, not physical distributed inference evidence.

### Browser decoder layer gate

For an exact local Qwen3 or Llama safetensors snapshot, export and verify the selected
layer, then publish it to the coordinator. Use an origin with an internal
administration token; the publisher reads `MYCELLIOS_INTERNAL_TOKEN` or
`MYCELLIOS_INTERNAL_TOKEN_FILE`. The generated bridge file contains no token.

```bash
python tests/run-browser-qwen3-layer-onnx.py --architecture <qwen3|llama> \
  --model <snapshot> --layer 0 --output runtime/layer-0.onnx \
  --fixture runtime/layer-0-fixture.json
python scripts/publish-browser-qwen3-layer.py --model <snapshot> \
  --graph runtime/layer-0.onnx --fixture runtime/layer-0-fixture.json \
  --coordinator-url <origin> --config-output runtime/browser-layer-bridge.json
```

For another model family, supply an ONNX graph and a
`mycellios-browser-layer/2` manifest with its exact model and graph digests,
declared tensor names/shapes, state outputs and a native-output canary. Register
a trusted `BrowserLayerAdapter` in the native process before stage creation,
then use `scripts/publish-browser-layer.py --model <snapshot> --graph <onnx>
--manifest <json> --coordinator-url <origin> --config-output <json>`. The
publisher checks the graph's declared inputs/outputs and canary numerically.

Set `MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE` to the absolute path of that generated
JSON in the selected Python stage environment. The selected layer must belong
to that stage's range. Keep the coordinator origin and artifact identity bound
to the same model snapshot. Run `tests/run-browser-layer-network.py` with the
generated config and fixture for a local browser/coordinator check, and
`tests/run-browser-layer-full-model.py` for the full-model regression on the
cached Qwen3-0.6B checkpoint. The latter fixture uses layer 0 and 28 layers.
For the two-range local regression, export and publish layer 1, then run
`tests/run-browser-layer-split-model.py` with its generated bridge config.

For a physical browser/native check, keep the fixture and browser on one
computer and run `tests/run-browser-layer-full-model.py` on the other through
an authenticated loopback SSH forward. Publish a verified real checkpoint
layer first. Copy the native source package and its `model_adapter_registry.json`,
and preserve the checkpoint's canonical `snapshots/<revision>` layout when
materializing cached files: a flat local copy has a different model identity.
Check source, configuration and weight hashes across hosts before execution.
Record host names, browser backend, exact layer, token parity, error bounds and
cleanup. The fixture's injected-request-failure case is not a real browser
disconnect; exercise actual Pause separately if claiming transport loss.
Close the fixture and forward and remove ephemeral test tokens afterwards.

For automatic discovery of an already published layer, configure the trusted
native stage host once with `MYCELLIOS_BROWSER_LAYER_COORDINATOR_URL` and
`MYCELLIOS_INTERNAL_TOKEN` or `MYCELLIOS_INTERNAL_TOKEN_FILE`. The origin must
use HTTPS, except for local loopback tests. The local auto-distribution agent
and remote launch daemon pass this explicit grant to their isolated Python
stages. At stage startup, the runtime looks up the active artifact by exact
model digest and assigned layer range, then attaches one compatible decoder
layer. The coordinator prefers an admitted WebGPU browser, validates the
model-specific numerical canary, and tries another eligible browser if that
canary or load fails. A disconnected or failed browser falls back to the
native layer. The explicit bridge file above remains available for a pinned
single-artifact run and takes precedence over discovery.

An operator must still export and publish a verified graph for each supported
model checkpoint before starting its stage. A browser click alone cannot turn
an unexported architecture, an unsupported ONNX operator, or an insufficient
device into a usable model layer. The browser may receive no layer when these
requirements are unmet. `--auto-discovery` on
`tests/run-browser-layer-full-model.py` verifies the lookup route without
passing a per-model bridge file to the stage.


`/browser/` admits a worker after the user presses **Start contributing** and
keeps that tab visible. For a regular Qwen3 model, copy
`config/browser-dense-bridge.example.json` to a private local file, set its
exact active model identity, coordinator HTTPS origin and one assigned layer,
then set `MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE` to its absolute path in the
Python stage environment. The standard Torch `StageRunner` publishes that
layer's dense SiLU MLP and uses two admitted browsers when present. It keeps
the complete native MLP for loss of browser residency or execution. This
currently applies to the Torch safetensors Qwen3 stage, not the GGUF backend.
Published weights and live activations reach untrusted browser devices; enable
this only for models and workloads allowed to leave the native stage.

For routed models, an operator can opt one SiLU expert from a
RAM-backed Qwen3/GLM4 MoE stage into browser placement. Copy
`config/browser-moe-bridge.example.json` to a private local file, then replace
the zero digest and example URL with the active model's exact SHA-256 identity
and the coordinator HTTPS origin. Choose an actual sparse `layer` and `expert`.
The five latency/bandwidth fields and two available VRAM budgets must come
from the intended hosts and devices; the example values are placeholders.
The budgets describe memory available to this expert mesh after other model
allocations. Keep `MYCELLIOS_INTERNAL_TOKEN` or
`MYCELLIOS_INTERNAL_TOKEN_FILE` in the native stage environment and set
`MYCELLIOS_BROWSER_MOE_BRIDGE_FILE` to the absolute path of the private JSON
file on the stage host. Never put the token in either JSON file or a public URL.

The stage publishes the exact float32 expert and retains its local RAM copy.
The planner selects the browser only when its profile and live residency favor
it. The coordinator requires two distinct, visible, admitted browser workers
and compares their outputs. A missing browser returns to the native expert.
The bridges currently cover one Qwen3 dense MLP or one routed MoE expert, not
an arbitrary transformer layer or the complete model on a phone. The browser
worker remains available without an operator bridge, but then receives only
its admission check unless another compatible model job is published.

For a local end-to-end check, run `npm run mobile:build`, start
`npx tsx tests/serve-browser-split-probe.ts`, open two independent browser
profiles at `http://127.0.0.1:8770/browser/`, press **Start contributing** in
both, and run `tests/run-browser-moe-generation.py` with `PYTHONPATH=python`
and `MYCELLIOS_INTERNAL_TOKEN=local-browser-split-probe` in its process
environment. The script creates a tiny model, compares complete token IDs,
checks that both browsers verified real expert work, and writes an ignored
local receipt under `runtime/`. Loopback and two browser profiles on one PC
remain a software check, not physical multi-host evidence.
`tests/run-browser-dense-generation.py` does the equivalent for a tiny dense
Qwen3 model; `--expect-fallback` verifies the native path after one browser is
paused. Both scripts generate test fixtures and leave production unchanged.
`tests/run-browser-qwen3-real-layer.py --snapshot <absolute-local-snapshot>`
compares one browser-owned layer against native Torch using an already cached
Qwen3 0.6B safetensors checkpoint; it does not generate a full answer.

## Development rules

- Preserve fail-closed schemas and identity checks.
- Do not replace measured values with projections or simulator output.
- Keep secrets in environment variables; examples contain environment-variable
  names, never credentials.
- Add a spec only for active medium/large work. Remove it after its durable
  decisions have been incorporated into canonical docs.
- Update `STATUS_AND_EVIDENCE.md` only when an implementation or immutable
  evidence artifact changes the stated boundary.
- Do not commit generated runtime state, model caches or local credentials.
- Follow the ownership and dependency rules in
  [`REPOSITORY_STRUCTURE.md`](REPOSITORY_STRUCTURE.md).
- Reduce a module's recorded ceiling in `config/module-size-budget.json` after
  extracting code; never raise a ceiling to make the architecture gate pass.

## Useful entry points

### Control plane

```bash
npm run dev:coordinator
npm run dev:worker
npm run dev:launch-agent -- --node-id node-a
```

### Two-host preparation and evidence

```bash
npm run preflight:two-host -- --config <file> --json
npm run model:auto-distribute -- --config <file> --prepare-only
npm run diagnostics:remote -- --help
```

The preflight is portable and non-mutating. The current setup, stage and GPU
campaign scripts ending in `.ps1` are Windows entry points. On Linux and macOS,
use the portable preflight and Python runtime commands; Windows-only operations
are unavailable rather than silently simulated.

### Quality and architecture

```bash
npm run verify:structure
npm run verify:docs
npm run verify:format
npm run verify:architecture
npm run verify:component-boundaries
npm run verify:bundle
```

### Safe tasks without a GPU

- Add a negative fixture for a fail-closed schema or CLI parser.
- Improve a stable diagnostic while preserving its error code.
- Reduce one tracked module below its recorded ceiling with characterization.
- Fix a canonical documentation link or command verified by the docs gate.
- Add a platform capability check to the dependency-free developer doctor.

These are candidate task shapes, not a promise that a GitHub label exists.
Check the live issue labels before referring contributors to `good first issue`.

The HTTP contract is in [`openapi.yaml`](openapi.yaml). Architecture changes
must also update [`ARCHITECTURE.md`](ARCHITECTURE.md).

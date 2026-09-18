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
local validation used `hmellor/tiny-random-LlamaForCausalLM` at revision
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

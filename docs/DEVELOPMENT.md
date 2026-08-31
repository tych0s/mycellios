# Development

## Prerequisites

- Node.js 24 or newer
- npm
- Python 3 and the runtime dependencies needed by the selected executor

## Install and verify

```bash
npm run doctor
npm ci
npm run check
npm run build
```

Use focused commands while iterating:

```bash
npm test -- --run tests/<area>.test.ts
python scripts/run-python-tests.py
npm run verify:architecture
npm run verify:native-boundary
npm run verify:bundle
```

`npm run doctor -- --json` provides machine-readable diagnostics for setup
scripts and support requests. Python 3.12 is required for full runtime work; a
missing Python interpreter is reported as a partial control-plane setup rather
than a successful full-runtime setup.

Physical tests are opt-in and must not be presented as passed when skipped:

```bash
npm run test:physical-launcher
npm run test:physical-external-cell
npm run test:physical-recovery
npm run test:physical-gpu-cell
```

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

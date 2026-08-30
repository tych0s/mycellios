# Mycellios

Mycellios is an experimental runtime for running one language model across
multiple heterogeneous computers. It assigns contiguous model ranges to nodes,
keeps each stage's cache close to its compute, streams activations between
stages and exposes the resulting route through a familiar streaming API.

The project is built around a simple goal: let several ordinary machines act
as one useful inference system without pretending that simulated or single-host
results are proof of a real network.

## Status

The native end-to-end path is implemented:

- model profiling and certified family adapters;
- automatic range planning and artifact preparation;
- local and authenticated remote launch agents;
- supervised stage startup, health checks and cleanup;
- direct runtime transport with controlled relay fallback;
- streaming inference, canaries and model registration;
- recovery contracts and evidence capture.

The software path is extensively tested on one host. A versioned result from
two physical computers is still the next release gate. Until that gate passes,
Mycellios does not claim measured multi-host throughput, WAN latency or remote
recovery.

See [Status and evidence](docs/STATUS_AND_EVIDENCE.md) for the exact boundary.

## How it fits together

```text
client
  │  POST /v1/chat/completions
  ▼
coordinator ── plans and activates one sealed route
  │
  ▼
root stage ──► middle stage(s) ──► tail stage
  ▲                                  │
  └──────── tokens / state ──────────┘
```

Every node runs Mycellios code. The coordinator owns admission, planning and
route lifecycle; launch agents start only pre-authorized commands; Python stage
processes load their assigned range and exchange activations over persistent
connections.

Read [Architecture](docs/ARCHITECTURE.md) for the component boundaries.

## Quick validation

Requirements: Node.js 24+, npm and Python 3 with the dependencies appropriate
for the runtime being tested.

```bash
npm ci
npm run typecheck
npm test
npm run build
```

Run the control-plane demonstration:

```bash
npm run demo
```

Prepare a deterministic two-node route without starting processes:

```bash
npm run model:auto-distribute -- \
  --config config/auto-distribute.two-host.example.json \
  --prepare-only
```

The example deliberately uses placeholder addresses and must be adapted to the
two machines. Continue with [Two-host quickstart](docs/TWO_HOST_QUICKSTART.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `src/coordinator` | API, admission, activation and route lifecycle |
| `src/distribution` | profiling, planning, manifests and launch supervision |
| `src/transport` | direct and relayed runtime channels |
| `src/worker` | node enrollment and execution tunnels |
| `python/distributed_runtime` | physical model stages and executors |
| `tests`, `python/tests` | software and opt-in physical verification |
| `config` | safe examples and physical campaign templates |
| `benchmarks` | retained reproducible evidence artifacts |
| `docs` | current operational documentation |

## Documentation

- [Documentation map](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Two-host quickstart](docs/TWO_HOST_QUICKSTART.md)
- [Status and evidence](docs/STATUS_AND_EVIDENCE.md)
- [Development](docs/DEVELOPMENT.md)
- [Repository structure](docs/REPOSITORY_STRUCTURE.md)
- [Security](docs/SECURITY.md)
- [OpenAPI contract](docs/openapi.yaml)

## Project direction

The immediate priority is reliability, not feature count:

1. make the two-host recipe reproducible;
2. capture exact parity, memory, TTFT, TPOT and throughput;
3. prove cleanup and recovery after a stage disappears;
4. then optimize scheduling, transport and larger-model support using measured
   bottlenecks.

New claims require committed evidence. New architecture work requires a clear
failure or measured limitation in the native path.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [Development](docs/DEVELOPMENT.md).
Project decisions follow [GOVERNANCE.md](GOVERNANCE.md), notable changes are in
[CHANGELOG.md](CHANGELOG.md), and security reports follow
[SECURITY.md](SECURITY.md).

## License

Mycellios is licensed under [GPL-3.0-only](LICENSE). Third-party notices are in
[THIRD_PARTY.md](THIRD_PARTY.md).

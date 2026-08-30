# Repository structure

Mycellios is currently a **single versioned npm workspace with several product
surfaces**, not a collection of independently released packages. This is
intentional while the runtime, applications and release tooling still share
contracts and one release identity.

## Owned areas

| Area | Owner and purpose |
| --- | --- |
| `src/contracts` | Stable schemas, identities and protocol types shared by all TypeScript surfaces |
| `src/coordinator` | API, persistence, admission and route lifecycle |
| `src/distribution` | Profiling, planning, manifests, launch and physical gates |
| `src/transport` | Direct and relayed runtime channels |
| `src/worker`, `src/node` | Enrolled execution nodes and installed native service |
| `python/distributed_runtime` | Native model execution data plane |
| `landing` | Public web and operator interface |
| `src/mobile` | Installable mobile contribution surface |
| `scripts` | Build, release and verification entry points—not reusable product code |
| `tests`, `python/tests` | Automated and opt-in physical verification |
| `config` | Versioned schemas, policies and safe examples |
| `benchmarks` | Small retained evidence artifacts, never generated build output |
| `deploy`, `container`, `sidecars` | Isolated packaging or OS integration contexts |

## Dependency direction

```text
web / mobile / node / coordinator
                 │
                 ▼
       contracts and domain modules
                 │
                 ▼
 distribution orchestration ──► Python runtime
```

- Contracts do not import applications, UI or infrastructure.
- Runtime orchestration does not depend on UI components.
- Applications may compose domain modules but should not duplicate protocol
  schemas.
- Build and deployment scripts call product entry points; product code does not
  import build scripts.

## Package manager

The supported path is Node.js 24+, npm and the root `package-lock.json`:

```bash
npm ci
npm run verify:structure
```

Nested package manifests under isolated container/OS build contexts are not npm
workspace members and must not influence root dependency resolution.

## Path toward a true package monorepo

Directory movement is deferred until after the two-host physical gate. A safe
future extraction proceeds by dependency direction:

1. `packages/contracts` for protocol-only schemas and types;
2. `packages/runtime-control` for reusable planning and activation logic;
3. `apps/coordinator`, `apps/web` and `apps/mobile`;
4. isolated packaging workspaces only when they need independent dependency
   graphs or releases.

Each extraction must preserve import boundaries, build provenance, a single
lockfile and the native release gates. Moving folders without those boundaries
would create the appearance of a monorepo without its benefits.

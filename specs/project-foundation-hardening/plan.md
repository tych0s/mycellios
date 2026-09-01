# Plan

## Current evidence

- Canonical documentation is concise, but has no roadmap document.
- The component model leaves `adapters`, `benchlab`, `economy`, `simulator` and
  `support` unclassified.
- The TypeScript import graph has one strongly connected component:
  `worker-hub` → `engine-profile-authority` →
  `engine-runtime-profile-scheduler` → `worker-hub`.
- The largest modules include `engine.py` (7,387 lines), `server.ts` (6,077),
  `Panel.tsx` (5,671) and several 2,000–4,000 line runtime modules.
- Graphify output is not configured in this npm repository, so the in-repo
  architecture verifier will be the reproducible dependency-graph authority.

## Approach

1. Add a canonical evidence-gated roadmap and link it from the documentation
   map and repository entry point.
2. Complete the component ownership map for all current TypeScript areas.
3. Move the engine runtime challenge request type to its protocol contract,
   removing the coordinator-only type dependency and the import cycle.
4. Add a dependency graph and module-size ratchet. Known oversized files are a
   finite debt list whose line ceilings may only stay equal or decrease.
5. Run structural, architectural, type and focused tests before wider checks.

## Decisions

- Prefer a ratchet over a mass refactor. Large modules are risky but stable;
  new behavior must be extracted while their ceilings decrease over time.
- Keep the current top-level layout until the physical gate. Package movement
  follows real contract boundaries, not aesthetics.
- Treat benchmarks and simulators as tooling, adapters and economy as engine
  code, and support as shared domain code.
- Do not add an external graph dependency for a check that can be implemented
  deterministically from TypeScript relative imports.

## Verification

- `npm run verify:structure`
- `npm run verify:component-boundaries`
- `npm run verify:architecture`
- `npm run typecheck`
- Focused component, contract and structure tests
- Full test/build only after the focused foundation is green

# Plan

## Decision

The repository remains a single npm package with multiple product surfaces
until the native two-host gate passes. This is more honest than a cosmetic
workspace split: `landing`, `src`, tests and release scripts still share source
contracts and one version.

After the gate, extraction can proceed in dependency order:

1. shared contracts and protocol schemas;
2. coordinator/runtime libraries;
3. web and mobile applications;
4. packaging and infrastructure adapters.

## Changes

- Remove pnpm lock/workspace files and declare npm in `package.json`.
- Add `AGENTS.md` as concise repository operating rules.
- Add `docs/REPOSITORY_STRUCTURE.md` and link it from canonical docs.
- Add a zero-dependency structure verifier and focused tests.
- Add the verifier to the public quality workflow.

## Verification

- `npm run verify:structure`
- focused verifier tests
- TypeScript typecheck and existing policy tests
- documentation link verification
- `git diff --check` for staged and unstaged changes

## Rejected alternatives

- Immediate directory migration to `apps/*` and `packages/*`: high churn with no
  product evidence benefit before the physical gate.
- Keeping both npm and pnpm: two locks can resolve different dependency graphs
  and make release provenance ambiguous.
- Adding Turborepo/Nx: unnecessary until independently versionable packages
  actually exist.

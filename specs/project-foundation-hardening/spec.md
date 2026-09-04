# Project foundation hardening

## Problem

Mycellios has a tested native runtime and a concise public documentation set,
but its repository health is not yet governed as rigorously as its runtime
behavior. The active direction has no canonical roadmap, some source areas are
outside the component-boundary model, a coordinator import cycle exists, and
very large modules can continue growing without an explicit reduction policy.

## Desired outcome

Contributors can understand what matters next, where code belongs, and which
architectural constraints every change must preserve. Automated checks reject
new dependency cycles, unowned source areas, unsupported documentation and
uncontrolled growth of known oversized modules.

## Actors

- Users evaluating the project's actual maturity.
- Contributors changing the TypeScript control plane or Python data plane.
- Maintainers preparing public candidates and physical evidence campaigns.

## Acceptance criteria

1. A single canonical roadmap connects current evidence to ordered release
   gates without promising unverified capabilities or calendar dates.
2. Every TypeScript product/tooling area has an explicit architectural owner.
3. The current TypeScript dependency cycle is removed and new cycles fail an
   automated architecture check.
4. Existing oversized modules have explicit non-growth budgets; new oversized
   modules fail the same check.
5. Repository, architecture, type and focused test gates pass.
6. Durable documentation explains how large modules will be reduced without a
   high-risk directory rewrite before the two-host gate.

## Constraints

- The two-physical-host native route remains the immediate product gate.
- npm and the single root lockfile remain authoritative.
- No new product surface, engine, runtime dependency or cosmetic monorepo move.
- Capability claims remain exclusive to `docs/STATUS_AND_EVIDENCE.md`.
- Existing unrelated worktree changes are preserved.
- Physical tests unavailable in this environment are reported as unavailable,
  not as passing.

## Non-goals

- Splitting every large file in one change.
- Moving the repository to `apps/` and `packages/` before runtime evidence.
- Redesigning the landing, mobile UI or coordinator product behavior.
- Claiming multi-host performance without committed physical evidence.

# Repository foundation

## Problem

Mycellios contains several product surfaces and runtimes but is still compiled
as one package. The repository also carries pnpm workspace files while every
supported build, CI and release path uses npm. Top-level ownership and rules are
implicit, so future work can reintroduce documentation and package-manager
drift.

## Desired outcome

Make the current repository intentional and enforceable without moving runtime
code before the two-host physical gate. Establish npm as the only JavaScript
package manager, document component ownership and the extraction path toward a
real monorepo, and add a fast structural verifier used by contributors and CI.

## Acceptance criteria

- Exactly one JavaScript package manager and root lockfile are canonical.
- The root manifest declares the package manager and exposes a structural gate.
- Repository areas and dependency direction are documented.
- Agent/contributor rules prevent new top-level clutter and duplicated docs.
- The structural gate fails on stale workspace files, duplicate docs or missing
  canonical areas.
- Existing runtime paths and imports are not moved in this phase.

## Non-goals

- Splitting the coordinator, landing and mobile surfaces into packages now.
- Changing runtime behavior or public APIs.
- Introducing a task runner or new dependency.
- Committing or publishing the work.

## Constraints

- Preserve the documentation reset already in progress.
- Preserve unrelated `.jarvis-dev.json` and `specs/repair-local-dev-runner/`
  changes.
- Use Node.js built-ins for the verifier.

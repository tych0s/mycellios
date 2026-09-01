# Public release readiness

## Objective

Publish a repository that is understandable, independently verifiable and safe
to contribute to without depending on private infrastructure.

## Requirements

- Preserve the native Mycellios product boundary and its canonical docs.
- Provide public CI for quality, security and native packaging.
- Pin third-party GitHub Actions to immutable revisions.
- Publish clear governance, ownership and project metadata.
- Keep third-party notices limited to software actually distributed or required.
- Generate the public repository only through the reproducible history filter.
- Certify the exact public candidate from a fresh anonymous clone.

## Non-goals

- Moving the repository into `apps/` and `packages/`.
- Refactoring large runtime modules before the two-host gate.
- Claiming physical multi-host results that have not been recorded.

## Acceptance

- Local structure, typechecks, tests, build, audit and secret scan pass.
- The public candidate contains only the allowlisted public workflows.
- The candidate contains no forbidden private, personal or competitive residue.
- Public `main` matches the certified candidate SHA.

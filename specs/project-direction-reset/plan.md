# Plan

## Canonical information architecture

- `README.md`: product entry point and fastest path to validation.
- `docs/README.md`: documentation map and source-of-truth rules.
- `docs/ARCHITECTURE.md`: components and end-to-end execution flow.
- `docs/TWO_HOST_QUICKSTART.md`: first physical multi-host campaign.
- `docs/STATUS_AND_EVIDENCE.md`: implemented, locally verified and physically
  unproven capabilities.
- `docs/DEVELOPMENT.md`: contributor workflow and verification.
- `docs/SECURITY.md`: runtime trust boundary and operational safety.
- `docs/openapi.yaml`: API contract.

## Cleanup strategy

Remove documents whose value is historical rather than operational: dated
research, handoffs, provider experiments, competitive material, resolved issue
reports, projections and duplicated designs. Remove completed feature specs,
checkpoints and obsolete publication specs; retain the public-history sanitation
procedure and specifications directly consumed by verification tooling.

Update references in code/configuration when a retained implementation points
to a removed document. Avoid behavior changes.

## Verification

- Search for broken local Markdown links and stale documentation paths.
- Run documentation/policy tests and the normal TypeScript checks.
- Run `git diff --check`.
- Verify unrelated `.jarvis-dev.json` and `specs/repair-local-dev-runner/`
  changes remain outside this work.

## Decision

The repository will describe measured reality, not accumulated ambition. New
design documents belong in a spec while work is active; completed decisions are
folded into canonical documentation or removed.

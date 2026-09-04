# Project direction reset

## Problem

Mycellios mixes current product guidance with historical research, handoffs,
experiments, provider-specific pilots, completed delivery specs and duplicated
status reports. A new contributor cannot reliably tell what the product is,
what already works, what remains unproven, or how to run the next decisive
experiment.

## Desired outcome

The repository presents one clear direction: Mycellios is an independent
runtime for distributed LLM inference across heterogeneous computers. The next
product milestone is a reproducible two-host physical run using the native
engine already present in the repository.

## Acceptance criteria

- The root README explains the product, honest status, architecture, quick
  validation and the next physical milestone without historical narrative.
- Current documentation has one obvious index and no competing status sources.
- A runnable two-host guide names every prerequisite and step without claiming
  evidence that has not been captured.
- Historical research, obsolete handoffs, provider pilots, competitive notes,
  duplicated plans and completed delivery specs no longer clutter the active
  documentation surface.
- References from code and verification tooling to retained canonical documents
  remain valid.
- Existing unrelated worktree changes are untouched.

## Non-goals

- Changing the runtime architecture or inference behavior.
- Claiming multi-host performance before a physical run exists.
- Integrating or copying a competing implementation.
- Publishing or pushing changes.

## Constraints

- Preserve the reproducible public-history sanitization machinery.
- Preserve API documentation and documentation required by automated policy
  checks.
- Keep Mycellios implementation and terminology independent.

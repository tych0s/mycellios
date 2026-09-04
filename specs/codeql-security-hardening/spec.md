# CodeQL security hardening

## Problem

The public repository is operationally certified but CodeQL reports 51 open
alerts across HTTP abuse protection, untrusted URL fetching, secret hashing,
client identifiers, escaping, error responses and one unsafe regular
expression. These findings weaken the public security posture even where an
existing control is functionally equivalent.

## Desired behavior

- Every coordinator HTTP route has a centrally enforced, bounded rate limit.
- Browser workers fetch expert manifests only from the expected same-origin
  endpoint for the offered artifact.
- API keys are stored with a password-hardening KDF and legacy hashes migrate
  after successful authentication.
- Client identifiers use cryptographically secure randomness.
- Error responses never expose internal exception details.
- Escaping and parsing perform one bounded, unambiguous transformation.
- The public candidate passes its existing quality, security and native-build
  workflows and has no open CodeQL alerts caused by the changed code.

## Non-goals

- No new product surface or distributed-runtime architecture change.
- No claim that physical multi-host validation has been completed.
- No unrelated refactor of the large coordinator modules.

## Acceptance criteria

1. Focused security tests cover each changed control and pass.
2. Repository structure, typechecks, full tests and build pass.
3. A regenerated anonymous public candidate contains no forbidden material or
   secrets and its GitHub workflows pass.
4. CodeQL reanalysis closes the 51 current alerts, or any genuinely invalid
   finding is documented with concrete evidence before dismissal.

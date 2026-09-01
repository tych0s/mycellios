# Plan

## Approach

- Register the official Fastify rate-limit plugin globally at coordinator
  startup, retaining narrower domain limits where they already exist.
- Validate mobile expert manifest URLs against origin, pathname and artifact
  identity before issuing a fetch.
- replace raw SHA-256 API-key storage with versioned scrypt records; accept and
  upgrade legacy records only after a constant-time successful comparison.
- Replace insecure ID fallbacks, multi-pass escaping, exception reflection and
  the calculator regex with bounded helpers.
- Add focused regression tests and regenerate the public history through the
  existing reproducible sanitizer.

## Security decisions

- Use a broad per-IP coordinator ceiling as defense in depth, not as a
  substitute for account, webhook, concurrency or websocket limits.
- Preserve compatibility for already issued API keys through opportunistic
  migration; do not invalidate users silently.
- Reject absolute, cross-origin and mismatched manifest paths instead of
  attempting URL normalization.

## Verification

- Focused Vitest suites for coordinator security, API access, mobile runtime,
  formatting/parsing and benchmark server behavior.
- `npm run verify:structure`, all typechecks, `npm test`, `npm run build`, and
  production dependency audit.
- Public-candidate historical scan, Gitleaks/CodeQL and all public workflows.

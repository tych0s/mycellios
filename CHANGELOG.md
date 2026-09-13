# Changelog

All notable user-facing changes will be recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Updated Fastify, Vitest and the transitive xmldom parser to patched versions.
- Made Python test failures fatal by default; CPU cell parity now uses a
  per-token FP32 roundoff budget backed by reference measurements and corruption
  rejection tests, while cache and ownership checks remain exact.
- Enrolled local development workers through signed admission and made cleanup
  deadlines keep the process alive until success or a nonzero failure result.
- Fixed Windows configuration writes, artifact cache traversal and process
  termination fallback; report Unix checkpoint control as unsupported on Windows.
- Consolidated active documentation around the native two-host release gate.
- Added repository structure checks and autonomous public quality workflows.
- Hardened the reproducible public-history boundary.
- Added an evidence-gated roadmap, documentation link validation and
  architecture health checks for ownership, cycles and module-size growth.
- Split web surfaces into route-level bundles and began reducing the oversized
  operator panel behind tested module boundaries.

## 0.2.77

Current experimental release baseline.

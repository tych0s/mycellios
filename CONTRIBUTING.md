# Contributing to Mycellios

Thank you for helping build exact distributed inference across heterogeneous
devices. Mycellios is both a research laboratory and a native runtime, so a
useful contribution must make clear which side of that boundary it changes.

## Before opening a pull request

For substantial behavior or protocol changes, open a proposal issue first. Bug
fixes, tests and documentation corrections can usually go directly to a pull
request. Security findings must follow [SECURITY.md](SECURITY.md), never a public
issue.

Describe the contribution as one or more of:

- production runtime or control plane;
- simulator, planner or research evidence;
- native packaging/release infrastructure;
- documentation or user interface.

Do not present simulated, single-host or mocked results as physical multi-host
evidence. Experimental external runtimes may be documented as research, but
must not become selectable production backends or packaged dependencies.

## Development setup

The baseline is Node.js 24, npm's committed lockfile and Python 3.12 for the
physical runtime.

```bash
npm ci
npm run verify:structure
npm run typecheck
npm run landing:typecheck
npm run mobile:typecheck
npm test
npm run build
```

Run the offline Python suite with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run-python-tests.py
```

The Python dependencies used in CI are installed from the hash-pinned lock in
`scripts/wheel-locks/linux-x64-cp312.txt`. Tests must not download model weights
or call external services implicitly.

Changes to the native node, workers, contracts, packaging or release scripts
also need the relevant checks from `.github/workflows/node-build.yml`. Physical
GPU/multi-PC campaigns are intentionally opt-in: attach the generated evidence
when your claim depends on real hardware, but do not make ordinary pull requests
depend on equipment that GitHub-hosted runners cannot reproduce.

## Pull request requirements

- Keep each pull request focused and explain the user-visible or architectural
  effect.
- Add or update tests for behavior changes and include reproduction steps for
  bugs.
- Preserve the native product boundary and fail closed on identity, provenance
  and artifact mismatches.
- Update canonical documentation when a protocol, security boundary, supported
  platform or verified claim changes.
- Never commit credentials, private endpoints, personal data, model weights or
  generated runtime caches.
- Record exact hardware, software versions and source revision for benchmark or
  physical-validation claims.
- Use conventional, imperative commit messages where practical.

By contributing, you agree that your contribution is licensed under the
repository's `GPL-3.0-only` license.

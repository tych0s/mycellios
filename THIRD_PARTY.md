# Third-party software

Unless a file states otherwise, Mycellios source code is licensed under the
GNU General Public License, version 3 only (`GPL-3.0-only`).

Mycellios uses open-source libraries for model loading, tensor execution,
transport and its user interfaces. These dependencies do not replace the
Mycellios planner, scheduler or distributed runtime.

The exact JavaScript dependency graph is recorded in `package-lock.json`. The
portable Python runtime is pinned by platform under `scripts/wheel-locks/`.
Release artifacts include a generated CycloneDX SBOM and provenance evidence;
those machine-readable files are authoritative for a particular build.

Core runtime dependencies include:

| Library | License | Role |
| --- | --- | --- |
| PyTorch | BSD-3-Clause | Tensor execution on supported compute backends |
| Transformers | Apache-2.0 | Model definitions, configuration and tokenizers |
| Accelerate | Apache-2.0 | Device-aware model loading |
| safetensors | Apache-2.0 | Tensor-file loading |
| NumPy | BSD-3-Clause | Numeric and tensor metadata utilities |
| aiohttp | Apache-2.0 | Python runtime transport |
| SentencePiece | Apache-2.0 | Tokenizer support |

The JavaScript application also distributes the packages declared under
`dependencies` in `package.json`; their individual licenses are preserved by
their packages and reported by the release SBOM.

Hardware SDKs and drivers may have separate vendor terms and are not
relicensed by Mycellios. Review the generated SBOM and applicable vendor terms
before redistributing a release artifact.

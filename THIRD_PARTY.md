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

## Adapted receipt invariant

The fail-closed zero-work invariant in
`src/contracts/distributed-execution-receipt.ts` is adapted from the receipt
coverage verifier in `leyten/shard` at revision
`fcf728096948c7686bcf0897e9acb75d1abda1d5` (`shard/receipt.py`). Shard is
licensed under Apache License 2.0, Copyright 2026 leyten. Mycellios uses its own
typed receipt schema, canonical encoding, trust pins and verification code; no
Shard runtime is included or required.

## Adapted activation-integrity challenge

The commit-first activation sketch in
`python/distributed_runtime/activation_integrity.py` is adapted from
`leyten/shard` at revision
`fcf728096948c7686bcf0897e9acb75d1abda1d5` (`shard/challenge.py`). Shard is
licensed under Apache License 2.0, Copyright 2026 leyten. The Mycellios version
is modified to use a domain-separated hash stream, sampling without
replacement, strict bounded wire validation and explicit fail-closed error
codes. It includes no c0mpute code and does not require either external runtime.

## Design references for the public-quality candidate

The comparison refreshed on 2026-09-18 used the following source snapshots:

| Reference | Revision and license record | Design input |
| --- | --- | --- |
| c0mpute | [`fcd4690fb6b19af51e801069ce6d35ae93b6db7a`](https://github.com/leyten/c0mpute/tree/fcd4690fb6b19af51e801069ce6d35ae93b6db7a); [README license: TBD](https://github.com/leyten/c0mpute/blob/fcd4690fb6b19af51e801069ce6d35ae93b6db7a/README.md#license) | Model-specific availability and worker preparation states. |
| shard | [`fcf728096948c7686bcf0897e9acb75d1abda1d5`](https://github.com/leyten/shard/tree/fcf728096948c7686bcf0897e9acb75d1abda1d5); [Apache-2.0 and third-party NOTICE](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/NOTICE) | Bounded transport and run-specific validation. |
| AntSeed | [`aef3462a52d45b4d6734e230f86c0ecd01cb1be5`](https://github.com/AntSeed/antseed/tree/aef3462a52d45b4d6734e230f86c0ecd01cb1be5); [GPLv3 license](https://github.com/AntSeed/antseed/blob/aef3462a52d45b4d6734e230f86c0ecd01cb1be5/LICENSE) | Evidence freshness, retry lifecycle and installed-runtime validation. |

The public-quality candidate implements its UI and reliability patterns
independently in Mycellios. This comparison adds no copied source, visual
assets or bundled dependencies from these repositories; it does not replace
the existing adapted-code notices above. In particular, c0mpute's unspecified
license is not treated as permission to copy its implementation or assets.
The resulting work plan is maintained in [`docs/ROADMAP.md`](docs/ROADMAP.md).

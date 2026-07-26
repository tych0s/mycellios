# Third-party libraries and research provenance

Mycellios is the product runtime. Projects studied during its design are not
selectable engines, sidecars or services in a Mycellios release. Their source
code and binaries are not incorporated into the production artifacts.

The repository can retain clearly isolated research notes or gated experiments.
Those files are source-only: the desktop and coordinator releases are assembled
from a closed allowlist, carry SHA-256 manifests and reject research or
simulation modules during verification.

The Mycellios kernel licence is pending a decision by the owner. Until a
`LICENSE` file exists, no open-source licence should be assumed for Mycellios
code.

## Libraries used by the native runtime

These are implementation libraries, not alternative inference services.
Versions and distribution artefacts are pinned by the portable-runtime policy
and must be re-verified when that policy changes.

| Library or platform | Observed licence | Product role |
|---|---|---|
| PyTorch | BSD-3-Clause | Tensor execution and native CPU/CUDA/ROCm/MPS backends |
| Hugging Face Transformers | Apache-2.0 | Model definitions and tokenizer/configuration support used by Mycellios loaders |
| Hugging Face Accelerate | Apache-2.0 | Device-aware loading support; it does not replace the Mycellios scheduler |
| safetensors | Apache-2.0 | Verified tensor-file loading |
| NumPy | BSD-3-Clause | Numeric and tensor metadata utilities |
| aiohttp | Apache-2.0 | Native Python runtime transport |
| SentencePiece | Apache-2.0 | Tokenizer support for compatible model artefacts |
| CUDA, ROCm, Metal/MPS and Vulkan components | Licence varies by component | Hardware APIs/toolchains selected by the native Mycellios runtime |

## Research references not shipped as product engines

The following projects informed experiments, comparisons or clean-room design
work. A reference here does not mean that Mycellios imports, launches, embeds or
connects to that project in production.

| Project | Observed licence | Provenance status |
|---|---|---|
| distributed runtime | GPL-3.0 | Research reference only; no product code incorporated |
| AI Horde | AGPL-3.0 | Architectural comparison only |
| external runtime A | Apache-2.0 | Historical experiment; not shipped or selectable |
| peer runtime | MIT | Research reference only |
| BloomBee | Apache-2.0 | Research reference only |
| parallel runtime | Apache-2.0 | Research reference only |
| PRIMA.cpp | MIT in the observed repository | Research reference only |
| external runtime B | Apache-2.0 | Research reference only |
| external GGUF runtime | MIT | Historical GGUF/RPC comparison; not a Mycellios product runtime |
| llmfit | MIT | Historical hardware-profiling comparison; not invoked by the product |
| Distributed Llama | MIT | LAN research reference only |
| Helix | Apache-2.0 | Planning reference; its optimiser/runtime is not included |
| DeServe | No licence declared in the observed repository | Paper/reference only; no code copied |
| NNTrainer | Apache-2.0 | Future research reference only |
| KVzip | MIT | KV-cache research reference only |
| LMCache | Apache-2.0 | Cache research reference only |
| FlowSpec | No licence declared in the observed repository | Paper/reference only; no code copied |
| MDI-LLM | MIT | Pipeline research reference only |
| NativeStage | MIT | Historical partial-GGUF experiment; not shipped or selectable |
| Speculative Pipeline Decoding | No licence declared in the observed repository | Paper/reference only; no code copied |
| Multi-Block Diffusion | MIT | Future research reference only |
| SpecPipe, Jupiter and Halo | No canonical repository recorded | Paper/reference comparison only |

Build toolchains such as the LunarG Vulkan SDK and w64devkit may be used on a
developer or CI machine. Using a compiler/toolchain does not make it a packaged
Mycellios inference engine.

This document is an engineering provenance record, not legal advice. Licences
and redistribution terms must be reviewed again before publishing any artefact
whose dependency set changes.

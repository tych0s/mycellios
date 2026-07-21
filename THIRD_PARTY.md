# External integrations and licences

This repository contains a proper implementation of the control planee. It has not been copied or incorporated source code of the projects studied.

The mycellios kernel license is pending decision by the owner before publishing it. Until a `LICENSE` file exists, an open source license should not be assumed.

| Proyecto | Licencia observada |Integration policy|
|---|---|---|
| distributed runtime | GPL-3.0 |Only external process/bridge or clean-room inspiration|
| AI Horde | AGPL-3.0 |External API or reimplemented concepts; not selling|
| [external runtime A](https://github.com/external runtime A/external-runtime-a) | Apache-2.0 |Sidecar experimental fixed by release/committee; private cell as a single deployment using OpenAI-compatible API|
| peer runtime | MIT | Sidecar Python opcional |
| BloomBee | Apache-2.0 |Sidecar Python with fixed environment and commit|
| parallel runtime | Apache-2.0 | Celda independiente mediante API OpenAI-compatible |
| PRIMA.cpp |MIT in the observed repository| Binario/proceso LAN aislado |
| external runtime B | Apache-2.0 | Celda independiente mediante API |
| external GGUF runtime | MIT |Runtime stock GGUF and cell RPC LAN; local build observed `b10068`/`571d0d540`. The base of the partial NativeStage backend is fixed separately in `c46583b86bed573c4ff30685dae59874f124e664`|
| Hugging Face Transformers | Apache-2.0 |Python unit fixed in `5.14.1`; own selective loader and native TP/EP as cell executor|
| Hugging Face Accelerate | Apache-2.0 |Python unit set to `1.14.0`, required by native TP/EP load; does not replace PyTorch|
| [llmfit](https://github.com/AlexsJones/llmfit) | MIT |Optional binary invoked without shell for hardware profile and recommendation; does not incorporate its code nor decides the layered GDLP distribution|
| Distributed Llama | MIT |LAN laboratory as an isolated process|
| Helix | Apache-2.0 |Planning ideas are used; Gurobi and his runtime are not included|
| DeServe |No license declared in the observed repository|Only ideas described in paper; do not copy code|
| [NNTrainer](https://github.com/nntrainer/nntrainer) | Apache-2.0 |Possible future sidecar for limited devices; not integrated or sold within the repository|
| [KVzip](https://github.com/snu-mllab/KVzip) | MIT |Future compression experiment of KV cache; no built-in code|
| [LMCache](https://github.com/LMCache/LMCache) | Apache-2.0 |Future experiment for trusted cells and compatible runtimes; do not use as a WAN cache among strangers|
| [FlowSpec](https://github.com/Leosang-lx/FlowSpec) |No license declared in the observed repository|Paper only and planning concepts; do not copy or redistribute code|
| [MDI-LLM](https://github.com/davmacario/MDI-LLM) | MIT |Research reference for recurring pipelines; not integrated|
| [NativeStage](https://github.com/fthrvi/native_stage) | MIT |Partial Backend Integrated flame by isolated process; source fixed in `0c16119713396ec6052400f3eb049c5e7a66cd94`, verified patches and sealed sub-GGGUF packages|
| LunarG Vulkan SDK |Licences by component|Local toolchain `1.4.350.0` used to compile the Vulkan gate; it is not edited or redistributed within the repository|
| w64devkit |Licences by component|Local portable toolchain `2.6.0` used for Windows builds; no version or redistribution within the repository|
| [Speculative Pipeline Decoding](https://github.com/yuyijiong/speculative_pipeline_decoding) |No license declared in the observed repository|Paper only, external evaluation and concepts; do not copy or redistribute code|
| [Multi-Block Diffusion](https://github.com/SJTU-DENG-Lab/mbd-lms) | MIT |Future and isolated WAN-native line; not integrated into autoregressive runtime|
|SpecPipe, Jupiter and Halo|No official localized linked repository|Only paper ideas and results; there is no code to incorporate into this project|

This table is an engineering guide, not legal advice. Licenses and commits must be re-verified before distributing an image that includes any external engine.

## Security rule

The local inference endpoints are linked to loopback. The domestic agent opens the connection to the coordinator. The RPC ports of external GGUF runtime, PRIMA.cpp or Distributed Call, nor the HTTP or external runtime A management ports, should not be published in WAN. A private external runtime A cell controls the admission, but does not constitute confidential execution or host attestation.

Critical reference: [advisory of external GGUF runtime RPC](https://github.com/ggml-org/external GGUF runtime/security/advisories/GHSA-j8rj-fmpv-wcxw).

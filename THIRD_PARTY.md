# Integraciones externas y licencias

Este repositorio contiene una implementación propia del plano de control. No se ha copiado ni incorporado código fuente de los proyectos estudiados.

La licencia del núcleo de mycellios queda pendiente de decisión del propietario antes de publicarlo. Hasta que exista un archivo `LICENSE`, no debe asumirse una licencia open source.

| Proyecto | Licencia observada | Política de integración |
|---|---|---|
| distributed runtime | GPL-3.0 | Solo proceso/bridge externo o inspiración clean-room |
| AI Horde | AGPL-3.0 | API externa o conceptos reimplementados; no vendoring |
| [external runtime A](https://github.com/external runtime A/external-runtime-a) | Apache-2.0 | Sidecar experimental fijado por release/commit; célula privada como un único deployment mediante API OpenAI-compatible |
| peer runtime | MIT | Sidecar Python opcional |
| BloomBee | Apache-2.0 | Sidecar Python con entorno y commit fijados |
| parallel runtime | Apache-2.0 | Celda independiente mediante API OpenAI-compatible |
| PRIMA.cpp | MIT en el repositorio observado | Binario/proceso LAN aislado |
| external runtime B | Apache-2.0 | Celda independiente mediante API |
| external GGUF runtime | MIT | Runtime stock GGUF y celda RPC LAN; build local observado `b10068`/`571d0d540`. La base del backend parcial NativeStage queda fijada por separado en `c46583b86bed573c4ff30685dae59874f124e664` |
| Hugging Face Transformers | Apache-2.0 | Dependencia Python fijada en `5.14.1`; loader selectivo propio y TP/EP nativo como executor de célula |
| Hugging Face Accelerate | Apache-2.0 | Dependencia Python fijada en `1.14.0`, requerida por la carga TP/EP nativa; no sustituye PyTorch |
| [llmfit](https://github.com/AlexsJones/llmfit) | MIT | Binario opcional invocado sin shell para perfil de hardware y recomendación; no se incorpora su código ni decide el reparto GDLP por capas |
| Distributed Llama | MIT | Laboratorio LAN como proceso aislado |
| Helix | Apache-2.0 | Se usan ideas de planificación; no se incluye Gurobi ni su runtime |
| DeServe | Sin licencia declarada en el repositorio observado | Solo ideas descritas en el paper; no copiar código |
| [NNTrainer](https://github.com/nntrainer/nntrainer) | Apache-2.0 | Posible sidecar futuro para dispositivos limitados; no integrado ni vendido dentro del repositorio |
| [KVzip](https://github.com/snu-mllab/KVzip) | MIT | Experimento futuro de compresión de KV cache; sin código incorporado |
| [LMCache](https://github.com/LMCache/LMCache) | Apache-2.0 | Experimento futuro para celdas de confianza y runtimes compatibles; no usar como caché WAN entre desconocidos |
| [FlowSpec](https://github.com/Leosang-lx/FlowSpec) | Sin licencia declarada en el repositorio observado | Solo paper y conceptos de planificación; no copiar ni redistribuir código |
| [MDI-LLM](https://github.com/davmacario/MDI-LLM) | MIT | Referencia de investigación para pipelines recurrentes; no integrado |
| [NativeStage](https://github.com/fthrvi/native_stage) | MIT | Backend parcial Llama integrado mediante proceso aislado; fuente fijada en `0c16119713396ec6052400f3eb049c5e7a66cd94`, parches verificados y paquetes sub-GGUF sellados |
| LunarG Vulkan SDK | Licencias por componente | Toolchain local `1.4.350.0` usado para compilar el gate Vulkan; no se versiona ni redistribuye dentro del repositorio |
| w64devkit | Licencias por componente | Toolchain portátil local `2.6.0` usado para builds Windows; no se versiona ni redistribuye dentro del repositorio |
| [Speculative Pipeline Decoding](https://github.com/yuyijiong/speculative_pipeline_decoding) | Sin licencia declarada en el repositorio observado | Solo paper, evaluación externa y conceptos; no copiar ni redistribuir código |
| [Multi-Block Diffusion](https://github.com/SJTU-DENG-Lab/mbd-lms) | MIT | Línea WAN-native futura y aislada; no integrada en el runtime autoregresivo |
| SpecPipe, Jupiter y HALO | Sin repositorio oficial enlazado localizado | Solo ideas y resultados del paper; no existe código que incorporar en este proyecto |

Esta tabla es una guía de ingeniería, no asesoramiento jurídico. Las licencias y commits deben volver a verificarse antes de distribuir una imagen que incluya cualquier motor externo.

## Regla de seguridad

Los endpoints locales de inferencia se enlazan a loopback. El agente doméstico abre la conexión hacia el coordinador. No se deben publicar en WAN los puertos RPC de external GGUF runtime, PRIMA.cpp o Distributed Llama, ni los puertos HTTP o de gestión de external runtime A. Una célula external runtime A privada controla la admisión, pero no constituye ejecución confidencial ni atestación de los hosts.

Referencia crítica: [advisory de external GGUF runtime RPC](https://github.com/ggml-org/external GGUF runtime/security/advisories/GHSA-j8rj-fmpv-wcxw).

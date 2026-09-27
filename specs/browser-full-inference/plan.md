# Plan

## 1. Establish what the browser exposes

Build a local capability probe and record core/compatibility adapter results on
the target desktop and phone browsers. Request only supported features and
limits. Keep diagnostics separate from any capability claim.

## 2. Execute a whole model in the browser

Evaluate current WebGPU engines with a pinned model and exact artifact hashes.
Prefer the smallest integration that demonstrates complete local inference.
Treat model format conversion and quantization as explicit transformations,
not hidden differences from the native baseline.

## 3. Measure parity honestly

Run native and browser on the same physical GPU and workload. Measure prompt
processing and token decoding independently, including cold/warm start,
sustained runs, memory and numerical output. Repeat enough times to quantify
variation. A passing browser run is not a parity result.

## 4. Optimize the measured bottleneck

Use GPU-resident weights/KV, static buffers, per-device kernel tuning and
dispatch fusion. First test multi-token GPU-resident decode chunks with
sampling and EOS handling in the complete model, because the timestamp profile
found large gaps between its queue submissions and a synthetic dependent
graph benefited strongly from eliminating per-token readback. Measure output
parity, TTFT, tokens/s, memory and user-visible streaming latency against the
original engine. Pursue subgroup-matrix hardware for prefill when browser
support becomes available. Reject changes that improve a synthetic kernel but
not end-to-end inference.

The wllama local probe found 60-66 tiny host writes into one uniform buffer
before most decode submissions. A wrapper that combined them into one wider
write per submission preserved visible output and drastically reduced API
calls, but did not improve full-request latency. If this path is pursued,
change the backend's parameter layout or generation so the shader consumes
prepacked GPU-resident parameters directly; measure shader time and total
request time before keeping it.

The temporary direct-stage probe now shows the next concrete backend design:
allocate a uniform buffer for each recurring graph stage, keep its image on
the GPU, and bind it directly instead of rewriting one shared buffer for
every token. It removed all warm uniform host writes on one complete model,
but did not establish an end-to-end speedup. The current backend still builds
18,368 uniform bind groups for 32 tokens; a safe cache based on actual
layout/resource identity found no reuse. Stabilize graph allocation and
layouts at backend compile time, then prebuild compatible bind groups and
compare exact token IDs plus native/browser timings. Do not carry the
temporary worker wrapper into production.

## 5. Connect the browser as one distributed Mycellios node

Add a typed browser-stage adapter under existing contracts and transport areas.
The local expert-owner probe exercised the existing resident-mesh route for
one real SmolLM2 SwiGLU MLP with two browser replicas. An opt-in bridge now
attaches one selected Qwen3/GLM4 SiLU MoE expert to a RAM-backed model stage;
the model's router, attention, KV and token generation stay native. The bridge
retains the native RAM expert as fallback and requires an exact model digest,
an operator-supplied measured profile, the internal coordinator token, and two
validated visible browser replicas. A generated tiny Qwen3-MoE fixture ran
three complete tokens through two same-host WebGPU browsers with exact token-ID
parity. Its profile deliberately forced that route and is not a calibrated
performance result. The same local route now supports one dense Qwen3 SiLU
MLP in a Torch safetensors stage. A tiny complete Qwen3 fixture produced three
exact token IDs through two same-host WebGPU browsers and through the native
fallback after one browser was paused. The GGUF backend, physical phones and
a fully browser-owned transformer stage remain outside this implementation.
A contiguous transformer layer still needs a browser-stage contract and
lifecycle implementation. The route must be heterogeneous: one browser may own
one stage while other stages run on native runtime nodes or other browser
nodes. Two browser tabs or two browser replicas are not an admission
requirement for a browser stage. First prove one browser stage and one native
stage on two physical machines with the same model and valid numerical output;
then compare network and compute costs. Foreground availability, model
residency and lost-device behavior must be visible to the same scheduler.

Browser registration should advertise executable backends, not just a yes/no
WebGPU flag. A WebGPU browser is eligible for a WebGPU stage after a real
model-specific capacity and numerical canary. A CPU-only browser may be
eligible for a CPU stage when it can load the assigned artifact and meet the
model's memory and timing constraints. The planner may assign zero layers to a
slow device; the click still joins the network and reports why no stage was
placed there. Admission must never invent usable VRAM from an adapter probe or
use the unrelated matrix benchmark as a model throughput measurement.

The browser stage uses `mycellios-browser-layer/2`: a content-addressed float32
ONNX graph, exact model digest, hidden input/output names, declared auxiliary
tensors, zero-initialized state tensors with full and incremental outputs, and
a native-output numerical canary. The browser runner and coordinator validate
declared names and shapes without knowing the model architecture. A trusted
native adapter supplies auxiliary tensors and commits incremental state; the
built-in rotary-KV adapter covers Qwen3 and Llama. Further model families can
register another adapter and publish a graph that satisfies the same ABI.
The publisher checks the graph hash, model digest, ONNX input/output names and
the native canary before registration. Model-specific export is still required;
unsupported ONNX operators, precision or memory limits are admission failures.
The coordinator rejects stale generation IDs, duplicate sequence positions,
artifact mismatches and backend mismatches, and may replace an idle resident
layer without evicting an active generation. An explicitly placed CPU browser
stage may execute with WASM. Keep browser capacity separate from the native
`distributedExecutor` claim until measured for that exact model and stage.

For the local ONNX gate, install the optional Python test dependencies
`onnx==1.23.0`, `onnxruntime==1.30.0` and `numpy==1.26.4` in the distribution
venv. Run `tests/run-browser-qwen3-layer-onnx.py` with `--architecture` set to
`qwen3` or `llama` and `--model` pointing to an exact safetensors snapshot,
then run
`node tests/serve-browser-qwen3-layer-onnx.mjs` and open
`http://127.0.0.1:8771/?real=1`. The test files use ignored `runtime/` outputs.

## 6. Propose platform changes only when necessary

If profiling proves a specific WebGPU API gap, prepare a minimal benchmark and
proposal for GPUWeb/Chromium. A future browser feature cannot be counted as a
shipped Mycellios capability until present in ordinary browsers.

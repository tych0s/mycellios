# Browser-only full-model inference

## Objective

A user opens Mycellios in an ordinary browser, presses Start, and executes a
complete model on that device's GPU without a native application, extension,
custom browser or mandatory flag. The performance target is a pinned
model/device/browser combination with native-equivalent prefill and decode,
responsive streaming, bounded memory and correct output. All measured
capabilities and limitations belong in `docs/STATUS_AND_EVIDENCE.md`.

## Definitions

- **Complete model:** all required weights, attention, KV cache, sampling and
  generation execute locally in the browser. A coordinator may select work
  and verify it, but cannot execute omitted model operations remotely.
- **Browser-only:** shipped JavaScript, WebAssembly and browser APIs such as
  WebGPU; no installed helper or native-messaging host. Mobile support must be
  measured on a physical device.
- **Native parity:** same source weights, tokenizer, raw input token IDs,
  precision target, context, batch and physical GPU. Record numerical output
  agreement, prefill and decode rate, TTFT, latency distribution, memory and
  sustained behavior before claiming equivalent performance.
- **GPU use:** distinguish WebGPU buffer sizes, OS GPU-engine scheduling time,
  device memory residency, GPU execution duration and physical compute-unit
  occupancy. None is interchangeable with another.

## Runtime design candidate

1. `start()` creates a long-lived browser compute session. Request the core,
   high-performance WebGPU adapter, the optional features needed by a chosen
   model, and only the limits its graph requires. Record adapter identity,
   granted features and limits. Handle rejection and device loss explicitly.
2. Load a signed/hash-checked model manifest and stream weight shards into
   persistent GPU buffers. Plan allocations around binding/buffer limits and
   observe actual creation failures. A limit is not a VRAM reservation.
3. Keep token IDs, embeddings, attention/KV, logits and sampling resident on
   GPU between dependent decode steps. Prepack position and KV metadata to
   reduce small host writes. Encode as much dependent work per command buffer
   as correctness allows.
4. Send bounded GPU decode chunks, read selected output IDs once per chunk,
   deliver them through a streaming interface, and roll back or mask any KV
   entries computed beyond EOS or the length cap. Tune initial and sustained
   chunk sizes separately to balance first-fragment latency and throughput.
5. Reject or fall back for sampling options without an equivalent GPU path:
   grammar, custom logit processors, penalties and detailed logprobs must not
   be silently skipped. Maintain exact context-window and sliding-KV behavior.
6. Keep browser UI and model execution separate where useful, but treat a Web
   Worker as an interface/lifecycle tool until an A/B test proves a compute
   advantage. Recover cleanly from page suspension and device loss.
7. After local full-model parity, adapt a model stage to Mycellios contracts
   and measure transport on two physical computers. Loopback cannot establish
   a physical distributed route.

The experimental pages under `tests/browser-webgpu-*.html` and
`tests/browser-webllm-*.html` exercise individual parts of this design. Their
results, model revisions, measurements and caveats are maintained only in
`docs/STATUS_AND_EVIDENCE.md`. The current WebLLM decode fork depends on
private runtime methods and must become a maintained backend interface before
product integration.

## Browser and driver boundary

The shipping browser route must use the capabilities exposed by WebGPU. A
website can request supported features and validation limits, but cannot
create its own driver privilege or exclusive GPU scheduler permission. A
native-messaging extension requires an installed host and therefore fails
the no-install objective. If profiling isolates a missing WebGPU capability,
prepare a minimal benchmark and a browser/GPUWeb proposal. Candidate areas
include subgroup matrix operations, reusable compute graphs and informative
memory-budget or sustained-compute signals. A JavaScript library can consume
such an API when implemented; it cannot grant the API to itself.

WebNN may be evaluated as a separate graph backend when it ships in ordinary
client browsers. Its accelerator preference remains a hint; any WebNN/WebGPU
tensor exchange must be measured for copies, synchronization and full-model
behavior before considering a hybrid decoder. The ordinary-browser gate must
not depend on a developer flag or a polyfill that delegates back to WebGPU.

## Acceptance gates

1. Pin browser build, model source revision, compiled library, native model
   source weights, GPU/driver and tokenizer. Verify artifact hashes and raw
   input token IDs for every compared request.
2. Run correctness and resource checks on multiple model sizes and prompt
   lengths, including EOS, length cap, second-turn KV reuse, context edge,
   sampling options, streaming and device loss. Report failures as failures.
3. Measure cold and warm load, prefill, decode, TTFT, first decoded fragment,
   full latency, throughput distribution, GPU buffer sizes, physical memory
   residency where available, and sustained behavior on the same machine.
4. Verify numerical/logit agreement and output token IDs across representative
   prompts before treating browser/native timing ratios as parity. Divergent
   outputs or unverified source weights prevent a strict parity conclusion.
5. Change a kernel, compiler path, scheduler or GPUWeb proposal only against a
   measured bottleneck, then repeat the full-model A/B check with the exact
   same input and output checks.
6. Prove a browser stage on two physical computers only after local model and
   native physical-GPU baselines exist. Record identities, transport, memory,
   TTFT/TPOT, aggregate and per-user tokens/s, parity and recovery.
7. Put all capability claims and dated evidence in
   `docs/STATUS_AND_EVIDENCE.md`; keep this spec as the design and gate.

## Research sources

- WebGPU specification: https://gpuweb.github.io/gpuweb/
- WebNN specification: https://www.w3.org/TR/webnn/
- Chrome WebGPU adapter selection on Windows: https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips
- Chrome compatibility mode: https://developer.chrome.com/blog/new-in-webgpu-146
- GPUWeb subgroup matrix draft: https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md
- GPUWeb memory budget issue: https://github.com/gpuweb/gpuweb/issues/5505
- Chrome native messaging host registration: https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging
- wllama llama.cpp/WebGPU browser package: https://github.com/ngxson/wllama
- LlamaWeb paper: https://arxiv.org/html/2605.20706v1
- WebLLM paper: https://arxiv.org/abs/2412.15803
- WebLLM v0.2.85 decode and sampling source: https://github.com/mlc-ai/web-llm/blob/v0.2.85/src/llm_chat.ts
- llama.cpp server token-ID API: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Dispatch overhead paper: https://arxiv.org/abs/2608.08730

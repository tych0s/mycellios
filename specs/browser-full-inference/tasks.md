# Tasks

- [x] Research current browser LLM engines, WebGPU capabilities and platform
  restrictions; record the technical hypothesis and acceptance gates.
- [x] Add a local browser capability probe for core versus compatibility mode
  and requested features/limits; run it on PC-DANI in Chrome 154.
- [x] Probe native WebNN and high/low WebGPU adapter preferences in the same
  ordinary Chrome session; record whether a WebNN benchmark can actually run.
- [x] Add and run a standalone loopback WebGPU matrix-compute probe with
  CPU correctness checks and GPU timestamp timing on PC-DANI.
- [x] Add an intentional device-loss and compute recovery exercise to the
  capability probe; run it in Chrome core and compatibility modes on PC-DANI.
- [x] Request adapter-level WebGPU buffer limits and verify a 512 MiB storage
  buffer with shader writes and readback at both ends in ordinary Chrome.
- [x] Load and generate from the same GGUF with llama.cpp/WASM/WebGPU in
  Chrome for both SmolLM2 135M F16 and Qwen3 0.6B Q8_0; compare native Vulkan
  and native/browser CPU baselines and record layer offload and output gaps.
- [x] Reproduce Qwen first-versus-reused output changes with and without
  prompt caching, and compare the near-tied sixth-token log probabilities
  in browser WebGPU and native Vulkan.
- [ ] Compare exact browser/native input token IDs and initial/reused Qwen KV
  tensors or layer outputs to locate the numerical difference before
  asserting cross-backend token parity.
- [x] Reverse a one-versus-four WASM-thread SmolLM2 WebGPU trial; reject the
  apparent speed difference because it disappeared on phase reversal.
- [ ] Trace the large wllama WebGPU latency drift with browser GPU and CPU
  timing while the model tab is visibly selected and the native path is idle.
- [x] Add a temporary-worker diagnostic for per-action WebGPU queue submissions,
  writes, maps and waits; compare three visible, focused complete-model
  requests and record the anomalous extra first-step submissions/maps.
- [x] Capture two Chrome GPU-process traces during Qwen requests and document
  that their service spans are not physical shader timings.
- [x] Profile wllama's per-token write sizes and buffer ranges; try a
  uniform-buffer shadow that coalesces writes before submission, verify
  visible output and reverse the comparison order. Reject the wrapper as a
  default because the four-phase trial did not improve end-to-end latency.
- [x] Compare narrow range merging, compute-scatter updates and per-stage
  uniform-buffer snapshots on the complete SmolLM2 WebGPU model. Verify
  visible output, count physical writes and record noisy wall times.
- [x] Redirect dynamically created bind groups to GPU-resident uniform
  buffers per stage; verify zero uniform host writes on a warm request and
  compare three direct-stage and three original 32-token requests.
- [x] Try an identity-safe bind-group cache with direct stage buffers; record
  zero reuse and retain layout/resource identity as a correctness condition.
- [ ] Stabilize graph layouts/resources in a maintained backend, then compare
  prebuilt bind groups and stage buffers on exact token IDs and repeated
  native/browser A/B workloads before treating them as a speed optimization.
- [x] Record a pinned native F16 GGUF/Vulkan baseline on one physical computer.
- [x] Execute a complete SmolLM2-135M q0f16 MLC model inside an unmodified
  browser, with fixed remote artifact revisions and local output metrics.
- [x] Instrument per-token buffer-map sizes and waits, and capture a Chrome
  GPU-process trace; record what those measurements can and cannot attribute.
- [x] Capture GPU timestamps for every compute pass in a warm complete-model
  request and separate gaps within and between queue submissions.
- [x] Build and verify a browser-only GPU-resident dependent decode probe against
  CPU and a per-token CPU-roundtrip variant on the same GPU.
- [x] Build a tiny transformer graph with GPU-resident token, KV cache, argmax
  and EOS; compare chunk sizes 1/4/16/64 against a CPU reference and roundtrips.
- [x] Run a local complete-model WebLLM decode fork that feeds the sampled GPU
  token into the next embedding and reads one packed token tensor per chunk;
  compare token IDs and alternate warm A/B requests on the Radeon 890M.
- [x] Roll back speculative GPU KV entries after chunked EOS/length, verify
  first-turn KV length and exact second-turn continuation for fixed prompts at
  chunk sizes 8 and 16, and compare an adaptive chunk schedule with fixed 8.
- [x] Exercise WebLLM's public streaming completion API with the complete
  model in original, adaptive and fixed-chunk modes; compare exact output and
  fragment timing on one fixed prompt.
- [x] Compare long-prompt chunks 8/16/32 and run 24-request original/chunk-16
  sessions in both phase orders, checking all output tokens and KV lengths.
- [x] Measure Chrome GPU-engine scheduling counters during sustained model
  and matrix workloads and test the pinned model in a browser Web Worker;
  distinguish those observations from physical utilization or native parity.
- [x] Test increasing long-output chunks `[1, 2, 4, 8, 16]` in two sessions,
  compare token and KV parity, rollback, maps, submissions, first decoded
  token and total time; verify fragment arrival through streaming API.
- [x] Run pinned complete SmolLM2-360M F16 and 1.7B Q4F16 models locally in
  Chrome, compare original and chunked token IDs, measure requested WebGPU
  buffer sizes, and verify a two-turn 1.7B KV continuation after rollback.
- [x] Capture the browser's exact short and long input token IDs; submit them
  to the loopback-only native Vulkan server, check output-token parity and
  divergence, compare short warm timings, and stop the native server.
- [x] Verify MD5 hashes for all eight pinned 135M MLC weight shards and compare
  every stored F16 weight against the local GGUF, including fused QKV layout,
  Q/K permutation and norm conversion.
- [x] Isolate the first long-prompt browser/native logit divergence with a
  shared fixed token prefix and an explicit comparison of logits and sampling.
- [ ] Compare selected intermediate layer outputs at the shared 58-token
  prefix to locate the operation that flips the two near-tied logits.
- [x] Identify the costly WebLLM pipeline IDs by generated WGSL operation and
  compare a subgroup MLP reduction against the original in the complete model.
- [x] Try a subgroup reduction in the final vocabulary projection, alternate
  it with the original, and compare targeted GPU time, response text and
  token IDs for the fixed prompt.
- [ ] Distinguish queue/CPU scheduling, copies and GPU activity within the
  inter-submission gaps beyond the current map/write correlation.
- [ ] Implement and A/B test GPU-resident multi-token decode chunks in a
  complete model with GPU-side EOS masking, general multi-turn KV state,
  supported sampling options and streaming output; the local fixed-settings
  fork has only validated selected two-turn continuations.
- [x] Identify the WebLLM v0.2.85 token readback and host-to-GPU reupload
  source path that a GPU-resident decode fork must replace.
- [x] Split full-model inter-submission GPU timestamp gaps by token readback
  and count the size distribution of queue writes; repeat under variable speed.
- [x] Test a queue-write staging wrapper on the complete model, verify the
  answer and reject it as a default without a repeatable end-to-end gain.
- [ ] Produce a reproducible same-device A/B report for prefill, decode, TTFT,
  memory, sustained rate and numerical parity.
- [ ] Optimize only measured gaps and repeat the A/B report.
- [ ] Prove a browser stage on a physical two-computer route.
- [x] Route the real SmolLM2 block-0 SwiGLU MLP weights through the existing
  browser expert owner and resident mesh on one local computer; verify two
  distinct visible WebGPU browser workers against a native numerical result.
- [x] Repeat same-GGUF browser WebGPU versus native Vulkan local completions
  with short and long SmolLM2 prompts, plus a Qwen diagnostic, and record
  batch-size and worker-duration measurements for the browser expert route.
- [ ] Update canonical documentation only when durable results are established.

## Connection recovery gate

- [ ] Exercise hiding the page beyond coordinator retention and returning to it;
  recover registration with the stable signed browser identity when the old
  ephemeral worker credentials are rejected. Cancel old leases before accepting
  work under a renewed registration.
- [x] Verify pause during registration recovery cancels that recovery and does
  not restart contribution. A rejected admission must remain rejected.
- [x] Verify coordinator removal and a superseding browser session do not cause
  competing automatic re-registration loops.
- [x] Verify renewal after coordinator restart and forced disconnected-session
  expiry in the isolated browser fixture, retaining the signed client identity.
- [ ] Repeat the public visible-page Pause/Resume route and save both the UI
  result and the coordinator's observed worker count. Keep browser connection
  evidence separate from an assigned model-layer inference.

# Status and evidence

Updated: 2026-09-27 (local browser layer and physical Pixel test).

## Local browser transformer-layer experiment (2026-09-27)

An opt-in local ONNX export probe executed one **complete** decoder layer from
the cached `Qwen/Qwen3-0.6B` checkpoint in the Codex in-app Chromium browser
on an AMD Radeon 890M (driver `32.0.31041.1004`).
The checkpoint SHA-256 was
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
the exported float32 layer graph was 62,947,836 bytes. ONNX Runtime Web 1.30.0
used its WebGPU execution provider and retained the attention key/value outputs
as GPU buffers between a three-token prefill and ten one-token decode steps.
All eleven browser outputs matched the corresponding native PyTorch layer
within 2.4e-6 maximum absolute error. In this one local run, model load took
2.57 s; the median of eight decode steps after the first two was 14.10 ms per
layer. This is a single-layer latency, not full-model tokens per second.

The same real-weight layer also ran through ONNX Runtime Web's WASM CPU backend
in that browser: all eleven outputs matched PyTorch within 4.1e-6 maximum
absolute error, and the median of the eight later decode steps was 4.20 ms in
one run. This does not establish that browser CPU is generally faster than
WebGPU; both measurements depend on the device, model shape, session and
browser. It does establish numerical feasibility for a CPU browser stage.

The graph was then extended to return incremental KV for a native handoff.
Actual browser-returned KV from both backends was fed into the native Qwen3
layer at each of ten continuation points. The maximum output difference from
an all-native layer was 1.5e-6 for WebGPU and 1.2e-6 for WASM. These are local
state-continuation checks, not a distributed failover test: no active network
stage was interrupted. Returning incremental KV changed measured latency; a
subsequent single run had 47.53 ms median WebGPU and 6.69 ms median WASM for
the eight later decode steps, with substantial WebGPU variation.

The ONNX execution trace also assigned integer shape operators to CPU. No
physical phone, two-host browser stage, automatic layer placement, complete
model route or same-GPU native/WebGPU performance comparison was tested. The
experiment does not establish zero CPU involvement, native-equivalent speed or
release readiness. Its local export and browser harness are under `tests/`;
the generated graph, weights, fixture and receipts stay in ignored `runtime/`.

## Implemented and connected

- model profiling, certified adapters and contiguous range planning;
- deterministic runtime manifests and launch descriptions;
- local and authenticated HTTP launch agents;
- staged startup, readiness, canary, registration and cleanup;
- streaming API and persistent runtime transport;
- direct-channel selection with bounded relay fallback;
- recovery orchestration and execution traces;
- commit-first activation-integrity sketches with a Torch-free control-plane
  verifier and optional binding inside signed physical receipts;
- physical campaign, evidence import and gate-report tooling.

## Verified in automated or single-host runs

- selective Llama and Qwen-family pipelines;
- real subprocesses and sockets on one host;
- CPU pipeline parity and greedy kill-and-resume recovery;
- local direct/relay transport behavior and rejection paths;
- tensor/expert cell contracts and supported CPU fixtures;
- coordinator, worker, activation, security and packaging policies.

The normal test suite validates software behavior. Opt-in physical tests are
skipped when their required model, GPU or host environment is unavailable.

## Economic and token status

The source includes usage-based pricing and an internal ledger that allocates
contributor credits from execution and contribution evidence. Each settlement
is balanced and produces a signed receipt bound to its job, pricing policy and
evidence. The readiness contract defines internal compute credits as
non-monetary, non-transferable and non-withdrawable.

The public-token readiness state is disabled and has no network selected. It
allows only `no_go` or `eligible_for_spec`; no public Mycellios token or
on-chain contributor settlement is active. Optional stablecoin billing
adapters are a separate inbound-payment path and do not change this boundary.

Any future token decision remains gated on demonstrated need and evidence for
legal, accounting, security, governance, smart-contract and liquidity readiness.
The repository does not commit to a chain, token supply or launch date.

The local development gate enrolls signed workers in its temporary coordinator
database. Its two workers are logical processes on one computer. Cleanup
deadlines remain active until completion and cleanup failures return nonzero.

Activation checkpoint capture/restore retains an owner-only Unix socket on
POSIX. The local candidate adds authenticated loopback control on Windows,
using a signed endpoint descriptor and a per-launch secret. Focused Windows
tests exercised a real TypeScript/Python capture/restore round trip. This
establishes the local control path, not remote recovery after losing a machine.
Windows configuration writes flush the payload before atomic rename;
directory fsync is available only on POSIX.

## Public-quality validation

These changes were validated locally against public main `0e3347f`. Publishing
the source does not establish deployment or physical-host evidence. The
comparison and improvement plan are in
[`ROADMAP.md`](ROADMAP.md), with exact external revisions and source provenance
in [`THIRD_PARTY.md`](../THIRD_PARTY.md).

The changes prevent cross-account session access and retain retry protection
for new and pre-update requests. They cancel work while waiting for capacity or
reading an SSE stream, reject late worker starts and prevent activation during
shutdown. The interface retains drafts and partial
answers, distinguishes intentional stop from failure, preserves selected models
when availability changes, and keeps coordinator connectivity separate from
inference capacity. Studio draft persistence, publication versioning, hot reload
and narrow-screen layout have regression coverage.

The strict Python suite passes: 1079 tests run, 1061 passed, 18 skipped, no
failures or errors. Rank entrypoints no longer import the dense model loader
solely for type annotations. Checkpoint requests still queued after timeout,
disconnect or shutdown cannot mutate stage state later. The CPU collective
fixture keeps its rendezvous port bound throughout process startup, removing
the observed Windows address-in-use race without relaxing parity or timeouts.
An additional Unix-only transport case is skipped on Windows; the original
platform/model prerequisite skips remain separate from passing evidence.
An additional focused checkpoint run passes 16 cases with one Unix-only skip.
It reproduces and fixes slow fragmented input monopolizing the serial control
listener: one 30-second deadline starts at accept and includes framing, queueing
and response. Both unauthenticated slow headers and authenticated slow payloads
expire, after which the same server accepts a legitimate capture.

The complete JavaScript suite passes: 274 test files and 1872 tests passed, with
6 files and 20 tests skipped for their declared prerequisites. The final run
includes the launcher and legacy-idempotency changes. Subsequent package
corrections pass six release-policy tests and ten registry/catalog tests.

Release staging exposed requirements for two optional service files absent from
the public checkout. It now requires the two repository-owned native policies,
and tests copy the real deployment files instead of inventing empty fixtures.
The storage policy now uses the node-update directory variable consumed by the
runtime. Installed adapter metadata resolves from the package, independently
of an external importer's entrypoint or working directory. Optional Content Hub
and Supabase configuration remain available through their existing settings.
No deployed configuration or stored data was migrated during these checks.

The final production build and sealed coordinator staging pass. Fresh child
processes resolve only staged dependencies, serve the staged landing/mobile
assets, expose loopback health with the matching build identity, and exit with
their listeners closed. Both the server library and actual coordinator CLI
entrypoint are exercised. On Windows the CLI check invokes its installed
SIGTERM handler; it does not claim a physical operating-system signal test.

The native development gate completed real CPU inference through two signed
logical workers and two stages on this computer, with matching canary/routed
output and drained resources. Browser checks covered home, chat and Studio at
desktop/narrow widths, including light/dark cancellation, partial errors,
unavailable capacity and draft persistence against an isolated API fixture.
Neither the fixture nor the native local run is evidence of two physical hosts.
The complete npm dependency audit reported zero known vulnerabilities.

## Physical two-computer repeat on a fixed commit (2026-09-28)

PC-DANI and MRPUCCHI ran the native route from the same public source commit
`ff835802815f3749d72b8deb4b2ea5978b479f3b` with `npm ci`, passing
`verify:structure` and `typecheck` on both hosts. The model was
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`; `model.safetensors` had
SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`
on both hosts. Both prepared launch descriptions had SHA-256
`61256c112bd9900fc70525d7e592c76b21c9c128c5f90a4dffa9ab7cc3363948`.
The route assigned `[0,15)` to PC-DANI (`192.168.1.79`, CPU) and `[15,30)`
to MRPUCCHI (`192.168.1.2`, NVIDIA RTX 2060 CUDA). Established TCP
connections were observed in both directions on stage port `18101` and return
port `18100`. This was a mixed CPU/GPU physical run, not a GPU comparison or a
monolithic parity test.

The canary returned 16 tokens with TTFT 161.9 ms, TPOT 87.7 ms, pipeline
1477.3 ms and 11.40 output tokens/s. Its text was `I'm sorry for any
misunderstanding, but as a chatbot, I don't`; it did not obey the `Reply with
only OK.` prompt. A separate API request answered `2 + 2 = 4.` with 9 output
tokens, TTFT 293.1 ms, TPOT 73.4 ms and pipeline 880.6 ms. Observed process
peak working sets were 970,891,264 bytes for the PC-DANI root and
1,331,494,912 bytes for the MRPUCCHI stage. These are individual observations,
not sustained throughput or GPU speedup claims.

For a controlled failure, a second long API request was started at
13:23:40.793 local time and the MRPUCCHI stage process was killed at
13:23:52.041 while it was running. The request returned HTTP 500 after
11.686 seconds. The coordinator then reported
`launch_agent_rpc_timeout:poll:5000`, exited with failure, and closed its
local API and return listeners. Automatic recovery was not observed. The
remote stage and launch agent were no longer listening; temporary inbound
firewall rules were removed and MRPUCCHI's original Python TCP block rule was
restored. This tests loss of the remote stage process, not loss of power or
network access to the whole PC. An earlier attempted fault happened after its
request completed and is excluded from this result.

The first launch attempt exposed a pre-existing Windows Firewall block rule
for `python.exe` on MRPUCCHI. A later attempt reached both stages but failed
health fetching because the local API bound to `127.0.0.1` while the local
configuration advertised `192.168.1.79`; the successful run advertised
`127.0.0.1`. Both are reproducible setup obstacles for a tester.

### Failure-handling repeat after local fixes (2026-09-28)

The coordinator and MRPUCCHI agent were relaunched with local, uncommitted
changes to the API error mapping and Windows launch-agent exit handling. These
results therefore do not establish a repeat on a new shared commit. The same
pinned model and two physical stage ranges were used. A long request began at
20:48:27.982 local time; the MRPUCCHI stage process was killed during it. The
API returned HTTP 503 (`service_unavailable`) in 2,517 ms. The coordinator
exited with `launch_process_exited:stage-92790ee51c7743c76aad:code=4294967295:signal=null`
instead of the earlier agent poll timeout. A subsequent authenticated agent
health check reported zero active processes and one retained tombstone. The
local API and return listeners closed; the test agent was stopped, temporary
firewall rules on both PCs were removed and MRPUCCHI's original Python block
rule was enabled again. This is graceful failure reporting and cleanup for a
killed stage process. Automatic recovery, loss of the entire PC and a third
party installation remain unverified.

### Whole-PC interruption (2026-09-28)

MRPUCCHI was rebooted during a long active request at 20:53 local time. This
was a physical loss of the remote computer, distinct from killing its stage
process. The request ended in a transport error after 43,738 ms, without an
HTTP response. The coordinator reported
`launch_agent_rpc_timeout:poll:5000`, marked the route failed and closed the
local API and return listeners. MRPUCCHI booted again at 20:53:48; SSH over
Tailscale and its automatic services were available by 20:54:11. Temporary
firewall rules were removed on both PCs and the original Python block rule was
restored. The initial attempt therefore **did not** meet the controlled-error
criterion. The generated Python root command used the 300-second startup
deadline for active socket operations too, so the supervisor stopped the root
before the request could time out and produce a 503.

The physical repeat used a separate 20-second active-operation deadline while
retaining the 300-second startup deadline. A long request began at 21:00:40.849
local time and MRPUCCHI was commanded to reboot during the request. The API
returned HTTP 503 (`service_unavailable`, `timed out waiting for request 5`)
after 29,156 ms, before the supervisor stopped the root. The supervisor later
reported the unavailable agent through `launch_agent_rpc_timeout:poll:5000`
and closed the API and return listeners. MRPUCCHI booted at 21:01:09 and SSH
over Tailscale, `sshd` and `Tailscale` services were reachable by 21:01:40.
The temporary stage and return firewall rules were removed and the original
Python block rule was enabled again. The run used local, uncommitted source
changes on the two PCs; it demonstrates controlled error reporting for this
small pinned model under this particular whole-PC reboot, not automatic
recovery or a general performance guarantee. The agent did not restart
automatically after the reboot, and the failed route was not resumed.

The lab USB has Jarvis and Tailscale material plus a provisional Mycellios Node
MSI and launcher; the one-click node path has not yet been run from the USB.
The repository has an MSI build and one-time enrollment flow, but no MSI or
pairing bundle was installed on a third PC in this campaign. The enrollment
bundle lifetime is at most 15 minutes and must be issued for each installation;
it is not a durable secret to leave on a USB. The contribute page previously
downloaded that bundle with a `.json` extension while the Windows MSI registers
`.mycellios-enrollment`; the local source now uses the registered extension.
That UI fix and the portable-runtime package build do not establish a
successful external installation or GPU contribution by an installed node.
The local Windows x64 portable runtime build completed with pinned Python
3.12.13 and CPU PyTorch 2.13.0; its relocated runtime and installed-stage
canary passed the archive verifier.
The local Windows x64 node payload was then assembled from the compiled app
and portable runtime, and `verify-node-installer` confirmed its complete file
layout and hashes. This is a provisional ignored build: its source revision
field is the current HEAD while its source provenance includes uncommitted
changes. A provisional 244,653,268-byte MSI was then built locally with WiX
3.14.1 from this payload. `verify-node-msi` passed administrative extraction,
exact staged-tree comparison and MSI metadata checks after adding the omitted
layout manifest to the MSI and shortening its verification path. The local
staging tree was mounted under a temporary drive letter for WiX because its
ordinary source path exceeded Windows Installer's path limit; the mapping was
removed after the build. This unsigned, uncommitted MSI is a lab artifact, not
a release. Local release-evidence generation and verification passed for this
MSI, with the required promotion signature still pending. It must be rebuilt
from a fixed commit before distribution. A later service-start review found
that this first MSI omitted the `distributed_runtime` Python source and the
Windows Job Object broker, while the packaged CPU-only PyTorch runtime was
sent through a CUDA-required startup path. The structural MSI checks therefore
did **not** prove that its service could reach Ready; this first MSI cannot be
used as the node pilot package. Local source now includes CPU node startup,
the Python data plane and the broker in the staged layout. Package
installation, enrollment and service participation have not been run in this
campaign.
A replacement provisional 249,811,116-byte Windows MSI was built locally from
the current uncommitted tree. Its 33,993-file layout includes
`python/distributed_runtime/physical_probe.py` and
`bin/mycellios-job-broker.exe`; `verify-node-installer`, `verify-node-msi`
and local release-evidence verification passed. The staged portable Python
reported `torch 2.13.0+cpu`, no CUDA API and zero GPU devices. The compiled
native runtime then selected `cpu-only`, `cpuEligible=true`, `cpu-ready`, and
no GPU physical identity; a second layout verification passed after that
probe because bytecode writes were disabled. The broker built locally as a
self-contained fallback after Native AOT failed for lack of the C++ linker;
seven broker tests passed, including physical child-process containment. CI
now sets up pinned .NET SDK 8.0.423 and requires the Native AOT build, but
that workflow has not run with these changes. The replacement MSI is unsigned
and has not been installed, enrolled, or assigned a live stage.
Further service review found that this replacement MSI also registered the
console `node.exe` directly with the Windows Service Control Manager. It has
no service entrypoint and must not be used for the USB pilot. Local source now
stages pinned WinSW 2.12.0 and registers that wrapper with a protected XML
definition. The later Windows payload was staged with the pinned wrapper and
`verify-node-installer` checked its 33,995-file layout and hashes. The wrapper
loaded a minimal XML for a non-installed service and reported `NonExistent`.
A third provisional, unsigned 256,082,132-byte MSI was built locally from this
uncommitted source and passed `verify-node-msi`: administrative extraction,
exact staged-tree comparison and MSI metadata checks. Its SHA-256 is
`829a5b023cc17cd2a01f2d04726906e20dab5400772f928cd92f0fc60c950d98`.
Focused tests, typecheck, structure, format and architecture checks passed.
On 28 September, this exact MSI was copied to MRPUCCHI and installed with
Windows Installer as an administrator. Its verbose log ended with
`Installation completed successfully` and `MainEngineThread is returning 0`;
Windows registered Mycellios Node 0.2.77. The installed portable Python ran
`torch 2.13.0+cpu` with no CUDA API. No pairing bundle was supplied, so
`MycelliosNode` is not registered as a service and neither Ready nor live
stage participation has been demonstrated. This was an MSI installation test,
not the one-click USB flow or a release.
The same MSI was then removed on MRPUCCHI with Windows Installer exit code 0.
The product registry entry, service, Program Files installation root and
ProgramData node configuration were all absent afterward. A revised local
candidate adds automatic provisioning of the existing pinned CUDA pack after
pairing, but it has not been installed or GPU-probed as an installed service.
That revised 256,090,372-byte unsigned MSI was built from the dirty local
source, with SHA-256
`f954dbd9447f171de7c79ae798a72b8e6446856a3c31e9603b63714c182ba7e0`.
Its 33,998-file staged layout and MSI administrative extraction comparison
passed. The exact bytes and checksum were copied to the lab USB, with zero
`.mycellios-enrollment` files; the revised candidate is not a release.
After correcting the development gate's model ID to the Hub's canonical
`HMellor/tiny-random-LlamaForCausalLM` spelling and refreshing its local
snapshot reference, `npm run dev:e2e:distributed -- --timeout 240` passed on
PC-DANI: two signed logical workers, two ready stages with non-overlapping
ranges, two direct runtime links, matching canary and routed output
`"eno Twe"`, and two completion tokens. The coordinator listener was absent
after the gate and no new gate temporary directory remained. This is a
single-host loopback regression check, not two-computer evidence.
The revised MSI was installed on MRPUCCHI with Windows Installer exit code 0.
The packaged accelerator setup detected its NVIDIA GeForce RTX 2060, downloaded
the pinned 2,594,590,371-byte CUDA 12.6 PyTorch wheel, verified its exact size
and SHA-256, installed it into an isolated machine runtime, and passed an
actual FP16 matrix operation with PyTorch `2.13.0+cu126`. A temporary WinSW
service then ran the same cached-runtime probe under
`NT AUTHORITY\LocalService`; it again reported `gpu-ready`, `cuda`, the RTX
2060 and `fp16_matmul`. The first temporary service attempt failed with Windows
`Access denied` after its ACL was applied to already copied files; the second
applied directory permissions before copying, as the product installer does,
and passed. The temporary service and directory were verified absent afterward.
The GPU pack test was run separately from the one-click path. The revised MSI
was then removed with Windows Installer exit code 0. A subsequent remote check
found no registered product, `MycelliosNode` service, Program Files install root
or node configuration. The CUDA runtime cache remained and Tailscale was
running. No installed-node Ready state or assigned inference has been shown.
An accountless coordinator on PC-DANI was then reachable from MRPUCCHI through
an SSH reverse tunnel over Tailscale. Installing the first GPU-auto lab MSI
and redeeming its local-only pairing exposed three package/service defects:
the staged MSI omitted the root `package.json`; Windows PowerShell 5.1 did not
preload the DPAPI assembly and lacked the three-argument `File.Move` overload;
and WinSW 2.12.0 under LocalService remained `Running` without a Node child
after the node exited for its first configuration reconciliation. The first
two produced service exit 1067. The WinSW behavior matches its upstream open
issue #1136. Local source now packages `package.json`, loads the DPAPI assembly,
uses an atomic move/replace compatible with PowerShell 5.1, grants the
protected-identity directory to LocalService, and wraps the Node process in a
small supervisor. A temporary installed-tree patch on MRPUCCHI showed the
supervisor recover from a forcibly killed Node child (new Ready PID and online
worker) and stop cleanly through SCM with health `stopped`; that patch was not
an MSI result. The old product and lab configuration were removed, while the
CUDA cache and Tailscale remained.
The replacement supervisor MSI, 256,115,028 bytes with SHA-256
`edb88e3e67cf929c5208341def1995be21712302ada79ed03e7ca55dfe0f0231`,
passed staged-layout and administrative-extraction comparison. It was installed
on clean MRPUCCHI with Windows Installer exit 0. Its packaged pairing entrypoint
redeemed a fresh local-only lab bundle, reused and physically probed the cached
CUDA pack, restarted the service, and returned `status: installed`,
`backend: cuda`, exit 0. SCM reported `Running`, health and diagnostics reported
Ready with CUDA 12.6 and an NVIDIA RTX 2060, and the private coordinator saw one
connected online worker. This was **not** the USB launcher or a production
account. No model stage or inference was assigned in this test.
The installed node initially lacked the runtime performance callback required
for scheduler eligibility. A subsequent **manual compiled-main patch** on this
lab installation enabled the physical CUDA calibration, and the private
coordinator stored validated CUDA performance evidence for its online worker.
That calibration fix is in local source but **not** in the supervisor MSI above;
it requires a rebuilt MSI and clean installation before package-level credit.
The rebuilt calibrated lab MSI is 256,110,932 bytes, SHA-256
`64992db049759cbebfd35c9ebd29282bcd756719ba64ad2769ec2e589b8c24dd`.
Its staged layout and administrative MSI extraction passed verification. The
same bytes were copied to MRPUCCHI, where the hash matched, then installed after
removing the previous lab product; Windows Installer and the packaged pairing
entrypoint both exited 0. The service reached `Running` and `Ready` with CUDA
12.6 on the RTX 2060. Against a private, accountless PC-DANI coordinator, the
installed worker completed the coordinator's physical runtime challenge: the
stored, coordinator-sealed profile reports CUDA and 21 samples. Stopping and
starting the installed service through SCM caused a fresh signed registration
and a second successful CUDA challenge. The private coordinator uses a local
SQLite database and an SSH reverse tunnel over Tailscale; this is package-level
calibration evidence, **not** an assigned stage, generation, production pairing,
USB-launcher run or reboot-persistence result. Its temporary coordinator token
was rotated during diagnosis, which briefly produced HTTP 401 responses; the
service re-registered after the lab coordinator was restarted without that
ephemeral token.
The same calibrated MSI then served an actual stage in a private two-PC route.
MRPUCCHI's installed `MycelliosNode` service launched SmolLM2-135M-Instruct
layers `[0,26)` on CUDA through its packaged Job Object broker; a PC-DANI
source worker ran `[26,30)` on CPU. Both workers supplied coordinator-accepted
physical runtime profiles and the relay measured links in both directions. The
model was pinned to revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
The first 16-token canary passed with TTFT 96.2 ms, TPOT 75.8 ms and 13.20
output tokens/s; a separate eight-token request answered `2 + 2 = 4`.
Observed peak working sets for the stage processes were 1,521,766,400 bytes
on MRPUCCHI and 692,690,944 bytes on PC-DANI. The remote process parent chain
was `MycelliosNode.exe` → `node.exe` → `mycellios-job-broker.exe` → `python.exe`.
Removing the requested model stopped both stage processes and closed their
ports while leaving the installed service connected.

In a later run, MRPUCCHI rebooted while a 256-token request was still open.
The client failed after 4.1 seconds; the coordinator entered `waiting_capacity`
and the local stage exited. MRPUCCHI's Tailscale SSH and `MycelliosNode` service
returned after boot without local interaction. The temporary SSH reverse
tunnel to the private coordinator had to be re-established from PC-DANI; only
then could the node reconnect and re-calibrate. The coordinator initially
crashed on a duplicate deployment-operation idempotency key when trying to
reactivate a previously successful route. Local source now allocates a unique
operation attempt within each generation; a focused regression passes. With
that source fix, the coordinator stayed alive through a second in-flight
MRPUCCHI reboot, retried after capacity returned, completed a fresh 16-token
canary (TTFT 114.9 ms, TPOT 69.7 ms, 14.34 tokens/s), and answered another
`2 + 2 = 4` request. Final model removal again left zero stage processes on
both PCs. This is lab recovery with a manually restored tunnel, not proof of
unattended production reconnection. The installed MSI predates the CPU identity
and coordinator retry source fixes, and the PC-DANI CPU worker ran from source.
On MRPUCCHI, the exact pinned WinSW executable from the staged package was
copied into an isolated temporary service probe. Windows SCM started it as
`NT AUTHORITY\LocalService`; the wrapped PowerShell child stayed running for
three seconds, then `sc stop` reached `Stopped`. The temporary service,
directory and copied files were deleted and verified absent. This proves the
wrapper can run under SCM on that PC; it does not prove the Mycellios MSI,
node process, enrollment or inference works as an installed service.
An older GitHub Actions Windows x64 MSI was downloaded for read-only inspection.
Its checksum matched the CI evidence, and an administrative extraction into a
short path produced all 33,901 paths and hashes recorded in that build's node
layout manifest. The companion uploaded staged tree lacked five hidden files,
so `verify-node-release-evidence` rejected that downloaded tree. The workflow
now requests hidden files in future artifact uploads, but that change has not
run in CI. This MSI identifies an older CI merge revision, is not Authenticode
signed and was not installed or enrolled on either computer. Administrative
extraction is not an installation or a service/inference test.
An earlier lab USB node launcher and matching provisional MSI/checksum were
placed on the current USB, without an enrollment bundle. Its non-mutating preflight
accepted a synthetic MSI/checksum and bundle, and rejected a changed hash,
short expiry, invalid nonce and unexpected coordinator path. A further
preflight on the actual USB accepted the exact 256,090,372-byte candidate MSI,
its matching SHA-256 and a synthetic bundle; that bundle was removed afterward.
That earlier USB copy did not install an MSI or redeem a real bundle.

On 29 September, a newer **dirty-tree lab candidate** incorporating the CPU
identity and deployment retry fixes was built as a 256,115,028-byte MSI,
SHA-256 `42e445baa6ba52e3f41da491572f5e9e7f79e73867b69deb6f209b9ee4f6e53d`.
Staged-layout and administrative-extraction verification passed. Its hash
matched after transfer to MRPUCCHI. After a targeted removal of the preceding
lab node (Tailscale and the CUDA cache stayed), the USB launcher preflight
passed against a fresh private lab enrollment bundle. The same launcher then
installed this MSI, redeemed the bundle and exited 0 with `backend: cuda`;
SCM reported `Running`, health `ready`, and CUDA diagnostics. The private
coordinator accepted a sealed 21-sample CUDA performance profile. Production
launcher mode rejected the lab's HTTP loopback URL as intended; the lab run
explicitly enabled loopback HTTP.

With that installed service on MRPUCCHI and a source CPU worker on PC-DANI,
the pinned SmolLM2-135M-Instruct revision above was split `[0,26)` on CUDA and
`[26,30)` on CPU. Both physical profiles were accepted. The 16-token canary
passed at TTFT 96.7 ms, TPOT 69.6 ms and 14.36 output tokens/s. A separate
eight-token request through the installed remote stage answered `2 + 2 = 4`
in 789 ms wall time. Removing the requested model stopped both stage processes
and left no listeners on the tested stage ports; the node service remained.
This verifies the actual launcher and package against a private two-PC lab
coordinator through a manually maintained SSH reverse tunnel over Tailscale.
It is not an account-owned production pairing, a fixed-commit release, or an
unattended reboot-reconnection result. The physical USB at
`D:\MYCELLIOS-LAB-VPN\Node` still carries the earlier 256,090,372-byte MSI,
SHA-256 `f954dbd9447f171de7c79ae798a72b8e6446856a3c31e9603b63714c182ba7e0`;
it has no account enrollment bundle and was not updated with this lab build.
After correcting a release fixture and a Windows exit assertion exposed by
the full run, the complete Vitest suite passed: 280 files and 1,960 tests,
with 5 files and 17 tests skipped. Typecheck, structure, architecture,
documentation and format gates
also passed.

A second 29 September **dirty-tree lab MSI** adds supervisor child IPC cleanup
and preserves a resource-governor failure in service health. Its size is
256,119,124 bytes, SHA-256
`f2b2cd04cd452af8a67a66c9671670b2ef88c7b1a303f43cc177ba65e94d48d0`.
The staged tree and administrative MSI extraction matched; the transferred
hash matched on MRPUCCHI. After targeted lab-node removal, the USB launcher
passed preflight, installed and redeemed a new private bundle, and exited 0
with CUDA `Ready`. A coordinator challenge accepted its physical CUDA profile
after the coordinator and installed service were restarted; the initial
registration did not produce accepted performance evidence, and that delay
remains unexplained. This test still used the temporary Tailscale SSH reverse
tunnel to a private PC-DANI coordinator.
Local CycloneDX SBOM, checksum and provenance files were generated and
verified against this MSI and its restored staged tree. Their signature state
is `pending`; signed-release promotion was not attempted.

The installed node applied an account-shaped `set-limits` command changing
`maxCpuPercent` from 90 to 89. Its main child changed PID 9652 → 7416 while
the service supervisor stayed running, and returned to `Ready`. Restoring 90
changed the child to PID 9388 and again reached `Ready`; the coordinator
reported the new snapshot. Forcing a later installed main child (PID 9556)
to exit made the packaged supervisor start PID 1628; health returned to
`Ready`, SCM remained `Running`, and the coordinator worker was `online` with
accepted CUDA evidence. The earlier installed lab candidate also applied
`pause` and `resume`: local contribution changed false then true, while its
coordinator worker changed `draining` then `online`. These prove physical
command execution through the native node, not the authenticated owner UI.

With the supervisor MSI, pinned SmolLM2-135M-Instruct split `[0,25)` on
MRPUCCHI CUDA and `[25,30)` on PC-DANI CPU. Both stages became ready; the
16-token canary passed with TTFT 121.6 ms, TPOT 71.2 ms and 14.04 output
tokens/s. A separate eight-token request answered `2 + 2 = 4` in 758 ms.
Removing the model left zero distributed stage processes and listeners on
both PCs, while the MRPUCCHI service remained `Ready`. This is another lab
package proof, not a fixed-commit release or unattended production route.
After this supervisor change, the full Vitest suite again passed 1,960 tests
(17 skipped across 5 skipped files).
The same staged package's bundled Windows Python on PC-DANI reported
PyTorch `2.13.0+cpu`, zero CUDA devices and its real CPU device name. Its
packaged runtime profiler completed 21 physical samples each for decode
memory, prefill compute and activation codec on CPU. This is a package
preflight, not an installed CPU-only node or coordinator challenge.

### Session recovery and a repeated installed-node route (29 September, lab)

Restarting the private PC-DANI coordinator while MRPUCCHI retained a session
from its prior random network secret left the installed worker running but
repeating WebSocket HTTP 401. A focused network test now verifies that a worker
re-registers after this rejection (13 tests passed in `worker-hardening.test.ts`).
The lab coordinator also now reads a private secret retained in its ignored
runtime state. With that secret unchanged, restarting only the coordinator
restored two connected workers without restarting MRPUCCHI; its service PID
remained 9176. This is coordinator-process recovery over the existing manually
maintained SSH reverse tunnel, not an unattended PC reboot or durable HTTPS
deployment.

A new **dirty-tree lab MSI** including the worker recovery fix was built,
its staged tree and administrative extraction verified, and its SHA-256
matched after transfer to MRPUCCHI: 256,123,268 bytes,
`6c6c7d03c46bb378fe1659a2633c2f80cb293f878a5af8e469e558493cdd7af4`.
The previous lab MSI uninstalled with exit 0 but left its separately registered
service and configuration in place. Those specific lab directories were backed
up and the service registration removed before the clean-install preflight;
Tailscale and the model cache remained. This is an **upgrade/reinstall obstacle**
for an unattended user path, not a successful automatic upgrade. The fresh USB
launcher then passed preflight, installed with exit 0, consumed the new
one-time pairing file, and reported CUDA `Ready`. The installed `agent.js` hash
matched the verified staged package.

The lab coordinator secret was then deliberately rotated and the coordinator
restarted while the newly installed MRPUCCHI service stayed at PID 11992.
The installed worker logged a WebSocket HTTP 401, re-registered without a
service restart, reconnected, and submitted an accepted RTX 2060 CUDA runtime
profile. Pinned SmolLM2-135M-Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac` then ran with MRPUCCHI CUDA
layers `[0,26)` and PC-DANI CPU layers `[26,30)`. The 16-token canary passed:
TTFT 106.2 ms, TPOT 77.5 ms, 12.90 output tokens/s. A separate eight-token
request returned `2 + 2 = 4` in 866 ms wall time. Deleting the model closed
its proxy and left zero stage processes on both hosts and zero listed stage
ports on MRPUCCHI; the Mycellios service remained running and both workers
connected. The focused test, typecheck, structure/build and architecture gates
passed after the source fix. This remains an unsigned, uncommitted lab build
using the manual tunnel, with no production account pairing or external tester.

The MSI generator now includes an uninstall-only `ServiceControl` for
`MycelliosNode`. The focused MSI/layout tests passed (9 tests). On MRPUCCHI, a
separate 32,768-byte probe MSI with that WiX service-control pattern uninstalled
with exit 0 and removed its deliberately registered disposable service;
MycelliosNode and Tailscale stayed running. This proves the Windows Installer
mechanism on that PC. The **full dirty-tree node MSI** with this change was
then built and verified by administrative extraction: 256,123,328 bytes,
SHA-256 `f67525b6928088a2926db774a7fed15dbff3ad8f8957f426d2d8bd3d2619a0c4`.
The transferred hash matched. A same-version Windows Installer repair returned
1603 because its secure-repair source lookup referred to an older MSI path
that was no longer present; it left the running node intact. After backing up
the previous lab configuration and clearing the previous service registration,
the fresh USB install of the new MSI passed preflight, consumed a new pairing
bundle, and reported CUDA `Ready` with two workers connected.

Uninstalling **that full MSI** returned 0, removed the MycelliosNode service,
and retained node configuration, identity, wrapper, cache and Tailscale. The
same MSI files were then reinstalled without consuming another pairing token.
Re-registering the retained wrapper required explicit service-account and
recovery settings: direct WinSW registration initially chose LocalSystem;
setting `NT AUTHORITY\LocalService` and the restart policy restored the
intended service configuration. MRPUCCHI returned to CUDA `Ready` with the
same node ID `node-4c3832ee0e884100ba0a6596a400192c`, automatic service
start and two connected workers. Service removal is now physically proven;
**one-click reinstall and same-version upgrade remain unproven** because the
launcher deliberately rejects retained configuration and this recovery needed
manual service registration. This is still an unsigned, uncommitted lab MSI.

After restoring that installation, MRPUCCHI was rebooted at
2026-09-29 01:36:40 UTC with no model active. Windows restarted Tailscale and
MycelliosNode automatically under LocalService, but the manually created
reverse SSH tunnel disappeared. MRPUCCHI had no listener at `127.0.0.1:18790`,
its worker process logged `ECONNREFUSED`, health stayed `starting`, and the
coordinator saw only the PC-DANI worker. Restoring the tunnel brought the
**same running service** back to CUDA `Ready` and two connected workers; no
service restart was needed. This is a physical failure of the current network
route across a PC reboot, not an unattended-reconnection pass.

An ignored lab-only PowerShell supervisor was then started on PC-DANI to
re-establish the same SSH reverse tunnel after its process exited. A second
MRPUCCHI reboot was requested at 01:40:10 UTC. The supervisor observed SSH
exit 255 and retried while the host was offline; Windows booted at 01:40:32
UTC, and the coordinator observed two connected/online workers by 03:41:23
Europe/Madrid time. MRPUCCHI's service remained automatic under
`NT AUTHORITY\LocalService`, Tailscale was running, and health returned to
CUDA `Ready`. This proves recovery **while a PC-DANI supervisor process remains
active**, not persistence of the tunnel across a PC-DANI reboot or a durable
HTTPS coordinator path.

On 2026-09-29, the USB launcher gained a retained-identity update path. Its
read-only preflight passed on MRPUCCHI without a new enrollment file. A
dirty-tree 0.2.78 MSI passed staged-layout and administrative-extraction
verification, but its physical upgrade from 0.2.77 **failed after MSI
replacement**: the service restorer tried to set permissions on the already
consumed `enrollment.json`. The 0.2.78 product was installed with its service
absent. The retained wrapper was manually registered under LocalService with
the recovery policy; the same node ID returned to CUDA `Ready` and two workers
connected. This failure is part of the evidence, not a successful one-click
upgrade.

The restorer was corrected to omit the consumed enrollment file only on a
retained-identity restore. The 0.2.79 lab MSI (256,123,376 bytes, SHA-256
`7dd73e6db428ac4a8f3f03734c488f7489cc9a213a4571588560a9576d4ea0f1`)
passed layout and full MSI extraction verification. Its source identity was
`sha256:a4d816f41ff85b814937f901fbb6f4f905f9abee40b6a19deae4507b8b42c8aa`
on dirty worktree HEAD `ff835802815f3749d72b8deb4b2ea5978b479f3b`.
After matching the transferred hash and passing USB preflight, the same
launcher upgraded MRPUCCHI from 0.2.78 to 0.2.79 without pairing or manual
service registration. It retained node ID
`node-4c3832ee0e884100ba0a6596a400192c`, registered an automatic
LocalService service, reached CUDA `Ready`, and restored two connected workers.

Post-upgrade, pinned `HuggingFaceTB/SmolLM2-135M-Instruct` revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac` assigned MRPUCCHI CUDA
layers `[0,26)` and PC-DANI CPU layers `[26,30)`. Both stages reached Ready;
the real 16-token canary passed with TTFT 121.3 ms, TPOT 98.5 ms and 10.15
output tokens/s. Deleting the model removed it from the coordinator and left
zero stage processes and stage ports on both PCs, while two workers remained
connected. Re-running the same USB file against the healthy 0.2.79 install
reported Ready without changing the service PID. In a separate physical
recovery test, only `MycelliosNode` was stopped and deregistered; the same USB
file recreated it in about 19 seconds with no new pairing. The node ID,
automatic LocalService account, CUDA Ready health, Tailscale and two connected
workers were verified afterward. These are lab results using a temporary
reverse SSH tunnel and an unsigned, uncommitted MSI, not a released external
installation or durable coordinator connection. The USB launches in these
physical runs used an already elevated SSH session; the standard-user UAC
handoff still needs a tester run. A later script revision moved retained
identity preflight behind that single UAC prompt and passed read-only preflight
and idempotent execution on MRPUCCHI without changing the service PID.

With no model active, MRPUCCHI reboot was requested again at 2026-09-29
02:37:13 UTC with the 0.2.79 MSI installed. The PC-DANI tunnel supervisor
observed the SSH disconnect and retried while the host was offline. After the
new Windows boot, Tailscale and MycelliosNode were `Running`, the latter
automatic under LocalService with the unchanged node ID. Health reached CUDA
`Ready` at 02:38:29 UTC and the coordinator again showed two connected/online
workers. This verifies installed-service recovery after reboot **while the
temporary PC-DANI tunnel supervisor stays active**; it does not verify an
unattended route after PC-DANI itself restarts.

A second activation of the same pinned model after that reboot again assigned
MRPUCCHI CUDA `[0,26)` and PC-DANI CPU `[26,30)`. Its real 16-token canary
passed with TTFT 122.9 ms, TPOT 132.9 ms and 7.53 output tokens/s. While the
stages were active, `Get-Process` reported peak working sets of 1,452.7 MiB
for MRPUCCHI's CUDA Python process and 662.0 MiB for PC-DANI's CPU Python
process. Windows WDDM `nvidia-smi` reported zero GPU memory use while the CUDA
stage was running, so that value is not a credible VRAM measurement. Deleting
the model again left zero `distributed_runtime` Python processes and zero
stage listeners on both PCs; the installed service stayed CUDA `Ready` with
two connected workers. The two canary rates describe individual lab runs and
do not establish a performance regression or improvement.

The 0.2.79 lab candidate also passed `npm test` (280 files, 1,963 tests;
5 files and 17 tests skipped), 33 focused Python `unittest` cases for physical
probing and the distributed server, TypeScript typecheck, repository structure,
architecture and documentation gates. A separate folder on the mounted lab USB
contains the verified 0.2.79 MSI, checksum and launcher, but deliberately has
no short-lived enrollment bundle. It is not ready for a first installation
until an account-owned bundle is issued, and it is not a signed release.

On 29 September, a separate dirty-tree 0.2.80 **lab candidate** was built
after rebuilding the pinned Windows Python runtime and Job Object broker.
The staged node layout and administrative MSI extraction both verified. The
MSI is 256,127,504 bytes with SHA-256
`14d2f03230467a6488044c5aefefbaeabf6f57400a174518aea880c9378c1e5f`;
its source ID is
`sha256:20ebaef654d23e7f99e21ecbe2aa7ef2e3337834e1c0266b23dc6b23a64556a5`
on dirty-tree HEAD `ff835802815f3749d72b8deb4b2ea5978b479f3b`.
The copied MSI on the mounted lab USB has the same hash and Windows Installer
reports product version 0.2.80. The first WiX link attempt failed on long
Torch paths; mapping the same worktree to a short temporary drive path let the
link and full MSI verification pass. The USB folder has no enrollment bundle
and is not suitable for a first installation.

The same MSI, checksum and launcher were copied to MRPUCCHI, where the hash
matched and the launcher preflight passed without a new pairing. The already
elevated SSH session ran the launcher and upgraded the existing 0.2.79 node to
0.2.80. Windows Installer registered one product; the node kept its ID
`node-4c3832ee0e884100ba0a6596a400192c` and coordinator binding. Its
automatic `MycelliosNode` service ran under `NT AUTHORITY\LocalService`,
reported CUDA `Ready`, and restored two connected/online control workers.

The pinned SmolLM2 revision activated as `lab-upgrade-0280-20260929` with
MRPUCCHI CUDA layers `[0,26)` and PC-DANI CPU layers `[26,30)`. Both stages
reached Ready. The 16-token canary passed with TTFT 126.9 ms, TPOT 102.2 ms
and 9.79 tokens/s. A first immediate coordinator chat request returned 503
`no_capacity` while the newly published cell deployment was still being
verified; its deployment subsequently reached `verificationState: verified`.
A second ordinary chat request completed as
`job_39d3cc38ba864b0190604eb63778c14f`, returning `2 + 2 is 4.` with
45 prompt and nine output tokens. Its lab-signed receipt
`sha256:2f7273815bad5f27974e9170b280e73bc453ad4500bb53580d9b753e256dee93`
reported TTFT 146 ms and active time 910 ms; its attached topology classified
one direct GPU-to-CPU stage boundary as physical. The signing key was the
ephemeral lab key, not a public release identity. While active, the MRPUCCHI CUDA Python
process reported a 1,523,032,064-byte peak working set and the PC-DANI CPU
process 693,653,504 bytes; these are process memory readings, not VRAM.
Deleting the requested model left zero distributed-runtime Python processes
and zero stage listeners on both PCs, with two control workers still online
and no requested model. Re-running the launcher against the healthy 0.2.80
installation returned Ready with the same service PID. This is physical lab
evidence over the temporary reverse SSH tunnel and an already elevated
session; the MSI remains unsigned, unpublished, built from a dirty worktree,
and untested through standard-user UAC or a durable coordinator route.
With no model active, MRPUCCHI reboot was requested at 04:23:17 UTC. Its new
Windows boot time was 04:23:42.5 UTC; Tailscale and the automatic LocalService
node were Running, diagnostics reported 0.2.80 CUDA `Ready`, and the lab
coordinator again had two connected/online workers. The PC-DANI SSH tunnel
supervisor was still running and retried through the outage, so this verifies
the installed service after reboot but not recovery without that supervisor.
The full 0.2.80 Vitest run initially reported 1,963 passing tests and 17
skips but exited nonzero when a worker fork terminated unexpectedly. Repeating
with `--maxWorkers=2` exited zero: 281 files and 1,968 tests passed, with
five files and 17 tests skipped. The strict Python runner passed 1,094 tests
with 19 skips, zero failures and zero errors. Format, component boundaries,
landing/mobile typechecks, documentation and `git diff --check` also passed.

On 29 September, the authenticated public Downloads page reported "Packages
unavailable" and no native packages published. The public `/health` endpoint
reported coordinator version 0.2.77 with zero connected/online workers. This
does not affect the separate two-worker local lab, but the public account and
download route cannot yet support the external native-node procedure.
The dirty worktree now aligns a new public release transaction with the MSI,
PKG and DEB filenames produced by the native build and keeps parsing of
previously committed ZIP/TAR.GZ transactions for rollback. The downloads API
and page prefer a committed native installer and report its real format; a
legacy archive remains selectable only when that is the available asset.
Four focused release/download test files passed (34 tests), including a
committed legacy archive restored after rolling back an installer release;
typecheck, structure, architecture and documentation gates passed. No native
installer has been published through this new route, so public availability
and an external download remain unverified.
The same dirty worktree now has a candidate assembler for the three native CI
artifact trees. It checks each staged payload and release evidence, requires
the same source revision/ID/version and CI run/attempt, verifies copied bytes,
and writes one update feed and checksum list before making the output directory
visible. Three synthetic CI job trees passed assembly into the public manifest;
mixed-run and changed-package fixtures were rejected. The focused assembler,
transaction CLI and upload tests passed (21 tests), along with typecheck,
structure, architecture, format, documentation and version checks. No real
three-platform CI artifact set, cryptographic signature check or public upload
has been exercised. The provenance `signature.state` field alone is not proof
 of code signing. The native workflow now includes a candidate assembly job
 after all three matrix builds. Its YAML passed local actionlint 1.7.12, but
 this new job has not run on GitHub.

The three real 0.2.77 CI artifacts from run `36347391341` were subsequently
downloaded and each ZIP matched the SHA-256 reported by GitHub. Their source
revision, source ID, run ID and attempt matched across Linux, macOS and Windows.
The assembler rejected the unmodified set with
`node_installer_layout_digest_mismatch`: GitHub's older upload had excluded
five hidden staged files on Linux, nine on macOS and five on Windows. The
macOS omissions included four NumPy `.dylibs` libraries needed at runtime.
The current dirty workflow sets `include-hidden-files: true`, but that change
has not run in CI, so no real complete three-platform set has passed the
assembler yet. A focused fixture now removes a required hidden runtime file
and confirms that the assembler rejects it (four focused tests passed).
The public transaction store checks release identity and file hashes but does
not cryptographically verify Windows, macOS or Linux package signatures at
commit time. The local 0.2.80 lab MSI returned `NotSigned` from Windows
Authenticode. No usable code-signing certificate was found in the inspected
PC-DANI certificate stores, and no signing-related GitHub Actions secret name
was listed for this repository. A public promotion still needs a verified
signing process and explicit platform signature checks; a CI candidate manifest
alone cannot satisfy that gate. The live public downloads API reported 0 of 3
packages available on 29 September.

The PC-DANI lab coordinator was restarted from the current dirty 0.2.80
worktree, and its local CPU worker now reads that package version rather than
an older hard-coded lab value. MRPUCCHI's installed 0.2.80 `MycelliosNode`
remained Automatic/Running, and the coordinator reported two connected/online
workers. The same pinned SmolLM2 revision activated as
`lab-aligned-0280-20260929`: MRPUCCHI CUDA layers `[0,26)` and PC-DANI CPU
layers `[26,30)`. The real 16-token canary reported TTFT 120.3 ms, TPOT
85.2 ms and 11.73 tokens/s. Three ordinary 16-token chat completions followed:
`job_0fe9127f587a4d9c8daebfd73c484dca`,
`job_7ecd5c8358cd429f82de10d117faec20`, and
`job_b6ec228535fb44f1bddec2498bd12da7`. Their receipts reported TTFT
3,617/1,875/2,036 ms and active time 7,430/6,718/6,253 ms respectively.
The last job's attached topology classified a direct physical CUDA-to-CPU
boundary. These normal-request timings are materially slower than the internal
canary and should be treated separately. The MRPUCCHI stage Python process
reported 1,632,641,024 peak working-set bytes and the PC-DANI stage
693,784,576 bytes, not VRAM measurements. Deleting the requested model left
zero distributed-runtime Python processes and checked stage listeners on both
PCs, while two control workers remained connected. This aligns the running
version numbers, but the lab still uses a dirty worktree and the temporary
reverse SSH tunnel; it does not prove identical committed source or unattended
coordinator reachability. A trial tailnet-only Tailscale TCP proxy worked on
PC-DANI but timed out from MRPUCCHI; it was removed after that failed test.

The timeout was traced to the current Tailscale packet filter: MRPUCCHI was
allowed to reach PC-DANI on TCP 18100, not on the trial port 18791. PC-DANI
now has a background, tailnet-only
Tailscale Serve TCP proxy from port 18100 to its loopback coordinator on 18790.
MRPUCCHI fetched the coordinator's 0.2.80 `/health` through that route. Its
automatic Windows IP Helper service has a persistent loopback port proxy from
`127.0.0.1:18790` to PC-DANI's tailnet port 18100; the temporary reverse SSH tunnel
supervisor was stopped, and the existing node configuration stayed on its
loopback URL. Two control workers reconnected without the SSH tunnel. The
extra trial port proxy on MRPUCCHI was removed.

With no model active, MRPUCCHI rebooted again. Its boot time changed from
04:23:42.5 to 04:45:49.5 UTC on 29 September. Tailscale, IP Helper and the
automatic `MycelliosNode` service were Running; its loopback proxy returned
coordinator `/health`, and the coordinator returned to two connected/online
workers without restarting the SSH tunnel. A fresh activation of the pinned
SmolLM2 revision as `lab-durable-0280-20260929` again assigned MRPUCCHI CUDA
`[0,26)` and PC-DANI CPU `[26,30)`. Its real 16-token canary reported TTFT
119.9 ms, TPOT 83.7 ms and 11.94 tokens/s. An ordinary 16-token chat request
completed as `job_a1a4e89ce7554681b5ebe0d83aeaca79`, with receipt
`sha256:00ddb0d9d77c975e0d7b4478065999fd88f7cf851648e05fd65b637ad3e82b94`,
TTFT 3,244 ms, active time 7,549 ms, and an attached topology classifying a
direct physical GPU-to-CPU boundary. Its generated text was cut at the
16-token limit, so this run proves completed inference and routing rather than
answer quality. Deleting the model left zero distributed-runtime Python
processes on both PCs while two control workers stayed connected. This proves
MRPUCCHI reboot recovery and a physical inference through the persistent
tailnet route. PC-DANI's coordinator is still an interactive lab process;
unattended PC-DANI reboot recovery, a sealed same-commit release, standard-user
installation and external availability remain unverified.

On 29 September a separate private-lab failure test activated the same pinned
SmolLM2 revision as `loss-mrpucchi-20260929`. The reservation placed PC-DANI
CPU layers `[0,4)` and MRPUCCHI CUDA layers `[4,30)`; the committed canary
returned 16 tokens with TTFT 109.8 ms, TPOT 79.0 ms and 12.67 tokens/s. A
normal streaming request to the root runtime at `127.0.0.1:18881` returned
HTTP 200 and its first token after 225 ms. After that token, `MycelliosNode`
was stopped on MRPUCCHI while the stream was still running. The stream ended
after 14.4 seconds with an SSE `service_unavailable` error, "pipeline connection
failed: socket closed while receiving a frame", followed by `[DONE]`. Thus the
client received an explicit stream error, not a successful complete answer.
Both computers then had zero distributed-runtime Python stage processes and
zero stage listeners. Removing the requested model succeeded while MRPUCCHI
was offline; restarting its service restored `Ready` and two connected/online
workers, with no requested model or stage process left behind. This tests
service loss during generation, not a full PC power or network outage.

A separate ordinary streaming request to the lab coordinator's
`/v1/chat/completions` for that committed model returned `503 no_capacity`
after a 45-second admission wait, despite the direct runtime canary passing.
The cause was the lab harness omitting the cell worker publication setting;
native control workers correctly do not advertise inference deployments. After
adding a loopback lab coordinator target and scoped lab credential to the
ignored harness, activation published a `mycellios-pipeline` worker. Its
deployment canary reached `verified`. A normal coordinator chat job
`job_0ba3a776fb4a4a2a8f98efb556df9353` completed with eight output tokens,
answer `2 + 2 = 4`, and 43 input tokens. The coordinator issued receipt
`sha256:b1c1a734cb67151351f16eed7da6958c1016d99b7d0df0060dd52ffbb4441774`
with TTFT 174 ms and active time 2,030 ms, signed by an **ephemeral lab** key.
Its attached topology classified the execution as physical, with CPU stage 0,
CUDA stage 1 and one physical relay boundary. This is a private-lab
coordinator result, not a production account or release result.

After the cell worker first appeared, the coordinator reactivated the model
once as its worker inventory changed. A first streaming probe used the default
temperature and failed with `adapter_error` because the root runtime at that
point rejected nonzero temperature; it did not cut MRPUCCHI. Once the second
activation and deployment verification finished, a coordinator SSE request
with `temperature=0` returned HTTP 200 and its first token after 366 ms. The
MRPUCCHI service was then stopped during generation. The stream ended after
14.3 seconds with the explicit `worker_disconnected` error and `[DONE]`, and
the coordinator recorded failed job `job_37c537464be04ae8845c5d8547417bc6`
with that failure code. Both hosts had zero distributed-runtime Python stage
processes and zero stage listeners. The model was removed while MRPUCCHI was
offline; restarting the service restored two connected/online native workers
and left no requested model active. There is no evidence of automatic
mid-stream continuation after a remote node disappears.

Later on 29 September, the dirty worktree added sampling parameter forwarding
to the Python batcher and a separate admission key per automatic cell identity.
The historical shared key is tried only when a cell has no dedicated key;
`worker_credential_reused` then triggers one retry with a dedicated key. The
focused TypeScript tests passed (23 cases), as did the Python server tests
(33 cases) and typecheck. In the physical lab, a new pinned SmolLM2 request
`lab-sampling-20260929` again assigned PC-DANI CPU layers `[0,5)` and MRPUCCHI
CUDA layers `[5,30)`. Both stages reached Ready; the 16-token canary passed
with TTFT 102.7 ms, TPOT 116.6 ms and 8.57 tokens/s. This time cell
registration succeeded and the requested model became active with three
connected workers. An ordinary coordinator chat request omitting temperature
completed with six output tokens instead of the earlier `adapter_error`.
Two requests with `temperature=0.7`, `top_p=0.8` and the same seed produced
the same 12-token output; this is a reproducibility spot check, not a sampling
quality or performance comparison. Deleting the requested model left zero
distributed-runtime stage processes and stage listeners on both PCs and two
connected native workers. The old shared key and one dedicated cell key both
remained on disk. This verifies the one-time migration retry for this lab cell,
not a released build or account-owned production enrollment.
The previously enrolled `lab-chat-20260929` cell was then reactivated with
the same pinned model and historical key. Its two physical stages reached
Ready, the 16-token canary passed (TTFT 92.5 ms, TPOT 128.0 ms, 7.82
tokens/s), and a coordinator chat job completed with eight output tokens.
After deletion, both hosts again had zero stage processes and the coordinator
had no requested model, with two native workers still connected. This checks
backward compatibility for the one previously enrolled lab cell; it does not
cover every preexisting installation or credential recovery after key loss.

## Physical two-computer CPU smoke test (local, uncommitted)

On 2026-09-26 the native automatic route generated through two Windows PCs on
the same LAN: PC-DANI (`192.168.1.79`) ran layers `[0,15)` and MRPUCCHI
(`192.168.1.2`) ran `[15,30)`. The remote stage was a Python process on
MRPUCCHI; both its outbound return connection and the root engine's connection
to `192.168.1.2:18101` were observed as established. The transport was direct
TCP with fp16 activations. Jarvis was used only to control MRPUCCHI, not as an
inference node.

The run used isolated source commit `70a8617ab784a0963e83ab15eb1f2b5bf0028096`
plus the uncommitted `buildIdentity: null` launch-agent fix. Both machines used
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`; the model weight SHA-256
matched on both hosts (`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`).
Health reported two stages and the canary returned 16 tokens. Its exact text
was `I'm sorry for any misunderstanding, but as a chatbot, I don't`; measured
TTFT was 694 ms, TPOT 188 ms and output rate 5.32 tokens/s. A separate live
request answered `2 + 2 = 4` with eight output tokens, TTFT 358 ms and TPOT
105 ms. The Python root process peaked at 966,189,056 working-set bytes and
the remote stage at 925,556,736 bytes. The root reported a 4.93 ms route-probe
round trip; ten PC-DANI-to-agent TCP connects had a 4.57 ms median. These are
single-run observations, not sustained performance benchmarks or one-way link
measurements.

The coordinator exited cleanly. Both stage processes and the remote launch
agent stopped, test listeners closed, temporary firewall rules were removed,
and MRPUCCHI's original TCP Block rule was restored. This proves a small CPU
model can execute across these two physical computers. It does not establish
GPU splitting, a combined-memory advantage, output parity against a monolithic
baseline, recovery after host loss or a signed physical receipt. The automated
canary checks non-empty output and metrics; it did not require the model to
obey the `Reply with only OK.` instruction. The activation `status.json` still
contains `state: running` after clean shutdown; process and port checks, not
that stale snapshot, establish cleanup.

## Production-coordinator inference from the two-computer CPU route

On 2026-09-26, the same SmolLM2 model split across PC-DANI `[0,15)` and
MRPUCCHI `[15,30)` completed a canary while connected to the production
Mycellios coordinator (version `0.2.77`). The first cell registered as
`wrk_a13065967d2447649fa882b2f2a284e3`, but the production scheduler
excluded its deployment: the cell's internal stage telemetry was interpreted
as dependencies on two additional public workers. A second cell,
`wrk_f4a7a612a30e4794981c70f9ad1e23f2`, advertised the same live root
engine as a self-contained internal pipeline, without the per-stage execution
telemetry. The coordinator verified its canary and listed
`two-host-smollm2-135m` as an available one-pipeline model.

A production API request to that model completed as
`job_609794ad936d4195a5679a1822c9fdd1`. It returned `2 + 2 is 4.` with
45 input and 9 output tokens. Its signed execution receipt is
`sha256:d9623ba355e452773143b2fc8dc9015fe59bf8ce8b33ac83144e84e7d205c555`
(`keyId: mycellios-prod-economic-receipt-v1`); the receipt reports 572 ms
time to first token and 1,897 ms active time. The public completed-job counter
rose from 54 to 55, and the job, receipt and public counter were read back
from production after completion.

This is evidence of a real coordinator-routed inference while the two physical
CPU stages were running. The production receipt labels the route `replica`, and
its topology record labels it `loopback` with no stages because the registered
cell does not expose its internal stage topology to the coordinator. Thus the
signed receipt alone is **not** proof of the physical two-host placement. The
physical placement is established by the local two-host process, socket and
canary observations above; the exact production job's stage-by-stage path was
not attested in the receipt. The public WebSocket ingress also returned HTTP
404 to valid worker upgrade attempts, so this run used a temporary SSH tunnel
to the same production coordinator. These nine output tokens are inference
usage, not issuance or settlement of a public blockchain token.

After shutdown, the public `/network` page displayed 55 completed jobs and the
new job in recent activity with 54 total tokens (45 input plus 9 output).
The temporary cells had deregistered, leaving the original two registered
nodes and zero connected nodes. The local API/stage and SSH tunnel listeners
were closed and the PC-DANI temporary firewall rule removed. On MRPUCCHI the
test agent and stage were stopped, ports 9750/18101 were free, its original
TCP/UDP Block rules were enabled, and the two temporary Allow rules removed.

## Local browser GPU probe (2026-09-26, uncommitted)

The self-contained page `tests/browser-webgpu-matmul.html` was served only on
`127.0.0.1` and run in Chrome 154 on the AMD Radeon 890M. A 1024 x 1024
float32 matrix multiplication executed through WebGPU core. The final result
matched three independently computed CPU cells (maximum relative error
3.14e-7). In one 12.213-second run with 16 repeated dispatches per submission,
the page measured 880 dispatches, 154.74 GFLOPS end to end, and 223.64 GFLOPS
within one GPU-timestamped compute pass. These figures measure this simple WGSL
kernel only. They do not establish peak hardware use, native performance parity,
mobile performance, or complete-model inference. No production service was
changed. The page can be repeated locally with
`node tests/serve-browser-webgpu.mjs` and its displayed Start button.

Further 1024 x 1024 trials of the same page gave 159.33 end-to-end GFLOPS
with the scalar F32 kernel, 161.91 with F16 inputs and F32 accumulation, and
98.41 with four outputs per thread in F32. The four-output variant was slower
in this trial despite passing the same correctness checks. A 1536 x 1536 F16
trial gave 162.04 GFLOPS. Each kernel/size figure is a single run under
changing load and clocks; the small F16 difference is not established as an
improvement. F16's maximum checked absolute error against F32 reference was
1.54e-4 across the two measured sizes.

For the scalar 1024 x 1024 F32 operation, changing only the number of
dispatches per submitted command buffer from 1 to 32 raised one-run sustained
throughput from 57.08 to 178.56 GFLOPS. The corresponding GPU-timestamped
pass rates were 237.13 and 218.07 GFLOPS. The page waits for completion after
each submission, so this isolates a scheduling/round-trip cost in that
particular test rather than an optimization already proven for autoregressive
generation.

The same scalar F32 kernel with 16 dispatches per submission measured 58.64
GFLOPS using 8 x 8 workgroups, 159.33 using 16 x 16, and 127.87 using
32 x 32, each over one 12-second run. All passed the CPU cell checks. This
small sweep favors 16 x 16 on the tested Radeon/browser; it is not a general
autotuning result.

During a 30.275-second browser run at 16 x 16 and 32 dispatches per
submission, the page reported 165.70 sustained GFLOPS and correct output.
Twelve one-second Windows GPU Engine samples for Chrome's GPU process on the
Radeon 3D engine averaged 76.7% utilization and peaked at 96.5%. This counter
records engine busy time for a process, not shader ALU occupancy, guaranteed
exclusive access, or the hardware's theoretical peak. It shows the page can
drive that engine close to full busy time briefly, while the 30-second average
was lower.

The local `tests/browser-webllm-smollm2.html` page also completed all model
operations in Chrome through WebLLM 0.2.85 and WebGPU on the same non-fallback
Radeon 890M. It loaded the public SmolLM2-135M-Instruct q0f16 MLC model,
pinned to model revision `06fc409f2212adff29d93cbc742c1d6957302df5`
and WebLLM binary-library revision
`025bcaf3780fa8254f5e5efd3bfea0a5397248f4`. Its fixed prompt used
41 tokens and produced the same 21-token answer in each of three runs. The
first run included shader preparation (35.642 s, 3.03 decode tokens/s); the
next two ran at 7.42 and 7.41 decode tokens/s, with 89.5 and 161.8 reported
prefill tokens/s. The page fetches public model/code artifacts from remote
providers but performs inference locally; it was only served from loopback.

A native reference used `llama-bench` build `571d0d540` (10068), Vulkan,
`-ngl 99`, and the local F16 GGUF with SHA-256
`ee7608cab6e375ae1c2446cb0e4c0193e2756fab3df2fd1327a7a04e571df56f`.
Verbose load output assigned all 30 transformer layers and KV cache to
Vulkan0. For `-p 41 -n 21 -r 3`, it measured 1280.0 prompt and 63.64 decode
tokens/s on average; `-p 256 -n 64 -r 3` measured 2076.5 and 39.20.
These native tests use synthetic token sequences, and the browser uses a
different model artifact and chat execution path. Their ratio is a diagnostic
signal, not a controlled native-parity result. No browser run has yet shown
native-equivalent performance or exclusive/full physical GPU use.

For a closer single-prompt comparison, a loopback-only `llama-server` process
used the same chat message, temperature 0, seed 1 and 64-token cap as the
browser page, with prompt caching disabled. Three native requests all returned
the exact same answer text as the browser; each counted 41 input tokens and
22 output tokens (the browser counted 21 output tokens). Native warm requests
reported 74.72 and 62.88 decode tokens/s, versus 7.42 and 7.41 in the pinned
browser run. Native warm wall times were 0.349 and 0.414 s, versus 3.306 and
3.102 s in the browser. Native prompt times were 43.35 and 38.35 ms; the
browser reported 0.458 and 0.253 s to first token. The loopback server was
stopped after the test. This matched-output result makes the performance gap
concrete for this small request, but different weight containers, execution
engines, output-token accounting and an unmeasured memory footprint still
prevent a strict native-equivalence claim.

A local diagnostic wrapper around WebGPU calls counted 132 queue submissions,
132 command buffers and 7,129 compute passes/dispatches for each 41-input,
21-output-token WebLLM request. It is instrumentation rather than a GPU trace,
and its additional JavaScript work may affect timing. It shows that this engine
already batches many compute passes into each submission; improving this model
requires profiling pass-level work and fusion, not merely changing the submit
batch size used by the synthetic matrix page.

An expanded wrapper also counted 7,458 `queue.writeBuffer()` calls and 22
buffer maps per request. In one warm request lasting 2.139 s, synchronous
time inside the wrapped submit, pass-begin/dispatch and write calls summed to
about 38 ms. This excludes shader execution, asynchronous browser/GPU-process
work and other JavaScript; it does not locate the remaining time precisely.
The 22 maps indicate a per-token readback path worth inspecting before
changing the model graph. A follow-up instrumented run found that each map
read exactly four bytes. Three requests returned the same answer and took
2.832, 2.456 and 2.235 s; the summed `mapAsync()` waits were 2.320, 2.102
and 1.874 s. Another three warm requests without a Chrome trace had map waits
of 2.089, 1.859 and 1.571 s against wall times of 2.605, 2.221 and 1.883 s.
`mapAsync()` can wait for earlier GPU work to finish, so these figures do not
isolate a four-byte transfer cost or prove that readback is the bottleneck.

A Chrome DevTools trace covering model load and the three traced requests
recorded 740 `WebGPU` spans on the GPU-process main thread, totalling about
1.406 s of span wall time and 0.923 s of thread CPU time. The trace was
collected without data loss but includes load and browser overhead; it does
not supply per-kernel GPU hardware timings or a causal breakdown for each
token. The trace itself was not saved as a durable artifact; pass-level GPU
timestamps were collected in the follow-up experiment below.

A later optional timestamp mode in the same complete-model page measured every
compute pass of a warm 41-input/21-output-token request on the Radeon 890M.
Chrome reported 7,129 pass durations summing to 356.965 ms while the request
took 2.574 s. The span from the first pass start to the last pass end was
2,535.816 ms. The 109 measured gaps that crossed a queue submission summed to
2,178.796 ms (median 16.707 ms); 7,019 gaps between passes within a submission
summed to 0.055 ms. The measured passes returned the same answer as the
unprofiled requests. These GPU timestamp values identify where time lies in
this specific run: most elapsed time between its first and last compute pass
was outside the measured compute passes. The gaps can include copies, browser
and driver scheduling, CPU-dependent work and waiting for the next token; they
are not a direct measurement of GPU idle time. Timestamp instrumentation itself
changes the workload, and the experiment does not establish the effect of any
proposed optimization on the full model.

The profiler then associated each inter-submission gap with completed
four-byte token maps. Two warm responses taking 2.276 and 2.404 s had 21
post-map gaps totalling 735.099 and 686.191 ms, versus 88 other gaps totalling
1,128.283 and 1,260.530 ms. Later, faster warm responses taking 1.074 and
0.829 s had post-map gaps of 130.625 and 92.390 ms, versus other gaps of
425.124 and 393.259 ms. In all four cases the same 7,129 compute passes,
132 submissions, 22 maps and output text were observed. Every one of the 109
measured inter-submission gaps followed at least one `queue.writeBuffer()` call.
Removing token readbacks alone therefore cannot explain or remove all observed
gap time; this is a scheduling hypothesis, not an optimization result.

A size counter found the same 7,458 `queue.writeBuffer()` calls per request
transferred only 81,832 bytes in total. Of those calls, 6,793 wrote at most
16 bytes, and the largest wrote 164 bytes. The synchronous JavaScript time in
the wrapped writes was tens of milliseconds, so their count and interleaving
make prepacked, GPU-resident per-step parameters a concrete experiment, not
proof that the writes themselves caused the entire GPU timestamp gap.

The faster browser samples show substantial run-to-run variance: in a later
three-request session the decode reports were 33.51, 35.16 and 28.92 tokens/s,
with the same prompt/output/token counts and GPU-call counts as earlier
roughly 10-token/s sessions. A contemporaneous loopback-only `llama-server`
rerun with the same prompt, temperature 0, seed 1, 64-token cap, Vulkan
`-ngl 99` and prompt caching disabled returned the exact same answer. One
request reported 114.15 decode tokens/s; the next seven reported 51.25-71.78
tokens/s and took 0.396-0.536 s wall. The native server was stopped afterward.
The different model containers/backend, 21 versus 22 output-token accounting,
and uncontrolled clock/thermal/background state still prevent a strict
native-parity or stable-speedup claim. The paired observations show a remaining
gap even in the fast browser state.

Capturing the generated WGSL associated the high-cost decode pipelines with
specific model operations. In one profiled request, the MLP gate/up projection
(`NT_matmul12_kernel`, 630 passes) used 119.219 ms GPU time, the MLP down
projection (`fused_NT_matmul13_add2_kernel`, 630 passes) 72.849 ms, the final
vocabulary projection (`NT_matmul14_kernel`, 22 passes) 58.391 ms, the QKV
projection (`NT_matmul10_kernel`, 630 passes) 43.187 ms and paged-KV attention
(`batch_decode_paged_kv_kernel`, 630 passes) 32.951 ms. The generated MLP
gate/up WGSL used a 64-thread workgroup and a shared-memory reduction with
multiple workgroup barriers. These timings identify operations, but their
absolute values vary with the same browser-speed fluctuations above.

An opt-in experiment replaced only that MLP gate/up reduction with a WGSL
`subgroupAdd()` reduction while retaining the model's other passes and the
same prompt. Chrome compiled it with `subgroups` and `subgroup_id`, and all
three requests returned the same answer text with no shader compilation
errors. In adjacent profiled runs, its 630 MLP passes totalled 60.687 ms,
versus 61.386 ms for the original kernel. Warm decode reports were 29.26 and
29.40 tokens/s for the variant, versus 33.90 and 32.94 for the subsequent
original run; an earlier unprofiled pair had overlapping, highly variable
ranges. This is not a demonstrated improvement or a numerical-logits parity
test. The variant remains experimental and is not a runtime default.

A second opt-in shader experiment replaced the 64-lane workgroup reduction
in the final F32 vocabulary projection (`NT_matmul14_kernel`) with
`subgroupAdd()` and one cross-subgroup sum, leaving its F16 weight reads and
F32 dot products intact. On Chrome 154 / Radeon 890M it compiled without
errors, and all fifteen variant/control requests returned the same 21-token
answer text. In an alternating original/variant/original/variant/original
sequence, the third request of each three-request phase was GPU timestamped.
The vocabulary kernel took 41.953, 21.846, 16.977, 53.953 and 40.650 ms,
respectively, over 22 calls. In the two adjacent variant-then-original
comparisons, the variant was slower for the targeted kernel (21.846 versus
16.977 ms; 53.953 versus 40.650 ms). Third-request wall times were 0.903,
0.591, 0.687, 1.192 and 0.830 s in that order, showing a large speed-state
shift and no repeatable end-to-end benefit. This variant is kept behind a
local test checkbox, not selected as the runtime default. Matching answer
text in those profiled requests was not a numerical-logit equivalence test.
A separate unprofiled variant-then-original pair captured the output IDs:
all three requests in each phase returned the same sequence of 21 IDs.
Excluding the first variant request with cold preparation, variant wall times
were 0.765 and 0.695 s; the original phase took 0.834, 0.762 and 1.004 s.
These few noisy wall samples do not establish a speed gain or parity for
other prompts.

Another opt-in experiment preserved WebGPU ordering while packing the 7,458
small host writes into mapped staging buffers before the next submission. It
replaced those queue writes with 110 staging submissions per request, but still
needed 7,458 GPU buffer-copy commands and increased command buffers from 132
to 242. Three requests returned the same 21-token answer. The packed variant's
two warm requests reported 37.90 and 33.84 decode tokens/s, versus 31.62 and
32.43 in a subsequent original-engine run. These overlapping, variable values
do not demonstrate an end-to-end gain. The experiment confirms that preserving
ordering by merely translating writes into copies is insufficient; a future
backend must use prepacked resident parameters directly and then remeasure.

The separate `tests/browser-webgpu-autoregressive.html` page tested a synthetic
64-step dependent recurrence with 32 compute passes per step and 1,024 u32
state values. A CPU-roundtrip variant mapped and rewrote the selected 4-byte
token each step (65 queue submissions including final output), while a
GPU-resident variant kept token and state on the GPU until the end (two queue
submissions including final output). Both matched a CPU reference token by
token and produced three distinct token values. Across two complete A/B runs
on this Radeon, CPU-roundtrip wall times were 1,299.1/1,627.8 ms and
1,416.4/1,560.0 ms; GPU-resident times were 94.3/104.7 ms and 80.2/89.9 ms.
The speedup of the paired averages was 14.71x and 17.50x. A third correctness
run returned 1,547.7/1,634.8 ms versus 72.4/118.7 ms (16.65x on paired
averages). This demonstrates a browser-only scheduling method for a small
dependent graph, not a full LLM, native parity, GPU exclusivity or mobile
performance. Its simple arithmetic, small state and fixed 64-step output do
not represent transformer memory traffic, EOS behavior or streaming latency.
The same page also accepts `?state=65536` (65,536 u32 values, 64 times the
original state). Two A/B repetitions again matched the CPU reference: paired
CPU-roundtrip means of 1,754.15 and 1,566.95 ms versus GPU-resident means of
71.8 and 69.7 ms (24.43x and 22.48x). This larger state still does not
approximate complete model weights, attention or KV memory traffic.

The separate `tests/browser-webgpu-tiny-transformer.html` page extended the
GPU-resident experiment to a complete but deliberately tiny one-layer
transformer graph: 32-wide embeddings and Q/K/V, causal attention with a KV
cache, a 64-wide MLP, 128-token vocabulary projection, argmax sampling and an
EOS flag. It used synthetic F32 weights and encoded 64 dependent decode steps
in Chrome on the same non-fallback Radeon 890M. Every measured route produced
the same 64-token sequence as an independent CPU implementation, including
27 distinct tokens and EOS at the final step. The CPU-roundtrip variant mapped
and rewrote each token; GPU-resident variants mapped output once per chunk of
1, 4, 16 or 64 steps without reuploading the token. Two complete ABBA-style
sessions gave CPU-roundtrip means of 243.0 and 246.6 ms. Paired GPU-resident
chunk-1 means were 230.9 and 245.15 ms (1.05x and 1.01x versus CPU), chunk-4
means 68.0 and 62.2 ms (3.57x and 3.96x), chunk-16 means 39.35 and 31.1 ms
(6.18x and 7.93x), and chunk-64 means 30.9 and 28.05 ms (7.86x and 8.79x).
These are full graph wall times, not isolated kernel timings. In the first
session, time to the first available chunk was 3.4-3.9 ms for CPU roundtrips,
7.6-8.9 ms for chunk 16, and 26-28 ms for chunk 64. Chunk size trades
streaming latency against throughput. A separate forced EOS at step 24 inside
a 16-step chunk matched the CPU sequence, left later outputs as sentinels, and
stopped issuing further chunks after that chunk. The graph still executes the
rest of the current chunk after EOS. This establishes the browser-only
algorithm for embedding, KV, sampling and bounded chunk output; its tiny
single-layer F32 weights and argmax sampler do not establish a complete
SmolLM2 implementation, native performance parity, sustained GPU occupancy or
mobile behavior.

An experimental `tests/browser-webllm-gpu-chunk.html` page applied bounded
GPU-resident decoding to the pinned complete SmolLM2-135M q0f16 MLC model,
without changing its weights or compiled WASM model. It replaces WebLLM
v0.2.85's per-token decode method only in this local test. The selected GPU
token feeds the next embedding directly, and the runtime's internal GPU buffer
copy packs token IDs into one GPU tensor per chunk before a single CPU map.
On the fixed 41-input/21-output-token prompt, chunk 8 issued 24 decode steps
(three beyond stop), reduced maps from 22 to four and queue submissions from
132 to 105. Three independent model-load sessions each ran a warm alternating
ABBA-style comparison with six original and six chunk-8 requests, excluding
the first two load/warmup requests. Median original versus chunk-8 decode
times were 0.8130 versus 0.5857 s, 0.6385 versus 0.4506 s, and 0.8594 versus
0.4939 s. Corresponding median full response times were 0.9390 versus 0.6925
s, 0.7415 versus 0.5710 s, and 0.9730 versus 0.6065 s. Every request returned
the same output text; the third session also captured and matched all 21
output token IDs in every request. The third session measured the first decoded
token becoming available at a median 146 ms from request start for the
original, versus 294 ms for chunk 8. Chunking therefore improved this short
nonstreaming request but delayed token-by-token streaming. The browser's own
`time_to_first_token_s` reports prefill's first token and does not show this
delay in the next decoded token.

For a longer 52-input/64-generated-token prompt capped by `max_tokens`, two
separate four-request sessions again matched all output token IDs and text
across original and chunk-8 modes. Chunk 8 mapped nine buffers versus 64
original maps and submitted 270 versus 384 command buffers, but decoded one
unneeded step beyond the length cap. Its decode times were 1.4600/1.5416 s
in the first session and 1.6204/2.3487 s in the second; the later original
reference in each session took 1.5757 and 2.3623 s. This longer workload did
not establish a consistent end-to-end gain, despite the fewer maps.

After KV rollback was added, a third four-request long-prompt session again
matched all 64 token IDs and text. Each route ended at KV length 115; chunk 8
removed one speculative KV step per request. The two chunk requests took
0.6023 and 0.6199 s of decode versus 1.1292 and 1.0268 s for the bracketing
original requests. Full response times were 0.657/0.676 s versus
1.372/1.093 s. The previously observed cross-session variance prevents
treating this one fast session as a stable sustained speedup.

A loopback-only native `llama-server` rerun of the long prompt used the F16
GGUF, Vulkan `-ngl 99`, temperature 0, seed 1, 64-token cap and disabled
prompt caching. Its load log assigned all 31 layers and KV to Vulkan0 on the
same Radeon. Six requests measured 67.01-92.56 predicted tokens/s with 52
prompt and 64 output tokens each. The native text differed from the browser
text despite the same input token count, and model containers, prompt
construction, backend and output-token accounting still differ. These values
show remaining distance but cannot prove or precisely quantify same-model
native parity. The native server was stopped and its listener verified absent.

The same page now rolls back the GPU KV state after speculative steps beyond
EOS or the length cap, using TVM's internal `vm.builtin.kv_state_popn` on each
active state. In a short-prompt sweep, chunk sizes 4, 8 and 16 all returned
the same 21 output token IDs and text as the original and ended at KV length
62. Chunk 16 removed 11 speculative KV steps. A separate two-turn test ran
the original and chunked first turns, then sent the same follow-up while
reusing each first turn's KV state. Both routes ended the first turn at KV
length 62 and the second at 86, with no reset during the follow-up. They
returned identical five-token follow-up IDs `[504, 6376, 314, 4461, 30]`
and text `The sky is blue.` for both chunk 8 and chunk 16. This verifies one
real-model continuation case; it does not establish general conversation
correctness. The experimental route falls back to the original decode near
the context limit or with sliding-window KV.

An adaptive chunk schedule `[1, 2, 4, 8]` was tested in two eight-request
sessions against fixed chunk 8 and the original. Every request in each
session returned the same 21 token IDs and text, with KV length 62. The
median time from request start to the second processed token, the first
decoded token, was 90.5 versus 161.9 ms for adaptive versus fixed chunk 8
in the first session, and 62.6 versus 125.8 ms in the second. Median full
response time was 0.298 versus 0.364 s in the first session and 0.304 versus
0.302 s in the second. This schedule improved early decoded-token visibility
relative to fixed chunk 8 in these two local sessions; total response time
remains variable. The page measures internal token processing, not actual
streamed arrival through the public API.

A separate six-request run used WebLLM's public `stream: true` completion API
with the original route, adaptive chunks and fixed chunk 8. All six requests
returned the same text and 21 output token IDs, and ended with KV length 62.
The original emitted 21 nonempty fragments per request; adaptive emitted six
and fixed chunk 8 emitted four. The first decoded fragment arrived at 102.4
and 65.0 ms for adaptive and at 98.6 and 123.5 ms for fixed chunk 8 in the
two measured passes. The original bracketing requests arrived at 282.5 and
71.6 ms, showing a large warm-state difference. This verifies progressive
streaming for the fixed request, including visible chunk aggregation; it does
not establish a stable latency ordering or general API compatibility.

Two further independent long-prompt sweeps compared fixed chunks 8, 16 and
32 against the original. Every request returned identical 64 token IDs and
text, with KV length 115; each chunked request rolled back one speculative
KV step. Across the four requests per size, median full response times were
0.702 s for chunk 8, 0.643 s for chunk 16 and 0.657 s for chunk 32, versus
1.322 s for the four bracketing original requests. Median time to the first
decoded token was 128, 191 and 367 ms for chunks 8, 16 and 32 respectively.
Chunk 16 reduced maps to five and submissions to 266, versus 64 maps and 384
submissions in the original. Chunk 32 reduced them further to three and 264,
but delayed output markedly and did not improve median total time here.
This is a measured local trade-off, not a universal chunk-size setting.

Two 24-request sustained sessions used 12 original and 12 chunk-16 requests
each, reversing the phase order in the second session. After discarding the
first two requests of each phase, median full response times were 1.324
versus 0.5875 s (original versus chunk 16) in the first order and 1.313
versus 0.696 s in reverse order. All 48 requests returned identical 64
token IDs and text with KV length 115. Median first decoded-token visibility
was 89.3 versus 172.4 ms and 80.1 versus 193.1 ms respectively. These
same-session comparisons support a throughput gain for this model, prompt,
device and browser, with a real streaming-latency cost. They do not establish
native parity, multi-model behavior or a performance guarantee across devices.

To reduce that cost, a new increasing schedule `[1, 2, 4, 8, 16]` was tested
in two independent eight-request long-prompt sessions. Every run matched all
64 output token IDs and text and ended at KV length 115. Each adaptive run
issued exactly 63 decode steps, made eight maps and 265 submissions, and
needed no speculative KV rollback; fixed chunk 16 made five maps and 266
submissions with one rollback. Adaptive median full response time was 0.800
versus 0.697 s for fixed 16 in one session and 0.657 versus 0.780 s in the
other. The adaptive first decoded-token times were 67.5/69.7/113.6 ms and
72.2/74.1/105.9 ms; fixed 16 took 187.9/249.0/211.1 ms and
254.2/174.0/217.4 ms. The schedule traded a few additional maps for much
earlier output while keeping a comparable, variable total time.

A six-request long-prompt `stream: true` run verified actual API delivery.
Adaptive produced eight nonempty fragments per request, fixed 16 produced
five, and the original produced 64; all returned the same 64 token IDs and
text with KV length 115. The first decoded fragment arrived at 79.3 and
58.6 ms for adaptive, versus 168.4 and 226.5 ms for fixed 16; bracketing
original requests took 199.2 and 114.9 ms. This demonstrates useful
progressive output for this fixed request, not general streaming support.

The pinned WebLLM v0.2.85 runtime already calls `requestAdapter()` with
`powerPreference: "high-performance"` and requests `shader-f16` and
`subgroups` when its adapter exposes them. Its device also requests larger
buffer, workgroup and binding limits within the adapter's reported limits.
The WebGPU `powerPreference` is a selection/power hint, and requested limits
relax validation rather than reserve the physical GPU. Adding a website
permission prompt cannot turn this logical device into exclusive driver
access. Sources: pinned WebLLM v0.2.85 `detectGPUDevice()` in its bundled
runtime and the [WebGPU adapter and limits specification](https://gpuweb.github.io/gpuweb/).

A separate zero-dependency `tests/browser-webnn-capabilities.html` page
probed another browser-native route on the same Chrome 154 / Windows machine.
The secure loopback page found `navigator.gpu` and the non-fallback Radeon
890M adapter, but `navigator.ml` and `MLGraphBuilder` were absent. Therefore
the WebNN F32 matrix branch could not run in this ordinary browser, and no
WebNN GPU or model-performance result is claimed. Requesting WebGPU adapters
with `powerPreference: "high-performance"` and `"low-power"` returned the
same reported AMD device `0x150e` in this local test. This agrees with
[Chrome's Windows WebGPU guidance](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips),
which says that preference does not select a different adapter there. The
[WebNN installation guide](https://webnn.io/en/learn/get-started/installation)
still describes an experimental browser flag and restart for early use;
that is not a zero-setup client route. The [WebNN specification](https://www.w3.org/TR/webnn/)
defines accelerated context hints and WebNN/WebGPU tensor exchange, but a
specification cannot supply an API missing from this Chrome configuration.
WebNN remains an optional future backend candidate if it becomes available
without a client flag and proves model-level correctness and speed.

During 24 continuous model requests, Windows' `GPU Engine` counter for
Chrome's GPU process reported roughly 47-52 on its active 3D engine in the
sampled steady portion. A separate 30.036-second 1024-square F32 matrix
run on the same Chrome/Radeon reported 233.37 sustained GFLOPS and passed
three CPU spot checks; its active 3D-engine counter reported roughly 80-112.
Values above 100 make this counter unsuitable as a literal fraction of
physical compute capacity. It establishes that this browser can submit
sustained heavy GPU work and that these two workloads had different scheduler
accounting. It does not prove 100% physical utilization or identify the
remaining model bottleneck by itself. Windows reports engine scheduling,
while 3D and compute engines may share physical cores, as described by
[Microsoft's GPU scheduling explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).

Finally, a browser-only Web Worker mode ran the same pinned model and long
prompt through WebLLM's worker engine. Six main-thread and six worker
requests returned the same text and 63 completion tokens. Excluding the
first two per mode, median full response times were 1.2062 and 1.1703 s.
This single main-then-worker session does not show a useful inference-speed
gain; a Worker remains a UI-thread isolation option, not a direct-driver path.

The browser-only page was then parameterized for two larger complete models
using the same pinned WebLLM 0.2.85 and compiled-library revision
`025bcaf3780fa8254f5e5efd3bfea0a5397248f4`. SmolLM2-360M-Instruct
q0f16 used model revision `df74591537f91ed5c9b7e95b9eea1bb982ab67ed`;
SmolLM2-1.7B-Instruct q4f16_1 used
`84f57f8580a9d8d623266b600ad4273bb9fd84c1`. Both model libraries
returned HTTP 200 at the pinned revision before testing. Chrome 154 on the
same non-fallback Radeon 890M loaded each and completed the 52-input-token,
64-output-token long prompt locally. Initial uncached load was 25.7 s for
360M and 51.3 s for 1.7B; cached reloads were 3.26 and 4.31 s. These load
times include network/cache and shader preparation and are not GPU-only.

For 360M, an eight-request original/adaptive/fixed-16 sequence returned the
same 64 output token IDs, text and KV length 115 in every request. Its later
original reference took 2.021 s wall; adaptive requests took 1.397, 1.416
and 1.164 s; fixed 16 took 1.259, 1.593 and 1.387 s. The first original
request took 5.130 s and included cold preparation, so it is excluded from
the warm comparison. A separate instrumented four-request session also
matched all token IDs and KV lengths. It counted 897,992,692 bytes of live
WebGPU buffers after model load and a 1,072,364,976-byte peak during
inference; all tracked buffers were destroyed on unload. This counts API
buffer sizes, not resident physical VRAM or driver allocations.

For 1.7B, the instrumented four-request original/chunk-8 sequence matched
all 64 output token IDs, text and KV length 115. Initial cold request took
10.356 s; two chunk-8 requests took 3.602 and 3.987 s, and the later
original reference took 4.004 s. Live WebGPU buffer sizes after load summed
to 1,788,842,100 bytes and peaked at 2,032,619,792 bytes; tracked live
bytes returned to zero on unload. A subsequent eight-request test without
buffer instrumentation also matched all tokens and KV lengths. Its later
original reference took 3.711 s; adaptive requests took 4.000, 4.236 and
3.808 s, and fixed-16 requests took 3.962, 3.993 and 3.384 s. The first
decoded token took roughly 1.0-1.15 s in adaptive requests and 1.59-1.98 s
in fixed-16 requests. These small samples show local full-model feasibility
at 1.7B and correctness of GPU chunking for the fixed prompt, but no clear
end-to-end speed gain at that size. The 135M tuning cannot be assumed to
generalize as model compute and memory traffic grow.

A separate 1.7B two-turn test used chunk 16 for the first response and the
original engine for the follow-up. The original and chunked first responses
had the same 33 token IDs and ended at KV length 74. Chunking rolled back 15
speculative KV steps. Neither route reset the conversation during the
follow-up; both consumed 19 new prompt tokens, returned the same seven
follow-up token IDs and text `The sky's color is blue.`, and ended at KV
length 100. This extends one controlled KV-continuation proof to a larger,
quantized model, but still does not validate arbitrary prompts or sampling.

For a closer native/browser comparison, the browser test intercepted
WebLLM's `getInputData()` and recorded the exact input token IDs. The short
request had 41 input IDs, matching WebLLM's usage count. A loopback-only
`llama-server` build `571d0d540` (10068), using the locally hashed F16 GGUF
above with Vulkan `-ngl 99`, `--no-cache-prompt`, temperature zero and no
penalties, accepted those same 41 IDs at `/completion`. Every native request
produced the browser's exact first 21 output IDs and then EOS token 2. Thus
input tokenization and this short deterministic output were matched. The
native endpoint's token-ID prompt support is
documented in the [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
The [MLC model card](https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC/blob/06fc409f2212adff29d93cbc742c1d6957302df5/README.md)
names the same Hugging Face base model but does not identify its source weight
commit. The native runtime's model record pins source revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The container weights were
therefore checked directly, as described below.

The pinned MLC revision's
[array index](https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC/blob/06fc409f2212adff29d93cbc742c1d6957302df5/ndarray-cache.json)
describes eight F16 parameter shards. All eight locally downloaded shards
passed their recorded MD5 hashes. `tests/compare-browser-native-weights.py`
compared all 182 MLC tensor records with the local F16 GGUF, covering every
one of the shards' 269,030,016 bytes. The 30 fused QKV matrices were split
into Q, K and V; Q/K received the [llama.cpp GGUF conversion's RoPE weight
permutation](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py),
and the GGUF F32 norm vectors were cast to F16. All 242 resulting component
comparisons matched byte for byte. This proves equality of the stored F16
model weights after those known container transformations; it does not prove
that WebGPU and Vulkan execute numerically identical kernels or sample the
same token on a close logit comparison. The MLC card's missing source commit
is still a metadata provenance gap, although it no longer prevents a direct
weight-equality claim for these two artifacts.

Seven native short requests after one cold request had median client wall
time 163.84 ms; a second six-request native series after the browser run had
median 205.76 ms. Each native response counted 41 input and 22 generated
tokens including EOS. In the intervening eight-request browser series, all
21 visible token IDs and text matched across modes. Three fixed-chunk-8
requests had median wall time 0.369 s; three increasing `[1,2,4,8]` requests
had median 0.436 s. The later original WebLLM request took 0.512 s, while
its first request took 1.060 s and included cold work. On this specific
matched-output request, the browser remains slower than the native reference.
The comparison uses the same stored F16 weights, but different inference
backends; the native path also generated an additional EOS token.

The long browser request recorded 52 exact input IDs and 64 output IDs.
Submitting the same 52 input IDs to the native endpoint also produced 64
tokens, but output diverged at the seventh generated ID: browser 260 versus
native 13739. Equal stored weights and input tokens therefore do not establish
numerical equivalence for longer generations. The gap could arise from
different kernel arithmetic, precision, RoPE implementation or decoding
semantics. The loopback native server was stopped, and port 8766 was verified
to have no listener.

A follow-up `suite=logits&longPrompt=1` browser run captured the F32 logits
immediately before WebLLM's softmax for the first ten generated tokens. At
zero-based step 6, after the common 52 input and six generated IDs, WebLLM
ranked token 260 (`20.82997131`) just above token 13739 (`20.77263832`):
a logit margin of `0.05733299`. A second Chrome run reproduced the same
values. Supplying that exact 58-ID prefix to `llama-server` with one greedy
token, `n_probs=10`, no penalties and prompt caching disabled ranked token
13739 above 260. Vulkan `-ngl 99` returned log probabilities
`-0.85600239` and `-0.96478993`, respectively, for a relative logit margin
of `0.10878754` in the opposite direction. The same GGUF on CPU with
`-ngl 0` also chose 13739; log probabilities were `-0.88394409` and
`-0.94635260`, a margin of `0.06240851`. Differences of these log
probabilities equal differences of pre-softmax logits for the two candidates.
The top three candidate IDs had the same order except for the first two.
This isolates the token disagreement to model logits at a near-tie, rather
than a different text tokenizer or a random sampling outcome. It does not
yet locate which numerical operation or runtime configuration causes the
logit difference. Both loopback native server modes were stopped afterward;
port 8766 had no listener.

The chunk experiment uses private WebLLM/TVM runtime fields and methods and
only supports fixed temperature-zero, no-penalty settings. It still computes
extra steps after EOS/length before rolling their KV state back. Grammar,
logprobs, other model architectures, general multi-turn paths and mobile
browsers have not been validated. It is a proof of a complete-model browser
algorithm, not a production backend or a claim of full-GPU or native-equivalent
performance.

The browser capability page deliberately destroyed a WebGPU device, observed
`device.lost` with reason `destroyed`, requested a fresh device with the
adapter's larger limits, and ran a recovery compute shader that returned 42.
This passed for both core and compatibility adapter requests in Chrome 154 on
this computer. It does not simulate an involuntary driver reset or tab kill.
The same adapter exposed `maxBufferSize=2147483648` and
`maxStorageBufferBindingSize=2147483644`, while a default device exposed
268435456 and 134217728 respectively. Requesting the adapter's larger limits
created a core WebGPU device that allocated a 536870912-byte storage buffer,
ran a shader writing its first and last 32-bit words, and read back the exact
values `[123, 456]`. This proves that a web page can request and use a
512 MiB buffer beyond the default device validation limits on this machine.
It does not reserve 2 GiB of physical memory, test a 2 GiB allocation, grant
exclusive scheduling, or bypass the browser's driver interface.

### Full GGUF via llama.cpp WebGPU in Chrome (local probe, 2026-09-26)

`tests/browser-wllama-smollm2.html` and `tests/serve-browser-wllama.mjs`
exercise the [wllama 3.6.1](https://github.com/ngxson/wllama) prebuilt
llama.cpp/WebGPU runtime without a native client process. The HTTP server
binds only `127.0.0.1:8767` and serves the page, runtime assets and local
GGUF bytes; it performs no inference. The pinned package was fetched from the
npm registry with `npm pack @wllama/wllama@3.6.1 --pack-destination
$env:TEMP\mycellios-wllama-probe` and extracted there. The test requires the
existing `runtime/models/SmolLM2-135M-Instruct-F16.gguf` artifact (270885888
bytes, SHA-256
`EE7608CAB6E375AE1C2446CB0E4C0193E2756FAB3DF2FD1327A7A04E571DF56F`).
Run `node tests/serve-browser-wllama.mjs`, open
`http://127.0.0.1:8767/browser-wllama-smollm2.html`, and click the two
buttons. No production endpoint or browser flag was used.

Chrome 154 reported AMD Radeon 890M, architecture `rdna-3`, WebGPU available
and `crossOriginIsolated=true`. Two load attempts completed in 7044 and
5599 ms, including reading 270885888 bytes over loopback and runtime
initialization. The llama.cpp log recorded `offloaded 31/31 layers to GPU`,
`WebGPU model buffer size = 256.63 MiB`, `WebGPU KV buffer size = 11.25 MiB`,
and a 54.00 MiB CPU model buffer. This proves full **layer offload** for this
135M model, not exclusive GPU access or that every operation and byte resides
on the GPU.

With prompt `The capital of France is`, greedy sampling and a 32-token cap,
the browser produced exactly the same visible text as the local native
llama.cpp Vulkan server using this GGUF. The first browser completion took
1823 ms (first output at 345 ms). After closing the native server, three
consecutive browser completions took 1773, 1329 and 1353 ms, median 1353 ms;
first output was at 245, 100 and 81 ms. Three native requests with the same
prompt/model/settings took 3102, 612 and 573 ms client wall time; the first
included cold prompt work, and the warm median was 592.5 ms. The native
server recorded 32 generated tokens and 469.62, 535.58 and 530.71 ms decode
time. Two intervening browser runs while the native process also held the
GPU took 7610 and 8099 ms and are excluded from the uncontended browser
series. Thus this local proof reaches complete-model browser inference, but
its observed warm end-to-end latency remains about 2.3 times native on this
short request. The WebGPU and Vulkan backends differ, and this small-model
single-prompt test is not evidence for larger models, mobile, sustained
throughput or equal native performance.

The same local page was then extended with `?model=qwen` to serve the existing
`Qwen3-0.6B-Q8_0.gguf` (639446688 bytes, SHA-256
`9465E63A22ADD5354D9BB4B99E90117043C7124007664907259BD16D043BB031`).
Chrome loaded it in 71067 ms including loopback transfer. The runtime logged
`offloaded 29/29 layers to GPU`, 604.15 MiB of WebGPU model buffers, 56.00
MiB of WebGPU KV buffers and a 157.65 MiB CPU model buffer. Five browser
32-token requests with the same prompt and greedy settings took 3905, 3512,
5055, 8702 and 8434 ms. The last two were measured after the native server
was stopped, but show a large unresolved slowdown; there is no stable
browser speed estimate for this model yet. Three native Vulkan requests took
3843, 2026 and 1949 ms client wall time, with the first including cold work.
The first browser output differed after ` Paris.`, while four later browser
outputs matched all 32 visible native tokens. This larger-model trial proves
complete-layer offload and generation in the browser, but also reveals a
numerical/reuse consistency question and variable latency requiring a
controlled, alternated rerun before performance or parity claims.

The cold-output difference reproduced after a fresh Qwen WebGPU load: the
first 32-token request returned ` Paris. The capital of France is also...`
in 3926 ms, while the second returned ` Paris. The capital of Italy is
Rome...` in 3461 ms. This load took 16674 ms, split into 6120 ms for local
fetch/Blob creation and 10554 ms for WASM/WebGPU initialization. Two
browser CPU/WASM requests with the same GGUF, prompt and settings both
returned the first (`France is also`) continuation in 7925 and 7193 ms;
that CPU load took 39561 ms, including 32533 ms local fetch. A separate
native llama.cpp CPU server (`-ngl 0`) also returned that continuation on
two 32-token requests in 1776 and 1874 ms. Native Vulkan returned the
`Italy is Rome` continuation on three requests. The two continuations
therefore correlate with backend arithmetic, although the WebGPU transition
between first and later requests remains unexplained. This probe cannot
claim deterministic cross-backend token parity for Qwen. In this short
small-model trial, native CPU was slightly faster than native Vulkan; a
"GPU versus browser" comparison alone would miss the best local baseline.

A SmolLM2 WebGPU `n_threads` probe compared one and four WASM threads in
successive sessions. The first one-thread block returned the same 32-token
text in 1927, 1857 and 2120 ms. The next four-thread block took 6833, 3964
and 7404 ms. Repeating one thread took 8269 and 10170 ms, so the apparent
one-thread advantage did not survive phase reversal. A further one-thread
request with the model tab selected, visible and focused took 11961 ms.
Native Vulkan warm requests in the later period took 1000, 915 and 812 ms,
slower than an earlier 573-612 ms range but far less variable than Chrome.
This trial identifies large browser-specific drift, not a validated thread
setting. Changing `n_threads` is not justified as a default optimization
until GPU scheduling, browser process activity and same-session controls
explain the drift.
One visible, focused one-thread request was also sampled with Windows
`GPU Engine` counters: it completed in 3204 ms; the Chrome process's 3D
engine counter rose from roughly 6-9% idle samples to 14.86%, 18.96% and
33.58% in the three samples around execution, then fell back. The Codex
desktop process had separate concurrent 3D activity. This single coarse
process-engine trace cannot determine shader occupancy or the fraction of
physical GPU capacity used by inference, but it does not support a claim
that this workload continuously saturated the GPU.

A Qwen follow-up isolated the first-versus-reused output change. With
`cache_prompt=false`, two browser WebGPU requests both produced the
`France is also` continuation, taking 7504 and 6298 ms in that session.
With prompt caching enabled and non-streaming top-five probabilities, the
first and second 16-token browser requests shared their first five output
IDs (`12095, 13, 576, 6722, 315`). At output index five, the first chose
`France` (ID 9625, log probability -1.647964) over `Italy` (ID 15344,
-1.763351), a relative margin of 0.115387. The reused request chose
`Italy` (-1.749390) over `France` (-1.787651), a margin of 0.038261 in the
opposite direction. The llama.cpp server log reported full-prompt cache
reuse and re-evaluation of one token (`n_past` lowered from five to four).

Native Vulkan with `cache_prompt=true` also changed continuation: its
first 16-token request chose `Italy` (-1.770673) narrowly over `France`
(-1.773463), margin 0.002790; the second chose `France` (-1.757805)
over `Italy` (-1.778893), margin 0.021088. Browser WebGPU and native Vulkan
therefore both show a near-tied sixth-token ranking that flips after prompt
reuse, although in opposite directions. This is stronger evidence for a
numerical/cache-state sensitivity than for an isolated first-use WebGPU
failure. It does not locate the exact operation or prove that their input
token IDs and KV tensors match. The native loopback server was stopped after
this comparison.

A local diagnostic patch to the temporary `@wllama/wllama@3.6.1` worker
(`tests/patch-wllama-worker-probe.mjs`) counted WebGPU calls around each WASM
action. It does not change the model or production runtime. With the native
server stopped, three visible, focused SmolLM2 F16 WebGPU 32-token requests
returned the same text in 2475, 1786 and 1284 ms, with first output at 369,
376 and 138 ms. Each request made 32 GPU-active `get_result` actions; each
such action normally made 10 `GPUQueue.submit()` calls, 580 `writeBuffer()`
calls, 15 `onSubmittedWorkDone()` calls and one `mapAsync()` call. Per-request
totals were 18560 writes and 480-481 queue waits. The first and third
requests had 320 submissions and 32 maps. The second had 380 submissions
and 92 maps because its first `get_result` action contained an additional
60 submissions and 60 maps, without additional writes. Why that first-step
work appeared is not yet established. Sum of `get_result` action durations
fell from about 2414 to 1674 to 1137 ms across the three requests, broadly
tracking wall time. The instrumentation and logging can affect these timings.
Promise wait durations may overlap and `mapAsync()` can include prior GPU
work; these counters do not distinguish physical shader execution from
browser scheduling or prove unused GPU capacity.

Two Chrome GPU-process traces during 16-token Qwen WebGPU requests gave
end-to-end times of 8070 and 3857 ms. The trace contained 649 `WebGPU`
service events over 8009 ms in the slower request, and 537 over 3807 ms in
the faster one. Their service-event durations summed to about 509 and 369
ms respectively; those are GPU-process CPU events, not hardware shader
timestamps or a valid estimate of GPU utilization. The GPU process can
also serve other tabs. Both traces reported no data loss, but the trace
streams were not saved as durable artifacts. Together with the worker
counts, these observations motivate a controlled native/browser A/B with
per-kernel GPU timestamps and a GPU-resident token loop. They do not
establish native-equivalent performance or full physical GPU access.

The same temporary worker probe then recorded `writeBuffer()` payload sizes
and destinations. A GPU-active SmolLM2 decode step made 580 logical writes
totalling 55,752 bytes. About nine submission groups each wrote 60-66 small
regions of a 18,944-byte uniform/COPY_DST buffer, usually spanning about
15-17 KiB while carrying about 4.3-4.8 KiB of payload. A separate 3,377,920
byte storage buffer received six writes. Thus the hot path has many tiny
host API calls, but simply reducing their number may increase transfer bytes.

An opt-in `--coalesce` mode in `tests/patch-wllama-worker-probe.mjs` kept a
CPU shadow of only that small uniform buffer and flushed one combined range
per submission, preserving submission order. It reduced the first decode
step from 580 logical writes to 15 physical calls (including the six
unmodified storage-buffer writes); across a 32-token request, physical calls
fell from 18,560 to 480. The first step transferred 159,460 physical bytes
instead of 55,752 logical bytes, and a full request transferred about
4.74-4.75 MB instead of 1.42-1.43 MB. All twelve 32-token requests in a
four-phase original/coalesced/original/coalesced local comparison returned
the same visible text. Wall times in milliseconds were original
`3046/1699/1535`, coalesced `1716/1871/1782`, original
`1902/1634/1675`, coalesced `2268/2245/1884`; each phase had a fresh model
load, and the first request of a phase can include warm-up effects. The two
later original requests in each original phase were faster than the two
later coalesced requests in the adjacent phase. This experiment shows a
large reduction in API call count without a repeatable end-to-end speed gain;
the wrapper is not a default optimization. It was validated only by visible
text on one model, prompt, browser and GPU. A maintained backend could avoid
the extra bytes by constructing GPU-resident, packed per-step parameters
directly and must be benchmarked separately.

A follow-up isolated this small-write path further on Chrome 154 and the
Radeon 890M, using the same SmolLM2 135M F16 GGUF and fixed 32-token prompt.
Gap-limited range merging had no useful middle setting: gaps up to 64 bytes
kept all 580 first-step writes; 128 bytes left 550 calls; 256 bytes or more
collapsed to 15 calls but widened the first-step transfer from 55,752 to
159,460 bytes. A separate local WebGPU microprobe
(`tests/browser-webgpu-uniform-scatter.html`) correctly read 64 small
uniform regions after either direct writes, one compute-scatter pass per
batch, or one wide write. It exercised nine batches for 32 iterations;
all six recorded runs passed the CPU check. Its wall times varied strongly
between repeated variants, so it is a capability/correctness probe rather
than model-speed evidence.

An opt-in `--scatter` worker experiment packed the uniform updates into one
host write and one GPU scatter pass per batch. It preserved the visible
32-token answer with all 31 model layers on WebGPU, reducing physical host
writes from 18,560 to 480 per request, but adding 288 compute passes and
roughly doubling the transferred update bytes. A reversed local comparison
gave scatter phases `1879/3228/1637` and `1700/2129/2117` ms and an
intervening original phase `2435/1920/1770` ms. There was no stable speed
gain; this is not a default optimization.

Comparing the CPU shadow of the uniform buffer at the same submission stage
of successive decoded tokens showed that only 3,581 of 5,285,376 compared
bytes changed in the first 32-token request. Across 279 stage comparisons,
262 were byte-for-byte equal; after the third decoded token, all nine
observed stage images remained equal for this fixed prompt. Immediate
successive writes themselves were rarely duplicates (427 of 18,368), because
the same buffer is reused for different stages within each token. This
motivated two GPU-resident stage-cache experiments, not a general claim
about other prompts, models or context lengths.

`--cache-stages` saved/restored a 18,944-byte stage image with GPU-to-GPU
copies and skipped the CPU writes on a cache hit. It preserved visible text,
with 262 hits/26 misses on the first request and 288/0 on later requests,
but still copied 5,455,872 bytes per request on the GPU. Request times were
2926, 2683 and 2090 ms, with no established gain. A better opt-in
`--direct-stages` experiment instead created one uniform buffer per stage,
redirected each dynamically created bind group to it, and only uploaded a
full stage image when that image changed. The second 32-token request made
288 stage hits and no misses; physical host writes fell to 192, all to the
other storage buffer (90,880 bytes), with no per-token uniform copies. The
first request made 262 hits/26 misses and 218 physical writes. All three
requests returned the same visible text, taking `2458/1778/1671` ms.
An adjacent fresh-load original phase returned the same text in
`1903/1970/1804` ms. These short, noisy phases do not prove a consistent
end-to-end speedup, native performance parity or exact output-token parity.
They do prove that the stage buffer can remain GPU-resident while the whole
model runs in an ordinary browser.

The worker created 18,368 uniform bind groups per 32-token request, one per
uniform update. A further `--direct-bind-cache` attempt keyed groups by the
actual layout/resource identities and offsets; it got zero safe cache hits
in two 32-token requests. Reusing groups without respecting layout and all
other resource bindings could change the computation. A maintained backend
would need to stabilize its graph resources/layouts before prebuilding these
groups. Even after eliminating uniform host writes, this path still performs
the same WebGPU submissions, map/wait cycles and bind-group construction;
direct access to the GPU driver or exclusive use of physical GPU capacity
has not been shown.

### Real model part through the browser worker (local probe, 2026-09-27)

`tests/serve-browser-split-probe.ts` starts an in-memory, loopback-only
coordinator using the existing `/browser/` worker UI. The companion
`tests/run-browser-split-smollm2.py` reads the actual layer-0 gate, up and down
FFN tensors from `SmolLM2-135M-Instruct-F16.gguf` (file SHA-256
`ee7608cab6e375ae1c2446cb0e4c0193e2756fab3df2fd1327a7a04e571df56f`).
It publishes their 10,616,832 float32 bytes through `BrowserExpertOwner`,
assigns the projection to the existing `ResidentExpertMesh` as a
`remote-resident` SwiGLU expert, and routes one 576-wide activation through
the coordinator. The coordinator requires two distinct, visible, verified
browser replicas and compares their output tensors before returning the
result to the native mesh. The activation is deterministic and synthetic; it
was not captured from an end-to-end prompt.

On one Windows PC with AMD Radeon 890M, Chrome 154 and the Codex in-app
browser each registered a separate WebGPU worker. Both reported WebGPU as the
backend after executing the real expert and each recorded four verified
tasks (one enrollment task and three expert executions). All three routed
results agreed with the PyTorch calculation under `rtol=2e-4, atol=2e-4`;
maximum absolute error was `0.0002899169921875`. From a fresh local
coordinator, the first complete route including both weight downloads,
canaries and execution took `650.194 ms`; two subsequent routes took
`91.122` and `109.192 ms`. A second three-route warm run took `101.386`,
`87.483` and `88.850 ms`. A separate fresh-coordinator trial routed 16
activation rows per request and passed the same numerical checks; its three
route times were `1065.475`, `101.787` and `131.459 ms`, with maximum absolute
error `0.00048828125`. These are local end-to-end route timings for a single
MLP operation, not GPU kernel times or a native GPU performance comparison.
The result receipts are `runtime/browser-split-probe-result.json` and
`runtime/browser-split-probe-16rows.json` (generated local state, not source).

To reproduce locally: run `npm run mobile:build`, start
`npx --no-install tsx tests/serve-browser-split-probe.ts`, open
`http://127.0.0.1:8770/browser/` in two independent visible browser contexts,
and press **Start contributing** in each. Once `/mobile/v1/workers` reports
two online WebGPU workers, resolve the repository Python environment with
`scripts/resolve-distribution-python.ps1`, set `PYTHONPATH=python`, and run
`tests/run-browser-split-smollm2.py --require-webgpu`. Pass `--rows 16` and a
distinct `--output` path to reproduce the 16-row variant. The server binds
only to `127.0.0.1` and uses transient database and artifact storage.

This proves that a real model part can run through the existing browser
contribution and model-division route. It does not execute the full model,
attention, KV cache, sampling or generation through that route. Both workers
shared one physical GPU, so this is not a two-computer distributed test.
That experiment did not test a physical mobile browser or native-equivalent speed.

### Same-device browser versus native performance (local, 2026-09-27)

On PC-DANI (AMD Radeon 890M, Windows driver `32.0.31041.1004`), a follow-up
compared the local GGUF through Chrome 154 / wllama 3.6.1 WebGPU with the same
GGUF through llama.cpp b10068 Vulkan. Browser and native generation did not
run at the same time. Both used four CPU threads, 512 context positions, a
128-token batch limit, greedy sampling, prompt caching disabled and a cap of
32 generated tokens. Browser phases were visible and focused. The first
request in each phase was omitted from warm medians; its value remains in the
raw local record. Native measurements include loopback HTTP client time;
browser measurements are from `performance.now()` around completion. Browser
model delivery was over loopback, so this does not include customer download
time. The browser log reported all 31 SmolLM2 or 29 Qwen layers offloaded to
WebGPU. Native requests reported the stated input and 32 generated tokens;
the browser streaming API did not return a usable token-count field.

| Model and prompt | Browser warm median | Native warm median | Observed latency ratio | Output check |
| --- | --- | --- | --- | --- |
| SmolLM2 135M F16, 5 input tokens | `1720 / 1424 / 1508 ms` across three browser phases | `787 / 410 / 347 ms` across adjacent native phases | `2.2 / 3.5 / 4.3` times native | All 21 requests returned the same visible text as native. |
| SmolLM2 135M F16, 145 input tokens | `1769 ms` | `456 / 556 ms` before/after browser | `3.9 / 3.2` times native | All requests returned the same visible text. |
| Qwen3 0.6B Q8_0, 5 input tokens | `2742 ms` | `1249 / 1172 ms` before/after browser | About `2.2 / 2.3` times native | Browser continued with “France is also”; native continued with “Italy is Rome”. Diagnostic timing only. |

Each short SmolLM2 phase had seven requests, leaving six warm samples. The
long SmolLM2 and Qwen browser phases had five requests, leaving four warm
samples. The short SmolLM2 browser's median first visible output in its three
warm phases was `213 / 190 / 202 ms`; the long prompt's was `444 ms`.
These first-output values are not directly comparable with native prompt
processing times because the native client was not streamed. For the matched
short SmolLM2 request, the measured browser latency implies about 54–77% less
completed-request throughput than the adjacent native phases. The range is
wide because native speed changed substantially during this session; there
is no single stable slowdown percentage. Background desktop activity and GPU
power state were not held constant. Equal visible text is weaker than
an exact token-ID and logit comparison.

The existing Mycellios browser expert route was measured separately after
publishing the actual SmolLM2 block-0 MLP weights. Two verified WebGPU browser
replicas on this same PC agreed with the PyTorch calculation for every run.
Ten routed requests plus five diagnostic executions were made at each batch
size. Excluding each first routed request, median mesh route time was
`59.7 ms` for one activation row, `115.9 ms` for 16 and `247.6 ms` for 64;
a repeated one-row phase gave `49.4 ms`. Direct coordinator HTTP execution
medians were `27.1 / 95.1 / 219.9 ms` for 1/16/64 rows. The primary browser
worker reported median local execution durations of `14.7 / 48.2 / 108.9 ms`;
these include its GPU setup, uploads, dispatch and readback, not shader time
alone, and do not describe the slower replica. Current code creates shader
modules/pipelines and uploads all 10.6 MB of expert weights on every task, so
this route has clear unoptimized work. The split probe does not generate
tokens; its times cannot be multiplied into an observed full-model slowdown.

The local data are in `runtime/browser-performance-2026-09-27.json`, its
referenced native phase files and the `runtime/browser-split-benchmark-*.json`
receipts. All are generated, ignored runtime state; the reproducible probe
pages/scripts remain under `tests/`. No production endpoint or physical mobile
device was used. These results show a substantial measured gap for the
current browser engines and split route on this GPU, while leaving optimized
GPU-resident browser stages and physical two-computer performance unproven.

An opt-in browser MoE bridge is now implemented for one SiLU expert in a
RAM-backed Qwen3/GLM4 MoE stage. The operator supplies an exact model identity,
the coordinator origin and a measured placement profile; the native RAM expert
remains available when browser residency is lost. The existing `/browser/`
button admits browser workers. A local regression test generated a complete
tiny Qwen3-MoE model, routed its selected expert through two independent,
visible WebGPU browser sessions on this same computer, and produced token IDs
`23, 29, 14`, identical to the model reference. Each browser verified three
real expert activations. The local, ignored receipt is
`runtime/browser-moe-generation-receipt.json`; the reproducible driver is
`tests/run-browser-moe-generation.py`. This is a complete-model software path
with a browser-owned part, not a two-computer result, mobile-device proof,
general dense-model browser stage, or measured production speedup. A separate
SmolLM2 expert rerun after caching browser GPU weights reported 5–10 ms for
its direct one-row browser execution samples, but end-to-end routed times did
not improve consistently across the two browser contexts; no throughput claim
is made from that rerun.

The regular Torch safetensors Qwen3 stage also has an opt-in browser bridge
for one dense SiLU MLP. A local tiny Qwen3 model completed three token steps
through two independent WebGPU browser sessions and matched the reference IDs
`23, 4, 26`; each browser verified three MLP activations. After one browser
was paused, the same model and prompt used three native MLP forwards and
returned the same IDs. The local receipts are
`runtime/browser-dense-generation-receipt.json` and
`runtime/browser-dense-fallback-receipt.json`, produced by
`tests/run-browser-dense-generation.py`. The browser route took `374 ms`
versus `123 ms` for the native fallback in this tiny same-host run. These
latencies are diagnostics for different execution states, not a controlled
native versus browser throughput benchmark. No production model, GGUF stage,
mobile device, physical second computer or public deployment was exercised.
The cached real Qwen3 0.6B safetensors checkpoint was also used for a
layer-0 activation comparison. Three browser WebGPU forwards through two
same-host sessions matched the native Torch stage within maximum absolute
error `2.87e-6` for shape `[1, 4, 1024]`; each browser verified three real
model activations. The second run, after the expert weights were already
resident in the browser, measured `140 / 131 / 129 ms` for the browser stage
and `10 / 12 / 8 ms` for the native stage in adjacent executions. This is one
layer, not a complete Qwen3 0.6B generation or a controlled speed comparison.
The ignored receipt is `runtime/browser-qwen3-real-layer-receipt.json`; its
reproducer is `tests/run-browser-qwen3-real-layer.py`.

A fresh local rerun of the browser route on 2026-09-27 used the rebuilt
`/browser/` worker in Chrome and the Codex in-app browser, both on one Windows
computer and both reporting WebGPU. The tiny dense Qwen3 generation again
matched all three reference token IDs (`23, 4, 26`), with three verified real
MLP forwards on each browser. The tiny MoE generation also matched all three
reference IDs (`23, 29, 14`) through its browser-resident expert. With one
browser paused, dense generation matched the same reference IDs through three
native fallback forwards. For the cached Qwen3-0.6B checkpoint, one real
layer's browser activation agreed with native Torch within `2.87e-6` maximum
absolute error. Its warm browser stage samples were `134 / 138 ms` versus
adjacent native samples of `17 / 15 ms`; the first browser sample included
loading and took `2204 ms`. These are local diagnostics, not a complete 0.6B
generation or a controlled speed ratio. The rerun receipts are the ignored
`runtime/browser-*-current.json` files.

The dense bridge now returns browser output to the native stage's device and
precision, and the release Python source allowlist includes both browser
bridges and their owner module. Focused Python bridge tests, native Python
product packaging tests, mobile compute hub tests and coordinator release
policy tests passed. The ordinary `npm run dev:e2e:distributed` local gate
also passed with two signed logical workers, two model stages, matching routed
and canary output, and complete cleanup. That gate is a separate native route
on one host; it does not exercise browser workers or prove two physical hosts.
The candidate physical preflight returned `ready: false` because host B still
has a documentation-only address and lacks authentication. The connected
Android device remained `offline` to ADB, so no physical mobile-browser test
ran. Three stale landing-page assertions were updated for the page's current
availability-dependent download copy. The full Vitest suite then passed with
one worker and the repository's Python 3.12 runtime: 275 files and 1,884
tests passed; 6 files and 20 tests were skipped. Structure, docs, format,
architecture, TypeScript typecheck, mobile build and landing build passed
separately. No public browser release or one-click automatic model assignment
has been validated.

The 2026-09-27 local GPU eligibility change requires a fresh WebGPU matrix
admission for each browser session before assigning model expert work. A CPU
admission remains visible as a CPU check but is not eligible for the replicated
expert route; a CPU fallback revokes GPU eligibility and resident expert
assignments. The
coordinator rejects expert load and execution results reported as CPU and
removes the first replica if loading the second fails. Ten focused mobile
hub tests passed, including CPU-only, GPU-loss and CPU-result cases.
After rebuilding the browser, Chrome and the Codex in-app browser on the same
Windows computer both reported WebGPU. A generated tiny dense Qwen3 model
returned the same three token IDs (`23, 4, 26`) as the native reference, with
three browser-owned MLP forwards verified on each browser and zero native
fallback forwards. This is one MLP in an otherwise native model and two
browsers on one physical host. It does not demonstrate automatic distribution
of a whole model, native-equivalent speed, mobile support, or public deployment.

### Browser-owned Qwen3 decoder layer on the local coordinator

The opt-in Qwen3 layer bridge uses the existing browser worker button and
coordinator WebSocket. A single validated, visible browser can receive one
content-addressed float32 ONNX decoder layer, including attention and MLP.
WebGPU and WASM CPU are separate eligible backends; the coordinator verifies
the graph hash, model digest, numerical canary, result shape, lease, request
position, and backend. Per-request browser KV is mirrored incrementally into
the native stage so that a failed browser step can continue with the native
layer. This route does not require two browser replicas.

The local Qwen3-0.6B checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca` exported a 62,947,897-byte
layer-0 graph with SHA-256
`e9d9b5a92cb59326c5537927447050c7f102e3671a267aa0bec948289331654f`.
One Codex in-app browser on this Windows host joined with the normal button,
loaded that graph over the coordinator, and executed 11 prefill/decode layer
steps on WebGPU. Maximum absolute differences against native were `2.38e-6`
for hidden output and `3.51e-4` for incremental key. The ignored local receipt
is `runtime/browser-layer-network-receipt.json`.

The same browser handled layer 0 inside all 28 layers of the native Qwen3-0.6B
Torch model. Four greedy generations matched the native token IDs
`[16, 17, 18, 19]`; the largest full-model hidden difference was `1.35e-4`.
In a second generation, the browser executed two steps and an injected browser
failure caused two native fallback steps; token IDs and the same hidden
tolerance remained intact. The ignored receipt is
`runtime/browser-layer-full-model-receipt.json`. Focused coordinator tests also
admitted a single CPU browser peer and rejected duplicate KV positions. The
real-device CPU branch and a physical phone have not been tested in this route.

A separate local split used native ranges `[0,1)` and `[1,28)` with the
browser assigned the second range's layer 1. The browser performed four layer
forwards and the combined model again generated `[16, 17, 18, 19]`; the
browser panel displayed four verified model steps. The ignored receipt is
`runtime/browser-layer-split-model-receipt.json`. Both native ranges and the
browser ran on the same physical Windows host, so this is mixed-executor
correctness evidence rather than a physical distributed performance result.

The final local build completed, including coordinator, browser and landing
bundles. The full Vitest run passed 1,889 tests in 276 files, with 20 tests in
6 files skipped by their normal conditions. Structure, architecture, format,
documentation, TypeScript checks, Python bridge parity, and focused mobile
hub/package checks passed. No physical two-host or real mobile CPU run was
available for this gate.

### Model-independent browser layer contract (local v2 gate)

The browser worker now reads `mycellios-browser-layer/2` manifests: hidden
tensor names and width, arbitrary named auxiliary float32 tensors, and zero or
more per-request state tensors with declared axes and incremental outputs. The
coordinator validates the same declared shapes and sequence before accepting
results. The native stage has a registered adapter boundary; its built-in
rotary-KV adapter currently covers Qwen3 and Llama. A separate generic
publisher checks model and graph identities, ONNX input/output names, and a
native-output canary. A stateless ONNX Add graph with different tensor names
also passed its publisher check, including rejection of a false canary.

With one local Windows in-app browser joined through the existing button, an
exact Qwen3-0.6B layer ran 11 WebGPU steps through v2: maximum hidden error
`2.38e-6`, incremental key error `3.51e-4`. On the same browser, a tiny
two-layer Llama checkpoint ran 11 WebGPU layer steps: maximum hidden error
`1.19e-7`, key error `5.96e-8`. Four complete-model generations with the
Llama browser layer matched native token IDs `[9, 0, 5, 21]`; largest hidden
error was `4.32e-7`. In a second generation, an injected browser failure led
to two native fallback steps with the same token IDs. The coordinator also
switched this one idle worker from Qwen3 to Llama; a focused test guards
against reassignment during an active generation. Ignored local receipts are
`runtime/browser-qwen3-network-v2-receipt.json`,
`runtime/browser-llama-network-switch-receipt.json`, and
`runtime/browser-llama-full-model-receipt.json`.

A separate synthetic stateless ONNX Add stage, with `activation`, `bias` and
`result` instead of decoder tensor names and with no KV tensors, was published
through the generic publisher and assigned to that WebGPU browser. Two
sequential executions returned `[8, 10]` and `[10, 12]` with empty state;
`runtime/browser-generic-abi-receipt.json` is the ignored local receipt. This
tests the ABI flexibility, not inference of an additional language model.

This is a reusable layer ABI, not proof that every model can run today. Each
model needs a matching exporter and native adapter, and its ONNX operators,
memory and backend must pass admission. These results used one physical host;
no physical two-host, phone, or performance-parity result exists for v2.
The v2 build, bundle budgets, structure, architecture, format and type checks
passed. The full Vitest invocation passed 1,876 tests and failed 12 because
`py -3.12` is absent on this Windows host and one process-tree timing test
failed under parallel load; those three affected files then passed all 32
tests with `MYCELLIOS_PYTHON` set to the local Python 3.14 runtime and one
worker. The two native bridge checks and generic publisher check passed in
that runtime. The local browser tests did not include a real phone.

This is an opt-in local Qwen3 float32 route with an operator-published artifact
and a native stage retaining its layer weights for fallback. It has not yet
shown automatic placement in the general native topology planner, a memory
saving across hosts, a physical two-host route, public deployment, or
native-equivalent browser performance. Browser admission may assign zero work
when no matching stage is active. The button alone is not evidence that every
model or phone can execute its allocated share.

### 2026-09-27 release validation (local, unpublished)

The complete JavaScript check passed its structure, architecture, format,
documentation, bundle, and TypeScript gates. Vitest completed 1,888 passing
tests and 20 skipped tests, with one process-tree timing timeout under the
parallel full run; all 21 tests in that file passed on a single-worker retry
using the prepared Python 3.12 environment. The strict Python runner passed
1,089 tests with 18 skips after correcting bridge-test discovery and imports.
The coordinator build completed. The local distributed E2E generated matching
tokens through two signed logical workers and cleaned up its processes; both
workers ran on one computer.

The two-host preflight is not ready: the checked-in example still has
documentation-only IP addresses and no prepared remote credentials, model
revision, or ranges. That candidate had no physical phone or two-computer
inference result and no sealed release revision or public deployment.

### Automatic published-layer lookup (local candidate)

A trusted native stage can now query the coordinator for the active browser
artifact matching its exact model digest and layer range. Registration stores
an active pointer so a republished layer remains unambiguous after coordinator
restart. The isolated local agent and remote launch daemon pass the opt-in
coordinator origin and internal credential from their own host environment;
the browser user does not supply a bridge file. The coordinator retries another
eligible browser if its preferred worker fails to load or to match the native
canary. Focused registry, restart, retry, credential filtering, and native
adapter tests passed.

With one Windows in-app WebGPU browser on the same computer, automatic lookup
executed a complete tiny Llama generation with token IDs `[9, 0, 5, 21]` and
a Qwen3-0.6B generation with token IDs `[16, 17, 18, 19]`, both matching their
native references. Each run included six browser forwards and two native
fallback forwards after an injected browser failure. Maximum hidden error was
`4.32e-7` for Llama and `1.34e-4` for Qwen3. A separate Qwen3 layer check ran
11 browser steps with maximum hidden error `2.38e-6` and incremental key error
`3.51e-4`. Ignored local receipts are
`runtime/browser-llama-auto-discovery-receipt.json`,
`runtime/browser-qwen3-auto-discovery-receipt.json`, and
`runtime/browser-qwen3-network-current-receipt.json`.

These local browser results do not demonstrate universal model export,
automatic placement from the general topology planner, two-computer native
inference, browser/native performance parity, or public deployment. Model
export and a matching trusted native adapter remain prerequisites.

The candidate's complete single-worker Vitest run passed 1,891 tests with 20
skips. The strict Python run passed 1,091 tests with 18 skips. Structure,
documentation, format, architecture, component boundaries, all TypeScript
checks, the mobile and landing bundle budgets, and the build passed. The
local two-logical-worker distributed E2E passed a routed generation and
cleanup with a 240-second whole-gate allowance. A first 120-second attempt
reached the active canary but exited during cleanup at its time limit; it is
not counted as a pass. None of these local checks is physical two-host proof.
The Android device did not join the coordinator during that candidate's tests.

### Physical Pixel browser layer (2026-09-27, local test)

After integrating `origin/main` at `5569a72`, a Pixel 8 Pro running Android
17 and Chrome 153 joined the local coordinator through wireless ADB port
reverse. The ordinary **Start contributing** button admitted its WebGPU
worker `abe3923a-e52e-4c65-b360-40a1eda1b24a`. No second browser worker was
connected. The source tree still contained uncommitted changes, so this is a
local test of a candidate, not a sealed release result.

The coordinator assigned layer 0 of the exact cached Qwen3-0.6B checkpoint
`c1899de289a04d12100db370d81485cdf75e47ca` to that phone. The graph
was 62,947,897 bytes, with SHA-256
`e9d9b5a92cb59326c5537927447050c7f102e3671a267aa0bec948289331654f`.
Eleven prefill/decode layer steps ran on phone WebGPU. Maximum absolute
differences from native PyTorch were `2.38e-6` for hidden output, `3.97e-4`
for incremental key, and `8.35e-7` for incremental value. The median reported
layer duration across those mixed steps was 40.2 ms; the first step reached
199 ms. These timings are not full-model throughput or a same-GPU comparison.
The ignored receipt is `runtime/browser-qwen3-pixel-network-receipt.json`.

With automatic published-layer lookup, the same phone executed six layer
forwards inside a complete native Qwen3-0.6B generation. Generated token IDs
`[16, 17, 18, 19]` matched the all-native reference. An injected browser
failure made the native layer execute two fallback forwards while preserving
the token IDs. Maximum full-model hidden error was `1.33e-4`. The phone's
verified task counter rose from 12 to 18 during that run. The ignored receipt
is `runtime/browser-qwen3-pixel-full-model-receipt.json`.

This proves a real phone browser can compute an assigned decoder layer in a
locally coordinated model and hand back its output and KV. The ADB reverse
tunnel, short generation, one assigned layer and CPU native remainder do not
establish deployment readiness, direct internet transport, sustained mobile
performance, whole-model browser execution, universal model support, or the
native two-computer release gate.

### Public HTTPS staging route (2026-09-27, temporary test)

A temporary Cloudflare tunnel exposed a separate local test coordinator over
HTTPS and WSS. The public `/browser/` page loaded, a WebSocket handshake
completed, and the Qwen3-0.6B layer graph was published through that HTTPS
origin. The Windows in-app browser joined via the public URL as a WebGPU
worker. Eleven real layer steps returned with maximum hidden output error
`2.38e-6` and incremental key error `3.51e-4` against the native fixture.
Full-model generation using this browser layer produced token IDs
`[16, 17, 18, 19]`, equal to the native reference, with six browser forwards
and two injected native fallback forwards; maximum hidden error was
`1.34e-4`. Receipts are ignored local files
`runtime/browser-qwen3-https-network-receipt.json` and
`runtime/browser-qwen3-https-full-model-receipt.json`.

The coordinator and the Windows browser were on the same physical computer,
despite using the public HTTPS route. This tests internet ingress and browser
execution, not physical multi-host inference or a production deployment.

## Not yet demonstrated by committed evidence

- a committed, signed receipt for the two-computer route above;
- a model that requires the combined memory of several hosts;
- GPU tensor or expert parallelism across machines;
- remote recovery after physically losing a stage;
- P50/P95 latency and sustained throughput on residential LAN/WAN;
- a reproducible comparison showing that Mycellios outperforms another system.
- activation-integrity enforcement using a distinct trusted recompute stage on
  a physical route (the contract exists, but no such receipt is committed).

## Evidence policy

A physical claim is accepted only when the artifact records at least:

- commit/build and model artifact identities;
- distinct machine identities and assigned layer ranges;
- transport mode and directed link measurements;
- exact canary/parity result;
- TTFT, TPOT, tokens per second and peak memory;
- process cleanup, plus the outcome of an intentional failure when recovery is
  claimed.

Existing files in `benchmarks/` are retained local/loopback evidence. They are
useful regression baselines, but none proves a two-computer route.

## Benchmark taxonomy

Physical reports separate control overhead, process startup, model loading,
activation transfer, TTFT, TPOT, sustained tokens/s, peak memory, cleanup and
recovery. Every result is bound to an exact source/build, operating systems,
hardware and drivers, model revision, assigned ranges, topology and transport.
An optimization comparison uses the same model and topology on both sides and
must retain output parity; loopback measurements cannot establish a multi-host
performance claim.

## Next gate

### Fixed candidate and access check, 30 September 2026

The Windows 0.2.81 lab MSI was built from commit
`d5058f687419f6a2d9cae8c24f8b4706c32acbc9`. Its SHA-256 is
`2e56bc83322ac60d8cbea3ef6b3defeb5d676a8f7c092a01f3631c34514e9a1d`;
Authenticode reports `NotSigned`. PR #37 remains a draft. Its current checks
passed, including the Windows, Linux and macOS native jobs and real candidate
assembly. This establishes CI packaging evidence, not public signing or a
physical installation by itself. Later in this session, the hash-checked MSI
upgraded MRPUCCHI to 0.2.81 and preserved its existing node identity.

PC-DANI's private lab coordinator now has a scheduled supervisor. With no
requested model, its coordinator process was deliberately terminated; the
supervisor started a new process and `/health` returned 0.2.81 after 10.97
seconds. PC-DANI's CPU worker subsequently reconnected. The evidence is retained
locally in `runtime/native-node-lab/coordinator-recovery-20260930.json`. This
is a local process recovery test; unattended PC-DANI reboot recovery and a
two-computer inference with 0.2.81 remain unverified.

MRPUCCHI answered a Tailscale ping, and the console's saved access rule permits
PC-DANI (`100.93.34.56`) to reach MRPUCCHI (`100.102.91.6`) on TCP 22. However,
Codex's local TCP probe returned Windows socket error 10013, including on the
requested escalated retry. The local firewall contains Codex sandbox outbound
restrictions. This does not establish a remote SSH authentication failure.
Jarvis's PC panel separately reported MRPUCCHI offline with its last signal at
28 September 20:37:20. Execution of the delivered `PREPARAR-MRPUCCHI.cmd`
bootstrap had not been confirmed at that check. The fixed physical repeat was unavailable
until a working authorized control route and the installed build are verified.

A later access check at 07:00 UTC reached MRPUCCHI's SSH server and was rejected
with `Permission denied (publickey)` for both the dedicated lab key and the
existing default key/agent route. The preceding local socket error must not be
treated as the current SSH failure. MRPUCCHI's signed native snapshots were
fresh, reporting 0.2.80, CUDA and `reconnecting`, while its worker remained
offline. With zero requested models and zero active commands, the operator
queued command `3e7a91f1-dabf-476f-a7bb-a9dc48535068` through the private lab's
validated command store to reapply the exact observed policy. MRPUCCHI returned
an authenticated `applied` result with no error at 07:04:59.857 UTC. That command
requests a process restart after acknowledgment. No newer snapshot appeared
during the following 154 seconds, and the worker did not reconnect; this proves
command delivery and acknowledgment, not successful restart or recovery.
Its result is retained in
`runtime/native-node-lab/reconnect-command-20260930.json`. The fixed MSI upgrade
was subsequently completed; a fresh physical inference remains a separate gate.

### Generic private USB bootstrap and actual reboot, 30 September 2026

The earlier SSH rejection used the wrong Windows account, `daniel`. The same
dedicated key successfully authenticated as `danie`, with administrator access.
The missing Jarvis signal was separately traced to its user-login startup:
MRPUCCHI had no interactive Windows user. Neither finding establishes a VPN or
remote-server outage.

The generic Windows x64 launcher was executed twice on MRPUCCHI, from the same
hash-checked package copied to its local disk. Both completed with exit code
zero. It created the managed `mycellios-control` administrator, retained existing
SSH administrators and the Mycellios identity, restricted its firewall rule to
the controller's Tailscale IP, configured service failure recovery and registered
the SYSTEM startup recovery task. A fresh SSH connection as that new account
confirmed administrator membership. Mycellios reported Ready with CUDA.

MRPUCCHI then underwent a real Windows reboot. At 08:15:14 UTC, its boot time
was 08:14:16.5 UTC, no interactive user was logged in, Tailscale, SSH and
Mycellios were Running, and Mycellios reported Ready with CUDA. Exactly one
Jarvis worker ran in session zero. Jarvis's database subsequently reported a
fresh MRPUCCHI heartbeat. This verifies unattended administrative access and
native-node startup on that PC; it does not verify an interactive desktop before
Windows login. Evidence is retained locally in
`runtime/native-node-lab/generic-usb-reboot-20260930.json`.

The private generic package contains ten separate pre-approved, non-reusable
Tailscale enrollments. The OAuth secret remains encrypted on PC-DANI and the
package contains only its public SSH key. Unused VPN enrollments expire after
90 days. The private coordinator issues a fresh 15-minute Mycellios enrollment
against a per-install claim, permanently binding each claim to one machine.
Three focused tests cover denial, expiration, same-machine retry, cross-machine
rejection, renewal and consumed enrollment; all passed. Package integrity,
structure, architecture, documentation and TypeScript checks passed. Tailscale's
bundled MSI has a valid vendor signature; the private Mycellios candidate remains
unsigned.

Daniel subsequently mounted the ESD-USB removable drive at D:. The complete
private package was copied into `D:\MYCELLIOS`, preserving Windows installation
media and previous folders. The sole entry file is `D:\INSTALAR-MYCELLIOS.cmd`.
Read-back preflight verified every static payload checksum and ten unused
single-PC tickets remained on the drive. No OAuth secret or private SSH key was
added to the new package. The generic Tailscale grant for controller SSH to
`tag:mycellios-lab` is prepared but awaits the required browser confirmation.
A first installation on a second clean PC, its interactive Jarvis pairing and a
physical inference through 0.2.81 are not established by the retained-node test.
PC-DANI's SYSTEM startup coordinator task was subsequently registered with
elevated Windows permissions; its configuration receipt is retained at
`runtime/native-node-lab/controller-startup-receipt.json`. Its actual reboot
has not been tested.

The single-file launcher now stages the complete static package and one ticket
in protected ProgramData, registers startup and ten-minute retries, and handles
installer reboot requests without reopening a USB file. A focused PowerShell
test verified allocation of one ticket, no copying of the unused ticket pool,
quoted profile paths, resume from disk after a simulated boot change, no repeat
restart on the same boot, and bounded reboot/retry counts. This test does not
establish a physical installer-required reboot.

At 09:16 UTC the revised launcher was executed as SYSTEM on the actual MRPUCCHI
against its existing installation. Exit code was zero, its original node identity
was unchanged, local health was Ready and the recovery task was Running. The
complete package was retained locally and the completed installation's resume
task was removed. This retained-node regression is separate from a first clean
installation or physical USB execution on another PC.

At 09:18 UTC a second SYSTEM invocation used only the staged ProgramData
installer with `-Resume`, without a USB source path or ticket pool on the PC.
It completed with exit zero, Ready/CUDA, the same identity, no interactive
Windows user, Running recovery, and no remaining installation resume task.
Evidence is saved in `runtime/native-node-lab/generic-usb-local-resume-20260930.json`.
Administrative SSH file transfer via its SFTP subsystem also passed. This is a
physical local-disk resume check, not a fresh installation or a new reboot test.

### Native 0.2.81 physical repeat and stream-failure finding, 30 September 2026

At 09:30 UTC the private coordinator activated the pinned SmolLM2 135M model
across PC-DANI and MRPUCCHI with both workers reporting 0.2.81. Native source
areas had no diff from `d5058f687419f6a2d9cae8c24f8b4706c32acbc9`
at this point; later administrative USB work was uncommitted. Both hosts' model
weights matched SHA-256
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`
at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
The automatically planned route assigned [0,4) to PC-DANI CPU/float32 and
[4,30) to MRPUCCHI RTX 2060 CUDA/float16, with fp16 activation transfer through
the authenticated runtime relay. This differs from the earlier equal split and
is not a controlled performance comparison.

The 16-token canary measured TTFT 164.3 ms, TPOT 84.3 ms, pipeline 1428.8 ms
and 11.86 output tokens/s. It returned the same apologetic text as the earlier
canary rather than following the instruction to answer only OK. A separate
coordinator API request answered `2 + 2 is 4.` with nine output tokens and
204 ms reported TTFT. Its execution receipt identifies both physical hosts,
assigned ranges and CPU/CUDA backends. Peak process working sets sampled after
the request were 738,410,496 bytes locally and 1,414,848,512 bytes remotely.
An additional whole-GPU sample reported 377 MiB used; this is neither per-model
peak VRAM nor proof of exclusive GPU use. Evidence is retained under
`runtime/native-node-lab/` in `activated-0281-20260930.json`,
`inference-0281-20260930.json` and `memory-0281-20260930.json`.

After a manual reactivation, a controlled fault terminated MRPUCCHI's stage at
09:40:01.464 UTC while a 256-token streaming request was still running. The
stream ended 206 ms later with only 101 text chunks, yet the coordinator marked
the partial answer completed with no failure code. The native pipeline adapter
ignored the Python runtime's SSE error payload and accepted DONE without a
terminal result. This is an incorrect success report, not recovery. An earlier
kill occurred after its request ended and is excluded from the active-inference
fault result. Model status subsequently became failed with no automatic retry;
manual deletion cleaned up the route and left both native workers connected.

A local uncommitted adapter correction now rejects native SSE errors, bare DONE,
premature EOF and terminal chunks without valid token usage. After partial output
it does not classify such failures as retryable. Twenty-one focused adapter and
worker-hardening tests passed, along with TypeScript, structure and architecture
gates. At that point the correction was not committed, published or included in
a replacement MSI.

After reloading the private coordinator with that uncommitted correction, the
same two physical nodes and pinned model passed another real activation canary.
At 09:52:16.071 UTC, MRPUCCHI's stage was terminated during a new streaming
request. This time the client received an explicit `adapter_error` rather than
a successful partial result; the message explained that exact replay after
streaming begins requires an explicit deterministic seed. The stream ended
287 ms after the fault and included 89 partial text chunks. This verifies
failure propagation, not automatic recovery or output completion. The test
differs from the fixed-commit baseline by the local adapter correction.
The coordinator recorded the request as failed. After removing the test model,
neither host retained a Python stage, local API/return listeners were closed,
both native workers were connected, and MRPUCCHI's installed service remained
Running/Ready. Cleanup evidence is in `streamfix-cleanup-20260930.json`.
Evidence is retained in `loss-stream-0281-streamfix-20260930.json`,
`fault-0281-streamfix-20260930.json` and `streamfix-fault-state-20260930.json`
under `runtime/native-node-lab/`. First clean-PC installation, the rebuilt
fixed-commit release and external testers remain separate gates.

### Private 0.2.82 upgrade, physical generation and reboot, 30 September 2026

The replacement Windows x64 MSI includes the streaming adapter correction.
Its administrative extraction and complete file-tree comparison passed, as did
41 focused adapter, worker, provisioning, MSI/layout and version tests, the
PowerShell USB staging/upgrade regression, TypeScript, structure, architecture
and documentation checks. The MSI is 256,143,888 bytes, SHA-256
`3ba4b6c6f1ac8474adff9d7ae4ca1843a5b2363a60101915573fadc29a7ef241`.
Its native source snapshot is
`sha256:8986f290b63132ea9df9fb6cda25029197d3a1197ae5458f286297e933508585`,
with parent revision `d5058f687419f6a2d9cae8c24f8b4706c32acbc9`.
This is an unsigned private candidate from uncommitted source.

The generic installer ran as SYSTEM on the actual MRPUCCHI and returned 0.
It retained `node-4c3832ee0e884100ba0a6596a400192c`, reached Ready/CUDA,
kept service recovery running and removed its installation resume task.
Only one MSI remained in the protected local package, and no unused ticket
pool was copied to the PC. This verifies a retained-node upgrade through the
generic installer, rather than a first installation on a clean PC.

Both physical workers then reported 0.2.82 and generated with SmolLM2-135M-
Instruct at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
The receipt assigns layers [0,3) to PC-DANI CPU and [3,30) to MRPUCCHI CUDA.
The 16-token activation canary measured TTFT 433.97 ms, TPOT 73.28 ms and
13.65 tokens/s. The subsequent API request reported TTFT 222 ms. Its response
described addition without answering 4, so this is generation evidence, not
a passing arithmetic test. A separate monolithic CUDA/float16 eager reference
used the same prompt, chat template, pinned snapshot and verified weight hash.
It produced exactly the same text and 17 completion tokens. This verifies
output parity for this one greedy prompt, not general semantic accuracy or a
performance comparison. Peak process working sets were
717,004,800 bytes on PC-DANI and 1,402,384,384 bytes on MRPUCCHI. These are
process observations, not exclusive GPU memory or performance percentiles.
Deleting the test model left zero stages, requested models and API/return
listeners, with both native workers connected.

MRPUCCHI was rebooted after cleanup with no interactive user. It booted at
10:53:33.5 UTC. At 10:55:27 UTC, administrative SSH worked, Tailscale, sshd
and MycelliosNode were running with automatic startup, and the node reported
Ready/CUDA with the retained identity. A later process check found one headless
Jarvis worker; Jarvis's operational database reported its fresh MRPUCCHI signal
at 10:59:43 UTC. Desktop capability remained null without a Windows login.
This verifies unattended access after an actual reboot of the upgraded PC.

The mounted ESD-USB at D: now contains these verified MSI bytes, 22 verified
static payload files and ten unused enrollments. Its single root entry remains
`D:\INSTALAR-MYCELLIOS.cmd`. The USB readback/preflight passed and its Windows
installation media were preserved. The generic SSH grant to tagged new PCs is
still pending; the existing individual MRPUCCHI grant supported these tests.

The local development gate initially failed with an unseeded default cache,
then with canonical-source/cached-artifact mismatches in older Tiny-Llama data.
Using its canonical `hmellor/tiny-random-LlamaForCausalLM` identifier, a prewarmed
model cache and fresh node caches completed a real two-worker generation.
This gate is CPU loopback with logical workers; its output was `eno Twe` and
does not establish physical multi-host execution or answer quality. Older
node caches were preserved separately. The default fixture/cache path is not
verified as working from a fresh checkout.

Evidence is retained under `runtime/native-node-lab/` in
`candidate-0282-20260930.json`, `upgrade-0282-20260930.json`,
`physical-repeat-0282-20260930.json`, `reboot-0282-after-20260930.json`,
`jarvis-after-reboot-0282-20260930.json`,
`monolithic-parity-0282-20260930.json`,
`usb-drive-one-click-0282-20260930.json` and
`dev-e2e-0282-fresh-cache-20260930.log`.
No commit, public release, first clean-PC USB execution or external tester
installation is claimed by these results.

### Private USB retry correction, 30 September 2026

The external PowerShell helper now retains MSI progress after a successful
fresh installation, before native pairing. Its regression executes the actual
installer entry with mocked Windows installation and pairing boundaries: two
pairing failures resume bootstrap while running the MSI once, and altered MSI
bytes are rejected. This is local regression evidence, not a clean-PC physical
installation. The existing physical 0.2.82 MSI is unchanged.

The verified static USB payload was refreshed without issuing new enrollment
keys. Its readback/preflight verifies package integrity only. Pending native
configuration, expired pairing after configuration creation and interrupted
accelerator provisioning remain separate recovery gaps; this test does not
establish their recovery or completion of the generic installation gate.

### Unreleased native installation continuation, 30 September 2026

Source now resumes complete bootstrap output, renews pending pairing without
changing the node identity, and restarts only the validated LocalService SCM
service. A protected bootstrap journal also reconstructs missing initial files
with the original node ID after validating every surviving file. Existing
identity metadata prevents resetting an already running node from this journal.
Consumed enrollment retries require the same nonce, identity, key and active
ownership; an expired unconsumed enrollment remains rejected by the coordinator.
GPU setup failures retain installation retries instead of reporting CPU fallback
as completed GPU setup. CPU-only and unsupported GPU devices remain permitted.

63 focused tests passed across bootstrap, continuation, service registration,
enrollment store/API/bootstrap, accelerator setup and private claim provisioning.
PowerShell MSI retry/package selection and USB staging regressions also passed,
as did TypeScript and architecture checks. These exercise real temporary files
and databases with mocked service/GPU boundaries; they are not physical first-
installation evidence. The mounted USB retains the previously verified 0.2.82
payload. This continuation source requires a new matching native MSI and
coordinator verification before replacing that payload.

The private 0.2.83 MSI subsequently passed extracted-tree verification and a
retained MRPUCCHI upgrade returned 0. Mycellios, SSH and Tailscale remained
Running, the original node ID remained registered and CUDA diagnostics were
ready. Physical SmolLM2-135M-Instruct inference then completed between MRPUCCHI
CUDA layers [0,29) and PC-DANI CPU layers [29,30), using the previously pinned
model revision. The execution trace recorded a direct physical boundary. API
TTFT was 130 ms; the separate 16-token activation canary measured TTFT
101.661 ms, TPOT 76.437 ms and 13.083 tokens/s. Observed peak process working
sets were 629641216 bytes on PC-DANI and 1544298496 bytes on MRPUCCHI; these
are not exclusive VRAM measurements. The response repeated the earlier
arithmetic explanation without a numeric answer. No new general accuracy or
performance comparison is claimed. Requested models and matching Python stage
processes were absent after cleanup. Evidence is retained privately in
`runtime/native-node-lab/physical-repeat-0283-20260930.json` and
`runtime/native-node-lab/upgrade-0283-20260930.json`. The source remains uncommitted.

An isolated new native service on that retained physical PC rejected an expired
pairing as expected, but installation continuation failed with
`node_install_resume_service_ownership_mismatch`: Spanish Windows translates
the `sc.exe qc` labels assumed by the candidate. The temporary service, its
data and its protected key were removed; the primary native node stayed Running.
The 0.2.83 MSI therefore does not pass first-installation recovery. The source
correction checks invariant CIM Name, PathName and StartName instead. Eight
focused continuation tests passed, including Spanish state output and rejection
of invalid CIM output, a foreign name or another account before any mutation.
A read-only probe on the actual Spanish MRPUCCHI accepted the real owned service;
stop/start operations were intercepted, so it is not a completed recovery test.
Receipts are `runtime/native-node-lab/fresh-native-0283-result.json` and
`runtime/native-node-lab/localized-resume-readonly-result.json`. The correction
requires a rebuilt native MSI and a repeated actual continuation test before
updating the mounted 0.2.82 USB. A separate PowerShell regression also verified
that downloaded enrollment JSON is written without BOM and parsed by Node.

### Private 0.2.84 recovery and USB refresh, 30 September 2026

The rebuilt private Windows x64 MSI passed administrative extraction, decompilation
and exact staged-tree comparison. Its SHA-256 is
`1273efbc0012f649a06766ad13277af8b5f4d5e4e08fe0491f668a1bf1d7de3c`;
the source ID is
`sha256:6a7fc68334af51b9edd494a5818ae48569bcd634e284c6a3769f284e8925cdff`.
It remains unsigned, uncommitted and unpublished. The generic retained upgrade
on MRPUCCHI returned 0, reached CUDA Ready and preserved the hashes of identity
metadata and its DPAPI-protected key. SSH and Tailscale stayed Running.

An isolated new LocalService node on the same retained physical PC rejected a
server-expired pairing. The actual installed resume entrypoint then renewed its
pairing and reached CUDA Ready with the same identity and protected key. A second
restart acknowledged the already consumed pairing despite an expired local
bundle, removed pending pairing and again reached Ready. Temporary services,
data and protected keys were removed, and their coordinator credentials revoked.
The primary node remained Running. This reused installed Windows and the shared
CUDA runtime; it is not a clean-PC USB installation or a cold GPU download.

The initial recovery harness rejected the configuration-file change and recorded
a failed assertion. The follow-up captured both files: the only changes were
revision 1 to 2 and insertion of the exact default work policy during the first
coordinator reconciliation. Runtime paths, resource limits and all other fields
were unchanged. The follow-up checks only that documented transition in addition
to exact identity/key preservation; it does not permit arbitrary configuration
changes. Receipts are `runtime/native-node-lab/fresh-native-0284-result.json`,
`runtime/native-node-lab/fresh-native-0284b-result.json` and
`runtime/native-node-lab/upgrade-0284-20260930.json`.

Another physical generation completed with the installed MRPUCCHI native GPU
worker 0.2.84 and the existing PC-DANI lab CPU worker 0.2.83. The observed route
was CPU [0,1), CUDA [1,30), with a direct physical boundary. API TTFT was 135 ms;
the 16-token activation canary measured TTFT 119.330 ms, TPOT 75.293 ms and
13.281 tokens/s. Observed peak process working sets were 674074624 and
1415569408 bytes, respectively. The same non-numeric arithmetic explanation
was returned. This mixed-version run verifies the updated GPU worker executes
an assigned partition, but is not the same-commit release gate or a performance
comparison. Evidence is `runtime/native-node-lab/physical-repeat-0284-mixed-20260930.json`.

A separate current PC-DANI CPU microbenchmark passed the unchanged planner
confidence gate: decode 19.799%, prefill 7.167%, codec 15.271% confidence
half-widths. This is standalone calibration, not proof that the coordinator
accepted the CPU node's challenged evidence. Twenty-seven focused packaging and
continuation tests and the PowerShell retry, staging/reboot and no-BOM regressions
passed, together with type, structure and architecture checks.

The mounted ESD-USB was updated to this private 0.2.84 candidate. Readback
verified 22 static files, exactly one MSI, unchanged root
`INSTALAR-MYCELLIOS.cmd` and unchanged bytes of its ten unused tickets. Writes
stayed inside `D:\MYCELLIOS`; Windows installation media was preserved. The
receipt is `runtime/native-node-lab/usb-drive-one-click-0284-20260930.json`.
The external USB helper was then corrected for installation on the controller
itself: matching target/controller Tailscale IPs skip creation of a self-proxy on
the coordinator's loopback port. A guarded PowerShell regression passed; the
actual production helper ran on PC-DANI with the same listener PID and healthy
coordinator before and after. The refreshed USB passed readback again, with
manifest SHA-256
`488d29b6153790fbe0d68d4c15243d02bcee34d93c15ff72fad8c6a243d88f35`.
Its native MSI was unchanged. This is a route-helper check, not a full controller
installation or reboot. The physical generation's subsequent cleanup found two
connected workers, zero requested models and no matching Python stage processes
on either PC; see `runtime/native-node-lab/cleanup-0284-20260930.json`.
At this checkpoint, generic tagged SSH access still awaited administrative
confirmation. The controller's SYSTEM startup configuration had a new elevated
receipt, while automatic approval review had rejected the proposed live handoff.
The subsequent saved grant and live handoff are recorded below; PC-DANI reboot,
first clean-PC USB execution, a fixed committed release and accompanied external
testers remain unverified.

### Private access and live SYSTEM handoff, 30 September 2026

The saved Tailscale grant now permits PC-DANI (`100.93.34.56`) to reach
`tag:mycellios-lab` on TCP 9750, 18101 and 22. The existing unrelated grants
were retained. The Machines panel independently showed both PC-DANI and
MRPUCCHI connected with **Expiry disabled**. This verifies the currently saved
private access configuration, not installation on another PC.

The live coordinator handoff passed without rebooting Windows. The previous
interactive task was disabled; the startup supervisor is Running as SYSTEM.
The actual serving process changed from PID 62388 to 39844 and its owner SID
is `S-1-5-18`. Both physical workers reconnected at version 0.2.84; see
`runtime/native-node-lab/controller-system-startup-0284-20260930.json`.
This proves the live serving identity and reconnection, not PC-DANI's first
post-reboot execution.

A subsequent physical generation assigned CPU layers `[0,1)` to PC-DANI and
CUDA layers `[1,30)` to the installed MRPUCCHI node. The execution receipt
reports one physical boundary using relay transport. API TTFT was 146 ms;
the separate 16-token canary measured TTFT 171.6823 ms, TPOT 86.1085 ms and
11.6132 tokens/s. Observed peak process working sets were 674,160,640 bytes
on PC-DANI and 1,414,955,008 bytes on MRPUCCHI. These are process memory
observations, not exclusive VRAM measurements or a browser/native comparison.
The first diagnostic runner could not see SYSTEM stage command lines from its
unelevated memory query; the elevated repetition recorded both actual stage
processes and passed. See
`runtime/native-node-lab/physical-repeat-0284-system-20260930.json`.
The source is still uncommitted and this remains a private candidate, not a
fixed-commit release or a first installation on a clean PC.

The physical loss test then stopped `MycelliosNode` on MRPUCCHI after the first
streamed token. The native service was Stopped at 15:25:22.910 UTC while VPN
and SSH remained Running. The stream had delivered 20 content chunks and
terminated with explicit `adapter_error` at 15:25:23.922 UTC. It did not replay
the partial answer: exact replay requires an explicit deterministic seed.
The SSE `[DONE]` event marks the end of this error stream, not successful
completion. The harness restored the service and both physical workers were
connected again by 15:25:39.761 UTC with zero requested models. See
`runtime/native-node-lab/loss-0284-system-20260930.json`. This is actual worker
service loss with administrative connectivity retained, not physical power
loss, a network partition or seamless failover. An earlier harness attempt
requested 512 tokens against the 256-token limit and issued no fault; its
failed receipt is retained separately and is excluded from this result.

After restoring MRPUCCHI, a new model request completed another physical
CPU/CUDA generation with API TTFT 142 ms. The separate recovery canary measured
TTFT 167.6377 ms, TPOT 91.5138 ms and 10.9273 tokens/s; it is not a controlled
performance comparison with the earlier run. The final elevated observation at
15:31:57 UTC found both physical workers connected, zero requested models and
no matching stage processes on either computer. See
`runtime/native-node-lab/physical-repeat-0284-system-recovered-20260930.json`
and `runtime/native-node-lab/cleanup-0284-system-20260930.json`. The mounted
USB's static hash and architecture preflight also passed again without installing
anything. First clean-PC USB execution, the controller's actual reboot,
fixed-commit packaging and accompanied external attempts still need their own
evidence.

The final source gate passed 52 focused tests in eight files covering the native
installer, enrollment, accelerator retry, USB claims, stream adapter and worker
hardening. Four PowerShell regressions passed for MSI continuation, protected
staging/reboot retries, UTF-8 enrollment output and controller route preservation.
TypeScript, repository structure, documentation and architecture verification
also passed. These gates do not replace the missing physical installations or
authorize publication.

Daniel subsequently deferred the separate clean-PC USB check and requested
moving to the next plan step. That check is deferred, not passed. Release
preparation continues without repeating the USB installation gate.
The public health endpoint still reported version 0.2.77 and its downloads
availability endpoint reported all three native packages unavailable. HTTP 200
from `/downloads/windows` returned the landing page, not an MSI. These public
observations are separate from the working private two-PC 0.2.84 route.

A prospective release-source snapshot was prepared from Git-visible files and
the explicit build artwork, excluding ignored local configuration, runtime
state, ticket pools, private credentials and previous binaries. The snapshot
contains 1,135 files and seals source ID
`sha256:58d6f84a6f50bc9883ac15a3fc76b55068b499890f2f66cec4c52856b6dc8050`;
see `runtime/native-node-lab/release-source-review-0284-20260930.json`.
It is an uncommitted review snapshot, not a compiled or deployed release, and
its source ID differs from the earlier private MSI's source snapshot. A new
build and installer verification must precede promotion of that source.

Build the Windows node installer and coordinator from a fixed commit including
the CPU identity and retry fixes, then repeat the launcher with account-owned
pairing against a durable coordinator. Verify an installed CPU-only node and
its coordinator-accepted calibration,
the authenticated owner Pause/Resume UI and unattended reboot reconnection.
The lab launcher, native command protocol, supervisor reload, assigned
 inference, session re-registration after HTTP 401 and stage cleanup now have
 physical evidence. Preserve their measurements and failure result, but do not
 treat this dirty-tree build as a reproducible release. Then test the same
 installation path with an external participant before claiming one-click
 readiness.
 A private Tailscale Serve TCP proxy on the existing allowed port 18100 and a
 persistent MRPUCCHI loopback port proxy replaced the manual SSH tunnel. The
 installed node reconnected after an unattended MRPUCCHI reboot and completed
 another physical inference. PC-DANI's own unattended reboot recovery is
 unverified despite the configured startup supervisor. This private
 lab route does not replace the public HTTPS, account-owned coordinator needed
 for external testers. The lab MSI physically upgrades a retained node and
 restores a missing service under LocalService without another pairing. Repeat
 that gate from a fixed commit and released artifact; the dirty-tree result is
 not a release.

### Official source build and coordinator deployment, 30 September 2026

The native recovery changes were committed and pushed as `bd935e36`. The
release revision `ef6078734f79f8a3c9fd25f40756d5078a6bc55f` increments only
`package.json` and `package-lock.json` to 0.2.85 so Windows can upgrade the
existing 0.2.84 lab installation. No inference code changed in that increment.
The general public quality workflow passed for `bd935e36`. The official
[0.2.85 native workflow](https://github.com/tych0s/mycellios/actions/runs/36749878987)
passed Windows x64, Linux x64, macOS arm64 and the three-platform candidate
assembly. This proves fixed-commit packaging; physical installation evidence
for the Windows MSI is recorded below. Platform signing is still unverified.

An official Dockerfile build from that release revision produced coordinator
image `sha256:9492ab1b55e1a9401ccfa258e221472729afbf00adcd3f3b804822b0c0818c4f`
with source identity
`sha256:566dbaa20f6e05b646979a2fc2be64c90966abe9a578bbfd3ba8626f0c9268ee`.
An isolated container without production credentials, data or network passed
health, landing, browser, downloads and snapshot HTTP checks. Initial canary
attempts omitted the writable expert storage setting and failed without
changing production; the corrected isolated configuration passed.

The production coordinator was located on Hetzner DB, not the earlier
Scaleway host. At 17:34:46 UTC it was replaced with that verified image after
checking there were no active jobs, retaining its environment, mounts, network,
resource limits and restart policy. A coherent SQLite backup and the stopped
previous container were retained on that server. Supabase, ingress and Traefik
were not restarted. Public root, www, browser page, browser entry JavaScript,
health and download metadata returned HTTP 200. Public health identifies the
exact 0.2.85 revision/source identity and persistence is connected without
errors. Local evidence is in
`runtime/native-node-lab/deployment-0285-20260930.json` and
`runtime/native-node-lab/public-verification-0285-20260930.json`.

This deployment uses the functional source already exercised in the two-PC
0.2.84 lab route plus the metadata-only version increment. It is not a new
same-commit physical installation test. Automatic permission review rejected
an attempt to reload PC-DANI's SYSTEM controller with `blocked by policy`;
that reload was not executed. The private PCs remain separate from production.
The initial deployment verification found zero connected workers. A later
public browser test connected one WebGPU browser node; native production
inference capacity and a public one-click inference remain unproven.
All three native downloads remain unavailable: CI packages were not promoted
without the required platform signature verification. The clean-PC USB check
remains deferred by Daniel, and no external tester installation is counted.

### Public browser connection and exact CI artifact, 30 September 2026

The public `/browser/` page on the deployed 0.2.85 coordinator was opened in
the Codex in-app browser and `Start contributing` was clicked. The page reached
`CONNECTED` and reported `GPU validated and waiting for compatible model
work`. The dashboard independently showed one online browser and zero active
models or evidenced routes. This proves public browser enrollment and the GPU
validation task, not a model layer execution or full distributed inference.
The screenshot is
`runtime/native-node-lab/public-browser-connected-0285-20260930.png`.

The manual workflow run `36749878987` produced the exact release revision
`ef6078734f79f8a3c9fd25f40756d5078a6bc55f`; the pull-request workflow
`36749853179` used a synthetic merge revision and is not the fixed revision
artifact. The manual candidate manifest and Windows MSI were downloaded and
the MSI's 252,764,784 bytes and SHA-256
`aa6c8f487a225b5b6230154d5257b4d0de303506b04ce35c6838269b6f14b76b`
matched that manifest. Windows Authenticode reports `NotSigned`. No code
signing certificate was found in either Windows certificate store and GitHub
lists no signing secrets. Candidate hashes and successful builds do not
replace platform signing; public native downloads remain gated.

Public native enrollment requires an authenticated account session. The
opened dashboard has no such session; its sign-in form is prepared without
creating credentials or bypassing account ownership.

### Fixed CI Windows MSI on MRPUCCHI, 30 September 2026

The checksum-verified Windows MSI from manual run `36749878987` was installed
over the retained 0.2.84 node on physical MRPUCCHI. The existing installation
helper restored its service and reported `Ready` with CUDA. The installed
manifest reports version 0.2.85, revision
`ef6078734f79f8a3c9fd25f40756d5078a6bc55f` and source identity matching the
production coordinator. The identity file and DPAPI-protected key were
byte-identical before and after the update. MycelliosNode, SSH and Tailscale
were all running after installation. Evidence:
`runtime/native-node-lab/ci-msi-upgrade-0285-20260930.json`.

This is a physical update of an existing lab node, not a clean-PC installation
or enrollment of that native node into the public coordinator. PC-DANI's
private SYSTEM controller remains version 0.2.84.

The fixed CI-installed MRPUCCHI node then completed another physical split
generation with that existing PC-DANI controller using the same pinned
SmolLM2-135M-Instruct model revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The observed planner assigned
MRPUCCHI CUDA layers [0,29) and PC-DANI CPU [29,30), with one direct physical
boundary. Generation TTFT was 134 ms; the activation canary measured TTFT
153.9803 ms, TPOT 95.8469 ms and 10.4333 tokens/s. These measurements are a
mixed-version integration result, not a controlled performance comparison or
output-parity assessment. The small model's returned text did not actually
answer the arithmetic question correctly; execution success does not imply
answer quality.

The measurement harness exited with an error because its local process-memory
collector returned empty JSON. Consequently this run has no verified live
stage memory measurement. The generation receipt and execution trace were
saved before that collector failed, and the harness's cleanup completed:
zero requested models, zero observed stage processes and both native workers
connected. Evidence is in
`runtime/native-node-lab/physical-repeat-0285-ci-mixed-20260930.json`,
`runtime/native-node-lab/inference-0285-ci-mixed-20260930.json` and
`runtime/native-node-lab/cleanup-0285-ci-mixed-20260930.json`.

### Browser continuation check, 30 September 2026

On a later inspection, the previously connected browser was no longer counted
by public health and its UI said it was waiting for coordinator confirmation.
The first explicit Pause/Start attempt reached a reconnecting loop. A second
explicit Pause/Start obtained admission challenge HTTP 200, registration HTTP
201 and WebSocket HTTP 101, then reached `CONNECTED` and GPU validated again.
Screenshot: `runtime/native-node-lab/public-browser-resumed-0285-20260930.png`.
The production container had not restarted (restart count zero). This is a
manual recovery result, not proof of unattended browser recovery.

Source inspection identifies a recovery gap worth testing: disconnected
browser registrations expire after 60 seconds in `mobile-compute-hub.ts`,
while `mobile/runtime.ts` retries the existing credentials when the WebSocket
closes. The hub rejects expired credentials with code 4401. The precise close
code of the observed failure was not captured, so expiry is a plausible cause,
not a confirmed diagnosis. Signed re-registration, cancellation and explicit
removal/supersession behavior remain open in the browser specification.

### Measured native repeat and next loss-test blocker, 30 September–1 October 2026

A later mixed-version physical repeat completed with the exact 0.2.85 CI MSI
on MRPUCCHI and the existing 0.2.84 controller on PC-DANI. The planner assigned
PC-DANI CPU [0,3) and MRPUCCHI CUDA [3,30). Generation TTFT was 224 ms; the
canary measured TTFT 137.3848 ms, TPOT 81.8881 ms and 12.2118 tokens/s.
These are separate runs with different partitions, not a controlled speed
comparison. Output parity and answer quality were not verified.

The memory collector was repaired to trace local Python descendants of the
owned coordinator listener, avoiding inaccessible SYSTEM command-line fields,
and to serialize process arrays explicitly. The local process observation had
working set 539,258,880 bytes and peak 717,258,752 bytes. MRPUCCHI's stage had
working set 1,394,110,464 bytes and peak 1,400,238,080 bytes. The local group
includes the pipeline server; these are process working sets, not exclusive
VRAM measurements. Cleanup verified zero requested models, zero owned local
Python descendants, zero observed remote stages and both native workers
connected. Receipts:
`runtime/native-node-lab/physical-repeat-0285-ci-measured-20260930.json` and
`runtime/native-node-lab/cleanup-0285-ci-measured-20260930.json`.

At 04:34 UTC on 1 October, the new loss test stopped at its preflight because
the expected pair was not simultaneously available. No service fault was
introduced. Its cleanup observation timed out. Subsequent snapshot retrieval
did show both expected native versions connected, but repeated five-second
`/health` requests timed out while the private listener PID 39844 remained
alive. Controller logs showed stale worker heartbeats and an unavailable CPU
calibration probe. MRPUCCHI SSH remained accessible. The cause of the private
controller's intermittent response is unconfirmed; this new loss/recovery
test is unavailable, not passing. Its failed-preflight receipt is
`runtime/native-node-lab/loss-0285-ci-20260930.json`.

The earlier automatic-review rejection still prevents an agent-driven SYSTEM
controller restart. A manual restart of the existing startup task has been
requested. Public account enrollment, signing, both-host fixed-release and
controller reboot gates remain open. The quickstart now includes two unassigned
external attempt slots and their acceptance checklist; no testers have been
invented, contacted or counted as installations.

### Browser registration recovery correction, 1 October 2026

The browser client now discards expired ephemeral credentials on WebSocket
4401, cancels old execution state, reinitializes its backend and renews its
signed registration with the existing browser identity. User pause aborts that
renewal. Coordinator removal, supersession and terminal protocol/rate-limit
closures pause contribution instead of causing a re-registration loop. A
failed admission remains failed. Version 0.2.86 is prepared for this change;
this section is local candidate evidence, not a claim of public deployment.

An actual in-app browser against an isolated loopback coordinator passed:
automatic recovery after restarting that coordinator with no additional Start
click; unchanged client ID and a new worker session with a verified GPU task;
pause while an admission challenge was held; terminal HTTP 401 admission
denial; superseding the original tab with a second tab; and explicit coordinator
removal. Both pause and denial left no worker registered, and supersession and
removal left the old contribution paused. The test controls deliberately
expire a disconnected registration; a natural long background-page interval
and the updated public HTTPS route still need verification. No model layer
inference was assigned by this lifecycle fixture.

The reproducible fixture is `tests/serve-browser-recovery-integration.ts` and
its procedure is in DEVELOPMENT. The local receipt is
`runtime/native-node-lab/browser-recovery-test-20261001.json`; screenshots
include `browser-session-superseded-20261001.png`,
`browser-admission-denied-20261001.png` and
`browser-coordinator-removal-20261001.png` under that same runtime directory.
The focused mobile/backend tests passed 19 tests and browser-layer hub tests
passed two. Browser build/bundle budget, mobile typecheck, root typecheck,
structure and architecture checks passed for the functional change.

The retried 0.2.85 physical worker-loss test first exceeded its five-minute
activation budget and cleaned up. A ten-minute-budget attempt did inject
MRPUCCHI service loss after the first streamed token: 20 chunks arrived, then
an explicit `adapter_error`, while SSH and VPN stayed running. The service
restored, both workers reconnected and cleanup observed zero requested models
and stages. Its harness rejected ordering because it compared MRPUCCHI wall
time with PC-DANI wall time; the remote timestamp was about 386 ms earlier
than the already observed first token. That rejected receipt is preserved as
`runtime/native-node-lab/loss-0285-ci-longstart-20261001.json`. The harness has
been corrected to use PC-DANI monotonic timestamps for dispatch and
acknowledgement; a passing rerun is still required. This is not seamless
failover evidence.

The corrected monotonic-clock rerun subsequently passed. Its first token was
observed before the local fault-dispatch marker; acknowledgement of the stopped
MRPUCCHI service preceded local stream termination. Seventeen chunks arrived
before an explicit `adapter_error`; SSH and VPN remained running. The native
service was restored, both native workers reconnected, and cleanup reported
zero requested models and stages. The passing receipt is
`runtime/native-node-lab/loss-0285-ci-monotonic-20261001.json`. The ten-minute
activation budget was retained to exercise loss after startup; the earlier
five-minute timeout remains an onboarding obstacle. This is bounded failure
and reconnection evidence, not uninterrupted failover or a post-recovery
generation result.

A subsequent physical generation after that service restoration completed on
PC-DANI CPU [0,2) and MRPUCCHI CUDA [2,30), with generation TTFT 234 ms. The
canary measured TTFT 185.3940 ms, TPOT 115.9598 ms and 8.6237 tokens/s. The
observed process peaks were 695,369,728 bytes on PC-DANI and 1,402,093,568
bytes on MRPUCCHI. Cleanup again verified zero requested models and stages,
with both native workers connected. Receipts:
`runtime/native-node-lab/physical-repeat-0285-ci-recovered-20261001.json` and
`runtime/native-node-lab/cleanup-0285-ci-recovered-20261001.json`. This completes
the mixed-version loss, reconnection and subsequent-generation check; it does
not close the both-host fixed-release or public account-owned route gate.

The browser recovery change was committed and pushed as
`785c4d23428f947468dad6f62f4495f31402e27b` for 0.2.86, source identity
`sha256:523333b99132b8bd15d52b494759b960cb322710642db47e839d6ffa78d57607`
for the clean pinned release checkout. The initial local working-tree identity
included three ignored local configuration/bytecode inputs. Comparing every
listed input found no tracked source differences; the clean checkout identity
was verified in the image before deployment.
The exact-revision manual native workflow
[36818114425](https://github.com/tych0s/mycellios/actions/runs/36818114425)
and the bounded coordinator image build were started. The coordinator image
subsequently passed its isolated canary and was deployed at
2026-10-01T05:19:52.982146Z, retaining the previous stopped container and a
consistent SQLite backup. Public HTTPS health verified version, revision and
clean source identity, with required persistence connected and no last error.
The public browser button then validated WebGPU and connected one browser
worker; its UI reports waiting for compatible model work. This verifies public
registration, not model-layer inference. Local receipts are
`runtime/native-node-lab/deployment-0286-20261001.json` and
`runtime/native-node-lab/public-health-0286-20261001.json`, with screenshot
`runtime/native-node-lab/public-browser-connected-0286-20261001.png`.

The last native workflow observation had Windows still building; the next
query received GitHub HTTP 403 API rate-limit exceeded. Its completion remains
unverified. Native downloads were not promoted by the coordinator deployment.
Trusted installer signing, both-host exact-release proof, authenticated public
native pairing and external tester installation remain open gates.

The visible public 0.2.86 browser subsequently passed Pause/Resume: clicking
Pause left the UI paused and public health reported connected=0, mobile=0;
clicking Start again validated WebGPU, restored the connected UI, and public
health reported connected=1, mobile=1. Receipts are
`runtime/native-node-lab/public-pause-0286-20261001.json` and
`runtime/native-node-lab/public-resume-0286-20261001.json`, with screenshot
`runtime/native-node-lab/public-browser-resumed-0286-20261001.png`. This does
not prove assigned model work or natural background-retention recovery.

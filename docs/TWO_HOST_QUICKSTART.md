# Two-host quickstart

This is the shortest path to answering the current product question: can the
native Mycellios pipeline generate through two physical computers?

The checked-in configuration is a template. Replace addresses, Python paths,
memory and model source with values true for the two machines.

## 1. Prepare both machines

On both hosts, use the same commit and install dependencies:

```bash
npm ci
python -m pip install -r python/requirements-distribution.txt
npm run typecheck
```

Confirm that each host can reach the other's stage/agent ports over a private
LAN or authenticated overlay network.

## 2. Configure the route

Copy `config/auto-distribute.two-host.example.json` to an ignored local file and
set:

- the model snapshot or pinned model identifier;
- the two private IP addresses;
- truthful available memory and reserve;
- the Python executable used on each host;
- a unique authentication environment variable for the remote agent.

Keep `runtime.apiAdvertiseHost` reachable from the coordinator on host A. The
example API binds to `127.0.0.1`, so its advertise host is also `127.0.0.1`;
the return endpoint and stage addresses still use the two LAN IPs.

Run the non-mutating preflight before profiling or launching anything:

```bash
npm run preflight:two-host -- \
  --config config/auto-distribute.two-host.local.json \
  --json
```

The checked-in example must return `dryRun: true` because its addresses are
documentation placeholders. A physical candidate must use two distinct,
non-loopback machine identities and authenticated endpoints. Warnings about an
unpinned model revision, unprepared ranges or unmeasured transport do not turn
planning inputs into evidence.

Generate and inspect the route without starting anything:

```bash
npm run model:auto-distribute -- \
  --config config/auto-distribute.two-host.local.json \
  --prepare-only
```

The artifacts appear under `runtime/auto-distribute/two-host-smoke/`. Inspect
`profile.json`, `runtime-manifest.json` and `python-launch.json`. Confirm that
there are two non-empty, non-overlapping ranges assigned to different hosts.

Preflight probes the configured local `runtime.pythonExecutable` with a bounded
version check. It does not contact host B or certify its Python environment.
Verify the remote interpreter and connectivity on that machine separately.

## 3. Start the remote launch agent

Set a high-entropy token on both the coordinator shell and host B without
writing it to disk. Copy the repository and generated launch description to
the same absolute paths expected by the configuration, then start host B:

```bash
npm run dev:launch-agent -- \
  --node-id host-b \
  --host 0.0.0.0 \
  --port 9750 \
  --auth-token-env MYCELLIOS_HOST_B_TOKEN \
  --allow-launch-file runtime/auto-distribute/two-host-smoke/python-launch.json
```

Restrict port `9750` to host A at the firewall.

## 4. Launch from host A

```bash
npm run model:auto-distribute -- \
  --config config/auto-distribute.two-host.local.json
```

Success requires all stages to become ready and the generation canary to pass.
Query the configured API endpoint only after that result.

## 5. Capture the evidence

Record the commit and model identities, stage ranges, host identities,
transport mode, memory, TTFT, TPOT and throughput. Stop host B during a second
run and record whether recovery succeeds and all processes are cleaned up.

Do not call the gate complete if the route used loopback, both stages ran on
one machine, a fallback produced synthetic output or cleanup is unverified.
An accepted signed receipt also requires TTFT, TPOT, tokens/s, peak memory and
passing assertions for exact output, non-empty ranges, build/model identity and
complete cleanup.

## Known boundary

Start with the ordinary two-host pipeline. For a CUDA assignment, confirm the
installed node reports a verified GPU backend and that the assigned stage
performs a real GPU operation. Tensor-parallel GPU cells have separate physical
campaign commands and stricter collective requirements. See
`STATUS_AND_EVIDENCE.md` for measured physical results and their limits.

## Windows troubleshooting and stopping

- If the remote agent answers but the root cannot reach the stage, check TCP
  `18101` from host A and TCP `18100` from host B. A Windows Firewall **block**
  rule for `python.exe` takes precedence over a new allow rule. Inspect active
  inbound rules for the exact Python executable, then use narrow, temporary
  rules for the two hosts. Restore any pre-existing block rule after the test.
- If both stages report ready but health fetching fails, confirm that
  `runtime.apiAdvertiseHost` points to the address where the root API actually
  binds. With `runtime.apiEndpoint.host` set to `127.0.0.1`, advertise
  `127.0.0.1` to the local coordinator. Keep the return endpoint on the LAN IP.
- Keep `runtime.connectTimeoutSeconds` large enough for stage startup. Set
  `runtime.operationTimeoutSeconds` for the expected maximum active request
  time when the supervisor may stop an unreachable stage earlier. The example
  leaves it unset; the pinned SmolLM2 lab run used 20 seconds. Large or slow
  models need a longer measured value. A timed-out request fails rather than
  silently retrying.
- After a failed launch, stop and restart the remote launch agent before
  retrying. Its process records can retain a terminal state for a repeated
  launch description. Confirm ports `9750`, `18101`, `18100` and the configured
  API port have the expected listeners or are closed before each retry.
- To pause the lab, stop the coordinator with Ctrl+C and stop the remote
  launch agent. Check that no stage Python process remains and that the stage,
  return and API ports have closed. Remove temporary firewall rules and
  restore the original inbound policy. The persisted `status.json` is a
  snapshot and does not replace those live checks.

## Accompanied external tester procedure

This procedure is a draft until a released native package has been installed
and exercised by someone outside the lab. It does not require installing
Jarvis or joining the lab Tailscale network on the tester's computer.

1. The operator checks that the Windows x64 package is actually available at
   `/downloads/windows`, verifies its published provenance and records the
   release revision. If it is unavailable, stop before creating a pairing
   bundle.
2. The tester confirms Windows x64, administrator access, Internet access and
   enough free disk for the package, runtime and selected model. For the NVIDIA
   CUDA trial, reserve at least 8 GiB of free space and allow a 2.6 GB runtime
   download. Record the start time and any warning or manual action.
3. On a trusted browser, sign in to the tester's Mycellios account and create
   a one-time pairing bundle from **Use a dedicated computer**. Put that bundle
   with the verified MSI, its checksum and the USB launcher on the drive.
   Prepare the bundle last: it expires within 15 minutes. The tester then runs
   the launcher once on the target PC and accepts Windows administrator
   elevation. No account sign-in is required on that PC. Never place a
   reusable account credential in the installer or on the USB drive.
4. Confirm from the trusted browser that the node appears under **Owned nodes**,
   reaches **Ready**, and reports its detected CPU/GPU and resource limits.
   Record the verified backend and any GPU fallback reason; a CUDA trial needs
   a real GPU operation under the installed service account. Record the time
   to connection. The operator assigns a small pinned model and verifies a real
   remote layer and answer with a receipt; a connected status alone is not an
   inference result.
5. Use **Pause** and **Resume** in **Owned nodes**; verify that paused capacity
   is no longer assigned. To end the pilot, use the available uninstall action
   or ask the operator to revoke the node. Record any service, permission,
   pairing or model-download error and the exact step where it occurred.

Stop the attempt if pairing expires, package verification fails, the node
never becomes ready or the assigned model cannot produce a verified answer.
The operator should retain timings and sanitized diagnostics, never the
pairing token or account credentials.

### Preparing the first two accompanied attempts

Use the same published revision and pinned model for both attempts. Before
inviting anyone, complete the account-owned enrollment, owner Pause/Resume,
assigned inference and restart checks on the operator's computers. Confirm
the package signature independently and verify that the public download
matches the recorded release identity. Current results belong in
[STATUS_AND_EVIDENCE.md](STATUS_AND_EVIDENCE.md).

Prepare these two slots for 1–7 October 2026; neither slot represents a real
tester, an invitation or a completed installation:

| Slot | Device to seek | Person and agreed time | Acceptance evidence |
| --- | --- | --- | --- |
| 1 | Windows x64 with a supported NVIDIA GPU | Unassigned | Account-owned Ready CUDA, assigned layer, response and cleanup |
| 2 | Windows x64 using CPU only | Unassigned | Account-owned Ready CPU, assigned layer, response and cleanup |

For each attempt, record the exact installer revision/hash, pinned model
revision, device/backend, start and connection times, every manual action,
assigned layer range, generation receipt, TTFT/TPOT, process memory and final
cleanup. Verify Pause/Resume and reconnection after a planned restart. Record
unavailable measurements as unavailable. If an installation or inference
fails, save its exact step and sanitized error, then fix that blocker before
expanding invitations. The owner selects the people and contacts them; this
checklist does not authorize sending messages.

### Invitation draft (not sent)

> Hola, estamos probando Mycellios con un grupo muy pequeño. Buscamos a alguien
> con un PC Windows x64 que pueda dedicar unos 45–60 minutos a instalar un nodo
> con un pendrive y comprobar una inferencia real. Estaremos contigo durante la
> prueba, mediremos el tiempo hasta conectar y podrás pausar o desinstalar el
> nodo al terminar. Si te encaja, acordamos un momento y te explicamos los
> requisitos antes de tocar tu equipo.

Choose two or three people who can consent to the accompanied test and record
their actual names separately before sending this invitation. No external
installation or contact is counted until it happens.

### Lab USB preparation after a fixed release exists

The lab's `PREPARAR-PC.cmd` currently handles Jarvis and Tailscale only. For a
native-node pilot, the operator can place `windows-usb-node-install.cmd` and
`windows-usb-node-install.ps1` from `scripts/` beside a `Node` folder on the
USB. That folder needs exactly one released Windows x64 MSI and its matching
`<MSI filename>.sha256` file. A new PC also needs exactly one fresh
`.mycellios-enrollment` downloaded from the tester's account. Prepare that
file last; the script requires more than 10 minutes of its 15-minute lifetime
to remain. Run the PowerShell script with `-PreflightOnly` before handing over
the USB; that read-only check requests administrator elevation to inspect a
retained identity. Then run the CMD file on the target PC. It requests
administrator elevation, pairs a new node and waits for a running service, local `Ready`
health and redemption of the one-time bundle. On an already paired node, the
same file can install a higher MSI version without another pairing; a healthy
matching version exits without restarting the service. If the MSI requires a
reboot during an update, reboot and run the same USB file again. A failed
first pairing needs diagnosis and a fresh unexpired bundle.

### Private multi-PC USB bootstrap

For the operator's own Windows x64 PCs, `scripts/windows-usb-lab-prepare.ps1`
assembles a separate private package. Supply its output directory, exact MSI
path and expected SHA-256, existing Jarvis package directory, public SSH key
and the private coordinator's claims path. The controller's restricted
Tailscale OAuth credential stays DPAPI-encrypted in its Windows profile.
`-EnrollmentCount` sets the number of single-PC VPN enrollments; unused
enrollments expire after 90 days. Keep this generated package private.
Use a new output directory for a different MSI version: the `Node` folder must
contain exactly one MSI and its checksum. Preserve existing unused tickets
when refreshing the private package, using `-Resume -WithoutNewKeys` after
copying them to the new protected output. Do not issue replacement VPN keys
for an already enrolled PC.

Before travel, run its `windows-usb-lab-install.ps1 -PreflightOnly`, establish
the controller's startup task with `windows-lab-controller-startup.ps1`, and
enable controller TCP 22 access to `tag:mycellios-lab` in the tailnet policy.
Register `registerLabUsbProvisioning` from `scripts/lab-usb-provisioning.ts`
on the private coordinator, using the protected claims file and the owner's
account. Keep the coordinator reachable through the private TCP proxy on
18100. This bootstrap is not a public enrollment endpoint.
The installer compares the target's Tailscale IP with the configured controller
IP. On the controller it keeps the existing local coordinator listener; only
other PCs receive a loopback proxy through the private controller address.

Copy the complete generated folder to `MYCELLIOS` on the USB without replacing
its Windows installation files. Copy `scripts/windows-usb-lab-entry.cmd` to the
USB root as `INSTALAR-MYCELLIOS.cmd`. On each target PC, open that single file and
accept administrator elevation. The launcher allocates a separate VPN key,
detects the target IP, establishes the `mycellios-control` administrator and
requests a fresh native enrollment when required. The initial Jarvis pairing
is approved by the owner from the controller; it needs no target-side Jarvis
login. Keep the controller online during this first installation.

The elevated launcher copies the complete verified static package and exactly
one enrollment ticket to protected local storage before installing anything.
On a package upgrade, it removes obsolete regular files owned by the previous
static manifest so the old MSI cannot conflict with its replacement.
A SYSTEM task retries failures every ten minutes, up to 72 attempts, and resumes
at startup. A requested installer reboot is announced 60 seconds in advance;
after reboot no file needs to be opened again and the USB is not required.
Reboot requests are limited to two. On success the installation resume task is
removed, while the separate service recovery task remains. This does not bypass
Windows administrator consent or the owner's initial Jarvis pairing approval.

Retain the non-secret receipt and installation log under
`C:\ProgramData\MycelliosUsb`. The receipt records the real SSH account, IP
and host public key; use it to verify a new SSH host before accepting its key.
The recovery task starts before Windows login. Interactive Jarvis desktop
control still requires a Windows user session; administrative SSH is a
separate recovery route. Current test evidence and pending gates are recorded
only in [STATUS_AND_EVIDENCE.md](STATUS_AND_EVIDENCE.md).

### Accompanied native account installation

Use this sequence for a supervised operator pilot. Before inviting a third
party, check the signing and download gates in
[STATUS_AND_EVIDENCE.md](STATUS_AND_EVIDENCE.md). Do not distribute an unsigned
candidate as the public installer.

1. Record the participant, machine, Windows version, CPU/GPU, start time and
   exact package version, commit and SHA-256. Keep credentials out of the log.
2. Sign in to the participant's own Mycellios account. Open Run, expand
   **Use a dedicated computer**, and create one pairing bundle immediately
   before installation. Pairing grants ownership to that account; do not reuse
   another participant's bundle.
3. Put the bundle beside the single verified MSI and its checksum in `Node`.
   Open `windows-usb-node-install.cmd` once and accept Windows administrator
   elevation. Leave the installer running while it prepares the runtime.
4. If enrollment expires during MSI installation, record the failure. Create
   a fresh bundle in the same account, replace the expired source bundle and
   reopen the same launcher. Keep its `msi-reboot.json` checkpoint; recovery
   validates the registered MSI and continues pairing without reinstalling it.
   Do not delete the installed identity or service configuration.
5. Refresh Owned nodes. Match the new node, check the observed backend and
   connection, then test Pause and Resume on that specific node. Wait for each
   command to be **applied**; queued alone is insufficient.
6. Request the pinned pilot model through the assigned network route. Save
   output, assigned ranges, TTFT, TPOT, process memory and cleanup observations.
   A connected node alone does not demonstrate model participation.
7. Record elapsed time, every intervention and the exact stopping point. Count
   an external attempt only after the named participant actually starts it.
   Count inference participation only with a receipt identifying that machine.

For a partial installation, preserve logs and protected configuration for
diagnosis. Do not disable security controls or copy private identity material
into a troubleshooting report. Pause contribution before leaving the machine
unused; distinguish pausing work from uninstalling or revoking ownership.

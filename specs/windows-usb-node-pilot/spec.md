# Windows USB node pilot

## Purpose

Prepare a single launch file for the already authorized two-PC lab. It should
install the native Mycellios MSI and redeem a fresh one-time enrollment bundle
without asking the user to log in on MRPUCCHI. The lab's Jarvis/Tailscale USB
preparation remains separate until a verified MSI exists.

## Inputs and boundaries

- One verified Windows x64 MSI and its matching SHA-256 checksum are supplied
  in a `Node` folder beside the launcher. A first installation also requires
  exactly one `.mycellios-enrollment` file; an already paired node needs no
  new enrollment. No reusable coordinator, Tailscale, or account credential
  is stored in the package.
- The pairing bundle is issued on PC-DANI by an authenticated Mycellios account
  shortly before use; its maximum lifetime is 15 minutes.
- Preflight checks architecture, regular files and checksum. On a first install
  it also checks bundle schema and expiry. It rejects partial or inconsistent
  existing node data.
- A later USB launch may update an already paired node without a new enrollment
  only when its protected configuration, identity, installation manifest and
  Windows Installer registration form one complete installation. A partial or
  inconsistent installation remains a hard stop. The update keeps the node ID,
  coordinator binding, encrypted identity and cache, restores the wrapper
  service as LocalService, and verifies fresh Ready health after MSI replacement.
  An unchanged healthy MSI is an idempotent success, not a destructive repair.
- Elevation is requested once. The launcher installs the MSI, then either runs
  the installed pairing entrypoint or restores the retained service and
  identity, and checks the service/configuration.
- A reboot-required MSI code is reported. The same USB file can resume an
  update after reboot; no stale pairing bundle is silently reused for a first
  installation.
- The bundled CPU Python runtime must let the installed service reach Ready
  and offer CPU stage capacity. A CUDA-only attestation remains mandatory for
  dedicated GPU headless workers; an installed CPU node must not claim a
  verified GPU or fabricate a GPU physical identity.
- On a supported NVIDIA host, the installation flow first redeems the short
  lived pairing through the CPU service, then provisions the existing pinned
  CUDA pack into protected machine data. It must verify the actual GPU with a
  float16 operation before restarting the service on that cached pack. CPU
  remains the explicit fallback if download, installation, or probing fails.
  Later service starts may use a verified cached pack without downloading it.
- The node payload includes the Python `distributed_runtime` package and, on
  Windows, the executable Job Object broker required by service isolation.
  Package verification rejects either missing component before installation.
- Windows registers a pinned WinSW 2.12.0 wrapper with the Service Control
  Manager. Its XML and executable are stored together in the protected service
  directory; the child Node process runs under LocalService. A Node supervisor
  keeps WinSW's managed process alive through normal configuration reloads and
  unexpected child exits, since WinSW 2.12.0 cannot signal a clean child exit
  under this restricted account. The package includes the wrapper's MIT license
  and rejects a missing wrapper. Directly
  registering the console `node.exe` as an SCM service is invalid.
- The MSI owns an uninstall-only service-control entry for `MycelliosNode` so
  Windows Installer stops and deregisters the wrapper service even though the
  first-install pairing step registered it outside MSI. Retained configuration
  and cache are separate; a verified retained installation can restore the
  service without a second pairing.
- The staged payload includes the root `package.json` needed by the Node service.
  Windows DPAPI setup loads `System.Security` explicitly and writes encrypted
  identity data using .NET Framework-compatible atomic move/replace calls.
- The installed node answers the coordinator's physical runtime-performance
  challenge so a connected GPU/CPU node can become eligible for model planning.

## Verification

- Check clean, invalid-hash, expired-bundle, valid retained-installation and
  inconsistent partial-installation paths without modifying a PC.
- Physically test a higher-version MSI over an already paired Windows node,
  then repeat the same USB file both when healthy and when only its service is
  absent. Verify node ID, LocalService account, backend and assigned inference.
- Run the real launcher on MRPUCCHI only after a package built from a fixed
  commit and a fresh enrollment bundle are available.
- Confirm the node is online and contributes a real assigned stage, then
  verify pause, process cleanup and persistence across reboot.
- Exercise the CPU-only physical probe through native service startup and
  confirm it publishes CPU eligibility and CPU diagnostics. Keep a focused
  regression that still rejects a CPU-only probe for dedicated GPU workers.
- Confirm the actual installed Windows service starts, stops and restarts across
  reboot; layout verification alone cannot establish service readiness.
- Verify service-control XML in the built MSI and uninstall behavior against a
  separately registered disposable service before claiming that the complete
  node MSI cleans up its own service on uninstall.

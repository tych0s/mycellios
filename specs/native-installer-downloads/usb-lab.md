# Generic private Windows USB bootstrap

## Scope

Prepare Daniel's Windows x64 PCs with one launcher, without target-side
account login. Preserve Windows installation media and existing node identities.
The installer establishes unattended Tailscale and key-only administrator SSH,
installs the checked native MSI, and keeps the current Jarvis desktop runtime.

## Design

- Controller identity, public SSH key and private coordinator address are
  configuration. Never hard-code the target hostname, IP or Windows username.
- Keep the controller's own loopback coordinator listener when its Tailscale IP
  matches the configured controller IP. Create the private port proxy only on
  other PCs; a self-proxy must not reserve the controller's server port.
- Create a dedicated local administrator with a random, unstored password;
  SSH admits the controller's public key. Keep other registered administrators.
- Allocate separate single-use, pre-approved, tagged VPN keys. The OAuth
  credential remains encrypted on the controller. Unused keys expire; installed
  machine credentials remain on their machine. No private key on removable media.
- A separate per-install claim obtains a fresh, pre-confirmed Mycellios enrollment
  over the private Tailscale route. Bind each claim permanently to one machine;
  retry on that machine only. Preserve the native enrollment expiry limit.
- Copy verified installation files to protected ProgramData before use. Register
  a startup recovery task; retain non-secret receipts, errors and actual account.
- Expose one root USB launcher, `INSTALAR-MYCELLIOS.cmd`. Stage the complete
  static package and exactly one ticket before installation; continue from local
  disk at startup and retry connection failures without another file opening.
  Bound retries and reboot requests; never bypass Windows elevation consent.
- Jarvis owns pairing and the interactive desktop. Its bootstrap requests owner
  approval remotely; existing pairing is retained. Without a logged-in session,
  administrator SSH remains available. Do not configure Windows auto-login.

## Acceptance

- Reject altered payloads, another tailnet, expired/reused claims and conflicting
  administrator accounts. Test claim binding, retries and consumption.
- Exercise the installer twice on the actual MRPUCCHI, verify its retained node
  identity, service startup, CUDA diagnostics and controller SSH command.
- Test service failure recovery and a real reboot with no interactive login.
- When staging an upgrade, remove obsolete regular files listed by the previous
  static manifest, including its old MSI and checksum. Preserve files outside
  that manifest and the already assigned ticket; reject paths escaping the
  protected package or passing through a reparse point.
- Physical inference remains a separate gate. A second clean PC installation,
  ARM support, public signing and external testers cannot be inferred from this.

## Interrupted first installation

- Write downloaded enrollment JSON as UTF-8 without BOM. Verify the actual
  PowerShell writer's output with the native Node JSON parser before packaging.

- Persist verified MSI progress after a successful fresh MSI install, before
  requesting native pairing. A subsequent attempt must match its registered
  ProductCode and MSI digest, obtain a current same-machine pairing and resume
  bootstrap without replaying the MSI. Remove progress after genuine Ready.
- Resume complete native bootstrap output only after checking the exact saved
  manifest, runtime paths, worker identity and any existing protected identity.
  Stop only the owned LocalService SCM service before atomically replacing a
  pending bundle obtained using the same machine-bound claim. Preserve node ID,
  key, resource settings and caches. Restore missing service files only after
  SCM confirms absence. Reject conflicting accounts and executable paths.
- Validate SCM ownership using invariant CIM properties, not translated
  `sc.exe qc` labels. Verify the Spanish Windows route and reject invalid JSON,
  another service name or another account before stopping any service.
- A consumed enrollment after a lost response must be acknowledged only for the
  same nonce, node identity, protected key and active ownership, including after
  its expiry. Unconsumed expired bundles remain denied by the coordinator.
- Resume accelerator provisioning after pairing without another enrollment.
  Keep network, installation, integrity, disk-space and physical-probe failures
  pending for retry; allow genuine CPU-only and unsupported GPU fallback.
  Ignore prior-process health when waiting for the newly started service.
- Persist a protected, non-secret bootstrap journal before writing configuration.
  Recover missing files from that exact manifest and original node ID only after
  all surviving files match the plan. Renew the bundle using the same claim.
  Reject journal recovery if a protected identity already exists; never rebuild
  a previously running node's resource settings from bootstrap defaults.
  Remove the journal before service registration. Local recovery tests are not
  evidence of a clean physical installation. Publish these helpers only with a native MSI that
  includes the resume entrypoint and the matching coordinator behavior.

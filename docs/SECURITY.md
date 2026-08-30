# Runtime security

Mycellios executes model code and moves activations between machines. Treat a
node as an explicit trust decision, not as an anonymous shell target.

## Boundaries

- Remote coordinators use HTTPS/WSS.
- Launch agents require authentication and accept only sealed, allowlisted
  launch descriptions.
- Tokens and private keys come from environment variables or platform secret
  storage; they are never stored in example configuration.
- Artifact, runtime, hardware and route identities are bound into health and
  evidence records where the relevant protocol supports them.
- Direct transport has explicit fallback rules and anti-replay/session
  protections; ambiguous mid-stream downgrade is rejected.
- Cleanup is part of the result. A campaign is not successful while supervised
  processes remain active.

## Operating a private test

- Use a private LAN or an authenticated overlay network.
- Restrict agent, rendezvous, stage and return ports to the participating
  machines.
- Give each launch agent a different high-entropy token.
- Pin the same repository/build revision and model artifact on every host.
- Inspect generated launch descriptions before starting agents.
- Rotate test tokens after the campaign.

Report vulnerabilities using the private process in the root
[`SECURITY.md`](../SECURITY.md).

# Runtime security

Mycellios executes model code and moves activations between machines. Treat a
node as an explicit trust decision, not as an anonymous shell target.

## Boundaries

- Remote coordinators use HTTPS/WSS.
- Launch agents require authentication and accept only sealed, allowlisted
  launch descriptions.
- Tokens and private keys come from environment variables or platform secret
  storage; they are never stored in example configuration.
- `MYCELLIOS_SUPABASE_URL` is the coordinator's server-side endpoint.
  Set `MYCELLIOS_SUPABASE_PUBLIC_URL` to the HTTPS origin reachable by browsers
  when the server uses an internal address. The public auth configuration must
  expose only that browser origin and the public anon key, never the service
  role key or an internal container hostname.
- Browser OAuth returns require a one-use state stored in the initiating tab
  and expire after ten minutes. Unsolicited token fragments are removed from
  the address bar without creating a session.
- Artifact, runtime, hardware and route identities are bound into health and
  evidence records where the relevant protocol supports them.
- Direct transport has explicit fallback rules and anti-replay/session
  protections; ambiguous mid-stream downgrade is rejected.
- Cleanup is part of the result. A campaign is not successful while supervised
  processes remain active.
- Account chat sessions are checked before cancellation, usage reservation and
  dispatch, including after waiting for capacity. Idempotency keys are scoped to
  the authenticated account; retrying an active request does not cancel it.
- Retry protection for older, unscoped idempotency records is retained only
  when the stored usage associates the original job with the same account.
  Internal account keys cannot collide with accepted external keys.
- A stored conversation with usage records from different accounts fails closed
  for every account. Investigate historical ownership before repairing such
  records; the request path must never guess an owner or reassign the session.
- Windows checkpoint control binds only to loopback and requires a per-launch
  secret. Endpoint metadata must be authenticated before the launcher connects;
  the secret is passed to the child environment, never written to the descriptor
  or included in diagnostic output. This local control port is not a remote
  launch-agent API.
- Checkpoint control uses one absolute deadline from connection acceptance
  through header/payload framing and queued execution. Dribbling bytes cannot
  reset that budget or hold the serial control server indefinitely.

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

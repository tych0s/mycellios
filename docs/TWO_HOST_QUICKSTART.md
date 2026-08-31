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

The generic automatic range executor is currently the safest first CPU path.
GPU cells have separate physical campaign commands and stricter collective
requirements. Prove the ordinary two-host pipeline first; then move the same
evidence discipline to GPU execution.

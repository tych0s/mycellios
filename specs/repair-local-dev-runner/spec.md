# Repair Local Dev Runner

## Problem

Jarvis Run local failed for Mycellios with `start_command_failed`. The local dev runner must start inside this checkout and expose the landing UI through the assigned Jarvis/Cloudflare dev URL without deployment or production changes.

## Actors

- Jarvis Dev Runner, which supplies `PORT`/`HOST` and starts the configured command.
- Local developer or reviewer opening `https://dev-mycellios.nodecodex.io`.
- Mycellios landing UI, backed by a local coordinator only for proxied API routes.

## Desired Behavior

- The configured local command starts the landing Vite server on the runner-provided `PORT` and `HOST`.
- The helper coordinator starts on a separate loopback port and the UI proxy targets that port.
- The command remains self-contained and does not require real secrets for the landing dev surface.
- Stopping the runner should stop both local processes.

## Non-Goals

- Do not deploy, push, commit, or modify production infrastructure.
- Do not create fake `.env` files or placeholder secrets.
- Do not change runtime/product behavior outside local dev startup.

## Acceptance Criteria

- `.jarvis-dev.json` no longer risks binding the coordinator to the same port as the UI.
- The command honors runner-provided `PORT`/`HOST` for the UI.
- A local startup smoke test reaches the UI HTTP endpoint.
- The working tree contains only scoped local-runner/spec changes.

## Constraints and Sources

- `AGENTS.md` is absent in this checkout.
- `graphify-out/GRAPH_REPORT.md` is absent in this checkout.
- `README.md` requires Node.js 24 or later and documents coordinator defaults.
- `vite.landing.config.ts` already reads `PORT` and `HOST` for the Vite server.
- Jarvis memory search returned no prior runner-specific decision for this project.

# Plan

## Relevant Files

- `.jarvis-dev.json`: Jarvis local dev command.
- `package.json`: `dev`, `landing:dev`, and `dev:coordinator` scripts.
- `vite.landing.config.ts`: existing Vite host/port behavior.

## Approach

1. Keep the landing UI as the externally exposed dev process.
2. Allocate a helper coordinator port from `MYCELLIOS_DEV_API_PORT` with a safe default.
3. If the runner-provided UI `PORT` equals the helper API port, choose a different helper port.
4. Point `MYCELLIOS_UI_API_TARGET` at the actual helper coordinator port.
5. Install a cleanup trap before starting background work so both processes stop together.

## Decisions

- Use POSIX shell syntax in `.jarvis-dev.json` because the current command already assumes shell execution.
- Keep the coordinator on loopback; only the Vite UI needs to honor the runner `HOST`.
- Do not modify Vite config unless startup proves it is necessary; it already honors `PORT` and `HOST`.

## Rejected Alternatives

- Running only `landing:dev`: rejected because proxied API/blog routes are already configured to use a local coordinator in the current runner command.
- Hard-coding a new coordinator port without collision handling: rejected because the assigned runner `PORT` can vary.
- Loading real secrets: rejected because the landing dev surface can start with local in-memory coordinator settings.

## Verification

- Run the configured command with an explicit `PORT` that used to collide with the fixed coordinator port.
- Fetch the UI HTTP endpoint on that port.
- Confirm process cleanup and check `git status --short`.

# Tasks

- [x] Update `.jarvis-dev.json` to avoid UI/coordinator port collisions.
  Verify: run the configured command with `PORT=4175 HOST=127.0.0.1`.

- [x] Confirm the landing UI is reachable on the assigned UI port.
  Verify: fetch `http://127.0.0.1:4175/`.

- [x] Reconcile artifacts and final status.
  Verify: `git status --short` shows only scoped changes.

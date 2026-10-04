import { spawn, type ChildProcess } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { join } from "node:path";

// WinSW 2.12 cannot mark a cleanly exited child as stopped under LocalService.
// Keep its managed process alive and restart the node child after a planned
// configuration reload or an unexpected exit.
const entrypoint = join(import.meta.dirname, "main.js");
const arguments_ = [entrypoint, ...process.argv.slice(2)];
const waitAbort = new AbortController();
let stopping = false;
let child: ChildProcess | null = null;
let stopTimer: NodeJS.Timeout | null = null;
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    if (stopping) return;
    stopping = true;
    waitAbort.abort();
    if (child?.connected) child.send({ type: "mycellios-node-stop" });
    else child?.kill(signal);
    stopTimer = setTimeout(() => child?.kill("SIGTERM"), 8_000);
    stopTimer.unref();
  });
}

let consecutiveShortRuns = 0;
while (!stopping) {
  const startedAt = Date.now();
  const running = spawn(process.execPath, arguments_, {
    cwd: join(import.meta.dirname, "../.."),
    env: process.env,
    stdio: ["ignore", "inherit", "inherit", "ipc"],
    windowsHide: true,
  });
  child = running;
  const exit = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
    running.once("error", () => undefined);
    running.once("close", (code, signal) => resolve({ code, signal }));
  });
  child = null;
  if (stopTimer) clearTimeout(stopTimer);
  stopTimer = null;
  if (stopping) break;
  consecutiveShortRuns = Date.now() - startedAt > 60_000 ? 0 : consecutiveShortRuns + 1;
  process.stderr.write(`node_service_child_exited:code=${exit.code ?? "null"}:signal=${exit.signal ?? "null"}\n`);
  const waitMs = Math.min(30_000, 1_000 * 2 ** Math.min(consecutiveShortRuns - 1, 5));
  await delay(waitMs, undefined, { signal: waitAbort.signal }).catch(() => undefined);
}

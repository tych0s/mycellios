import {
  spawn,
  type ChildProcess,
  type SpawnOptions,
} from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { buildIsolatedProcessEnvironment } from "./process-environment.js";

interface TreeRootProcess {
  readonly pid?: number | undefined;
  kill(signal?: NodeJS.Signals | number): boolean;
}

/**
 * POSIX children become process-group leaders so a controlled shutdown can
 * signal the ordinary descendant tree. Windows uses taskkill /T instead.
 */
export function processTreeSpawnOptions(): Pick<SpawnOptions, "detached"> {
  return {
    detached: process.platform !== "win32",
  };
}

/**
 * Request termination of the root and its ordinary descendants.
 *
 * This is not containment: a hostile descendant can escape a POSIX process
 * group, and Windows has no kill-on-close guarantee without a Job Object.
 */
export async function terminateProcessTree(
  child: TreeRootProcess,
  force: boolean,
  helperTimeoutMs: number,
): Promise<void> {
  const pid = child.pid;
  if (!Number.isSafeInteger(pid) || (pid ?? 0) < 1) return;
  if (process.platform === "win32") {
    const invoked = await terminateWindowsTree(pid!, force, helperTimeoutMs);
    if (!invoked && force) bestEffortDirectKill(child, "SIGKILL");
    return;
  }

  try {
    process.kill(-pid!, force ? "SIGKILL" : "SIGTERM");
  } catch (error) {
    if (!isMissingProcess(error)) {
      bestEffortDirectKill(child, force ? "SIGKILL" : "SIGTERM");
    }
  }
}

async function terminateWindowsTree(
  pid: number,
  force: boolean,
  timeoutMs: number,
): Promise<boolean> {
  const systemRoot = process.env.SystemRoot ?? process.env.WINDIR;
  if (!systemRoot) return false;
  const executable = join(systemRoot, "System32", "taskkill.exe");
  if (!existsSync(executable)) return false;
  const args = ["/PID", String(pid), "/T", ...(force ? ["/F"] : [])];
  const helper = spawn(executable, args, {
    shell: false,
    windowsHide: true,
    stdio: "ignore",
    env: buildIsolatedProcessEnvironment(),
  });
  await waitForHelper(helper, timeoutMs);
  return true;
}

async function waitForHelper(
  helper: ChildProcess,
  timeoutMs: number,
): Promise<void> {
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      bestEffortDirectKill(helper, "SIGKILL");
      finish();
    }, Math.max(100, timeoutMs));
    timer.unref?.();
    helper.once("error", finish);
    helper.once("close", finish);
  });
}

function bestEffortDirectKill(
  child: TreeRootProcess,
  signal: NodeJS.Signals,
): void {
  try {
    child.kill(signal);
  } catch {
    // The process may already be gone.
  }
}

function isMissingProcess(error: unknown): boolean {
  return (
    !!error
    && typeof error === "object"
    && "code" in error
    && (error as { code?: unknown }).code === "ESRCH"
  );
}

import { createHash } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  realpathSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import {
  isAbsolute,
  join,
  relative,
  resolve,
} from "node:path";

export interface ExecutorWorkspaceLease {
  readonly path: string;
  cleanup(): void;
}

export interface CreateExecutorWorkspaceOptions {
  launchId: string;
  processId: string;
  root?: string;
}

/**
 * Allocate one private temporary write area for an exact physical process.
 *
 * The trusted runtime and model cache remain outside this directory. This is a
 * write-isolation primitive, not a filesystem sandbox.
 */
export function createExecutorWorkspace(
  options: CreateExecutorWorkspaceOptions,
): ExecutorWorkspaceLease {
  const launchId = safeIdentity(options.launchId, "executor_workspace_launch_id");
  const processId = safeIdentity(options.processId, "executor_workspace_process_id");
  const requestedRoot = resolve(
    options.root ?? join(tmpdir(), "mycellios-executors"),
  );
  mkdirSync(requestedRoot, { recursive: true, mode: 0o700 });
  const root = realpathSync(requestedRoot);
  const identity = createHash("sha256")
    .update("mycellios-executor-workspace/1\0")
    .update(launchId)
    .update("\0")
    .update(processId)
    .digest("hex")
    .slice(0, 20);
  const path = realpathSync(mkdtempSync(join(root, `exec-${identity}-`)));
  assertContainedPath(root, path);
  let cleaned = false;

  return {
    path,
    cleanup(): void {
      if (cleaned) return;
      assertContainedPath(root, path);
      rmSync(path, {
        recursive: true,
        force: true,
        maxRetries: 2,
        retryDelay: 50,
      });
      cleaned = true;
    },
  };
}

function assertContainedPath(root: string, target: string): void {
  const child = relative(root, target);
  if (
    child.length === 0
    || child === ".."
    || child.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)
    || isAbsolute(child)
  ) {
    throw new Error("executor_workspace_path_escapes_root");
  }
}

function safeIdentity(value: unknown, name: string): string {
  if (
    typeof value !== "string"
    || value.length < 1
    || value.length > 512
    || /[\u0000-\u001f\u007f]/u.test(value)
  ) {
    throw new Error(`${name}_is_invalid`);
  }
  return value;
}

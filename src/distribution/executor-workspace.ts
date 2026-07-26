import { createHash } from "node:crypto";
import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
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
  measure(maxEntries: number): ExecutorWorkspaceUsage;
  cleanup(): void;
}

export interface ExecutorWorkspaceUsage {
  bytes: number;
  entries: number;
  entryLimitExceeded: boolean;
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
    measure(maxEntries: number): ExecutorWorkspaceUsage {
      if (
        !Number.isSafeInteger(maxEntries)
        || maxEntries < 1
        || maxEntries > 1_000_000
      ) {
        throw new Error("executor_workspace_entry_limit_is_invalid");
      }
      assertContainedPath(root, path);
      return measureWorkspace(path, maxEntries);
    },
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

function measureWorkspace(
  root: string,
  maxEntries: number,
): ExecutorWorkspaceUsage {
  const directories = [root];
  let bytes = 0;
  let entries = 0;
  while (directories.length > 0) {
    const directory = directories.pop()!;
    let children;
    try {
      children = readdirSync(directory, { withFileTypes: true });
    } catch (error) {
      if (isMissingPath(error)) continue;
      throw error;
    }
    for (const child of children) {
      entries += 1;
      if (entries > maxEntries) {
        return { bytes, entries, entryLimitExceeded: true };
      }
      const childPath = join(directory, child.name);
      if (child.isDirectory()) {
        directories.push(childPath);
        continue;
      }
      try {
        // lstat deliberately does not follow symlinks outside the lease.
        bytes += lstatSync(childPath).size;
      } catch (error) {
        if (!isMissingPath(error)) throw error;
      }
      if (!Number.isSafeInteger(bytes)) {
        throw new Error("executor_workspace_usage_is_not_a_safe_integer");
      }
    }
  }
  return { bytes, entries, entryLimitExceeded: false };
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

function isMissingPath(error: unknown): boolean {
  return (
    !!error
    && typeof error === "object"
    && "code" in error
    && (
      (error as { code?: unknown }).code === "ENOENT"
      || (error as { code?: unknown }).code === "ENOTDIR"
    )
  );
}

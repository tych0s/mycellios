import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, readdir, rename } from "node:fs/promises";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import {
  nodeInstallationManifestSchema,
  nodeUninstallReceiptSchema,
  nodeUninstallRequestSchema,
  type NodeInstallationManifest,
  type NodeUninstallReceipt,
} from "../contracts/node-uninstall.js";

export interface NodeUninstallScheduleInput {
  nodeId: string;
  generation: number;
  commandId: string;
  retain: { cache: boolean; logs: boolean; configuration: boolean; identity: boolean };
}

export interface NodeUninstallLifecycleApi {
  schedule(input: NodeUninstallScheduleInput): Promise<NodeUninstallReceipt>;
  arm(requestId: string, requestDigest: string): Promise<void>;
}

export type UninstallHelperRunner = (executable: string, args: readonly string[]) => Promise<void>;

export class NodeUninstallScheduler implements NodeUninstallLifecycleApi {
  constructor(
    private readonly manifestPath: string,
    private readonly requestDirectory: string,
    private readonly runner: UninstallHelperRunner = detachedRunner,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async schedule(input: NodeUninstallScheduleInput): Promise<NodeUninstallReceipt> {
    const { manifest, bytes, digest: manifestDigest } = await loadManifest(this.manifestPath);
    if (manifest.platform !== process.platform) throw new Error("node_uninstall_manifest_platform_mismatch");
    assertInstallationLayout(manifest);
    const id = randomUUID();
    const request = nodeUninstallRequestSchema.parse({
      schema: "mycellios-node-uninstall-request/1",
      id,
      nodeId: input.nodeId,
      generation: input.generation,
      commandId: input.commandId,
      manifestPath: resolve(this.manifestPath),
      manifestDigest,
      retain: input.retain,
      createdAt: this.now().toISOString(),
    });
    const serialized = `${canonicalJson(request)}\n`;
    const requestDigest = digest(serialized);
    await mkdir(resolve(this.requestDirectory), { recursive: true, mode: 0o700 });
    const requestPath = join(resolve(this.requestDirectory), `${id}.json`);
    const temporary = `${requestPath}.tmp-${process.pid}`;
    const file = await open(temporary, "wx", 0o600);
    try { await file.writeFile(serialized, "utf8"); await file.sync(); } finally { await file.close(); }
    await rename(temporary, requestPath);
    await this.runner(manifest.nodeExecutable, [manifest.helperEntrypoint, "--request", requestPath, "--request-digest", requestDigest]);
    void bytes;
    return nodeUninstallReceiptSchema.parse({
      schema: "mycellios-node-uninstall-receipt/1",
      requestId: id,
      requestDigest,
      state: "scheduled",
      platform: manifest.platform,
      retained: Object.entries(input.retain).filter(([, retained]) => retained).map(([name]) => name),
      createdAt: this.now().toISOString(),
    });
  }

  async arm(requestId: string, requestDigest: string): Promise<void> {
    if (!/^[0-9a-f-]{36}$/.test(requestId)) throw new Error("node_uninstall_request_id_invalid");
    if (!/^sha256:[a-f0-9]{64}$/.test(requestDigest)) throw new Error("node_uninstall_request_digest_invalid");
    const path = join(resolve(this.requestDirectory), `${requestId}.armed`);
    const temporary = `${path}.tmp-${process.pid}`;
    const file = await open(temporary, "wx", 0o600);
    try { await file.writeFile(`${requestDigest}\n`, "utf8"); await file.sync(); } finally { await file.close(); }
    await rename(temporary, path);
  }

  async armAcknowledged(commandIds: ReadonlySet<string>): Promise<number> {
    let armed = 0;
    let entries: string[];
    try { entries = await readdir(resolve(this.requestDirectory)); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0; throw error; }
    for (const entry of entries.filter((name) => /^[0-9a-f-]{36}\.json$/.test(name))) {
      const path = join(resolve(this.requestDirectory), entry);
      const bytes = await readFile(path, "utf8");
      const request = nodeUninstallRequestSchema.parse(JSON.parse(bytes));
      if (!commandIds.has(request.commandId)) continue;
      try {
        await this.arm(request.id, digest(bytes));
        armed += 1;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      }
    }
    return armed;
  }
}

export async function waitForUninstallArm(input: {
  requestPath: string; requestId: string; requestDigest: string; timeoutMs?: number;
}): Promise<void> {
  const armPath = join(resolve(input.requestPath, ".."), `${input.requestId}.armed`);
  const deadline = Date.now() + (input.timeoutMs ?? 5 * 60_000);
  while (Date.now() < deadline) {
    try {
      const value = await readFile(armPath, "utf8");
      if (value === `${input.requestDigest}\n`) return;
      throw new Error("node_uninstall_arm_digest_mismatch");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
  throw new Error("node_uninstall_arm_timeout");
}

export async function verifyUninstallRequest(input: {
  requestPath: string;
  requestDigest: string;
  expectedRequestDirectory: string;
}): Promise<{ request: ReturnType<typeof nodeUninstallRequestSchema.parse>; manifest: NodeInstallationManifest }> {
  const requestPath = resolve(input.requestPath);
  const requestRoot = `${resolve(input.expectedRequestDirectory)}${process.platform === "win32" ? "\\" : "/"}`;
  if (!requestPath.startsWith(requestRoot)) throw new Error("node_uninstall_request_path_outside_root");
  const requestBytes = await readFile(requestPath, "utf8");
  if (digest(requestBytes) !== input.requestDigest) throw new Error("node_uninstall_request_digest_mismatch");
  const request = nodeUninstallRequestSchema.parse(JSON.parse(requestBytes));
  const loaded = await loadManifest(request.manifestPath);
  if (loaded.digest !== request.manifestDigest) throw new Error("node_uninstall_manifest_digest_mismatch");
  if (loaded.manifest.platform !== process.platform) throw new Error("node_uninstall_manifest_platform_mismatch");
  assertInstallationLayout(loaded.manifest);
  return { request, manifest: loaded.manifest };
}

async function loadManifest(path: string): Promise<{ manifest: NodeInstallationManifest; bytes: string; digest: `sha256:${string}` }> {
  const bytes = await readFile(resolve(path), "utf8");
  return { manifest: nodeInstallationManifestSchema.parse(JSON.parse(bytes)), bytes, digest: digest(bytes) };
}

function assertInstallationLayout(manifest: NodeInstallationManifest): void {
  const separator = process.platform === "win32" ? "\\" : "/";
  const normalize = (value: string) => process.platform === "win32" ? resolve(value).toLowerCase() : resolve(value);
  const root = `${normalize(manifest.installRoot)}${separator}`;
  for (const path of [manifest.helperEntrypoint]) {
    if (!normalize(path).startsWith(root)) throw new Error("node_uninstall_executable_outside_install_root");
  }
  const protectedRoots = new Set(["/", "C:\\", "C:/"]);
  for (const path of [manifest.installRoot, manifest.configPath, manifest.identityPath, manifest.cachePath, manifest.logsPath, manifest.statePath, manifest.receiptPath, manifest.serviceDefinitionPath]) {
    if (protectedRoots.has(resolve(path)) || resolve(path).length < 4) throw new Error("node_uninstall_target_is_unsafe");
  }
  for (const path of [manifest.configPath, manifest.identityPath, manifest.cachePath, manifest.logsPath, manifest.statePath, manifest.receiptPath, manifest.serviceDefinitionPath]) {
    if (`${normalize(path)}${separator}`.startsWith(root)) throw new Error("node_uninstall_data_path_inside_install_root");
  }
  const receipt = normalize(manifest.receiptPath);
  for (const removed of [manifest.configPath, manifest.identityPath, manifest.cachePath, manifest.logsPath, manifest.statePath, manifest.serviceDefinitionPath]) {
    const removedRoot = `${normalize(removed)}${separator}`;
    if (receipt === normalize(removed) || `${receipt}${separator}`.startsWith(removedRoot)) {
      throw new Error("node_uninstall_receipt_path_is_not_durable");
    }
  }
}

async function detachedRunner(executable: string, args: readonly string[]): Promise<void> {
  const child = spawn(executable, [...args], { detached: true, stdio: "ignore", shell: false, windowsHide: true });
  await new Promise<void>((resolvePromise, reject) => {
    child.once("error", reject);
    child.once("spawn", resolvePromise);
  });
  child.unref();
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(",")}}`;
  return JSON.stringify(value);
}

function digest(value: string): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

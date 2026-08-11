import { open, readFile, rename, rm } from "node:fs/promises";
import { dirname } from "node:path";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { nodeUninstallReceiptSchema, type NodeInstallationManifest, type NodeUninstallReceipt, type NodeUninstallRequest } from "../contracts/node-uninstall.js";

export type PlatformCommandRunner = (executable: string, args: readonly string[], allowFailure?: boolean) => Promise<void>;
export type PathRemover = (path: string) => Promise<void>;

export async function executeVerifiedUninstall(input: {
  request: NodeUninstallRequest;
  manifest: NodeInstallationManifest;
  requestDigest: string;
  run?: PlatformCommandRunner;
  remove?: PathRemover;
  now?: () => Date;
}): Promise<NodeUninstallReceipt> {
  const existing = await readReceipt(input.manifest.receiptPath);
  if (existing) {
    if (existing.requestId !== input.request.id || existing.requestDigest !== input.requestDigest) {
      throw new Error("node_uninstall_receipt_conflict");
    }
    return existing;
  }
  const run = input.run ?? runPlatformCommand;
  await disableService(input.manifest, run);
  const remove = input.remove ?? (async (path) => rm(path, { recursive: true, force: true }));
  const selected = new Set<string>([input.manifest.serviceDefinitionPath]);
  if (!input.request.retain.configuration) selected.add(input.manifest.configPath);
  if (!input.request.retain.identity) selected.add(input.manifest.identityPath);
  if (!input.request.retain.cache) selected.add(input.manifest.cachePath);
  if (!input.request.retain.logs) selected.add(input.manifest.logsPath);
  selected.add(input.manifest.statePath);
  for (const path of [...selected].sort((left, right) => right.length - left.length)) await remove(path);
  await remove(input.manifest.installRoot);
  if (input.manifest.platform === "linux") await run("systemctl", ["daemon-reload"]);
  const receipt = nodeUninstallReceiptSchema.parse({
    schema: "mycellios-node-uninstall-receipt/1",
    requestId: input.request.id,
    requestDigest: input.requestDigest,
    state: "completed",
    platform: input.manifest.platform,
    retained: Object.entries(input.request.retain).filter(([, retained]) => retained).map(([name]) => name),
    createdAt: (input.now ?? (() => new Date()))().toISOString(),
  });
  await writeReceipt(input.manifest.receiptPath, receipt);
  return receipt;
}

async function disableService(manifest: NodeInstallationManifest, run: PlatformCommandRunner): Promise<void> {
  if (manifest.platform === "linux") {
    await run("systemctl", ["disable", "--now", manifest.serviceName], true);
    return;
  }
  if (manifest.platform === "darwin") {
    await run("launchctl", ["bootout", `system/${manifest.serviceName}`], true);
    return;
  }
  await run("sc.exe", ["stop", manifest.serviceName], true);
  await run("sc.exe", ["delete", manifest.serviceName], true);
}

async function runPlatformCommand(executable: string, args: readonly string[], allowFailure = false): Promise<void> {
  const child = spawn(executable, [...args], { shell: false, windowsHide: true, stdio: "ignore" });
  const code = await new Promise<number>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (status) => resolve(status ?? -1));
  });
  if (code !== 0 && !allowFailure) throw new Error(`node_uninstall_service_command_failed:${executable}:${code}`);
}

async function readReceipt(path: string): Promise<NodeUninstallReceipt | null> {
  try { return nodeUninstallReceiptSchema.parse(JSON.parse(await readFile(path, "utf8"))); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

async function writeReceipt(path: string, receipt: NodeUninstallReceipt): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const temporary = `${path}.tmp-${process.pid}`;
  const file = await open(temporary, "wx", 0o600);
  try { await file.writeFile(`${JSON.stringify(receipt, null, 2)}\n`, "utf8"); await file.sync(); } finally { await file.close(); }
  await rename(temporary, path);
}

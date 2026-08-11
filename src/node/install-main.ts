import { readFile } from "node:fs/promises";
import { bootstrapNodeInstallation } from "./installation-bootstrap.js";
import { NodeConfigurationStore } from "./config-store.js";
import { nodeInstallationManifestSchema } from "../contracts/node-uninstall.js";
import { registerNativeNodeService } from "./service-registration.js";
import { defaultNodeInstallationManifest } from "./default-installation.js";
import { resolve } from "node:path";
import { rm } from "node:fs/promises";

const enrollmentPath = argument("--enrollment");
if (!enrollmentPath) throw new Error("usage: install-main --enrollment <path> [--manifest <path>]");
const manifestPath = argument("--manifest");
const manifest = manifestPath
  ? nodeInstallationManifestSchema.parse(JSON.parse(await readFile(manifestPath, "utf8")) as unknown)
  : defaultNodeInstallationManifest({ platform: supportedPlatform(), installRoot: argument("--install-root") ?? resolve(import.meta.dirname, "../..") });
const result = await bootstrapNodeInstallation({ manifest, enrollmentSourcePath: enrollmentPath });
const config = (await new NodeConfigurationStore(result.configPath).load()).config;
await registerNativeNodeService({ manifest, config });
if (resolve(enrollmentPath) !== resolve(result.enrollmentPath)) await rm(resolve(enrollmentPath), { force: true });
process.stdout.write(`${JSON.stringify({ status: "installed", nodeId: result.nodeId })}\n`);

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function supportedPlatform(): "linux" | "darwin" | "win32" {
  if (process.platform === "linux" || process.platform === "darwin" || process.platform === "win32") return process.platform;
  throw new Error(`node_install_platform_is_unsupported:${process.platform}`);
}

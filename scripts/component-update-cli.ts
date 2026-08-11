import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import type {
  ComponentUpdateChannel,
  ComponentUpdateManifest,
} from "../src/contracts/component-update-manifest.js";
import {
  createDevelopmentSigningIdentity,
  prepareComponentRelease,
  publishComponentRelease,
  readComponentUpdateEnrollment,
  readPreparedComponentRelease,
  shipComponentRelease,
  type ComponentUpdateTarget,
  type PrepareComponentReleaseOptions,
  type SupportedComponentArchitecture,
  type SupportedComponentPlatform,
} from "../src/update/component-update-publisher.js";

const workspace = resolve(import.meta.dirname, "..");
const localRoot = resolve(workspace, ".codex-runtime", "component-updates");

const [command = "help", ...argumentTokens] = process.argv.slice(2);
const args = parseArguments(argumentTokens);

switch (command) {
  case "keygen": {
    const channel = readChannel(args);
    if (channel !== "dev") {
      throw new Error(
        "Stable signing keys require the offline production key ceremony; this development CLI only creates dev keys.",
      );
    }
    print(await createDevelopmentSigningIdentity({
      localRoot,
      keyId: option(args, "key-id"),
      outputDirectory: option(args, "output"),
      persistToFiles: true,
    }));
    break;
  }
  case "prepare":
    print(await prepareComponentRelease(prepareOptions(args)));
    break;
  case "publish": {
    const releaseDirectory = resolve(requiredOption(args, "release"));
    const release = await readPreparedComponentRelease(releaseDirectory);
    print(await publishComponentRelease({
      releaseDirectory,
      coordinatorUrl:
        option(args, "coordinator")
        ?? process.env.MYCELLIOS_COMPONENT_UPDATE_FEED_URL
        ?? await manifestCoordinator(releaseDirectory, release.manifestFile),
      target: release.target,
      ...authorizationOptions(args),
    }));
    break;
  }
  case "ship": {
    const prepared = prepareOptions(args);
    print(await shipComponentRelease({
      ...prepared,
      coordinatorUrl:
        option(args, "coordinator")
        ?? process.env.MYCELLIOS_COMPONENT_UPDATE_FEED_URL
        ?? prepared.feedUrl,
      ...authorizationOptions(args),
    }));
    break;
  }
  case "coordinator":
    await startDevelopmentCoordinator(args);
    break;
  case "help":
  case "--help":
  case "-h":
    printHelp();
    break;
  default:
    throw new Error(`Unknown component update command: ${command}`);
}

function prepareOptions(
  options: ReadonlyMap<string, string>,
): PrepareComponentReleaseOptions {
  const channel = readChannel(options);
  const keyDirectory = resolve(
    option(options, "key-directory") ?? join(localRoot, "keys", channel),
  );
  return {
    workspace,
    localRoot,
    feedUrl:
      option(options, "feed")
      ?? option(options, "coordinator")
      ?? process.env.MYCELLIOS_COMPONENT_UPDATE_FEED_URL
      ?? "http://127.0.0.1:8787",
    target: {
      channel,
      platform: readPlatform(options),
      arch: readArchitecture(options),
    },
    processEnvironment: { ...process.env },
    keyDirectory,
    ...(option(options, "private-key-file")
      ? { privateKeyFile: option(options, "private-key-file") }
      : {}),
    ...(option(options, "public-key-file")
      ? { publicKeyFile: option(options, "public-key-file") }
      : {}),
    ...(option(options, "key-id")
      ? { keyId: option(options, "key-id") }
      : {}),
    ...(option(options, "output")
      ? { outputDirectory: option(options, "output") }
      : {}),
    ...(option(options, "sequence")
      ? {
        sequence: positiveSafeInteger(
          option(options, "sequence")!,
          "sequence",
        ),
      }
      : {}),
    ...(option(options, "revision")
      ? { revision: option(options, "revision") }
      : {}),
  };
}

function authorizationOptions(
  options: ReadonlyMap<string, string>,
): {
  adminToken?: string;
  adminTokenFile?: string;
} {
  const adminTokenFile =
    option(options, "admin-token-file")
    ?? process.env.MYCELLIOS_MODEL_ADMIN_TOKEN_FILE;
  if (adminTokenFile) return { adminTokenFile };
  const adminToken = process.env.MYCELLIOS_MODEL_ADMIN_TOKEN?.trim();
  return adminToken ? { adminToken } : {};
}

async function manifestCoordinator(
  releaseDirectory: string,
  manifestFile: string,
): Promise<string> {
  const manifest = JSON.parse(
    await readFile(resolve(releaseDirectory, manifestFile), "utf8"),
  ) as ComponentUpdateManifest;
  const artifactUrl = manifest.components[0]?.artifact.url;
  if (!artifactUrl) {
    throw new Error("Prepared release does not contain an artifact URL.");
  }
  return new URL(artifactUrl).origin;
}

async function startDevelopmentCoordinator(
  options: ReadonlyMap<string, string>,
): Promise<void> {
  const channel = readChannel(options);
  if (channel !== "dev") {
    throw new Error(
      "The local source coordinator helper only accepts the dev channel.",
    );
  }
  const keyDirectory = resolve(
    option(options, "key-directory") ?? join(localRoot, "keys", channel),
  );
  const publicKey = await readComponentUpdateEnrollment(
    resolve(
      option(options, "public-key-file")
        ?? join(keyDirectory, "public.json"),
    ),
    channel,
  );
  process.env.MYCELLIOS_COMPONENT_UPDATES_DIST = resolve(
    option(options, "store") ?? join(localRoot, "store"),
  );
  process.env.GPU_MESH_HOST = "127.0.0.1";
  const port = option(options, "port")
    ? positiveSafeInteger(option(options, "port")!, "port")
    : 8_787;
  if (port > 65_535) throw new Error("port must be at most 65535.");
  process.env.GPU_MESH_PORT = String(port);
  process.env.GPU_MESH_DB = ":memory:";
  process.env.MYCELLIOS_COMPONENT_UPDATE_DEV_KEY_ID = publicKey.keyId;
  process.env.MYCELLIOS_COMPONENT_UPDATE_DEV_PUBLIC_KEY = publicKey.spki;
  await import("../src/coordinator/main.js");
}

function readChannel(
  options: ReadonlyMap<string, string>,
): ComponentUpdateChannel {
  const channel =
    option(options, "channel")
    ?? process.env.MYCELLIOS_COMPONENT_UPDATE_CHANNEL
    ?? "dev";
  if (channel !== "dev" && channel !== "stable") {
    throw new Error("Component channel must be dev or stable.");
  }
  return channel;
}

function readPlatform(
  options: ReadonlyMap<string, string>,
): SupportedComponentPlatform {
  const platform = option(options, "platform") ?? process.platform;
  if (!["win32", "linux", "darwin"].includes(platform)) {
    throw new Error("Component platform must be win32, linux or darwin.");
  }
  return platform as SupportedComponentPlatform;
}

function readArchitecture(
  options: ReadonlyMap<string, string>,
): SupportedComponentArchitecture {
  const architecture = option(options, "arch") ?? process.arch;
  if (!["x64", "arm64"].includes(architecture)) {
    throw new Error("Component architecture must be x64 or arm64.");
  }
  return architecture as SupportedComponentArchitecture;
}

function parseArguments(tokens: readonly string[]): Map<string, string> {
  const output = new Map<string, string>();
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (!token.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
    const equals = token.indexOf("=");
    const key = token.slice(2, equals < 0 ? undefined : equals);
    const value =
      equals >= 0 ? token.slice(equals + 1) : tokens[++index];
    if (!key || value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for --${key || "option"}.`);
    }
    if (output.has(key)) throw new Error(`Duplicate option: --${key}.`);
    output.set(key, value);
  }
  return output;
}

function option(
  options: ReadonlyMap<string, string>,
  name: string,
): string | undefined {
  const value = options.get(name)?.trim();
  return value || undefined;
}

function requiredOption(
  options: ReadonlyMap<string, string>,
  name: string,
): string {
  const value = option(options, name);
  if (!value) throw new Error(`Missing required option --${name}.`);
  return value;
}

function positiveSafeInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return parsed;
}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function printHelp(): void {
  process.stdout.write(
    [
      "Mycellios signed component updates",
      "",
      "  npm run dev:update:keygen",
      "  npm run dev:update:coordinator",
      "  npm run dev:update:ship -- --coordinator http://127.0.0.1:8787",
      "",
      "Remote ship uses the same command with an HTTPS coordinator and",
      "MYCELLIOS_MODEL_ADMIN_TOKEN_FILE. Enroll enrollment.json once in the",
      "desktop Developer settings; the dev agent then polls every 20 seconds.",
      "",
    ].join("\n"),
  );
}

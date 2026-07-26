import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { verifyWanPilotSnapshot } from "./wan-pilot-gate.js";

try {
  const argumentsMap = parseArguments(process.argv.slice(2));
  const source = required(argumentsMap, "--snapshot");
  const model = required(argumentsMap, "--model");
  const minimumPhysicalNodes = optionalInteger(argumentsMap, "--min-physical-nodes", 2);
  const minimumGpuCloudNodes = optionalInteger(argumentsMap, "--min-gpu_cloud-nodes", 2);
  const snapshot = await loadSnapshot(source);
  const report = verifyWanPilotSnapshot(snapshot, {
    model,
    minimumPhysicalNodes,
    minimumGpuCloudNodes,
  });
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} catch (error) {
  const failure = {
    schema: "gdlp-wan-pilot-gate-failure/1",
    passed: false,
    error: error instanceof Error ? error.message : String(error),
  };
  process.stderr.write(`${JSON.stringify(failure, null, 2)}\n`);
  process.exitCode = 1;
}

async function loadSnapshot(source: string): Promise<unknown> {
  if (/^https:\/\//i.test(source)) {
    const response = await fetch(source, {
      redirect: "error",
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) throw new Error(`wan_pilot_snapshot_http_${response.status}`);
    const contentLength = Number(response.headers.get("content-length") ?? 0);
    if (contentLength > 8 * 1024 * 1024) throw new Error("wan_pilot_snapshot_is_too_large");
    const text = await response.text();
    if (Buffer.byteLength(text, "utf8") > 8 * 1024 * 1024) {
      throw new Error("wan_pilot_snapshot_is_too_large");
    }
    return JSON.parse(text) as unknown;
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(source)) {
    throw new Error("wan_pilot_snapshot_url_must_use_https");
  }
  return JSON.parse(await readFile(resolve(source), "utf8")) as unknown;
}

function parseArguments(args: string[]): Map<string, string> {
  const parsed = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    const value = args[index + 1];
    if (!flag?.startsWith("--") || value === undefined || value.startsWith("--")) {
      throw new Error(`wan_pilot_cli_argument_is_invalid:${flag ?? "missing"}`);
    }
    if (parsed.has(flag)) throw new Error(`wan_pilot_cli_argument_is_duplicate:${flag}`);
    parsed.set(flag, value);
  }
  return parsed;
}

function required(values: Map<string, string>, name: string): string {
  const value = values.get(name)?.trim();
  if (!value) throw new Error(`wan_pilot_cli_requires:${name}`);
  return value;
}

function optionalInteger(values: Map<string, string>, name: string, fallback: number): number {
  const raw = values.get(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) throw new Error(`wan_pilot_cli_integer_is_invalid:${name}`);
  return value;
}

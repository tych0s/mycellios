import { readFile } from "node:fs/promises";
import { delimiter, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { LocalProcessAgent } from "./launch-supervisor.js";
import { LaunchAgentRpcServer } from "./launch-agent-rpc.js";
import { PythonPhysicalProbe } from "./physical-probe.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";

export interface LaunchAgentDaemonCliOptions {
  host: string;
  port: number;
  nodeId: string;
  authTokenEnv: string;
  allowLaunchFile?: string;
  physicalProbePython: string;
  physicalProbeTimeoutMs: number;
  cwd?: string;
  maxOutputBytes: number;
  startTimeoutMs: number;
  stopTimeoutMs: number;
  stopGraceMs: number;
  maxProcesses: number;
}

async function main(): Promise<void> {
  const options = parseLaunchAgentDaemonArguments(process.argv.slice(2));
  const authToken = readLaunchAgentDaemonAuthToken(
    options.authTokenEnv,
    process.env,
  );
  const allowedLaunch =
    options.allowLaunchFile === undefined
      ? undefined
      : await loadAllowedPythonLaunchDescription(options.allowLaunchFile);
  const physicalProbe = new PythonPhysicalProbe({
    pythonExecutable: options.physicalProbePython,
    ...(options.cwd === undefined ? {} : { cwd: options.cwd }),
    env: {
      PYTHONPATH: [
        resolve(options.cwd ?? process.cwd(), "python"),
        process.env.PYTHONPATH,
      ]
        .filter((value): value is string => value !== undefined && value.length > 0)
        .join(delimiter),
    },
    timeoutMs: options.physicalProbeTimeoutMs,
  });
  const local = new LocalProcessAgent({
    id: `local-process:${options.nodeId}`,
    ...(options.cwd === undefined ? {} : { cwd: options.cwd }),
    allowedExecutables: [
      allowedLaunch?.configuration.pythonExecutable ??
        options.physicalProbePython,
    ],
    maxOutputBytesPerStream: options.maxOutputBytes,
    stopGraceMs: options.stopGraceMs,
  });
  const daemon = new LaunchAgentRpcServer({
    agent: local,
    nodeId: options.nodeId,
    ...(authToken === undefined ? {} : { authToken }),
    ...(allowedLaunch === undefined
      ? {}
      : { allowedLaunchDescriptions: [allowedLaunch] }),
    physicalProbe,
    physicalProbeTimeoutMs: options.physicalProbeTimeoutMs,
    maxOutputBytesPerStream: options.maxOutputBytes,
    startTimeoutMs: options.startTimeoutMs,
    stopTimeoutMs: options.stopTimeoutMs,
    maxProcesses: options.maxProcesses,
  });
  const address = await daemon.listen(options.port, options.host);
  process.stdout.write(
    `${JSON.stringify({
      schema: "gdlp-launch-agent-daemon/1",
      event: "ready",
      nodeId: options.nodeId,
      agentId: local.id,
      host: address.host,
      port: address.port,
      url: address.url,
    })}\n`,
  );

  let closing: Promise<void> | null = null;
  const close = (signal: NodeJS.Signals) => {
    closing ??= daemon.close(`daemon_${signal.toLowerCase()}`);
    void closing.then(
      () => process.exit(0),
      (error) => {
        process.stderr.write(`${renderError(error)}\n`);
        process.exit(1);
      },
    );
  };
  process.once("SIGINT", close);
  process.once("SIGTERM", close);
}

export function parseLaunchAgentDaemonArguments(
  argumentsValue: string[],
): LaunchAgentDaemonCliOptions {
  const values = new Map<string, string>();
  const allowed = new Set([
    "--host",
    "--port",
    "--node-id",
    "--auth-token-env",
    "--allow-launch-file",
    "--physical-probe-python",
    "--physical-probe-timeout-ms",
    "--cwd",
    "--max-output-bytes",
    "--start-timeout-ms",
    "--stop-timeout-ms",
    "--stop-grace-ms",
    "--max-processes",
  ]);
  for (let index = 0; index < argumentsValue.length; index += 2) {
    const name = argumentsValue[index];
    const value = argumentsValue[index + 1];
    if (!name || !allowed.has(name) || value === undefined || values.has(name)) {
      throw new Error(`launch_agent_daemon_argument_is_invalid:${name ?? "missing"}`);
    }
    values.set(name, value);
  }
  const nodeId = values.get("--node-id");
  if (!nodeId) throw new Error("launch_agent_daemon_node_id_is_required");
  const host = values.get("--host") ?? "127.0.0.1";
  if (!host.trim() || /[\0\r\n]/.test(host)) {
    throw new Error("launch_agent_daemon_host_is_invalid");
  }
  const cwd = values.get("--cwd");
  if (cwd !== undefined && (!cwd.trim() || /[\0\r\n]/.test(cwd))) {
    throw new Error("launch_agent_daemon_cwd_is_invalid");
  }
  const authTokenEnv =
    values.get("--auth-token-env") ?? "GDLP_LAUNCH_AGENT_TOKEN";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(authTokenEnv)) {
    throw new Error("launch_agent_daemon_auth_token_env_is_invalid");
  }
  const allowLaunchFile = values.get("--allow-launch-file");
  if (
    allowLaunchFile !== undefined &&
    (!allowLaunchFile.trim() || /[\0\r\n]/.test(allowLaunchFile))
  ) {
    throw new Error("launch_agent_daemon_allow_launch_file_is_invalid");
  }
  const physicalProbePython =
    values.get("--physical-probe-python") ?? "python";
  if (!physicalProbePython.trim() || /[\0\r\n]/.test(physicalProbePython)) {
    throw new Error("launch_agent_daemon_physical_probe_python_is_invalid");
  }
  return {
    host,
    port: integerOption(values, "--port", 9_750, 0, 65_535),
    nodeId,
    authTokenEnv,
    ...(allowLaunchFile === undefined ? {} : { allowLaunchFile }),
    physicalProbePython,
    physicalProbeTimeoutMs: integerOption(
      values,
      "--physical-probe-timeout-ms",
      30_000,
      1,
      300_000,
    ),
    ...(cwd === undefined ? {} : { cwd }),
    maxOutputBytes: integerOption(
      values,
      "--max-output-bytes",
      64 * 1024,
      1_024,
      16 * 1024 * 1024,
    ),
    startTimeoutMs: integerOption(
      values,
      "--start-timeout-ms",
      30_000,
      1,
      300_000,
    ),
    stopTimeoutMs: integerOption(
      values,
      "--stop-timeout-ms",
      15_000,
      1,
      300_000,
    ),
    stopGraceMs: integerOption(
      values,
      "--stop-grace-ms",
      5_000,
      1,
      300_000,
    ),
    maxProcesses: integerOption(
      values,
      "--max-processes",
      1_024,
      1,
      1_000_000,
    ),
  };
}

export function readLaunchAgentDaemonAuthToken(
  name: string,
  environment: NodeJS.ProcessEnv,
): string | undefined {
  const value = environment[name];
  return value === undefined || value.length === 0 ? undefined : value;
}

export async function loadAllowedPythonLaunchDescription(
  pathValue: string,
): Promise<PythonPipelineLaunchDescription> {
  if (typeof pathValue !== "string" || !pathValue.trim() || /[\0\r\n]/.test(pathValue)) {
    throw new Error("launch_agent_daemon_allow_launch_file_is_invalid");
  }
  const source = await readFile(resolve(pathValue), "utf8");
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    throw new Error("launch_agent_daemon_allow_launch_file_is_not_json");
  }
  validatePythonLaunchDescription(value);
  return structuredClone(value);
}

function integerOption(
  values: Map<string, string>,
  name: string,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const raw = values.get(name);
  if (raw === undefined) return fallback;
  if (!/^(0|[1-9][0-9]*)$/.test(raw)) {
    throw new Error(`launch_agent_daemon_integer_is_invalid:${name}`);
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`launch_agent_daemon_integer_is_invalid:${name}`);
  }
  return value;
}

function renderError(value: unknown): string {
  const error = value instanceof Error ? value : new Error(String(value));
  return JSON.stringify({
    schema: "gdlp-launch-agent-daemon/1",
    event: "error",
    error: error.message,
  });
}

const entryPath = process.argv[1];
if (
  entryPath !== undefined &&
  resolve(fileURLToPath(import.meta.url)) === resolve(entryPath)
) {
  void main().catch((error) => {
    process.stderr.write(`${renderError(error)}\n`);
    process.exitCode = 1;
  });
}

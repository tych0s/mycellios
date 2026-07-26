import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  HttpLaunchAgent,
  type HttpLaunchAgentOptions,
} from "./launch-agent-rpc.js";
import {
  extractPhysicalGpuRankWork,
  parsePhysicalGpuCampaignCliConfig,
  validateRankWorkAgainstCampaign,
  type PhysicalGpuCampaignCliRemoteAgent,
} from "./physical-gpu-campaign-cli.js";
import {
  buildPhysicalGpuCampaignGateReport,
  type PhysicalGpuCampaignReportHostBinding,
  type PhysicalGpuCampaignReportInput,
} from "./physical-gpu-campaign-report.js";
import {
  runPhysicalGpuConveyorAb,
  validatePhysicalGpuConveyorLaunchPair,
  type PhysicalGpuConveyorAbDependencies,
  type PhysicalGpuConveyorAbReport,
} from "./physical-gpu-conveyor-ab.js";
import { validatePhysicalProbe } from "./physical-probe.js";

export interface PhysicalGpuConveyorAbCliArguments {
  baselineLaunchPath: string;
  conveyorLaunchPath: string;
  configPath: string;
  outputPath: string;
  rounds: number;
}

export interface PhysicalGpuConveyorAbCliDependencies {
  cwd?: string;
  environment?: NodeJS.ProcessEnv;
  readText?: (absolutePath: string) => Promise<string>;
  pathExists?: (absolutePath: string) => Promise<boolean>;
  writeJsonExclusive?: (absolutePath: string, value: unknown) => Promise<void>;
  createAgent?: (
    options: HttpLaunchAgentOptions,
  ) => PhysicalGpuCampaignCliRemoteAgent;
  runAb?: typeof runPhysicalGpuConveyorAb;
  abDependencies?: Omit<PhysicalGpuConveyorAbDependencies, "physicalGate">;
  writeStderr?: (message: string) => void;
}

export function parsePhysicalGpuConveyorAbCliArguments(
  argv: string[],
): PhysicalGpuConveyorAbCliArguments {
  const allowed = new Set([
    "--baseline-launch",
    "--conveyor-launch",
    "--config",
    "--output",
    "--rounds",
  ]);
  const values = new Map<string, string>();
  if (argv.length % 2 !== 0) {
    throw new Error("physical_gpu_conveyor_ab_cli_arguments_are_invalid");
  }
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (
      name === undefined
      || value === undefined
      || !allowed.has(name)
      || values.has(name)
      || value.length === 0
      || /[\0\r\n]/.test(value)
    ) {
      throw new Error(
        `physical_gpu_conveyor_ab_cli_argument_is_invalid:${name ?? "missing"}`,
      );
    }
    values.set(name, value);
  }
  const baselineLaunchPath = requiredFlag(values, "--baseline-launch");
  const conveyorLaunchPath = requiredFlag(values, "--conveyor-launch");
  const configPath = requiredFlag(values, "--config");
  const outputPath = requiredFlag(values, "--output");
  const roundsText = values.get("--rounds") ?? "1";
  if (!/^[1-9][0-9]{0,2}$/.test(roundsText)) {
    throw new Error("physical_gpu_conveyor_ab_cli_rounds_is_invalid");
  }
  const rounds = Number(roundsText);
  if (rounds > 100) {
    throw new Error("physical_gpu_conveyor_ab_cli_rounds_is_invalid");
  }
  return {
    baselineLaunchPath,
    conveyorLaunchPath,
    configPath,
    outputPath,
    rounds,
  };
}

export async function executePhysicalGpuConveyorAbCli(
  argv: string[],
  dependencies: PhysicalGpuConveyorAbCliDependencies = {},
): Promise<number> {
  const writeStderr =
    dependencies.writeStderr ?? ((message: string) => process.stderr.write(message));
  let args: PhysicalGpuConveyorAbCliArguments;
  try {
    args = parsePhysicalGpuConveyorAbCliArguments(argv);
  } catch (error) {
    writeStderr(`${normalizeError(error).message}\n`);
    return 1;
  }

  const cwd = resolve(dependencies.cwd ?? process.cwd());
  const paths = {
    baseline: resolve(cwd, args.baselineLaunchPath),
    conveyor: resolve(cwd, args.conveyorLaunchPath),
    config: resolve(cwd, args.configPath),
    output: resolve(cwd, args.outputPath),
  };
  if (new Set(Object.values(paths)).size !== 4) {
    writeStderr("physical_gpu_conveyor_ab_cli_all_paths_must_differ\n");
    return 1;
  }
  const pathExists = dependencies.pathExists ?? fileExists;
  if (await pathExists(paths.output)) {
    writeStderr("physical_gpu_conveyor_ab_cli_output_already_exists\n");
    return 1;
  }

  const readText = dependencies.readText ?? ((path) => readFile(path, "utf8"));
  const writeJsonExclusive = dependencies.writeJsonExclusive ?? exclusiveWriteJson;
  const createAgent =
    dependencies.createAgent
    ?? ((options: HttpLaunchAgentOptions) =>
      new HttpLaunchAgent(options) as PhysicalGpuCampaignCliRemoteAgent);
  const runAb = dependencies.runAb ?? runPhysicalGpuConveyorAb;
  const environment = dependencies.environment ?? process.env;
  const secrets: string[] = [];

  try {
    const [baselineSource, conveyorSource, configSource] = await Promise.all([
      readText(paths.baseline),
      readText(paths.conveyor),
      readText(paths.config),
    ]);
    const pair = validatePhysicalGpuConveyorLaunchPair(
      parseJson(baselineSource, "baseline_launch"),
      parseJson(conveyorSource, "conveyor_launch"),
    );
    const config = parsePhysicalGpuCampaignCliConfig(
      parseJson(configSource, "config"),
    );

    const agents = config.agents.map((agentConfig) => {
      const token = environment[agentConfig.authTokenEnv];
      if (token === undefined || token.length === 0) {
        throw new Error(
          `physical_gpu_conveyor_ab_cli_auth_token_env_is_missing:${agentConfig.authTokenEnv}`,
        );
      }
      secrets.push(token);
      return createAgent({
        endpoint: agentConfig.endpoint,
        id: `local-process:${agentConfig.nodeId}`,
        authToken: token,
      });
    });
    const agentByNode = new Map(
      config.agents.map((agent, index) => [
        agent.nodeId,
        { config: agent, agent: agents[index]! },
      ]),
    );
    const probes = await Promise.all(
      config.hosts.map(async (host) => {
        const binding = agentByNode.get(host.agentNodeId);
        if (binding === undefined) {
          throw new Error("physical_gpu_conveyor_ab_cli_host_agent_is_missing");
        }
        const probe = await binding.agent.physicalEvidence(host.nonce);
        validatePhysicalProbe(probe, host.nonce);
        return probe;
      }),
    );
    const hosts = config.hosts.map(
      (host, index): PhysicalGpuCampaignReportHostBinding => {
        const binding = agentByNode.get(host.agentNodeId);
        if (binding === undefined) {
          throw new Error("physical_gpu_conveyor_ab_cli_host_agent_is_missing");
        }
        return {
          hostId: host.hostId,
          agentId: binding.agent.id,
          agentEndpoint: binding.config.endpoint,
          rankNodeId: host.rankNodeId,
          device: host.device,
          offeredVramBytes: host.offeredVramBytes,
          vendor: host.vendor,
          expectedProbeNonce: host.nonce,
          probe: structuredClone(probes[index]!),
        };
      },
    );

    const report = await runAb(
      {
        baselineLaunch: pair.baseline,
        conveyorLaunch: pair.conveyor,
        campaign: {
          agents: config.agents.map((agentConfig, index) => ({
            nodeId: agentConfig.nodeId,
            agent: agents[index]!,
          })),
          apiBaseUrl: config.apiBaseUrl,
          canaries: structuredClone(config.canaries),
          warmups: config.warmups,
          iterations: config.iterations,
          concurrencies: [...config.concurrencies],
        },
        rounds: args.rounds,
      },
      {
        ...dependencies.abDependencies,
        physicalGate: async (_arm, launch, campaign) => {
          const rankWork = extractPhysicalGpuRankWork(
            launch,
            campaign.lifecycle.supervisorStopped,
            hosts,
          );
          validateRankWorkAgainstCampaign(rankWork, campaign);
          const input: PhysicalGpuCampaignReportInput = {
            capturedAt: new Date().toISOString(),
            networkScope: config.networkScope,
            samplesTruncated: false,
            campaign,
            launch,
            hosts,
            networkLinks: structuredClone(config.networkLinks),
            reference: structuredClone(config.reference),
            rankWork,
          };
          const gateReport = buildPhysicalGpuCampaignGateReport(input);
          return {
            passed: gateReport.gate.passed,
            reportSealSha256: gateReport.seal.digest,
            report: gateReport,
          };
        },
      },
    );
    await writeJsonExclusive(paths.output, report);
    if (!report.passed) {
      writeStderr(`physical_gpu_conveyor_ab_failed:${report.failures.join(",")}\n`);
      return 1;
    }
    return 0;
  } catch (error) {
    writeStderr(`${redactSecrets(normalizeError(error).message, secrets)}\n`);
    return 1;
  }
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await readFile(path);
    return true;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return false;
    throw error;
  }
}

async function exclusiveWriteJson(
  path: string,
  value: PhysicalGpuConveyorAbReport,
): Promise<void> {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
}

function requiredFlag(values: ReadonlyMap<string, string>, name: string): string {
  const value = values.get(name);
  if (value === undefined) {
    throw new Error(`physical_gpu_conveyor_ab_cli_${name.slice(2).replaceAll("-", "_")}_is_required`);
  }
  return value;
}

function parseJson(source: string, name: string): unknown {
  try {
    return JSON.parse(source) as unknown;
  } catch {
    throw new Error(`physical_gpu_conveyor_ab_cli_${name}_json_is_invalid`);
  }
}

function redactSecrets(message: string, secrets: readonly string[]): string {
  let result = message;
  for (const secret of [...new Set(secrets)].sort(
    (left, right) => right.length - left.length,
  )) {
    if (secret.length > 0) result = result.replaceAll(secret, "[REDACTED]");
  }
  return result;
}

function normalizeError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

async function main(): Promise<void> {
  process.exitCode = await executePhysicalGpuConveyorAbCli(process.argv.slice(2));
}

const entryPath = process.argv[1];
if (
  entryPath !== undefined
  && resolve(fileURLToPath(import.meta.url)) === resolve(entryPath)
) {
  void main().catch((error) => {
    process.stderr.write(`${normalizeError(error).message}\n`);
    process.exitCode = 1;
  });
}

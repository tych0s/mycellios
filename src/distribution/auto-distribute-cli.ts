import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  compileAutoDistribution,
  parseAutoDistributionConfig,
  profileCompatibleModel,
  runAutoDistribution,
  writeAutoDistributionArtifacts,
} from "./auto-distribute.js";

export interface AutoDistributeCliArguments {
  configPath: string;
  prepareOnly: boolean;
}

export function parseAutoDistributeCliArguments(argv: readonly string[]): AutoDistributeCliArguments {
  let configPath: string | null = null;
  let prepareOnly = false;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]!;
    if (argument === "--prepare-only") {
      prepareOnly = true;
      continue;
    }
    if (argument === "--config") {
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) throw new Error("auto_distribute_config_requires_value");
      if (configPath !== null) throw new Error("auto_distribute_duplicate_config");
      configPath = value;
      index += 1;
      continue;
    }
    throw new Error(`auto_distribute_unknown_argument:${argument}`);
  }
  if (configPath === null) throw new Error("auto_distribute_config_is_required");
  return { configPath, prepareOnly };
}

export async function executeAutoDistributeCli(argv: readonly string[], cwd = process.cwd()): Promise<void> {
  const args = parseAutoDistributeCliArguments(argv);
  const config = parseAutoDistributionConfig(
    JSON.parse(await readFile(resolve(cwd, args.configPath), "utf8")) as unknown,
  );
  process.stderr.write(`Profiling ${config.model.source} and resolving a certified adapter...\n`);
  const profile = await profileCompatibleModel(config, cwd);
  process.stderr.write(`Certified adapter: ${profile.compatibility.adapterId ?? "none"}\n`);
  const compilation = compileAutoDistribution(config, profile);
  const artifacts = await writeAutoDistributionArtifacts(config, compilation, cwd);
  process.stderr.write(
    `Automatic route: ${compilation.boundaries.join(" -> ")} across ${compilation.boundaries.length - 1} stages\n`,
  );
  process.stderr.write(`Artifacts: ${artifacts}\n`);
  if (args.prepareOnly) return;
  process.stderr.write("Launching stages, running the canary, then registering the model...\n");
  await runAutoDistribution(config, compilation, cwd);
}

async function main(): Promise<void> {
  try {
    await executeAutoDistributeCli(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`auto-distribute: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 2;
  }
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (invokedPath === import.meta.url) await main();

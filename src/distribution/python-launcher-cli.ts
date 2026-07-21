import { randomUUID } from "node:crypto";
import {
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  compilePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";

export const PYTHON_LAUNCH_REQUEST_SCHEMA = "gdlp-python-launch-request/1";

export interface PythonLaunchRequestV1 {
  schema: typeof PYTHON_LAUNCH_REQUEST_SCHEMA;
  manifest: Record<string, unknown>;
  options: PythonLaunchCompilerOptions;
}

export interface PythonLauncherCliArguments {
  inputPath: string;
  outputPath: string | null;
}

export interface PythonLauncherCliDependencies {
  cwd?: string;
  readText?: (path: string) => Promise<string>;
  writeTextAtomic?: (path: string, content: string) => Promise<void>;
  writeStdout?: (content: string) => void;
}

/**
 * Parse the deliberately small, file-oriented CLI surface. Keeping this
 * separate from execution makes imports and embedding side-effect free.
 */
export function parsePythonLauncherCliArguments(
  argv: readonly string[],
): PythonLauncherCliArguments {
  let inputPath: string | null = null;
  let outputPath: string | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]!;
    if (argument === "--input" || argument.startsWith("--input=")) {
      if (inputPath !== null) throw new Error("python_launcher_cli_duplicate_input");
      const parsed = optionValue(argument, argv[index + 1], "--input");
      inputPath = parsed.value;
      index += parsed.consumedNext ? 1 : 0;
      continue;
    }
    if (argument === "--out" || argument.startsWith("--out=")) {
      if (outputPath !== null) throw new Error("python_launcher_cli_duplicate_output");
      const parsed = optionValue(argument, argv[index + 1], "--out");
      outputPath = parsed.value;
      index += parsed.consumedNext ? 1 : 0;
      continue;
    }
    if (argument.startsWith("-")) {
      throw new Error(`python_launcher_cli_unknown_option:${argument}`);
    }
    throw new Error("python_launcher_cli_positional_arguments_are_not_supported");
  }

  if (inputPath === null) throw new Error("python_launcher_cli_input_is_required");
  return { inputPath, outputPath };
}

/** Validate only the versioned request envelope; the compiler validates its payloads. */
export function parsePythonLaunchRequest(value: unknown): PythonLaunchRequestV1 {
  if (!isRecord(value)) throw new Error("python_launch_request_must_be_an_object");
  const keys = Object.keys(value).sort();
  if (
    keys.length !== 3 ||
    keys[0] !== "manifest" ||
    keys[1] !== "options" ||
    keys[2] !== "schema"
  ) {
    throw new Error("python_launch_request_must_have_exact_keys");
  }
  if (value.schema !== PYTHON_LAUNCH_REQUEST_SCHEMA) {
    throw new Error("unsupported_python_launch_request_schema");
  }
  if (!isRecord(value.manifest)) {
    throw new Error("python_launch_request_manifest_must_be_an_object");
  }
  if (!isRecord(value.options)) {
    throw new Error("python_launch_request_options_must_be_an_object");
  }
  return {
    schema: PYTHON_LAUNCH_REQUEST_SCHEMA,
    manifest: value.manifest,
    options: value.options as unknown as PythonLaunchCompilerOptions,
  };
}

/** Compile the request into data only. No process or socket is opened here. */
export function compilePythonLaunchRequest(
  value: unknown,
): PythonPipelineLaunchDescription {
  const request = parsePythonLaunchRequest(value);
  return compilePythonLaunchDescription(request.manifest, request.options);
}

/**
 * Read, compile and emit one launch description. Dependencies are injectable
 * so callers can embed this without mutating global process IO.
 */
export async function executePythonLauncherCli(
  argv: readonly string[],
  dependencies: PythonLauncherCliDependencies = {},
): Promise<PythonPipelineLaunchDescription> {
  const parsed = parsePythonLauncherCliArguments(argv);
  const cwd = dependencies.cwd ?? process.cwd();
  const inputPath = resolve(cwd, parsed.inputPath);
  const outputPath = parsed.outputPath === null ? null : resolve(cwd, parsed.outputPath);
  if (outputPath === inputPath) {
    throw new Error("python_launcher_cli_output_must_not_overwrite_input");
  }
  const readText = dependencies.readText ?? ((path: string) => readFile(path, "utf8"));
  const source = await readText(inputPath);
  const request = parseJson(source);
  const description = compilePythonLaunchRequest(request);
  const rendered = `${JSON.stringify(description, null, 2)}\n`;

  if (outputPath === null) {
    const writeStdout =
      dependencies.writeStdout ?? ((content: string) => void process.stdout.write(content));
    writeStdout(rendered);
  } else {
    const writeTextAtomic = dependencies.writeTextAtomic ?? writePythonLauncherOutputAtomic;
    await writeTextAtomic(outputPath, rendered);
  }
  return description;
}

/** Write beside the destination first, then publish with one rename. */
export async function writePythonLauncherOutputAtomic(
  path: string,
  content: string,
): Promise<void> {
  const directory = dirname(path);
  await mkdir(directory, { recursive: true });
  const temporaryPath = resolve(
    directory,
    `.${path.split(/[\\/]/).at(-1) ?? "python-launch"}.${process.pid}.${randomUUID()}.tmp`,
  );
  try {
    await writeFile(temporaryPath, content, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true }).catch(() => undefined);
    throw error;
  }
}

function optionValue(
  argument: string,
  next: string | undefined,
  name: "--input" | "--out",
): { value: string; consumedNext: boolean } {
  if (argument === name) {
    if (next === undefined || next.startsWith("--") || next.length === 0) {
      throw new Error(`python_launcher_cli_option_requires_value:${name}`);
    }
    return { value: next, consumedNext: true };
  }
  const value = argument.slice(name.length + 1);
  if (value.length === 0) {
    throw new Error(`python_launcher_cli_option_requires_value:${name}`);
  }
  return { value, consumedNext: false };
}

function parseJson(source: string): unknown {
  try {
    return JSON.parse(source) as unknown;
  } catch {
    throw new Error("python_launcher_cli_input_is_not_json");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function main(): Promise<void> {
  try {
    await executePythonLauncherCli(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(
      `python-launcher-cli: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 2;
  }
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (invokedPath === import.meta.url) await main();

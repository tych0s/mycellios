import { resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { executeDistributedDevelopmentGate } from "../src/program/distributed-development-gate.js";

export * from "../src/program/distributed-development-gate.js";

function renderError(value: unknown): string {
  if (value instanceof AggregateError) {
    return [
      value.stack ?? value.message,
      ...value.errors.map((error, index) =>
        `cleanup error ${index + 1}: ${renderError(error)}`
      ),
    ].join("\n");
  }
  return value instanceof Error ? value.stack ?? value.message : String(value);
}

async function main(): Promise<void> {
  try {
    await executeDistributedDevelopmentGate(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`${renderError(error)}\n`);
    process.exitCode = 2;
  }
}

const entryPath = process.argv[1];
if (
  entryPath !== undefined
  && resolve(fileURLToPath(import.meta.url)) === resolve(entryPath)
) {
  void main();
}

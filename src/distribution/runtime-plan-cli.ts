import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "./runtime-manifest.js";

const inputPath = process.argv[2];
if (!inputPath) {
  console.error("Usage: npm run plan:distribution -- <runtime-plan.json>");
  process.exitCode = 2;
} else {
  const source = await readFile(resolve(inputPath), "utf8");
  const request = JSON.parse(source) as RuntimePlanRequest;
  const manifest = buildRuntimePipelineManifest(request);
  process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
}

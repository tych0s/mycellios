// Local coordinator for the browser/native Qwen3 layer integration test.

import { resolve } from "node:path";
import { createCoordinator } from "../src/coordinator/server.js";

const port = Number(process.env.MYCELLIOS_BROWSER_LAYER_TEST_PORT ?? 8772);
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error("MYCELLIOS_BROWSER_LAYER_TEST_PORT must be an unprivileged TCP port.");
}
const internalToken = process.env.MYCELLIOS_INTERNAL_TOKEN?.trim()
  || "local-browser-layer-test-secret";
const artifactDirectory = process.env.MYCELLIOS_BROWSER_LAYER_TEST_ARTIFACTS?.trim()
  || "runtime/browser-layer-test-artifacts";
const runtime = await createCoordinator({
  host: "127.0.0.1", port, databasePath: ":memory:",
  requestTimeoutMs: 60_000,
  internalToken,
  mobileExpertArtifactsPath: resolve(artifactDirectory),
}, { logger: false });
await runtime.app.listen({ host: "127.0.0.1", port });
process.stdout.write(`Browser layer integration: http://127.0.0.1:${port}/browser/\n`);
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => { void runtime.close().then(() => process.exit(0)); });
}

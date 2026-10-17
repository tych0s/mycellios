import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createCoordinator } from "../src/coordinator/server.js";

const host = "127.0.0.1";
const port = Number(process.env.MYCELLIOS_BROWSER_SPLIT_PORT || 8770);
const internalToken = "local-browser-split-probe";
const artifactPath = mkdtempSync(join(tmpdir(), "mycellios-browser-split-"));

const runtime = await createCoordinator({
  host,
  port,
  databasePath: ":memory:",
  requestTimeoutMs: 60_000,
  internalToken,
  mobileAssetsPath: resolve("mobile-dist"),
  mobileExpertArtifactsPath: artifactPath,
}, { logger: false });

await runtime.app.listen({ host, port });
process.stdout.write(`Browser split probe: http://${host}:${port}/browser/\n`);
process.stdout.write(`Internal probe token: ${internalToken}\n`);

async function close(): Promise<void> {
  await runtime.close();
}
process.once("SIGINT", () => { void close(); });
process.once("SIGTERM", () => { void close(); });

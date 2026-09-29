import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const WINSW_VERSION = "2.12.0";
export const WINSW_SHA256 = "05b82d46ad331cc16bdc00de5c6332c1ef818df8ceefcd49c726553209b3a0da";
const URL = `https://github.com/winsw/winsw/releases/download/v${WINSW_VERSION}/WinSW-x64.exe`;

export async function fetchWindowsServiceWrapper(output) {
  const target = resolve(output);
  const existing = await readFile(target).catch((error) => {
    if (error.code === "ENOENT") return null;
    throw error;
  });
  if (existing && sha256(existing) === WINSW_SHA256) return target;
  const response = await fetch(URL);
  if (!response.ok) throw new Error(`node_service_wrapper_download_failed:${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (sha256(bytes) !== WINSW_SHA256) throw new Error("node_service_wrapper_digest_mismatch");
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes);
  return target;
}

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const output = process.argv.find((value) => value.startsWith("--output="))?.slice(9);
  if (!output) throw new Error("usage: fetch-windows-service-wrapper --output=<path>");
  process.stdout.write(`${await fetchWindowsServiceWrapper(output)}\n`);
}

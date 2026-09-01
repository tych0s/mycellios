import { execFileSync } from "node:child_process";
import { lstat, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createHash, randomBytes } from "node:crypto";
import { importPhysicalGateEvidence } from "../src/program/physical-gate-evidence.js";

const receipts = values("--receipt");
if (receipts.length === 0) throw new Error("usage: import-physical-gate-evidence --receipt=<path> [--receipt=<path>]");
const trustPath = resolve(value("--trust") ?? "config/physical-evidence-trust.json");
const outputPath = resolve(value("--output") ?? "config/gate-program-evidence.json");
const sourceSha = value("--source-sha") ?? execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const sourceIdentity = value("--source-identity") ?? `sha256:${createHash("sha256").update(
  execFileSync("git", ["ls-tree", "-r", "HEAD"),
).digest("hex")}`;
const documents = await Promise.all(receipts.map(readReceipt));
const [trust, existing] = await Promise.all([readJson(trustPath), readJson(outputPath)]);
const imported = importPhysicalGateEvidence({ receipts: documents, trust, sourceSha, sourceIdentity, existing });
await writeAtomic(outputPath, imported);
process.stdout.write(`Imported ${documents.length} signed physical receipt(s) for ${sourceSha}.\n`);

async function readReceipt(pathValue: string): Promise<unknown> {
  const path = resolve(pathValue); const stats = await lstat(path);
  if (!stats.isFile() || stats.isSymbolicLink() || stats.size > 2 * 1024 * 1024) throw new Error(`physical_gate_receipt_file_is_unsafe:${path}`);
  return JSON.parse(await readFile(path, "utf8")) as unknown;
}
async function readJson(path: string): Promise<unknown> { return JSON.parse(await readFile(path, "utf8")) as unknown; }
async function writeAtomic(path: string, document: unknown): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}-${randomBytes(6).toString("hex")}`; const file = await open(temporary, "wx", 0o600);
  try { await file.writeFile(`${JSON.stringify(document, null, 2)}\n`, "utf8"); await file.sync(); await file.close(); await rename(temporary, path); }
  catch (error) { await file.close().catch(() => undefined); await rm(temporary, { force: true }); throw error; }
}
function values(name: string): string[] { return process.argv.slice(2).flatMap((argument) => argument.startsWith(`${name}=`) ? [argument.slice(name.length + 1)] : []); }
function value(name: string): string | undefined { return values(name).at(-1); }

import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { gateReceiptSchema, type GateReceipt } from "../contracts/gate-receipt.js";

export function sha256(value: string | Uint8Array): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

export async function readGateReceipt(file: string): Promise<GateReceipt> {
  return gateReceiptSchema.parse(JSON.parse(await readFile(file, "utf8")));
}

export async function writeGateReceiptAtomic(file: string, receipt: GateReceipt): Promise<void> {
  const parsed = gateReceiptSchema.parse(receipt);
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, `${JSON.stringify(parsed, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporary, file);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

export function receiptMatches(receipt: GateReceipt, sourceSha: string, inputDigest: string): boolean {
  return receipt.status === "pass" && receipt.sourceSha === sourceSha && receipt.inputDigest === inputDigest;
}

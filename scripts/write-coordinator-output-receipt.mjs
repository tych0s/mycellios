import { resolve } from "node:path";
import { writeCoordinatorOutputReceipt } from "./coordinator-release-policy.mjs";

const workspace = resolve(import.meta.dirname, "..");
const receipt = writeCoordinatorOutputReceipt(workspace);
process.stdout.write(
  `Native coordinator outputs sealed: ${receipt.receiptId} for ${receipt.sourceId}.\n`,
);

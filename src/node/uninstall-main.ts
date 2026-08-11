import { dirname } from "node:path";
import { executeVerifiedUninstall } from "./uninstall-executor.js";
import { verifyUninstallRequest, waitForUninstallArm } from "./uninstall-helper.js";

const requestPath = argument("--request");
const requestDigest = argument("--request-digest");
if (!requestPath || !requestDigest || !/^sha256:[a-f0-9]{64}$/.test(requestDigest)) {
  throw new Error("node_uninstall_arguments_invalid");
}
const verified = await verifyUninstallRequest({
  requestPath,
  requestDigest,
  expectedRequestDirectory: dirname(requestPath),
});
await waitForUninstallArm({ requestPath, requestId: verified.request.id, requestDigest: requestDigest as `sha256:${string}` });
await executeVerifiedUninstall({ ...verified, requestDigest: requestDigest as `sha256:${string}` });

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

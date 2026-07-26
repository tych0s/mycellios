import type { WorkerCapabilities } from "../contracts/types.js";
import type {
  WorkerAdmissionIdentity,
  WorkerProtocolRange,
} from "../contracts/worker-admission.js";
import { sha256CanonicalEvidence } from "./json.js";

export function workerRegistrationDigest(input: {
  identity: WorkerAdmissionIdentity;
  capabilities: WorkerCapabilities;
  protocol: WorkerProtocolRange;
}): string {
  return sha256CanonicalEvidence({
    schema: "mycellios-worker-registration/1",
    identity: input.identity,
    capabilities: input.capabilities,
    protocol: input.protocol,
  });
}

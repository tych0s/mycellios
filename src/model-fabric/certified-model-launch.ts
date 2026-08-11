import {
  assertProductiveModelCertification,
  verifyModelCertification,
  type ModelCertification,
} from "../contracts/model-certification.js";
import { parseModelDistributionManifest } from "../contracts/model-distribution-manifest.js";

export interface CertifiedModelLaunchEnvironment {
  topology: ModelCertification["topology"];
  hardware: Omit<ModelCertification["hardware"], "minimumMemoryBytes"> & { availableMemoryBytes: number };
  contextTokens: number;
  codec: string;
  workerProtocol: number;
  tensorAbi: string;
}

export function authorizeCertifiedModelLaunch(input: {
  distributionManifest: unknown;
  certification: unknown;
  pinnedCertificationKey: { keyId: string; spki: string };
  expectedAdapterContractId: string;
  expectedComponentManifestId: string;
  environment: CertifiedModelLaunchEnvironment;
  now?: Date;
}): { distributionManifestId: string; certificationId: string } {
  const distribution = parseModelDistributionManifest(input.distributionManifest);
  const certification = verifyModelCertification(input.certification, {
    pinnedKey: input.pinnedCertificationKey,
    expectedDistributionManifestId: distribution.manifestId,
    ...(input.now ? { now: input.now } : {}),
  });
  assertProductiveModelCertification(certification);
  if (certification.adapterContractId !== input.expectedAdapterContractId) throw new Error("certified_model_adapter_mismatch");
  if (certification.componentManifestId !== input.expectedComponentManifestId) throw new Error("certified_model_component_mismatch");
  if (JSON.stringify(certification.topology) !== JSON.stringify(input.environment.topology)) throw new Error("certified_model_topology_mismatch");
  const expectedHardware = certification.hardware;
  const actualHardware = input.environment.hardware;
  if (expectedHardware.platform !== actualHardware.platform
    || expectedHardware.arch !== actualHardware.arch
    || expectedHardware.backend !== actualHardware.backend
    || expectedHardware.deviceFamily !== actualHardware.deviceFamily
    || expectedHardware.driverFingerprint !== actualHardware.driverFingerprint
    || actualHardware.availableMemoryBytes < expectedHardware.minimumMemoryBytes) {
    throw new Error("certified_model_hardware_mismatch");
  }
  if (input.environment.contextTokens > certification.context.maximumTokens) throw new Error("certified_model_context_exceeds_certification");
  if (!certification.context.codecs.includes(input.environment.codec)) throw new Error("certified_model_codec_mismatch");
  if (input.environment.workerProtocol < certification.context.workerProtocol.min
    || input.environment.workerProtocol > certification.context.workerProtocol.max) throw new Error("certified_model_protocol_mismatch");
  if (input.environment.tensorAbi !== certification.context.tensorAbi
    || input.environment.tensorAbi !== distribution.tensorAbi) throw new Error("certified_model_tensor_abi_mismatch");
  return { distributionManifestId: distribution.manifestId, certificationId: certification.certificationId };
}

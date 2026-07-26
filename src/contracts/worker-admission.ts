import { z } from "zod";

const SHA256 = /^sha256:[a-f0-9]{64}$/;
const BASE64URL = /^[A-Za-z0-9_-]+$/;

export const WORKER_PROTOCOL_MIN = 1;
export const WORKER_PROTOCOL_MAX = 1;

export const workerAdmissionIdentitySchema = z.object({
  kind: z.enum(["device", "cell", "browser"]),
  id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/),
}).strict();

export const workerProtocolRangeSchema = z.object({
  min: z.number().int().min(1).max(65_535),
  max: z.number().int().min(1).max(65_535),
}).strict().superRefine((range, context) => {
  if (range.min > range.max) {
    context.addIssue({
      code: "custom",
      path: ["min"],
      message: "protocol min must not exceed max",
    });
  }
});

export const workerAdmissionPublicKeySchema = z.object({
  algorithm: z.enum(["ed25519", "ecdsa-p256-sha256"]),
  spki: z.string().min(40).max(256).regex(BASE64URL),
}).strict();

export const workerAdmissionChallengeRequestSchema = z.object({
  identity: workerAdmissionIdentitySchema,
  publicKey: workerAdmissionPublicKeySchema,
  protocol: workerProtocolRangeSchema,
  registrationDigest: z.string().regex(SHA256),
}).strict();

export const workerAdmissionProofSchema = z.object({
  challengeId: z.string().uuid(),
  publicKey: workerAdmissionPublicKeySchema,
  protocol: workerProtocolRangeSchema,
  registrationDigest: z.string().regex(SHA256),
  signature: z.string().min(64).max(256).regex(BASE64URL),
}).strict();

export const workerAdmissionChallengeResponseSchema = z.object({
  challengeId: z.string().uuid(),
  expiresAt: z.string().datetime(),
  protocolVersion: z.number().int().min(1).max(65_535),
  signingPayload: z.string().min(1).max(4_096),
}).strict();

export type WorkerAdmissionIdentity = z.infer<typeof workerAdmissionIdentitySchema>;
export type WorkerProtocolRange = z.infer<typeof workerProtocolRangeSchema>;
export type WorkerAdmissionPublicKey = z.infer<typeof workerAdmissionPublicKeySchema>;
export type WorkerAdmissionProof = z.infer<typeof workerAdmissionProofSchema>;
export type WorkerAdmissionChallengeRequest =
  z.infer<typeof workerAdmissionChallengeRequestSchema>;
export type WorkerAdmissionChallengeResponse =
  z.infer<typeof workerAdmissionChallengeResponseSchema>;

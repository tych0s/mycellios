import {
  createHash,
  createPublicKey,
  randomBytes,
  randomUUID,
  verify,
  type KeyObject,
} from "node:crypto";
import { canonicalEvidenceJson } from "../core/json.js";
import type { MeshDatabase, WorkerAdmissionCredentialRow } from "../storage/database.js";
import {
  WORKER_PROTOCOL_MAX,
  WORKER_PROTOCOL_MIN,
  workerAdmissionProofSchema,
  workerCredentialRotationProofSchema,
  type WorkerAdmissionChallengeRequest,
  type WorkerAdmissionChallengeResponse,
  type WorkerAdmissionIdentity,
  type WorkerAdmissionProof,
  type WorkerAdmissionPublicKey,
  type WorkerProtocolRange,
  type WorkerCredentialRotationChallengeRequest,
  type WorkerCredentialRotationProof,
} from "../contracts/worker-admission.js";

const BASE64URL = /^[A-Za-z0-9_-]+$/;

export const WORKER_ADMISSION_CHALLENGE_TTL_MS = 60_000;
export const MAX_PENDING_WORKER_ADMISSION_CHALLENGES = 1_024;

interface PendingChallenge {
  identity: WorkerAdmissionIdentity;
  publicKey: WorkerAdmissionPublicKey;
  fingerprint: string;
  registrationDigest: string;
  protocol: WorkerProtocolRange;
  protocolVersion: number;
  signingPayload: string;
  expiresAt: number;
}

interface PendingRotationChallenge {
  identity: WorkerAdmissionIdentity;
  currentKey: KeyObject;
  currentAlgorithm: WorkerAdmissionPublicKey["algorithm"];
  currentFingerprint: string;
  nextKey: KeyObject;
  nextPublicKey: WorkerAdmissionPublicKey;
  nextFingerprint: string;
  protocolVersion: number;
  signingPayload: string;
  expiresAt: number;
}

export class WorkerAdmissionError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly statusCode: 400 | 401 | 403 | 404 | 409 | 426 | 429,
  ) {
    super(message);
    this.name = "WorkerAdmissionError";
  }
}

export class WorkerAdmissionAuthority {
  private readonly challenges = new Map<string, PendingChallenge>();
  private readonly rotationChallenges = new Map<string, PendingRotationChallenge>();

  constructor(
    private readonly database: MeshDatabase,
    private readonly now: () => number = Date.now,
  ) {}

  issue(
    input: WorkerAdmissionChallengeRequest,
  ): WorkerAdmissionChallengeResponse {
    this.pruneExpired();
    if (
      this.challenges.size + this.rotationChallenges.size
      >= MAX_PENDING_WORKER_ADMISSION_CHALLENGES
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_busy",
        "The coordinator has too many pending worker admission challenges.",
        429,
      );
    }
    const protocolVersion = selectWorkerProtocolVersion(input.protocol);
    const normalized = normalizePublicKey(input.publicKey);
    const challengeId = randomUUID();
    const expiresAt = this.now() + WORKER_ADMISSION_CHALLENGE_TTL_MS;
    const signingPayload = canonicalEvidenceJson({
      schema: "mycellios-worker-admission/1",
      challengeId,
      nonce: randomBytes(32).toString("base64url"),
      identity: input.identity,
      publicKey: normalized.publicKey,
      protocol: input.protocol,
      protocolVersion,
      registrationDigest: input.registrationDigest,
      expiresAt,
    });
    this.challenges.set(challengeId, {
      identity: input.identity,
      publicKey: normalized.publicKey,
      fingerprint: normalized.fingerprint,
      registrationDigest: input.registrationDigest,
      protocol: input.protocol,
      protocolVersion,
      signingPayload,
      expiresAt,
    });
    return {
      challengeId,
      expiresAt: new Date(expiresAt).toISOString(),
      protocolVersion,
      signingPayload,
    };
  }

  issueRotation(
    input: WorkerCredentialRotationChallengeRequest,
  ): WorkerAdmissionChallengeResponse {
    this.pruneExpired();
    if (
      this.challenges.size + this.rotationChallenges.size
      >= MAX_PENDING_WORKER_ADMISSION_CHALLENGES
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_busy",
        "The coordinator has too many pending worker admission challenges.",
        429,
      );
    }
    const protocolVersion = selectWorkerProtocolVersion(input.protocol);
    const current = normalizePublicKey(input.currentPublicKey);
    const next = normalizePublicKey(input.nextPublicKey);
    if (current.fingerprint === next.fingerprint) {
      throw new WorkerAdmissionError(
        "worker_rotation_key_unchanged",
        "The replacement worker key must be different from the current key.",
        400,
      );
    }
    const existing = this.database.getWorkerAdmissionCredential(
      input.identity.kind,
      input.identity.id,
    );
    if (!existing) {
      throw new WorkerAdmissionError(
        "worker_credential_unknown",
        "This worker identity has not been enrolled.",
        404,
      );
    }
    if (existing.status === "revoked") {
      throw new WorkerAdmissionError(
        "worker_credential_revoked",
        "This worker credential has been revoked.",
        403,
      );
    }
    if (
      existing.algorithm !== current.publicKey.algorithm
      || existing.publicKey !== current.publicKey.spki
      || existing.fingerprint !== current.fingerprint
    ) {
      throw new WorkerAdmissionError(
        "worker_rotation_current_key_mismatch",
        "The current worker key does not match the enrolled credential.",
        409,
      );
    }
    const challengeId = randomUUID();
    const expiresAt = this.now() + WORKER_ADMISSION_CHALLENGE_TTL_MS;
    const signingPayload = canonicalEvidenceJson({
      schema: "mycellios-worker-credential-rotation/1",
      challengeId,
      nonce: randomBytes(32).toString("base64url"),
      identity: input.identity,
      currentPublicKey: current.publicKey,
      currentFingerprint: current.fingerprint,
      nextPublicKey: next.publicKey,
      nextFingerprint: next.fingerprint,
      protocol: input.protocol,
      protocolVersion,
      expiresAt,
    });
    this.rotationChallenges.set(challengeId, {
      identity: input.identity,
      currentKey: current.key,
      currentAlgorithm: current.publicKey.algorithm,
      currentFingerprint: current.fingerprint,
      nextKey: next.key,
      nextPublicKey: next.publicKey,
      nextFingerprint: next.fingerprint,
      protocolVersion,
      signingPayload,
      expiresAt,
    });
    return {
      challengeId,
      expiresAt: new Date(expiresAt).toISOString(),
      protocolVersion,
      signingPayload,
    };
  }

  verifyRotation(input: {
    identity: WorkerAdmissionIdentity;
    proof: WorkerCredentialRotationProof;
  }): {
    protocolVersion: number;
    previousFingerprint: string;
    credentialFingerprint: string;
  } {
    const proof = workerCredentialRotationProofSchema.parse(input.proof);
    const challenge = this.rotationChallenges.get(proof.challengeId);
    this.rotationChallenges.delete(proof.challengeId);
    if (!challenge) {
      throw new WorkerAdmissionError(
        "worker_rotation_challenge_unknown",
        "The credential rotation challenge is unknown or was already consumed.",
        401,
      );
    }
    if (challenge.expiresAt <= this.now()) {
      throw new WorkerAdmissionError(
        "worker_rotation_challenge_expired",
        "The credential rotation challenge expired.",
        401,
      );
    }
    if (
      challenge.identity.kind !== input.identity.kind
      || challenge.identity.id !== input.identity.id
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_identity_mismatch",
        "The credential rotation identity does not match its challenge.",
        401,
      );
    }
    if (
      !verifyAdmissionSignature(
        challenge.currentKey,
        challenge.currentAlgorithm,
        challenge.signingPayload,
        proof.currentSignature,
      )
      || !verifyAdmissionSignature(
        challenge.nextKey,
        challenge.nextPublicKey.algorithm,
        challenge.signingPayload,
        proof.nextSignature,
      )
    ) {
      throw new WorkerAdmissionError(
        "worker_rotation_signature_invalid",
        "Credential rotation requires valid signatures from both keys.",
        401,
      );
    }
    const rotation = this.database.rotateWorkerAdmissionCredential({
      identityKind: input.identity.kind,
      identityId: input.identity.id,
      expectedFingerprint: challenge.currentFingerprint,
      algorithm: challenge.nextPublicKey.algorithm,
      publicKey: challenge.nextPublicKey.spki,
      fingerprint: challenge.nextFingerprint,
      protocolVersion: challenge.protocolVersion,
    });
    if (rotation.state === "not_found") {
      throw new WorkerAdmissionError(
        "worker_credential_unknown",
        "This worker identity has not been enrolled.",
        404,
      );
    }
    if (rotation.state === "revoked") {
      throw new WorkerAdmissionError(
        "worker_credential_revoked",
        "This worker credential has been revoked.",
        403,
      );
    }
    if (rotation.state === "fingerprint_mismatch") {
      throw new WorkerAdmissionError(
        "worker_rotation_raced",
        "The worker credential changed while rotation was pending.",
        409,
      );
    }
    if (rotation.state === "fingerprint_in_use") {
      throw new WorkerAdmissionError(
        "worker_credential_reused",
        "The replacement key is already bound to another worker identity.",
        409,
      );
    }
    if (!rotation.credential) throw new Error("worker_rotation_credential_missing");
    return {
      protocolVersion: challenge.protocolVersion,
      previousFingerprint: challenge.currentFingerprint,
      credentialFingerprint: rotation.credential.fingerprint,
    };
  }

  verify(input: {
    identity: WorkerAdmissionIdentity;
    registrationDigest: string;
    proof: WorkerAdmissionProof;
  }): {
    protocolVersion: number;
    credentialFingerprint: string;
    enrollment: "enrolled" | "accepted";
  } {
    const proof = workerAdmissionProofSchema.parse(input.proof);
    const challenge = this.challenges.get(proof.challengeId);
    // Challenges are one-shot even when the proof is malformed. This prevents
    // online signature guessing and makes replay deterministic.
    this.challenges.delete(proof.challengeId);
    if (!challenge) {
      throw new WorkerAdmissionError(
        "worker_admission_challenge_unknown",
        "The worker admission challenge is unknown or was already consumed.",
        401,
      );
    }
    if (challenge.expiresAt <= this.now()) {
      throw new WorkerAdmissionError(
        "worker_admission_challenge_expired",
        "The worker admission challenge expired.",
        401,
      );
    }
    if (
      challenge.identity.kind !== input.identity.kind
      || challenge.identity.id !== input.identity.id
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_identity_mismatch",
        "The signed admission identity does not match the registration.",
        401,
      );
    }
    if (
      challenge.registrationDigest !== input.registrationDigest
      || proof.registrationDigest !== input.registrationDigest
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_registration_changed",
        "The worker registration changed after the admission challenge was issued.",
        401,
      );
    }
    if (
      challenge.publicKey.algorithm !== proof.publicKey.algorithm
      || challenge.publicKey.spki !== proof.publicKey.spki
      || challenge.protocol.min !== proof.protocol.min
      || challenge.protocol.max !== proof.protocol.max
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_proof_mismatch",
        "The worker admission proof does not match its challenge.",
        401,
      );
    }
    const normalized = normalizePublicKey(proof.publicKey);
    if (
      normalized.fingerprint !== challenge.fingerprint
      || !verifyAdmissionSignature(
        normalized.key,
        proof.publicKey.algorithm,
        challenge.signingPayload,
        proof.signature,
      )
    ) {
      throw new WorkerAdmissionError(
        "worker_admission_signature_invalid",
        "The worker could not prove possession of its device key.",
        401,
      );
    }

    const admission = this.database.admitWorkerCredential({
      identityKind: input.identity.kind,
      identityId: input.identity.id,
      algorithm: proof.publicKey.algorithm,
      publicKey: proof.publicKey.spki,
      fingerprint: challenge.fingerprint,
      protocolVersion: challenge.protocolVersion,
    });
    if (admission.state === "revoked") {
      throw new WorkerAdmissionError(
        "worker_credential_revoked",
        "This worker credential has been revoked.",
        403,
      );
    }
    if (admission.state === "key_mismatch") {
      throw new WorkerAdmissionError(
        "worker_identity_key_mismatch",
        "This stable worker identity is already bound to another device key.",
        409,
      );
    }
    if (admission.state === "fingerprint_in_use") {
      throw new WorkerAdmissionError(
        "worker_credential_reused",
        "This device key is already bound to another worker identity.",
        409,
      );
    }
    return {
      protocolVersion: challenge.protocolVersion,
      credentialFingerprint: admission.credential.fingerprint,
      enrollment: admission.state,
    };
  }

  private pruneExpired(): void {
    const now = this.now();
    for (const [challengeId, challenge] of this.challenges) {
      if (challenge.expiresAt <= now) this.challenges.delete(challengeId);
    }
    for (const [challengeId, challenge] of this.rotationChallenges) {
      if (challenge.expiresAt <= now) this.rotationChallenges.delete(challengeId);
    }
  }
}

export function selectWorkerProtocolVersion(range: WorkerProtocolRange): number {
  const minimum = Math.max(range.min, WORKER_PROTOCOL_MIN);
  const maximum = Math.min(range.max, WORKER_PROTOCOL_MAX);
  if (minimum > maximum) {
    throw new WorkerAdmissionError(
      "worker_protocol_incompatible",
      `Worker protocol ${range.min}-${range.max} is incompatible with coordinator ${WORKER_PROTOCOL_MIN}-${WORKER_PROTOCOL_MAX}.`,
      426,
    );
  }
  return maximum;
}

function normalizePublicKey(input: WorkerAdmissionPublicKey): {
  publicKey: WorkerAdmissionPublicKey;
  fingerprint: string;
  key: KeyObject;
} {
  let der: Buffer;
  try {
    der = decodeBase64Url(input.spki);
  } catch {
    throw new WorkerAdmissionError(
      "worker_admission_public_key_invalid",
      "The worker admission public key is not valid base64url.",
      400,
    );
  }
  let key: KeyObject;
  try {
    key = createPublicKey({ key: der, format: "der", type: "spki" });
  } catch {
    throw new WorkerAdmissionError(
      "worker_admission_public_key_invalid",
      "The worker admission public key is not a valid SPKI key.",
      400,
    );
  }
  if (input.algorithm === "ed25519" && key.asymmetricKeyType !== "ed25519") {
    throw new WorkerAdmissionError(
      "worker_admission_algorithm_mismatch",
      "The worker admission key is not Ed25519.",
      400,
    );
  }
  if (input.algorithm === "ecdsa-p256-sha256") {
    const details = key.asymmetricKeyDetails;
    if (key.asymmetricKeyType !== "ec" || details?.namedCurve !== "prime256v1") {
      throw new WorkerAdmissionError(
        "worker_admission_algorithm_mismatch",
        "The browser admission key is not ECDSA P-256.",
        400,
      );
    }
  }
  const canonicalDer = key.export({ format: "der", type: "spki" });
  const spki = Buffer.from(canonicalDer).toString("base64url");
  return {
    publicKey: { algorithm: input.algorithm, spki },
    fingerprint: `sha256:${createHash("sha256").update(canonicalDer).digest("hex")}`,
    key,
  };
}

function verifyAdmissionSignature(
  key: KeyObject,
  algorithm: WorkerAdmissionPublicKey["algorithm"],
  payload: string,
  signature: string,
): boolean {
  let decoded: Buffer;
  try {
    decoded = decodeBase64Url(signature);
  } catch {
    return false;
  }
  try {
    if (algorithm === "ed25519") {
      return verify(null, Buffer.from(payload, "utf8"), key, decoded);
    }
    return verify(
      "sha256",
      Buffer.from(payload, "utf8"),
      { key, dsaEncoding: "ieee-p1363" },
      decoded,
    );
  } catch {
    return false;
  }
}

function decodeBase64Url(value: string): Buffer {
  if (!BASE64URL.test(value)) throw new Error("invalid_base64url");
  const decoded = Buffer.from(value, "base64url");
  if (decoded.length === 0 || decoded.toString("base64url") !== value) {
    throw new Error("non_canonical_base64url");
  }
  return decoded;
}

export function admissionCredentialSummary(
  credential: WorkerAdmissionCredentialRow,
): Pick<
  WorkerAdmissionCredentialRow,
  "identityKind" | "identityId" | "fingerprint" | "status" | "protocolVersion" | "lastSeenAt"
> {
  return {
    identityKind: credential.identityKind,
    identityId: credential.identityId,
    fingerprint: credential.fingerprint,
    status: credential.status,
    protocolVersion: credential.protocolVersion,
    lastSeenAt: credential.lastSeenAt,
  };
}

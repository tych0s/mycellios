import { generateKeyPairSync, sign } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import type { WorkerCapabilities } from "../src/contracts/types.js";
import { workerRegistrationDigest } from "../src/core/worker-admission-digest.js";
import {
  WorkerAdmissionAuthority,
  WorkerAdmissionError,
} from "../src/coordinator/worker-admission.js";
import { MeshDatabase } from "../src/storage/database.js";
import {
  generateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "../src/worker/admission-credential.js";

const databases: MeshDatabase[] = [];

afterEach(() => {
  for (const database of databases.splice(0)) database.close();
});

describe("signed worker admission", () => {
  it("enrolls one stable key and accepts it again without creating a duplicate identity", () => {
    const authority = createAuthority();
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const first = admit(authority, signer, "desktop-a");
    const second = admit(authority, signer, "desktop-a");

    expect(first.enrollment).toBe("enrolled");
    expect(second).toEqual({
      protocolVersion: 1,
      credentialFingerprint: first.credentialFingerprint,
      enrollment: "accepted",
    });
  });

  it("consumes every challenge once and rejects replay", () => {
    const authority = createAuthority();
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const identity = { kind: "device" as const, id: "desktop-replay" };
    const capabilities = workerCapabilities();
    const protocol = { min: 1, max: 1 };
    const registrationDigest = workerRegistrationDigest({
      identity,
      capabilities,
      protocol,
    });
    const challenge = authority.issue({
      identity,
      publicKey: signer.publicKey,
      protocol,
      registrationDigest,
    });
    const proof = {
      challengeId: challenge.challengeId,
      publicKey: signer.publicKey,
      protocol,
      registrationDigest,
      signature: signer.sign(challenge.signingPayload),
    };

    expect(authority.verify({ identity, registrationDigest, proof }).enrollment).toBe("enrolled");
    expect(() => authority.verify({ identity, registrationDigest, proof })).toThrowError(
      expect.objectContaining({ code: "worker_admission_challenge_unknown" }),
    );
  });

  it("rejects a registration changed after the device signed its challenge", () => {
    const authority = createAuthority();
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const identity = { kind: "device" as const, id: "desktop-tampered" };
    const capabilities = workerCapabilities();
    const protocol = { min: 1, max: 1 };
    const registrationDigest = workerRegistrationDigest({
      identity,
      capabilities,
      protocol,
    });
    const challenge = authority.issue({
      identity,
      publicKey: signer.publicKey,
      protocol,
      registrationDigest,
    });
    const changedDigest = workerRegistrationDigest({
      identity,
      capabilities: {
        ...capabilities,
        gpus: [{ ...capabilities.gpus[0]!, offeredVramMb: 65_536 }],
      },
      protocol,
    });

    expect(() => authority.verify({
      identity,
      registrationDigest: changedDigest,
      proof: {
        challengeId: challenge.challengeId,
        publicKey: signer.publicKey,
        protocol,
        registrationDigest,
        signature: signer.sign(challenge.signingPayload),
      },
    })).toThrowError(
      expect.objectContaining({ code: "worker_admission_registration_changed" }),
    );
  });

  it("does not let a new key silently take over an existing stable identity", () => {
    const authority = createAuthority();
    admit(
      authority,
      workerAdmissionSigner(generateWorkerAdmissionCredential()),
      "desktop-bound",
    );
    expect(() => admit(
      authority,
      workerAdmissionSigner(generateWorkerAdmissionCredential()),
      "desktop-bound",
    )).toThrowError(
      expect.objectContaining({ code: "worker_identity_key_mismatch" }),
    );
  });

  it("accepts the persistent ECDSA P-256 proof used by browser workers", () => {
    const authority = createAuthority();
    const { publicKey, privateKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
    const admissionPublicKey = {
      algorithm: "ecdsa-p256-sha256" as const,
      spki: publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    };
    const identity = { kind: "browser" as const, id: "browser-stable-a" };
    const protocol = { min: 1, max: 1 };
    const registrationDigest = workerRegistrationDigest({
      identity,
      capabilities: workerCapabilities(),
      protocol,
    });
    const challenge = authority.issue({
      identity,
      publicKey: admissionPublicKey,
      protocol,
      registrationDigest,
    });
    const signature = sign(
      "sha256",
      Buffer.from(challenge.signingPayload, "utf8"),
      { key: privateKey, dsaEncoding: "ieee-p1363" },
    ).toString("base64url");

    expect(authority.verify({
      identity,
      registrationDigest,
      proof: {
        challengeId: challenge.challengeId,
        publicKey: admissionPublicKey,
        protocol,
        registrationDigest,
        signature,
      },
    })).toMatchObject({
      protocolVersion: 1,
      enrollment: "enrolled",
    });
  });

  it("rotates a credential only when both the current and replacement keys sign", () => {
    const authority = createAuthority();
    const current = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const next = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const enrolled = admit(authority, current, "desktop-rotate");
    const identity = { kind: "device" as const, id: "desktop-rotate" };
    const challenge = authority.issueRotation({
      identity,
      currentPublicKey: current.publicKey,
      nextPublicKey: next.publicKey,
      protocol: { min: 1, max: 1 },
    });

    const rotated = authority.verifyRotation({
      identity,
      proof: {
        challengeId: challenge.challengeId,
        currentSignature: current.sign(challenge.signingPayload),
        nextSignature: next.sign(challenge.signingPayload),
      },
    });

    expect(rotated.previousFingerprint).toBe(enrolled.credentialFingerprint);
    expect(rotated.credentialFingerprint).not.toBe(enrolled.credentialFingerprint);
    expect(admit(authority, next, "desktop-rotate").enrollment).toBe("accepted");
    expect(() => admit(authority, current, "desktop-rotate")).toThrowError(
      expect.objectContaining({ code: "worker_identity_key_mismatch" }),
    );
  });

  it("revokes a credential idempotently and blocks later admission or rotation", () => {
    const authority = createAuthority();
    const current = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const enrolled = admit(authority, current, "desktop-revoked");
    const database = databases.at(-1)!;
    const first = database.revokeWorkerAdmissionCredential({
      identityKind: "device",
      identityId: "desktop-revoked",
      expectedFingerprint: enrolled.credentialFingerprint,
      reason: "device reported stolen",
    });
    const second = database.revokeWorkerAdmissionCredential({
      identityKind: "device",
      identityId: "desktop-revoked",
      expectedFingerprint: enrolled.credentialFingerprint,
      reason: "device reported stolen",
    });

    expect(first.state).toBe("revoked");
    expect(second.state).toBe("already_revoked");
    expect(() => admit(authority, current, "desktop-revoked")).toThrowError(
      expect.objectContaining({ code: "worker_credential_revoked" }),
    );
    expect(() => authority.issueRotation({
      identity: { kind: "device", id: "desktop-revoked" },
      currentPublicKey: current.publicKey,
      nextPublicKey: workerAdmissionSigner(generateWorkerAdmissionCredential()).publicKey,
      protocol: { min: 1, max: 1 },
    })).toThrowError(
      expect.objectContaining({ code: "worker_credential_revoked" }),
    );
  });

  it("rejects protocol ranges that do not overlap the coordinator", () => {
    const authority = createAuthority();
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    expect(() => authority.issue({
      identity: { kind: "device", id: "desktop-new-protocol" },
      publicKey: signer.publicKey,
      protocol: { min: 2, max: 4 },
      registrationDigest: `sha256:${"a".repeat(64)}`,
    })).toThrowError(
      expect.objectContaining({
        code: "worker_protocol_incompatible",
        statusCode: 426,
      } satisfies Partial<WorkerAdmissionError>),
    );
  });
});

function createAuthority(): WorkerAdmissionAuthority {
  const database = new MeshDatabase(":memory:");
  databases.push(database);
  return new WorkerAdmissionAuthority(database);
}

function admit(
  authority: WorkerAdmissionAuthority,
  signer: ReturnType<typeof workerAdmissionSigner>,
  id: string,
) {
  const identity = { kind: "device" as const, id };
  const capabilities = workerCapabilities();
  const protocol = { min: 1, max: 1 };
  const registrationDigest = workerRegistrationDigest({
    identity,
    capabilities,
    protocol,
  });
  const challenge = authority.issue({
    identity,
    publicKey: signer.publicKey,
    protocol,
    registrationDigest,
  });
  return authority.verify({
    identity,
    registrationDigest,
    proof: {
      challengeId: challenge.challengeId,
      publicKey: signer.publicKey,
      protocol,
      registrationDigest,
      signature: signer.sign(challenge.signingPayload),
    },
  });
}

function workerCapabilities(): WorkerCapabilities {
  return {
    region: "es-mad",
    agentVersion: "test",
    gpus: [{
      id: "gpu-0",
      vendor: "nvidia",
      model: "Synthetic GPU",
      physicalVramMb: 8_192,
      offeredVramMb: 4_096,
      freeOfferedVramMb: 4_096,
    }],
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    deployments: [],
    network: { coordinatorRttMs: 10, uplinkMbps: 100, downlinkMbps: 100 },
  };
}

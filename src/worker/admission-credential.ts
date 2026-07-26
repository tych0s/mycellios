import {
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  sign,
} from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import type { WorkerAdmissionPublicKey } from "../contracts/worker-admission.js";

export interface WorkerAdmissionCredential {
  schema: "mycellios-worker-credential/1";
  algorithm: "ed25519";
  publicKeySpki: string;
  privateKeyPkcs8: string;
}

export interface WorkerAdmissionSigner {
  publicKey: WorkerAdmissionPublicKey;
  sign(payload: string): string;
}

export function generateWorkerAdmissionCredential(): WorkerAdmissionCredential {
  const pair = generateKeyPairSync("ed25519");
  return {
    schema: "mycellios-worker-credential/1",
    algorithm: "ed25519",
    publicKeySpki: Buffer.from(
      pair.publicKey.export({ format: "der", type: "spki" }),
    ).toString("base64url"),
    privateKeyPkcs8: Buffer.from(
      pair.privateKey.export({ format: "der", type: "pkcs8" }),
    ).toString("base64url"),
  };
}

export function parseWorkerAdmissionCredential(value: unknown): WorkerAdmissionCredential {
  if (!value || typeof value !== "object") throw new Error("worker_credential_invalid");
  const record = value as Partial<WorkerAdmissionCredential>;
  if (
    record.schema !== "mycellios-worker-credential/1"
    || record.algorithm !== "ed25519"
    || typeof record.publicKeySpki !== "string"
    || typeof record.privateKeyPkcs8 !== "string"
  ) {
    throw new Error("worker_credential_invalid");
  }
  const publicDer = decodeCanonicalBase64Url(record.publicKeySpki);
  const privateDer = decodeCanonicalBase64Url(record.privateKeyPkcs8);
  let privateKey;
  let publicKey;
  try {
    privateKey = createPrivateKey({ key: privateDer, format: "der", type: "pkcs8" });
    publicKey = createPublicKey({ key: publicDer, format: "der", type: "spki" });
  } catch {
    throw new Error("worker_credential_invalid");
  }
  if (
    privateKey.asymmetricKeyType !== "ed25519"
    || publicKey.asymmetricKeyType !== "ed25519"
  ) {
    throw new Error("worker_credential_algorithm_invalid");
  }
  const derived = Buffer.from(
    createPublicKey(privateKey).export({ format: "der", type: "spki" }),
  ).toString("base64url");
  if (derived !== record.publicKeySpki) throw new Error("worker_credential_key_mismatch");
  return {
    schema: record.schema,
    algorithm: record.algorithm,
    publicKeySpki: record.publicKeySpki,
    privateKeyPkcs8: record.privateKeyPkcs8,
  };
}

export function workerAdmissionSigner(
  credential: WorkerAdmissionCredential,
): WorkerAdmissionSigner {
  const parsed = parseWorkerAdmissionCredential(credential);
  const privateKey = createPrivateKey({
    key: Buffer.from(parsed.privateKeyPkcs8, "base64url"),
    format: "der",
    type: "pkcs8",
  });
  return {
    publicKey: {
      algorithm: "ed25519",
      spki: parsed.publicKeySpki,
    },
    sign(payload: string): string {
      return sign(null, Buffer.from(payload, "utf8"), privateKey).toString("base64url");
    },
  };
}

export function loadOrCreateWorkerAdmissionCredential(
  path: string,
): WorkerAdmissionCredential {
  if (existsSync(path)) {
    return parseWorkerAdmissionCredential(
      JSON.parse(readFileSync(path, "utf8")) as unknown,
    );
  }
  mkdirSync(dirname(path), { recursive: true });
  const credential = generateWorkerAdmissionCredential();
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(credential)}\n`, {
    encoding: "utf8",
    mode: 0o600,
    flag: "wx",
  });
  renameSync(temporary, path);
  return credential;
}

function decodeCanonicalBase64Url(value: string): Buffer {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) throw new Error("worker_credential_invalid");
  const decoded = Buffer.from(value, "base64url");
  if (decoded.length === 0 || decoded.toString("base64url") !== value) {
    throw new Error("worker_credential_invalid");
  }
  return decoded;
}

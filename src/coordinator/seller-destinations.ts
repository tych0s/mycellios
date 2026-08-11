import { createHash, createPublicKey, verify } from "node:crypto";
import type { MeshDatabase } from "../storage/database.js";
import type { PayoutMethod } from "./payouts.js";

export interface SellerDestinationAttestation {
  schema: "mycellios.seller-destination.v1";
  destinationId: string;
  sellerId: string;
  payoutMethod: PayoutMethod;
  destinationKind: "provider_account" | "wallet";
  destinationReference: string;
  destinationFingerprint: string;
  verifiedAt: number;
  expiresAt: number;
  verifierKeyId: string;
  signature: string;
}

export interface SellerPayoutDestination {
  id: string;
  sellerId: string;
  payoutMethod: PayoutMethod;
  destinationKind: "provider_account" | "wallet";
  destinationReference: string;
  destinationFingerprint: string;
  status: "active" | "revoked";
  verifiedAt: number;
  expiresAt: number;
  revokedAt: number | null;
}

export class SellerDestinationError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "SellerDestinationError";
  }
}

export class SellerDestinationManager {
  constructor(
    private readonly database: MeshDatabase,
    private readonly trustedVerifierKeys: ReadonlyMap<string, string>,
    private readonly now: () => number = Date.now,
  ) {}

  record(attestation: SellerDestinationAttestation): SellerPayoutDestination {
    validateAttestation(attestation, this.now());
    const publicKey = this.trustedVerifierKeys.get(attestation.verifierKeyId);
    if (!publicKey) throw new SellerDestinationError("untrusted_destination_verifier", "The destination verifier is not trusted.");
    let valid = false;
    try {
      valid = verify(
        null,
        sellerDestinationSigningBytes(attestation),
        createPublicKey(publicKey),
        Buffer.from(attestation.signature, "base64url"),
      );
    } catch {
      valid = false;
    }
    if (!valid) throw new SellerDestinationError("invalid_destination_signature", "The destination attestation signature is invalid.");

    return this.database.transaction(() => {
      const existing = this.getById(attestation.destinationId);
      if (existing) {
        if (!destinationMatches(existing, attestation)) {
          throw new SellerDestinationError("destination_replay_conflict", "The destination id was reused with different details.");
        }
        return existing;
      }
      const now = this.now();
      this.database.raw.prepare(
        `UPDATE seller_payout_destinations SET status = 'revoked', revoked_at = ?
         WHERE seller_id = ? AND payout_method = ? AND status = 'active'`,
      ).run(now, attestation.sellerId, attestation.payoutMethod);
      this.database.raw.prepare(
        `INSERT INTO seller_payout_destinations(
           id, seller_id, payout_method, destination_kind, destination_reference,
           destination_fingerprint, verifier_key_id, attestation_signature, status,
           verified_at, expires_at, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)`,
      ).run(
        attestation.destinationId, attestation.sellerId, attestation.payoutMethod,
        attestation.destinationKind, attestation.destinationReference,
        attestation.destinationFingerprint, attestation.verifierKeyId,
        attestation.signature, attestation.verifiedAt, attestation.expiresAt, now,
      );
      return this.getById(attestation.destinationId)!;
    });
  }

  active(sellerId: string, method: PayoutMethod): SellerPayoutDestination | null {
    const row = this.database.raw.prepare(
      `SELECT id, seller_id, payout_method, destination_kind, destination_reference,
              destination_fingerprint, status, verified_at, expires_at, revoked_at
       FROM seller_payout_destinations
       WHERE seller_id = ? AND payout_method = ? AND status = 'active'`,
    ).get(sellerId, method) as DestinationRow | undefined;
    if (!row || Number(row.expires_at) <= this.now()) return null;
    return mapDestination(row);
  }

  revoke(sellerId: string, method: PayoutMethod): boolean {
    return this.database.raw.prepare(
      `UPDATE seller_payout_destinations SET status = 'revoked', revoked_at = ?
       WHERE seller_id = ? AND payout_method = ? AND status = 'active'`,
    ).run(this.now(), sellerId, method).changes === 1;
  }

  private getById(id: string): SellerPayoutDestination | null {
    const row = this.database.raw.prepare(
      `SELECT id, seller_id, payout_method, destination_kind, destination_reference,
              destination_fingerprint, status, verified_at, expires_at, revoked_at
       FROM seller_payout_destinations WHERE id = ?`,
    ).get(id) as DestinationRow | undefined;
    return row ? mapDestination(row) : null;
  }
}

export function sellerDestinationFingerprint(kind: string, reference: string): string {
  return createHash("sha256").update(JSON.stringify([kind, reference]), "utf8").digest("hex");
}

export function sellerDestinationSigningBytes(attestation: Omit<SellerDestinationAttestation, "signature">): Buffer {
  return Buffer.from(JSON.stringify([
    attestation.schema, attestation.destinationId, attestation.sellerId,
    attestation.payoutMethod, attestation.destinationKind, attestation.destinationReference,
    attestation.destinationFingerprint, attestation.verifiedAt, attestation.expiresAt,
    attestation.verifierKeyId,
  ]), "utf8");
}

interface DestinationRow {
  id: string; seller_id: string; payout_method: PayoutMethod;
  destination_kind: "provider_account" | "wallet"; destination_reference: string;
  destination_fingerprint: string; status: "active" | "revoked";
  verified_at: number; expires_at: number; revoked_at: number | null;
}

function mapDestination(row: DestinationRow): SellerPayoutDestination {
  return {
    id: row.id, sellerId: row.seller_id, payoutMethod: row.payout_method,
    destinationKind: row.destination_kind, destinationReference: row.destination_reference,
    destinationFingerprint: row.destination_fingerprint, status: row.status,
    verifiedAt: Number(row.verified_at), expiresAt: Number(row.expires_at),
    revokedAt: row.revoked_at === null ? null : Number(row.revoked_at),
  };
}

function validateAttestation(input: SellerDestinationAttestation, now: number): void {
  if (input.schema !== "mycellios.seller-destination.v1") throw new SellerDestinationError("unsupported_destination_schema", "The destination schema is unsupported.");
  for (const value of [input.destinationId, input.sellerId, input.verifierKeyId]) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) throw new SellerDestinationError("invalid_destination", "The destination identity is invalid.");
  }
  if (!['stable', 'spore'].includes(input.payoutMethod)
    || !['provider_account', 'wallet'].includes(input.destinationKind)
    || input.destinationReference.length < 4 || input.destinationReference.length > 512
    || /[\u0000-\u001f\u007f]/.test(input.destinationReference)) {
    throw new SellerDestinationError("invalid_destination", "The payout destination is invalid.");
  }
  if (input.destinationFingerprint !== sellerDestinationFingerprint(input.destinationKind, input.destinationReference)) {
    throw new SellerDestinationError("destination_fingerprint_mismatch", "The destination fingerprint is invalid.");
  }
  if (!Number.isSafeInteger(input.verifiedAt) || !Number.isSafeInteger(input.expiresAt)
    || input.verifiedAt > now + 60_000 || input.expiresAt <= now || input.expiresAt <= input.verifiedAt) {
    throw new SellerDestinationError("invalid_destination_validity", "The destination validity window is invalid.");
  }
  if (!/^[A-Za-z0-9_-]{16,1024}$/.test(input.signature)) throw new SellerDestinationError("invalid_destination_signature", "The destination signature encoding is invalid.");
}

function destinationMatches(stored: SellerPayoutDestination, input: SellerDestinationAttestation): boolean {
  return stored.sellerId === input.sellerId && stored.payoutMethod === input.payoutMethod
    && stored.destinationKind === input.destinationKind
    && stored.destinationReference === input.destinationReference
    && stored.destinationFingerprint === input.destinationFingerprint
    && stored.verifiedAt === input.verifiedAt && stored.expiresAt === input.expiresAt;
}

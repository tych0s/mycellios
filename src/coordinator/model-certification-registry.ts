import { modelCertificationSchema, verifyModelCertification, type ModelCertification } from "../contracts/model-certification.js";
import type { MeshDatabase } from "../storage/database.js";

export class ModelCertificationRegistry {
  private readonly keys = new Map<string, string>();
  constructor(private readonly database: MeshDatabase, pinnedKeys: readonly { keyId: string; spki: string }[]) {
    for (const key of pinnedKeys) {
      if (this.keys.has(key.keyId)) throw new Error("model_certification_key_id_is_duplicated");
      this.keys.set(key.keyId, key.spki);
    }
    if (this.keys.size === 0) throw new Error("model_certification_pinned_key_is_required");
  }

  publish(value: unknown, now = new Date()): { certification: ModelCertification; alreadyPublished: boolean } {
    const parsed = modelCertificationSchema.parse(value);
    const spki = this.keys.get(parsed.keyId);
    if (!spki) throw new Error("model_certification_key_is_not_pinned");
    const certification = verifyModelCertification(parsed, { pinnedKey: { keyId: parsed.keyId, spki }, now });
    return this.database.transaction(() => {
      const existing = this.get(certification.certificationId);
      if (existing) {
        if (JSON.stringify(existing) !== JSON.stringify(certification)) throw new Error("model_certification_identity_conflict");
        return { certification: existing, alreadyPublished: true };
      }
      this.database.raw.prepare(`INSERT INTO model_certifications(certification_id, model_family, decision, topology_kind, platform, backend, certification_json, reviewed_at, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
        .run(certification.certificationId, certification.modelFamily, certification.decision, certification.topology.kind,
          certification.hardware.platform, certification.hardware.backend, JSON.stringify(certification),
          Date.parse(certification.review.reviewedAt), Date.parse(certification.expiresAt), now.getTime());
      return { certification, alreadyPublished: false };
    });
  }

  get(certificationId: string): ModelCertification | null {
    const row = this.database.raw.prepare("SELECT certification_json FROM model_certifications WHERE certification_id = ?")
      .get(certificationId) as { certification_json: string } | undefined;
    return row ? modelCertificationSchema.parse(JSON.parse(row.certification_json)) : null;
  }

  list(limit = 100): ModelCertification[] {
    const rows = this.database.raw.prepare("SELECT certification_json FROM model_certifications ORDER BY reviewed_at DESC, certification_id DESC LIMIT ?")
      .all(Math.max(1, Math.min(100, Math.trunc(limit)))) as Array<{ certification_json: string }>;
    return rows.map((row) => modelCertificationSchema.parse(JSON.parse(row.certification_json)));
  }
}

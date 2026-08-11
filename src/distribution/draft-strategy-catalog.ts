import { createPublicKey, type KeyLike } from "node:crypto";
import { z } from "zod";

import {
  draftStrategyCertificationSchema,
  draftStrategyDescriptorIdentity,
  draftStrategyDescriptorSchema,
  type DraftStrategyDescriptor,
} from "../contracts/engine-family.js";
import type { PythonPipelineLaunchDescription } from "./python-launcher.js";
import {
  DraftStrategyResolutionError,
  NativeDraftStrategyRegistry,
  type ResolvedDraftStrategy,
} from "./draft-strategy-registry.js";

export const DRAFT_STRATEGY_CATALOG_SCHEMA =
  "mycellios-draft-strategy-catalog/1" as const;

const keyIdSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/);

export const draftStrategyCatalogSchema = z.object({
  schema: z.literal(DRAFT_STRATEGY_CATALOG_SCHEMA),
  entries: z.array(z.object({
    descriptor: draftStrategyDescriptorSchema,
    certification: draftStrategyCertificationSchema,
  }).strict()).min(1).max(1_024),
}).strict();

export const draftStrategyKeyringSchema = z.array(z.object({
  keyId: keyIdSchema,
  spki: z.string().min(1).max(512).regex(/^[A-Za-z0-9_-]+$/),
}).strict()).min(1).max(128);

export type DraftStrategyCatalogDocument = z.infer<
  typeof draftStrategyCatalogSchema
>;

export interface DraftStrategyCatalogResolutionContext {
  targetDescriptorDigest: `sha256:${string}`;
  tokenizerDigest: `sha256:${string}`;
  vocabularyDigest: `sha256:${string}`;
  backend: DraftStrategyDescriptor["resource"]["allowedBackends"][number];
  availableRamBytes: number;
  availableVramBytes: number;
  now: Date;
}

export class NativeDraftStrategyCatalog {
  readonly #registry: NativeDraftStrategyRegistry;
  readonly #descriptors: DraftStrategyDescriptor[];

  constructor(documentValue: unknown, keyringValue: unknown) {
    const document = draftStrategyCatalogSchema.parse(documentValue);
    const keys = parsePinnedKeyring(keyringValue);
    this.#registry = new NativeDraftStrategyRegistry(keys);
    this.#descriptors = [];
    const descriptorIds = new Set<string>();
    for (const entry of document.entries) {
      const descriptorId = draftStrategyDescriptorIdentity(entry.descriptor);
      if (descriptorIds.has(descriptorId)) {
        throw new DraftStrategyResolutionError(
          "draft_strategy_catalog_descriptor_is_duplicated",
        );
      }
      descriptorIds.add(descriptorId);
      this.#registry.registerDescriptor(entry.descriptor);
      this.#registry.registerCertification(entry.certification);
      this.#descriptors.push(entry.descriptor);
    }
  }

  resolveForLaunch(
    launch: PythonPipelineLaunchDescription,
    context: DraftStrategyCatalogResolutionContext,
  ): ResolvedDraftStrategy {
    const policy = launch.sourceManifest.plans.decode.speculation;
    const selected = policy.strategies.find(
      (strategy) => strategy.id === policy.defaultStrategyId,
    );
    if (!selected || selected.kind === "autoregressive" || policy.mode === "disabled") {
      throw new DraftStrategyResolutionError(
        "draft_strategy_catalog_launch_is_not_speculative",
      );
    }
    const candidates = this.#descriptors.filter(
      (descriptor) => descriptor.strategyId === selected.id,
    );
    const resolutions: ResolvedDraftStrategy[] = [];
    for (const descriptor of candidates) {
      try {
        resolutions.push(this.#registry.resolve({
          descriptorDigest: draftStrategyDescriptorIdentity(descriptor),
          targetDescriptorDigest: context.targetDescriptorDigest,
          tokenizerDigest: context.tokenizerDigest,
          vocabularyDigest: context.vocabularyDigest,
          backend: context.backend,
          availableRamBytes: context.availableRamBytes,
          availableVramBytes: context.availableVramBytes,
          requestedDraftTokens: selected.maxDraftTokens,
          requestedInflightWaves: launch.configuration.speculativeInflightWaves ?? 1,
          now: context.now,
        }));
      } catch (error) {
        if (!(error instanceof DraftStrategyResolutionError)) throw error;
      }
    }
    if (resolutions.length === 0) {
      throw new DraftStrategyResolutionError(
        "draft_strategy_catalog_has_no_compatible_entry",
      );
    }
    if (resolutions.length !== 1) {
      throw new DraftStrategyResolutionError(
        "draft_strategy_catalog_resolution_is_ambiguous",
      );
    }
    return resolutions[0]!;
  }
}

function parsePinnedKeyring(value: unknown): ReadonlyMap<string, KeyLike> {
  const keyring = draftStrategyKeyringSchema.parse(value);
  const keys = new Map<string, KeyLike>();
  for (const entry of keyring) {
    if (Buffer.from(entry.spki, "base64url").toString("base64url") !== entry.spki) {
      throw new DraftStrategyResolutionError("draft_strategy_keyring_spki_is_not_canonical");
    }
    let key;
    try {
      key = createPublicKey({
        key: Buffer.from(entry.spki, "base64url"),
        format: "der",
        type: "spki",
      });
    } catch {
      throw new DraftStrategyResolutionError("draft_strategy_keyring_key_is_invalid");
    }
    if (key.asymmetricKeyType !== "ed25519") {
      throw new DraftStrategyResolutionError("draft_strategy_keyring_key_is_not_ed25519");
    }
    if (keys.has(entry.keyId)) {
      throw new DraftStrategyResolutionError("draft_strategy_keyring_key_id_is_duplicated");
    }
    keys.set(entry.keyId, key);
  }
  return keys;
}

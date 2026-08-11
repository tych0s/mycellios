import {
  createPrivateKey,
  createPublicKey,
  sign as signBytes,
  verify as verifyBytes,
  type KeyLike,
  type KeyObject,
} from "node:crypto";
import { z } from "zod";

import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";
import { MAX_COMPONENT_ARTIFACT_BYTES } from "./component-update-policy.js";

export const COMPONENT_UPDATE_MANIFEST_SCHEMA =
  "mycellios-component-update/3" as const;

const channelSchema = z.enum(["dev", "stable"]);
const sequenceSchema = z.number().int().positive().safe();
const revisionSchema = z.string().regex(/^[0-9a-f]{40}$/);
const sha256IdentitySchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const sourceIdSchema = sha256IdentitySchema;
const manifestIdSchema = sha256IdentitySchema;
const keyIdSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const boundedIdentifierSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9][A-Za-z0-9._+/-]*$/);
const semanticVersionSchema = z
  .string()
  .max(128)
  .regex(
    /^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/,
  );

const workerProtocolSchema = z
  .object({
    min: z.number().int().min(1).max(65_535),
    max: z.number().int().min(1).max(65_535),
  })
  .strict()
  .refine(
    ({ min, max }) => min <= max,
    "component_update_worker_protocol_range_is_invalid",
  );

const compatibilitySchema = z
  .object({
    workerProtocol: workerProtocolSchema,
    runtimeAbi: boundedIdentifierSchema,
    minBootstrapVersion: semanticVersionSchema,
  })
  .strict();
const sourceProvenanceSchema = z
  .object({
    baseRevision: revisionSchema,
    sourceTreeDirty: z.boolean(),
    sourceTreeDigest: sha256IdentitySchema,
  })
  .strict();

const httpsArtifactUrlSchema = z
  .string()
  .url()
  .max(4_096)
  .refine(isSafeArtifactUrl, "component_update_artifact_url_is_invalid");

const artifactSchema = z
  .object({
    url: httpsArtifactUrlSchema,
    sha256: sha256IdentitySchema,
    bytes: z
      .number()
      .int()
      .positive()
      .max(MAX_COMPONENT_ARTIFACT_BYTES),
    format: z.enum([
      "json-gzip-v1",
      "raw",
      "zip",
      "tar.gz",
      "tar.zst",
      "nupkg",
    ]),
    filesManifestSha256: sha256IdentitySchema.optional(),
  })
  .strict();

const componentDependencySchema = z
  .object({
    id: z
      .string()
      .min(1)
      .max(128)
      .regex(/^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/),
    minVersion: semanticVersionSchema,
    maxVersionExclusive: semanticVersionSchema.optional(),
  })
  .strict()
  .refine(
    ({ minVersion, maxVersionExclusive }) =>
      maxVersionExclusive === undefined
      || compareSemanticVersions(minVersion, maxVersionExclusive) < 0,
    "component_update_dependency_version_range_is_invalid",
  );

const componentRequirementsSchema = z
  .object({
    backend: z.enum(["any", "cpu", "cuda", "rocm", "metal", "vulkan", "webgpu"]),
    driver: z
      .object({
        api: boundedIdentifierSchema,
        minVersion: semanticVersionSchema.optional(),
        maxVersionExclusive: semanticVersionSchema.optional(),
      })
      .strict()
      .refine(
        ({ minVersion, maxVersionExclusive }) =>
          minVersion === undefined
          || maxVersionExclusive === undefined
          || compareSemanticVersions(minVersion, maxVersionExclusive) < 0,
        "component_update_driver_version_range_is_invalid",
      )
      .nullable(),
    runtimeAbi: boundedIdentifierSchema,
    workerProtocol: workerProtocolSchema,
    dependencies: z
      .array(componentDependencySchema)
      .max(64)
      .superRefine((dependencies, context) => {
        const seen = new Set<string>();
        for (let index = 0; index < dependencies.length; index += 1) {
          const id = dependencies[index]!.id;
          if (seen.has(id)) {
            context.addIssue({
              code: "custom",
              message: "component_update_dependency_is_duplicated",
              path: [index],
            });
          }
          seen.add(id);
          if (index > 0 && dependencies[index - 1]!.id.localeCompare(id) > 0) {
            context.addIssue({
              code: "custom",
              message: "component_update_dependencies_are_not_canonical",
              path: [index],
            });
          }
        }
      }),
  })
  .strict();

const componentSchema = z
  .object({
    id: z
      .string()
      .min(1)
      .max(128)
      .regex(/^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/),
    version: boundedIdentifierSchema,
    platform: z.enum(["any", "win32", "linux", "darwin"]),
    arch: z.enum(["any", "x64", "arm64"]),
    artifact: artifactSchema,
    requirements: componentRequirementsSchema,
    restartScope: z.enum(["none", "runtime", "agent", "application"]),
  })
  .strict();

const componentsSchema = z
  .array(componentSchema)
  .min(1)
  .max(128)
  .superRefine((components, context) => {
    const seen = new Set<string>();
    for (let index = 0; index < components.length; index += 1) {
      const key = componentSortKey(components[index]!);
      if (seen.has(key)) {
        context.addIssue({
          code: "custom",
          message: "component_update_component_target_is_duplicated",
          path: [index],
        });
      }
      seen.add(key);
    }
  });

const canonicalComponentsSchema = componentsSchema.superRefine(
  (components, context) => {
    for (let index = 1; index < components.length; index += 1) {
      if (
        compareComponentTargets(components[index - 1]!, components[index]!) > 0
      ) {
        context.addIssue({
          code: "custom",
          message: "component_update_components_are_not_canonical",
          path: [index],
        });
        return;
      }
    }
  },
);

const unsignedManifestShape = {
  schema: z.literal(COMPONENT_UPDATE_MANIFEST_SCHEMA),
  channel: channelSchema,
  sequence: sequenceSchema,
  revision: revisionSchema,
  provenance: sourceProvenanceSchema,
  sourceId: sourceIdSchema,
  compatibility: compatibilitySchema,
  components: canonicalComponentsSchema,
} as const;

const buildInputSchema = z
  .object({
    channel: channelSchema,
    sequence: sequenceSchema,
    revision: revisionSchema,
    provenance: sourceProvenanceSchema,
    sourceId: sourceIdSchema,
    compatibility: compatibilitySchema,
    components: componentsSchema,
  })
  .strict();

export const unsignedComponentUpdateManifestSchema = z
  .object(unsignedManifestShape)
  .strict();

const signatureSchema = z
  .string()
  .max(128)
  .refine(
    (value) => canonicalBase64urlBytes(value, 64) !== null,
    "component_update_manifest_signature_is_invalid",
  );

export const componentUpdateManifestSchema = z
  .object({
    ...unsignedManifestShape,
    manifestId: manifestIdSchema,
    keyId: keyIdSchema,
    signature: signatureSchema,
  })
  .strict();

const pinnedKeySchema = z
  .object({
    keyId: keyIdSchema,
    spki: z
      .string()
      .max(2_048)
      .refine(
        (value) => canonicalBase64urlBytes(value) !== null,
        "component_update_manifest_pinned_key_is_invalid",
      ),
  })
  .strict();

export type ComponentUpdateChannel = z.infer<typeof channelSchema>;
export type ComponentUpdateCompatibility = z.infer<
  typeof compatibilitySchema
>;
export type ComponentUpdateArtifact = z.infer<typeof artifactSchema>;
export type ComponentUpdateRequirements = z.infer<typeof componentRequirementsSchema>;
export type ComponentUpdateComponent = z.infer<typeof componentSchema>;
export type UnsignedComponentUpdateManifest = z.infer<
  typeof unsignedComponentUpdateManifestSchema
>;
export type ComponentUpdateManifest = z.infer<
  typeof componentUpdateManifestSchema
>;
export type ComponentUpdateManifestBuildInput = z.input<
  typeof buildInputSchema
>;

export interface ComponentUpdateManifestSigningOptions {
  keyId: string;
  privateKey: KeyLike;
}

export interface PinnedComponentUpdateKey {
  keyId: string;
  /**
   * Canonical base64url encoding of a DER SubjectPublicKeyInfo document.
   * This value comes from the bootstrap trust store, never from a manifest.
   */
  spki: string;
}

export interface ComponentUpdateManifestVerificationOptions {
  pinnedKey: PinnedComponentUpdateKey;
  expectedChannel?: ComponentUpdateChannel | undefined;
  /**
   * Lowest sequence already accepted for this channel. Equality is allowed so
   * an interrupted, idempotent install can verify the same manifest again.
   */
  minimumSequence?: number | undefined;
}

/**
 * Builds the one canonical unsigned representation. Callers may supply
 * components in any order; the signed representation is always target-sorted.
 */
export function buildComponentUpdateManifest(
  input: ComponentUpdateManifestBuildInput,
): UnsignedComponentUpdateManifest {
  const parsed = buildInputSchema.parse(input);
  const components = [...parsed.components].sort(compareComponentTargets);
  return unsignedComponentUpdateManifestSchema.parse({
    schema: COMPONENT_UPDATE_MANIFEST_SCHEMA,
    channel: parsed.channel,
    sequence: parsed.sequence,
    revision: parsed.revision,
    provenance: parsed.provenance,
    sourceId: parsed.sourceId,
    compatibility: parsed.compatibility,
    components,
  });
}

export function computeComponentUpdateManifestId(
  manifest: UnsignedComponentUpdateManifest,
): `sha256:${string}` {
  const parsed = unsignedComponentUpdateManifestSchema.parse(manifest);
  return sha256CanonicalEvidence(parsed) as `sha256:${string}`;
}

/**
 * Signs the manifest identity plus the external key selector. The public key
 * is deliberately absent: verification authority belongs to the bootstrap's
 * pinned trust store.
 */
export function signComponentUpdateManifest(
  manifest: UnsignedComponentUpdateManifest,
  options: ComponentUpdateManifestSigningOptions,
): ComponentUpdateManifest {
  const parsed = unsignedComponentUpdateManifestSchema.parse(manifest);
  const keyId = keyIdSchema.parse(options.keyId);
  const privateKey = ed25519PrivateKey(options.privateKey);
  const manifestId = computeComponentUpdateManifestId(parsed);
  const signingDocument = {
    ...parsed,
    manifestId,
    keyId,
  };
  const signature = signBytes(
    null,
    Buffer.from(canonicalEvidenceJson(signingDocument), "utf8"),
    privateKey,
  ).toString("base64url");
  return componentUpdateManifestSchema.parse({
    ...signingDocument,
    signature,
  });
}

/**
 * Parses the strict v2 envelope and checks its content address. This does not
 * establish publisher authenticity; use verifyComponentUpdateManifest for
 * trust decisions.
 */
export function parseComponentUpdateManifest(
  value: unknown,
): ComponentUpdateManifest {
  const manifest = componentUpdateManifestSchema.parse(value);
  const expectedId = computeComponentUpdateManifestId(unsignedPart(manifest));
  if (manifest.manifestId !== expectedId) {
    throw new Error("component_update_manifest_identity_mismatch");
  }
  return manifest;
}

/**
 * Verifies only with the SPKI selected by the caller's pinned trust store.
 * No field in the untrusted manifest can introduce or replace a trust root.
 */
export function verifyComponentUpdateManifest(
  value: unknown,
  options: ComponentUpdateManifestVerificationOptions,
): ComponentUpdateManifest {
  const manifest = parseComponentUpdateManifest(value);
  const pinnedKey = pinnedKeySchema.parse(options.pinnedKey);

  if (manifest.keyId !== pinnedKey.keyId) {
    throw new Error("component_update_manifest_key_id_mismatch");
  }
  if (
    options.expectedChannel !== undefined
    && manifest.channel !== channelSchema.parse(options.expectedChannel)
  ) {
    throw new Error("component_update_manifest_channel_mismatch");
  }
  if (options.minimumSequence !== undefined) {
    const minimumSequence = z
      .number()
      .int()
      .nonnegative()
      .safe()
      .parse(options.minimumSequence);
    if (manifest.sequence < minimumSequence) {
      throw new Error("component_update_manifest_rollback_rejected");
    }
  }

  const publicKey = ed25519PublicKeyFromSpki(pinnedKey.spki);
  const verified = verifyBytes(
    null,
    Buffer.from(
      canonicalEvidenceJson(signingPart(manifest)),
      "utf8",
    ),
    publicKey,
    Buffer.from(manifest.signature, "base64url"),
  );
  if (!verified) {
    throw new Error("component_update_manifest_signature_verification_failed");
  }
  return manifest;
}

function unsignedPart(
  manifest: ComponentUpdateManifest,
): UnsignedComponentUpdateManifest {
  return {
    schema: manifest.schema,
    channel: manifest.channel,
    sequence: manifest.sequence,
    revision: manifest.revision,
    provenance: manifest.provenance,
    sourceId: manifest.sourceId,
    compatibility: manifest.compatibility,
    components: manifest.components,
  };
}

function signingPart(manifest: ComponentUpdateManifest): {
  schema: typeof COMPONENT_UPDATE_MANIFEST_SCHEMA;
  channel: ComponentUpdateChannel;
  sequence: number;
  revision: string;
  provenance: ComponentUpdateManifest["provenance"];
  sourceId: string;
  compatibility: ComponentUpdateCompatibility;
  components: ComponentUpdateComponent[];
  manifestId: string;
  keyId: string;
} {
  return {
    ...unsignedPart(manifest),
    manifestId: manifest.manifestId,
    keyId: manifest.keyId,
  };
}

function ed25519PrivateKey(value: KeyLike): KeyObject {
  let key: KeyObject;
  try {
    key = isKeyObject(value) ? value : createPrivateKey(value);
  } catch {
    throw new Error("component_update_manifest_private_key_is_invalid");
  }
  if (key.type !== "private" || key.asymmetricKeyType !== "ed25519") {
    throw new Error("component_update_manifest_private_key_is_not_ed25519");
  }
  return key;
}

function ed25519PublicKeyFromSpki(spki: string): KeyObject {
  let key: KeyObject;
  const encoded = Buffer.from(spki, "base64url");
  try {
    key = createPublicKey({
      key: encoded,
      format: "der",
      type: "spki",
    });
  } catch {
    throw new Error("component_update_manifest_pinned_key_is_invalid");
  }
  const canonical = key.export({ format: "der", type: "spki" });
  if (
    key.asymmetricKeyType !== "ed25519"
    || encoded.length > 128
    || !Buffer.isBuffer(canonical)
    || !canonical.equals(encoded)
  ) {
    throw new Error("component_update_manifest_pinned_key_is_not_ed25519");
  }
  return key;
}

function isKeyObject(value: KeyLike): value is KeyObject {
  return (
    typeof value === "object"
    && value !== null
    && "type" in value
    && "export" in value
    && typeof (value as { export?: unknown }).export === "function"
  );
}

function compareSemanticVersions(left: string, right: string): number {
  const parse = (value: string): [number, number, number] => {
    const [core] = value.split("-", 1);
    const [major, minor, patch] = core!.split(".").map(Number);
    return [major!, minor!, patch!];
  };
  const a = parse(left);
  const b = parse(right);
  for (let index = 0; index < 3; index += 1) {
    if (a[index] !== b[index]) return a[index]! - b[index]!;
  }
  return 0;
}

function canonicalBase64urlBytes(
  value: string,
  expectedBytes?: number,
): Buffer | null {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) return null;
  const decoded = Buffer.from(value, "base64url");
  if (
    decoded.length === 0
    || decoded.toString("base64url") !== value
    || (expectedBytes !== undefined && decoded.length !== expectedBytes)
  ) {
    return null;
  }
  return decoded;
}

function isSafeArtifactUrl(value: string): boolean {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    const loopback =
      host === "localhost"
      || host === "127.0.0.1"
      || host === "::1";
    return (
      (url.protocol === "https:" || (url.protocol === "http:" && loopback))
      && url.hostname.length > 0
      && url.username === ""
      && url.password === ""
      && url.hash === ""
    );
  } catch {
    return false;
  }
}

function componentSortKey(
  component: Pick<ComponentUpdateComponent, "id" | "platform" | "arch">,
): string {
  return `${component.id}\u0000${component.platform}\u0000${component.arch}`;
}

function compareComponentTargets(
  left: Pick<ComponentUpdateComponent, "id" | "platform" | "arch">,
  right: Pick<ComponentUpdateComponent, "id" | "platform" | "arch">,
): number {
  const leftKey = componentSortKey(left);
  const rightKey = componentSortKey(right);
  return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
}

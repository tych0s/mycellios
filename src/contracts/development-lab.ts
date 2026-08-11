import { isIP } from "node:net";
import { z } from "zod";

import { canonicalEvidenceJson } from "../core/json.js";

export const DEVELOPMENT_LAB_INVITATION_SCHEMA =
  "mycellios-development-lab-invitation/1" as const;
export const DEVELOPMENT_LAB_INVITATION_PREFIX =
  "mycellios-dev-lab:" as const;

const MAX_INVITATION_CODE_BYTES = 4_096;
const LAB_ID_PATTERN = /^lab_[A-Za-z0-9_-]{22}$/;
const INVITATION_TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const KEY_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export const developmentLabIdSchema = z
  .string()
  .regex(LAB_ID_PATTERN);

export const developmentLabInvitationTokenSchema = z
  .string()
  .regex(INVITATION_TOKEN_PATTERN)
  .refine((value) => canonicalBase64url(value, 32));

export const developmentLabPublicKeySchema = z
  .object({
    keyId: z.string().min(1).max(128).regex(KEY_ID_PATTERN),
    spki: z
      .string()
      .min(1)
      .max(2_048)
      .regex(/^[A-Za-z0-9_-]+$/),
  })
  .strict();

export const developmentLabCreateRequestSchema = z
  .object({
    name: z.string().trim().min(1).max(128).optional(),
    keyId: developmentLabPublicKeySchema.shape.keyId,
    spki: developmentLabPublicKeySchema.shape.spki,
    invitationTtlSeconds: z.number().int().min(60).max(7 * 24 * 60 * 60)
      .optional(),
  })
  .strict();

export const developmentLabReinviteRequestSchema = z
  .object({
    ttlSeconds: z.number().int().min(60).max(7 * 24 * 60 * 60).optional(),
  })
  .strict();

export const developmentLabRedeemRequestSchema = z
  .object({
    labId: developmentLabIdSchema,
    token: developmentLabInvitationTokenSchema,
  })
  .strict();

export const developmentLabInvitationSchema = z
  .object({
    schema: z.literal(DEVELOPMENT_LAB_INVITATION_SCHEMA),
    coordinatorUrl: z.string().url().max(2_048),
    labId: developmentLabIdSchema,
    token: developmentLabInvitationTokenSchema,
    expiresAt: z.string().datetime({ offset: true }),
  })
  .strict();

export const developmentLabSummarySchema = z
  .object({
    labId: developmentLabIdSchema,
    name: z.string().min(1).max(128),
    keyId: developmentLabPublicKeySchema.shape.keyId,
    spki: developmentLabPublicKeySchema.shape.spki,
    status: z.enum(["active", "revoked"]),
    createdAt: z.string().datetime({ offset: true }),
    updatedAt: z.string().datetime({ offset: true }),
    revokedAt: z.string().datetime({ offset: true }).nullable(),
  })
  .strict();

export const developmentLabCreateResponseSchema = z
  .object({
    data: developmentLabSummarySchema.extend({
      feedBaseUrl: z.string().url().max(2_048),
      invitation: z
        .object({
          token: developmentLabInvitationTokenSchema,
          expiresAt: z.string().datetime({ offset: true }),
        })
        .strict(),
    }),
  })
  .strict();

export const developmentLabRedeemResponseSchema = z
  .object({
    data: developmentLabSummarySchema.extend({
      feedBaseUrl: z.string().url().max(2_048),
    }),
  })
  .strict();

export type DevelopmentLabInvitation = z.infer<
  typeof developmentLabInvitationSchema
>;
export type DevelopmentLabCreateRequest = z.infer<
  typeof developmentLabCreateRequestSchema
>;
export type DevelopmentLabCreateResponse = z.infer<
  typeof developmentLabCreateResponseSchema
>;
export type DevelopmentLabRedeemRequest = z.infer<
  typeof developmentLabRedeemRequestSchema
>;
export type DevelopmentLabRedeemResponse = z.infer<
  typeof developmentLabRedeemResponseSchema
>;
export type DevelopmentLabSummary = z.infer<
  typeof developmentLabSummarySchema
>;

export interface DevelopmentLabInvitationCodecOptions {
  /**
   * Production enrollment requires HTTPS. Tests and an explicitly local
   * developer coordinator may opt into HTTP only for a loopback hostname.
   */
  allowLoopbackHttp?: boolean;
}

export function encodeDevelopmentLabInvitation(
  input: DevelopmentLabInvitation,
  options: DevelopmentLabInvitationCodecOptions = {},
): string {
  const invitation = validateDevelopmentLabInvitation(input, options);
  const encoded = Buffer
    .from(canonicalEvidenceJson(invitation), "utf8")
    .toString("base64url");
  const code = `${DEVELOPMENT_LAB_INVITATION_PREFIX}${encoded}`;
  if (Buffer.byteLength(code, "utf8") > MAX_INVITATION_CODE_BYTES) {
    throw new Error("development_lab_invitation_too_large");
  }
  return code;
}

export function parseDevelopmentLabInvitation(
  code: string,
  options: DevelopmentLabInvitationCodecOptions = {},
): DevelopmentLabInvitation {
  if (
    typeof code !== "string"
    || Buffer.byteLength(code, "utf8") > MAX_INVITATION_CODE_BYTES
    || !code.startsWith(DEVELOPMENT_LAB_INVITATION_PREFIX)
  ) {
    throw new Error("development_lab_invitation_invalid");
  }
  const encoded = code.slice(DEVELOPMENT_LAB_INVITATION_PREFIX.length);
  if (!canonicalBase64url(encoded)) {
    throw new Error("development_lab_invitation_invalid");
  }
  let candidate: unknown;
  try {
    candidate = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"));
  } catch {
    throw new Error("development_lab_invitation_invalid");
  }
  const invitation = validateDevelopmentLabInvitation(candidate, options);
  const canonical = Buffer
    .from(canonicalEvidenceJson(invitation), "utf8")
    .toString("base64url");
  if (canonical !== encoded) {
    throw new Error("development_lab_invitation_not_canonical");
  }
  return invitation;
}

function validateDevelopmentLabInvitation(
  input: unknown,
  options: DevelopmentLabInvitationCodecOptions,
): DevelopmentLabInvitation {
  const invitation = developmentLabInvitationSchema.parse(input);
  let coordinator: URL;
  try {
    coordinator = new URL(invitation.coordinatorUrl);
  } catch {
    throw new Error("development_lab_coordinator_url_invalid");
  }
  const loopback = isLoopbackHostname(coordinator.hostname);
  if (
    coordinator.username !== ""
    || coordinator.password !== ""
    || coordinator.search !== ""
    || coordinator.hash !== ""
    || (
      coordinator.protocol !== "https:"
      && !(
        options.allowLoopbackHttp === true
        && coordinator.protocol === "http:"
        && loopback
      )
    )
  ) {
    throw new Error("development_lab_coordinator_url_invalid");
  }
  if (Date.parse(invitation.expiresAt) <= 0) {
    throw new Error("development_lab_invitation_expiry_invalid");
  }
  return {
    ...invitation,
    coordinatorUrl: coordinator.toString().replace(/\/+$/, ""),
  };
}

function canonicalBase64url(value: string, expectedBytes?: number): boolean {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) return false;
  const bytes = Buffer.from(value, "base64url");
  return (
    bytes.length > 0
    && bytes.toString("base64url") === value
    && (expectedBytes === undefined || bytes.length === expectedBytes)
  );
}

function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[(.*)\]$/, "$1");
  if (normalized === "localhost" || normalized === "::1") return true;
  if (isIP(normalized) === 4) {
    return normalized.split(".", 1)[0] === "127";
  }
  if (isIP(normalized) === 6 && normalized.startsWith("::ffff:")) {
    const mapped = normalized.slice("::ffff:".length);
    return isIP(mapped) === 4 && mapped.split(".", 1)[0] === "127";
  }
  return false;
}

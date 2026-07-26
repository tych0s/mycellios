import { z } from "zod";

export const NATIVE_BUILD_PROVENANCE_SCHEMA =
  "mycellios-native-build-provenance/1" as const;
export const NATIVE_BUILD_PROVENANCE_FILE =
  "mycellios-native-build-provenance.json";

const strongSourceIdSchema = z
  .string()
  .regex(/^sha256:[0-9a-f]{64}$/)
  .transform((value): `sha256:${string}` => value as `sha256:${string}`);

export const nativeBuildIdentitySchema = z
  .object({
    schema: z.literal(NATIVE_BUILD_PROVENANCE_SCHEMA),
    version: z.string().min(1).max(128),
    sourceId: strongSourceIdSchema,
  })
  .strict();

export const nativeBuildProvenanceDocumentSchema = nativeBuildIdentitySchema
  .extend({
    files: z
      .array(
        z
          .object({
            path: z
              .string()
              .min(1)
              .max(4_096)
              .refine(isSafePortablePath, "Invalid portable source path"),
            bytes: z.number().int().nonnegative().safe(),
            sha256: z.string().regex(/^[0-9a-f]{64}$/),
          })
          .strict(),
      )
      .min(1),
  })
  .strict();

export type NativeBuildIdentity = z.infer<typeof nativeBuildIdentitySchema>;
export type NativeBuildProvenanceDocument = z.infer<
  typeof nativeBuildProvenanceDocumentSchema
>;

function isSafePortablePath(portable: string): boolean {
  if (
    portable.includes("\\")
    || portable.startsWith("/")
    || portable.endsWith("/")
    || /^[A-Za-z]:/u.test(portable)
    || /[\u0000-\u001f\u007f]/u.test(portable)
  ) {
    return false;
  }
  return portable
    .split("/")
    .every((segment) => segment.length > 0 && segment !== "." && segment !== "..");
}

import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { z } from "zod";

const revision = z.string().regex(/^[a-f0-9]{40}$/);
const packageManifestSchema = z.union([z.object({
  schema: z.enum(["mycellios-node-package/1", "mycellios-node-package/2"]),
  sourceRevision: revision,
}).passthrough(), z.object({
  schema: z.literal("mycellios-node-runtime-identity/1"), sourceRevision: revision,
  version: z.string().min(1), sourceId: z.string().regex(/^sha256:[a-f0-9]{64}$/), provenanceSha256: z.string().regex(/^[a-f0-9]{64}$/),
}).strict()]);

/** Resolve only a CI-sealed, packaged, or real checkout revision. */
export async function resolveNodeSourceRevision(
  environment: NodeJS.ProcessEnv = process.env,
  execute: typeof execFileSync = execFileSync,
): Promise<string> {
  const explicit = environment.MYCELLIOS_SOURCE_REVISION?.trim();
  if (explicit) return revision.parse(explicit);
  try {
    const document = JSON.parse(await readFile(new URL("../manifest.json", import.meta.url), "utf8"));
    return packageManifestSchema.parse(document).sourceRevision;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw new Error("mycellios_node_build_manifest_is_invalid", { cause: error });
  }
  try {
    return revision.parse(execute("git", ["rev-parse", "--verify", "HEAD"], { encoding: "utf8" }).trim());
  } catch (error) {
    throw new Error("mycellios_node_source_revision_is_unavailable", { cause: error });
  }
}

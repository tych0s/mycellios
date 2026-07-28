import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  NATIVE_BUILD_PROVENANCE_FILE,
  buildNativeSourceProvenance,
} from "./native-build-provenance.mjs";

export function prepareCoordinatorRuntimeIdentity(
  sourceRoot,
  runtimeRoot,
  revision,
) {
  const normalizedRevision = String(revision ?? "").trim().toLowerCase();
  if (!/^[0-9a-f]{40}$/.test(normalizedRevision)) {
    throw new Error(
      "Coordinator runtime identity requires an exact 40-character Git SHA.",
    );
  }

  const source = resolve(sourceRoot);
  const runtime = resolve(runtimeRoot);
  const provenance = buildNativeSourceProvenance(source);
  writeFileSync(resolve(runtime, "REVISION"), `${normalizedRevision}\n`, "utf8");
  writeFileSync(
    resolve(runtime, NATIVE_BUILD_PROVENANCE_FILE),
    `${JSON.stringify(provenance, null, 2)}\n`,
    "utf8",
  );
  return {
    revision: normalizedRevision,
    sourceId: provenance.sourceId,
    version: provenance.version,
  };
}

function cliRevision() {
  return process.argv
    .slice(2)
    .find((argument) => argument.startsWith("--revision="))
    ?.slice("--revision=".length);
}

if (
  process.argv[1]
  && import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  const workspace = resolve(import.meta.dirname, "..");
  const identity = prepareCoordinatorRuntimeIdentity(
    workspace,
    workspace,
    cliRevision(),
  );
  process.stdout.write(
    `Coordinator runtime identity ready: ${identity.revision} / ${identity.sourceId}\n`,
  );
}

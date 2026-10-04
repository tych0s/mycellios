import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { verifyNodeReleaseEvidence } from "./verify-node-release-evidence.mjs";

const TARGETS = Object.freeze([
  { target: "linux-x64", extension: "deb" },
  { target: "macos-arm64", extension: "pkg" },
  { target: "windows-x64", extension: "msi" },
]);
const VERSION = /^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/;
const REVISION = /^[0-9a-f]{40}$/;
const SOURCE_ID = /^sha256:[0-9a-f]{64}$/;

export async function assembleNodeReleaseAssets(input) {
  if (!VERSION.test(input.version)) throw new Error("node_release_version_invalid");
  if (!REVISION.test(input.revision)) throw new Error("node_release_revision_invalid");
  if (!SOURCE_ID.test(input.sourceId)) throw new Error("node_release_source_id_invalid");
  if (typeof input.publishedAt !== "string"
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(input.publishedAt)
    || new Date(input.publishedAt).toISOString() !== input.publishedAt) {
    throw new Error("node_release_published_at_invalid");
  }

  const artifactsRoot = resolve(input.artifactsRoot), outputRoot = resolve(input.outputRoot);
  if (outputRoot === artifactsRoot || outputRoot.startsWith(`${artifactsRoot}${sep}`)
    || artifactsRoot.startsWith(`${outputRoot}${sep}`)) {
    throw new Error("node_release_input_and_output_overlap");
  }
  await assertDirectory(artifactsRoot, "node_release_artifacts_root_invalid");
  if (await lstat(outputRoot).catch((error) => missingOnly(error))) {
    throw new Error("node_release_output_already_exists");
  }

  const packages = [];
  let builder = null;
  let signatureState = null;
  for (const { target, extension } of TARGETS) {
    const jobRoot = join(artifactsRoot, `mycellios-node-${target}`);
    const name = `mycellios-node-${input.version}-${target}.${extension}`;
    const artifact = join(jobRoot, name);
    const stagedRoot = join(jobRoot, "node-package", target);
    const evidenceRoot = join(jobRoot, "evidence", target);
    const evidencePath = join(evidenceRoot, `${name}.provenance.json`);
    const sbomPath = join(evidenceRoot, `${name}.cdx.json`);
    const checksumPath = join(evidenceRoot, `${name}.sha256`);
    await Promise.all([
      assertDirectory(jobRoot, `node_release_job_root_invalid:${target}`),
      assertDirectory(stagedRoot, `node_release_staged_root_invalid:${target}`),
      assertDirectory(evidenceRoot, `node_release_evidence_root_invalid:${target}`),
      ...[artifact, evidencePath, sbomPath, checksumPath].map((path) =>
        assertFile(path, `node_release_file_invalid:${target}:${basename(path)}`)),
    ]);
    const evidence = await verifyNodeReleaseEvidence({
      artifact, stagedRoot, evidencePath, sbomPath, checksumPath,
    });
    if (evidence.source?.revision !== input.revision
      || evidence.source?.sourceId !== input.sourceId
      || evidence.source?.version !== input.version
      || evidence.artifact?.name !== name
      || evidence.artifact?.format !== extension
      || evidence.artifact?.target !== target) {
      throw new Error(`node_release_source_or_target_mismatch:${target}`);
    }
    if (!evidence.builder?.workflow || !evidence.builder?.runId || !evidence.builder?.runAttempt) {
      throw new Error(`node_release_builder_identity_missing:${target}`);
    }
    const thisBuilder = JSON.stringify([
      evidence.builder?.workflow,
      evidence.builder?.runId,
      evidence.builder?.runAttempt,
    ]);
    if (builder !== null && builder !== thisBuilder) {
      throw new Error(`node_release_builder_run_mismatch:${target}`);
    }
    builder = thisBuilder;
    if (!new Set(["pending", "signed"]).has(evidence.signature?.state)) {
      throw new Error(`node_release_signature_state_invalid:${target}`);
    }
    if (signatureState !== null && signatureState !== evidence.signature.state) {
      throw new Error(`node_release_signature_states_mixed:${target}`);
    }
    signatureState = evidence.signature.state;
    packages.push({
      target, name, bytes: evidence.artifact.bytes,
      sha256: evidence.artifact.sha256, artifact,
    });
  }

  const parent = dirname(outputRoot);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(join(parent, ".mycellios-release-assets-"));
  try {
    for (const item of packages) {
      const destination = join(temporary, item.name);
      await copyFile(item.artifact, destination);
      if (await sha256File(destination) !== item.sha256
        || (await lstat(destination)).size !== item.bytes) {
        throw new Error(`node_release_copy_mismatch:${item.target}`);
      }
    }
    const feedName = "mycellios-node-latest.json";
    const feed = {
      schema: "mycellios-node-update-feed/1",
      version: input.version,
      publishedAt: input.publishedAt,
      packages: packages.map(({ target, name, bytes, sha256 }) => ({ target, name, bytes, sha256 })),
    };
    await writeFile(join(temporary, feedName), `${JSON.stringify(feed, null, 2)}\n`, { flag: "wx" });
    const checksums = [
      ...packages.map(({ name, sha256 }) => ({ name, sha256 })),
      { name: feedName, sha256: await sha256File(join(temporary, feedName)) },
    ].sort((left, right) => left.name.localeCompare(right.name, "en"));
    await writeFile(join(temporary, "sha256sums.txt"),
      `${checksums.map(({ sha256, name }) => `${sha256}  ${name}`).join("\n")}\n`, { flag: "wx" });
    if (await lstat(outputRoot).catch((error) => missingOnly(error))) {
      throw new Error("node_release_output_already_exists");
    }
    await rename(temporary, outputRoot);
    return {
      outputRoot, version: input.version, revision: input.revision,
      sourceId: input.sourceId, signatureState, packages: feed.packages,
    };
  } catch (error) {
    if (!resolve(temporary).startsWith(`${resolve(parent)}${sep}`)
      || !basename(temporary).startsWith(".mycellios-release-assets-")) {
      throw new Error("node_release_temporary_cleanup_path_unsafe", { cause: error });
    }
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

async function assertDirectory(path, error) {
  const details = await lstat(path).catch((caught) => missingOnly(caught));
  if (!details?.isDirectory() || details.isSymbolicLink()) throw new Error(error);
}

async function assertFile(path, error) {
  const details = await lstat(path).catch((caught) => missingOnly(caught));
  if (!details?.isFile() || details.isSymbolicLink() || details.size < 1) throw new Error(error);
}

function missingOnly(error) {
  if (error.code === "ENOENT") return null;
  throw error;
}

async function sha256File(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

function parseArguments(argv) {
  const values = new Map();
  for (const argument of argv) {
    const match = /^--([a-z-]+)=(.+)$/.exec(argument);
    if (!match || values.has(match[1])) throw new Error(`node_release_argument_invalid:${argument}`);
    values.set(match[1], match[2]);
  }
  const expected = ["artifacts", "output", "published-at", "revision", "source-id", "version"];
  if (JSON.stringify([...values.keys()].sort()) !== JSON.stringify(expected)) {
    throw new Error("node_release_arguments_incomplete_or_unknown");
  }
  return {
    artifactsRoot: values.get("artifacts"), outputRoot: values.get("output"),
    publishedAt: values.get("published-at"), revision: values.get("revision"),
    sourceId: values.get("source-id"), version: values.get("version"),
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const result = await assembleNodeReleaseAssets(parseArguments(process.argv.slice(2)));
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

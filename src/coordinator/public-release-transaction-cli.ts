import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  buildReleaseTransactionManifest,
  expectedPublicReleaseAssets,
  type ReleaseAssetEvidence,
  type ReleaseTransactionManifest,
} from "./release-upload.js";

export interface PreparePublicReleaseTransactionInput {
  assetsRoot: string;
  outputPath: string;
  transactionId: string;
  sourceId: string;
  revision: string;
  version: string;
}

const SHA256_SUMS = "sha256sums.txt";

export async function preparePublicReleaseTransaction(
  input: PreparePublicReleaseTransactionInput,
): Promise<ReleaseTransactionManifest> {
  const assetsRoot = resolve(input.assetsRoot);
  const rootDetails = await lstat(assetsRoot);
  if (!rootDetails.isDirectory() || rootDetails.isSymbolicLink()) {
    throw new Error("public_release_assets_root_is_not_a_real_directory");
  }

  const expected = expectedPublicReleaseAssets(input.version);
  const allowedNames = new Set([
    ...expected.map(({ fileName }) => fileName),
    SHA256_SUMS,
  ]);
  const observedNames = (await readdir(assetsRoot)).sort();
  const unexpected = observedNames.filter((name) => !allowedNames.has(name));
  if (unexpected.length > 0) {
    throw new Error(
      `public_release_assets_contain_unexpected_files:${unexpected.join(",")}`,
    );
  }

  const assets: ReleaseAssetEvidence[] = [];
  for (const expectedAsset of expected) {
    const path = resolve(assetsRoot, expectedAsset.fileName);
    const details = await lstat(path);
    if (!details.isFile() || details.isSymbolicLink() || details.size < 1) {
      throw new Error(
        `public_release_asset_is_not_a_nonempty_regular_file:${expectedAsset.fileName}`,
      );
    }
    assets.push({
      ...expectedAsset,
      fileSize: details.size,
      fileSha256: await sha256File(path),
    });
  }
  if (observedNames.includes(SHA256_SUMS)) {
    await verifyHumanReadableChecksums(assetsRoot, assets);
  }

  const manifest = buildReleaseTransactionManifest({
    transactionId: input.transactionId,
    sourceId: input.sourceId as `sha256:${string}`,
    revision: input.revision,
    version: input.version,
    assets,
  });
  const outputPath = resolve(input.outputPath);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
  return manifest;
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function verifyHumanReadableChecksums(
  assetsRoot: string,
  assets: ReleaseAssetEvidence[],
): Promise<void> {
  const path = resolve(assetsRoot, SHA256_SUMS);
  const details = await lstat(path);
  if (!details.isFile() || details.isSymbolicLink()) {
    throw new Error("public_release_checksums_are_not_a_regular_file");
  }
  const expected = assets
    .filter(({ fileName }) => fileName.startsWith("mycellios-"))
    .sort((left, right) => left.fileName.localeCompare(right.fileName, "en"))
    .map(({ fileName, fileSha256 }) => `${fileSha256}  ${fileName}`)
    .join("\n") + "\n";
  if (await readFile(path, "utf8") !== expected) {
    throw new Error("public_release_checksums_do_not_match_assets");
  }
}

function parseArguments(argv: string[]): PreparePublicReleaseTransactionInput {
  const values = new Map<string, string>();
  for (const argument of argv) {
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      throw new Error(`public_release_argument_invalid:${argument}`);
    }
    const name = argument.slice(2, separator);
    const value = argument.slice(separator + 1);
    if (!value || values.has(name)) {
      throw new Error(`public_release_argument_invalid:${name}`);
    }
    values.set(name, value);
  }
  const expected = [
    "assets",
    "output",
    "revision",
    "source-id",
    "transaction-id",
    "version",
  ];
  if (
    JSON.stringify([...values.keys()].sort())
      !== JSON.stringify(expected)
  ) {
    throw new Error("public_release_arguments_incomplete_or_unknown");
  }
  return {
    assetsRoot: values.get("assets")!,
    outputPath: values.get("output")!,
    revision: values.get("revision")!,
    sourceId: values.get("source-id")!,
    transactionId: values.get("transaction-id")!,
    version: values.get("version")!,
  };
}

async function main(): Promise<void> {
  const manifest = await preparePublicReleaseTransaction(
    parseArguments(process.argv.slice(2)),
  );
  process.stdout.write(
    `${JSON.stringify({
      transactionId: manifest.transactionId,
      releaseId: manifest.releaseId,
      assets: manifest.assets.length,
    })}\n`,
  );
}

if (
  process.argv[1]
  && import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  await main();
}

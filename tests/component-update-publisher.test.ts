import {
  createServer,
  type IncomingMessage,
  type Server,
} from "node:http";
import {
  generateKeyPairSync,
} from "node:crypto";
import {
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildComponentUpdateManifest,
  signComponentUpdateManifest,
  type ComponentUpdateManifest,
} from "../src/contracts/component-update-manifest.js";
import { buildComponentFilesPackage } from "../src/update/component-files.js";
import {
  COMPONENT_PUBLICATION_EVIDENCE_SCHEMA,
  COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
  ComponentPublicationError,
  PREPARED_COMPONENT_RELEASE_SCHEMA,
  createDevelopmentSigningIdentity,
  publishComponentRelease,
  validateComponentUpdateEnrollment,
  type ComponentUpdateEnrollment,
  type PreparedComponentRelease,
} from "../src/update/component-update-publisher.js";

const directories: string[] = [];
const servers: Server[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => new Promise<void>(
    (resolvePromise) => server.close(() => resolvePromise()),
  )));
  await Promise.all(
    directories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true })
    ),
  );
});

describe("component update publisher", () => {
  it("hands a development identity to secure storage without returning or writing its private key", async () => {
    const localRoot = await temporaryDirectory("identity");
    let capturedPrivateType: string | null = null;
    const result = await createDevelopmentSigningIdentity({
      localRoot,
      keyId: "mycellios-dev-safe-storage",
      persistToFiles: false,
      storeIdentity(identity) {
        capturedPrivateType =
          typeof identity.privateKey === "string"
          || Buffer.isBuffer(identity.privateKey)
            ? "serialized"
            : identity.privateKey.type;
        expect(identity.enrollment.keyId).toBe(
          "mycellios-dev-safe-storage",
        );
      },
    });

    expect(capturedPrivateType).toBe("private");
    expect(result.privateKeyFile).toBeNull();
    expect(result.publicKeyFile).toBeNull();
    expect(JSON.stringify(result)).not.toContain("PRIVATE KEY");
    expect(result.desktopEnrollment).toEqual(
      validateComponentUpdateEnrollment(
        result.desktopEnrollment,
        "dev",
      ),
    );
  });

  it("rejects a non-canonical or non-Ed25519 enrollment", () => {
    const { publicKey } = generateKeyPairSync("ed25519");
    const spki = publicKey.export({ format: "der", type: "spki" });
    const base = {
      schema: COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
      channel: "dev",
      keyId: "mycellios-dev-test",
    } as const;

    expect(() => validateComponentUpdateEnrollment({
      ...base,
      spki: Buffer.concat([spki, Buffer.from([0])]).toString("base64url"),
    }, "dev")).toThrow(/SPKI/);

    const rsa = generateKeyPairSync("rsa", { modulusLength: 2_048 });
    expect(() => validateComponentUpdateEnrollment({
      ...base,
      spki: rsa.publicKey
        .export({ format: "der", type: "spki" })
        .toString("base64url"),
    }, "dev")).toThrow(/SPKI/);
  });

  it("publishes idempotent content and returns readback and reconciliation evidence", async () => {
    const fixture = await publicationFixture();
    const result = await publishComponentRelease({
      releaseDirectory: fixture.releaseDirectory,
      coordinatorUrl: fixture.coordinator,
      target: fixture.target,
    });

    expect(result).toMatchObject({
      manifestId: fixture.manifest.manifestId,
      sequence: fixture.manifest.sequence,
      artifactStatus: 201,
      manifestStatus: 201,
      publicVerification: true,
      commitPhase: "readback-verified",
      reconciliation: {
        schema: COMPONENT_PUBLICATION_EVIDENCE_SCHEMA,
        safeToRetry: true,
        publicManifestVerified: true,
        artifactReadbackVerified: true,
      },
    });
  });

  it("reports the durable commit boundary when publication fails after staging the artifact", async () => {
    const fixture = await publicationFixture({ rejectManifest: true });
    const failure = await publishComponentRelease({
      releaseDirectory: fixture.releaseDirectory,
      coordinatorUrl: fixture.coordinator,
      target: fixture.target,
    }).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ComponentPublicationError);
    expect(failure).toMatchObject({
      commitPhase: "artifact-staged",
      reconciliation: {
        safeToRetry: true,
        artifactStatus: 201,
        manifestStatus: 503,
        publicManifestVerified: false,
        artifactReadbackVerified: false,
      },
    });
  });

  it("reconciles a committed manifest when its PUT response is lost", async () => {
    const fixture = await publicationFixture({ dropManifestResponse: true });
    const result = await publishComponentRelease({
      releaseDirectory: fixture.releaseDirectory,
      coordinatorUrl: fixture.coordinator,
      target: fixture.target,
    });

    expect(result).toMatchObject({
      manifestId: fixture.manifest.manifestId,
      sequence: fixture.manifest.sequence,
      artifactStatus: 201,
      manifestStatus: null,
      publicVerification: true,
      commitPhase: "readback-verified",
      reconciliation: {
        publicManifestVerified: true,
        artifactReadbackVerified: true,
      },
    });
  });
});

async function publicationFixture(
  options: {
    rejectManifest?: boolean;
    dropManifestResponse?: boolean;
  } = {},
): Promise<{
  releaseDirectory: string;
  coordinator: string;
  target: PreparedComponentRelease["target"];
  manifest: ComponentUpdateManifest;
}> {
  const root = await temporaryDirectory("publication");
  const source = join(root, "source");
  const releaseDirectory = join(root, "release");
  await mkdir(source, { recursive: true });
  await mkdir(releaseDirectory, { recursive: true });
  await writeFile(join(source, "runtime.py"), "READY = True\n");
  const artifact = buildComponentFilesPackage(source);
  let publishedManifest: ComponentUpdateManifest | null = null;
  let publishedArtifact: Buffer | null = null;
  const server = createServer(async (request, response) => {
    const path = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
    if (
      request.method === "PUT"
      && path.startsWith("/public/v1/admin/component-updates/artifacts/")
    ) {
      publishedArtifact = await requestBody(request);
      response.writeHead(201).end();
      return;
    }
    if (
      request.method === "PUT"
      && path.endsWith("/dev/win32/x64/manifest")
    ) {
      if (options.rejectManifest) {
        response.writeHead(503).end("not ready");
        return;
      }
      publishedManifest = JSON.parse(
        (await requestBody(request)).toString("utf8"),
      ) as ComponentUpdateManifest;
      if (options.dropManifestResponse) {
        response.destroy();
        return;
      }
      response.writeHead(201).end();
      return;
    }
    if (
      request.method === "GET"
      && path === "/updates/v1/dev/win32/x64/manifest.json"
      && publishedManifest
    ) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(publishedManifest));
      return;
    }
    if (
      request.method === "GET"
      && path.startsWith("/updates/v1/artifacts/")
      && publishedArtifact
    ) {
      response.writeHead(200, {
        "content-type": "application/octet-stream",
        "content-length": publishedArtifact.length,
      });
      response.end(publishedArtifact);
      return;
    }
    response.writeHead(404).end();
  });
  servers.push(server);
  await new Promise<void>((resolvePromise) => {
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Test publication server did not bind.");
  }
  const coordinator = `http://127.0.0.1:${address.port}`;
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  const enrollment: ComponentUpdateEnrollment = {
    schema: COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
    channel: "dev",
    keyId: "mycellios-dev-publisher-test",
    spki: publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url"),
  };
  const target = {
    channel: "dev",
    platform: "win32",
    arch: "x64",
  } as const;
  const manifest = signComponentUpdateManifest(
    buildComponentUpdateManifest({
      channel: "dev",
      sequence: 42,
      revision: "a".repeat(40),
      provenance: {
        baseRevision: "a".repeat(40),
        sourceTreeDirty: false,
        sourceTreeDigest: artifact.filesManifestSha256,
      },
      sourceId: `sha256:${"b".repeat(64)}`,
      compatibility: {
        workerProtocol: { min: 1, max: 1 },
        runtimeAbi: "mycellios-distribution-runtime/4",
        minBootstrapVersion: "0.2.70",
      },
      components: [{
        id: "python-product",
        version: "0.2.70-dev.42",
        platform: "win32",
        arch: "x64",
        artifact: {
          url:
            `${coordinator}/updates/v1/artifacts/`
            + artifact.artifactSha256.slice("sha256:".length),
          sha256: artifact.artifactSha256,
          bytes: artifact.packageBytes.length,
          format: artifact.format,
          filesManifestSha256: artifact.filesManifestSha256,
        },
        requirements: {
          backend: "any",
          driver: null,
          runtimeAbi: "mycellios-distribution-runtime/4",
          workerProtocol: { min: 3, max: 4 },
          dependencies: [],
        },
        restartScope: "runtime",
      }],
    }),
    { keyId: enrollment.keyId, privateKey },
  );
  const release: PreparedComponentRelease = {
    schema: PREPARED_COMPONENT_RELEASE_SCHEMA,
    target,
    artifactFile: "artifact.blob",
    manifestFile: "manifest.json",
    enrollmentFile: "enrollment.json",
    manifestId: manifest.manifestId,
    sequence: manifest.sequence,
    provenance: {
      baseRevision: "a".repeat(40),
      sourceTreeDirty: false,
      releaseRevision: "a".repeat(40),
      filesManifestSha256: artifact.filesManifestSha256,
    },
  };
  await Promise.all([
    writeFile(join(releaseDirectory, release.artifactFile), artifact.packageBytes),
    writeFile(
      join(releaseDirectory, release.manifestFile),
      `${JSON.stringify(manifest)}\n`,
    ),
    writeFile(
      join(releaseDirectory, release.enrollmentFile),
      `${JSON.stringify(enrollment)}\n`,
    ),
    writeFile(
      join(releaseDirectory, "release.json"),
      `${JSON.stringify(release)}\n`,
    ),
  ]);
  return { releaseDirectory, coordinator, target, manifest };
}

async function temporaryDirectory(label: string): Promise<string> {
  const directory = await mkdtemp(
    join(tmpdir(), `mycellios-component-${label}-`),
  );
  directories.push(directory);
  return directory;
}

async function requestBody(request: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

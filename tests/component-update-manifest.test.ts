import {
  generateKeyPairSync,
  type KeyObject,
} from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  buildComponentUpdateManifest,
  parseComponentUpdateManifest,
  signComponentUpdateManifest,
  verifyComponentUpdateManifest,
  type ComponentUpdateManifest,
  type ComponentUpdateManifestBuildInput,
} from "../src/contracts/component-update-manifest.js";
import { MAX_COMPONENT_ARTIFACT_BYTES } from "../src/contracts/component-update-policy.js";

function fixtureInput(
  overrides: Partial<ComponentUpdateManifestBuildInput> = {},
): ComponentUpdateManifestBuildInput {
  return {
    channel: "dev",
    sequence: 42,
    revision: "a".repeat(40),
    provenance: {
      baseRevision: "a".repeat(40),
      sourceTreeDirty: false,
      sourceTreeDigest: `sha256:${"9".repeat(64)}`,
    },
    sourceId: `sha256:${"b".repeat(64)}`,
    compatibility: {
      workerProtocol: { min: 3, max: 4 },
      runtimeAbi: "mycellios-distribution-runtime/4",
      minBootstrapVersion: "1.2.0",
    },
    components: [
      {
        id: "worker-runtime",
        version: "0.8.0-dev.42",
        platform: "linux",
        arch: "x64",
        artifact: {
          url: "https://updates.example.test/dev/worker-runtime.tar.zst",
          sha256: `sha256:${"c".repeat(64)}`,
          bytes: 12_345,
          format: "tar.zst",
          filesManifestSha256: `sha256:${"e".repeat(64)}`,
        },
        requirements: {
          backend: "cuda",
          driver: { api: "nvidia", minVersion: "550.0.0" },
          runtimeAbi: "mycellios-distribution-runtime/4",
          workerProtocol: { min: 3, max: 4 },
          dependencies: [],
        },
        restartScope: "runtime",
      },
      {
        id: "desktop-agent",
        version: "0.8.0-dev.42",
        platform: "win32",
        arch: "x64",
        artifact: {
          url: "https://updates.example.test/dev/desktop-agent.zip",
          sha256: `sha256:${"d".repeat(64)}`,
          bytes: 23_456,
          format: "json-gzip-v1",
          filesManifestSha256: `sha256:${"f".repeat(64)}`,
        },
        requirements: {
          backend: "any",
          driver: null,
          runtimeAbi: "mycellios-distribution-runtime/4",
          workerProtocol: { min: 3, max: 4 },
          dependencies: [{ id: "worker-runtime", minVersion: "0.8.0" }],
        },
        restartScope: "agent",
      },
    ],
    ...overrides,
  };
}

function signingFixture(): {
  privateKey: KeyObject;
  publicSpki: string;
  keyId: string;
} {
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  return {
    privateKey,
    publicSpki: publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url"),
    keyId: "mycellios-dev-2026-01",
  };
}

function signedFixture(): {
  manifest: ComponentUpdateManifest;
  privateKey: KeyObject;
  publicSpki: string;
  keyId: string;
} {
  const keys = signingFixture();
  return {
    ...keys,
    manifest: signComponentUpdateManifest(
      buildComponentUpdateManifest(fixtureInput()),
      {
        keyId: keys.keyId,
        privateKey: keys.privateKey,
      },
    ),
  };
}

describe("component update manifest", () => {
  it("builds, signs and verifies a canonical v3 manifest with an external pin", () => {
    const { manifest, publicSpki, keyId } = signedFixture();

    expect(manifest.components.map(({ id }) => id)).toEqual([
      "desktop-agent",
      "worker-runtime",
    ]);
    expect(manifest.manifestId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(manifest).not.toHaveProperty("publicKey");
    expect(manifest).not.toHaveProperty("publisherPublicKey");
    expect(
      verifyComponentUpdateManifest(manifest, {
        pinnedKey: { keyId, spki: publicSpki },
        expectedChannel: "dev",
        minimumSequence: 42,
      }),
    ).toEqual(manifest);
  });

  it("produces the same identity and deterministic Ed25519 signature for reordered input", () => {
    const keys = signingFixture();
    const first = signComponentUpdateManifest(
      buildComponentUpdateManifest(fixtureInput()),
      { keyId: keys.keyId, privateKey: keys.privateKey },
    );
    const input = fixtureInput();
    const second = signComponentUpdateManifest(
      buildComponentUpdateManifest({
        ...input,
        compatibility: {
          minBootstrapVersion: input.compatibility.minBootstrapVersion,
          runtimeAbi: input.compatibility.runtimeAbi,
          workerProtocol: {
            max: input.compatibility.workerProtocol.max,
            min: input.compatibility.workerProtocol.min,
          },
        },
        components: [...input.components].reverse(),
      }),
      { keyId: keys.keyId, privateKey: keys.privateKey },
    );

    expect(second.manifestId).toBe(first.manifestId);
    expect(second.signature).toBe(first.signature);
  });

  it("rejects unknown fields, embedded trust material, and non-v3 schemas", () => {
    const { manifest } = signedFixture();

    expect(() =>
      parseComponentUpdateManifest({ ...manifest, unexpected: true }),
    ).toThrow();
    expect(() =>
      parseComponentUpdateManifest({
        ...manifest,
        publisherPublicKey: "attacker-controlled",
      }),
    ).toThrow();
    expect(() =>
      parseComponentUpdateManifest({
        ...manifest,
        schema: "mycellios-component-update/2",
      }),
    ).toThrow();
    expect(() =>
      parseComponentUpdateManifest({
        ...manifest,
        compatibility: {
          ...manifest.compatibility,
          futureFlag: true,
        },
      }),
    ).toThrow();
  });

  it("rejects unsafe artifacts, invalid protocol ranges, and duplicate targets", () => {
    const input = fixtureInput();
    expect(() =>
      buildComponentUpdateManifest({
        ...input,
        components: [
          {
            ...input.components[0]!,
            artifact: {
              ...input.components[0]!.artifact,
              url: "http://updates.example.test/worker.zip",
            },
          },
        ],
      }),
    ).toThrow();
    expect(() =>
      buildComponentUpdateManifest({
        ...input,
        components: [{
          ...input.components[0]!,
          artifact: {
            ...input.components[0]!.artifact,
            bytes: MAX_COMPONENT_ARTIFACT_BYTES + 1,
          },
        }],
      }),
    ).toThrow();
    expect(() =>
      buildComponentUpdateManifest({
        ...input,
        components: [
          {
            ...input.components[0]!,
            artifact: {
              ...input.components[0]!.artifact,
              url: "http://127.0.0.1:8787/updates/v1/artifacts/test",
            },
          },
        ],
      }),
    ).not.toThrow();
    expect(() =>
      buildComponentUpdateManifest({
        ...input,
        compatibility: {
          ...input.compatibility,
          workerProtocol: { min: 5, max: 4 },
        },
      }),
    ).toThrow();
    expect(() =>
      buildComponentUpdateManifest({
        ...input,
        components: [
          input.components[0]!,
          {
            ...input.components[0]!,
            version: "different-version",
          },
        ],
      }),
    ).toThrow();
  });

  it("requires strict runtime, driver, protocol and dependency metadata", () => {
    const input = fixtureInput();
    const component = input.components[0]!;
    expect(() => buildComponentUpdateManifest({
      ...input,
      components: [{ ...component, requirements: undefined as never }],
    })).toThrow();
    expect(() => buildComponentUpdateManifest({
      ...input,
      components: [{
        ...component,
        requirements: {
          ...component.requirements,
          workerProtocol: { min: 5, max: 4 },
        },
      }],
    })).toThrow();
    expect(() => buildComponentUpdateManifest({
      ...input,
      components: [{
        ...component,
        requirements: {
          ...component.requirements,
          dependencies: [
            { id: "runtime-core", minVersion: "2.0.0" },
            { id: "runtime-core", minVersion: "2.1.0" },
          ],
        },
      }],
    })).toThrow("component_update_dependency_is_duplicated");
    expect(() => buildComponentUpdateManifest({
      ...input,
      components: [{
        ...component,
        requirements: {
          ...component.requirements,
          driver: {
            api: "nvidia",
            minVersion: "560.0.0",
            maxVersionExclusive: "550.0.0",
          },
        },
      }],
    })).toThrow("component_update_driver_version_range_is_invalid");
  });

  it("detects content tampering before trusting the signature", () => {
    const { manifest } = signedFixture();
    expect(() =>
      parseComponentUpdateManifest({
        ...manifest,
        sequence: manifest.sequence + 1,
      }),
    ).toThrow("component_update_manifest_identity_mismatch");
  });

  it("rejects the wrong external key, key id, channel, rollback, and signature", () => {
    const { manifest, publicSpki, keyId } = signedFixture();
    const other = signingFixture();

    expect(() =>
      verifyComponentUpdateManifest(manifest, {
        pinnedKey: { keyId, spki: other.publicSpki },
      }),
    ).toThrow("component_update_manifest_signature_verification_failed");
    expect(() =>
      verifyComponentUpdateManifest(manifest, {
        pinnedKey: { keyId: "other-key", spki: publicSpki },
      }),
    ).toThrow("component_update_manifest_key_id_mismatch");
    expect(() =>
      verifyComponentUpdateManifest(manifest, {
        pinnedKey: { keyId, spki: publicSpki },
        expectedChannel: "stable",
      }),
    ).toThrow("component_update_manifest_channel_mismatch");
    expect(() =>
      verifyComponentUpdateManifest(manifest, {
        pinnedKey: { keyId, spki: publicSpki },
        minimumSequence: manifest.sequence + 1,
      }),
    ).toThrow("component_update_manifest_rollback_rejected");
    expect(() =>
      verifyComponentUpdateManifest(
        {
          ...manifest,
          signature: `${
            manifest.signature.startsWith("A") ? "B" : "A"
          }${manifest.signature.slice(1)}`,
        },
        { pinnedKey: { keyId, spki: publicSpki } },
      ),
    ).toThrow("component_update_manifest_signature_verification_failed");
  });

  it("requires Ed25519 signing and pinned verification keys", () => {
    const rsa = generateKeyPairSync("rsa", { modulusLength: 2_048 });
    const unsigned = buildComponentUpdateManifest(fixtureInput());
    expect(() =>
      signComponentUpdateManifest(unsigned, {
        keyId: "rsa-is-not-allowed",
        privateKey: rsa.privateKey,
      }),
    ).toThrow("component_update_manifest_private_key_is_not_ed25519");

    const { manifest, keyId } = signedFixture();
    const rsaSpki = rsa.publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url");
    expect(() =>
      verifyComponentUpdateManifest(manifest, {
        pinnedKey: { keyId, spki: rsaSpki },
      }),
    ).toThrow("component_update_manifest_pinned_key_is_not_ed25519");
  });
});

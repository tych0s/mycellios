import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  DRAFT_STRATEGY_CERTIFICATION_SCHEMA,
  DRAFT_STRATEGY_DESCRIPTOR_SCHEMA,
  draftStrategyDescriptorIdentity,
  signDraftStrategyCertification,
} from "../src/contracts/engine-family.js";
import {
  DRAFT_STRATEGY_CATALOG_SCHEMA,
  NativeDraftStrategyCatalog,
} from "../src/distribution/draft-strategy-catalog.js";
import { DraftStrategyResolutionError } from "../src/distribution/draft-strategy-registry.js";
import type { PythonPipelineLaunchDescription } from "../src/distribution/python-launcher.js";

const digest = (character: string) => `sha256:${character.repeat(64)}` as const;
const keys = generateKeyPairSync("ed25519");
const spki = keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url");

function entry(component = digest("1")) {
  const descriptor = {
    schema: DRAFT_STRATEGY_DESCRIPTOR_SCHEMA,
    strategyId: "ngram",
    version: "1.0.0",
    kind: "ngram" as const,
    componentDigest: component,
    targetDescriptorDigest: digest("2"),
    tokenizerDigest: digest("3"),
    vocabularyDigest: digest("4"),
    resource: {
      minimumRamBytes: 0,
      minimumVramBytes: 0,
      allowedBackends: ["cuda" as const],
    },
    limits: { minDraftTokens: 1, maxDraftTokens: 8, maxInflightWaves: 4 },
    evidenceDigest: digest("5"),
  };
  return {
    descriptor,
    certification: signDraftStrategyCertification({
      schema: DRAFT_STRATEGY_CERTIFICATION_SCHEMA,
      descriptorDigest: draftStrategyDescriptorIdentity(descriptor),
      sourceId: digest("6"),
      status: "certified",
      validFrom: "2026-08-01T00:00:00.000Z",
      expiresAt: "2026-09-01T00:00:00.000Z",
      publisherKeyId: "release-2026",
    }, keys.privateKey),
  };
}

function document(entries = [entry()]) {
  return { schema: DRAFT_STRATEGY_CATALOG_SCHEMA, entries };
}

function keyring() {
  return [{ keyId: "release-2026", spki }];
}

function launch(): PythonPipelineLaunchDescription {
  return {
    sourceManifest: {
      plans: {
        decode: {
          speculation: {
            mode: "adaptive",
            defaultStrategyId: "ngram",
            strategies: [{ id: "ngram", kind: "ngram", maxDraftTokens: 2 }],
          },
        },
      },
    },
    configuration: { speculativeInflightWaves: 2 },
  } as unknown as PythonPipelineLaunchDescription;
}

function context() {
  return {
    targetDescriptorDigest: digest("2"),
    tokenizerDigest: digest("3"),
    vocabularyDigest: digest("4"),
    backend: "cuda" as const,
    availableRamBytes: 0,
    availableVramBytes: 0,
    now: new Date("2026-08-10T00:00:00.000Z"),
  };
}

describe("native draft strategy catalog", () => {
  it("loads only pinned signatures and resolves one exact launch entry", () => {
    const catalog = new NativeDraftStrategyCatalog(document(), keyring());
    expect(catalog.resolveForLaunch(launch(), context()).descriptor.strategyId)
      .toBe("ngram");
  });

  it("fails closed when compatibility evidence is missing or wrong", () => {
    const catalog = new NativeDraftStrategyCatalog(document(), keyring());
    expect(() => catalog.resolveForLaunch(launch(), {
      ...context(),
      vocabularyDigest: digest("9"),
    })).toThrowError(new DraftStrategyResolutionError(
      "draft_strategy_catalog_has_no_compatible_entry",
    ));
  });

  it("rejects unpinned, malformed and ambiguous catalogs", () => {
    expect(() => new NativeDraftStrategyCatalog(document(), [{
      keyId: "other",
      spki,
    }])).toThrow(/draft_strategy_certification_key_is_unknown/);
    expect(() => new NativeDraftStrategyCatalog(document(), [{
      keyId: "release-2026",
      spki: Buffer.from("not-a-key").toString("base64url"),
    }])).toThrowError(new DraftStrategyResolutionError(
      "draft_strategy_keyring_key_is_invalid",
    ));

    const catalog = new NativeDraftStrategyCatalog(
      document([entry(digest("1")), entry(digest("7"))]),
      keyring(),
    );
    expect(() => catalog.resolveForLaunch(launch(), context())).toThrowError(
      new DraftStrategyResolutionError(
        "draft_strategy_catalog_resolution_is_ambiguous",
      ),
    );
  });
});

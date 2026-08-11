import { generateKeyPairSync } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";

import {
  DeveloperWorkflowController,
  readDevelopmentEnrollment,
} from "../src/program/developer-workflow.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(
    roots.splice(0).map((root) =>
      rm(root, { recursive: true, force: true }),
    ),
  );
});

describe("desktop developer workflow", () => {
  it("is fail-closed outside a source checkout", () => {
    const controller = new DeveloperWorkflowController({
      workspace: join(tmpdir(), "missing-mycellios-source"),
      sourceMode: false,
    });

    expect(controller.status).toMatchObject({
      available: false,
      channel: { state: "stopped", enrolled: false },
    });
    expect(() => controller.startDistributedGate()).toThrow(
      "available only from a Mycellios source checkout",
    );
  });

  it("accepts only a canonical Ed25519 enrollment", async () => {
    const root = await mkdtemp(join(tmpdir(), "mycellios-dev-enrollment-"));
    roots.push(root);
    const { publicKey } = generateKeyPairSync("ed25519");
    const spki = publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url");
    const path = join(root, "public.json");
    await writeFile(
      path,
      JSON.stringify({
        schema: "mycellios-component-update-public-key/1",
        channel: "dev",
        keyId: "mycellios-dev-test-1",
        spki,
      }),
    );

    await expect(readDevelopmentEnrollment(path)).resolves.toMatchObject({
      keyId: "mycellios-dev-test-1",
      spki,
    });

    const nonCanonical = Buffer.concat([
      Buffer.from(spki, "base64url"),
      Buffer.from([0]),
    ]).toString("base64url");
    await writeFile(
      path,
      JSON.stringify({
        schema: "mycellios-component-update-public-key/1",
        channel: "dev",
        keyId: "mycellios-dev-test-1",
        spki: nonCanonical,
      }),
    );
    await expect(readDevelopmentEnrollment(path)).rejects.toThrow(
      "not Ed25519",
    );
  });

  it("selects an enrolled HTTPS lab without starting the loopback publisher", () => {
    const { publicKey } = generateKeyPairSync("ed25519");
    const spki = publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url");
    const controller = new DeveloperWorkflowController({
      workspace: process.cwd(),
      sourceMode: true,
    });

    controller.configureEnrolledChannel({
      feedUrl: "https://www.mycellios.com/development-labs/lab_abcdefghijklmnopqrstuv/",
      keyId: "mycellios-dev-test-1",
      spki,
      adminToken: "kept-only-in-main",
    });

    expect(controller.usesLocalFeed).toBe(false);
    expect(controller.feedUrl).toBe(
      "https://www.mycellios.com/development-labs/lab_abcdefghijklmnopqrstuv",
    );
    expect(controller.status.channel).toMatchObject({
      state: "ready",
      enrolled: true,
      keyId: "mycellios-dev-test-1",
    });
    expect(JSON.stringify(controller.status)).not.toContain("kept-only-in-main");
    expect(() => controller.configureEnrolledChannel({
      feedUrl: "http://updates.example.test/development-labs/lab_abcdefghijklmnopqrstuv",
      keyId: "mycellios-dev-test-1",
      spki,
    })).toThrow("require HTTPS");
  });
});

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  WINDOWS_JOB_BROKER_SCHEMA,
  normalizeWindowsJobBrokerExecutable,
  writeWindowsJobBrokerRequest,
} from "../src/distribution/windows-job-broker.js";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("Windows Job Object broker contract", () => {
  it("writes one exact, parent-bound request inside the private workspace", () => {
    const workspace = mkdtempSync(join(tmpdir(), "mycellios-job-request-"));
    temporaryRoots.push(workspace);
    const requestPath = writeWindowsJobBrokerRequest(
      workspace,
      {
        executable: process.execPath,
        args: ["-e", "console.log('value with spaces and \"quotes\"')"],
      },
      resolve("."),
      12_345,
    );

    expect(requestPath).toBe(
      join(workspace, "windows-job-broker-request.json"),
    );
    expect(JSON.parse(readFileSync(requestPath, "utf8"))).toEqual({
      schema: WINDOWS_JOB_BROKER_SCHEMA,
      executable: resolve(process.execPath),
      args: ["-e", "console.log('value with spaces and \"quotes\"')"],
      cwd: resolve("."),
      parentPid: 12_345,
    });
  });

  it("rejects relative paths and request-file replacement", () => {
    const workspace = mkdtempSync(join(tmpdir(), "mycellios-job-request-"));
    temporaryRoots.push(workspace);
    const command = { executable: process.execPath, args: [] };
    expect(() =>
      writeWindowsJobBrokerRequest(workspace, command, "relative"),
    ).toThrow("windows_job_broker_cwd_must_be_absolute");

    writeWindowsJobBrokerRequest(workspace, command, resolve("."));
    expect(() =>
      writeWindowsJobBrokerRequest(workspace, command, resolve(".")),
    ).toThrow();
  });

  it("pins an existing absolute broker executable", () => {
    const packagedBroker = resolve(
      "build",
      "windows-job-broker",
      "mycellios-job-broker.exe",
    );
    if (!existsSync(packagedBroker)) return;
    expect(normalizeWindowsJobBrokerExecutable(packagedBroker)).toBe(
      packagedBroker,
    );
    expect(() =>
      normalizeWindowsJobBrokerExecutable("mycellios-job-broker.exe"),
    ).toThrow("windows_job_broker_executable_is_invalid");
  });
});

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { writeWindowsJobBrokerRequest } from "../src/distribution/windows-job-broker.js";

const brokerExecutable = resolve(
  "build",
  "windows-job-broker",
  "mycellios-job-broker.exe",
);
const physical = process.platform === "win32" && existsSync(brokerExecutable)
  ? describe
  : describe.skip;
const temporaryRoots: string[] = [];
const survivorPids = new Set<number>();

afterEach(async () => {
  for (const pid of survivorPids) {
    if (!processIsAlive(pid)) continue;
    spawnSync(
      resolve(process.env.SystemRoot ?? "C:\\Windows", "System32", "taskkill.exe"),
      ["/PID", String(pid), "/T", "/F"],
      { stdio: "ignore", shell: false, windowsHide: true },
    );
  }
  survivorPids.clear();
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

physical("Windows Job Object broker physical containment", () => {
  it("rejects a request not bound to the broker's actual parent", async () => {
    const workspace = temporaryWorkspace();
    const marker = join(workspace, "must-not-exist.txt");
    const request = writeWindowsJobBrokerRequest(
      workspace,
      {
        executable: process.execPath,
        args: [
          "-e",
          "require('node:fs').writeFileSync(process.argv[1], 'ran')",
          marker,
        ],
      },
      resolve("."),
      process.pid + 1,
    );
    const child = spawn(brokerExecutable, [request], {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const output = collect(child);
    await expect(childExit(child)).resolves.toEqual({
      code: 70,
      signal: null,
    });
    expect((await output).stderr).toContain(
      "trusted_parent_pid_does_not_match_broker_parent",
    );
    expect(existsSync(marker)).toBe(false);
  });

  it("preserves exact argv, output and target exit status", async () => {
    const workspace = temporaryWorkspace();
    const script = [
      "const expected = ['space value', 'quote\"value', 'trailing\\\\'];",
      "if (JSON.stringify(process.argv.slice(1)) !== JSON.stringify(expected)) process.exit(91);",
      "console.log('broker-target-ready');",
      "process.exit(7);",
    ].join("");
    const request = writeWindowsJobBrokerRequest(
      workspace,
      {
        executable: process.execPath,
        args: [
          "-e",
          script,
          "space value",
          "quote\"value",
          "trailing\\",
        ],
      },
      resolve("."),
    );
    const child = spawn(brokerExecutable, [request], {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const output = collect(child);
    const exit = await childExit(child);

    expect(exit).toEqual({ code: 7, signal: null });
    expect((await output).stdout).toContain("broker-target-ready");
    expect((await output).stderr).toBe("");
    expect(existsSync(request)).toBe(false);
  });

  it("kills the target and its detached descendant when the broker dies", async () => {
    const workspace = temporaryWorkspace();
    const pidPath = join(workspace, "job-pids.json");
    const descendantScript = "setInterval(() => undefined, 1000);";
    const targetScript = [
      "const {spawn}=require('node:child_process');",
      "const {writeFileSync}=require('node:fs');",
      `const descendant=spawn(process.execPath,['-e',${JSON.stringify(descendantScript)}],`,
      "{detached:true,stdio:'ignore',windowsHide:true});",
      "writeFileSync(process.argv[1],JSON.stringify({target:process.pid,descendant:descendant.pid}));",
      "setInterval(() => undefined, 1000);",
    ].join("");
    const request = writeWindowsJobBrokerRequest(
      workspace,
      {
        executable: process.execPath,
        args: ["-e", targetScript, pidPath],
      },
      resolve("."),
    );
    const broker = spawn(brokerExecutable, [request], {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const pids = await waitForPids(pidPath);
    survivorPids.add(pids.target);
    survivorPids.add(pids.descendant);
    expect(processIsAlive(pids.target)).toBe(true);
    expect(processIsAlive(pids.descendant)).toBe(true);

    // Node uses TerminateProcess for SIGKILL on Windows. It does not request a
    // recursive taskkill, so descendant closure here comes from Job Objects.
    expect(broker.kill("SIGKILL")).toBe(true);
    await childExit(broker);
    await waitFor(() =>
      !processIsAlive(pids.target) && !processIsAlive(pids.descendant),
    );

    expect(processIsAlive(pids.target)).toBe(false);
    expect(processIsAlive(pids.descendant)).toBe(false);
    survivorPids.delete(pids.target);
    survivorPids.delete(pids.descendant);
  });

  it("kills the job when its trusted parent exits", async () => {
    const workspace = temporaryWorkspace();
    const pidPath = join(workspace, "parent-exit-pids.json");
    const descendantScript = "setInterval(() => undefined, 1000);";
    const targetScript = [
      "const {spawn}=require('node:child_process');",
      "const {writeFileSync}=require('node:fs');",
      `const descendant=spawn(process.execPath,['-e',${JSON.stringify(descendantScript)}],`,
      "{detached:true,stdio:'ignore',windowsHide:true});",
      "writeFileSync(process.argv[1],JSON.stringify({target:process.pid,descendant:descendant.pid}));",
      "setInterval(() => undefined, 1000);",
    ].join("");
    const wrapperScript = [
      "const {existsSync,writeFileSync}=require('node:fs');",
      "const {join}=require('node:path');",
      "const {spawn}=require('node:child_process');",
      "const [broker,workspace,pidPath,targetScript]=process.argv.slice(1);",
      "const requestPath=join(workspace,'windows-job-broker-request.json');",
      "writeFileSync(requestPath,JSON.stringify({",
      "schema:'mycellios-windows-job-broker/1',",
      "executable:process.execPath,args:['-e',targetScript,pidPath],",
      "cwd:process.cwd(),parentPid:process.pid}));",
      "spawn(broker,[requestPath],{stdio:'ignore',windowsHide:true});",
      "const deadline=Date.now()+5000;",
      "const timer=setInterval(()=>{",
      "if(existsSync(pidPath)){clearInterval(timer);process.exit(0);}",
      "if(Date.now()>deadline){clearInterval(timer);process.exit(92);}",
      "},10);",
    ].join("");
    const wrapper = spawn(
      process.execPath,
      [
        "-e",
        wrapperScript,
        brokerExecutable,
        workspace,
        pidPath,
        targetScript,
      ],
      {
        cwd: resolve("."),
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    await expect(childExit(wrapper)).resolves.toEqual({
      code: 0,
      signal: null,
    });
    const pids = await waitForPids(pidPath);
    survivorPids.add(pids.target);
    survivorPids.add(pids.descendant);
    await waitFor(() =>
      !processIsAlive(pids.target) && !processIsAlive(pids.descendant),
    );

    expect(processIsAlive(pids.target)).toBe(false);
    expect(processIsAlive(pids.descendant)).toBe(false);
    survivorPids.delete(pids.target);
    survivorPids.delete(pids.descendant);
  });
});

function temporaryWorkspace(): string {
  const workspace = mkdtempSync(join(tmpdir(), "mycellios-job-physical-"));
  temporaryRoots.push(workspace);
  return workspace;
}

function collect(child: ChildProcess): Promise<{ stdout: string; stderr: string }> {
  let stdout = "";
  let stderr = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    stdout += chunk.toString("utf8");
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString("utf8");
  });
  return new Promise((resolveOutput) => {
    child.once("close", () => resolveOutput({ stdout, stderr }));
  });
}

function childExit(
  child: ChildProcess,
): Promise<{ code: number | null; signal: NodeJS.Signals | null }> {
  return new Promise((resolveExit, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => resolveExit({ code, signal }));
  });
}

async function waitForPids(
  path: string,
): Promise<{ target: number; descendant: number }> {
  await waitFor(() => existsSync(path));
  const value = JSON.parse(readFileSync(path, "utf8")) as {
    target?: unknown;
    descendant?: unknown;
  };
  if (
    !Number.isInteger(value.target)
    || !Number.isInteger(value.descendant)
  ) {
    throw new Error("physical_job_broker_did_not_publish_valid_pids");
  }
  return {
    target: value.target as number,
    descendant: value.descendant as number,
  };
}

async function waitFor(predicate: () => boolean, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolveWait) => setTimeout(resolveWait, 25));
  }
  throw new Error("windows_job_broker_physical_wait_timed_out");
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

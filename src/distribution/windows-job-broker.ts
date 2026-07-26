import { spawn } from "node:child_process";
import { existsSync, statSync, writeFileSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { buildIsolatedProcessEnvironment } from "./process-environment.js";

export const WINDOWS_JOB_BROKER_SCHEMA =
  "mycellios-windows-job-broker/1" as const;
export const WINDOWS_JOB_BROKER_PROBE =
  "mycellios-windows-job-broker/1:ready" as const;

export interface WindowsJobBrokerRequest {
  schema: typeof WINDOWS_JOB_BROKER_SCHEMA;
  executable: string;
  args: string[];
  cwd: string;
  parentPid: number;
}

export function normalizeWindowsJobBrokerExecutable(value: string): string {
  if (
    typeof value !== "string"
    || value.length === 0
    || value !== value.trim()
    || /[\0\r\n]/.test(value)
    || !isAbsolute(value)
    || !value.toLowerCase().endsWith(".exe")
  ) {
    throw new Error("windows_job_broker_executable_is_invalid");
  }
  const executable = resolve(value);
  if (!existsSync(executable) || !statSync(executable).isFile()) {
    throw new Error("windows_job_broker_executable_is_missing");
  }
  return executable;
}

export function writeWindowsJobBrokerRequest(
  workspacePath: string,
  command: { executable: string; args: readonly string[] },
  cwdValue: string,
  parentPid = process.pid,
): string {
  if (!isAbsolute(workspacePath) || !isAbsolute(command.executable)) {
    throw new Error("windows_job_broker_target_paths_must_be_absolute");
  }
  if (
    !Number.isSafeInteger(parentPid)
    || parentPid < 1
    || command.args.length > 512
    || command.args.some(
      (argument) =>
        typeof argument !== "string"
        || argument.length > 32_000
        || argument.includes("\0"),
    )
  ) {
    throw new Error("windows_job_broker_request_is_invalid");
  }
  if (!isAbsolute(cwdValue)) {
    throw new Error("windows_job_broker_cwd_must_be_absolute");
  }
  const cwd = resolve(cwdValue);
  const request: WindowsJobBrokerRequest = {
    schema: WINDOWS_JOB_BROKER_SCHEMA,
    executable: resolve(command.executable),
    args: [...command.args],
    cwd,
    parentPid,
  };
  const requestPath = join(workspacePath, "windows-job-broker-request.json");
  writeFileSync(requestPath, `${JSON.stringify(request)}\n`, {
    encoding: "utf8",
    flag: "wx",
    mode: 0o600,
  });
  return requestPath;
}

export async function probeWindowsJobBroker(
  executableValue: string,
  timeoutMs = 5_000,
): Promise<void> {
  const executable = normalizeWindowsJobBrokerExecutable(executableValue);
  await new Promise<void>((resolveProbe, reject) => {
    const child = spawn(executable, ["--probe"], {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: buildIsolatedProcessEnvironment(),
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolveProbe();
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(new Error("windows_job_broker_probe_timed_out"));
    }, Math.max(100, timeoutMs));
    timer.unref?.();
    child.stdout.on("data", (chunk: Buffer) => {
      stdout = `${stdout}${chunk.toString("utf8")}`.slice(-4_096);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4_096);
    });
    child.once("error", (error) => finish(error));
    child.once("close", (code) => {
      if (code !== 0 || stdout.trim() !== WINDOWS_JOB_BROKER_PROBE) {
        finish(
          new Error(
            `windows_job_broker_probe_failed:code=${code ?? "null"}:${
              stderr.trim() || stdout.trim() || "no_output"
            }`,
          ),
        );
        return;
      }
      finish();
    });
  });
}

import { describe, expect, it } from "vitest";
import type {
  LaunchCapturedOutput,
  LaunchProcessHandle,
} from "../src/distribution/launch-supervisor.js";
import { stageStartupDiagnostic } from "../src/desktop/stage-startup-diagnostic.js";

describe("stageStartupDiagnostic", () => {
  it("keeps the bounded tail that explains an early Python exit", () => {
    const handle = fixtureHandle({
      stdout: "",
      stderr: [
        "Fetching model files",
        "Traceback (most recent call last):",
        '  File "C:\\Users\\ExampleUser\\runtime\\server.py", line 12, in main',
        "RuntimeError: CUDA kernel image is unavailable",
      ].join("\n"),
      stdoutTruncated: false,
      stderrTruncated: false,
    });

    expect(stageStartupDiagnostic(handle)).toBe(
      "Traceback (most recent call last): | File \"[PATH]\", line 12, in main | "
        + "RuntimeError: CUDA kernel image is unavailable",
    );
  });

  it("redacts credentials, email addresses and Unix paths", () => {
    const handle = fixtureHandle({
      stdout: "",
      stderr:
        "Bearer abc.def secret=visible exampleuser@example.com /home/exampleuser/models/model.py",
      stdoutTruncated: false,
      stderrTruncated: false,
    });

    expect(stageStartupDiagnostic(handle)).toBe(
      "Bearer [REDACTED] secret=[REDACTED] [EMAIL] [PATH]",
    );
  });

  it("does not fail activation when process output is unavailable", () => {
    expect(stageStartupDiagnostic(null)).toBeNull();
    expect(stageStartupDiagnostic({
      ...fixtureHandle({
        stdout: "",
        stderr: "",
        stdoutTruncated: false,
        stderrTruncated: false,
      }),
      output: () => {
        throw new Error("output unavailable");
      },
    })).toBeNull();
  });
});

function fixtureHandle(output: LaunchCapturedOutput): LaunchProcessHandle {
  return {
    ready: Promise.resolve(),
    exited: Promise.resolve({ code: 1, signal: null }),
    stop: async () => undefined,
    output: () => ({ ...output }),
  };
}

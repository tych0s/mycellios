import { existsSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";

const outputDirectory = resolve("build", "windows-job-broker");
const executable = resolve(outputDirectory, "mycellios-job-broker.exe");
const localDotnet = resolve(".codex-runtime", "dotnet-sdk", "dotnet.exe");
const dotnetExecutable =
  process.env.MYCELLIOS_DOTNET?.trim()
  || (existsSync(localDotnet) ? localDotnet : "dotnet");

if (process.platform !== "win32" || process.arch !== "x64") {
  console.log(`Windows Job Object broker skipped on ${process.platform}/${process.arch}.`);
  process.exit(0);
}

rmSync(outputDirectory, { recursive: true, force: true });
const project = resolve(
  "sidecars",
  "windows-job-broker",
  "Mycellios.JobBroker.csproj",
);
const restore = dotnet([
  "restore",
  project,
  "--runtime",
  "win-x64",
  "--locked-mode",
  "--source",
  "https://api.nuget.org/v3/index.json",
  "/p:PublishAot=true",
]);
if (restore.error) throw restore.error;
if (restore.status !== 0) {
  throw new Error(
    restore.stderr?.trim()
      || restore.stdout?.trim()
      || `dotnet restore exited with ${restore.status ?? "no status"}`,
  );
}
const commonArguments = [
  "publish",
  project,
  "--configuration",
  "Release",
  "--runtime",
  "win-x64",
  "--self-contained",
  "true",
  "--output",
  outputDirectory,
  "--no-restore",
];
let result = dotnet([
  ...commonArguments,
  "/p:PublishAot=true",
]);
if (result.status !== 0) {
  const nativeAotFailure = result.stderr?.trim() || result.stdout?.trim();
  if (
    process.env.CI === "true"
    || process.env.MYCELLIOS_REQUIRE_NATIVE_AOT === "1"
  ) {
    throw new Error(
      nativeAotFailure
      || "The required Native AOT Windows Job Object broker build failed.",
    );
  }
  console.warn(
    `Native AOT broker build unavailable; using the self-contained single-file fallback.\n${nativeAotFailure}`,
  );
  rmSync(outputDirectory, { recursive: true, force: true });
  result = dotnet([
    ...commonArguments,
    "/p:PublishAot=false",
    "/p:PublishSingleFile=true",
    "/p:PublishTrimmed=true",
    "/p:IncludeNativeLibrariesForSelfExtract=true",
    "/p:EnableCompressionInSingleFile=true",
  ]);
}
if (result.error) throw result.error;
if (result.status !== 0) {
  throw new Error(
    result.stderr?.trim()
      || result.stdout?.trim()
      || `dotnet publish exited with ${result.status ?? "no status"}`,
  );
}
if (!existsSync(executable)) {
  throw new Error(`Windows Job Object broker was not produced: ${executable}`);
}

const probe = spawnSync(executable, ["--probe"], {
  encoding: "utf8",
  shell: false,
  windowsHide: true,
  timeout: 5_000,
});
if (
  probe.error
  || probe.status !== 0
  || probe.stdout.trim() !== "mycellios-windows-job-broker/1:ready"
) {
  throw probe.error
    ?? new Error(
      probe.stderr?.trim()
        || `Windows Job Object broker probe failed with ${probe.status ?? "no status"}`,
    );
}

console.log(`Windows Job Object broker ready: ${executable}`);

function dotnet(arguments_) {
  return spawnSync(
    dotnetExecutable,
    arguments_,
    {
      cwd: resolve("."),
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
      shell: false,
      windowsHide: true,
      env: {
        ...process.env,
        ...(dotnetExecutable === "dotnet"
          ? {}
          : { DOTNET_ROOT: dirname(dotnetExecutable) }),
        DOTNET_CLI_TELEMETRY_OPTOUT: "1",
        DOTNET_NOLOGO: "1",
      },
    },
  );
}

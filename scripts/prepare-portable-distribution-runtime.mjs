import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const workspace = resolve(import.meta.dirname, "..");
const venvPython = process.platform === "win32"
  ? join(workspace, "runtime", "distribution-venv", "Scripts", "python.exe")
  : join(workspace, "runtime", "distribution-venv", "bin", "python3");
if (!existsSync(venvPython)) {
  throw new Error("Distribution venv is missing; run npm run setup:distribution-runtime first.");
}

const probe = spawnSync(venvPython, ["-c", [
  "import json,sys,sysconfig",
  "print(json.dumps({'base':sys.base_prefix,'base_exe':sys._base_executable,'venv_site':sysconfig.get_paths()['purelib']}))",
].join("; ")], { encoding: "utf8", windowsHide: true });
if (probe.status !== 0) throw new Error(probe.stderr || "Could not inspect the distribution Python runtime.");
const info = JSON.parse(probe.stdout.trim());
const output = join(workspace, "build", "distribution-runtime");
const resolvedOutput = resolve(output);
const resolvedBuild = resolve(workspace, "build");
if (dirname(resolvedOutput) !== resolvedBuild) throw new Error("Portable runtime output escaped build directory.");
rmSync(resolvedOutput, { recursive: true, force: true });
mkdirSync(resolvedOutput, { recursive: true });

const ignore = (source) => {
  const name = source.split(/[\\/]/).at(-1)?.toLowerCase();
  return name === "__pycache__" || name === "site-packages" || name === "test" || name === "tests";
};
cpSync(info.base, resolvedOutput, { recursive: true, filter: (source) => !ignore(source) });

const baseSiteProbe = spawnSync(venvPython, ["-c", "import sys,sysconfig; print(sysconfig.get_path('purelib', vars={'base':sys.base_prefix,'platbase':sys.base_prefix}))"], { encoding: "utf8", windowsHide: true });
if (baseSiteProbe.status !== 0) throw new Error(baseSiteProbe.stderr || "Could not resolve portable site-packages path.");
const baseSite = baseSiteProbe.stdout.trim();
const siteRelative = relative(info.base, baseSite);
if (siteRelative.startsWith("..")) throw new Error("Python site-packages escaped the base runtime.");
const outputSite = join(resolvedOutput, siteRelative);
mkdirSync(outputSite, { recursive: true });
cpSync(info.venv_site, outputSite, { recursive: true, filter: (source) => !source.includes("__pycache__") });

const packagedPython = process.platform === "win32"
  ? join(resolvedOutput, "python.exe")
  : join(resolvedOutput, "bin", "python3");
if (!existsSync(packagedPython)) throw new Error(`Portable Python executable was not created at ${packagedPython}`);
const verifyScript = [
  "import accelerate,json,safetensors,sys,torch,transformers",
  "print(json.dumps({'pythonVersion':'.'.join(map(str,sys.version_info[:3])),'pythonAbi':f'cp{sys.version_info[0]}{sys.version_info[1]}','torchVersion':torch.__version__,'transformersVersion':transformers.__version__,'accelerateVersion':accelerate.__version__,'safetensorsVersion':safetensors.__version__,'cudaVersion':torch.version.cuda,'hipVersion':getattr(torch.version,'hip',None),'cudaAvailable':bool(torch.cuda.is_available())},sort_keys=True))",
].join("; ");
const verify = spawnSync(packagedPython, ["-c", verifyScript], { encoding: "utf8", windowsHide: true });
if (verify.status !== 0) throw new Error(verify.stderr || "Portable distribution runtime verification failed.");
const runtime = JSON.parse(verify.stdout.trim());
if (
  !String(runtime.pythonVersion).startsWith("3.12.") ||
  runtime.pythonAbi !== "cp312" ||
  runtime.torchVersion !== "2.13.0+cpu" ||
  runtime.cudaVersion !== null ||
  runtime.hipVersion !== null ||
  runtime.cudaAvailable !== false
) {
  throw new Error(`Portable runtime is not the certified CPython 3.12 CPU bootstrap: ${verify.stdout.trim()}`);
}
const manifest = {
  schema: "mycellios-distribution-runtime/2",
  platform: process.platform,
  arch: process.arch,
  pythonVersion: runtime.pythonVersion,
  pythonAbi: runtime.pythonAbi,
  executable: process.platform === "win32" ? "python.exe" : "bin/python3",
  torchVersion: runtime.torchVersion,
  transformersVersion: runtime.transformersVersion,
  accelerateVersion: runtime.accelerateVersion,
  safetensorsVersion: runtime.safetensorsVersion,
  backend: "cpu",
};
writeFileSync(
  join(resolvedOutput, "runtime-manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
  "utf8",
);
const archive = join(workspace, "build", "distribution-runtime.tar.gz");
rmSync(archive, { force: true });
const packed = spawnSync("tar", ["-czf", archive, "-C", resolvedOutput, "."], { encoding: "utf8", windowsHide: true });
if (packed.status !== 0 || !existsSync(archive)) throw new Error(packed.stderr || "Could not archive the portable distribution runtime.");
process.stdout.write(
  `Portable distribution runtime v2 ready: Python ${manifest.pythonVersion}, ${manifest.torchVersion} (${manifest.backend})\n`,
);

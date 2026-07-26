import { existsSync, readFileSync, realpathSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import {
  NATIVE_PYTHON_IMPORT_SMOKE_MODULES,
  NATIVE_PYTHON_PRODUCT_FILES,
  NATIVE_PYTHON_PRODUCT_MANIFEST,
  verifyNativePythonProductSource,
} from "./native-python-product-policy.mjs";

export const INSTALLED_STAGE_CANARY_SCHEMA =
  "mycellios-installed-stage-canary/1";
export const INSTALLED_STAGE_CANARY_MARKER =
  "MYCELLIOS_INSTALLED_STAGE_CANARY=";

const REQUIRED_SOURCE_FILES = [
  ...NATIVE_PYTHON_PRODUCT_FILES,
  NATIVE_PYTHON_PRODUCT_MANIFEST,
].map((path) => path.split("/"));

/**
 * Execute the ABI canary with the exact interpreter and source tree that will
 * ship. Dependencies are injectable so failure behavior can be unit-tested
 * without pretending a mocked process is physical evidence.
 */
export function runInstalledStageCanary(options, dependencies = {}) {
  const spawn = dependencies.spawnSync ?? spawnSync;
  const fileExists = dependencies.existsSync ?? existsSync;
  const readFile = dependencies.readFileSync ?? readFileSync;
  const verifyProductSource =
    dependencies.verifyNativePythonProductSource
    ?? verifyNativePythonProductSource;
  const pythonExecutable = resolveRequiredPath(
    options?.pythonExecutable,
    "installed canary Python executable",
  );
  const pythonSourceRoot = resolveRequiredPath(
    options?.pythonSourceRoot,
    "installed canary Python source root",
  );
  if (!fileExists(pythonExecutable)) {
    throw new Error(`Installed stage canary Python is missing: ${pythonExecutable}.`);
  }
  for (const parts of REQUIRED_SOURCE_FILES) {
    const required = join(pythonSourceRoot, ...parts);
    if (!fileExists(required)) {
      throw new Error(`Installed stage canary source is missing: ${required}.`);
    }
  }
  verifyProductSource(pythonSourceRoot);
  const registryContract = readAdapterRegistryContract(pythonSourceRoot, readFile);

  const bootstrap = [
    "import importlib,json,runpy,sys",
    "source=sys.argv[1]",
    "sys.path.insert(0,source)",
    "modules=json.loads(sys.argv[2])",
    "[importlib.import_module(module) for module in modules]",
    "sys.argv=['mycellios-installed-stage-canary']",
    "runpy.run_module('distributed_runtime.installed_stage_canary',run_name='__main__')",
  ].join("; ");
  const result = spawn(
    pythonExecutable,
    [
      "-I",
      "-B",
      "-c",
      bootstrap,
      pythonSourceRoot,
      JSON.stringify(NATIVE_PYTHON_IMPORT_SMOKE_MODULES),
    ],
    {
      cwd: dirname(pythonSourceRoot),
      encoding: "utf8",
      env: {
        ...process.env,
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
        HF_DATASETS_OFFLINE: "1",
        TOKENIZERS_PARALLELISM: "false",
        PYTHONNOUSERSITE: "1",
        PYTHONDONTWRITEBYTECODE: "1",
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
      },
      maxBuffer: 16 * 1024 * 1024,
      shell: false,
      windowsHide: true,
    },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      result.stderr?.trim() ||
        `Installed stage canary exited with ${result.status ?? "no status"}.`,
    );
  }
  const markerLines = String(result.stdout ?? "")
    .split(/\r?\n/)
    .filter((line) => line.startsWith(INSTALLED_STAGE_CANARY_MARKER));
  if (markerLines.length !== 1) {
    throw new Error(
      "Installed stage canary did not emit exactly one authenticated result marker.",
    );
  }
  let evidence;
  try {
    evidence = JSON.parse(
      markerLines[0].slice(INSTALLED_STAGE_CANARY_MARKER.length),
    );
  } catch {
    throw new Error("Installed stage canary emitted invalid JSON evidence.");
  }
  validateInstalledStageCanaryEvidence(evidence, {
    ...options,
    expectedAdapterRegistryId: registryContract.registryId,
    expectedAdapterContractId: registryContract.adapterContractId,
  });
  verifyProductSource(pythonSourceRoot);
  return evidence;
}

export function validateInstalledStageCanaryEvidence(evidence, options = {}) {
  if (
    !isObject(evidence) ||
    evidence.schema !== INSTALLED_STAGE_CANARY_SCHEMA ||
    evidence.ok !== true ||
    evidence.engine !== "python-torch" ||
    evidence.adapter !== "transformers-llama-v1" ||
    !/^sha256:[0-9a-f]{64}$/.test(String(evidence.adapterContractId ?? "")) ||
    !/^sha256:[0-9a-f]{64}$/.test(String(evidence.adapterRegistryId ?? "")) ||
    evidence.loader !== "selective-safetensors" ||
    evidence.batchSize !== 2 ||
    !positiveInteger(evidence.physicalBatchCalls, 2) ||
    !positiveInteger(evidence.physicalBatchItems, 4) ||
    evidence.sequenceTokens !== 3 ||
    !positiveInteger(evidence.kvBytes) ||
    evidence.copiedKvBytes !== evidence.kvBytes ||
    evidence.batchQueueBatches !== 1 ||
    !/^[0-9a-f]{64}$/.test(String(evidence.outputTokenSha256 ?? "")) ||
    !isObject(evidence.parity) ||
    evidence.parity.sequentialVsBatch !== true ||
    evidence.parity.fork !== true ||
    evidence.parity.rollback !== true
  ) {
    throw new Error(
      `Installed stage canary parity evidence is invalid: ${JSON.stringify(evidence)}.`,
    );
  }
  if (
    options.expectedAdapterRegistryId
    && evidence.adapterRegistryId !== options.expectedAdapterRegistryId
  ) {
    throw new Error("Installed stage canary used a different adapter registry.");
  }
  if (
    options.expectedAdapterContractId
    && evidence.adapterContractId !== options.expectedAdapterContractId
  ) {
    throw new Error("Installed stage canary used a different adapter contract.");
  }
  if (
    options.expectedTorchVersion &&
    evidence.torchVersion !== options.expectedTorchVersion
  ) {
    throw new Error(
      `Installed stage canary used Torch ${evidence.torchVersion}; expected ${options.expectedTorchVersion}.`,
    );
  }
  if (
    options.expectedTransformersVersion &&
    evidence.transformersVersion !== options.expectedTransformersVersion
  ) {
    throw new Error(
      `Installed stage canary used Transformers ${evidence.transformersVersion}; expected ${options.expectedTransformersVersion}.`,
    );
  }
  if (
    options.expectedPythonPrefix &&
    !sameExistingPath(evidence.pythonPrefix, options.expectedPythonPrefix)
  ) {
    throw new Error(
      `Installed stage canary escaped its runtime: ${evidence.pythonPrefix}.`,
    );
  }
}

function readAdapterRegistryContract(pythonSourceRoot, readFile) {
  const path = join(
    pythonSourceRoot,
    "distributed_runtime",
    "model_adapter_registry.json",
  );
  let document;
  try {
    document = JSON.parse(readFile(path, "utf8"));
  } catch {
    throw new Error(`Installed stage canary adapter registry is invalid: ${path}.`);
  }
  const adapter = Array.isArray(document?.adapters)
    ? document.adapters.find((entry) => entry?.id === "transformers-llama-v1")
    : undefined;
  if (
    document?.schema !== "mycellios-transformers-stage-adapters/2"
    || !/^sha256:[0-9a-f]{64}$/.test(String(document.registryId ?? ""))
    || !adapter
  ) {
    throw new Error(`Installed stage canary adapter registry is unsupported: ${path}.`);
  }
  // The Python runtime independently verifies the self-hash and exact
  // implementation set during import. This preflight binds the JS verifier to
  // the same file and to the exact adapter entry it expects the canary to load.
  const adapterContractId = `sha256:${createHash("sha256")
    .update(canonicalJson(adapter))
    .digest("hex")}`;
  return { registryId: document.registryId, adapterContractId };
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
}

function resolveRequiredPath(value, name) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${name} is required.`);
  }
  return resolve(value);
}

function positiveInteger(value, minimum = 1) {
  return Number.isInteger(value) && value >= minimum;
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function sameExistingPath(left, right) {
  if (typeof left !== "string" || !left.trim()) return false;
  const normalize = (value) => {
    const path = realpathSync.native(resolve(value)).replaceAll("\\", "/");
    return process.platform === "win32" ? path.toLowerCase() : path;
  };
  try {
    return normalize(left) === normalize(right);
  } catch {
    return false;
  }
}

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";

export const MODEL_ADAPTER_REGISTRY_SCHEMA =
  "mycellios-transformers-stage-adapters/2" as const;
export const MODEL_ADAPTER_EVIDENCE_SCOPE = "software-contract-only" as const;

export interface ModelAdapterContract {
  readonly id: string;
  readonly modelType: string;
  readonly architectures: readonly [string];
  readonly requiredLayerModules: readonly string[];
  readonly semanticFeatures: readonly string[];
  readonly attentionScope: "full-only";
  readonly parallelismScope: "pipeline-stage" | "pipeline-only";
  readonly implementationContract: string;
  readonly adapterContractId: string;
}

export interface ModelAdapterRegistry {
  readonly schema: typeof MODEL_ADAPTER_REGISTRY_SCHEMA;
  readonly registryVersion: 2;
  /**
   * This registry proves that Mycellios has an exact software adapter. It is
   * deliberately not evidence that a backend/device tuple passed a physical
   * campaign; ExecutorCompatibilityRegistry owns that stronger decision.
   */
  readonly evidenceScope: typeof MODEL_ADAPTER_EVIDENCE_SCOPE;
  readonly registryId: string;
  readonly adapters: readonly ModelAdapterContract[];
}

interface StoredModelAdapterContract {
  id: string;
  modelType: string;
  architectures: [string];
  requiredLayerModules: string[];
  semanticFeatures: string[];
  attentionScope: "full-only";
  parallelismScope: "pipeline-stage" | "pipeline-only";
  implementationContract: string;
}

interface StoredModelAdapterRegistry {
  schema: typeof MODEL_ADAPTER_REGISTRY_SCHEMA;
  registryVersion: 2;
  evidenceScope: typeof MODEL_ADAPTER_EVIDENCE_SCOPE;
  registryId: string;
  adapters: StoredModelAdapterContract[];
}

/**
 * Executable contracts present in this TypeScript/Python release.
 *
 * The JSON remains the canonical model catalogue. This small code-owned map
 * is deliberately limited to implementation availability: a locally edited
 * registry cannot make the coordinator advertise a family that the installed
 * runtime does not actually implement.
 */
const INSTALLED_IMPLEMENTATION_CONTRACTS: ReadonlyMap<
  string,
  {
    readonly modelType: string;
    readonly parallelismScope: StoredModelAdapterContract["parallelismScope"];
  }
> = new Map([
  ["mycellios-selective-llama/1", {
    modelType: "llama",
    parallelismScope: "pipeline-stage",
  }],
  ["mycellios-selective-qwen3/1", {
    modelType: "qwen3",
    parallelismScope: "pipeline-stage",
  }],
  ["mycellios-selective-qwen3-moe/1", {
    modelType: "qwen3_moe",
    parallelismScope: "pipeline-only",
  }],
  ["mycellios-selective-glm4-moe/1", {
    modelType: "glm4_moe",
    parallelismScope: "pipeline-only",
  }],
]);

const source = loadStoredRegistry();
const registryId = source.document.registryId;
const contracts = source.document.adapters.map((adapter) => Object.freeze({
  ...adapter,
  architectures: Object.freeze([...adapter.architectures]) as readonly [string],
  requiredLayerModules: Object.freeze([...adapter.requiredLayerModules]),
  semanticFeatures: Object.freeze([...adapter.semanticFeatures]),
  adapterContractId: sha256CanonicalEvidence(adapter),
}));

export const MODEL_ADAPTER_REGISTRY_ID = registryId;
export const MODEL_ADAPTER_REGISTRY_SOURCE = source.path;

const registry: ModelAdapterRegistry = Object.freeze({
  ...source.document,
  registryId,
  adapters: Object.freeze(contracts),
});

const byModelIdentity = new Map(
  contracts.map((adapter) => [
    `${adapter.modelType}\0${adapter.architectures[0]}`,
    adapter,
  ]),
);

export function modelAdapterRegistry(): ModelAdapterRegistry {
  return registry;
}

export function resolveModelAdapterContract(
  modelType: string | null,
  architecture: string | null,
): ModelAdapterContract | null {
  if (modelType === null || architecture === null) return null;
  return byModelIdentity.get(`${modelType}\0${architecture}`) ?? null;
}

function loadStoredRegistry(): {
  path: string;
  document: StoredModelAdapterRegistry;
} {
  const resourcesPath = (process as NodeJS.Process & { resourcesPath?: string }).resourcesPath;
  const candidates = [
    ...(resourcesPath
      ? [resolve(resourcesPath, "python", "distributed_runtime", "model_adapter_registry.json")]
      : []),
    // Both src/contracts and dist/contracts belong to this installation. A
    // library importer or service working directory cannot select another copy.
    resolve(import.meta.dirname, "../..", "python", "distributed_runtime", "model_adapter_registry.json"),
  ];
  const path = candidates.find((candidate) => existsSync(candidate));
  if (!path) {
    throw new Error("mycellios_model_adapter_registry_is_missing");
  }
  let value: unknown;
  try {
    value = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    throw new Error("mycellios_model_adapter_registry_is_invalid_json");
  }
  return { path, document: validateStoredRegistry(value) };
}

function validateStoredRegistry(value: unknown): StoredModelAdapterRegistry {
  assertExactRecord(
    value,
    ["schema", "registryVersion", "evidenceScope", "registryId", "adapters"],
    "mycellios_model_adapter_registry_is_invalid",
  );
  if (
    value.schema !== MODEL_ADAPTER_REGISTRY_SCHEMA
    || value.registryVersion !== 2
    || value.evidenceScope !== MODEL_ADAPTER_EVIDENCE_SCOPE
    || typeof value.registryId !== "string"
    || !Array.isArray(value.adapters)
    || value.adapters.length < 1
  ) {
    throw new Error("mycellios_model_adapter_registry_is_unsupported");
  }

  const adapters = value.adapters.map((entry, index) => validateStoredAdapter(entry, index));
  const ids = new Set<string>();
  const modelIdentities = new Set<string>();
  const implementations = new Set<string>();
  for (const adapter of adapters) {
    if (ids.has(adapter.id)) {
      throw new Error("mycellios_model_adapter_registry_has_duplicate_adapter");
    }
    ids.add(adapter.id);
    const key = `${adapter.modelType}\0${adapter.architectures[0]}`;
    if (modelIdentities.has(key)) {
      throw new Error("mycellios_model_adapter_registry_has_ambiguous_model_identity");
    }
    modelIdentities.add(key);
    const installed = INSTALLED_IMPLEMENTATION_CONTRACTS.get(
      adapter.implementationContract,
    );
    if (
      installed === undefined
      || installed.modelType !== adapter.modelType
      || installed.parallelismScope !== adapter.parallelismScope
      || implementations.has(adapter.implementationContract)
    ) {
      throw new Error("mycellios_model_adapter_registry_implementation_is_not_installed");
    }
    implementations.add(adapter.implementationContract);
  }
  if (implementations.size !== INSTALLED_IMPLEMENTATION_CONTRACTS.size) {
    throw new Error("mycellios_model_adapter_registry_implementation_is_missing");
  }
  const identityBody = {
    schema: MODEL_ADAPTER_REGISTRY_SCHEMA,
    registryVersion: 2 as const,
    evidenceScope: MODEL_ADAPTER_EVIDENCE_SCOPE,
    adapters,
  };
  const registryId = sha256CanonicalEvidence(identityBody);
  if (value.registryId !== registryId) {
    throw new Error("mycellios_model_adapter_registry_identity_mismatch");
  }
  const document: StoredModelAdapterRegistry = {
    ...identityBody,
    registryId,
  };
  // Exercise the strict canonicalizer during startup, not only when hashing.
  canonicalEvidenceJson(document);
  return document;
}

function validateStoredAdapter(value: unknown, index: number): StoredModelAdapterContract {
  const error = `mycellios_model_adapter_registry_adapter_${index}_is_invalid`;
  assertExactRecord(
    value,
    [
      "id",
      "modelType",
      "architectures",
      "requiredLayerModules",
      "semanticFeatures",
      "attentionScope",
      "parallelismScope",
      "implementationContract",
    ],
    error,
  );
  if (
    !nonEmptyString(value.id)
    || !nonEmptyString(value.modelType)
    || !singleString(value.architectures)
    || !uniqueStringList(value.requiredLayerModules)
    || !uniqueStringList(value.semanticFeatures)
    || value.attentionScope !== "full-only"
    || (value.parallelismScope !== "pipeline-stage"
      && value.parallelismScope !== "pipeline-only")
    || !nonEmptyString(value.implementationContract)
  ) {
    throw new Error(error);
  }
  return {
    id: value.id,
    modelType: value.modelType,
    architectures: [value.architectures[0]],
    requiredLayerModules: [...value.requiredLayerModules],
    semanticFeatures: [...value.semanticFeatures],
    attentionScope: value.attentionScope,
    parallelismScope: value.parallelismScope,
    implementationContract: value.implementationContract,
  };
}

function assertExactRecord(
  value: unknown,
  fields: readonly string[],
  error: string,
): asserts value is Record<string, unknown> {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
    || Object.keys(value).sort().join("\0") !== [...fields].sort().join("\0")
  ) {
    throw new Error(error);
  }
}

function nonEmptyString(value: unknown): value is string {
  return (
    typeof value === "string"
    && value.trim().length > 0
    && /^[\x20-\x7e]+$/.test(value)
  );
}

function singleString(value: unknown): value is [string] {
  return Array.isArray(value) && value.length === 1 && nonEmptyString(value[0]);
}

function uniqueStringList(value: unknown): value is string[] {
  return (
    Array.isArray(value)
    && value.length > 0
    && value.every(nonEmptyString)
    && new Set(value).size === value.length
  );
}

import type { StoredRequestedModel, StoredWorker } from "../storage/store.js";
import type { HubCatalogModel, HubCatalogPage, HubCatalogSearchInput } from "../contracts/types.js";
import {
  MODEL_ADAPTER_EVIDENCE_SCOPE,
  MODEL_ADAPTER_REGISTRY_ID,
  resolveModelAdapterContract,
} from "../contracts/model-adapter-registry.js";
import type { ActivationIncident } from "./activation-incident.js";

const MIB = 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES = 64 * MIB;
const HUB_MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

export async function searchHubModelCatalog(
  query: string,
  fetcher: typeof fetch = fetch,
  options: Omit<HubCatalogSearchInput, "query"> = {},
): Promise<HubCatalogPage> {
  const normalizedQuery = query.trim().slice(0, 80);
  const limit = Math.max(10, Math.min(100, Math.round(options.limit ?? 50)));
  const sort = options.sort === "likes" || options.sort === "lastModified" ? options.sort : "downloads";
  const parameters = new URLSearchParams({
    pipeline_tag: "text-generation",
    sort,
    direction: "-1",
    limit: String(limit),
  });
  for (const property of [
    "author",
    "config",
    "downloads",
    "gated",
    "lastModified",
    "likes",
    "pipeline_tag",
    "safetensors",
    "tags",
  ]) parameters.append("expand", property);
  if (normalizedQuery) parameters.set("search", normalizedQuery);
  if (options.cursor) parameters.set("cursor", options.cursor.slice(0, 4_096));

  const response = await fetcher(`https://huggingface.co/api/models?${parameters.toString()}`, {
    headers: { accept: "application/json" },
    redirect: "follow",
    signal: AbortSignal.timeout(15_000),
  });
  if (!response.ok) throw new Error(`huggingface_catalog_http_${response.status}`);
  const value = await response.json() as unknown;
  if (!Array.isArray(value)) throw new Error("huggingface_catalog_is_invalid");

  const models = value.flatMap((entry): HubCatalogModel[] => {
    if (!isRecord(entry) || typeof entry.id !== "string" || !HUB_MODEL_ID.test(entry.id)) return [];
    if (entry.private === true) return [];
    const config = isRecord(entry.config) ? entry.config : {};
    const modelType = typeof config.model_type === "string" ? config.model_type : null;
    const architecture = Array.isArray(config.architectures) && typeof config.architectures[0] === "string"
      ? config.architectures[0]
      : null;
    const adapter = resolveModelAdapterContract(modelType, architecture);
    const gated = entry.gated !== false && entry.gated !== undefined && entry.gated !== null;
    const tags = Array.isArray(entry.tags) ? entry.tags.filter((tag): tag is string => typeof tag === "string") : [];
    const hasSafetensors = tags.includes("safetensors");
    const parameterCount = hubSafetensorsParameterCount(entry.safetensors);
    const compatible = adapter !== null && !gated && hasSafetensors;
    const compatibilityReason = gated
      ? "Access approval is required on Hugging Face."
      : !hasSafetensors
        ? "No public Safetensors checkpoint was advertised."
        : adapter === null
          ? "This architecture does not have a registered native Mycellios adapter yet."
          : null;
    return [{
      id: entry.id,
      author: typeof entry.author === "string" ? entry.author : entry.id.split("/")[0]!,
      downloads: nonNegativeNumber(entry.downloads),
      likes: nonNegativeNumber(entry.likes),
      lastModified: typeof entry.lastModified === "string" ? entry.lastModified : null,
      pipelineTag: typeof entry.pipeline_tag === "string" ? entry.pipeline_tag : null,
      modelType,
      architecture,
      adapterId: adapter?.id ?? null,
      adapterContractId: adapter?.adapterContractId ?? null,
      adapterRegistryId: MODEL_ADAPTER_REGISTRY_ID,
      adapterEvidenceScope: MODEL_ADAPTER_EVIDENCE_SCOPE,
      compatible,
      gated,
      compatibilityReason,
      parameterCount,
      estimatedMemoryMiB: estimateCatalogMemoryMiB(parameterCount),
      memoryEstimateSource: parameterCount === null ? null : "hub_metadata",
    }];
  });

  return {
    data: models,
    nextCursor: nextHubCursor(response.headers.get("link")),
  };
}

function nextHubCursor(linkHeader: string | null): string | null {
  if (!linkHeader) return null;
  for (const link of linkHeader.split(",")) {
    if (!/rel\s*=\s*"?next"?/i.test(link)) continue;
    const target = link.match(/<([^>]+)>/)?.[1];
    if (!target) continue;
    try {
      return new URL(target).searchParams.get("cursor");
    } catch {
      return null;
    }
  }
  return null;
}

function hubSafetensorsParameterCount(value: unknown): number | null {
  if (!isRecord(value)) return null;
  const total = value.total;
  return typeof total === "number" && Number.isSafeInteger(total) && total > 0 ? total : null;
}

function estimateCatalogMemoryMiB(parameterCount: number | null): number | null {
  if (parameterCount === null) return null;
  const weightBytes = parameterCount * 2;
  const runtimeBytes = 2 * 256 * MIB;
  const loaderAndActivationReserve = Math.ceil(weightBytes * 0.1) + 2 * 64 * MIB;
  return Math.ceil((weightBytes + runtimeBytes + loaderAndActivationReserve) / MIB);
}

export type RequestedModelStatus =
  | "profiling"
  | "waiting_capacity"
  | "ready"
  | "activating"
  | "active"
  | "incompatible"
  | "failed";

export interface HubModelCapacityProfile {
  schema: "mycellios-hub-model-capacity/1";
  adapterId: string | null;
  adapterContractId: string | null;
  adapterRegistryId: string;
  adapterEvidenceScope: typeof MODEL_ADAPTER_EVIDENCE_SCOPE;
  compatible: boolean;
  incompatibilityReason: string | null;
  architecture: string | null;
  modelType: string | null;
  totalLayers: number;
  hiddenSize: number;
  weightBytes: number;
  requiredVramMiB: number;
  minimumStageVramMiB: number;
  minimumNodes: number;
  contextTokens: number;
  source: string;
  revision: string;
}

export interface RequestedModelCapacityView {
  id: string;
  source: string;
  revision: string | null;
  status: RequestedModelStatus;
  autoActivate: boolean;
  adapterId: string | null;
  compatible: boolean | null;
  requiredVramMiB: number | null;
  availableVramMiB: number;
  missingVramMiB: number | null;
  requiredNodes: number;
  availableNodes: number;
  missingNodes: number;
  weightBytes: number | null;
  contextTokens: number;
  message: string;
  activationIncident: ActivationIncident | null;
  activationProgress: readonly ModelActivationProgressEvent[];
  activationRequestedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ModelActivationProgressEvent {
  phase: string;
  message: string;
  at: string;
  state: "running" | "completed" | "failed";
  nodeId?: string;
  processId?: string;
  device?: string;
  details?: readonly string[];
}

export interface ModelExecutionCapacityNode {
  id: string;
  availableVramMiB: number;
}

export async function inspectHubModelCapacity(
  input: {
    source: string;
    revision: string | null;
    contextTokens: number;
    minimumNodes: number;
  },
  fetcher: typeof fetch = fetch,
): Promise<HubModelCapacityProfile> {
  if (!HUB_MODEL_ID.test(input.source)) throw new Error("invalid_huggingface_model_id");
  const revision = input.revision ?? "main";
  const base = `https://huggingface.co/${input.source}/resolve/${encodeURIComponent(revision)}`;
  const config = await fetchJsonRecord(`${base}/config.json`, fetcher, "model_config");
  const modelType = typeof config.model_type === "string" ? config.model_type : null;
  const architecture = Array.isArray(config.architectures) && typeof config.architectures[0] === "string"
    ? config.architectures[0]
    : null;
  const adapter = resolveModelAdapterContract(modelType, architecture);
  const totalLayers = positiveConfigInteger(config, "num_hidden_layers", "n_layer", "num_layers");
  const hiddenSize = positiveConfigInteger(config, "hidden_size", "n_embd", "d_model");
  const attentionHeads = positiveConfigInteger(config, "num_attention_heads", "n_head");
  const kvHeads = optionalPositiveInteger(config.num_key_value_heads) ?? attentionHeads;
  const headDim = optionalPositiveInteger(config.head_dim) ?? Math.ceil(hiddenSize / attentionHeads);
  const weightBytes = await hubSafetensorsWeightBytes(base, fetcher);
  const kvBytes = 2 * kvHeads * headDim * 2 * totalLayers * input.contextTokens;
  const runtimeBytes = input.minimumNodes * 256 * MIB;
  const loaderAndActivationReserve = Math.ceil(weightBytes * 0.1) + input.minimumNodes * 64 * MIB;
  const requiredVramMiB = Math.ceil((weightBytes + kvBytes + runtimeBytes + loaderAndActivationReserve) / MIB);
  const minimumStageVramMiB = Math.max(
    512,
    Math.ceil(requiredVramMiB / input.minimumNodes * 0.55),
  );
  return {
    schema: "mycellios-hub-model-capacity/1",
    adapterId: adapter?.id ?? null,
    adapterContractId: adapter?.adapterContractId ?? null,
    adapterRegistryId: MODEL_ADAPTER_REGISTRY_ID,
    adapterEvidenceScope: MODEL_ADAPTER_EVIDENCE_SCOPE,
    compatible: adapter !== null,
    incompatibilityReason: adapter === null
      ? `No native adapter in registry ${MODEL_ADAPTER_REGISTRY_ID} for model_type=${modelType ?? "unknown"}, architecture=${architecture ?? "unknown"}`
      : null,
    architecture,
    modelType,
    totalLayers,
    hiddenSize,
    weightBytes,
    requiredVramMiB,
    minimumStageVramMiB,
    minimumNodes: input.minimumNodes,
    contextTokens: input.contextTokens,
    source: input.source,
    revision,
  };
}

export function requestedModelCapacityViews(input: {
  requests: StoredRequestedModel[];
  workers: StoredWorker[];
  connectedWorkerIds: ReadonlySet<string>;
  activeModelIds: ReadonlySet<string>;
  executionNodesForModel?: (modelId: string) => readonly ModelExecutionCapacityNode[];
  activationProgressForModel?: (modelId: string) => readonly ModelActivationProgressEvent[];
  activationStatusMessageForModel?: (modelId: string) => string | null;
  activationIncidentForModel?: (modelId: string) => ActivationIncident | null;
  activationAvailable?: boolean;
}): RequestedModelCapacityView[] {
  const workerCapacity = input.workers
    .filter((worker) => input.connectedWorkerIds.has(worker.id) && worker.status === "online")
    .map((worker) => {
      const offered = worker.capabilities.gpus.reduce(
        (sum, gpu) => sum + Math.min(gpu.offeredVramMb, gpu.freeOfferedVramMb),
        0,
      );
      const reserved = worker.capabilities.deployments.reduce(
        (sum, deployment) => sum + deployment.peakVramMb,
        0,
      );
      return { id: worker.id, availableVramMiB: Math.max(0, offered - reserved) };
    });
  return input.requests.map((request) => {
    const capacityNodes = input.executionNodesForModel
      ? input.executionNodesForModel(request.id)
      : workerCapacity;
    const profile = parseProfile(request.profile);
    const requiredNodes = profile?.minimumNodes ?? request.minimumNodes;
    const minimumStage = profile?.minimumStageVramMiB ?? 512;
    const eligible = capacityNodes.filter((node) => node.availableVramMiB >= minimumStage);
    const availableVramMiB = eligible.reduce((sum, node) => sum + node.availableVramMiB, 0);
    const requiredVramMiB = profile?.requiredVramMiB ?? null;
    const missingVramMiB = requiredVramMiB === null
      ? null
      : Math.max(0, requiredVramMiB - availableVramMiB);
    const missingNodes = Math.max(0, requiredNodes - eligible.length);
    const active = input.activeModelIds.has(request.id);
    const compatible = profile?.compatible ?? null;
    let status: RequestedModelStatus;
    let message: string;
    if (active) {
      status = "active";
      message = "Model active and available for inference.";
    } else if (request.profileError) {
      status = "failed";
      message = request.profileError;
    } else if (request.activationError) {
      status = "failed";
      message = `Automatic activation failed: ${request.activationError}`;
    } else if (!profile) {
      status = "profiling";
      message = "Reading model metadata and calculating capacity.";
    } else if (!profile.compatible) {
      status = "incompatible";
      message = profile.incompatibilityReason ?? "This architecture has no registered native adapter.";
    } else if ((missingVramMiB ?? 0) > 0 || missingNodes > 0) {
      status = "waiting_capacity";
      message = capacityMessage(missingVramMiB ?? 0, missingNodes);
    } else if (request.autoActivate && (input.activationAvailable ?? true)) {
      status = "activating";
      message = input.activationStatusMessageForModel?.(request.id)
        ?? "Capacity reached. Automatic activation is queued.";
    } else {
      status = "ready";
      message = request.autoActivate
        ? "Capacity reached. The automatic deployment service is not configured."
        : "Capacity reached. Ready to activate.";
    }
    return {
      id: request.id,
      source: request.source,
      revision: request.revision,
      status,
      autoActivate: request.autoActivate,
      adapterId: profile?.adapterId ?? null,
      compatible,
      requiredVramMiB,
      availableVramMiB,
      missingVramMiB,
      requiredNodes,
      availableNodes: eligible.length,
      missingNodes,
      weightBytes: profile?.weightBytes ?? null,
      contextTokens: request.contextTokens,
      message,
      activationIncident: input.activationIncidentForModel?.(request.id) ?? null,
      activationProgress: input.activationProgressForModel?.(request.id) ?? [],
      activationRequestedAt: request.activationRequestedAt === null
        ? null
        : new Date(request.activationRequestedAt).toISOString(),
      createdAt: new Date(request.createdAt).toISOString(),
      updatedAt: new Date(request.updatedAt).toISOString(),
    };
  });
}

export function shouldQueueAutomaticActivation(view: RequestedModelCapacityView): boolean {
  return view.status === "activating" && view.autoActivate && view.activationRequestedAt === null;
}

async function hubSafetensorsWeightBytes(base: string, fetcher: typeof fetch): Promise<number> {
  const indexResponse = await fetcher(`${base}/model.safetensors.index.json`, {
    headers: { accept: "application/json" },
    redirect: "follow",
    signal: AbortSignal.timeout(30_000),
  });
  if (indexResponse.ok) {
    const index = await indexResponse.json() as unknown;
    if (
      isRecord(index) &&
      isRecord(index.metadata) &&
      Number.isSafeInteger(index.metadata.total_size) &&
      (index.metadata.total_size as number) > 0
    ) {
      return index.metadata.total_size as number;
    }
    throw new Error("safetensors_index_has_no_total_size");
  }
  if (indexResponse.status !== 404) {
    throw new Error(`safetensors_index_http_${indexResponse.status}`);
  }
  const file = `${base}/model.safetensors`;
  const lengthResponse = await fetcher(file, {
    headers: { range: "bytes=0-7", "accept-encoding": "identity" },
    redirect: "follow",
    signal: AbortSignal.timeout(30_000),
  });
  if (lengthResponse.status !== 206) throw new Error("safetensors_range_request_not_supported");
  const lengthBytes = new Uint8Array(await lengthResponse.arrayBuffer());
  if (lengthBytes.length !== 8) throw new Error("safetensors_header_length_is_invalid");
  const headerLength = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > MAX_SAFETENSORS_HEADER_BYTES) {
    throw new Error("safetensors_header_length_is_invalid");
  }
  const headerResponse = await fetcher(file, {
    headers: {
      range: `bytes=8-${7 + headerLength}`,
      "accept-encoding": "identity",
    },
    redirect: "follow",
    signal: AbortSignal.timeout(30_000),
  });
  if (headerResponse.status !== 206) throw new Error("safetensors_header_range_not_supported");
  const header = await headerResponse.json() as unknown;
  if (!isRecord(header)) throw new Error("safetensors_header_is_invalid");
  let total = 0;
  for (const [name, tensor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (
      !isRecord(tensor) ||
      !Array.isArray(tensor.data_offsets) ||
      tensor.data_offsets.length !== 2 ||
      !tensor.data_offsets.every((value) => Number.isSafeInteger(value) && (value as number) >= 0)
    ) {
      throw new Error("safetensors_tensor_metadata_is_invalid");
    }
    total += (tensor.data_offsets[1] as number) - (tensor.data_offsets[0] as number);
  }
  if (!Number.isSafeInteger(total) || total <= 0) throw new Error("safetensors_weight_size_is_invalid");
  return total;
}

async function fetchJsonRecord(
  url: string,
  fetcher: typeof fetch,
  name: string,
): Promise<Record<string, unknown>> {
  const response = await fetcher(url, {
    headers: { accept: "application/json" },
    redirect: "follow",
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`${name}_http_${response.status}`);
  const value = await response.json() as unknown;
  if (!isRecord(value)) throw new Error(`${name}_is_invalid`);
  return value;
}

function positiveConfigInteger(config: Record<string, unknown>, ...names: string[]): number {
  for (const name of names) {
    const value = optionalPositiveInteger(config[name]);
    if (value !== null) return value;
  }
  throw new Error(`model_config_missing_${names[0]}`);
}

function optionalPositiveInteger(value: unknown): number | null {
  return Number.isSafeInteger(value) && (value as number) > 0 ? value as number : null;
}

function nonNegativeNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0;
}

function parseProfile(value: Record<string, unknown> | null): HubModelCapacityProfile | null {
  if (!value || value.schema !== "mycellios-hub-model-capacity/1") return null;
  if (value.adapterRegistryId !== MODEL_ADAPTER_REGISTRY_ID) return null;
  if (value.adapterEvidenceScope !== MODEL_ADAPTER_EVIDENCE_SCOPE) return null;
  const adapter = resolveModelAdapterContract(
    typeof value.modelType === "string" ? value.modelType : null,
    typeof value.architecture === "string" ? value.architecture : null,
  );
  if (
    value.compatible === true
    && (
      adapter === null
      || value.adapterId !== adapter.id
      || value.adapterContractId !== adapter.adapterContractId
    )
  ) {
    return null;
  }
  if (
    value.compatible === false
    && (value.adapterId !== null || value.adapterContractId !== null)
  ) {
    return null;
  }
  return value as unknown as HubModelCapacityProfile;
}

function capacityMessage(missingVramMiB: number, missingNodes: number): string {
  const parts: string[] = [];
  if (missingVramMiB > 0) parts.push(`about ${missingVramMiB} MiB more capacity`);
  if (missingNodes > 0) parts.push(`${missingNodes} more compatible node${missingNodes === 1 ? "" : "s"}`);
  return `Waiting for ${parts.join(" and ")}.`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

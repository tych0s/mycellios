import { z } from "zod";
import type { ChatCompletionRequest } from "./types.js";

export const FEDERATED_NETWORK_IDS = [
  "external-runtime-a",
  "ai-horde",
  "peer-runtime",
  "chutes",
  "akashml",
  "gpu_cloud",
  "vast",
  "clore",
] as const;

export type FederatedNetworkId = typeof FEDERATED_NETWORK_IDS[number];
export type FederatedNetworkClass = "community" | "token-api" | "rental";
export type FederatedNetworkActualState =
  | "disabled"
  | "blocked"
  | "discovering"
  | "ready"
  | "degraded"
  | "draining"
  | "circuit-open"
  | "error";
export type FederatedNodeScope = "physical" | "logical" | "aggregate";
export type InferenceRouteKind = "native-replica" | "native-pipeline" | "federated";

export interface FederatedModel {
  canonicalId: string;
  externalId: string;
  displayName: string;
  contextTokens?: number;
  verifiedAt: number | null;
  advertisedOnly: boolean;
  estimatedInputUsdPerMillion?: number;
  estimatedOutputUsdPerMillion?: number;
}

export interface FederatedNetwork {
  id: FederatedNetworkId;
  name: string;
  class: FederatedNetworkClass;
  desiredEnabled: boolean;
  actualState: FederatedNetworkActualState;
  priority: number;
  dailyBudgetUsd: number;
  monthlyBudgetUsd: number;
  configured: boolean;
  experimental: boolean;
  models: FederatedModel[];
  nodeCount: number;
  lastCanaryAt: number | null;
  ttftMs: number | null;
  reliability: number;
  spentTodayUsd: number;
  spentMonthUsd: number;
  consecutivePretokenFailures: number;
  retryAt: number | null;
  lastError: string | null;
}

export interface FederatedNode {
  id: string;
  networkId: FederatedNetworkId;
  scope: FederatedNodeScope;
  label: string;
  models: string[];
  status: "online" | "degraded" | "offline" | "unknown";
  reliability: number;
  routable: boolean;
  individuallySelectable: boolean;
  lastVerifiedAt: number | null;
  capacity?: {
    gpuModel?: string;
    vramMb?: number;
  };
}

export interface FederatedRouteAttempt {
  id: string;
  requestId: string;
  provider: FederatedNetworkId;
  canonicalModel: string;
  externalModel: string;
  routeKind: InferenceRouteKind;
  startedAt: number;
  firstTokenAt: number | null;
  completedAt: number | null;
  inputTokens: number;
  outputTokens: number;
  reservedCostUsd: number;
  actualCostUsd: number;
  result: "running" | "completed" | "failed" | "cancelled";
  fallbackReason: string | null;
  failureCode: string | null;
}

export interface ManagedRental {
  id: string;
  provider: Extract<FederatedNetworkId, "gpu_cloud" | "vast" | "clore">;
  state:
    | "requested"
    | "starting"
    | "awaiting-worker"
    | "verified"
    | "draining"
    | "stopping"
    | "stopped"
    | "failed";
  image: string;
  requestedHardware: {
    gpuModel?: string;
    minimumVramMb: number;
    spot: boolean;
  };
  workerId: string | null;
  reservedCostUsd: number;
  createdAt: number;
  updatedAt: number;
  drainStartedAt: number | null;
  stoppedAt: number | null;
  lastError: string | null;
}

export interface FederationSettings {
  enabled: boolean;
  dailyBudgetUsd: number;
  monthlyBudgetUsd: number;
  autoscalingEnabled: boolean;
  maxRentals: number;
  updatedAt: number;
}

export interface FederatedNetworkSettings {
  id: FederatedNetworkId;
  enabled: boolean;
  priority: number;
  dailyBudgetUsd: number;
  monthlyBudgetUsd: number;
  updatedAt: number;
}

export interface FederationSnapshot {
  enabled: boolean;
  notice: string;
  readyNetworks: number;
  routableNodes: number;
  verifiedModels: number;
  spentTodayUsd: number;
  spentMonthUsd: number;
  externalRequestsAllowed: boolean;
}

export const federationSettingsUpdateSchema = z.object({
  enabled: z.boolean().optional(),
  dailyBudgetUsd: z.number().finite().min(0).max(1_000_000).optional(),
  monthlyBudgetUsd: z.number().finite().min(0).max(10_000_000).optional(),
  autoscalingEnabled: z.boolean().optional(),
  maxRentals: z.number().int().min(0).max(4).optional(),
}).strict();

export const federatedNetworkUpdateSchema = z.object({
  enabled: z.boolean().optional(),
  priority: z.number().int().min(0).max(1_000).optional(),
  dailyBudgetUsd: z.number().finite().min(0).max(1_000_000).optional(),
  monthlyBudgetUsd: z.number().finite().min(0).max(10_000_000).optional(),
}).strict();

export const federationProbeSchema = z.object({
  model: z.string().trim().min(1).max(300).optional(),
}).strict();

export interface FederatedProviderModel {
  canonicalId: string;
  externalId: string;
  displayName?: string;
  contextTokens?: number;
  estimatedInputUsdPerMillion?: number;
  estimatedOutputUsdPerMillion?: number;
}

export interface FederatedProviderNode {
  externalId: string;
  label?: string;
  scope: FederatedNodeScope;
  models: string[];
  status: FederatedNode["status"];
  reliability?: number;
  individuallySelectable?: boolean;
  capacity?: FederatedNode["capacity"];
}

export type FederatedInferenceEvent =
  | { type: "heartbeat"; at: number }
  | { type: "token"; text: string; index: number; at: number }
  | {
      type: "completed";
      text: string;
      inputTokens: number;
      outputTokens: number;
      finishReason: string;
      actualCostUsd?: number;
    };

export interface FederatedInferenceInput {
  requestId: string;
  request: ChatCompletionRequest;
  canonicalModel: string;
  externalModel: string;
  signal: AbortSignal;
}

export interface FederatedProviderAdapter {
  readonly id: FederatedNetworkId;
  readonly class: FederatedNetworkClass;
  readonly configured: boolean;
  discover(signal: AbortSignal): Promise<{
    models: FederatedProviderModel[];
    nodes: FederatedProviderNode[];
  }>;
  infer(input: FederatedInferenceInput): AsyncIterable<FederatedInferenceEvent>;
  estimateMaximumCostUsd(input: {
    request: ChatCompletionRequest;
    model: FederatedProviderModel;
  }): number;
  close?(): Promise<void>;
}

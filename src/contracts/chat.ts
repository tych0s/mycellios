import type { ChatActivity } from "./chat-activity.js";
import type { RequestRecoveryTrace } from "./recovery-trace.js";
import type { RuntimeRecoveryEvidence } from "./runtime-recovery.js";
import type { ChatMessage } from "./types.js";

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  sessionId: string;
  maxTokens?: number;
}

export interface ChatResponse {
  requestId: string;
  model: string;
  text: string;
  promptTokens: number;
  outputTokens: number;
  totalTokens: number;
  routeClass: string;
  affinityHit: boolean;
  sessionId: string;
  reusedKvTokens: number;
  ttftMs: number;
  activeMs: number;
  recoveryTrace?: RequestRecoveryTrace;
  runtimeRecovery?: RuntimeRecoveryEvidence;
  activity?: ChatActivity;
}

export interface ChatStreamUpdate {
  requestId: string;
  model: string;
  delta: string;
  text: string;
  outputTokens: number;
  routeClass: string;
  affinityHit: boolean;
  sessionId: string;
  reusedKvTokens: number;
  ttftMs: number;
  elapsedMs: number;
  phase?: "connecting" | "waiting_first_token" | "recovering" | "streaming";
  statusMessage?: string;
  attempt?: number;
  maximumAttempts?: number;
  affectedWorkerId?: string;
  affectedNodeId?: string;
}

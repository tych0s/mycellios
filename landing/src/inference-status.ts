import type { ChatStreamUpdate } from "../../src/contracts/chat";
import { shortId } from "./panel-ui-utilities";

export function friendlyInferenceError(message: string): string {
  const normalized = message.toLowerCase();
  if (normalized.includes("first token")) return "The route did not return a first token. Check model availability and try again.";
  if (normalized.includes("disconnected from the distributed pipeline")) return "A device in the distributed route disconnected. Check its connection before retrying.";
  if (normalized.includes("no_capacity") || normalized.includes("no_candidates") || normalized.includes("no candidate") || normalized.includes("unavailable")) return "The model currently has no available capacity. Choose another model or try again when a route is ready.";
  if (normalized.includes("timeout") || normalized.includes("deadline")) return "The network took too long to respond. Your message is ready to retry.";
  if (normalized.includes("429") || normalized.includes("rate_limit")) return "Your request limit was reached. Wait a moment before sending again.";
  if (normalized.includes("401") || normalized.includes("invalid_network_token") || normalized.includes("invalid network token") || normalized.includes("administrator token") || normalized.includes("credential")) return "Your session or network credential is no longer valid. Sign in again before retrying.";
  return message;
}

export function inferenceWaitingStatus(update: ChatStreamUpdate): string {
  if (update.phase === "recovering") {
    if (update.statusMessage) return update.statusMessage;
    return update.affectedNodeId
      ? `Node ${shortId(update.affectedNodeId)} disconnected. Reconnecting the route and retrying…`
      : "A stage lost its connection. Reconnecting the route and retrying…";
  }
  if (update.phase === "connecting") return "Looking for an available route…";
  if (update.phase === "waiting_first_token") return "Route ready. Waiting for the first token…";
  return "Waiting for the model's first token…";
}

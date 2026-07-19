import type { ChatCompletionRequest } from "../contracts/types.js";
import { canonicalJson, sha256Text } from "./json.js";

export function inputHashForRequest(request: ChatCompletionRequest): string {
  const canonical = canonicalJson({
    model: request.model,
    messages: request.messages,
    max_tokens: request.max_tokens,
    temperature: request.temperature,
    top_p: request.top_p,
    seed: request.seed,
  });
  return sha256Text(canonical);
}

export function estimateInputTokens(request: ChatCompletionRequest): number {
  const characters = request.messages.reduce(
    (total, message) => total + message.role.length + message.content.length,
    0,
  );
  return Math.max(1, Math.ceil(characters / 4));
}

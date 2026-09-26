import { createHash } from "node:crypto";
import type { FastifyRequest } from "fastify";

export interface SupportAssistantRateState {
  windowStartedAt: number;
  requests: number;
  active: number;
}

export function supportAssistantRateKey(request: FastifyRequest, sessionId: string): string {
  const userAgent = request.headers["user-agent"] ?? "unknown";
  return createHash("sha256")
    .update(`${request.ip}\n${userAgent}\n${sessionId}`)
    .digest("hex")
    .slice(0, 24);
}

export function supportAssistantClientRateKey(request: FastifyRequest): string {
  const userAgent = request.headers["user-agent"] ?? "unknown";
  return createHash("sha256")
    .update(`${request.ip}\n${userAgent}\nassistant-client`)
    .digest("hex")
    .slice(0, 24);
}

export function claimSupportAssistantRequest(
  states: Map<string, SupportAssistantRateState>,
  key: string,
  requestLimit = 24,
  activeLimit = 2,
  now = Date.now(),
): (() => void) | null {
  const windowMs = 10 * 60_000;
  const previous = states.get(key);
  if (!previous && states.size >= 2_000) {
    for (const [candidateKey, candidate] of states) {
      if (now - candidate.windowStartedAt >= windowMs && candidate.active === 0) {
        states.delete(candidateKey);
      }
    }
    if (states.size >= 2_000) return null;
  }
  const state = !previous || now - previous.windowStartedAt >= windowMs
    ? { windowStartedAt: now, requests: 0, active: 0 }
    : previous;
  // Session claims allow two overlapping reconnect attempts. The shared
  // client claim also bounds requests when the caller rotates session IDs.
  if (state.requests >= requestLimit || state.active >= activeLimit) return null;
  state.requests += 1;
  state.active += 1;
  states.set(key, state);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    state.active = Math.max(0, state.active - 1);
  };
}

import type { FastifyInstance, FastifyRequest } from "fastify";
import {
  BillingInboundError,
  StablecoinBillingInbound,
  StripeBillingInbound,
  applyBillingCommand,
  type SignedStablecoinWatcherEvent,
} from "./billing-inbound.js";
import { BillingError, type BillingManager } from "./billing.js";

const STRIPE_BODY_LIMIT_BYTES = 1_000_000;
const INBOUND_WINDOW_MS = 10 * 60_000;
const INBOUND_REQUESTS_PER_WINDOW = 240;

interface InboundRateState {
  startedAt: number;
  requests: number;
}

export interface BillingInboundRoutesOptions {
  manager: BillingManager;
  stripe?: StripeBillingInbound;
  stablecoin?: StablecoinBillingInbound;
  now?: () => number;
}

export async function registerBillingInboundRoutes(
  app: FastifyInstance,
  options: BillingInboundRoutesOptions,
): Promise<void> {
  const rateStates = new Map<string, InboundRateState>();
  const now = options.now ?? Date.now;

  await app.register(async (stripeScope) => {
    // Stripe signs the exact bytes sent. Keep this parser encapsulated so the
    // coordinator's normal application/json parser remains unchanged.
    stripeScope.removeContentTypeParser("application/json");
    stripeScope.addContentTypeParser(
      "application/json",
      { parseAs: "buffer", bodyLimit: STRIPE_BODY_LIMIT_BYTES },
      (_request, body, done) => done(null, body),
    );
    stripeScope.post("/webhooks/v1/stripe", async (request, reply) => {
      if (!claimInboundRequest(rateStates, request, now())) {
        reply.header("retry-after", Math.ceil(INBOUND_WINDOW_MS / 1_000));
        return reply.code(429).send({ error: { code: "billing_inbound_rate_limited" } });
      }
      if (!options.stripe) {
        return reply.code(503).send({ error: { code: "stripe_webhook_not_configured" } });
      }
      const signature = firstHeader(request.headers["stripe-signature"]);
      if (!signature) {
        return reply.code(401).send({ error: { code: "stripe_signature_required" } });
      }
      const rawBody = request.body;
      if (!Buffer.isBuffer(rawBody)) {
        return reply.code(400).send({ error: { code: "stripe_raw_body_required" } });
      }
      try {
        const command = options.stripe.verifyAndNormalize(rawBody, signature);
        if (!command) return { accepted: true, ignored: true };
        const result = applyBillingCommand(options.manager, command);
        return reply.code(result.duplicate ? 200 : 202).send({
          accepted: true,
          duplicate: result.duplicate,
        });
      } catch (error) {
        return sendBillingInboundError(reply, error);
      }
    });
  });

  app.post("/webhooks/v1/stablecoin", async (request, reply) => {
    if (!claimInboundRequest(rateStates, request, now())) {
      reply.header("retry-after", Math.ceil(INBOUND_WINDOW_MS / 1_000));
      return reply.code(429).send({ error: { code: "billing_inbound_rate_limited" } });
    }
    if (!options.stablecoin) {
      return reply.code(503).send({ error: { code: "stablecoin_watcher_not_configured" } });
    }
    if (typeof request.body !== "object" || request.body === null || Array.isArray(request.body)) {
      return reply.code(400).send({ error: { code: "invalid_stablecoin_payload" } });
    }
    try {
      const command = options.stablecoin.verifyAndNormalize(
        request.body as SignedStablecoinWatcherEvent,
      );
      const result = applyBillingCommand(options.manager, command);
      return reply.code(result.duplicate ? 200 : 202).send({
        accepted: true,
        duplicate: result.duplicate,
      });
    } catch (error) {
      return sendBillingInboundError(reply, error);
    }
  });
}

function claimInboundRequest(
  states: Map<string, InboundRateState>,
  request: FastifyRequest,
  now: number,
): boolean {
  const key = request.ip;
  const previous = states.get(key);
  const state = !previous || now - previous.startedAt >= INBOUND_WINDOW_MS
    ? { startedAt: now, requests: 0 }
    : previous;
  if (state.requests >= INBOUND_REQUESTS_PER_WINDOW) return false;
  state.requests += 1;
  states.set(key, state);
  if (states.size > 2_000) {
    for (const [candidate, value] of states) {
      if (now - value.startedAt >= INBOUND_WINDOW_MS) states.delete(candidate);
    }
  }
  return true;
}

function firstHeader(value: string | string[] | undefined): string | undefined {
  const selected = Array.isArray(value) ? value[0] : value;
  return selected?.trim() || undefined;
}

function sendBillingInboundError(reply: Parameters<FastifyInstance["setErrorHandler"]>[0] extends never
  ? never
  : import("fastify").FastifyReply, error: unknown) {
  if (error instanceof BillingInboundError) {
    const unauthorized = error.code.includes("signature")
      || error.code === "untrusted_stablecoin_watcher";
    return reply.code(unauthorized ? 401 : 400).send({
      error: { code: error.code, message: error.message },
    });
  }
  if (error instanceof BillingError) {
    return reply.code(409).send({
      error: { code: error.code, message: error.message },
    });
  }
  throw error;
}

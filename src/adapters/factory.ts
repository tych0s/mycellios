import type { WorkerConfig } from "../contracts/schemas.js";
import type { InferenceAdapter } from "./base.js";
import { MockAdapter } from "./mock.js";
import { local model runtimeAdapter } from "./local-model-runtime.js";
import { OpenAICompatibleAdapter } from "./openai-compatible.js";

export function createAdapter(config: WorkerConfig): InferenceAdapter {
  switch (config.adapter.kind) {
    case "mock":
      return new MockAdapter(config.adapter);
    case "local-model-runtime":
      return new local model runtimeAdapter(config.adapter);
    case "externalggufruntime":
      return new OpenAICompatibleAdapter({ ...config.adapter, kind: "externalggufruntime" });
    case "openai-compatible": {
      const apiKey = config.adapter.apiKeyEnv
        ? process.env[config.adapter.apiKeyEnv]
        : undefined;
      if (config.adapter.apiKeyEnv && !apiKey) {
        throw new Error(
          `Missing API key environment variable: ${config.adapter.apiKeyEnv}`,
        );
      }
      return new OpenAICompatibleAdapter({
        ...config.adapter,
        kind: "openai-compatible",
        ...(apiKey ? { apiKey } : {}),
      });
    }
  }
}

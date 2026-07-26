import type { WorkerConfig } from "../contracts/schemas.js";
import type { InferenceAdapter } from "./base.js";
import { MycelliosNativeControlAdapter } from "./mycellios-native-control.js";
import { MycelliosPipelineAdapter } from "./mycellios-pipeline.js";
import { MockAdapter } from "./mock.js";

export function createAdapter(config: WorkerConfig): InferenceAdapter {
  switch (config.adapter.kind) {
    case "mycellios-native":
      return new MycelliosNativeControlAdapter();
    case "mycellios-pipeline":
      return new MycelliosPipelineAdapter({
        baseUrl: config.adapter.baseUrl,
        model: config.adapter.model,
        modelDigest: config.deployment.modelDigest!,
        activationId: config.deployment.activationId!,
      });
    case "mock":
      if (process.env.NODE_ENV === "production") {
        throw new Error("mock_adapter_is_not_available_in_production");
      }
      return new MockAdapter(config.adapter);
  }
}

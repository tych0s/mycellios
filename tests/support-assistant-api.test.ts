import { afterEach, describe, expect, it } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import {
  buildSupportAssistantMessages,
  DEFAULT_SUPPORT_ASSISTANT_SETTINGS,
  resolveSupportAssistantModel,
} from "../src/support/assistant.js";

describe("network support assistant", () => {
  let runtime: CoordinatorRuntime | null = null;

  afterEach(async () => {
    await runtime?.close();
    runtime = null;
  });

  async function start() {
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
      modelAdminToken: "model-admin-secret",
    });
    return runtime;
  }

  it("exposes only safe public configuration and truthfully reports no network model", async () => {
    const coordinator = await start();

    const response = await coordinator.app.inject({
      method: "GET",
      url: "/public/v1/assistant/config",
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toMatchObject({
      enabled: true,
      available: false,
      provider: "mycellios-network",
      configuredModel: null,
      selectedModel: null,
      availableModels: [],
    });
    expect(response.json()).not.toHaveProperty("systemPrompt");
    expect(response.json()).not.toHaveProperty("temperature");
  });

  it("protects private assistant instructions with administrator authorization", async () => {
    const coordinator = await start();

    const anonymous = await coordinator.app.inject({
      method: "GET",
      url: "/public/v1/admin/assistant",
    });
    const authorized = await coordinator.app.inject({
      method: "GET",
      url: "/public/v1/admin/assistant",
      headers: { authorization: "Bearer model-admin-secret" },
    });

    expect(anonymous.statusCode).toBe(401);
    expect(authorized.statusCode).toBe(200);
    expect(authorized.json().settings.systemPrompt).toContain("official mycellios");
    expect(authorized.json().runtime.provider).toBe("mycellios-network");
  });

  it("persists editable public copy and private behaviour without exposing the prompt", async () => {
    const coordinator = await start();
    const welcomeMessage = "Hola, soy el soporte privado de esta red mycellios.";
    const systemPrompt = "You are the official support agent. Use only verified live mycellios network facts.";

    const saved = await coordinator.app.inject({
      method: "PUT",
      url: "/public/v1/admin/assistant",
      headers: { authorization: "Bearer model-admin-secret" },
      payload: {
        enabled: true,
        modelId: null,
        systemPrompt,
        welcomeMessage,
        suggestions: ["Ayúdame con mi GPU", "¿Qué modelos están conectados?"],
        maxOutputTokens: 768,
        temperature: 0.1,
        allowDeviceControl: false,
      },
    });
    const publicConfig = await coordinator.app.inject({
      method: "GET",
      url: "/public/v1/assistant/config",
    });
    const privateConfig = await coordinator.app.inject({
      method: "GET",
      url: "/public/v1/admin/assistant",
      headers: { authorization: "Bearer model-admin-secret" },
    });

    expect(saved.statusCode).toBe(200);
    expect(publicConfig.json()).toMatchObject({
      welcomeMessage,
      suggestions: ["Ayúdame con mi GPU", "¿Qué modelos están conectados?"],
      allowDeviceControl: false,
    });
    expect(publicConfig.json()).not.toHaveProperty("systemPrompt");
    expect(privateConfig.json().settings).toMatchObject({
      systemPrompt,
      maxOutputTokens: 768,
      temperature: 0.1,
      allowDeviceControl: false,
    });
  });

  it("rejects an unavailable configured model instead of inventing capacity", async () => {
    const coordinator = await start();

    const response = await coordinator.app.inject({
      method: "PUT",
      url: "/public/v1/admin/assistant",
      headers: { authorization: "Bearer model-admin-secret" },
      payload: {
        enabled: true,
        modelId: "external-or-offline-model",
        systemPrompt: DEFAULT_SUPPORT_ASSISTANT_SETTINGS.systemPrompt,
        welcomeMessage: DEFAULT_SUPPORT_ASSISTANT_SETTINGS.welcomeMessage,
        suggestions: DEFAULT_SUPPORT_ASSISTANT_SETTINGS.suggestions,
        maxOutputTokens: 512,
        temperature: 0.2,
        allowDeviceControl: true,
      },
    });

    expect(response.statusCode).toBe(409);
    expect(response.json().error.code).toBe("assistant_model_not_available");
  });

  it("does not answer when no real model is connected to the network", async () => {
    const coordinator = await start();

    const response = await coordinator.app.inject({
      method: "POST",
      url: "/public/v1/assistant/chat",
      payload: {
        session_id: "browser-session",
        messages: [{ role: "user", content: "¿Qué GPU necesito?" }],
        page: "landing:/",
        platform: "test",
      },
    });

    expect(response.statusCode).toBe(503);
    expect(response.json().error.code).toBe("assistant_model_unavailable");
  });
});

describe("support assistant context", () => {
  it("resolves only an actually available network model", () => {
    expect(resolveSupportAssistantModel(DEFAULT_SUPPORT_ASSISTANT_SETTINGS, ["qwen-network"])).toBe("qwen-network");
    expect(resolveSupportAssistantModel(
      { ...DEFAULT_SUPPORT_ASSISTANT_SETTINGS, modelId: "offline-model" },
      ["qwen-network"],
    )).toBeNull();
  });

  it("adds verified live state and safety rules ahead of the conversation", () => {
    const messages = buildSupportAssistantMessages(
      DEFAULT_SUPPORT_ASSISTANT_SETTINGS,
      {
        registeredNodes: 4,
        connectedNodes: 2,
        offeredMemoryGb: 12,
        availableModels: ["qwen-network"],
        requestedModels: [{ id: "large-model", status: "waiting_capacity" }],
      },
      [{ role: "user", content: "Cambia mi GPU a máxima potencia" }],
      { page: "panel:/network", platform: "Windows" },
    );

    expect(messages[0]).toMatchObject({ role: "system" });
    expect(messages[0]?.content).toContain("Connected nodes: 2");
    expect(messages[0]?.content).toContain("Available inference models: qwen-network");
    expect(messages[0]?.content).toContain("cannot silently change");
    expect(messages.at(-1)).toEqual({ role: "user", content: "Cambia mi GPU a máxima potencia" });
  });
});

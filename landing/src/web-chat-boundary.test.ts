import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("web chat boundary", () => {
  it("routes panel and support chat only through coordinator HTTP/SSE", () => {
    const panel = readFileSync(new URL("./Panel.tsx", import.meta.url), "utf8");
    const assistant = readFileSync(new URL("./SupportAssistant.tsx", import.meta.url), "utf8");
    expect(panel).not.toContain("desktopBridge");
    expect(panel).not.toContain("DesktopBridge");
    expect(panel).not.toContain("DashboardSnapshot");
    expect(panel).not.toContain("const desktop = false");
    expect(panel).not.toMatch(/view === ["'](?:machine|settings)["']/);
    expect(panel).not.toMatch(/desktopBridge\??\.(?:streamChat|sendChat)/);
    expect(panel).not.toMatch(/desktopBridge\??\.(?:getSupportAssistant|getFleetContribution|saveSupportAssistant|setFleetContribution)/);
    expect(assistant).not.toContain("DesktopBridge");
    expect(assistant).not.toContain("DesktopSettings");
    expect(assistant).not.toContain('navigate("machine")');
    expect(`${panel}\n${assistant}`).not.toContain("window.desktopAPI");
    expect(panel).toContain('fetch(apiRequestUrl("/v1/chat/completions", authConfig.publicApiBaseUrl)');
    expect(assistant).toContain("/public/v1/assistant/chat");
  });
});

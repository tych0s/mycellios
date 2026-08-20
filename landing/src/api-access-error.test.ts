import { describe, expect, it } from "vitest";
import { apiAccessErrorPresentation, apiAccessSessionNeedsRefresh } from "./ApiAccessPanel";

describe("apiAccessErrorPresentation", () => {
  it("explains invalid network tokens and offers re-authentication", () => {
    expect(apiAccessErrorPresentation("invalid_network_token")).toMatchObject({
      title: "La sesión de API ha caducado",
      action: "Iniciar sesión",
      requiresAuth: true,
    });
  });
  it.each([
    "invalid_network_token",
    "Invalid network token",
    "invalid_access_token",
    "Account session is invalid",
    "Session is invalid or expired",
  ])("automatically refreshes rejected authenticated sessions: %s", (message) => {
    expect(apiAccessSessionNeedsRefresh(new Error(message))).toBe(true);
  });
  it("does not refresh ordinary API failures", () => {
    expect(apiAccessSessionNeedsRefresh(new Error("HTTP 503"))).toBe(false);
  });
  it("keeps ordinary API errors actionable", () => {
    expect(apiAccessErrorPresentation("HTTP 503")).toMatchObject({
      title: "No se pudo comprobar la API",
      detail: "HTTP 503",
      action: "Reintentar",
      requiresAuth: false,
    });
  });
});

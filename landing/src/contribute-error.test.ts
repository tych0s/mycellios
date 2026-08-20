import { describe, expect, it } from "vitest";
import { contributionErrorPresentation } from "./Contribute";

describe("contributionErrorPresentation", () => {
  it("explains an expired network token and routes the user back to auth", () => {
    expect(contributionErrorPresentation("invalid_network_token")).toEqual({
      title: "La sesión de contribución ha caducado",
      detail: "La credencial de red de este navegador ya no es válida. Vuelve a iniciar sesión para reconectar tu contribución.",
      action: "Iniciar sesión",
      requiresAuth: true,
    });
  });

  it("keeps unknown failures useful without exposing a detached technical banner", () => {
    expect(contributionErrorPresentation("Coordinator unavailable")).toMatchObject({
      title: "No se pudo actualizar la contribución",
      detail: "Coordinator unavailable",
      action: "Reintentar",
      requiresAuth: false,
    });
  });
});

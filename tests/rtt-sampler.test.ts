import { describe, expect, it } from "vitest";
import { UNMEASURED_RTT_MS } from "../src/core/rtt.js";

/**
 * Política de aristas sin medir en `auto-distribute`.
 *
 * RECONCILIACIÓN (25-jul-2026): `main` ya resolvió el cero del RTT en la sonda
 * y en `estimateLinkLatencyMs`, con `UNMEASURED_LINK_LATENCY_MS = 65` — la
 * mediana medida de Exp15 — como prior informado. Lo que `main` NO tocó es el
 * relleno por defecto de `auto-distribute`, que seguía en `sameRegion ? 1 : 35`.
 *
 * Este fichero fija esa mitad, alineada a la misma constante: dos convenciones
 * distintas de "sin medir" en el mismo repositorio se contradicen en silencio.
 */
describe("prior de arista sin medir", () => {
  it("no premia al enlace desconocido, que era el defecto", () => {
    // El relleno viejo daba 1 ms a dos nodos que compartían etiqueta de región.
    // La etiqueta no es una medición: dos nodos `us` midieron 65 y 132 ms.
    expect(UNMEASURED_RTT_MS).toBeGreaterThan(35);
  });

  it("es un prior plausible, no un castigo que excluya al nodo", () => {
    // Un centinela enorme haría que un plan sin medidas fuese incomparable y
    // que ningún nodo nuevo pudiese entrar jamás. 65 ms es un enlace típico.
    expect(UNMEASURED_RTT_MS).toBeLessThan(500);
  });

  it("coincide con el prior que ya usa el coordinador", () => {
    // `UNMEASURED_LINK_LATENCY_MS` en `connected-executor-activation.ts`.
    // Si alguien mueve uno sin el otro, los dos caminos de planificación
    // empiezan a discrepar sobre qué vale un enlace desconocido.
    expect(UNMEASURED_RTT_MS).toBe(65);
  });

  it("es finito y positivo", () => {
    expect(Number.isFinite(UNMEASURED_RTT_MS)).toBe(true);
    expect(UNMEASURED_RTT_MS).toBeGreaterThan(0);
  });
});

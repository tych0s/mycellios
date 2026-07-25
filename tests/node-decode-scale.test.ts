import { describe, expect, it } from "vitest";
import { deriveDecodeScales } from "../src/distribution/node-scale.js";

/**
 * El SEGUNDO cero del planificador.
 *
 * `coordinatorRttMs` estaba a 0 y ya se mide. Pero `decodeScale` seguía fijado
 * a 1 en las dos rutas que construyen perfiles de nodo reales, así que
 * `ProportionalComputePlanner` dividía por un vector de unos y **el reparto
 * proporcional degeneraba a reparto igual**: el planificador no podía
 * distinguir una 4090 de una 1050 Ti.
 *
 * Y tiene una consecuencia retroactiva: es la razón estructural por la que
 * Exp7 no pudo ver ningún efecto al desequilibrar capas. No es que el reparto
 * desigual no sirva — es que el sistema no tenía con qué desequilibrar.
 */
describe("deriveDecodeScales", () => {
  it("normaliza al nodo más rápido, que queda en 1", () => {
    const { scales, referenceTokensPerSecond } = deriveDecodeScales([
      { nodeId: "rapido", measuredTokensPerSecond: 40 },
      { nodeId: "lento", measuredTokensPerSecond: 10 },
    ]);
    expect(referenceTokensPerSecond).toBe(40);
    expect(scales.find((s) => s.nodeId === "rapido")!.decodeScale).toBe(1);
    // decodeScale es un multiplicador de COSTE: cuatro veces más lento = 4.
    expect(scales.find((s) => s.nodeId === "lento")!.decodeScale).toBe(4);
  });

  it("un nodo sin medir conserva 1 pero se declara NO medido", () => {
    // Lo importante no es el 1: es que el llamante pueda saber que no lo sabe,
    // en vez de que una suposición pase por informada.
    const { scales, fullyMeasured } = deriveDecodeScales([
      { nodeId: "medido", measuredTokensPerSecond: 20 },
      { nodeId: "ciego", measuredTokensPerSecond: null },
    ]);
    expect(fullyMeasured).toBe(false);
    const ciego = scales.find((s) => s.nodeId === "ciego")!;
    expect(ciego.decodeScale).toBe(1);
    expect(ciego.measured).toBe(false);
    expect(scales.find((s) => s.nodeId === "medido")!.measured).toBe(true);
  });

  it("sin ninguna medida no inventa nada y lo dice", () => {
    const { scales, fullyMeasured, referenceTokensPerSecond } = deriveDecodeScales([
      { nodeId: "a", measuredTokensPerSecond: null },
      { nodeId: "b", measuredTokensPerSecond: null },
    ]);
    expect(fullyMeasured).toBe(false);
    expect(referenceTokensPerSecond).toBeNull();
    expect(scales.every((s) => s.decodeScale === 1 && !s.measured)).toBe(true);
  });

  it("acota el nodo patológico por los dos lados", () => {
    // Sin cota, un nodo mil veces más lento se lleva una rebanada de una capa
    // (o ninguna), y una cadena con un hueco es peor que una cadena desigual.
    const { scales } = deriveDecodeScales([
      { nodeId: "rapido", measuredTokensPerSecond: 1000 },
      { nodeId: "penoso", measuredTokensPerSecond: 0.01 },
    ]);
    const penoso = scales.find((s) => s.nodeId === "penoso")!;
    expect(penoso.decodeScale).toBeLessThanOrEqual(20);
    expect(penoso.decodeScale).toBeGreaterThanOrEqual(0.1);
  });

  it("marca fullyMeasured solo cuando TODOS traen medida", () => {
    expect(deriveDecodeScales([
      { nodeId: "a", measuredTokensPerSecond: 10 },
      { nodeId: "b", measuredTokensPerSecond: 20 },
    ]).fullyMeasured).toBe(true);
  });

  it("una flota homogénea da escalas iguales — y ahí el reparto igual SÍ es el correcto", () => {
    // Es exactamente el montaje de Exp7: dos nodos idénticos. Con escalas
    // iguales, repartir por igual es óptimo y desequilibrar sólo puede perder.
    // El fallo de Exp7 no fue el resultado, fue extrapolarlo a flotas mixtas.
    const { scales } = deriveDecodeScales([
      { nodeId: "a", measuredTokensPerSecond: 30 },
      { nodeId: "b", measuredTokensPerSecond: 30 },
    ]);
    expect(scales.map((s) => s.decodeScale)).toEqual([1, 1]);
  });

  it("ignora medidas absurdas en vez de propagarlas", () => {
    const { scales } = deriveDecodeScales([
      { nodeId: "bueno", measuredTokensPerSecond: 25 },
      { nodeId: "cero", measuredTokensPerSecond: 0 },
      { nodeId: "nan", measuredTokensPerSecond: Number.NaN },
    ]);
    expect(scales.find((s) => s.nodeId === "cero")!.measured).toBe(false);
    expect(scales.find((s) => s.nodeId === "nan")!.measured).toBe(false);
    expect(scales.find((s) => s.nodeId === "bueno")!.decodeScale).toBe(1);
  });

  it("no rompe con la lista vacía", () => {
    const { scales, fullyMeasured } = deriveDecodeScales([]);
    expect(scales).toEqual([]);
    expect(fullyMeasured).toBe(false);
  });
});

import { describe, expect, it } from "vitest";
import { UNMEASURED_RTT_MS, derivedOneWayLatencyMs } from "../src/core/rtt.js";

/**
 * Política de "sin medir" para el RTT.
 *
 * RECONCILIACIÓN (25-jul-2026): dos sesiones atacaron este cero a la vez y las
 * dos mitades son complementarias, no rivales.
 *
 * - La **mecánica de la sonda** viene de `main`: etiqueta cada ping con
 *   `RTT_PROBE_PAYLOAD` (porque `ws` emite pongs no solicitados por su propio
 *   keepalive y emparejar cualquiera corrompe la estimación), usa
 *   `process.hrtime.bigint()` monotónico, y suaviza con EMA α=0,2.
 * - La **política de ausencia** viene de aquí: mientras no haya muestra se
 *   publica el centinela, no un cero. Un cero es indistinguible de "enlace
 *   perfecto" aguas abajo, y ése era el defecto original.
 *
 * Este fichero fija la segunda mitad, que es la que no está en `main`.
 */
describe("política de RTT sin medir", () => {
  it("el centinela es mucho más caro que cualquier enlace medido plausible", () => {
    // De esta propiedad depende toda la política: medir tiene que ser la única
    // forma de que un nodo entre en un plan. Si alguien bajase el centinela por
    // debajo de un RTT transcontinental real, se invierte en silencio.
    const peorEnlaceRealista = 500;
    expect(UNMEASURED_RTT_MS).toBeGreaterThan(peorEnlaceRealista * 10);
  });

  it("nunca es cero: un cero se lee aguas abajo como enlace perfecto", () => {
    expect(UNMEASURED_RTT_MS).toBeGreaterThan(0);
  });

  it("es finito, para que un plan sin medidas siga siendo comparable", () => {
    // Con `Infinity` la aritmética de comparación de planes se rompe y el
    // planificador no puede ordenar candidatos: preferimos "carísimo" a "NaN".
    expect(Number.isFinite(UNMEASURED_RTT_MS)).toBe(true);
  });
});

describe("derivedOneWayLatencyMs", () => {
  it("preserva la aritmética de la ruta relevada", () => {
    // (RTT_A + RTT_B) / 2 modela A->coordinador->B, que es lo que corremos hoy.
    expect(derivedOneWayLatencyMs(40, 60)).toBe(50);
  });

  it("propaga 'sin medir' si CUALQUIERA de los extremos no ha medido", () => {
    // El caso que importa: un extremo desconocido no puede quedar enmascarado
    // por promediarlo con un extremo rápido.
    expect(derivedOneWayLatencyMs(UNMEASURED_RTT_MS, 1)).toBe(UNMEASURED_RTT_MS);
    expect(derivedOneWayLatencyMs(1, UNMEASURED_RTT_MS)).toBe(UNMEASURED_RTT_MS);
  });

  it("una arista sin medir sale más cara que cualquiera medida", () => {
    const peorMedida = derivedOneWayLatencyMs(500, 500);
    expect(derivedOneWayLatencyMs(UNMEASURED_RTT_MS, 10)).toBeGreaterThan(peorMedida);
  });

  it("nunca devuelve cero ni negativo", () => {
    expect(derivedOneWayLatencyMs(0, 0)).toBeGreaterThan(0);
  });
});

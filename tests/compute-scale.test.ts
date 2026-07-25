import { describe, expect, it } from "vitest";
import {
  MAX_ASSUMED_SCALE_MULTIPLIER,
  NEUTRAL_DECODE_SCALE,
  deriveDecodeScales,
  fleetHasMeasuredThroughput,
} from "../src/core/compute-scale.js";

describe("deriveDecodeScales", () => {
  it("normaliza contra el nodo más rápido, que queda en 1,0", () => {
    const scales = deriveDecodeScales([
      { nodeId: "rapido", tokensPerSecond: 100, source: "measured" },
      { nodeId: "lento", tokensPerSecond: 25, source: "measured" },
    ]);
    expect(scales.get("rapido")!.decodeScale).toBeCloseTo(1);
    // 4x más lento => 4x más tiempo de decode por capa.
    expect(scales.get("lento")!.decodeScale).toBeCloseTo(4);
  });

  it("menor es más rápido, que es la convención del cost-model", () => {
    // `optimisticRemainingCompute` toma el MÍNIMO como el nodo más rápido; si
    // esta convención se invirtiera, el planificador daría las capas al nodo
    // más lento y el error sería invisible en los tests de forma.
    const scales = deriveDecodeScales([
      { nodeId: "a", tokensPerSecond: 200, source: "measured" },
      { nodeId: "b", tokensPerSecond: 50, source: "measured" },
    ]);
    expect(scales.get("a")!.decodeScale).toBeLessThan(
      scales.get("b")!.decodeScale,
    );
  });

  it("throughput NO medido no cuenta como medido", () => {
    // Éste es el defecto que se corrige: un valor por defecto usado como si
    // fuera telemetría. Reparte capas, así que mentir aquí crea un cuello serie.
    const scales = deriveDecodeScales([
      { nodeId: "a", tokensPerSecond: 100, source: "default" },
      { nodeId: "b", tokensPerSecond: 10, source: "estimated" },
    ]);
    expect(fleetHasMeasuredThroughput(scales)).toBe(false);
    for (const entry of scales.values()) {
      expect(entry.decodeScale).toBe(NEUTRAL_DECODE_SCALE);
      expect(entry.source).toBe("unmeasured-fleet");
    }
  });

  it("una flota sin medir se comporta como antes, pero lo declara", () => {
    const scales = deriveDecodeScales([
      { nodeId: "a", tokensPerSecond: 0 },
      { nodeId: "b", tokensPerSecond: 0 },
    ]);
    expect([...scales.values()].map((entry) => entry.decodeScale)).toEqual([1, 1]);
    expect([...scales.values()].every((e) => e.source === "unmeasured-fleet")).toBe(
      true,
    );
  });

  it("un nodo sin medir hereda la peor escala medida, no una absurda", () => {
    // A diferencia del centinela de RTT, aquí NO vale un valor enorme:
    // `decodeScale` reparte capas y un nodo con escala gigante recibiría casi
    // ninguna, lo que puede hacer que el modelo no quepa.
    const scales = deriveDecodeScales([
      { nodeId: "rapido", tokensPerSecond: 100, source: "measured" },
      { nodeId: "lento", tokensPerSecond: 50, source: "measured" },
      { nodeId: "desconocido", tokensPerSecond: 0 },
    ]);
    expect(scales.get("desconocido")!.decodeScale).toBeCloseTo(2);
    expect(scales.get("desconocido")!.source).toBe("assumed-worst");
  });

  it("la penalización al no medido está acotada", () => {
    const scales = deriveDecodeScales([
      { nodeId: "rapido", tokensPerSecond: 1000, source: "measured" },
      { nodeId: "lentisimo", tokensPerSecond: 1, source: "measured" },
      { nodeId: "desconocido", tokensPerSecond: 0 },
    ]);
    // Sin tope serían 1000x y el nodo no recibiría ninguna capa.
    expect(scales.get("desconocido")!.decodeScale).toBe(
      MAX_ASSUMED_SCALE_MULTIPLIER,
    );
  });

  it("nodos idénticos dan escalas idénticas, sin ruido espurio", () => {
    const scales = deriveDecodeScales([
      { nodeId: "a", tokensPerSecond: 42, source: "measured" },
      { nodeId: "b", tokensPerSecond: 42, source: "measured" },
    ]);
    expect(scales.get("a")!.decodeScale).toBe(scales.get("b")!.decodeScale);
  });

  it("nunca devuelve cero, negativo ni infinito", () => {
    const scales = deriveDecodeScales([
      { nodeId: "a", tokensPerSecond: 100, source: "measured" },
      { nodeId: "raro", tokensPerSecond: Number.NaN, source: "measured" },
      { nodeId: "cero", tokensPerSecond: 0, source: "measured" },
    ]);
    for (const entry of scales.values()) {
      expect(Number.isFinite(entry.decodeScale)).toBe(true);
      expect(entry.decodeScale).toBeGreaterThan(0);
    }
  });
});

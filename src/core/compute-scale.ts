/**
 * Derivación de `decodeScale` a partir de throughput medido.
 *
 * EL SEGUNDO CERO
 * ---------------
 * `decodeScale: 1` estaba cableado en `connected-executor-activation.ts` y
 * `desktop/main.ts` — las dos rutas que construyen perfiles de nodo reales.
 * Con un vector de unos:
 *
 * - `planners.ts::proportionalSplit` calcula `speeds = 1/decodeScale` y
 *   **degenera al reparto igual**, así que el `ProportionalComputePlanner` que
 *   ya existe nunca reparte por rendimiento en producción;
 * - `cost-model.ts` minimiza el máximo de un vector constante, que cualquier
 *   asignación satisface.
 *
 * Es el mismo defecto que el `coordinatorRttMs: 0`, doce líneas más arriba en
 * el mismo fichero: el consumidor está escrito y le falta la telemetría.
 *
 * SEMÁNTICA
 * ---------
 * `decodeScale` multiplica el tiempo de decode por capa (`cost-model.ts`:
 * `decodeCompute = layerDecode × decodeScale × batchSize / speedup`).
 * **Menor es más rápido.** `optimisticRemainingCompute` toma el mínimo como
 * "el más rápido", así que la convención es firme.
 *
 * Se normaliza contra el nodo **más rápido de la flota**: el más rápido queda
 * en 1,0 y los demás por encima. Al planificador solo le importan las
 * proporciones —tanto el reparto proporcional como el minimax son invariantes
 * a la escala—, y normalizar así mantiene el 1,0 como suelo, que es el valor
 * que el modelo asume al perfilar `decodeMsAtUnit`.
 *
 * QUÉ CUENTA COMO MEDIDO
 * ----------------------
 * Solo `throughputSource === "measured"`. Un valor `"default"` o `"estimated"`
 * usado como si fuera medido es inventar datos — exactamente el defecto que
 * se acaba de corregir en el lado del RTT, y aquí sería peor: el RTT mal
 * puesto elige mal el nodo, pero un `decodeScale` mal puesto reparte mal las
 * **capas**, y una etapa infradotada es un cuello serie en cada token.
 */

/** Cómo se obtuvo el `decodeScale` de un nodo. Viaja con el dato. */
export type ComputeScaleSource = "measured" | "assumed-worst" | "unmeasured-fleet";

export interface NodeThroughputSample {
  nodeId: string;
  tokensPerSecond: number;
  /** El discriminador del documento de capacidades. */
  source?: "measured" | "estimated" | "configured" | "default" | undefined;
}

export interface ComputeScaleResult {
  decodeScale: number;
  source: ComputeScaleSource;
}

/** Escala neutra: la que había cableada. Se conserva como valor de "no sé". */
export const NEUTRAL_DECODE_SCALE = 1;

/**
 * Tope de la penalización a un nodo sin medir.
 *
 * A diferencia del centinela de RTT (10 s, deliberadamente absurdo para que el
 * nodo quede fuera de cualquier plan), aquí NO se puede usar un valor enorme:
 * `decodeScale` reparte capas, y un nodo con escala gigante recibiría casi
 * ninguna, lo que puede hacer que el modelo **no quepa**. Un nodo sin medir se
 * trata como "no mejor que el peor que sí medimos", que es conservador y
 * acotado.
 */
export const MAX_ASSUMED_SCALE_MULTIPLIER = 4;

function isMeasured(sample: NodeThroughputSample): boolean {
  return (
    sample.source === "measured" &&
    Number.isFinite(sample.tokensPerSecond) &&
    sample.tokensPerSecond > 0
  );
}

/**
 * Deriva `decodeScale` por nodo.
 *
 * Degrada en tres escalones, y cada uno se etiqueta en el resultado para que
 * quien lea un plan sepa sobre qué datos se tomó:
 *
 * 1. **Ningún nodo medido** → todos a 1,0 (`unmeasured-fleet`). Es el
 *    comportamiento de hoy, pero declarado como lo que es en vez de disfrazado
 *    de dato.
 * 2. **Nodo medido** → `fastest / this` (`measured`).
 * 3. **Nodo sin medir con flota parcialmente medida** → la peor escala medida
 *    (`assumed-worst`), acotada por `MAX_ASSUMED_SCALE_MULTIPLIER`.
 */
export function deriveDecodeScales(
  samples: NodeThroughputSample[],
): Map<string, ComputeScaleResult> {
  const result = new Map<string, ComputeScaleResult>();
  const measured = samples.filter(isMeasured);

  if (measured.length === 0) {
    for (const sample of samples) {
      result.set(sample.nodeId, {
        decodeScale: NEUTRAL_DECODE_SCALE,
        source: "unmeasured-fleet",
      });
    }
    return result;
  }

  const fastest = Math.max(...measured.map((sample) => sample.tokensPerSecond));
  const scaleFor = (tokensPerSecond: number): number =>
    fastest / Math.max(tokensPerSecond, 1e-9);

  const worstMeasured = Math.min(
    MAX_ASSUMED_SCALE_MULTIPLIER,
    Math.max(...measured.map((sample) => scaleFor(sample.tokensPerSecond))),
  );

  for (const sample of samples) {
    if (isMeasured(sample)) {
      result.set(sample.nodeId, {
        decodeScale: scaleFor(sample.tokensPerSecond),
        source: "measured",
      });
    } else {
      result.set(sample.nodeId, {
        decodeScale: worstMeasured,
        source: "assumed-worst",
      });
    }
  }
  return result;
}

/** `true` si el plan se tomó sobre rendimiento realmente medido. */
export function fleetHasMeasuredThroughput(
  scales: Map<string, ComputeScaleResult>,
): boolean {
  return [...scales.values()].some((entry) => entry.source === "measured");
}

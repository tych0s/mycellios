/**
 * Medición de RTT y política de "sin medir".
 *
 * Contexto (auditoría 2026-07-25): el planificador de colocación estaba
 * alimentado con `coordinatorRttMs: 0` constante — nadie escribía nunca ese
 * campo. Peor que el cero era el relleno por defecto de aristas sin medir
 * (`sameRegion ? 1 : 35`), que PREMIABA justo al enlace del que no sabemos
 * nada. En una tubería serie donde el TPOT lo domina `(S-1)·2·RTT`, una arista
 * desconocida colocada en la cadena por optimismo cuesta el token entero.
 *
 * La política aquí es la contraria y es deliberada: **lo que no se ha medido
 * se cobra caro**. Un enlace sin muestras vale `UNMEASURED_RTT_MS`, de modo que
 * el planificador lo evita mientras exista cualquier alternativa medida, y solo
 * lo usa cuando no queda otra. Medir es entonces la única forma de que un nodo
 * entre en un plan, que es exactamente el incentivo que queremos.
 */

/**
 * Coste asignado a un enlace del que no tenemos ninguna muestra.
 *
 * 10 s no pretende estimar nada: es un centinela lo bastante grande como para
 * que cualquier arista medida gane siempre, y lo bastante finito como para que
 * un plan compuesto solo por enlaces desconocidos siga siendo representable en
 * vez de convertirse en `Infinity` y romper la aritmética de comparación.
 */
export const UNMEASURED_RTT_MS = 10_000;

/**
 * Coste one-way de una arista derivado de dos RTT extremo-a-coordinador.
 *
 * Modela la ruta **relevada** A→coordinador→B, que es la que corremos hoy:
 * `RTT_A/2 + RTT_B/2 = (RTT_A + RTT_B)/2`. Es deliberadamente la misma
 * aritmética que ya usaban `connected-executor-activation.ts` y
 * `desktop/main.ts`; aquí solo se centraliza y se le añade la propagación de
 * "sin medir". Bajar este coste para modelar un camino directo A→B sería una
 * decisión distinta y necesita la matriz nodo↔nodo real: sin ella, dividir más
 * solo produce planes que se ven mejor sobre el papel.
 *
 * La degradación es la parte importante: si CUALQUIERA de los dos extremos está
 * sin medir, la arista entera queda sin medir. Un extremo desconocido no puede
 * quedar enmascarado por promediarlo con un extremo rápido.
 */
export function derivedOneWayLatencyMs(
  fromRttMs: number,
  toRttMs: number,
): number {
  if (fromRttMs >= UNMEASURED_RTT_MS || toRttMs >= UNMEASURED_RTT_MS) {
    return UNMEASURED_RTT_MS;
  }
  return Math.max(0.1, (fromRttMs + toRttMs) / 2);
}

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
 * Coste one-way asignado a una arista de la que no tenemos ninguna muestra.
 *
 * 65 ms es la MEDIANA MEDIDA de Exp15, no un castigo: un enlace desconocido se
 * trata como un enlace típico de esta flota. Es la misma constante y el mismo
 * criterio que `estimateLinkLatencyMs` en `coordinator/connected-executor-activation.ts`,
 * y mantenerlos alineados importa — dos convenciones distintas de "sin medir"
 * en el mismo repositorio se contradicen en silencio.
 *
 * Lo que sí se corrige aquí es el relleno ANTIGUO de `auto-distribute`
 * (`sameRegion ? 1 : 35`), que premiaba al enlace del que no sabemos nada: la
 * etiqueta de región no es una medición, y dos nodos `us` midieron 65 y 132 ms.
 */
export const UNMEASURED_RTT_MS = 65;

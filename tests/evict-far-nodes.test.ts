import { describe, expect, it } from "vitest";
import { evictFarNodes } from "../src/distribution/planners.js";
import type { ComputeNodeProfile, DistributionTopology } from "../src/distribution/types.js";

/**
 * Exp15 measured 54 ms on the best node and 437 ms on the worst, and showed that
 * ONE far node drags a four-hop chain from ~3.1 tok/s to ~0.55. Since the cost
 * model sums latency along the route, an outlier taxes every token that crosses
 * it — so refusing to route through it is the cheapest win available.
 */
function node(id: string): ComputeNodeProfile {
  return {
    id,
    memoryMiB: 4_096,
    reserveMiB: 256,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    powerWatts: 60,
    availability: 0.99,
  } as ComputeNodeProfile;
}

function mesh(latencies: Record<string, number>): DistributionTopology {
  const ids = Object.keys(latencies);
  const links = [];
  for (const from of ids) {
    for (const to of ids) {
      if (from === to) continue;
      links.push({
        from,
        to,
        // Un enlace cuesta lo peor de sus dos extremos: un nodo lejano
        // contamina todas sus aristas, que es como se comporta la realidad.
        oneWayLatencyMs: Math.max(latencies[from]!, latencies[to]!),
        jitterP95Ms: 0,
        bandwidthMbps: 100,
        lossRate: 0,
      });
    }
  }
  return { nodes: ids.map(node), links };
}

describe("evictFarNodes", () => {
  it("drops the one node that would tax every route through it", () => {
    // Cuatro nodos sanos a ~65 ms y uno en Taiwán a 437, como en Exp15.
    const { topology, evicted } = evictFarNodes(
      mesh({ a: 65, b: 60, c: 70, d: 55, taiwan: 437 }),
    );
    expect(evicted).toEqual(["taiwan"]);
    expect(topology.nodes.map((n) => n.id)).toEqual(["a", "b", "c", "d"]);
    // Y sus aristas se van con él: dejarlas sería invitar al planificador a usarlas.
    expect(topology.links.some((l) => l.from === "taiwan" || l.to === "taiwan")).toBe(false);
  });

  it("keeps a merely mediocre node, because the rule is for outliers", () => {
    const { evicted } = evictFarNodes(mesh({ a: 65, b: 60, c: 70, slow: 120 }));
    expect(evicted).toEqual([]);
  });

  it("is relative, so a fast LAN cell is not judged by WAN standards", () => {
    // Todos rápidos: nadie sobra aunque uno sea 3x el resto en términos absolutos
    // pequeños... salvo que de verdad sea un outlier relativo.
    const { evicted } = evictFarNodes(mesh({ a: 1, b: 1.2, c: 1.1, d: 1.3 }));
    expect(evicted).toEqual([]);
  });

  it("evicts the relative outlier even when every number is small", () => {
    const { evicted } = evictFarNodes(mesh({ a: 1, b: 1.2, c: 1.1, odd: 40 }));
    expect(evicted).toEqual(["odd"]);
  });

  it("refuses to empty the fleet: if it would ban everyone, the problem is us", () => {
    // Un solo nodo sano y tres lejanos: la medida es sospechosa, no el enjambre.
    const { topology, evicted } = evictFarNodes(mesh({ ok: 20, x: 400, y: 420, z: 440 }));
    expect(evicted).toEqual([]);
    expect(topology.nodes).toHaveLength(4);
  });

  it("never drops below the minimum node count", () => {
    const { evicted } = evictFarNodes(mesh({ a: 60, b: 65, far: 900 }), { minimumNodes: 3 });
    expect(evicted).toEqual([]);
  });

  it("leaves an unmeasured node alone, because unmeasured is not slow", () => {
    // `ghost` no aparece en ningún enlace: la regla no puede opinar sobre él, y
    // penalizarlo encogería el enjambre según se retrasa la instrumentación.
    const base = mesh({ a: 65, b: 60, c: 70, taiwan: 437 });
    const topology: DistributionTopology = {
      nodes: [...base.nodes, node("ghost")],
      links: base.links,
    };
    const { topology: pruned, evicted } = evictFarNodes(topology);
    expect(evicted).toEqual(["taiwan"]);
    expect(pruned.nodes.map((n) => n.id)).toContain("ghost");
  });

  it("does nothing without enough measured nodes to define a fleet median", () => {
    const { evicted } = evictFarNodes(mesh({ a: 10, b: 900 }));
    expect(evicted).toEqual([]);
  });

  it("ignores zero and non-finite latencies instead of treating them as fast", () => {
    const topology = mesh({ a: 65, b: 60, c: 70, taiwan: 437 });
    topology.links.push({
      from: "a", to: "c", oneWayLatencyMs: 0,
      jitterP95Ms: 0, bandwidthMbps: 100, lossRate: 0,
    });
    const { evicted } = evictFarNodes(topology);
    expect(evicted).toEqual(["taiwan"]);
  });
});

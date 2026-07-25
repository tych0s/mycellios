import { describe, expect, it } from "vitest";
import {
  AUTO_DISTRIBUTE_SCHEMA,
  MODEL_PROFILE_SCHEMA,
  compileAutoDistribution,
  parseAutoDistributionConfig,
  type AutoDistributionConfig,
  type CompiledModelProfile,
} from "../src/distribution/auto-distribute.js";
import type { DistributedModelProfile } from "../src/distribution/types.js";
import { UNMEASURED_RTT_MS } from "../src/core/rtt.js";

const MIB = 1024 * 1024;
const SNAPSHOT_HEX = "0000000000003039";
const ARTIFACT_IDENTITY = `sha256:${SNAPSHOT_HEX}${"a".repeat(48)}`;

/**
 * PRUEBA DE ACTIVACIÓN del centinela de aristas sin medir.
 *
 * Que la suite entera siga verde no demuestra que este cambio haga nada — la
 * mayoría de los tests pasan enlaces explícitos y nunca tocan la rama del
 * relleno por defecto. Este fichero construye a propósito el único escenario
 * donde el comportamiento viejo y el nuevo eligen nodos DISTINTOS, para que el
 * cambio quede probado por su efecto y no por su presencia.
 *
 * Escenario:
 *  - `node-root` y `node-unknown` comparten región, y NO hay medida entre ellos.
 *  - `node-measured` está en otra región, con un enlace MEDIDO de 40 ms.
 *
 * Comportamiento viejo (`sameRegion ? 1 : 35`): `node-unknown` recibía 1 ms por
 * el mero hecho de compartir etiqueta de región y ganaba al medido de 40 ms.
 * Ese es exactamente el defecto: la etiqueta de región no es una medición, y
 * Exp15 midió 65 y 132 ms entre dos nodos ambos etiquetados `us`.
 *
 * Comportamiento nuevo: sin muestra se cobra el centinela, así que gana el
 * enlace medido aunque sea nominalmente más lento.
 */
describe("centinela de aristas sin medir", () => {
  it("prefiere el nodo MEDIDO a 40 ms antes que el no medido de la misma región", () => {
    const compiled = compileAutoDistribution(
      parseAutoDistributionConfig(configFixture()),
      profileFixture(),
    );

    const elegidos = compiled.manifest.plans.decode.stages.map(
      (stage) => stage.anchor.memberId,
    );
    // El enlace medido está declarado y el otro no: si esta precondición se
    // rompe, el test dejaría de probar lo que dice probar.
    const coste = new Map(
      compiled.request.topology.links.map((link) => [
        `${link.from}->${link.to}`,
        link.oneWayLatencyMs,
      ]),
    );
    expect(coste.get("node-root->node-measured")).toBe(40);
    expect(coste.get("node-root->node-unknown")).toBe(UNMEASURED_RTT_MS);

    expect(elegidos).toContain("node-measured");
    // La aserción que hace fallar al código viejo: bajo el relleno optimista,
    // `node-unknown` entraba con 1 ms y desplazaba al medido.
    expect(elegidos).not.toContain("node-unknown");
  });

  it("el centinela es más caro que cualquier enlace medido plausible", () => {
    // Propiedad de la que depende toda la política: medir tiene que ser la
    // única forma de que un nodo entre en un plan. Si alguien bajase el
    // centinela por debajo de un RTT transcontinental real, la política se
    // invierte en silencio y volvemos al defecto original.
    const peorEnlaceRealista = 500;
    expect(UNMEASURED_RTT_MS).toBeGreaterThan(peorEnlaceRealista * 10);
  });
});

function configFixture(): AutoDistributionConfig {
  const node = (
    id: string,
    region: string,
    port: number,
  ): AutoDistributionConfig["nodes"][number] => ({
    id,
    region,
    endpoint: { host: "127.0.0.1", port },
    memoryMiB: 2_048,
    reserveMiB: 256,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    powerWatts: 65,
    availability: 0.999,
    agent: { kind: "local" },
  });

  return {
    schema: AUTO_DISTRIBUTE_SCHEMA,
    model: {
      source: "local-compatible-model",
      revision: null,
      publicName: "sentinel-test-model",
    },
    // Orden de declaración ADVERSO a propósito: el no medido va antes que el
    // medido. Si la colocación vuelve a depender del orden en vez del coste,
    // este test lo detecta.
    nodes: [
      node("node-root", "region-a", 22_001),
      node("node-unknown", "region-a", 22_002),
      node("node-measured", "region-b", 22_003),
    ],
    // Solo se declaran las aristas con `node-measured`. Las que tocan a
    // `node-unknown` quedan deliberadamente sin declarar: ése es el caso bajo
    // prueba.
    links: [
      measuredLink("node-root", "node-measured"),
      measuredLink("node-measured", "node-root"),
    ],
    distribution: {
      minimumStages: 2,
      maximumStages: 2,
      allowLossyActivation: false,
    },
    workload: {
      promptTokens: 32,
      outputTokens: 16,
      contextTokens: 128,
      concurrentSequences: 1,
      minRouteAvailability: 0.9,
      batchWindowMs: 1,
      p95: false,
    },
    runtime: {
      pythonExecutable: "runtime/distribution-venv/Scripts/python.exe",
      pythonPath: "python",
      hfHome: "runtime/hf-cache",
      apiEndpoint: { host: "127.0.0.1", port: 8_088 },
      apiAdvertiseHost: "127.0.0.1",
      returnEndpoint: { host: "127.0.0.1", port: 30_000 },
      returnBindHost: "127.0.0.1",
      threadsPerStage: 1,
      connectTimeoutSeconds: 60,
    },
  } as AutoDistributionConfig;
}

function measuredLink(from: string, to: string) {
  return {
    from,
    to,
    oneWayLatencyMs: 40,
    jitterP95Ms: 2,
    bandwidthMbps: 200,
    lossRate: 0,
    availability: 0.999,
  };
}

function profileFixture(): CompiledModelProfile {
  return {
    schema: MODEL_PROFILE_SCHEMA,
    source: {
      model: "local-compatible-model",
      revision: null,
      snapshotCommit: null,
      snapshotIdentityUint64Hex: SNAPSHOT_HEX,
      artifactIdentity: ARTIFACT_IDENTITY,
      canonicalSource: `content-addressed://${ARTIFACT_IDENTITY}`,
      canonicalRevision: ARTIFACT_IDENTITY,
      format: "safetensors",
    },
    inspection: {
      architecture: "LlamaForCausalLM",
      calibrationRequired: true,
    },
    compatibility: {
      selectiveSafetensors: true,
      requiresAdapter: false,
      adapterId: "transformers-llama-v1",
      reasons: [],
    },
    model: modelProfile(),
  };
}

function modelProfile(): DistributedModelProfile {
  return {
    id: "local-compatible-model",
    layers: Array.from({ length: 8 }, (_, index) => ({
      index,
      weightBytes: 32 * MIB,
      activationElements: 512,
      kvBytesPerToken: 128,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: 16 * MIB,
    lmHeadBytes: 16 * MIB,
    tiedEmbeddingAndHead: false,
    runtimeOverheadBytesPerStage: 256 * MIB,
    embeddingDecodeMsAtUnit: 0,
    lmHeadDecodeMsAtUnit: 0,
    embeddingPrefillMsPerTokenAtUnit: 0,
    lmHeadPrefillMsPerTokenAtUnit: 0,
  };
}

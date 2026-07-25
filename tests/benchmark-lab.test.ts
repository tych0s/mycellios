import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  compareRunWithHistory,
  createRunIdentity,
  loadBenchmarkRuns,
  saveBenchmarkRun,
  type RunIdentity,
} from "../src/benchlab/history.js";
import { importPhysicalCampaign } from "../src/benchlab/physical-import.js";
import {
  buildRealBenchmarkRun,
  type ApiBenchmarkDocument,
} from "../src/benchlab/real-suite.js";

const IDENTITY: RunIdentity = {
  runId: "run-v1",
  version: "1.0.0",
  label: "v1.0.0 test",
  gitCommit: "0123456789abcdef",
  gitBranch: "test",
  gitDirty: false,
  build: {
    release: "1.0.0",
    releaseSource: "override",
    revision: "0123456789abcdef",
    revisionSource: "git",
  },
};

describe("benchmark lab", () => {
  it("turns measured API samples into loopback evidence without inventing devices", () => {
    const run = realRun(IDENTITY);
    const measurement = run.measurements[0]!;
    expect(run.suite).toBe("real-runtime");
    expect(measurement.evidence).toBe("loopback");
    expect(measurement.metrics.tokensPerSecond).toBe(2.5);
    expect(measurement.metrics.ttftMsP95).toBe(1300);
    expect(measurement.inventory.connectedDevices).toBe(1);
    expect(measurement.inventory.profiles.map((profile) => profile.kind)).toEqual(["cpu", "gpu"]);
    expect(measurement.notes.join(" ")).toContain("no es una simulación");
  });

  it("flags throughput regressions against the same scenario and evidence class", () => {
    const baseline = realRun(IDENTITY);
    const next = realRun({ ...IDENTITY, runId: "run-v2", version: "1.1.0" });
    next.startedAt = "2026-07-22T00:00:00.000Z";
    next.finishedAt = "2026-07-22T00:00:01.000Z";
    baseline.startedAt = "2026-07-21T00:00:00.000Z";
    baseline.finishedAt = "2026-07-21T00:00:01.000Z";
    const target = next.measurements.find((measurement) => measurement.metrics.tokensPerSecond !== null)!;
    target.metrics.tokensPerSecond = target.metrics.tokensPerSecond! * 0.8;
    const compared = compareRunWithHistory(next, [baseline]);
    const result = compared.measurements.find((measurement) => measurement.id === target.id)!;
    expect(result.status).toBe("regression");
    expect(result.comparison.tokensPerSecondPct).toBeCloseTo(-20, 1);
    expect(compared.status).toBe("regression");
  });

  it("persists and reloads versioned history", () => {
    const cwd = mkdtempSync(join(tmpdir(), "mycellios-benchlab-"));
    const run = realRun(IDENTITY);
    const path = saveBenchmarkRun(cwd, run);
    expect(JSON.parse(readFileSync(path, "utf8")).schema).toBe("mycellios-benchmark-run/2");
    expect(loadBenchmarkRuns(cwd)).toHaveLength(1);
  });

  it("uses deployment release and revision without a .git directory", () => {
    const cwd = mkdtempSync(join(tmpdir(), "mycellios-release-"));
    writeFileSync(join(cwd, "package.json"), JSON.stringify({ version: "0.2.19" }));
    writeFileSync(join(cwd, "REVISION"), `${"a".repeat(40)}\n`);

    const fromFile = createRunIdentity(cwd, undefined, undefined, {});
    expect(fromFile.version).toBe("0.2.19");
    expect(fromFile.gitCommit).toBe("a".repeat(40));
    expect(fromFile.gitDirty).toBeNull();
    expect(fromFile.build).toEqual({
      release: "0.2.19",
      releaseSource: "package",
      revision: "a".repeat(40),
      revisionSource: "revision-file",
    });

    const fromEnvironment = createRunIdentity(cwd, undefined, undefined, {
      MYCELLIOS_RELEASE: "0.3.0",
      MYCELLIOS_REVISION: "b".repeat(40),
    });
    expect(fromEnvironment.version).toBe("0.3.0");
    expect(fromEnvironment.gitCommit).toBe("b".repeat(40));
    expect(fromEnvironment.build.releaseSource).toBe("environment");
    expect(fromEnvironment.build.revisionSource).toBe("environment");
  });

  it("migrates schema v1 explicitly and blocks comparisons without a model digest", () => {
    const cwd = mkdtempSync(join(tmpdir(), "mycellios-benchlab-v1-"));
    const current = realRun(IDENTITY);
    const legacy = structuredClone(current) as unknown as Record<string, unknown>;
    legacy.schema = "mycellios-benchmark-run/1";
    legacy.gitCommit = null;
    legacy.gitBranch = null;
    legacy.gitDirty = false;
    delete legacy.build;
    for (const measurement of legacy.measurements as Array<Record<string, unknown>>) {
      delete measurement.scenarioFingerprint;
      delete measurement.topology;
      delete (measurement.model as Record<string, unknown>).digest;
    }
    writeFileSync(join(cwd, "legacy.json"), JSON.stringify(legacy));

    const [migrated] = loadBenchmarkRuns(cwd, ".");
    expect(migrated?.schema).toBe("mycellios-benchmark-run/2");
    expect(migrated?.gitDirty).toBeNull();
    expect(migrated?.measurements[0]?.model.digest).toBeNull();
    expect(migrated?.measurements[0]?.scenarioFingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);

    const compared = compareRunWithHistory(current, [migrated!]);
    expect(compared.measurements[0]?.comparison.baselineRunId).toBeNull();
  });

  it("imports real campaign metrics without relabelling them as simulated", () => {
    const observation = {
      schema: "gdlp-physical-gpu-campaign-cli-observation/1",
      capturedAt: "2026-07-21T08:00:00.000Z",
      passed: true,
      launchId: "launch-1",
      pipelineId: "pipeline-1",
      probes: [
        {
          probe: {
            devices: [
              { name: "NVIDIA test GPU", totalMemoryBytes: 4 * 1024 ** 3 },
            ],
          },
        },
      ],
      campaign: {
        passed: true,
        apiHealth: {
          model: "physical-model",
          artifactIdentity: `sha256:${"c".repeat(64)}`,
          canonicalModelSource: "hf://physical/model",
          canonicalModelRevision: "revision-1",
          pipelineSnapshotIdentity: "pipeline-snapshot-1",
          codec: "fp16",
          stages: 2,
          boundaries: [0, 14, 28],
        },
        samples: [
          {
            phase: "measure",
            concurrency: 1,
            passed: true,
            promptTokens: 32,
            completionTokens: 16,
          },
        ],
        summary: {
          byConcurrency: [
            { concurrency: 1, aggregateOutputTokensPerSecondIncludingTtft: 12.5 },
          ],
          serverTtftMs: { p50: 120, p95: 180 },
          serverTpotMs: { p50: 70, p95: 85 },
        },
      },
    };
    const config = {
      schema: "gdlp-physical-gpu-campaign-config/1",
      networkScope: "lan",
      hosts: [{ rankNodeId: "node-a" }, { rankNodeId: "node-b" }],
    };
    const run = importPhysicalCampaign(IDENTITY, observation, config);
    expect(run.suite).toBe("physical-import");
    expect(run.measurements[0]!.evidence).toBe("physical");
    expect(run.measurements[0]!.inventory.profiles[0]!.label).toBe("NVIDIA test GPU");
    expect(run.measurements[0]!.metrics.tokensPerSecond).toBe(12.5);
  });
});

function realRun(identity: RunIdentity) {
  const benchmark: ApiBenchmarkDocument = {
    schema_version: 2,
    kind: "openai_api_continuous_scheduler",
    configuration: {
      base_url: "http://127.0.0.1:8082",
      model: "qwen-real",
      output_tokens: 10,
      warm_batches_per_scenario: 1,
      measured_batches_per_scenario: 2,
    },
    rows: [
      {
        concurrency: 1,
        measured_requests: 2,
        actual_completion_tokens: 20,
        aggregate_actual_tok_s: 2.4,
        per_request_actual_tok_s_mean: 2.5,
        server_token_ttft_mean_ms: 1_100,
        server_token_ttft_p95_ms: 1_300,
        server_token_tpot_mean_ms: 390,
        server_token_tpot_p95_ms: 420,
        nonempty: 2,
        request_samples: [
          {
            prompt_tokens: 20,
            completion_tokens: 10,
            response_ms: 4_000,
            server_token_ttft_ms: 1_000,
            server_token_tpot_ms: 380,
            text_nonempty: true,
          },
          {
            prompt_tokens: 20,
            completion_tokens: 10,
            response_ms: 4_000,
            server_token_ttft_ms: 1_300,
            server_token_tpot_ms: 420,
            text_nonempty: true,
          },
        ],
      },
    ],
    server_after_measurement: {
      status: "ready",
      error: null,
      model: "qwen-real",
      artifact_identity: `sha256:${"a".repeat(64)}`,
      canonical_model_source: "hf://Qwen/Qwen3-0.6B",
      canonical_model_revision: "revision-1",
      pipeline_snapshot_identity: "snapshot-1",
      stages: 2,
      boundaries: [0, 14, 28],
      codec: "fp16",
    },
  };
  return buildRealBenchmarkRun(
    identity,
    benchmark,
    {
      hostname: "test-host",
      platform: "win32",
      ramMb: 32 * 1024,
      gpus: [
        {
          id: "gpu-0",
          vendor: "amd",
          model: "AMD test GPU",
          physicalVramMb: 512,
          sharedMemoryMb: 16 * 1024,
        },
      ],
    },
    null,
    "2026-07-21T00:00:00.000Z",
  );
}

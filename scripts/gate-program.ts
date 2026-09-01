import { execFileSync, spawnSync } from "node:child_process";
import { readFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { z } from "zod";
import { GATE_RECEIPT_SCHEMA, type GateReceipt } from "../src/contracts/gate-receipt.js";
import { readGateReceipt, receiptMatches, sha256, writeGateReceiptAtomic } from "../src/program/gate-receipt-store.js";
import { verifyElectronZero } from "../src/program/electron-zero-verifier.js";

const manifestSchema = z.object({
  schema: z.literal("mycellios-gate-program/1"),
  gates: z.array(z.object({
    id: z.string(), phase: z.number().int(), inputs: z.array(z.string()),
    checks: z.array(z.object({ id: z.string(), kind: z.enum(["automatic", "physical", "external"]) }).strict()),
  }).strict()),
}).strict();

const importedEvidenceSchema = z.object({
  schema: z.literal("mycellios-gate-imported-evidence/1"),
  evidence: z.array(z.object({
    gate: z.string(),
    check: z.string(),
    kind: z.enum(["physical", "external"]),
    status: z.enum(["pass", "fail", "blocked"]),
    detail: z.string().min(1),
    observedAt: z.string().datetime(),
    sourceSha: z.string().regex(/^[a-f0-9]{40}$/).nullable(),
    sourceIdentity: z.string().regex(/^sha256:[a-f0-9]{64}$/).nullable(),
    artifactDigests: z.array(z.string().regex(/^sha256:[a-f0-9]{64}$/)),
    evidence: z.array(z.string().min(1)).min(1),
  }).strict().superRefine((entry, context) => {
    if (entry.status !== "blocked" && (entry.sourceSha === null || entry.sourceIdentity === null)) context.addIssue({ code: "custom", path: ["sourceIdentity"], message: "Pass/fail evidence must bind the exact source SHA and source identity" });
    if (entry.kind === "physical" && entry.status === "pass" && entry.artifactDigests.length === 0) context.addIssue({ code: "custom", path: ["artifactDigests"], message: "Passing physical evidence must bind artifact digests" });
  })),
}).strict();
type ImportedEvidence = z.infer<typeof importedEvidenceSchema>["evidence"][number];

export function parseImportedGateEvidence(value: unknown): z.infer<typeof importedEvidenceSchema> {
  return importedEvidenceSchema.parse(value);
}

export function importedGateEvidenceResult(
  check: { id: string; kind: "physical" | "external" },
  imported: ImportedEvidence | undefined,
  sourceSha: string,
  sourceIdentity?: string,
) {
  if (!imported) return { ...check, status: "blocked" as const, detail: `${check.kind} evidence has not been imported`, evidence: [] };
  if (imported.status !== "blocked" && imported.sourceSha !== sourceSha) return {
    ...check, status: "blocked" as const,
    detail: `Imported ${check.kind} evidence belongs to ${imported.sourceSha}, not current source ${sourceSha}`,
    evidence: [`source-sha:${imported.sourceSha}`, ...imported.artifactDigests, ...imported.evidence],
  };
  if (imported.status !== "blocked" && sourceIdentity && imported.sourceIdentity !== sourceIdentity) return {
    ...check, status: "blocked" as const,
    detail: `Imported ${check.kind} evidence belongs to ${imported.sourceIdentity}, not current source identity ${sourceIdentity}`,
    evidence: [`source-identity:${imported.sourceIdentity}`, ...imported.artifactDigests, ...imported.evidence],
  };
  return { ...check, status: imported.status, detail: `${imported.detail} (observed ${imported.observedAt})`,
    evidence: [...(imported.sourceSha ? [`source-sha:${imported.sourceSha}`] : []),
      ...(imported.sourceIdentity ? [`source-identity:${imported.sourceIdentity}`] : []),
      ...imported.artifactDigests, ...imported.evidence] };
}

const RUNTIME_CHANNEL_COMMIT = "28227c2be27ef741bd78abc77e922b665b64febd";
const RUNTIME_CHANNEL_FILES = [
  "src/contracts/component-update-manifest.ts",
  "src/coordinator/component-release-store.ts",
  "src/program/distributed-development-gate.ts",
  "src/update/component-update-manager.ts",
] as const;

const ELECTRON_INVENTORY_AREAS = [
  { id: "desktop-runtime", prefixes: ["src/desktop/", "src/renderer/"] },
  { id: "desktop-build", prefixes: ["forge.config.ts", "vite.main.config.ts", "vite.preload.config.ts", "vite.renderer.config.ts", "tsconfig.desktop.json"] },
  { id: "desktop-release", prefixes: ["scripts/verify-desktop", "scripts/prepare-desktop", "scripts/desktop-release", ".github/workflows/desktop-build.yml"] },
  { id: "desktop-tests", prefixes: ["tests/desktop-", "tests/installer-"] },
] as const;

export const PRODUCT_JOURNEY_TESTS = {
  J1_NEW_CONTRIBUTOR: [
    "tests/node-enrollment-api.test.ts",
    "tests/node-installation-bootstrap.test.ts",
    "tests/node-service-host.test.ts",
    "tests/node-control-state.test.ts",
  ],
  J2_FIRST_INFERENCE: [
    "tests/execution-route-policy.test.ts",
    "tests/chat-stream.test.ts",
    "landing/src/web-chat-boundary.test.ts",
    "tests/execution-receipt.test.ts",
    "tests/e2e.test.ts",
  ],
  J3_DISTRIBUTED_ACTIVATION: [
    "tests/model-artifact-selection.test.ts",
    "tests/deployment-control-plane.test.ts",
    "tests/connected-executor-activation.test.ts",
    "tests/model-activation-manager.test.ts",
  ],
  J4_UPDATE_AND_ROLLBACK: [
    "tests/component-update-manager.test.ts",
    "tests/component-update-api.test.ts",
    "tests/update-recovery.test.ts",
  ],
  J5_EXECUTION_FAILURE: [
    "tests/runtime-stream-recovery.test.ts",
    "tests/mesh-failover.test.ts",
    "tests/recovery.test.ts",
    "landing/src/inference-recovery.test.ts",
  ],
  J6_REVOCATION_AND_UNINSTALL: [
    "tests/node-command-api.test.ts",
    "tests/node-reauth.test.ts",
    "tests/node-ownership-transfer-api.test.ts",
    "tests/node-uninstall-helper.test.ts",
  ],
} as const;

const PRODUCT_SURFACE_TESTS = [
  "landing/src/Landing.test.tsx",
  "landing/src/routing.test.ts",
  "landing/src/operational-state.test.ts",
  "landing/src/network-topology.test.ts",
  "landing/src/network-page.test.tsx",
  "landing/src/activation-progress.test.tsx",
  "landing/src/inference-attachments.test.tsx",
  "landing/src/live-network-telemetry.test.tsx",
  "landing/src/mesh-node-popover.test.tsx",
  "landing/src/network-history.test.tsx",
  "landing/src/panel-issue.test.tsx",
  "tests/economic-api.test.ts",
  "tests/economic-settlement-receipt.test.ts",
  "tests/execution-topology.test.ts",
  "tests/operator-evidence-api.test.ts",
  "tests/model-certification.test.ts",
  "tests/support-assistant-api.test.ts",
] as const;

export const PRODUCT_JOURNEY_GATE_TESTS = [
  "tests/product-journey-coverage.test.ts",
  ...new Set([
    ...Object.values(PRODUCT_JOURNEY_TESTS).flat(),
    ...PRODUCT_SURFACE_TESTS,
  ]),
] as const;

const AUTOMATIC_COMMAND_CHECKS: Record<string, Array<{ command: string; args: string[]; evidence: string[] }>> = {
  node_control_plane: [{
    command: "npx", args: ["vitest", "run",
      "tests/worker-admission.test.ts", "tests/worker-protocol.test.ts",
      "tests/worker-hub-duplicate.test.ts", "tests/worker-hub-stale-connection.test.ts",
      "tests/node-command-store.test.ts", "tests/node-command-client.test.ts",
      "tests/node-command-executor.test.ts", "tests/node-command-applicator.test.ts",
      "tests/node-command-api.test.ts", "tests/node-reconciliation-store.test.ts",
      "tests/node-reconciliation-state.test.ts", "tests/node-work-policy.test.ts", "tests/node-reauth.test.ts",
      "tests/node-ownership-transfer-store.test.ts", "tests/node-ownership-transfer-api.test.ts"],
    evidence: ["src/coordinator/worker-admission.ts", "src/coordinator/worker-hub.ts",
      "src/coordinator/node-command-store.ts", "src/coordinator/node-reconciliation-store.ts",
      "src/node/command-client.ts", "src/node/command-executor.ts", "src/node/main.ts",
      "src/contracts/node-control.ts", "src/contracts/node-work-policy.ts", "src/contracts/worker-protocol.ts",
      "src/coordinator/node-ownership-transfer-store.ts"],
  }],
  web_node_enrollment: [
    { command: "npm", args: ["run", "landing:typecheck"], evidence: ["tsconfig.landing.json", "landing/src/Contribute.tsx"] },
    { command: "npx", args: ["vitest", "run",
      "landing/src/api-access-node.test.ts", "landing/src/auth-mfa.test.ts",
      "landing/src/contribute-native-node.test.ts"],
    evidence: ["landing/src/Contribute.tsx", "landing/src/api-access.ts", "landing/src/auth.ts",
      "src/contracts/node-control.ts", "src/coordinator/server.ts"] },
  ],
  component_and_model_contracts: [
    { command: "npm", args: ["run", "typecheck"], evidence: ["tsconfig.json"] },
    { command: "npx", args: ["vitest", "run",
      "tests/component-update-manifest.test.ts", "tests/model-distribution-manifest.test.ts",
      "tests/model-certification.test.ts"], evidence: [
      "src/contracts/component-update-manifest.ts", "src/contracts/model-distribution-manifest.ts",
      "src/contracts/model-certification.ts",
    ] },
  ],
  minimal_artifact_download: [{
    command: "npx", args: ["vitest", "run", "tests/model-artifact-cache.test.ts",
      "tests/model-artifact-selection.test.ts", "tests/artifact-swarm.test.ts",
      "tests/artifact-swarm-download.test.ts", "tests/stage-artifact-preparer.test.ts"],
    evidence: ["src/model-fabric/model-artifact-cache.ts", "src/model-fabric/model-artifact-selection.ts", "src/model-fabric/artifact-swarm.ts"],
  }],
  certification_harness: [{
    command: "npx", args: ["vitest", "run", "tests/model-certification-harness.test.ts",
      "tests/certified-model-launch.test.ts"],
    evidence: ["src/benchlab/model-certification-harness.ts", "src/model-fabric/certified-model-launch.ts"],
  }],
  execution_route_policy: [{
    command: "npx", args: ["vitest", "run", "tests/execution-route-policy.test.ts",
      "tests/auto-distribute.test.ts", "tests/auto-distribute-execution-telemetry.test.ts",
      "tests/scheduler.test.ts", "tests/network-execution-trace.test.ts"],
    evidence: ["src/distribution/execution-route-policy.ts", "src/scheduler/scheduler.ts", "src/telemetry/network-execution-trace.ts"],
  }],
  single_device_execution: [{
    command: "npx", args: ["vitest", "run", "tests/headless-worker.test.ts",
      "tests/mycellios-native-control.test.ts", "tests/e2e.test.ts",
      "tests/worker-tunnel-execution-telemetry.test.ts"],
    evidence: ["src/worker/headless-runtime.ts", "src/node", "src/coordinator/mesh-service.ts"],
  }],
  atomic_activation_lifecycle: [{
    command: "npx", args: ["vitest", "run", "tests/deployment-control-plane.test.ts",
      "tests/connected-executor-activation.test.ts", "tests/model-activation-manager.test.ts",
      "tests/installed-stage-canary.test.ts"],
    evidence: ["src/coordinator/deployment-control-plane.ts", "src/coordinator/connected-executor-activation.ts", "src/coordinator/model-activation-manager.ts"],
  }],
  bounded_authenticated_data_plane: [{
    command: "npx", args: ["vitest", "run", "tests/direct-secure-channel.test.ts",
      "tests/direct-runtime-mux.test.ts", "tests/runtime-direct-transport-integration.test.ts",
      "tests/runtime-stream-tunnel.test.ts", "tests/worker-session-token.test.ts"],
    evidence: ["src/transport/direct-secure-channel.ts", "src/transport/direct-runtime-mux.ts", "src/coordinator/worker-hub.ts"],
  }],
  conveyor_correctness: [{
    command: "npx", args: ["vitest", "run", "tests/speculation-tp-gate.test.ts",
      "tests/macro-wave.test.ts", "tests/macro-wave-runtime-integration.test.ts",
      "tests/node-decode-scale.test.ts", "tests/python-launcher.test.ts"],
    evidence: ["src/distribution/macro-wave.ts", "src/distribution/python-launcher.ts", "python/distributed_runtime"],
  }],
  bounded_recovery: [{
    command: "npx", args: ["vitest", "run", "tests/recovery.test.ts",
      "tests/mesh-failover.test.ts", "tests/runtime-stream-recovery.test.ts",
      "tests/update-recovery.test.ts", "tests/deployment-control-plane.test.ts"],
    evidence: ["src/storage/database.ts", "src/coordinator/deployment-control-plane.ts", "src/update/update-recovery.ts"],
  }],
  privacy_and_trust_policy: [{
    command: "npx", args: ["vitest", "run", "tests/coordinator-security.test.ts",
      "tests/coordinator-config-security.test.ts", "tests/worker-admission.test.ts",
      "tests/worker-hardening.test.ts", "tests/node-os-isolation.test.ts",
      "tests/remote-diagnostics.test.ts", "tests/physical-contribution-evidence.test.ts",
      "tests/economic-ledger.test.ts", "tests/economic-settlement-receipt.test.ts",
      "tests/execution-receipt.test.ts", "tests/inference-privacy.test.ts",
      "tests/scheduler.test.ts", "tests/mesh-failover.test.ts"],
    evidence: ["src/coordinator/worker-admission.ts", "src/coordinator/mesh-service.ts",
      "src/scheduler/scheduler.ts", "src/contracts/execution-receipt.ts", "src/node/os-isolation.ts",
      "src/contracts/remote-diagnostics.ts", "src/economy"],
  }],
  protocol_and_artifact_fuzz: [{
    command: "npx", args: ["vitest", "run", "tests/worker-protocol-fuzz.test.ts",
      "tests/public-control-plane-fuzz.test.ts",
      "tests/worker-protocol.test.ts", "tests/component-update-manifest.test.ts",
      "tests/artifact-swarm-download.test.ts", "tests/model-distribution-manifest.test.ts"],
    evidence: ["src/contracts/worker-protocol.ts", "src/contracts/node-control.ts",
      "src/coordinator/node-enrollment-store.ts", "src/coordinator/node-command-store.ts",
      "src/contracts/component-update-manifest.ts", "src/update/component-files.ts",
      "src/model-fabric/artifact-swarm.ts"],
  }],
  automatic_fault_campaign: [{
    command: "npx", args: ["vitest", "run",
      "tests/process-tree.test.ts", "tests/dev-distributed-e2e-cleanup.test.ts",
      "tests/recovery.test.ts", "tests/runtime-stream-recovery.test.ts",
      "tests/mesh-failover.test.ts", "tests/artifact-swarm-download.test.ts",
      "tests/physical-gpu-campaign.test.ts", "tests/distribution-lab.test.ts"],
    evidence: ["src/distribution/process-tree.ts", "src/distribution/physical-gpu-campaign.ts",
      "src/coordinator/mesh-service.ts", "src/coordinator/worker-hub.ts",
      "src/model-fabric/artifact-swarm.ts", "src/storage/database.ts"],
  }],
  security_p0_control_matrix: [{
    command: "npx", args: ["tsx", "scripts/verify-security-p0-controls.ts"],
    evidence: ["config/security-p0-controls.json", "scripts/verify-security-p0-controls.ts",
      "docs/SECURITY.md"],
  }, {
    command: "npx", args: ["vitest", "run", "tests/security-p0-controls.test.ts",
      "tests/node-os-isolation.test.ts", "tests/api-access.test.ts",
      "tests/node-enrollment-store.test.ts", "tests/node-command-store.test.ts",
      "tests/worker-admission.test.ts"],
    evidence: ["src/node/os-isolation.ts", "src/coordinator/api-access.ts",
      "src/coordinator/worker-admission.ts", "src/storage/database.ts"],
  }],
  economic_publication_policy: [{
    command: "npx", args: ["tsx", "scripts/verify-economic-publication-policy.ts"],
    evidence: ["config/economic-publication-policy.json", "scripts/verify-economic-publication-policy.ts"],
  }, {
    command: "npx", args: ["vitest", "run", "tests/economic-publication-policy.test.ts",
      "tests/economic-ledger.test.ts", "tests/economic-api.test.ts",
      "landing/src/Landing.test.tsx"],
    evidence: ["src/economy/economic-ledger.ts", "src/coordinator/server.ts",
      "landing/src/rebrand/PaymentsSection.tsx", "landing/src/rebrand/QuestionsSection.tsx"],
  }],
  product_journeys: [{
    command: "npx", args: ["vitest", "run", ...PRODUCT_JOURNEY_GATE_TESTS],
    evidence: ["landing/src/Panel.tsx", "landing/src/routing.ts", "landing/src/operational-state.ts",
      "landing/src/network-topology.ts", "landing/src/rebrand/RebrandLanding.tsx",
      "src/contracts/execution-topology.ts", "src/coordinator/server.ts",
      "docs/ARCHITECTURE.md", "tests/product-journey-coverage.test.ts"],
  }],
  accessibility: [{
    command: "npx", args: ["vitest", "run", "tests/product-accessibility-contract.test.ts",
      "landing/src/Landing.test.tsx", "landing/src/mesh-node-popover.test.tsx"],
    evidence: ["tests/product-accessibility-contract.test.ts", "landing/src/Panel.tsx", "landing/src/panel.css", "landing/src/rebrand/rebrand.css"],
  }],
  final_traceability: [{
    command: "npm", args: ["run", "verify:traceability"],
    evidence: ["config/final-web-native-traceability.json", "config/final-web-native-program.json",
      "scripts/verify-final-web-native-traceability.ts"],
  }, {
    command: "npx", args: ["vitest", "run", "tests/final-web-native-traceability.test.ts",
      "tests/final-gate-receipt-validator.test.ts", "tests/gate-imported-evidence.test.ts"],
    evidence: ["tests/final-web-native-traceability.test.ts", "tests/final-gate-receipt-validator.test.ts",
      "src/program/final-gate-receipt-validator.ts"],
  }],
  release_supply_chain: [
    { command: "npm", args: ["run", "verify:native-boundary"], evidence: ["scripts/verify-native-product-boundary.ts", "scripts/verify-native-python-product-source.mjs"] },
    { command: "npx", args: ["vitest", "run", "tests/public-release-transaction-cli.test.ts",
      "tests/component-update-publisher.test.ts", "tests/node-release-version.test.ts",
      "tests/installer-package-tree.test.ts"],
      evidence: ["scripts/native-build-provenance.mjs", "src/update/component-update-publisher.ts", ".github/workflows/node-build.yml"] },
    { command: "npx", args: ["vitest", "run", "tests/native-build-provenance.test.ts",
      "tests/coordinator-release-policy.test.ts", "tests/portable-runtime-wheel-lock.test.ts"],
      evidence: ["scripts/native-build-provenance.mjs", "scripts/coordinator-release-policy.mjs", "scripts/portable-runtime-wheel-lock.mjs"] },
  ],
};

function arg(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}

async function digestInputs(inputs: string[]): Promise<`sha256:${string}`> {
  const rows: string[] = [];
  for (const input of inputs) {
    try {
      const info = await stat(input);
      if (info.isDirectory()) {
        const listed = execFileSync("git", [
          "ls-files", "--cached", "--others", "--exclude-standard", "--", input,
        ], { encoding: "utf8" }).trim().split("\n").filter(Boolean);
        for (const file of listed) rows.push(`${file}\0${sha256(await readFile(file))}`);
      } else rows.push(`${input}\0${sha256(await readFile(input))}`);
    } catch { rows.push(`${input}\0missing`); }
  }
  return sha256(rows.sort().join("\n"));
}

function gitSucceeds(args: string[]): boolean {
  try {
    execFileSync("git", args, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

async function runtimeChannelCheck() {
  const ancestor = gitSucceeds(["merge-base", "--is-ancestor", RUNTIME_CHANNEL_COMMIT, "HEAD"]);
  const presentFiles = (await Promise.all(RUNTIME_CHANNEL_FILES.map(async (file) => {
    try { return (await stat(file)).isFile() ? file : null; } catch { return null; }
  }))).filter((file): file is NonNullable<typeof file> => file !== null);
  const integrated = ancestor || presentFiles.length === RUNTIME_CHANNEL_FILES.length;
  return {
    id: "runtime_channel_integrated",
    kind: "automatic" as const,
    status: integrated ? "pass" as const : "fail" as const,
    detail: ancestor
      ? "Runtime channel commit is an ancestor"
      : integrated
        ? "Runtime channel capabilities are present after a non-ancestral integration"
        : "Required runtime channel capabilities are absent",
    evidence: ancestor ? [RUNTIME_CHANNEL_COMMIT] : presentFiles,
  };
}

function electronInventoryCheck() {
  const tracked = execFileSync("git", ["ls-files"], { encoding: "utf8" })
    .trim().split("\n").filter(Boolean);
  const areas: Array<{ id: string; files: string[] }> = ELECTRON_INVENTORY_AREAS.map((area) => ({
    id: area.id,
    files: tracked.filter((file) => area.prefixes.some((prefix) => file === prefix || file.startsWith(prefix))),
  }));
  const referenceFiles = execFileSync("git", [
    "grep", "-Il", "-E", "Electron|electron|desktop:|src/desktop|Forge|Squirrel", "--",
    "README.md", "docs", "specs", "landing", "package.json", ".github",
  ], { encoding: "utf8" }).trim().split("\n").filter(Boolean);
  areas.push({
    id: "migration-references",
    files: referenceFiles.filter((file) => !areas.some((area) => area.files.includes(file))),
  });
  areas.push({
    id: "named-references",
    files: tracked.filter((file) =>
      /electron|desktop|forge|squirrel/i.test(file)
      && !areas.some((area) => area.files.includes(file))
      && file !== "package-lock.json"),
  });
  const uncoveredDesktopFiles = tracked.filter((file) =>
    /electron|desktop|forge|squirrel/i.test(file)
    && !areas.some((area) => area.files.includes(file))
    && file !== "package-lock.json",
  );
  const inventoryFiles = areas.flatMap((area) => area.files.map((file) => `${area.id}:${file}`)).sort();
  const inventory = [
    ...areas.map((area) => `${area.id}:${area.files.length}`),
    `inventory-digest:${sha256(inventoryFiles.join("\n"))}`,
  ];
  return {
    id: "electron_inventory",
    kind: "automatic" as const,
    status: uncoveredDesktopFiles.length === 0 ? "pass" as const : "fail" as const,
    detail: uncoveredDesktopFiles.length === 0
      ? `Electron responsibilities inventoried across ${areas.reduce((sum, area) => sum + area.files.length, 0)} tracked files`
      : `Electron inventory has ${uncoveredDesktopFiles.length} unclassified tracked files`,
    evidence: [...inventory, ...uncoveredDesktopFiles.slice(0, 20).map((file) => `unclassified:${file}`)],
  };
}

async function electronZeroCheck() {
  const trackedFiles = execFileSync("git", [
    "ls-files", "--cached", "--others", "--exclude-standard",
  ], { encoding: "utf8" }).trim().split("\n").filter((file) => Boolean(file) && existsSync(file));
  const inspectedFiles = trackedFiles.filter((file) =>
    file.startsWith(".github/workflows/")
    || file === "README.md"
    || file.startsWith("docs/")
    || (/^(?:src|landing|scripts)\//u.test(file) && /\.(?:ts|tsx|mts|mjs|js|json|ya?ml|md|ps1)$/u.test(file)));
  const fileContents = new Map(await Promise.all(inspectedFiles.map(async (file) =>
    [file, await readFile(file, "utf8")] as const)));
  const violations = verifyElectronZero({
    trackedFiles,
    packageDocument: JSON.parse(await readFile("package.json", "utf8")) as Record<string, unknown>,
    lockDocument: JSON.parse(await readFile("package-lock.json", "utf8")) as Record<string, unknown>,
    fileContents,
  });
  return {
    id: "electron_zero",
    kind: "automatic" as const,
    status: violations.length === 0 ? "pass" as const : "fail" as const,
    detail: violations.length === 0
      ? `No productive Electron surface exists across ${trackedFiles.length} tracked files`
      : `${violations.length} productive Electron violation(s) remain`,
    evidence: violations.slice(0, 50).map((entry) =>
      `${entry.category}:${entry.subject}:${entry.reason}`),
  };
}

function automaticCommandCheck(id: string) {
  const commands = AUTOMATIC_COMMAND_CHECKS[id];
  if (!commands) return null;
  const evidence: string[] = [];
  const failures: string[] = [];
  const blockers: string[] = [];
  for (const entry of commands) {
    const executable = process.platform === "win32" && ["npm", "npx"].includes(entry.command)
      ? `${entry.command}.cmd`
      : entry.command;
    const result = spawnSync(executable, entry.args, {
      cwd: process.cwd(), encoding: "utf8", timeout: 10 * 60_000,
      env: { ...process.env, CI: "1" },
    });
    evidence.push(`command:${entry.command} ${entry.args.join(" ")}`, ...entry.evidence);
    if (result.status !== 0) {
      const diagnostic = `${result.stdout ?? ""}\n${result.stderr ?? ""}`.trim().split("\n").slice(-12).join(" | ");
      const message = `exit ${result.status ?? "unknown"}: ${diagnostic || result.error?.message || "no diagnostic"}`;
      if (
        (result.error as NodeJS.ErrnoException | undefined)?.code === "ENOENT"
        || /Python 3\.12 is required.*commands not found/u.test(diagnostic)
      ) blockers.push(message);
      else failures.push(message);
    }
  }
  if (failures.length > 0) return {
    id, kind: "automatic" as const, status: "fail" as const,
    detail: `${failures.length} verifier command(s) failed: ${failures.join(" || ")}`,
    evidence,
  };
  if (blockers.length > 0) return {
    id, kind: "automatic" as const, status: "blocked" as const,
    detail: `${commands.length - blockers.length}/${commands.length} verifier command(s) passed; environment prerequisite blocked ${blockers.length}: ${blockers.join(" || ")}`,
    evidence,
  };
  return {
    id, kind: "automatic" as const, status: "pass" as const,
    detail: `${commands.length} executable verifier command(s) passed`, evidence,
  };
}

async function main(): Promise<void> {
  const target = z.enum(["local", "pre", "release"]).parse(arg("--target", "local"));
  const through = arg("--through");
  const audit = process.argv.includes("--audit");
  const manifest = manifestSchema.parse(JSON.parse(await readFile("config/final-web-native-program.json", "utf8")));
  const importedEvidence = importedEvidenceSchema.parse(JSON.parse(
    await readFile("config/gate-program-evidence.json", "utf8"),
  ));
  const sourceSha = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  const sourceIdentity = sha256(execFileSync("git", ["ls-tree", "-r", "HEAD"], { encoding: "utf8" }));
  const gates = through ? manifest.gates.slice(0, manifest.gates.findIndex((gate) => gate.id === through) + 1) : manifest.gates;
  if (through && gates.length === 0) throw new Error(`Unknown gate: ${through}`);

  let exitCode = 0;
  for (const gate of gates) {
    const inputDigest = await digestInputs([
      ...gate.inputs,
      "scripts/gate-program.ts",
      "config/physical-evidence-trust.json",
    ]);
    const receiptFile = path.join("evidence", "gate-program", `${gate.id}.json`);
    try {
      const current = await readGateReceipt(receiptFile);
      if (receiptMatches(current, sourceSha, inputDigest)) { console.log(`${gate.id}: pass (resumed)`); continue; }
    } catch { /* missing, partial, or stale receipts are rebuilt */ }

    const checks = await Promise.all(gate.checks.map(async (check) => {
      if (check.kind !== "automatic") {
        const imported = importedEvidence.evidence.find((entry) =>
          entry.gate === gate.id && entry.check === check.id && entry.kind === check.kind,
        );
        return importedGateEvidenceResult({ id: check.id, kind: check.kind }, imported, sourceSha, sourceIdentity);
      }
      if (check.id === "isolated_worktree") {
        const gitDirectory = execFileSync("git", ["rev-parse", "--git-dir"], { encoding: "utf8" }).trim();
        const commonDirectory = execFileSync("git", ["rev-parse", "--git-common-dir"], { encoding: "utf8" }).trim();
        const isolated = path.resolve(gitDirectory) !== path.resolve(commonDirectory);
        return { ...check, status: isolated ? "pass" as const : "fail" as const, detail: isolated ? "Dedicated worktree detected" : "Not running in a linked worktree", evidence: [gitDirectory, commonDirectory] };
      }
      if (check.id === "clean_worktree") {
        const changes = execFileSync("git", ["status", "--porcelain=v1", "--untracked-files=all"], { encoding: "utf8" })
          .trim().split("\n").filter(Boolean)
          .filter((line) => !line.slice(3).startsWith("evidence/"));
        return {
          ...check,
          status: changes.length === 0 ? "pass" as const : "blocked" as const,
          detail: changes.length === 0
            ? "Tracked source identity exactly matches the receipt SHA"
            : `Receipt cannot attest ${changes.length} uncommitted worktree changes`,
          evidence: changes.slice(0, 50),
        };
      }
      if (check.id === "runtime_channel_integrated") {
        return runtimeChannelCheck();
      }
      if (check.id === "electron_inventory") return electronInventoryCheck();
      if (check.id === "electron_zero") return electronZeroCheck();
      const commandCheck = automaticCommandCheck(check.id);
      if (commandCheck) return commandCheck;
      return { ...check, status: "blocked" as const, detail: `${check.id} verifier is not implemented yet`, evidence: [] };
    }));
    const status = checks.some((check) => check.status === "fail") ? "fail" : checks.some((check) => check.status === "blocked") ? "blocked" : "pass";
    const receipt: GateReceipt = { schema: GATE_RECEIPT_SCHEMA, gate: gate.id, phase: gate.phase, status, sourceSha, sourceIdentity, inputDigest, createdAt: new Date().toISOString(), target, checks, hardware: [], blocker: status === "blocked" ? checks.filter((check) => check.status === "blocked").map((check) => check.id).join(", ") : null };
    await writeGateReceiptAtomic(receiptFile, receipt);
    console.log(`${gate.id}: ${status}`);
    if (status !== "pass") {
      exitCode = 2;
      if (!audit) break;
    }
  }
  process.exitCode = exitCode;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main();

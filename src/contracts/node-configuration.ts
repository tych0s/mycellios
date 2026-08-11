import { z } from "zod";
import { createPublicKey } from "node:crypto";
import { DEFAULT_NODE_WORK_POLICY, nodeWorkPolicySchema } from "./node-work-policy.js";

export const MYCELLIOS_NODE_CONFIGURATION_SCHEMA = "mycellios-node-configuration/1" as const;

const nodeId = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/);
const absolutePath = z.string().min(1).refine((value) => /^(?:[A-Za-z]:[\\/]|\/)/.test(value), {
  message: "Path must be absolute",
});
const pinnedUpdateKey = z.object({
  keyId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  spki: z.string().min(1).max(256).regex(/^[A-Za-z0-9_-]+$/),
}).strict();
const componentUpdates = z.object({
  feedUrl: z.string().url(),
  pinnedKeys: z.object({
    dev: z.array(pinnedUpdateKey).max(8),
    stable: z.array(pinnedUpdateKey).max(8),
  }).strict(),
}).strict().superRefine((updates, context) => {
  const url = new URL(updates.feedUrl);
  const loopback = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]).has(url.hostname.toLowerCase());
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    context.addIssue({ code: "custom", path: ["feedUrl"], message: "Remote component feeds must use HTTPS" });
  }
  if (url.username || url.password || url.search || url.hash) {
    context.addIssue({ code: "custom", path: ["feedUrl"], message: "Component feed URL must not contain credentials, query or fragment" });
  }
  for (const channel of ["dev", "stable"] as const) {
    const seen = new Set<string>();
    for (const [index, key] of updates.pinnedKeys[channel].entries()) {
      if (seen.has(key.keyId)) context.addIssue({ code: "custom", path: ["pinnedKeys", channel, index, "keyId"], message: "Pinned key IDs must be unique per channel" });
      seen.add(key.keyId);
      if (!isEd25519Spki(key.spki)) context.addIssue({ code: "custom", path: ["pinnedKeys", channel, index, "spki"], message: "Pinned key must be canonical Ed25519 SPKI" });
    }
  }
});

export const nodeConfigurationSchema = z.object({
  schema: z.literal(MYCELLIOS_NODE_CONFIGURATION_SCHEMA),
  revision: z.number().int().positive(),
  nodeId,
  coordinator: z.object({
    url: z.string().url(),
    identityPath: absolutePath,
  }).strict(),
  worker: z.object({
    configPath: absolutePath,
  }).strict(),
  runtime: z.object({
    pythonExecutable: absolutePath,
    pythonPath: absolutePath,
    cachePath: absolutePath,
    stagePort: z.number().int().min(1_024).max(65_535),
  }).strict(),
  limits: z.object({
    maxConcurrency: z.number().int().min(1).max(64),
    maxCpuPercent: z.number().int().min(1).max(100),
    maxRamMiB: z.number().int().min(256),
    maxVramMiB: z.number().int().min(0),
    maxDiskMiB: z.number().int().min(512),
    maxTemperatureC: z.number().int().min(40).max(110),
  }).strict(),
  policy: nodeWorkPolicySchema.optional(),
  isolation: z.discriminatedUnion("mode", [
    z.object({
      mode: z.literal("windows-job-object"),
      brokerExecutable: absolutePath,
    }).strict(),
    z.object({
      mode: z.literal("linux-cgroup-v2"),
    }).strict(),
    z.object({
      mode: z.literal("macos-launchd-limits"),
    }).strict(),
  ]),
  updateChannel: z.enum(["dev", "stable"]),
  componentUpdates: componentUpdates.optional(),
  uninstall: z.object({ manifestPath: absolutePath }).strict().optional(),
}).strict();

const legacyNodeConfigurationSchema = z.object({
  schema: z.literal("mycellios-node-configuration/0"),
  nodeId,
  coordinatorUrl: z.string().url(),
  credentialPath: absolutePath,
  workerConfigPath: absolutePath,
  pythonExecutable: absolutePath,
  pythonPath: absolutePath,
  cachePath: absolutePath,
  stagePort: z.number().int().min(1_024).max(65_535),
}).strict();

export type NodeConfiguration = z.infer<typeof nodeConfigurationSchema>;

export function parseNodeConfiguration(value: unknown): NodeConfiguration {
  const current = nodeConfigurationSchema.safeParse(value);
  if (current.success) return current.data;
  const legacy = legacyNodeConfigurationSchema.safeParse(value);
  if (!legacy.success) throw current.error;
  return nodeConfigurationSchema.parse({
    schema: MYCELLIOS_NODE_CONFIGURATION_SCHEMA,
    revision: 1,
    nodeId: legacy.data.nodeId,
    coordinator: {
      url: legacy.data.coordinatorUrl,
      identityPath: legacy.data.credentialPath,
    },
    worker: { configPath: legacy.data.workerConfigPath },
    runtime: {
      pythonExecutable: legacy.data.pythonExecutable,
      pythonPath: legacy.data.pythonPath,
      cachePath: legacy.data.cachePath,
      stagePort: legacy.data.stagePort,
    },
    limits: {
      maxConcurrency: 1,
      maxCpuPercent: 90,
      maxRamMiB: 8_192,
      maxVramMiB: 0,
      maxDiskMiB: 32_768,
      maxTemperatureC: 85,
    },
    policy: DEFAULT_NODE_WORK_POLICY,
    isolation: { mode: "linux-cgroup-v2" },
    updateChannel: "stable",
  });
}

function isEd25519Spki(value: string): boolean {
  try {
    const encoded = Buffer.from(value, "base64url");
    if (encoded.toString("base64url") !== value) return false;
    const key = createPublicKey({ key: encoded, format: "der", type: "spki" });
    return key.asymmetricKeyType === "ed25519"
      && Buffer.from(key.export({ format: "der", type: "spki" })).equals(encoded);
  } catch {
    return false;
  }
}

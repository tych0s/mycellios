import { z } from "zod";

const absolutePath = z.string().min(1).refine((value) => /^(?:[A-Za-z]:[\\/]|\/)/.test(value) && !/[\r\n\0]/.test(value), {
  message: "Path must be absolute and canonical",
});
const sha256 = z.string().regex(/^sha256:[a-f0-9]{64}$/);

export const nodeInstallationManifestSchema = z.object({
  schema: z.literal("mycellios-node-installation/1"),
  platform: z.enum(["linux", "darwin", "win32"]),
  serviceName: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  installRoot: absolutePath,
  nodeExecutable: absolutePath,
  helperEntrypoint: absolutePath,
  serviceDefinitionPath: absolutePath,
  configPath: absolutePath,
  identityPath: absolutePath,
  cachePath: absolutePath,
  logsPath: absolutePath,
  statePath: absolutePath,
  receiptPath: absolutePath,
}).strict();

export const nodeUninstallRequestSchema = z.object({
  schema: z.literal("mycellios-node-uninstall-request/1"),
  id: z.string().uuid(),
  nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  generation: z.number().int().positive(),
  commandId: z.string().uuid(),
  manifestPath: absolutePath,
  manifestDigest: sha256,
  retain: z.object({
    cache: z.boolean(), logs: z.boolean(), configuration: z.boolean(), identity: z.boolean(),
  }).strict(),
  createdAt: z.string().datetime({ offset: true }),
}).strict();

export const nodeUninstallReceiptSchema = z.object({
  schema: z.literal("mycellios-node-uninstall-receipt/1"),
  requestId: z.string().uuid(),
  requestDigest: sha256,
  state: z.enum(["scheduled", "completed"]),
  platform: z.enum(["linux", "darwin", "win32"]),
  retained: z.array(z.enum(["cache", "logs", "configuration", "identity"])),
  createdAt: z.string().datetime({ offset: true }),
}).strict();

export type NodeInstallationManifest = z.infer<typeof nodeInstallationManifestSchema>;
export type NodeUninstallRequest = z.infer<typeof nodeUninstallRequestSchema>;
export type NodeUninstallReceipt = z.infer<typeof nodeUninstallReceiptSchema>;

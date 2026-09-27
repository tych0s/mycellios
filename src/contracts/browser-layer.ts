import { z } from "zod";

const hash = z.string().regex(/^[0-9a-f]{64}$/);
const digest = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const tensor = z.string().min(4).max(8 * 1024 * 1024);
const name = z.string().regex(/^[A-Za-z][A-Za-z0-9_]{0,63}$/);
const dimension = z.union([z.number().int().positive().max(131_072),
  z.enum(["tokens", "past", "total"])]);
const shape = z.array(dimension).min(1).max(6);

const auxiliary = z.object({ name, shape }).strict().refine((item) =>
  !item.shape.includes("past"), "auxiliary inputs cannot contain an empty past axis");
const state = z.object({
  name, inputName: name, outputName: name, deltaOutputName: name,
  shape, tokenAxis: z.number().int().nonnegative().max(5),
}).strict().refine((entry) => entry.tokenAxis < entry.shape.length
  && entry.shape[entry.tokenAxis] === "past"
  && entry.shape.every((value, index) => index === entry.tokenAxis || typeof value === "number"),
"state must have one past axis and fixed other dimensions");

/** The graph ABI describes tensors and state; model-specific export stays outside the browser. */
const manifestBaseSchema = z.object({
  schema: z.literal("mycellios-browser-layer/2"),
  artifactId: hash, graphSha256: hash, modelDigest: digest,
  layer: z.number().int().nonnegative().max(100_000),
  hidden: z.object({ inputName: name, outputName: name,
    width: z.number().int().positive().max(16_384) }).strict(),
  auxiliary: z.array(auxiliary).max(16),
  state: z.array(state).max(16),
  maxContextTokens: z.number().int().positive().max(131_072),
  canary: z.object({
    tokens: z.number().int().positive().max(64),
    hiddenBase64: tensor,
    auxiliaryBase64: z.record(name, tensor),
    expectedBase64: tensor,
  }).strict(),
}).strict();

export const browserLayerManifestInputSchema = manifestBaseSchema.omit({ artifactId: true });

export const browserLayerManifestSchema = manifestBaseSchema.superRefine((manifest, ctx) => {
  if (manifest.canary.tokens > manifest.maxContextTokens) {
    ctx.addIssue({ code: "custom", message: "canary exceeds model context" });
  }
  const inputs = [manifest.hidden.inputName, ...manifest.auxiliary.map((item) => item.name),
    ...manifest.state.map((item) => item.inputName)];
  const outputs = [manifest.hidden.outputName, ...manifest.state.flatMap((item) =>
    [item.outputName, item.deltaOutputName])];
  if (new Set(inputs).size !== inputs.length || new Set(outputs).size !== outputs.length
    || new Set(manifest.state.map((item) => item.name)).size !== manifest.state.length) {
    ctx.addIssue({ code: "custom", message: "browser layer tensor names must be unique" });
  }
  if (Object.keys(manifest.canary.auxiliaryBase64).sort().join("|")
    !== manifest.auxiliary.map((item) => item.name).sort().join("|")) {
    ctx.addIssue({ code: "custom", message: "canary auxiliary tensors do not match manifest" });
  }
});

export type BrowserLayerManifest = z.infer<typeof browserLayerManifestSchema>;

export const browserLayerExecuteSchema = z.object({
  artifactId: hash, requestId: z.string().uuid(),
  position: z.number().int().nonnegative().max(131_072),
  tokens: z.number().int().positive().max(4_096),
  hiddenBase64: tensor,
  auxiliaryBase64: z.record(name, tensor),
}).strict();

export type BrowserLayerExecute = z.infer<typeof browserLayerExecuteSchema>;

export function tensorElements(shape: (number | "tokens" | "past" | "total")[],
  tokens: number, past: number): number {
  let count = 1;
  for (const dimension of shape) {
    count *= dimension === "tokens" ? tokens
      : dimension === "past" ? past
        : dimension === "total" ? past + tokens : dimension;
    if (!Number.isSafeInteger(count) || count > 2_000_000) {
      throw new Error("browser layer tensor exceeds supported shape");
    }
  }
  return count;
}

export function tensorShape(shape: (number | "tokens" | "past" | "total")[],
  tokens: number, past: number): number[] {
  return shape.map((dimension) => dimension === "tokens" ? tokens
    : dimension === "past" ? past
      : dimension === "total" ? past + tokens : dimension);
}

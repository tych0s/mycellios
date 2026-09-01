import { readFile } from "node:fs/promises";
import { z } from "zod";

const traceabilitySchema = z.object({
  schema: z.literal("mycellios-final-web-native-traceability/1"),
  requirements: z.array(z.object({
    id: z.string().regex(/^FR-\d{2}$/),
    tasks: z.array(z.string().regex(/^P\d-[A-Z]$/)).min(1),
    gates: z.array(z.string().min(1)).min(1),
  }).strict()),
  acceptanceCriteria: z.array(z.object({
    id: z.string().regex(/^AC-\d{2}$/),
    checks: z.array(z.string().regex(/^G\d_[A-Z_]+\/[a-z0-9_]+$/)).min(1),
    evidenceClasses: z.array(z.enum(["automatic", "physical", "external"])).min(1),
  }).strict()),
}).strict();

const programSchema = z.object({
  schema: z.literal("mycellios-gate-program/1"),
  gates: z.array(z.object({
    id: z.string(),
    phase: z.number().int(),
    inputs: z.array(z.string()),
    checks: z.array(z.object({
      id: z.string(),
      kind: z.enum(["automatic", "physical", "external"]),
    }).strict()),
  }).strict()),
}).strict();

function orderedIds(prefix: "FR" | "AC", first: number, last: number): string[] {
  return Array.from({ length: last - first + 1 }, (_, index) => `${prefix}-${String(first + index).padStart(2, "0")}`);
}

function unique(values: string[], label: string): void {
  const duplicates = values.filter((value, index) => values.indexOf(value) !== index);
  if (duplicates.length > 0) throw new Error(`${label}_duplicates:${[...new Set(duplicates)].join(",")}`);
}

function equalIds(actual: string[], expected: string[], label: string): void {
  const missing = expected.filter((id) => !actual.includes(id));
  const unknown = actual.filter((id) => !expected.includes(id));
  if (missing.length || unknown.length) throw new Error(`${label}_mismatch:missing=${missing.join(",") || "none"}:unknown=${unknown.join(",") || "none"}`);
}

export async function verifyFinalWebNativeTraceability(root = process.cwd()): Promise<{ requirements: number; acceptanceCriteria: number; checkLinks: number; consistencyChecks: number }> {
  const [traceabilityRaw, programRaw] = await Promise.all([
    readFile(`${root}/config/final-web-native-traceability.json`, "utf8"),
    readFile(`${root}/config/final-web-native-program.json`, "utf8"),
  ]);
  const traceability = traceabilitySchema.parse(JSON.parse(traceabilityRaw));
  const program = programSchema.parse(JSON.parse(programRaw));
  const gateChecks = new Map<string, "automatic" | "physical" | "external">(
    program.gates.flatMap((gate) => gate.checks.map((check) => [`${gate.id}/${check.id}`, check.kind] as const)),
  );
  const gateIds = new Set(program.gates.map((gate) => gate.id));

  if (gateChecks.get("G7_FINAL_RELEASE/final_traceability") !== "automatic") {
    throw new Error("final_traceability_not_enforced_by_G7");
  }

  unique(traceability.requirements.map((entry) => entry.id), "traceability_requirements");
  unique(traceability.acceptanceCriteria.map((entry) => entry.id), "traceability_acceptance");
  equalIds(traceability.requirements.map((entry) => entry.id), orderedIds("FR", 0, 15), "traceability_requirements");
  equalIds(traceability.acceptanceCriteria.map((entry) => entry.id), orderedIds("AC", 1, 20), "traceability_acceptance");

  for (const requirement of traceability.requirements) {
    for (const gate of requirement.gates) if (!gateIds.has(gate)) throw new Error(`${requirement.id}_unknown_gate:${gate}`);
  }
  for (const criterion of traceability.acceptanceCriteria) {
    const kinds = new Set(criterion.checks.map((check) => {
      const kind = gateChecks.get(check);
      if (!kind) throw new Error(`${criterion.id}_unknown_check:${check}`);
      return kind;
    }));
    for (const evidenceClass of criterion.evidenceClasses) {
      if (!kinds.has(evidenceClass)) throw new Error(`${criterion.id}_missing_${evidenceClass}_check`);
    }
  }

  const orderedPhases = program.gates.map((gate) => gate.phase);
  if (orderedPhases.some((phase, index) => index > 0 && phase <= orderedPhases[index - 1]!)) {
    throw new Error("gate_phases_are_not_strictly_increasing");
  }
  unique(program.gates.map((gate) => gate.id), "program_gates");

  return {
    requirements: traceability.requirements.length,
    acceptanceCriteria: traceability.acceptanceCriteria.length,
    checkLinks: traceability.acceptanceCriteria.reduce((total, entry) => total + entry.checks.length, 0),
    consistencyChecks: 2,
  };
}

const invokedDirectly = process.argv[1]?.endsWith("verify-final-web-native-traceability.ts") ?? false;
if (invokedDirectly) {
  const result = await verifyFinalWebNativeTraceability();
  console.log(`Final web/native traceability verified: ${result.requirements} FRs, ${result.acceptanceCriteria} ACs, ${result.checkLinks} gate-check links, ${result.consistencyChecks} ordering/dependency checks.`);
}

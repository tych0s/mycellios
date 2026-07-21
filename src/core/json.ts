import { createHash } from "node:crypto";

export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, child]) => child !== undefined)
    .sort(([left], [right]) => left.localeCompare(right));
  return `{${entries
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`)
    .join(",")}}`;
}

export function sha256Text(value: string): string {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

/**
 * Canonical JSON for evidence and other content-addressed artifacts.
 *
 * Unlike `canonicalJson`, this deliberately accepts only the JSON data model.
 * It rejects values that JSON.stringify would silently erase or coerce (for
 * example undefined and non-finite numbers), accessors, sparse arrays and
 * non-plain objects. Object keys use the ECMAScript/JCS UTF-16 ordinal order,
 * never the process locale.
 */
export function canonicalEvidenceJson(value: unknown): string {
  return renderCanonicalEvidence(value, "$", new Set<object>());
}

export function sha256CanonicalEvidence(value: unknown): string {
  return sha256Text(canonicalEvidenceJson(value));
}

function renderCanonicalEvidence(
  value: unknown,
  path: string,
  ancestors: Set<object>,
): string {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError(`canonical_evidence_non_finite_number:${path}`);
    }
    return JSON.stringify(value);
  }
  if (typeof value !== "object") {
    throw new TypeError(`canonical_evidence_unsupported_value:${path}:${typeof value}`);
  }
  if (ancestors.has(value)) {
    throw new TypeError(`canonical_evidence_cycle:${path}`);
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      const ownKeys = Reflect.ownKeys(value);
      for (const key of ownKeys) {
        if (key === "length") continue;
        if (typeof key !== "string" || !/^(?:0|[1-9][0-9]*)$/.test(key)) {
          throw new TypeError(`canonical_evidence_array_property:${path}`);
        }
      }
      const rendered: string[] = [];
      for (let index = 0; index < value.length; index += 1) {
        if (!Object.prototype.hasOwnProperty.call(value, index)) {
          throw new TypeError(`canonical_evidence_sparse_array:${path}[${index}]`);
        }
        rendered.push(renderCanonicalEvidence(value[index], `${path}[${index}]`, ancestors));
      }
      return `[${rendered.join(",")}]`;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(`canonical_evidence_non_plain_object:${path}`);
    }
    const keys = Reflect.ownKeys(value);
    if (keys.some((key) => typeof key !== "string")) {
      throw new TypeError(`canonical_evidence_symbol_key:${path}`);
    }
    const stringKeys = (keys as string[]).sort(compareUtf16Ordinal);
    const rendered = stringKeys.map((key) => {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (
        descriptor === undefined ||
        !descriptor.enumerable ||
        !("value" in descriptor)
      ) {
        throw new TypeError(`canonical_evidence_non_data_property:${path}.${key}`);
      }
      return `${JSON.stringify(key)}:${renderCanonicalEvidence(
        descriptor.value,
        `${path}.${key}`,
        ancestors,
      )}`;
    });
    return `{${rendered.join(",")}}`;
  } finally {
    ancestors.delete(value);
  }
}

function compareUtf16Ordinal(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

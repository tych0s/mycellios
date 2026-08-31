import path from "node:path";

export type Component = "control" | "engine" | "shared" | "legacy" | "tooling" | "unknown";

const roots: ReadonlyArray<readonly [string, Component]> = [
  ["landing/", "control"], ["src/mobile/", "control"], ["src/coordinator/", "control"],
  ["src/support/", "shared"],
  ["src/node/", "engine"], ["src/worker/", "engine"], ["src/distribution/", "engine"], ["src/transport/", "engine"],
  ["src/adapters/", "engine"],
  ["src/model-fabric/", "engine"], ["src/scheduler/", "engine"], ["src/telemetry/", "engine"],
  ["python/distributed_runtime/", "engine"],
  ["src/contracts/", "shared"], ["src/core/", "shared"], ["src/performance/", "shared"],
  ["src/storage/", "shared"], ["src/update/", "shared"],
  ["src/economy/", "engine"],
  ["src/desktop/", "legacy"], ["src/renderer/", "legacy"],
  ["src/program/", "tooling"], ["src/benchlab/", "tooling"], ["src/simulator/", "tooling"],
  ["src/demo.ts", "tooling"], ["scripts/", "tooling"], ["tests/", "tooling"],
];

export function componentFor(portablePath: string): Component {
  const normalized = portablePath.replaceAll("\\", "/").replace(/^\.\//, "");
  return roots.find(([root]) => normalized.startsWith(root))?.[1] ?? "unknown";
}

export function resolveImport(fromFile: string, specifier: string): string | null {
  if (!specifier.startsWith(".")) return null;
  return path.posix.normalize(path.posix.join(path.posix.dirname(fromFile.replaceAll("\\", "/")), specifier));
}

export function boundaryViolation(fromFile: string, specifier: string): string | null {
  const target = resolveImport(fromFile, specifier);
  if (!target) return null;
  const sourceComponent = componentFor(fromFile);
  const targetComponent = componentFor(target);
  if (sourceComponent === "shared" && (targetComponent === "control" || targetComponent === "engine" || targetComponent === "legacy")) {
    return `shared_must_not_depend_on_${targetComponent}`;
  }
  if (sourceComponent === "engine" && (targetComponent === "control" || targetComponent === "legacy")) {
    return `engine_must_not_depend_on_${targetComponent}`;
  }
  if (sourceComponent === "control" && targetComponent === "legacy") {
    return "control_must_not_depend_on_legacy";
  }
  if ((fromFile.startsWith("landing/") || fromFile.startsWith("src/mobile/")) && targetComponent === "engine") {
    return "web_must_use_control_api_not_engine_internals";
  }
  return null;
}

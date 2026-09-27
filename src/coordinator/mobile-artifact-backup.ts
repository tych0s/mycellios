import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import type { SupabasePersistence } from "../storage/supabase-sync.js";

export function queueExistingMobileArtifacts(
  persistence: SupabasePersistence,
  directory: string,
): void {
  let files: string[];
  try {
    files = readdirSync(directory);
  } catch {
    return;
  }
  for (const file of files) {
    const path = resolve(directory, file);
    if (file.endsWith(".bin")) {
      const weightsHash = file.slice(0, -4);
      if (!/^[a-f0-9]{64}$/.test(weightsHash)) continue;
      const sizeBytes = statSync(path).size;
      persistence.registerArtifactBackup({
        id: `mobile-weight-${weightsHash}`,
        localPath: path,
        storagePath: `mobile-experts/weights/${file}`,
        contentType: "application/octet-stream",
        sha256: weightsHash,
        sizeBytes,
        metadata: { kind: "mobile-expert-weights", weightsHash },
      });
      continue;
    }
    if (!file.endsWith(".json")) continue;
    const artifactId = file.slice(0, -5);
    if (!/^[a-f0-9]{64}$/.test(artifactId)) continue;
    const body = readFileSync(path);
    persistence.registerArtifactBackup({
      id: `mobile-manifest-${artifactId}`,
      localPath: path,
      storagePath: `mobile-experts/manifests/${file}`,
      contentType: "application/json",
      sha256: createHash("sha256").update(body).digest("hex"),
      sizeBytes: body.length,
      metadata: { kind: "mobile-expert-manifest", artifactId },
    });
  }
  const layerDirectory = resolve(directory, "layers");
  let layerFiles: string[];
  try {
    layerFiles = readdirSync(layerDirectory);
  } catch {
    return;
  }
  for (const file of layerFiles) {
    const active = /^([0-9a-f]{64})-([0-9]+)\.active$/.exec(file);
    if (active) {
      const localPath = resolve(layerDirectory, file);
      const body = readFileSync(localPath);
      const artifactId = body.toString("utf8");
      if (!/^[0-9a-f]{64}$/.test(artifactId)) continue;
      persistence.registerArtifactBackup({
        id: `mobile-layer-active-${active[1]}-${active[2]}`,
        localPath,
        storagePath: `mobile-experts/layers/active/${file}`,
        contentType: "text/plain",
        sha256: createHash("sha256").update(body).digest("hex"),
        sizeBytes: body.length,
        metadata: { kind: "mobile-layer-active", modelDigest: `sha256:${active[1]}`,
          layer: Number(active[2]), artifactId },
      });
      continue;
    }
    const match = /^([0-9a-f]{64})\.(onnx|json)$/.exec(file);
    if (!match) continue;
    const id = match[1]!;
    const graph = match[2] === "onnx";
    const localPath = resolve(layerDirectory, file);
    const sizeBytes = statSync(localPath).size;
    persistence.registerArtifactBackup({
      id: graph ? `mobile-layer-graph-${id}` : `mobile-layer-manifest-${id}`,
      localPath,
      storagePath: graph
        ? `mobile-experts/layers/graphs/${file}`
        : `mobile-experts/layers/manifests/${file}`,
      contentType: graph ? "application/octet-stream" : "application/json",
      sha256: graph ? id : createHash("sha256").update(readFileSync(localPath)).digest("hex"),
      sizeBytes,
      metadata: graph
        ? { kind: "mobile-layer-graph", graphSha256: id }
        : { kind: "mobile-layer-manifest", artifactId: id },
    });
  }
}

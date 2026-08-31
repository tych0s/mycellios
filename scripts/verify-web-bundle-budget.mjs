import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { basename, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const scriptPattern = /\.(?:m?js)$/;

function manifestScripts(manifest) {
  return [...new Set(Object.values(manifest)
    .map((entry) => entry?.file)
    .filter((file) => typeof file === "string" && scriptPattern.test(file)))];
}

function outputScripts(root, outputDirectory) {
  const assets = resolve(root, outputDirectory, "assets");
  if (!existsSync(assets)) return [];
  return readdirSync(assets, { withFileTypes: true })
    .filter((entry) => entry.isFile() && scriptPattern.test(entry.name))
    .map((entry) => `assets/${entry.name}`);
}

function dependencyClosure(manifest, entryKey) {
  if (!manifest[entryKey]) throw new Error(`critical_path_entry_missing:${entryKey}`);
  const seen = new Set();
  const visit = (key) => {
    if (seen.has(key)) return;
    const entry = manifest[key];
    if (!entry) throw new Error(`manifest_import_missing:${key}`);
    seen.add(key);
    for (const imported of entry.imports ?? []) visit(imported);
  };
  visit(entryKey);
  return [...seen].map((key) => manifest[key].file).filter((file) => scriptPattern.test(file));
}

export function verifyWebBundleBudget({ root = process.cwd(), outputDirectory, manifest, definition }) {
  const declaredScripts = manifestScripts(manifest);
  const emittedScripts = outputScripts(root, outputDirectory);
  for (const file of emittedScripts) {
    if (!declaredScripts.includes(file)) throw new Error(`bundle_not_in_manifest:${file}`);
  }

  const results = [];
  const covered = new Set();
  for (const [role, budget] of Object.entries(definition.budgets)) {
    const matches = declaredScripts.filter((candidate) => basename(candidate).startsWith(budget.match));
    if (matches.length === 0) throw new Error(`bundle_role_missing:${role}:${budget.match}`);
    for (const file of matches) {
      const bytes = statSync(resolve(root, outputDirectory, file)).size;
      if (bytes > budget.maximumBytes) throw new Error(`bundle_budget_exceeded:${role}:${bytes}>${budget.maximumBytes}`);
      covered.add(file);
      results.push({ kind: "asset", role, file, bytes, maximumBytes: budget.maximumBytes });
    }
  }

  for (const file of declaredScripts.filter((candidate) => !covered.has(candidate))) {
    const bytes = statSync(resolve(root, outputDirectory, file)).size;
    if (bytes > definition.defaultMaximumBytes) {
      throw new Error(`unknown_bundle_exceeds_default:${file}:${bytes}>${definition.defaultMaximumBytes}`);
    }
    results.push({ kind: "default", role: "default", file, bytes, maximumBytes: definition.defaultMaximumBytes });
  }

  for (const [route, budget] of Object.entries(definition.criticalPaths)) {
    const files = new Set(dependencyClosure(manifest, budget.entry));
    for (const match of budget.highPriorityMatches) {
      const matches = declaredScripts.filter((candidate) => basename(candidate).startsWith(match));
      if (matches.length === 0) throw new Error(`critical_path_preload_missing:${route}:${match}`);
      for (const file of matches) files.add(file);
    }
    const bytes = [...files].reduce((total, file) => total + statSync(resolve(root, outputDirectory, file)).size, 0);
    if (bytes > budget.maximumBytes) throw new Error(`critical_path_budget_exceeded:${route}:${bytes}>${budget.maximumBytes}`);
    results.push({ kind: "critical-path", role: route, file: [...files].sort().join(","), bytes, maximumBytes: budget.maximumBytes });
  }

  if (definition.sourceMapsAllowed === false) {
    const maps = readdirSync(resolve(root, outputDirectory, "assets"), { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith(".map"));
    if (maps.length > 0) throw new Error(`public_source_maps_forbidden:${maps[0].name}`);
  }
  return results;
}

export function loadAndVerifyWebBundleBudget(root = process.cwd(), selectedSurface) {
  const policy = JSON.parse(readFileSync(resolve(root, "config/web-bundle-budget.json"), "utf8"));
  if (policy.schema !== "mycellios.web-bundle-budget.v2") throw new Error("unsupported_bundle_budget_schema");
  const surfaces = selectedSurface ? { [selectedSurface]: policy.surfaces[selectedSurface] } : policy.surfaces;
  if (selectedSurface && !surfaces[selectedSurface]) throw new Error(`unknown_bundle_surface:${selectedSurface}`);
  return Object.entries(surfaces).flatMap(([surface, definition]) => {
    const manifestPath = resolve(root, definition.outputDirectory, ".vite/manifest.json");
    if (!existsSync(manifestPath)) throw new Error(`${surface}_manifest_missing:run npm run ${surface}:build first`);
    return verifyWebBundleBudget({ root, outputDirectory: definition.outputDirectory,
      manifest: JSON.parse(readFileSync(manifestPath, "utf8")), definition })
      .map((result) => ({ surface, ...result }));
  });
}

const invokedDirectly = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (invokedDirectly) {
  const surfaceArgument = process.argv.find((argument) => argument.startsWith("--surface="));
  const results = loadAndVerifyWebBundleBudget(process.cwd(), surfaceArgument?.slice("--surface=".length));
  for (const result of results.filter((entry) => entry.kind !== "default")) {
    console.log(`${result.surface}/${result.kind}/${result.role}: ${result.bytes}/${result.maximumBytes} bytes`);
  }
  console.log(`Web bundle policy verified: ${results.length} asset and critical-path checks.`);
}

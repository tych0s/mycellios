export interface ElectronZeroInputs {
  trackedFiles: readonly string[];
  packageDocument: Record<string, unknown>;
  lockDocument: Record<string, unknown>;
  fileContents: ReadonlyMap<string, string>;
}

export interface ElectronZeroViolation {
  category: "source" | "build" | "dependency" | "script" | "workflow" | "artifact" | "documentation";
  subject: string;
  reason: string;
}

const PRODUCTIVE_TEXT_PATH = /^(?:src|landing|scripts|\.github\/workflows)\//u;
const PRODUCTIVE_TEXT_EXEMPT = new Set([
  "src/program/electron-zero-verifier.ts",
  "src/program/component-boundaries.ts",
  "scripts/verify-electron-zero.ts",
  "scripts/verify-final-web-native-traceability.ts",
  "scripts/gate-program.ts",
]);
const PRODUCTIVE_ELECTRON_TOKEN = /\belectron(?:-forge)?\b|\bsquirrel\b|\.asar\b|\.nupkg\b|\/updates\/win32\b/iu;
const LEGACY_RELEASE_FEED_TOKEN = /\bRELEASES\b/u;

const PRODUCTIVE_PATHS = [
  /^src\/desktop\//u,
  /^src\/renderer\//u,
  /^forge\.config\./u,
  /^vite\.(?:main|preload|renderer)\.config\./u,
  /^tsconfig\.desktop\.json$/u,
  /^scripts\/(?:desktop-|prepare-desktop|verify-desktop)/u,
] as const;
const ELECTRON_PACKAGE = /^(?:electron$|electron-squirrel-startup$|electron-installer(?:-|$)|electron-winstaller$|@electron\/|@electron-forge\/)/u;
const ELECTRON_ARTIFACT = /(?:^|\/)(?:RELEASES|[^/]+\.nupkg|[^/]+\.asar)$/iu;
const OPERATIONAL_DOC = /^(?:README\.md|docs\/(?!archive\/|history\/).+\.(?:md|mdx))$/u;
const HISTORICAL_DOCUMENT_MARKER = "<!-- mycellios:historical-non-operational -->";
const OPERATIONAL_INSTRUCTION = /\b(?:electron-forge|npm\s+run\s+desktop:|open|launch|start|install|download|publish|package|build)\b[^\n]{0,100}\b(?:Electron|electron|Forge|Squirrel|ASAR)\b|\b(?:Electron|electron|Forge|Squirrel|ASAR)\b[^\n]{0,100}\b(?:app|client|installer|runtime|package|release|download|build)\b/iu;

export function verifyElectronZero(inputs: ElectronZeroInputs): ElectronZeroViolation[] {
  const violations: ElectronZeroViolation[] = [];
  for (const file of inputs.trackedFiles) {
    if (PRODUCTIVE_PATHS.some((pattern) => pattern.test(file))) {
      violations.push({
        category: file.startsWith("src/") ? "source" : "build",
        subject: file,
        reason: "productive Electron path remains tracked",
      });
    }
    if (ELECTRON_ARTIFACT.test(file)) {
      violations.push({ category: "artifact", subject: file, reason: "Electron update/package artifact remains tracked" });
    }
    if (file.startsWith(".github/workflows/")) {
      const source = (inputs.fileContents.get(file) ?? "").replaceAll("verify:electron-zero", "");
      if (/electron-forge|electron-builder|electron-packager|@electron\/|\belectron\b|forge\.config|\bsquirrel\b|\.asar\b|desktop:/iu.test(source)) {
        violations.push({ category: "workflow", subject: file, reason: "workflow still builds or publishes Electron" });
      }
    }
    const documentation = inputs.fileContents.get(file) ?? "";
    if (
      OPERATIONAL_DOC.test(file)
      && !documentation.startsWith(HISTORICAL_DOCUMENT_MARKER)
      && OPERATIONAL_INSTRUCTION.test(documentation)
    ) {
      violations.push({ category: "documentation", subject: file, reason: "operational Electron instruction remains" });
    }
    if (PRODUCTIVE_TEXT_PATH.test(file) && !PRODUCTIVE_TEXT_EXEMPT.has(file)) {
      const productiveSource = (inputs.fileContents.get(file) ?? "")
        .replaceAll("verify:electron-zero", "");
      if (
        PRODUCTIVE_ELECTRON_TOKEN.test(productiveSource)
        || LEGACY_RELEASE_FEED_TOKEN.test(productiveSource)
      ) {
        violations.push({
          category: file.startsWith(".github/workflows/") ? "workflow" : "source",
          subject: file,
          reason: "productive source retains an Electron, Squirrel, ASAR, or legacy update-feed token",
        });
      }
    }
  }

  const scripts = record(inputs.packageDocument.scripts);
  for (const [name, command] of Object.entries(scripts)) {
    if (name.startsWith("desktop:") || (
      name !== "verify:electron-zero"
      && /electron-forge|electron-builder|electron-packager|@electron\/|\belectron\b|forge\.config|\bsquirrel\b|\.asar\b/iu.test(String(command))
    )) {
      violations.push({ category: "script", subject: name, reason: "package script retains an Electron lifecycle" });
    }
  }
  for (const section of ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies"] as const) {
    for (const name of Object.keys(record(inputs.packageDocument[section]))) {
      if (ELECTRON_PACKAGE.test(name)) {
        violations.push({ category: "dependency", subject: `${section}:${name}`, reason: "direct Electron dependency remains" });
      }
    }
  }
  for (const name of Object.keys(record(inputs.packageDocument.overrides))) {
    if (ELECTRON_PACKAGE.test(name)) {
      violations.push({ category: "dependency", subject: `overrides:${name}`, reason: "Electron override remains" });
    }
  }
  for (const key of Object.keys(record(inputs.lockDocument.packages))) {
    const name = key.startsWith("node_modules/") ? key.split("node_modules/").at(-1) ?? "" : "";
    if (name && ELECTRON_PACKAGE.test(name)) {
      violations.push({ category: "dependency", subject: `lock:${name}`, reason: "transitive Electron package remains in lockfile" });
    }
  }
  return [...new Map(violations.map((violation) => [
    `${violation.category}\u0000${violation.subject}\u0000${violation.reason}`,
    violation,
  ])).values()].sort((left, right) =>
    left.category.localeCompare(right.category) || left.subject.localeCompare(right.subject));
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

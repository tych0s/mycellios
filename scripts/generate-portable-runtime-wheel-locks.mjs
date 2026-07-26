import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";

const workspace = resolve(import.meta.dirname, "..");
const outputDirectory = resolve(import.meta.dirname, "wheel-locks");
const targets = [
  {
    platform: "win32",
    arch: "x64",
    report: "build/mycellios-win-lock-report.json",
  },
  {
    platform: "linux",
    arch: "x64",
    report: "build/mycellios-linux-x64-lock-report.json",
  },
  {
    platform: "darwin",
    arch: "arm64",
    report: "build/mycellios-darwin-arm64-lock-report.json",
  },
];

mkdirSync(outputDirectory, { recursive: true });
for (const target of targets) {
  const reportPath = resolve(workspace, target.report);
  const report = JSON.parse(readFileSync(reportPath, "utf8"));
  if (!Array.isArray(report.install) || report.install.length === 0) {
    throw new Error(`Pip report has no install closure: ${reportPath}.`);
  }
  const observed = new Set();
  const artifacts = report.install.map((entry, index) => {
    const name = normalizeName(entry?.metadata?.name);
    const version = requiredToken(entry?.metadata?.version, `version ${index}`);
    const url = requiredHttpsUrl(entry?.download_info?.url, index);
    const hash = entry?.download_info?.archive_info?.hash;
    if (typeof hash !== "string" || !/^sha256=[0-9a-f]{64}$/.test(hash)) {
      throw new Error(`Pip report artifact ${index} has no exact SHA-256.`);
    }
    const filename = decodeURIComponent(basename(new URL(url).pathname));
    if (!filename.endsWith(".whl") || /[\0\r\n]/.test(filename)) {
      throw new Error(`Pip report artifact ${index} is not a wheel.`);
    }
    if (observed.has(name)) {
      throw new Error(`Pip report resolves ${name} more than once.`);
    }
    observed.add(name);
    return {
      name,
      version,
      url,
      sha256: hash.slice("sha256=".length),
    };
  }).sort((left, right) => left.name.localeCompare(right.name, "en"));
  const lines = [
    "# mycellios-python-wheel-lock/1",
    `# target=${target.platform}/${target.arch}/cp312`,
    "# Regenerate deliberately from a reviewed pip JSON report; never edit hashes by hand.",
    ...artifacts.map(
      (artifact) =>
        `${artifact.name} @ ${artifact.url} --hash=sha256:${artifact.sha256} # version=${artifact.version}`,
    ),
    "",
  ];
  const output = resolve(
    outputDirectory,
    `${target.platform}-${target.arch}-cp312.txt`,
  );
  writeFileSync(output, lines.join("\n"), "utf8");
  process.stdout.write(`${output}: ${artifacts.length} sealed wheels\n`);
}

function normalizeName(value) {
  const name = requiredToken(value, "package name")
    .toLowerCase()
    .replace(/[_.]+/g, "-");
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name)) {
    throw new Error(`Invalid Python package name: ${value}.`);
  }
  return name;
}

function requiredToken(value, label) {
  if (
    typeof value !== "string"
    || value.length < 1
    || value.length > 256
    || /[\0\r\n]/.test(value)
  ) {
    throw new Error(`Invalid ${label}.`);
  }
  return value;
}

function requiredHttpsUrl(value, index) {
  const raw = requiredToken(value, `artifact URL ${index}`);
  const url = new URL(raw);
  if (
    url.protocol !== "https:"
    || url.username
    || url.password
    || url.search
    || ![
      "files.pythonhosted.org",
      "download.pytorch.org",
      "download-r2.pytorch.org",
    ].includes(url.hostname)
  ) {
    throw new Error(`Unapproved Python artifact URL: ${raw}.`);
  }
  return url.href;
}

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const metadata = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const version = metadata.version;
const makerDirectory = resolve(root, "out", "make", "squirrel.windows", "x64");
const updateDirectory = resolve(root, "updates", "win32", "x64");
const downloadDirectory = resolve(root, "landing-dist", "downloads");
const fullPackageName = `mycellios-${version}-full.nupkg`;
const expected = ["RELEASES", "mycellios-setup.exe", fullPackageName];

for (const fileName of expected) {
  const path = resolve(makerDirectory, fileName);
  if (!existsSync(path)) throw new Error(`Missing Windows release artifact: ${path}`);
}

mkdirSync(updateDirectory, { recursive: true });
for (const fileName of readdirSync(updateDirectory)) {
  if (fileName === "RELEASES" || fileName === "latest.json" || fileName.endsWith(".nupkg") || fileName.endsWith("-setup.exe")) {
    rmSync(resolve(updateDirectory, fileName), { force: true });
  }
}

for (const fileName of expected) {
  copyFileSync(resolve(makerDirectory, fileName), resolve(updateDirectory, fileName));
}

mkdirSync(downloadDirectory, { recursive: true });
copyFileSync(
  resolve(makerDirectory, "mycellios-setup.exe"),
  resolve(downloadDirectory, "mycellios-windows-x64.exe"),
);
writeFileSync(
  resolve(updateDirectory, "latest.json"),
  `${JSON.stringify(
    {
      version,
      publishedAt: new Date().toISOString(),
      files: expected.map((fileName) => basename(fileName)),
    },
    null,
    2,
  )}\n`,
  "utf8",
);

console.log(`Prepared mycellios ${version} Windows update feed at ${updateDirectory}`);

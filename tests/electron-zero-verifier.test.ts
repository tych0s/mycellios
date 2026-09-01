import { describe, expect, it } from "vitest";
import { verifyElectronZero } from "../src/program/electron-zero-verifier.js";

describe("ELECTRON_ZERO verifier", () => {
  it("accepts a native-node and web-only product tree", () => {
    expect(verifyElectronZero({
      trackedFiles: ["src/node/main.ts", "landing/src/Panel.tsx", ".github/workflows/node-build.yml"],
      packageDocument: { scripts: { "node:start": "tsx src/node/main.ts" }, dependencies: { fastify: "1" } },
      lockDocument: { packages: { "": {}, "node_modules/fastify": {} } },
      fileContents: new Map([[".github/workflows/node-build.yml", "run: npm run node:start"]]),
    })).toEqual([]);
  });

  it("rejects productive paths, direct and transitive packages, scripts, workflows and artifacts", () => {
    const violations = verifyElectronZero({
      trackedFiles: [
        "src/desktop/main.ts",
        "forge.config.ts",
        ".github/workflows/desktop-build.yml",
        "release/RELEASES",
      ],
      packageDocument: {
        scripts: { "desktop:start": "electron-forge start" },
        devDependencies: { electron: "1", "@electron-forge/cli": "1" },
        overrides: { "@electron/rebuild": "1" },
      },
      lockDocument: { packages: { "node_modules/electron": {}, "node_modules/electron-winstaller": {} } },
      fileContents: new Map([[".github/workflows/desktop-build.yml", "run: npm run desktop:start"]]),
    });
    expect(new Set(violations.map((violation) => violation.category))).toEqual(new Set([
      "source", "build", "dependency", "script", "workflow", "artifact",
    ]));
  });

  it("allows historical specs but rejects active operational documentation", () => {
    const violations = verifyElectronZero({
      trackedFiles: ["specs/legacy.md", "docs/install.md"],
      packageDocument: {},
      lockDocument: {},
      fileContents: new Map([
        ["specs/legacy.md", "Electron was the previous shell."],
        ["docs/install.md", "Download the Electron installer for the desktop app."],
      ]),
    });
    expect(violations).toEqual([expect.objectContaining({ category: "documentation", subject: "docs/install.md" })]);
  });
});

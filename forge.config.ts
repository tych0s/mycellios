import type { ForgeConfig } from "@electron-forge/shared-types";
import { MakerDeb } from "@electron-forge/maker-deb";
import { MakerDMG } from "@electron-forge/maker-dmg";
import { MakerRpm } from "@electron-forge/maker-rpm";
import { MakerSquirrel } from "@electron-forge/maker-squirrel";
import { MakerZIP } from "@electron-forge/maker-zip";
import { FusesPlugin } from "@electron-forge/plugin-fuses";
import { VitePlugin } from "@electron-forge/plugin-vite";
import { FuseV1Options, FuseVersion } from "@electron/fuses";
import { existsSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import {
  NATIVE_BUILD_PROVENANCE_FILE,
  buildNativeSourceProvenance,
} from "./scripts/native-build-provenance.mjs";

const windowsSign = process.env.WINDOWS_CERTIFICATE_FILE
  ? {
      certificateFile: process.env.WINDOWS_CERTIFICATE_FILE,
      ...(process.env.WINDOWS_CERTIFICATE_PASSWORD
        ? { certificatePassword: process.env.WINDOWS_CERTIFICATE_PASSWORD }
        : {}),
      description: "mycellios distributed intelligence",
    }
  : undefined;

const portableRuntimeArchive = "build/distribution-runtime.tar.gz";

function commandLineOption(name: string): string | undefined {
  const equalsPrefix = `--${name}=`;
  const equalsValue = process.argv.find((value) => value.startsWith(equalsPrefix));
  if (equalsValue) return equalsValue.slice(equalsPrefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const targetPlatform = commandLineOption("platform") ?? process.platform;
const targetArch = commandLineOption("arch") ?? process.arch;
const macDistributionIdentity = process.env.MYCELLIOS_MAC_SIGN_IDENTITY?.trim();
const macSign =
  targetPlatform === "darwin"
    ? macDistributionIdentity
      ? { identity: macDistributionIdentity }
      : {
          identity: "-",
          identityValidation: false,
          preAutoEntitlements: false,
          optionsForFile: () => ({ timestamp: "none" }),
        }
    : undefined;
const macNotarize =
  macDistributionIdentity &&
  process.env.APPLE_ID &&
  process.env.APPLE_APP_SPECIFIC_PASSWORD &&
  process.env.APPLE_TEAM_ID
    ? {
        appleId: process.env.APPLE_ID,
        appleIdPassword: process.env.APPLE_APP_SPECIFIC_PASSWORD,
        teamId: process.env.APPLE_TEAM_ID,
      }
    : undefined;
const bundlesPortableRuntime =
  (targetPlatform === "win32" && targetArch === "x64") ||
  (targetPlatform === "linux" && targetArch === "x64") ||
  (targetPlatform === "darwin" && (targetArch === "arm64" || targetArch === "x64"));

function portableRuntimeMatchesTarget(): boolean {
  if (!bundlesPortableRuntime || !existsSync(portableRuntimeArchive)) return false;
  for (const entry of ["./runtime-manifest.json", "runtime-manifest.json"]) {
    const result = spawnSync("tar", ["-xOf", portableRuntimeArchive, entry], {
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      shell: false,
      windowsHide: true,
    });
    if (result.status !== 0 || !result.stdout.trim()) continue;
    try {
      const manifest = JSON.parse(result.stdout) as { platform?: unknown; arch?: unknown };
      return manifest.platform === targetPlatform && manifest.arch === targetArch;
    } catch {
      return false;
    }
  }
  return false;
}

const includePortableRuntime = portableRuntimeMatchesTarget();

const config: ForgeConfig = {
  hooks: {
    packageAfterCopy: async (_forgeConfig, buildPath) => {
      const provenance = buildNativeSourceProvenance(import.meta.dirname);
      await writeFile(
        resolve(buildPath, NATIVE_BUILD_PROVENANCE_FILE),
        `${JSON.stringify(provenance, null, 2)}\n`,
        "utf8",
      );
    },
  },
  packagerConfig: {
    asar: true,
    executableName: "mycellios",
    appBundleId: "app.mycellios.desktop",
    appCategoryType: "public.app-category.utilities",
    win32metadata: {
      CompanyName: "mycellios",
      FileDescription: "mycellios distributed intelligence",
      ProductName: "mycellios",
      InternalName: "mycellios",
      OriginalFilename: "mycellios.exe",
      "requested-execution-level": "asInvoker",
    },
    icon: "build/icons/app-icon-v2",
    ...(targetPlatform === "darwin" && targetArch === "arm64"
      ? { extendInfo: { LSMinimumSystemVersion: "14.0" } }
      : {}),
    extraResource: [
      "assets/mycellios-logo.png",
      "build/icons",
      "mobile-dist",
      "landing-dist",
      "build/python",
      ...(includePortableRuntime
        ? [portableRuntimeArchive]
        : []),
    ],
    ...(windowsSign ? { windowsSign } : {}),
    ...(macSign ? { osxSign: macSign } : {}),
    ...(macNotarize ? { osxNotarize: macNotarize } : {}),
  },
  rebuildConfig: {},
  makers: [
    new MakerSquirrel({
      name: "mycellios",
      setupExe: "mycellios-setup.exe",
      setupIcon: "build/icons/app-icon-v2.ico",
      ...(windowsSign ? { windowsSign } : {}),
    }),
    new MakerZIP({}, ["darwin", "linux"]),
    new MakerDMG(
      {
        name: "mycellios",
        icon: "build/icons/app-icon-v2.icns",
        format: "ULFO",
      },
      ["darwin"],
    ),
    new MakerDeb({
      options: {
        name: "mycellios",
        productName: "mycellios",
        genericName: "Distributed inference network",
        description: "Control and contribute compute to the mycellios distributed inference network.",
        categories: ["Network", "Utility"],
        icon: "build/icons/app-icon-v2.png",
      },
    }),
    new MakerRpm({
      options: {
        name: "mycellios",
        productName: "mycellios",
        genericName: "Distributed inference network",
        description: "Control and contribute compute to the mycellios distributed inference network.",
        license: "UNLICENSED",
        categories: ["Network", "Utility"],
        icon: "build/icons/app-icon-v2.png",
      },
    }),
  ],
  plugins: [
    new VitePlugin({
      build: [
        {
          entry: "src/desktop/main.ts",
          config: "vite.main.config.ts",
          target: "main",
        },
        {
          entry: "src/desktop/preload.ts",
          config: "vite.preload.config.ts",
          target: "preload",
        },
      ],
      renderer: [
        {
          name: "main_window",
          config: "vite.renderer.config.ts",
        },
      ],
    }),
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: true,
    }),
  ],
};

export default config;

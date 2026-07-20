import type { ForgeConfig } from "@electron-forge/shared-types";
import { MakerDeb } from "@electron-forge/maker-deb";
import { MakerDMG } from "@electron-forge/maker-dmg";
import { MakerRpm } from "@electron-forge/maker-rpm";
import { MakerSquirrel } from "@electron-forge/maker-squirrel";
import { MakerZIP } from "@electron-forge/maker-zip";
import { FusesPlugin } from "@electron-forge/plugin-fuses";
import { VitePlugin } from "@electron-forge/plugin-vite";
import { FuseV1Options, FuseVersion } from "@electron/fuses";

const windowsSign = process.env.WINDOWS_CERTIFICATE_FILE
  ? {
      certificateFile: process.env.WINDOWS_CERTIFICATE_FILE,
      ...(process.env.WINDOWS_CERTIFICATE_PASSWORD
        ? { certificatePassword: process.env.WINDOWS_CERTIFICATE_PASSWORD }
        : {}),
      description: "mycellios distributed intelligence",
    }
  : undefined;

const macSign = process.env.MYCELLIOS_MAC_SIGN_IDENTITY
  ? { identity: process.env.MYCELLIOS_MAC_SIGN_IDENTITY }
  : undefined;

const macNotarize =
  macSign &&
  process.env.APPLE_ID &&
  process.env.APPLE_APP_SPECIFIC_PASSWORD &&
  process.env.APPLE_TEAM_ID
    ? {
        appleId: process.env.APPLE_ID,
        appleIdPassword: process.env.APPLE_APP_SPECIFIC_PASSWORD,
        teamId: process.env.APPLE_TEAM_ID,
      }
    : undefined;

const config: ForgeConfig = {
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
    icon: "build/icons/icon",
    extraResource: ["assets/mycellios-logo.png", "build/icons", "mobile-dist"],
    ...(windowsSign ? { windowsSign } : {}),
    ...(macSign ? { osxSign: macSign } : {}),
    ...(macNotarize ? { osxNotarize: macNotarize } : {}),
  },
  rebuildConfig: {},
  makers: [
    new MakerSquirrel({
      name: "mycellios",
      setupExe: "mycellios-setup.exe",
      setupIcon: "build/icons/icon.ico",
      ...(windowsSign ? { windowsSign } : {}),
    }),
    new MakerZIP({}, ["darwin", "linux"]),
    new MakerDMG(
      {
        name: "mycellios",
        icon: "build/icons/icon.icns",
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
        icon: "build/icons/icon.png",
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
        icon: "build/icons/icon.png",
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

import type { FastifyInstance } from "fastify";
import { lstatSync } from "node:fs";
import { resolve } from "node:path";
import {
  PUBLIC_DOWNLOADS_SCHEMA,
  publicDownloadAvailabilitySchema,
} from "../contracts/public-downloads.js";
import type { NativeReleaseTransactionStore } from "./release-upload.js";

const NATIVE_DOWNLOADS = [
  { id: "windows-x64", label: "Windows 10/11 · x64", format: "MSI", suffix: "windows-x64.msi", legacyFormat: "ZIP", legacyFileName: "mycellios-node-windows-x64.zip", path: "/downloads/windows" },
  { id: "macos-arm64", label: "macOS · Apple Silicon", format: "PKG", suffix: "macos-arm64.pkg", legacyFormat: "TAR.GZ", legacyFileName: "mycellios-node-macos-arm64.tar.gz", path: "/downloads/macos-arm64" },
  { id: "linux-x64", label: "Linux · x64", format: "DEB", suffix: "linux-x64.deb", legacyFormat: "TAR.GZ", legacyFileName: "mycellios-node-linux-x64.tar.gz", path: "/downloads/linux" },
] as const;

export function registerPublicDownloadRoutes(
  app: FastifyInstance,
  options: {
    version: string;
    downloadsRoot: string | null;
    transactions: NativeReleaseTransactionStore | null;
  },
): void {
  const resolveAsset = async (fileName: string) => options.transactions
    ? options.transactions.publicAssetPath("downloads", fileName)
    : options.downloadsRoot
      ? resolve(options.downloadsRoot, fileName)
      : null;
  const available = (filePath: string | null): boolean => {
    if (!filePath) return false;
    try {
      const details = lstatSync(filePath);
      return details.isFile() && !details.isSymbolicLink() && details.size > 0;
    } catch {
      return false;
    }
  };

  const packageFor = async (target: typeof NATIVE_DOWNLOADS[number]) => {
    const fileName = `mycellios-node-${options.version}-${target.suffix}`;
    if (available(await resolveAsset(fileName))) {
      return { id: target.id, label: target.label, format: target.format, fileName, path: target.path, available: true };
    }
    if (available(await resolveAsset(target.legacyFileName))) {
      return { id: target.id, label: target.label, format: target.legacyFormat, fileName: target.legacyFileName, path: target.path, available: true };
    }
    return { id: target.id, label: target.label, format: target.format, fileName, path: target.path, available: false };
  };

  app.get("/public/v1/downloads", async (_request, reply) => {
    const packages = await Promise.all(NATIVE_DOWNLOADS.map(packageFor));
    reply.header("Cache-Control", "public, max-age=30");
    return publicDownloadAvailabilitySchema.parse({
      schema: PUBLIC_DOWNLOADS_SCHEMA,
      version: options.version,
      packages,
    });
  });

  for (const target of NATIVE_DOWNLOADS) {
    app.get(target.path, async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      const selected = await packageFor(target);
      if (!selected.available) {
        return reply.redirect(`/downloads?availability=unavailable&platform=${target.id}`);
      }
      return reply.redirect(`/downloads/${selected.fileName}?v=${options.version}`);
    });
  }
  app.get("/downloads/linux-deb", async (_request, reply) => reply.redirect("/downloads/linux"));
  app.get("/downloads/linux-rpm", async (_request, reply) =>
    reply.redirect("/downloads?availability=unsupported&platform=linux-rpm"));
}

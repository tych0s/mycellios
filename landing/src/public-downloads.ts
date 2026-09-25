import type { PublicDownloadAvailability } from "../../src/contracts/public-downloads.js";

const DOWNLOADS_SCHEMA = "mycellios-public-downloads/1";

export const PUBLIC_DOWNLOAD_OPTIONS = [
  { id: "windows-x64", label: "Windows", detail: "Windows 10/11 · x64", format: "ZIP" },
  { id: "macos-arm64", label: "macOS", detail: "Apple Silicon", format: "TAR.GZ" },
  { id: "linux-x64", label: "Linux", detail: "Linux · x64", format: "TAR.GZ" },
] as const;

export async function fetchPublicDownloadAvailability(): Promise<PublicDownloadAvailability> {
  const response = await fetch("/public/v1/downloads", {
    headers: { accept: "application/json" },
  });
  if (!response.ok) throw new Error(`download_availability_http_${response.status}`);
  const value: unknown = await response.json();
  if (!isPublicDownloadAvailability(value)) throw new Error("download_availability_invalid");
  return value;
}

function isPublicDownloadAvailability(value: unknown): value is PublicDownloadAvailability {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  if (candidate.schema !== DOWNLOADS_SCHEMA || typeof candidate.version !== "string" || !Array.isArray(candidate.packages) || candidate.packages.length !== PUBLIC_DOWNLOAD_OPTIONS.length) return false;
  return candidate.packages.every((item) => {
    if (!item || typeof item !== "object") return false;
    const entry = item as Record<string, unknown>;
    return ["windows-x64", "macos-arm64", "linux-x64"].includes(String(entry.id))
      && typeof entry.label === "string"
      && ["ZIP", "TAR.GZ"].includes(String(entry.format))
      && typeof entry.fileName === "string"
      && typeof entry.path === "string"
      && entry.path.startsWith("/downloads/")
      && typeof entry.available === "boolean";
  });
}

import { z } from "zod";

export const PUBLIC_DOWNLOADS_SCHEMA = "mycellios-public-downloads/1";

export const publicDownloadAvailabilitySchema = z.object({
  schema: z.literal(PUBLIC_DOWNLOADS_SCHEMA),
  version: z.string().min(1),
  packages: z.array(z.object({
    id: z.enum(["windows-x64", "macos-arm64", "linux-x64"]),
    label: z.string().min(1),
    format: z.enum(["ZIP", "TAR.GZ"]),
    fileName: z.string().min(1),
    path: z.string().startsWith("/downloads/"),
    available: z.boolean(),
  }).strict()).length(3),
}).strict();

export type PublicDownloadAvailability = z.infer<typeof publicDownloadAvailabilitySchema>;
export type PublicDownloadPackage = PublicDownloadAvailability["packages"][number];

import { afterEach, describe, expect, it, vi } from "vitest";
import type { PublicDownloadAvailability } from "../../src/contracts/public-downloads.js";
import { downloadAvailabilityNotice } from "./Downloads";
import { fetchPublicDownloadAvailability } from "./public-downloads";

afterEach(() => vi.unstubAllGlobals());

describe("public package availability", () => {
  const availability: PublicDownloadAvailability = {
    schema: "mycellios-public-downloads/1",
    version: "0.2.77",
    packages: [
      { id: "windows-x64", label: "Windows", format: "ZIP", fileName: "mycellios.zip", path: "/downloads/mycellios.zip", available: true },
      { id: "macos-arm64", label: "macOS", format: "TAR.GZ", fileName: "mycellios-macos.tar.gz", path: "/downloads/mycellios-macos.tar.gz", available: false },
      { id: "linux-x64", label: "Linux", format: "TAR.GZ", fileName: "mycellios-linux.tar.gz", path: "/downloads/mycellios-linux.tar.gz", available: false },
    ],
  };

  it("waits for current package status instead of trusting an old URL hint", () => {
    expect(downloadAvailabilityNotice(null, false, "windows-x64", "unavailable")).toBeNull();
    expect(downloadAvailabilityNotice(availability, false, "windows-x64", "unavailable")).toBeNull();
  });

  it("reports only the requested package if its current release is missing", () => {
    expect(downloadAvailabilityNotice(availability, false, "macos-arm64", "unavailable")).toEqual({
      title: "Package unavailable",
      detail: "The macOS TAR.GZ package is not currently published.",
    });
  });

  it("shows a single status-check failure and does not infer missing releases", () => {
    expect(downloadAvailabilityNotice(null, true, "windows-x64", "unavailable")).toEqual({
      title: "Could not check package status",
      detail: "Current package availability could not be checked. Review the public releases or try the browser worker.",
    });
  });

  it("distinguishes unsupported RPM from the current Linux archive status", () => {
    expect(downloadAvailabilityNotice(availability, false, "linux-x64", "unsupported")).toEqual({
      title: "Package format unavailable",
      detail: "No Fedora / RHEL RPM package is offered. Check whether the Linux x64 archive is currently published, or try the browser worker.",
    });
    expect(downloadAvailabilityNotice(null, true, "linux-x64", "unsupported")?.title).toBe("Could not check package status");
  });

  it("rejects a status response that duplicates a platform", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({
      ...availability,
      packages: [availability.packages[0], availability.packages[0], availability.packages[2]],
    })));
    await expect(fetchPublicDownloadAvailability()).rejects.toThrow("download_availability_invalid");
  });

  it("bounds the status request so Downloads can show its fallback", async () => {
    const request = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      throw new DOMException("Timed out", "TimeoutError");
    });
    vi.stubGlobal("fetch", request);
    await expect(fetchPublicDownloadAvailability()).rejects.toMatchObject({ name: "TimeoutError" });
    expect(request).toHaveBeenCalledOnce();
  });

  it("cancels the status request when the page leaves", async () => {
    const controller = new AbortController();
    const request = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    }));
    vi.stubGlobal("fetch", request);
    const pending = fetchPublicDownloadAvailability(controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(request.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
  });
});

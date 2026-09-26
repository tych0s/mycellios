import { useEffect, useState, type ReactNode } from "react";
import { ArrowRight, Cpu, Download, ExternalLink, Laptop } from "lucide-react";
import type { PublicDownloadAvailability } from "../../src/contracts/public-downloads.js";
import { fetchPublicDownloadAvailability, PUBLIC_DOWNLOAD_OPTIONS } from "./public-downloads";

const RELEASES_URL = "https://github.com/tych0s/mycellios/releases";

export function downloadAvailabilityNotice(
  availability: PublicDownloadAvailability | null,
  availabilityFailed: boolean,
  requestedPlatform: string | null,
  availabilityReason: string | null,
): { title: string; detail: string } | null {
  if (availabilityFailed) return {
    title: "Could not check package status",
    detail: availabilityReason === "unsupported"
      ? "No Fedora / RHEL RPM package is offered. Current archive availability could not be checked; review the public releases or try the browser worker."
      : "Current package availability could not be checked. Review the public releases or try the browser worker.",
  };
  if (availabilityReason === "unsupported") return {
    title: "Package format unavailable",
    detail: "No Fedora / RHEL RPM package is offered. Check whether the Linux x64 archive is currently published, or try the browser worker.",
  };
  if (!availability) return null;
  const requestedOption = PUBLIC_DOWNLOAD_OPTIONS.find((option) => option.id === requestedPlatform);
  if (requestedOption) {
    if (availability.packages.find((item) => item.id === requestedOption.id)?.available) return null;
    return {
      title: "Package unavailable",
      detail: `The ${requestedOption.label} ${requestedOption.format} package is not currently published.`,
    };
  }
  if (availability.packages.some((item) => item.available)) return null;
  return {
    title: "Packages unavailable",
    detail: "No native packages are currently published. Try the browser worker or check the public releases.",
  };
}

export function DownloadsPage({
  title,
  publicLink,
  external,
}: {
  title: ReactNode;
  publicLink: (path: string) => string;
  external: boolean;
}) {
  const [availability, setAvailability] = useState<PublicDownloadAvailability | null>(null);
  const [availabilityFailed, setAvailabilityFailed] = useState(false);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    void fetchPublicDownloadAvailability(controller.signal).then((result) => {
      if (active) setAvailability(result);
    }).catch(() => {
      if (active) setAvailabilityFailed(true);
    });
    return () => { active = false; controller.abort(); };
  }, []);

  const params = new URLSearchParams(window.location.search);
  const requestedPlatform = params.get("platform");
  const notice = downloadAvailabilityNotice(availability, availabilityFailed, requestedPlatform, params.get("availability"));

  return <section>
    {title}
    {notice && <div className="download-availability-note" role="status"><strong>{notice.title}</strong><p>{notice.detail}</p><a href={RELEASES_URL} target="_blank" rel="noreferrer">View published releases <ExternalLink /></a><a href="/browser/">Try the browser worker <ArrowRight /></a></div>}
    <div className="download-grid">{PUBLIC_DOWNLOAD_OPTIONS.map((option) => {
      const pkg = availability?.packages.find((item) => item.id === option.id);
      const available = pkg?.available === true;
      const copy = availabilityFailed
        ? "Availability could not be confirmed. Check the public release page."
        : available && availability
      ? `Version ${availability.version} · published ${option.format} archive.`
          : availability
            ? `No ${option.format} archive is published for the current version.`
            : "Checking published release files…";
      return <article className={`download-card${available ? " ready" : " unavailable"}`} key={option.id}>
        <span>{option.id === "windows-x64" ? <Laptop /> : <Cpu />}{option.label.toUpperCase()} · {option.format}</span>
        <h2>{option.detail}</h2>
        <p>{copy}</p>
        {available && pkg
          ? <a className="download-card-action" href={publicLink(pkg.path)} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>Download {option.format} <Download /></a>
          : availabilityFailed
            ? <a className="download-card-action" href={RELEASES_URL} target="_blank" rel="noreferrer">Check releases <ExternalLink /></a>
            : <span className="download-card-action">{availability ? "Not published" : "Checking…"}</span>}
      </article>;
    })}</div>
  </section>;
}

import { useEffect, useState, type ReactNode } from "react";
import { ArrowRight, Cpu, Download, ExternalLink, Laptop } from "lucide-react";
import type { PublicDownloadAvailability } from "../../src/contracts/public-downloads.js";
import { fetchPublicDownloadAvailability, PUBLIC_DOWNLOAD_OPTIONS } from "./public-downloads";

const RELEASES_URL = "https://github.com/tych0s/mycellios/releases";

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
    void fetchPublicDownloadAvailability().then((result) => {
      if (active) setAvailability(result);
    }).catch(() => {
      if (active) setAvailabilityFailed(true);
    });
    return () => { active = false; };
  }, []);

  const params = new URLSearchParams(window.location.search);
  const requestedPlatform = params.get("platform");
  const requestedOption = PUBLIC_DOWNLOAD_OPTIONS.find((option) => option.id === requestedPlatform);
  const availabilityReason = params.get("availability");
  const hasAvailablePackage = availability?.packages.some((item) => item.available) === true;
  const showUnavailableNotice = availabilityReason === "unavailable" || availabilityReason === "unsupported"
    || (availability && !hasAvailablePackage);
  const notice = availabilityReason === "unsupported"
    ? "There is no Fedora / RHEL RPM package. The published Linux archive supports x64 systems."
    : requestedOption
      ? `The ${requestedOption.label} ${requestedOption.format} package is not currently published.`
      : "No native packages are currently published. Try the browser worker or check the public releases.";

  return <section>
    {title}
    {showUnavailableNotice && <div className="download-availability-note" role="status"><strong>Package unavailable</strong><p>{notice}</p><a href={RELEASES_URL} target="_blank" rel="noreferrer">View published releases <ExternalLink /></a><a href="/browser/">Try the browser worker <ArrowRight /></a></div>}
    {availabilityFailed && <div className="download-availability-note" role="status"><strong>Could not check package status</strong><p>Review the public releases or try the browser worker.</p><a href={RELEASES_URL} target="_blank" rel="noreferrer">View published releases <ExternalLink /></a></div>}
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
          : <span className="download-card-action">{availabilityFailed ? "Check releases" : availability ? "Not published" : "Checking…"}</span>}
      </article>;
    })}</div>
  </section>;
}

/// <reference lib="dom" />

const SITE_ORIGIN = "https://www.mycellios.com";
const BRAND_IMAGE_URL = `${SITE_ORIGIN}/assets/brand/og-image.png`;
const BRAND_LOGO_URL = `${SITE_ORIGIN}/assets/brand/app-icon.png`;
const ORGANIZATION_ID = `${SITE_ORIGIN}/#organization`;
const WEBSITE_ID = `${SITE_ORIGIN}/#website`;
const SOFTWARE_ID = `${SITE_ORIGIN}/#software`;

type JsonValue =
  | boolean
  | number
  | string
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

interface SeoPage {
  title: string;
  description: string;
  canonicalPath: string;
  robots: "index, follow, max-image-preview:large" | "noindex, nofollow, noarchive";
  structuredData?: JsonValue;
  staticContent?: {
    heading: string;
    paragraphs: readonly string[];
    links: readonly { href: string; label: string }[];
  };
}

const organization = {
  "@type": "Organization",
  "@id": ORGANIZATION_ID,
  name: "mycellios",
  url: `${SITE_ORIGIN}/`,
  slogan: "Distributed intelligence, rooted in nature.",
  logo: {
    "@type": "ImageObject",
    url: BRAND_LOGO_URL,
  },
  email: "hello@mycellios.com",
};

const website = {
  "@type": "WebSite",
  "@id": WEBSITE_ID,
  url: `${SITE_ORIGIN}/`,
  name: "mycellios",
  publisher: { "@id": ORGANIZATION_ID },
};

const softwareApplication = {
  "@type": "SoftwareApplication",
  "@id": SOFTWARE_ID,
  name: "mycellios",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Windows 10/11, macOS (Apple Silicon), Linux (x64)",
  description:
    "Open-source software coordinating AI inference across model-stage workers. Multi-host GPU performance and combined-memory execution remain under development.",
  url: `${SITE_ORIGIN}/`,
  publisher: { "@id": ORGANIZATION_ID },
};

function webPage(name: string, description: string, path: string): JsonValue {
  return {
    "@context": "https://schema.org",
    "@type": "WebPage",
    name,
    description,
    url: `${SITE_ORIGIN}${path}`,
    isPartOf: { "@id": WEBSITE_ID },
    about: { "@id": SOFTWARE_ID },
  };
}

export const SEO_PAGES = {
  "/": {
    title: "mycellios — The open compute layer for distributed AI",
    description:
      "Mycellios develops open-source distributed AI inference. Explore model-stage coordination, public network data, browser workers, and current evidence.",
    canonicalPath: "/",
    robots: "index, follow, max-image-preview:large",
    staticContent: {
      heading: "Distributed AI inference with mycellios",
      paragraphs: [
        "mycellios is an open-source project building a distributed AI inference runtime. Its coordinator plans model stages across worker computers and streams responses through a familiar API.",
        "The project is under active development. Multi-host GPU performance, models that need combined host memory, and recovery after a worker fails have not been demonstrated as general production capabilities.",
      ],
      links: [
        { href: "/network", label: "Inspect the public network snapshot" },
        { href: "/earn", label: "Learn about contributing to the test network" },
        { href: "https://github.com/tych0s/mycellios", label: "View the open-source project" },
      ],
    },
    structuredData: {
      "@context": "https://schema.org",
      "@graph": [organization, website, softwareApplication],
    },
  },
  "/network": {
    title: "Live distributed AI network | mycellios",
    description:
      "Inspect the public mycellios network snapshot: connected workers, available models, and recorded inference activity when data is available.",
    canonicalPath: "/network",
    robots: "index, follow, max-image-preview:large",
    staticContent: {
      heading: "Mycellios public network status",
      paragraphs: [
        "This page reads a public snapshot from the Mycellios coordinator and reports registered or connected workers, available models, and recorded activity when that data is available.",
        "Values are observed or derived from the snapshot. Revenue, rewards, and throughput remain unpublished unless an auditable public source is available.",
      ],
      links: [
        { href: "/", label: "About the Mycellios project" },
        { href: "/blog", label: "Read Mycellios research and field notes" },
      ],
    },
    structuredData: webPage(
      "Live distributed AI network | mycellios",
      "A public snapshot of Mycellios workers, models, and recorded inference activity when data is available.",
      "/network",
    ),
  },
  "/create": {
    title: "Design an AI identity | Mycellios Studio",
    description:
      "Design an AI identity with personality, memory, knowledge, tools, and launch channels. Preview a local draft and sign in to publish.",
    canonicalPath: "/create",
    robots: "index, follow, max-image-preview:large",
    staticContent: {
      heading: "Design an AI identity",
      paragraphs: [
        "Draft an AI identity by choosing its personality, memory policy, planned knowledge sources, tools, and launch channel.",
        "Drafts stay in this browser. Source labels do not connect content, and tools remain controlled; publishing requires an account and an explicitly configured connection.",
      ],
      links: [
        { href: "/", label: "Explore the Mycellios project" },
        { href: "/network", label: "Inspect the public network snapshot" },
      ],
    },
    structuredData: webPage(
      "Design an AI identity | Mycellios Studio",
      "A visual builder for drafting and previewing AI identities before publishing to web, Telegram, or API channels.",
      "/create",
    ),
  },
  "/earn": {
    title: "Contribute compute to the Mycellios test network",
    description:
      "Join the Mycellios test network with a compatible native client or browser worker. Contributions are voluntary, and rewards are not live.",
    canonicalPath: "/earn",
    robots: "index, follow, max-image-preview:large",
    staticContent: {
      heading: "Contribute compute to the Mycellios test network",
      paragraphs: [
        "Join the Mycellios test network with a compatible browser worker or native client. You choose what your device shares and can pause it at any time.",
        "Rewards are not live, and Mycellios does not currently promise USDC or token payments. Contributions are measured for reliability and useful work.",
      ],
      links: [
        { href: "/browser/", label: "Try the browser worker" },
        { href: "/downloads", label: "Check native package availability" },
      ],
    },
    structuredData: webPage(
      "Contribute compute to the Mycellios test network",
      "Join the Mycellios test network with a compatible native client or browser worker. Contributions are voluntary, and rewards are not live.",
      "/earn",
    ),
  },
  "/spore": {
    title: "$SPORE participation preview | mycellios",
    description:
      "Preview the proposed SPORE participation flow. No token, staking contract, treasury, price feed or reward program is live.",
    canonicalPath: "/spore",
    robots: "noindex, nofollow, noarchive",
    staticContent: {
      heading: "SPORE participation preview",
      paragraphs: ["This is a preview only. No token, staking contract, treasury, price feed, or reward program is live."],
      links: [{ href: "/", label: "Return to Mycellios" }],
    },
  },
  "/spore/treasury": {
    title: "SPORE treasury preview | mycellios",
    description:
      "Preview the proposed SPORE treasury model. No treasury contract or reward distributions are live.",
    canonicalPath: "/spore/treasury",
    robots: "noindex, nofollow, noarchive",
    staticContent: {
      heading: "SPORE treasury preview",
      paragraphs: ["This is a proposed model preview. No treasury contract or reward distributions are live."],
      links: [{ href: "/", label: "Return to Mycellios" }],
    },
  },
  "/spore/data": {
    title: "Network data preview | mycellios",
    description:
      "Preview the planned mycellios network telemetry. No live feed or token price data is available.",
    canonicalPath: "/spore/data",
    robots: "noindex, nofollow, noarchive",
    staticContent: {
      heading: "Network data preview",
      paragraphs: ["Planned Mycellios network telemetry is not a live feed, and token price data is not available."],
      links: [{ href: "/network", label: "View the public network snapshot" }],
    },
  },
  "/downloads": {
    title: "Native package availability | mycellios",
    description:
      "Check which mycellios native packages are published for Windows, macOS, and Linux, or contribute from your browser without installing software.",
    canonicalPath: "/downloads",
    robots: "index, follow, max-image-preview:large",
    staticContent: {
      heading: "Mycellios native package availability",
      paragraphs: [
        "Check published native releases for Windows, macOS, and Linux. Download links appear only for packages found in the public release store.",
        "If a package is unavailable, review the public releases or try the browser worker on a compatible device without installing a native client.",
      ],
      links: [
        { href: "/browser/", label: "Try the browser worker" },
        { href: "https://github.com/tych0s/mycellios/releases", label: "View published releases" },
      ],
    },
    structuredData: {
      "@context": "https://schema.org",
      ...softwareApplication,
      url: `${SITE_ORIGIN}/downloads`,
    },
  },
  "/browser/": {
    title: "Browser worker for desktop, tablet, and phone | mycellios",
    description:
      "Connect a desktop, tablet, or phone browser to mycellios and contribute voluntary, measurable compute that you can pause anytime.",
    canonicalPath: "/browser/",
    robots: "index, follow, max-image-preview:large",
    staticContent: {
      heading: "Connect a browser to Mycellios",
      paragraphs: [
        "An opt-in worker lets compatible desktop, tablet, or phone browsers contribute measurable compute to the Mycellios test network. You can pause it at any time.",
        "Available tasks depend on browser capabilities and network state. Network rewards are not live, and no payment is promised for browser contribution.",
      ],
      links: [
        { href: "/earn", label: "Learn about contributing compute" },
        { href: "/network", label: "Inspect the public network snapshot" },
      ],
    },
    structuredData: {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      name: "mycellios browser worker",
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Any compatible desktop, tablet, or phone browser",
      description:
        "A universal browser worker for voluntarily contributing compatible compute to the mycellios network.",
      url: `${SITE_ORIGIN}/browser/`,
      publisher: { "@id": ORGANIZATION_ID },
    },
  },
  "/admin": {
    title: "Network administration | mycellios",
    description: "Operational administration for the mycellios network.",
    canonicalPath: "/admin",
    robots: "noindex, nofollow, noarchive",
    staticContent: {
      heading: "Mycellios network administration",
      paragraphs: ["Operational administration for the Mycellios network."],
      links: [{ href: "/", label: "Return to Mycellios" }],
    },
  },
  "/dashboard": {
    title: "Dashboard | mycellios",
    description: "Sign in to manage your Mycellios account and network workspace.",
    canonicalPath: "/dashboard",
    robots: "noindex, nofollow, noarchive",
    staticContent: {
      heading: "Mycellios dashboard",
      paragraphs: ["Sign in to manage your Mycellios account and network workspace."],
      links: [{ href: "/", label: "Return to Mycellios" }],
    },
  },
  "/account": {
    title: "Account and subscription | mycellios",
    description: "Secure account, subscription, billing and usage management for mycellios.",
    canonicalPath: "/account",
    robots: "noindex, nofollow, noarchive",
    staticContent: {
      heading: "Mycellios account and subscription",
      paragraphs: ["Manage your Mycellios account, subscription, billing, and usage."],
      links: [{ href: "/", label: "Return to Mycellios" }],
    },
  },
} as const satisfies Record<string, SeoPage>;

export type SeoPath = keyof typeof SEO_PAGES;

export const INDEXABLE_SEO_PATHS = ["/", "/network", "/create", "/earn", "/downloads", "/browser/"] as const;
export const LANDING_SEO_PATHS = [
  "/",
  "/network",
  "/create",
  "/earn",
  "/spore",
  "/spore/treasury",
  "/spore/data",
  "/downloads",
  "/admin",
  "/dashboard",
  "/account",
] as const;

export function canonicalUrl(page: SeoPage): string {
  return `${SITE_ORIGIN}${page.canonicalPath}`;
}

export function seoPageForPath(pathname: string): SeoPage {
  if (pathname === "/browser" || pathname.startsWith("/browser/") || pathname === "/mobile" || pathname.startsWith("/mobile/")) return SEO_PAGES["/browser/"];
  if (pathname in SEO_PAGES) return SEO_PAGES[pathname as SeoPath];
  return SEO_PAGES["/"];
}

function upsertMeta(attribute: "name" | "property", key: string, content: string): void {
  let element = document.head.querySelector<HTMLMetaElement>(`meta[${attribute}="${key}"]`);
  if (!element) {
    element = document.createElement("meta");
    element.setAttribute(attribute, key);
    document.head.append(element);
  }
  element.content = content;
}

function upsertCanonical(href: string): void {
  let element = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!element) {
    element = document.createElement("link");
    element.rel = "canonical";
    document.head.append(element);
  }
  element.href = href;
}

function upsertStructuredData(data: JsonValue | undefined): void {
  const existing = document.head.querySelector<HTMLScriptElement>("#seo-structured-data");
  if (!data) {
    existing?.remove();
    return;
  }
  const element = existing ?? document.createElement("script");
  element.id = "seo-structured-data";
  element.type = "application/ld+json";
  element.textContent = JSON.stringify(data);
  if (!existing) document.head.append(element);
}

export function applySeoMetadata(pathname = window.location.pathname): void {
  const page = seoPageForPath(pathname);
  const url = canonicalUrl(page);

  document.documentElement.lang = "en";
  document.title = page.title;
  upsertMeta("name", "description", page.description);
  upsertMeta("name", "robots", page.robots);
  upsertMeta("property", "og:type", "website");
  upsertMeta("property", "og:site_name", "mycellios");
  upsertMeta("property", "og:locale", "en_US");
  upsertMeta("property", "og:url", url);
  upsertMeta("property", "og:title", page.title);
  upsertMeta("property", "og:description", page.description);
  upsertMeta("property", "og:image", BRAND_IMAGE_URL);
  upsertMeta("property", "og:image:alt", "mycellios — distributed intelligence, rooted in nature");
  upsertMeta("name", "twitter:card", "summary_large_image");
  upsertMeta("name", "twitter:title", page.title);
  upsertMeta("name", "twitter:description", page.description);
  upsertMeta("name", "twitter:image", BRAND_IMAGE_URL);
  upsertCanonical(url);
  upsertStructuredData(page.structuredData);
}

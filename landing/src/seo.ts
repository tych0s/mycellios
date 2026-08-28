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
  operatingSystem: "Windows 10/11, macOS, Ubuntu, Debian, Fedora, RHEL",
  description:
    "A distributed intelligence network that connects capacity across different machines to run AI models that cannot fit on a single device.",
  url: `${SITE_ORIGIN}/`,
  downloadUrl: `${SITE_ORIGIN}/downloads`,
  publisher: { "@id": ORGANIZATION_ID },
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
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
      "mycellios connects capacity across different machines to run AI models that cannot fit on a single device.",
    canonicalPath: "/",
    robots: "index, follow, max-image-preview:large",
    structuredData: {
      "@context": "https://schema.org",
      "@graph": [organization, website, softwareApplication],
    },
  },
  "/network": {
    title: "Live distributed AI network | mycellios",
    description:
      "Inspect the live mycellios network: connected compute nodes, shared memory, available models, verified tasks, and current distributed capacity.",
    canonicalPath: "/network",
    robots: "index, follow, max-image-preview:large",
    structuredData: webPage(
      "Live distributed AI network | mycellios",
      "Public status and capacity for the live mycellios distributed AI network.",
      "/network",
    ),
  },
  "/create": {
    title: "Build persistent AI identities | Mycellios Studio",
    description:
      "Create, configure, and test a persistent AI identity with personality, memory, trusted knowledge, tools, and launch channels in Mycellios Studio.",
    canonicalPath: "/create",
    robots: "index, follow, max-image-preview:large",
    structuredData: webPage(
      "Build persistent AI identities | Mycellios Studio",
      "A visual builder for creating, testing, and preparing persistent AI identities for web, Telegram, and API channels.",
      "/create",
    ),
  },
  "/join": {
    title: "Join a distributed AI network | mycellios",
    description:
      "Connect a computer to an existing mycellios network or create a private network that pools memory and compute across your own machines.",
    canonicalPath: "/join",
    robots: "index, follow, max-image-preview:large",
    structuredData: webPage(
      "Join a distributed AI network | mycellios",
      "Connect a machine to mycellios or create a private distributed AI network.",
      "/join",
    ),
  },
  "/downloads": {
    title: "Download mycellios for Windows, macOS, and Linux",
    description:
      "Download the mycellios desktop app for Windows, macOS, Ubuntu, Debian, Fedora, or RHEL and connect your machine to a distributed AI network.",
    canonicalPath: "/downloads",
    robots: "index, follow, max-image-preview:large",
    structuredData: {
      "@context": "https://schema.org",
      ...softwareApplication,
      url: `${SITE_ORIGIN}/downloads`,
    },
  },
  "/mobile/": {
    title: "Contribute mobile compute | mycellios",
    description:
      "Connect a compatible mobile browser to the mycellios public test network and contribute voluntary, measurable compute that you can pause anytime.",
    canonicalPath: "/mobile/",
    robots: "index, follow, max-image-preview:large",
    structuredData: {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      name: "mycellios mobile compute cell",
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Any compatible mobile browser",
      description:
        "A browser-based compute cell for voluntarily contributing compatible mobile hardware to the mycellios network.",
      url: `${SITE_ORIGIN}/mobile/`,
      publisher: { "@id": ORGANIZATION_ID },
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
      },
    },
  },
  "/admin": {
    title: "Network administration | mycellios",
    description: "Operational administration for the mycellios network.",
    canonicalPath: "/admin",
    robots: "noindex, nofollow, noarchive",
  },
  "/account": {
    title: "Account and subscription | mycellios",
    description: "Secure account, subscription, billing and usage management for mycellios.",
    canonicalPath: "/account",
    robots: "noindex, nofollow, noarchive",
  },
} as const satisfies Record<string, SeoPage>;

export type SeoPath = keyof typeof SEO_PAGES;

export const INDEXABLE_SEO_PATHS = ["/", "/network", "/create", "/join", "/downloads", "/mobile/"] as const;
export const LANDING_SEO_PATHS = ["/", "/network", "/create", "/join", "/downloads", "/admin"] as const;

export function canonicalUrl(page: SeoPage): string {
  return `${SITE_ORIGIN}${page.canonicalPath}`;
}

export function seoPageForPath(pathname: string): SeoPage {
  if (pathname === "/mobile" || pathname.startsWith("/mobile/")) return SEO_PAGES["/mobile/"];
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

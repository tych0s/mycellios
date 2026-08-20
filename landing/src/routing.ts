export type LandingSurface = "landing" | "network" | "panel";

export type PanelView =
  | "overview"
  | "nodes"
  | "network"
  | "models"
  | "jobs"
  | "inference"
  | "tests"
  | "downloads"
  | "contribute"
  | "logs"
  | "admin";

export type PanelMode = "simple" | "developer";
export type PanelRole = "public" | "viewer" | "operator" | "admin" | "owner";
export type PanelWidth = "compact" | "standard" | "wide";

export interface PanelRoute {
  id: PanelView;
  label: string;
  width: PanelWidth;
  modes: readonly PanelMode[];
  roles: readonly PanelRole[];
  primaryAction: string;
}

const EVERY_ROLE: readonly PanelRole[] = ["public", "viewer", "operator", "admin", "owner"];
const OPERATIONS_ROLES: readonly PanelRole[] = ["operator", "admin", "owner"];
const ADMIN_ROLES: readonly PanelRole[] = ["admin", "owner"];

export const PANEL_ROUTES: readonly PanelRoute[] = [
  { id: "overview", label: "Overview", width: "standard", modes: ["simple", "developer"], roles: EVERY_ROLE, primaryAction: "Run inference" },
  { id: "nodes", label: "Nodes", width: "wide", modes: ["developer"], roles: EVERY_ROLE, primaryAction: "Connect node" },
  { id: "network", label: "Network", width: "wide", modes: ["simple", "developer"], roles: EVERY_ROLE, primaryAction: "Change range" },
  { id: "models", label: "Models", width: "wide", modes: ["developer"], roles: EVERY_ROLE, primaryAction: "Request model" },
  { id: "jobs", label: "Jobs", width: "wide", modes: ["developer"], roles: EVERY_ROLE, primaryAction: "Inspect job" },
  { id: "inference", label: "Inference", width: "standard", modes: ["simple", "developer"], roles: EVERY_ROLE, primaryAction: "Send prompt" },
  { id: "tests", label: "Tests", width: "wide", modes: ["developer"], roles: EVERY_ROLE, primaryAction: "Run benchmark" },
  { id: "downloads", label: "Downloads", width: "standard", modes: ["simple", "developer"], roles: EVERY_ROLE, primaryAction: "Download node" },
  { id: "contribute", label: "Run", width: "standard", modes: ["simple", "developer"], roles: EVERY_ROLE, primaryAction: "Connect device" },
  { id: "logs", label: "Logs", width: "wide", modes: ["developer"], roles: OPERATIONS_ROLES, primaryAction: "Export logs" },
  { id: "admin", label: "Admin", width: "wide", modes: ["developer"], roles: ADMIN_ROLES, primaryAction: "Save configuration" },
];

const PANEL_ROUTE_BY_ID = new Map(PANEL_ROUTES.map((route) => [route.id, route]));
const LEGACY_PANEL_VIEWS: Readonly<Record<string, PanelView>> = { history: "network", tasks: "jobs", chat: "inference", join: "contribute" };

export function normalizePanelView(value: string | null): PanelView | null {
  if (!value) return null;
  const normalized = LEGACY_PANEL_VIEWS[value] ?? value;
  return PANEL_ROUTE_BY_ID.has(normalized as PanelView) ? normalized as PanelView : null;
}

export function canInspectOperationalEvidence(role: PanelRole, accessToken: string | null | undefined): boolean {
  return Boolean(accessToken && (role === "operator" || role === "admin" || role === "owner"));
}

export function visiblePanelRoutes(mode: PanelMode, role: PanelRole): PanelRoute[] {
  return PANEL_ROUTES.filter((route) => route.modes.includes(mode) && route.roles.includes(role));
}

export function panelRoute(view: PanelView): PanelRoute {
  const route = PANEL_ROUTE_BY_ID.get(view);
  if (!route) throw new Error(`Unknown panel view: ${view}`);
  return route;
}

export function panelViewFromLocation(pathname: string, search: string, mobile = false): PanelView {
  const requested = normalizePanelView(new URLSearchParams(search).get("view"));
  if (requested) return requested;
  if (mobile || pathname.startsWith("/browser") || pathname.startsWith("/mobile")) return "contribute";
  if (pathname === "/join") return "contribute";
  if (pathname === "/downloads") return "downloads";
  if (pathname === "/admin") return "admin";
  return "overview";
}

export function panelLocation(view: PanelView, mobile = false, dashboard = false): string {
  if (mobile) return view === "contribute" ? "/browser/" : `/browser/?view=${view}`;
  if (dashboard) return view === "overview" ? "/dashboard" : `/dashboard?view=${view}`;
  if (view === "downloads") return "/downloads";
  if (view === "admin") return "/admin";
  return view === "overview" ? "/network?view=overview" : `/network?view=${view}`;
}

const PANEL_PATHS = new Set(["/network", "/dashboard", "/admin", "/downloads", "/account", "/browser", "/mobile"]);

export function resolveLandingSurface(pathname: string, search: string): LandingSurface {
  if (pathname === "/network" && !new URLSearchParams(search).has("view")) return "network";
  if (PANEL_PATHS.has(pathname)) return "panel";
  return "landing";
}

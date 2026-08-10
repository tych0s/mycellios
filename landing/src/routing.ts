export type LandingSurface = "landing" | "network" | "panel";

const PANEL_PATHS = new Set(["/network", "/admin", "/join", "/downloads"]);

export function resolveLandingSurface(pathname: string, search: string): LandingSurface {
  if (pathname === "/network" && !new URLSearchParams(search).has("view")) return "network";
  if (PANEL_PATHS.has(pathname)) return "panel";
  return "landing";
}



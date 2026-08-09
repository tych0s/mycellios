import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Panel } from "./Panel";
import { RebrandLanding } from "./rebrand/RebrandLanding";
import { CreateStudio } from "./rebrand/CreateStudio";
import { applySeoMetadata } from "./seo";

const root = document.getElementById("root");
if (!root) throw new Error("Missing landing root element");

const panelRoutes = new Set(["/network", "/admin", "/join", "/downloads"]);
// Phase 2 (merged landing): `/rebrand` no longer serves a separate concept — the
// rebrand design is the landing. Redirect to `/` before applying SEO metadata so
// the canonical tags describe the merged page.
const isRebrandRoute = window.location.pathname === "/rebrand" || window.location.pathname === "/rebrand/";
if (isRebrandRoute) window.history.replaceState({}, "", "/");
applySeoMetadata();

createRoot(root).render(
  <StrictMode>
    {window.location.pathname === "/create" ? <CreateStudio /> : panelRoutes.has(window.location.pathname) ? <Panel /> : <RebrandLanding />}
  </StrictMode>,
);

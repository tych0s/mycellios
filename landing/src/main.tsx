import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Panel } from "./Panel";
import { NetworkPage } from "./NetworkPage";
import { RebrandLanding } from "./rebrand/RebrandLanding";
import { CreateStudio } from "./rebrand/CreateStudio";
import { EarnStudio } from "./rebrand/EarnStudio";
import { SporeStudio } from "./rebrand/SporeStudio";
import { applySeoMetadata } from "./seo";
import { resolveLandingSurface } from "./routing";

const root = document.getElementById("root");
if (!root) throw new Error("Missing landing root element");

// Phase 2 (merged landing): `/rebrand` no longer serves a separate concept — the
// rebrand design is the landing. Redirect to `/` before applying SEO metadata so
// the canonical tags describe the merged page.
const isRebrandRoute = window.location.pathname === "/rebrand" || window.location.pathname === "/rebrand/";
if (isRebrandRoute) window.history.replaceState({}, "", "/");
applySeoMetadata();

const surface = resolveLandingSurface(window.location.pathname, window.location.search);

createRoot(root).render(
  <StrictMode>
    {window.location.pathname === "/create" ? <CreateStudio /> : window.location.pathname === "/earn" ? <EarnStudio /> : window.location.pathname.startsWith("/spore") ? <SporeStudio /> : surface === "network" ? <NetworkPage /> : surface === "panel" ? <Panel /> : <RebrandLanding />}
  </StrictMode>,
);

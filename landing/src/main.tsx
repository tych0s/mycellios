import { lazy, StrictMode, Suspense, useEffect, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { loadMushroomStage, wantsMushroom } from "./rebrand/HeroMushroom";
import { applySeoMetadata } from "./seo";
import { resolveLandingSurface } from "./routing";

const Panel = lazy(() => import("./Panel").then((module) => ({ default: module.Panel })));
const NetworkPage = lazy(() => import("./NetworkPage").then((module) => ({ default: module.NetworkPage })));
const RebrandLanding = lazy(() => import("./rebrand/RebrandLanding").then((module) => ({ default: module.RebrandLanding })));
const CreateStudio = lazy(() => import("./rebrand/CreateStudio").then((module) => ({ default: module.CreateStudio })));
const EarnStudio = lazy(() => import("./rebrand/EarnStudio").then((module) => ({ default: module.EarnStudio })));
const SporeStudio = lazy(() => import("./rebrand/SporeStudio").then((module) => ({ default: module.SporeStudio })));

function LoadedRoute({ children }: { children: ReactNode }) {
  useEffect(() => {
    const loader = document.getElementById("mycellios-loader");
    if (!loader) return;
    let timer: number | undefined;
    const removeLoader = () => loader.remove();
    let frame = requestAnimationFrame(() => { frame = requestAnimationFrame(() => {
      loader.dataset.state = "leaving";
      loader.addEventListener("transitionend", removeLoader, { once: true });
      timer = window.setTimeout(removeLoader, 400);
    }); });
    return () => {
      cancelAnimationFrame(frame);
      if (timer !== undefined) window.clearTimeout(timer);
      loader.removeEventListener("transitionend", removeLoader);
    };
  }, []);
  return children;
}

const root = document.getElementById("root");
if (!root) throw new Error("Missing landing root element");

// Phase 2 (merged landing): `/rebrand` no longer serves a separate concept — the
// rebrand design is the landing. Redirect to `/` before applying SEO metadata so
// the canonical tags describe the merged page.
const isRebrandRoute = window.location.pathname === "/rebrand" || window.location.pathname === "/rebrand/";
if (isRebrandRoute) window.history.replaceState({}, "", "/");
if (window.location.pathname === "/join" || window.location.pathname === "/join/") {
  window.history.replaceState({}, "", "/earn");
}
applySeoMetadata();

const path = window.location.pathname.length > 1
  ? window.location.pathname.replace(/\/+$/, "")
  : window.location.pathname;
const surface = resolveLandingSurface(path, window.location.search);

const isCreate = path === "/create";
const isEarn = path === "/earn";
const isSpore = path.startsWith("/spore");
const isNetwork = surface === "network";
const isPanel = surface === "panel";
const isLanding = !isCreate && !isEarn && !isSpore && !isNetwork && !isPanel;

const page = isCreate ? <CreateStudio />
  : isEarn ? <EarnStudio />
  : isSpore ? <SporeStudio />
  : isNetwork ? <NetworkPage />
  : isPanel ? <Panel accountEntry={path === "/account"} mobileEntry={path === "/browser" || path === "/mobile"} />
  : <RebrandLanding />;

/*
 * Start the hero renderer's download here rather than letting the component ask
 * for it after mount.
 *
 * The mushroom chunk is the largest asset on the page, and a dynamic import
 * inside an effect cannot begin until this bundle has been fetched, parsed and
 * rendered — so the two biggest downloads ran strictly back to back and the
 * organism landed a second or two after the text it belongs with. Requesting it
 * now overlaps them: same chunk, same lazy boundary, no bytes added to this
 * bundle, but the network is busy with it while React is still mounting.
 *
 * The condition is derived from the same mutually exclusive flags that select
 * the lazy route. A separate paraphrase of the router would eventually make a
 * phone or studio route download 190KB it never draws. `wantsMushroom()` then
 * applies the same reduced-motion and width tests the component uses.
 */
if (isLanding && wantsMushroom()) {
  void loadMushroomStage().catch(() => {
    /* The component awaits the same shared promise and reports there. */
  });
}

// Vite can evaluate this entry again after an imported module changes. Reuse
// its React root so the previous mounted tree is updated, never duplicated.
const applicationRoot: Root = import.meta.hot?.data.reactRoot ?? createRoot(root);
if (import.meta.hot) import.meta.hot.data.reactRoot = applicationRoot;
applicationRoot.render(
  <StrictMode>
    <Suspense fallback={null}><LoadedRoute>{page}</LoadedRoute></Suspense>
  </StrictMode>,
);

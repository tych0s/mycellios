import { describe, expect, it } from "vitest";
import {
  PANEL_ROUTES,
  canInspectOperationalEvidence,
  normalizePanelView,
  panelLocation,
  panelRoute,
  panelViewFromLocation,
  resolveLandingSurface,
  visiblePanelRoutes,
} from "./routing";

describe("panel information architecture", () => {
  it("requires an authenticated operational role for signed evidence", () => {
    expect(canInspectOperationalEvidence("viewer", "token")).toBe(false);
    expect(canInspectOperationalEvidence("operator", null)).toBe(false);
    expect(canInspectOperationalEvidence("operator", "token")).toBe(true);
    expect(canInspectOperationalEvidence("admin", "token")).toBe(true);
    expect(canInspectOperationalEvidence("owner", "token")).toBe(true);
  });
  it("defines every required product view with one primary action and a stable width", () => {
    const required = ["overview", "nodes", "network", "models", "jobs", "inference", "tests", "downloads"];
    expect(PANEL_ROUTES.filter((route) => required.includes(route.id)).map((route) => route.id)).toEqual(required);
    for (const id of required) {
      expect(panelRoute(id as Parameters<typeof panelRoute>[0]).primaryAction).not.toBe("");
      expect(["compact", "standard", "wide"]).toContain(panelRoute(id as Parameters<typeof panelRoute>[0]).width);
    }
  });

  it("keeps privileged surfaces out of unauthorized navigation", () => {
    const publicViews = visiblePanelRoutes("developer", "public").map((route) => route.id);
    const operatorViews = visiblePanelRoutes("developer", "operator").map((route) => route.id);
    const ownerViews = visiblePanelRoutes("developer", "owner").map((route) => route.id);
    expect(publicViews).not.toContain("logs");
    expect(publicViews).not.toContain("admin");
    expect(operatorViews).toContain("logs");
    expect(operatorViews).not.toContain("admin");
    expect(ownerViews).toContain("admin");
  });

  it("resolves canonical and legacy deep links without losing bare Network", () => {
    expect(panelViewFromLocation("/network", "?view=network")).toBe("network");
    expect(panelViewFromLocation("/network", "?view=history")).toBe("network");
    expect(panelViewFromLocation("/network", "?view=tasks")).toBe("jobs");
    expect(panelViewFromLocation("/dashboard", "?view=join")).toBe("contribute");
    expect(panelViewFromLocation("/join", "")).toBe("contribute");
    expect(normalizePanelView("join")).toBe("contribute");
    expect(normalizePanelView("unknown")).toBeNull();
    expect(panelLocation("overview")).toBe("/network?view=overview");
    expect(panelLocation("overview", false, true)).toBe("/dashboard");
    expect(panelLocation("inference", false, true)).toBe("/dashboard?view=inference");
    expect(normalizePanelView("settings")).toBeNull();
    expect(resolveLandingSurface("/network", "?view=globe")).toBe("network");
    expect(resolveLandingSurface("/network", "?view=overview")).toBe("panel");
  });

  it("routes Stripe returns to the account surface", () => {
    expect(resolveLandingSurface("/account", "?checkout=success")).toBe("panel");
    expect(panelViewFromLocation("/account", "?checkout=success")).toBe("overview");
  });

  it("routes the dashboard login destination to the Panel overview", () => {
    expect(resolveLandingSurface("/dashboard", "")).toBe("panel");
    expect(panelViewFromLocation("/dashboard", "")).toBe("overview");
  });

  it("routes browser worker entries to the contribution panel", () => {
    expect(resolveLandingSurface("/browser", "")).toBe("panel");
    expect(resolveLandingSurface("/mobile", "")).toBe("panel");
    expect(panelViewFromLocation("/browser", "")).toBe("contribute");
    expect(panelViewFromLocation("/mobile", "")).toBe("contribute");
  });

  it("keeps the web contribution route and exposes no retired desktop views", () => {
    expect(visiblePanelRoutes("simple", "owner").map((route) => route.id)).toContain("contribute");
    expect(PANEL_ROUTES.map((route) => route.id)).not.toContain("machine");
    expect(PANEL_ROUTES.map((route) => route.id)).not.toContain("settings");
  });
});

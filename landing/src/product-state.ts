import type { ComponentUpdateState, DesktopUpdateState } from "../../src/contracts/dashboard";

export type ProductStateId =
  | "loading" | "downloading" | "canarying" | "ready" | "paused"
  | "draining" | "failed" | "updating" | "rollback";

export interface ProductStatePresentation {
  id: ProductStateId;
  label: string;
  title: string;
  copy: string;
  nextAction: string;
  tone: "neutral" | "progress" | "success" | "warning" | "danger";
}

const STATES: Record<ProductStateId, ProductStatePresentation> = {
  loading: { id: "loading", label: "Loading", title: "Reading verified state", copy: "Mycellios is loading the latest coordinator evidence.", nextAction: "Keep this view open", tone: "progress" },
  downloading: { id: "downloading", label: "Downloading", title: "Downloading a signed component", copy: "The active runtime remains unchanged until integrity verification passes.", nextAction: "Keep Mycellios running", tone: "progress" },
  canarying: { id: "canarying", label: "Canarying", title: "Testing before activation", copy: "The candidate runtime is isolated and must pass its local canary before promotion.", nextAction: "Wait for the canary result", tone: "progress" },
  ready: { id: "ready", label: "Ready", title: "Verified and ready", copy: "Identity, configuration and runtime evidence are current.", nextAction: "No action required", tone: "success" },
  paused: { id: "paused", label: "Paused", title: "Updates are paused", copy: "The current verified runtime stays active and no candidate is downloaded.", nextAction: "Enable updates when ready", tone: "neutral" },
  draining: { id: "draining", label: "Draining", title: "Finishing accepted work", copy: "No new jobs are accepted while current work reaches a safe boundary.", nextAction: "Wait for active work to finish", tone: "warning" },
  failed: { id: "failed", label: "Failed", title: "The operation stopped safely", copy: "Mycellios did not promote unverified state and preserved the previous safe boundary.", nextAction: "Review the failure and retry", tone: "danger" },
  updating: { id: "updating", label: "Updating", title: "Applying a verified update", copy: "The signed candidate is progressing through validation and activation.", nextAction: "Keep Mycellios running", tone: "progress" },
  rollback: { id: "rollback", label: "Rollback", title: "Restoring the previous runtime", copy: "The candidate was rejected and the last verified component set is being restored.", nextAction: "Wait for health verification", tone: "warning" },
};

export function componentProductState(state: ComponentUpdateState, message = ""): ProductStatePresentation {
  if (/roll(?:ed|ing)?[ -]?back|rollback/i.test(message)) return STATES.rollback;
  switch (state) {
    case "disabled": return STATES.paused;
    case "idle":
    case "up-to-date": return STATES.ready;
    case "checking":
    case "updating": return STATES.updating;
    case "downloading": return STATES.downloading;
    case "waiting-idle": return STATES.draining;
    case "canarying": return STATES.canarying;
    case "activating": return /canary/i.test(message) ? STATES.canarying : STATES.updating;
    case "rollback": return STATES.rollback;
    case "error": return STATES.failed;
  }
}

export function desktopUpdateProductState(state: DesktopUpdateState, message = ""): ProductStatePresentation {
  if (/roll(?:ed|ing)?[ -]?back|rollback/i.test(message)) return STATES.rollback;
  switch (state) {
    case "unsupported":
    case "development":
    case "idle":
    case "up-to-date":
    case "ready": return STATES.ready;
    case "checking": return STATES.updating;
    case "downloading": return STATES.downloading;
    case "error": return STATES.failed;
  }
}

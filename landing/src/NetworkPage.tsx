import { useCallback, useEffect, useId, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { ArrowRight, CheckCircle2, Database, ExternalLink, RefreshCw, ShieldCheck } from "lucide-react";
import { LandingHeader } from "./rebrand/RebrandLanding";
import {
  BORDER_LINES,
  LAND_POINTS,
  buildArc,
  placeWorker,
  rotatePoint,
  sphere,
  visitorLocation,
  type Vec3,
} from "./network-globe";
import "./network-page.css";

export interface NetworkGpu {
  id: string;
  vendor: string;
  model: string;
  offeredVramMb: number;
}

export interface NetworkDeployment {
  deploymentId: string;
  model: string;
  mode: string;
  freeSlots: number;
}

export interface NetworkWorker {
  id: string;
  kind: "desktop" | "browser" | "cell";
  status: "online" | "suspect" | "offline" | "draining";
  connected: boolean;
  region: string;
  offeredVramMb: number;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: string;
  gpus: NetworkGpu[];
  deployments: NetworkDeployment[];
}

export interface NetworkJob {
  id: string;
  model: string;
  status: string;
  workerId: string | null;
  inputTokens: number;
  outputTokens: number;
  updatedAt: string;
}

export interface NetworkSnapshot {
  capturedAt: string;
  version: string;
  summary: {
    registered: number;
    connected: number;
    online: number;
    mobile: number;
    offeredVramMb: number;
    completedJobs: number;
  };
  workers: NetworkWorker[];
  models: Array<{ id: string; replicas: number; pipelines: number }>;
  jobs: NetworkJob[];
}

export type NetworkHistoryRange = "24h" | "7d" | "30d" | "90d";
export type NetworkViewMode = "overview" | "globe";

export interface NetworkTelemetrySample {
  capturedAt: string;
  registeredNodes: number;
  connectedNodes: number;
  onlineNodes: number;
  browserNodes: number;
  activeModels: number;
  modelReplicas: number;
  modelPipelines: number;
  offeredVramMb: number;
  freeVramMb: number;
  inflightJobs: number;
  runningJobs: number;
  completedJobs: number;
}

export interface NetworkTelemetryHistory {
  capturedAt: string;
  intervalMinutes: number;
  retentionDays: number;
  range: NetworkHistoryRange;
  samples: NetworkTelemetrySample[];
}

const EMPTY_SNAPSHOT: NetworkSnapshot = {
  capturedAt: new Date(0).toISOString(),
  version: "—",
  summary: { registered: 0, connected: 0, online: 0, mobile: 0, offeredVramMb: 0, completedJobs: 0 },
  workers: [],
  models: [],
  jobs: [],
};

const SNAPSHOT_COMMAND = "curl -s https://www.mycellios.com/public/v1/snapshot";
const HISTORY_RANGES: Array<{ value: NetworkHistoryRange; label: string }> = [
  { value: "24h", label: "24 hours" },
  { value: "7d", label: "7 days" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
];

type NodeState = "serving" | "standby" | "joining";

export function nodeState(worker: NetworkWorker): NodeState {
  if (!worker.connected) return "joining";
  if (worker.status === "online") return worker.deployments.length > 0 ? "serving" : "standby";
  if (worker.status === "draining" || worker.status === "suspect") return "standby";
  return "joining";
}

export function deriveNetworkMetrics(snapshot: NetworkSnapshot) {
  const connectedWorkers = snapshot.workers.filter((worker) => worker.connected);
  const activeWorkers = connectedWorkers.filter((worker) => worker.status === "online");
  const activeJobs = snapshot.jobs.filter((job) =>
    ["queued", "leasing", "running", "streaming"].includes(job.status));
  const regions = new Set(
    activeWorkers
      .map((worker) => worker.region.trim().toLowerCase())
      .filter((region) => region && region !== "auto"),
  );
  const gpuCount = activeWorkers.reduce(
    (total, worker) => total + (worker.gpus.length || (worker.kind === "browser" ? 1 : 0)),
    0,
  );
  return {
    activeWorkers,
    activeJobs,
    gpuCount,
    regions: regions.size,
    memoryGb: snapshot.summary.offeredVramMb / 1_024,
    availableModels: snapshot.models.length,
    tokensServed: snapshot.jobs.reduce(
      (total, job) => total + job.inputTokens + job.outputTokens,
      0,
    ),
  };
}

export interface DailyJobBucket {
  /** UTC day, ISO date only (YYYY-MM-DD). */
  day: string;
  count: number;
}

/**
 * Buckets completed jobs per UTC day from the persisted cumulative counter:
 * every positive sample-to-sample delta lands in the day of the later sample.
 * Counter resets (negative deltas) are ignored rather than synthesised, and
 * the first sample contributes nothing — there is no earlier baseline.
 */
export function deriveDailyJobs(samples: NetworkTelemetrySample[]): DailyJobBucket[] {
  const ordered = samples
    .filter((sample) => Number.isFinite(Date.parse(sample.capturedAt)))
    .slice()
    .sort((a, b) => Date.parse(a.capturedAt) - Date.parse(b.capturedAt));
  if (ordered.length === 0) return [];
  const deltas = new Map<string, number>();
  for (let index = 1; index < ordered.length; index += 1) {
    const delta = ordered[index]!.completedJobs - ordered[index - 1]!.completedJobs;
    if (delta > 0) {
      const day = ordered[index]!.capturedAt.slice(0, 10);
      deltas.set(day, (deltas.get(day) ?? 0) + delta);
    }
  }
  const start = Date.parse(`${ordered[0]!.capturedAt.slice(0, 10)}T00:00:00Z`);
  const end = Date.parse(`${ordered.at(-1)!.capturedAt.slice(0, 10)}T00:00:00Z`);
  const buckets: DailyJobBucket[] = [];
  for (let time = start; time <= end; time += 86_400_000) {
    const day = new Date(time).toISOString().slice(0, 10);
    buckets.push({ day, count: deltas.get(day) ?? 0 });
  }
  return buckets;
}

interface GlobeNode {
  worker: NetworkWorker;
  vector: Vec3;
  located: boolean;
  country: string | null;
  state: NodeState;
  /** Screen position from the last frame, used for hit-testing. */
  x: number;
  y: number;
  front: boolean;
}

const INK = { serving: "34, 197, 94", standby: "255, 255, 255" };

export function NetworkPage({
  initialSnapshot,
  initialHistory,
  initialViewMode,
  animate = true,
}: {
  initialSnapshot?: NetworkSnapshot;
  initialHistory?: NetworkTelemetryHistory;
  initialViewMode?: NetworkViewMode;
  animate?: boolean;
}) {
  const [viewMode, setViewMode] = useState<NetworkViewMode>(() => initialViewMode
    ?? (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("view") === "globe" ? "globe" : "overview"));
  const [snapshot, setSnapshot] = useState(initialSnapshot ?? EMPTY_SNAPSHOT);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(initialSnapshot === undefined);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [historyRange, setHistoryRange] = useState<NetworkHistoryRange>("30d");
  const [history, setHistory] = useState<NetworkTelemetryHistory | null>(initialHistory ?? null);
  const [historyLoading, setHistoryLoading] = useState(initialSnapshot === undefined && initialHistory === undefined);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const selectViewMode = useCallback((next: NetworkViewMode) => {
    setViewMode(next);
    setSelectedId(null);
    if (typeof window !== "undefined") {
      window.history.replaceState({}, "", next === "globe" ? "/network?view=globe" : "/network");
    }
  }, []);

  useEffect(() => {
    const syncFromLocation = () => setViewMode(new URLSearchParams(window.location.search).get("view") === "globe" ? "globe" : "overview");
    window.addEventListener("popstate", syncFromLocation);
    return () => window.removeEventListener("popstate", syncFromLocation);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/public/v1/snapshot", { cache: "no-store" });
      if (!response.ok) throw new Error(`Network snapshot returned HTTP ${response.status}.`);
      setSnapshot(await response.json() as NetworkSnapshot);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The network snapshot is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (initialSnapshot) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15_000);
    return () => window.clearInterval(timer);
  }, [initialSnapshot, refresh]);

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(`/public/v1/history?range=${historyRange}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Network history returned HTTP ${response.status}.`);
      setHistory(await response.json() as NetworkTelemetryHistory);
      setHistoryError(null);
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "The network history is unavailable.");
    } finally {
      setHistoryLoading(false);
    }
  }, [historyRange]);

  useEffect(() => {
    if (initialHistory) return;
    void refreshHistory();
  }, [initialHistory, refreshHistory]);

  const metrics = useMemo(() => deriveNetworkMetrics(snapshot), [snapshot]);

  const dailyJobs = useMemo(() => deriveDailyJobs(history?.samples ?? []), [history]);

  const recentJobs = useMemo(() => snapshot.jobs
    .slice()
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt))
    .slice(0, 8), [snapshot.jobs]);

  const nodes = useMemo<GlobeNode[]>(() => snapshot.workers.map((worker) => {
    const placement = placeWorker(worker.region, worker.id);
    return {
      worker,
      vector: sphere(placement.lon, placement.lat),
      located: placement.located,
      country: placement.country,
      state: nodeState(worker),
      x: 0,
      y: 0,
      front: false,
    };
  }), [snapshot.workers]);

  // Serving nodes form the visible routes. A single closed ring keeps the arc
  // count proportional to the network instead of drawing every pair.
  const arcs = useMemo(() => {
    const serving = nodes.filter((node) => node.state === "serving");
    if (serving.length < 2) return [] as Float32Array[];
    const pairs = serving.length === 2 ? 1 : serving.length;
    const built: Float32Array[] = [];
    for (let index = 0; index < pairs; index += 1) {
      built.push(buildArc(serving[index]!.vector, serving[(index + 1) % serving.length]!.vector));
    }
    return built;
  }, [nodes]);

  // Refs mirror the render inputs so the animation loop never re-subscribes and
  // never forces a React render per frame.
  const nodesRef = useRef(nodes);
  const arcsRef = useRef(arcs);
  const selectedRef = useRef<string | null>(selectedId);
  const hoveredRef = useRef<string | null>(null);
  nodesRef.current = nodes;
  arcsRef.current = arcs;
  selectedRef.current = selectedId;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    const visitor = visitorLocation();
    const targetYaw = -visitor.lon * (Math.PI / 180);
    const targetTilt = Math.max(-1, Math.min(1, visitor.lat * (Math.PI / 180) * 0.9));

    const view = {
      yaw: reduceMotion ? targetYaw : targetYaw - 0.55,
      tilt: reduceMotion ? targetTilt : 0.18,
      velocity: 0,
      easing: !reduceMotion && animate,
    };
    let width = 0;
    let height = 0;
    let centerX = 0;
    let centerY = 0;
    let radius = 0;
    let pulse = 0;
    const scratch: Vec3 = [0, 0, 0];

    const layout = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.imageSmoothingEnabled = false;
      centerX = width / 2;
      centerY = height / 2;
      radius = Math.min(width, height) * 0.44;
    };

    const draw = () => {
      if (!width || !height) return;
      const sinYaw = Math.sin(view.yaw);
      const cosYaw = Math.cos(view.yaw);
      const sinTilt = Math.sin(view.tilt);
      const cosTilt = Math.cos(view.tilt);
      const project = (source: ArrayLike<number>, offset: number) =>
        rotatePoint(source, offset, sinYaw, cosYaw, sinTilt, cosTilt, scratch);

      context.clearRect(0, 0, width, height);

      // Sphere edge.
      context.strokeStyle = "rgba(255,255,255,0.09)";
      context.lineWidth = 1;
      context.beginPath();
      context.arc(centerX, centerY, radius, 0, Math.PI * 2);
      context.stroke();

      // Land cells on the near hemisphere, brightness by depth.
      const landCount = LAND_POINTS.length / 3;
      for (let index = 0; index < landCount; index += 1) {
        const point = project(LAND_POINTS, index * 3);
        if (point[2] <= 0.02) continue;
        const alpha = 0.06 + 0.34 * point[2];
        context.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
        context.fillRect((centerX + point[0] * radius) | 0, (centerY - point[1] * radius) | 0, 2, 2);
      }

      // Country borders, cut wherever they cross the horizon.
      context.strokeStyle = "rgba(255,255,255,0.2)";
      context.lineWidth = 0.6;
      context.beginPath();
      for (const line of BORDER_LINES) {
        let pen = false;
        for (let vertex = 0; vertex < line.length / 3; vertex += 1) {
          const point = project(line, vertex * 3);
          if (point[2] > 0.02) {
            const x = centerX + point[0] * radius;
            const y = centerY - point[1] * radius;
            if (pen) context.lineTo(x, y);
            else context.moveTo(x, y);
            pen = true;
          } else {
            pen = false;
          }
        }
      }
      context.stroke();

      // Route arcs between serving nodes, with a travelling pulse.
      for (const arc of arcsRef.current) {
        const count = arc.length / 3;
        context.strokeStyle = "rgba(34,197,94,0.5)";
        context.lineWidth = 1;
        context.beginPath();
        let pen = false;
        for (let index = 0; index < count; index += 1) {
          const point = project(arc, index * 3);
          if (point[2] > 0) {
            const x = centerX + point[0] * radius;
            const y = centerY - point[1] * radius;
            if (pen) context.lineTo(x, y);
            else context.moveTo(x, y);
            pen = true;
          } else {
            pen = false;
          }
        }
        context.stroke();
        const head = Math.floor((pulse % 1) * (count - 1));
        const point = project(arc, head * 3);
        if (point[2] > 0) {
          context.fillStyle = "#fff";
          context.fillRect(
            ((centerX + point[0] * radius) | 0) - 1,
            ((centerY - point[1] * radius) | 0) - 1,
            3,
            3,
          );
        }
      }

      // Nodes. Screen positions are cached here for pointer hit-testing.
      context.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      context.textBaseline = "middle";
      for (const node of nodesRef.current) {
        const point = project(node.vector, 0);
        if (point[2] <= 0.02) {
          node.front = false;
          continue;
        }
        const x = centerX + point[0] * radius;
        const y = centerY - point[1] * radius;
        node.x = x;
        node.y = y;
        node.front = true;
        const ix = x | 0;
        const iy = y | 0;
        // Fade nodes in as they round the horizon so they never pop.
        const edge = Math.min(1, (point[2] - 0.02) * 4);
        const selected = node.worker.id === selectedRef.current;
        const hovered = node.worker.id === hoveredRef.current;

        if (node.state === "serving") {
          context.fillStyle = `rgba(${INK.serving},${(0.2 * edge).toFixed(2)})`;
          context.fillRect(ix - 4, iy - 4, 8, 8);
          context.fillStyle = `rgba(${INK.serving},${edge.toFixed(2)})`;
          context.fillRect(ix - 2, iy - 2, 4, 4);
        } else {
          const alpha = (node.state === "standby" ? 0.8 : 0.34) * edge;
          context.fillStyle = `rgba(${INK.standby},${alpha.toFixed(2)})`;
          context.fillRect(ix - 2, iy - 2, 4, 4);
        }

        if (selected || hovered) {
          context.strokeStyle = selected ? "rgba(34,197,94,0.95)" : "rgba(255,255,255,0.7)";
          context.lineWidth = 1;
          context.strokeRect(ix - 6.5, iy - 6.5, 13, 13);
          const label = (node.located ? node.worker.region : shortId(node.worker.id)).toLowerCase();
          const textWidth = context.measureText(label).width;
          const textX = ix + 11 + textWidth > width - 4 ? ix - 11 - textWidth : ix + 11;
          context.fillStyle = selected ? "rgba(34,197,94,0.95)" : "rgba(255,255,255,0.85)";
          context.fillText(label, textX, iy);
        }
      }
    };

    const shortestAngle = (from: number, to: number, fraction: number) => {
      const delta = ((((to - from + Math.PI) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) - Math.PI;
      return from + delta * fraction;
    };

    let frame = 0;
    let previous = 0;
    const tick = (timestamp: number) => {
      const elapsed = previous ? Math.min(timestamp - previous, 80) : 16;
      previous = timestamp;
      if (!reduceMotion && animate) {
        if (dragRef.current) {
          // The pointer owns the camera.
        } else if (Math.abs(view.velocity) > 0.0003) {
          view.yaw += view.velocity;
          view.velocity *= 0.93;
        } else if (view.easing) {
          view.yaw = shortestAngle(view.yaw, targetYaw, 0.05);
          view.tilt += (targetTilt - view.tilt) * 0.05;
          if (Math.abs(shortestAngle(view.yaw, targetYaw, 1)) < 0.01
            && Math.abs(targetTilt - view.tilt) < 0.01) {
            view.easing = false;
          }
        }
        pulse += elapsed / 2_600;
      }
      draw();
      frame = window.requestAnimationFrame(tick);
    };

    // Pointer handling lives here so it can mutate the camera without a render.
    const pointerPosition = (event: PointerEvent) => {
      const bounds = canvas.getBoundingClientRect();
      return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    };
    const nodeAt = (x: number, y: number) => {
      let best: GlobeNode | null = null;
      let bestDistance = 14;
      for (const node of nodesRef.current) {
        if (!node.front) continue;
        const distance = Math.abs(node.x - x) + Math.abs(node.y - y);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = node;
        }
      }
      return best;
    };

    const onPointerDown = (event: PointerEvent) => {
      dragRef.current = { x: event.clientX, y: event.clientY, moved: false };
      view.velocity = 0;
      view.easing = false;
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("dragging");
    };
    const onPointerMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (drag) {
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        const gain = 1 / (radius || 300);
        view.yaw += dx * gain;
        view.tilt = Math.max(-1.2, Math.min(1.2, view.tilt + dy * gain));
        view.velocity = dx * gain;
        if (Math.abs(dx) + Math.abs(dy) > 0) drag.moved = drag.moved || Math.abs(dx) + Math.abs(dy) > 5;
        drag.x = event.clientX;
        drag.y = event.clientY;
        if (reduceMotion) draw();
        return;
      }
      const { x, y } = pointerPosition(event);
      const hit = nodeAt(x, y);
      const hitId = hit?.worker.id ?? null;
      if (hitId !== hoveredRef.current) {
        hoveredRef.current = hitId;
        canvas.style.cursor = hit ? "pointer" : "";
        if (reduceMotion) draw();
      }
    };
    const onPointerUp = (event: PointerEvent) => {
      const drag = dragRef.current;
      dragRef.current = null;
      canvas.classList.remove("dragging");
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (drag && !drag.moved) {
        const { x, y } = pointerPosition(event);
        const hit = nodeAt(x, y);
        setSelectedId(hit ? hit.worker.id : null);
      }
    };
    const onPointerCancel = () => {
      dragRef.current = null;
      canvas.classList.remove("dragging");
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerCancel);

    let resizeTimer = 0;
    const onResize = () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        layout();
        draw();
      }, 40);
    };
    // The canvas' CSS box changes when Overview/Globe changes. Observing the
    // element itself keeps its bitmap and projection in lockstep with that box;
    // a window resize listener alone leaves the previous bitmap stretched.
    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(() => {
        layout();
        draw();
      });
    resizeObserver?.observe(canvas);
    window.addEventListener("resize", onResize);

    layout();
    if (reduceMotion || !animate) draw();
    else frame = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(resizeTimer);
      resizeObserver?.disconnect();
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerCancel);
    };
  }, [animate, viewMode]);

  const selectedNode = nodes.find((node) => node.worker.id === selectedId) ?? null;
  const hasData = snapshot.capturedAt !== EMPTY_SNAPSHOT.capturedAt;
  const stale = hasData && Date.now() - Date.parse(snapshot.capturedAt) > 60_000;

  useEffect(() => {
    if (selectedId && !snapshot.workers.some((worker) => worker.id === selectedId)) {
      setSelectedId(null);
    }
  }, [selectedId, snapshot.workers]);

  useEffect(() => {
    if (!selectedId) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setSelectedId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId]);

  const joining = nodes.filter((node) => node.state !== "serving").slice(0, 3);

  const copyCommand = useCallback(() => {
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_400);
    };
    if (navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(SNAPSHOT_COMMAND).then(done, done);
      return;
    }
    done();
  }, []);

  return <main className={`net net-view-${viewMode}`}>
    <div className="net-stage">
      {viewMode === "globe" && <canvas
        ref={canvasRef}
        className="net-globe"
        aria-label={`Interactive globe showing ${nodes.length} announced Mycellios nodes`}
        role="img"
      />}

      <h1 className="net-visually-hidden">Mycellios network</h1>

      <LandingHeader active="network" />

      <div className="net-hud net-view-switcher" role="group" aria-label="Network view">
        <button type="button" className={viewMode === "overview" ? "active" : ""} aria-pressed={viewMode === "overview"} onClick={() => selectViewMode("overview")}>Overview</button>
        <button type="button" className={viewMode === "globe" ? "active" : ""} aria-pressed={viewMode === "globe"} onClick={() => selectViewMode("globe")}>Globe</button>
      </div>

      {viewMode === "overview" && <section className="net-dashboard" aria-label="Live network overview">
        <header className="net-dashboard-head">
          <span className="net-hero-kicker"><i className={error || stale ? "degraded" : ""} />{error ? "LAST VERIFIED SNAPSHOT" : loading ? "CONNECTING" : stale ? "STALE SNAPSHOT" : "LIVE NETWORK"}</span>
        </header>
        <div className="net-hero-metrics">
          <span><strong>{snapshot.summary.completedJobs.toLocaleString("en-US")}</strong>completed jobs</span>
          <span><strong>{snapshot.summary.connected}</strong>connected nodes</span>
          <span><strong>{snapshot.summary.online}</strong>online now</span>
          <span><strong>{metrics.gpuCount}</strong>GPUs online</span>
          <span><strong>{metrics.availableModels}</strong>models serving</span>
          <span><strong>{metrics.activeJobs.length}</strong>active tasks</span>
          <span><strong>{formatMemory(metrics.memoryGb)}</strong>memory offered</span>
        </div>
        <div className="net-dashboard-visuals">
          <article className="net-globe-card">
            <canvas
              ref={canvasRef}
              className="net-globe"
              aria-label={`Interactive globe showing ${nodes.length} announced Mycellios nodes`}
              role="img"
            />
            <header><span>LIVE TOPOLOGY</span><strong>{snapshot.summary.online} online · {metrics.regions} regions</strong></header>
            <button type="button" onClick={() => selectViewMode("globe")} aria-label="Open full-screen globe"><ArrowRight /></button>
          </article>
          <DashboardChart title="Offered capacity" unit="GB" samples={history?.samples ?? []} series={[{ key: "offeredVramMb", label: "Offered", tone: "bronze", divisor: 1024 }, { key: "freeVramMb", label: "Free", tone: "sage", divisor: 1024 }]} />
          <DashboardChart title="Node availability" unit="nodes" samples={history?.samples ?? []} series={[{ key: "connectedNodes", label: "Connected", tone: "ink" }, { key: "onlineNodes", label: "Online", tone: "sage" }]} />
          <DashboardChart title="Model fabric" unit="models" samples={history?.samples ?? []} series={[{ key: "activeModels", label: "Models", tone: "ink" }, { key: "modelReplicas", label: "Replicas", tone: "bronze" }]} />
          <DashboardChart title="Inference load" unit="jobs" samples={history?.samples ?? []} series={[{ key: "inflightJobs", label: "In flight", tone: "bronze" }, { key: "runningJobs", label: "Running", tone: "sage" }]} />
          <DashboardChart title="Completed work" unit="jobs" samples={history?.samples ?? []} series={[{ key: "completedJobs", label: "Completed", tone: "sage" }]} />
        </div>
      </section>}

      {viewMode === "overview" && <a className="net-hud net-scroll-cue" href="#network-evidence"><span>SCROLL</span>30-day history · node ledger · methodology <b aria-hidden="true">↓</b></a>}

      <div className="net-hud net-legend" role="list" aria-label="Node state legend">
        <span role="listitem"><i style={{ background: "#22c55e" }} />serving</span>
        <span role="listitem"><i style={{ background: "rgba(255,255,255,0.82)" }} />standby</span>
        <span role="listitem"><i style={{ background: "rgba(255,255,255,0.32)" }} />joining</span>
      </div>

      <div className="net-hud net-stats net-stats-left" aria-label="Network capacity">
        <Stat value={String(metrics.gpuCount)} label="GPUS ONLINE" />
        <Stat value={String(metrics.regions)} label="REGIONS" />
        <Stat value={String(metrics.availableModels)} label="MODELS SERVING" />
      </div>

      <div
        className={`net-hud net-stats net-stats-right${selectedNode ? " hidden" : ""}`}
        aria-label="Network activity"
      >
        <Stat value={String(metrics.activeJobs.length)} label="ACTIVE TASKS" />
        <Stat value={metrics.tokensServed.toLocaleString("en-US")} label="TOKENS IN SNAPSHOT" />
        <Stat value={formatMemory(metrics.memoryGb)} unit label="GPU MEMORY POOLED" />
      </div>

      <div className="net-hud net-ticker">
        <div className="net-ticker-label">{joining.length > 0 ? "joining now" : "node activity"}</div>
        {joining.length === 0
          ? <div className="net-ticker-row empty">
              {nodes.length === 0 ? "Waiting for the first node." : "Every announced node is serving."}
            </div>
          : joining.map((node, index) => <div
              key={node.worker.id}
              className="net-ticker-row"
              style={{ opacity: 1 - index * 0.28 }}
            >
              <span className="mark">▸</span>
              <b>{describeHardware(node.worker)}</b>
              {` ${node.state === "standby" ? "on standby" : "joining"} · `}
              <span className="loc">
                {node.located ? `${node.worker.region}${node.country ? `, ${node.country}` : ""}` : "region not announced"}
              </span>
            </div>)}
      </div>

      <div className="net-hud net-cmd">
        <code>{SNAPSHOT_COMMAND}</code>
        <button type="button" onClick={copyCommand}>{copied ? "copied" : "copy"}</button>
      </div>

      <div className="net-hud net-status">
        <span className={error || stale ? "degraded" : ""}>
          <i />
          {error ? "REFRESH FAILED" : loading ? "CONNECTING" : stale ? "STALE" : "LIVE"}
        </span>
        <span>{hasData ? `${relativeTime(snapshot.capturedAt)} · v${snapshot.version}` : "—"}</span>
        <a href="/earn">connect a device →</a>
      </div>

      {selectedNode && <NodePanel
        node={selectedNode}
        jobs={snapshot.jobs.filter((job) => job.workerId === selectedNode.worker.id)}
        onClose={() => setSelectedId(null)}
      />}

      {!loading && nodes.length === 0 && <div className="net-empty">
        <strong>No nodes announced</strong>
        <span>The coordinator is reachable, but no worker is currently visible.</span>
        <a href="/earn">Connect a device →</a>
      </div>}

      {error && <div className="net-error" role="alert">
        <span>{hasData ? "Live refresh failed. Showing the last successful snapshot." : error}</span>
        <button onClick={() => void refresh()}>Retry</button>
      </div>}
    </div>

    <section className="net-data" id="network-evidence" aria-labelledby="net-data-title">
      <div className="net-data-shell">
        <header className="net-data-intro">
          <div>
            <span className="net-kicker">PUBLIC NETWORK EVIDENCE</span>
            <h2 id="net-data-title">A network you can inspect.</h2>
          </div>
          <p>Capacity, availability and work are published from the coordinator’s observed state. No simulated nodes, projected earnings or configured throughput appear here.</p>
        </header>

        <div className="net-proof-strip" aria-label="Evidence provenance">
          <span><i className="physical" />Observed now <strong>{relativeTime(snapshot.capturedAt)}</strong></span>
          <span><i className="stored" />Persisted history <strong>every {history?.intervalMinutes ?? 10} min</strong></span>
          <span><i className="sealed" />Disclosure <strong>public, redacted</strong></span>
        </div>

        <section className="net-signal-index" aria-labelledby="net-signals-title">
          <header className="net-section-head">
            <div><span className="net-kicker">SIGNAL INDEX</span><h2 id="net-signals-title">The network, in signals.</h2></div>
            <p>Compact surfaces for the questions visitors ask first. Every value is observed, derived, or deliberately withheld.</p>
          </header>
          <div className="net-signal-groups">
            <article className="net-signal-group">
              <header><strong>LIVE</strong><span>coordinator snapshot</span></header>
              <div className="net-signal-cards">
                <SignalCard label="Workers online" value={metrics.activeWorkers.length} tone="healthy" />
                <SignalCard label="Native nodes" value={metrics.activeWorkers.filter((worker) => worker.kind !== "browser").length} />
                <SignalCard label="Browser nodes" value={snapshot.summary.mobile} />
                <SignalCard label="Models serving" value={metrics.availableModels} />
                <SignalCard label="Tasks in flight" value={metrics.activeJobs.length} />
                <SignalCard label="Memory offered" value={formatMemory(metrics.memoryGb)} />
              </div>
            </article>
            <article className="net-signal-group">
              <header><strong>NETWORK</strong><span>persisted telemetry</span></header>
              <div className="net-signal-cards">
                <SignalCard label="Completed jobs" value={snapshot.summary.completedJobs.toLocaleString("en-US")} />
                <SignalCard label="Samples stored" value={history?.samples.length.toLocaleString("en-US") ?? "—"} />
                <SignalCard label="Retention" value={history ? `${history.retentionDays}d` : "—"} />
                <SignalCard label="Sampling" value={history ? `${history.intervalMinutes}m` : "—"} />
              </div>
            </article>
            <article className="net-signal-group net-signal-group-muted">
              <header><strong>TRANSPARENCY</strong><span>no unsupported claims</span></header>
              <div className="net-signal-cards">
                <SignalCard label="Revenue" value="Not published" />
                <SignalCard label="Rewards" value="Not active" />
                <SignalCard label="$SPORE" value="Not active" />
                <SignalCard label="Throughput" value="Not measured" />
              </div>
            </article>
          </div>
        </section>

        <section className="net-wall" aria-labelledby="net-wall-network-title">
          <header className="net-wall-head">
            <h2 id="net-wall-network-title">network</h2>
            <div className="net-range" role="group" aria-label="History range">
              {HISTORY_RANGES.map(({ value, label }) => <button type="button" key={value} className={historyRange === value ? "active" : ""} aria-pressed={historyRange === value} onClick={() => setHistoryRange(value)}>{label}</button>)}
            </div>
          </header>

          {historyError && <div className="net-history-alert" role="alert"><Database /><span><strong>History unavailable</strong>{historyError}</span><button type="button" onClick={() => void refreshHistory()}><RefreshCw />Retry</button></div>}
          {historyLoading && !history && <div className="net-history-loading"><span />Reading persisted samples…</div>}
          {history && history.samples.length > 0 ? <div className="net-wall-grid">
            <StripedBarChart
              full
              title="jobs completed per day"
              note="derived from the persisted cumulative counter"
              buckets={dailyJobs}
            />
            <StepAreaChart
              title="capacity offered (GB)"
              note="observed offered vram ÷ 1024"
              samples={history.samples}
              metric="offeredVramMb"
              divisor={1024}
              unit="GB"
              tone="bronze"
            />
            <StepLineChart
              title="nodes online"
              note="observed online nodes"
              samples={history.samples}
              metric="onlineNodes"
              unit="nodes"
              tone="ivory"
            />
            <StepLineChart
              full
              highResolution
              title="workers online"
              note={`sampled every ${history.intervalMinutes} min · ${history.samples.length} persisted observations`}
              samples={history.samples}
              metric="onlineNodes"
              unit="nodes"
              tone="ivory"
            />
          </div> : !historyLoading && !historyError ? <div className="net-no-history"><strong>No historical samples yet.</strong><span>The first observation will appear after the coordinator persists it.</span></div> : null}
        </section>

        <section className="net-wall" aria-labelledby="net-wall-snapshot-title">
          <header className="net-wall-head">
            <h2 id="net-wall-snapshot-title">snapshot</h2>
            <span className="net-wall-note">{hasData ? `observed ${relativeTime(snapshot.capturedAt)}` : "awaiting first snapshot"}</span>
          </header>
          <div className="net-wall-stats">
            <WallStat value={snapshot.summary.registered.toLocaleString("en-US")} label="registered nodes" />
            <WallStat value={snapshot.summary.connected.toLocaleString("en-US")} label="connected" />
            <WallStat value={snapshot.summary.online.toLocaleString("en-US")} label="online now" />
            <WallStat value={String(metrics.gpuCount)} label="GPUs online" />
            <WallStat value={String(metrics.availableModels)} label="models serving" />
            <WallStat value={snapshot.summary.completedJobs.toLocaleString("en-US")} label="completed jobs" />
          </div>
        </section>

        <section className="net-wall" aria-labelledby="net-wall-spore-title">
          <header className="net-wall-head">
            <h2 id="net-wall-spore-title">$SPORE</h2>
            <span className="net-wall-note">no public economic feed — states, not projections</span>
          </header>
          <div className="net-wall-stats">
            <WallStat muted value="Not active" label="token" />
            <WallStat muted value="Not active" label="rewards" />
            <WallStat muted value="Not published" label="buyback" />
            <WallStat muted value="—" label="stakers" />
          </div>
          <div className="net-wall-panel net-wall-feed-empty">
            <strong>economic feed not published yet</strong>
            <span>Nothing here is projected, simulated or configured. When a public, auditable economic source exists, its charts will render in this panel.</span>
          </div>
        </section>

        <section className="net-wall" aria-labelledby="net-wall-activity-title">
          <header className="net-wall-head">
            <h2 id="net-wall-activity-title">recent activity</h2>
            <span className="net-wall-note">latest jobs in the current snapshot</span>
          </header>
          <div className="net-wall-panel net-wall-activity">
            {recentJobs.length > 0 ? <div className="net-wall-activity-table" role="table" aria-label="Recent jobs">
              <div className="net-wall-activity-row head" role="row"><span>job</span><span>model</span><span>status</span><span>tokens</span><span>age</span></div>
              {recentJobs.map((job) => <div className="net-wall-activity-row" role="row" key={job.id}>
                <span className="id">{shortId(job.id)}</span>
                <span>{job.model}</span>
                <span className={`status s-${job.status}`}>{job.status}</span>
                <span>{(job.inputTokens + job.outputTokens).toLocaleString("en-US")}</span>
                <span>{relativeTime(job.updatedAt)}</span>
              </div>)}
            </div> : <div className="net-wall-empty"><strong>No jobs observed in the current snapshot.</strong><span>Completed and in-flight jobs appear here as the coordinator records them.</span></div>}
          </div>
        </section>

        <section className="net-ledger" aria-labelledby="net-ledger-title">
          <header className="net-section-head">
            <div><span className="net-kicker">CURRENT SUPPLY</span><h2 id="net-ledger-title">Nodes behind the snapshot.</h2></div>
            <a href="/join">Connect a device <ArrowRight /></a>
          </header>
          <div className="net-ledger-table" role="table" aria-label="Public node ledger">
            <div className="net-ledger-row head" role="row"><span>Node</span><span>State</span><span>Hardware</span><span>Work</span><span>Last seen</span></div>
            {metrics.activeWorkers.slice(0, 12).map((worker) => <div className="net-ledger-row" role="row" key={worker.id}>
              <span><strong>{worker.region && worker.region !== "auto" ? worker.region : shortId(worker.id)}</strong><small>{worker.kind} · {shortId(worker.id)}</small></span>
              <span><i className={`net-state-dot ${nodeState(worker)}`} />{nodeState(worker)}</span>
              <span><strong>{describeHardware(worker)}</strong><small>{formatMemory(worker.offeredVramMb / 1024)}</small></span>
              <span><strong>{worker.jobsCompleted.toLocaleString("en-US")}</strong><small>completed jobs</small></span>
              <span>{relativeTime(worker.lastSeenAt)}</span>
            </div>)}
            {metrics.activeWorkers.length === 0 && <div className="net-ledger-empty">No online nodes are present in the current snapshot.</div>}
          </div>
        </section>

        <section className="net-method" aria-labelledby="net-method-title">
          <div><span className="net-kicker">MEASUREMENT CONTRACT</span><h2 id="net-method-title">What these numbers mean.</h2></div>
          <div className="net-method-copy">
            <p><ShieldCheck /> <span><strong>Observed</strong>Workers, models, memory and jobs come from the coordinator’s live public snapshot.</span></p>
            <p><Database /> <span><strong>Derived</strong>Regions and totals are calculated only from that snapshot; the formula stays visible in the open source client.</span></p>
            <p><CheckCircle2 /> <span><strong>Deliberately absent</strong>Revenue, rewards and throughput stay unpublished until they have an auditable public source.</span></p>
          </div>
          <div className="net-method-links"><a href="/docs/protocol">Read the architecture <ExternalLink /></a><a href="https://github.com/tych0s/mycellios" target="_blank" rel="noreferrer">Inspect the source <ExternalLink /></a></div>
        </section>
      </div>
    </section>
  </main>;
}

function SignalCard({ label, value, tone }: { label: string; value: string | number; tone?: "healthy" }) {
  return <div className={`net-signal-card${tone ? ` ${tone}` : ""}`}><strong>{value}</strong><span>{label}</span></div>;
}

function WallStat({ value, label, muted = false }: { value: string; label: string; muted?: boolean }) {
  return <article className={`net-wall-stat${muted ? " muted" : ""}`}><strong>{value}</strong><span>{label}</span></article>;
}

type WallTone = "ivory" | "bronze" | "sage";

interface WallChartGeo {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
  plotWidth: number;
  plotHeight: number;
  ceiling: number;
  x: (index: number) => number;
  y: (value: number) => number;
}

interface WallTipLine {
  label: string;
  value: string;
  tone?: WallTone;
}

/**
 * Shared chart frame: y ticks at 0/mid/max, three x labels, a pointer-driven
 * vertical guide and a small mono tooltip. The hover is pure React state on
 * pointer events, so server rendering never touches window.
 */
function WallChartFrame({ count, max, banded = false, height = 220, xLabels, tooltip, ariaLabel, children }: {
  count: number;
  max: number;
  banded?: boolean;
  height?: number;
  xLabels: [string, string, string];
  tooltip: (index: number) => { title: string; lines: WallTipLine[] };
  ariaLabel: string;
  children: (geo: WallChartGeo) => ReactNode;
}) {
  const width = 720;
  const left = 46;
  const right = 14;
  const top = 16;
  const bottom = 30;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const ceiling = Math.max(1, max);
  const x = banded
    ? (index: number) => left + ((index + 0.5) / Math.max(count, 1)) * plotWidth
    : (index: number) => count <= 1 ? left + plotWidth / 2 : left + (index / (count - 1)) * plotWidth;
  const y = (value: number) => top + (1 - value / ceiling) * plotHeight;
  const geo: WallChartGeo = { width, height, left, right, top, bottom, plotWidth, plotHeight, ceiling, x, y };

  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || count === 0) return;
    const bounds = svg.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const px = ((event.clientX - bounds.left) / bounds.width) * width;
    const ratio = Math.min(1, Math.max(0, (px - left) / plotWidth));
    setHover(banded
      ? Math.min(count - 1, Math.floor(ratio * count))
      : Math.round(ratio * (count - 1)));
  };

  const tip = hover !== null && count > 0 ? tooltip(Math.max(0, Math.min(count - 1, hover))) : null;
  const tipWidth = tip
    ? Math.max(104, Math.max(tip.title.length, ...tip.lines.map((line) => `${line.label} ${line.value}`.length)) * 6.4 + 22)
    : 0;
  const tipHeight = tip ? 24 + tip.lines.length * 15 : 0;
  const tipX = tip && hover !== null
    ? (x(hover) + 12 + tipWidth > width - right ? x(hover) - 12 - tipWidth : x(hover) + 12)
    : 0;

  return <svg
    ref={svgRef}
    className="net-wall-chart"
    viewBox={`0 0 ${width} ${height}`}
    role="img"
    aria-label={ariaLabel}
    onPointerMove={onPointerMove}
    onPointerLeave={() => setHover(null)}
  >
    {[0, 0.5, 1].map((ratio) => <g key={ratio}>
      <line className="grid" x1={left} x2={width - right} y1={y(ceiling * ratio)} y2={y(ceiling * ratio)} />
      <text x={left - 8} y={y(ceiling * ratio) + 3.5} textAnchor="end">{formatChartValue(ceiling * ratio)}</text>
    </g>)}
    {count > 0 && children(geo)}
    <text className="axis" x={left} y={height - 8}>{xLabels[0]}</text>
    <text className="axis" x={left + plotWidth / 2} y={height - 8} textAnchor="middle">{xLabels[1]}</text>
    <text className="axis" x={width - right} y={height - 8} textAnchor="end">{xLabels[2]}</text>
    {tip && hover !== null && <g className="net-wall-tip" pointerEvents="none">
      <line className="guide" x1={x(hover)} x2={x(hover)} y1={top} y2={top + plotHeight} />
      <rect x={tipX} y={top + 4} width={tipWidth} height={tipHeight} rx={5} />
      <text className="tip-title" x={tipX + 11} y={top + 4 + 15}>{tip.title}</text>
      {tip.lines.map((line, index) => <text key={line.label} className={`tip-line ${line.tone ?? ""}`} x={tipX + 11} y={top + 4 + 30 + index * 15}>{line.label} {line.value}</text>)}
    </g>}
  </svg>;
}

/** Step-after line: the value holds from its sample until the next one. */
function stepLinePath(values: number[], geo: WallChartGeo): string {
  let path = "";
  for (let index = 0; index < values.length; index += 1) {
    const px = geo.x(index);
    const py = geo.y(values[index]!);
    path += index === 0
      ? `M ${px.toFixed(2)} ${py.toFixed(2)}`
      : ` L ${px.toFixed(2)} ${geo.y(values[index - 1]!).toFixed(2)} L ${px.toFixed(2)} ${py.toFixed(2)}`;
  }
  return path;
}

function stepAreaPath(values: number[], geo: WallChartGeo): string {
  const line = stepLinePath(values, geo);
  if (!line) return "";
  const baseline = geo.y(0).toFixed(2);
  return `${line} L ${geo.x(values.length - 1).toFixed(2)} ${baseline} L ${geo.x(0).toFixed(2)} ${baseline} Z`;
}

function sampleXLabels(points: NetworkTelemetrySample[]): [string, string, string] {
  return [
    formatChartDate(points[0]!.capturedAt),
    formatChartDate(points[Math.floor((points.length - 1) / 2)]!.capturedAt),
    formatChartDate(points.at(-1)!.capturedAt),
  ];
}

function WallPanel({ title, note, full = false, children }: { title: string; note?: string | undefined; full?: boolean; children: ReactNode }) {
  return <article className={`net-wall-panel${full ? " full" : ""}`}>
    <header className="net-wall-panel-head"><strong>{title}</strong>{note ? <span>{note}</span> : null}</header>
    {children}
  </article>;
}

/** Daily buckets rendered as stacks of thin horizontal stripes, like the reference wall. */
function StripedBarChart({ title, note, full = false, buckets }: { title: string; note?: string; full?: boolean; buckets: DailyJobBucket[] }) {
  const max = Math.max(0, ...buckets.map((bucket) => bucket.count));
  return <WallPanel title={title} note={note} full={full}>
    {buckets.length > 0 ? <WallChartFrame
      count={buckets.length}
      max={max}
      banded
      ariaLabel={`${title}: ${buckets.length} daily buckets derived from the persisted counter`}
      xLabels={[
        buckets[0]!.day.slice(5),
        buckets[Math.floor((buckets.length - 1) / 2)]!.day.slice(5),
        buckets.at(-1)!.day.slice(5),
      ]}
      tooltip={(index) => ({
        title: buckets[index]!.day,
        lines: [{ label: "jobs completed", value: buckets[index]!.count.toLocaleString("en-US") }],
      })}
    >{(geo) => <g className="net-wall-bars">
      {buckets.map((bucket, index) => {
        if (bucket.count <= 0) return null;
        const band = geo.plotWidth / buckets.length;
        const barWidth = Math.min(30, Math.max(2.5, band * 0.6));
        const cx = geo.x(index);
        const barTop = geo.y(bucket.count);
        const barBottom = geo.y(0);
        const stripes: ReactNode[] = [];
        for (let sy = barTop; sy < barBottom - 0.5; sy += 3) {
          stripes.push(<rect key={sy} x={cx - barWidth / 2} y={sy} width={barWidth} height={Math.min(2, barBottom - sy)} />);
        }
        return <g key={bucket.day}>{stripes}</g>;
      })}
    </g>}</WallChartFrame> : <div className="net-wall-empty"><strong>No persisted samples in this range yet.</strong></div>}
  </WallPanel>;
}

/** Step-area series filled with a dot-grid pattern. */
function StepAreaChart({ title, note, full = false, samples, metric, divisor = 1, unit, tone }: {
  title: string;
  note?: string;
  full?: boolean;
  samples: NetworkTelemetrySample[];
  metric: ChartMetricKey;
  divisor?: number;
  unit: string;
  tone: WallTone;
}) {
  const patternId = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const points = downsample(samples, 180);
  const values = points.map((sample) => sample[metric] / divisor);
  const max = Math.max(0, ...values);
  const seriesLabel = title.replace(/\s*\(.*\)$/, "");
  return <WallPanel title={title} note={note} full={full}>
    <WallChartFrame
      count={points.length}
      max={max}
      ariaLabel={`${title}: ${points.length} persisted observations`}
      xLabels={sampleXLabels(points)}
      tooltip={(index) => ({
        title: formatChartDate(points[index]!.capturedAt),
        lines: [{ label: seriesLabel, value: `${formatChartValue(values[index]!)} ${unit}`, tone }],
      })}
    >{(geo) => <>
      <defs>
        <pattern id={patternId} className={`net-wall-dots ${tone}`} width="6" height="6" patternUnits="userSpaceOnUse">
          <circle cx="1.7" cy="1.7" r="1" />
        </pattern>
      </defs>
      {values.length === 1
        ? <circle className={`net-wall-dot ${tone}`} cx={geo.x(0)} cy={geo.y(values[0]!)} r="2.5" />
        : <>
          <path className={`net-wall-area ${tone}`} d={stepAreaPath(values, geo)} fill={`url(#${patternId})`} />
          <path className={`net-wall-line ${tone}`} d={stepLinePath(values, geo)} />
        </>}
    </>}</WallChartFrame>
  </WallPanel>;
}

/** High-resolution step line over every persisted sample. */
function StepLineChart({ title, note, full = false, highResolution = false, samples, metric, unit, tone }: {
  title: string;
  note?: string;
  full?: boolean;
  highResolution?: boolean;
  samples: NetworkTelemetrySample[];
  metric: ChartMetricKey;
  unit: string;
  tone: WallTone;
}) {
  const points = downsample(samples, highResolution ? 400 : 180);
  const values = points.map((sample) => sample[metric]);
  const max = Math.max(0, ...values);
  return <WallPanel title={title} note={note} full={full}>
    <WallChartFrame
      count={points.length}
      max={max}
      ariaLabel={`${title}: ${points.length} persisted observations`}
      xLabels={sampleXLabels(points)}
      tooltip={(index) => ({
        title: formatChartDate(points[index]!.capturedAt),
        lines: [{ label: title, value: `${formatChartValue(values[index]!)} ${unit}`, tone }],
      })}
    >{(geo) => values.length === 1
      ? <circle className={`net-wall-dot ${tone}`} cx={geo.x(0)} cy={geo.y(values[0]!)} r="2.5" />
      : <path className={`net-wall-line ${tone}`} d={stepLinePath(values, geo)} />}
    </WallChartFrame>
  </WallPanel>;
}

type ChartMetricKey = keyof Pick<NetworkTelemetrySample, "offeredVramMb" | "freeVramMb" | "connectedNodes" | "onlineNodes" | "browserNodes" | "activeModels" | "modelReplicas" | "modelPipelines" | "inflightJobs" | "runningJobs" | "completedJobs">;

function DashboardChart({ title, unit, samples, series }: { title: string; unit: string; samples: NetworkTelemetrySample[]; series: Array<{ key: ChartMetricKey; label: string; tone: "ink" | "sage" | "bronze"; divisor?: number }> }) {
  const points = downsample(samples, 48);
  const width = 360;
  const height = 190;
  const value = (sample: NetworkTelemetrySample, key: ChartMetricKey, divisor = 1) => sample[key] / divisor;
  const ceiling = Math.max(1, ...points.flatMap((sample) => series.map((item) => value(sample, item.key, item.divisor))));
  const x = (index: number) => 16 + (index / Math.max(points.length - 1, 1)) * 328;
  const y = (amount: number) => 12 + (1 - amount / ceiling) * 154;
  const latest = points.at(-1);
  return <article className="net-dashboard-chart">
    <header><div><span>30-DAY HISTORY</span><h2>{title}</h2></div><strong>{latest ? formatChartValue(value(latest, series[0]!.key, series[0]!.divisor)) : "—"} <small>{unit}</small></strong></header>
    {points.length > 0 ? <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}: ${samples.length} persisted observations`}>
      {[0, .5, 1].map((ratio) => <line key={ratio} x1="16" x2="344" y1={12 + ratio * 154} y2={12 + ratio * 154} />)}
      {series.map((item) => <polyline className={item.tone} key={item.key} points={points.map((sample, index) => `${x(index)},${y(value(sample, item.key, item.divisor))}`).join(" ")} />)}
    </svg> : <div className="net-dashboard-chart-empty">Waiting for persisted samples</div>}
    <footer>{series.map((item) => <span className={item.tone} key={item.key}><i />{item.label}</span>)}</footer>
  </article>;
}

function downsample<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const stride = (items.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (_, index) => items[Math.round(index * stride)]!);
}

function formatChartValue(value: number): string {
  return value >= 100 ? Math.round(value).toLocaleString("en-US") : value >= 10 ? value.toFixed(0) : value.toFixed(1).replace(/\.0$/, "");
}

function formatChartDate(value: string): string {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(parsed) : "—";
}

function Stat({ value, label, unit = false }: { value: string; label: string; unit?: boolean }) {
  // Values like "56.0 GB" carry their unit in the string; the unit renders at
  // body size so the numeral keeps the display face's weight.
  const [amount, suffix] = unit ? splitUnit(value) : [value, ""];
  return <div className="net-stat">
    <div className="net-stat-value">{amount}{suffix && <small> {suffix}</small>}</div>
    <div className="net-stat-label">{label}</div>
  </div>;
}

function splitUnit(value: string): [string, string] {
  const match = /^([\d.,]+)\s*(\D.*)$/.exec(value);
  return match ? [match[1]!, match[2]!] : [value, ""];
}

function NodePanel({ node, jobs, onClose }: {
  node: GlobeNode;
  jobs: NetworkJob[];
  onClose: () => void;
}) {
  const { worker, state, located, country } = node;
  const gpu = worker.gpus[0];
  const deployment = worker.deployments[0];
  const reliability = Number.isFinite(worker.reliability)
    ? Math.max(0, Math.min(1, worker.reliability))
    : null;
  const stateLabel = state === "serving" ? "serving" : state === "standby" ? "standby" : "joining";
  return <div className="net-hud net-panel" aria-label={`Node detail for ${located ? worker.region : worker.id}`}>
    <div className="net-panel-top">
      <span className="net-panel-city">{located ? worker.region : "Unlocated node"}</span>
      <span className="net-panel-cc">{country ?? worker.kind}</span>
      <button className="net-panel-close" type="button" onClick={onClose} aria-label="Close node detail">×</button>
    </div>
    <div className={`net-panel-state ${state}`}>{stateLabel}</div>
    <div className="net-panel-id">{worker.id}</div>
    <div className="net-panel-rows">
      <Row label="gpu" value={gpu
        ? `${gpu.model.toLowerCase()} · ${formatMemory(gpu.offeredVramMb / 1_024)}`
        : worker.kind === "browser" ? "browser compute" : "not announced"} />
      <Row label="device" value={worker.kind} />
      <Row label="memory offered" value={formatMemory(worker.offeredVramMb / 1_024)} />
      <Row label="serving" value={deployment ? `${deployment.model} · ${deployment.mode}` : "—"} />
      <Row label="models held" value={String(worker.deployments.length || "none")} />
      {reliability !== null && <>
        <Row label="reliability" value={`${Math.round(reliability * 100)}%`} />
        <div className="net-panel-bar"><div className="fill" style={{ width: `${(reliability * 100).toFixed(1)}%` }} /></div>
      </>}
      <Row label="jobs completed" value={worker.jobsCompleted.toLocaleString("en-US")} />
      <Row label="jobs in snapshot" value={String(jobs.length)} />
      <Row label="last seen" value={relativeTime(worker.lastSeenAt)} />
    </div>
    {!located && <p className="net-panel-note">
      This node did not announce a region, so its globe position is schematic.
    </p>}
  </div>;
}

function Row({ label, value }: { label: string; value: string }) {
  return <div className="net-panel-row"><span className="kk">{label}</span><span className="vv">{value}</span></div>;
}

function describeHardware(worker: NetworkWorker): string {
  const gpu = worker.gpus[0];
  if (gpu?.model) return gpu.model.toLowerCase();
  if (worker.kind === "browser") return "browser worker";
  return worker.kind;
}

function formatMemory(gigabytes: number): string {
  if (!Number.isFinite(gigabytes) || gigabytes <= 0) return "0 GB";
  if (gigabytes >= 1_024) return `${(gigabytes / 1_024).toFixed(gigabytes >= 10_240 ? 0 : 2)} TB`;
  return `${gigabytes.toFixed(gigabytes >= 100 ? 0 : 1)} GB`;
}

function shortId(value: string): string {
  return value.length <= 20 ? value : `${value.slice(0, 10)}…${value.slice(-6)}`;
}

function relativeTime(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "unknown";
  const elapsed = Math.max(0, Date.now() - timestamp);
  if (elapsed < 60_000) return "just now";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
  return `${Math.floor(elapsed / 86_400_000)}d ago`;
}

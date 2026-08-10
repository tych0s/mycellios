import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

const EMPTY_SNAPSHOT: NetworkSnapshot = {
  capturedAt: new Date(0).toISOString(),
  version: "—",
  summary: { registered: 0, connected: 0, online: 0, mobile: 0, offeredVramMb: 0, completedJobs: 0 },
  workers: [],
  models: [],
  jobs: [],
};

const SNAPSHOT_COMMAND = "curl -s https://www.mycellios.com/public/v1/snapshot";

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
  animate = true,
}: {
  initialSnapshot?: NetworkSnapshot;
  animate?: boolean;
}) {
  const [snapshot, setSnapshot] = useState(initialSnapshot ?? EMPTY_SNAPSHOT);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(initialSnapshot === undefined);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);

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

  const metrics = useMemo(() => deriveNetworkMetrics(snapshot), [snapshot]);

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
      }, 120);
    };
    window.addEventListener("resize", onResize);

    layout();
    if (reduceMotion || !animate) draw();
    else frame = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(resizeTimer);
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerCancel);
    };
  }, [animate]);

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

  return <div className="net">
    <div className="net-stage">
      <canvas
        ref={canvasRef}
        className="net-globe"
        aria-label={`Interactive globe showing ${nodes.length} announced Mycellios nodes`}
        role="img"
      />

      <h1 className="net-visually-hidden">Mycellios network</h1>

      <a className="net-hud net-brand" href="/">
        <span className="net-brand-name">mycellios</span>
        <span className="net-brand-sub">/ network</span>
      </a>

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
        <a href="/join">connect a device →</a>
      </div>

      {selectedNode && <NodePanel
        node={selectedNode}
        jobs={snapshot.jobs.filter((job) => job.workerId === selectedNode.worker.id)}
        onClose={() => setSelectedId(null)}
      />}

      {!loading && nodes.length === 0 && <div className="net-empty">
        <strong>No nodes announced</strong>
        <span>The coordinator is reachable, but no worker is currently visible.</span>
        <a href="/join">Connect a device →</a>
      </div>}

      {error && <div className="net-error" role="alert">
        <span>{hasData ? "Live refresh failed. Showing the last successful snapshot." : error}</span>
        <button onClick={() => void refresh()}>Retry</button>
      </div>}
    </div>
  </div>;
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

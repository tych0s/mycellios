import Graph from "graphology";
import Sigma from "sigma";
import { useEffect, useRef } from "react";
import type {
  DashboardModel,
  DashboardSnapshot,
  DashboardWorker,
} from "../../desktop/contracts";

export type GraphSelection =
  | { kind: "network"; label: string }
  | { kind: "device"; label: string }
  | { kind: "model"; label: string; model: DashboardModel }
  | { kind: "worker"; label: string; worker: DashboardWorker };

interface NetworkGraphProps {
  snapshot: DashboardSnapshot;
  selectedId: string | null;
  onSelect: (id: string, selection: GraphSelection) => void;
}

interface NodeMetadata {
  selection: GraphSelection;
}

const COLORS = {
  core: "#f2d59d",
  local: "#50e6e6",
  model: "#9b67f6",
  online: "#54e3bd",
  suspect: "#f1b96e",
  draining: "#78a8ff",
  offline: "#5c6870",
};

export function NetworkGraph({ snapshot, selectedId, onSelect }: NetworkGraphProps) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current) return;
    const graph = new Graph();
    const metadata = new Map<string, NodeMetadata>();

    addNode(
      graph,
      metadata,
      "network",
      {
        x: 0,
        y: 0,
        size: selectedId === "network" ? 26 : 23,
        label: "mycellios network",
        color: COLORS.core,
        forceLabel: true,
        zIndex: 10,
      },
      { kind: "network", label: "mycellios network" },
    );

    const modelRadius = 2.7;
    snapshot.models.forEach((model, index) => {
      const angle = radialAngle(index, Math.max(1, snapshot.models.length), -Math.PI / 2);
      const id = `model:${model.id}`;
      addNode(
        graph,
        metadata,
        id,
        {
          x: Math.cos(angle) * modelRadius,
          y: Math.sin(angle) * modelRadius,
          size: selectedId === id ? 17 : 14,
          label: model.id,
          color: COLORS.model,
          forceLabel: true,
          zIndex: 7,
        },
        { kind: "model", label: model.id, model },
      );
      graph.addEdge("network", id, { size: 1.8, color: "#7964a5", zIndex: 2 });
    });

    const workerRadius = snapshot.workers.length <= 8 ? 5.2 : 6.2;
    snapshot.workers.forEach((worker, index) => {
      const angle = radialAngle(index, Math.max(1, snapshot.workers.length), -Math.PI / 2);
      const id = `worker:${worker.id}`;
      const label = worker.gpus[0]?.model ?? `${worker.region} · ${shortId(worker.id)}`;
      addNode(
        graph,
        metadata,
        id,
        {
          x: Math.cos(angle) * workerRadius,
          y: Math.sin(angle) * workerRadius,
          size: selectedId === id ? 14 : 11,
          label,
          color: COLORS[worker.status],
          forceLabel: snapshot.workers.length <= 14,
          zIndex: 5,
        },
        { kind: "worker", label, worker },
      );
      const workerModels = new Set(worker.deployments.map((deployment) => deployment.model));
      let linked = false;
      for (const model of snapshot.models) {
        if (!workerModels.has(model.id)) continue;
        graph.addEdge(`model:${model.id}`, id, {
          size: worker.connected ? 2.2 : 1,
          color: worker.connected ? "#4eaaa6" : "#39454c",
          zIndex: 1,
        });
        linked = true;
      }
      if (!linked) {
        graph.addEdge("network", id, { size: 1.2, color: "#3b6b70", zIndex: 1 });
      }
    });

    const localId = "device:local";
    const localAngle = Math.PI * 0.84;
    const localLabel = snapshot.localHardware.hostname || "This machine";
    addNode(
      graph,
      metadata,
      localId,
      {
        x: Math.cos(localAngle) * 5.2,
        y: Math.sin(localAngle) * 5.2,
        size: selectedId === localId ? 14 : 11,
        label: `${localLabel} · this machine`,
        color: COLORS.local,
        forceLabel: true,
        zIndex: 6,
      },
      { kind: "device", label: localLabel },
    );
    graph.addEdge("network", localId, {
      size: snapshot.settings.contributionEnabled ? 2.5 : 1,
      color: snapshot.settings.contributionEnabled ? "#51ded3" : "#315c62",
      zIndex: 3,
    });

    const renderer = new Sigma(graph, container.current, {
      allowInvalidContainer: false,
      defaultNodeColor: COLORS.local,
      defaultEdgeColor: "#40535a",
      enableEdgeEvents: false,
      hideEdgesOnMove: false,
      labelColor: { color: "#dce9e8" },
      labelFont: "Inter, system-ui, sans-serif",
      labelSize: 12,
      labelWeight: "500",
      minCameraRatio: 0.22,
      maxCameraRatio: 3,
      renderEdgeLabels: false,
      renderLabels: true,
      zIndex: true,
    });

    renderer.on("clickNode", ({ node }) => {
      const value = metadata.get(node);
      if (value) onSelect(node, value.selection);
    });
    renderer.on("enterNode", () => {
      if (container.current) container.current.style.cursor = "pointer";
    });
    renderer.on("leaveNode", () => {
      if (container.current) container.current.style.cursor = "grab";
    });
    renderer.getCamera().animatedReset({ duration: 650 });

    return () => renderer.kill();
  }, [snapshot, selectedId, onSelect]);

  return <div className="network-graph" ref={container} aria-label="mycellios network map" />;
}

function addNode(
  graph: Graph,
  metadata: Map<string, NodeMetadata>,
  id: string,
  attributes: Record<string, unknown>,
  selection: GraphSelection,
) {
  graph.addNode(id, attributes);
  metadata.set(id, { selection });
}

function radialAngle(index: number, count: number, offset = 0): number {
  return offset + (index / count) * Math.PI * 2;
}

function shortId(id: string): string {
  return id.length > 13 ? `${id.slice(0, 6)}…${id.slice(-4)}` : id;
}

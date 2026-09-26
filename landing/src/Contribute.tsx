import {
  Activity,
  AlertTriangle,
  ChevronDown,
  Cpu,
  Download,
  MonitorSmartphone,
  Power,
  Radio,
  RefreshCw,
  Server,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { NodeCommand } from "../../src/contracts/node-control";
import {
  getMobileWorkerSnapshot,
  initializeMobileWorker,
  setMobileWorkerLevel,
  subscribeMobileWorker,
  toggleMobileWorker,
  type MobileWorkerLevel,
  type MobileWorkerSnapshot,
} from "../../src/mobile/runtime";
import {
  acceptNodeOwnershipTransfer,
  createNodeEnrollmentBundle,
  createNodeOwnershipTransfer,
  enqueueOwnedNodeCommand,
  loadOwnedNodes,
  revokeOwnedNode,
  type OwnedNode,
} from "./api-access";
import type { AuthSession, PublicAuthConfig } from "./auth";
import { verifyTotpStepUp } from "./auth";

interface ContributeProps {
  accessToken: string | null;
  authConfig: PublicAuthConfig;
  authSession: AuthSession | null;
  getValidSession(forceRefresh?: boolean): Promise<AuthSession | null>;
  onSessionElevated(session: AuthSession): void;
  onSignIn(): void;
  onOpenAccount(): void;
  onNavigate(view: "overview" | "nodes" | "inference"): void;
  network: {
    connectedNodes: number;
    completedJobs: number;
    traceCount: number;
  };
}

export function Contribute(props: ContributeProps) {
  const [worker, setWorker] = useState<MobileWorkerSnapshot>(() =>
    getMobileWorkerSnapshot(),
  );
  const [nodes, setNodes] = useState<OwnedNode[]>([]);
  const [loadingNodes, setLoadingNodes] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const failedNodeLoadToken = useRef<string | null>(null);
  const nodeLoadInFlight = useRef(false);
  const nodeLoadBlocked = useRef(false);
  const startButtonRef = useRef<HTMLButtonElement>(null);
  const errorPresentation = error ? contributionErrorPresentation(error) : null;

  useEffect(() => {
    initializeMobileWorker();
    const unsubscribe = subscribeMobileWorker(setWorker);
    const location = new URL(window.location.href);
    if (location.searchParams.get("autostart") === "1") {
      location.searchParams.delete("autostart");
      window.history.replaceState(
        window.history.state,
        "",
        `${location.pathname}${location.search}${location.hash}`,
      );
      window.requestAnimationFrame(() => startButtonRef.current?.focus());
    }
    return unsubscribe;
  }, []);

  const refreshNodes = useCallback(async (manual = false) => {
    if (nodeLoadInFlight.current) return;
    if (manual) {
      nodeLoadBlocked.current = false;
      failedNodeLoadToken.current = null;
    } else if (nodeLoadBlocked.current) {
      return;
    }
    if (!props.authSession) {
      setNodes([]);
      return;
    }
    if (failedNodeLoadToken.current === props.authSession.accessToken) return;
    nodeLoadInFlight.current = true;
    setLoadingNodes(true);
    let attemptedToken: string | null = props.authSession.accessToken;
    try {
      let session = await props.getValidSession(false);
      if (!session) {
        setNodes([]);
        return;
      }
      try {
        setNodes(await loadOwnedNodes(session.accessToken));
      } catch (cause) {
        if (!isInvalidSessionError(cause)) throw cause;
        session = await props.getValidSession(true);
        if (!session) {
          setNodes([]);
          return;
        }
        attemptedToken = session.accessToken;
        setNodes(await loadOwnedNodes(session.accessToken));
      }
      failedNodeLoadToken.current = null;
      setError(null);
    } catch (cause) {
      failedNodeLoadToken.current = attemptedToken;
      nodeLoadBlocked.current = true;
      setError(errorText(cause));
    } finally {
      nodeLoadInFlight.current = false;
      setLoadingNodes(false);
    }
  }, [props.authSession, props.getValidSession]);

  useEffect(() => {
    void refreshNodes();
  }, [refreshNodes]);

  return (
    <section
      className={`contribute-page ${worker.running ? "is-running" : ""}`}
    >
      <div className="panel-page-title">
        <div>
          <h1>Your devices</h1>
        </div>
      </div>
      {errorPresentation && (
        <div className="contribute-error" role="alert">
          <AlertTriangle aria-hidden="true" />
          <span>
            <strong>{errorPresentation.title}</strong>
            <small>{errorPresentation.detail}</small>
          </span>
          <button
            type="button"
            onClick={() => {
              setError(null);
              if (errorPresentation.requiresAuth) props.onSignIn();
              else void refreshNodes(true);
            }}
          >
            {errorPresentation.action}
          </button>
        </div>
      )}
      {notice && (
        <div className="owned-nodes-feedback" role="status">
          {notice}
        </div>
      )}
      <BrowserContribution worker={worker} startButtonRef={startButtonRef} />
      <NativeNodeSetup
        {...props}
        nodes={nodes}
        loading={loadingNodes}
        onRefresh={() => refreshNodes(true)}
        onNotice={setNotice}
        onError={setError}
      />
    </section>
  );
}

function BrowserContribution({
  worker,
  startButtonRef,
}: {
  worker: MobileWorkerSnapshot;
  startButtonRef: React.RefObject<HTMLButtonElement | null>;
}) {
  const connected = worker.running && worker.connection === "online";
  return (
    <article className="contribute-browser-card">
      <header>
        <div className="contribute-browser-icon"><MonitorSmartphone /></div>
        <div>
          <h2>This browser</h2>
          <p>{worker.statusText}</p>
        </div>
        <span className={`state-badge ${connected ? "online" : "paused"}`}>
          {connected ? "Connected" : "Paused"}
        </span>
      </header>

      <div className="contribute-browser-action">
        <div className="contribute-browser-facts" aria-label="Browser worker details">
          <span><Cpu /><small>Engine</small><strong>{worker.backend === "webgpu" ? "WebGPU" : "CPU"}</strong></span>
          <span><Activity /><small>Work</small><strong>{worker.verifiedTasks > 0 ? `${worker.verifiedTasks} verified` : "No tasks yet"}</strong></span>
        </div>
        <button
          ref={startButtonRef}
          disabled={worker.starting}
          onClick={() => void toggleMobileWorker()}
        >
          {worker.starting ? <Radio className="spin" /> : <Power />}
          {worker.running ? "Pause contribution" : "Start contributing"}
        </button>
      </div>

      <div className="contribute-power-row">
        <span>Power</span>
        <div className="contribute-levels" role="group" aria-label="Browser power level">
          {(["low", "balanced", "maximum"] as MobileWorkerLevel[]).map((level, index) => (
            <button
              key={level}
              className={worker.level === level ? "active" : ""}
              disabled={worker.running || worker.starting}
              onClick={() => setMobileWorkerLevel(level)}
              aria-label={`${level} browser power`}
            >
              <span className="contribute-level-icon" aria-hidden="true">
                {Array.from({ length: index + 1 }, (_, ray) => <Zap key={ray} />)}
              </span>
              <span>{level}</span>
            </button>
          ))}
        </div>
      </div>
    </article>
  );
}

function NativeNodeSetup(
  props: ContributeProps & {
    nodes: OwnedNode[];
    loading: boolean;
    onRefresh(): Promise<void>;
    onNotice(value: string | null): void;
    onError(value: string | null): void;
  },
) {
  const [busy, setBusy] = useState<string | null>(null);
  const installer = detectInstaller();
  const createBundle = async () => {
    if (!props.accessToken || !props.authSession) {
      props.onSignIn();
      return;
    }
    setBusy("pair");
    props.onError(null);
    try {
      const bundle = await createNodeEnrollmentBundle(
        props.accessToken,
        props.authSession.user.id,
        window.location.origin,
      );
      const blob = new Blob([`${JSON.stringify(bundle, null, 2)}\n`], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `mycellios-pair-${bundle.enrollmentId.slice(0, 8)}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      props.onNotice(
        "Pairing bundle created and confirmed. Open it only with the native installer; it expires and can be redeemed once.",
      );
    } catch (cause) {
      props.onError(errorText(cause));
    } finally {
      setBusy(null);
    }
  };
  const acceptTransfer = async () => {
    if (!props.accessToken || !props.authSession) {
      props.onSignIn();
      return;
    }
    const transferId = window.prompt("Paste the ownership transfer ID.")?.trim();
    const transferToken = window.prompt("Paste the one-time ownership transfer token.")?.trim();
    const nodeId = window.prompt("Type the complete node ID you are accepting.")?.trim();
    const code = window.prompt("Enter the six-digit authenticator code to accept ownership.")?.trim();
    if (!transferId || !transferToken || !nodeId || !code) return;
    setBusy("accept-transfer");
    props.onError(null);
    try {
      const elevated = await verifyTotpStepUp(props.authConfig, props.authSession, code);
      props.onSessionElevated(elevated);
      await acceptNodeOwnershipTransfer(elevated.accessToken, {
        transferId, transferToken, nodeId, confirmation: nodeId,
      });
      props.onNotice(`Ownership accepted for ${nodeId.slice(0, 12)}. The node will reconnect under its new generation.`);
      await props.onRefresh();
    } catch (cause) {
      props.onError(errorText(cause));
    } finally {
      setBusy(null);
    }
  };
  return (
    <details className="contribute-native-disclosure">
      <summary>
        <span className="contribute-native-summary-icon"><Server /></span>
        <span><strong>Use a dedicated computer</strong><small>Check native package availability for persistent contribution.</small></span>
        <b className={props.nodes.length > 0 ? "" : "empty"} aria-label={`${props.nodes.length} paired nodes`}>{props.nodes.length || ""}</b>
        <ChevronDown />
      </summary>
      <div className="contribute-detail-grid">
      <article className="contribute-settings-card">
        <div className="contribute-card-head">
          <div>
            <span className="card-kicker">NATIVE NODE</span>
            <h2>Persistent contribution</h2>
          </div>
          <Server />
        </div>
        <p>
          When a native package is available, install the signed service and
          use a one-time pairing bundle. The browser never scans localhost.
        </p>
        <div className="page-actions">
          <a href="/downloads">
            <Download />
            Check {installer.label} package
          </a>
          {props.accessToken ? <>
            <button type="button" disabled={busy !== null} onClick={() => void createBundle()}>
              <ShieldCheck />
              {busy === "pair" ? "Creating…" : "Create pairing bundle"}
            </button>
            <button type="button" disabled={busy !== null} onClick={() => void acceptTransfer()}>
              <ShieldCheck />
              Accept ownership transfer
            </button>
          </> : (
            <button type="button" onClick={props.onSignIn}><ShieldCheck />Sign in to pair a node</button>
          )}
        </div>
      </article>
      <article className="contribute-activity-card">
        <div className="contribute-card-head">
          <div>
            <span className="card-kicker">OWNED NODES</span>
            <h2>Durable control</h2>
          </div>
          <button
            aria-label="Refresh nodes"
            disabled={props.loading}
            onClick={() => void props.onRefresh()}
          >
            <RefreshCw className={props.loading ? "spin" : ""} />
          </button>
        </div>
        {props.nodes.length === 0 ? (
          <p>
            {props.accessToken
              ? "No native node is paired with this account yet."
              : "Sign in to see your native nodes."}
          </p>
        ) : (
          props.nodes.map((node) => (
            <OwnedNodeCard
              key={node.nodeId}
              node={node}
              {...props}
              busy={busy}
              setBusy={setBusy}
            />
          ))
        )}
      </article>
      </div>
    </details>
  );
}

function OwnedNodeCard({
  node,
  accessToken,
  authSession,
  authConfig,
  onSessionElevated,
  onNotice,
  onError,
  onRefresh,
  busy,
  setBusy,
}: ContributeProps & {
  node: OwnedNode;
  onNotice(value: string | null): void;
  onError(value: string | null): void;
  onRefresh(): Promise<void>;
  busy: string | null;
  setBusy(value: string | null): void;
}) {
  const [limits, setLimits] = useState(
    () =>
      node.observed?.limits ??
      node.desired?.limits ?? {
        maxConcurrency: 1,
        maxCpuPercent: 80,
        maxRamMiB: 4096,
        maxVramMiB: 0,
        maxDiskMiB: 8192,
        maxTemperatureC: 85,
      },
  );
  const [policy, setPolicy] = useState(
    () =>
      node.observed?.policy ??
      node.desired?.policy ?? {
        schedule: [
          {
            days: [0, 1, 2, 3, 4, 5, 6],
            startMinuteUtc: 0,
            endMinuteUtc: 1_440,
          },
        ],
        modelAllowlist: [],
      },
  );
  const [models, setModels] = useState(() => policy.modelAllowlist.join("\n"));
  const run = async (
    type: NodeCommand["type"],
    payload: NodeCommand["payload"],
    token = accessToken,
  ) => {
    if (!token || !authSession) return;
    setBusy(`${node.nodeId}:${type}`);
    onError(null);
    try {
      await enqueueOwnedNodeCommand(token, authSession.user.id, node, {
        type,
        payload,
      } as Pick<NodeCommand, "type" | "payload">);
      onNotice(`${type} queued for ${node.nodeId.slice(0, 8)}.`);
      await onRefresh();
    } catch (cause) {
      onError(errorText(cause));
    } finally {
      setBusy(null);
    }
  };
  const revoke = async () => {
    if (!authSession) return;
    const code = window.prompt(
      "Enter the six-digit authenticator code to revoke this node.",
    );
    if (!code) return;
    const confirmation = window.prompt(
      `Type the complete node ID ${node.nodeId} to confirm permanent credential revocation.`,
    );
    if (confirmation !== node.nodeId) {
      onError("Revocation cancelled: node identity did not match.");
      return;
    }
    setBusy(`${node.nodeId}:revoke`);
    try {
      const elevated = await verifyTotpStepUp(authConfig, authSession, code);
      onSessionElevated(elevated);
      await revokeOwnedNode(
        elevated.accessToken,
        node,
        "owner_requested",
        confirmation,
      );
      onNotice("Node credential revoked and active sessions disconnected.");
      await onRefresh();
    } catch (cause) {
      onError(errorText(cause));
    } finally {
      setBusy(null);
    }
  };
  const uninstall = async () => {
    if (!authSession) return;
    const retentionChoice = window.prompt(
      "Choose exactly KEEP-DATA to remove only the service and application, or DELETE-ALL to also erase cache, logs, configuration and device identity.",
    );
    const retain = uninstallRetentionChoice(retentionChoice);
    if (!retain) {
      onError("Uninstall cancelled: choose KEEP-DATA or DELETE-ALL exactly.");
      return;
    }
    const code = window.prompt(
      "Enter the six-digit authenticator code to uninstall this node.",
    );
    if (!code) return;
    if (
      window.prompt(
        `Type the complete node ID ${node.nodeId} to confirm ${retentionChoice === "KEEP-DATA" ? "service removal while retaining cache, logs, configuration and identity" : "permanent removal of the service and all node data"}.`,
      ) !== node.nodeId
    ) {
      onError("Uninstall cancelled: node identity did not match.");
      return;
    }
    try {
      const elevated = await verifyTotpStepUp(authConfig, authSession, code);
      onSessionElevated(elevated);
      await run(
        "uninstall",
        {
          version: 1,
          retain,
        },
        elevated.accessToken,
      );
    } catch (cause) {
      onError(errorText(cause));
    }
  };
  const transferOwnership = async () => {
    if (!authSession) return;
    const targetAccountId = window.prompt("Enter the destination account ID. The destination account must accept separately.")?.trim();
    const code = window.prompt("Enter the six-digit authenticator code to authorize ownership transfer.")?.trim();
    const confirmation = window.prompt(`Type the complete node ID ${node.nodeId} to create a one-time transfer bundle.`)?.trim();
    if (!targetAccountId || !code || confirmation !== node.nodeId) {
      if (confirmation && confirmation !== node.nodeId) onError("Transfer cancelled: node identity did not match.");
      return;
    }
    setBusy(`${node.nodeId}:transfer`);
    onError(null);
    try {
      const elevated = await verifyTotpStepUp(authConfig, authSession, code);
      onSessionElevated(elevated);
      const transfer = await createNodeOwnershipTransfer(elevated.accessToken, node, targetAccountId, confirmation);
      const blob = new Blob([`${JSON.stringify(transfer, null, 2)}\n`], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `mycellios-transfer-${node.nodeId.slice(0, 8)}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      onNotice("One-time ownership transfer bundle created. Share it only with the destination account; it expires in 15 minutes.");
    } catch (cause) {
      onError(errorText(cause));
    } finally {
      setBusy(null);
    }
  };
  const pending = busy?.startsWith(node.nodeId) ?? false;
  const disabled = pending || node.status !== "active";
  const policyWindow = policy.schedule[0] ?? {
    days: [0, 1, 2, 3, 4, 5, 6],
    startMinuteUtc: 0,
    endMinuteUtc: 1_440,
  };
  const presentation = nodePresentation(node);
  return (
    <div className="owned-node-card">
      <header>
        <div>
          <strong>{node.nodeId.slice(0, 12)}</strong>
          <small>
            generation {node.generation} ·{" "}
            {node.connected ? "connected" : "offline"}
          </small>
        </div>
        <span className={`state-badge ${presentation.tone}`}>
          {presentation.label}
        </span>
      </header>
      <p>{presentation.summary}</p>
      <NodeDiagnostics node={node} />
      <div className="page-actions">
        <button
          disabled={disabled}
          onClick={() => void run("pause", { version: 1 })}
        >
          Pause
        </button>
        <button
          disabled={disabled}
          onClick={() => void run("resume", { version: 1 })}
        >
          Resume
        </button>
        <button
          disabled={disabled}
          onClick={() => void run("drain", { version: 1, deadlineMs: 300_000 })}
        >
          Drain
        </button>
        <button
          disabled={disabled}
          onClick={() => void run("update", { version: 1, channel: "stable" })}
        >
          Update
        </button>
        <button
          disabled={disabled}
          onClick={() =>
            void run("rollback", {
              version: 1,
              componentIds: ["python-product"],
            })
          }
        >
          Rollback
        </button>
        <button disabled={disabled} onClick={() => void revoke()}>
          Revoke
        </button>
        <button disabled={disabled} onClick={() => void transferOwnership()}>
          Transfer ownership
        </button>
        <button
          disabled={
            disabled || node.observed?.runtime.uninstallAvailable !== true
          }
          onClick={() => void uninstall()}
        >
          Uninstall all
        </button>
      </div>
      <details>
        <summary>
          <span>
            Resource limits
            <small>
              {limits.maxCpuPercent}% CPU · {limits.maxRamMiB} MiB RAM
            </small>
          </span>
          <ChevronDown />
        </summary>
        <div className="node-limit-grid">
          {(
            [
              ["maxConcurrency", "Concurrent jobs"],
              ["maxCpuPercent", "CPU %"],
              ["maxRamMiB", "RAM MiB"],
              ["maxVramMiB", "VRAM MiB"],
              ["maxDiskMiB", "Disk MiB"],
              ["maxTemperatureC", "Temperature °C"],
            ] as const
          ).map(([key, label]) => (
            <label key={key}>
              <span>{label}</span>
              <input
                type="number"
                min={key === "maxVramMiB" ? 0 : 1}
                value={limits[key]}
                onChange={(event) =>
                  setLimits((current) => ({
                    ...current,
                    [key]: Number(event.target.value),
                  }))
                }
              />
            </label>
          ))}
        </div>
        <button
          disabled={disabled}
          onClick={() => void run("set-limits", { version: 1, ...limits })}
        >
          Apply limits and restart safely
        </button>
      </details>
      <details>
        <summary>
          <span>
            Schedule and allowed models
            <small>
              {policyWindow.days.length} days ·{" "}
              {policy.modelAllowlist.length || "any certified"} models
            </small>
          </span>
          <ChevronDown />
        </summary>
        <p>
          Times are UTC. An empty model list permits any coordinator-certified
          model.
        </p>
        <div
          className="node-schedule-days"
          role="group"
          aria-label="Contribution days UTC"
        >
          {DAY_LABELS.map((label, day) => (
            <label key={label}>
              <input
                type="checkbox"
                checked={policyWindow.days.includes(day)}
                onChange={(event) =>
                  setPolicy((current) => ({
                    ...current,
                    schedule: [
                      {
                        ...policyWindow,
                        days: event.target.checked
                          ? [...new Set([...policyWindow.days, day])].sort()
                          : policyWindow.days.filter((value) => value !== day),
                      },
                    ],
                  }))
                }
              />
              {label}
            </label>
          ))}
        </div>
        <div className="node-limit-grid">
          <label>
            <span>Start UTC</span>
            <input
              type="time"
              value={minuteToTime(policyWindow.startMinuteUtc)}
              onChange={(event) =>
                setPolicy((current) => ({
                  ...current,
                  schedule: [
                    {
                      ...policyWindow,
                      startMinuteUtc: timeToMinute(event.target.value),
                    },
                  ],
                }))
              }
            />
          </label>
          <label>
            <span>End UTC</span>
            <input
              type="time"
              value={minuteToTime(
                policyWindow.endMinuteUtc === 1_440
                  ? 0
                  : policyWindow.endMinuteUtc,
              )}
              onChange={(event) =>
                setPolicy((current) => ({
                  ...current,
                  schedule: [
                    {
                      ...policyWindow,
                      endMinuteUtc: timeToMinute(event.target.value) || 1_440,
                    },
                  ],
                }))
              }
            />
          </label>
        </div>
        <label>
          <span>Exact model IDs, one per line</span>
          <textarea
            rows={4}
            value={models}
            onChange={(event) => setModels(event.target.value)}
            placeholder="org/model-id"
          />
        </label>
        <button
          disabled={disabled}
          onClick={() =>
            void run("set-policy", {
              version: 1,
              policy: {
                ...policy,
                modelAllowlist: [
                  ...new Set(
                    models
                      .split(/\r?\n/)
                      .map((value) => value.trim())
                      .filter(Boolean),
                  ),
                ],
              },
            })
          }
        >
          Apply policy and restart safely
        </button>
      </details>
      {node.commands[0] && (
        <small>
          Latest: {node.commands[0].type} · {node.commands[0].state}
        </small>
      )}
    </div>
  );
}

function NodeDiagnostics({ node }: { node: OwnedNode }) {
  const observed = node.observed;
  const diagnostics = observed?.diagnostics;
  if (!observed)
    return (
      <div className="node-diagnostics-empty">
        <Radio className="spin" />
        <span>
          <strong>Waiting for telemetry</strong>
          <small>The signed node snapshot has not arrived yet.</small>
        </span>
      </div>
    );
  const accelerator = diagnostics?.capacity.primaryAccelerator;
  const resources = diagnostics?.resources;
  return (
    <details
      className="node-diagnostics"
      open={observed.state === "degraded" || observed.state === "failed"}
    >
      <summary>
        <span>
          <Activity />
          <span>
            <strong>Runtime and diagnostics</strong>
            <small>
              {observed.build.version} ·{" "}
              {observed.runtime.backend ?? "runtime pending"} ·{" "}
              {diagnostics?.capacity.acceleratorCount ?? 0} accelerator(s)
            </small>
          </span>
        </span>
        <ChevronDown />
      </summary>
      <div className="node-diagnostics-grid">
        <span>
          <small>Build</small>
          <strong>{observed.build.version}</strong>
          <em>{observed.build.sourceRevision}</em>
        </span>
        <span>
          <small>Runtime</small>
          <strong>{observed.runtime.ready ? "Ready" : "Starting"}</strong>
          <em>{observed.runtime.abi}</em>
        </span>
        <span>
          <small>Accelerator</small>
          <strong>{accelerator?.name ?? "Not reported"}</strong>
          <em>
            {accelerator
              ? `${accelerator.offeredMemoryMiB} / ${accelerator.totalMemoryMiB} MiB offered`
              : "Capacity pending"}
          </em>
        </span>
        <span>
          <small>Resources</small>
          <strong>
            {resources
              ? `${resources.cpuPercent.toFixed(0)}% CPU · ${resources.ramMiB} MiB`
              : "Probe pending"}
          </strong>
          <em>
            {resources
              ? `${resources.diskMiB} MiB disk${resources.temperatureC === null ? "" : ` · ${resources.temperatureC.toFixed(0)} °C`}`
              : "No measurement yet"}
          </em>
        </span>
      </div>
      {diagnostics?.incident && (
        <div className="node-incident" role="status">
          <AlertTriangle />
          <span>
            <strong>{diagnostics.incident.code}</strong>
            <small>
              {new Date(diagnostics.incident.occurredAt).toLocaleString()}
            </small>
          </span>
        </div>
      )}
    </details>
  );
}

export function nodePresentation(
  node: Pick<OwnedNode, "connected" | "status" | "observed">,
) {
  if (node.status === "revoked")
    return {
      label: "Revoked",
      tone: "error",
      summary: "Credentials revoked. This node can no longer reconnect.",
    };
  if (!node.observed)
    return {
      label: node.connected ? "Pairing" : "Waiting",
      tone: "paused",
      summary: "Waiting for a signed node snapshot.",
    };
  if (!node.connected && node.observed.state !== "failed")
    return {
      label: "Reconnecting",
      tone: "paused",
      summary: `Build ${node.observed.build.version} · last observed ${node.observed.state}.`,
    };
  const labels: Record<typeof node.observed.state, string> = {
    loading: "Loading",
    pairing: "Pairing",
    canary: "Canary",
    ready: "Ready",
    reconnecting: "Reconnecting",
    degraded: "Degraded",
    failed: "Failed",
    revoked: "Revoked",
  };
  const state = node.observed.state;
  return {
    label: labels[state],
    tone:
      state === "ready"
        ? "online"
        : state === "failed" || state === "revoked"
          ? "error"
          : "paused",
    summary: `Build ${node.observed.build.version} · ${node.observed.runtime.backend ?? "runtime pending"} · observed ${new Date(node.observed.observedAt).toLocaleString()}.`,
  };
}

function errorText(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

export interface ContributionErrorPresentation {
  title: string;
  detail: string;
  action: string;
  requiresAuth: boolean;
}

export function contributionErrorPresentation(message: string): ContributionErrorPresentation {
  const normalized = message.toLowerCase();
  if (isInvalidSessionError(message)) {
    return {
      title: "Your contribution session has expired",
      detail: "This browser's network credential is no longer valid. Sign in again to reconnect your contribution.",
      action: "Sign in",
      requiresAuth: true,
    };
  }
  return {
    title: "Could not update contribution",
    detail: message || "The coordinator could not confirm this browser's status.",
    action: "Retry",
    requiresAuth: false,
  };
}

function isInvalidSessionError(error: unknown): boolean {
  const normalized = errorText(error).toLowerCase();
  return normalized.includes("invalid_network_token")
    || normalized.includes("invalid network token")
    || normalized.includes("invalid_access_token")
    || normalized.includes("account session is invalid")
    || normalized.includes("session is invalid or expired");
}

const DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] as const;
function timeToMinute(value: string) {
  const [hours = "0", minutes = "0"] = value.split(":");
  return Number(hours) * 60 + Number(minutes);
}
function minuteToTime(value: number) {
  return `${String(Math.floor(value / 60) % 24).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}
export function uninstallRetentionChoice(
  choice: string | null,
): { cache: boolean; logs: boolean; configuration: boolean; identity: boolean } | null {
  if (choice === "KEEP-DATA") {
    return { cache: true, logs: true, configuration: true, identity: true };
  }
  if (choice === "DELETE-ALL") {
    return { cache: false, logs: false, configuration: false, identity: false };
  }
  return null;
}
export function detectInstallerPlatform(userAgent: string, platform: string) {
  const value = `${userAgent} ${platform}`.toLowerCase();
  if (value.includes("win"))
    return { label: "Windows", path: "/downloads/windows" };
  if (value.includes("mac"))
    return { label: "macOS", path: "/downloads/macos-arm64" };
  return { label: "Linux", path: "/downloads/linux" };
}
function detectInstaller() {
  return detectInstallerPlatform(navigator.userAgent, navigator.platform);
}

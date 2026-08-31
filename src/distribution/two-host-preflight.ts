import type { AutoDistributionConfig } from "./auto-distribute.js";
import { TWO_HOST_PREFLIGHT_SCHEMA, twoHostPreflightReportSchema, type TwoHostPreflightDiagnostic, type TwoHostPreflightReport } from "../contracts/two-host-preflight.js";

export interface TwoHostPreflightCapabilities {
  python312: boolean;
  configuredEnvironmentNames: ReadonlySet<string>;
  preparedStageRanges?: readonly { nodeId: string; startLayer: number; endLayer: number }[];
}

export function evaluateTwoHostPreflight(config: AutoDistributionConfig, capabilities: TwoHostPreflightCapabilities): TwoHostPreflightReport {
  const diagnostics: TwoHostPreflightDiagnostic[] = [];
  const error = (code: TwoHostPreflightDiagnostic["code"], subject: string, message: string) => diagnostics.push({ code, severity: "error", subject, message });
  const warning = (code: TwoHostPreflightDiagnostic["code"], subject: string, message: string) => diagnostics.push({ code, severity: "warning", subject, message });

  if (config.nodes.length !== 2) error("node_count_invalid", "nodes", "The first physical gate requires exactly two configured nodes.");
  if (new Set(config.nodes.map((node) => node.id)).size !== config.nodes.length) error("node_identity_duplicate", "nodes", "Node identities must be distinct.");
  if (new Set(config.nodes.map((node) => `${node.endpoint.host}:${node.endpoint.port}`)).size !== config.nodes.length) error("stage_endpoint_duplicate", "nodes", "Stage endpoints must be distinct.");

  const nodes = config.nodes.map((node) => {
    const endpointClass = classifyHost(node.endpoint.host);
    if (endpointClass === "loopback") error("stage_endpoint_loopback", node.id, "A physical node cannot use a loopback stage endpoint.");
    if (endpointClass === "placeholder") error("stage_endpoint_placeholder", node.id, "Replace documentation-only addresses before a physical run.");
    return { id: node.id, remote: node.agent.kind !== "local", endpointClass };
  });
  const remoteNodes = config.nodes.filter((node) => node.agent.kind === "http");
  if (remoteNodes.length === 0) error("remote_agent_missing", "nodes", "At least one node must use an authenticated remote HTTP launch agent.");
  for (const node of config.nodes) {
    if (node.agent.kind !== "http") continue;
    if (!node.agent.endpoint.startsWith("https://") && !isPrivateHttp(node.agent.endpoint)) {
      error("remote_agent_insecure", node.id, "Remote agent HTTP is allowed only on a private address; otherwise use HTTPS.");
    }
    if (!node.agent.authTokenEnv || !capabilities.configuredEnvironmentNames.has(node.agent.authTokenEnv)) {
      error("remote_auth_missing", node.id, `Configure the environment variable named by authTokenEnv for ${node.id}.`);
    }
  }
  if (!capabilities.python312) error("python_runtime_unavailable", "runtime", "Python 3.12 is required on both hosts.");
  if (config.model.revision === null) warning("model_revision_unpinned", "model", "Pin an immutable model revision before accepting physical evidence.");

  const ranges = capabilities.preparedStageRanges ?? [];
  const preparedRanges = ranges.length === config.nodes.length && ranges.every((range) => range.endLayer > range.startLayer)
    && new Set(ranges.map((range) => range.nodeId)).size === config.nodes.length;
  if (!preparedRanges) warning("stage_range_unproven", "artifacts", "Run prepare-only and preflight the generated non-empty stage ranges before launch.");
  const measuredTransport = config.links.length >= 2 && config.links.every((link) => link.evidence?.source === "runtime-probe");
  if (!measuredTransport) warning("transport_unmeasured", "links", "Configured link estimates are planning inputs, not measured transport evidence.");

  const remoteAuthentication = remoteNodes.length > 0 && config.nodes.filter((node) => node.agent.kind === "http").every((node) => {
    if (node.agent.kind !== "http") return false;
    return Boolean(node.agent.authTokenEnv && capabilities.configuredEnvironmentNames.has(node.agent.authTokenEnv));
  });
  const dryRun = nodes.some((node) => node.endpointClass === "placeholder");
  return twoHostPreflightReportSchema.parse({ schema: TWO_HOST_PREFLIGHT_SCHEMA,
    ready: !diagnostics.some((diagnostic) => diagnostic.severity === "error"), dryRun,
    model: { source: config.model.source, revisionPinned: config.model.revision !== null }, nodes,
    capabilities: { python312: capabilities.python312, remoteAuthentication, measuredTransport, preparedRanges }, diagnostics });
}

function classifyHost(host: string): "routable" | "loopback" | "placeholder" {
  const normalized = host.trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (normalized === "localhost" || normalized === "::1" || normalized.startsWith("127.")) return "loopback";
  const octets = normalized.split(".").map(Number);
  if (octets.length === 4 && octets.every((octet) => Number.isInteger(octet) && octet >= 0 && octet <= 255)) {
    if ((octets[0] === 192 && octets[1] === 0 && octets[2] === 2)
      || (octets[0] === 198 && octets[1] === 51 && octets[2] === 100)
      || (octets[0] === 203 && octets[1] === 0 && octets[2] === 113)) return "placeholder";
  }
  return "routable";
}

function isPrivateHttp(endpoint: string): boolean {
  try {
    const url = new URL(endpoint);
    const host = url.hostname;
    return url.protocol === "http:" && (/^10\./.test(host) || /^192\.168\./.test(host)
      || /^172\.(?:1[6-9]|2\d|3[01])\./.test(host));
  } catch { return false; }
}

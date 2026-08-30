import { createSocket, type RemoteInfo, type Socket as DatagramSocket } from "node:dgram";
import { networkInterfaces } from "node:os";

const SSDP_ADDRESS = "239.255.255.250";
const SSDP_PORT = 1_900;
const DEFAULT_DISCOVERY_TIMEOUT_MS = 700;
const DEFAULT_REQUEST_TIMEOUT_MS = 1_500;
const MAX_DISCOVERY_RESPONSES = 16;
const MAX_SSDP_PACKET_BYTES = 16 * 1024;
const MAX_DESCRIPTION_BYTES = 256 * 1024;
const MAX_SOAP_BYTES = 64 * 1024;

export interface UpnpGatewayLocation {
  location: string;
  localAddress: string;
  gatewayAddress: string;
}

export type UpnpGatewayDiscovery = (
  timeoutMs: number,
) => Promise<UpnpGatewayLocation[]>;

export type UpnpFetch = (
  input: string | URL,
  init?: RequestInit,
) => Promise<Response>;

export interface TcpPortMapping {
  externalHost: string;
  externalPort: number;
  /**
   * Optional lease-loss signal. A mapper must fire it only after the public
   * endpoint can no longer be renewed with the original identity.
   */
  onInvalidated?(listener: () => void): () => void;
  close(): Promise<void>;
}

export interface TcpPortMapper {
  mapTcpPort(input: {
    internalPort: number;
    internalHosts: string[];
    description: string;
  }): Promise<TcpPortMapping | null>;
}

export interface NativeUpnpPortMapperOptions {
  discovery?: UpnpGatewayDiscovery;
  fetch?: UpnpFetch;
  discoveryTimeoutMs?: number;
  requestTimeoutMs?: number;
}

interface UpnpService {
  serviceType: string;
  controlUrl: URL;
  supportsAddAny: boolean;
}

/**
 * Small, native UPnP IGD client used only to make the already authenticated
 * Mycellios direct listener reachable. It never trusts DNS names or redirects:
 * discovery, device description and SOAP control all stay on the private IPv4
 * router that answered SSDP.
 */
export class NativeUpnpPortMapper implements TcpPortMapper {
  private readonly discovery: UpnpGatewayDiscovery;
  private readonly request: UpnpFetch;
  private readonly discoveryTimeoutMs: number;
  private readonly requestTimeoutMs: number;

  constructor(options: NativeUpnpPortMapperOptions = {}) {
    this.discovery = options.discovery ?? discoverUpnpGateways;
    this.request = options.fetch ?? globalThis.fetch;
    this.discoveryTimeoutMs = boundedInteger(
      options.discoveryTimeoutMs ?? DEFAULT_DISCOVERY_TIMEOUT_MS,
      250,
      3_000,
      "upnp_discovery_timeout_is_invalid",
    );
    this.requestTimeoutMs = boundedInteger(
      options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      250,
      5_000,
      "upnp_request_timeout_is_invalid",
    );
  }

  async mapTcpPort(input: {
    internalPort: number;
    internalHosts: string[];
    description: string;
  }): Promise<TcpPortMapping | null> {
    const internalPort = boundedPort(input.internalPort);
    const allowedInternalHosts = new Set(
      input.internalHosts.filter(isPrivateIpv4),
    );
    if (allowedInternalHosts.size === 0) return null;
    const locations = await this.discovery(this.discoveryTimeoutMs);
    for (const discovered of locations.slice(0, MAX_DISCOVERY_RESPONSES)) {
      if (
        !allowedInternalHosts.has(discovered.localAddress)
        || !isPrivateIpv4(discovered.gatewayAddress)
      ) continue;
      let location: URL;
      try {
        location = validateRouterUrl(
          new URL(discovered.location),
          discovered.gatewayAddress,
        );
      } catch {
        continue;
      }
      try {
        const descriptionXml = await this.fetchBounded(
          location,
          { method: "GET", redirect: "manual" },
          MAX_DESCRIPTION_BYTES,
        );
        const service = parseControlService(descriptionXml, location);
        const externalHost = await this.getExternalAddress(service);
        if (!isPublicIpv4(externalHost)) {
          throw new Error("upnp_external_address_is_not_public");
        }
        const externalPort = await this.addMapping({
          service,
          internalHost: discovered.localAddress,
          internalPort,
          description: sanitizeDescription(input.description),
        });
        if (!Number.isSafeInteger(externalPort) || externalPort < 1 || externalPort > 65_535) {
          throw new Error("upnp_external_port_is_invalid");
        }
        let closed = false;
        return {
          externalHost,
          externalPort,
          close: async () => {
            if (closed) return;
            closed = true;
            await this.deleteMapping(service, externalPort);
          },
        };
      } catch {
        // A malformed or unreachable IGD is not evidence of public reachability.
        // Try another bounded SSDP result and otherwise leave relay available.
      }
    }
    return null;
  }

  private async getExternalAddress(service: UpnpService): Promise<string> {
    const xml = await this.soap(service, "GetExternalIPAddress", "");
    const address = extractXmlText(xml, "NewExternalIPAddress");
    if (!address) throw new Error("upnp_external_address_is_missing");
    return address;
  }

  private async addMapping(input: {
    service: UpnpService;
    internalHost: string;
    internalPort: number;
    description: string;
  }): Promise<number> {
    if (input.service.supportsAddAny) {
      try {
        const response = await this.soap(
          input.service,
          "AddAnyPortMapping",
          soapFields({
            NewRemoteHost: "",
            NewExternalPort: "0",
            NewProtocol: "TCP",
            NewInternalPort: String(input.internalPort),
            NewInternalClient: input.internalHost,
            NewEnabled: "1",
            NewPortMappingDescription: input.description,
            NewLeaseDuration: "0",
          }),
        );
        const allocated = Number(extractXmlText(response, "NewReservedPort"));
        return boundedPort(allocated);
      } catch {
        // Some devices advertise WANIPConnection:2 but reject AddAny. A fixed
        // mapping is still standards-compliant and remains fully validated.
      }
    }
    await this.soap(
      input.service,
      "AddPortMapping",
      soapFields({
        NewRemoteHost: "",
        NewExternalPort: String(input.internalPort),
        NewProtocol: "TCP",
        NewInternalPort: String(input.internalPort),
        NewInternalClient: input.internalHost,
        NewEnabled: "1",
        NewPortMappingDescription: input.description,
        NewLeaseDuration: "0",
      }),
    );
    return input.internalPort;
  }

  private async deleteMapping(service: UpnpService, externalPort: number): Promise<void> {
    await this.soap(
      service,
      "DeletePortMapping",
      soapFields({
        NewRemoteHost: "",
        NewExternalPort: String(boundedPort(externalPort)),
        NewProtocol: "TCP",
      }),
    );
  }

  private async soap(service: UpnpService, action: string, fields: string): Promise<string> {
    const envelope =
      "<?xml version=\"1.0\"?>"
      + "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" "
      + "s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">"
      + `<s:Body><u:${action} xmlns:u="${xmlEscape(service.serviceType)}">`
      + fields
      + `</u:${action}></s:Body></s:Envelope>`;
    return this.fetchBounded(
      service.controlUrl,
      {
        method: "POST",
        redirect: "manual",
        headers: {
          "content-type": "text/xml; charset=\"utf-8\"",
          soapaction: `"${service.serviceType}#${action}"`,
        },
        body: envelope,
      },
      MAX_SOAP_BYTES,
    );
  }

  private async fetchBounded(
    url: URL,
    init: RequestInit,
    maximumBytes: number,
  ): Promise<string> {
    const controller = new AbortController();
    let timer: NodeJS.Timeout | undefined;
    try {
      const response = await Promise.race([
        this.request(url, { ...init, signal: controller.signal }),
        new Promise<never>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error("upnp_request_timed_out")),
            this.requestTimeoutMs,
          );
          timer.unref();
        }),
      ]);
      if (response.status !== 200) throw new Error("upnp_request_failed");
      const declaredLength = Number(response.headers.get("content-length"));
      if (Number.isFinite(declaredLength) && declaredLength > maximumBytes) {
        throw new Error("upnp_response_is_too_large");
      }
      return await readResponseBody(response, maximumBytes);
    } finally {
      if (timer) clearTimeout(timer);
      controller.abort();
    }
  }
}

export async function discoverUpnpGateways(
  timeoutMs = DEFAULT_DISCOVERY_TIMEOUT_MS,
): Promise<UpnpGatewayLocation[]> {
  const boundedTimeout = boundedInteger(
    timeoutMs,
    250,
    3_000,
    "upnp_discovery_timeout_is_invalid",
  );
  const localAddresses = privateInterfaceAddresses();
  if (localAddresses.length === 0) return [];
  const results: UpnpGatewayLocation[] = [];
  const sockets: DatagramSocket[] = [];
  const search = Buffer.from([
    "M-SEARCH * HTTP/1.1",
    `HOST: ${SSDP_ADDRESS}:${SSDP_PORT}`,
    "MAN: \"ssdp:discover\"",
    "MX: 1",
    "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1",
    "",
    "",
  ].join("\r\n"), "ascii");

  await Promise.all(localAddresses.map(async (localAddress) => {
    const socket = createSocket("udp4");
    sockets.push(socket);
    socket.on("message", (message, remote) => {
      const discovered = parseSsdpResponse(message, remote, localAddress);
      if (
        discovered
        && results.length < MAX_DISCOVERY_RESPONSES
        && !results.some((entry) =>
          entry.location === discovered.location
          && entry.localAddress === discovered.localAddress
        )
      ) results.push(discovered);
    });
    await new Promise<void>((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      socket.once("error", finish);
      socket.bind(0, localAddress, () => {
        try {
          socket.setMulticastInterface(localAddress);
          socket.setMulticastTTL(2);
          socket.send(search, SSDP_PORT, SSDP_ADDRESS, finish);
        } catch {
          finish();
        }
      });
    });
  }));

  await new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, boundedTimeout);
    timer.unref();
  });
  for (const socket of sockets) {
    try {
      socket.close();
    } catch {
      // Already closed after a local socket failure.
    }
  }
  return results;
}

function parseSsdpResponse(
  message: Buffer,
  remote: RemoteInfo,
  localAddress: string,
): UpnpGatewayLocation | null {
  if (
    message.byteLength < 1
    || message.byteLength > MAX_SSDP_PACKET_BYTES
    || !isPrivateIpv4(remote.address)
    || !isPrivateIpv4(localAddress)
  ) return null;
  const text = message.toString("latin1");
  if (!/^HTTP\/1\.[01] 200(?:\s|$)/i.test(text)) return null;
  const headers = new Map<string, string>();
  for (const line of text.split(/\r?\n/).slice(1, 128)) {
    const separator = line.indexOf(":");
    if (separator < 1) continue;
    headers.set(line.slice(0, separator).trim().toLowerCase(), line.slice(separator + 1).trim());
  }
  const locationValue = headers.get("location");
  if (!locationValue) return null;
  try {
    const location = validateRouterUrl(new URL(locationValue), remote.address);
    return {
      location: location.toString(),
      localAddress,
      gatewayAddress: remote.address,
    };
  } catch {
    return null;
  }
}

function parseControlService(xml: string, location: URL): UpnpService {
  const services = [...xml.matchAll(/<service\b[^>]*>([\s\S]*?)<\/service\s*>/gi)]
    .slice(0, 64)
    .map((match) => {
      const body = match[1] ?? "";
      return {
        serviceType: extractXmlText(body, "serviceType"),
        controlUrl: extractXmlText(body, "controlURL"),
      };
    })
    .filter((entry) =>
      entry.serviceType !== null
      && entry.controlUrl !== null
      && /^urn:schemas-upnp-org:service:(?:WANIP|WANPPP)Connection:[12]$/.test(
        entry.serviceType,
      )
    )
    .sort((left, right) =>
      Number(right.serviceType?.endsWith(":2")) - Number(left.serviceType?.endsWith(":2"))
    );
  const selected = services[0];
  if (!selected?.serviceType || !selected.controlUrl) {
    throw new Error("upnp_control_service_is_missing");
  }
  const controlUrl = new URL(decodeXmlText(selected.controlUrl), location);
  if (
    controlUrl.protocol !== "http:"
    || controlUrl.username !== ""
    || controlUrl.password !== ""
    || controlUrl.hostname !== location.hostname
    || !isPrivateIpv4(controlUrl.hostname)
  ) {
    throw new Error("upnp_control_url_is_untrusted");
  }
  return {
    serviceType: selected.serviceType,
    controlUrl,
    supportsAddAny: selected.serviceType ===
      "urn:schemas-upnp-org:service:WANIPConnection:2",
  };
}

function validateRouterUrl(url: URL, expectedAddress: string): URL {
  if (
    url.protocol !== "http:"
    || url.username !== ""
    || url.password !== ""
    || url.hostname !== expectedAddress
    || !isPrivateIpv4(url.hostname)
    || url.port === "0"
  ) {
    throw new Error("upnp_location_is_untrusted");
  }
  return url;
}

function privateInterfaceAddresses(): string[] {
  return Array.from(new Set(
    Object.values(networkInterfaces()).flatMap((addresses) =>
      (addresses ?? [])
        .filter((address) =>
          address.family === "IPv4"
          && !address.internal
          && isPrivateIpv4(address.address)
        )
        .map((address) => address.address)
    ),
  ));
}

export function isPrivateIpv4(host: string): boolean {
  const octets = ipv4Octets(host);
  if (!octets) return false;
  const [a, b] = octets;
  return (
    a === 10
    || (a === 172 && b! >= 16 && b! <= 31)
    || (a === 192 && b === 168)
  );
}

export function isPublicIpv4(host: string): boolean {
  const octets = ipv4Octets(host);
  if (!octets) return false;
  const [a, b, c] = octets;
  if (
    a === 0
    || a === 10
    || a === 127
    || (a === 100 && b! >= 64 && b! <= 127)
    || (a === 169 && b === 254)
    || (a === 172 && b! >= 16 && b! <= 31)
    || (a === 192 && b === 168)
    || (a === 192 && b === 0 && c === 0)
    || (a === 192 && b === 0 && c === 2)
    || (a === 198 && (b === 18 || b === 19))
    || (a === 198 && b === 51 && c === 100)
    || (a === 203 && b === 0 && c === 113)
    || a! >= 224
  ) return false;
  return true;
}

function ipv4Octets(host: string): [number, number, number, number] | null {
  const parts = host.split(".");
  if (parts.length !== 4) return null;
  const octets = parts.map((part) => {
    if (!/^(?:0|[1-9]\d{0,2})$/.test(part)) return Number.NaN;
    return Number(part);
  });
  if (octets.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return null;
  return octets as [number, number, number, number];
}

async function readResponseBody(response: Response, maximumBytes: number): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximumBytes) throw new Error("upnp_response_is_too_large");
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder("utf-8", { fatal: true }).decode(body);
}

function extractXmlText(xml: string, localName: string): string | null {
  const escaped = localName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(
    `<(?:[A-Za-z_][\\w.-]*:)?${escaped}\\b[^>]*>([^<]{0,4096})<\\/(?:[A-Za-z_][\\w.-]*:)?${escaped}\\s*>`,
    "i",
  ).exec(xml);
  return match?.[1]?.trim() ?? null;
}

function decodeXmlText(value: string): string {
  if (/&(?!(?:amp|lt|gt|quot|apos);)/i.test(value)) {
    throw new Error("upnp_xml_entity_is_invalid");
  }
  const entities: Record<string, string> = {
    amp: "&",
    lt: "<",
    gt: ">",
    quot: "\"",
    apos: "'",
  };
  return value.replace(/&(amp|lt|gt|quot|apos);/gi, (_entity, name: string) => (
    entities[name.toLowerCase()]!
  ));
}

function soapFields(fields: Record<string, string>): string {
  return Object.entries(fields)
    .map(([name, value]) => `<${name}>${xmlEscape(value)}</${name}>`)
    .join("");
}

function xmlEscape(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&apos;");
}

function sanitizeDescription(value: string): string {
  const normalized = value.replace(/[\u0000-\u001f\u007f]/g, "").trim().slice(0, 48);
  return normalized || "Mycellios direct transport";
}

function boundedPort(value: number): number {
  return boundedInteger(value, 1, 65_535, "upnp_port_is_invalid");
}

function boundedInteger(value: number, minimum: number, maximum: number, code: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(code);
  return value;
}

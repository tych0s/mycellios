import { describe, expect, it, vi } from "vitest";
import {
  orderDirectCandidates,
  RuntimeDirectTransport,
  type RuntimeDirectTransportCallbacks,
} from "../src/worker/runtime-direct-transport.js";
import {
  NativeUpnpPortMapper,
  type TcpPortMapper,
  type TcpPortMapping,
  type UpnpFetch,
} from "../src/worker/upnp-port-mapper.js";

const DEVICE_DESCRIPTION = `<?xml version="1.0"?>
<root><device><serviceList><service>
  <serviceType>urn:schemas-upnp-org:service:WANIPConnection:2</serviceType>
  <controlURL>/upnp/control/wanip</controlURL>
</service></serviceList></device></root>`;

describe("native UPnP IGD TCP port mapping", () => {
  it("announces only a validated public mapping and deletes it on close", async () => {
    const actions: string[] = [];
    const request: UpnpFetch = vi.fn(async (input, init) => {
      const url = new URL(input);
      expect(url.hostname).toBe("192.168.1.1");
      const action = new Headers(init?.headers).get("soapaction");
      if (!action) return xml(DEVICE_DESCRIPTION);
      actions.push(action);
      if (action.endsWith("#GetExternalIPAddress\"")) {
        return xml("<NewExternalIPAddress>8.8.8.8</NewExternalIPAddress>");
      }
      if (action.endsWith("#AddAnyPortMapping\"")) {
        expect(String(init?.body)).toContain("<NewInternalClient>192.168.1.20</NewInternalClient>");
        return xml("<NewReservedPort>45678</NewReservedPort>");
      }
      if (action.endsWith("#DeletePortMapping\"")) {
        expect(String(init?.body)).toContain("<NewExternalPort>45678</NewExternalPort>");
        return xml("<DeletePortMappingResponse/>");
      }
      return new Response("", { status: 500 });
    });
    const mapper = new NativeUpnpPortMapper({
      discovery: async () => [{
        location: "http://192.168.1.1:1900/igd.xml",
        localAddress: "192.168.1.20",
        gatewayAddress: "192.168.1.1",
      }],
      fetch: request,
      discoveryTimeoutMs: 250,
      requestTimeoutMs: 250,
    });
    const mapping = await mapper.mapTcpPort({
      internalPort: 34_567,
      internalHosts: ["192.168.1.20"],
      description: "Mycellios test",
    });
    expect(mapping).toMatchObject({
      externalHost: "8.8.8.8",
      externalPort: 45_678,
    });
    await mapping?.close();
    await mapping?.close();
    expect(actions.filter((action) => action.endsWith("#DeletePortMapping\""))).toHaveLength(1);
  });

  it("uses the fixed AddPortMapping contract for IGD v1 routers", async () => {
    const actions: string[] = [];
    const request: UpnpFetch = vi.fn(async (_input, init) => {
      const action = new Headers(init?.headers).get("soapaction");
      if (!action) {
        return xml(DEVICE_DESCRIPTION.replace(
          "WANIPConnection:2",
          "WANIPConnection:1",
        ));
      }
      actions.push(action);
      if (action.endsWith("#GetExternalIPAddress\"")) {
        return xml("<NewExternalIPAddress>1.1.1.1</NewExternalIPAddress>");
      }
      return xml("<ok/>");
    });
    const mapper = new NativeUpnpPortMapper({
      discovery: gatewayDiscovery,
      fetch: request,
      discoveryTimeoutMs: 250,
      requestTimeoutMs: 250,
    });
    const mapping = await mapper.mapTcpPort({
      internalPort: 34_567,
      internalHosts: ["192.168.1.20"],
      description: "Mycellios test",
    });
    expect(mapping?.externalPort).toBe(34_567);
    expect(actions.some((action) => action.endsWith("#AddPortMapping\""))).toBe(true);
    expect(actions.some((action) => action.endsWith("#AddAnyPortMapping\""))).toBe(false);
    await mapping?.close();
  });

  it("rejects private metadata locations, cross-host control URLs and non-public WAN addresses", async () => {
    const noFetch = vi.fn<UpnpFetch>();
    const metadataMapper = new NativeUpnpPortMapper({
      discovery: async () => [{
        location: "http://169.254.169.254/latest/meta-data",
        localAddress: "192.168.1.20",
        gatewayAddress: "192.168.1.1",
      }],
      fetch: noFetch,
      discoveryTimeoutMs: 250,
      requestTimeoutMs: 250,
    });
    await expect(metadataMapper.mapTcpPort({
      internalPort: 34_567,
      internalHosts: ["192.168.1.20"],
      description: "Mycellios test",
    })).resolves.toBeNull();
    expect(noFetch).not.toHaveBeenCalled();

    const crossHostFetch: UpnpFetch = vi.fn(async (_input, init) =>
      init?.method === "GET"
        ? xml(DEVICE_DESCRIPTION.replace(
          "/upnp/control/wanip",
          "http://192.168.1.2/upnp/control/wanip",
        ))
        : xml("")
    );
    const crossHostMapper = new NativeUpnpPortMapper({
      discovery: gatewayDiscovery,
      fetch: crossHostFetch,
      discoveryTimeoutMs: 250,
      requestTimeoutMs: 250,
    });
    await expect(crossHostMapper.mapTcpPort({
      internalPort: 34_567,
      internalHosts: ["192.168.1.20"],
      description: "Mycellios test",
    })).resolves.toBeNull();
    expect(crossHostFetch).toHaveBeenCalledTimes(1);

    const cgnatFetch: UpnpFetch = vi.fn(async (_input, init) => {
      const action = new Headers(init?.headers).get("soapaction");
      return action
        ? xml("<NewExternalIPAddress>100.64.1.2</NewExternalIPAddress>")
        : xml(DEVICE_DESCRIPTION);
    });
    const cgnatMapper = new NativeUpnpPortMapper({
      discovery: gatewayDiscovery,
      fetch: cgnatFetch,
      discoveryTimeoutMs: 250,
      requestTimeoutMs: 250,
    });
    await expect(cgnatMapper.mapTcpPort({
      internalPort: 34_567,
      internalHosts: ["192.168.1.20"],
      description: "Mycellios test",
    })).resolves.toBeNull();
    expect(cgnatFetch).toHaveBeenCalledTimes(2);
  });

  it("does not advertise a false public candidate when mapping fails", async () => {
    const mapper: TcpPortMapper = {
      mapTcpPort: vi.fn(async () => {
        throw new Error("router_offline");
      }),
    };
    const transport = new RuntimeDirectTransport(
      "node-a",
      callbacks(),
      {
        listenHost: "127.0.0.1",
        candidateHosts: ["192.168.1.20"],
        portMapper: mapper,
      },
    );
    const advertisement = await transport.start();
    expect(advertisement?.candidates).toContainEqual(
      expect.objectContaining({ host: "192.168.1.20", scope: "configured" }),
    );
    expect(advertisement?.candidates.some(({ scope }) => scope === "public-mapped")).toBe(false);
    await transport.close();
  });

  it("publishes an injected mapping and cleans it with the listener lifecycle", async () => {
    const close = vi.fn(async () => undefined);
    const mapper: TcpPortMapper = {
      mapTcpPort: vi.fn(async () => ({
        externalHost: "8.8.4.4",
        externalPort: 51_234,
        close,
      })),
    };
    const transport = new RuntimeDirectTransport(
      "node-a",
      callbacks(),
      {
        listenHost: "127.0.0.1",
        candidateHosts: ["192.168.1.20"],
        portMapper: mapper,
      },
    );
    const advertisement = await transport.start();
    expect(advertisement?.candidates).toContainEqual({
      host: "8.8.4.4",
      port: 51_234,
      scope: "public-mapped",
    });
    await transport.close();
    await transport.close();
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("withdraws an expired public lease from the live advertisement", async () => {
    let invalidate!: () => void;
    const mapping: TcpPortMapping = {
      externalHost: "8.8.4.4",
      externalPort: 51_234,
      onInvalidated: (listener) => {
        invalidate = listener;
        return () => undefined;
      },
      close: vi.fn(async () => undefined),
    };
    const mapper: TcpPortMapper = {
      mapTcpPort: vi.fn(async () => mapping),
    };
    const changed = vi.fn();
    const transport = new RuntimeDirectTransport(
      "node-a",
      { ...callbacks(), onAdvertisementChanged: changed },
      {
        listenHost: "127.0.0.1",
        candidateHosts: ["192.168.1.20"],
        portMapper: mapper,
      },
    );
    expect((await transport.start())?.candidates.some(
      ({ scope }) => scope === "public-mapped",
    )).toBe(true);
    invalidate();
    expect((await transport.start())?.candidates.some(
      ({ scope }) => scope === "public-mapped",
    )).toBe(false);
    expect(changed).toHaveBeenCalledWith(expect.objectContaining({
      candidates: expect.not.arrayContaining([
        expect.objectContaining({ scope: "public-mapped" }),
      ]),
    }));
    await transport.close();
  });

  it("orders same-LAN first and a mapped WAN endpoint before foreign LAN addresses", () => {
    const candidates = [
      { host: "10.20.0.8", port: 40_000, scope: "lan" as const },
      { host: "8.8.8.8", port: 40_001, scope: "public-mapped" as const },
      { host: "192.168.1.30", port: 40_002, scope: "lan" as const },
    ];
    expect(orderDirectCandidates(candidates, ["192.168.1.10"]).map(({ host }) => host))
      .toEqual(["192.168.1.30", "8.8.8.8", "10.20.0.8"]);
    expect(orderDirectCandidates(candidates, ["172.16.5.10"]).map(({ host }) => host))
      .toEqual(["8.8.8.8", "10.20.0.8", "192.168.1.30"]);
  });
});

async function gatewayDiscovery() {
  return [{
    location: "http://192.168.1.1:1900/igd.xml",
    localAddress: "192.168.1.20",
    gatewayAddress: "192.168.1.1",
  }];
}

function xml(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/xml; charset=utf-8" },
  });
}

function callbacks(): RuntimeDirectTransportCallbacks {
  return {
    send: vi.fn(() => true),
    isTargetPortAuthorized: vi.fn(() => true),
    onSourceCommitted: vi.fn(),
    onSourceFailed: vi.fn(),
    onSourceClosed: vi.fn(),
  };
}

import {
  isPublicIpv4,
  NativeUpnpPortMapper,
  type NativeUpnpPortMapperOptions,
  type TcpPortMapper,
  type TcpPortMapping,
} from "./upnp-port-mapper.js";
import {
  PcpNatPmpPortMapper,
  type PcpNatPmpPortMapperOptions,
} from "./pcp-nat-pmp-port-mapper.js";

export interface NativeTcpPortMapperOptions {
  upnp?: NativeUpnpPortMapperOptions;
  pcpNatPmp?: PcpNatPmpPortMapperOptions;
}

/**
 * Deterministic native mapping chain. UPnP remains first because a successful
 * IGD SOAP lease is explicit and already broadly deployed. PCP (nonce-bound)
 * and then NAT-PMP are used only when UPnP yields no verified mapping.
 */
export class NativeTcpPortMapper implements TcpPortMapper {
  private readonly composite: CompositeTcpPortMapper;

  constructor(options: NativeTcpPortMapperOptions = {}) {
    this.composite = new CompositeTcpPortMapper([
      new NativeUpnpPortMapper(options.upnp),
      new PcpNatPmpPortMapper(options.pcpNatPmp),
    ]);
  }

  mapTcpPort(input: {
    internalPort: number;
    internalHosts: string[];
    description: string;
  }): Promise<TcpPortMapping | null> {
    return this.composite.mapTcpPort(input);
  }
}

export class CompositeTcpPortMapper implements TcpPortMapper {
  constructor(private readonly mappers: TcpPortMapper[]) {
    if (mappers.length < 1 || mappers.length > 8) {
      throw new Error("native_port_mapper_chain_is_invalid");
    }
  }

  async mapTcpPort(input: {
    internalPort: number;
    internalHosts: string[];
    description: string;
  }): Promise<TcpPortMapping | null> {
    for (const mapper of this.mappers) {
      try {
        const mapping = await mapper.mapTcpPort(input);
        if (!mapping) continue;
        if (
          !isPublicIpv4(mapping.externalHost)
          || !Number.isSafeInteger(mapping.externalPort)
          || mapping.externalPort < 1
          || mapping.externalPort > 65_535
        ) {
          await mapping.close().catch(() => undefined);
          continue;
        }
        return mapping;
      } catch {
        // Mapping is optional; the next native protocol or relay owns fallback.
      }
    }
    return null;
  }
}

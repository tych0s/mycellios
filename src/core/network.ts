import type { NetworkInterfaceInfo } from "node:os";

/** Prefer physical LAN adapters over VPN/overlay addresses for stage traffic. */
export function selectPreferredLanAddress(
  interfaces: NodeJS.Dict<NetworkInterfaceInfo[]>,
): string {
  const candidates = Object.entries(interfaces).flatMap(([name, addresses]) =>
    (addresses ?? [])
      .filter((address) =>
        address.family === "IPv4"
        && !address.internal
        && !address.address.startsWith("169.254."),
      )
      .map((address) => ({ name, address: address.address })),
  );
  candidates.sort((left, right) =>
    adapterScore(left.name, left.address) - adapterScore(right.name, right.address)
    || left.name.localeCompare(right.name)
    || left.address.localeCompare(right.address),
  );
  return candidates[0]?.address ?? "127.0.0.1";
}

function adapterScore(name: string, address: string): number {
  if (/wi-?fi|wireless|ethernet/i.test(name)) return 0;
  if (address.startsWith("192.168.")) return 10;
  const second = Number(address.split(".")[1]);
  if (address.startsWith("172.") && second >= 16 && second <= 31) return 20;
  if (address.startsWith("10.")) return 30;
  return 40;
}

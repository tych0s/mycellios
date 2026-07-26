import { randomBytes } from "node:crypto";
import { once } from "node:events";
import { describe, expect, it } from "vitest";
import {
  createDirectSessionGrant,
  DirectSecureChannel,
  DirectSecureServer,
} from "../src/transport/direct-secure-channel.js";

describe("native encrypted direct transport", () => {
  it("authenticates both peers and carries encrypted binary frames both ways", async () => {
    const grant = createDirectSessionGrant({
      connectionId: "direct-a-b",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: Date.now() + 60_000,
      secret: Buffer.alloc(32, 7),
    });
    let accepted: DirectSecureChannel | null = null;
    const server = new DirectSecureServer({
      resolveGrant: (connectionId) => connectionId === grant.connectionId ? grant : null,
      onChannel: (channel) => {
        accepted = channel;
      },
    });
    const address = await server.listen();
    const client = await DirectSecureChannel.connect({
      host: address.host,
      port: address.port,
      grant,
    });
    await waitUntil(() => accepted !== null);
    const serverChannel = accepted!;

    const serverData = once(serverChannel, "data");
    await client.send(Buffer.from("activation-bytes"));
    expect((await serverData)[0]).toEqual(Buffer.from("activation-bytes"));

    const clientData = once(client, "data");
    await serverChannel.send(Buffer.from([0, 1, 2, 255]));
    expect((await clientData)[0]).toEqual(Buffer.from([0, 1, 2, 255]));

    await client.close();
    serverChannel.destroy();
    await server.close();
  });

  it("rejects a peer that does not possess the coordinator grant", async () => {
    const grant = createDirectSessionGrant({
      connectionId: "direct-auth",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: Date.now() + 60_000,
      secret: Buffer.alloc(32, 8),
    });
    const rejections: Error[] = [];
    const server = new DirectSecureServer({
      resolveGrant: () => grant,
      onChannel: () => {
        throw new Error("unauthorized channel was accepted");
      },
      onRejected: (error) => rejections.push(error),
    });
    const address = await server.listen();
    const forged = { ...grant, secret: randomBytes(32).toString("base64url") };
    await expect(DirectSecureChannel.connect({
      host: address.host,
      port: address.port,
      grant: forged,
      timeoutMs: 1_000,
    })).rejects.toThrow();
    await waitUntil(() => rejections.length > 0);
    expect(rejections[0]?.message).toBe("direct_client_authentication_failed");
    await server.close();
  });

  it("rejects a peer identity even when it knows the session secret", async () => {
    const grant = createDirectSessionGrant({
      connectionId: "direct-peer-bound",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: Date.now() + 60_000,
    });
    const rejections: Error[] = [];
    const server = new DirectSecureServer({
      resolveGrant: () => grant,
      onChannel: () => {
        throw new Error("wrong peer was accepted");
      },
      onRejected: (error) => rejections.push(error),
    });
    const address = await server.listen();
    await expect(DirectSecureChannel.connect({
      host: address.host,
      port: address.port,
      grant: { ...grant, sourceNodeId: "node-c" },
      timeoutMs: 1_000,
    })).rejects.toThrow();
    await waitUntil(() => rejections.length > 0);
    expect(rejections[0]?.message).toBe("direct_transport_grant_identity_mismatch");
    await server.close();
  });

  it("consumes a grant once to prevent handshake replay", async () => {
    const grant = createDirectSessionGrant({
      connectionId: "direct-once",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: Date.now() + 60_000,
    });
    const channels: DirectSecureChannel[] = [];
    const rejections: Error[] = [];
    const server = new DirectSecureServer({
      resolveGrant: () => grant,
      onChannel: (channel) => channels.push(channel),
      onRejected: (error) => rejections.push(error),
    });
    const address = await server.listen();
    const first = await DirectSecureChannel.connect({
      host: address.host,
      port: address.port,
      grant,
    });
    await waitUntil(() => channels.length === 1);
    await expect(DirectSecureChannel.connect({
      host: address.host,
      port: address.port,
      grant,
      timeoutMs: 1_000,
    })).rejects.toThrow();
    await waitUntil(() => rejections.length > 0);
    expect(rejections.at(-1)?.message).toBe("direct_transport_grant_was_already_consumed");
    await first.close();
    channels[0]?.destroy();
    await server.close();
  });

  it("rejects expired grants and oversized frames without fallback claims", async () => {
    const expired = createDirectSessionGrant({
      connectionId: "direct-expired",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: 1_000,
    });
    await expect(DirectSecureChannel.connect({
      host: "127.0.0.1",
      port: 9,
      grant: expired,
      now: () => 1_001,
    })).rejects.toThrow("direct_transport_grant_is_expired");

    const grant = createDirectSessionGrant({
      connectionId: "direct-bounds",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: Date.now() + 60_000,
    });
    const accepted: DirectSecureChannel[] = [];
    const server = new DirectSecureServer({
      maximumFrameBytes: 1_024,
      resolveGrant: () => grant,
      onChannel: (channel) => {
        accepted.push(channel);
      },
    });
    const address = await server.listen();
    const client = await DirectSecureChannel.connect({
      host: address.host,
      port: address.port,
      grant,
      maximumFrameBytes: 1_024,
    });
    await expect(client.send(Buffer.alloc(1_025))).rejects.toThrow(
      "direct_frame_size_is_invalid",
    );
    await client.close();
    for (const channel of accepted) channel.destroy();
    await server.close();
  });
});

async function waitUntil(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("condition was not reached");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

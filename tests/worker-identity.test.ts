import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";
import { addWorker } from "./helpers.js";

describe("stable worker identity", () => {
  let database: MeshDatabase;
  let store: MeshStore;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    store = new MeshStore(database);
  });

  afterEach(() => database.close());

  it("reuses and reactivates the same worker row for a native device", () => {
    const identity = { kind: "device" as const, id: "desktop-stable-test" };
    const first = addWorker(store, { id: "first", identity, offeredVramMb: 4_096 });
    expect(store.deregisterWorker(first.id)).toBe(true);

    const second = addWorker(store, { id: "second", identity, offeredVramMb: 6_144 });

    expect(second.id).toBe(first.id);
    expect(second.identityKind).toBe("device");
    expect(second.identityId).toBe(identity.id);
    expect(second.capabilities.gpus[0]?.offeredVramMb).toBe(6_144);
    expect(store.listWorkers()).toHaveLength(1);
  });

  it("keeps independently identified cells separate from physical devices", () => {
    const device = addWorker(store, {
      id: "device",
      identity: { kind: "device", id: "shared-name" },
    });
    const cell = addWorker(store, {
      id: "cell",
      identity: { kind: "cell", id: "shared-name" },
    });

    expect(cell.id).not.toBe(device.id);
    expect(store.listWorkers()).toHaveLength(2);
  });
});

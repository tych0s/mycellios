import { describe, expect, it } from "vitest";
import {
  RentalCapacityManager,
  type RentalProviderDriver,
  type RentalQuote,
} from "../src/coordinator/rental-capacity-manager.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

describe("RentalCapacityManager", () => {
  it("waits for sustained economic demand and tears down only its labelled rental", async () => {
    const database = new MeshDatabase(":memory:");
    const store = new MeshStore(database);
    const hub = new WorkerHub(store);
    let now = Date.UTC(2026, 6, 29, 10);
    store.saveFederationSettings({
      enabled: true,
      dailyBudgetUsd: 10,
      monthlyBudgetUsd: 100,
      autoscalingEnabled: true,
      maxRentals: 4,
    });
    store.saveFederatedNetworkSettings({
      id: "vast",
      enabled: true,
      priority: 500,
      dailyBudgetUsd: 5,
      monthlyBudgetUsd: 50,
    });
    const driver = new FakeRentalDriver();
    const manager = new RentalCapacityManager(
      store,
      hub,
      [driver],
      "https://www.mycellios.com",
      `registry.example/mycellios@sha256:${"a".repeat(64)}`,
      { now: () => now, reconcileIntervalMs: 60_000 },
    );
    manager.recordUnmetDemand({
      model: "test-8b",
      minimumVramMb: 16_384,
      batch: false,
      projectedTokenCostUsdPerHour: 2,
    });

    await manager.reconcile();
    expect(driver.createCalls).toHaveLength(0);

    now += 5 * 60_000;
    manager.recordUnmetDemand({
      model: "test-8b",
      minimumVramMb: 16_384,
      batch: false,
      projectedTokenCostUsdPerHour: 2,
    });
    await manager.reconcile();

    expect(driver.createCalls).toHaveLength(1);
    const created = driver.createCalls[0]!;
    expect(created.environment.MYCELLIOS_NETWORK_TOKEN).toBeUndefined();
    expect(created.environment.MYCELLIOS_WORKER_CREDENTIAL_PATH).toBe(
      "/var/lib/mycellios/worker-credential.json",
    );
    expect(created.labels["mycellios-managed"]).toBe("true");
    expect(store.listManagedRentals()[0]).toEqual(expect.objectContaining({
      provider: "vast",
      state: "awaiting-worker",
      workerId: null,
    }));

    await manager.emergencyStop();

    expect(driver.terminated).toEqual(["provider-instance-1"]);
    expect(store.listManagedRentals()[0]?.state).toBe("stopped");
    manager.close();
    hub.close();
    database.close();
  });
});

class FakeRentalDriver implements RentalProviderDriver {
  readonly id = "vast" as const;
  readonly configured = true;
  readonly createCalls: Array<Parameters<RentalProviderDriver["create"]>[0]> = [];
  readonly terminated: string[] = [];
  private existsState = false;

  async quote(): Promise<RentalQuote> {
    return {
      offerId: "offer-1",
      hourlyUsd: 1,
      gpuModel: "Test GPU",
      vramMb: 24_576,
      reliability: 0.99,
      verifiedHost: true,
      spot: false,
    };
  }

  async create(input: Parameters<RentalProviderDriver["create"]>[0]) {
    this.createCalls.push(input);
    this.existsState = true;
    return { externalId: "provider-instance-1" };
  }

  async terminate(externalId: string) {
    this.terminated.push(externalId);
    this.existsState = false;
  }

  async exists() {
    return this.existsState;
  }
}

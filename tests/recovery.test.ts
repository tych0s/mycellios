import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MeshService } from "../src/coordinator/mesh-service.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import { Scheduler } from "../src/scheduler/scheduler.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

describe("coordinator crash recovery", () => {
  const directories: string[] = [];
  afterEach(() => {
    for (const directory of directories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("fails orphaned inference jobs on restart", () => {
    const directory = mkdtempSync(join(tmpdir(), "gpu-mesh-recovery-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    let database = new MeshDatabase(path);
    let store = new MeshStore(database);
    store.createJob({
      id: "job-orphan",
      sessionId: "session",
      model: "model",
      workloadClass: "interactive",
      deadlineAt: Date.now() + 60_000,
    });
    database.close();

    database = new MeshDatabase(path);
    store = new MeshStore(database);
    const hub = new WorkerHub(store);
    new MeshService(store, new Scheduler(store), hub, 10_000);

    expect(store.getJob("job-orphan")?.status).toBe("failed");
    expect(store.getJob("job-orphan")?.failureCode).toBe("coordinator_restarted");
    hub.close();
    database.close();
  });
});

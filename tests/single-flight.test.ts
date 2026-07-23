import { describe, expect, it } from "vitest";
import { SingleFlight } from "../src/desktop/single-flight.js";

describe("SingleFlight", () => {
  it("shares one in-flight initialization across concurrent callers", async () => {
    const flight = new SingleFlight();
    let resolve!: () => void;
    const blocked = new Promise<void>((done) => {
      resolve = done;
    });
    let calls = 0;
    const operation = async () => {
      calls += 1;
      await blocked;
    };

    const first = flight.run(operation);
    const second = flight.run(operation);

    expect(second).toBe(first);
    expect(calls).toBe(0);
    await Promise.resolve();
    expect(calls).toBe(1);
    resolve();
    await Promise.all([first, second, flight.wait()]);
    expect(calls).toBe(1);
  });

  it("allows a clean retry after a failed initialization", async () => {
    const flight = new SingleFlight();
    let calls = 0;

    await expect(flight.run(async () => {
      calls += 1;
      throw new Error("initialization failed");
    })).rejects.toThrow("initialization failed");
    await flight.run(async () => {
      calls += 1;
    });

    expect(calls).toBe(2);
  });
});

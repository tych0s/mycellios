import { describe, expect, it } from "vitest";
import { formatCompactTokens, formatMemory, formatPower, relativeTime, relativeTimeEs, shortFingerprint, shortId } from "./panel-ui-utilities";

describe("panel UI utilities", () => {
  it("formats capacity and identity consistently", () => {
    expect(formatCompactTokens(1_250_000)).toBe("1.3M");
    expect(formatMemory(12_288)).toBe("12 GB");
    expect(formatPower(1_250)).toBe("1.3 kW");
    expect(shortId("abcdefghijklmnop")).toBe("abcdef…mnop");
    expect(shortFingerprint(`sha256:${"a".repeat(64)}`)).toBe("aaaaaaaaaa…aaaaaaaa");
  });
  it("uses an injectable clock for stable relative labels", () => {
    const now = Date.parse("2026-08-30T13:00:00Z");
    expect(relativeTime("2026-08-30T12:58:30Z", now)).toBe("1m ago");
    expect(relativeTime("2026-08-28T13:00:00Z", now)).toBe("2d ago");
    expect(relativeTime("2026-07-16T13:00:00Z", now)).toBe("Jul 16, 2026");
    expect(relativeTime("invalid", now)).toBe("Unknown");
    expect(relativeTimeEs("2026-08-30T12:58:30Z", now)).toBe("hace 1 min");
  });
});

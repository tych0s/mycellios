import { describe, expect, it } from "vitest";
import { STORY_SECONDS, beatFor, nextPlayed, scrollTravel, storyProgress } from "./use-motion";

/*
 * These guard the one thing a reader complained about: how much work the story
 * section demands. The hook itself needs a DOM, but the rules that decide the
 * pacing are pure, and they are what regress — a height bumped back up or an
 * autoplay clock quietly slowed down would pass every rendering test.
 */

/** Plays `seconds` of pinned frames at 60fps and returns the progress reached. */
function play(seconds: number, pinned = true): number {
  let played = 0;
  for (let frame = 0; frame < Math.round(seconds * 60); frame++) played = nextPlayed(played, 1 / 60, pinned);
  return played;
}

describe("the story plays itself instead of charging the reader for it", () => {
  it("reaches the end on its own within a few seconds of being pinned", () => {
    expect(STORY_SECONDS).toBeLessThanOrEqual(6);
    expect(play(STORY_SECONDS)).toBeCloseTo(1, 5);
  });

  it("shows every beat while it plays, not just the last one", () => {
    const seen = new Set<number>();
    let played = 0;
    for (let frame = 0; frame < Math.round(STORY_SECONDS * 60); frame++) {
      played = nextPlayed(played, 1 / 60, true);
      seen.add(beatFor(played, 4));
    }
    expect([...seen].sort()).toEqual([0, 1, 2, 3]);
  });

  it("does not play at a reader the section has not pinned yet", () => {
    expect(play(3, false)).toBe(0);
  });

  it("never runs backwards once finished", () => {
    expect(nextPlayed(1, 1 / 60, true)).toBe(1);
    expect(nextPlayed(1, 1 / 60, false)).toBe(1);
  });

  it("survives a suspended tab without skipping the story", () => {
    // A background tab parks requestAnimationFrame; the first frame back can
    // carry minutes. Uncapped, that one frame would jump straight to the end.
    expect(nextPlayed(0, 600, true)).toBeLessThan(0.1);
  });
});

describe("scrolling scrubs ahead rather than being the only transport", () => {
  it("rewinds immediately while the reader scrolls back up", () => {
    expect(storyProgress(0.25, 0.9, true)).toBe(0.25);
    expect(storyProgress(0.9, 0.25, true)).toBe(0.9);
  });

  it("lets autoplay resume from the settled scroll position", () => {
    expect(storyProgress(0.25, 0.4, false)).toBe(0.4);
    expect(storyProgress(0.4, 0.25, false)).toBe(0.4);
  });

  it("lets a reader reach the end before the last pixel of the section", () => {
    // 200vh section, 100vh viewport: one viewport of travel. The payoff has to
    // land with room left, or it is gone the instant it arrives.
    expect(scrollTravel(-820, 2000, 1000)).toBeCloseTo(1, 5);
    expect(scrollTravel(-1000, 2000, 1000)).toBe(1);
  });

  it("starts at zero and is monotonic through the pin", () => {
    expect(scrollTravel(0, 2000, 1000)).toBe(0);
    expect(scrollTravel(500, 2000, 1000)).toBe(0);
    let previous = -1;
    for (let top = 0; top >= -1000; top -= 50) {
      const travel = scrollTravel(top, 2000, 1000);
      expect(travel).toBeGreaterThanOrEqual(previous);
      previous = travel;
    }
  });

  it("costs at most one viewport of wheel travel to get through", () => {
    // The section is 200vh tall and pins for 100vh, so the reader pays one
    // screen of scrolling — the old 420vh cost them more than three.
    const css = new URL("./rebrand.css", import.meta.url);
    return import("node:fs/promises").then(async (fs) => {
      const text = await fs.readFile(css, "utf8");
      for (const [, value] of text.matchAll(/\.rb-story \{[^}]*height:(\d+)vh/g)) {
        expect(Number(value)).toBeLessThanOrEqual(200);
      }
    });
  });
});

describe("beats", () => {
  it("maps travel onto the four beats and clamps at both ends", () => {
    expect(beatFor(0, 4)).toBe(0);
    expect(beatFor(0.3, 4)).toBe(1);
    expect(beatFor(0.6, 4)).toBe(2);
    expect(beatFor(0.9, 4)).toBe(3);
    expect(beatFor(1, 4)).toBe(3);
    expect(beatFor(1.4, 4)).toBe(3);
    expect(beatFor(-0.2, 4)).toBe(0);
  });
});

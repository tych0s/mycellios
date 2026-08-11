/// <reference lib="dom" />
import { useEffect, useState, type RefObject } from "react";

/*
 * Motion primitives for the landing.
 *
 * The landing is a scroll story, so every effect here is driven by the reader's
 * own scrolling rather than by autoplaying timers. Three rules hold throughout:
 *
 *  - work happens inside requestAnimationFrame, never inside the scroll event;
 *  - progress is published as a CSS custom property so the animation itself
 *    runs on the compositor instead of re-rendering React on every frame;
 *  - `prefers-reduced-motion` collapses each hook to its finished state.
 */

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** How long the whole story takes to play by itself, once it is pinned. */
export const STORY_SECONDS = 4.4;

/* A frame delta is never trusted beyond this. Background tabs suspend
   requestAnimationFrame, so the first frame after the reader returns can carry
   a delta of many seconds — without a clamp that single frame would jump the
   story straight to its end. */
const MAX_FRAME_SECONDS = 0.1;

/* Scroll travel reaches 1 at this fraction of the pin, leaving the rest as a
   settled tail. A reader who crosses the section quickly still sees the payoff
   land with room to spare instead of at the final pixel. */
const SETTLE = 0.82;

/**
 * Advances the autoplay clock by one frame. Exported for its own sake: this is
 * the rule that decides whether the reader has to work for the story.
 */
export function nextPlayed(played: number, deltaSeconds: number, pinned: boolean): number {
  if (!pinned || played >= 1) return Math.min(1, played);
  const step = Math.min(Math.max(0, deltaSeconds), MAX_FRAME_SECONDS) / STORY_SECONDS;
  return Math.min(1, played + step);
}

/** Maps 0→1 travel onto a beat index, with the last beat reached before 1. */
export function beatFor(progress: number, beats: number): number {
  return Math.min(beats - 1, Math.max(0, Math.floor(progress * beats)));
}

/** Fraction of the pin the reader has scrolled through, with the settled tail. */
export function scrollTravel(top: number, height: number, viewport: number): number {
  const distance = Math.max(1, height - viewport);
  return Math.min(1, Math.max(0, -top / distance / SETTLE));
}

/**
 * Drives a pinned story section: writes 0→1 into `--p` on the element and
 * returns the current beat index. React only re-renders on beat changes, so the
 * per-frame cost stays in CSS.
 *
 * The section plays *itself*. As soon as it is pinned an autoplay clock starts,
 * and progress is the further of the clock and the reader's own scroll — so
 * standing still shows the whole story in {@link STORY_SECONDS}, while a reader
 * who wants to move on can scrub past it at their own speed. Neither one can
 * pull the story backwards, which is what made the section feel like a toll
 * gate when scroll was the only transport.
 */
export function useStoryProgress(ref: RefObject<HTMLElement | null>, beats: number): number {
  const [beat, setBeat] = useState(0);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (prefersReducedMotion()) {
      element.style.setProperty("--p", "1");
      setBeat(beats - 1);
      return;
    }

    let frame = 0;
    let played = 0;
    let last = 0;

    const update = (now: number) => {
      frame = 0;
      const rect = element.getBoundingClientRect();
      const viewport = window.innerHeight;
      // Pinned means the sticky child fills the viewport, which is exactly the
      // window in which the reader is being held — and so the only window in
      // which playing at them is fair.
      const pinned = rect.top <= 0 && rect.bottom >= viewport;
      played = nextPlayed(played, last ? (now - last) / 1000 : 0, pinned);
      last = now;

      const progress = Math.max(scrollTravel(rect.top, element.offsetHeight, viewport), played);
      element.style.setProperty("--p", progress.toFixed(4));
      setBeat(beatFor(progress, beats));

      // The loop sustains itself only while there is something left to play;
      // after that the scroll listener alone keeps `--p` honest.
      if (pinned && played < 1) frame = window.requestAnimationFrame(update);
      else last = 0;
    };
    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };

    schedule();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [ref, beats]);

  return beat;
}

/**
 * Counts a headline figure up once it is actually on screen.
 *
 * The initial state is the final value so server-rendered markup already
 * carries the real number for crawlers; the client rewinds to zero on mount and
 * replays the count when the element scrolls into view.
 */
export function useCountUp(ref: RefObject<HTMLElement | null>, target: number, duration = 1500): number {
  const [value, setValue] = useState(target);

  useEffect(() => {
    const element = ref.current;
    if (!element || prefersReducedMotion()) return;

    setValue(0);
    let frame = 0;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          observer.disconnect();
          const start = performance.now();
          const step = (now: number) => {
            const t = Math.min(1, (now - start) / duration);
            setValue(Math.round(target * (1 - (1 - t) ** 3)));
            if (t < 1) frame = window.requestAnimationFrame(step);
          };
          frame = window.requestAnimationFrame(step);
        }
      },
      { threshold: 0.45 },
    );
    observer.observe(element);
    return () => {
      observer.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [ref, target, duration]);

  return value;
}

/**
 * Adds `is-visible` to every `.rb-reveal` inside a root the first time it
 * appears. One observer serves the whole page instead of one per section.
 */
export function useRevealOnScroll(ref: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const targets = root.querySelectorAll(".rb-reveal");
    if (prefersReducedMotion()) {
      targets.forEach((element) => element.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }),
      { threshold: 0.14, rootMargin: "0px 0px -6% 0px" },
    );
    targets.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [ref]);
}

/** Marks an element once it has been on screen, for one-shot CSS transitions. */
export function useInView(ref: RefObject<HTMLElement | null>, threshold = 0.3): boolean {
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (prefersReducedMotion()) {
      setSeen(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          setSeen(true);
          observer.disconnect();
        }),
      { threshold },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, threshold]);

  return seen;
}

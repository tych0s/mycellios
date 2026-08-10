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

/**
 * Scrubs a tall sticky section: writes 0→1 travel into `--p` on the element and
 * returns the current beat index. React only re-renders on beat changes, so the
 * per-frame cost stays in CSS.
 */
export function useStickyProgress(ref: RefObject<HTMLElement | null>, beats: number): number {
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
    const update = () => {
      frame = 0;
      const rect = element.getBoundingClientRect();
      const distance = Math.max(1, element.offsetHeight - window.innerHeight);
      const progress = Math.min(1, Math.max(0, -rect.top / distance));
      element.style.setProperty("--p", progress.toFixed(4));
      setBeat(Math.min(beats - 1, Math.floor(progress * beats)));
    };
    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };

    update();
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

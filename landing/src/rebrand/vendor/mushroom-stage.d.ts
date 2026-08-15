/*
 * Public surface of the vendored renderer.
 *
 * A sibling declaration file rather than an ambient `declare module`: the
 * import is a relative path, and relative specifiers are resolved against the
 * file system, so an ambient module declaration for "./vendor/…" is only ever
 * matched from one directory and is silently ignored everywhere else. TypeScript
 * picks this up automatically for `mushroom-stage.js`.
 */

/** Registers the <mushroom-stage> custom element. Safe to call repeatedly. */
export function defineMushroomStage(): void;

/*
 * The variant table is declared here because it is read from TypeScript.
 *
 * The tests project these profiles through a real three.js camera to check that
 * a body is cut by its section's rule rather than hovering over it, so the
 * shapes have to be typed or every one of those checks is an `any`. Tuples,
 * not `number[]`: `const [sx, sy, sz] = V.capScale` and `camera.target` are
 * spread into three-argument calls, and an array type makes each element
 * possibly-undefined under `noUncheckedIndexedAccess` for no reason — the
 * lengths are fixed by the renderer.
 */

/** A lathe row: radius from the axis, height along it. */
export type StageRow = readonly [number, number];

/** One drawn body's geometry, independent of where it stands. */
export interface StageProfile {
  readonly capTop: readonly StageRow[];
  readonly capUnder: readonly StageRow[];
  readonly stem: readonly StageRow[];
  readonly capScale: readonly [number, number, number];
  readonly stemBend: number;
  readonly gillFactor: number;
}

/** A colony member: a profile plus its place in the patch of ground. */
export interface StageColonyBody extends StageProfile {
  readonly at: readonly [number, number, number];
  readonly scale: number;
  readonly rotY: number;
}

/** Multipliers on everything in the variant that emits light. */
export interface StageTune {
  readonly halo: number;
  readonly glow: number;
  readonly shell: number;
  readonly bloom: number;
  readonly threshold: number;
}

export interface StageCamera {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly fov: number;
  readonly target: readonly [number, number, number];
}

/** One entry in the table: a body (or a group), framed and lit. */
export interface StageVariant extends StageProfile {
  readonly camera: StageCamera;
  readonly tune: StageTune;
  /** Present only on group variants, where the top-level profile is the middle member's. */
  readonly colony?: readonly StageColonyBody[];
}

export declare const VARIANTS: {
  readonly hero: StageVariant;
  readonly gutter: StageVariant;
  readonly colony: StageVariant & { readonly colony: readonly StageColonyBody[] };
};

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

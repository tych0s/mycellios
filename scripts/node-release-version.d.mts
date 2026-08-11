/** MAJOR, MINOR y PATCH ya separados en numeros. */
export type ParsedVersion = readonly [number, number, number];

export function parseVersion(value: unknown): ParsedVersion;

/** Negativo si a < b, cero si iguales, positivo si a > b. Numerico, no textual. */
export function compareVersions(a: string, b: string): number;

/** La mayor de la lista, o `null` si viene vacia. */
export function highestVersion(versions: readonly string[]): string | null;

export function nextPatch(version: string): string;

/**
 * Versiones tomadas de los tags `vX.Y.Z` del repositorio, o `null` si no se
 * puede leer git. `null` no es lo mismo que lista vacia: significa "no lo se".
 */
export function publishedVersions(cwd?: string): string[] | null;

/**
 * Version declarada en `package.json`. Lanza si `package-lock.json` no coincide
 * en sus DOS campos, que es la condicion que el propio workflow exige.
 */
export function currentVersion(): string;

/** La siguiente libre: por delante de lo declarado y de todo tag publicado. */
export function resolveNextVersion(
  declared: string,
  published: readonly string[] | null,
): string;

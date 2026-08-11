/**
 * Component manifests pin the on-disk/runtime contract separately from the
 * worker WebSocket range. Bump this only when a side-by-side component can no
 * longer be consumed by the current bootstrap.
 */
export const MYCELLIOS_RUNTIME_ABI =
  "mycellios-distribution-runtime/4" as const;

/**
 * Oldest desktop bootstrap that understands the v1 signed component layout.
 * This is intentionally independent from package.json; ordinary source
 * releases must not force every development node through a new installer.
 */
export const COMPONENT_UPDATE_BOOTSTRAP_MIN_VERSION = "0.2.70" as const;

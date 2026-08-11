import type { PinnedComponentUpdateKey } from "../contracts/component-update-manifest.js";

/**
 * Stable trust is an immutable bootstrap decision, never renderer-owned
 * settings. The first stable signing key is deliberately enrolled only during
 * a production key ceremony and desktop release; development keys cannot
 * promote themselves into this list.
 */
export const STABLE_COMPONENT_UPDATE_KEYS:
  readonly PinnedComponentUpdateKey[] = Object.freeze([]);

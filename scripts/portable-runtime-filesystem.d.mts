export function assertNoEscapingSymlinks(root: string): void;

export function normalizeCopiedInternalAbsoluteSymlinks(
  copiedRoot: string,
  sourceRoot: string,
): void;

export function samePath(left: string, right: string): boolean;

export function verifyNodePkg(input: { pkg: string; stagedRoot: string; version: string }): Promise<{ identifier: string; version: string }>;

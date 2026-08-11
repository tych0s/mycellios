export function verifyNodeMsi(input: { msi: string; stagedRoot: string; version: string; sourceRevision: string }): Promise<{ version: string; sourceRevision: string }>;

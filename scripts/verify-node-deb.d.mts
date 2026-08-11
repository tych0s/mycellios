export function verifyNodeDeb(input: { deb: string; stagedRoot: string; version: string }): Promise<{ package: string | undefined; version: string | undefined; architecture: string | undefined }>;

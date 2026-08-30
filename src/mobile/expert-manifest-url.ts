const SHA256_HEX = /^[a-f0-9]{64}$/;

export function trustedExpertManifestPath(artifactId: string, candidate: string): string {
  if (!SHA256_HEX.test(artifactId)) throw new Error("invalid expert artifact id");
  const expectedPath = `/mobile/v1/experts/${artifactId}/manifest`;
  if (candidate !== expectedPath) throw new Error("untrusted expert manifest URL");
  return expectedPath;
}

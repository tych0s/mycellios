export const ROOT_DOCUMENTS: readonly string[];

export interface DocumentationVerificationResult {
  documents: number;
  localLinks: number;
}

export function verifyDocumentationLinks(
  root?: string,
  files?: readonly string[],
): DocumentationVerificationResult;

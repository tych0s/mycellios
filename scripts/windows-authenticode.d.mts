export interface AuthenticodeReport {
  status: string;
  sha256: string;
  signerThumbprint: string;
  timestampThumbprint: string;
}
export function verifyWindowsAuthenticode(input: { artifact: string; sha256: string; signerThumbprint?: string }): AuthenticodeReport;
export function validateWindowsAuthenticodeReport(report: unknown, input: { sha256: string; signerThumbprint?: string }): AuthenticodeReport;

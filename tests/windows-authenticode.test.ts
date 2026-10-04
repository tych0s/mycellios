import { describe, expect, it } from "vitest";
import { validateWindowsAuthenticodeReport } from "../scripts/windows-authenticode.mjs";

const expected = { sha256: "a".repeat(64), signerThumbprint: "b".repeat(40) };
const valid = { status: "Valid", sha256: expected.sha256, signerThumbprint: expected.signerThumbprint, timestampThumbprint: "c".repeat(40) };

describe("Windows release signature trust binding", () => {
  it.each(["NotSigned", "HashMismatch", "NotTrusted", "UnknownError"])("rejects Windows status %s", (status) => {
    expect(() => validateWindowsAuthenticodeReport({ ...valid, status }, expected)).toThrow("authenticode_not_valid");
  });
  it("requires an independently supplied publisher certificate", () => {
    expect(() => validateWindowsAuthenticodeReport(valid, { sha256: expected.sha256 })).toThrow("expected_signer_is_required");
  });
  it("rejects a valid signature made by another publisher", () => {
    expect(() => validateWindowsAuthenticodeReport({ ...valid, signerThumbprint: "d".repeat(40) }, expected)).toThrow("signer_mismatch");
  });
  it("rejects inspection of different bytes", () => {
    expect(() => validateWindowsAuthenticodeReport({ ...valid, sha256: "d".repeat(64) }, expected)).toThrow("artifact_mismatch");
  });
  it("requires a timestamp", () => {
    expect(() => validateWindowsAuthenticodeReport({ ...valid, timestampThumbprint: null }, expected)).toThrow("timestamp_missing");
  });
  it("accepts a matching publisher independent of thumbprint case", () => {
    expect(validateWindowsAuthenticodeReport(valid, expected).signerThumbprint).toBe(expected.signerThumbprint.toUpperCase());
  });
});

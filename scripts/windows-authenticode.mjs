import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";

/** Inspect the actual file, never a caller's declaration of signature state. */
export function verifyWindowsAuthenticode(input) {
  if (process.platform !== "win32") throw new Error("node_release_authenticode_requires_windows");
  const expected = validateExpected(input);
  const script = `
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$path=$env:MYCELLIOS_SIGNATURE_ARTIFACT
$handle=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
try {
  $signature=Get-AuthenticodeSignature -LiteralPath $path
  $hash=Get-FileHash -LiteralPath $path -Algorithm SHA256
  [ordered]@{
    status=$signature.Status.ToString()
    sha256=$hash.Hash.ToLowerInvariant()
    signerThumbprint=if($signature.SignerCertificate){$signature.SignerCertificate.Thumbprint}else{$null}
    timestampThumbprint=if($signature.TimeStamperCertificate){$signature.TimeStamperCertificate.Thumbprint}else{$null}
  } | ConvertTo-Json -Compress
} finally { $handle.Dispose() }
`;
  const powershellRoot = join(process.env.SystemRoot ?? "C:/Windows", "System32", "WindowsPowerShell", "v1.0");
  const result = spawnSync(join(powershellRoot, "powershell.exe"), ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
    Buffer.from(script, "utf16le").toString("base64")], {
    encoding: "utf8", timeout: 120_000, windowsHide: true, shell: false,
    // A caller running PowerShell 7 can otherwise leak incompatible modules
    // into the Windows PowerShell 5 signature inspector.
    env: { ...process.env, PSModulePath: join(powershellRoot, "Modules"), MYCELLIOS_SIGNATURE_ARTIFACT: resolve(input.artifact) },
  });
  if (result.error || result.status !== 0) throw new Error("node_release_authenticode_inspection_failed", {
    cause: result.error ?? new Error(result.stderr.trim().slice(0, 2_000)),
  });
  let report;
  try { report = JSON.parse(result.stdout.trim()); }
  catch { throw new Error("node_release_authenticode_report_is_invalid"); }
  return validateWindowsAuthenticodeReport(report, expected);
}

export function validateWindowsAuthenticodeReport(report, input) {
  const expected = validateExpected(input);
  if (report?.status !== "Valid") throw new Error("node_release_authenticode_not_valid");
  if (report.sha256 !== expected.sha256) throw new Error("node_release_authenticode_artifact_mismatch");
  if (typeof report.signerThumbprint !== "string"
    || report.signerThumbprint.toUpperCase() !== expected.signerThumbprint) {
    throw new Error("node_release_authenticode_signer_mismatch");
  }
  if (typeof report.timestampThumbprint !== "string" || !/^[a-f0-9]{40}$/i.test(report.timestampThumbprint)) {
    throw new Error("node_release_authenticode_timestamp_missing");
  }
  return { ...report, signerThumbprint: expected.signerThumbprint };
}

function validateExpected(input) {
  if (!/^[a-f0-9]{64}$/.test(input.sha256 ?? "")) throw new Error("node_release_authenticode_expected_hash_is_invalid");
  if (!/^[a-f0-9]{40}$/i.test(input.signerThumbprint ?? "")) throw new Error("node_release_authenticode_expected_signer_is_required");
  return { sha256: input.sha256, signerThumbprint: input.signerThumbprint.toUpperCase() };
}

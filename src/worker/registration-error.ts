import { readResponseTextLimited } from "./response-limit.js";

export class WorkerRegistrationError extends Error {
  constructor(
    readonly status: number,
    readonly code: string | null,
  ) {
    super(`Worker registration failed with HTTP ${status}${code ? ` (${code})` : ""}`);
    this.name = "WorkerRegistrationError";
  }
}

export async function readWorkerRegistrationError(response: Response): Promise<WorkerRegistrationError> {
  const failure = await readResponseTextLimited(response, 16 * 1024);
  let code: string | null = null;
  try {
    const body = JSON.parse(failure) as { error?: { code?: unknown } };
    if (typeof body.error?.code === "string" && /^[a-z][a-z0-9_]{0,127}$/.test(body.error.code)) {
      code = body.error.code;
    }
  } catch {
    // Preserve the HTTP status when an upstream proxy did not return JSON.
  }
  return new WorkerRegistrationError(response.status, code);
}

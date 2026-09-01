import process from "node:process";

interface DiagnosticEvent {
  id: string;
  sourceId: string;
  appVersion: string;
  platform: string;
  arch: string;
  level: "info" | "warning" | "error";
  source: string;
  event: string;
  message: string;
  details?: string;
  occurredAt: string;
  receivedAt: string;
}

interface DiagnosticResponse {
  capturedAt: string;
  events: DiagnosticEvent[];
}

const args = process.argv.slice(2);
if (args.includes("--help")) {
  console.log(`Read recent Mycellios desktop diagnostics.

Usage:
  npm run diagnostics:remote -- [options]

Options:
  --url <base-url>       Coordinator URL (default: MYCELLIOS_PUBLIC_API_BASE_URL)
  --level <level>        info, warning or error
  --source <uuid>        Diagnostics source ID
  --since <iso-date>     Only events at or after this date
  --limit <count>        1-500 events (default: 100)
  --json                 Print the unmodified JSON response

Authentication:
  MYCELLIOS_MODEL_ADMIN_TOKEN or MYCELLIOS_ADMIN_ACCESS_TOKEN`);
  process.exit(0);
}

const baseUrl = option("--url")
  ?? process.env.MYCELLIOS_PUBLIC_API_BASE_URL
  ?? "https://www.mycellios.com";
const token = process.env.MYCELLIOS_MODEL_ADMIN_TOKEN
  ?? process.env.MYCELLIOS_ADMIN_ACCESS_TOKEN;
if (!token?.trim()) {
  throw new Error(
    "Set MYCELLIOS_MODEL_ADMIN_TOKEN or MYCELLIOS_ADMIN_ACCESS_TOKEN through Infisical.",
  );
}

const query = new URLSearchParams();
query.set("limit", option("--limit") ?? "100");
const level = option("--level");
const sourceId = option("--source");
const since = option("--since");
if (level) query.set("level", level);
if (sourceId) query.set("sourceId", sourceId);
if (since) query.set("since", since);

const url = new URL(`/public/v1/admin/diagnostics?${query}`, baseUrl);
const response = await fetch(url, {
  headers: { authorization: `Bearer ${token.trim()}` },
  signal: AbortSignal.timeout(10_000),
  redirect: "error",
});
if (!response.ok) {
  const body = await response.text();
  throw new Error(`Diagnostics request failed (${response.status}): ${body.slice(0, 500)}`);
}
const diagnostics = await response.json() as DiagnosticResponse;
if (args.includes("--json")) {
  console.log(JSON.stringify(diagnostics, null, 2));
  process.exit(0);
}

if (diagnostics.events.length === 0) {
  console.log("No diagnostic events matched the filters.");
  process.exit(0);
}

for (const event of diagnostics.events) {
  console.log(
    `${event.occurredAt} ${event.level.toUpperCase().padEnd(7)} `
    + `${event.source}/${event.event} [${event.sourceId.slice(0, 8)}]`,
  );
  console.log(`  ${event.message}`);
  if (event.details) console.log(`  ${event.details}`);
}

function option(name: string): string | undefined {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  const value = args[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`${name} requires a value.`);
  }
  return value;
}


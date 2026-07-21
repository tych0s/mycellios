import { execFile } from "node:child_process";
import { createReadStream, statSync } from "node:fs";
import { createServer, type ServerResponse } from "node:http";
import { extname, join, resolve } from "node:path";
import { loadBenchmarkRuns } from "./history.js";
import { runAndPersistRealSuite } from "./cli.js";

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
};

export interface BenchmarkServerOptions {
  cwd: string;
  host: string;
  port: number;
  open: boolean;
}

export async function startBenchmarkServer(options: BenchmarkServerOptions): Promise<void> {
  const dashboardDirectory = resolve(options.cwd, "benchmarks", "dashboard");
  let runInFlight = false;
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", `http://${options.host}:${options.port}`);
      if (request.method === "GET" && url.pathname === "/api/runs") {
        sendJson(response, 200, { runs: loadBenchmarkRuns(options.cwd).reverse() });
        return;
      }
      if (request.method === "POST" && url.pathname === "/api/run") {
        if (runInFlight) {
          sendJson(response, 409, { error: "Ya hay una ejecución en curso." });
          return;
        }
        runInFlight = true;
        try {
          const result = await runAndPersistRealSuite({ cwd: options.cwd });
          sendJson(response, 201, { run: result.run });
        } finally {
          runInFlight = false;
        }
        return;
      }
      if (request.method !== "GET") {
        sendJson(response, 405, { error: "Método no permitido." });
        return;
      }
      const fileName = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
      if (!new Set(["index.html", "dashboard.css", "dashboard.js"]).has(fileName)) {
        sendJson(response, 404, { error: "No encontrado." });
        return;
      }
      serveFile(response, join(dashboardDirectory, fileName));
    } catch (error) {
      sendJson(response, 500, { error: error instanceof Error ? error.message : String(error) });
    }
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(options.port, options.host, () => resolveListen());
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : options.port;
  const url = `http://${options.host}:${port}`;
  console.log(`Banco de pruebas disponible en ${url}`);
  console.log("Pulsa Ctrl+C para cerrarlo.");
  if (options.open) openUrl(url);
}

function serveFile(response: ServerResponse, path: string): void {
  const stat = statSync(path);
  response.writeHead(200, {
    "content-type": MIME_TYPES[extname(path)] ?? "application/octet-stream",
    "content-length": stat.size,
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "content-security-policy":
      "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'",
  });
  createReadStream(path).pipe(response);
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
  });
  response.end(body);
}

function openUrl(url: string): void {
  const command = process.platform === "win32" ? "rundll32" : process.platform === "darwin" ? "open" : "xdg-open";
  const args = process.platform === "win32" ? ["url.dll,FileProtocolHandler", url] : [url];
  const child = execFile(command, args, { windowsHide: true }, () => undefined);
  child.unref();
}

function parseArguments(values: string[]): BenchmarkServerOptions {
  const parsed: BenchmarkServerOptions = {
    cwd: process.cwd(),
    host: "127.0.0.1",
    port: 4317,
    open: false,
  };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    const next = values[index + 1];
    if (value === "--port" && next) parsed.port = boundedPort(next);
    else if (value === "--host" && next) parsed.host = next;
    else if (value === "--open") {
      parsed.open = true;
      continue;
    } else throw new Error(`Argumento desconocido: ${value}`);
    index += 1;
  }
  return parsed;
}

function boundedPort(value: string): number {
  const port = Number.parseInt(value, 10);
  if (!Number.isInteger(port) || port < 0 || port > 65_535) throw new Error("Puerto no válido.");
  return port;
}

if (isEntryPoint()) {
  void startBenchmarkServer(parseArguments(process.argv.slice(2))).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}

function isEntryPoint(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  return import.meta.url === new URL(`file:///${entry.replaceAll("\\", "/")}`).href;
}

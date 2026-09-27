import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { resolve, basename } from "node:path";

const files = new Map([
  ["/", ["tests/browser-qwen3-layer-onnx.html", "text/html; charset=utf-8"]],
  ["/model.onnx", ["runtime/browser-qwen3-layer.onnx", "application/octet-stream"]],
  ["/fixture.json", ["runtime/browser-qwen3-layer-fixture.json", "application/json"]],
  ["/real-model.onnx", ["runtime/browser-qwen3-real-layer.onnx", "application/octet-stream"]],
  ["/real-fixture.json", ["runtime/browser-qwen3-real-layer-fixture.json", "application/json"]],
]);
const dist = resolve("node_modules/onnxruntime-web/dist");
const server = createServer(async (request, response) => {
  const pathname = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
  if (pathname === "/receipt" && request.method === "POST") {
    try {
      const chunks = [];
      let bytes = 0;
      for await (const chunk of request) {
        bytes += chunk.length;
        if (bytes > 5 * 1024 * 1024) {
          response.writeHead(413).end();
          return;
        }
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks);
      const receipt = JSON.parse(body.toString("utf8"));
      if (receipt.schema !== "mycellios-local-browser-layer-receipt/1"
          || !["wasm", "webgpu"].includes(receipt.backend)
          || typeof receipt.realCheckpoint !== "boolean"
          || !Array.isArray(receipt.steps)) {
        response.writeHead(400).end();
        return;
      }
      const name = `browser-qwen3-${receipt.realCheckpoint ? "real" : "tiny"}-${receipt.backend}-receipt.json`;
      await writeFile(resolve("runtime", name), body);
      response.writeHead(201, { "content-type": "application/json" }).end(JSON.stringify({ name }));
    } catch {
      response.writeHead(400).end();
    }
    return;
  }
  let entry = files.get(pathname);
  if (!entry && pathname.startsWith("/ort/")) {
    const name = basename(pathname);
    if (name === pathname.slice(5) && /^ort[\w.-]+\.(?:js|mjs|wasm)$/.test(name)) {
      entry = [resolve(dist, name), name.endsWith(".wasm")
        ? "application/wasm" : "text/javascript; charset=utf-8"];
    }
  }
  if (!entry) { response.writeHead(404).end(); return; }
  try {
    const data = await readFile(entry[0]);
    response.writeHead(200, {
      "content-type": entry[1], "cache-control": "no-store",
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    }).end(data);
  } catch {
    response.writeHead(404).end();
  }
});
server.listen(8771, "127.0.0.1", () => {
  process.stdout.write("Qwen3 ONNX layer probe: http://127.0.0.1:8771/\n");
});

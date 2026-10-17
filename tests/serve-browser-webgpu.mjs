import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const pages = new Set([
  '/browser-webgpu-matmul.html',
  '/browser-webgpu-capabilities.html',
  '/browser-webllm-smollm2.html',
  '/browser-webgpu-autoregressive.html',
  '/browser-webgpu-tiny-transformer.html',
  '/browser-webgpu-uniform-scatter.html',
  '/browser-webllm-gpu-chunk.html',
  '/browser-webnn-capabilities.html',
]);

createServer(async (request, response) => {
  const path = new URL(request.url, 'http://127.0.0.1').pathname;
  if (!pages.has(path)) {
    response.writeHead(404).end('Not found');
    return;
  }
  try {
    const html = await readFile(join(root, path.slice(1)));
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' }).end(html);
  } catch {
    response.writeHead(500).end('Unable to read probe');
  }
}).listen(Number(process.env.MYCELLIOS_WEBGPU_PROBE_PORT || 8765), '127.0.0.1', () => {
  const port = Number(process.env.MYCELLIOS_WEBGPU_PROBE_PORT || 8765);
  process.stdout.write(`WebGPU probes: http://127.0.0.1:${port}/browser-webgpu-matmul.html\n`);
});

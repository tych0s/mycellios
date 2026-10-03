import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const tests = dirname(fileURLToPath(import.meta.url));
const root = dirname(tests);
const packageRoot = process.env.WLLAMA_PROBE_PACKAGE || join(tmpdir(), 'mycellios-wllama-probe', 'package');
const routes = new Map([
  ['/browser-wllama-smollm2.html', [join(tests, 'browser-wllama-smollm2.html'), 'text/html; charset=utf-8']],
  ['/wllama/index.js', [join(packageRoot, 'esm', 'index.js'), 'text/javascript; charset=utf-8']],
  ['/wllama/wasm/wllama.wasm', [join(packageRoot, 'esm', 'wasm', 'wllama.wasm'), 'application/wasm']],
  ['/model.gguf', [join(root, 'runtime', 'models', 'SmolLM2-135M-Instruct-F16.gguf'), 'application/octet-stream']],
  ['/qwen.gguf', [join(root, 'runtime', 'models', 'Qwen3-0.6B-Q8_0.gguf'), 'application/octet-stream']],
]);

createServer(async (request, response) => {
  const pathname = new URL(request.url, 'http://127.0.0.1').pathname;
  const route = routes.get(pathname);
  if (!route) return response.writeHead(404).end('Not found');
  const [path, contentType] = route;
  try {
    const file = await stat(path);
    response.writeHead(200, {
      'Content-Type': contentType,
      'Content-Length': file.size,
      'Cache-Control': 'no-store',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404).end('Probe asset unavailable');
  }
}).listen(8767, '127.0.0.1', () => {
  process.stdout.write('Wllama probe: http://127.0.0.1:8767/browser-wllama-smollm2.html\n');
});

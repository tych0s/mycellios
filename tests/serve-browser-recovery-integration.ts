import { createCoordinator } from '../src/coordinator/server.js';
import { resolve } from 'node:path';
const runtime = await createCoordinator({
  host: '127.0.0.1', port: 18910, databasePath: ':memory:', requestTimeoutMs: 10_000,
  mobileAssetsPath: resolve('mobile-dist'),
  mobileExpertArtifactsPath: resolve('runtime/browser-recovery-proof/artifacts'),
}, {logger: false});
let held = false;
let release: (() => void) | undefined;
let waiting = 0;
let denied = false;
let gate = Promise.resolve();
runtime.app.addHook('onRequest', async (request, reply) => {
  if (denied && request.method === 'POST' && request.url === '/mobile/v1/admission-challenge') {
    return reply.code(401).send({error: {code: 'lab_admission_denied'}});
  }
  if (held && request.method === 'POST' && request.url === '/mobile/v1/admission-challenge') {
    waiting++; await gate; waiting--;
  }
});
runtime.app.get('/lab/state', async () => ({waiting, workers: runtime.mobileHub.listWorkers().map(w => ({
  id: w.id, clientId: w.clientId, connected: w.connected, visible: w.visible, verifiedTasks: w.verifiedTasks,
}))}));
runtime.app.post('/lab/drop', async () => {
  const before = runtime.mobileHub.listWorkers().map(w => ({id: w.id, clientId: w.clientId}));
  const states = (runtime.mobileHub as unknown as {workers: Map<string, {socket: {close(code: number, reason: string): void} | null}>}).workers;
  for (const worker of states.values()) worker.socket?.close(1001, 'lab transport fault');
  await new Promise(r => setTimeout(r, 500));
  const expired = runtime.mobileHub.expireDisconnectedWorkers(Date.now() + 60_001);
  return {before, expired};
});
runtime.app.post('/lab/hold', async () => {
  held = true; gate = new Promise<void>(r => {release = r;}); return {held};
});
runtime.app.post('/lab/release', async () => {
  held = false; release?.(); return {held};
});
runtime.app.post('/lab/deny', async () => {denied = true; return {denied};});
runtime.app.post('/lab/remove', async () => ({removed: runtime.mobileHub.listWorkers()
  .map(w => runtime.mobileHub.removeWorker(w.id)).filter(Boolean).length}));
runtime.app.post('/lab/shutdown', async () => {
  held = false; release?.();
  setTimeout(() => void runtime.close().then(() => process.exit(0)), 200);
  return {closing: true};
});
await runtime.app.listen({host: '127.0.0.1', port: 18910});
console.log('RECOVERY_CANARY_READY 18910');

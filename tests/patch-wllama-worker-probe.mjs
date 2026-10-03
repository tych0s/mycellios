import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const packageRoot = process.env.WLLAMA_PROBE_PACKAGE || join(tmpdir(), 'mycellios-wllama-probe', 'package');
const bundle = join(packageRoot, 'esm', 'index.js');
const backup = join(packageRoot, 'esm', 'index.mycellios-uninstrumented.js');
const original = await readFile(bundle, 'utf8');

if (process.argv.includes('--restore')) {
  const saved = await readFile(backup, 'utf8');
  await writeFile(bundle, saved);
  process.stdout.write(`Restored ${bundle}\n`);
  process.exit(0);
}

const pkg = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8'));
if (pkg.name !== '@wllama/wllama' || pkg.version !== '3.6.1') {
  throw new Error(`Expected @wllama/wllama 3.6.1, got ${pkg.name}@${pkg.version}`);
}
if (original.includes('MYCELLIOS_WORKER_GPU_PROBE')) {
  throw new Error('Worker probe is already installed; use --restore first');
}
const gapFlag = process.argv.find((arg) => arg.startsWith('--coalesce-gap='));
const coalesceGap = gapFlag ? Number(gapFlag.slice('--coalesce-gap='.length))
  : process.argv.includes('--coalesce') ? Infinity : -1;
const scatterMode = process.argv.includes('--scatter');
const cacheStages = process.argv.includes('--cache-stages');
const directBindCache = process.argv.includes('--direct-bind-cache');
const directStages = process.argv.includes('--direct-stages') || directBindCache;
if (coalesceGap !== Infinity && (!Number.isSafeInteger(coalesceGap) || coalesceGap < -1)) {
  throw new Error('Expected --coalesce-gap=N with a nonnegative integer number of bytes');
}
if (Number(scatterMode) + Number(cacheStages) + Number(directStages) + Number(coalesceGap >= 0) > 1) {
  throw new Error('Choose one write optimization');
}

const marker = 'var LLAMA_CPP_WORKER_CODE = ';
const start = original.indexOf(marker);
if (start < 0 || original.indexOf(marker, start + marker.length) >= 0) {
  throw new Error('Expected exactly one embedded llama.cpp worker');
}
const literalStart = start + marker.length;
const literalEnd = original.indexOf(';\n', literalStart);
if (literalEnd < 0) throw new Error('Worker string terminator not found');
const workerSource = JSON.parse(original.slice(literalStart, literalEnd));
if (!workerSource.startsWith('// Start the main llama.cpp')) {
  throw new Error('Unexpected embedded worker source');
}

const instrumentation = String.raw`
// MYCELLIOS_WORKER_GPU_PROBE: local diagnostics only.
(() => {
  const coalesceGap = ${coalesceGap};
  const scatterMode = ${scatterMode};
  const cacheStages = ${cacheStages};
  const directStages = ${directStages};
  const directBindCache = ${directBindCache};
  const coalesceUniform = coalesceGap >= 0;
  const stats = { submits: 0, writes: 0, writeBytes: 0,
    physicalWrites: 0, physicalWriteBytes: 0, maps: 0, mapWaitMs: 0,
    queueWaits: 0, queueWaitMs: 0, scatterPasses: 0, scatterWords: 0,
    uniformWrites: 0, unchangedUniformWrites: 0, uniformBytes: 0,
    unchangedUniformBytes: 0, sameBatchComparedBytes: 0,
    sameBatchChangedBytes: 0, sameBatchEqualBuffers: 0,
    cacheHits: 0, cacheMisses: 0, cacheCopyBytes: 0, cacheSavedWrites: 0,
    directHits: 0, directMisses: 0, directSavedWrites: 0,
    bindGroupCacheHits: 0,
    bindGroups: 0, uniformBindGroups: 0, setBindGroups: 0,
    uniformSetBindGroups: 0 };
  const snapshot = () => ({ ...stats });
  let previous = snapshot();
  let actionWriteSizes = new Map();
  let actionTargets = new Set();
  let previousWrite = null;
  let contiguousPairs = 0;
  let reportedSizes = false;
  let pendingWrites = [];
  let actionBatches = [];
  let currentActionName = '';
  let actionBatchIndex = 0;
  let cacheStageIndex = 0;
  const touchedUniform = new Set();
  const previousBatches = new WeakMap();
  const mergeRanges = (ranges, gap) => ranges.map((range) => [...range])
    .sort((a, b) => a[0] - b[0]).reduce((out, range) => {
      if (out.length && range[0] - out[out.length - 1][1] <= gap) {
        out[out.length - 1][1] = Math.max(out[out.length - 1][1], range[1]);
      } else out.push(range);
      return out;
    }, []);
  const summarize = (writes) => [...new Set(writes.map((x) => x.buffer))].map((buffer) => {
    const ranges = writes.filter((x) => x.buffer === buffer).map((x) => [x.offset, x.offset + x.bytes]);
    const min = Math.min(...ranges.map((x) => x[0]));
    const max = Math.max(...ranges.map((x) => x[1]));
    const merged = mergeRanges(ranges, 0);
    const plans = buffer.usage === 72 && buffer.size <= 65536
      ? Object.fromEntries([0, 32, 64, 128, 256, 512, 1024, Infinity].map((gap) => {
          const segments = mergeRanges(ranges, gap);
          return [gap === Infinity ? 'all' : String(gap), { calls: segments.length,
            bytes: segments.reduce((n, x) => n + x[1] - x[0], 0) }];
        })) : undefined;
    return { bufferSize: buffer.size, usage: buffer.usage, writes: ranges.length,
      bytes: ranges.reduce((n, x) => n + x[1] - x[0], 0), min, max,
      unionBytes: merged.reduce((n, x) => n + x[1] - x[0], 0), spans: merged.length,
      plans };
  });
  self.__mycelliosGpuStats = (name) => {
    currentActionName = name;
    actionBatchIndex = 0;
    cacheStageIndex = 0;
    touchedUniform.clear();
    actionWriteSizes = new Map();
    actionTargets = new Set();
    previousWrite = null;
    contiguousPairs = 0;
    pendingWrites = [];
    actionBatches = [];
    return snapshot();
  };
  self.__mycelliosReportAction = (name, before, started) => {
    const after = snapshot();
    const delta = Object.fromEntries(Object.keys(after).map((key) => [key, after[key] - before[key]]));
    const detail = name === 'get_result' && delta.writes && !reportedSizes
      ? { writeSizes: Object.fromEntries([...actionWriteSizes].sort((a, b) => b[1] - a[1])),
          uniqueTargets: actionTargets.size, contiguousPairs,
          batches: actionBatches }
      : {};
    if (Object.keys(detail).length) reportedSizes = true;
    postMessage({ verb: 'console.log', args: ['MYCELLIOS_GPU_PROBE_ACTION', JSON.stringify({
      epochMs: Date.now(), name, durationMs: performance.now() - started, ...delta, ...detail })] });
  };
  const queueProto = self.GPUQueue?.prototype;
  const bufferProto = self.GPUBuffer?.prototype;
  const deviceProto = self.GPUDevice?.prototype;
  const passProto = self.GPUComputePassEncoder?.prototype;
  const scatterTargets = new WeakMap();
  const cacheTargets = new WeakMap();
  const directTargets = new WeakMap();
  const uniformBindGroupSet = new WeakSet();
  const bindGroupCache = new Map();
  const resourceIds = new WeakMap();
  let nextResourceId = 1;
  const resourceId = (value) => {
    if (!resourceIds.has(value)) resourceIds.set(value, nextResourceId++);
    return resourceIds.get(value);
  };
  const uniformObserved = new WeakMap();
  if ((scatterMode || cacheStages) && deviceProto) {
    const createBuffer = deviceProto.createBuffer;
    deviceProto.createBuffer = function (descriptor) {
      if (descriptor.usage === 72 && descriptor.size <= 65536) {
        const buffer = createBuffer.call(this, { ...descriptor,
          usage: descriptor.usage | (scatterMode ? 128 : 4) });
        if (scatterMode) scatterTargets.set(buffer, { device: this, words: new Map(),
          patchBuffer: null, patchCapacity: 0, bindGroup: null, pipeline: null });
        if (cacheStages) cacheTargets.set(buffer, { device: this, pending: [],
          snapshots: [], cacheBuffer: null });
        return buffer;
      }
      return createBuffer.call(this, descriptor);
    };
  }
  if (deviceProto) {
    const createBindGroup = deviceProto.createBindGroup;
    deviceProto.createBindGroup = function (descriptor) {
      let resolved = descriptor;
      let replacedUniform = false;
      if (directStages && currentActionName === 'get_result' && actionBatchIndex < 16) {
        const entries = descriptor.entries?.map((entry) => {
          const buffer = entry.resource?.buffer;
          if (!(buffer instanceof self.GPUBuffer) || buffer.usage !== 72 || buffer.size > 65536) {
            return entry;
          }
          let state = directTargets.get(buffer);
          if (!state) {
            state = { device: this, buffers: [], snapshots: [], pending: [], pendingStage: -1 };
            directTargets.set(buffer, state);
          }
          if (!state.buffers[actionBatchIndex]) {
            state.buffers[actionBatchIndex] = this.createBuffer({ size: buffer.size, usage: buffer.usage });
          }
          replacedUniform = true;
          return { ...entry, resource: { ...entry.resource,
            buffer: state.buffers[actionBatchIndex] } };
        });
        if (entries?.some((entry, index) => entry !== descriptor.entries[index])) {
          resolved = { ...descriptor, entries };
        }
      }
      let cacheKey;
      if (directBindCache && replacedUniform && resolved.entries?.every((entry) => {
        const resource = entry.resource?.buffer || entry.resource;
        return resource && typeof resource === 'object';
      })) {
        cacheKey = resourceId(resolved.layout) + '|' + resolved.entries.map((entry) => {
          const resource = entry.resource?.buffer || entry.resource;
          return [entry.binding, resourceId(resource), entry.resource?.offset || 0,
            entry.resource?.size || 0].join(':');
        }).join('|');
        const cached = bindGroupCache.get(cacheKey);
        if (cached) {
          stats.bindGroupCacheHits++;
          return cached;
        }
      }
      const group = createBindGroup.call(this, resolved);
      stats.bindGroups++;
      if (cacheKey && bindGroupCache.size < 4096) bindGroupCache.set(cacheKey, group);
      if (resolved.entries?.some((entry) => {
        const buffer = entry.resource?.buffer || entry.resource;
        return buffer instanceof self.GPUBuffer &&
          (buffer.usage === 72 || scatterTargets.has(buffer) || cacheTargets.has(buffer));
      })) {
        stats.uniformBindGroups++;
        uniformBindGroupSet.add(group);
      }
      return group;
    };
  }
  if (passProto) {
    const setBindGroup = passProto.setBindGroup;
    passProto.setBindGroup = function (index, group, ...rest) {
      stats.setBindGroups++;
      if (uniformBindGroupSet.has(group)) stats.uniformSetBindGroups++;
      return setBindGroup.call(this, index, group, ...rest);
    };
  }
  if (queueProto) {
    const submit = queueProto.submit;
    const writeBuffer = queueProto.writeBuffer;
    const pendingByQueue = new WeakMap();
    const scatterPendingByQueue = new WeakMap();
    const cachePendingByQueue = new WeakMap();
    const directPendingByQueue = new WeakMap();
    const uniformShadows = new WeakMap();
    const flush = (queue) => {
      const pending = pendingByQueue.get(queue);
      if (!pending) return;
      for (const [buffer, ranges] of pending) {
        const shadow = uniformShadows.get(buffer);
        for (const [min, max] of mergeRanges(ranges, coalesceGap)) {
          const bytes = max - min;
          writeBuffer.call(queue, buffer, min, shadow.subarray(min, max));
          stats.physicalWrites++;
          stats.physicalWriteBytes += bytes;
        }
      }
      pending.clear();
    };
    const scatterCode = [
      'struct Words { data: array<u32> };',
      '@group(0) @binding(0) var<storage, read> patches: Words;',
      '@group(0) @binding(1) var<storage, read_write> destination: Words;',
      '@compute @workgroup_size(64)',
      'fn main(@builtin(global_invocation_id) id: vec3<u32>) {',
      '  if (id.x >= patches.data[0]) { return; }',
      '  let pos = 1u + id.x * 2u;',
      '  destination.data[patches.data[pos]] = patches.data[pos + 1u];',
      '}',
    ].join('\n');
    const flushScatter = (queue) => {
      const pending = scatterPendingByQueue.get(queue);
      if (!pending?.size) return [];
      const commandBuffers = [];
      for (const buffer of pending) {
        const state = scatterTargets.get(buffer);
        const words = state.words;
        if (!words.size) continue;
        if (!state.pipeline) {
          const module = state.device.createShaderModule({ code: scatterCode });
          state.pipeline = state.device.createComputePipeline({ layout: 'auto',
            compute: { module, entryPoint: 'main' } });
        }
        const packed = new Uint32Array(1 + 2 * words.size);
        packed[0] = words.size;
        let index = 1;
        for (const [offset, value] of words) {
          packed[index++] = offset;
          packed[index++] = value;
        }
        if (packed.byteLength > state.patchCapacity) {
          state.patchBuffer?.destroy();
          state.patchCapacity = packed.byteLength;
          state.patchBuffer = state.device.createBuffer({ size: state.patchCapacity, usage: 136 });
          state.bindGroup = state.device.createBindGroup({
            layout: state.pipeline.getBindGroupLayout(0), entries: [
              { binding: 0, resource: { buffer: state.patchBuffer } },
              { binding: 1, resource: { buffer } },
            ] });
        }
        writeBuffer.call(queue, state.patchBuffer, 0, packed);
        stats.physicalWrites++;
        stats.physicalWriteBytes += packed.byteLength;
        const encoder = state.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.pipeline);
        pass.setBindGroup(0, state.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(words.size / 64));
        pass.end();
        commandBuffers.push(encoder.finish());
        stats.scatterPasses++;
        stats.scatterWords += words.size;
        words.clear();
      }
      pending.clear();
      return commandBuffers;
    };
    const flushCache = (queue) => {
      const pending = cachePendingByQueue.get(queue);
      if (!pending?.size) return [];
      const commandBuffers = [];
      for (const buffer of pending) {
        const state = cacheTargets.get(buffer);
        if (!state.pending.length) continue;
        const stage = cacheStageIndex++;
        if (stage >= 16) {
          for (const write of state.pending) {
            writeBuffer.call(queue, buffer, write.offset, write.data);
            stats.physicalWrites++;
            stats.physicalWriteBytes += write.data.byteLength;
          }
          state.pending = [];
          continue;
        }
        const current = uniformObserved.get(buffer).values;
        const before = state.snapshots[stage];
        let equal = Boolean(before);
        if (before) for (let i = 0; i < current.length; i++) {
          if (before[i] !== current[i]) { equal = false; break; }
        }
        if (!state.cacheBuffer) state.cacheBuffer = state.device.createBuffer({
          size: buffer.size * 16, usage: 12 });
        const encoder = state.device.createCommandEncoder();
        if (equal) {
          encoder.copyBufferToBuffer(state.cacheBuffer, stage * buffer.size,
            buffer, 0, buffer.size);
          stats.cacheHits++;
          stats.cacheSavedWrites += state.pending.length;
        } else {
          for (const write of state.pending) {
            writeBuffer.call(queue, buffer, write.offset, write.data);
            stats.physicalWrites++;
            stats.physicalWriteBytes += write.data.byteLength;
          }
          encoder.copyBufferToBuffer(buffer, 0, state.cacheBuffer,
            stage * buffer.size, buffer.size);
          state.snapshots[stage] = current.slice();
          stats.cacheMisses++;
        }
        stats.cacheCopyBytes += buffer.size;
        commandBuffers.push(encoder.finish());
        state.pending = [];
      }
      pending.clear();
      return commandBuffers;
    };
    const flushDirect = (queue) => {
      const pending = directPendingByQueue.get(queue);
      if (!pending?.size) return;
      for (const buffer of pending) {
        const state = directTargets.get(buffer);
        const stage = state.pendingStage;
        const current = uniformObserved.get(buffer).values;
        const before = state.snapshots[stage];
        let equal = Boolean(before);
        if (before) for (let i = 0; i < current.length; i++) {
          if (before[i] !== current[i]) { equal = false; break; }
        }
        if (equal) {
          stats.directHits++;
          stats.directSavedWrites += state.pending.length;
        } else {
          writeBuffer.call(queue, state.buffers[stage], 0, current);
          stats.physicalWrites++;
          stats.physicalWriteBytes += current.byteLength;
          stats.directMisses++;
          state.snapshots[stage] = current.slice();
        }
        state.pending = [];
        state.pendingStage = -1;
      }
      pending.clear();
    };
    queueProto.submit = function (...args) {
      stats.submits++;
      if (currentActionName === 'get_result' && touchedUniform.size) {
        for (const buffer of touchedUniform) {
          const current = uniformObserved.get(buffer).values;
          const prior = previousBatches.get(buffer) || [];
          const before = prior[actionBatchIndex];
          if (before) {
            let changed = 0;
            for (let i = 0; i < current.length; i++) if (current[i] !== before[i]) changed++;
            stats.sameBatchComparedBytes += current.length;
            stats.sameBatchChangedBytes += changed;
            if (!changed) stats.sameBatchEqualBuffers++;
          }
          prior[actionBatchIndex] = current.slice();
          previousBatches.set(buffer, prior);
        }
        actionBatchIndex++;
      }
      touchedUniform.clear();
      if (!reportedSizes && pendingWrites.length) {
        actionBatches.push(summarize(pendingWrites));
        pendingWrites = [];
      }
      flush(this);
      if (directStages) flushDirect(this);
      if (scatterMode) {
        const prefix = flushScatter(this);
        if (prefix.length) return submit.call(this, [...prefix, ...args[0]]);
      }
      if (cacheStages) {
        const prefix = flushCache(this);
        if (prefix.length) return submit.call(this, [...prefix, ...args[0]]);
      }
      return submit.apply(this, args);
    };
    queueProto.writeBuffer = function (...args) {
      stats.writes++;
      const data = args[2];
      const elementBytes = ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT || 1 : 1;
      const bytes = (args[4] === undefined
        ? data.byteLength / elementBytes - (args[3] || 0) : args[4]) * elementBytes;
      stats.writeBytes += bytes;
      actionWriteSizes.set(bytes, (actionWriteSizes.get(bytes) || 0) + 1);
      actionTargets.add(args[0]);
      if (!reportedSizes) pendingWrites.push({ buffer: args[0], offset: args[1], bytes });
      if (previousWrite?.buffer === args[0] && previousWrite.end === args[1]) contiguousPairs++;
      previousWrite = { buffer: args[0], end: args[1] + bytes };
      if ((args[0].usage === 72 || scatterTargets.has(args[0]) || cacheTargets.has(args[0]))
          && args[0].size <= 65536) {
        touchedUniform.add(args[0]);
        let observed = uniformObserved.get(args[0]);
        if (!observed) {
          observed = { values: new Uint8Array(args[0].size), known: new Uint8Array(args[0].size) };
          uniformObserved.set(args[0], observed);
        }
        const byteOffset = (args[3] || 0) * elementBytes;
        const source = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset + byteOffset, bytes)
          : new Uint8Array(data, byteOffset, bytes);
        let unchanged = true;
        for (let i = 0; i < bytes; i++) {
          const targetOffset = args[1] + i;
          if (!observed.known[targetOffset] || observed.values[targetOffset] !== source[i]) unchanged = false;
          observed.values[targetOffset] = source[i];
          observed.known[targetOffset] = 1;
        }
        stats.uniformWrites++;
        stats.uniformBytes += bytes;
        if (unchanged) {
          stats.unchangedUniformWrites++;
          stats.unchangedUniformBytes += bytes;
        }
      }
      if (coalesceUniform && args[0].usage === 72 && args[0].size <= 65536) {
        let shadow = uniformShadows.get(args[0]);
        if (!shadow) {
          shadow = new Uint8Array(args[0].size);
          uniformShadows.set(args[0], shadow);
        }
        const dataOffset = (args[3] || 0) * elementBytes;
        const source = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset + dataOffset, bytes)
          : new Uint8Array(data, dataOffset, bytes);
        shadow.set(source, args[1]);
        let pending = pendingByQueue.get(this);
        if (!pending) {
          pending = new Map();
          pendingByQueue.set(this, pending);
        }
        let ranges = pending.get(args[0]);
        if (!ranges) {
          ranges = [];
          pending.set(args[0], ranges);
        }
        ranges.push([args[1], args[1] + bytes]);
        return;
      }
      if (scatterMode && scatterTargets.has(args[0])) {
        const state = scatterTargets.get(args[0]);
        const dataOffset = (args[3] || 0) * elementBytes;
        const view = ArrayBuffer.isView(data)
          ? new DataView(data.buffer, data.byteOffset + dataOffset, bytes)
          : new DataView(data, dataOffset, bytes);
        for (let offset = 0; offset < bytes; offset += 4) {
          state.words.set((args[1] + offset) / 4, view.getUint32(offset, true));
        }
        let pending = scatterPendingByQueue.get(this);
        if (!pending) {
          pending = new Set();
          scatterPendingByQueue.set(this, pending);
        }
        pending.add(args[0]);
        return;
      }
      if (cacheStages && currentActionName === 'get_result' && cacheTargets.has(args[0])) {
        const state = cacheTargets.get(args[0]);
        const dataOffset = (args[3] || 0) * elementBytes;
        const source = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset + dataOffset, bytes)
          : new Uint8Array(data, dataOffset, bytes);
        state.pending.push({ offset: args[1], data: source.slice() });
        let pending = cachePendingByQueue.get(this);
        if (!pending) {
          pending = new Set();
          cachePendingByQueue.set(this, pending);
        }
        pending.add(args[0]);
        return;
      }
      if (directStages && currentActionName === 'get_result' && directTargets.has(args[0])
          && actionBatchIndex < 16) {
        const state = directTargets.get(args[0]);
        state.pending.push(bytes);
        state.pendingStage = actionBatchIndex;
        let pending = directPendingByQueue.get(this);
        if (!pending) {
          pending = new Set();
          directPendingByQueue.set(this, pending);
        }
        pending.add(args[0]);
        return;
      }
      stats.physicalWrites++;
      stats.physicalWriteBytes += bytes;
      return writeBuffer.apply(this, args);
    };
    const onSubmittedWorkDone = queueProto.onSubmittedWorkDone;
    queueProto.onSubmittedWorkDone = function (...args) {
      flush(this);
      if (directStages) flushDirect(this);
      if (scatterMode) {
        const prefix = flushScatter(this);
        if (prefix.length) {
          submit.call(this, prefix);
          stats.submits++;
        }
      }
      if (cacheStages) {
        const prefix = flushCache(this);
        if (prefix.length) {
          submit.call(this, prefix);
          stats.submits++;
        }
      }
      const started = performance.now();
      stats.queueWaits++;
      return onSubmittedWorkDone.apply(this, args).then((value) => {
        stats.queueWaitMs += performance.now() - started;
        return value;
      });
    };
  }
  if (bufferProto) {
    const mapAsync = bufferProto.mapAsync;
    bufferProto.mapAsync = function (...args) {
      const started = performance.now();
      stats.maps++;
      return mapAsync.apply(this, args).then((value) => {
        stats.mapWaitMs += performance.now() - started;
        return value;
      });
    };
  }
  postMessage({ verb: 'console.log', args: ['MYCELLIOS_GPU_PROBE_READY', JSON.stringify({
    gpuQueue: Boolean(queueProto), gpuBuffer: Boolean(bufferProto),
    gpuDevice: Boolean(deviceProto), gpuComputePass: Boolean(passProto),
    coalesceGap, scatterMode, cacheStages, directStages, directBindCache })] });
  setInterval(() => {
    const current = snapshot();
    const delta = Object.fromEntries(Object.keys(current).map((key) => [key, current[key] - previous[key]]));
    previous = current;
    if (delta.submits || delta.maps || delta.queueWaits) {
      postMessage({ verb: 'console.log', args: ['MYCELLIOS_GPU_PROBE_SAMPLE', JSON.stringify({
        epochMs: Date.now(), elapsedMs: Math.round(performance.now()), ...delta })] });
    }
  }, 1000);
})();
`;

const actionStart = `  const argAction = args[0];\n`;
const actionEnd = `  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError`;
if (workerSource.split(actionStart).length !== 2 || workerSource.split(actionEnd).length !== 2) {
  throw new Error('Unexpected worker action function');
}
const instrumentedWorker = (instrumentation + workerSource)
  .replace(actionStart, `${actionStart}  const probeBefore = self.__mycelliosGpuStats?.(argAction);\n  const probeStart = performance.now();\n`)
  .replace(actionEnd, `  } catch (err) {\n    handleError(err);\n  } finally {\n    if (probeBefore) self.__mycelliosReportAction?.(argAction, probeBefore, probeStart);\n  }\n};\n\nfunction handleError`);
const patched = original.slice(0, literalStart) + JSON.stringify(instrumentedWorker)
  + original.slice(literalEnd);
let saved;
try {
  saved = await readFile(backup, 'utf8');
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
if (saved && saved !== original) {
  throw new Error('Current bundle differs from the saved original');
}
if (!saved) await writeFile(backup, original, { flag: 'wx' });
await writeFile(bundle, patched);
process.stdout.write(`Instrumented ${bundle}\n`);

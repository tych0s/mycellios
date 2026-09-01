"""Puente TCP<->WS con RECONEXIÓN y reanudación por offset (un nodo-etapa).

El gateway de Salad cierra los WS a los ~3-4 min; mycellios usa UNA conexión TCP
persistente por enlace. Así que el puente mantiene el TCP local SIEMPRE abierto y
un gestor de WS que reconecta al relé tras cada corte, reanudando el byte-stream
por offset sin perder ni duplicar (ver relay.py para el protocolo).

Config por entorno:
  RELAY_URL, SALAD_API_KEY (opcional si relé auth:false),
  GDLP_BRIDGE_ENDPOINTS = JSON [{"mode":"accept"|"dial","host","port","room"}, ...]
"""
import asyncio
import json
import hashlib
import hmac
import os
import struct
import sys
import time

KEEPALIVE_SEC = 15
UA = "gdlp-bridge/2.0"
HELLO, HELLO_ACK, DATA, ACK, ACKW, KEEPALIVE = 0x10, 0x11, 0x20, 0x21, 0x22, 0x01
PING, PONG = 0x30, 0x31   # RTT puente<->relé: diagnostica si el sobrecoste por
                          # paso del decode es latencia de red o cómputo
RTT_EVERY_SEC = 20


def _install_deps():
    if os.environ.get("GDLP_SKIP_PIP") == "1":
        return
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "aiohttp"], check=True)


RELAY_SECRET = os.environ.get("GDLP_RELAY_SECRET", "").encode()


def hello_signature(room, eid, gen):
    """Debe coincidir byte a byte con `relay.hello_signature`."""
    mac = hmac.new(RELAY_SECRET, digestmod=hashlib.sha256)
    mac.update(room.encode())
    mac.update(b"\x00")
    mac.update(eid)
    mac.update(struct.pack(">Q", gen))
    return mac.digest()


def enc_hello(room, eid, gen, recv):
    rb = room.encode()
    frame = (bytes([HELLO]) + struct.pack(">H", len(rb)) + rb + eid
             + struct.pack(">Q", gen) + struct.pack(">Q", recv))
    # El relé exige HMAC salvo que corra en modo anónimo explícito (tests).
    if RELAY_SECRET:
        frame += hello_signature(room, eid, gen)
    return frame


def enc_data(off, payload):
    return bytes([DATA]) + struct.pack(">Q", off) + payload


def enc_ackw(off):
    return bytes([ACKW]) + struct.pack(">Q", off)


class Endpoint:
    def __init__(self, relay_url, key, room, host, port, eid=None, gen=1):
        self.relay_url = relay_url
        self.key = key
        self.room = room
        self.host = host
        self.port = port
        self.eid = eid if eid else os.urandom(16)
        self.gen = int(gen)
        self.reader = None
        self.writer = None
        self.out_offset = 0     # total leído del TCP local (siguiente offset a asignar)
        self.ack_out = 0        # confirmado recibido por el relé -> out_buf empieza aquí
        self.out_buf = bytearray()
        self.in_written = 0     # total escrito al TCP local (entrante)
        self.lock = asyncio.Lock()
        self.new_data = asyncio.Event()
        self.local_eof = False
        self.stats = {"tcp_to_ws": 0, "ws_to_tcp": 0, "reconnects": 0}

    async def local_reader(self):
        """Además de bombear, mide la FRACCIÓN DE BURBUJA del enlace: el túnel ve
        TODO el tráfico inter-etapa, así que los huecos entre llegadas son tiempo
        en que la etapa de destino está OCIOSA esperando trabajo."""
        last = None
        try:
            while True:
                data = await self.reader.read(65536)
                if not data:
                    break
                now = time.time()
                if last is not None:
                    gap = now - last
                    s = self.stats
                    s["gap_n"] = s.get("gap_n", 0) + 1
                    s["gap_sum"] = s.get("gap_sum", 0.0) + gap
                    if gap > 0.005:  # >5 ms sin datos = burbuja apreciable
                        s["bubble_sum"] = s.get("bubble_sum", 0.0) + gap
                        s["bubble_n"] = s.get("bubble_n", 0) + 1
                last = now
                async with self.lock:
                    self.out_buf += data
                    self.out_offset += len(data)
                    self.stats["tcp_to_ws"] += len(data)
                self.new_data.set()
        finally:
            self.local_eof = True
            self.new_data.set()

    async def _connect_ws(self, session):
        import aiohttp
        hdr = {"User-Agent": UA}
        if self.key:
            hdr["Salad-Api-Key"] = self.key
        ws = await session.ws_connect(self.relay_url, headers=hdr, max_msg_size=32 * 1024 * 1024,
                                      heartbeat=KEEPALIVE_SEC, autoping=True)
        async with self.lock:
            recv_in = self.in_written
        await ws.send_bytes(enc_hello(self.room, self.eid, self.gen, recv_in))
        first = await ws.receive()
        from aiohttp import WSMsgType
        if first.type != WSMsgType.BINARY or not first.data or first.data[0] != HELLO_ACK:
            await ws.close()
            raise ConnectionError("sin HELLO_ACK")
        resume = struct.unpack(">Q", first.data[1:9])[0]
        return ws, resume

    async def _rtt_prober(self, ws):
        """Ping periódico al relé: mide el RTT de UN salto del túnel. El camino
        inter-etapa real es puente->relé->puente (~2 saltos)."""
        try:
            while not ws.closed:
                await ws.send_bytes(bytes([PING]) + struct.pack(">d", time.time()))
                s = self.stats
                if s.get("gap_n"):
                    span = s["gap_sum"]
                    bub = s.get("bubble_sum", 0.0)
                    print(f"[bridge] BURBUJA room={self.room} "
                          f"ociosa={bub/span*100:.1f}% de {span:.1f}s activos "
                          f"(huecos>5ms: {s.get('bubble_n',0)}/{s['gap_n']}, "
                          f"gap_medio={span/s['gap_n']*1000:.1f}ms)", flush=True)
                await asyncio.sleep(RTT_EVERY_SEC)
        except Exception:
            pass

    async def _sender(self, ws):
        from aiohttp import WSMsgType
        async with self.lock:
            send_ptr = max(self.ack_out, min(getattr(self, "_resume", 0), self.out_offset))
        while not ws.closed:
            chunk = None
            base = 0
            async with self.lock:
                if send_ptr < self.out_offset:
                    idx = send_ptr - self.ack_out
                    if idx < 0:
                        send_ptr = self.ack_out
                        idx = 0
                    chunk = bytes(self.out_buf[idx:])
                    base = send_ptr
                    send_ptr = self.out_offset
            if chunk:
                await ws.send_bytes(enc_data(base, chunk))
            else:
                try:
                    await ws.send_bytes(enc_ackw(self.in_written))  # confirma entrante
                except Exception:
                    break
                if self.local_eof and send_ptr >= self.out_offset:
                    # nada más que enviar y el local cerró: deja que el receiver termine
                    await asyncio.sleep(0.2)
                try:
                    await asyncio.wait_for(self.new_data.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                self.new_data.clear()

    async def _receiver(self, ws):
        from aiohttp import WSMsgType
        async for msg in ws:
            if msg.type != WSMsgType.BINARY or not msg.data:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
                continue
            t = msg.data[0]
            if t == KEEPALIVE:
                continue
            if t == PONG:
                sent = struct.unpack(">d", msg.data[1:9])[0]
                rtt_ms = (time.time() - sent) * 1000.0
                self.stats["rtt_ms_last"] = round(rtt_ms, 2)
                s = self.stats
                s["rtt_n"] = s.get("rtt_n", 0) + 1
                s["rtt_sum"] = s.get("rtt_sum", 0.0) + rtt_ms
                s["rtt_ms_avg"] = round(s["rtt_sum"] / s["rtt_n"], 2)
                s["rtt_ms_min"] = round(min(s.get("rtt_ms_min", 1e9), rtt_ms), 2)
                print(f"[bridge] RTT room={self.room} last={rtt_ms:.1f}ms "
                      f"avg={s['rtt_ms_avg']}ms min={s['rtt_ms_min']}ms n={s['rtt_n']}", flush=True)
                continue
            if t == ACK:
                acked = struct.unpack(">Q", msg.data[1:9])[0]
                async with self.lock:
                    if acked > self.ack_out:
                        del self.out_buf[:acked - self.ack_out]
                        self.ack_out = acked
            elif t == DATA:
                off = struct.unpack(">Q", msg.data[1:9])[0]
                payload = msg.data[9:]
                end = off + len(payload)
                if end <= self.in_written:
                    continue  # duplicado
                start = self.in_written - off if off < self.in_written else 0
                if off > self.in_written:
                    print(f"[bridge] HUECO entrante room={self.room} off={off}>{self.in_written}", flush=True)
                    continue
                new = payload[start:]
                self.writer.write(new)
                await self.writer.drain()
                self.in_written = end
                self.stats["ws_to_tcp"] += len(new)

    async def run(self, session):
        rtask = asyncio.create_task(self.local_reader())
        backoff = 0.3
        try:
            while not (self.local_eof and len(self.out_buf) == 0):
                try:
                    ws, resume = await self._connect_ws(session)
                    self._resume = resume
                    backoff = 0.3
                    print(f"[bridge] room={self.room} WS conectado (reanuda envío en {resume}, reconexiones={self.stats['reconnects']})", flush=True)
                    st = asyncio.create_task(self._sender(ws))
                    rt = asyncio.create_task(self._receiver(ws))
                    pt = asyncio.create_task(self._rtt_prober(ws))
                    done, pending = await asyncio.wait({st, rt}, return_when=asyncio.FIRST_COMPLETED)
                    pt.cancel()
                    for p in pending:
                        p.cancel()
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    self.stats["reconnects"] += 1
                except Exception as e:
                    print(f"[bridge] room={self.room} WS error: {e}; reintento en {backoff:.1f}s", flush=True)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                if self.local_eof and len(self.out_buf) == 0:
                    break
        finally:
            rtask.cancel()
            try:
                self.writer.close()
            except Exception:
                pass
            print(f"[bridge] room={self.room} FIN tcp->ws={self.stats['tcp_to_ws']} ws->tcp={self.stats['ws_to_tcp']} reconexiones={self.stats['reconnects']}", flush=True)


async def run_accept(session, relay, key, ep):
    host, port, room = ep["host"], int(ep["port"]), ep["room"]
    eid = bytes.fromhex(ep["eid"]) if ep.get("eid") else None
    gen = ep.get("gen", 1)

    async def on_conn(reader, writer):
        print(f"[bridge] ACCEPT '{room}' conexión local entrante", flush=True)
        e = Endpoint(relay, key, room, host, port, eid=eid, gen=gen)
        e.reader, e.writer = reader, writer
        await e.run(session)

    server = await asyncio.start_server(on_conn, host, port)
    print(f"[bridge] ACCEPT escuchando {host}:{port} -> sala '{room}'", flush=True)
    async with server:
        await server.serve_forever()


async def run_dial(session, relay, key, ep):
    host, port, room = ep["host"], int(ep["port"]), ep["room"]
    reader = writer = None
    for _ in range(1200):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            break
        except OSError:
            await asyncio.sleep(0.5)
    if writer is None:
        print(f"[bridge] DIAL '{room}' no pudo conectar a {host}:{port}", flush=True)
        return
    print(f"[bridge] DIAL '{room}' conectado a listener local {host}:{port}", flush=True)
    e = Endpoint(relay, key, room, host, port, eid=bytes.fromhex(ep["eid"]) if ep.get("eid") else None, gen=ep.get("gen", 1))
    e.reader, e.writer = reader, writer
    await e.run(session)


async def _main():
    import aiohttp
    relay = os.environ.get("RELAY_URL")
    key = os.environ.get("SALAD_API_KEY", "")
    eps = json.loads(os.environ.get("GDLP_BRIDGE_ENDPOINTS", "[]"))
    if not relay or not eps:
        sys.exit("faltan RELAY_URL y/o GDLP_BRIDGE_ENDPOINTS")
    print(f"[bridge] relé={relay} endpoints={json.dumps(eps)}", flush=True)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for ep in eps:
            if ep["mode"] == "accept":
                tasks.append(asyncio.create_task(run_accept(session, relay, key, ep)))
            elif ep["mode"] == "dial":
                tasks.append(asyncio.create_task(run_dial(session, relay, key, ep)))
            else:
                sys.exit(f"modo desconocido: {ep['mode']}")
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    _install_deps()
    asyncio.run(_main())

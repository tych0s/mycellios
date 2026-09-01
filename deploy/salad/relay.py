"""Relé TCP-sobre-WS con RECONEXIÓN y reanudación por offset (nodo CPU-only).

Salad/Cloudflare cierra los WebSocket a los ~3-4 min PASE LO QUE PASE (medido:
194-256 s, ni el keepalive lo evita). Como mycellios usa UNA conexión TCP
persistente por enlace, el túnel DEBE sobrevivir a esos cortes sin perder ni
duplicar bytes. Solución (patrón de producción con `sequence`): cada lado
numera su stream por offset de bytes; el relé y los puentes PERSISTEN sus
contadores y buffers; al reconectar un WS se reenvía solo el hueco.

El relé empareja 2 "slots" por sala (identificados por endpoint_id estable, así
sobreviven reconexiones). Cada slot tiene un stream SALIENTE (bytes que manda al
relé, destinados al par). El relé bufferiza el saliente de cada slot hasta que el
par confirma haberlo ESCRITO en su TCP local (ACKW), y lo reenvía desde el offset
pedido cuando el par (re)aparece.

Frames BINARIOS: [tipo][...]
  0x10 HELLO      [2B room_len][room][16B endpoint_id][8B gen][8B recv_offset][32B hmac?]  bridge->relé
  0x11 HELLO_ACK  [8B resume_send]                                       relé->bridge
  0x20 DATA       [8B offset][payload]
  0x21 ACK        [8B acked]   relé->bridge: recibido tu saliente hasta acked
  0x22 ACKW       [8B written] bridge->relé: escribí mi entrante hasta written
  0x01 KEEPALIVE  (reduce muertes por inactividad antes del corte duro)

SEGURIDAD (25-07-2026). El relé vive en una URL pública y el nombre de sala
("f0"/"ret") es adivinable, así que un HELLO ajeno bastaba para destruir una
tubería viva. Tres defensas:
  1. HMAC obligatorio en el HELLO (GDLP_RELAY_SECRET), ligado a sala+eid+gen.
     Se puede desactivar SOLO con GDLP_RELAY_ALLOW_ANONYMOUS=1 (tests locales).
  2. El control de admisión de slots va ANTES del reset por generación: un
     tercer endpoint no puede vaciar los buffers de los dos legítimos.
  3. Cota al buffer de salida: un par atascado ya no puede tumbar el relé por
     memoria (el contenedor tiene 2 GB y el mensaje máximo son 32 MB).
"""
import asyncio
import hmac
import hashlib
import json
import os
import socket
import struct
import sys
import time

KEEPALIVE_SEC = 15
MAXLIFE_SEC = float(os.environ.get("GDLP_WS_MAXLIFE_SEC", "0"))  # 0=off; >0 cierra el WS tras N s (test)
RELAY_SECRET = os.environ.get("GDLP_RELAY_SECRET", "").encode()
ALLOW_ANONYMOUS = os.environ.get("GDLP_RELAY_ALLOW_ANONYMOUS") == "1"
# Un enlace sano nunca acumula: el par confirma la escritura (ACKW) y el buffer
# se vacía. Acumular decenas de MB significa que el par murió o está atascado;
# en ese caso conviene cortar ese enlace y no perder el relé entero.
MAX_BUFFER_BYTES = int(os.environ.get("GDLP_RELAY_MAX_BUFFER_BYTES", str(64 * 1024 * 1024)))


def hello_signature(room_name, eid, gen):
    """HMAC del HELLO. Liga la sala, la identidad y la generación: sin el
    secreto no se puede entrar en una sala ajena ni subir su generación."""
    mac = hmac.new(RELAY_SECRET, digestmod=hashlib.sha256)
    mac.update(room_name.encode())
    mac.update(b"\x00")
    mac.update(eid)
    mac.update(struct.pack(">Q", gen))
    return mac.digest()

HELLO, HELLO_ACK, DATA, ACK, ACKW, KEEPALIVE = 0x10, 0x11, 0x20, 0x21, 0x22, 0x01
PING, PONG = 0x30, 0x31   # medición de RTT puente<->relé (diagnóstico del sobrecoste)

STATE = {"role": "relay", "started": time.time(), "rooms": {}, "total_bytes": 0}


def _install_deps():
    if os.environ.get("GDLP_SKIP_PIP") == "1":
        return
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "aiohttp"], check=True)


def enc_data(off, payload):
    return bytes([DATA]) + struct.pack(">Q", off) + payload


def enc_ack(off):
    return bytes([ACK]) + struct.pack(">Q", off)


def enc_hello_ack(resume):
    return bytes([HELLO_ACK]) + struct.pack(">Q", resume)


class Slot:
    __slots__ = ("eid", "ws", "out_base", "out_buf", "fwd_ptr")

    def __init__(self, eid):
        self.eid = eid
        self.ws = None
        self.out_base = 0            # offset de out_buf[0]
        self.out_buf = bytearray()   # saliente bufferizado [out_base, out_base+len)
        self.fwd_ptr = 0             # cuánto del saliente se ha mandado al par actual

    @property
    def recv(self):
        return self.out_base + len(self.out_buf)


class Room:
    def __init__(self, name):
        self.name = name
        self.slots = {}   # eid -> Slot
        self.gen = 0      # generación: un /start nuevo (gen mayor) resetea los buffers

    def peer(self, eid):
        for e, s in self.slots.items():
            if e != eid:
                return s
        return None


ROOMS = {}


def parse_hello(data):
    """Descompone un HELLO en (room_name, eid, gen, recv_offset, signature).

    `signature` es b"" si el frame no la trae. Separada del handler para que la
    ruta crítica de seguridad se pueda probar sin levantar un WebSocket.
    """
    rlen = struct.unpack(">H", data[1:3])[0]
    room_name = data[3:3 + rlen].decode()
    eid = bytes(data[3 + rlen:3 + rlen + 16])
    gen = struct.unpack(">Q", data[3 + rlen + 16:3 + rlen + 24])[0]
    recv_offset = struct.unpack(">Q", data[3 + rlen + 24:3 + rlen + 32])[0]
    signature = bytes(data[3 + rlen + 32:3 + rlen + 64])
    return room_name, eid, gen, recv_offset, signature


def authorize_hello(room_name, eid, gen, signature):
    """¿Se acepta este HELLO? Devuelve None si sí, o el motivo del rechazo."""
    if RELAY_SECRET:
        if not hmac.compare_digest(signature, hello_signature(room_name, eid, gen)):
            return "hmac_invalido"
        return None
    if not ALLOW_ANONYMOUS:
        return "falta_secreto"
    return None


def admit_slot(room, eid, gen):
    """Admite `eid` en `room` y aplica el reset por generación.

    ORDEN CRÍTICO: la admisión va ANTES del reset. Al revés, un tercer endpoint
    vaciaba los buffers de los dos slots legítimos y sólo DESPUÉS era rechazado
    — un solo HELLO con `gen` alto mataba una tubería viva, y con gen=2^63 la
    dejaba envenenada para siempre.

    Devuelve (slot, None) si entra, o (None, motivo) si se rechaza.
    """
    slot = room.slots.get(eid)
    if slot is None and len(room.slots) >= 2:
        return None, "sala_llena"
    if gen > room.gen:
        for s in room.slots.values():
            s.out_base = 0
            s.out_buf = bytearray()
            s.fwd_ptr = 0
        room.gen = gen
    if slot is None:
        slot = Slot(eid)
        room.slots[eid] = slot
    return slot, None


def _stat(room):
    STATE["rooms"][room.name] = {
        "slots": len(room.slots),
        "buffered": sum(len(s.out_buf) for s in room.slots.values()),
        "recv": {("s%d" % i): s.recv for i, s in enumerate(room.slots.values())},
    }


async def _main():
    from aiohttp import web, WSMsgType

    routes = web.RouteTableDef()

    @routes.get("/healthz")
    async def healthz(req):
        return web.Response(text="ok")

    @routes.get("/status")
    async def status(req):
        return web.json_response(STATE)

    @routes.get("/relay")
    async def relay(req):
        ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024, heartbeat=KEEPALIVE_SEC)
        await ws.prepare(req)

        first = await ws.receive()
        if first.type != WSMsgType.BINARY or not first.data or first.data[0] != HELLO:
            await ws.close()
            return ws
        room_name, eid, gen, recv_offset, signature = parse_hello(first.data)

        # (1) AUTENTICACIÓN antes de tocar cualquier estado de la sala.
        denied = authorize_hello(room_name, eid, gen, signature)
        if denied:
            print(f"[relay] HELLO RECHAZADO room={room_name} ({denied})", flush=True)
            await ws.close()
            return ws

        room = ROOMS.setdefault(room_name, Room(room_name))
        # (2) ADMISIÓN antes del reset por generación (ver `admit_slot`).
        prev_gen = room.gen
        slot, denied = admit_slot(room, eid, gen)
        if denied:
            print(f"[relay] HELLO RECHAZADO room={room_name} ({denied}, {len(room.slots)} slots)", flush=True)
            await ws.close()
            return ws
        if room.gen > prev_gen:
            print(f"[relay] room={room_name} RESET a gen={room.gen}", flush=True)
        slot.ws = ws
        print(f"[relay] HELLO room={room_name} eid={eid.hex()[:8]} recv_in={recv_offset} slots={len(room.slots)}", flush=True)

        peer = room.peer(eid)
        # el par debe REENVIAR su saliente a este slot desde recv_offset (lo que este ya escribió)
        if peer is not None:
            # descarta saliente del par ya escrito por nosotros
            discard = recv_offset - peer.out_base
            if discard > 0:
                del peer.out_buf[:discard]
                peer.out_base = recv_offset
            peer.fwd_ptr = max(peer.out_base, recv_offset)
        # dile a este slot desde dónde reanudar su ENVÍO (lo que el relé ya recibió)
        await ws.send_bytes(enc_hello_ack(slot.recv))
        # reenvía el saliente pendiente del par hacia nosotros
        if peer is not None and peer.ws is not None and len(peer.out_buf) > 0:
            start = peer.fwd_ptr - peer.out_base
            if start < len(peer.out_buf):
                await ws.send_bytes(enc_data(peer.fwd_ptr, bytes(peer.out_buf[start:])))
                peer.fwd_ptr = peer.recv
        _stat(room)

        async def _keepalive():
            try:
                while not ws.closed:
                    await asyncio.sleep(KEEPALIVE_SEC)
                    await ws.send_bytes(bytes([KEEPALIVE]))
            except Exception:
                pass

        async def _maxlife():
            if MAXLIFE_SEC > 0:
                await asyncio.sleep(MAXLIFE_SEC)
                try:
                    await ws.close()
                except Exception:
                    pass

        ka = asyncio.create_task(_keepalive())
        ml = asyncio.create_task(_maxlife())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.BINARY or not msg.data:
                    if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                        break
                    continue
                t = msg.data[0]
                if t == KEEPALIVE:
                    continue
                if t == PING:
                    # devuelve la marca de tiempo intacta -> el puente calcula el RTT
                    await ws.send_bytes(bytes([PONG]) + msg.data[1:])
                    continue
                if t == DATA:
                    off = struct.unpack(">Q", msg.data[1:9])[0]
                    payload = msg.data[9:]
                    recv = slot.recv
                    end = off + len(payload)
                    if end <= recv:
                        continue  # duplicado completo
                    if off > recv:
                        print(f"[relay] HUECO room={room_name} off={off}>recv={recv}; cerrando", flush=True)
                        break
                    new = payload[recv - off:]
                    # (3) COTA AL BUFFER. Sin esto, un par que no confirma
                    # escrituras (muerto, atascado u hostil) hace crecer out_buf
                    # sin límite hasta tumbar el relé por memoria.
                    if len(slot.out_buf) + len(new) > MAX_BUFFER_BYTES:
                        print(f"[relay] BUFFER DESBORDADO room={room_name} eid={eid.hex()[:8]} "
                              f"buffered={len(slot.out_buf)} nuevo={len(new)} max={MAX_BUFFER_BYTES}; "
                              "el par no confirma escrituras, cerrando este enlace", flush=True)
                        break
                    slot.out_buf += new
                    STATE["total_bytes"] += len(new)
                    await ws.send_bytes(enc_ack(slot.recv))  # confirma recepción a este slot
                    peer = room.peer(eid)
                    if peer is not None and peer.ws is not None:
                        start = slot.fwd_ptr - slot.out_base
                        if start < len(slot.out_buf):
                            try:
                                await peer.ws.send_bytes(enc_data(slot.fwd_ptr, bytes(slot.out_buf[start:])))
                                slot.fwd_ptr = slot.recv
                            except Exception:
                                pass
                    _stat(room)
                elif t == ACKW:
                    # este slot escribió su ENTRANTE (=saliente del par) hasta y -> descarta buffer del par
                    y = struct.unpack(">Q", msg.data[1:9])[0]
                    peer = room.peer(eid)
                    if peer is not None:
                        discard = y - peer.out_base
                        if discard > 0:
                            del peer.out_buf[:min(discard, len(peer.out_buf))]
                            peer.out_base = y
                    _stat(room)
        finally:
            ka.cancel()
            ml.cancel()
            if slot.ws is ws:
                slot.ws = None   # el slot PERSISTE (buffers intactos) para reconexión
            print(f"[relay] WS cerrado room={room_name} eid={eid.hex()[:8]} (slot persiste)", flush=True)
        return ws

    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("GDLP_PORT", "8000"))
    bind = os.environ.get("GDLP_BIND", "::")
    site = web.TCPSite(runner, bind, port)
    await site.start()
    STATE["ready"] = True
    print(f"[relay] listo en {bind}:{port} (reconexión+reanudación)", flush=True)
    while True:
        await asyncio.sleep(30)


def _run():
    try:
        _install_deps()
        asyncio.run(_main())
    except BaseException as e:
        import traceback
        traceback.print_exc()
        STATE["error"] = f"{type(e).__name__}: {e}"
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _DS(ThreadingHTTPServer):
            address_family = socket.AF_INET6

            def server_bind(self):
                try:
                    self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                except OSError:
                    pass
                super().server_bind()

        class Hh(BaseHTTPRequestHandler):
            def do_GET(self):
                b = json.dumps(STATE).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        _DS(("::", int(os.environ.get("GDLP_PORT", "8000"))), Hh).serve_forever()


if __name__ == "__main__":
    _run()

#!/usr/bin/env python
"""Sonda de latencia: mide el RTT de ESTE nodo al relé y publica el resultado.

Sirve para responder la pregunta que decide la topología del fleet: ¿cuánto se
gana poniendo los nodos CERCA unos de otros? Un tramo nodo<->relé es exactamente
uno de los cuatro tramos que paga cada token en el pipeline actual, así que este
número se traduce directo a tok/s.

Corre en un contenedor solo-CPU (python:3.11-slim), habla el mismo protocolo que
`bridge.py` (HELLO + PING/PONG contra `relay.py`) y expone `/stats` por HTTP.

Env: GDLP_RELAY_URL, GDLP_PROBE_ID, GDLP_PROBE_SECONDS, GDLP_PORT.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import struct
import subprocess
import sys
import time
import urllib.request

HELLO = 0x10
PING, PONG = 0x30, 0x31

RELAY_URL = os.environ.get("GDLP_RELAY_URL", "")
PROBE_ID = os.environ.get("GDLP_PROBE_ID", "probe")
SECONDS = float(os.environ.get("GDLP_PROBE_SECONDS", "180"))
INTERVAL = float(os.environ.get("GDLP_PROBE_INTERVAL", "0.5"))
PORT = int(os.environ.get("GDLP_PORT", "8000"))
# Plazo de muerte ABSOLUTO. Sin esto, si el orquestador muere de forma dura (cierras la
# terminal, se cae la red, apagón) su `finally` nunca corre y el contenedor factura para
# siempre. Con restart_policy=never, salir del proceso para la instancia.
TTL = float(os.environ.get("GDLP_PROBE_TTL", str(SECONDS + 600)))

#: Muestras por par antes de dar la arista por medida. 30 basta para una mediana
#: estable sin alargar el arranque: el orquestador espera a que la matriz esté
#: completa antes de planificar.
PEER_SAMPLES = int(os.environ.get("GDLP_PEER_SAMPLES", "30"))
PEER_INTERVAL = float(os.environ.get("GDLP_PEER_INTERVAL", "0.3"))

STATE: dict = {"probe": PROBE_ID, "relay_url": RELAY_URL, "samples": [],
               "geo": None, "done": False, "error": None,
               "connects": 0, "started": time.time(),
               "peer_url": None, "peer_samples": [], "peer_error": None,
               # Matriz all-pairs: {peer_id: {"url":..., "samples":[...], "error":...}}
               "peers": {}}


def _ensure_aiohttp():
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "aiohttp"], check=True)


def _geolocate():
    """Dónde ha caído este nodo. Sólo diagnóstico: si falla, seguimos igual —
    el dato que importa es el RTT, no la etiqueta geográfica."""
    for url in ("https://ipinfo.io/json", "https://ifconfig.co/json"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gdlp-probe"})
            with urllib.request.urlopen(req, timeout=8) as r:
                d = json.loads(r.read().decode())
            return {"country": d.get("country") or d.get("country_iso"),
                    "region": d.get("region"), "city": d.get("city"),
                    "org": (d.get("org") or d.get("asn_org")), "source": url}
        except Exception:
            continue
    return None


def _peer_ping_once(url: str) -> float | None:
    """Un ida-y-vuelta HTTP contra el gateway público de OTRO nodo.

    Responde la pregunta del ×1,8: si un nodo puede alcanzar a otro por su URL
    pública en un tiempo parecido al del relé, la cadena pasa de 4 tramos a 2
    SIN necesitar perforación de NAT. Si el borde de Salad añade su propio
    sobrecoste, no compensa y hay que ir a la conexión directa de verdad."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gdlp-probe"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        STATE["peer_error"] = f"{type(exc).__name__}: {exc}"
        return None


async def _peer_loop():
    """Mide contra CADA par publicado, en rueda.

    Un solo par respondía «¿se puede quitar el relé?». La matriz all-pairs
    responde la que de verdad decide la colocación: **qué nodo va junto a cuál**.
    El planificador ya consume aristas dirigidas y penaliza lo no medido con un
    centinela caro (`src/core/rtt.ts`), así que una arista que esta sonda no
    cubra queda excluida de facto de cualquier plan.

    Se recorren los pares por turnos en vez de en paralelo a propósito: N sondas
    simultáneas desde el mismo nodo compiten por su propio enlace de salida y se
    miden a sí mismas. Por turnos es más lento y es lo correcto.
    """
    loop = asyncio.get_running_loop()
    while True:
        peers = STATE["peers"]
        if not peers:
            await asyncio.sleep(1.0)
            continue
        for peer_id in list(peers):
            entry = peers.get(peer_id)
            if not entry or not entry.get("url"):
                continue
            if len(entry["samples"]) >= PEER_SAMPLES:
                continue
            ms = await loop.run_in_executor(None, _peer_ping_once, entry["url"])
            if ms is not None:
                entry["samples"].append(ms)
            else:
                entry["error"] = STATE.get("peer_error")
            await asyncio.sleep(PEER_INTERVAL)
        await asyncio.sleep(0.5)


def _stats(samples: list[float]):
    s = sorted(samples)
    if not s:
        return {}
    return {"n": len(s), "min_ms": round(s[0], 2), "p50_ms": round(statistics.median(s), 2),
            "p90_ms": round(s[int(len(s) * 0.9)], 2) if len(s) > 4 else None,
            "max_ms": round(s[-1], 2), "mean_ms": round(statistics.fmean(s), 2)}


def _summary():
    s = sorted(STATE["samples"])
    if not s:
        return {}
    return {"n": len(s), "min_ms": round(s[0], 2), "p50_ms": round(statistics.median(s), 2),
            "p90_ms": round(s[int(len(s) * 0.9)], 2) if len(s) > 4 else None,
            "max_ms": round(s[-1], 2), "mean_ms": round(statistics.fmean(s), 2)}


async def _probe_loop():
    import aiohttp

    deadline = time.time() + SECONDS
    eid = os.urandom(16)
    room = f"probe-{PROBE_ID}"
    hello = (bytes([HELLO]) + struct.pack(">H", len(room)) + room.encode()
             + eid + struct.pack(">Q", 1) + struct.pack(">Q", 0))

    while time.time() < deadline:
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=20)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.ws_connect(RELAY_URL, max_msg_size=1 << 20, heartbeat=20) as ws:
                    STATE["connects"] += 1
                    await ws.send_bytes(hello)
                    pending: dict[int, float] = {}
                    seq = 0

                    async def sender():
                        nonlocal seq
                        while time.time() < deadline and not ws.closed:
                            pending[seq] = time.perf_counter()
                            await ws.send_bytes(bytes([PING]) + struct.pack(">Q", seq)
                                                + struct.pack(">d", time.time()))
                            seq += 1
                            await asyncio.sleep(INTERVAL)

                    task = asyncio.create_task(sender())
                    try:
                        async for msg in ws:
                            if msg.type is not aiohttp.WSMsgType.BINARY or not msg.data:
                                continue
                            if msg.data[0] == PONG and len(msg.data) >= 9:
                                sq = struct.unpack(">Q", msg.data[1:9])[0]
                                t0 = pending.pop(sq, None)
                                if t0 is not None:
                                    STATE["samples"].append((time.perf_counter() - t0) * 1000.0)
                            if time.time() >= deadline:
                                break
                    finally:
                        task.cancel()
        except Exception as exc:  # el cap duro de WS de Salad corta ~cada 194 s
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[probe] reconecto tras {STATE['error']}", flush=True)
            await asyncio.sleep(2)

    STATE["done"] = True
    STATE["summary"] = _summary()
    print(f"[probe] {PROBE_ID} FIN {json.dumps(STATE['summary'])}", flush=True)


async def _main():
    from aiohttp import web

    STATE["geo"] = await asyncio.get_running_loop().run_in_executor(None, _geolocate)
    print(f"[probe] {PROBE_ID} geo={STATE['geo']}", flush=True)

    routes = web.RouteTableDef()

    @routes.get("/healthz")
    async def healthz(req):
        return web.Response(text="ok")

    @routes.get("/stats")
    async def stats(req):
        skip = ("samples", "peer_samples", "peers")
        # `complete` es lo que el orquestador espera antes de planificar: una
        # matriz a medias produciría aristas con centinela y el planificador
        # descartaría nodos que en realidad son buenos.
        peer_stats = {
            peer_id: {
                "url": entry.get("url"),
                "error": entry.get("error"),
                "complete": len(entry["samples"]) >= PEER_SAMPLES,
                **_stats(entry["samples"]),
            }
            for peer_id, entry in STATE["peers"].items()
        }
        return web.json_response({**{k: v for k, v in STATE.items() if k not in skip},
                                  "summary": _summary(),
                                  "peer_summary": _stats(STATE["peer_samples"]),
                                  "peers": peer_stats,
                                  "peers_complete": bool(peer_stats) and all(
                                      entry["complete"] for entry in peer_stats.values()
                                  ),
                                  "elapsed": round(time.time() - STATE["started"], 1)})

    @routes.post("/peer")
    async def peer(req):
        """Publica uno o varios pares.

        Acepta la forma antigua `{"url": ...}` y la nueva
        `{"peers": {"<id>": "<url>", ...}}`. La compatibilidad no es cortesía:
        un orquestador viejo apuntando a una sonda nueva debe seguir midiendo
        en vez de fallar en silencio y dejar la arista sin medir.
        """
        body = await req.json()
        single = body.get("url")
        if single:
            STATE["peer_url"] = single
            STATE["peers"].setdefault(
                body.get("id") or "peer", {"url": single, "samples": [], "error": None}
            )["url"] = single
        for peer_id, url in (body.get("peers") or {}).items():
            entry = STATE["peers"].setdefault(
                peer_id, {"url": url, "samples": [], "error": None}
            )
            if entry["url"] != url:
                # Un par reubicado invalida sus muestras: son de otra ruta.
                entry.update({"url": url, "samples": [], "error": None})
        print(f"[probe] pares asignados: {sorted(STATE['peers'])}", flush=True)
        return web.json_response({"ok": True, "peers": sorted(STATE["peers"])})

    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    # el gateway de Salad entra por IPv6: hay que escuchar en dual-stack
    await web.TCPSite(runner, os.environ.get("GDLP_BIND", "::"), PORT).start()
    print(f"[probe] {PROBE_ID} HTTP en :{PORT}, midiendo {SECONDS:.0f}s contra {RELAY_URL}", flush=True)

    asyncio.create_task(_peer_loop())
    await _probe_loop()
    # sigue sirviendo /stats hasta que el orquestador recoja y borre... pero no para siempre
    while time.time() - STATE["started"] < TTL:
        await asyncio.sleep(30)
    print(f"[probe] {PROBE_ID} TTL de {TTL:.0f}s agotado; salgo para dejar de facturar", flush=True)


if __name__ == "__main__":
    if not RELAY_URL:
        sys.exit("falta GDLP_RELAY_URL")
    _ensure_aiohttp()
    asyncio.run(_main())

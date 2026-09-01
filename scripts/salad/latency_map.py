#!/usr/bin/env python
"""Mapa de latencia del fleet: ¿cuánto se gana eligiendo nodos CERCANOS?

Mide el RTT real nodo<->relé desde varios países a la vez. Ese tramo es
exactamente uno de los cuatro que paga cada token en el pipeline, así que el
resultado se traduce directo a tok/s con el modelo ya calibrado:

    tok/s = 1 / ((tramos * RTT + cómputo) / 1000)

Todo en contenedores solo-CPU (céntimos). Salad permite fijar el país con
`country_codes`, así que se puede comparar "mismo país que el relé" contra
"donde caiga" SIN tocar el pipeline ni encender una sola GPU.

  SALAD_API_KEY=... python latency_map.py --org X --project Y \
      --relay-country us --probes "us:3,gb:1,any:2" --seconds 180
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_orchestrate import (  # noqa: E402
    DEPLOY, call, container_exists, gw, standalone_command, wait_running,
)

CPU_RESOURCES = {"cpu": 2, "memory": 2048, "storage_amount": 8 * 1073741824}
IMAGE = "docker.io/library/python:3.11-slim"


def parse_probes(spec: str) -> list[str]:
    """'us:3,gb:1,any:2' -> ['us','us','us','gb','any','any']"""
    out: list[str] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        country, _, count = chunk.partition(":")
        out.extend([country.strip().lower()] * int(count or 1))
    return out


def create(org, project, name, env, command, auth, country, retries=3):
    body = {"name": name,
            "container": {"image": IMAGE, "resources": CPU_RESOURCES,
                          "environment_variables": env, "command": command},
            "autostart_policy": True, "restart_policy": "never", "replicas": 1,
            "networking": {"protocol": "http", "port": 8000, "auth": auth}}
    if country and country != "any":
        body["country_codes"] = [country]
    last = None
    for attempt in range(retries):
        st, cg = call("POST", f"/organizations/{org}/projects/{project}/containers", body)
        if st in (200, 201):
            return
        last = (st, cg)
        if container_exists(org, project, name):
            print(f"  aviso: create {name} devolvió {st} pero existe; sigo")
            return
        if st in (500, 502, 503, 504, 520, 521, 522, 524) and attempt + 1 < retries:
            time.sleep(3 * (attempt + 1))
            continue
        break
    raise RuntimeError(f"create {name} falló: {last[0]} {last[1]}")


def toks(rtt_ms: float, hops: int, compute_ms: float) -> float:
    return 1000.0 / (hops * rtt_ms + compute_ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--tag", default="lat")
    ap.add_argument("--relay-country", default="us")
    ap.add_argument("--probes", default="us:3,gb:1,any:2")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--boot-timeout-min", type=float, default=12)
    ap.add_argument("--compute-ms", type=float, default=60.0,
                    help="cómputo por token del presupuesto medido")
    ap.add_argument("--peer-test", action="store_true",
                    help="cada sonda mide también el RTT al gateway público de otra "
                         "sonda (¿se puede pasar de 4 tramos a 2 HOY?). Obliga a "
                         "auth:false en las sondas: sin la clave en el contenedor no "
                         "podrían llamarse entre ellas, y la clave NO viaja a Salad.")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    plan = parse_probes(args.probes)
    print(f"plan: relé en {args.relay_country!r}, sondas {plan}")

    created: list[tuple[str, str]] = []   # (name, base)
    t0 = time.time()
    result = {"relay_country": args.relay_country, "plan": plan,
              "seconds": args.seconds, "probes": []}
    try:
        relay_name = f"gdlp-relay-{args.tag}{stamp}"
        renv = {"GDLP_PORT": "8000", "PYTHONUNBUFFERED": "1"}
        rcmd = standalone_command(os.path.join(DEPLOY, "relay.py"), renv)
        rbase = f"/organizations/{args.org}/projects/{args.project}/containers/{relay_name}"
        created.append((relay_name, rbase))
        create(args.org, args.project, relay_name, renv, rcmd, auth=False,
               country=args.relay_country)
        relay_dns = wait_running(rbase, args.boot_timeout_min, t0)
        relay_url = f"wss://{relay_dns}/relay"
        print(f"relé listo en {relay_dns}")

        probes = []
        for i, country in enumerate(plan):
            pid = f"{country}{i}"
            name = f"gdlp-p{i}-{args.tag}{stamp}"
            penv = {"GDLP_PORT": "8000", "PYTHONUNBUFFERED": "1",
                    "GDLP_RELAY_URL": relay_url, "GDLP_PROBE_ID": pid,
                    "GDLP_PROBE_SECONDS": str(args.seconds)}
            pcmd = standalone_command(os.path.join(DEPLOY, "rtt_probe.py"), penv)
            base = f"/organizations/{args.org}/projects/{args.project}/containers/{name}"
            created.append((name, base))
            create(args.org, args.project, name, penv, pcmd,
                   auth=not args.peer_test, country=country)
            probes.append({"id": pid, "requested_country": country, "name": name, "base": base})
            print(f"sonda {pid} creada ({country})")

        for p in probes:
            try:
                p["dns"] = wait_running(p["base"], args.boot_timeout_min, t0)
            except Exception as exc:
                p["dns"], p["error"] = None, f"no arrancó: {exc}"
                print(f"  sonda {p['id']} NO arrancó: {exc}")

        # Emparejamiento en anillo: cada sonda hace ping HTTP al gateway público de
        # la siguiente. Mide si "nodo -> nodo" (2 tramos) es viable en Salad HOY,
        # sin perforación de NAT — la palanca del x1,8.
        live = [p for p in probes if p.get("dns")]
        for i, p in enumerate(live if args.peer_test else []):
            peer = live[(i + 1) % len(live)]
            if peer is p:
                continue
            p["peer_id"] = peer["id"]
            try:
                st, raw = gw(p["dns"], "/peer", method="POST",
                             body={"url": f"https://{peer['dns']}/healthz"},
                             auth=not args.peer_test, retries=3, timeout=20)
                print(f"  {p['id']} -> par {peer['id']} ({st})")
            except Exception as exc:
                print(f"  {p['id']}: no pude asignar par ({type(exc).__name__})")

        deadline = time.time() + args.seconds + 300
        pending = [p for p in probes if p.get("dns")]
        while pending and time.time() < deadline:
            time.sleep(20)
            still = []
            for p in pending:
                try:
                    st, raw = gw(p["dns"], "/stats", auth=not args.peer_test, retries=2, timeout=20)
                    d = json.loads(raw) if st == 200 else None
                except Exception as exc:
                    print(f"  {p['id']}: sin respuesta ({type(exc).__name__})")
                    still.append(p)
                    continue
                if not isinstance(d, dict):
                    print(f"  {p['id']}: HTTP {st} {str(raw)[:80]}")
                    still.append(p)
                    continue
                p["stats"] = d
                s = d.get("summary") or {}
                print(f"  {p['id']:>6} t+{int(time.time()-t0)}s n={s.get('n', 0)} "
                      f"p50={s.get('p50_ms')}ms min={s.get('min_ms')}ms geo={(d.get('geo') or {}).get('country')}")
                if not d.get("done"):
                    still.append(p)
            pending = still

        for p in probes:
            s = (p.get("stats") or {}).get("summary") or {}
            geo = (p.get("stats") or {}).get("geo") or {}
            result["probes"].append({
                "id": p["id"], "requested_country": p["requested_country"],
                "actual_country": geo.get("country"), "city": geo.get("city"),
                "org": geo.get("org"), "rtt": s, "error": p.get("error"),
                "peer_id": p.get("peer_id"),
                "peer_rtt": (p.get("stats") or {}).get("peer_summary") or {}})

        print("\n" + "=" * 96)
        print(f"{'sonda':>7} {'pedido':>7} {'real':>5} {'ciudad':<15} "
              f"{'RTT relé':>9} {'min':>7} {'RTT par':>9} {'n':>5}   tok/s (2 tramos vía relé)")
        print("-" * 96)
        for r in result["probes"]:
            rtt = r["rtt"].get("p50_ms")
            proj = f"{toks(rtt, 2, args.compute_ms):.2f}" if rtt else "—"
            print(f"{r['id']:>7} {r['requested_country']:>7} {str(r['actual_country'] or '?'):>5} "
                  f"{str(r['city'] or '?'):<15} {str(rtt or '—'):>9} "
                  f"{str(r['rtt'].get('min_ms') or '—'):>7} "
                  f"{str(r['peer_rtt'].get('p50_ms') or '—'):>9} "
                  f"{str(r['rtt'].get('n') or 0):>5}   {proj}")

        same = [r["rtt"]["p50_ms"] for r in result["probes"]
                if r["requested_country"] == args.relay_country and r["rtt"].get("p50_ms")]
        other = [r["rtt"]["p50_ms"] for r in result["probes"]
                 if r["requested_country"] != args.relay_country and r["rtt"].get("p50_ms")]
        if same and other:
            ms, mo = statistics.median(same), statistics.median(other)
            result["verdict"] = {"same_country_p50": ms, "other_p50": mo,
                                 "ratio": round(mo / ms, 2) if ms else None,
                                 "toks_same": round(toks(ms, 2, args.compute_ms), 2),
                                 "toks_other": round(toks(mo, 2, args.compute_ms), 2)}
            print("-" * 84)
            print(f"mismo país que el relé: p50 {ms:.1f} ms -> {toks(ms, 2, args.compute_ms):.2f} tok/s")
            print(f"distinto país:          p50 {mo:.1f} ms -> {toks(mo, 2, args.compute_ms):.2f} tok/s")
        print("=" * 84)

    finally:
        # BORRAR PRIMERO, guardar después: cada contenedor vivo factura. Si `json.dump`
        # peta (fichero abierto en un editor de Windows, disco lleno) y va delante, la
        # excepción sale del finally y no se borra NADA.
        print("teardown…")
        pendientes = []
        for name, base in created:
            borrado, st = False, None
            for intento in range(4):
                try:
                    st, _ = call("DELETE", base)
                    borrado = st < 300 or st == 404   # 202 aceptado; 404 ya no estaba
                except Exception as exc:              # el borde de Salad/Cloudflare falla
                    st = f"{type(exc).__name__}: {exc}"
                if borrado:
                    break
                time.sleep(2 * (intento + 1))
            print(f"  borrado {name}: {st}" if borrado else f"  ¡FALLO al borrar {name}: {st}!")
            if not borrado:
                pendientes.append(name)
        if pendientes:
            print("\n" + "!" * 70)
            print("ATENCIÓN: estos contenedores SIGUEN FACTURANDO, bórralos a mano:")
            for name in pendientes:
                print(f"  {name}")
            print("!" * 70)
        try:
            out = os.path.join(args.out, "result.json")
            json.dump(result, open(out, "w"), indent=1)
            print(f"\nresultado -> {out}")
        except Exception as exc:
            print(f"\nno pude guardar result.json ({type(exc).__name__}: {exc}); lo vuelco aquí:")
            print(json.dumps(result, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()

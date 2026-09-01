#!/usr/bin/env python
"""Orquestador reutilizable de SaladCloud para el programa de pruebas GDLP-2.

Ciclo de vida SEGURO: crear -> vigilar -> recoger artefactos -> TEARDOWN SIEMPRE
-> registrar coste. El teardown está en `finally`: pase lo que pase (éxito,
error, Ctrl-C, timeout) el contenedor se borra y deja de facturar.

Uso:
  set SALAD_API_KEY=...   (o exporta; NUNCA se committea)
  python scripts/salad/orchestrate.py \
      --org nodecodex --project mycellios \
      --image <REGISTRY>/mycellios-salad:<TAG> \
      --gpu "3060 (12" --mode probe --model Qwen/Qwen3-0.6B \
      --out docs/benchmarks/salad-s0-<fecha>/

Lecciones horneadas (fallo 24-07):
  * User-Agent obligatorio (sin él, Cloudflare devuelve 403 code 1010).
  * price() usa la prioridad configurable (por defecto 'high' = más disponible).
  * imagen PRE-HORNEADA (nada de pip en runtime); este orquestador NO instala nada.
  * gateway auth con el mismo Salad-Api-Key; puede tardar minutos en 503 (arranque).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = "https://api.salad.com/api/public"
UA = "gdlp-orchestrator/1.0"


def _key() -> str:
    k = os.environ.get("SALAD_API_KEY")
    if not k:
        sys.exit("falta SALAD_API_KEY en el entorno (no se committea)")
    return k


def call(method: str, path: str, body=None, timeout=45):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Salad-Api-Key": _key(), "Content-Type": "application/json", "User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:800]}


def gateway_get(dns: str, path: str, timeout=20):
    url = path if dns.startswith("http") else f"https://{dns}{path}"
    req = urllib.request.Request(url, headers={"Salad-Api-Key": _key(), "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def pick_gpu(org: str, gpu_filter: str, priority: str):
    st, gpus = call("GET", f"/organizations/{org}/gpu-classes")
    if st != 200:
        sys.exit(f"gpu-classes fallo: {st} {gpus}")
    items = gpus.get("items", [])
    match = [g for g in items if gpu_filter.lower() in g.get("name", "").lower()]
    if not match:
        sys.exit(f"sin GPU que case '{gpu_filter}'. Hay: {[g['name'] for g in items]}")

    def price(g):
        for p in g.get("prices", []):
            if p.get("priority") == priority and p.get("price"):
                return float(p["price"])
        return 9e9

    match.sort(key=price)
    return match[0], price(match[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
                    help="imagen pre-horneada mycellios-salad; por defecto pytorch estándar para la vía --standalone-script")
    ap.add_argument("--standalone-script", default=None,
                    help="vía SIN-imagen: inyecta este .py autocontenido por env y lo ejecuta con python -c "
                         "(pip visible en /log). Úsalo mientras la imagen pre-horneada no esté en un registro.")
    ap.add_argument("--payload-tgz", default=None,
                    help="inyecta un tarball (p.ej. el runtime mycellios) por env PAYLOAD_* para que el "
                         "script autocontenido lo extraiga. Viable hasta ~200 KB (verificado con Salad).")
    ap.add_argument("--gpu", default="3060 (12", help="subcadena del nombre de la clase de GPU")
    ap.add_argument("--priority", default="high", choices=["high", "medium", "low", "batch"])
    ap.add_argument("--mode", default="probe")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--memory", type=int, default=12288)
    ap.add_argument("--storage-gb", type=int, default=30)
    ap.add_argument("--gdlp-args", default="[]", help="JSON list para GDLP_ARGS (modo server)")
    ap.add_argument("--env", action="append", default=[], help="EXTRA=valor (repetible)")
    ap.add_argument("--boot-timeout-min", type=int, default=25)
    ap.add_argument("--run-timeout-min", type=int, default=45)
    ap.add_argument("--out", required=True, help="carpeta de artefactos")
    args = ap.parse_args()

    gpu, price = pick_gpu(args.org, args.gpu, args.priority)
    name = "gdlp-" + args.mode + "-" + datetime.datetime.now().strftime("%m%d-%H%M%S")
    os.makedirs(args.out, exist_ok=True)
    print(f"GPU {gpu['name']} (~{price}$/h @ {args.priority}); grupo {name}")

    env = {"GDLP_MODE": args.mode, "GDLP_MODEL": args.model, "GDLP_ARGS": args.gdlp_args, "PYTHONUNBUFFERED": "1"}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    if args.payload_tgz:
        import base64

        pb64 = base64.b64encode(open(args.payload_tgz, "rb").read()).decode()
        pchunks = [pb64[i : i + 990] for i in range(0, len(pb64), 990)]
        for i, c in enumerate(pchunks):
            env[f"PAYLOAD_{i}"] = c
        env["PAYLOAD_PARTS"] = str(len(pchunks))
        print(f"payload {args.payload_tgz}: {len(pb64)//1024} KB base64 en {len(pchunks)} env vars")

    command = None
    if args.standalone_script:
        import base64

        b64 = base64.b64encode(open(args.standalone_script, "rb").read()).decode()
        chunks = [b64[i : i + 900] for i in range(0, len(b64), 900)]
        for i, c in enumerate(chunks):
            env[f"GDLP_SCRIPT_{i}"] = c
        env["GDLP_SCRIPT_PARTS"] = str(len(chunks))
        loader = (
            "import base64,os;"
            "n=int(os.environ['GDLP_SCRIPT_PARTS']);"
            "src=base64.b64decode(''.join(os.environ[f'GDLP_SCRIPT_{i}'] for i in range(n)));"
            "exec(compile(src,'<probe>','exec'))"
        )
        command = ["python", "-c", loader]
        print(f"vía SIN-imagen: {args.standalone_script} inyectado en {len(chunks)} trozos sobre {args.image}")

    container = {
        "image": args.image,
        "resources": {
            "cpu": args.cpu,
            "memory": args.memory,
            "gpu_classes": [gpu["id"]],
            "storage_amount": args.storage_gb * 1073741824,
        },
        "environment_variables": env,
    }
    if command is not None:
        container["command"] = command

    body = {
        "name": name,
        "container": container,
        "autostart_policy": True,
        "restart_policy": "never",
        "replicas": 1,
        "networking": {"protocol": "http", "port": int(env.get("GDLP_PORT", "8000")), "auth": True},
    }
    st, cg = call("POST", f"/organizations/{args.org}/projects/{args.project}/containers", body)
    if st not in (200, 201):
        sys.exit(f"create fallo: {st} {cg}")
    t0 = time.time()
    json.dump(
        {"request": body, "gpu": gpu["name"], "price_per_hour": price, "created_ts": t0},
        open(os.path.join(args.out, "manifest.json"), "w"),
        indent=1,
    )
    base = f"/organizations/{args.org}/projects/{args.project}/containers/{name}"

    try:
        dns = None
        boot_deadline = t0 + args.boot_timeout_min * 60
        while time.time() < boot_deadline:
            st, cur = call("GET", base)
            status = cur.get("current_state", {}).get("status")
            dns = cur.get("networking", {}).get("dns")
            running = cur.get("current_state", {}).get("instance_status_counts", {}).get("running_count", 0)
            print(f"  estado={status} running={running} dns={dns} t+{int(time.time()-t0)}s", flush=True)
            if status == "running" and running >= 1 and dns:
                break
            time.sleep(15)
        else:
            raise TimeoutError("no llegó a running en el boot-timeout")

        run_deadline = time.time() + args.run_timeout_min * 60
        last = None
        while time.time() < run_deadline:
            try:
                res = gateway_get(dns, "/results.json")
                last = res
                json.dump(res, open(os.path.join(args.out, "results-latest.json"), "w"), indent=1)
                print(f"  probe status={res.get('status')} t+{int(time.time()-t0)}s", flush=True)
                if res.get("status") in ("done", "error"):
                    json.dump(res, open(os.path.join(args.out, "results-final.json"), "w"), indent=1)
                    # guardar tambien el log de texto del contenedor
                    try:
                        txt = urllib.request.urlopen(
                            urllib.request.Request(f"https://{dns}/log", headers={"Salad-Api-Key": _key(), "User-Agent": UA}),
                            timeout=20,
                        ).read().decode()
                        open(os.path.join(args.out, "container.log"), "w", encoding="utf-8").write(txt)
                    except Exception:
                        pass
                    print("RESULTADO FINAL recogido:", res.get("status"))
                    break
            except Exception as e:
                print(f"  gateway {type(e).__name__} (arrancando) t+{int(time.time()-t0)}s", flush=True)
            time.sleep(15)
    finally:
        # TEARDOWN A PRUEBA DE FALLOS. `call` sólo captura HTTPError, así que un
        # corte de red o un timeout aquí dejaba el contenedor VIVO y facturando
        # sin que nadie se enterase. Mismo fallo que ya se corrigió en
        # `latency_map.py`. Se reintenta, y lo que sobreviva se anuncia con el
        # comando exacto para rematarlo a mano.
        st = None
        for attempt in range(4):
            try:
                st, _ = call("DELETE", base, timeout=30)
                if st in (200, 202, 204, 404):  # 404 = ya no existe
                    break
                print(f"  TEARDOWN http={st} (intento {attempt + 1}/4)")
            except Exception as error:  # noqa: BLE001 - nada puede saltarse el borrado
                st = f"EXC:{type(error).__name__}"
                print(f"  TEARDOWN excepción {error} (intento {attempt + 1}/4)")
            time.sleep(2 * (attempt + 1))
        alive = None
        if st not in (200, 202, 204, 404):
            print()
            print(f"!!! ATENCION: {name} PUEDE SEGUIR FACTURANDO. Bórralo a mano:")
            print(f"  curl -X DELETE -H 'Salad-Api-Key: $SALAD_API_KEY' "
                  f"-H 'User-Agent: {UA}' '{API}{base}'")
        else:
            # Confirmación positiva: que el DELETE devuelva 2xx no prueba que muriera.
            try:
                lst, listing = call("GET", f"/organizations/{args.org}/projects/"
                                           f"{args.project}/containers", timeout=30)
                if lst == 200:
                    alive = [c["name"] for c in listing.get("items", [])]
                    print(f"  verificado: {len(alive)} contenedores vivos en el proyecto")
            except Exception as error:  # noqa: BLE001
                print(f"  (no se pudo verificar el listado: {error})")
        elapsed_h = (time.time() - t0) / 3600
        cost = elapsed_h * price
        print(f"TEARDOWN http={st}; sesión {elapsed_h:.3f} h; coste GPU ~{cost:.4f} $")
        json.dump(
            {"elapsed_h": elapsed_h, "est_cost_usd": round(cost, 4), "gpu": gpu["name"],
             "teardown_http": st, "containers_alive_after_teardown": alive},
            open(os.path.join(args.out, "session-cost.json"), "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()

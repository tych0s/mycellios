#!/usr/bin/env python
"""Sesión de agente CALIENTE en Salad: crea (con reintento anti-nodo-malo),
espera a /ready, manda N trabajos por HTTP, y hace teardown al final.

Invierte el peaje de arranque UNA vez y luego cada experimento es casi
instantáneo. Uso:

  export SALAD_API_KEY=...
  python scripts/salad/agent_session.py --org nodecodex --project mycellios \
     --gpu "3060 (12" --payload-tgz <runtime.tgz> --agent-script deploy/salad/bench_agent.py \
     --model Qwen/Qwen3-0.6B --out docs/benchmarks/salad-s1-<fecha>/ \
     --job "s1?lambda=3.0&dur=60&window=50" --job "probe?model=Qwen/Qwen3-0.6B"

  --retries 3        reintenta en OTRO nodo si el actual no arranca en --boot-timeout-min
  --ready-timeout-min 12   espera de /ready tras running
  --keep-alive       NO hace teardown (para mandar más trabajos luego con --attach <dns>)
"""
from __future__ import annotations
import argparse, base64, datetime, json, os, sys, time, urllib.error, urllib.parse, urllib.request

# La consola de Windows (cp1252) no puede imprimir no-ASCII -> forzar UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = "https://api.salad.com/api/public"
UA = "gdlp-agent/1.0"


def key():
    k = os.environ.get("SALAD_API_KEY")
    if not k:
        sys.exit("falta SALAD_API_KEY")
    return k


def call(method, path, body=None, timeout=45):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={"Salad-Api-Key": key(), "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(); return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:600]}


def gw(dns, path, timeout=None):
    url = path if dns.startswith("http") else f"https://{dns}{path}"
    req = urllib.request.Request(url, headers={"Salad-Api-Key": key(), "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout or 30) as r:
        return r.status, json.loads(r.read().decode())


def pick_gpu(org, flt, prio):
    st, g = call("GET", f"/organizations/{org}/gpu-classes")
    if st != 200: sys.exit(f"gpu-classes {st} {g}")
    m = [x for x in g.get("items", []) if flt.lower() in x.get("name", "").lower()]
    if not m: sys.exit(f"sin GPU '{flt}': {[x['name'] for x in g['items']]}")
    def price(x):
        for p in x.get("prices", []):
            if p.get("priority") == prio and p.get("price"): return float(p["price"])
        return 9e9
    m.sort(key=price); return m[0], price(m[0])


def chunk_env(env, path, prefix):
    b = base64.b64encode(open(path, "rb").read()).decode()
    parts = [b[i:i + 990] for i in range(0, len(b), 990)]
    for i, c in enumerate(parts): env[f"{prefix}_{i}"] = c
    env[f"{prefix}_PARTS"] = str(len(parts))
    return len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True); ap.add_argument("--project", required=True)
    ap.add_argument("--gpu", default="3060 (12"); ap.add_argument("--priority", default="high")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    ap.add_argument("--agent-script", required=True); ap.add_argument("--payload-tgz", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cpu", type=int, default=4); ap.add_argument("--memory", type=int, default=16384)
    ap.add_argument("--storage-gb", type=int, default=30)
    ap.add_argument("--job", action="append", default=[], help="p.ej. 's1?lambda=3.0&dur=60' (repetible)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--boot-timeout-min", type=int, default=10)
    ap.add_argument("--ready-timeout-min", type=int, default=14)
    ap.add_argument("--job-timeout-min", type=int, default=20)
    ap.add_argument("--keep-alive", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gpu, price = pick_gpu(args.org, args.gpu, args.priority)
    print(f"GPU {gpu['name']} ~{price}$/h")

    base_env = {"GDLP_MODEL": args.model, "PYTHONUNBUFFERED": "1"}
    n_pay = chunk_env(base_env, args.payload_tgz, "PAYLOAD")
    b64 = base64.b64encode(open(args.agent_script, "rb").read()).decode()
    sc = [b64[i:i + 990] for i in range(0, len(b64), 990)]
    for i, c in enumerate(sc): base_env[f"AGENT_{i}"] = c
    base_env["AGENT_PARTS"] = str(len(sc))
    loader = ("import base64,os;n=int(os.environ['AGENT_PARTS']);"
              "exec(compile(base64.b64decode(''.join(os.environ[f'AGENT_{i}'] for i in range(n))),'<agent>','exec'))")
    print(f"payload {n_pay} vars + agente {len(sc)} vars")

    dns = None; name = None; t0 = time.time()
    try:
        for attempt in range(1, args.retries + 1):
            name = f"gdlp-agent-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
            body = {"name": name, "container": {"image": args.image,
                    "resources": {"cpu": args.cpu, "memory": args.memory, "gpu_classes": [gpu["id"]],
                                  "storage_amount": args.storage_gb * 1073741824},
                    "environment_variables": base_env, "command": ["python", "-c", loader]},
                    "autostart_policy": True, "restart_policy": "never", "replicas": 1,
                    "networking": {"protocol": "http", "port": 8000, "auth": True}}
            st, cg = call("POST", f"/organizations/{args.org}/projects/{args.project}/containers", body)
            if st not in (200, 201): sys.exit(f"create {st} {cg}")
            base = f"/organizations/{args.org}/projects/{args.project}/containers/{name}"
            print(f"intento {attempt}/{args.retries}: {name}")
            # esperar running
            deadline = time.time() + args.boot_timeout_min * 60; running = False
            while time.time() < deadline:
                st, cur = call("GET", base)
                status = cur.get("current_state", {}).get("status"); dns = cur.get("networking", {}).get("dns")
                nrun = cur.get("current_state", {}).get("instance_status_counts", {}).get("running_count", 0)
                print(f"  {status} run={nrun} t+{int(time.time()-t0)}s", flush=True)
                if status == "running" and nrun >= 1 and dns: running = True; break
                time.sleep(15)
            if not running:
                print("  nodo lento -> descarto y reintento en otro"); call("DELETE", base); continue
            # esperar /ready
            rdl = time.time() + args.ready_timeout_min * 60; ready = False
            while time.time() < rdl:
                try:
                    code, r = gw(dns, "/status", timeout=15)
                    print(f"  agente status={r.get('status')} phase={r.get('phase')} t+{int(time.time()-t0)}s", flush=True)
                    if r.get("ready"): ready = True; break
                    if r.get("status") == "error":
                        print("  agente ERROR:", r.get("error"))
                        open(os.path.join(args.out, "boot-error.txt"), "w").write(str(r.get("error")))
                        break
                except Exception as e:
                    print(f"  gateway {type(e).__name__} t+{int(time.time()-t0)}s", flush=True)
                time.sleep(15)
            if ready:
                print(f"AGENTE LISTO tras {int(time.time()-t0)}s en {name}")
                open(os.path.join(args.out, "agent.txt"), "w").write(f"{name}\n{dns}\n")
                break
            print("  agente no llegó a ready -> descarto y reintento"); call("DELETE", base); dns = None
        else:
            sys.exit("no se consiguió un nodo bueno tras los reintentos")

        # mandar trabajos: lanzar (async) -> sondear /job?id=... hasta done/error
        # (peticiones cortas; el gateway corta las largas a ~100 s).
        results = []
        for spec in args.job:
            print(f"--- JOB {spec} ---")
            if "?" in spec:
                jobname, qs = spec.split("?", 1)
                run_path = f"/run?job={jobname}&{qs}"
            else:
                run_path = f"/run?job={spec}"
            try:
                code, r = gw(dns, run_path, timeout=30)
                jid = r.get("job_id")
                if not jid:
                    raise RuntimeError(f"sin job_id: {r}")
                print(f"  lanzado {jid}; sondeando…")
                jdl = time.time() + args.job_timeout_min * 60
                jr = None
                while time.time() < jdl:
                    time.sleep(15)
                    try:
                        _, jr = gw(dns, f"/job?id={jid}", timeout=20)
                    except Exception as e:
                        print(f"    sonda {type(e).__name__}")
                        continue
                    if jr.get("status") in ("done", "error"):
                        break
                res = (jr or {}).get("result")
                results.append({"spec": spec, "job_id": jid, "status": (jr or {}).get("status"), "result": res, "error": (jr or {}).get("error")})
                print("  ->", json.dumps(res.get("summary") if res else (jr or {}).get("error") or res, ensure_ascii=False)[:400] if res or jr else "timeout")
            except Exception as e:
                results.append({"spec": spec, "error": f"{type(e).__name__}: {e}"})
                print("  JOB fallo:", e)
            json.dump(results, open(os.path.join(args.out, "jobs-results.json"), "w"), indent=1)
        # guardar log del agente
        try:
            import urllib.request as U
            txt = U.urlopen(U.Request(f"https://{dns}/log", headers={"Salad-Api-Key": key(), "User-Agent": UA}), timeout=20).read().decode()
            open(os.path.join(args.out, "agent.log"), "w", encoding="utf-8").write(txt)
        except Exception:
            pass
    finally:
        if name and not args.keep_alive:
            st, _ = call("DELETE", f"/organizations/{args.org}/projects/{args.project}/containers/{name}")
            h = (time.time() - t0) / 3600
            print(f"TEARDOWN http={st}; sesión {h:.3f} h; coste ~{h*price:.4f} $")
            json.dump({"elapsed_h": h, "est_cost_usd": round(h * price, 4), "gpu": gpu["name"]},
                      open(os.path.join(args.out, "session-cost.json"), "w"), indent=1)
        elif args.keep_alive:
            print(f"AGENTE SIGUE VIVO: {name} @ {dns} (usa --attach o borra manualmente para parar el coste)")


if __name__ == "__main__":
    main()

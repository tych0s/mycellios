#!/usr/bin/env python
"""Adopta contenedores YA CREADOS (huérfanos) y ejecuta el A/B mismo-nodo.

Uso cuando un orquestador muere y deja nodos vivos: en vez de tirarlos (ya
pagaron el pull de la imagen, 15-25 min), este script los descubre por --tag,
espera a que estén booted, barre los arms (/start+/stop con generación) y hace
TEARDOWN SIEMPRE en finally.

  python adopt_run.py --tag e1- --stages 3 --model Qwen/Qwen3-0.6B \
     --total-layers 28 --arms-json '[...]' --bench-lambdas 4,8 --out <dir>
"""
from __future__ import annotations
import argparse, base64, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_orchestrate as P  # reutiliza call/gw/node_plan/wait_* /patch_myargs

ORG, PROJECT = "nodecodex", "mycellios"


def discover(tag):
    st, d = P.call("GET", f"/organizations/{ORG}/projects/{PROJECT}/containers")
    if st != 200:
        sys.exit(f"list fallo: {st} {d}")
    items = [c for c in d.get("items", []) if tag in c.get("name", "")]
    relay = [c for c in items if "relay" in c["name"]]
    nodes = sorted([c for c in items if "relay" not in c["name"]], key=lambda c: c["name"])
    if not relay or not nodes:
        sys.exit(f"no encontré relé/nodos con tag '{tag}' (items={[c['name'] for c in items]})")
    return relay[0], nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stages", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--total-layers", type=int, required=True)
    ap.add_argument("--codec", default="fp16")
    ap.add_argument("--arms-json", required=True)
    ap.add_argument("--bench-lambdas", default="4,8")
    ap.add_argument("--bench-dur", type=int, default=25)
    ap.add_argument("--bench-tokens", type=int, default=16)
    ap.add_argument("--bench-timeout-min", type=int, default=15)
    ap.add_argument("--ready-timeout-min", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = args.stages
    os.makedirs(args.out, exist_ok=True)
    boundaries = P.balanced_boundaries(args.total_layers, n)
    plans = P.node_plan(n, boundaries, args.total_layers, args.model, args.codec, False)
    for _role, _eps, _my, _ev in plans:
        for _ep in _eps:
            _ep["eid"] = os.urandom(16).hex()

    relay_c, node_cs = discover(args.tag)
    if len(node_cs) != n:
        sys.exit(f"esperaba {n} nodos, encontré {len(node_cs)}: {[c['name'] for c in node_cs]}")
    relay_dns = relay_c.get("networking", {}).get("dns")
    relay_url = f"wss://{relay_dns}/relay"
    node_dns = [c.get("networking", {}).get("dns") for c in node_cs]
    bases = [f"/organizations/{ORG}/projects/{PROJECT}/containers/{c['name']}" for c in node_cs]
    relay_base = f"/organizations/{ORG}/projects/{PROJECT}/containers/{relay_c['name']}"
    print(f"ADOPTADO tag={args.tag}: relé={relay_dns}; nodos={[c['name'] for c in node_cs]}")

    t0 = time.time()
    result = {"adopted": True, "tag": args.tag, "stages": n, "boundaries": boundaries,
              "model": args.model, "relay_dns": relay_dns, "arms": {}}
    arm_list = json.loads(args.arms_json)
    root_dns = node_dns[0]

    def _bench():
        bench = []
        for lam in [x.strip() for x in args.bench_lambdas.split(",") if x.strip()]:
            sc, txt = P.gw(root_dns, f"/run?job=bench&mode=open&lam={lam}&dur={args.bench_dur}&tokens={args.bench_tokens}")
            jid = json.loads(txt).get("job_id")
            print(f"  lambda={lam} job={jid}", flush=True)
            j = {"status": "running"}
            deadline = time.time() + args.bench_timeout_min * 60
            while time.time() < deadline:
                sc, txt = P.gw(root_dns, f"/job?id={jid}")
                j = json.loads(txt)
                if j.get("status") in ("done", "error"):
                    break
                time.sleep(10)
            if j.get("status") == "done":
                res = j["result"]
                summ = res["summary"]
                print(f"  lambda={lam} -> {summ} B_eff={res.get('beff')} ragged={res.get('ragged_requested')}", flush=True)
                bench.append({"lambda_req": lam, "summary": summ, "beff": res.get("beff"),
                              "ragged_requested": res.get("ragged_requested")})
            else:
                print(f"  lambda={lam} FALLO: {j.get('error')}", flush=True)
                bench.append({"lambda_req": lam, "error": j.get("error")})
        return bench

    def _inference():
        sc, txt = P.gw(root_dns, "/v1/models")
        mid = json.loads(txt)["data"][0]["id"]
        ti = time.time()
        sc, txt = P.gw(root_dns, "/v1/chat/completions", method="POST", body={
            "model": mid, "messages": [{"role": "user", "content": "List three colors, comma separated."}],
            "max_tokens": 24, "temperature": 0, "stream": False, "seed": 7}, timeout=120)
        dt = time.time() - ti
        out = json.loads(txt); usage = out.get("usage", {}); toks = usage.get("completion_tokens")
        return {"text": out["choices"][0]["message"]["content"], "usage": usage, "seconds": round(dt, 2),
                "tok_s_single": round(toks / dt, 2) if toks and dt else None}

    try:
        # esperar running+booted de todos (el pull puede seguir en curso)
        for k, c in enumerate(node_cs):
            if not node_dns[k]:
                node_dns[k] = P.wait_running(bases[k], 30, t0)
        root_dns = node_dns[0]
        all_booted = True
        for k, dns in enumerate(node_dns):
            all_booted = P.wait_booted(dns, args.ready_timeout_min, t0, f"nodo{k}") and all_booted
        result["all_booted"] = all_booted

        if all_booted:
            print("TODOS BOOTED. Barriendo arms sobre los mismos nodos...", flush=True)
            for ai, arm in enumerate(arm_list):
                label = arm.get("label", f"arm{ai}")
                gen = ai + 1
                # /stop SIEMPRE (también antes del 1er arm): en un nodo adoptado puede
                # quedar un mycellios/puente de un run anterior -> "Address already in use".
                print(f"=== limpieza previa a '{label}' (/stop gen={gen}) ===", flush=True)
                for dns in node_dns:
                    try:
                        P.gw(dns, "/stop", method="POST", body={})
                    except Exception as e:
                        print(f"  /stop: {e}")
                time.sleep(6)
                print(f"=== ARM '{label}' gen={gen} cfg={ {k: v for k, v in arm.items() if k != 'label'} } ===", flush=True)
                ragged = bool(arm.get("ragged", False))
                for k in range(n):
                    role, eps, my, env_extra = plans[k]
                    my_p = P.patch_myargs(my, codec=arm.get("codec"), max_active=arm.get("max_active"),
                                          max_batch=arm.get("max_batch"))
                    eps_gen = [dict(ep, gen=gen) for ep in eps]
                    cfg = {"role": role, "relay_url": relay_url, "bridge_endpoints": eps_gen,
                           "my_args": my_p, "ragged": ragged}
                    sc, txt = P.gw(node_dns[k], "/start", method="POST", body=cfg)
                    print(f"  /start nodo{k}: {sc} {txt[:70]}", flush=True)
                ready = True
                for k, dns in enumerate(node_dns):
                    ready = P.wait_ready(dns, args.ready_timeout_min, t0, f"nodo{k}") and ready
                arm_res = {"ready": ready, "arm_cfg": {k: v for k, v in arm.items() if k != "label"}}
                if ready:
                    try:
                        arm_res["inference"] = _inference()
                        print(f"  SALIDA[{label}]:", repr(arm_res["inference"]["text"])[:70], flush=True)
                        arm_res["bench"] = _bench()
                    except Exception as e:
                        arm_res["error"] = f"{type(e).__name__}: {e}"
                        print(f"  arm '{label}' error midiendo (sigo): {e}", flush=True)
                else:
                    for k, dns in enumerate(node_dns):
                        try:
                            sc, txt = P.gw(dns, "/log", timeout=20)
                            open(os.path.join(args.out, f"node{k}-{label}.log"), "w", encoding="utf-8").write(txt)
                        except Exception:
                            pass
                result["arms"][label] = arm_res
                json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)
        json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)
    finally:
        print("=== TEARDOWN ===", flush=True)
        for b, name in [(relay_base, relay_c["name"])] + list(zip(bases, [c["name"] for c in node_cs])):
            st, _ = P.call("DELETE", b)
            print(f"  DELETE {name} http={st}", flush=True)
        elapsed_h = (time.time() - t0) / 3600
        result["elapsed_h"] = round(elapsed_h, 4)
        json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)
        print(f"sesión adoptada {elapsed_h:.3f} h", flush=True)


if __name__ == "__main__":
    main()

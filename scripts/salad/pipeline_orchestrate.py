#!/usr/bin/env python
"""Orquestador multi-nodo de SaladCloud: modelo repartido en N GPUs SEPARADAS.

Despliega:
  - 1 relé CPU-only barato (relay.py, auth:false) -> DNS público.
  - N nodos GPU (pipeline_node.py + runtime + bridge.py) que tunelizan sus
    enlaces inter-etapa contra el relé. Nodo 0 = root+stage0 (expone la API por
    su gateway); nodos 1..N-1 = stage_cli.

Cablea puertos/salas/flags por nodo (espejo de RuntimeStreamTunnel, validado en
local token-exacto). Espera ready, prueba la inferencia end-to-end, mide tok/s,
y hace TEARDOWN de TODO en finally con contabilidad de coste.

Uso:
  SALAD_API_KEY=... python pipeline_orchestrate.py --stages 2 \
     --model HuggingFaceTB/SmolLM2-135M-Instruct --total-layers 30 \
     --payload-tgz dr_full.tgz --out docs/benchmarks/salad-s2-<fecha>/
"""
from __future__ import annotations
import argparse, base64, datetime, json, os, secrets, sys, time
import urllib.error, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

API = "https://api.salad.com/api/public"
UA = "gdlp-pipeline-orch/1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DEPLOY = os.path.join(REPO_ROOT, "deploy", "salad")


def key():
    k = os.environ.get("SALAD_API_KEY")
    if not k:
        sys.exit("falta SALAD_API_KEY")
    return k


def call(method, path, body=None, timeout=45):
    req = urllib.request.Request(API + path,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 method=method,
                                 headers={"Salad-Api-Key": key(), "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:800]}


def gw(dns, path, method="GET", body=None, timeout=30, auth=True, retries=5):
    hdr = {"User-Agent": UA}
    if auth:
        hdr["Salad-Api-Key"] = key()
    data = None
    if body is not None:
        hdr["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f"https://{dns}{path}", data=data, method=method, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            # 503 = "aún no listo" (respuesta legítima del nodo), no una excepción de red
            return e.code, e.read().decode()
        except Exception as e:
            # transitorio (ConnectionReset/timeout/URLError por la WAN flaky): reintenta
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def pick_gpu(org, gpu_filter, priority, max_price):
    st, gpus = call("GET", f"/organizations/{org}/gpu-classes")
    if st != 200:
        sys.exit(f"gpu-classes fallo: {st} {gpus}")
    items = gpus.get("items", [])

    def price(g):
        for p in g.get("prices", []):
            if p.get("priority") == priority and p.get("price"):
                return float(p["price"])
        return 9e9
    cands = [g for g in items if (not gpu_filter or gpu_filter.lower() in g.get("name", "").lower())]
    cands = [g for g in cands if price(g) <= max_price]
    if not cands:
        sys.exit(f"sin GPU <= {max_price}$/h que case '{gpu_filter}'. Hay: {[(g['name'], price(g)) for g in items][:20]}")
    cands.sort(key=price)
    return cands[0], price(cands[0])


def pick_gpus_per_stage(org, gpu_filter, stages, priority, max_price):
    """Una clase de GPU POR ETAPA.

    `--gpu-filter` acepta una lista separada por comas: 'rtx 3060,1050' monta una
    cadena HETEROGENEA (etapa 0 en una 3060, etapa 1 en una 1050 Ti). Un solo
    filtro se replica en todas las etapas, que es el comportamiento anterior.

    Por que existe: hasta hoy el orquestador elegia UNA clase y la aplicaba a
    todas las etapas, asi que TODOS los experimentos de la campania corrieron
    sobre nodos homogeneos por construccion. Exp7 concluyo "el reparto desigual
    de capas no resuelve" midiendo una sola clase de GPU -- pero repartir capas
    en proporcion a la capacidad solo puede notarse si las capacidades DIFIEREN.
    La herramienta era la que forzaba la homogeneidad, no el fenomeno.
    """
    filters = [f.strip() for f in gpu_filter.split(",")] if gpu_filter else [""]
    if len(filters) == 1:
        filters = filters * stages
    if len(filters) != stages:
        sys.exit(f"--gpu-filter tiene {len(filters)} entradas para {stages} etapas: "
                 "pon una sola (se replica) o exactamente una por etapa")
    chosen = []
    for index, filt in enumerate(filters):
        gpu, price = pick_gpu(org, filt, priority, max_price)
        chosen.append((gpu, price))
        print(f"  etapa {index}: filtro '{filt}' -> {gpu['name']} ({price}$/h)")
    names = {gpu["name"] for gpu, _ in chosen}
    if len(names) == 1 and len(filters) > 1:
        print(f"  ⚠️  AVISO: los {len(filters)} filtros resolvieron a la MISMA clase ({names.pop()}). "
              "La cadena es HOMOGENEA y un A/B de fronteras no puede resolver nada: "
              "es exactamente el defecto de alcance de Exp7. Afina los filtros.")
    return chosen


def balanced_boundaries(total, n):
    b = [round(i * total / n) for i in range(n + 1)]
    b[0], b[-1] = 0, total
    for i in range(1, len(b)):
        if b[i] <= b[i - 1]:
            b[i] = b[i - 1] + 1
    if b[-1] != total:
        sys.exit(f"no caben {n} etapas en {total} capas")
    return b


# puertos loopback fijos por nodo (cada nodo aislado -> se pueden reutilizar)
P_FWD_ACCEPT = 41001   # root: bridge accept del forward-0
P_RET_DIAL = 41002     # root: bridge dial al return listener del root
P_LISTEN = 41010       # stage: puerto de escucha (bridge dial entra)
P_NEXT = 41011         # stage middle: puerto next (bridge accept)
P_RET_ACCEPT = 41012   # stage last: puerto return dial (bridge accept)
P_RET_DUMMY = 41099    # stage middle: return no usado
INTERNAL = 8071        # root: puerto API interno del server


def nakshatra_stage_args(cfg):
    """Banderas del motor nativo para una etapa, o `[]` si no se pide.

    Es el brazo D de EXP17: «C + etapa nakshatra/GGUF». No hay motor nuevo que
    escribir —`stage_cli.py` ya acepta estas banderas y `stage.py` ya construye
    el `NakshatraStageRunner`—; lo único que faltaba era pasarlas desde aquí.

    Fail-closed a propósito: si se pide el motor nativo y falta cualquier pieza
    obligatoria, se aborta en vez de caer en silencio al motor de PyTorch. Un
    brazo D que en realidad corrió PyTorch y se reporta como nativo es el mismo
    fallo de exp13 —medir un mecanismo que nunca se activó—, y aquí sería aún
    más difícil de detectar porque los tokens saldrían idénticos.
    """
    if not cfg:
        return []
    required = ("package", "daemon_bin", "pipeline_id", "context_tokens")
    missing = [name for name in required if not cfg.get(name)]
    if missing:
        raise SystemExit(
            "nakshatra: faltan campos obligatorios "
            f"{missing}. Se aborta en vez de caer al motor de PyTorch en "
            "silencio: un brazo 'nativo' que corrió PyTorch invalida la medida."
        )
    args = [
        "--nakshatra-package", str(cfg["package"]),
        "--nakshatra-daemon-bin", str(cfg["daemon_bin"]),
        "--nakshatra-pipeline-id", str(cfg["pipeline_id"]),
        "--nakshatra-context-tokens", str(cfg["context_tokens"]),
        "--nakshatra-gpu-layers", str(cfg.get("gpu_layers", -1)),
        "--nakshatra-compute-api", str(cfg.get("compute_api", "cuda")),
    ]
    for key, flag in (
        ("package_id", "--nakshatra-package-id"),
        ("manifest_sha256", "--nakshatra-manifest-sha256"),
    ):
        if cfg.get(key):
            args += [flag, str(cfg[key])]
    return args


def node_plan(n, boundaries, total, model, codec, ragged, nakshatra=None):
    """Devuelve, por nodo, (role, bridge_endpoints, my_args, env_extra).

    `nakshatra` activa el motor nativo en las etapas NO raíz. La raíz sigue en
    PyTorch: tokeniza, planifica y muestrea, y `server.py` no expone estas
    banderas. Es justo el reparto que quiere el brazo D —aislar el motor de la
    etapa, no cambiar el sistema entero de golpe.
    """
    plans = []
    for k in range(n):
        env_extra = {}
        if k == 0:
            role = "root"
            eps = [
                {"mode": "accept", "host": "127.0.0.1", "port": P_FWD_ACCEPT, "room": "f0"},
                {"mode": "dial", "host": "127.0.0.1", "port": P_RET_DIAL, "room": "ret"},
            ]
            my = ["--model", model, "--stages", str(n),
                  "--boundaries", ",".join(str(x) for x in boundaries),
                  "--threads-per-stage", "2", "--codec", codec, "--device", "cuda",
                  "--max-batch-size", "8", "--max-active-sequences", "8",
                  "--prefill-chunk-tokens", "128", "--no-speculation-probes",
                  "--first-stage-host", "127.0.0.1", "--first-stage-port", str(P_FWD_ACCEPT),
                  "--return-bind-host", "127.0.0.1", "--return-advertise-host", "127.0.0.1",
                  "--return-port", str(P_RET_DIAL), "--port", str(INTERNAL)]
            if ragged:
                env_extra["GDLP_RAGGED_GROUPING"] = "1"
        else:
            role = "stage"
            last = (k == n - 1)
            in_room = f"f{k-1}"
            eps = [{"mode": "dial", "host": "127.0.0.1", "port": P_LISTEN, "room": in_room}]
            my = ["--model", model, "--layer-start", str(boundaries[k]),
                  "--layer-end", str(boundaries[k + 1]), "--total-layers", str(total),
                  "--threads", "2", "--codec", codec, "--device", "cuda",
                  "--listen-host", "127.0.0.1", "--listen-port", str(P_LISTEN)]
            my += nakshatra_stage_args(nakshatra)
            if last:
                eps.append({"mode": "accept", "host": "127.0.0.1", "port": P_RET_ACCEPT, "room": "ret"})
                my += ["--return-host", "127.0.0.1", "--return-port", str(P_RET_ACCEPT)]
            else:
                out_room = f"f{k}"
                eps.append({"mode": "accept", "host": "127.0.0.1", "port": P_NEXT, "room": out_room})
                my += ["--next-host", "127.0.0.1", "--next-port", str(P_NEXT),
                       "--next-layer-end", str(boundaries[k + 2]),
                       "--return-host", "127.0.0.1", "--return-port", str(P_RET_DUMMY)]
            if ragged:
                env_extra["GDLP_RAGGED_GROUPING"] = "1"
        plans.append((role, eps, my, env_extra))
    return plans


def patch_boundaries(my_args, role, k, boundaries):
    """Reparte las capas según `boundaries` para ESTE nodo (arm A/B de split).

    Permite probar splits DESEQUILIBRADOS (p.ej. dar menos capas a la raíz, que
    además tokeniza/planifica/muestrea, o ajustar al nodo más lento) frente al
    reparto por capas iguales que usa el sistema por defecto.
    """
    a = list(my_args)

    def setflag(flag, val):
        if flag in a:
            a[a.index(flag) + 1] = str(val)

    if role == "root":
        setflag("--boundaries", ",".join(str(x) for x in boundaries))
    else:
        setflag("--layer-start", boundaries[k])
        setflag("--layer-end", boundaries[k + 1])
        if "--next-layer-end" in a:
            setflag("--next-layer-end", boundaries[k + 2])
    return a


def verify_payload_matches_worktree(payload_path):
    """Abortar si el runtime que se va a inyectar no es el del árbol de trabajo.

    El paquete se genera a mano y se reutiliza entre corridas, así que es fácil
    seguir midiendo una foto vieja del código sin enterarse: pasó de verdad —
    varios experimentos corrieron durante horas contra un runtime anterior al
    merge, sin el arreglo de KV ni el batching ragged, y los resultados parecían
    normales. Un experimento que mide código que no es el tuyo es peor que no
    tener experimento, porque parece dato.
    """
    import tarfile

    source = os.path.join(REPO_ROOT, "python", "distributed_runtime")
    if not os.path.isdir(source):
        return
    local = {
        f for f in os.listdir(source) if f.endswith(".py")
    }
    try:
        with tarfile.open(payload_path, "r:gz") as tar:
            packed = {
                os.path.basename(name)
                for name in tar.getnames()
                if name.endswith(".py")
            }
    except Exception as error:
        sys.exit(f"no pude leer el payload {payload_path}: {error}")
    missing = sorted(local - packed)
    if missing:
        sys.exit(
            f"PAYLOAD DESACTUALIZADO: a {payload_path} le faltan {len(missing)} módulos "
            f"del árbol de trabajo ({', '.join(missing[:6])}"
            f"{'…' if len(missing) > 6 else ''}).\n"
            "Regenéralo antes de medir:\n"
            '  python -c "import tarfile,os; '
            "t=tarfile.open(r'<destino>.tgz','w:gz'); "
            "t.add(os.path.join('python','distributed_runtime'), arcname='distributed_runtime', "
            "filter=lambda i: None if '__pycache__' in i.name else i); t.close()\""
        )
    newest = max(
        (os.path.getmtime(os.path.join(source, f)) for f in local), default=0
    )
    if newest > os.path.getmtime(payload_path):
        sys.exit(
            f"PAYLOAD MÁS ANTIGUO QUE EL CÓDIGO: {payload_path} se generó antes del "
            "último cambio en python/distributed_runtime. Regenéralo antes de medir."
        )


def patch_speculation(my_args, role, mode, draft_tokens=None):
    """Enciende la decodificación especulativa EXACTA en la raíz.

    Existe en el runtime (`--speculation ngram`) pero está apagada por defecto, y
    todos nuestros bancos anteriores además pasaban `--no-speculation-probes`, que
    desactiva el sondeo con el que el controlador decide si especular compensa.
    Para medirla de verdad hay que quitar ese flag Y encender el modo.
    """
    if role != "root" or mode is None:
        return my_args
    a = [x for x in my_args if x != "--no-speculation-probes"]
    if "--speculation" in a:
        a[a.index("--speculation") + 1] = str(mode)
    else:
        a.extend(["--speculation", str(mode)])
    if draft_tokens is not None:
        if "--speculative-max-draft-tokens" in a:
            a[a.index("--speculative-max-draft-tokens") + 1] = str(draft_tokens)
        else:
            a.extend(["--speculative-max-draft-tokens", str(draft_tokens)])
    return a


def patch_myargs(my_args, role=None, codec=None, max_active=None, max_batch=None,
                 root_window=None, stage_window=None, threads=None, prefill_chunk=None):
    """Parchea flags de my_args para un arm (mismo-nodo).

    setflag: sustituye si el flag ya está. addflag: lo AÑADE si falta (para
    parámetros que el plan base no fija, como las ventanas de coalescencia).
    Cada flag se aplica solo al rol que lo acepta (root=server, stage=stage_cli).
    """
    a = list(my_args)

    def setflag(flag, val):
        if val is None:
            return
        if flag in a:
            a[a.index(flag) + 1] = str(val)

    def addflag(flag, val, roles):
        if val is None or (role is not None and role not in roles):
            return
        if flag in a:
            a[a.index(flag) + 1] = str(val)
        else:
            a.extend([flag, str(val)])

    setflag("--codec", codec)
    setflag("--max-active-sequences", max_active)
    setflag("--max-batch-size", max_batch)
    setflag("--threads-per-stage", threads)
    setflag("--threads", threads)
    setflag("--prefill-chunk-tokens", prefill_chunk)
    # ventanas de coalescencia: la del root agrupa peticiones; la de etapa agrupa
    # el batch físico. Son EL knob que decide cuántas secuencias se fusionan.
    addflag("--root-batch-window-ms", root_window, ("root",))     # solo server.py
    addflag("--physical-batch-window-ms", stage_window, ("stage",))  # solo stage_cli.py
    return a


def chunk_env(env, name, raw_bytes):
    b64 = base64.b64encode(raw_bytes).decode()
    parts = [b64[i:i + 990] for i in range(0, len(b64), 990)]
    for i, c in enumerate(parts):
        env[f"{name}_{i}"] = c
    env[f"{name}_PARTS"] = str(len(parts))
    return len(parts)


def standalone_command(script_path, env):
    b64 = base64.b64encode(open(script_path, "rb").read()).decode()
    parts = [b64[i:i + 900] for i in range(0, len(b64), 900)]
    for i, c in enumerate(parts):
        env[f"GDLP_SCRIPT_{i}"] = c
    env["GDLP_SCRIPT_PARTS"] = str(len(parts))
    loader = ("import base64,os;n=int(os.environ['GDLP_SCRIPT_PARTS']);"
              "src=base64.b64decode(''.join(os.environ[f'GDLP_SCRIPT_{i}'] for i in range(n)));"
              "exec(compile(src,'<node>','exec'))")
    return ["python", "-c", loader]


def container_exists(org, project, name):
    st, d = call("GET", f"/organizations/{org}/projects/{project}/containers/{name}")
    return st == 200


def create_container(org, project, name, image, resources, env, command, auth, retries=3,
                     country=None):
    """Crea con reintentos ante errores TRANSITORIOS del borde (5xx/520 de
    Cloudflare). OJO: un 520 puede llegar DESPUÉS de que el contenedor se haya
    creado de verdad -> antes de reintentar o abortar, comprobamos si existe
    (si no, quedaría huérfano facturando fuera del teardown).

    `country` fija el país del nodo (`country_codes`). Sin él, cada contenedor
    cae donde quiera y la cadena puede acabar cruzando continentes: Exp15 midió
    **65 ms en el mismo país del relé contra 190 ms en otro**, y como el coste
    por token suma un tramo por etapa, esa diferencia se multiplica por la
    longitud de la cadena. Fijarlo cuesta un céntimo y diez minutos.

    ⚠️ La etiqueta de país NO basta por sí sola: dos tiradas con el mismo
    `country_codes=['us']` dieron medianas de 65 y 132,7 ms (Exp15 §principio 3).
    Reduce la varianza, no la elimina — el RTT hay que MEDIRLO igual.
    """
    body = {"name": name,
            "container": {"image": image, "resources": resources, "environment_variables": env, "command": command},
            "autostart_policy": True, "restart_policy": "never", "replicas": 1,
            "networking": {"protocol": "http", "port": 8000, "auth": auth}}
    if country and country != "any":
        body["country_codes"] = [country]
    last = None
    for attempt in range(retries):
        st, cg = call("POST", f"/organizations/{org}/projects/{project}/containers", body)
        if st in (200, 201):
            return body
        last = (st, cg)
        if container_exists(org, project, name):
            print(f"  aviso: create {name} devolvió {st} pero el contenedor SÍ existe; sigo")
            return body
        if st in (500, 502, 503, 504, 520, 521, 522, 524) and attempt + 1 < retries:
            print(f"  create {name} error transitorio {st}; reintento…")
            time.sleep(3 * (attempt + 1))
            continue
        break
    raise RuntimeError(f"create {name} fallo: {last[0]} {last[1]}")


def wait_running(base, timeout_min, t0):
    dns = None
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        st, cur = call("GET", base)
        status = cur.get("current_state", {}).get("status")
        dns = cur.get("networking", {}).get("dns")
        running = cur.get("current_state", {}).get("instance_status_counts", {}).get("running_count", 0)
        print(f"    {base.split('/')[-1]} estado={status} running={running} dns={dns} t+{int(time.time()-t0)}s", flush=True)
        if status == "running" and running >= 1 and dns:
            return dns
        time.sleep(15)
    raise TimeoutError(f"{base} no llegó a running")


def wait_ready(dns, timeout_min, t0, label):
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        try:
            sc, txt = gw(dns, "/ready")
            j = json.loads(txt)
            print(f"    {label} ready={j.get('ready')} status={j.get('status')} t+{int(time.time()-t0)}s", flush=True)
            if sc == 200 and j.get("ready"):
                return True
        except Exception as e:
            print(f"    {label} /ready {type(e).__name__} t+{int(time.time()-t0)}s", flush=True)
        time.sleep(10)
    return False


def wait_booted(dns, timeout_min, t0, label):
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        raw = None
        try:
            sc, raw = gw(dns, "/booted")
            j = json.loads(raw)
            print(f"    {label} booted={j.get('booted')} status={j.get('status')} t+{int(time.time()-t0)}s", flush=True)
            if sc == 200 and j.get("booted"):
                return True
        except Exception as e:
            # Print WHAT came back, not just the exception type: a non-JSON body is
            # almost always the gateway's own error page, and without the status and
            # first bytes there is no way to tell a booting node from a dead one.
            snippet = (raw or "")[:120].replace("\n", " ") if raw is not None else "(sin cuerpo)"
            print(
                f"    {label} /booted {type(e).__name__} t+{int(time.time()-t0)}s "
                f"resp={snippet!r}",
                flush=True,
            )
        time.sleep(10)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="example")
    ap.add_argument("--project", default="mycellios")
    ap.add_argument("--stages", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--total-layers", type=int, required=True)
    ap.add_argument("--codec", default="fp16")
    ap.add_argument("--ragged", action="store_true")
    ap.add_argument("--bench", action="store_true", help="tras la inferencia, sweep de throughput on-node (en el root)")
    ap.add_argument("--bench-lambdas", default="3,6,10", help="lista de lambda req/s para el sweep open-loop")
    ap.add_argument("--bench-dur", type=int, default=40)
    ap.add_argument("--bench-tokens", type=int, default=16)
    ap.add_argument("--bench-timeout-min", type=int, default=15)
    ap.add_argument("--gpu-filter", default="", help="subcadena de clase GPU; vacío = la más barata")
    ap.add_argument("--priority", default="high")
    ap.add_argument("--max-gpu-price", type=float, default=0.20, help="$/h máx por GPU")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    ap.add_argument("--payload-tgz", required=True)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--memory", type=int, default=8192)
    ap.add_argument("--storage-gb", type=int, default=20)
    ap.add_argument("--boot-timeout-min", type=int, default=25)
    ap.add_argument("--ready-timeout-min", type=int, default=15)
    ap.add_argument("--tag", default="", help="prefijo para los nombres de contenedor (evita colisiones al lanzar en paralelo)")
    ap.add_argument("--country", default=None,
                    help="Código de país ISO para relé Y nodos (p.ej. 'us'). Sin él, cada "
                         "contenedor cae donde quiera y la cadena puede cruzar continentes: "
                         "Exp15 midió 65 ms en el mismo país del relé contra 190 ms en otro, "
                         "y el coste por token suma un tramo por etapa. OJO: la etiqueta NO "
                         "basta (dos tiradas 'us' dieron 65 y 132,7 ms) — reduce la varianza, "
                         "no sustituye a medir el RTT.")
    ap.add_argument("--arms", default=None, help="A/B MISMO-NODO: lista p.ej. 'strict,ragged' -> 1 deploy, cada arm vía /stop+/start")
    ap.add_argument("--arms-json", default=None,
                    help='A/B genérico MISMO-NODO: JSON de arms con overrides, p.ej. '
                         '[{"label":"mas8","max_active":8},{"label":"mas32","max_active":32}]. '
                         'Claves: label, ragged(bool), codec(str), max_active(int), max_batch(int).')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = args.stages
    os.makedirs(args.out, exist_ok=True)
    boundaries = balanced_boundaries(args.total_layers, n)
    stage_gpus = pick_gpus_per_stage(args.org, args.gpu_filter, n, args.priority, args.max_gpu_price)
    total_hourly = sum(price for _, price in stage_gpus)  # relé ~0
    gpu_names = [gpu["name"] for gpu, _ in stage_gpus]
    heterogeneous = len(set(gpu_names)) > 1
    print(f"GPUs por etapa {gpu_names} = ~{total_hourly:.3f}$/h "
          f"({'HETEROGENEA' if heterogeneous else 'homogenea'}); boundaries {boundaries}")
    if total_hourly > 0.50:
        sys.exit(f"ABORTA: {total_hourly:.3f}$/h > 0.50 (mandato de presupuesto)")

    verify_payload_matches_worktree(args.payload_tgz)
    plans = node_plan(n, boundaries, args.total_layers, args.model, args.codec, args.ragged)
    # eid estable por endpoint (persiste entre arms A/B -> el relé reconoce el slot)
    import os as _osrand
    for _role, _eps, _my, _ev in plans:
        for _ep in _eps:
            _ep["eid"] = _osrand.urandom(16).hex()
    payload = open(args.payload_tgz, "rb").read()
    bridge_src = open(os.path.join(DEPLOY, "bridge.py"), "rb").read()
    stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")

    created = []  # (name, base, dns, price)
    t0 = time.time()
    # El manifest registra la GPU DE CADA ETAPA, no una sola. El manifest de Exp7
    # declaraba una unica clase y por eso costo tres meses descubrir que aquel
    # experimento no podia responder a la pregunta que decia responder.
    manifest = {"gpu_per_stage": gpu_names, "heterogeneous": heterogeneous,
                "price_per_hour_per_stage": [price for _, price in stage_gpus],
                "price_per_hour_total": round(total_hourly, 4),
                "stages": n, "boundaries": boundaries,
                "model": args.model, "codec": args.codec, "ragged": args.ragged, "nodes": []}
    result = {"manifest": manifest}
    try:
        # 1) RELÉ (CPU-only, auth:false en el borde de Salad)
        # El relé queda en una URL pública y el nombre de sala es adivinable, así
        # que la autenticación la pone el propio protocolo: un secreto por RUN,
        # compartido con los puentes, que firma cada HELLO (ver deploy/salad/relay.py).
        # Sin esto, un HELLO ajeno con `gen` alto mataba una tubería viva.
        relay_secret = secrets.token_hex(32)
        relay_name = f"gdlp-relay-{args.tag}{stamp}"
        renv = {"GDLP_PORT": "8000", "PYTHONUNBUFFERED": "1", "GDLP_RELAY_SECRET": relay_secret}
        rcmd = standalone_command(os.path.join(DEPLOY, "relay.py"), renv)
        created.append([relay_name, f"/organizations/{args.org}/projects/{args.project}/containers/{relay_name}", None, 0.0])
        create_container(args.org, args.project, relay_name, "docker.io/library/python:3.11-slim",
                         {"cpu": 2, "memory": 2048, "storage_amount": 8 * 1073741824}, renv, rcmd,
                         auth=False, country=args.country)
        print(f"relé {relay_name} creado (CPU-only, HELLO firmado por HMAC)")
        relay_dns = wait_running(created[-1][1], args.boot_timeout_min, t0)
        created[-1][2] = relay_dns
        relay_url = f"wss://{relay_dns}/relay"
        print(f"relé DNS={relay_dns}  RELAY_URL={relay_url}")

        # 2) NODOS GPU: crear SOLO con lo de boot (pip+extract+preload). La config
        #    del run (relay, endpoints, flags) va luego por /start (barrera).
        for k in range(n):
            role, eps, my, env_extra = plans[k]
            name = f"gdlp-n{k}-{args.tag}{stamp}"
            env = {"GDLP_PORT": "8000", "GDLP_ROLE": role, "PYTHONUNBUFFERED": "1",
                   "GDLP_MODEL": args.model, "HF_HOME": "/opt/hf",
                   "GDLP_RELAY_SECRET": relay_secret}
            chunk_env(env, "PAYLOAD", payload)
            env["GDLP_BRIDGE_B64"] = base64.b64encode(bridge_src).decode()
            cmd = standalone_command(os.path.join(DEPLOY, "pipeline_node.py"), env)
            stage_gpu, stage_price = stage_gpus[k]
            resources = {"cpu": args.cpu, "memory": args.memory, "gpu_classes": [stage_gpu["id"]],
                         "storage_amount": args.storage_gb * 1073741824}
            # REGISTRAR ANTES DE CREAR: si el create falla a medias (p.ej. 520 tras
            # crearse de verdad), el teardown del finally lo cubre igual.
            base = f"/organizations/{args.org}/projects/{args.project}/containers/{name}"
            # Las dos ramas aportaban piezas COMPLEMENTARIAS aquí: el precio y
            # la GPU POR ETAPA (necesarios para un reparto heterogéneo) y el
            # país (Exp15: 65 ms mismo país vs 190 ms distinto). Se conservan
            # ambas.
            created.append([name, base, None, stage_price])
            create_container(args.org, args.project, name, args.image, resources, env, cmd,
                             auth=True, country=args.country)
            manifest["nodes"].append({"index": k, "role": role, "name": name, "endpoints": eps,
                                      "my_args": my, "gpu": stage_gpu["name"],
                                      "price_per_hour": stage_price})
            print(f"nodo {k} ({role}) {name} creado")
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)

        # esperar running + dns
        for entry in created[1:]:
            entry[2] = wait_running(entry[1], args.boot_timeout_min, t0)
        # BARRERA: esperar a que TODOS estén booted (pip+extract+preload)
        all_booted = True
        for k, entry in enumerate(created[1:]):
            all_booted = wait_booted(entry[2], args.ready_timeout_min, t0, f"nodo{k}") and all_booted
        result["all_booted"] = all_booted
        root_dns = created[1][2]
        result["root_dns"] = root_dns
        # arms A/B sobre los MISMOS nodos (p.ej. strict,ragged): 1 deploy, cada
        # arm vía /stop+/start con gen -> el relé resetea sus buffers. Controla la
        # heterogeneidad de nodos (S0: ~1.6x entre hosts) que contamina el A/B
        # entre despliegues distintos.
        if args.arms_json:
            arm_list = json.loads(args.arms_json)
        elif args.arms:
            arm_list = [{"label": a.strip(), "ragged": a.strip() == "ragged"} for a in args.arms.split(",") if a.strip()]
        else:
            arm_list = [{"label": ("ragged" if args.ragged else "strict"), "ragged": args.ragged}]

        def _start_arm(gen, arm):
            ragged = bool(arm.get("ragged", args.ragged))
            for k in range(n):
                role, eps, my, env_extra = plans[k]
                my_p = patch_myargs(my, role=role, codec=arm.get("codec"),
                                    max_active=arm.get("max_active"), max_batch=arm.get("max_batch"),
                                    root_window=arm.get("root_window"), stage_window=arm.get("stage_window"),
                                    threads=arm.get("threads"), prefill_chunk=arm.get("prefill_chunk"))
                if arm.get("boundaries"):
                    my_p = patch_boundaries(my_p, role, k, arm["boundaries"])
                my_p = patch_speculation(
                    my_p, role, arm.get("speculation"), arm.get("draft_tokens")
                )
                eps_gen = [dict(ep, gen=gen) for ep in eps]   # mismo eid, gen del arm
                cfg = {"role": role, "relay_url": relay_url, "bridge_endpoints": eps_gen,
                       "my_args": my_p, "ragged": ragged}
                sc, txt = gw(created[1 + k][2], "/start", method="POST", body=cfg)
                print(f"  /start nodo{k} (gen={gen} {arm.get('label')}): {sc} {txt[:70]}")
            ok = True
            for k, entry in enumerate(created[1:]):
                ok = wait_ready(entry[2], args.ready_timeout_min, t0, f"nodo{k}") and ok
            return ok

        def _stop_arm():
            for k, entry in enumerate(created[1:]):
                try:
                    gw(entry[2], "/stop", method="POST", body={})
                except Exception as e:
                    print(f"  /stop nodo{k}: {e}")
            time.sleep(6)

        def _inference():
            sc, txt = gw(root_dns, "/v1/models")
            mid = json.loads(txt)["data"][0]["id"]
            ti = time.time()
            sc, txt = gw(root_dns, "/v1/chat/completions", method="POST", body={
                "model": mid, "messages": [{"role": "user", "content": "List three colors, comma separated."}],
                "max_tokens": 24, "temperature": 0, "stream": False, "seed": 7}, timeout=120)
            dt = time.time() - ti
            out = json.loads(txt); usage = out.get("usage", {}); toks = usage.get("completion_tokens")
            return {"text": out["choices"][0]["message"]["content"], "usage": usage, "seconds": round(dt, 2),
                    "tok_s_single": round(toks / dt, 2) if toks and dt else None}

        def _bench(arm=None):
            # tokens de salida por-arm (permite medir la degradación con respuestas
            # largas); si el arm no lo fija, usa el global.
            toks = (arm or {}).get("bench_tokens", args.bench_tokens)
            # lambda por-arm: imprescindible para comparar longitudes de salida a
            # CARGA DE TOKENS CONSTANTE (si no, alargar la salida multiplica la
            # carga ofrecida y se mide saturación, no el efecto de la longitud).
            lam_spec = str((arm or {}).get("bench_lambda", args.bench_lambdas))
            bench = []
            for lam in [x.strip() for x in lam_spec.split(",") if x.strip()]:
                sc, txt = gw(root_dns, f"/run?job=bench&mode=open&lam={lam}&dur={args.bench_dur}&tokens={toks}")
                jid = json.loads(txt).get("job_id")
                print(f"  lambda={lam} job={jid}")
                j = {"status": "running"}
                deadline = time.time() + args.bench_timeout_min * 60
                while time.time() < deadline:
                    sc, txt = gw(root_dns, f"/job?id={jid}")
                    j = json.loads(txt)
                    if j.get("status") in ("done", "error"):
                        break
                    time.sleep(10)
                if j.get("status") == "done":
                    res = j["result"]
                    summ = res["summary"]
                    # B_eff = prueba de que el grouping ragged está ACTIVO (strict~1,0)
                    print(f"  lambda={lam} -> {summ} B_eff={res.get('beff')} ragged={res.get('ragged_requested')}")
                    bench.append({"lambda_req": lam, "summary": summ, "beff": res.get("beff"),
                                  "ragged_requested": res.get("ragged_requested")})
                else:
                    print(f"  lambda={lam} FALLO: {j.get('error')}")
                    bench.append({"lambda_req": lam, "error": j.get("error")})
            return bench

        result["arms"] = {}
        if all_booted:
            print("TODOS BOOTED. Barriendo arms sobre los mismos nodos...")
            for ai, arm in enumerate(arm_list):
                label = arm.get("label", f"arm{ai}")
                gen = ai + 1
                if ai > 0:
                    print(f"=== reconfig -> arm '{label}' (/stop + /start, gen={gen}) ===")
                    _stop_arm()
                print(f"=== ARM '{label}' (gen={gen}, cfg={ {kk: vv for kk, vv in arm.items() if kk != 'label'} }) — MISMOS nodos ===")
                ready = _start_arm(gen, arm)
                arm_res = {"ready": ready, "arm_cfg": {kk: vv for kk, vv in arm.items() if kk != "label"}}
                if ready:
                    # La inferencia y el bench van en try SEPARADOS a proposito.
                    # Cuando estaban juntos, un fallo transitorio de la inferencia
                    # se llevaba el bench entero del brazo: en exp17 el relé de
                    # Salad se reinicio a mitad de corrida, la inferencia del brazo
                    # de TRATAMIENTO murio con KeyError 'choices' y el experimento
                    # perdio justo el brazo que respondia a su pregunta. Un error
                    # transitorio no puede costar un brazo.
                    for attempt in (1, 2):
                        try:
                            arm_res["inference"] = _inference()
                            print(f"  SALIDA[{label}]:", repr(arm_res['inference']['text'])[:70], arm_res['inference']['usage'])
                            break
                        except Exception as e:
                            arm_res.setdefault("inference_errors", []).append(f"{type(e).__name__}: {e}")
                            print(f"  arm '{label}' inferencia intento {attempt} fallo: {e}")
                            if attempt == 1:
                                time.sleep(15)  # margen para que el relé se reenganche
                    if args.bench:
                        try:
                            print(f"=== BENCH arm '{label}' (tokens={arm.get('bench_tokens', args.bench_tokens)}) ===")
                            arm_res["bench"] = _bench(arm)
                        except Exception as e:
                            arm_res["error"] = f"{type(e).__name__}: {e}"
                            print(f"  arm '{label}' error en bench (sigo): {e}")
                else:
                    print(f"  arm '{label}' no llegó a ready; recogiendo logs")
                    for k, entry in enumerate(created[1:]):
                        try:
                            sc, txt = gw(entry[2], "/log", timeout=20)
                            open(os.path.join(args.out, f"node{k}-{label}.log"), "w", encoding="utf-8").write(txt)
                        except Exception:
                            pass
                result["arms"][label] = arm_res
                json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)
        result["all_ready"] = any(a.get("ready") for a in result["arms"].values())
        json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)
    finally:
        # TEARDOWN A PRUEBA DE FALLOS. Antes, este bucle llamaba a `call("DELETE")`
        # sin protección: `call` sólo captura HTTPError, así que un corte de red o
        # un timeout durante el borrado abortaba el bucle entero y **dejaba
        # facturando el resto de los contenedores**. Es el mismo fallo que ya se
        # corrigió en `scripts/salad/latency_map.py` y que aquí seguía vivo.
        # Reglas: cada borrado aislado, con reintentos, y lo que no muera se
        # anuncia a gritos con el comando para rematarlo a mano.
        print("=== TEARDOWN ===")
        survivors = []
        for name, base, dns, pr in created:
            deleted = False
            for attempt in range(4):
                try:
                    st, _ = call("DELETE", base, timeout=30)
                    # 404 = ya no existe, que es el estado que queremos.
                    if st in (200, 202, 204, 404):
                        print(f"  DELETE {name} http={st}")
                        deleted = True
                        break
                    print(f"  DELETE {name} http={st} (intento {attempt + 1}/4)")
                except Exception as error:  # noqa: BLE001 - nada puede saltarse el borrado
                    print(f"  DELETE {name} EXCEPCION {type(error).__name__}: {error} "
                          f"(intento {attempt + 1}/4)")
                time.sleep(2 * (attempt + 1))
            if not deleted:
                survivors.append((name, base))
        if survivors:
            print()
            print("!!! ATENCION: NO se pudieron borrar estos contenedores y "
                  "SIGUEN FACTURANDO:")
            for name, base in survivors:
                print(f"  - {name}")
                print(f"    curl -X DELETE -H 'Salad-Api-Key: $SALAD_API_KEY' "
                      f"-H 'User-Agent: {UA}' '{API}{base}'")
            result["undeleted_containers"] = [name for name, _ in survivors]
        else:
            # Confirmación positiva: no basta con que el DELETE devuelva 2xx.
            try:
                st, listing = call("GET", f"/organizations/{args.org}/projects/"
                                          f"{args.project}/containers", timeout=30)
                if st == 200:
                    alive = [c["name"] for c in listing.get("items", [])
                             if c.get("name", "").startswith("gdlp-")]
                    result["containers_alive_after_teardown"] = alive
                    print(f"  verificado: {len(alive)} contenedores gdlp-* vivos tras el teardown")
            except Exception as error:  # noqa: BLE001
                print(f"  (no se pudo verificar el listado: {error})")
        elapsed_h = (time.time() - t0) / 3600
        # Suma de los precios REALES por etapa: con una cadena heterogenea, el
        # viejo `price * n` habria facturado todo al precio de la primera GPU.
        cost = elapsed_h * total_hourly  # relé ~0
        result["elapsed_h"] = round(elapsed_h, 4)
        result["est_cost_usd"] = round(cost, 4)
        print(f"sesión {elapsed_h:.3f} h; coste GPU ~{cost:.4f} $ "
              f"({' + '.join(f'{g}@{p}$/h' for g, p in zip(gpu_names, [pp for _, pp in stage_gpus]))})")
        json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

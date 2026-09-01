# deploy/salad — nodo mycellios en SaladCloud

> **Archivo de investigación, no integración de producto.** Estos artefactos
> conservan campañas reproducibles realizadas sobre infraestructura externa.
> Mycellios no ofrece SaladCloud como servicio, runtime, dependencia o destino
> de despliegue soportado. Esta carpeta no entra en el instalador ni se expone
> mediante comandos de `package.json`. Las ideas útiles deben implementarse y
> validarse en el runtime nativo.

Base reutilizable para correr mycellios en GPUs de consumo de SaladCloud.
**Diseño completo y checklist:** [`docs/DISENO_INTEGRACION_SALAD_MYCELLIOS.md`](../../docs/DISENO_INTEGRACION_SALAD_MYCELLIOS.md).

## Piezas

- `Dockerfile` — imagen pre-horneada (torch-CUDA + deps SELLADAS + runtime mycellios). **Regla nº1: nada de pip en el arranque.**
- `entrypoint.py` — entrypoint OBSERVABLE: sirve `/status` `/log` `/results.json` desde el segundo 0. Modos por env `GDLP_MODE`: `probe` (S0, listo), `server` (S1/S4), `worker` (dial-out, pendiente de cablear).
- El probe S0 vive en `python/distributed_runtime/salad_probe.py`; el orquestador en `scripts/salad/orchestrate.py`.

## Un comando (fase S0)

```bash
export SALAD_API_KEY=...   # nunca se committea
python scripts/salad/orchestrate.py \
  --org example --project mycellios \
  --image <REGISTRY>/mycellios-salad:<TAG> \
  --gpu "3060 (12" --mode probe --model Qwen/Qwen3-0.6B \
  --out docs/benchmarks/salad-s0-<fecha>/
```

Crea → vigila → recoge → **teardown siempre** → apunta coste. Presupuesto: ≤0,50 $/h.

## Construir la imagen (solo si cambió código/deps)

```bash
docker build -t <REGISTRY>/mycellios-salad:<TAG> -f deploy/salad/Dockerfile .
docker push <REGISTRY>/mycellios-salad:<TAG>
```

Requiere Docker + credenciales del registro (las pone el fundador).

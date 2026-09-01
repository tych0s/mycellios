# Imagen mycellios para nodos SaladCloud (GPU de consumo, CUDA).
# LECCIÓN nº1 (fallo del 24-07): NADA de `pip install` en el arranque del
# contenedor — es lento (7 GB de base + descarga de wheels) e INVISIBLE, y si
# falla el nodo se queda colgado sin logs. Todo se hornea aquí, una vez.
#
# Build (en una máquina con Docker + GPU-less está bien, es solo empaquetado):
#   docker build -t <REGISTRY>/mycellios-salad:<TAG> -f deploy/salad/Dockerfile .
#   docker push <REGISTRY>/mycellios-salad:<TAG>
# Salad tira de la imagen pública (o con registry_authentication si es privada).
#
# TORCH: se instala el wheel CUDA cuya versión se fija por ARG para poder casar
# la versión sellada. La exactitud cross-hardware la valida S0 (token-hash CUDA
# vs la referencia CPU); si una combinación torch/cuda rompe el hash, se cambia
# el ARG y se reconstruye — por eso la versión es un parámetro, no un supuesto.

ARG CUDA_TAG=12.1.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ARG TORCH_VERSION=2.5.1
ARG TORCH_CUDA=cu121
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Python 3.12 (misma serie que el venv sellado 3.12.13) + utilidades mínimas.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git ca-certificates curl \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Torch CUDA primero (capa grande, se cachea) y luego las deps SELLADAS.
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Versiones EXACTAS del runtime (deben coincidir con python/requirements-distribution.txt;
# transformers 5.14.1 es la versión sellada de paged_stage.py — crítica para exactitud).
COPY python/requirements-distribution.txt /tmp/requirements-distribution.txt
RUN python -m pip install -r /tmp/requirements-distribution.txt

# Código del runtime mycellios (solo lo necesario para servir/medir).
COPY python/distributed_runtime /opt/mycellios/distributed_runtime
COPY deploy/salad/entrypoint.py /opt/mycellios/entrypoint.py
ENV PYTHONPATH=/opt/mycellios

# HuggingFace cache dentro de la imagen si se quiere pre-descargar modelos
# (opcional; por defecto se descargan en caliente — outbound es gratis en Salad).
ENV HF_HOME=/opt/hf-cache
ENV PYTHONDONTWRITEBYTECODE=1

# El runtime no necesita privilegios. La caché y el código quedan accesibles
# para un usuario dedicado, evitando ejecutar modelos o descargas como root.
RUN groupadd --system --gid 10001 mycellios \
    && useradd --system --uid 10001 --gid mycellios \
        --home-dir /opt/mycellios --shell /usr/sbin/nologin mycellios \
    && mkdir -p /opt/hf-cache \
    && chown -R mycellios:mycellios /opt/mycellios /opt/hf-cache
USER mycellios:mycellios

# El gateway de Salad expone UN puerto; el entrypoint sirve status/log/results ahí.
EXPOSE 8000

# Entrypoint OBSERVABLE (lección nº2): arranca el servidor HTTP de estado en el
# segundo 0, captura todo el stdout, y ejecuta el modo pedido por env GDLP_MODE.
ENTRYPOINT ["python", "/opt/mycellios/entrypoint.py"]

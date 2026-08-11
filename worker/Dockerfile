ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    VIDIOAI_RUNTIME_DEPS_DIR=/cache/runtime-deps \
    VIDIOAI_MODELS_DIR=/models \
    VIDIOAI_WORK_DIR=/work \
    VIDIOAI_OUTPUTS_DIR=/outputs

# Dépendances système du worker.
# python3-venv permet d'éviter le blocage PEP 668.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        tini \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vidioai-worker

# ------------------------------------------------------------
# Environnement Python dédié au worker
# ------------------------------------------------------------
#
# IMPORTANT :
# --system-site-packages permet au venv de conserver l'accès au
# PyTorch/CUDA déjà fourni par l'image pytorch/pytorch.
#
RUN python -m venv --system-site-packages /opt/vidioai-venv

ENV PATH="/opt/vidioai-venv/bin:${PATH}"

# Vérification : on veut bien utiliser le Python du venv
# tout en voyant PyTorch fourni par l'image de base.
RUN python --version \
    && python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

# ------------------------------------------------------------
# Dépendances Python VidioAI
# ------------------------------------------------------------

COPY requirements.txt ./

RUN python -m pip install \
    --no-cache-dir \
    --requirement requirements.txt

# Le cas NF4 réel est un contrat de l'image, pas seulement un test de l'hôte.
RUN python -c "import bitsandbytes, torch; from bitsandbytes.nn import Linear4bit; assert bitsandbytes.__version__ == '0.49.2'; assert torch.version.cuda == '12.8'; assert Linear4bit is not None"

# ------------------------------------------------------------
# Utilisateur non-root
# ------------------------------------------------------------

RUN useradd \
        --system \
        --uid 10002 \
        --create-home \
        worker \
    && install -d \
        -m 0750 \
        -o worker \
        -g worker \
        /models \
        /cache \
        /work \
        /outputs

COPY --chown=worker:worker app ./app

USER worker

EXPOSE 8000

HEALTHCHECK \
    --interval=15s \
    --timeout=5s \
    --retries=5 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

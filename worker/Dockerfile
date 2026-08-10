ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM $PYTORCH_IMAGE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    VIDIOAI_MODELS_DIR=/models \
    VIDIOAI_WORK_DIR=/work \
    VIDIOAI_OUTPUTS_DIR=/outputs

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vidioai-worker
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements.txt

RUN useradd --system --uid 10002 --create-home worker \
    && install -d -m 0750 -o worker -g worker /models /work /outputs

COPY --chown=worker:worker app ./app
USER worker

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

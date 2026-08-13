# Le contexte Docker est la racine du dépôt afin d'embarquer l'autorité
# partagée model-packs/workflows dans la même image immuable que le backend.
ARG VIDIOAI_VERSION=dev

# Étape de compilation reproductible du serveur Rust.
# Les versions verrouillées de `all-smi` et `sysinfo` nécessitent respectivement
# Rust 1.96 et 1.95. Le tag explicite garde le build reproductible.
FROM rust:1.96-bookworm AS builder
WORKDIR /src
# `luwen-api` compile ses schémas Protobuf via `prost-build`. `all-smi`
# active aussi le backend AMD qui doit être lié aux bibliothèques Linux DRM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libdrm-dev libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/src ./src
# Les caches BuildKit rendent les reconstructions après une modification Rust
# incrémentales. Le binaire final est copié hors du mount avant la fin du RUN.
RUN --mount=type=cache,id=vidioai-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=vidioai-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=vidioai-cargo-target,target=/src/target \
    cargo build --release --locked \
    && cp /src/target/release/rust_ai /tmp/vidioai-backend

# FFmpeg/FFprobe servent au moteur vidéo. Les bibliothèques DRM sont conservées
# au runtime pour que la détection GPU AMD de `all-smi` puisse être chargée.
FROM debian:bookworm-slim AS runtime
ARG VIDIOAI_VERSION
LABEL org.opencontainers.image.version="${VIDIOAI_VERSION}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        awscli ca-certificates curl ffmpeg libdrm-amdgpu1 libdrm2 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --create-home vidioai
WORKDIR /app
# Les volumes Docker héritent des droits du point de montage lors de leur
# première création. Le backend non-root peut ainsi initialiser son état local.
RUN mkdir -p /opt/vidioai/model-packs /opt/vidioai/workflows \
        /registry /state /models /outputs /cache /work \
    && chown -R vidioai:vidioai \
        /opt/vidioai /registry /state /models /outputs /cache /work
COPY --from=builder /tmp/vidioai-backend /usr/local/bin/vidioai-backend
COPY --chown=vidioai:vidioai model-packs /opt/vidioai/model-packs
COPY --chown=vidioai:vidioai workflows /opt/vidioai/workflows
USER vidioai
ENV VIDIOAI_VERSION=${VIDIOAI_VERSION} \
    VIDIOAI_STATE_DIR=/state \
    VIDIOAI_MODELS_DIR=/models \
    VIDIOAI_OUTPUTS_DIR=/outputs \
    VIDIOAI_CACHE_DIR=/cache \
    VIDIOAI_WORK_DIR=/work \
    VIDIOAI_MODEL_PACKS_DIR=/opt/vidioai/model-packs \
    VIDIOAI_WORKFLOWS_DIR=/opt/vidioai/workflows \
    VIDIOAI_MODEL_PACK_REGISTRY_DIR=/registry
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vidioai-backend"]

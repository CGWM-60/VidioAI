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
COPY Cargo.toml Cargo.lock ./
COPY src ./src
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
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        awscli ca-certificates curl ffmpeg libdrm-amdgpu1 libdrm2 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --create-home vidioai
WORKDIR /app
# Les volumes Docker héritent des droits du point de montage lors de leur
# première création. Le backend non-root peut ainsi initialiser son état local.
RUN mkdir -p /data /state /models /outputs /cache /work \
    && chown -R vidioai:vidioai /data /state /models /outputs /cache /work
COPY --from=builder /tmp/vidioai-backend /usr/local/bin/vidioai-backend
USER vidioai
ENV VIDIOAI_STATE_DIR=/state \
    VIDIOAI_MODELS_DIR=/models \
    VIDIOAI_OUTPUTS_DIR=/outputs \
    VIDIOAI_CACHE_DIR=/cache \
    VIDIOAI_WORK_DIR=/work
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vidioai-backend"]

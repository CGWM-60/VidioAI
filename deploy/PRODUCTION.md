# Exploitation VidioAI

La procédure canonique, les variables, le bootstrap GPU, le stockage S3, le
drainage, le smoke test matériel et le rollback sont documentés à la racine dans
[`DEPLOIEMENT_SCALEWAY_GPU.md`](../DEPLOIEMENT_SCALEWAY_GPU.md).

Raccourcis :

```bash
# Runner de build/CI, jamais l'Instance GPU
VIDIOAI_REGISTRY=... ./deploy/scripts/build-release.sh <version>

# Instance GPU préalablement bootstrappée
./deploy/scripts/deploy.sh <version>
./deploy/scripts/smoke-test.sh http://127.0.0.1:8080
./deploy/scripts/shutdown.sh
./deploy/scripts/rollback.sh
```

`compose.production.yml` ne possède aucune directive `build` et refuse les
variables critiques absentes. Le proxy écoute la boucle locale tant qu'un TLS
et une authentification utilisateur n'ont pas été installés en frontal.

Le bundle contient aussi `deploy/bin/vidioai-host-agent` compilé pour Linux et
son unité systemd. `bootstrap-server.sh` installe le binaire hors Docker, génère
le token interne si nécessaire, démarre le service et valide `/health` puis
`/system` avant d'autoriser le déploiement Compose.

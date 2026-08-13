# Exploitation VidioAI

La procédure canonique, les variables, le bootstrap GPU, le stockage S3, le
drainage, le smoke test matériel et le rollback sont documentés à la racine dans
[`DEPLOIEMENT_SCALEWAY_GPU.md`](../DEPLOIEMENT_SCALEWAY_GPU.md).

Raccourcis :

```bash
# Runner de build/CI, jamais l'Instance GPU
VIDIOAI_REGISTRY=... ./deploy/scripts/build-release.sh <version>

# Instance GPU préalablement bootstrappée : commande d'installation canonique
./deploy/release-install.sh <version>
./deploy/scripts/smoke-test.sh http://127.0.0.1:8080
./deploy/scripts/shutdown.sh
./deploy/scripts/rollback.sh
```

`release-install.sh` accepte une archive locale (`VIDIOAI_RELEASE_ARCHIVE`),
une URL (`VIDIOAI_RELEASE_URL` ou `VIDIOAI_RELEASE_BASE_URL`) ou la release S3
`s3://$AWS_S3_BUCKET/releases/<version>/`. Il vérifie le manifest, les trois
tags et leurs digests avant tout arrêt, copie le bundle vérifié, réutilise
`shutdown.sh` puis `deploy.sh`, et termine par des contrôles HTTP sans lancer de
génération GPU.

ComfyUI est un moteur headless interne optionnel. Définir
`COMPOSE_PROFILES=comfyui` dans `.env.production` l'active; son image est
épinglée par digest par défaut et reste remplaçable avec
`VIDIOAI_COMFYUI_IMAGE`. Le worker le joint via `COMFYUI_URL`.

Les ModelPacks livrés dans les images restent le socle de secours immuable. Le
backend initialise puis met à jour le registre indépendant dans
`$VIDIOAI_STATE_DIR/model-pack-registry`; ce répertoire est monté en lecture-
écriture dans le backend et en lecture seule dans le worker. Les activations,
rollbacks et publications passent par le Lab administrateur, conservent les
anciennes versions et sont vérifiés par SHA-256 avant rechargement à chaud.

`compose.production.yml` ne possède aucune directive `build` et refuse les
variables critiques absentes. Le proxy écoute la boucle locale tant qu'un TLS
et une authentification utilisateur n'ont pas été installés en frontal.

Le bundle contient aussi `deploy/bin/vidioai-host-agent` compilé pour Linux et
son unité systemd. `bootstrap-server.sh` installe le binaire hors Docker, génère
le token interne si nécessaire, démarre le service et valide `/health` puis
`/system` avant d'autoriser le déploiement Compose.

## Scratch GPU obligatoire

`bootstrap-server.sh` refuse GPU_PRODUCTION si `/scratch` n'est pas un mount
inscriptible distinct de `/`. Il crée `/scratch/vidioai/{models,cache,work,worker-work}`
et écrit explicitement `VIDIOAI_SCRATCH_DIR=/scratch/vidioai` dans
`.env.production`. Le preflight contrôle ensuite le Compose résolu et le
filesystem réel avant que le Worker puisse démarrer.

Si des données existent encore dans `/var/lib/vidioai/scratch`, le bootstrap les
copie et les compare sans supprimer la source. La procédure explicite est :

```bash
sudo ./deploy/scripts/migrate-scratch.sh prepare
./deploy/scripts/deploy.sh <version>
sudo ./deploy/scripts/migrate-scratch.sh verify
# Seulement après vérification et avec confirmation destructive explicite :
sudo VIDIOAI_CONFIRM_SCRATCH_CLEANUP=DELETE_OLD_SCRATCH \
  ./deploy/scripts/migrate-scratch.sh cleanup
```

# Déploiement VidioAI sur Scaleway GPU

## Architecture des ressources physiques

Le backend et le worker restent non privilégiés dans Docker. Les informations
de la machine sont collectées par `/usr/local/bin/vidioai-host-agent`, un petit
service Rust natif géré par systemd :

```text
Hôte Linux / macOS
├── vidioai-host-agent :8091  (token interne)
└── Docker
    ├── backend ── HOST_AGENT_URL ──► host-gateway:8091
    ├── frontend
    └── worker GPU ──► NVIDIA/CUDA
```

`/api/system` préfère toujours `source: host` et ne bascule sur
`source: container` qu'avec un diagnostic explicite. `/api/resources` conserve
séparément les ressources physiques, celles du worker et l'état de la queue.
En `GPU_PRODUCTION`, l'agent natif, un GPU NVIDIA physique, CUDA dans le worker,
le Scratch, SQLite, FFmpeg et S3 sont tous obligatoires pour la readiness.

En local, la commande canonique est :

```bash
./scripts/dev.sh
# Ctrl+C, ou depuis un autre terminal :
./scripts/stop.sh
```

Sur Apple Silicon, Metal et la mémoire unifiée sont identifiés ; les métriques
Metal que macOS ne publie pas de façon fiable restent `null`.

Ce document est la procédure exécutable du dépôt. Il décrit l'état réellement
implémenté au 9 août 2026 et distingue les contrôles locaux de la validation qui
nécessite physiquement une carte NVIDIA.

## 1. Architecture livrée

```text
Navigateur
  └─ proxy Nginx (boucle locale / tunnel SSH)
      ├─ frontend Next.js
      └─ backend Rust/Axum
          ├─ queue durable SQLite
          ├─ assets et manifests
          ├─ Object Storage S3-compatible
          └─ API interne authentifiée
              └─ worker Python
                  ├─ Hugging Face snapshot_download
                  ├─ Diffusers + PyTorch
                  ├─ CUDA / NVIDIA
                  └─ sortie PNG réelle
```

Le backend ne contient ni PyTorch ni CUDA. Le worker est le seul service ayant
accès au GPU. Les moteurs `Vidio Canvas Local` et `Vidio Motion Local` ont
`engine_type=procedural`; les dépôts Hugging Face ont `engine_type=ai`.

Le worker livré exécute réellement `TEXT_TO_IMAGE`. Les autres endpoints IA
répondent `501` et les modèles vidéo sont marqués `RUNTIME_UNAVAILABLE`, afin
qu'aucune vidéo FFmpeg ne soit faussement attribuée à un modèle IA.

## 2. Profils et readiness

- `LOCAL` autorise les moteurs procéduraux sans CUDA. Le worker est facultatif.
- `GPU_PRODUCTION` exige le worker, PyTorch, CUDA, une NVIDIA, SQLite, FFmpeg,
  les volumes inscriptibles, S3 si activé, et des jetons administrateur/worker.

Routes :

- `GET /healthcheck` : alias historique de liveness ;
- `GET /api/health` : liveness du backend, sans dépendance externe ;
- `GET /api/ready` : readiness détaillée et code 503 si une dépendance requise
  manque ;
- `GET /api/resources` : OS/CPU/RAM/disques réels, GPU worker réel, modèles
  chargés et métriques de queue ;
- `POST /api/admin/drain`, `/resume`, `/stop` : cycle
  `ACCEPTING_JOBS → DRAINING → STOPPING` protégé par Bearer token.

`READY` pour un modèle signifie strictement : poids installés et validés,
runtime disponible et compatible, puis test d'inférence réussi. Les états sont
`NOT_INSTALLED`, `DOWNLOADING`, `INSTALLED`, `RUNTIME_UNAVAILABLE`,
`INCOMPATIBLE`, `READY` et `FAILED`.

## 3. Stockage

Les chemins sont indépendants et configurables :

```text
/var/lib/vidioai/state             SQLite + settings
/var/lib/vidioai/outputs           assets et manifests locaux
/var/lib/vidioai/scratch/models    cache de poids
/var/lib/vidioai/scratch/cache     cache applicatif
/var/lib/vidioai/scratch/work      échange backend ↔ worker
/var/lib/vidioai/scratch/worker-work
```

Hiérarchie du cache :

1. L1 : pipeline chargé en VRAM ;
2. L2 : snapshot validé sur Scratch ;
3. L3 : `s3://<bucket>/models/<repository>/main/`.

Les résultats sont publiés sous
`s3://<bucket>/outputs/<generation-id>/<fichier>`. Les bundles sont sous
`releases/<version>/` et les snapshots d'état sous `state/snapshots/<date>/`.

## 4. Développement local

```bash
docker compose up --build
```

Sans worker GPU, tester :

```bash
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8080/api/ready
curl http://127.0.0.1:8080/api/resources
```

La suite complète locale est :

```bash
make test
docker compose config --quiet
```

Les tests worker n'installent pas PyTorch et ne simulent jamais CUDA. Ils
vérifient notamment que `GPU_PRODUCTION` reste non prêt sans vraie NVIDIA, qu'un
manifest sans poids est refusé et qu'un snapshot L3 valide est réutilisé.

## 5. Construire une release hors du GPU

Prérequis sur le runner : Rust, Node 22, Python 3.12+, Docker Buildx, `jq`, AWS
CLI et une authentification au Container Registry.

```bash
export VIDIOAI_REGISTRY=rg.fr-par.scw.cloud/vidioai
export AWS_S3_BUCKET=mon-bucket-prive
export AWS_ENDPOINT_URL_S3=https://s3.fr-par.scw.cloud
export AWS_DEFAULT_REGION=fr-par
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'

./deploy/scripts/build-release.sh 2026.08.09-1
```

Le script exécute Clippy/tests Rust, lint/build Next.js, tests worker, construit
les trois images `linux/amd64`, les pousse sans tag `latest`, capture leurs
digests, crée `output/vidioai-release-<version>.tar.gz` et le publie dans S3 si
le bucket est fourni.

## 6. Préparer Scaleway

Actions cloud qui ne peuvent pas être réalisées depuis le dépôt :

1. Créer un namespace Container Registry privé.
2. Créer un bucket Object Storage privé dans la même région.
3. Créer des identités IAM séparées pour la CI et le serveur, avec le minimum
   de permissions Registry/Object Storage.
4. Créer une Instance GPU NVIDIA avec suffisamment de VRAM et un Scratch assez
   grand pour les poids et le travail temporaire.
5. Ne pas ouvrir l'application sur Internet pendant la validation. Le proxy de
   `compose.production.yml` écoute uniquement `127.0.0.1`.

Le script optionnel de création exige un type explicite afin d'éviter toute
machine coûteuse créée par défaut :

```bash
export SCW_DEFAULT_PROJECT_ID='...'
export SCW_DEFAULT_ZONE=fr-par-2
export VIDIOAI_SERVER_TYPE='TYPE_GPU_CHOISI'
./deploy/scaleway/create-server.sh
```

## 7. Bootstrap du serveur

Copier le bundle dans `/opt/vidioai`, puis :

```bash
sudo /opt/vidioai/deploy/scripts/bootstrap-server.sh
```

Le script installe Docker, AWS CLI et NVIDIA Container Toolkit si nécessaire,
prépare les bind mounts et leurs ACL pour les UID non-root 10001/10002, puis
valide `nvidia-smi` depuis un conteneur CUDA. Il échoue si aucune NVIDIA réelle
n'est disponible.

## 8. Configuration de production

```bash
cd /opt/vidioai
cp .env.production.example .env.production
chmod 600 .env.production
```

Remplacer toutes les valeurs secrètes. Générer au minimum deux jetons distincts
de 32 octets :

```bash
openssl rand -hex 32   # VIDIOAI_WORKER_TOKEN
openssl rand -hex 32   # VIDIOAI_ADMIN_TOKEN
```

Les variables S3 utilisent les noms AWS standards : `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL_S3`,
`AWS_S3_BUCKET` et `AWS_S3_STORAGE_CLASS`.

Authentifier Docker au Registry avant le déploiement :

```bash
docker login rg.fr-par.scw.cloud
```

## 9. Déployer

```bash
cd /opt/vidioai
./deploy/scripts/deploy.sh 2026.08.09-1
```

Le script enregistre la vraie version courante comme version précédente,
demande le drainage, fait uniquement `docker compose pull` puis `up --wait`, et
lance les smoke tests. Il n'exécute ni build ni installation Python sur le GPU.

Accès privé depuis le poste de travail :

```bash
ssh -L 8080:127.0.0.1:8080 utilisateur@ip-du-serveur
```

Ouvrir ensuite `http://127.0.0.1:8080`. Avant toute exposition publique,
terminer TLS dans un load balancer/reverse proxy et ajouter une authentification
utilisateur. CORS n'est pas un mécanisme d'authentification.

## 10. Smoke test matériel réel

La readiness valide déjà le worker et CUDA. Pour télécharger, valider et charger
un snapshot, puis produire et relire un vrai PNG :

```bash
export VIDIOAI_SMOKE_PROFILE=GPU_PRODUCTION
export VIDIOAI_SMOKE_AI_MODEL_ID=stable-image-core
./deploy/scripts/smoke-test.sh http://127.0.0.1:8080
```

Selon le débit Hugging Face et la taille du modèle, augmenter
`SMOKE_MODEL_ATTEMPTS`. Un dépôt gated nécessite `HF_TOKEN`.

## 11. Arrêt et rollback

Arrêt sûr avec drainage, attente de queue vide, arrêt Compose et snapshot S3 :

```bash
./deploy/scripts/shutdown.sh
```

Rollback vers la version réellement précédente :

```bash
./deploy/scripts/rollback.sh
```

Le rollback tire les anciennes images, attend leur healthcheck, exécute les
smoke tests puis échange correctement `.current-version` et `.previous-version`.

## 12. État de validation

### Implémenté et testé localement

- worker API séparé, authentifié, états/capacités/annulation ;
- téléchargement complet HF, validations de poids et checksums ;
- chemin backend → worker → Diffusers → PNG ;
- séparation `procedural` / `ai` ;
- SQLite WAL et reprise des jobs actifs en `interrupted` ;
- bind mounts, profils, readiness, resources, drainage ;
- interface S3, cache L1/L2/L3 et structure des objets ;
- Dockerfiles production, Compose sans build, scripts release/deploy/rollback ;
- Clippy strict, 9 tests Rust, 6 tests worker, lint et build des 12 routes Next.

### Implémenté, validation matérielle restante sur Scaleway

- import PyTorch CUDA depuis l'image worker ;
- détection NVIDIA/CUDA/VRAM/température/utilisation par `nvidia-smi` ;
- chargement SDXL en VRAM et inférence T2I réelle ;
- performance, capacité VRAM, coût et temps de téléchargement réels ;
- accès réel au Registry et au bucket privés de votre compte.

Rien dans le code ne remplace ces validations par un faux GPU. Si CUDA manque en
`GPU_PRODUCTION`, `/api/ready` renvoie 503 et le smoke test échoue.

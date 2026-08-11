# Post-mortem du runtime modèles — cible 2026.08.11-3

Ce document décrit le comportement observé dans le code à `a595958` avant toute correction du runtime. Il distingue les faits reproductibles des informations de production qui ne sont pas présentes dans le dépôt.

## Incident observé

Le backend place un job d'installation à `progress = 20`, étape `downloading`, immédiatement avant l'appel `worker.install(...)`. Le Worker peut ensuite rejeter le modèle pendant son préflight avec `PIPELINE_UNSUPPORTED`, avant le téléchargement des poids. L'affichage « échec à 20 % » ne prouve donc ni un échec réseau ni un téléchargement partiel : il localise l'échec dans le contrat d'installation Worker.

Le `repo_id` exact visible dans la capture mentionnée par la mission n'est pas récupérable depuis l'état local : `backend/data/jobs.sqlite` ne contient aucun job, aucun log de production n'est versionné et la capture elle-même n'est pas jointe. Les mentions de `Wan-AI/Wan2.2-TI2V-5B-Diffusers` dans les anciennes consignes et tests sont des exemples ou des candidats de test, pas une preuve de l'identité du job fautif. Aucun fixture spécifique ne doit être créé avant obtention de cette preuve.

## Analyse du code avant correction

| Mécanisme actuel | Problème précis | Impact production | Pourquoi les tests passent | Principe de correction |
|---|---|---|---|---|
| `RuntimeManager._imports()` retourne un tuple imbriqué | Le contrat n'est ni nommé ni vérifié ; chaque appelant dépend d'un unpacking positionnel différent | Une erreur d'unpacking peut sortir de `runtime_status()` | Un test affirme explicitement l'ancien contrat à deux valeurs au lieu de tester ses consommateurs | Introduire un objet `RuntimeImports` nommé et faire consommer ses attributs par tous les chemins |
| `/ready` appelle directement `runtime_status()` | Seuls certains `WorkerError` sont convertis ; `ValueError`, `ImportError` ou un contrat interne invalide peuvent devenir HTTP 500 | L'orchestrateur voit une panne applicative au lieu d'une readiness négative diagnostiquée | Les tests ne simulent que l'erreur métier attendue | Garantir une réponse structurée 200/503 pour toute panne de readiness, sans masquer `/health` |
| Préflight Hugging Face limité à `model_index.json` et `config.json` | L'absence d'une classe immédiatement résolue est traitée comme une incompatibilité certaine | Des pipelines Diffusers valides sont bloquées avant téléchargement | Les fixtures injectent souvent directement `class_name` ou `capabilities` | Modéliser `SUPPORTED`, `UNKNOWN`, `UNSUPPORTED`; réserver le rejet au cas prouvé |
| Compatibilité backend et UI booléenne | `UNKNOWN` est aplati en `false` | L'UI affiche « runtime non compatible » et désactive l'installation alors qu'une validation locale est encore possible | Les contrats Rust ne possèdent aucun troisième état à tester | Propager le statut tri-state et sa raison du Worker au frontend |
| `PipelineRegistry` peut conditionner l'installation | La présence d'un adapter spécialisé reste une barrière malgré le fallback générique | `PIPELINE_UNSUPPORTED` reste fréquent | Les tests construisent une registry avec métadonnées synthétiques déjà favorables | Faire de la classe Diffusers réellement importable la source de vérité et des adapters spécialisés une optimisation |
| Fixtures vidéo synthétiques | Les tests fabriquent classe, tags et capacités, notamment une fausse pipeline Wan injectée globalement | Une version de Diffusers sans la vraie classe peut tout de même être déclarée compatible en CI | `conftest.py` remplace le module Diffusers et masque les incompatibilités de version | Utiliser des métadonnées publiques figées et tester plusieurs familles contre les dépendances de l'image finale |
| Test du generic adapter isolé | Le code VidioAI interne est court-circuité par des mocks et des capacités inventées | Les ruptures entre inspection, résolution, registry, kwargs et normalisation ne sont pas détectées | Chaque unité réussit indépendamment | Ajouter un contrat traversant toute la chaîne interne, en ne remplaçant que réseau, CUDA et poids géants |
| `load_model()` exécute `_validate_loaded_pipeline()` | Charger implique une vraie inférence, même pendant le parcours d'installation backend | Coût GPU, timeout et confusion entre `INSTALLED`, `LOADED`, `READY` | Les pipelines de test renvoient instantanément un petit résultat artificiel | Séparer download, install/validate, load et readiness; réserver l'inférence à un test explicite |
| États Worker et timeline frontend divergents | La timeline annonce téléchargement, chargement CUDA, inférence, cache et démarrage comme une seule installation | Progression trompeuse et erreurs difficiles à diagnostiquer | Aucun test frontend ne vérifie la succession réelle des états | Exposer et tester des états distincts, un message et un code d'erreur persistants |
| Asset I2V toléré comme chaîne brute | Les adapters peuvent transmettre `asset_id` au pipeline si le backend n'a pas fourni un chemin | Une pipeline reçoit un identifiant opaque à la place d'une image | Les tests directs utilisent eux-mêmes des chaînes et valident ce faux contrat | Exiger des entrées normalisées et échouer clairement avant l'appel pipeline si l'asset n'est pas résolu |
| Résolution/fps/frames présents mais testés séparément | Le contrat 480p/720p n'est pas prouvé dans une génération vidéo complète | Une résolution valide isolément peut échouer dans la pipeline ou l'encodage | Les tests ne traversent pas le resolver, le profil, l'adapter et ffmpeg ensemble | Ajouter un contrat CPU complet 480p/720p et `N * k + 1` jusqu'à `ffprobe` |
| `OutputNormalizer` est strict, mais les smokes le sont moins | Certains scripts vérifient seulement que `ffprobe` s'exécute | Un MP4 vide, mauvais codec ou mono-frame peut passer une couche d'acceptance | Le test unitaire H.264 n'est pas relié aux scripts de release | Vérifier codec H.264, dimensions, durée et nombre de frames à chaque frontière finale |
| Image Worker construite puis immédiatement poussée | Le script de release ne teste pas l'image finale avant publication | Les versions réellement installées peuvent différer de l'environnement de tests | La CI Python installe seulement `requirements-test.txt`, sans Diffusers/Transformers | Construire, lancer et tester l'image exacte avant son push; publier seulement après succès |
| Healthcheck Worker limité à `/health` | Il ne prouve ni imports, ni CUDA, ni authentification `/ready` | Un conteneur sain peut être inutilisable pour les modèles | Compose valide la vie du processus, pas le runtime | Ajouter un contrat conteneur authentifié pour `/ready`, imports, ffmpeg et montages |
| `gpu-acceptance.sh` sélectionne plusieurs modèles/capacités | La validation finale peut multiplier téléchargements et inférences | Gaspillage de la session L40S et résultats non reproductibles | Le script privilégie une matrice large plutôt que les trois preuves requises | Une seule session finale, un modèle explicite, ordre T2V 480p → I2V 480p → T2V 720p, diagnostics automatiques |

## Frontières déjà correctes à conserver

- Le frontend envoie déjà `quality`, `aspect_ratio`, `duration_seconds` et `fps` pour la vidéo.
- `ResolutionResolver` sait viser 480p/720p en respectant ratio, multiples et bornes.
- `ModelProfile` sait normaliser les frames selon une contrainte temporelle.
- `OutputNormalizer` encode via `libx264` et contrôle déjà codec, dimensions, durée et nombre de frames.
- Le backend résout normalement les assets persistants vers des chemins temporaires avant l'appel Worker.
- Le compose production monte le Scratch persistant sur les répertoires modèles/cache/work attendus.

Ces éléments ne constituent toutefois pas une preuve de bout en bout tant qu'ils ne sont pas traversés par les nouveaux contrats.

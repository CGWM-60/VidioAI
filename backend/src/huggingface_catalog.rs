//! Catalogue public alimenté par Hugging Face Hub.
//!
//! Ce module est volontairement indépendant d'Axum : il connaît l'API distante,
//! sa normalisation et son cache, mais pas les routes HTTP de VidioAI. Le token
//! `HF_TOKEN` reste ainsi confiné au backend et n'apparaît jamais dans un DTO.

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{fs, sync::RwLock};

use crate::hardware_estimator::{
    HardwareEstimate, HardwareEstimator, HardwareFile, HardwareMetadata, SafetensorsMetadata,
};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_TTL_SECONDS: u64 = 15 * 60;
const MAX_PAGE_SIZE: usize = 60;
const MAX_REMOTE_RESULTS: usize = 240;
const GIB: u64 = 1024 * 1024 * 1024;
// Toute évolution du contrat normalisé invalide le cache disque précédent. Cela
// évite notamment qu'un ancien booléen `runtime_supported` masque la raison
// détaillée calculée par la nouvelle matrice de pipelines.
// Toute modification d'un champ calculé (comme `runtime_reason`) doit invalider
// les anciennes entrées, sinon l'interface continuerait à afficher un diagnostic
// obsolète après la mise à jour du binaire.
const CACHE_SCHEMA_VERSION: u32 = 8;

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Capacités stables exposées par VidioAI, jamais les tags bruts du Hub.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelCapability {
    Chat,
    Vision,
    TextToImage,
    ImageToImage,
    Inpainting,
    Outpainting,
    ImageVariation,
    ImageUpscale,
    ControlledImageGeneration,
    TextToVideo,
    ImageToVideo,
    MultiImageToVideo,
    StartEndImageToVideo,
    KeyframesToVideo,
    VideoToVideo,
    VideoInpainting,
    VideoUpscale,
    Audio,
    TextToSpeech,
    SpeechToText,
    CapabilityUnknown,
}

impl ModelCapability {
    /// Retourne exactement le libellé public utilisé par l'API JSON.
    ///
    /// Il ne faut pas dériver ce texte avec `Debug`: `TextToVideo` deviendrait
    /// alors `TEXTTOVIDEO`, tandis que le contrat Serde expose
    /// `TEXT_TO_VIDEO`. Centraliser ce mapping garde donc les explications
    /// humaines cohérentes avec les valeurs réellement consommées par le
    /// frontend.
    pub(crate) fn api_name(&self) -> &'static str {
        match self {
            Self::Chat => "CHAT",
            Self::Vision => "VISION",
            Self::TextToImage => "TEXT_TO_IMAGE",
            Self::ImageToImage => "IMAGE_TO_IMAGE",
            Self::Inpainting => "INPAINTING",
            Self::Outpainting => "OUTPAINTING",
            Self::ImageVariation => "IMAGE_VARIATION",
            Self::ImageUpscale => "IMAGE_UPSCALE",
            Self::ControlledImageGeneration => "CONTROLLED_IMAGE_GENERATION",
            Self::TextToVideo => "TEXT_TO_VIDEO",
            Self::ImageToVideo => "IMAGE_TO_VIDEO",
            Self::MultiImageToVideo => "MULTI_IMAGE_TO_VIDEO",
            Self::StartEndImageToVideo => "START_END_IMAGE_TO_VIDEO",
            Self::KeyframesToVideo => "KEYFRAMES_TO_VIDEO",
            Self::VideoToVideo => "VIDEO_TO_VIDEO",
            Self::VideoInpainting => "VIDEO_INPAINTING",
            Self::VideoUpscale => "VIDEO_UPSCALE",
            Self::Audio => "AUDIO",
            Self::TextToSpeech => "TEXT_TO_SPEECH",
            Self::SpeechToText => "SPEECH_TO_TEXT",
            Self::CapabilityUnknown => "CAPABILITY_UNKNOWN",
        }
    }
}

/// Famille d'interface. Les capacités détaillées restent la source d'autorité.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelKind {
    Chat,
    Image,
    Video,
    Vision,
    Audio,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVariant {
    pub id: String,
    pub label: String,
    pub ram_required: u64,
    pub vram_required: u64,
    pub download_bytes: u64,
    /// Les exigences distantes sont des ordres de grandeur, jamais une promesse.
    pub estimated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryFile {
    pub path: String,
    pub size: Option<u64>,
    pub lfs_sha256: Option<String>,
}

/// Modèle normalisé commun aux résultats de recherche, détails et installations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub storage_id: String,
    pub name: String,
    pub description: String,
    pub kind: ModelKind,
    pub capabilities: Vec<ModelCapability>,
    pub variants: Vec<ModelVariant>,
    pub repository: String,
    pub author: Option<String>,
    pub revision: String,
    pub last_modified: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub library: Option<String>,
    pub architecture: Option<String>,
    /// Configuration structurée retournée par l'API Hugging Face. Elle n'est
    /// jamais importée ni exécutée et sert uniquement aux comparaisons du Lab.
    #[serde(default)]
    pub config: Value,
    pub license: Option<String>,
    pub files: Vec<RepositoryFile>,
    pub estimated_size_bytes: Option<u64>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub trending_score: Option<f64>,
    pub gated: bool,
    pub private: bool,
    pub disabled: bool,
    pub accessibility: String,
    /// `true` lorsque les fichiers de configuration du repository ont été lus
    /// avec succès. Pour un repo gated/privé, cela prouve que HF_TOKEN possède
    /// l'autorisation nécessaire sans jamais exposer le token.
    #[serde(default)]
    pub access_authorized: bool,
    /// Distingue un refus réel d'un résultat de liste qui n'a pas encore testé
    /// l'accès aux fichiers. La vérification définitive a lieu sur la fiche
    /// exacte avant toute installation.
    #[serde(default)]
    pub access_checked: bool,
    pub source_available: bool,
    pub quality_valid: bool,
    pub runtime_name: Option<String>,
    pub runtime_supported: bool,
    /// Explication destinée à l'API et à l'interface. Le matériel et le runtime
    /// sont volontairement deux axes différents : un modèle vidéo peut tenir
    /// dans la VRAM tout en n'ayant aucun exécuteur VidioAI implémenté.
    #[serde(default)]
    pub runtime_reason: String,
    /// Classe Diffusers détectée dans `model_index.json`/la configuration HF.
    /// Elle reste informative : le Worker relit et valide le manifest téléchargé.
    #[serde(default)]
    pub pipeline_class: Option<String>,
    /// Capacités que le Worker sait réellement exécuter pour ce repository.
    /// Ce champ ne recopie jamais aveuglément les tags Hugging Face.
    #[serde(default)]
    pub runtime_capabilities: Vec<ModelCapability>,
    pub installable: bool,
    pub local: bool,
    /// Estimation sans la machine courante. `platform::model_view` y applique
    /// ensuite le benchmark local et le profil retourné par `/api/system`.
    #[serde(default)]
    pub hardware: HardwareEstimate,
}

/// Paramètres que le handler peut convertir directement depuis la query string.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogQuery {
    pub search: Option<String>,
    pub task: Option<String>,
    pub category: Option<String>,
    pub author: Option<String>,
    pub sort: Option<String>,
    pub page: Option<usize>,
    pub limit: Option<usize>,
    pub installed: Option<bool>,
    pub compatible: Option<bool>,
}

impl CatalogQuery {
    pub fn page(&self) -> usize {
        self.page.unwrap_or(1).max(1)
    }

    pub fn limit(&self) -> usize {
        self.limit.unwrap_or(30).clamp(1, MAX_PAGE_SIZE)
    }

    fn normalized_search(&self) -> Option<String> {
        self.search
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(normalize_repository_reference)
    }

    fn cache_key(&self) -> String {
        // Le nombre demandé fait partie de la clé : une page profonde ne doit
        // pas réutiliser un cache plus court produit pour la première page.
        let source = format!(
            "{}|{}|{}|{}|{}|{}",
            self.normalized_search().unwrap_or_default(),
            self.task
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase(),
            self.category
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase(),
            self.author
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase(),
            self.sort
                .as_deref()
                .unwrap_or("trending")
                .to_ascii_lowercase(),
            self.page().saturating_mul(self.limit()).saturating_add(1),
        );
        format!("{:x}", Sha256::digest(source.as_bytes()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResult {
    saved_at: u64,
    #[serde(default)]
    invalidated: bool,
    models: Vec<CatalogModel>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistentCache {
    #[serde(default)]
    schema_version: u32,
    results: HashMap<String, CachedResult>,
}

impl Default for PersistentCache {
    fn default() -> Self {
        Self {
            schema_version: CACHE_SCHEMA_VERSION,
            results: HashMap::new(),
        }
    }
}

/// Résultat du service avant enrichissement avec l'état local et le hardware.
#[derive(Debug, Clone)]
pub struct CatalogResult {
    pub models: Vec<CatalogModel>,
    pub stale: bool,
    pub last_sync: Option<u64>,
}

pub struct HuggingFaceCatalogService {
    endpoint: String,
    token: Option<String>,
    ttl_seconds: u64,
    cache_path: PathBuf,
    http: reqwest::Client,
    cache: RwLock<PersistentCache>,
}

impl HuggingFaceCatalogService {
    /// Charge le cache disque au démarrage. Un JSON corrompu est ignoré : le Hub
    /// pourra le reconstruire sans empêcher VidioAI de démarrer.
    pub async fn initialize(cache_dir: &Path) -> Self {
        let cache_path = cache_dir.join("huggingface-catalog.json");
        let cache = fs::read(&cache_path)
            .await
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .filter(|cache: &PersistentCache| cache.schema_version == CACHE_SCHEMA_VERSION)
            .unwrap_or_default();
        let timeout = std::env::var("HF_CATALOG_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        let ttl_seconds = std::env::var("HF_CATALOG_TTL_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_TTL_SECONDS);
        let endpoint = std::env::var("HF_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_owned())
            .trim_end_matches('/')
            .to_owned();
        let token = std::env::var("HF_TOKEN")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(timeout))
            .user_agent(format!("VidioAI/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .unwrap_or_default();
        Self {
            endpoint,
            token,
            ttl_seconds,
            cache_path,
            http,
            cache: RwLock::new(cache),
        }
    }

    fn request(&self, url: String) -> reqwest::RequestBuilder {
        let request = self.http.get(url);
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn persist_cache(&self) {
        let bytes = {
            let cache = self.cache.read().await;
            serde_json::to_vec_pretty(&*cache).ok()
        };
        let Some(bytes) = bytes else { return };
        if let Some(parent) = self.cache_path.parent() {
            let _ = fs::create_dir_all(parent).await;
        }
        let temporary = self.cache_path.with_extension("json.tmp");
        if fs::write(&temporary, bytes).await.is_ok() {
            let _ = fs::rename(temporary, &self.cache_path).await;
        }
    }

    /// Invalide le cache sans effacer sa dernière valeur. La prochaine lecture
    /// tentera le Hub, puis pourra toujours servir cette copie en mode dégradé.
    pub async fn clear_cache(&self) {
        for result in self.cache.write().await.results.values_mut() {
            result.invalidated = true;
        }
        self.persist_cache().await;
    }

    pub async fn search(&self, query: &CatalogQuery, force: bool) -> Result<CatalogResult, String> {
        let key = query.cache_key();
        let cached = self.cache.read().await.results.get(&key).cloned();
        if !force
            && let Some(value) = &cached
            && !value.invalidated
        {
            let fresh = unix_now().saturating_sub(value.saved_at) <= self.ttl_seconds;
            // Une valeur expirée reste immédiatement utilisable. La mise à jour
            // explicite invalide la clé puis effectue l'appel HF ; une simple
            // navigation ne doit jamais se bloquer sur le réseau distant.
            return Ok(CatalogResult {
                models: value.models.clone(),
                stale: !fresh,
                last_sync: Some(value.saved_at),
            });
        }

        match self.fetch_search(query).await {
            Ok(models) => {
                let saved_at = unix_now();
                self.cache.write().await.results.insert(
                    key,
                    CachedResult {
                        saved_at,
                        invalidated: false,
                        models: models.clone(),
                    },
                );
                self.persist_cache().await;
                Ok(CatalogResult {
                    models,
                    stale: false,
                    last_sync: Some(saved_at),
                })
            }
            Err(error) => cached.map_or(Err(error), |value| {
                Ok(CatalogResult {
                    models: value.models,
                    stale: true,
                    last_sync: Some(value.saved_at),
                })
            }),
        }
    }

    /// Récupère un repository précis, y compris lorsqu'il n'est jamais apparu
    /// dans une recherche VidioAI. Le cache utilise une clé dédiée au détail.
    pub async fn model(&self, reference: &str, force: bool) -> Result<CatalogResult, String> {
        let repository = normalize_repository_reference(reference);
        validate_repository_id(&repository)?;
        let query = CatalogQuery {
            search: Some(repository.clone()),
            page: Some(1),
            limit: Some(1),
            ..CatalogQuery::default()
        };
        let key = format!("detail-{}", query.cache_key());
        let cached = self.cache.read().await.results.get(&key).cloned();
        if !force
            && let Some(value) = &cached
            && !value.invalidated
            && unix_now().saturating_sub(value.saved_at) <= self.ttl_seconds
        {
            return Ok(CatalogResult {
                models: value.models.clone(),
                stale: false,
                last_sync: Some(value.saved_at),
            });
        }

        match self.fetch_model(&repository).await {
            Ok(model) => {
                let saved_at = unix_now();
                let models = vec![model];
                self.cache.write().await.results.insert(
                    key,
                    CachedResult {
                        saved_at,
                        invalidated: false,
                        models: models.clone(),
                    },
                );
                self.persist_cache().await;
                Ok(CatalogResult {
                    models,
                    stale: false,
                    last_sync: Some(saved_at),
                })
            }
            Err(error) => cached.map_or(Err(error), |value| {
                Ok(CatalogResult {
                    models: value.models,
                    stale: true,
                    last_sync: Some(value.saved_at),
                })
            }),
        }
    }

    async fn fetch_search(&self, query: &CatalogQuery) -> Result<Vec<CatalogModel>, String> {
        if let Some(search) = query.normalized_search()
            && validate_repository_id(&search).is_ok()
        {
            return self.fetch_model(&search).await.map(|model| vec![model]);
        }

        let requested = query
            .page()
            .saturating_mul(query.limit())
            // La validation retire les repositories incomplets ; demander une
            // fenêtre plus large permet de remplir la page après normalisation.
            .saturating_mul(3)
            .saturating_add(1)
            .min(MAX_REMOTE_RESULTS);
        let pipelines = requested_pipelines(query);
        let mut raw_models = Vec::new();
        if pipelines.is_empty() {
            raw_models.extend(self.fetch_list(query, None, requested).await?);
        } else {
            // Une catégorie VidioAI peut agréger plusieurs pipeline_tag HF. Les
            // recherches sont indépendantes : les exécuter en série multipliait
            // le timeout par trois pour VIDEO et AUDIO.
            let searches = pipelines
                .into_iter()
                .map(|pipeline| self.fetch_list(query, Some(pipeline), requested));
            for models in futures_util::future::join_all(searches).await {
                raw_models.extend(models?);
            }
        }

        let mut seen = HashSet::new();
        let mut models: Vec<_> = raw_models
            .into_iter()
            .map(normalize_model)
            .filter(|model| seen.insert(model.id.clone()))
            .filter(|model| model.quality_valid || model.runtime_supported)
            .filter(|model| {
                !matches!(
                    model.capabilities.as_slice(),
                    [ModelCapability::CapabilityUnknown]
                )
            })
            .collect();
        sort_models(&mut models, query.sort.as_deref().unwrap_or("trending"));
        Ok(models)
    }

    async fn fetch_list(
        &self,
        query: &CatalogQuery,
        pipeline: Option<&str>,
        limit: usize,
    ) -> Result<Vec<HfRawModel>, String> {
        let request = self
            .request(format!("{}/api/models", self.endpoint))
            .query(&[
                ("limit", limit.to_string()),
                ("full", "true".into()),
                ("config", "true".into()),
            ]);
        // `full=true` n'inclut pas le résumé Safetensors dans les listes. Le Hub
        // n'accepte actuellement qu'une propriété `expand` à la fois : un appel
        // parallèle très léger récupère donc uniquement les paramètres/dtypes.
        let safetensors_request = self
            .request(format!("{}/api/models", self.endpoint))
            .query(&[
                ("limit", limit.to_string()),
                ("expand", "safetensors".into()),
            ]);
        let request = apply_list_filters(request, query, pipeline);
        let safetensors_request = apply_list_filters(safetensors_request, query, pipeline);
        let (response, safetensors_response) =
            tokio::join!(request.send(), safetensors_request.send());
        let response = response.map_err(clean_network_error)?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("Hugging Face a répondu HTTP {status}."));
        }
        let bytes = response.bytes().await.map_err(clean_network_error)?;
        let mut models: Vec<HfRawModel> = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Réponse JSON Hugging Face invalide : {error}"))?;
        if let Ok(response) = safetensors_response
            && response.status().is_success()
            && let Ok(projections) = response.json::<Vec<HfSafetensorsProjection>>().await
        {
            let by_id: HashMap<_, _> = projections
                .into_iter()
                .filter_map(|projection| projection.safetensors.map(|value| (projection.id, value)))
                .collect();
            for model in &mut models {
                if let Some(value) = by_id.get(&model.id) {
                    model.safetensors = Some(value.clone());
                }
            }
        }
        Ok(models)
    }

    async fn fetch_model(&self, repository: &str) -> Result<CatalogModel, String> {
        let response = self
            .request(format!("{}/api/models/{repository}", self.endpoint))
            .query(&[("blobs", "true"), ("securityStatus", "true")])
            .send()
            .await
            .map_err(clean_network_error)?;
        match response.status() {
            status if status.is_success() => {
                let bytes = response.bytes().await.map_err(clean_network_error)?;
                let mut raw = serde_json::from_slice::<HfRawModel>(&bytes)
                    .map_err(|error| format!("Réponse JSON Hugging Face invalide : {error}"))?;
                // L'objet `config` de l'API Hub est volontairement résumé. La
                // fiche détaillée complète donc les configurations légères sans
                // télécharger les poids.
                raw.access_authorized = self.enrich_configuration(repository, &mut raw).await;
                raw.access_checked = true;
                Ok(normalize_model(raw))
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Ok(access_required_placeholder(repository))
            }
            StatusCode::NOT_FOUND => {
                Err(format!("Repository Hugging Face {repository} introuvable."))
            }
            status => Err(format!("Hugging Face a répondu HTTP {status}.")),
        }
    }

    /// Télécharge uniquement les petits JSON utiles à l'estimation. Le nombre
    /// est borné et les échecs sont tolérés : le service retombe alors sur les
    /// métadonnées Safetensors et les tailles LFS déjà reçues.
    async fn enrich_configuration(&self, repository: &str, raw: &mut HfRawModel) -> bool {
        let revision = raw.sha.as_deref().unwrap_or("main");
        let interesting = raw
            .siblings
            .iter()
            .map(|file| file.rfilename.as_str())
            .filter(|path| {
                *path == "config.json"
                    || *path == "model_index.json"
                    || *path == "modular_model_index.json"
                    || ([
                        "transformer/",
                        "unet/",
                        "vae/",
                        "text_encoder/",
                        "text_encoder_2/",
                        "text_encoder_3/",
                        "image_encoder/",
                        "controlnet/",
                    ]
                    .iter()
                    .any(|prefix| path.starts_with(prefix))
                        && path.ends_with("config.json"))
            })
            .filter(|path| !path.contains("..") && !path.contains(['?', '#', '\\']))
            .take(12)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if interesting.is_empty() {
            return false;
        }

        let requests = interesting.iter().map(|path| async move {
            let response = self
                .request(format!(
                    "{}/{}/resolve/{}/{}",
                    self.endpoint, repository, revision, path
                ))
                .send()
                .await
                .ok()?;
            if !response.status().is_success() {
                return None;
            }
            let value = response.json::<Value>().await.ok()?;
            Some((path.clone(), value))
        });
        let mut merged = raw.config.as_object().cloned().unwrap_or_default();
        let mut components = serde_json::Map::new();
        let mut fetched_configuration = false;
        for result in futures_util::future::join_all(requests)
            .await
            .into_iter()
            .flatten()
        {
            fetched_configuration = true;
            match result.0.as_str() {
                "config.json" => {
                    merged.insert("_root_config".into(), result.1);
                }
                "model_index.json" => {
                    merged.insert("_model_index".into(), result.1);
                }
                "modular_model_index.json" => {
                    merged.insert("_modular_model_index".into(), result.1);
                }
                _ => {
                    components.insert(result.0, result.1);
                }
            }
        }
        if !components.is_empty() {
            merged.insert("_component_configs".into(), Value::Object(components));
        }
        raw.config = Value::Object(merged);
        fetched_configuration
    }
}

#[derive(Debug, Clone, Deserialize)]
struct HfRawModel {
    id: String,
    author: Option<String>,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    gated: Value,
    #[serde(skip)]
    access_authorized: bool,
    #[serde(skip)]
    access_checked: bool,
    #[serde(default)]
    disabled: bool,
    sha: Option<String>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    #[serde(rename = "trendingScore")]
    trending_score: Option<f64>,
    #[serde(rename = "usedStorage")]
    used_storage: Option<u64>,
    #[serde(default)]
    siblings: Vec<HfRawSibling>,
    #[serde(default)]
    config: Value,
    #[serde(rename = "cardData", default)]
    card_data: Value,
    /// Le Hub expose le nombre de paramètres ventilé par dtype lorsqu'il a pu
    /// analyser les fichiers Safetensors du repository.
    safetensors: Option<HfRawSafetensors>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HfRawSafetensors {
    #[serde(default)]
    parameters: BTreeMap<String, u64>,
    total: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct HfSafetensorsProjection {
    id: String,
    safetensors: Option<HfRawSafetensors>,
}

fn apply_list_filters(
    mut request: reqwest::RequestBuilder,
    query: &CatalogQuery,
    pipeline: Option<&str>,
) -> reqwest::RequestBuilder {
    if let Some(search) = query.normalized_search() {
        request = request.query(&[("search", search)]);
    }
    if let Some(author) = query
        .author
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        request = request.query(&[("author", author)]);
    }
    if let Some(pipeline) = pipeline {
        request = request.query(&[("pipeline_tag", pipeline)]);
    }
    if let Some(sort) = hf_sort(query.sort.as_deref()) {
        request = request.query(&[("sort", sort), ("direction", "-1")]);
    }
    request
}

#[derive(Debug, Clone, Deserialize)]
struct HfRawSibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<HfRawLfs>,
}

#[derive(Debug, Clone, Deserialize)]
struct HfRawLfs {
    size: Option<u64>,
    sha256: Option<String>,
}

fn clean_network_error(error: impl std::fmt::Display) -> String {
    // Le message ne contient volontairement ni URL complète signée ni headers.
    format!("Catalogue Hugging Face indisponible : {error}")
}

fn normalize_repository_reference(value: &str) -> String {
    value
        .trim()
        .trim_end_matches('/')
        .strip_prefix("https://huggingface.co/")
        .or_else(|| {
            value
                .trim()
                .trim_end_matches('/')
                .strip_prefix("http://huggingface.co/")
        })
        .unwrap_or(value.trim().trim_end_matches('/'))
        .split(['?', '#'])
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn validate_repository_id(value: &str) -> Result<(), String> {
    let parts: Vec<_> = value.split('/').collect();
    let valid_part = |part: &str| {
        !part.is_empty()
            && part.len() <= 128
            && part
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character))
    };
    if parts.len() == 2 && parts.iter().all(|part| valid_part(part)) {
        Ok(())
    } else {
        Err("Le modelId Hugging Face doit respecter organisation/modèle.".into())
    }
}

/// Identifiant de dossier et de worker sans séparateur de chemin.
pub fn storage_id(repository: &str) -> String {
    repository
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || "-_ .".contains(character) {
                if character == ' ' { '-' } else { character }
            } else {
                '-'
            }
        })
        .collect::<String>()
        .replace("--", "-")
}

fn normalize_model(raw: HfRawModel) -> CatalogModel {
    let gated = !raw.gated.is_null() && raw.gated != Value::Bool(false);
    let files: Vec<_> = raw
        .siblings
        .into_iter()
        .map(|file| RepositoryFile {
            path: file.rfilename,
            size: file
                .size
                .or_else(|| file.lfs.as_ref().and_then(|lfs| lfs.size)),
            lfs_sha256: file.lfs.and_then(|lfs| lfs.sha256),
        })
        .collect();
    let pipeline = raw.pipeline_tag.as_deref().unwrap_or_default();
    let capabilities = normalize_capabilities(pipeline, &raw.tags);
    let kind = primary_kind(&capabilities);
    let has_weights = files.iter().any(|file| is_weight_file(&file.path));
    let has_modular_index = files
        .iter()
        .any(|file| file.path.rsplit('/').next() == Some("modular_model_index.json"));
    let has_config = files.iter().any(|file| {
        matches!(
            file.path.rsplit('/').next(),
            Some("config.json" | "model_index.json" | "modular_model_index.json")
        )
    });
    // Un repository ModularPipeline peut être un pur manifeste : les poids
    // référencés sont matérialisés par le Worker pendant l'installation.
    let quality_valid =
        !raw.disabled && !files.is_empty() && has_config && (has_weights || has_modular_index);
    let architecture = architecture_from_config(&raw.config);
    let license = license_from(&raw.card_data, &raw.tags);
    let estimated_size_bytes = raw.used_storage.or_else(|| {
        let known: Vec<_> = files.iter().filter_map(|file| file.size).collect();
        (!known.is_empty()).then(|| known.into_iter().sum())
    });
    let (runtime_name, runtime_supported, runtime_reason, runtime_capabilities) = runtime_match(
        raw.library_name.as_deref(),
        &capabilities,
        &files,
        architecture.as_deref(),
    );
    let access_authorized = (!gated && !raw.private) || raw.access_authorized;
    let access_checked = (!gated && !raw.private) || raw.access_checked;
    let accessibility = if access_authorized && (raw.private || gated) {
        "AUTHORIZED"
    } else if !access_checked && (raw.private || gated) {
        "UNVERIFIED"
    } else if raw.private {
        "PRIVATE"
    } else if gated {
        "ACCESS_REQUIRED"
    } else {
        "PUBLIC"
    }
    .to_owned();
    let installable = runtime_supported && quality_valid && (access_authorized || !access_checked);
    let hardware = HardwareEstimator::estimate(&HardwareMetadata {
        pipeline_tag: raw.pipeline_tag.clone(),
        library_name: raw.library_name.clone(),
        tags: raw.tags.clone(),
        architecture: architecture.clone(),
        config: raw.config.clone(),
        card_data: raw.card_data.clone(),
        safetensors: raw.safetensors.as_ref().map(|value| SafetensorsMetadata {
            parameters: value.parameters.clone(),
            total: value.total,
        }),
        files: files
            .iter()
            .map(|file| HardwareFile {
                path: file.path.clone(),
                size: file.size,
            })
            .collect(),
        repository_size: estimated_size_bytes,
    });
    let variants = estimated_variant(estimated_size_bytes, &hardware);
    let name = raw.id.rsplit('/').next().unwrap_or(&raw.id).to_owned();
    let description = raw
        .card_data
        .get("description")
        .or_else(|| raw.card_data.get("summary"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            format!(
                "Modèle Hugging Face{}{}.",
                raw.pipeline_tag
                    .as_deref()
                    .map(|tag| format!(" pour {tag}"))
                    .unwrap_or_default(),
                raw.library_name
                    .as_deref()
                    .map(|library| format!(" · bibliothèque {library}"))
                    .unwrap_or_default()
            )
        });
    CatalogModel {
        storage_id: storage_id(&raw.id),
        id: raw.id.clone(),
        name,
        description,
        kind,
        capabilities,
        variants,
        repository: raw.id,
        author: raw.author,
        revision: raw.sha.unwrap_or_else(|| "main".into()),
        last_modified: raw.last_modified,
        pipeline_tag: raw.pipeline_tag,
        tags: raw.tags,
        library: raw.library_name,
        architecture: architecture.clone(),
        config: raw.config,
        license,
        files,
        estimated_size_bytes,
        downloads: raw.downloads,
        likes: raw.likes,
        trending_score: raw.trending_score,
        gated,
        private: raw.private,
        disabled: raw.disabled,
        accessibility,
        access_authorized,
        access_checked,
        source_available: true,
        quality_valid,
        runtime_name,
        runtime_supported,
        runtime_reason,
        pipeline_class: architecture.clone(),
        runtime_capabilities,
        installable,
        local: false,
        hardware,
    }
}

fn access_required_placeholder(repository: &str) -> CatalogModel {
    CatalogModel {
        id: repository.into(),
        storage_id: storage_id(repository),
        name: repository.rsplit('/').next().unwrap_or(repository).into(),
        description: "Les métadonnées de ce repository nécessitent une autorisation Hugging Face."
            .into(),
        kind: ModelKind::Unknown,
        capabilities: vec![ModelCapability::CapabilityUnknown],
        variants: Vec::new(),
        repository: repository.into(),
        author: repository.split('/').next().map(str::to_owned),
        revision: "main".into(),
        last_modified: None,
        pipeline_tag: None,
        tags: Vec::new(),
        library: None,
        architecture: None,
        config: Value::Object(Default::default()),
        license: None,
        files: Vec::new(),
        estimated_size_bytes: None,
        downloads: None,
        likes: None,
        trending_score: None,
        gated: true,
        private: false,
        disabled: false,
        accessibility: "ACCESS_REQUIRED".into(),
        access_authorized: false,
        access_checked: true,
        source_available: true,
        quality_valid: false,
        runtime_name: None,
        runtime_supported: false,
        runtime_reason:
            "Métadonnées inaccessibles : fournissez un HF_TOKEN autorisé pour analyser le pipeline."
                .into(),
        pipeline_class: None,
        runtime_capabilities: Vec::new(),
        installable: false,
        local: false,
        hardware: HardwareEstimate::default(),
    }
}

fn normalize_capabilities(pipeline: &str, tags: &[String]) -> Vec<ModelCapability> {
    let mut values: HashSet<_> = std::iter::once(pipeline)
        .chain(tags.iter().map(String::as_str))
        .filter_map(|value| match value.to_ascii_lowercase().as_str() {
            "text-generation" | "text2text-generation" | "conversational" => {
                Some(ModelCapability::Chat)
            }
            "image-text-to-text"
            | "visual-question-answering"
            | "image-to-text"
            | "document-question-answering" => Some(ModelCapability::Vision),
            "text-to-image" => Some(ModelCapability::TextToImage),
            "image-to-image" => Some(ModelCapability::ImageToImage),
            "inpainting" | "image-inpainting" => Some(ModelCapability::Inpainting),
            "outpainting" | "image-outpainting" => Some(ModelCapability::Outpainting),
            "image-variation" | "variation" => Some(ModelCapability::ImageVariation),
            "super-resolution" | "upscale" | "image-upscale" => Some(ModelCapability::ImageUpscale),
            "controlnet" | "controlled-image-generation" => {
                Some(ModelCapability::ControlledImageGeneration)
            }
            "text-to-video" => Some(ModelCapability::TextToVideo),
            "image-to-video" => Some(ModelCapability::ImageToVideo),
            "multi-image-to-video" => Some(ModelCapability::MultiImageToVideo),
            "start-end-image-to-video" => Some(ModelCapability::StartEndImageToVideo),
            "keyframes-to-video" => Some(ModelCapability::KeyframesToVideo),
            "video-to-video" => Some(ModelCapability::VideoToVideo),
            "video-inpainting" => Some(ModelCapability::VideoInpainting),
            "video-upscale" | "video-super-resolution" => Some(ModelCapability::VideoUpscale),
            "text-to-speech" | "text-to-audio" => Some(ModelCapability::TextToSpeech),
            "automatic-speech-recognition" | "speech-to-text" => {
                Some(ModelCapability::SpeechToText)
            }
            "audio-to-audio" | "audio-classification" => Some(ModelCapability::Audio),
            _ => None,
        })
        .collect();
    if values.contains(&ModelCapability::TextToSpeech)
        || values.contains(&ModelCapability::SpeechToText)
    {
        values.insert(ModelCapability::Audio);
    }
    if values.contains(&ModelCapability::Vision) {
        values.insert(ModelCapability::Chat);
    }
    if values.is_empty() {
        values.insert(ModelCapability::CapabilityUnknown);
    }
    let order = [
        ModelCapability::Chat,
        ModelCapability::Vision,
        ModelCapability::TextToImage,
        ModelCapability::ImageToImage,
        ModelCapability::Inpainting,
        ModelCapability::Outpainting,
        ModelCapability::ImageVariation,
        ModelCapability::ImageUpscale,
        ModelCapability::ControlledImageGeneration,
        ModelCapability::TextToVideo,
        ModelCapability::ImageToVideo,
        ModelCapability::MultiImageToVideo,
        ModelCapability::StartEndImageToVideo,
        ModelCapability::KeyframesToVideo,
        ModelCapability::VideoToVideo,
        ModelCapability::VideoInpainting,
        ModelCapability::VideoUpscale,
        ModelCapability::Audio,
        ModelCapability::TextToSpeech,
        ModelCapability::SpeechToText,
        ModelCapability::CapabilityUnknown,
    ];
    order
        .into_iter()
        .filter(|capability| values.contains(capability))
        .collect()
}

fn primary_kind(capabilities: &[ModelCapability]) -> ModelKind {
    if capabilities.iter().any(|capability| {
        matches!(
            capability,
            ModelCapability::TextToVideo
                | ModelCapability::ImageToVideo
                | ModelCapability::MultiImageToVideo
                | ModelCapability::StartEndImageToVideo
                | ModelCapability::KeyframesToVideo
                | ModelCapability::VideoToVideo
                | ModelCapability::VideoInpainting
                | ModelCapability::VideoUpscale
        )
    }) {
        ModelKind::Video
    } else if capabilities.iter().any(|capability| {
        matches!(
            capability,
            ModelCapability::TextToImage
                | ModelCapability::ImageToImage
                | ModelCapability::Inpainting
                | ModelCapability::Outpainting
                | ModelCapability::ImageVariation
                | ModelCapability::ImageUpscale
                | ModelCapability::ControlledImageGeneration
        )
    }) {
        ModelKind::Image
    } else if capabilities.contains(&ModelCapability::Vision) {
        ModelKind::Vision
    } else if capabilities.iter().any(|capability| {
        matches!(
            capability,
            ModelCapability::Audio | ModelCapability::TextToSpeech | ModelCapability::SpeechToText
        )
    }) {
        ModelKind::Audio
    } else if capabilities.contains(&ModelCapability::Chat) {
        ModelKind::Chat
    } else {
        ModelKind::Unknown
    }
}

fn architecture_from_config(config: &Value) -> Option<String> {
    config
        .get("_modular_model_index")
        .and_then(|value| value.get("_class_name"))
        .and_then(Value::as_str)
        .or_else(|| {
            config
                .get("_model_index")
                .and_then(|value| value.get("_class_name"))
                .and_then(Value::as_str)
        })
        .or_else(|| config.get("_class_name").and_then(Value::as_str))
        .or_else(|| {
            config
                .get("diffusers")
                .and_then(|value| value.get("_class_name"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            config
                .get("_root_config")
                .and_then(|value| value.get("_class_name"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            config
                .get("_root_config")
                .and_then(|value| value.get("architectures"))
                .and_then(Value::as_array)
                .and_then(|values| values.first())
                .and_then(Value::as_str)
        })
        .or_else(|| {
            config
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|values| values.first())
                .and_then(Value::as_str)
        })
        .map(str::to_owned)
}

fn license_from(card_data: &Value, tags: &[String]) -> Option<String> {
    card_data
        .get("license")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            tags.iter()
                .find_map(|tag| tag.strip_prefix("license:").map(str::to_owned))
        })
}

fn is_weight_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [".safetensors", ".bin", ".gguf", ".pt", ".pth"]
        .iter()
        .any(|extension| lower.ends_with(extension))
}

fn runtime_match(
    library: Option<&str>,
    capabilities: &[ModelCapability],
    files: &[RepositoryFile],
    architecture: Option<&str>,
) -> (Option<String>, bool, String, Vec<ModelCapability>) {
    let library = library.unwrap_or_default().to_ascii_lowercase();
    let has_safetensors = files
        .iter()
        .any(|file| file.path.to_ascii_lowercase().ends_with(".safetensors"));
    let has_diffusers_index = files
        .iter()
        .any(|file| file.path.rsplit('/').next() == Some("model_index.json"));
    let has_modular_index = files
        .iter()
        .any(|file| file.path.rsplit('/').next() == Some("modular_model_index.json"));

    if library == "diffusers"
        && has_modular_index
        && architecture.is_some_and(|value| value.ends_with("Pipeline"))
    {
        let runtime_capabilities = infer_runtime_capabilities(capabilities, architecture, files);
        let detected = architecture.expect("architecture vérifiée ci-dessus");
        return (
            Some("Diffusers ModularPipeline".into()),
            true,
            format!(
                "Manifest modular_model_index.json détecté ({detected}); le Worker validera ModularPipeline, ses blocks et ses composants au chargement."
            ),
            runtime_capabilities,
        );
    }

    if library == "diffusers"
        && has_safetensors
        && has_diffusers_index
        && architecture.is_some_and(|value| value.ends_with("Pipeline"))
    {
        let runtime_capabilities = infer_runtime_capabilities(capabilities, architecture, files);
        let detected = architecture.expect("architecture vérifiée ci-dessus");
        let labels = runtime_capabilities
            .iter()
            .map(ModelCapability::api_name)
            .collect::<Vec<_>>()
            .join(", ");
        return (
            Some("Diffusers PipelineResolver".into()),
            true,
            format!(
                "Pipeline Diffusers déclarée ({detected}); la présence de la classe et sa signature seront vérifiées par le Worker avant téléchargement{}.",
                if labels.is_empty() {
                    String::new()
                } else {
                    format!(" : {labels}")
                }
            ),
            runtime_capabilities,
        );
    }
    if capabilities.contains(&ModelCapability::Chat)
        && library == "transformers"
        && architecture.is_some_and(|value| value.contains("CausalLM"))
    {
        // Le moteur existe, mais son installateur révisionné n'est pas encore
        // relié au Model Runtime Manager. Il ne doit donc pas être installable.
        return (
            Some("mistral.rs (détection uniquement)".into()),
            false,
            "Architecture de chat détectée, mais aucun chargeur de chat n'est relié au Worker GPU."
                .into(),
            Vec::new(),
        );
    }
    if library == "diffusers" {
        let requested = capabilities
            .iter()
            .map(ModelCapability::api_name)
            .collect::<Vec<_>>()
            .join(", ");
        return (
            Some("Diffusers (incomplet)".into()),
            false,
            format!(
                "Manifest Diffusers détecté, mais incomplet pour une exécution sûre (fichiers requis absents ou classe non reconnue) : {requested}."
            ),
            Vec::new(),
        );
    }
    (
        None,
        false,
        format!(
            "Bibliothèque '{}' non prise en charge par le Worker GPU actuel.",
            if library.is_empty() {
                "non renseignée"
            } else {
                &library
            }
        ),
        Vec::new(),
    )
}

fn infer_runtime_capabilities(
    advertised: &[ModelCapability],
    architecture: Option<&str>,
    files: &[RepositoryFile],
) -> Vec<ModelCapability> {
    let mut values = HashSet::new();
    let class_name = architecture.unwrap_or_default().to_ascii_lowercase();
    let has_mask_files = files
        .iter()
        .map(|file| file.path.to_ascii_lowercase())
        .any(|path| path.contains("mask"));

    if advertised.contains(&ModelCapability::TextToImage) || class_name.contains("text2image") {
        values.insert(ModelCapability::TextToImage);
    }
    if advertised.contains(&ModelCapability::ImageToImage)
        || class_name.contains("img2img")
        || class_name.contains("image2image")
    {
        values.insert(ModelCapability::ImageToImage);
    }
    if advertised.contains(&ModelCapability::TextToVideo) || class_name.contains("text2video") {
        values.insert(ModelCapability::TextToVideo);
    }

    if advertised.contains(&ModelCapability::ImageToVideo)
        || class_name.contains("image2video")
        || class_name.contains("img2vid")
    {
        values.insert(ModelCapability::ImageToVideo);
    }
    if advertised.contains(&ModelCapability::VideoToVideo)
        || class_name.contains("video2video")
        || class_name.contains("vid2vid")
    {
        values.insert(ModelCapability::VideoToVideo);
    }

    if advertised.contains(&ModelCapability::Inpainting)
        || class_name.contains("inpaint")
        || has_mask_files
    {
        values.insert(ModelCapability::Inpainting);
    }
    if advertised.contains(&ModelCapability::Outpainting) || class_name.contains("outpaint") {
        values.insert(ModelCapability::Outpainting);
    }
    if advertised.contains(&ModelCapability::ImageVariation) || class_name.contains("variation") {
        values.insert(ModelCapability::ImageVariation);
    }
    if advertised.contains(&ModelCapability::ImageUpscale) || class_name.contains("upscale") {
        values.insert(ModelCapability::ImageUpscale);
    }
    if advertised.contains(&ModelCapability::ControlledImageGeneration)
        || class_name.contains("control")
    {
        values.insert(ModelCapability::ControlledImageGeneration);
    }
    if advertised.contains(&ModelCapability::MultiImageToVideo) {
        values.insert(ModelCapability::MultiImageToVideo);
    }

    if advertised.contains(&ModelCapability::StartEndImageToVideo) {
        values.insert(ModelCapability::StartEndImageToVideo);
    }

    if advertised.contains(&ModelCapability::KeyframesToVideo) {
        values.insert(ModelCapability::KeyframesToVideo);
    }

    if advertised.contains(&ModelCapability::VideoInpainting)
        || (class_name.contains("video") && class_name.contains("inpaint"))
    {
        values.insert(ModelCapability::VideoInpainting);
    }
    if advertised.contains(&ModelCapability::VideoUpscale)
        || (class_name.contains("video") && class_name.contains("upscale"))
    {
        values.insert(ModelCapability::VideoUpscale);
    }

    let order = [
        ModelCapability::TextToImage,
        ModelCapability::ImageToImage,
        ModelCapability::Inpainting,
        ModelCapability::Outpainting,
        ModelCapability::ImageVariation,
        ModelCapability::ImageUpscale,
        ModelCapability::ControlledImageGeneration,
        ModelCapability::TextToVideo,
        ModelCapability::ImageToVideo,
        ModelCapability::MultiImageToVideo,
        ModelCapability::StartEndImageToVideo,
        ModelCapability::KeyframesToVideo,
        ModelCapability::VideoToVideo,
        ModelCapability::VideoInpainting,
        ModelCapability::VideoUpscale,
    ];
    order
        .into_iter()
        .filter(|capability| values.contains(capability))
        .collect()
}

fn estimated_variant(size: Option<u64>, hardware: &HardwareEstimate) -> Vec<ModelVariant> {
    let Some(download_bytes) = size else {
        return Vec::new();
    };
    let ram_required = hardware
        .estimated_ram
        .as_ref()
        .map(|range| range.max_bytes)
        .unwrap_or_else(|| download_bytes.saturating_mul(2).saturating_add(2 * GIB));
    let vram_required = hardware.estimated_vram_recommended.unwrap_or_default();
    vec![ModelVariant {
        id: "auto".into(),
        label: "Estimation automatique".into(),
        ram_required,
        vram_required,
        download_bytes,
        estimated: true,
    }]
}

fn requested_pipelines(query: &CatalogQuery) -> Vec<&'static str> {
    if let Some(task) = query.task.as_deref() {
        return match task.to_ascii_uppercase().as_str() {
            "CHAT" => vec!["text-generation"],
            "VISION" => vec!["image-text-to-text", "visual-question-answering"],
            "TEXT_TO_IMAGE" => vec!["text-to-image"],
            "IMAGE_TO_IMAGE"
            | "INPAINTING"
            | "OUTPAINTING"
            | "IMAGE_VARIATION"
            | "IMAGE_UPSCALE"
            | "CONTROLLED_IMAGE_GENERATION" => vec!["image-to-image"],
            "TEXT_TO_VIDEO" => vec!["text-to-video", "video-generation"],
            "IMAGE_TO_VIDEO"
            | "MULTI_IMAGE_TO_VIDEO"
            | "START_END_IMAGE_TO_VIDEO"
            | "KEYFRAMES_TO_VIDEO" => vec!["image-to-video"],
            "VIDEO_TO_VIDEO" | "VIDEO_INPAINTING" | "VIDEO_UPSCALE" => {
                vec!["video-to-video"]
            }
            "AUDIO" => vec!["text-to-speech", "automatic-speech-recognition"],
            "TEXT_TO_SPEECH" => vec!["text-to-speech"],
            "SPEECH_TO_TEXT" => vec!["automatic-speech-recognition"],
            _ => Vec::new(),
        };
    }
    match query
        .category
        .as_deref()
        .unwrap_or_default()
        .to_ascii_uppercase()
        .as_str()
    {
        "CHAT" => vec!["text-generation"],
        "IMAGE" => vec!["text-to-image", "image-to-image"],
        "VIDEO" => vec![
            "text-to-video",
            "image-to-video",
            "video-to-video",
            "video-generation",
        ],
        "VISION" => vec!["image-text-to-text", "visual-question-answering"],
        "AUDIO" => vec![
            "text-to-speech",
            "automatic-speech-recognition",
            "audio-to-audio",
        ],
        _ => Vec::new(),
    }
}

fn hf_sort(sort: Option<&str>) -> Option<&'static str> {
    match sort.unwrap_or("trending").to_ascii_lowercase().as_str() {
        "downloads" => Some("downloads"),
        "likes" => Some("likes"),
        "updated" | "last_modified" => Some("lastModified"),
        "trending" | "popular" => Some("trendingScore"),
        _ => None,
    }
}

fn sort_models(models: &mut [CatalogModel], sort: &str) {
    match sort.to_ascii_lowercase().as_str() {
        "name" => models.sort_by_key(|model| model.name.to_ascii_lowercase()),
        "downloads" => models.sort_by_key(|model| std::cmp::Reverse(model.downloads.unwrap_or(0))),
        "likes" => models.sort_by_key(|model| std::cmp::Reverse(model.likes.unwrap_or(0))),
        "updated" | "last_modified" => {
            models.sort_by_key(|model| std::cmp::Reverse(model.last_modified.clone()))
        }
        "compatibility" | "recommended" => models.sort_by_key(|model| {
            std::cmp::Reverse((
                model.runtime_supported,
                model.quality_valid,
                model.downloads.unwrap_or(0),
            ))
        }),
        _ => models.sort_by(|left, right| {
            right
                .trending_score
                .unwrap_or_default()
                .total_cmp(&left.trending_score.unwrap_or_default())
                .then_with(|| {
                    right
                        .downloads
                        .unwrap_or(0)
                        .cmp(&left.downloads.unwrap_or(0))
                })
        }),
    }
}

/// Les moteurs intégrés restent déclarés localement : ce ne sont pas des
/// modèles publics et ils ne constituent donc pas un catalogue HF de secours.
pub fn local_runtime_models() -> Vec<CatalogModel> {
    vec![
        CatalogModel {
            id: "vidio-canvas-local".into(),
            storage_id: "vidio-canvas-local".into(),
            name: "Vidio Canvas Local".into(),
            description: "Moteur image procédural intégré, disponible hors ligne.".into(),
            kind: ModelKind::Image,
            capabilities: vec![
                ModelCapability::TextToImage,
                ModelCapability::ImageToImage,
                ModelCapability::Inpainting,
                ModelCapability::Outpainting,
                ModelCapability::ImageVariation,
                ModelCapability::ImageUpscale,
                ModelCapability::ControlledImageGeneration,
            ],
            variants: vec![ModelVariant {
                id: "builtin".into(),
                label: "Intégré".into(),
                ram_required: 256 * 1024 * 1024,
                vram_required: 0,
                download_bytes: 0,
                estimated: false,
            }],
            repository: "local/vidio-canvas".into(),
            author: Some("VidioAI".into()),
            revision: env!("CARGO_PKG_VERSION").into(),
            last_modified: None,
            pipeline_tag: Some("text-to-image".into()),
            tags: vec!["local".into()],
            library: None,
            architecture: Some("CanvasEngine".into()),
            config: Value::Object(Default::default()),
            license: Some("Projet local".into()),
            files: Vec::new(),
            estimated_size_bytes: Some(0),
            downloads: None,
            likes: None,
            trending_score: None,
            gated: false,
            private: true,
            disabled: false,
            accessibility: "LOCAL".into(),
            access_authorized: true,
            access_checked: true,
            source_available: true,
            quality_valid: true,
            runtime_name: Some("Vidio Canvas".into()),
            runtime_supported: true,
            runtime_reason: "Moteur procédural local intégré couvrant toutes les familles image (hors Worker GPU).".into(),
            pipeline_class: Some("CanvasEngine".into()),
            runtime_capabilities: vec![
                ModelCapability::TextToImage,
                ModelCapability::ImageToImage,
                ModelCapability::Inpainting,
                ModelCapability::Outpainting,
                ModelCapability::ImageVariation,
                ModelCapability::ImageUpscale,
                ModelCapability::ControlledImageGeneration,
            ],
            installable: false,
            local: true,
            hardware: HardwareEstimate {
                source: crate::hardware_estimator::HardwareSource::Official,
                confidence: crate::hardware_estimator::EstimateConfidence::High,
                weights_memory: Some(crate::hardware_estimator::MemoryRange {
                    min_bytes: 0,
                    max_bytes: 0,
                }),
                estimated_vram_min: Some(0),
                estimated_vram_recommended: Some(0),
                estimated_ram: Some(crate::hardware_estimator::MemoryRange {
                    min_bytes: 256 * 1024 * 1024,
                    max_bytes: 256 * 1024 * 1024,
                }),
                recommended_backend: Some("CPU".into()),
                recommended_precision: None,
                compatible_with_current_machine: None,
                optimization_required: false,
                compatibility_level: "UNKNOWN".into(),
                parameter_count: None,
                tensor_dtypes: Vec::new(),
                components: Vec::new(),
                supports_cpu_offload: false,
                notes: vec!["Exigence du moteur procédural intégré VidioAI.".into()],
                benchmark: None,
            },
        },
        CatalogModel {
            id: "vidio-motion-local".into(),
            storage_id: "vidio-motion-local".into(),
            name: "Vidio Motion Local".into(),
            description: "Moteur vidéo FFmpeg intégré, disponible hors ligne.".into(),
            kind: ModelKind::Video,
            capabilities: vec![
                ModelCapability::TextToVideo,
                ModelCapability::ImageToVideo,
                ModelCapability::MultiImageToVideo,
                ModelCapability::StartEndImageToVideo,
                ModelCapability::KeyframesToVideo,
                ModelCapability::VideoToVideo,
                ModelCapability::VideoInpainting,
                ModelCapability::VideoUpscale,
                ModelCapability::Audio,
            ],
            variants: vec![ModelVariant {
                id: "builtin".into(),
                label: "Intégré".into(),
                ram_required: GIB,
                vram_required: 0,
                download_bytes: 0,
                estimated: false,
            }],
            repository: "local/vidio-motion".into(),
            author: Some("VidioAI".into()),
            revision: env!("CARGO_PKG_VERSION").into(),
            last_modified: None,
            pipeline_tag: Some("text-to-video".into()),
            tags: vec!["local".into()],
            library: None,
            architecture: Some("FfmpegMotionEngine".into()),
            config: Value::Object(Default::default()),
            license: Some("Projet local".into()),
            files: Vec::new(),
            estimated_size_bytes: Some(0),
            downloads: None,
            likes: None,
            trending_score: None,
            gated: false,
            private: true,
            disabled: false,
            accessibility: "LOCAL".into(),
            access_authorized: true,
            access_checked: true,
            source_available: true,
            quality_valid: true,
            runtime_name: Some("FFmpeg".into()),
            runtime_supported: true,
            runtime_reason: "Moteur vidéo local FFmpeg intégré couvrant toutes les familles vidéo (hors Worker Diffusers).".into(),
            pipeline_class: Some("FfmpegMotionEngine".into()),
            runtime_capabilities: vec![
                ModelCapability::TextToVideo,
                ModelCapability::ImageToVideo,
                ModelCapability::MultiImageToVideo,
                ModelCapability::StartEndImageToVideo,
                ModelCapability::KeyframesToVideo,
                ModelCapability::VideoToVideo,
                ModelCapability::VideoInpainting,
                ModelCapability::VideoUpscale,
            ],
            installable: false,
            local: true,
            hardware: HardwareEstimate {
                source: crate::hardware_estimator::HardwareSource::Official,
                confidence: crate::hardware_estimator::EstimateConfidence::High,
                weights_memory: Some(crate::hardware_estimator::MemoryRange {
                    min_bytes: 0,
                    max_bytes: 0,
                }),
                estimated_vram_min: Some(0),
                estimated_vram_recommended: Some(0),
                estimated_ram: Some(crate::hardware_estimator::MemoryRange {
                    min_bytes: GIB,
                    max_bytes: GIB,
                }),
                recommended_backend: Some("CPU".into()),
                recommended_precision: None,
                compatible_with_current_machine: None,
                optimization_required: false,
                compatibility_level: "UNKNOWN".into(),
                parameter_count: None,
                tensor_dtypes: Vec::new(),
                components: Vec::new(),
                supports_cpu_offload: false,
                notes: vec!["Exigence du moteur FFmpeg intégré VidioAI.".into()],
                benchmark: None,
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        CatalogQuery, HfRawModel, ModelCapability, normalize_model, normalize_repository_reference,
        storage_id,
    };
    use serde_json::json;

    #[test]
    fn repository_urls_are_normalized_without_leaking_query_parameters() {
        assert_eq!(
            normalize_repository_reference("https://huggingface.co/Wan-AI/Wan2.1?foo=bar"),
            "Wan-AI/Wan2.1"
        );
        assert_eq!(storage_id("Wan-AI/Wan2.1"), "Wan-AI-Wan2.1");
    }

    #[test]
    fn organization_model_ids_remain_lossless_for_every_production_example() {
        // Ces identifiants couvrent points, tirets et surtout le slash qui ne
        // doit jamais être confondu avec un séparateur de route HTTP.
        for repository in [
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-video-diffusion-img2vid",
        ] {
            assert_eq!(normalize_repository_reference(repository), repository);
            super::validate_repository_id(repository).expect("ID HF organisation/modèle valide");
            assert!(!storage_id(repository).contains('/'));
        }
    }

    #[test]
    fn diffusers_t2i_requires_weights_and_model_index() {
        let raw: HfRawModel = serde_json::from_value(json!({
            "id": "org/model", "pipeline_tag": "text-to-image", "library_name": "diffusers",
            "sha": "abc", "tags": ["safetensors", "license:apache-2.0"],
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "unet/diffusion_pytorch_model.safetensors", "size": 42}
            ],
            "config": {"diffusers": {"_class_name": "StableDiffusionPipeline"}}
        }))
        .unwrap();
        let model = normalize_model(raw);
        assert!(model.runtime_supported);
        assert!(model.quality_valid);
        assert_eq!(model.capabilities, vec![ModelCapability::TextToImage]);
        assert_eq!(
            model.runtime_capabilities,
            vec![ModelCapability::TextToImage]
        );
        assert!(model.runtime_reason.contains("Worker avant téléchargement"));
    }

    #[test]
    fn diffusers_video_is_runtime_supported_when_manifest_is_complete() {
        let raw: HfRawModel = serde_json::from_value(json!({
            "id": "stabilityai/stable-video-diffusion-img2vid",
            "pipeline_tag": "image-to-video",
            "library_name": "diffusers",
            "sha": "abc",
            "tags": ["safetensors"],
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "unet/diffusion_pytorch_model.safetensors", "size": 42}
            ],
            "config": {"diffusers": {"_class_name": "StableVideoDiffusionPipeline"}}
        }))
        .unwrap();
        let model = normalize_model(raw);
        assert!(model.quality_valid);
        assert!(model.runtime_supported);
        assert!(
            model
                .runtime_capabilities
                .contains(&ModelCapability::ImageToVideo)
        );
        assert!(
            !model
                .runtime_capabilities
                .contains(&ModelCapability::MultiImageToVideo)
        );
        assert!(model.runtime_reason.contains("Worker avant téléchargement"));
    }

    #[test]
    fn wan_diffusers_pipeline_is_runtime_supported_from_model_index_class() {
        let raw: HfRawModel = serde_json::from_value(json!({
            "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "library_name": "diffusers",
            "sha": "abc",
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "transformer/diffusion_pytorch_model.safetensors", "size": 42}
            ],
            "config": {"_model_index": {"_class_name": "WanPipeline"}}
        }))
        .unwrap();
        let model = normalize_model(raw);
        assert!(model.runtime_supported);
        assert!(model.runtime_capabilities.is_empty());
    }

    #[test]
    fn model_index_pipeline_class_has_priority_over_component_architecture() {
        let raw: HfRawModel = serde_json::from_value(json!({
            "id": "org/generic-video", "pipeline_tag": "text-to-video",
            "library_name": "diffusers", "sha": "abc",
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "transformer/model.safetensors", "size": 42}
            ],
            "config": {
                "architectures": ["Transformer3DModel"],
                "_model_index": {"_class_name": "FutureVideoPipeline"}
            }
        }))
        .unwrap();
        let model = normalize_model(raw);
        assert_eq!(model.pipeline_class.as_deref(), Some("FutureVideoPipeline"));
        assert!(model.runtime_supported);
    }

    #[test]
    fn wan_derivative_quantized_stays_runtime_supported_when_pipeline_matches() {
        let raw: HfRawModel = serde_json::from_value(json!({
            "id": "AsadIsmail/Wan2.2-TI2V-5B-ternary",
            "library_name": "diffusers",
            "sha": "abc",
            "tags": ["wan", "text-to-video", "image-to-video"],
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "transformer/diffusion_pytorch_model.safetensors", "size": 42}
            ],
            "config": {"_model_index": {"_class_name": "WanPipeline"}}
        }))
        .unwrap();
        let model = normalize_model(raw);
        assert!(model.runtime_supported);
        assert!(
            model
                .runtime_capabilities
                .contains(&ModelCapability::TextToVideo)
        );
    }

    #[test]
    fn gated_list_entry_remains_installable_until_access_is_checked() {
        let raw: HfRawModel = serde_json::from_value(json!({
            "id": "black-forest-labs/FLUX.1-dev",
            "pipeline_tag": "text-to-image",
            "library_name": "diffusers",
            "gated": "auto",
            "sha": "abc",
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "transformer/diffusion_pytorch_model.safetensors", "size": 42}
            ],
            "config": {"diffusers": {"_class_name": "FluxPipeline"}}
        }))
        .unwrap();

        let model = normalize_model(raw);
        assert!(model.gated);
        assert!(!model.access_checked);
        assert!(!model.access_authorized);
        assert_eq!(model.accessibility, "UNVERIFIED");
        assert!(model.installable);
    }

    #[test]
    fn exact_gated_access_check_distinguishes_authorized_and_denied() {
        let raw_json = json!({
            "id": "black-forest-labs/FLUX.1-dev",
            "pipeline_tag": "text-to-image",
            "library_name": "diffusers",
            "gated": "auto",
            "sha": "abc",
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "transformer/diffusion_pytorch_model.safetensors", "size": 42}
            ],
            "config": {"diffusers": {"_class_name": "FluxPipeline"}}
        });
        let mut authorized: HfRawModel = serde_json::from_value(raw_json.clone()).unwrap();
        authorized.access_checked = true;
        authorized.access_authorized = true;
        let authorized = normalize_model(authorized);
        assert_eq!(authorized.accessibility, "AUTHORIZED");
        assert!(authorized.installable);

        let mut denied: HfRawModel = serde_json::from_value(raw_json).unwrap();
        denied.access_checked = true;
        let denied = normalize_model(denied);
        assert_eq!(denied.accessibility, "ACCESS_REQUIRED");
        assert!(!denied.installable);
    }

    #[test]
    fn query_limits_are_bounded() {
        let query = CatalogQuery {
            page: Some(0),
            limit: Some(5_000),
            ..Default::default()
        };
        assert_eq!(query.page(), 1);
        assert_eq!(query.limit(), 60);
    }
}

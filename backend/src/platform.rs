//! Services applicatifs de VidioAI (étapes 4 à 12).
//!
//! Ce module volontairement documenté centralise le premier socle fonctionnel :
//! configuration persistante, catalogue de modèles, jobs, événements temps réel,
//! runtime, assets et générations d'images. Les types exposés ici constituent le
//! contrat JSON consommé par Next.js. Ils évitent surtout de coder les modèles ou
//! les états de progression en dur dans React.

use async_trait::async_trait;
use axum::{
    Json, Router,
    body::Body,
    extract::{
        DefaultBodyLimit, Multipart, Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgb, imageops::FilterType};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    future::Future,
    path::{Path as FilePath, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs,
    io::AsyncReadExt,
    process::Command,
    sync::{Mutex, RwLock, broadcast, mpsc},
    time::{Duration, Instant, sleep},
};
use uuid::Uuid;

use crate::engine_ia::engine::text_messages_ia;
use crate::hardware_benchmark_store::HardwareBenchmarkStore;
use crate::hardware_estimator::{
    CurrentMachine, HardwareBenchmark, HardwareEstimate, HardwareEstimator,
};
use crate::host_agent::{HostAgentClient, HostSnapshot, ResourceSource, resolve_system};
use crate::huggingface_catalog::{
    CatalogModel as CatalogEntry, CatalogQuery, CatalogResult, HuggingFaceCatalogService,
    ModelCapability, ModelKind, ModelVariant, RepositoryFile, local_runtime_models, storage_id,
};
use crate::job_store::JobStore;
use crate::object_storage::{
    ObjectStorage, S3Storage, SnapshotManifest, TransferProgressCallback, UploadProgress,
    UploadProgressCallback, is_snapshot_file, model_s3_prefix,
};
use crate::worker::{
    GenerateResponse, WorkerBenchmarkObservation, WorkerClient, WorkerCompatibility, WorkerReady,
    WorkerResources,
};

/// Taille maximale d'un asset reçu. La limite protège le processus avant même
/// que les données ne soient décodées par la crate `image`.
const MAX_IMAGE_BYTES: usize = 25 * 1024 * 1024;
const MAX_VIDEO_BYTES: usize = 512 * 1024 * 1024;

/// Retourne un timestamp Unix simple et stable pour les contrats API.
pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

// -----------------------------------------------------------------------------
// Erreurs HTTP homogènes
// -----------------------------------------------------------------------------

/// Une erreur applicative est toujours renvoyée sous la forme
/// `{ "error": "..." }`, ce qui permet au frontend d'afficher le vrai message.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
        }
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        eprintln!("Erreur plateforme VidioAI : {error}");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "Une erreur interne empêche l'opération.".to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

// -----------------------------------------------------------------------------
// Étape 4 : configuration persistante et arborescence de stockage
// -----------------------------------------------------------------------------

/// Configuration réellement persistée dans `settings.json`.
///
/// Les chemins sont stockés en absolu : un redémarrage lancé depuis un autre
/// dossier de travail ne doit jamais changer leur signification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub models_dir: PathBuf,
    pub outputs_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub work_dir: PathBuf,
    pub auto_unload_minutes: u64,
    pub automatic_optimization: bool,
}

impl AppSettings {
    /// Construit les valeurs du premier démarrage sous `VIDIOAI_DATA_DIR`, ou
    /// dans `backend/data` lorsque la variable n'est pas définie.
    fn defaults(base: &FilePath) -> Self {
        // Chaque niveau de stockage peut être monté séparément en production.
        // `VIDIOAI_DATA_DIR` reste le repli compatible avec les installations
        // locales historiques.
        let configured = |name: &str, fallback: PathBuf| {
            std::env::var_os(name)
                .map(PathBuf::from)
                .unwrap_or(fallback)
        };
        Self {
            models_dir: configured("VIDIOAI_MODELS_DIR", base.join("models")),
            outputs_dir: configured("VIDIOAI_OUTPUTS_DIR", base.join("outputs")),
            cache_dir: configured("VIDIOAI_CACHE_DIR", base.join("cache")),
            work_dir: configured("VIDIOAI_WORK_DIR", base.join("work")),
            auto_unload_minutes: 15,
            automatic_optimization: true,
        }
    }

    /// Tous les dossiers sont validés par une seule routine afin que le premier
    /// démarrage et `PUT /api/settings` aient exactement les mêmes garanties.
    async fn ensure_directories(&self) -> Result<(), ApiError> {
        for (label, directory) in [
            ("modèles", &self.models_dir),
            ("sorties", &self.outputs_dir),
            ("cache", &self.cache_dir),
            ("temporaire", &self.work_dir),
        ] {
            if directory.as_os_str().is_empty() {
                return Err(ApiError::bad_request(format!(
                    "Le dossier {label} ne peut pas être vide."
                )));
            }

            // Un fichier à la place d'un dossier est une configuration invalide.
            if fs::metadata(directory)
                .await
                .is_ok_and(|meta| meta.is_file())
            {
                return Err(ApiError::bad_request(format!(
                    "Le chemin {} désigne un fichier, pas un dossier.",
                    directory.display()
                )));
            }

            fs::create_dir_all(directory).await.map_err(|error| {
                ApiError::bad_request(format!(
                    "Impossible de créer le dossier {} : {error}",
                    directory.display()
                ))
            })?;
        }
        Ok(())
    }
}

/// Le store connaît à la fois la valeur courante et l'emplacement du fichier.
pub struct SettingsStore {
    config_path: PathBuf,
    value: RwLock<AppSettings>,
}

impl SettingsStore {
    /// Charge la configuration existante ou crée automatiquement les valeurs et
    /// dossiers du premier démarrage.
    async fn initialize() -> Result<Self, ApiError> {
        let default_base = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("data");
        let base = std::env::var_os("VIDIOAI_STATE_DIR")
            .or_else(|| std::env::var_os("VIDIOAI_DATA_DIR"))
            .map(PathBuf::from)
            .unwrap_or(default_base);
        let config_path = std::env::var_os("VIDIOAI_CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| base.join("settings.json"));

        let mut settings = match fs::read(&config_path).await {
            Ok(content) => serde_json::from_slice(&content).map_err(|error| {
                ApiError::internal(format!("Configuration illisible : {error}"))
            })?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                AppSettings::defaults(&base)
            }
            Err(error) => return Err(ApiError::internal(error)),
        };

        // Les montages de conteneur sont une contrainte d'exploitation et ont
        // priorité sur un ancien settings.json restauré depuis le volume state.
        for (name, target) in [
            ("VIDIOAI_MODELS_DIR", &mut settings.models_dir),
            ("VIDIOAI_OUTPUTS_DIR", &mut settings.outputs_dir),
            ("VIDIOAI_CACHE_DIR", &mut settings.cache_dir),
            ("VIDIOAI_WORK_DIR", &mut settings.work_dir),
        ] {
            if let Some(value) = std::env::var_os(name) {
                *target = PathBuf::from(value);
            }
        }

        settings.ensure_directories().await?;
        let store = Self {
            config_path,
            value: RwLock::new(settings),
        };
        store.persist().await?;
        Ok(store)
    }

    async fn get(&self) -> AppSettings {
        self.value.read().await.clone()
    }

    fn state_dir(&self) -> PathBuf {
        self.config_path
            .parent()
            .unwrap_or_else(|| FilePath::new("."))
            .to_path_buf()
    }

    /// Écriture atomique : le JSON complet est écrit dans un fichier temporaire,
    /// puis renommé. Une interruption ne peut donc pas laisser un JSON tronqué.
    async fn persist(&self) -> Result<(), ApiError> {
        let value = self.value.read().await.clone();
        let parent = self
            .config_path
            .parent()
            .unwrap_or_else(|| FilePath::new("."));
        fs::create_dir_all(parent)
            .await
            .map_err(ApiError::internal)?;
        let temporary = self.config_path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(&value).map_err(ApiError::internal)?;
        fs::write(&temporary, bytes)
            .await
            .map_err(ApiError::internal)?;
        fs::rename(&temporary, &self.config_path)
            .await
            .map_err(ApiError::internal)?;
        Ok(())
    }

    async fn replace(&self, value: AppSettings) -> Result<AppSettings, ApiError> {
        value.ensure_directories().await?;
        *self.value.write().await = value.clone();
        self.persist().await?;
        Ok(value)
    }
}

// -----------------------------------------------------------------------------
// Étape 5 : catalogue générique de modèles
// -----------------------------------------------------------------------------

/// DTO enrichi avec les informations propres à la machine courante.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInputProfile {
    pub min_input_images: usize,
    pub max_input_images: usize,
    pub supported_image_roles: Vec<String>,
    pub supports_start_end_frames: bool,
    pub supports_reference_images: bool,
    pub supports_keyframes: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelView {
    pub id: String,
    pub name: String,
    pub description: String,
    pub kind: ModelKind,
    pub capabilities: Vec<ModelCapability>,
    pub declared_capabilities: Vec<ModelCapability>,
    pub display_capabilities: Vec<ModelCapability>,
    pub variants: Vec<ModelVariant>,
    pub installed: bool,
    pub cache_status: String,
    pub cache_error: Option<String>,
    pub runtime_dependencies: Vec<serde_json::Value>,
    pub runtime_precision: Option<serde_json::Value>,
    /// `true` uniquement lorsqu'un runtime exploitable a été validé. Un simple
    /// manifeste Hugging Face ne doit jamais être présenté comme un modèle prêt.
    pub runtime_ready: bool,
    /// État lisible par le frontend sans avoir à déduire la situation depuis
    /// plusieurs booléens (`builtin`, `not_installed` ou `ready`).
    pub installation_state: String,
    pub compatible: bool,
    pub recommended_variant: Option<String>,
    pub loaded: bool,
    /// Dépôt de référence vérifiable. Le catalogue n'utilise volontairement
    /// que des organisations officielles sur Hugging Face.
    pub repository: String,
    pub repository_url: String,
    pub license: String,
    pub engine: String,
    /// `procedural` désigne exclusivement les moteurs locaux Canvas/FFmpeg ;
    /// `ai` implique des poids réels et un runtime worker validé.
    pub engine_type: String,
    /// Indique si le worker livré sait réellement exécuter au moins une des
    /// capacités du modèle. Les modèles vidéo restent catalogués sans être
    /// présentés comme installables par le runtime T2I actuel.
    pub runtime_supported: bool,
    /// Décision tri-state du Worker. UNKNOWN autorise une installation de
    /// validation, mais ne prétend jamais que le runtime est déjà supporté.
    pub runtime_compatibility: String,
    /// Justification fournie par la matrice pipeline/library/architecture.
    pub runtime_reason: String,
    pub pipeline_class: Option<String>,
    pub runtime_capabilities: Vec<ModelCapability>,
    pub input_profile: ModelInputProfile,
    /// Alias métier explicite demandé par le contrat catalogue.
    pub vidioai_supported: bool,
    pub source_available: bool,
    pub hardware_compatible: bool,
    /// Capacités libres observées au moment de la réponse. Elles expliquent la
    /// décision et ne doivent pas être confondues avec la capacité physique.
    pub available_ram_bytes: u64,
    pub available_vram_bytes: u64,
    pub installable: bool,
    pub compatibility_checks: Vec<CompatibilityCheck>,
    pub accessibility: String,
    pub access_authorized: bool,
    pub access_checked: bool,
    pub gated: bool,
    pub private: bool,
    pub author: Option<String>,
    pub revision: String,
    pub installed_revision: Option<String>,
    pub update_available: bool,
    pub last_modified: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub library: Option<String>,
    pub architecture: Option<String>,
    pub files: Vec<RepositoryFile>,
    pub estimated_size_bytes: Option<u64>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub quality_valid: bool,
    pub compatibility_level: String,
    /// Objet normalisé détaillant la provenance et les fourchettes. Le frontend
    /// n'a plus à interpréter une variante comme une mesure exacte.
    pub hardware: HardwareEstimate,
}

/// Une décision de compatibilité est exposée comme une liste de contrôles
/// lisibles plutôt qu'un opaque « supporté/non supporté ».
#[derive(Debug, Clone, Serialize)]
pub struct CompatibilityCheck {
    pub key: &'static str,
    pub label: &'static str,
    pub ok: bool,
    pub detail: String,
}

// -----------------------------------------------------------------------------
// Étapes 6 et 7 : jobs, queue et événements WebSocket
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobKind {
    InstallModel,
    RestoreModel,
    CacheModel,
    GenerateImage,
    GenerateVideo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Dispatching,
    Running,
    SavingOutput,
    Completed,
    Failed,
    Cancelled,
    /// Le backend a redémarré alors que le job était encore actif. Cet état est
    /// durable et évite de présenter une inférence disparue comme « running ».
    Interrupted,
    /// Le job n'avait pas commencé. Son payload reste consultable pour qu'un
    /// opérateur ou futur ordonnanceur puisse le relancer explicitement.
    PendingRetry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Uuid,
    pub kind: JobKind,
    pub target_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    pub status: JobStatus,
    pub stage: String,
    pub progress: u8,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer: Option<UploadProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependency: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventEnvelope {
    pub event: String,
    pub timestamp: u64,
    pub data: serde_json::Value,
}

// -----------------------------------------------------------------------------
// Étape 8 : état du runtime des modèles
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEntry {
    pub model_id: String,
    pub state: String,
    pub device: String,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub last_used_at: u64,
}

// -----------------------------------------------------------------------------
// Étapes 9, 11 et 12 : assets et générations communes
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AssetKind {
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub id: Uuid,
    pub kind: AssetKind,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_seconds: Option<f64>,
    #[serde(default)]
    pub fps: Option<f64>,
    pub created_at: u64,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GenerationMode {
    TextToImage,
    ImageToImage,
    TextToVideo,
    ImageToVideo,
    VideoToVideo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    pub id: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<Uuid>,
    pub kind: AssetKind,
    pub mode: GenerationMode,
    #[serde(default)]
    pub capability: Option<ModelCapability>,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub model_id: String,
    pub input_asset_id: Option<Uuid>,
    #[serde(default)]
    pub mask_asset_id: Option<Uuid>,
    #[serde(default)]
    pub control_asset_id: Option<Uuid>,
    #[serde(default)]
    pub input_images: Vec<GenerationInputImage>,
    pub output_asset_id: Option<Uuid>,
    pub status: GenerationStatus,
    pub progress: u8,
    pub error: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    #[serde(default)]
    pub duration_seconds: Option<u32>,
    #[serde(default)]
    pub requested_duration_seconds: Option<f64>,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub requested_quality: Option<String>,
    #[serde(default)]
    pub requested_aspect_ratio: Option<String>,
    #[serde(default)]
    pub requested_fps: Option<u32>,
    #[serde(default)]
    pub requested_frames: Option<u32>,
    #[serde(default)]
    pub inference_frames: Option<u32>,
    #[serde(default)]
    pub actual_width: Option<u32>,
    #[serde(default)]
    pub actual_height: Option<u32>,
    #[serde(default)]
    pub actual_fps: Option<f64>,
    #[serde(default)]
    pub actual_frames: Option<u32>,
    #[serde(default)]
    pub actual_duration: Option<f64>,
    #[serde(default)]
    pub audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationInputImage {
    pub asset_id: Uuid,
    #[serde(default)]
    pub order: usize,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Deserialize)]
pub struct GenerateImageRequest {
    pub mode: GenerationMode,
    #[serde(default)]
    pub capability: Option<ModelCapability>,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub model_id: Option<String>,
    pub input_asset_id: Option<Uuid>,
    pub mask_asset_id: Option<Uuid>,
    pub control_asset_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateVideoRequest {
    pub mode: GenerationMode,
    #[serde(default)]
    pub capability: Option<ModelCapability>,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub model_id: Option<String>,
    pub input_asset_id: Option<Uuid>,
    #[serde(default)]
    pub input_images: Vec<GenerationInputImage>,
    pub duration_seconds: Option<u32>,
    #[serde(alias = "resolution")]
    pub quality: Option<String>,
    pub aspect_ratio: Option<String>,
    pub fps: Option<u32>,
    #[serde(default)]
    pub audio: bool,
}

// -----------------------------------------------------------------------------
// Étape 21 : projets persistants reliant assets, générations et conversations
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub asset_ids: Vec<Uuid>,
    pub generation_ids: Vec<Uuid>,
    pub chat_ids: Vec<Uuid>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Deserialize)]
pub struct ProjectInput {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub asset_ids: Vec<Uuid>,
    #[serde(default)]
    pub generation_ids: Vec<Uuid>,
    #[serde(default)]
    pub chat_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: Uuid,
    pub role: String,
    pub content: String,
    pub attachment_ids: Vec<Uuid>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: Uuid,
    pub title: String,
    pub model_id: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Deserialize)]
pub struct CreateChatInput {
    pub title: Option<String>,
    pub model_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SendMessageInput {
    pub content: String,
    #[serde(default)]
    pub attachment_ids: Vec<Uuid>,
}

#[derive(Debug, Serialize)]
pub struct ChatTurn {
    pub user: ChatMessage,
    pub assistant: ChatMessage,
    pub suggested_action: Option<String>,
}

/// Recharge l'historique au redémarrage. Un manifeste corrompu est ignoré
/// individuellement afin qu'un ancien job ne puisse pas empêcher le serveur de
/// démarrer et que les autres résultats restent accessibles.
async fn load_generations(settings: &AppSettings) -> HashMap<Uuid, Generation> {
    let directory = settings.outputs_dir.join("generations");
    let _ = fs::create_dir_all(&directory).await;
    let Ok(mut entries) = fs::read_dir(directory).await else {
        return HashMap::new();
    };
    let mut generations = HashMap::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        if let Ok(bytes) = fs::read(entry.path()).await
            && let Ok(generation) = serde_json::from_slice::<Generation>(&bytes)
        {
            generations.insert(generation.id, generation);
        }
    }
    generations
}

async fn load_projects(settings: &AppSettings) -> HashMap<Uuid, Project> {
    let path = settings.outputs_dir.join("projects.json");
    fs::read(path)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<Project>>(&bytes).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|project| (project.id, project))
        .collect()
}

async fn load_chats(settings: &AppSettings) -> HashMap<Uuid, ChatSession> {
    let path = settings.outputs_dir.join("chats.json");
    fs::read(path)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<ChatSession>>(&bytes).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|chat| (chat.id, chat))
        .collect()
}

/// État partagé injecté dans tous les handlers Axum.
pub struct AppState {
    settings: SettingsStore,
    /// Client Hugging Face et cache disque partagés entre toutes les requêtes.
    catalog: HuggingFaceCatalogService,
    compatibility_cache: RwLock<HashMap<String, (u64, WorkerCompatibility)>>,
    jobs: RwLock<HashMap<Uuid, Job>>,
    /// Réservation atomique des restaurations par repository@revision.
    restore_claims: Mutex<HashMap<String, Uuid>>,
    runtime: RwLock<HashMap<String, RuntimeEntry>>,
    generations: RwLock<HashMap<Uuid, Generation>>,
    projects: RwLock<HashMap<Uuid, Project>>,
    chats: RwLock<HashMap<Uuid, ChatSession>>,
    cancelled_generations: RwLock<HashSet<Uuid>>,
    events: broadcast::Sender<EventEnvelope>,
    job_store: JobStore,
    hardware_benchmark_store: HardwareBenchmarkStore,
    /// Client de l'agent natif. `None` est autorisé en LOCAL et interdit par la
    /// readiness GPU_PRODUCTION.
    host_agent: Option<HostAgentClient>,
    worker: Option<WorkerClient>,
    object_storage: Arc<S3Storage>,
    mode: RwLock<ApplicationMode>,
    profile: ApplicationProfile,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ApplicationMode {
    AcceptingJobs,
    Draining,
    Stopping,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ApplicationProfile {
    Local,
    GpuProduction,
}

impl ApplicationProfile {
    fn from_env() -> Self {
        if std::env::var("APP_ENV").is_ok_and(|value| value.eq_ignore_ascii_case("GPU_PRODUCTION"))
        {
            Self::GpuProduction
        } else {
            Self::Local
        }
    }
}

impl AppState {
    pub async fn initialize() -> Result<Arc<Self>, ApiError> {
        let settings = SettingsStore::initialize().await?;
        let current_settings = settings.get().await;
        let catalog = HuggingFaceCatalogService::initialize(&current_settings.cache_dir).await;
        let generations = load_generations(&current_settings).await;
        let projects = load_projects(&current_settings).await;
        let chats = load_chats(&current_settings).await;
        let database_path = settings.state_dir().join("jobs.sqlite");
        let job_store = JobStore::open(database_path.clone())
            .await
            .map_err(ApiError::internal)?;
        let hardware_benchmark_store = HardwareBenchmarkStore::open(database_path)
            .await
            .map_err(ApiError::internal)?;
        let jobs = job_store
            .load_and_interrupt_active()
            .await
            .map_err(ApiError::internal)?
            .into_iter()
            .map(|job| (job.id, job))
            .collect();
        let (events, _) = broadcast::channel(256);
        let state = Arc::new(Self {
            settings,
            catalog,
            compatibility_cache: RwLock::new(HashMap::new()),
            jobs: RwLock::new(jobs),
            restore_claims: Mutex::new(HashMap::new()),
            runtime: RwLock::new(HashMap::new()),
            generations: RwLock::new(generations),
            projects: RwLock::new(projects),
            chats: RwLock::new(chats),
            cancelled_generations: RwLock::new(HashSet::new()),
            events,
            job_store,
            hardware_benchmark_store,
            host_agent: HostAgentClient::from_env(),
            worker: WorkerClient::from_env(),
            object_storage: Arc::new(S3Storage::from_env()),
            mode: RwLock::new(ApplicationMode::AcceptingJobs),
            profile: ApplicationProfile::from_env(),
        });

        // Les manifests de génération sont durables eux aussi. Un rendu qui
        // était actif avant l'arrêt ne peut pas se terminer tout seul.
        let interrupted_generations = {
            let mut generations = state.generations.write().await;
            let mut interrupted = Vec::new();
            for generation in generations.values_mut() {
                if matches!(
                    generation.status,
                    GenerationStatus::Queued | GenerationStatus::Running
                ) {
                    generation.status = GenerationStatus::Failed;
                    generation.error =
                        Some("Génération interrompue par un redémarrage du backend.".into());
                    generation.updated_at = unix_now();
                    interrupted.push(generation.clone());
                }
            }
            interrupted
        };
        for generation in interrupted_generations {
            update_generation(&state, generation).await;
        }

        // Le moniteur applique réellement `auto_unload_minutes`. Son intervalle
        // court garde le test observable, sans effectuer de travail coûteux.
        let monitor_state = state.clone();
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(30)).await;
                let timeout = monitor_state.settings.get().await.auto_unload_minutes * 60;
                let now = unix_now();
                let removed = {
                    let mut runtime = monitor_state.runtime.write().await;
                    let expired: Vec<_> = runtime
                        .values()
                        .filter(|entry| now.saturating_sub(entry.last_used_at) >= timeout)
                        .map(|entry| entry.model_id.clone())
                        .collect();
                    for id in &expired {
                        runtime.remove(id);
                    }
                    expired
                };
                for model_id in removed {
                    if let Some(worker) = &monitor_state.worker
                        && let Err(error) = worker.unload(&storage_id(&model_id)).await
                    {
                        eprintln!("Auto-déchargement worker de {model_id} impossible : {error}");
                    }
                    monitor_state.emit(
                        "resources.updated",
                        &json!({
                            "model_id": model_id, "state": "auto_unloaded"
                        }),
                    );
                }
            }
        });

        Ok(state)
    }

    /// L'absence de client WebSocket n'est jamais une erreur métier : `send`
    /// peut légitimement échouer lorsque personne n'écoute encore.
    fn emit<T: Serialize>(&self, event: &str, payload: &T) {
        if let Ok(data) = serde_json::to_value(payload) {
            let _ = self.events.send(EventEnvelope {
                event: event.to_string(),
                timestamp: unix_now(),
                data,
            });
        }
    }

    async fn update_job(
        &self,
        id: Uuid,
        status: JobStatus,
        stage: &str,
        progress: u8,
        message: &str,
    ) {
        let updated = {
            let mut jobs = self.jobs.write().await;
            let Some(job) = jobs.get_mut(&id) else { return };
            let now = unix_now();
            if matches!(
                status,
                JobStatus::Dispatching | JobStatus::Running | JobStatus::SavingOutput
            ) && job.started_at.is_none()
            {
                job.started_at = Some(now);
            }
            if matches!(
                status,
                JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
            ) {
                job.completed_at = Some(now);
            }
            if status == JobStatus::Failed {
                let candidate = message.split(':').next().unwrap_or("JOB_FAILED").trim();
                let code = if !candidate.is_empty()
                    && candidate
                        .chars()
                        .all(|character| character.is_ascii_uppercase() || character == '_')
                {
                    candidate
                } else {
                    "JOB_FAILED"
                };
                job.error = Some(json!({
                    "code": code,
                    "message": message,
                    "retryable": matches!(
                        code,
                        "JOB_DISPATCH_TIMEOUT" | "WORKER_START_TIMEOUT" | "WORKER_LOST"
                    ),
                }));
            }
            job.status = status;
            job.stage = stage.to_string();
            job.progress = progress.min(100);
            job.message = message.to_string();
            job.updated_at = now;
            job.clone()
        };
        // Les versions précédentes étiquetaient aussi les générations comme
        // `model.install.*`. Le type du job détermine désormais son événement.
        let event = match (&updated.kind, &updated.status) {
            (JobKind::InstallModel, JobStatus::Completed) => "model.install.completed",
            (JobKind::InstallModel, JobStatus::Failed) => "model.install.failed",
            (JobKind::InstallModel, _) => "model.install.progress",
            (JobKind::CacheModel, _) => "model.cache.progress",
            _ => "job.updated",
        };
        if let Err(error) = self.job_store.upsert(&updated).await {
            eprintln!("Persistance du job {} impossible : {error}", updated.id);
        }
        match updated.status {
            JobStatus::Dispatching => eprintln!("JOB_DISPATCH id={}", updated.id),
            JobStatus::Running => eprintln!(
                "JOB_PROGRESS id={} progress={} stage={}",
                updated.id, updated.progress, updated.stage
            ),
            JobStatus::SavingOutput => eprintln!("JOB_OUTPUT_SAVING id={}", updated.id),
            JobStatus::Completed => eprintln!("JOB_COMPLETED id={}", updated.id),
            JobStatus::Failed => eprintln!(
                "JOB_FAILED id={} code={}",
                updated.id,
                updated
                    .error
                    .as_ref()
                    .and_then(|error| error.get("code"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("JOB_FAILED")
            ),
            _ => {}
        }
        self.emit(event, &updated);
        self.emit("queue.updated", &self.queue().await);
    }

    async fn set_job_result(&self, id: Uuid, result: serde_json::Value) {
        let updated = {
            let mut jobs = self.jobs.write().await;
            let Some(job) = jobs.get_mut(&id) else { return };
            job.result = Some(result);
            job.updated_at = unix_now();
            job.clone()
        };
        let _ = self.job_store.upsert(&updated).await;
        self.emit("job.updated", &updated);
    }

    async fn classify_generation_error(&self, id: Uuid, video: bool, message: &str) -> String {
        let candidate = message.split(':').next().unwrap_or_default().trim();
        if !candidate.is_empty()
            && candidate
                .chars()
                .all(|character| character.is_ascii_uppercase() || character == '_')
        {
            return message.to_owned();
        }
        let saving_output = self
            .jobs
            .read()
            .await
            .get(&id)
            .is_some_and(|job| job.status == JobStatus::SavingOutput);
        let code = if video
            && (saving_output
                || message.to_ascii_lowercase().contains("ffmpeg")
                || message.to_ascii_lowercase().contains("ffprobe"))
        {
            "VIDEO_ENCODING_FAILED"
        } else if saving_output {
            "OUTPUT_WRITE_FAILED"
        } else {
            "GENERATION_FAILED"
        };
        format!("{code}: {message}")
    }

    async fn update_cache_progress(&self, id: Uuid, transfer: UploadProgress) {
        let percent = transfer.percent();
        let overall = (90.0 + percent * 0.09).round().clamp(90.0, 99.0) as u8;
        let file = transfer
            .current_file
            .as_deref()
            .unwrap_or("validation du cache");
        let updated = {
            let mut jobs = self.jobs.write().await;
            let Some(job) = jobs.get_mut(&id) else { return };
            job.status = JobStatus::Running;
            job.started_at.get_or_insert_with(unix_now);
            job.stage = "saving_cache".into();
            job.progress = overall;
            job.message = format!(
                "Sauvegarde dans le cache S3 · {file} · {:.2}% · {} fichier(s) déjà présent(s)",
                percent, transfer.files_skipped
            );
            job.transfer = Some(transfer);
            job.cache_status = Some("CACHE_UPLOADING".into());
            job.cache_error = None;
            job.updated_at = unix_now();
            job.clone()
        };
        if let Err(error) = self.job_store.upsert(&updated).await {
            eprintln!("Persistance du job {} impossible : {error}", updated.id);
        }
        self.emit("model.cache.progress", &updated);
        self.emit("model.install.progress", &updated);
        self.emit("queue.updated", &self.queue().await);
    }

    async fn update_cache_status(&self, id: Uuid, status: &str, error: Option<String>) {
        let updated = {
            let mut jobs = self.jobs.write().await;
            let Some(job) = jobs.get_mut(&id) else { return };
            job.cache_status = Some(status.into());
            job.cache_error = error;
            job.updated_at = unix_now();
            job.clone()
        };
        let _ = self.job_store.upsert(&updated).await;
        self.emit("model.cache.progress", &updated);
    }

    async fn update_dependency_progress(
        &self,
        id: Uuid,
        worker_state: &str,
        dependencies: &[serde_json::Value],
    ) {
        let dependency = dependencies.first().cloned();
        let label = dependency
            .as_ref()
            .and_then(|value| value.get("package"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("dépendances runtime");
        let updated = {
            let mut jobs = self.jobs.write().await;
            let Some(job) = jobs.get_mut(&id) else { return };
            job.status = JobStatus::Running;
            let (stage, progress, action) = match worker_state {
                "DOWNLOADING_DEPENDENCY" => {
                    ("installing_dependencies", 85, "téléchargement en cours")
                }
                "INSTALLING_DEPENDENCIES" => ("installing_dependencies", 87, "installation"),
                _ => ("resolving_dependencies", 84, "résolution"),
            };
            job.stage = stage.into();
            job.progress = progress;
            job.message = format!("Préparation du runtime · {label} — {action}");
            job.dependency = dependency;
            job.updated_at = unix_now();
            job.clone()
        };
        let _ = self.job_store.upsert(&updated).await;
        self.emit("model.dependency.progress", &updated);
        self.emit("model.install.progress", &updated);
    }

    async fn insert_job(&self, job: Job) -> Result<(), ApiError> {
        let id = job.id;
        let kind = job.kind.clone();
        self.job_store
            .upsert(&job)
            .await
            .map_err(ApiError::internal)?;
        self.jobs.write().await.insert(id, job);
        eprintln!("JOB_CREATE id={id} kind={kind:?}");
        self.emit("queue.updated", &self.queue().await);
        Ok(())
    }

    async fn queue(&self) -> Vec<Job> {
        let mut queue: Vec<_> = self
            .jobs
            .read()
            .await
            .values()
            .filter(|job| {
                matches!(
                    job.status,
                    JobStatus::Queued
                        | JobStatus::Dispatching
                        | JobStatus::Running
                        | JobStatus::SavingOutput
                        | JobStatus::PendingRetry
                )
            })
            .cloned()
            .collect();
        queue.sort_by_key(|job| job.created_at);
        queue
    }

    async fn ensure_accepting_jobs(&self) -> Result<(), ApiError> {
        if *self.mode.read().await == ApplicationMode::AcceptingJobs {
            Ok(())
        } else {
            Err(ApiError::unavailable(
                "VidioAI est en drainage et n'accepte plus de nouveaux jobs.",
            ))
        }
    }
}

/// Monte toutes les routes de la plateforme sous `/api` dans `main.rs`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/settings", get(get_settings).put(put_settings))
        .route("/ready", get(get_ready))
        .route("/health", get(get_api_health))
        .route("/system", get(get_system))
        .route("/resources", get(get_resources))
        .route("/admin/drain", post(start_drain))
        .route("/admin/resume", post(resume_jobs))
        .route("/admin/stop", post(stop_jobs))
        .route("/dashboard", get(get_dashboard))
        .route("/projects", get(list_projects).post(create_project))
        .route(
            "/projects/{id}",
            get(get_project).put(update_project).delete(delete_project),
        )
        .route("/chats", get(list_chats).post(create_chat))
        .route("/chats/{id}", get(get_chat).delete(delete_chat))
        .route("/chats/{id}/messages", post(send_chat_message))
        .route("/models", get(get_models))
        .route("/models/catalog/refresh", post(refresh_models))
        .route("/models/installed", get(list_installed_models))
        .route("/models/cloud", get(list_cloud_models))
        .route("/models/cloud/restore", post(restore_cloud_models))
        // Les routes query/body sont la forme canonique : un ID Hugging Face
        // contient un slash et ne doit pas dépendre du décodage du reverse proxy.
        .route(
            "/models/by-id",
            get(get_model_from_query).delete(delete_model_from_query),
        )
        .route("/models/install", post(install_model_from_body))
        .route("/models/cache", post(cache_model_from_body))
        .route("/models/load", post(load_model_from_body))
        .route("/models/unload", post(unload_model_from_body))
        // Compatibilité conservée pour les anciens clients et les IDs locaux
        // sans slash. Les nouvelles interfaces n'utilisent plus ces routes.
        .route("/models/{id}", get(get_model).delete(delete_model))
        .route("/models/{id}/install", post(install_model))
        .route("/models/{id}/load", post(load_model))
        .route("/models/{id}/unload", post(unload_model))
        .route("/router/classify", post(classify_request))
        .route("/models/route", post(route_model))
        .route("/optimizer", post(optimize_request))
        .route("/jobs/{id}", get(get_job))
        .route("/queue", get(get_queue))
        .route("/events", get(events_upgrade))
        .route(
            "/assets",
            get(list_assets)
                .post(upload_asset)
                .layer(DefaultBodyLimit::max(MAX_VIDEO_BYTES + 1024 * 1024)),
        )
        .route("/assets/{id}", get(get_asset).delete(delete_asset))
        .route("/images/generate", post(generate_image))
        .route("/videos/generate", post(generate_video))
        .route("/generations", get(list_generations))
        .route(
            "/generations/{id}",
            get(get_generation).delete(delete_generation),
        )
        .route("/generations/{id}/cancel", post(cancel_generation))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct ClassifyInput {
    prompt: String,
    input_asset_id: Option<Uuid>,
}
#[derive(Debug, Serialize)]
struct ClassifyOutput {
    action: String,
    capability: ModelCapability,
    confidence: f32,
}

/// Routeur d'intention transparent : les règles sont volontairement simples et
/// déterministes. Une couche LLM pourra les remplacer sans changer le contrat.
async fn classify_request(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ClassifyInput>,
) -> Result<Json<ClassifyOutput>, ApiError> {
    let text = input.prompt.to_lowercase();
    let asset_kind = if let Some(id) = input.input_asset_id {
        Some(read_asset_manifest(&state, id).await?.0.kind)
    } else {
        None
    };
    let asks_video = ["vidéo", "video", "anime", "mouvement"]
        .iter()
        .any(|word| text.contains(word));
    let asks_image = ["image", "photo", "illustration", "dessine"]
        .iter()
        .any(|word| text.contains(word));
    let (action, capability) = match (asset_kind.clone(), asks_video, asks_image) {
        (Some(AssetKind::Video), _, _) => ("VIDEO_TO_VIDEO", ModelCapability::VideoToVideo),
        (Some(AssetKind::Image), true, _) => ("IMAGE_TO_VIDEO", ModelCapability::ImageToVideo),
        (Some(AssetKind::Image), false, true) => ("IMAGE_TO_IMAGE", ModelCapability::ImageToImage),
        (None, true, _) => ("TEXT_TO_VIDEO", ModelCapability::TextToVideo),
        (None, false, true) => ("TEXT_TO_IMAGE", ModelCapability::TextToImage),
        _ => ("CHAT", ModelCapability::Chat),
    };
    Ok(Json(ClassifyOutput {
        action: action.into(),
        capability,
        confidence: if asks_video || asks_image || asset_kind.is_some() {
            0.92
        } else {
            0.72
        },
    }))
}

#[derive(Debug, Deserialize)]
struct ModelRouteInput {
    capability: ModelCapability,
    #[serde(default)]
    preset: String,
}
#[derive(Debug, Serialize)]
struct ModelRouteOutput {
    model: ModelView,
    score: i32,
    reason: String,
}

async fn route_model(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ModelRouteInput>,
) -> Result<Json<ModelRouteOutput>, ApiError> {
    let task = match &input.capability {
        ModelCapability::Chat => "CHAT",
        ModelCapability::Vision => "VISION",
        ModelCapability::TextToImage => "TEXT_TO_IMAGE",
        ModelCapability::ImageToImage => "IMAGE_TO_IMAGE",
        ModelCapability::Inpainting => "INPAINTING",
        ModelCapability::Outpainting => "OUTPAINTING",
        ModelCapability::ImageVariation => "IMAGE_VARIATION",
        ModelCapability::ImageUpscale => "IMAGE_UPSCALE",
        ModelCapability::ControlledImageGeneration => "CONTROLLED_IMAGE_GENERATION",
        ModelCapability::TextToVideo => "TEXT_TO_VIDEO",
        ModelCapability::ImageToVideo => "IMAGE_TO_VIDEO",
        ModelCapability::MultiImageToVideo => "MULTI_IMAGE_TO_VIDEO",
        ModelCapability::StartEndImageToVideo => "START_END_IMAGE_TO_VIDEO",
        ModelCapability::KeyframesToVideo => "KEYFRAMES_TO_VIDEO",
        ModelCapability::VideoToVideo => "VIDEO_TO_VIDEO",
        ModelCapability::VideoInpainting => "VIDEO_INPAINTING",
        ModelCapability::VideoUpscale => "VIDEO_UPSCALE",
        ModelCapability::Audio => "AUDIO",
        ModelCapability::TextToSpeech => "TEXT_TO_SPEECH",
        ModelCapability::SpeechToText => "SPEECH_TO_TEXT",
        ModelCapability::CapabilityUnknown => "UNKNOWN",
    };
    let query = CatalogQuery {
        task: Some(task.into()),
        limit: Some(60),
        ..CatalogQuery::default()
    };
    let remote = state
        .catalog
        .search(&query, false)
        .await
        .map(|result| result.models)
        .unwrap_or_default();
    let mut entries = local_runtime_models();
    entries.extend(remote);
    let mut candidates = Vec::new();
    for entry in entries
        .into_iter()
        .filter(|entry| entry.capabilities.contains(&input.capability))
    {
        let view = model_view(&state, &entry).await;
        if !view.installed || !view.compatible {
            continue;
        }
        let local_bonus = if entry.repository.starts_with("local/") {
            35
        } else {
            0
        };
        let loaded_bonus = if view.loaded { 30 } else { 0 };
        let quality_bonus = if input.preset.eq_ignore_ascii_case("quality")
            && !entry.repository.starts_with("local/")
        {
            25
        } else {
            0
        };
        candidates.push((local_bonus + loaded_bonus + quality_bonus + 40, view));
    }
    let (score, model) = candidates
        .into_iter()
        .max_by_key(|(score, _)| *score)
        .ok_or_else(|| {
            ApiError::conflict("Aucun modèle installé et compatible pour cette capacité.")
        })?;
    Ok(Json(ModelRouteOutput {
        model,
        score,
        reason: "Modèle installé, compatible et adapté au preset demandé.".into(),
    }))
}

#[derive(Debug, Deserialize)]
struct OptimizeInput {
    mode: GenerationMode,
    #[serde(default)]
    preset: String,
    #[serde(alias = "resolution")]
    quality: Option<String>,
    duration_seconds: Option<u32>,
}
#[derive(Debug, Serialize)]
struct OptimizeOutput {
    quality: String,
    duration_seconds: u32,
    device: String,
    warnings: Vec<String>,
}

async fn optimize_request(
    State(state): State<Arc<AppState>>,
    Json(input): Json<OptimizeInput>,
) -> Json<OptimizeOutput> {
    let (hardware, _) = resolve_system(state.host_agent.as_ref()).await;
    let vram = hardware.total_vram_bytes().unwrap_or_default();
    let mut quality = input.quality.unwrap_or_else(|| "480p".into());
    let mut duration = input.duration_seconds.unwrap_or(6).clamp(2, 15);
    let mut warnings = Vec::new();
    if input.preset.eq_ignore_ascii_case("fast") {
        quality = "480p".into();
        duration = duration.min(6);
    }
    if input.preset.eq_ignore_ascii_case("minimum_cost") {
        quality = "480p".into();
        duration = duration.min(4);
    }
    if matches!(
        input.mode,
        GenerationMode::TextToVideo | GenerationMode::ImageToVideo | GenerationMode::VideoToVideo
    ) && vram < 8 * 1024 * 1024 * 1024
        && quality == "1080p"
    {
        quality = "720p".into();
        warnings.push("Résolution réduite à 720p : moins de 8 Go de VRAM détectés.".into());
    }
    Json(OptimizeOutput {
        quality,
        duration_seconds: duration,
        device: if vram > 0 { "GPU".into() } else { "CPU".into() },
        warnings,
    })
}

#[derive(Debug, Serialize)]
struct ReadyStatus {
    ready: bool,
    storage_writable: bool,
    scratch_writable: bool,
    scratch_mount_ok: bool,
    scratch_filesystem: Option<String>,
    scratch_total_bytes: u64,
    scratch_available_bytes: u64,
    ffmpeg: bool,
    catalog_models: usize,
    message: String,
    profile: ApplicationProfile,
    mode: ApplicationMode,
    queue: bool,
    worker: bool,
    host_agent: bool,
    system_source: ResourceSource,
    runtime: bool,
    gpu: bool,
    s3: bool,
    errors: Vec<String>,
}

fn worker_runtime_flags(status: Option<&WorkerReady>) -> (bool, bool, bool) {
    status.map_or((false, false, false), |status| {
        (
            status.ready,
            status.runtime_available,
            status.cuda_available,
        )
    })
}

/// Readiness plus stricte que `/healthcheck` : elle vérifie que l'application
/// peut réellement écrire une sortie et lancer le moteur vidéo requis.
async fn get_ready(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ReadyStatus>) {
    let settings = state.settings.get().await;
    let sentinel = settings.outputs_dir.join(".readiness");
    let storage_writable = fs::write(&sentinel, b"ready").await.is_ok();
    let _ = fs::remove_file(sentinel).await;
    // Le Scratch de production couvre les poids, le cache et les fichiers de
    // travail. Chacun est testé, car un seul montage en lecture seule suffit à
    // faire échouer une installation ou une inférence.
    let mut scratch_writable = true;
    for directory in [
        &settings.models_dir,
        &settings.cache_dir,
        &settings.work_dir,
    ] {
        let sentinel = directory.join(".readiness");
        scratch_writable &= fs::write(&sentinel, b"ready").await.is_ok();
        let _ = fs::remove_file(sentinel).await;
    }
    let ffmpeg = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success());
    let queue = state.job_store.ping().await;
    let mut errors = Vec::new();
    let host_health = match &state.host_agent {
        Some(client) => match client.health().await {
            Ok(()) => true,
            Err(error) => {
                errors.push(error);
                false
            }
        },
        None => false,
    };
    let (system_profile, system_error) = resolve_system(state.host_agent.as_ref()).await;
    let host_metrics = system_profile.source == ResourceSource::Host;
    if state.host_agent.is_some()
        && let Some(error) = system_error
    {
        errors.push(error);
    }
    let worker_status = match &state.worker {
        Some(worker) => match worker.health().await {
            Ok(health) if health.status == "ok" && health.service == "vidioai-gpu-worker" => {
                match worker.ready().await {
                    Ok(status) => Some(status),
                    Err(error) => {
                        errors.push(format!("Worker non prêt : {error}"));
                        None
                    }
                }
            }
            Ok(_) => {
                errors.push("Le service worker a répondu avec une identité inattendue.".into());
                None
            }
            Err(error) => {
                errors.push(format!("Worker indisponible : {error}"));
                None
            }
        },
        None => None,
    };
    let worker_required = state.profile == ApplicationProfile::GpuProduction;
    let (worker_ready, runtime, gpu) = worker_runtime_flags(worker_status.as_ref());
    let worker_scratch_mount_ok = worker_status
        .as_ref()
        .is_some_and(|status| status.scratch_mount_ok);
    let s3 = if state.object_storage.enabled() {
        match state.object_storage.health().await {
            Ok(()) => true,
            Err(error) => {
                errors.push(format!("Stockage objet indisponible : {error}"));
                false
            }
        }
    } else {
        true
    };
    if !storage_writable {
        errors.push("Le volume de sorties n'est pas inscriptible.".into());
    }
    if !scratch_writable {
        errors.push("Le Scratch modèles/cache/travail n'est pas entièrement inscriptible.".into());
    }
    if !ffmpeg {
        errors.push("FFmpeg est absent ou inutilisable.".into());
    }
    if !queue {
        errors.push("La file SQLite est indisponible.".into());
    }
    if worker_required && !worker_ready {
        errors.push("Le worker GPU de production n'est pas prêt.".into());
    }
    if worker_required && !gpu {
        errors.push("CUDA/NVIDIA est obligatoire dans GPU_PRODUCTION.".into());
    }
    if worker_required && !worker_scratch_mount_ok {
        errors.push("Le Worker n'atteste pas le filesystem Scratch dédié.".into());
    }
    if state.profile == ApplicationProfile::GpuProduction && (!host_health || !host_metrics) {
        errors.push("Le Host Agent natif est obligatoire dans GPU_PRODUCTION.".into());
    }
    let host_nvidia = system_profile.physical_nvidia().is_some();
    if state.profile == ApplicationProfile::GpuProduction && !host_nvidia {
        errors.push("Aucun GPU NVIDIA physique n'est détecté par le Host Agent.".into());
    }
    if host_nvidia && !gpu {
        errors.push("GPU détecté sur l'hôte mais non exposé au GPU Worker.".into());
    }
    if state.profile == ApplicationProfile::GpuProduction
        && std::env::var("VIDIOAI_ADMIN_TOKEN").map_or(true, |value| {
            value.len() < 32 || value.starts_with("replace-with")
        })
    {
        errors.push(
            "VIDIOAI_ADMIN_TOKEN doit être un secret aléatoire d'au moins 32 caractères.".into(),
        );
    }
    let ready = storage_writable
        && scratch_writable
        && ffmpeg
        && queue
        && s3
        && (state.host_agent.is_none() || (host_health && host_metrics))
        && (state.profile != ApplicationProfile::GpuProduction || host_nvidia)
        && (!worker_required || (worker_ready && runtime && gpu && worker_scratch_mount_ok));
    let payload = ReadyStatus {
        ready,
        storage_writable,
        scratch_writable,
        scratch_mount_ok: worker_scratch_mount_ok,
        scratch_filesystem: worker_status
            .as_ref()
            .and_then(|status| status.scratch_filesystem.clone()),
        scratch_total_bytes: worker_status
            .as_ref()
            .map_or(0, |status| status.scratch_total_bytes),
        scratch_available_bytes: worker_status
            .as_ref()
            .map_or(0, |status| status.scratch_available_bytes),
        ffmpeg,
        catalog_models: local_runtime_models().len(),
        message: if ready {
            "VidioAI est prêt à accepter des jobs.".into()
        } else {
            "Une ou plusieurs dépendances obligatoires ne sont pas prêtes.".into()
        },
        profile: state.profile,
        mode: *state.mode.read().await,
        queue,
        worker: worker_ready,
        host_agent: host_health && host_metrics,
        system_source: system_profile.source,
        runtime,
        gpu,
        s3,
        errors,
    };
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(payload),
    )
}

/// Liveness volontairement minimale : si cette route répond, le processus
/// Axum et son ordonnanceur fonctionnent. Les dépendances appartiennent à
/// `/api/ready`, afin de ne pas provoquer de redémarrages en boucle.
async fn get_api_health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "vidioai-backend",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[derive(Debug, Serialize)]
struct SystemView {
    #[serde(flatten)]
    snapshot: HostSnapshot,
    profile: ApplicationProfile,
    host_agent_error: Option<String>,
}

/// Le navigateur ne contacte jamais directement le service natif. Le backend
/// sélectionne l'hôte ou le fallback conteneur et joint le diagnostic éventuel.
async fn get_system(State(state): State<Arc<AppState>>) -> Json<SystemView> {
    let (snapshot, host_agent_error) = resolve_system(state.host_agent.as_ref()).await;
    Json(SystemView {
        snapshot,
        profile: state.profile,
        host_agent_error,
    })
}

#[derive(Debug, Serialize)]
struct ResourcesView {
    measured_at: u64,
    profile: ApplicationProfile,
    mode: ApplicationMode,
    system: HostSnapshot,
    system_error: Option<String>,
    worker: Option<WorkerResources>,
    worker_source: Option<ResourceSource>,
    worker_error: Option<String>,
    queue_active: usize,
    queue_total: usize,
    loaded_models: usize,
    object_storage_enabled: bool,
}

/// Vue de ressources sans valeurs inventées. En particulier, si le worker ou
/// NVIDIA est absent, `worker` vaut `null` et l'erreur explique la cause.
async fn get_resources(State(state): State<Arc<AppState>>) -> Json<ResourcesView> {
    let (system, system_error) = resolve_system(state.host_agent.as_ref()).await;
    let (worker, worker_error) = match &state.worker {
        Some(client) => match client.resources().await {
            Ok(resources) => (Some(resources), None),
            Err(error) => (None, Some(error)),
        },
        None => (None, Some("VIDIOAI_WORKER_URL non configurée".into())),
    };
    let jobs = state.jobs.read().await;
    Json(ResourcesView {
        measured_at: unix_now(),
        profile: state.profile,
        mode: *state.mode.read().await,
        system,
        system_error,
        worker_source: worker.as_ref().map(|_| ResourceSource::Worker),
        worker,
        worker_error,
        queue_active: jobs
            .values()
            .filter(|job| {
                matches!(
                    job.status,
                    JobStatus::Queued
                        | JobStatus::Dispatching
                        | JobStatus::Running
                        | JobStatus::SavingOutput
                )
            })
            .count(),
        queue_total: jobs.len(),
        loaded_models: state.runtime.read().await.len(),
        object_storage_enabled: state.object_storage.enabled(),
    })
}

fn authorize_admin(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let expected = std::env::var("VIDIOAI_ADMIN_TOKEN").ok();
    if expected.is_none() && state.profile == ApplicationProfile::Local {
        return Ok(());
    }
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if supplied == expected.as_deref() {
        Ok(())
    } else {
        Err(ApiError::unauthorized("Jeton administrateur invalide."))
    }
}

async fn set_application_mode(
    state: Arc<AppState>,
    headers: HeaderMap,
    mode: ApplicationMode,
) -> Result<Json<serde_json::Value>, ApiError> {
    authorize_admin(&state, &headers)?;
    *state.mode.write().await = mode;
    state.emit("application.mode", &mode);
    Ok(Json(
        json!({ "mode": mode, "active_jobs": state.queue().await.len() }),
    ))
}

async fn start_drain(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    set_application_mode(state, headers, ApplicationMode::Draining).await
}

async fn resume_jobs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    set_application_mode(state, headers, ApplicationMode::AcceptingJobs).await
}

async fn stop_jobs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    set_application_mode(state, headers, ApplicationMode::Stopping).await
}

#[derive(Debug, Serialize)]
struct DashboardView {
    generations_total: usize,
    generations_running: usize,
    images_created: usize,
    videos_created: usize,
    storage_bytes: u64,
    installed_models: usize,
    loaded_models: usize,
    recent_generations: Vec<Generation>,
}

/// Agrégat unique pour éviter la multiplication des appels sur l'accueil.
async fn get_dashboard(State(state): State<Arc<AppState>>) -> Json<DashboardView> {
    let generations = state.generations.read().await;
    let mut recent: Vec<_> = generations.values().cloned().collect();
    recent.sort_by_key(|item| std::cmp::Reverse(item.created_at));
    recent.truncate(6);
    let images_created = generations
        .values()
        .filter(|item| item.kind == AssetKind::Image && item.status == GenerationStatus::Completed)
        .count();
    let videos_created = generations
        .values()
        .filter(|item| item.kind == AssetKind::Video && item.status == GenerationStatus::Completed)
        .count();
    let running = generations
        .values()
        .filter(|item| {
            matches!(
                item.status,
                GenerationStatus::Queued | GenerationStatus::Running
            )
        })
        .count();
    let total = generations.len();
    drop(generations);

    let settings = state.settings.get().await;
    let mut storage_bytes = 0;
    if let Ok(mut entries) = fs::read_dir(settings.outputs_dir.join("assets")).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if let Ok(metadata) = entry.metadata().await
                && metadata.is_file()
                && entry.path().extension().and_then(|value| value.to_str()) != Some("json")
            {
                storage_bytes += metadata.len();
            }
        }
    }
    // Les deux moteurs intégrés sont toujours présents. Chaque autre dossier
    // n'est compté que s'il contient le pointeur atomique du worker.
    let mut installed_models = local_runtime_models().len();
    if let Ok(mut entries) = fs::read_dir(&settings.models_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if fs::metadata(entry.path().join("active.json")).await.is_ok() {
                installed_models += 1;
            }
        }
    }
    Json(DashboardView {
        generations_total: total,
        generations_running: running,
        images_created,
        videos_created,
        storage_bytes,
        installed_models,
        loaded_models: state.runtime.read().await.len(),
        recent_generations: recent,
    })
}

async fn persist_projects(state: &AppState) -> Result<(), ApiError> {
    let mut projects: Vec<_> = state.projects.read().await.values().cloned().collect();
    projects.sort_by_key(|project| project.created_at);
    let path = state.settings.get().await.outputs_dir.join("projects.json");
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&projects).map_err(ApiError::internal)?,
    )
    .await
    .map_err(ApiError::internal)?;
    fs::rename(temporary, path)
        .await
        .map_err(ApiError::internal)
}

async fn list_projects(State(state): State<Arc<AppState>>) -> Json<Vec<Project>> {
    let mut projects: Vec<_> = state.projects.read().await.values().cloned().collect();
    projects.sort_by_key(|project| std::cmp::Reverse(project.updated_at));
    Json(projects)
}

async fn create_project(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ProjectInput>,
) -> Result<(StatusCode, Json<Project>), ApiError> {
    let name = input.name.trim();
    if name.len() < 2 || name.len() > 80 {
        return Err(ApiError::bad_request(
            "Le nom doit contenir entre 2 et 80 caractères.",
        ));
    }
    let project = Project {
        id: Uuid::new_v4(),
        name: name.into(),
        description: input.description.trim().into(),
        asset_ids: input.asset_ids,
        generation_ids: input.generation_ids,
        chat_ids: input.chat_ids,
        created_at: unix_now(),
        updated_at: unix_now(),
    };
    state
        .projects
        .write()
        .await
        .insert(project.id, project.clone());
    persist_projects(&state).await?;
    Ok((StatusCode::CREATED, Json(project)))
}

async fn get_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Project>, ApiError> {
    state
        .projects
        .read()
        .await
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| ApiError::not_found("Projet inconnu."))
}

async fn update_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(input): Json<ProjectInput>,
) -> Result<Json<Project>, ApiError> {
    let mut projects = state.projects.write().await;
    let project = projects
        .get_mut(&id)
        .ok_or_else(|| ApiError::not_found("Projet inconnu."))?;
    let name = input.name.trim();
    if name.len() < 2 || name.len() > 80 {
        return Err(ApiError::bad_request(
            "Le nom doit contenir entre 2 et 80 caractères.",
        ));
    }
    project.name = name.into();
    project.description = input.description.trim().into();
    project.asset_ids = input.asset_ids;
    project.generation_ids = input.generation_ids;
    project.chat_ids = input.chat_ids;
    project.updated_at = unix_now();
    let result = project.clone();
    drop(projects);
    persist_projects(&state).await?;
    Ok(Json(result))
}

async fn delete_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    state
        .projects
        .write()
        .await
        .remove(&id)
        .ok_or_else(|| ApiError::not_found("Projet inconnu."))?;
    // Les assets et générations ne sont pas supprimés : ils peuvent appartenir
    // à plusieurs projets et restent récupérables depuis l'historique global.
    persist_projects(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn persist_chats(state: &AppState) -> Result<(), ApiError> {
    let mut chats: Vec<_> = state.chats.read().await.values().cloned().collect();
    chats.sort_by_key(|chat| chat.created_at);
    let path = state.settings.get().await.outputs_dir.join("chats.json");
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&chats).map_err(ApiError::internal)?,
    )
    .await
    .map_err(ApiError::internal)?;
    fs::rename(temporary, path)
        .await
        .map_err(ApiError::internal)
}

async fn list_chats(State(state): State<Arc<AppState>>) -> Json<Vec<ChatSession>> {
    let mut chats: Vec<_> = state.chats.read().await.values().cloned().collect();
    chats.sort_by_key(|chat| std::cmp::Reverse(chat.updated_at));
    Json(chats)
}

async fn create_chat(
    State(state): State<Arc<AppState>>,
    Json(input): Json<CreateChatInput>,
) -> Result<(StatusCode, Json<ChatSession>), ApiError> {
    let model_id = input
        .model_id
        .unwrap_or_else(|| "Qwen/Qwen2.5-0.5B-Instruct".into());
    let model = resolve_model(&state, &model_id).await?;
    if !model.capabilities.contains(&ModelCapability::Chat) {
        return Err(ApiError::bad_request("Modèle de chat inconnu."));
    }
    let chat = ChatSession {
        id: Uuid::new_v4(),
        title: input
            .title
            .unwrap_or_else(|| "Nouvelle conversation".into()),
        model_id,
        messages: Vec::new(),
        created_at: unix_now(),
        updated_at: unix_now(),
    };
    state.chats.write().await.insert(chat.id, chat.clone());
    persist_chats(&state).await?;
    Ok((StatusCode::CREATED, Json(chat)))
}

async fn get_chat(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ChatSession>, ApiError> {
    state
        .chats
        .read()
        .await
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| ApiError::not_found("Conversation inconnue."))
}

async fn delete_chat(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    state
        .chats
        .write()
        .await
        .remove(&id)
        .ok_or_else(|| ApiError::not_found("Conversation inconnue."))?;
    persist_chats(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn send_chat_message(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(input): Json<SendMessageInput>,
) -> Result<Json<ChatTurn>, ApiError> {
    let content = input.content.trim();
    if content.is_empty() || content.len() > 4000 {
        return Err(ApiError::bad_request(
            "Le message doit contenir entre 1 et 4000 caractères.",
        ));
    }
    for asset_id in &input.attachment_ids {
        read_asset_manifest(&state, *asset_id).await?;
    }

    let (model_id, history) = {
        let chats = state.chats.read().await;
        let chat = chats
            .get(&id)
            .ok_or_else(|| ApiError::not_found("Conversation inconnue."))?;
        let history = chat
            .messages
            .iter()
            .rev()
            .take(8)
            .rev()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n");
        (chat.model_id.clone(), history)
    };
    let user = ChatMessage {
        id: Uuid::new_v4(),
        role: "user".into(),
        content: content.into(),
        attachment_ids: input.attachment_ids,
        created_at: unix_now(),
    };

    // Le même routeur que l'API publique détecte les demandes créatives. Le
    // chat renvoie alors une action exploitable par l'interface, sans lancer une
    // génération coûteuse sans confirmation explicite de l'utilisateur.
    let lower = content.to_lowercase();
    let asks_video = ["vidéo", "video", "anime", "mouvement"]
        .iter()
        .any(|word| lower.contains(word));
    let asks_image = ["image", "photo", "illustration", "dessine"]
        .iter()
        .any(|word| lower.contains(word));
    let suggested_action = if asks_video {
        Some("TEXT_TO_VIDEO".to_string())
    } else if asks_image {
        Some("TEXT_TO_IMAGE".to_string())
    } else {
        None
    };

    let answer = if let Some(action) = &suggested_action {
        format!(
            "J’ai préparé une action {action}. Vérifiez le prompt et les paramètres, puis confirmez la génération dans le studio."
        )
    } else {
        let repository = resolve_model(&state, &model_id)
            .await
            .map(|model| model.repository)
            .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct".into());
        let complete_prompt = if history.is_empty() {
            content.to_string()
        } else {
            format!("Contexte récent :\n{history}\n\nUtilisateur : {content}")
        };
        text_messages_ia(repository, complete_prompt)
            .await
            .map_err(ApiError::internal)?
    };
    let assistant = ChatMessage {
        id: Uuid::new_v4(),
        role: "assistant".into(),
        content: answer,
        attachment_ids: Vec::new(),
        created_at: unix_now(),
    };
    {
        let mut chats = state.chats.write().await;
        let chat = chats
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found("Conversation inconnue."))?;
        chat.messages.push(user.clone());
        chat.messages.push(assistant.clone());
        if chat.title == "Nouvelle conversation" {
            chat.title = content.chars().take(48).collect();
        }
        chat.updated_at = unix_now();
    }
    persist_chats(&state).await?;
    state.emit(
        "chat.message.completed",
        &json!({ "chat_id": id, "message": assistant }),
    );
    Ok(Json(ChatTurn {
        user,
        assistant,
        suggested_action,
    }))
}

async fn get_settings(State(state): State<Arc<AppState>>) -> Json<AppSettings> {
    Json(state.settings.get().await)
}

async fn put_settings(
    State(state): State<Arc<AppState>>,
    Json(settings): Json<AppSettings>,
) -> Result<Json<AppSettings>, ApiError> {
    state.settings.replace(settings).await.map(Json)
}

#[derive(Debug, Deserialize)]
struct InstalledPointer {
    revision: String,
    #[serde(default)]
    repository: Option<String>,
}

/// Résout d'abord les deux moteurs internes, puis interroge le Hub. Cette
/// routine unique évite que les routes détail/installation/génération divergent.
async fn resolve_model(state: &AppState, id: &str) -> Result<CatalogEntry, ApiError> {
    if let Some(model) = local_runtime_models()
        .into_iter()
        .find(|model| model.id == id)
    {
        return Ok(model);
    }
    state
        .catalog
        .model(id, false)
        .await
        .map_err(ApiError::unavailable)?
        .models
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::not_found("Modèle Hugging Face inconnu."))
}

/// Lit le pointeur écrit atomiquement par le worker. Cette vérification locale
/// évite un appel HTTP au worker pour chaque carte d'une page de catalogue.
async fn installed_revision(state: &AppState, entry: &CatalogEntry) -> Option<String> {
    if entry.local {
        return Some(entry.revision.clone());
    }
    let path = state
        .settings
        .get()
        .await
        .models_dir
        .join(&entry.storage_id)
        .join("active.json");
    fs::read(path)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<InstalledPointer>(&bytes).ok())
        .map(|pointer| pointer.revision)
}

/// Complète l'observation interne avec l'identité publique du modèle. Cette
/// séparation empêche le worker d'avoir à connaître les conventions du Hub.
async fn record_worker_benchmark(
    state: &AppState,
    model_id: &str,
    revision: &str,
    observation: &WorkerBenchmarkObservation,
) {
    let benchmark = HardwareBenchmark {
        model_id: model_id.to_owned(),
        revision: revision.to_owned(),
        gpu: observation.gpu.clone(),
        vram_idle_bytes: observation.vram_idle_bytes,
        vram_after_load_bytes: observation.vram_after_load_bytes,
        vram_peak_bytes: observation.vram_peak_bytes,
        ram_peak_bytes: observation.ram_peak_bytes,
        runtime: observation.runtime.clone(),
        precision: observation.precision.clone(),
        resolution_width: observation.resolution_width,
        resolution_height: observation.resolution_height,
        frames: observation.frames,
        duration_seconds: observation.duration_seconds,
        fps: observation.fps,
        batch: observation.batch,
        attention_implementation: observation.attention_implementation.clone(),
        vae_tiling: observation.vae_tiling,
        cpu_offload: observation.cpu_offload,
        model_offload: observation.model_offload,
        inference_seconds: observation.inference_seconds,
        measured_at: unix_now(),
    };
    if let Err(error) = state.hardware_benchmark_store.record(&benchmark).await {
        // Une mesure non persistée ne doit jamais faire échouer un chargement ou
        // une génération déjà réussis. L'erreur reste toutefois observable.
        eprintln!("Benchmark matériel non persisté : {error}");
    }
}

/// Instantané partagé par toutes les cartes d'une même réponse catalogue. Sans
/// ce contexte, une page de 20 modèles interrogerait 20 fois le Host Agent et
/// le Worker, avec des résultats susceptibles de varier au milieu de la liste.
struct ModelMachineContext {
    profile: HostSnapshot,
    runtime_available: bool,
    cuda_available: bool,
    available_ram_bytes: u64,
    available_vram_bytes: u64,
}

async fn model_machine_context(state: &AppState) -> ModelMachineContext {
    let host = resolve_system(state.host_agent.as_ref());
    let worker = async {
        let Some(worker) = &state.worker else {
            return (None, None);
        };
        // La disponibilité globale CUDA ne dépend jamais de l'installation du
        // modèle affiché. C'était la cause des faux « non supporté » du catalogue.
        let (ready, resources) = tokio::join!(worker.ready(), worker.resources());
        (ready.ok(), resources.ok())
    };
    let ((profile, _), (ready, resources)) = tokio::join!(host, worker);
    let available_ram_bytes = profile
        .ram
        .available_bytes
        .or(profile.ram.total_bytes)
        .unwrap_or(u64::MAX);
    let available_vram_bytes = resources
        .as_ref()
        .and_then(|value| value.gpu.as_ref())
        .map(|gpu| gpu.vram_total_bytes.saturating_sub(gpu.vram_used_bytes))
        .or_else(|| {
            profile
                .physical_nvidia()
                .and_then(|gpu| gpu.vram_available_bytes.or(gpu.vram_total_bytes))
        })
        .unwrap_or_default();
    ModelMachineContext {
        profile,
        runtime_available: ready
            .as_ref()
            .is_some_and(|status| status.runtime_available),
        cuda_available: ready
            .as_ref()
            .is_some_and(|status| status.ready && status.cuda_available),
        available_ram_bytes,
        available_vram_bytes,
    }
}

/// Enrichit les métadonnées publiques avec l'état du disque, du runtime et les
/// ressources réellement disponibles sur la machine courante.
async fn model_view(state: &AppState, entry: &CatalogEntry) -> ModelView {
    let machine = model_machine_context(state).await;
    model_view_with_machine(state, entry, &machine).await
}

fn model_input_profile(runtime_capabilities: &[ModelCapability]) -> ModelInputProfile {
    let supports_start_end = runtime_capabilities.contains(&ModelCapability::StartEndImageToVideo);
    let supports_multi_reference = runtime_capabilities
        .contains(&ModelCapability::MultiImageToVideo)
        || runtime_capabilities.contains(&ModelCapability::KeyframesToVideo);

    if supports_start_end {
        ModelInputProfile {
            min_input_images: 1,
            max_input_images: 2,
            supported_image_roles: vec![
                "start".into(),
                "end".into(),
                "start_frame".into(),
                "end_frame".into(),
            ],
            supports_start_end_frames: true,
            supports_reference_images: supports_multi_reference,
            supports_keyframes: runtime_capabilities.contains(&ModelCapability::KeyframesToVideo),
        }
    } else if supports_multi_reference {
        ModelInputProfile {
            min_input_images: 1,
            max_input_images: 8,
            supported_image_roles: vec!["reference".into(), "keyframe".into()],
            supports_start_end_frames: false,
            supports_reference_images: true,
            supports_keyframes: true,
        }
    } else {
        ModelInputProfile {
            min_input_images: 1,
            max_input_images: 1,
            supported_image_roles: Vec::new(),
            supports_start_end_frames: false,
            supports_reference_images: false,
            supports_keyframes: false,
        }
    }
}

fn worker_reports_ready(status: &crate::worker::WorkerModelStatus) -> bool {
    status.state == "READY"
        && status.weights_valid
        && status.runtime_available
        && status.runtime_compatible
}

async fn model_view_with_machine(
    state: &AppState,
    entry: &CatalogEntry,
    machine: &ModelMachineContext,
) -> ModelView {
    let runtime_check = if entry.local || !machine.runtime_available {
        None
    } else if let Some(worker) = &state.worker {
        let cache_key = format!("{}@{}", entry.id, entry.revision);
        let cached = state
            .compatibility_cache
            .read()
            .await
            .get(&cache_key)
            .filter(|(created_at, _)| unix_now().saturating_sub(*created_at) < 300)
            .map(|(_, compatibility)| compatibility.clone());
        if cached.is_some() {
            cached
        } else {
            let compatibility = worker
                .compatibility(
                    entry.pipeline_class.as_deref(),
                    entry.library.as_deref(),
                    entry.pipeline_tag.as_deref(),
                    &entry.tags,
                )
                .await
                .ok();
            if let Some(value) = &compatibility {
                state
                    .compatibility_cache
                    .write()
                    .await
                    .insert(cache_key, (unix_now(), value.clone()));
            }
            compatibility
        }
    } else {
        None
    };
    let runtime_compatibility = if entry.local {
        "SUPPORTED".to_owned()
    } else if let Some(check) = &runtime_check {
        match check.compatibility_status.as_str() {
            "SUPPORTED" | "UNKNOWN" | "UNSUPPORTED" => check.compatibility_status.clone(),
            _ if check.runtime_supported => "SUPPORTED".into(),
            _ => "UNKNOWN".into(),
        }
    } else {
        "UNKNOWN".into()
    };
    let runtime_supported = runtime_compatibility == "SUPPORTED";
    let runtime_allowed = runtime_compatibility != "UNSUPPORTED";
    let runtime_capabilities = if entry.local {
        entry.runtime_capabilities.clone()
    } else {
        runtime_check
            .as_ref()
            .map(|check| {
                check
                    .runtime_capabilities
                    .iter()
                    .filter_map(|value| {
                        serde_json::from_value::<ModelCapability>(serde_json::Value::String(
                            value.clone(),
                        ))
                        .ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let runtime_reason = if entry.local {
        entry.runtime_reason.clone()
    } else if let Some(check) = &runtime_check {
        check.runtime_reason.clone()
    } else if !machine.runtime_available {
        "RUNTIME_UNAVAILABLE: le Worker Diffusers n'est pas disponible.".into()
    } else {
        "PIPELINE_CLASS_NOT_AVAILABLE: compatibility check Worker impossible.".into()
    };
    let pipeline_class = runtime_check
        .as_ref()
        .and_then(|check| check.pipeline_class.clone())
        .or_else(|| entry.pipeline_class.clone());
    let installed_revision = installed_revision(state, entry).await;
    let downloaded = entry.local || installed_revision.is_some();
    let worker_status = if !entry.local && downloaded {
        if let Some(worker) = &state.worker {
            worker.model_status(&entry.storage_id).await.ok()
        } else {
            None
        }
    } else {
        None
    };
    let installed = if entry.local {
        true
    } else {
        worker_status
            .as_ref()
            .is_some_and(|status| downloaded && status.installed && status.weights_valid)
    };
    let runtime_dependencies = worker_status
        .as_ref()
        .map(|status| status.runtime_dependencies.clone())
        .unwrap_or_default();
    let runtime_precision = worker_status
        .as_ref()
        .and_then(|status| status.precision_plan.clone());
    let cache_job = state
        .jobs
        .read()
        .await
        .values()
        .filter(|job| job.target_id == entry.id && job.cache_status.is_some())
        .max_by_key(|job| job.updated_at)
        .cloned();
    let cache_status = cache_job
        .as_ref()
        .and_then(|job| job.cache_status.clone())
        .unwrap_or_else(|| {
            if state.object_storage.enabled() && installed {
                "CACHE_UNKNOWN".into()
            } else {
                "CACHE_DISABLED".into()
            }
        });
    let cache_error = cache_job.and_then(|job| job.cache_error);
    let runtime_ready = entry.local || worker_status.as_ref().is_some_and(worker_reports_ready);
    let available_ram = machine.available_ram_bytes;
    let available_vram = machine.available_vram_bytes;
    // Un benchmark attaché à cette révision remplace l'estimation du Hub. Une
    // mesure d'une ancienne révision ne doit jamais être réutilisée.
    let mut hardware_estimate = entry.hardware.clone();
    if let Ok(Some(benchmark)) = state
        .hardware_benchmark_store
        .latest(&entry.id, &entry.revision)
        .await
    {
        hardware_estimate = HardwareEstimator::with_benchmark(hardware_estimate, benchmark);
    }
    hardware_estimate = HardwareEstimator::with_machine(
        hardware_estimate,
        Some(CurrentMachine {
            ram_bytes: available_ram,
            vram_bytes: available_vram,
            // La présence physique CUDA ne suffit pas pour exécuter le modèle.
            // Seul le worker atteste que le runtime est réellement accessible.
            cuda_available: machine.cuda_available,
        }),
    );
    let recommended = entry.variants.iter().find(|variant| {
        variant.ram_required <= available_ram
            && (variant.vram_required == 0 || variant.vram_required <= available_vram)
    });
    let storage_compatible = machine
        .profile
        .storage
        .available_bytes
        .is_none_or(|available| {
            entry
                .estimated_size_bytes
                // Téléchargement + snapshot temporaire + marge de validation.
                .is_none_or(|size| size.saturating_mul(2) <= available)
        });
    let mut hardware_compatible = hardware_estimate
        .compatible_with_current_machine
        .unwrap_or(false);
    if !storage_compatible {
        hardware_compatible = false;
        hardware_estimate.compatible_with_current_machine = Some(false);
        hardware_estimate.compatibility_level = "UNSUPPORTED".into();
        hardware_estimate
            .notes
            .push("Espace disque insuffisant pour le snapshot et sa validation atomique.".into());
    }
    let compatibility_level = hardware_estimate.compatibility_level.clone();
    let installation_state = if entry.local {
        "READY".to_owned()
    } else if runtime_compatibility == "UNSUPPORTED" {
        "RUNTIME_UNAVAILABLE".to_owned()
    } else if runtime_compatibility == "UNKNOWN" && !downloaded {
        "COMPATIBILITY_UNKNOWN".to_owned()
    } else if runtime_ready {
        "READY".to_owned()
    } else if worker_status
        .as_ref()
        .is_some_and(|status| status.state.eq_ignore_ascii_case("FAILED"))
    {
        "FAILED".to_owned()
    } else if installed {
        worker_status
            .as_ref()
            .map(|status| status.state.clone())
            .unwrap_or_else(|| "INSTALLED".into())
    } else if downloaded {
        "DOWNLOADED".to_owned()
    } else {
        "NOT_INSTALLED".into()
    };

    let runtime_detail = runtime_reason.clone();
    // Les résultats de liste HF ne prouvent ni l'autorisation ni son absence.
    // Ils restent installables jusqu'au fetch exact effectué par resolve_model.
    let access_ok = entry.access_authorized || !entry.access_checked;
    let compatibility_checks = vec![
        CompatibilityCheck {
            key: "source",
            label: "Source Hugging Face",
            ok: entry.source_available,
            detail: if entry.source_available {
                format!("Repository {} accessible.", entry.repository)
            } else {
                "Métadonnées ou fichiers sources indisponibles.".into()
            },
        },
        CompatibilityCheck {
            key: "access",
            label: "Droits d'accès",
            ok: access_ok,
            detail: if entry.access_authorized && (entry.gated || entry.private) {
                "HF_TOKEN autorisé : les fichiers de configuration sont accessibles.".into()
            } else if !entry.access_checked && (entry.gated || entry.private) {
                "Accès gated/privé non vérifié dans la liste ; contrôle exact avant téléchargement."
                    .into()
            } else if entry.gated {
                "Repository gated : HF_TOKEN doit disposer de l'autorisation.".into()
            } else if entry.private {
                "Repository privé : HF_TOKEN autorisé requis.".into()
            } else {
                "Repository public.".into()
            },
        },
        CompatibilityCheck {
            key: "hardware",
            label: "Configuration matérielle",
            ok: hardware_compatible,
            detail: format!(
                "VRAM disponible {:.1} Go · RAM disponible {:.1} Go{}.",
                available_vram as f64 / 1_073_741_824.0,
                available_ram as f64 / 1_073_741_824.0,
                if storage_compatible {
                    ""
                } else {
                    " · espace disque insuffisant"
                }
            ),
        },
        CompatibilityCheck {
            key: "runtime",
            label: "Pipeline VidioAI",
            ok: runtime_allowed && (entry.local || machine.runtime_available),
            detail: runtime_detail,
        },
        CompatibilityCheck {
            key: "files",
            label: "Fichiers requis",
            ok: entry.quality_valid,
            detail: if entry.quality_valid {
                "Manifest et poids détectés dans le repository.".into()
            } else {
                "Manifest Diffusers ou poids Safetensors incomplets.".into()
            },
        },
    ];
    let compatible = hardware_compatible
        && runtime_allowed
        && entry.source_available
        && entry.quality_valid
        && (entry.local || machine.runtime_available);

    let display_capabilities = if runtime_capabilities.is_empty() {
        entry.capabilities.clone()
    } else {
        runtime_capabilities.clone()
    };
    ModelView {
        id: entry.id.clone(),
        name: entry.name.clone(),
        description: entry.description.clone(),
        kind: entry.kind.clone(),
        capabilities: display_capabilities.clone(),
        declared_capabilities: entry.capabilities.clone(),
        display_capabilities,
        variants: entry.variants.clone(),
        installed,
        cache_status,
        cache_error,
        runtime_dependencies,
        runtime_precision,
        runtime_ready,
        installation_state,
        compatible,
        recommended_variant: recommended.map(|variant| variant.id.clone()),
        loaded: state.runtime.read().await.contains_key(&entry.id),
        repository: entry.repository.clone(),
        repository_url: if entry.local {
            String::new()
        } else {
            format!("https://huggingface.co/{}", entry.repository)
        },
        license: entry
            .license
            .clone()
            .unwrap_or_else(|| "Non renseignée".into()),
        engine: entry
            .runtime_name
            .clone()
            .unwrap_or_else(|| "Non supporté".into()),
        engine_type: if entry.local { "procedural" } else { "ai" }.into(),
        runtime_supported,
        runtime_compatibility: runtime_compatibility.clone(),
        runtime_reason,
        pipeline_class,
        runtime_capabilities: runtime_capabilities.clone(),
        input_profile: model_input_profile(&runtime_capabilities),
        vidioai_supported: runtime_supported,
        source_available: entry.source_available,
        hardware_compatible,
        available_ram_bytes: available_ram,
        available_vram_bytes: available_vram,
        installable: entry.source_available
            && entry.quality_valid
            && runtime_allowed
            && compatible
            && access_ok,
        compatibility_checks,
        accessibility: entry.accessibility.clone(),
        access_authorized: entry.access_authorized,
        access_checked: entry.access_checked,
        gated: entry.gated,
        private: entry.private,
        author: entry.author.clone(),
        revision: entry.revision.clone(),
        update_available: installed_revision
            .as_deref()
            .is_some_and(|revision| revision != entry.revision),
        installed_revision,
        last_modified: entry.last_modified.clone(),
        pipeline_tag: entry.pipeline_tag.clone(),
        tags: entry.tags.clone(),
        library: entry.library.clone(),
        architecture: entry.architecture.clone(),
        files: entry.files.clone(),
        estimated_size_bytes: entry.estimated_size_bytes,
        downloads: entry.downloads,
        likes: entry.likes,
        quality_valid: entry.quality_valid,
        compatibility_level,
        hardware: hardware_estimate,
    }
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    items: Vec<ModelView>,
    page: usize,
    limit: usize,
    has_more: bool,
    total: usize,
    stale: bool,
    last_sync: Option<u64>,
    source: &'static str,
}

fn entry_matches_query(entry: &CatalogEntry, query: &CatalogQuery) -> bool {
    if let Some(search) = query
        .search
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let needle = search
            .trim_start_matches("https://huggingface.co/")
            .to_ascii_lowercase();
        if !format!("{} {} {}", entry.id, entry.name, entry.description)
            .to_ascii_lowercase()
            .contains(&needle)
        {
            return false;
        }
    }
    if let Some(author) = query.author.as_deref()
        && !entry
            .author
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(author))
    {
        return false;
    }
    if let Some(category) = query.category.as_deref()
        && !format!("{:?}", entry.kind).eq_ignore_ascii_case(category)
    {
        return false;
    }
    if let Some(task) = query.task.as_deref() {
        let serialized = serde_json::to_value(&entry.capabilities).unwrap_or_default();
        if !serialized.as_array().is_some_and(|items| {
            items.iter().any(|item| {
                item.as_str()
                    .is_some_and(|value| value.eq_ignore_ascii_case(task))
            })
        }) {
            return false;
        }
    }
    true
}

async fn get_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CatalogQuery>,
) -> Result<Json<ModelListResponse>, ApiError> {
    let (result, source) = match state.catalog.search(&query, false).await {
        Ok(result) => (result, "huggingface"),
        Err(error) => {
            // Le catalogue distant ne doit jamais rendre les moteurs locaux
            // inutilisables. Le détail/l'installation continueront à retourner
            // une erreur précise lorsqu'un repository HF est réellement requis.
            eprintln!("Catalogue Hugging Face en mode local dégradé : {error}");
            (
                CatalogResult {
                    models: Vec::new(),
                    stale: true,
                    last_sync: None,
                },
                "local-fallback",
            )
        }
    };
    let mut entries = local_runtime_models();
    entries.extend(result.models);
    let machine = model_machine_context(&state).await;
    let matching = entries
        .iter()
        .filter(|entry| entry_matches_query(entry, &query))
        .collect::<Vec<_>>();
    // Les enrichissements Worker sont bornés : l'ouverture du catalogue ne
    // peut pas lancer des dizaines de preflights en concurrence avec une
    // génération réelle. Les décisions sont en outre mises en cache 5 minutes.
    let mut candidates = Vec::with_capacity(matching.len());
    for chunk in matching.chunks(4) {
        candidates.extend(
            futures_util::future::join_all(
                chunk
                    .iter()
                    .map(|entry| model_view_with_machine(&state, entry, &machine)),
            )
            .await,
        );
    }
    let mut views = Vec::with_capacity(candidates.len());
    for view in candidates {
        if query
            .installed
            .is_some_and(|expected| view.installed != expected)
            || query
                .compatible
                .is_some_and(|expected| view.compatible != expected)
        {
            continue;
        }
        views.push(view);
    }
    match query.sort.as_deref().unwrap_or("trending") {
        "name" => views.sort_by_key(|model| model.name.to_ascii_lowercase()),
        "downloads" => {
            views.sort_by_key(|model| std::cmp::Reverse(model.downloads.unwrap_or_default()))
        }
        "likes" => views.sort_by_key(|model| std::cmp::Reverse(model.likes.unwrap_or_default())),
        "updated" | "last_modified" => {
            views.sort_by_key(|model| std::cmp::Reverse(model.last_modified.clone()))
        }
        "compatibility" | "recommended" => views.sort_by_key(|model| {
            std::cmp::Reverse((
                model.vidioai_supported,
                model.hardware_compatible,
                model.runtime_ready,
                model.downloads.unwrap_or_default(),
            ))
        }),
        _ => {}
    }
    let page = query.page();
    let limit = query.limit();
    let offset = page.saturating_sub(1).saturating_mul(limit);
    let total = views.len();
    let has_more = offset.saturating_add(limit) < total;
    let items = views.into_iter().skip(offset).take(limit).collect();
    Ok(Json(ModelListResponse {
        items,
        page,
        limit,
        has_more,
        total,
        stale: result.stale,
        last_sync: result.last_sync,
        source,
    }))
}

async fn refresh_models(State(state): State<Arc<AppState>>) -> StatusCode {
    state.catalog.clear_cache().await;
    state.compatibility_cache.write().await.clear();
    StatusCode::NO_CONTENT
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ModelView>, ApiError> {
    get_model_by_id(&state, &id).await
}

#[derive(Debug, Deserialize)]
struct ModelIdQuery {
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct ModelActionInput {
    model_id: String,
}

async fn get_model_from_query(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ModelIdQuery>,
) -> Result<Json<ModelView>, ApiError> {
    get_model_by_id(&state, &query.model_id).await
}

async fn get_model_by_id(state: &AppState, id: &str) -> Result<Json<ModelView>, ApiError> {
    let entry = resolve_model(state, id).await?;
    Ok(Json(model_view(state, &entry).await))
}

async fn install_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<Job>), ApiError> {
    start_model_install(state, id, None).await
}

#[derive(Debug, Deserialize)]
struct InstallModelInput {
    model_id: String,
    revision: Option<String>,
}

/// Route recommandée : le repository reste dans le JSON et ne dépend donc pas
/// du traitement des slashs par un reverse proxy.
async fn install_model_from_body(
    State(state): State<Arc<AppState>>,
    Json(input): Json<InstallModelInput>,
) -> Result<(StatusCode, Json<Job>), ApiError> {
    start_model_install(state, input.model_id, input.revision).await
}

async fn start_model_install(
    state: Arc<AppState>,
    id: String,
    requested_revision: Option<String>,
) -> Result<(StatusCode, Json<Job>), ApiError> {
    state.ensure_accepting_jobs().await?;
    let mut entry = resolve_model(&state, &id).await?;
    if let Some(revision) = requested_revision {
        if revision != entry.revision {
            return Err(ApiError::conflict(
                "La révision demandée n'est plus la révision publiée par Hugging Face.",
            ));
        }
        entry.revision = revision;
    }
    let view = model_view(&state, &entry).await;
    entry.runtime_capabilities = view.runtime_capabilities.clone();
    entry.pipeline_class = view.pipeline_class.clone();
    // Une installation déjà à la révision courante est idempotemment refusée,
    // mais une révision distante plus récente suit le même job atomique : c'est
    // la voie de mise à jour, sans route spéciale ni perte de l'ID original.
    if view.installed && !view.update_available {
        return Err(ApiError::conflict("Ce modèle est déjà installé."));
    }
    if (entry.gated || entry.private) && entry.access_checked && !entry.access_authorized {
        return Err(ApiError::unauthorized(
            "Ce modèle nécessite un accès Hugging Face autorisé via HF_TOKEN.",
        ));
    }
    if view.runtime_compatibility == "UNSUPPORTED" || !entry.quality_valid {
        return Err(ApiError::conflict(
            "Le modèle ne possède pas les fichiers requis par un runtime VidioAI validé.",
        ));
    }
    if !view.compatible {
        return Err(ApiError::conflict(
            "Ce modèle n'est pas installable ; consultez compatibility_checks pour la cause précise.",
        ));
    }

    let job = Job {
        id: Uuid::new_v4(),
        kind: JobKind::InstallModel,
        target_id: entry.id.clone(),
        model_id: Some(entry.id.clone()),
        capability: None,
        status: JobStatus::Queued,
        stage: "checking".into(),
        progress: 0,
        message: "Vérification du modèle".into(),
        transfer: None,
        dependency: None,
        cache_status: if state.object_storage.enabled() {
            Some("CACHE_PENDING".into())
        } else {
            None
        },
        cache_error: None,
        started_at: None,
        completed_at: None,
        error: None,
        result: None,
        created_at: unix_now(),
        updated_at: unix_now(),
    };
    state.insert_job(job.clone()).await?;
    let worker_state = state.clone();
    let worker_job = job.clone();
    tokio::spawn(async move {
        run_install(worker_state, worker_job, entry).await;
    });
    Ok((StatusCode::ACCEPTED, Json(job)))
}

/// Relance exclusivement la publication L2 -> L3 d'un snapshot déjà installé.
/// Aucun appel Hugging Face et aucun téléchargement de poids n'est effectué.
async fn cache_model_from_body(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ModelActionInput>,
) -> Result<(StatusCode, Json<Job>), ApiError> {
    state.ensure_accepting_jobs().await?;
    if !state.object_storage.enabled() {
        return Err(ApiError::conflict("Le cache S3 est désactivé."));
    }
    let entry = resolve_model(&state, &input.model_id).await?;
    if entry.local {
        return Err(ApiError::conflict(
            "Les moteurs procéduraux ne possèdent pas de snapshot à publier.",
        ));
    }
    let worker = state
        .worker
        .as_ref()
        .ok_or_else(|| ApiError::unavailable("Le worker GPU n'est pas configuré."))?;
    let status = worker
        .model_status(&entry.storage_id)
        .await
        .map_err(ApiError::unavailable)?;
    if !status.installed || !status.weights_valid {
        return Err(ApiError::conflict(
            "Le snapshot local doit être installé et validé avant sa sauvegarde S3.",
        ));
    }
    let revision = status
        .revision
        .clone()
        .unwrap_or_else(|| entry.revision.clone());
    let snapshot_root = state
        .settings
        .get()
        .await
        .models_dir
        .join(&entry.storage_id)
        .join(&revision);
    if !matches!(fs::metadata(&snapshot_root).await, Ok(metadata) if metadata.is_dir()) {
        return Err(ApiError::conflict("Le snapshot local validé est absent."));
    }
    let job = Job {
        id: Uuid::new_v4(),
        kind: JobKind::CacheModel,
        target_id: entry.id.clone(),
        model_id: Some(entry.id.clone()),
        capability: None,
        status: JobStatus::Queued,
        stage: "saving_cache".into(),
        progress: 90,
        message: "Sauvegarde dans le cache S3".into(),
        transfer: None,
        dependency: None,
        cache_status: Some("CACHE_PENDING".into()),
        cache_error: None,
        started_at: None,
        completed_at: None,
        error: None,
        result: None,
        created_at: unix_now(),
        updated_at: unix_now(),
    };
    state.insert_job(job.clone()).await?;
    let task_state = state.clone();
    let task_job = job.clone();
    tokio::spawn(async move {
        match upload_model_cache(
            task_state.clone(),
            task_job.id,
            &entry.repository,
            &revision,
            &snapshot_root,
        )
        .await
        {
            Ok(()) => {
                task_state
                    .update_cache_status(task_job.id, "CACHE_READY", None)
                    .await;
                task_state
                    .update_job(
                        task_job.id,
                        JobStatus::Completed,
                        "installed",
                        100,
                        "Cache S3 validé; aucun téléchargement Hugging Face n'a été relancé",
                    )
                    .await;
            }
            Err(error) => {
                task_state
                    .update_cache_status(task_job.id, "CACHE_FAILED", Some(error.clone()))
                    .await;
                let progress = task_state
                    .jobs
                    .read()
                    .await
                    .get(&task_job.id)
                    .map(|job| job.progress)
                    .unwrap_or(90)
                    .min(99);
                task_state
                    .update_job(
                        task_job.id,
                        JobStatus::Failed,
                        "saving_cache",
                        progress,
                        &format!("CACHE_FAILED retryable: {error}"),
                    )
                    .await;
            }
        }
    });
    Ok((StatusCode::ACCEPTED, Json(job)))
}

#[derive(Debug, Serialize)]
struct CloudModelView {
    repository: String,
    revision: String,
    name: String,
    size_bytes: u64,
    files: usize,
    created_at: u64,
    capabilities: Vec<String>,
    cloud_state: &'static str,
    local_state: &'static str,
    local: bool,
    cloud: bool,
    valid: bool,
    manifest_uri: String,
}

#[derive(Debug, Serialize)]
struct InstalledModelView {
    id: String,
    storage_id: String,
    repository: String,
    revision: String,
    state: String,
    stage: Option<String>,
    loaded: bool,
    capabilities: Vec<String>,
    precision: String,
    precision_plan: Option<serde_json::Value>,
    pipeline_class: Option<String>,
    size_bytes: u64,
    device: Option<String>,
    memory_strategy: Option<String>,
    memory_plan: Option<serde_json::Value>,
    vram_bytes: u64,
    vram_peak_bytes: u64,
    ram_peak_bytes: u64,
    cpu_offload: bool,
    disk_offload: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct InstalledModelsResponse {
    items: Vec<InstalledModelView>,
    loaded: usize,
    gpu: Option<serde_json::Value>,
    memory: Option<serde_json::Value>,
}

async fn directory_size(root: &FilePath) -> u64 {
    let mut pending = vec![root.to_path_buf()];
    let mut total = 0_u64;
    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::metadata(&path).await else {
            continue;
        };
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        } else if metadata.is_dir()
            && let Ok(mut entries) = fs::read_dir(path).await
        {
            while let Ok(Some(entry)) = entries.next_entry().await {
                pending.push(entry.path());
            }
        }
    }
    total
}

async fn list_installed_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<InstalledModelsResponse>, ApiError> {
    let settings = state.settings.get().await;
    let resources = if let Some(worker) = &state.worker {
        worker.resources().await.ok()
    } else {
        None
    };
    let mut items = Vec::new();
    let mut directories = fs::read_dir(&settings.models_dir)
        .await
        .map_err(ApiError::internal)?;
    while let Some(entry) = directories.next_entry().await.map_err(ApiError::internal)? {
        let storage_id = entry.file_name().to_string_lossy().to_string();
        if storage_id.starts_with('.')
            || !entry
                .file_type()
                .await
                .map_err(ApiError::internal)?
                .is_dir()
        {
            continue;
        }
        let pointer_path = entry.path().join("active.json");
        let Ok(pointer_bytes) = fs::read(&pointer_path).await else {
            continue;
        };
        let Ok(pointer) = serde_json::from_slice::<InstalledPointer>(&pointer_bytes) else {
            continue;
        };
        let snapshot = entry.path().join(&pointer.revision);
        if !fs::metadata(&snapshot)
            .await
            .is_ok_and(|metadata| metadata.is_dir())
        {
            continue;
        }
        let manifest = fs::read(snapshot.join("vidioai-model.json"))
            .await
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .unwrap_or_default();
        let repository = pointer
            .repository
            .or_else(|| {
                manifest
                    .get("repository")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| storage_id.clone());
        let status = if let Some(worker) = &state.worker {
            worker.model_status(&storage_id).await.ok()
        } else {
            None
        };
        let capabilities = status
            .as_ref()
            .map(|status| status.capabilities.clone())
            .filter(|values| !values.is_empty())
            .unwrap_or_else(|| {
                manifest
                    .get("capabilities")
                    .or_else(|| manifest.get("requested_capabilities"))
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect()
            });
        let precision_plan = status
            .as_ref()
            .and_then(|status| status.precision_plan.clone());
        let memory_plan = status
            .as_ref()
            .and_then(|status| status.memory_plan.clone());
        let benchmark = status.as_ref().and_then(|status| status.benchmark.as_ref());
        let memory_strategy = memory_plan
            .as_ref()
            .and_then(|plan| plan.get("strategy"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let disk_offload = memory_strategy.as_deref() == Some("DISK_OFFLOAD");
        let manifest_size = manifest
            .get("files")
            .and_then(serde_json::Value::as_array)
            .map(|files| {
                files
                    .iter()
                    .filter_map(|file| file.get("size").and_then(serde_json::Value::as_u64))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        items.push(InstalledModelView {
            id: repository.clone(),
            storage_id,
            repository,
            revision: pointer.revision,
            state: status
                .as_ref()
                .map(|status| status.state.clone())
                .unwrap_or_else(|| "INSTALLED".into()),
            stage: status.as_ref().and_then(|status| status.stage.clone()),
            loaded: status.as_ref().is_some_and(|status| status.ready),
            capabilities,
            precision: precision_plan
                .as_ref()
                .and_then(|plan| plan.get("resolved"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("AUTO")
                .to_owned(),
            precision_plan,
            pipeline_class: status
                .as_ref()
                .and_then(|status| status.pipeline_class.clone())
                .or_else(|| {
                    manifest
                        .get("pipeline_class")
                        .or_else(|| manifest.get("class_name"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                }),
            size_bytes: if manifest_size > 0 {
                manifest_size
            } else {
                directory_size(&snapshot).await
            },
            device: status.as_ref().and_then(|status| status.device.clone()),
            memory_strategy,
            memory_plan,
            vram_bytes: benchmark.map_or(0, |value| value.vram_after_load_bytes),
            vram_peak_bytes: benchmark.map_or(0, |value| value.vram_peak_bytes),
            ram_peak_bytes: benchmark
                .and_then(|value| value.ram_peak_bytes)
                .unwrap_or(0),
            cpu_offload: benchmark.is_some_and(|value| value.cpu_offload),
            disk_offload,
            error: status.and_then(|status| status.error),
        });
    }
    items.sort_by(|left, right| left.repository.cmp(&right.repository));
    let loaded = items.iter().filter(|item| item.loaded).count();
    Ok(Json(InstalledModelsResponse {
        items,
        loaded,
        gpu: resources
            .as_ref()
            .and_then(|resources| serde_json::to_value(&resources.gpu).ok())
            .filter(|value| !value.is_null()),
        memory: resources.and_then(|resources| resources.memory),
    }))
}

#[derive(Debug, Serialize)]
struct CloudModelsResponse {
    items: Vec<CloudModelView>,
}

#[derive(Debug, Deserialize)]
struct CloudRestoreSelection {
    repository: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
struct CloudRestoreInput {
    models: Vec<CloudRestoreSelection>,
}

fn restore_identity(repository: &str, revision: &str) -> String {
    format!("{repository}@{revision}")
}

fn select_cloud_manifests(
    available: Vec<SnapshotManifest>,
    requested: &[CloudRestoreSelection],
) -> Result<Vec<SnapshotManifest>, String> {
    let by_identity = available
        .into_iter()
        .map(|manifest| {
            (
                (manifest.repository.clone(), manifest.revision.clone()),
                manifest,
            )
        })
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    requested
        .iter()
        .map(|selection| {
            let identity = (selection.repository.clone(), selection.revision.clone());
            if !seen.insert(identity.clone()) {
                return Err(
                    "CLOUD_SELECTION_DUPLICATE: snapshot sélectionné plusieurs fois".into(),
                );
            }
            by_identity
                .get(&identity)
                .cloned()
                .ok_or_else(|| "CLOUD_SNAPSHOT_NOT_FOUND: snapshot S3 valide inconnu".into())
        })
        .collect()
}

async fn list_cloud_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<CloudModelsResponse>, ApiError> {
    if !state.object_storage.enabled() {
        return Ok(Json(CloudModelsResponse { items: Vec::new() }));
    }
    let manifests = state
        .object_storage
        .list_snapshots()
        .await
        .map_err(ApiError::unavailable)?;
    let settings = state.settings.get().await;
    let ready_models = if let Some(worker) = &state.worker {
        worker
            .resources()
            .await
            .ok()
            .map(|resources| {
                resources
                    .loaded_models
                    .into_iter()
                    .filter_map(|model| {
                        let ready = model
                            .get("state")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|value| value == "READY");
                        ready.then(|| model.get("model_id")?.as_str().map(str::to_owned))?
                    })
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default()
    } else {
        HashSet::new()
    };
    let restore_jobs = state
        .jobs
        .read()
        .await
        .values()
        .filter(|job| job.kind == JobKind::RestoreModel)
        .cloned()
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        let model_storage_id = storage_id(&manifest.repository);
        let active_revision = fs::read(
            settings
                .models_dir
                .join(&model_storage_id)
                .join("active.json"),
        )
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<InstalledPointer>(&bytes).ok())
        .map(|pointer| pointer.revision);
        let local_state = if ready_models.contains(&model_storage_id) {
            "READY"
        } else if active_revision.as_deref() == Some(manifest.revision.as_str()) {
            "INSTALLED"
        } else {
            "ABSENT"
        };
        let cloud_state = restore_jobs
            .iter()
            .filter(|job| {
                job.target_id == restore_identity(&manifest.repository, &manifest.revision)
            })
            .max_by_key(|job| job.updated_at)
            .map(|job| match job.status {
                JobStatus::Queued
                | JobStatus::Dispatching
                | JobStatus::Running
                | JobStatus::SavingOutput => "RESTORING",
                JobStatus::Completed => "READY",
                JobStatus::Failed | JobStatus::Interrupted => "FAILED",
                JobStatus::Cancelled | JobStatus::PendingRetry => "AVAILABLE",
            })
            .unwrap_or("AVAILABLE");
        result.push(CloudModelView {
            name: manifest
                .repository
                .split('/')
                .next_back()
                .unwrap_or(&manifest.repository)
                .to_owned(),
            manifest_uri: state
                .object_storage
                .snapshot_uri(&manifest.repository, &manifest.revision)
                .map_err(ApiError::internal)?,
            repository: manifest.repository,
            revision: manifest.revision,
            size_bytes: manifest.total_size,
            files: manifest.files.len(),
            created_at: manifest.created_at,
            capabilities: manifest.capabilities,
            cloud_state,
            local_state,
            local: local_state != "ABSENT",
            cloud: true,
            valid: true,
        });
    }
    Ok(Json(CloudModelsResponse { items: result }))
}

async fn restore_cloud_models(
    State(state): State<Arc<AppState>>,
    Json(input): Json<CloudRestoreInput>,
) -> Result<(StatusCode, Json<Vec<Job>>), ApiError> {
    state.ensure_accepting_jobs().await?;
    if !state.object_storage.enabled() {
        return Err(ApiError::conflict("Le stockage S3 est désactivé."));
    }
    if input.models.is_empty() || input.models.len() > 20 {
        return Err(ApiError::bad_request(
            "Sélectionnez entre 1 et 20 snapshots.",
        ));
    }
    let available = state
        .object_storage
        .list_snapshots()
        .await
        .map_err(ApiError::unavailable)?;
    let selected =
        select_cloud_manifests(available, &input.models).map_err(ApiError::bad_request)?;

    let pending = selected
        .into_iter()
        .map(|manifest| {
            let identity = restore_identity(&manifest.repository, &manifest.revision);
            let job = Job {
                id: Uuid::new_v4(),
                kind: JobKind::RestoreModel,
                target_id: identity.clone(),
                model_id: Some(manifest.repository.clone()),
                capability: None,
                status: JobStatus::Queued,
                stage: "queued".into(),
                progress: 0,
                message: "Restauration S3 ajoutée à la file".into(),
                transfer: None,
                dependency: None,
                cache_status: Some("CLOUD_AVAILABLE".into()),
                cache_error: None,
                started_at: None,
                completed_at: None,
                error: None,
                result: None,
                created_at: unix_now(),
                updated_at: unix_now(),
            };
            (identity, job, manifest)
        })
        .collect::<Vec<_>>();

    {
        let mut claims = state.restore_claims.lock().await;
        if pending
            .iter()
            .any(|(identity, _, _)| claims.contains_key(identity))
        {
            return Err(ApiError::conflict(
                "RESTORE_ALREADY_RUNNING: une restauration est déjà active pour ce repository@revision.",
            ));
        }
        for (identity, job, _) in &pending {
            claims.insert(identity.clone(), job.id);
        }
    }

    for (_, job, _) in &pending {
        if let Err(error) = state.insert_job(job.clone()).await {
            let mut claims = state.restore_claims.lock().await;
            for (identity, _, _) in &pending {
                claims.remove(identity);
            }
            return Err(error);
        }
    }

    let mut jobs = Vec::with_capacity(pending.len());
    for (identity, job, manifest) in pending {
        let task_state = state.clone();
        let task_job = job.clone();
        tokio::spawn(async move {
            run_cloud_restore(task_state, task_job, manifest, identity).await;
        });
        jobs.push(job);
    }
    Ok((StatusCode::ACCEPTED, Json(jobs)))
}

async fn snapshot_file_sha256(path: &FilePath) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .await
        .map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .await
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

async fn snapshot_matches_manifest(root: &FilePath, manifest: &SnapshotManifest) -> bool {
    for file in &manifest.files {
        let relative = FilePath::new(&file.path);
        if !is_snapshot_file(relative) {
            return false;
        }
        let path = root.join(relative);
        if !fs::metadata(&path)
            .await
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == file.size)
        {
            return false;
        }
        if let Some(expected) = &file.sha256
            && snapshot_file_sha256(&path).await.as_deref() != Ok(expected.as_str())
        {
            return false;
        }
    }
    true
}

async fn seed_restore_staging(
    source: &FilePath,
    staging: &FilePath,
    manifest: &SnapshotManifest,
) -> Result<(), String> {
    for file in &manifest.files {
        let relative = FilePath::new(&file.path);
        if !is_snapshot_file(relative) {
            return Err("S3_MANIFEST_PATH_INVALID: chemin de fichier invalide".into());
        }
        let existing = source.join(relative);
        if !fs::metadata(&existing)
            .await
            .is_ok_and(|metadata| metadata.is_file())
        {
            continue;
        }
        let target = staging.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|error| error.to_string())?;
        }
        if fs::hard_link(&existing, &target).await.is_err() {
            fs::copy(&existing, &target)
                .await
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

async fn promote_restored_snapshot(
    staging: &FilePath,
    final_path: &FilePath,
    quarantine: &FilePath,
    manifest: &SnapshotManifest,
) -> Result<(), String> {
    if snapshot_matches_manifest(final_path, manifest).await {
        fs::remove_dir_all(staging)
            .await
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let had_previous = fs::metadata(final_path).await.is_ok();
    if had_previous {
        fs::rename(final_path, quarantine)
            .await
            .map_err(|error| error.to_string())?;
    }
    if let Err(error) = fs::rename(staging, final_path).await {
        if had_previous {
            let _ = fs::rename(quarantine, final_path).await;
        }
        return Err(error.to_string());
    }
    if had_previous {
        let _ = fs::remove_dir_all(quarantine).await;
    }
    Ok(())
}

async fn run_cloud_restore(
    state: Arc<AppState>,
    job: Job,
    manifest: SnapshotManifest,
    identity: String,
) {
    state
        .update_job(
            job.id,
            JobStatus::Dispatching,
            "dispatching",
            2,
            "Validation du manifeste S3",
        )
        .await;
    let result: Result<serde_json::Value, String> = async {
        let settings = state.settings.get().await;
        let model_id = storage_id(&manifest.repository);
        let model_root = settings.models_dir.join(&model_id);
        let snapshot_root = model_root.join(&manifest.revision);
        let restore_root = settings
            .models_dir
            .join(".restore")
            .join(job.id.to_string());
        let quarantine = settings
            .models_dir
            .join(".restore")
            .join(format!("{}.previous", job.id));
        fs::create_dir_all(&restore_root)
            .await
            .map_err(|error| error.to_string())?;
        seed_restore_staging(&snapshot_root, &restore_root, &manifest).await?;
        let (sender, mut receiver) = mpsc::unbounded_channel::<UploadProgress>();
        let callback: TransferProgressCallback = Arc::new(move |progress| {
            let _ = sender.send(progress);
        });
        let mut restore = Box::pin(state.object_storage.restore_snapshot(
            &manifest.repository,
            &manifest.revision,
            &restore_root,
            Some(callback),
        ));
        let restored = loop {
            tokio::select! {
                result = &mut restore => break result?,
                progress = receiver.recv() => {
                    if let Some(progress) = progress {
                        let percent = progress.percent();
                        let overall = (5.0 + percent * 0.8).round().clamp(5.0, 85.0) as u8;
                        let updated = {
                            let mut jobs = state.jobs.write().await;
                            jobs.get_mut(&job.id).map(|current| {
                                current.status = JobStatus::Running;
                                current.stage = "restoring".into();
                                current.progress = overall;
                                current.message = format!("Restauration S3 · {:.2}%", percent);
                                current.transfer = Some(progress);
                                current.started_at.get_or_insert_with(unix_now);
                                current.updated_at = unix_now();
                                current.clone()
                            })
                        };
                        if let Some(updated) = updated {
                            let _ = state.job_store.upsert(&updated).await;
                            state.emit("job.updated", &updated);
                        }
                    }
                }
            }
        };
        if !restored {
            return Err("S3_SNAPSHOT_NOT_FOUND: manifeste absent".into());
        }
        fs::create_dir_all(&model_root)
            .await
            .map_err(|error| error.to_string())?;
        promote_restored_snapshot(&restore_root, &snapshot_root, &quarantine, &manifest).await?;
        let pointer = json!({
            "model_id": model_id,
            "repository": manifest.repository,
            "revision": manifest.revision,
        });
        let active_pointer = model_root.join("active.json");
        let previous_pointer = fs::read(&active_pointer).await.ok();
        let temporary = model_root.join(format!("active.json.{}.tmp", job.id));
        fs::write(
            &temporary,
            serde_json::to_vec_pretty(&pointer).map_err(|e| e.to_string())?,
        )
        .await
        .map_err(|error| error.to_string())?;
        fs::rename(temporary, &active_pointer)
            .await
            .map_err(|error| error.to_string())?;

        let worker = state
            .worker
            .as_ref()
            .ok_or("WORKER_UNAVAILABLE: worker GPU absent")?;
        state
            .update_job(
                job.id,
                JobStatus::Running,
                "validating",
                90,
                "Validation locale du snapshot",
            )
            .await;
        let installed_result = worker
            .install(
                &model_id,
                &manifest.repository,
                &manifest.revision,
                &manifest.capabilities,
            )
            .await;
        let _installed = match installed_result {
            Ok(installed) if installed.installed && installed.weights_valid => installed,
            Ok(_) => {
                if let Some(previous) = previous_pointer {
                    fs::write(&active_pointer, previous)
                        .await
                        .map_err(|error| error.to_string())?;
                } else {
                    let _ = fs::remove_file(&active_pointer).await;
                }
                return Err("SNAPSHOT_INVALID: le worker a refusé les poids restaurés".into());
            }
            Err(error) => {
                if let Some(previous) = previous_pointer {
                    fs::write(&active_pointer, previous)
                        .await
                        .map_err(|restore_error| restore_error.to_string())?;
                } else {
                    let _ = fs::remove_file(&active_pointer).await;
                }
                return Err(error);
            }
        };
        Ok(json!({
            "repository": manifest.repository,
            "revision": manifest.revision,
            "model_id": model_id,
            "local_state": "INSTALLED",
            "cloud_state": "READY",
        }))
    }
    .await;

    match result {
        Ok(value) => {
            state.set_job_result(job.id, value).await;
            state
                .update_job(
                    job.id,
                    JobStatus::Completed,
                    "completed",
                    100,
                    "Snapshot restauré et validé",
                )
                .await;
        }
        Err(error) => {
            let progress = state
                .jobs
                .read()
                .await
                .get(&job.id)
                .map(|current| current.progress.min(99))
                .unwrap_or(0);
            state
                .update_job(job.id, JobStatus::Failed, "failed", progress, &error)
                .await;
        }
    }
    let settings = state.settings.get().await;
    let _ = fs::remove_dir_all(
        settings
            .models_dir
            .join(".restore")
            .join(job.id.to_string()),
    )
    .await;
    state.restore_claims.lock().await.remove(&identity);
}

/// Mesure les octets effectivement présents dans le staging du Worker. Cette
/// valeur alimente la progression pendant que `snapshot_download` s'exécute ;
/// aucun compteur temporel ou pourcentage simulé n'est utilisé.
async fn staged_download_bytes(models_dir: &FilePath, storage_id: &str) -> u64 {
    let downloads = models_dir.join(".downloads");
    let prefix = format!("download-{storage_id}-");
    let mut roots = Vec::new();
    let Ok(mut entries) = fs::read_dir(downloads).await else {
        return 0;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            roots.push(entry.path());
        }
    }
    let mut bytes = 0_u64;
    while let Some(path) = roots.pop() {
        if let Ok(metadata) = fs::metadata(&path).await {
            if metadata.is_file() {
                bytes = bytes.saturating_add(metadata.len());
            } else if metadata.is_dir()
                && let Ok(mut children) = fs::read_dir(path).await
            {
                while let Ok(Some(child)) = children.next_entry().await {
                    roots.push(child.path());
                }
            }
        }
    }
    bytes
}

async fn restore_model_cache(
    state: &AppState,
    entry: &CatalogEntry,
    model_root: &FilePath,
) -> Result<bool, String> {
    let snapshot_root = model_root.join(&entry.revision);
    let restored = state
        .object_storage
        .restore_snapshot(&entry.repository, &entry.revision, &snapshot_root, None)
        .await?;
    if !restored {
        return Ok(false);
    }
    fs::create_dir_all(model_root)
        .await
        .map_err(|error| error.to_string())?;
    let pointer = model_root.join("active.json");
    let temporary = model_root.join("active.json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&json!({
            "model_id": entry.storage_id,
            "repository": entry.repository,
            "revision": entry.revision,
        }))
        .map_err(|error| error.to_string())?,
    )
    .await
    .map_err(|error| error.to_string())?;
    fs::rename(temporary, pointer)
        .await
        .map_err(|error| error.to_string())?;
    Ok(true)
}

async fn upload_model_cache(
    state: Arc<AppState>,
    job_id: Uuid,
    repository: &str,
    revision: &str,
    snapshot_root: &FilePath,
) -> Result<(), String> {
    // Valide l'identité avant de démarrer le moindre transfert.
    model_s3_prefix(repository, revision)?;
    let (sender, mut receiver) = mpsc::unbounded_channel::<UploadProgress>();
    let callback: UploadProgressCallback = Arc::new(move |progress| {
        let _ = sender.send(progress);
    });
    let mut upload = Box::pin(state.object_storage.upload_snapshot(
        repository,
        revision,
        snapshot_root,
        Some(callback),
    ));
    let result = loop {
        tokio::select! {
            result = &mut upload => break result,
            progress = receiver.recv() => {
                if let Some(progress) = progress {
                    state.update_cache_progress(job_id, progress).await;
                }
            }
        }
    };
    while let Ok(progress) = receiver.try_recv() {
        state.update_cache_progress(job_id, progress).await;
    }
    result.map(|_| ())
}

/// Worker d'installation : téléchargement en flux, progression mesurée sur le
/// staging, vérification par SHA-256, installation atomique et état READY.
async fn run_install(state: Arc<AppState>, job: Job, entry: CatalogEntry) {
    let result: Result<Option<String>, String> = async {
        let worker = state.worker.as_ref().ok_or_else(|| {
            "VIDIOAI_WORKER_URL est obligatoire pour installer des poids IA.".to_owned()
        })?;
        state
            .update_job(
                job.id,
                JobStatus::Running,
                "restoring_cache",
                5,
                "Recherche du snapshot dans le cache S3",
            )
            .await;
        let settings = state.settings.get().await;
        let model_root = settings.models_dir.join(&entry.storage_id);
        if state.object_storage.enabled() {
            // L1 est la VRAM du worker, L2 le volume scratch `models_dir`, L3 le
            // préfixe S3 validé par manifeste. Un échec n'empêche pas HF de remplir L2.
            let _ = restore_model_cache(&state, &entry, &model_root).await;
        }
        state
            .update_job(
                job.id,
                JobStatus::Running,
                "downloading",
                20,
                "Téléchargement atomique du snapshot Hugging Face",
            )
            .await;
        let install_capabilities = if entry.runtime_capabilities.is_empty() {
            entry.capabilities
                .iter()
                .map(|capability| capability.api_name().to_owned())
                .collect::<Vec<_>>()
        } else {
            entry.runtime_capabilities
                .iter()
                .map(|capability| capability.api_name().to_owned())
                .collect::<Vec<_>>()
        };
        let mut installation = Box::pin(worker.install(
            &entry.storage_id,
            &entry.repository,
            &entry.revision,
            &install_capabilities,
        ));
        let installed = loop {
            tokio::select! {
                result = &mut installation => break result?,
                _ = sleep(Duration::from_secs(2)) => {
                    let downloaded = staged_download_bytes(&settings.models_dir, &entry.storage_id).await;
                    if downloaded > 0 {
                        let progress = entry.estimated_size_bytes
                            .filter(|total| *total > 0)
                            .map(|total| 20 + ((downloaded.min(total) as f64 / total as f64) * 54.0) as u8)
                            .unwrap_or(35)
                            .min(74);
                        let message = if let Some(total) = entry.estimated_size_bytes {
                            format!(
                                "Snapshot Hugging Face : {:.1} / {:.1} Go reçus",
                                downloaded as f64 / 1_073_741_824.0,
                                total as f64 / 1_073_741_824.0,
                            )
                        } else {
                            format!(
                                "Snapshot Hugging Face : {:.1} Go reçus (taille totale inconnue)",
                                downloaded as f64 / 1_073_741_824.0,
                            )
                        };
                        state.update_job(
                            job.id,
                            JobStatus::Running,
                            "downloading",
                            progress,
                            &message,
                        ).await;
                    }
                    if let Ok(status) = worker.model_status(&entry.storage_id).await
                        && matches!(
                            status.state.as_str(),
                            "RESOLVING_DEPENDENCIES"
                                | "DOWNLOADING_DEPENDENCY"
                                | "INSTALLING_DEPENDENCIES"
                        )
                    {
                        state.update_dependency_progress(
                            job.id,
                            &status.state,
                            &status.runtime_dependencies,
                        ).await;
                    }
                }
            }
        };
        if !installed.installed || !installed.weights_valid {
            return Err("Le worker n'a pas validé les poids téléchargés.".into());
        }
        state
            .update_job(
                job.id,
                JobStatus::Running,
                "validating_snapshot",
                80,
                "Snapshot et poids validés; aucun chargement GPU pendant l'installation",
            )
            .await;
        state
            .update_job(
                job.id,
                JobStatus::Running,
                "resolving_dependencies",
                84,
                "Vérification des dépendances runtime déclarées par le snapshot",
            )
            .await;
        if !installed.runtime_dependencies.is_empty() {
            state
                .update_dependency_progress(
                    job.id,
                    "RESOLVING_DEPENDENCIES",
                    &installed.runtime_dependencies,
                )
                .await;
        }
        let resolved_revision = installed
            .revision
            .as_deref()
            .unwrap_or(&entry.revision)
            .to_owned();
        let snapshot_root = model_root.join(&resolved_revision);
        let mut cache_error = None;
        if state.object_storage.enabled() {
            state
                .update_job(
                    job.id,
                    JobStatus::Running,
                    "saving_cache",
                    90,
                    "Sauvegarde dans le cache S3",
                )
                .await;
            match upload_model_cache(
                state.clone(),
                job.id,
                &entry.repository,
                &resolved_revision,
                &snapshot_root,
            )
            .await
            {
                Ok(()) => state
                    .update_cache_status(job.id, "CACHE_READY", None)
                    .await,
                Err(error) => {
                    state
                        .update_cache_status(job.id, "CACHE_FAILED", Some(error.clone()))
                        .await;
                    cache_error = Some(error);
                }
            }
        }
        Ok(cache_error)
    }
    .await;

    match result {
        Ok(cache_error) => {
            let message = cache_error.map_or_else(
                || "Modèle installé; chargement runtime disponible séparément".to_owned(),
                |error| {
                    format!(
                        "Modèle installé localement; CACHE_FAILED retryable: {error}. La sauvegarde S3 peut être relancée séparément"
                    )
                },
            );
            state
                .update_job(job.id, JobStatus::Completed, "installed", 100, &message)
                .await
        }
        Err(error) => {
            let last_progress = state
                .jobs
                .read()
                .await
                .get(&job.id)
                .map(|item| item.progress)
                .unwrap_or(0)
                .min(99);
            let message = format!("Installation échouée à {last_progress}% · {error}");
            state
                .update_job(job.id, JobStatus::Failed, "failed", last_progress, &message)
                .await
        }
    }
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Job>, ApiError> {
    state
        .job_store
        .get(id)
        .await
        .map_err(ApiError::internal)?
        .map(Json)
        .ok_or_else(|| ApiError::not_found("Job inconnu."))
}

async fn get_queue(State(state): State<Arc<AppState>>) -> Json<Vec<Job>> {
    Json(state.queue().await)
}

async fn delete_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    delete_model_by_id(&state, &id).await
}

async fn delete_model_from_query(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ModelIdQuery>,
) -> Result<StatusCode, ApiError> {
    delete_model_by_id(&state, &query.model_id).await
}

async fn delete_model_by_id(state: &AppState, id: &str) -> Result<StatusCode, ApiError> {
    let entry = resolve_model(state, id).await?;
    if entry.local {
        return Err(ApiError::conflict(
            "Le moteur intégré ne peut pas être supprimé.",
        ));
    }
    if let Some(worker) = &state.worker {
        let _ = worker.unload(&entry.storage_id).await;
    }
    state.runtime.write().await.remove(id);
    let directory = state
        .settings
        .get()
        .await
        .models_dir
        .join(&entry.storage_id);
    if fs::metadata(&directory).await.is_err() {
        return Err(ApiError::not_found("Ce modèle n'est pas installé."));
    }
    fs::remove_dir_all(directory)
        .await
        .map_err(ApiError::internal)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<RuntimeEntry>, ApiError> {
    load_model_by_id(&state, &id).await
}

async fn load_model_from_body(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ModelActionInput>,
) -> Result<Json<RuntimeEntry>, ApiError> {
    load_model_by_id(&state, &input.model_id).await
}

async fn installed_storage_id(state: &AppState, id: &str) -> Option<(String, String)> {
    let settings = state.settings.get().await;
    let mut entries = fs::read_dir(settings.models_dir).await.ok()?;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let stored_id = entry.file_name().to_string_lossy().to_string();
        let bytes = fs::read(entry.path().join("active.json")).await.ok();
        let Some(pointer) =
            bytes.and_then(|bytes| serde_json::from_slice::<InstalledPointer>(&bytes).ok())
        else {
            continue;
        };
        if stored_id == storage_id(id) || pointer.repository.as_deref() == Some(id) {
            return Some((stored_id, pointer.revision));
        }
    }
    None
}

async fn load_model_by_id(state: &AppState, id: &str) -> Result<Json<RuntimeEntry>, ApiError> {
    if let Some((installed_storage_id, revision)) = installed_storage_id(state, id).await {
        let worker = state
            .worker
            .as_ref()
            .ok_or_else(|| ApiError::unavailable("Le worker GPU n'est pas configuré."))?;
        let status = worker
            .load(&installed_storage_id, id, &revision)
            .await
            .map_err(ApiError::unavailable)?;
        if status.state != "READY" || !status.ready || !status.runtime_compatible {
            return Err(ApiError::unavailable(status.error.unwrap_or_else(|| {
                "Le runtime worker n'a pas atteint READY.".into()
            })));
        }
        if let Some(observation) = &status.benchmark {
            record_worker_benchmark(state, id, &revision, observation).await;
        }
        let runtime = RuntimeEntry {
            model_id: id.to_owned(),
            state: "ready".into(),
            device: status.device.unwrap_or_else(|| "GPU".into()),
            ram_bytes: status
                .benchmark
                .as_ref()
                .and_then(|benchmark| benchmark.ram_peak_bytes)
                .unwrap_or(0),
            vram_bytes: status
                .benchmark
                .as_ref()
                .map_or(0, |benchmark| benchmark.vram_after_load_bytes),
            last_used_at: unix_now(),
        };
        state
            .runtime
            .write()
            .await
            .insert(id.to_owned(), runtime.clone());
        state.emit("resources.updated", &runtime);
        return Ok(Json(runtime));
    }
    let entry = resolve_model(state, id).await?;
    let view = model_view(state, &entry).await;
    if !view.installed {
        return Err(ApiError::conflict(
            "Installez le modèle avant de le charger.",
        ));
    }
    if !view.compatible {
        return Err(ApiError::conflict(
            "La mémoire disponible est insuffisante.",
        ));
    }
    if let Some(existing) = state.runtime.read().await.get(id).cloned() {
        if !entry.local {
            if let Some(worker) = &state.worker {
                let status = worker
                    .model_status(&entry.storage_id)
                    .await
                    .map_err(ApiError::unavailable)?;
                if status.state == "READY" && status.runtime_compatible && status.weights_valid {
                    return Ok(Json(existing));
                }
                state.runtime.write().await.remove(id);
            } else {
                state.runtime.write().await.remove(id);
            }
        } else {
            return Ok(Json(existing));
        }
    }
    if !entry.local {
        let worker = state
            .worker
            .as_ref()
            .ok_or_else(|| ApiError::unavailable("Le worker GPU n'est pas configuré."))?;
        let status = worker
            .load(&entry.storage_id, &entry.repository, &entry.revision)
            .await
            .map_err(ApiError::unavailable)?;
        if status.state != "READY" || !status.runtime_compatible {
            return Err(ApiError::unavailable(
                "Le runtime worker n'a pas atteint READY.",
            ));
        }
        if let Some(observation) = &status.benchmark {
            record_worker_benchmark(state, &entry.id, &entry.revision, observation).await;
        }
        let runtime = RuntimeEntry {
            model_id: id.to_owned(),
            state: "ready".into(),
            device: "GPU".into(),
            ram_bytes: 0,
            vram_bytes: entry
                .variants
                .first()
                .map_or(0, |variant| variant.vram_required),
            last_used_at: unix_now(),
        };
        state
            .runtime
            .write()
            .await
            .insert(id.to_owned(), runtime.clone());
        state.emit("resources.updated", &runtime);
        return Ok(Json(runtime));
    }
    let variant_id = view.recommended_variant.as_deref().unwrap_or("builtin");
    let variant = entry
        .variants
        .iter()
        .find(|variant| variant.id == variant_id)
        .unwrap_or(&entry.variants[0]);
    let runtime = RuntimeEntry {
        model_id: id.to_owned(),
        state: "warming_up".into(),
        device: if variant.vram_required > 0 {
            "GPU".into()
        } else {
            "CPU".into()
        },
        ram_bytes: variant.ram_required,
        vram_bytes: variant.vram_required,
        last_used_at: unix_now(),
    };
    state.runtime.write().await.insert(id.to_owned(), runtime);
    sleep(Duration::from_millis(180)).await;
    let mut runtimes = state.runtime.write().await;
    let ready = runtimes.get_mut(id).expect("runtime inséré");
    ready.state = "ready".into();
    let ready = ready.clone();
    state.emit("resources.updated", &ready);
    Ok(Json(ready))
}

async fn unload_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    unload_model_by_id(&state, &id).await
}

async fn unload_model_from_body(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ModelActionInput>,
) -> Result<StatusCode, ApiError> {
    unload_model_by_id(&state, &input.model_id).await
}

async fn unload_model_by_id(state: &AppState, id: &str) -> Result<StatusCode, ApiError> {
    let worker_handled = if let Some(worker) = &state.worker {
        worker
            .unload(&storage_id(id))
            .await
            .map_err(ApiError::unavailable)?;
        true
    } else {
        false
    };
    let removed = state.runtime.write().await.remove(id).is_some();
    if !removed && !worker_handled {
        return Err(ApiError::not_found("Ce modèle n'est pas chargé."));
    }
    state.emit(
        "resources.updated",
        &json!({ "model_id": id, "state": "unloaded" }),
    );
    Ok(StatusCode::NO_CONTENT)
}

async fn events_upgrade(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| events_socket(socket, state))
}

/// Une connexion reçoit les événements diffusés par tous les workers. Le `select!`
/// surveille aussi la fermeture du navigateur pour libérer immédiatement la tâche.
async fn events_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut receiver = state.events.subscribe();
    loop {
        tokio::select! {
            event = receiver.recv() => match event {
                Ok(event) => {
                    let Ok(payload) = serde_json::to_string(&event) else { continue };
                    if socket.send(Message::Text(payload.into())).await.is_err() { break; }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            },
            message = socket.recv() => match message {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                _ => {}
            }
        }
    }
}

/// Emplacement d'un asset et de son manifeste. Le chemin n'est jamais accepté du
/// client : seul l'UUID sert de clé, ce qui bloque les traversées de répertoires.
async fn asset_paths(
    state: &AppState,
    id: Uuid,
    extension: &str,
) -> Result<(PathBuf, PathBuf), ApiError> {
    let directory = state.settings.get().await.outputs_dir.join("assets");
    fs::create_dir_all(&directory)
        .await
        .map_err(ApiError::internal)?;
    Ok((
        directory.join(format!("{id}.{extension}")),
        directory.join(format!("{id}.json")),
    ))
}

async fn save_asset(
    state: &AppState,
    bytes: &[u8],
    filename: String,
    mime_type: String,
    kind: AssetKind,
    dimensions: Option<(u32, u32)>,
    extension: &str,
) -> Result<Asset, ApiError> {
    let id = Uuid::new_v4();
    let (binary_path, manifest_path) = asset_paths(state, id, extension).await?;
    fs::write(&binary_path, bytes)
        .await
        .map_err(ApiError::internal)?;
    let asset = Asset {
        id,
        kind,
        filename,
        mime_type,
        size_bytes: bytes.len() as u64,
        width: dimensions.map(|value| value.0),
        height: dimensions.map(|value| value.1),
        duration_seconds: None,
        fps: None,
        created_at: unix_now(),
        url: format!("/api/assets/{id}"),
    };
    fs::write(
        manifest_path,
        serde_json::to_vec_pretty(&asset).map_err(ApiError::internal)?,
    )
    .await
    .map_err(ApiError::internal)?;
    Ok(asset)
}

/// Publie une sortie sous la structure stable demandée par le stockage objet :
/// `outputs/<generation-id>/<nom-fichier>`. Le disque local reste la copie L2.
async fn publish_generation_asset(
    state: &AppState,
    generation_id: Uuid,
    asset: &Asset,
) -> Result<(), String> {
    if !state.object_storage.enabled() {
        return Ok(());
    }
    let (_, binary_path) = read_asset_manifest(state, asset.id)
        .await
        .map_err(|error| error.message)?;
    state
        .object_storage
        .upload_file(
            &binary_path,
            &format!("outputs/{generation_id}/{}", asset.filename),
        )
        .await
}

async fn read_asset_manifest(state: &AppState, id: Uuid) -> Result<(Asset, PathBuf), ApiError> {
    let directory = state.settings.get().await.outputs_dir.join("assets");
    let manifest_path = directory.join(format!("{id}.json"));
    let bytes = fs::read(&manifest_path)
        .await
        .map_err(|_| ApiError::not_found("Asset inconnu."))?;
    let asset: Asset = serde_json::from_slice(&bytes).map_err(ApiError::internal)?;
    let extension = match asset.mime_type.as_str() {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "video/mp4" => "mp4",
        "audio/mpeg" => "mp3",
        _ => "bin",
    };
    Ok((asset, directory.join(format!("{id}.{extension}"))))
}

/// Réécrit uniquement le manifeste d'un asset après enrichissement par
/// FFprobe. Le fichier binaire n'est jamais déplacé ou réencodé ici.
async fn persist_asset_manifest(state: &AppState, asset: &Asset) -> Result<(), ApiError> {
    let path = state
        .settings
        .get()
        .await
        .outputs_dir
        .join("assets")
        .join(format!("{}.json", asset.id));
    fs::write(
        path,
        serde_json::to_vec_pretty(asset).map_err(ApiError::internal)?,
    )
    .await
    .map_err(ApiError::internal)
}

/// Lit résolution, durée et cadence depuis le flux vidéo principal. Pour un
/// upload, l'absence de ces informations signifie que le fichier n'est pas une
/// vidéo exploitable et l'API le refuse avant de créer un asset.
async fn probe_video(path: &FilePath) -> Option<(u32, u32, f64, f64)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let stream = value.get("streams")?.as_array()?.first()?;
    let width = stream.get("width")?.as_u64()? as u32;
    let height = stream.get("height")?.as_u64()? as u32;
    let duration = value
        .get("format")?
        .get("duration")?
        .as_str()?
        .parse()
        .ok()?;
    let fraction = stream.get("r_frame_rate")?.as_str()?;
    let (numerator, denominator) = fraction.split_once('/').unwrap_or((fraction, "1"));
    let fps = numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?.max(1.0);
    Some((width, height, duration, fps))
}

/// Vérifie qu'un fichier contient bien un flux audio décodable et récupère sa
/// durée. Le type MIME envoyé par le navigateur n'est jamais considéré comme
/// une preuve suffisante.
async fn probe_audio(path: &FilePath) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let has_audio = value
        .get("streams")?
        .as_array()?
        .iter()
        .any(|stream| stream.get("codec_type").and_then(|kind| kind.as_str()) == Some("audio"));
    if !has_audio {
        return None;
    }
    value.get("format")?.get("duration")?.as_str()?.parse().ok()
}

async fn upload_asset(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<Asset>), ApiError> {
    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::bad_request(format!(
            "INVALID_MULTIPART: impossible de lire le flux multipart ({error})"
        ))
    })? {
        if field.name() != Some("file") {
            continue;
        }
        let filename = field.file_name().unwrap_or("asset").to_string();
        let mime = field
            .content_type()
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = field.bytes().await.map_err(|error| {
            ApiError::bad_request(format!(
                "INVALID_MULTIPART: contenu du champ `file` illisible ({error})"
            ))
        })?;
        let maximum = if mime == "video/mp4" {
            MAX_VIDEO_BYTES
        } else {
            MAX_IMAGE_BYTES
        };
        if bytes.len() > maximum {
            return Err(ApiError::bad_request(if mime == "video/mp4" {
                "La vidéo dépasse 512 Mo."
            } else {
                "Le fichier dépasse 25 Mo."
            }));
        }

        let mut video_metadata = None;
        let mut audio_duration = None;
        let (kind, dimensions, extension) = if mime.starts_with("image/") {
            let decoded = image::load_from_memory(&bytes)
                .map_err(|_| ApiError::bad_request("Le fichier n'est pas une image décodable."))?;
            let extension = match mime.as_str() {
                "image/jpeg" => "jpg",
                "image/webp" => "webp",
                _ => "png",
            };
            (AssetKind::Image, Some(decoded.dimensions()), extension)
        } else if mime == "video/mp4" {
            let temporary = state
                .settings
                .get()
                .await
                .work_dir
                .join(format!("upload-{}.mp4", Uuid::new_v4()));
            fs::write(&temporary, &bytes)
                .await
                .map_err(ApiError::internal)?;
            video_metadata = probe_video(&temporary).await;
            let _ = fs::remove_file(&temporary).await;
            if video_metadata.is_none() {
                return Err(ApiError::bad_request(
                    "Le fichier MP4 ne contient pas de flux vidéo décodable.",
                ));
            }
            (AssetKind::Video, None, "mp4")
        } else if mime == "audio/mpeg" {
            let temporary = state
                .settings
                .get()
                .await
                .work_dir
                .join(format!("upload-{}.mp3", Uuid::new_v4()));
            fs::write(&temporary, &bytes)
                .await
                .map_err(ApiError::internal)?;
            audio_duration = probe_audio(&temporary).await;
            let _ = fs::remove_file(&temporary).await;
            if audio_duration.is_none() {
                return Err(ApiError::bad_request(
                    "Le fichier MP3 ne contient pas de flux audio décodable.",
                ));
            }
            (AssetKind::Audio, None, "mp3")
        } else {
            return Err(ApiError::bad_request(
                "Type refusé. Utilisez PNG, JPEG, WebP, MP4 ou MP3.",
            ));
        };
        let mut asset = save_asset(
            &state,
            &bytes,
            filename,
            mime,
            kind.clone(),
            dimensions,
            extension,
        )
        .await?;
        if let Some((width, height, duration, fps)) = video_metadata {
            asset.width = Some(width);
            asset.height = Some(height);
            asset.duration_seconds = Some(duration);
            asset.fps = Some(fps);
            persist_asset_manifest(&state, &asset).await?;
        } else if let Some(duration) = audio_duration {
            asset.duration_seconds = Some(duration);
            persist_asset_manifest(&state, &asset).await?;
        }
        return Ok((StatusCode::CREATED, Json(asset)));
    }
    Err(ApiError::bad_request(
        "Le champ multipart `file` est obligatoire.",
    ))
}

/// Inventorie les manifests plutôt que les fichiers binaires. Le client reçoit
/// donc toujours les mêmes URLs contrôlées et jamais un chemin du serveur.
async fn list_assets(State(state): State<Arc<AppState>>) -> Json<Vec<Asset>> {
    let directory = state.settings.get().await.outputs_dir.join("assets");
    let mut assets = Vec::new();
    if let Ok(mut entries) = fs::read_dir(directory).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if let Ok(bytes) = fs::read(entry.path()).await
                && let Ok(asset) = serde_json::from_slice::<Asset>(&bytes)
            {
                assets.push(asset);
            }
        }
    }
    assets.sort_by_key(|asset| std::cmp::Reverse(asset.created_at));
    Json(assets)
}

async fn get_asset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Response, ApiError> {
    let (asset, path) = read_asset_manifest(&state, id).await?;
    let bytes = fs::read(path)
        .await
        .map_err(|_| ApiError::not_found("Fichier asset absent."))?;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, asset.mime_type)
        .header(header::CACHE_CONTROL, "private, max-age=3600")
        .body(Body::from(bytes))
        .map_err(ApiError::internal)
}

async fn delete_asset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    let (_, binary) = read_asset_manifest(&state, id).await?;
    let manifest = state
        .settings
        .get()
        .await
        .outputs_dir
        .join("assets")
        .join(format!("{id}.json"));
    let _ = fs::remove_file(binary).await;
    fs::remove_file(manifest)
        .await
        .map_err(ApiError::internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// -----------------------------------------------------------------------------
// Étape 10 : interface commune aux moteurs IA
// -----------------------------------------------------------------------------

/// Un moteur reçoit toujours un contexte générique et retourne de vrais octets.
/// L'implémentation Stable Diffusion future respectera le même contrat.
#[async_trait]
trait AiEngine: Send + Sync {
    async fn load(&self) -> Result<(), String>;
    async fn unload(&self) -> Result<(), String>;
    async fn health(&self) -> bool;
    async fn text_to_image(&self, prompt: &str) -> Result<Vec<u8>, String>;
    async fn image_to_image(&self, prompt: &str, input: &[u8]) -> Result<Vec<u8>, String>;
    async fn cancel(&self) -> Result<(), String>;
}

/// Moteur local léger mais réellement fonctionnel : il génère un PNG unique à
/// partir du prompt. Il sert de runtime de référence et de test de bout en bout,
/// sans prétendre remplacer un modèle de diffusion téléchargé.
struct CanvasEngine;

impl CanvasEngine {
    fn palette(prompt: &str) -> [u8; 3] {
        let digest = Sha256::digest(prompt.as_bytes());
        [digest[0].max(32), digest[1].max(32), digest[2].max(32)]
    }

    fn encode_png(image: DynamicImage) -> Result<Vec<u8>, String> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|error| error.to_string())?;
        Ok(cursor.into_inner())
    }
}

#[async_trait]
impl AiEngine for CanvasEngine {
    async fn load(&self) -> Result<(), String> {
        Ok(())
    }
    async fn unload(&self) -> Result<(), String> {
        Ok(())
    }
    async fn health(&self) -> bool {
        true
    }

    async fn text_to_image(&self, prompt: &str) -> Result<Vec<u8>, String> {
        let palette = Self::palette(prompt);
        let mut image = ImageBuffer::new(1024, 1024);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let fx = x as f32 / 1023.0;
            let fy = y as f32 / 1023.0;
            let glow =
                (((fx - 0.68).powi(2) + (fy - 0.32).powi(2)).sqrt() * 255.0).min(255.0) as u8;
            let wave = (((x as f32 * 0.025).sin() + (y as f32 * 0.018).cos() + 2.0) * 28.0) as u8;
            *pixel = Rgb([
                palette[0].saturating_add(wave / 3).saturating_sub(glow / 3),
                palette[1].saturating_add(wave / 4).saturating_sub(glow / 4),
                palette[2].saturating_add(wave / 2).saturating_sub(glow / 5),
            ]);
        }
        Self::encode_png(DynamicImage::ImageRgb8(image))
    }

    async fn image_to_image(&self, prompt: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        let source = image::load_from_memory(input).map_err(|error| error.to_string())?;
        let palette = Self::palette(prompt);
        let mut image = source
            .resize_to_fill(1024, 1024, FilterType::Lanczos3)
            .to_rgb8();
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let light = (((x + y) % 255) as u8) / 8;
            for channel in 0..3 {
                pixel[channel] =
                    ((u16::from(pixel[channel]) * 3 + u16::from(palette[channel])) / 4) as u8;
                pixel[channel] = pixel[channel].saturating_add(light);
            }
        }
        Self::encode_png(DynamicImage::ImageRgb8(image))
    }

    async fn cancel(&self) -> Result<(), String> {
        Ok(())
    }
}

fn image_endpoint(capability: &ModelCapability) -> &'static str {
    match capability {
        ModelCapability::TextToImage => "/v1/generate/text-to-image",
        ModelCapability::ImageToImage => "/v1/generate/image-to-image",
        ModelCapability::Inpainting => "/v1/generate/inpainting",
        ModelCapability::Outpainting => "/v1/generate/outpainting",
        ModelCapability::ImageVariation => "/v1/generate/image-variation",
        ModelCapability::ImageUpscale => "/v1/generate/image-upscale",
        ModelCapability::ControlledImageGeneration => "/v1/generate/controlled-image-generation",
        _ => "/v1/generate/image-to-image",
    }
}

fn image_capability_mode_default(mode: &GenerationMode) -> Option<ModelCapability> {
    match mode {
        GenerationMode::TextToImage => Some(ModelCapability::TextToImage),
        GenerationMode::ImageToImage => Some(ModelCapability::ImageToImage),
        _ => None,
    }
}

fn is_valid_image_capability_for_mode(mode: &GenerationMode, capability: &ModelCapability) -> bool {
    match mode {
        GenerationMode::TextToImage => *capability == ModelCapability::TextToImage,
        GenerationMode::ImageToImage => matches!(
            capability,
            ModelCapability::ImageToImage
                | ModelCapability::Inpainting
                | ModelCapability::Outpainting
                | ModelCapability::ImageVariation
                | ModelCapability::ImageUpscale
                | ModelCapability::ControlledImageGeneration
        ),
        GenerationMode::TextToVideo
        | GenerationMode::ImageToVideo
        | GenerationMode::VideoToVideo => false,
    }
}

fn video_capability_mode_default(mode: &GenerationMode) -> Option<ModelCapability> {
    match mode {
        GenerationMode::TextToVideo => Some(ModelCapability::TextToVideo),
        GenerationMode::ImageToVideo => Some(ModelCapability::ImageToVideo),
        GenerationMode::VideoToVideo => Some(ModelCapability::VideoToVideo),
        GenerationMode::TextToImage | GenerationMode::ImageToImage => None,
    }
}

fn video_endpoint(capability: &ModelCapability) -> &'static str {
    match capability {
        ModelCapability::TextToVideo => "/v1/generate/text-to-video",
        ModelCapability::ImageToVideo => "/v1/generate/image-to-video",
        ModelCapability::MultiImageToVideo => "/v1/generate/multi-image-to-video",
        ModelCapability::StartEndImageToVideo => "/v1/generate/start-end-image-to-video",
        ModelCapability::KeyframesToVideo => "/v1/generate/keyframes-to-video",
        ModelCapability::VideoToVideo => "/v1/generate/video-to-video",
        ModelCapability::VideoInpainting => "/v1/generate/video-inpainting",
        ModelCapability::VideoUpscale => "/v1/generate/video-upscale",
        _ => "/v1/generate/video-to-video",
    }
}

fn is_valid_video_capability_for_mode(mode: &GenerationMode, capability: &ModelCapability) -> bool {
    match mode {
        GenerationMode::TextToVideo => *capability == ModelCapability::TextToVideo,
        GenerationMode::ImageToVideo => matches!(
            capability,
            ModelCapability::ImageToVideo
                | ModelCapability::MultiImageToVideo
                | ModelCapability::StartEndImageToVideo
                | ModelCapability::KeyframesToVideo
        ),
        GenerationMode::VideoToVideo => matches!(
            capability,
            ModelCapability::VideoToVideo
                | ModelCapability::VideoInpainting
                | ModelCapability::VideoUpscale
        ),
        GenerationMode::TextToImage | GenerationMode::ImageToImage => false,
    }
}

async fn generate_image(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateImageRequest>,
) -> Result<(StatusCode, Json<Generation>), ApiError> {
    state.ensure_accepting_jobs().await?;
    let prompt = request.prompt.trim();
    if prompt.len() < 3 || prompt.len() > 1000 {
        return Err(ApiError::bad_request(
            "Le prompt doit contenir entre 3 et 1000 caractères.",
        ));
    }
    if request.mode == GenerationMode::ImageToImage && request.input_asset_id.is_none() {
        return Err(ApiError::bad_request(
            "input_asset_id est obligatoire en Image → Image.",
        ));
    }
    let model_id = match request.model_id {
        Some(model_id) => model_id,
        None if state.profile == ApplicationProfile::GpuProduction => {
            return Err(ApiError::bad_request(
                "model_id est obligatoire en GPU_PRODUCTION ; aucun moteur procédural ne sera choisi automatiquement.",
            ));
        }
        None => "vidio-canvas-local".into(),
    };
    let entry = resolve_model(&state, &model_id).await?;
    if state.profile == ApplicationProfile::GpuProduction && entry.local {
        return Err(ApiError::conflict(
            "Les moteurs procéduraux sont désactivés pour la génération d'image GPU_PRODUCTION.",
        ));
    }
    let requested_capability = request
        .capability
        .clone()
        .or_else(|| image_capability_mode_default(&request.mode))
        .ok_or_else(|| ApiError::bad_request("Capacité image invalide."))?;

    let expected_capability = match request.mode {
        GenerationMode::TextToImage => ModelCapability::TextToImage,
        GenerationMode::ImageToImage => ModelCapability::ImageToImage,
        _ => {
            return Err(ApiError::bad_request(
                "Utilisez /api/videos/generate pour un mode vidéo.",
            ));
        }
    };
    if !is_valid_image_capability_for_mode(&request.mode, &requested_capability) {
        return Err(ApiError::bad_request(
            "Capacité image invalide pour ce mode.",
        ));
    }
    let view = model_view(&state, &entry).await;
    if !view.runtime_capabilities.contains(&expected_capability)
        || !view.runtime_capabilities.contains(&requested_capability)
    {
        return Err(ApiError::conflict(
            "Ce modèle ne supporte pas ce mode de génération.",
        ));
    }
    if !view.installed {
        return Err(ApiError::conflict("Installez le modèle avant de générer."));
    }
    if model_id != "vidio-canvas-local" && !view.runtime_ready {
        return Err(ApiError::conflict(
            "Le modèle IA est installé mais son runtime n'est pas READY. Chargez-le avant de générer.",
        ));
    }

    let job_id = Uuid::new_v4();
    let generation = Generation {
        id: Uuid::new_v4(),
        job_id: Some(job_id),
        kind: AssetKind::Image,
        mode: request.mode,
        capability: Some(requested_capability),
        prompt: prompt.to_string(),
        negative_prompt: request.negative_prompt,
        model_id,
        input_asset_id: request.input_asset_id,
        mask_asset_id: request.mask_asset_id,
        control_asset_id: request.control_asset_id,
        input_images: vec![],
        output_asset_id: None,
        status: GenerationStatus::Queued,
        progress: 0,
        error: None,
        created_at: unix_now(),
        updated_at: unix_now(),
        duration_seconds: None,
        requested_duration_seconds: None,
        resolution: None,
        requested_quality: None,
        requested_aspect_ratio: None,
        requested_fps: None,
        requested_frames: None,
        inference_frames: None,
        actual_width: None,
        actual_height: None,
        actual_fps: None,
        actual_frames: None,
        actual_duration: None,
        audio: false,
    };
    state
        .generations
        .write()
        .await
        .insert(generation.id, generation.clone());
    let job = Job {
        id: job_id,
        kind: JobKind::GenerateImage,
        target_id: generation.id.to_string(),
        model_id: Some(generation.model_id.clone()),
        capability: generation
            .capability
            .as_ref()
            .map(|value| value.api_name().to_owned()),
        status: JobStatus::Queued,
        stage: "queued".into(),
        progress: 0,
        message: "Génération ajoutée à la file".into(),
        transfer: None,
        dependency: None,
        cache_status: None,
        cache_error: None,
        started_at: None,
        completed_at: None,
        error: None,
        result: None,
        created_at: unix_now(),
        updated_at: unix_now(),
    };
    state.insert_job(job.clone()).await?;
    let worker_state = state.clone();
    let worker_generation = generation.clone();
    tokio::spawn(async move {
        run_generation(worker_state, worker_generation, job.id).await;
    });
    Ok((StatusCode::ACCEPTED, Json(generation)))
}

async fn update_generation(state: &AppState, generation: Generation) {
    state
        .generations
        .write()
        .await
        .insert(generation.id, generation.clone());
    let directory = state.settings.get().await.outputs_dir.join("generations");
    if fs::create_dir_all(&directory).await.is_ok()
        && let Ok(bytes) = serde_json::to_vec_pretty(&generation)
    {
        let temporary = directory.join(format!("{}.json.tmp", generation.id));
        let final_path = directory.join(format!("{}.json", generation.id));
        if fs::write(&temporary, bytes).await.is_ok() {
            let _ = fs::rename(temporary, final_path).await;
        }
    }
    // Le nom de l'événement expose directement la transition réelle au client.
    // `generation.updated` est conservé pour les consommateurs plus anciens.
    let event = match generation.status {
        GenerationStatus::Queued | GenerationStatus::Running => "generation.progress",
        GenerationStatus::Completed => "generation.completed",
        GenerationStatus::Failed => "generation.failed",
        GenerationStatus::Cancelled => "generation.cancelled",
    };
    state.emit(event, &generation);
    state.emit("generation.updated", &generation);
}

async fn await_worker_generation<F>(
    state: &AppState,
    job_id: Uuid,
    request: F,
) -> Result<GenerateResponse, String>
where
    F: Future<Output = Result<GenerateResponse, String>>,
{
    let worker = state
        .worker
        .as_ref()
        .ok_or("WORKER_UNAVAILABLE: worker GPU absent")?;
    let timeout_seconds = std::env::var("VIDIOAI_WORKER_START_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30)
        .clamp(5, 300);
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
    let mut accepted = false;
    let mut request = Box::pin(request);
    loop {
        tokio::select! {
            result = &mut request => return result,
            _ = sleep(Duration::from_millis(700)) => {
                match worker.job_status(&job_id.to_string()).await {
                    Ok(status) => {
                        accepted = true;
                        if status.state == "FAILED" {
                            return Err(format!(
                                "{}: {}",
                                status.error_code.unwrap_or_else(|| "GENERATION_FAILED".into()),
                                status.error.unwrap_or_else(|| "Le worker a échoué".into()),
                            ));
                        }
                        state
                            .update_job(
                                job_id,
                                JobStatus::Running,
                                "generating",
                                status.progress.min(95),
                                "Inférence mesurée par le Worker",
                            )
                            .await;
                    }
                    Err(_) if !accepted && Instant::now() >= deadline => {
                        let _ = worker.cancel(&job_id.to_string()).await;
                        return Err(format!(
                            "WORKER_START_TIMEOUT: job non accepté après {timeout_seconds} secondes"
                        ));
                    }
                    Err(_) => {}
                }
            }
        }
    }
}

/// Pipeline commun T2I/I2I : charge le runtime, exécute le moteur, transforme le
/// résultat en Asset persistant puis clôt la Generation et son Job.
async fn run_generation(state: Arc<AppState>, mut generation: Generation, job_id: Uuid) {
    generation.status = GenerationStatus::Running;
    generation.progress = 12;
    generation.updated_at = unix_now();
    update_generation(&state, generation.clone()).await;
    state
        .update_job(
            job_id,
            JobStatus::Dispatching,
            "dispatching",
            12,
            "Dispatch vers le moteur image",
        )
        .await;

    let result: Result<Asset, String> = async {
        if state
            .cancelled_generations
            .read()
            .await
            .contains(&generation.id)
        {
            return Err("__cancelled__".into());
        }
        let procedural = generation.model_id == "vidio-canvas-local";
        let engine = CanvasEngine;
        if procedural {
            engine.load().await?;
            if !engine.health().await {
                return Err("Le moteur image procédural n'est pas sain.".into());
            }
        }
        generation.progress = 38;
        generation.updated_at = unix_now();
        update_generation(&state, generation.clone()).await;
        state
            .update_job(
                job_id,
                JobStatus::Running,
                "generating",
                38,
                "Création de l'image",
            )
            .await;

        // Les moteurs proceduraux restent 1024x1024 par defaut.
        // Pour les pipelines IA, les dimensions reelles du Worker sont
        // authoritative et sont propagees jusqu'a l'Asset.
        let mut output_dimensions = (1024_u32, 1024_u32);
        let bytes = match generation.mode {
            GenerationMode::TextToImage if !procedural => {
                let worker = state.worker.as_ref().ok_or("Worker GPU absent")?;
                let relative = PathBuf::from("generations").join(format!("{}.png", generation.id));
                let capability = generation
                    .capability
                    .clone()
                    .unwrap_or(ModelCapability::TextToImage);
                let worker_result = await_worker_generation(
                    &state,
                    job_id,
                    worker.generate_image(
                        image_endpoint(&capability),
                        &job_id.to_string(),
                        &storage_id(&generation.model_id),
                        &generation.prompt,
                        generation.negative_prompt.as_deref(),
                        &relative,
                        None,
                        None,
                        None,
                        Some(capability.api_name()),
                    ),
                )
                .await?;
                let worker_width = worker_result.actual_width.unwrap_or(worker_result.width);
                let worker_height = worker_result.actual_height.unwrap_or(worker_result.height);
                if worker_result.job_id != job_id.to_string()
                    || worker_result.state != "COMPLETED"
                    || worker_result.output_relative_path != relative.to_string_lossy()
                    || worker_width == 0
                    || worker_height == 0
                    || worker_result.sha256.len() != 64
                {
                    return Err(format!(
                        "WORKER_RESULT_INVALID: state={} path={} width={} height={} sha256_len={}",
                        worker_result.state,
                        worker_result.output_relative_path,
                        worker_width,
                        worker_height,
                        worker_result.sha256.len(),
                    ));
                }
                output_dimensions = (worker_width, worker_height);
                if let Some(observation) = &worker_result.benchmark
                    && let Ok(entry) = resolve_model(&state, &generation.model_id).await
                {
                    record_worker_benchmark(&state, &entry.id, &entry.revision, observation).await;
                }
                let path = state.settings.get().await.work_dir.join(&relative);
                let content = fs::read(&path).await.map_err(|error| {
                    format!("Sortie worker introuvable sur le volume partagé: {error}")
                })?;
                let _ = fs::remove_file(path).await;
                content
            }
            GenerationMode::TextToImage => engine.text_to_image(&generation.prompt).await?,
            GenerationMode::ImageToImage if !procedural => {
                let worker = state.worker.as_ref().ok_or("Worker GPU absent")?;
                let relative = PathBuf::from("generations").join(format!("{}.png", generation.id));
                let id = generation.input_asset_id.ok_or("Asset source absent")?;
                let (_, path) = read_asset_manifest(&state, id)
                    .await
                    .map_err(|error| error.message)?;
                let input = fs::read(path).await.map_err(|error| error.to_string())?;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|value| value.as_nanos().to_string())
                    .unwrap_or_else(|_| "0".to_string());
                let input_path = state
                    .settings
                    .get()
                    .await
                    .work_dir
                    .join(format!("{}.input-{}.bin", generation.id, timestamp));
                std::fs::write(&input_path, &input).map_err(|error| error.to_string())?;
                let mut temporary_paths = vec![input_path.clone()];
                let mask_path = if let Some(asset_id) = generation.mask_asset_id {
                    let (asset, path) = read_asset_manifest(&state, asset_id)
                        .await
                        .map_err(|error| error.message)?;
                    if asset.kind != AssetKind::Image {
                        return Err("Le masque/contrôle doit être une image.".into());
                    }
                    let bytes = fs::read(path).await.map_err(|error| error.to_string())?;
                    let output = state
                        .settings
                        .get()
                        .await
                        .work_dir
                        .join(format!("{}.mask-{}.png", generation.id, timestamp));
                    fs::write(&output, bytes)
                        .await
                        .map_err(|error| error.to_string())?;
                    Some(output)
                } else {
                    None
                };
                if let Some(path) = &mask_path {
                    temporary_paths.push(path.clone());
                }
                let control_path = if let Some(asset_id) = generation.control_asset_id {
                    let (asset, path) = read_asset_manifest(&state, asset_id)
                        .await
                        .map_err(|error| error.message)?;
                    if asset.kind != AssetKind::Image {
                        return Err("Le masque/contrôle doit être une image.".into());
                    }
                    let bytes = fs::read(path).await.map_err(|error| error.to_string())?;
                    let output = state
                        .settings
                        .get()
                        .await
                        .work_dir
                        .join(format!("{}.control-{}.png", generation.id, timestamp));
                    fs::write(&output, bytes)
                        .await
                        .map_err(|error| error.to_string())?;
                    Some(output)
                } else {
                    None
                };
                if let Some(path) = &control_path {
                    temporary_paths.push(path.clone());
                }
                let capability = generation
                    .capability
                    .clone()
                    .unwrap_or(ModelCapability::ImageToImage);
                let worker_result = await_worker_generation(
                    &state,
                    job_id,
                    worker.generate_image(
                        image_endpoint(&capability),
                        &job_id.to_string(),
                        &storage_id(&generation.model_id),
                        &generation.prompt,
                        generation.negative_prompt.as_deref(),
                        &relative,
                        Some(&input_path.to_string_lossy()),
                        mask_path
                            .as_ref()
                            .map(|path| path.to_string_lossy())
                            .as_deref(),
                        control_path
                            .as_ref()
                            .map(|path| path.to_string_lossy())
                            .as_deref(),
                        Some(capability.api_name()),
                    ),
                )
                .await?;
                let worker_width = worker_result.actual_width.unwrap_or(worker_result.width);
                let worker_height = worker_result.actual_height.unwrap_or(worker_result.height);
                if worker_result.job_id != job_id.to_string()
                    || worker_result.state != "COMPLETED"
                    || worker_result.output_relative_path != relative.to_string_lossy()
                    || worker_width == 0
                    || worker_height == 0
                    || worker_result.sha256.len() != 64
                {
                    return Err(format!(
                        "WORKER_RESULT_INVALID: state={} path={} width={} height={} sha256_len={}",
                        worker_result.state,
                        worker_result.output_relative_path,
                        worker_width,
                        worker_height,
                        worker_result.sha256.len(),
                    ));
                }
                output_dimensions = (worker_width, worker_height);
                let path = state.settings.get().await.work_dir.join(&relative);
                let content = fs::read(&path).await.map_err(|error| {
                    format!("Sortie worker introuvable sur le volume partagé: {error}")
                })?;
                let _ = fs::remove_file(path).await;
                for path in temporary_paths {
                    let _ = fs::remove_file(path).await;
                }
                content
            }
            GenerationMode::ImageToImage => {
                let id = generation.input_asset_id.ok_or("Asset source absent")?;
                let (_, path) = read_asset_manifest(&state, id)
                    .await
                    .map_err(|error| error.message)?;
                let input = fs::read(path).await.map_err(|error| error.to_string())?;
                engine.image_to_image(&generation.prompt, &input).await?
            }
            _ => return Err("Mode incompatible avec le moteur image.".into()),
        };
        if state
            .cancelled_generations
            .read()
            .await
            .contains(&generation.id)
        {
            return Err("__cancelled__".into());
        }
        generation.progress = 82;
        generation.updated_at = unix_now();
        update_generation(&state, generation.clone()).await;
        state
            .update_job(
                job_id,
                JobStatus::SavingOutput,
                "saving_output",
                82,
                "Enregistrement de l'asset final",
            )
            .await;
        let asset = save_asset(
            &state,
            &bytes,
            format!("generation-{}.png", generation.id),
            "image/png".into(),
            AssetKind::Image,
            Some(output_dimensions),
            "png",
        )
        .await
        .map_err(|error| error.message)?;
        publish_generation_asset(&state, generation.id, &asset).await?;
        if procedural {
            engine.unload().await?;
        }
        Ok(asset)
    }
    .await;

    match result {
        Ok(asset) => {
            generation.output_asset_id = Some(asset.id);
            generation.actual_width = asset.width;
            generation.actual_height = asset.height;
            generation.status = GenerationStatus::Completed;
            generation.progress = 100;
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            state
                .set_job_result(
                    job_id,
                    json!({
                        "asset_id": asset.id,
                        "url": asset.url,
                        "mime_type": asset.mime_type,
                        "width": asset.width,
                        "height": asset.height,
                        "asset": asset,
                        "generation": generation,
                    }),
                )
                .await;
            state
                .update_job(
                    job_id,
                    JobStatus::Completed,
                    "completed",
                    100,
                    "Image disponible",
                )
                .await;
        }
        Err(error) if error == "__cancelled__" => {
            let _ = CanvasEngine.cancel().await;
            generation.status = GenerationStatus::Cancelled;
            generation.error = None;
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            state
                .update_job(
                    job_id,
                    JobStatus::Cancelled,
                    "cancelled",
                    generation.progress,
                    "Génération annulée",
                )
                .await;
        }
        Err(error) => {
            // Le contrat commun prévoit l'annulation même si le moteur local n'a
            // aucune ressource distante à interrompre.
            let _ = CanvasEngine.cancel().await;
            generation.status = GenerationStatus::Failed;
            generation.progress = 100;
            generation.error = Some(error.clone());
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            let job_error = state.classify_generation_error(job_id, false, &error).await;
            state
                .update_job(job_id, JobStatus::Failed, "failed", 100, &job_error)
                .await;
        }
    }
}

// -----------------------------------------------------------------------------
// Étapes 13 à 15 : génération vidéo T2V, I2V et V2V
// -----------------------------------------------------------------------------

async fn generate_video(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateVideoRequest>,
) -> Result<(StatusCode, Json<Generation>), ApiError> {
    state.ensure_accepting_jobs().await?;
    if !matches!(
        request.mode,
        GenerationMode::TextToVideo | GenerationMode::ImageToVideo | GenerationMode::VideoToVideo
    ) {
        return Err(ApiError::bad_request(
            "Le mode demandé n'est pas un mode vidéo.",
        ));
    }
    let prompt = request.prompt.trim();
    if prompt.len() < 3 || prompt.len() > 1000 {
        return Err(ApiError::bad_request(
            "Le prompt doit contenir entre 3 et 1000 caractères.",
        ));
    }
    let duration = request.duration_seconds.unwrap_or(6).clamp(2, 15);
    let quality = request.quality.unwrap_or_else(|| "480p".into());
    if !matches!(quality.as_str(), "480p" | "720p" | "1080p") {
        return Err(ApiError::bad_request(
            "La qualité doit être 480p, 720p ou 1080p.",
        ));
    }
    let aspect_ratio = request.aspect_ratio.unwrap_or_else(|| "16:9".into());
    if !matches!(aspect_ratio.as_str(), "16:9" | "9:16" | "1:1") {
        return Err(ApiError::bad_request(
            "Le ratio doit être 16:9, 9:16 ou 1:1.",
        ));
    }
    let requested_fps = request.fps.unwrap_or(24);
    if !(1..=60).contains(&requested_fps) {
        return Err(ApiError::bad_request(
            "Le FPS doit être compris entre 1 et 60.",
        ));
    }

    let mode = request.mode.clone();
    let requested_capability = request
        .capability
        .clone()
        .or_else(|| video_capability_mode_default(&mode))
        .ok_or_else(|| ApiError::bad_request("Capacité vidéo invalide."))?;
    let expected = match mode {
        GenerationMode::TextToVideo => ModelCapability::TextToVideo,
        GenerationMode::ImageToVideo => ModelCapability::ImageToVideo,
        GenerationMode::VideoToVideo => ModelCapability::VideoToVideo,
        _ => unreachable!(),
    };
    let model_id = request
        .model_id
        .unwrap_or_else(|| "vidio-motion-local".into());
    let entry = resolve_model(&state, &model_id).await?;
    if !is_valid_video_capability_for_mode(&mode, &requested_capability) {
        return Err(ApiError::bad_request(
            "Capacité vidéo invalide pour ce mode.",
        ));
    }
    let view = model_view(&state, &entry).await;
    if !view.runtime_capabilities.contains(&expected)
        || !view.runtime_capabilities.contains(&requested_capability)
    {
        return Err(ApiError::conflict(
            "Ce modèle ne supporte pas le mode vidéo choisi.",
        ));
    }
    if !view.installed {
        return Err(ApiError::conflict("Installez le modèle avant de générer."));
    }
    if model_id != "vidio-motion-local" && !view.runtime_ready {
        return Err(ApiError::conflict(
            "MODEL_NOT_READY: le modèle est installé mais son runtime worker n'est pas READY.",
        ));
    }
    let mut input_images = request.input_images.clone();
    if mode != GenerationMode::TextToVideo {
        if mode == GenerationMode::ImageToVideo {
            if let Some(asset_id) = request.input_asset_id
                && input_images.is_empty()
            {
                input_images.push(GenerationInputImage {
                    asset_id,
                    order: 0,
                    role: "start_frame".into(),
                });
            }
            if input_images.is_empty() {
                return Err(ApiError::bad_request(
                    "input_images est obligatoire pour le mode IMAGE_TO_VIDEO.",
                ));
            }
            let unique_ids = input_images
                .iter()
                .map(|item| item.asset_id)
                .collect::<HashSet<_>>();
            if unique_ids.len() != input_images.len() {
                return Err(ApiError::bad_request(
                    "Les images d'entrée ne doivent pas être dupliquées.",
                ));
            }
            let mut ordered = input_images.clone();
            ordered.sort_by_key(|item| item.order);
            for item in &ordered {
                let (asset, _) = read_asset_manifest(&state, item.asset_id).await?;
                if asset.kind != AssetKind::Image {
                    return Err(ApiError::bad_request(
                        "Le type de l'asset source ne correspond pas au mode.",
                    ));
                }
            }
            let _ = ordered;
        } else {
            let id = request.input_asset_id.ok_or_else(|| {
                ApiError::bad_request("input_asset_id est obligatoire pour ce mode.")
            })?;
            let (asset, _) = read_asset_manifest(&state, id).await?;
            if asset.kind != AssetKind::Video {
                return Err(ApiError::bad_request(
                    "Le type de l'asset source ne correspond pas au mode.",
                ));
            }
        }
    }

    let job_id = Uuid::new_v4();
    let generation = Generation {
        id: Uuid::new_v4(),
        job_id: Some(job_id),
        kind: AssetKind::Video,
        mode,
        capability: Some(requested_capability),
        prompt: prompt.to_string(),
        negative_prompt: request.negative_prompt,
        model_id,
        input_asset_id: request.input_asset_id,
        mask_asset_id: None,
        control_asset_id: None,
        input_images: input_images.clone(),
        output_asset_id: None,
        status: GenerationStatus::Queued,
        progress: 0,
        error: None,
        created_at: unix_now(),
        updated_at: unix_now(),
        duration_seconds: Some(duration),
        requested_duration_seconds: Some(f64::from(duration)),
        resolution: Some(quality.clone()),
        requested_quality: Some(quality),
        requested_aspect_ratio: Some(aspect_ratio),
        requested_fps: Some(requested_fps),
        requested_frames: Some(duration.saturating_mul(requested_fps)),
        inference_frames: None,
        actual_width: None,
        actual_height: None,
        actual_fps: None,
        actual_frames: None,
        actual_duration: None,
        audio: request.audio,
    };
    update_generation(&state, generation.clone()).await;
    let job = Job {
        id: job_id,
        kind: JobKind::GenerateVideo,
        target_id: generation.id.to_string(),
        model_id: Some(generation.model_id.clone()),
        capability: generation
            .capability
            .as_ref()
            .map(|value| value.api_name().to_owned()),
        status: JobStatus::Queued,
        stage: "queued".into(),
        progress: 0,
        message: "Vidéo ajoutée à la file".into(),
        transfer: None,
        dependency: None,
        cache_status: None,
        cache_error: None,
        started_at: None,
        completed_at: None,
        error: None,
        result: None,
        created_at: unix_now(),
        updated_at: unix_now(),
    };
    state.insert_job(job.clone()).await?;
    let worker_state = state.clone();
    let returned = generation.clone();
    tokio::spawn(async move {
        run_video_generation(worker_state, generation, job.id).await;
    });
    Ok((StatusCode::ACCEPTED, Json(returned)))
}

/// Encode un MP4 H.264 lisible par les navigateurs. T2V construit d'abord une
/// image déterministe depuis le prompt ; I2V anime l'image fournie ; V2V
/// réencode et stylise la vidéo d'origine sans jamais l'écraser.
fn procedural_video_dimensions(quality: &str, aspect_ratio: &str) -> (u32, u32) {
    let short_side = match quality {
        "1080p" => 1080,
        "720p" => 720,
        _ => 480,
    };
    let mut long_side = ((f64::from(short_side) * 16.0 / 9.0).round()) as u32;
    if !long_side.is_multiple_of(2) {
        long_side += 1;
    }
    match aspect_ratio {
        "9:16" => (short_side, long_side),
        "1:1" => (short_side, short_side),
        _ => (long_side, short_side),
    }
}

async fn run_video_generation(state: Arc<AppState>, mut generation: Generation, job_id: Uuid) {
    generation.status = GenerationStatus::Running;
    generation.progress = 8;
    generation.updated_at = unix_now();
    update_generation(&state, generation.clone()).await;
    state
        .update_job(
            job_id,
            JobStatus::Dispatching,
            "dispatching",
            8,
            "Dispatch vers le moteur vidéo",
        )
        .await;

    let settings = state.settings.get().await;
    let temporary_dir = settings.work_dir.join(generation.id.to_string());
    let _ = fs::create_dir_all(&temporary_dir).await;
    let output_path = temporary_dir.join("result.mp4");
    let progress_path = temporary_dir.join("ffmpeg-progress.txt");
    let duration = generation.duration_seconds.unwrap_or(6);
    let quality = generation
        .requested_quality
        .as_deref()
        .unwrap_or("480p")
        .to_owned();
    let aspect_ratio = generation
        .requested_aspect_ratio
        .as_deref()
        .unwrap_or("16:9")
        .to_owned();
    let requested_fps = generation.requested_fps.unwrap_or(24);
    let (width, height) = procedural_video_dimensions(&quality, &aspect_ratio);

    let result: Result<Asset, String> = async {
        let procedural = generation.model_id == "vidio-motion-local";
        if !procedural {
            let worker = state.worker.as_ref().ok_or("Worker GPU absent")?;
            let capability = generation
                .capability
                .clone()
                .or_else(|| video_capability_mode_default(&generation.mode))
                .ok_or("Capacité vidéo invalide")?;
            let endpoint = video_endpoint(&capability);
            let relative = PathBuf::from("generations").join(format!("{}.mp4", generation.id));

            let mut temporary_inputs = Vec::new();
            let mut video_input_path: Option<String> = None;
            let mut input_images_payload = Vec::new();

            match generation.mode {
                GenerationMode::TextToVideo => {}
                GenerationMode::ImageToVideo => {
                    let mut ordered = generation.input_images.clone();
                    ordered.sort_by_key(|item| item.order);
                    for (index, item) in ordered.iter().enumerate() {
                        let (asset, source_path) = read_asset_manifest(&state, item.asset_id)
                            .await
                            .map_err(|error| error.message)?;
                        if asset.kind != AssetKind::Image {
                            return Err("Le type de l'asset source ne correspond pas au mode.".into());
                        }
                        let bytes = fs::read(&source_path).await.map_err(|error| error.to_string())?;
                        let temporary_path = temporary_dir.join(format!("input-image-{index}.png"));
                        fs::write(&temporary_path, bytes).await.map_err(|error| error.to_string())?;
                        temporary_inputs.push(temporary_path.clone());
                        if index == 0 {
                            video_input_path = Some(temporary_path.to_string_lossy().to_string());
                        }
                        input_images_payload.push(json!({
                            "order": item.order,
                            "role": item.role,
                            "source": temporary_path.to_string_lossy(),
                        }));
                    }
                    if input_images_payload.is_empty()
                        && let Some(asset_id) = generation.input_asset_id
                    {
                        let (asset, source_path) = read_asset_manifest(&state, asset_id)
                            .await
                            .map_err(|error| error.message)?;
                        if asset.kind != AssetKind::Image {
                            return Err("Le type de l'asset source ne correspond pas au mode.".into());
                        }
                        let bytes = fs::read(&source_path).await.map_err(|error| error.to_string())?;
                        let temporary_path = temporary_dir.join("input-image-0.png");
                        fs::write(&temporary_path, bytes).await.map_err(|error| error.to_string())?;
                        temporary_inputs.push(temporary_path.clone());
                        video_input_path = Some(temporary_path.to_string_lossy().to_string());
                        input_images_payload.push(json!({
                            "order": 0,
                            "role": "start_frame",
                            "source": temporary_path.to_string_lossy(),
                        }));
                    }
                }
                GenerationMode::VideoToVideo => {
                    let id = generation.input_asset_id.ok_or("Asset source absent")?;
                    let (asset, source_path) = read_asset_manifest(&state, id)
                        .await
                        .map_err(|error| error.message)?;
                    if asset.kind != AssetKind::Video {
                        return Err("Le type de l'asset source ne correspond pas au mode.".into());
                    }
                    let bytes = fs::read(&source_path).await.map_err(|error| error.to_string())?;
                    let temporary_path = temporary_dir.join("input-video.mp4");
                    fs::write(&temporary_path, bytes).await.map_err(|error| error.to_string())?;
                    temporary_inputs.push(temporary_path.clone());
                    video_input_path = Some(temporary_path.to_string_lossy().to_string());
                }
                GenerationMode::TextToImage | GenerationMode::ImageToImage => {
                    return Err("Mode incompatible avec la génération vidéo.".into());
                }
            }

            generation.progress = 35;
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            state
                .update_job(
                    job_id,
                    JobStatus::Running,
                    "generating",
                    35,
                    "Inférence vidéo par le runtime Worker",
                )
                .await;

            let worker_result = await_worker_generation(
                &state,
                job_id,
                worker.generate_video(
                    endpoint,
                    &job_id.to_string(),
                    &storage_id(&generation.model_id),
                    &generation.prompt,
                    generation.negative_prompt.as_deref(),
                    &relative,
                    video_input_path.as_deref(),
                    if input_images_payload.is_empty() {
                        None
                    } else {
                        Some(serde_json::Value::Array(input_images_payload))
                    },
                    None,
                    Some(capability.api_name()),
                    &quality,
                    &aspect_ratio,
                    duration,
                    requested_fps,
                ),
            )
            .await?;

            for path in temporary_inputs {
                let _ = fs::remove_file(path).await;
            }

            if worker_result.job_id != job_id.to_string()
                || worker_result.state != "COMPLETED"
                || worker_result.output_relative_path != relative.to_string_lossy()
            {
                return Err("Le worker a renvoyé un résultat incohérent.".into());
            }
            generation.actual_width = worker_result.actual_width.or(Some(worker_result.width));
            generation.actual_height = worker_result.actual_height.or(Some(worker_result.height));
            generation.actual_fps = worker_result.actual_fps;
            generation.actual_frames = worker_result.actual_frames;
            generation.actual_duration = worker_result.actual_duration;
            generation.requested_duration_seconds = worker_result.requested_duration_seconds
                .or(generation.requested_duration_seconds);
            generation.requested_fps = worker_result.requested_fps
                .or(generation.requested_fps);
            generation.requested_frames = worker_result.requested_frames
                .or(generation.requested_frames);
            generation.inference_frames = worker_result.inference_frames;
            generation.requested_quality = worker_result.requested_quality
                .or_else(|| generation.requested_quality.clone());
            generation.requested_aspect_ratio = worker_result.requested_aspect_ratio
                .or_else(|| generation.requested_aspect_ratio.clone());

            let shared_output = state.settings.get().await.work_dir.join(&relative);
            let bytes = fs::read(&shared_output).await.map_err(|error| {
                format!("Sortie worker introuvable sur le volume partagé: {error}")
            })?;
            let _ = fs::remove_file(shared_output).await;
            fs::write(&output_path, &bytes)
                .await
                .map_err(|error| format!("Copie sortie worker impossible : {error}"))?;

            state
                .update_job(
                    job_id,
                    JobStatus::SavingOutput,
                    "saving_output",
                    82,
                    "Enregistrement du MP4 final",
                )
                .await;
            let mut asset = save_asset(
                &state,
                &bytes,
                format!("generation-{}.mp4", generation.id),
                "video/mp4".into(),
                AssetKind::Video,
                Some((width, height)),
                "mp4",
            )
            .await
            .map_err(|error| error.message)?;
            if let Some((asset_width, asset_height, asset_duration, fps)) =
                probe_video(&output_path).await
            {
                asset.width = Some(asset_width);
                asset.height = Some(asset_height);
                asset.duration_seconds = Some(asset_duration);
                asset.fps = Some(fps);
                persist_asset_manifest(&state, &asset)
                    .await
                    .map_err(|error| error.message)?;
            }
            publish_generation_asset(&state, generation.id, &asset).await?;
            return Ok(asset);
        }

        let source_path = if generation.mode == GenerationMode::TextToVideo {
            let worker = state.worker.as_ref().ok_or("Worker GPU absent")?;
            let relative = PathBuf::from("generations").join(format!("{}.png", generation.id));
            let worker_result = await_worker_generation(
                &state,
                job_id,
                worker.generate_video(
                    "/v1/generate/text-to-video",
                    &job_id.to_string(),
                    &storage_id(&generation.model_id),
                    &generation.prompt,
                    None,
                    &relative,
                    None,
                    None,
                    None,
                    Some("TEXT_TO_VIDEO"),
                    &quality,
                    &aspect_ratio,
                    duration,
                    requested_fps,
                ),
            )
            .await?;
            if worker_result.job_id != job_id.to_string()
                || worker_result.state != "COMPLETED"
                || worker_result.output_relative_path != relative.to_string_lossy()
            {
                return Err("Le worker a renvoyé un résultat incohérent.".into());
            }
            generation.actual_width = worker_result.actual_width.or(Some(worker_result.width));
            generation.actual_height = worker_result.actual_height.or(Some(worker_result.height));
            generation.actual_fps = worker_result.actual_fps;
            generation.actual_frames = worker_result.actual_frames;
            generation.actual_duration = worker_result.actual_duration;
            generation.requested_duration_seconds = worker_result.requested_duration_seconds
                .or(generation.requested_duration_seconds);
            generation.requested_fps = worker_result.requested_fps
                .or(generation.requested_fps);
            generation.requested_frames = worker_result.requested_frames
                .or(generation.requested_frames);
            generation.inference_frames = worker_result.inference_frames;
            generation.requested_quality = worker_result.requested_quality
                .or_else(|| generation.requested_quality.clone());
            generation.requested_aspect_ratio = worker_result.requested_aspect_ratio
                .or_else(|| generation.requested_aspect_ratio.clone());
            let path = state.settings.get().await.work_dir.join(&relative);
            let bytes = fs::read(&path).await.map_err(|error| {
                format!("Sortie worker introuvable sur le volume partagé: {error}")
            })?;
            let _ = fs::remove_file(path).await;
            let prompt_path = temporary_dir.join("prompt.png");
            fs::write(&prompt_path, bytes).await.map_err(|error| error.to_string())?;
            prompt_path
        } else {
            let id = generation.input_asset_id.ok_or("Asset source absent")?;
            read_asset_manifest(&state, id).await.map_err(|error| error.message)?.1
        };

        generation.progress = 24;
        generation.updated_at = unix_now();
        update_generation(&state, generation.clone()).await;
        state.update_job(job_id, JobStatus::Running, "encoding", 24, "Encodage MP4 en cours").await;

        let mut command = Command::new("ffmpeg");
        command.arg("-y").arg("-loglevel").arg("error");
        if generation.mode == GenerationMode::VideoToVideo {
            command.arg("-i").arg(&source_path)
                .arg("-t").arg(duration.to_string())
                .arg("-vf").arg(format!("scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},eq=contrast=1.04:saturation=1.10"));
            if generation.audio {
                command.args(["-map", "0:v:0", "-map", "0:a?", "-c:a", "aac"]);
            } else {
                command.arg("-an");
            }
        } else {
            command.args(["-loop", "1", "-i"]).arg(&source_path);
            if generation.audio {
                command.args(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest"]);
            }
            command.arg("-t").arg(duration.to_string())
                .arg("-vf").arg(format!("scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},zoompan=z='min(zoom+0.0007,1.08)':d=1:s={width}x{height}:fps={requested_fps},format=yuv420p"));
            if generation.audio { command.args(["-c:a", "aac"]); }
        }
        command.arg("-r").arg(requested_fps.to_string())
            .args(["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            // FFmpeg écrit ici le temps réellement encodé. La progression n'est
            // donc plus un compteur artificiel basé sur une temporisation.
            .arg("-progress").arg(&progress_path).arg("-nostats")
            .arg(&output_path).stdout(Stdio::null()).stderr(Stdio::null());
        let mut child = command.spawn().map_err(|error| format!("FFmpeg indisponible : {error}"))?;
        let mut last_progress = generation.progress;

        loop {
            if state.cancelled_generations.read().await.contains(&generation.id) {
                let _ = child.kill().await;
                return Err("__cancelled__".into());
            }
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                if !status.success() { return Err("FFmpeg n'a pas pu produire la vidéo.".into()); }
                break;
            }
            if let Ok(contents) = fs::read_to_string(&progress_path).await {
                let encoded_microseconds = contents
                    .lines()
                    .rev()
                    .find_map(|line| {
                        line.strip_prefix("out_time_us=")
                            .or_else(|| line.strip_prefix("out_time_ms="))
                            .and_then(|value| value.parse::<u64>().ok())
                    })
                    .unwrap_or_default();
                let ratio = (encoded_microseconds as f64
                    / (f64::from(duration) * 1_000_000.0))
                    .clamp(0.0, 1.0);
                let measured_progress = 24 + (ratio * 64.0).round() as u8;
                if measured_progress > last_progress {
                    last_progress = measured_progress.min(88);
                    generation.progress = last_progress;
                    generation.updated_at = unix_now();
                    update_generation(&state, generation.clone()).await;
                    state
                        .update_job(
                            job_id,
                            JobStatus::Running,
                            "encoding",
                            last_progress,
                            "Encodage MP4 mesuré par FFmpeg",
                        )
                        .await;
                }
            }
            sleep(Duration::from_millis(350)).await;
        }

        let bytes = fs::read(&output_path).await.map_err(|error| error.to_string())?;
        state
            .update_job(
                job_id,
                JobStatus::SavingOutput,
                "saving_output",
                90,
                "Enregistrement du MP4 final",
            )
            .await;
        let mut asset = save_asset(
            &state, &bytes, format!("generation-{}.mp4", generation.id),
            "video/mp4".into(), AssetKind::Video, Some((width, height)), "mp4",
        ).await.map_err(|error| error.message)?;
        if let Some((asset_width, asset_height, asset_duration, fps)) = probe_video(&output_path).await {
            asset.width = Some(asset_width); asset.height = Some(asset_height);
            asset.duration_seconds = Some(asset_duration); asset.fps = Some(fps);
            persist_asset_manifest(&state, &asset).await.map_err(|error| error.message)?;
        }
        publish_generation_asset(&state, generation.id, &asset).await?;
        Ok(asset)
    }.await;

    let _ = fs::remove_dir_all(&temporary_dir).await;
    match result {
        Ok(asset) => {
            generation.actual_width = asset.width.or(generation.actual_width);
            generation.actual_height = asset.height.or(generation.actual_height);
            generation.actual_fps = asset.fps.or(generation.actual_fps);
            generation.actual_duration = asset.duration_seconds.or(generation.actual_duration);
            if generation.actual_frames.is_none() {
                generation.actual_frames = asset
                    .duration_seconds
                    .zip(asset.fps)
                    .map(|(duration, fps)| (duration * fps).round() as u32);
            }
            generation.output_asset_id = Some(asset.id);
            generation.status = GenerationStatus::Completed;
            generation.progress = 100;
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            state
                .set_job_result(
                    job_id,
                    json!({
                        "asset_id": asset.id,
                        "url": asset.url,
                        "mime_type": asset.mime_type,
                        "width": asset.width,
                        "height": asset.height,
                        "duration": asset.duration_seconds,
                        "fps": asset.fps,
                        "frames": generation.actual_frames,
                        "codec": "h264",
                        "asset": asset,
                        "generation": generation,
                    }),
                )
                .await;
            state
                .update_job(
                    job_id,
                    JobStatus::Completed,
                    "completed",
                    100,
                    "Vidéo disponible",
                )
                .await;
        }
        Err(error) if error == "__cancelled__" => {
            generation.status = GenerationStatus::Cancelled;
            generation.error = None;
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            state
                .update_job(
                    job_id,
                    JobStatus::Cancelled,
                    "cancelled",
                    generation.progress,
                    "Génération annulée",
                )
                .await;
        }
        Err(error) => {
            generation.status = GenerationStatus::Failed;
            generation.error = Some(error.clone());
            generation.updated_at = unix_now();
            update_generation(&state, generation.clone()).await;
            let job_error = state.classify_generation_error(job_id, true, &error).await;
            state
                .update_job(
                    job_id,
                    JobStatus::Failed,
                    "failed",
                    generation.progress,
                    &job_error,
                )
                .await;
        }
    }
}

async fn list_generations(State(state): State<Arc<AppState>>) -> Json<Vec<Generation>> {
    let mut generations: Vec<_> = state.generations.read().await.values().cloned().collect();
    generations.sort_by_key(|generation| std::cmp::Reverse(generation.created_at));
    Json(generations)
}

async fn cancel_generation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Generation>, ApiError> {
    let generation = state
        .generations
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Génération inconnue."))?;
    if !matches!(
        generation.status,
        GenerationStatus::Queued | GenerationStatus::Running
    ) {
        return Err(ApiError::conflict("Cette génération est déjà terminée."));
    }
    state.cancelled_generations.write().await.insert(id);
    if let Some(worker) = &state.worker {
        let worker_job = state
            .jobs
            .read()
            .await
            .values()
            .find(|job| job.target_id == id.to_string())
            .map(|job| job.id.to_string());
        if let Some(worker_job) = worker_job {
            let _ = worker.cancel(&worker_job).await;
        }
    }
    Ok(Json(generation))
}

async fn delete_generation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    let generation = state
        .generations
        .write()
        .await
        .remove(&id)
        .ok_or_else(|| ApiError::not_found("Génération inconnue."))?;
    if matches!(
        generation.status,
        GenerationStatus::Queued | GenerationStatus::Running
    ) {
        state.cancelled_generations.write().await.insert(id);
    }
    let path = state
        .settings
        .get()
        .await
        .outputs_dir
        .join("generations")
        .join(format!("{id}.json"));
    let _ = fs::remove_file(path).await;
    Ok(StatusCode::NO_CONTENT)
}

async fn get_generation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Generation>, ApiError> {
    state
        .generations
        .read()
        .await
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| ApiError::not_found("Génération inconnue."))
}

#[cfg(test)]
mod tests {
    use super::{
        AppSettings, CanvasEngine, CloudModelView, CloudRestoreSelection, GenerateVideoRequest,
        GenerationMode, ModelCapability, ModelIdQuery, image_endpoint,
        is_valid_image_capability_for_mode, is_valid_video_capability_for_mode,
        local_runtime_models, procedural_video_dimensions, promote_restored_snapshot,
        restore_identity, seed_restore_staging, select_cloud_manifests, video_endpoint,
        worker_runtime_flags,
    };
    use crate::object_storage::{SnapshotFile, SnapshotManifest};
    use crate::worker::{WorkerModelStatus, WorkerReady};
    use serde_json::json;

    #[test]
    fn cloud_catalog_exposes_valid_manifest_and_restores_only_selection() {
        let manifest = |repository: &str| SnapshotManifest {
            repository: repository.into(),
            revision: "revision-1".into(),
            files: vec![SnapshotFile {
                path: "model.safetensors".into(),
                size: 42,
                sha256: Some("a".repeat(64)),
            }],
            total_size: 42,
            created_at: 1,
            schema_version: 1,
            capabilities: vec!["IMAGE_TO_VIDEO".into()],
        };
        let selected = select_cloud_manifests(
            vec![manifest("owner/first"), manifest("owner/second")],
            &[CloudRestoreSelection {
                repository: "owner/second".into(),
                revision: "revision-1".into(),
            }],
        )
        .unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].repository, "owner/second");

        let view = CloudModelView {
            repository: selected[0].repository.clone(),
            revision: selected[0].revision.clone(),
            name: "second".into(),
            size_bytes: selected[0].total_size,
            files: selected[0].files.len(),
            created_at: selected[0].created_at,
            capabilities: selected[0].capabilities.clone(),
            cloud_state: "AVAILABLE",
            local_state: "ABSENT",
            local: false,
            cloud: true,
            valid: true,
            manifest_uri: "s3://bucket/models/owner/second/revision-1/manifest.json".into(),
        };
        assert_eq!(
            serde_json::to_value(view).unwrap()["cloud_state"],
            "AVAILABLE"
        );
    }

    #[test]
    fn cloud_restore_lock_identity_includes_the_exact_revision() {
        assert_eq!(
            restore_identity("owner/model", "revision-1"),
            "owner/model@revision-1"
        );
        assert_ne!(
            restore_identity("owner/model", "revision-1"),
            restore_identity("owner/model", "revision-2")
        );
    }

    #[tokio::test]
    async fn cloud_restore_staging_is_unique_and_promoted_only_when_complete() {
        use sha2::{Digest, Sha256};

        let root = std::env::temp_dir().join(format!("vidioai-restore-{}", uuid::Uuid::new_v4()));
        let final_path = root.join("model/revision");
        let staging = root.join(".restore/job-id");
        let quarantine = root.join(".restore/job-id.previous");
        tokio::fs::create_dir_all(&final_path).await.unwrap();
        tokio::fs::write(final_path.join("model.safetensors"), b"old")
            .await
            .unwrap();
        let expected = format!("{:x}", Sha256::digest(b"new-valid-weights"));
        let manifest = SnapshotManifest {
            repository: "owner/model".into(),
            revision: "revision".into(),
            files: vec![SnapshotFile {
                path: "model.safetensors".into(),
                size: 17,
                sha256: Some(expected),
            }],
            total_size: 17,
            created_at: 1,
            schema_version: 1,
            capabilities: vec![],
        };

        seed_restore_staging(&final_path, &staging, &manifest)
            .await
            .unwrap();
        tokio::fs::write(staging.join("model.safetensors"), b"new-valid-weights")
            .await
            .unwrap();
        promote_restored_snapshot(&staging, &final_path, &quarantine, &manifest)
            .await
            .unwrap();

        assert_eq!(
            tokio::fs::read(final_path.join("model.safetensors"))
                .await
                .unwrap(),
            b"new-valid-weights"
        );
        assert!(tokio::fs::metadata(&staging).await.is_err());
        assert!(tokio::fs::metadata(&quarantine).await.is_err());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[test]
    fn query_model_ids_preserve_encoded_hugging_face_slashes() {
        for repository in [
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-video-diffusion-img2vid",
        ] {
            let encoded = repository.replace('/', "%2F");
            let query: ModelIdQuery =
                serde_urlencoded::from_str(&format!("model_id={encoded}")).unwrap();
            assert_eq!(query.model_id, repository);
        }
    }

    #[test]
    fn generation_modes_have_stable_json_names() {
        assert_eq!(
            serde_json::to_string(&GenerationMode::TextToImage).unwrap(),
            "\"TEXT_TO_IMAGE\""
        );
    }

    #[test]
    fn local_catalog_only_contains_builtin_runtimes() {
        let repositories: Vec<_> = local_runtime_models()
            .into_iter()
            .map(|entry| entry.repository)
            .collect();
        assert_eq!(repositories, ["local/vidio-canvas", "local/vidio-motion"]);
    }

    #[test]
    fn video_modes_have_stable_json_names() {
        assert_eq!(
            serde_json::to_string(&GenerationMode::ImageToVideo).unwrap(),
            "\"IMAGE_TO_VIDEO\""
        );
        assert_eq!(
            serde_json::to_string(&GenerationMode::VideoToVideo).unwrap(),
            "\"VIDEO_TO_VIDEO\""
        );
    }

    #[test]
    fn video_request_accepts_semantic_quality_and_legacy_resolution_alias() {
        let current: GenerateVideoRequest = serde_json::from_value(json!({
            "mode": "TEXT_TO_VIDEO",
            "prompt": "Une ville futuriste en mouvement",
            "quality": "480p",
            "aspect_ratio": "9:16",
            "fps": 24
        }))
        .unwrap();
        assert_eq!(current.quality.as_deref(), Some("480p"));
        assert_eq!(current.aspect_ratio.as_deref(), Some("9:16"));
        assert_eq!(current.fps, Some(24));

        let legacy: GenerateVideoRequest = serde_json::from_value(json!({
            "mode": "TEXT_TO_VIDEO",
            "prompt": "Une ville futuriste en mouvement",
            "resolution": "720p"
        }))
        .unwrap();
        assert_eq!(legacy.quality.as_deref(), Some("720p"));
    }

    #[test]
    fn procedural_video_dimensions_cover_required_qualities_and_ratios() {
        assert_eq!(procedural_video_dimensions("480p", "16:9"), (854, 480));
        assert_eq!(procedural_video_dimensions("720p", "9:16"), (720, 1280));
        assert_eq!(procedural_video_dimensions("480p", "1:1"), (480, 480));
    }

    #[test]
    fn canvas_palette_is_deterministic() {
        assert_eq!(
            CanvasEngine::palette("ville futuriste"),
            CanvasEngine::palette("ville futuriste")
        );
        assert_ne!(
            CanvasEngine::palette("ville futuriste"),
            CanvasEngine::palette("forêt")
        );
    }

    #[test]
    fn backend_routes_all_image_capabilities_to_expected_worker_endpoints() {
        assert_eq!(
            image_endpoint(&ModelCapability::TextToImage),
            "/v1/generate/text-to-image"
        );
        assert_eq!(
            image_endpoint(&ModelCapability::ImageToImage),
            "/v1/generate/image-to-image"
        );
        assert_eq!(
            image_endpoint(&ModelCapability::Inpainting),
            "/v1/generate/inpainting"
        );
        assert_eq!(
            image_endpoint(&ModelCapability::Outpainting),
            "/v1/generate/outpainting"
        );
        assert_eq!(
            image_endpoint(&ModelCapability::ImageVariation),
            "/v1/generate/image-variation"
        );
        assert_eq!(
            image_endpoint(&ModelCapability::ImageUpscale),
            "/v1/generate/image-upscale"
        );
        assert_eq!(
            image_endpoint(&ModelCapability::ControlledImageGeneration),
            "/v1/generate/controlled-image-generation"
        );
    }

    #[test]
    fn backend_routes_all_video_capabilities_to_expected_worker_endpoints() {
        assert_eq!(
            video_endpoint(&ModelCapability::TextToVideo),
            "/v1/generate/text-to-video"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::ImageToVideo),
            "/v1/generate/image-to-video"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::MultiImageToVideo),
            "/v1/generate/multi-image-to-video"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::StartEndImageToVideo),
            "/v1/generate/start-end-image-to-video"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::KeyframesToVideo),
            "/v1/generate/keyframes-to-video"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::VideoToVideo),
            "/v1/generate/video-to-video"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::VideoInpainting),
            "/v1/generate/video-inpainting"
        );
        assert_eq!(
            video_endpoint(&ModelCapability::VideoUpscale),
            "/v1/generate/video-upscale"
        );
    }

    #[test]
    fn backend_validates_image_mode_capability_families() {
        assert!(is_valid_image_capability_for_mode(
            &GenerationMode::TextToImage,
            &ModelCapability::TextToImage
        ));
        for capability in [
            ModelCapability::ImageToImage,
            ModelCapability::Inpainting,
            ModelCapability::Outpainting,
            ModelCapability::ImageVariation,
            ModelCapability::ImageUpscale,
            ModelCapability::ControlledImageGeneration,
        ] {
            assert!(is_valid_image_capability_for_mode(
                &GenerationMode::ImageToImage,
                &capability
            ));
        }
        assert!(!is_valid_image_capability_for_mode(
            &GenerationMode::TextToImage,
            &ModelCapability::ImageToImage
        ));
        assert!(!is_valid_image_capability_for_mode(
            &GenerationMode::ImageToImage,
            &ModelCapability::TextToImage
        ));
    }

    #[test]
    fn backend_validates_video_mode_capability_families() {
        assert!(is_valid_video_capability_for_mode(
            &GenerationMode::TextToVideo,
            &ModelCapability::TextToVideo
        ));
        for capability in [
            ModelCapability::ImageToVideo,
            ModelCapability::MultiImageToVideo,
            ModelCapability::StartEndImageToVideo,
            ModelCapability::KeyframesToVideo,
        ] {
            assert!(is_valid_video_capability_for_mode(
                &GenerationMode::ImageToVideo,
                &capability
            ));
        }
        for capability in [
            ModelCapability::VideoToVideo,
            ModelCapability::VideoInpainting,
            ModelCapability::VideoUpscale,
        ] {
            assert!(is_valid_video_capability_for_mode(
                &GenerationMode::VideoToVideo,
                &capability
            ));
        }
        assert!(!is_valid_video_capability_for_mode(
            &GenerationMode::VideoToVideo,
            &ModelCapability::ImageToVideo
        ));
        assert!(!is_valid_video_capability_for_mode(
            &GenerationMode::ImageToVideo,
            &ModelCapability::VideoUpscale
        ));
    }

    #[tokio::test]
    async fn invalid_settings_reject_a_file_instead_of_a_directory() {
        let root = std::env::temp_dir().join(format!("vidioai-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let file = root.join("not-a-directory");
        tokio::fs::write(&file, b"test").await.unwrap();
        let settings = AppSettings {
            models_dir: file,
            outputs_dir: root.join("outputs"),
            cache_dir: root.join("cache"),
            work_dir: root.join("work"),
            auto_unload_minutes: 15,
            automatic_optimization: true,
        };
        assert!(settings.ensure_directories().await.is_err());
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    #[test]
    fn backend_ready_is_false_when_worker_not_ready() {
        let status = WorkerModelStatus {
            model_id: "wan".into(),
            state: "INSTALLED".into(),
            repository: Some("Wan-AI/Wan2.2-TI2V-5B-Diffusers".into()),
            revision: Some("rev".into()),
            installed: true,
            loaded: false,
            ready: false,
            weights_valid: true,
            runtime_available: true,
            runtime_compatible: true,
            validation_test: false,
            error: None,
            benchmark: None,
            runtime_dependencies: vec![],
            precision_plan: None,
            memory_plan: None,
            capabilities: vec![],
            capability: None,
            pipeline_class: None,
            device: None,
            stage: None,
        };
        assert!(!super::worker_reports_ready(&status));
    }

    #[test]
    fn backend_ready_is_true_only_when_worker_confirms_ready() {
        let status = WorkerModelStatus {
            model_id: "wan".into(),
            state: "READY".into(),
            repository: Some("Wan-AI/Wan2.2-TI2V-5B-Diffusers".into()),
            revision: Some("rev".into()),
            installed: true,
            loaded: true,
            ready: true,
            weights_valid: true,
            runtime_available: true,
            runtime_compatible: true,
            validation_test: true,
            error: None,
            benchmark: None,
            runtime_dependencies: vec![],
            precision_plan: None,
            memory_plan: None,
            capabilities: vec![],
            capability: None,
            pipeline_class: None,
            device: None,
            stage: None,
        };
        assert!(super::worker_reports_ready(&status));
    }

    #[test]
    fn backend_ready_does_not_require_a_hidden_inference_during_load() {
        let status = WorkerModelStatus {
            model_id: "video-model".into(),
            state: "READY".into(),
            repository: Some("example/video-model".into()),
            revision: Some("rev".into()),
            installed: true,
            loaded: true,
            ready: true,
            weights_valid: true,
            runtime_available: true,
            runtime_compatible: true,
            validation_test: false,
            error: None,
            benchmark: None,
            runtime_dependencies: vec![],
            precision_plan: None,
            memory_plan: None,
            capabilities: vec![],
            capability: None,
            pipeline_class: None,
            device: None,
            stage: None,
        };
        assert!(super::worker_reports_ready(&status));
    }

    #[test]
    fn worker_cuda_and_runtime_flags_map_without_contradiction() {
        let status = WorkerReady {
            ready: true,
            profile: "GPU_PRODUCTION".into(),
            runtime_available: true,
            cuda_available: true,
            gpu_required: true,
            scratch_mount_ok: true,
            scratch_filesystem: Some("device:contract".into()),
            scratch_total_bytes: 1_500_000_000_000,
            scratch_available_bytes: 1_400_000_000_000,
            errors: vec![],
        };
        assert_eq!(worker_runtime_flags(Some(&status)), (true, true, true));
    }
}

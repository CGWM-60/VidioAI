//! Registre versionné et activable des ModelPacks.
//!
//! Chaque artefact contient le pack complet et un SHA-256 de sa représentation
//! canonique. L'index, l'artefact et le pointeur actif sont écrits par rename
//! atomique. Un téléchargement S3 n'est jamais activé avant validation.

use crate::{model_pack::ModelPack, object_storage::ObjectStorage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{fs, sync::RwLock};
use uuid::Uuid;

pub const REGISTRY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionedPackManifest {
    pub schema_version: u32,
    pub id: String,
    pub version: String,
    pub sha256: String,
    pub min_vidioai_version: String,
    pub workflow_version: String,
    #[serde(default)]
    pub workflows: Vec<WorkflowArtifactRecord>,
    pub created_at: u64,
    pub pack: ModelPack,
}

impl VersionedPackManifest {
    pub fn new(
        pack: ModelPack,
        version: String,
        min_vidioai_version: String,
        workflow_version: String,
        created_at: u64,
    ) -> Result<Self, String> {
        pack.validate()?;
        let sha256 = pack_sha256(&pack)?;
        Ok(Self {
            schema_version: REGISTRY_SCHEMA_VERSION,
            id: pack.id.clone(),
            version,
            sha256,
            min_vidioai_version,
            workflow_version,
            workflows: Vec::new(),
            created_at,
            pack,
        })
    }

    pub fn validate(&self, expected_id: &str, expected_version: &str) -> Result<(), String> {
        if self.schema_version != REGISTRY_SCHEMA_VERSION
            || self.id != expected_id
            || self.pack.id != expected_id
            || self.version != expected_version
        {
            return Err("MODEL_PACK_ARTIFACT_INVALID: identité/version incohérente".into());
        }
        self.pack.validate()?;
        let actual = pack_sha256(&self.pack)?;
        if actual != self.sha256 {
            return Err(format!(
                "MODEL_PACK_CHECKSUM_MISMATCH: attendu {}, obtenu {actual}",
                self.sha256
            ));
        }
        let expected_prefix = format!("workflows/{}/", self.workflow_version);
        if self.workflows.iter().any(|workflow| {
            !workflow.template.starts_with(&expected_prefix)
                || workflow
                    .template
                    .split('/')
                    .any(|part| part == ".." || part.is_empty())
                || workflow.sha256.len() != 64
                || !workflow.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        }) {
            return Err("WORKFLOW_INVALID: métadonnées de workflow versionné invalides".into());
        }
        if self.workflows.len() != self.pack.workflow_by_capability.len()
            || self
                .pack
                .workflow_by_capability
                .iter()
                .any(|(capability, template)| {
                    let expected_filename = Path::new(template)
                        .file_name()
                        .and_then(|value| value.to_str());
                    !self.workflows.iter().any(|workflow| {
                        workflow.capability == *capability
                            && Path::new(&workflow.template)
                                .file_name()
                                .and_then(|value| value.to_str())
                                == expected_filename
                    })
                })
        {
            return Err("WORKFLOW_INVALID: mapping capability/template incohérent".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackVersionRecord {
    pub id: String,
    pub family: String,
    pub version: String,
    pub sha256: String,
    pub min_vidioai_version: String,
    pub workflow_version: String,
    #[serde(default)]
    pub workflows: Vec<WorkflowArtifactRecord>,
    pub active: bool,
    pub source: String,
    pub created_at: u64,
    pub published_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowArtifactRecord {
    pub capability: String,
    pub template: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RegistryIndex {
    schema_version: u32,
    packs: Vec<PackVersionRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PackRegistryResponse {
    pub packs: Vec<PackVersionRecord>,
    pub manifests: Vec<VersionedPackManifest>,
}

#[derive(Debug, Clone)]
pub struct VersionedPackRegistry {
    root: PathBuf,
    index: Arc<RwLock<RegistryIndex>>,
}

impl VersionedPackRegistry {
    pub async fn open(
        root: PathBuf,
        bundled: Vec<ModelPack>,
        bundled_workflows: Option<&Path>,
        now: u64,
    ) -> Result<Self, String> {
        fs::create_dir_all(root.join("model-packs"))
            .await
            .map_err(|error| error.to_string())?;
        fs::create_dir_all(root.join("active"))
            .await
            .map_err(|error| error.to_string())?;
        fs::create_dir_all(root.join("workflows"))
            .await
            .map_err(|error| error.to_string())?;
        let index_path = root.join("registry.json");
        let mut index = match fs::read(&index_path).await {
            Ok(bytes) => serde_json::from_slice::<RegistryIndex>(&bytes)
                .map_err(|error| format!("MODEL_PACK_REGISTRY_INVALID: {error}"))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => RegistryIndex {
                schema_version: REGISTRY_SCHEMA_VERSION,
                packs: Vec::new(),
            },
            Err(error) => return Err(error.to_string()),
        };
        if index.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err("MODEL_PACK_REGISTRY_INVALID: schema non supporté".into());
        }
        for pack in bundled {
            // schema_version décrit le format du contrat, pas la version du pack.
            // Le patch semver est dérivé du contenu canonique : modifier un pack
            // crée donc automatiquement une nouvelle version visible/rollbackable
            // sans casser la compatibilité du schema.
            let fingerprint = pack_sha256(&pack)?;
            let patch = u64::from_str_radix(&fingerprint[..8], 16)
                .map_err(|error| format!("MODEL_PACK_VERSION_INVALID: {error}"))?;
            let version = format!("{}.0.{patch}", pack.schema_version);
            if index
                .packs
                .iter()
                .any(|entry| entry.id == pack.id && entry.version == version)
            {
                continue;
            }
            let mut manifest = VersionedPackManifest::new(
                pack,
                version.clone(),
                env!("CARGO_PKG_VERSION").into(),
                "1".into(),
                now,
            )?;
            let workflows = if let Some(source_root) = bundled_workflows {
                seed_workflows(&root, source_root, &manifest).await?
            } else {
                Vec::new()
            };
            manifest.workflows = workflows.clone();
            write_artifact(&root, &manifest).await?;
            let has_active = index
                .packs
                .iter()
                .any(|entry| entry.id == manifest.id && entry.active);
            index.packs.push(PackVersionRecord {
                id: manifest.id.clone(),
                family: manifest.pack.family.clone(),
                version: manifest.version.clone(),
                sha256: manifest.sha256.clone(),
                min_vidioai_version: manifest.min_vidioai_version.clone(),
                workflow_version: manifest.workflow_version.clone(),
                workflows,
                active: !has_active,
                source: "bundled".into(),
                created_at: manifest.created_at,
                published_at: None,
            });
            if !has_active {
                write_active_pointer(&root, &manifest).await?;
            }
        }
        persist_index(&root, &index).await?;
        let registry = Self {
            root,
            index: Arc::new(RwLock::new(index)),
        };
        // Valide toutes les versions actives avant de rendre le backend prêt.
        registry.active_packs().await?;
        Ok(registry)
    }

    pub async fn list(&self) -> PackRegistryResponse {
        let mut packs = self.index.read().await.packs.clone();
        packs.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| right.created_at.cmp(&left.created_at))
        });
        let mut manifests = Vec::new();
        for pack in &packs {
            if let Ok(manifest) = self.read_verified(&pack.id, &pack.version).await {
                manifests.push(manifest);
            }
        }
        PackRegistryResponse { packs, manifests }
    }

    pub async fn active_version(&self, id: &str) -> Option<PackVersionRecord> {
        self.index
            .read()
            .await
            .packs
            .iter()
            .find(|entry| entry.id == id && entry.active)
            .cloned()
    }

    pub async fn active_packs(&self) -> Result<Vec<ModelPack>, String> {
        let entries = self
            .index
            .read()
            .await
            .packs
            .iter()
            .filter(|entry| entry.active)
            .cloned()
            .collect::<Vec<_>>();
        let mut packs = Vec::with_capacity(entries.len());
        for entry in entries {
            packs.push(self.read_verified(&entry.id, &entry.version).await?.pack);
        }
        Ok(packs)
    }

    pub async fn activate(
        &self,
        id: &str,
        version: &str,
        current_vidioai_version: &str,
    ) -> Result<PackVersionRecord, String> {
        let manifest = self.read_verified(id, version).await?;
        if version_is_newer(&manifest.min_vidioai_version, current_vidioai_version) {
            return Err(format!(
                "MODEL_PACK_VERSION_INCOMPATIBLE: VidioAI {} requis",
                manifest.min_vidioai_version
            ));
        }
        write_active_pointer(&self.root, &manifest).await?;
        let mut index = self.index.write().await;
        if !index
            .packs
            .iter()
            .any(|entry| entry.id == id && entry.version == version)
        {
            index.packs.push(PackVersionRecord {
                id: manifest.id.clone(),
                family: manifest.pack.family.clone(),
                version: manifest.version.clone(),
                sha256: manifest.sha256.clone(),
                min_vidioai_version: manifest.min_vidioai_version.clone(),
                workflow_version: manifest.workflow_version.clone(),
                workflows: workflow_records(&self.root, &manifest).await?,
                active: false,
                source: "downloaded".into(),
                created_at: manifest.created_at,
                published_at: None,
            });
        }
        for entry in index.packs.iter_mut().filter(|entry| entry.id == id) {
            entry.active = entry.version == version;
        }
        let active = index
            .packs
            .iter()
            .find(|entry| entry.id == id && entry.active)
            .cloned()
            .ok_or_else(|| "MODEL_PACK_REGISTRY_INVALID: activation perdue".to_owned())?;
        persist_index(&self.root, &index).await?;
        Ok(active)
    }

    pub async fn rollback(
        &self,
        id: &str,
        requested_version: Option<&str>,
        current_vidioai_version: &str,
    ) -> Result<PackVersionRecord, String> {
        let index = self.index.read().await;
        let current = index
            .packs
            .iter()
            .find(|entry| entry.id == id && entry.active)
            .ok_or_else(|| "MODEL_PACK_NOT_FOUND: pack actif inconnu".to_owned())?;
        let version = if let Some(version) = requested_version {
            version.to_owned()
        } else {
            index
                .packs
                .iter()
                .filter(|entry| entry.id == id && entry.version != current.version)
                .max_by_key(|entry| entry.created_at)
                .map(|entry| entry.version.clone())
                .ok_or_else(|| {
                    "MODEL_PACK_ROLLBACK_UNAVAILABLE: aucune version antérieure".to_owned()
                })?
        };
        drop(index);
        self.activate(id, &version, current_vidioai_version).await
    }

    pub async fn publish(
        &self,
        id: &str,
        version: Option<&str>,
        storage: &dyn ObjectStorage,
        now: u64,
    ) -> Result<PackVersionRecord, String> {
        let selected = if let Some(version) = version {
            self.index
                .read()
                .await
                .packs
                .iter()
                .find(|entry| entry.id == id && entry.version == version)
                .cloned()
        } else {
            self.active_version(id).await
        }
        .ok_or_else(|| "MODEL_PACK_NOT_FOUND: version à publier inconnue".to_owned())?;
        self.read_verified(id, &selected.version).await?;
        let path = artifact_path(&self.root, id, &selected.version);
        if storage.enabled() {
            storage
                .upload_file(&path, &artifact_key(id, &selected.version)?)
                .await?;
            for workflow in &selected.workflows {
                let filename = Path::new(&workflow.template)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| "WORKFLOW_INVALID: nom absent".to_owned())?;
                validate_segment(filename)?;
                let local = self.root.join(&workflow.template);
                storage
                    .upload_file(
                        &local,
                        &format!(
                            "model-packs/{}/{}/workflows/{filename}",
                            selected.id, selected.version
                        ),
                    )
                    .await?;
            }
            let mut published = self
                .index
                .read()
                .await
                .packs
                .iter()
                .filter(|entry| entry.published_at.is_some())
                .cloned()
                .collect::<Vec<_>>();
            let mut published_selected = selected.clone();
            published_selected.published_at = Some(now);
            published_selected.source = "s3".into();
            if let Some(entry) = published.iter_mut().find(|entry| {
                entry.id == published_selected.id && entry.version == published_selected.version
            }) {
                *entry = published_selected;
            } else {
                published.push(published_selected);
            }
            let remote_index = PublishedRegistryIndex {
                schema_version: REGISTRY_SCHEMA_VERSION,
                versions: published,
            };
            let temporary = self
                .root
                .join(format!("published-index.{}.tmp", Uuid::new_v4()));
            fs::write(
                &temporary,
                serde_json::to_vec_pretty(&remote_index).map_err(|error| error.to_string())?,
            )
            .await
            .map_err(|error| error.to_string())?;
            let upload = storage
                .upload_file(&temporary, "model-packs/registry.json")
                .await;
            let _ = fs::remove_file(&temporary).await;
            upload?;
        }
        let mut index = self.index.write().await;
        let entry = index
            .packs
            .iter_mut()
            .find(|entry| entry.id == id && entry.version == selected.version)
            .ok_or_else(|| "MODEL_PACK_NOT_FOUND: version à publier inconnue".to_owned())?;
        entry.published_at = Some(now);
        entry.source = if storage.enabled() { "s3" } else { "local" }.into();
        let result = entry.clone();
        persist_index(&self.root, &index).await?;
        Ok(result)
    }

    pub async fn ensure_local_from_storage(
        &self,
        id: &str,
        version: &str,
        storage: &dyn ObjectStorage,
    ) -> Result<(), String> {
        let destination = artifact_path(&self.root, id, version);
        if fs::try_exists(&destination).await.unwrap_or(false) {
            return self.read_verified(id, version).await.map(|_| ());
        }
        if !storage.enabled() {
            return Err("MODEL_PACK_NOT_FOUND: artefact local absent et S3 désactivé".into());
        }
        let parent = destination
            .parent()
            .ok_or_else(|| "MODEL_PACK_PATH_INVALID: parent absent".to_owned())?;
        fs::create_dir_all(parent)
            .await
            .map_err(|error| error.to_string())?;
        let temporary = parent.join(format!("{}.download.tmp", Uuid::new_v4()));
        let downloaded = storage
            .download_file(&artifact_key(id, version)?, &temporary)
            .await?;
        if !downloaded {
            return Err("MODEL_PACK_NOT_FOUND: artefact S3 absent".into());
        }
        let bytes = fs::read(&temporary)
            .await
            .map_err(|error| error.to_string())?;
        let manifest: VersionedPackManifest = serde_json::from_slice(&bytes)
            .map_err(|error| format!("MODEL_PACK_ARTIFACT_INVALID: {error}"))?;
        if let Err(error) = manifest.validate(id, version) {
            let _ = fs::remove_file(&temporary).await;
            return Err(error);
        }
        for workflow in &manifest.workflows {
            let filename = Path::new(&workflow.template)
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| "WORKFLOW_INVALID: nom absent".to_owned())?;
            validate_segment(filename)?;
            let destination = self.root.join(&workflow.template);
            let parent = destination
                .parent()
                .ok_or_else(|| "WORKFLOW_INVALID: parent absent".to_owned())?;
            fs::create_dir_all(parent)
                .await
                .map_err(|error| error.to_string())?;
            let workflow_temporary =
                parent.join(format!("{filename}.download.{}.tmp", Uuid::new_v4()));
            let downloaded = storage
                .download_file(
                    &format!("model-packs/{id}/{version}/workflows/{filename}"),
                    &workflow_temporary,
                )
                .await?;
            if !downloaded {
                let _ = fs::remove_file(&temporary).await;
                return Err(format!("WORKFLOW_INVALID: artefact S3 {filename} absent"));
            }
            let workflow_bytes = fs::read(&workflow_temporary)
                .await
                .map_err(|error| error.to_string())?;
            validate_workflow_bytes(&workflow_bytes)?;
            let actual = format!("{:x}", Sha256::digest(&workflow_bytes));
            if actual != workflow.sha256 {
                let _ = fs::remove_file(&workflow_temporary).await;
                let _ = fs::remove_file(&temporary).await;
                return Err(format!(
                    "WORKFLOW_CHECKSUM_MISMATCH: {filename} attendu {}, obtenu {actual}",
                    workflow.sha256
                ));
            }
            fs::rename(workflow_temporary, destination)
                .await
                .map_err(|error| error.to_string())?;
        }
        fs::rename(&temporary, &destination)
            .await
            .map_err(|error| error.to_string())
    }

    pub async fn synchronize_from_storage(
        &self,
        storage: &dyn ObjectStorage,
    ) -> Result<Vec<PackVersionRecord>, String> {
        if !storage.enabled() {
            return Ok(Vec::new());
        }
        let temporary = self
            .root
            .join(format!("remote-index.{}.tmp", Uuid::new_v4()));
        if !storage
            .download_file("model-packs/registry.json", &temporary)
            .await?
        {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&temporary)
            .await
            .map_err(|error| error.to_string())?;
        let _ = fs::remove_file(&temporary).await;
        let remote: PublishedRegistryIndex = serde_json::from_slice(&bytes)
            .map_err(|error| format!("MODEL_PACK_REGISTRY_INVALID: {error}"))?;
        if remote.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err("MODEL_PACK_REGISTRY_INVALID: schema S3 non supporté".into());
        }
        let mut candidates = Vec::new();
        for mut record in remote.versions {
            validate_segment(&record.id)?;
            validate_segment(&record.version)?;
            record.active = false;
            record.source = "s3".into();
            if record.published_at.is_none() {
                return Err("MODEL_PACK_REGISTRY_INVALID: version S3 non publiée".into());
            }
            self.ensure_local_from_storage(&record.id, &record.version, storage)
                .await?;
            let manifest = self.read_verified(&record.id, &record.version).await?;
            if record.sha256 != manifest.sha256
                || record.workflow_version != manifest.workflow_version
                || record.workflows != manifest.workflows
                || record.min_vidioai_version != manifest.min_vidioai_version
            {
                return Err(
                    "MODEL_PACK_REGISTRY_INVALID: index S3 différent du manifest vérifié".into(),
                );
            }
            candidates.push(record);
        }
        let mut discovered = Vec::new();
        let mut index = self.index.write().await;
        for record in candidates {
            if index
                .packs
                .iter()
                .any(|entry| entry.id == record.id && entry.version == record.version)
            {
                continue;
            }
            discovered.push(record.clone());
            index.packs.push(record);
        }
        persist_index(&self.root, &index).await?;
        Ok(discovered)
    }

    async fn read_verified(
        &self,
        id: &str,
        version: &str,
    ) -> Result<VersionedPackManifest, String> {
        validate_segment(id)?;
        validate_segment(version)?;
        let bytes = fs::read(artifact_path(&self.root, id, version))
            .await
            .map_err(|error| format!("MODEL_PACK_NOT_FOUND: {error}"))?;
        let manifest: VersionedPackManifest = serde_json::from_slice(&bytes)
            .map_err(|error| format!("MODEL_PACK_ARTIFACT_INVALID: {error}"))?;
        manifest.validate(id, version)?;
        verify_workflows(&self.root, &manifest).await?;
        Ok(manifest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PublishedRegistryIndex {
    schema_version: u32,
    versions: Vec<PackVersionRecord>,
}

fn pack_sha256(pack: &ModelPack) -> Result<String, String> {
    let bytes = serde_json::to_vec(pack).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_segment(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || value.starts_with('.')
        || !value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return Err("MODEL_PACK_PATH_INVALID: identifiant/version invalide".into());
    }
    Ok(())
}

fn artifact_key(id: &str, version: &str) -> Result<String, String> {
    validate_segment(id)?;
    validate_segment(version)?;
    Ok(format!("model-packs/{id}/{version}/manifest.json"))
}

fn artifact_path(root: &Path, id: &str, version: &str) -> PathBuf {
    root.join("model-packs")
        .join(id)
        .join(version)
        .join("manifest.json")
}

async fn seed_workflows(
    root: &Path,
    source_root: &Path,
    manifest: &VersionedPackManifest,
) -> Result<Vec<WorkflowArtifactRecord>, String> {
    validate_segment(&manifest.workflow_version)?;
    let destination_root = root.join("workflows").join(&manifest.workflow_version);
    fs::create_dir_all(&destination_root)
        .await
        .map_err(|error| error.to_string())?;
    let mut records = Vec::new();
    for (capability, relative) in &manifest.pack.workflow_by_capability {
        let filename = Path::new(relative)
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "WORKFLOW_INVALID: nom de template absent".to_owned())?;
        validate_segment(filename)?;
        let source = source_root.join(relative);
        let destination = destination_root.join(filename);
        let bytes = if fs::try_exists(&destination).await.unwrap_or(false) {
            fs::read(&destination)
                .await
                .map_err(|error| error.to_string())?
        } else {
            fs::read(&source)
                .await
                .map_err(|error| format!("WORKFLOW_INVALID: {}: {error}", source.display()))?
        };
        let _: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("WORKFLOW_INVALID: {}: {error}", source.display()))?;
        if !fs::try_exists(&destination).await.unwrap_or(false) {
            let temporary = destination_root.join(format!("{filename}.{}.tmp", Uuid::new_v4()));
            fs::write(&temporary, &bytes)
                .await
                .map_err(|error| error.to_string())?;
            fs::rename(temporary, &destination)
                .await
                .map_err(|error| error.to_string())?;
        }
        records.push(WorkflowArtifactRecord {
            capability: capability.clone(),
            template: format!("workflows/{}/{filename}", manifest.workflow_version),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        });
    }
    records.sort_by(|left, right| left.capability.cmp(&right.capability));
    Ok(records)
}

async fn workflow_records(
    root: &Path,
    manifest: &VersionedPackManifest,
) -> Result<Vec<WorkflowArtifactRecord>, String> {
    validate_segment(&manifest.workflow_version)?;
    let mut records = Vec::new();
    for (capability, template) in &manifest.pack.workflow_by_capability {
        let filename = Path::new(template)
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "WORKFLOW_INVALID: nom absent".to_owned())?;
        validate_segment(filename)?;
        let path = root
            .join("workflows")
            .join(&manifest.workflow_version)
            .join(filename);
        let bytes = fs::read(path)
            .await
            .map_err(|error| format!("WORKFLOW_INVALID: {error}"))?;
        records.push(WorkflowArtifactRecord {
            capability: capability.clone(),
            template: format!("workflows/{}/{filename}", manifest.workflow_version),
            sha256: format!("{:x}", Sha256::digest(bytes)),
        });
    }
    Ok(records)
}

fn validate_workflow_bytes(bytes: &[u8]) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("WORKFLOW_INVALID: {error}"))?;
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
        || value
            .get("workflow")
            .and_then(serde_json::Value::as_object)
            .is_none_or(|nodes| nodes.is_empty())
    {
        return Err("WORKFLOW_INVALID: schema ou nodes absents".into());
    }
    Ok(())
}

async fn verify_workflows(root: &Path, manifest: &VersionedPackManifest) -> Result<(), String> {
    for workflow in &manifest.workflows {
        let path = root.join(&workflow.template);
        let bytes = fs::read(&path)
            .await
            .map_err(|error| format!("WORKFLOW_INVALID: {}: {error}", path.display()))?;
        validate_workflow_bytes(&bytes)?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != workflow.sha256 {
            return Err(format!(
                "WORKFLOW_CHECKSUM_MISMATCH: {} attendu {}, obtenu {actual}",
                workflow.template, workflow.sha256
            ));
        }
    }
    Ok(())
}

async fn write_artifact(root: &Path, manifest: &VersionedPackManifest) -> Result<(), String> {
    manifest.validate(&manifest.id, &manifest.version)?;
    let destination = artifact_path(root, &manifest.id, &manifest.version);
    let parent = destination
        .parent()
        .ok_or_else(|| "MODEL_PACK_PATH_INVALID: parent absent".to_owned())?;
    fs::create_dir_all(parent)
        .await
        .map_err(|error| error.to_string())?;
    let temporary = parent.join(format!("{}.tmp", Uuid::new_v4()));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(manifest).map_err(|error| error.to_string())?,
    )
    .await
    .map_err(|error| error.to_string())?;
    fs::rename(temporary, destination)
        .await
        .map_err(|error| error.to_string())
}

async fn write_active_pointer(root: &Path, manifest: &VersionedPackManifest) -> Result<(), String> {
    let destination = root.join("active").join(format!("{}.json", manifest.id));
    let temporary = root
        .join("active")
        .join(format!("{}.{}.tmp", manifest.id, Uuid::new_v4()));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(manifest).map_err(|error| error.to_string())?,
    )
    .await
    .map_err(|error| error.to_string())?;
    fs::rename(temporary, destination)
        .await
        .map_err(|error| error.to_string())
}

async fn persist_index(root: &Path, index: &RegistryIndex) -> Result<(), String> {
    let destination = root.join("registry.json");
    let temporary = root.join(format!("registry.{}.tmp", Uuid::new_v4()));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(index).map_err(|error| error.to_string())?,
    )
    .await
    .map_err(|error| error.to_string())?;
    fs::rename(temporary, destination)
        .await
        .map_err(|error| error.to_string())
}

fn version_is_newer(required: &str, current: &str) -> bool {
    fn parts(value: &str) -> [u64; 3] {
        let mut result = [0; 3];
        for (index, part) in value.split(['.', '-']).take(3).enumerate() {
            result[index] = part.parse().unwrap_or(u64::MAX);
        }
        result
    }
    parts(required) > parts(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_pack::{
        CatalogModelStatus, EngineKind, ModelPackComponents, ModelPackDefaults,
        ModelPackMemoryPolicy,
    };
    use crate::object_storage::{
        SnapshotManifest, TransferCancellationToken, TransferProgressCallback, UploadOutcome,
        UploadProgressCallback,
    };
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, sync::Arc};
    use tokio::sync::RwLock;

    #[derive(Debug, Clone, Default)]
    struct MemoryObjectStorage {
        objects: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
    }

    #[async_trait]
    impl ObjectStorage for MemoryObjectStorage {
        fn enabled(&self) -> bool {
            true
        }

        fn snapshot_uri(&self, repository: &str, revision: &str) -> Result<String, String> {
            Ok(format!("memory://{repository}/{revision}"))
        }

        async fn health(&self) -> Result<(), String> {
            Ok(())
        }

        async fn upload_file(&self, local: &Path, key: &str) -> Result<(), String> {
            let bytes = fs::read(local).await.map_err(|error| error.to_string())?;
            self.objects.write().await.insert(key.into(), bytes);
            Ok(())
        }

        async fn download_file(&self, key: &str, local: &Path) -> Result<bool, String> {
            let Some(bytes) = self.objects.read().await.get(key).cloned() else {
                return Ok(false);
            };
            fs::write(local, bytes)
                .await
                .map_err(|error| error.to_string())?;
            Ok(true)
        }

        async fn restore_snapshot(
            &self,
            _repository: &str,
            _revision: &str,
            _local: &Path,
            _progress: Option<TransferProgressCallback>,
        ) -> Result<bool, String> {
            Ok(false)
        }

        async fn list_snapshots(&self) -> Result<Vec<SnapshotManifest>, String> {
            Ok(Vec::new())
        }

        async fn upload_snapshot(
            &self,
            _repository: &str,
            _revision: &str,
            _local: &Path,
            _progress: Option<UploadProgressCallback>,
            _cancellation: Option<TransferCancellationToken>,
        ) -> Result<UploadOutcome, String> {
            Err("unused".into())
        }
    }

    fn pack() -> ModelPack {
        ModelPack {
            schema_version: 1,
            id: "fixture-pack".into(),
            family: "fixture".into(),
            status: CatalogModelStatus::Ready,
            engine: EngineKind::Comfyui,
            capabilities: vec!["TEXT_TO_IMAGE".into()],
            architectures: vec!["Fixture".into()],
            pipeline_classes: vec!["FixturePipeline".into()],
            workflow_by_capability: BTreeMap::from([(
                "TEXT_TO_IMAGE".into(),
                "fixture.json".into(),
            )]),
            inputs: json!({}),
            outputs: json!({}),
            components: ModelPackComponents {
                checkpoint: Some("checkpoint.safetensors".into()),
                vae: Some("vae.safetensors".into()),
                text_encoders: vec![],
                loras: vec![],
            },
            defaults: ModelPackDefaults {
                sampler: Some("euler".into()),
                scheduler: Some("normal".into()),
                steps: Some(10),
                cfg: Some(4.0),
                resolution: json!({"width": 512, "height": 512}),
                fps: None,
                frames: Some(1),
                dtype: "BF16".into(),
                quantization: None,
            },
            memory_policy: ModelPackMemoryPolicy {
                min_vram_bytes: 1,
                safety_reserve_bytes: 1,
                supports_cpu_offload: true,
                supports_sequential_offload: true,
                supports_quantization: false,
                component_placement: BTreeMap::new(),
            },
            presets: BTreeMap::from([
                ("FAST".into(), json!({})),
                ("BALANCED".into(), json!({})),
                ("QUALITY".into(), json!({})),
            ]),
        }
    }

    async fn workflow_source() -> PathBuf {
        let source = std::env::temp_dir().join(format!("vidioai-workflows-{}", Uuid::new_v4()));
        fs::create_dir_all(&source).await.unwrap();
        fs::write(
            source.join("fixture.json"),
            serde_json::to_vec(&json!({
                "schema_version": 1,
                "workflow": {"1": {"class_type": "Fixture", "inputs": {}}},
                "bindings": {}
            }))
            .unwrap(),
        )
        .await
        .unwrap();
        source
    }

    #[tokio::test]
    async fn bundled_pack_is_versioned_checksummed_and_atomically_active() {
        let root = std::env::temp_dir().join(format!("vidioai-pack-registry-{}", Uuid::new_v4()));
        let source = workflow_source().await;
        let registry = VersionedPackRegistry::open(root.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        let response = registry.list().await;
        assert_eq!(response.packs.len(), 1);
        assert_eq!(response.packs[0].sha256.len(), 64);
        assert!(response.packs[0].active);
        assert_eq!(registry.active_packs().await.unwrap(), [pack()]);
        assert!(root.join("active/fixture-pack.json").is_file());
        let _ = fs::remove_dir_all(root).await;
        let _ = fs::remove_dir_all(source).await;
    }

    #[tokio::test]
    async fn checksum_mismatch_is_rejected_before_activation() {
        let root = std::env::temp_dir().join(format!("vidioai-pack-registry-{}", Uuid::new_v4()));
        let source = workflow_source().await;
        let registry = VersionedPackRegistry::open(root.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        let path = artifact_path(&root, "fixture-pack", "1.0.0");
        let mut artifact: Value = serde_json::from_slice(&fs::read(&path).await.unwrap()).unwrap();
        artifact["sha256"] = json!("0".repeat(64));
        fs::write(&path, serde_json::to_vec(&artifact).unwrap())
            .await
            .unwrap();
        let error = registry
            .activate("fixture-pack", "1.0.0", env!("CARGO_PKG_VERSION"))
            .await
            .unwrap_err();
        assert!(error.starts_with("MODEL_PACK_CHECKSUM_MISMATCH"));
        let _ = fs::remove_dir_all(root).await;
        let _ = fs::remove_dir_all(source).await;
    }

    #[tokio::test]
    async fn published_pack_and_workflow_restore_into_an_empty_registry() {
        let source = workflow_source().await;
        let first = std::env::temp_dir().join(format!("vidioai-pack-registry-{}", Uuid::new_v4()));
        let second = std::env::temp_dir().join(format!("vidioai-pack-registry-{}", Uuid::new_v4()));
        let storage = MemoryObjectStorage::default();
        let publisher = VersionedPackRegistry::open(first.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        publisher
            .publish("fixture-pack", None, &storage, 11)
            .await
            .unwrap();
        assert!(
            storage
                .objects
                .read()
                .await
                .contains_key("model-packs/registry.json")
        );
        assert!(
            storage
                .objects
                .read()
                .await
                .contains_key("model-packs/fixture-pack/1.0.0/workflows/fixture.json")
        );

        let consumer = VersionedPackRegistry::open(second.clone(), Vec::new(), None, 20)
            .await
            .unwrap();
        let discovered = consumer.synchronize_from_storage(&storage).await.unwrap();
        assert_eq!(discovered.len(), 1);
        consumer
            .ensure_local_from_storage("fixture-pack", "1.0.0", &storage)
            .await
            .unwrap();
        consumer
            .activate("fixture-pack", "1.0.0", env!("CARGO_PKG_VERSION"))
            .await
            .unwrap();
        assert_eq!(consumer.active_packs().await.unwrap(), [pack()]);

        let _ = fs::remove_dir_all(source).await;
        let _ = fs::remove_dir_all(first).await;
        let _ = fs::remove_dir_all(second).await;
    }
}

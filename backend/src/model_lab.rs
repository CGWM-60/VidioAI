//! Registre persistant du Model Lab.
//!
//! Le Lab n'exécute aucun code provenant de Hugging Face. Il compare seulement
//! les métadonnées et fichiers obtenus par l'API HTTP existante, puis conserve
//! chaque couple repository/révision comme une entrée indépendante.

use crate::{
    huggingface_catalog::CatalogModel,
    model_pack::{CatalogModelStatus, ModelPack},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{fs, sync::RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LabLifecycle {
    Discovered,
    Analyzed,
    Installed,
    Experimental,
    Validated,
    Ready,
}

impl LabLifecycle {
    fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Discovered, Self::Analyzed)
                | (Self::Analyzed, Self::Installed)
                | (Self::Installed, Self::Experimental)
                | (Self::Experimental, Self::Validated)
                | (Self::Validated, Self::Ready)
        ) || self == next
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DifferenceKind {
    #[serde(rename = "IDENTIQUE")]
    Identical,
    #[serde(rename = "MODIFIÉ")]
    Modified,
    #[serde(rename = "AJOUTÉ")]
    Added,
    #[serde(rename = "SUPPRIMÉ")]
    Removed,
    #[serde(rename = "INCONNU")]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDifference {
    pub field: String,
    pub status: DifferenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FingerprintFile {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelFingerprint {
    pub architecture: Option<String>,
    pub pipeline_class: Option<String>,
    pub capabilities: Vec<String>,
    pub configs: BTreeMap<String, Value>,
    pub vae: Option<String>,
    pub text_encoders: Vec<String>,
    pub scheduler: Option<String>,
    pub files: Vec<FingerprintFile>,
    pub revision: Option<String>,
    pub size_bytes: Option<u64>,
}

impl ModelFingerprint {
    pub fn from_catalog(model: &CatalogModel) -> Self {
        let files = model
            .files
            .iter()
            .map(|file| FingerprintFile {
                path: file.path.clone(),
                size: file.size,
                sha256: file.lfs_sha256.clone(),
            })
            .collect::<Vec<_>>();
        let paths = files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>();
        let vae = paths
            .iter()
            .find(|path| path.to_ascii_lowercase().contains("vae"))
            .map(|path| (*path).to_owned());
        let text_encoders = paths
            .iter()
            .filter(|path| path.to_ascii_lowercase().contains("text_encoder"))
            .map(|path| (*path).to_owned())
            .collect();
        let scheduler = paths
            .iter()
            .find(|path| path.to_ascii_lowercase().contains("scheduler"))
            .map(|path| (*path).to_owned());
        let mut configs = model
            .config
            .as_object()
            .map(|values| {
                values
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        configs.insert("library".into(), json!(model.library));
        configs.insert("pipeline_tag".into(), json!(model.pipeline_tag));
        configs.insert("tags".into(), json!(model.tags));
        configs.insert("kind".into(), json!(model.kind));
        configs.insert("quality_valid".into(), json!(model.quality_valid));
        configs.insert("trust_remote_code".into(), json!(false));
        Self {
            architecture: model.architecture.clone(),
            pipeline_class: model.pipeline_class.clone(),
            capabilities: model
                .capabilities
                .iter()
                .filter_map(|capability| serde_json::to_value(capability).ok())
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect(),
            configs,
            vae,
            text_encoders,
            scheduler,
            files,
            revision: Some(model.revision.clone()),
            size_bytes: model.estimated_size_bytes,
        }
    }

    pub fn from_pack(pack: &ModelPack) -> Self {
        let mut files = BTreeSet::new();
        if let Some(checkpoint) = &pack.components.checkpoint {
            files.insert(checkpoint.clone());
        }
        if let Some(vae) = &pack.components.vae {
            files.insert(vae.clone());
        }
        files.extend(pack.components.text_encoders.iter().cloned());
        files.extend(pack.components.loras.iter().cloned());
        let mut configs = BTreeMap::new();
        configs.insert("defaults".into(), json!(pack.defaults));
        configs.insert("inputs".into(), pack.inputs.clone());
        configs.insert("outputs".into(), pack.outputs.clone());
        Self {
            architecture: pack.architectures.first().cloned(),
            pipeline_class: pack.pipeline_classes.first().cloned(),
            capabilities: pack.capabilities.clone(),
            configs,
            vae: pack.components.vae.clone(),
            text_encoders: pack.components.text_encoders.clone(),
            scheduler: pack.defaults.scheduler.clone(),
            files: files
                .into_iter()
                .map(|path| FingerprintFile {
                    path,
                    size: None,
                    sha256: None,
                })
                .collect(),
            revision: None,
            size_bytes: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelPackCandidate {
    pub id: String,
    pub family: String,
    pub version: String,
    pub status: CatalogModelStatus,
    pub engine: String,
    pub workflow_version: String,
    pub based_on_pack: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LabLifecycleEvent {
    pub status: LabLifecycle,
    pub at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromotionRecord {
    pub repository: String,
    pub revision: String,
    pub family: String,
    pub pack_version: String,
    pub workflow_version: String,
    pub validated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabModel {
    pub id: Uuid,
    pub model_id: String,
    pub repository: String,
    pub revision: String,
    pub lifecycle: LabLifecycle,
    pub history: Vec<LabLifecycleEvent>,
    pub fingerprint: ModelFingerprint,
    pub closest_pack: Option<String>,
    pub closest_model: Option<Uuid>,
    pub differences: Vec<ModelDifference>,
    pub similarity_score: f32,
    pub risk: String,
    pub model_pack_candidate: ModelPackCandidate,
    pub install_job_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed_storage_id: Option<String>,
    pub promotion: Option<PromotionRecord>,
    pub update_available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_revision: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LabAnalysisResponse {
    pub model: LabModel,
    pub closest_pack: Option<String>,
    pub closest_model: Option<Uuid>,
    pub differences: Vec<ModelDifference>,
    pub similarity_score: f32,
    pub risk: String,
    pub model_pack_candidate: ModelPackCandidate,
}

impl From<LabModel> for LabAnalysisResponse {
    fn from(model: LabModel) -> Self {
        Self {
            closest_pack: model.closest_pack.clone(),
            closest_model: model.closest_model,
            differences: model.differences.clone(),
            similarity_score: model.similarity_score,
            risk: model.risk.clone(),
            model_pack_candidate: model.model_pack_candidate.clone(),
            model,
        }
    }
}

fn compare_value(
    field: impl Into<String>,
    expected: Option<Value>,
    actual: Option<Value>,
) -> ModelDifference {
    let status = match (&expected, &actual) {
        (Some(left), Some(right)) if left == right => DifferenceKind::Identical,
        (Some(_), Some(_)) => DifferenceKind::Modified,
        (None, Some(_)) => DifferenceKind::Added,
        (Some(_), None) => DifferenceKind::Removed,
        (None, None) => DifferenceKind::Unknown,
    };
    ModelDifference {
        field: field.into(),
        status,
        expected,
        actual,
    }
}

pub fn compare_fingerprints(
    expected: Option<&ModelFingerprint>,
    actual: &ModelFingerprint,
) -> Vec<ModelDifference> {
    let expected = expected.cloned().unwrap_or(ModelFingerprint {
        architecture: None,
        pipeline_class: None,
        capabilities: Vec::new(),
        configs: BTreeMap::new(),
        vae: None,
        text_encoders: Vec::new(),
        scheduler: None,
        files: Vec::new(),
        revision: None,
        size_bytes: None,
    });
    let mut differences = vec![
        compare_value(
            "architecture",
            expected.architecture.map(Value::String),
            actual.architecture.clone().map(Value::String),
        ),
        compare_value(
            "pipeline_class",
            expected.pipeline_class.map(Value::String),
            actual.pipeline_class.clone().map(Value::String),
        ),
        compare_value(
            "capabilities",
            (!expected.capabilities.is_empty()).then(|| json!(expected.capabilities)),
            (!actual.capabilities.is_empty()).then(|| json!(actual.capabilities)),
        ),
        compare_value(
            "configs",
            (!expected.configs.is_empty()).then(|| json!(expected.configs)),
            (!actual.configs.is_empty()).then(|| json!(actual.configs)),
        ),
        compare_value(
            "vae",
            expected.vae.map(Value::String),
            actual.vae.clone().map(Value::String),
        ),
        compare_value(
            "text_encoders",
            (!expected.text_encoders.is_empty()).then(|| json!(expected.text_encoders)),
            (!actual.text_encoders.is_empty()).then(|| json!(actual.text_encoders)),
        ),
        compare_value(
            "scheduler",
            expected.scheduler.map(Value::String),
            actual.scheduler.clone().map(Value::String),
        ),
        compare_value(
            "revision",
            expected.revision.map(Value::String),
            actual.revision.clone().map(Value::String),
        ),
        compare_value(
            "size_bytes",
            expected.size_bytes.map(Value::from),
            actual.size_bytes.map(Value::from),
        ),
    ];

    let expected_files = expected
        .files
        .into_iter()
        .map(|file| (file.path.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let actual_files = actual
        .files
        .iter()
        .cloned()
        .map(|file| (file.path.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let paths = expected_files
        .keys()
        .chain(actual_files.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    differences.extend(paths.into_iter().map(|path| {
        compare_value(
            format!("files.{path}"),
            expected_files
                .get(&path)
                .and_then(|file| serde_json::to_value(file).ok()),
            actual_files
                .get(&path)
                .and_then(|file| serde_json::to_value(file).ok()),
        )
    }));
    differences
}

pub fn score_and_risk(differences: &[ModelDifference]) -> (f32, String) {
    let known = differences
        .iter()
        .filter(|difference| difference.status != DifferenceKind::Unknown)
        .count();
    let identical = differences
        .iter()
        .filter(|difference| difference.status == DifferenceKind::Identical)
        .count();
    let score = if known == 0 {
        0.0
    } else {
        identical as f32 * 100.0 / known as f32
    };
    let critical_identity_change = differences.iter().any(|difference| {
        matches!(difference.field.as_str(), "architecture" | "pipeline_class")
            && difference.status != DifferenceKind::Identical
    });
    let removed = differences
        .iter()
        .any(|difference| difference.status == DifferenceKind::Removed);
    let modified = differences
        .iter()
        .any(|difference| difference.status == DifferenceKind::Modified);
    let risk = if critical_identity_change {
        "CRITICAL"
    } else if removed {
        "HIGH"
    } else if modified {
        "MEDIUM"
    } else if differences
        .iter()
        .any(|difference| difference.status == DifferenceKind::Unknown)
    {
        "UNKNOWN"
    } else {
        "LOW"
    };
    (score, risk.into())
}

#[derive(Debug, Clone)]
pub struct ModelLabStore {
    path: PathBuf,
    items: Arc<RwLock<Vec<LabModel>>>,
}

impl ModelLabStore {
    pub async fn open(path: PathBuf) -> Result<Self, String> {
        let items = match fs::read(&path).await {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|error| format!("MODEL_LAB_REGISTRY_INVALID: {error}"))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error.to_string()),
        };
        Ok(Self {
            path,
            items: Arc::new(RwLock::new(items)),
        })
    }

    pub async fn list(&self) -> Vec<LabModel> {
        let mut items = self.items.read().await.clone();
        items.sort_by_key(|item| std::cmp::Reverse(item.updated_at));
        items
    }

    pub async fn find_revision(&self, repository: &str, revision: &str) -> Option<LabModel> {
        self.items
            .read()
            .await
            .iter()
            .find(|item| {
                (item.repository == repository || item.model_id == repository)
                    && item.revision == revision
            })
            .cloned()
    }

    pub async fn effective_status(
        &self,
        repository: &str,
        revision: &str,
    ) -> Option<CatalogModelStatus> {
        self.find_revision(repository, revision)
            .await
            .map(|model| match model.lifecycle {
                LabLifecycle::Ready => CatalogModelStatus::Ready,
                LabLifecycle::Discovered | LabLifecycle::Analyzed => {
                    CatalogModelStatus::Downloadable
                }
                _ => CatalogModelStatus::Experimental,
            })
    }

    pub async fn analyzed(
        &self,
        model: &CatalogModel,
        closest_pack: Option<&ModelPack>,
        pack_version: &str,
        workflow_version: &str,
        now: u64,
    ) -> Result<LabModel, String> {
        let fingerprint = ModelFingerprint::from_catalog(model);
        let validated = self
            .items
            .read()
            .await
            .iter()
            .filter(|item| {
                item.revision != model.revision
                    && matches!(
                        item.lifecycle,
                        LabLifecycle::Validated | LabLifecycle::Ready
                    )
            })
            .cloned()
            .collect::<Vec<_>>();
        let previous = validated
            .iter()
            .filter(|item| item.repository == model.repository)
            .max_by_key(|item| item.updated_at)
            .cloned()
            .or_else(|| {
                closest_pack.and_then(|pack| {
                    validated
                        .iter()
                        .filter(|item| item.model_pack_candidate.family == pack.family)
                        .max_by(|left, right| {
                            left.similarity_score
                                .total_cmp(&right.similarity_score)
                                .then_with(|| left.updated_at.cmp(&right.updated_at))
                        })
                        .cloned()
                })
            });
        // `pack_baseline` doit vivre jusqu'après la comparaison.
        let pack_baseline = closest_pack.map(ModelFingerprint::from_pack);
        let baseline = previous
            .as_ref()
            .map(|item| &item.fingerprint)
            .or(pack_baseline.as_ref());
        let differences = compare_fingerprints(baseline, &fingerprint);
        let (similarity_score, risk) = score_and_risk(&differences);
        let family = closest_pack
            .map(|pack| pack.family.clone())
            .or_else(|| model.architecture.clone())
            .unwrap_or_else(|| "unknown".into());
        let slug = model
            .repository
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect::<String>();
        let candidate = ModelPackCandidate {
            id: format!("lab-{slug}"),
            family,
            version: "0.1.0-lab".into(),
            status: CatalogModelStatus::Experimental,
            engine: closest_pack
                .map(|pack| pack.engine_name().to_owned())
                .unwrap_or_else(|| "diffusers".into()),
            workflow_version: workflow_version.into(),
            based_on_pack: closest_pack.map(|pack| pack.id.clone()),
        };
        let mut items = self.items.write().await;
        let existing = items.iter().position(|item| {
            item.repository == model.repository && item.revision == model.revision
        });
        let mut record = existing
            .map(|index| items[index].clone())
            .unwrap_or_else(|| LabModel {
                id: Uuid::new_v4(),
                model_id: model.id.clone(),
                repository: model.repository.clone(),
                revision: model.revision.clone(),
                lifecycle: LabLifecycle::Discovered,
                history: vec![LabLifecycleEvent {
                    status: LabLifecycle::Discovered,
                    at: now,
                }],
                fingerprint: fingerprint.clone(),
                closest_pack: None,
                closest_model: None,
                differences: Vec::new(),
                similarity_score: 0.0,
                risk: "UNKNOWN".into(),
                model_pack_candidate: candidate.clone(),
                install_job_id: None,
                installed_storage_id: None,
                promotion: None,
                update_available: previous.is_some(),
                available_revision: previous.as_ref().map(|_| model.revision.clone()),
                created_at: now,
                updated_at: now,
            });
        if record.lifecycle == LabLifecycle::Discovered {
            record.lifecycle = LabLifecycle::Analyzed;
            record.history.push(LabLifecycleEvent {
                status: LabLifecycle::Analyzed,
                at: now,
            });
        }
        record.fingerprint = fingerprint;
        record.closest_pack = closest_pack.map(|pack| pack.id.clone());
        record.closest_model = previous.as_ref().map(|item| item.id);
        record.differences = differences;
        record.similarity_score = similarity_score;
        record.risk = risk;
        record.model_pack_candidate = ModelPackCandidate {
            version: pack_version.into(),
            ..candidate
        };
        record.update_available = previous.is_some();
        record.available_revision = previous.as_ref().map(|_| model.revision.clone());
        record.updated_at = now;
        if let Some(index) = existing {
            items[index] = record.clone();
        } else {
            for item in items.iter_mut().filter(|item| {
                item.repository == model.repository && item.revision != model.revision
            }) {
                item.update_available = true;
                item.available_revision = Some(model.revision.clone());
            }
            items.push(record.clone());
        }
        self.persist_locked(&items).await?;
        Ok(record)
    }

    pub async fn note_available_revision(
        &self,
        repository: &str,
        revision: &str,
        now: u64,
    ) -> Result<(), String> {
        let mut items = self.items.write().await;
        let mut changed = false;
        for item in items.iter_mut().filter(|item| {
            item.repository == repository
                && item.revision != revision
                && matches!(
                    item.lifecycle,
                    LabLifecycle::Validated | LabLifecycle::Ready
                )
        }) {
            item.update_available = true;
            item.available_revision = Some(revision.into());
            item.updated_at = now;
            changed = true;
        }
        if changed {
            self.persist_locked(&items).await?;
        }
        Ok(())
    }

    pub async fn attach_install_job(
        &self,
        id: Uuid,
        job_id: Uuid,
        installed_storage_id: String,
        now: u64,
    ) -> Result<LabModel, String> {
        let mut items = self.items.write().await;
        let item = items
            .iter_mut()
            .find(|item| item.id == id)
            .ok_or_else(|| "MODEL_LAB_NOT_FOUND: entrée Lab inconnue".to_owned())?;
        if item.lifecycle != LabLifecycle::Analyzed {
            return Err("MODEL_LAB_TRANSITION_INVALID: analyse requise avant installation".into());
        }
        item.install_job_id = Some(job_id);
        item.installed_storage_id = Some(installed_storage_id);
        item.updated_at = now;
        let result = item.clone();
        self.persist_locked(&items).await?;
        Ok(result)
    }

    pub async fn mark_experimental(&self, id: Uuid, now: u64) -> Result<LabModel, String> {
        let mut items = self.items.write().await;
        let item = items
            .iter_mut()
            .find(|item| item.id == id)
            .ok_or_else(|| "MODEL_LAB_NOT_FOUND: entrée Lab inconnue".to_owned())?;
        transition(item, LabLifecycle::Installed, now)?;
        transition(item, LabLifecycle::Experimental, now)?;
        let result = item.clone();
        self.persist_locked(&items).await?;
        Ok(result)
    }

    pub async fn validate_for_promotion(&self, id: Uuid, now: u64) -> Result<LabModel, String> {
        let mut items = self.items.write().await;
        let item = items
            .iter_mut()
            .find(|item| item.id == id)
            .ok_or_else(|| "MODEL_LAB_NOT_FOUND: entrée Lab inconnue".to_owned())?;
        if item.lifecycle != LabLifecycle::Experimental {
            return Err("MODEL_LAB_PROMOTION_INVALID: modèle EXPERIMENTAL requis".into());
        }
        transition(item, LabLifecycle::Validated, now)?;
        item.promotion = Some(PromotionRecord {
            repository: item.repository.clone(),
            revision: item.revision.clone(),
            family: item.model_pack_candidate.family.clone(),
            pack_version: item.model_pack_candidate.version.clone(),
            workflow_version: item.model_pack_candidate.workflow_version.clone(),
            validated_at: now,
        });
        let result = item.clone();
        self.persist_locked(&items).await?;
        Ok(result)
    }

    pub async fn mark_ready(&self, id: Uuid, now: u64) -> Result<LabModel, String> {
        let mut items = self.items.write().await;
        let item = items
            .iter_mut()
            .find(|item| item.id == id)
            .ok_or_else(|| "MODEL_LAB_NOT_FOUND: entrée Lab inconnue".to_owned())?;
        transition(item, LabLifecycle::Ready, now)?;
        let result = item.clone();
        self.persist_locked(&items).await?;
        Ok(result)
    }

    async fn persist_locked(&self, items: &[LabModel]) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|error| error.to_string())?;
        }
        let bytes = serde_json::to_vec_pretty(items).map_err(|error| error.to_string())?;
        let temporary = self.path.with_extension(format!("{}.tmp", Uuid::new_v4()));
        fs::write(&temporary, bytes)
            .await
            .map_err(|error| error.to_string())?;
        fs::rename(&temporary, &self.path)
            .await
            .map_err(|error| error.to_string())
    }
}

fn transition(item: &mut LabModel, next: LabLifecycle, now: u64) -> Result<(), String> {
    if !item.lifecycle.can_transition_to(next) {
        return Err(format!(
            "MODEL_LAB_TRANSITION_INVALID: {:?} -> {:?}",
            item.lifecycle, next
        ));
    }
    item.lifecycle = next;
    item.updated_at = now;
    if item.history.last().is_none_or(|event| event.status != next) {
        item.history.push(LabLifecycleEvent {
            status: next,
            at: now,
        });
    }
    Ok(())
}

pub fn closest_pack<'a>(
    model: &CatalogModel,
    packs: impl Iterator<Item = &'a ModelPack>,
) -> Option<&'a ModelPack> {
    let capabilities = model
        .capabilities
        .iter()
        .filter_map(|capability| serde_json::to_value(capability).ok())
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    packs
        .filter_map(|pack| {
            let score = usize::from(model.architecture.as_ref().is_some_and(|architecture| {
                pack.architectures
                    .iter()
                    .any(|value| value.eq_ignore_ascii_case(architecture))
            })) * 100
                + usize::from(model.pipeline_class.as_ref().is_some_and(|pipeline| {
                    pack.pipeline_classes
                        .iter()
                        .any(|value| value == "*" || value.eq_ignore_ascii_case(pipeline))
                })) * 75
                + pack
                    .capabilities
                    .iter()
                    .filter(|capability| capabilities.contains(*capability))
                    .count();
            (score > 0).then_some((score, pack))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, pack)| pack)
}

pub fn registry_path(state_dir: &Path) -> PathBuf {
    state_dir.join("model-lab.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        huggingface_catalog::{ModelCapability, RepositoryFile, local_runtime_models},
        model_pack::ModelPackRegistry,
    };

    fn fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            architecture: Some("WanTransformer3DModel".into()),
            pipeline_class: Some("WanPipeline".into()),
            capabilities: vec!["TEXT_TO_VIDEO".into()],
            configs: BTreeMap::from([("dtype".into(), json!("bf16"))]),
            vae: Some("vae/model.safetensors".into()),
            text_encoders: vec!["text_encoder/model.safetensors".into()],
            scheduler: Some("flow".into()),
            files: vec![FingerprintFile {
                path: "transformer/model.safetensors".into(),
                size: Some(10),
                sha256: Some("a".repeat(64)),
            }],
            revision: Some("a".repeat(40)),
            size_bytes: Some(10),
        }
    }

    #[test]
    fn comparison_uses_the_five_exact_difference_states() {
        let mut actual = fingerprint();
        let mut expected = fingerprint();
        actual.pipeline_class = Some("WanNewPipeline".into());
        actual.scheduler = None;
        expected.vae = None;
        actual.revision = None;
        expected.revision = None;
        let differences = compare_fingerprints(Some(&expected), &actual);
        assert!(
            differences
                .iter()
                .any(|item| item.status == DifferenceKind::Identical)
        );
        assert!(
            differences
                .iter()
                .any(|item| item.status == DifferenceKind::Modified)
        );
        assert!(
            differences
                .iter()
                .any(|item| item.status == DifferenceKind::Added)
        );
        assert!(
            differences
                .iter()
                .any(|item| item.status == DifferenceKind::Removed)
        );
        assert!(
            differences
                .iter()
                .any(|item| item.status == DifferenceKind::Unknown)
        );
    }

    #[test]
    fn score_is_informational_and_never_changes_lifecycle() {
        let differences = compare_fingerprints(Some(&fingerprint()), &fingerprint());
        let (score, risk) = score_and_risk(&differences);
        assert!(score > 90.0);
        assert_eq!(risk, "LOW");
        assert_eq!(LabLifecycle::Analyzed, LabLifecycle::Analyzed);
    }

    fn catalog_fixture(repository: &str, revision: &str) -> CatalogModel {
        let mut model = local_runtime_models().remove(0);
        model.id = repository.into();
        model.storage_id = repository.replace('/', "-");
        model.repository = repository.into();
        model.revision = revision.into();
        model.local = false;
        model.architecture = Some("FluxTransformer2DModel".into());
        model.pipeline_class = Some("FluxPipeline".into());
        model.capabilities = vec![ModelCapability::TextToImage];
        model.config = json!({"_class_name": "FluxPipeline", "torch_dtype": "bfloat16"});
        model.files = vec![
            RepositoryFile {
                path: "transformer/model.safetensors".into(),
                size: Some(10),
                lfs_sha256: Some("a".repeat(64)),
            },
            RepositoryFile {
                path: "vae/model.safetensors".into(),
                size: Some(2),
                lfs_sha256: Some("b".repeat(64)),
            },
        ];
        model.estimated_size_bytes = Some(12);
        model
    }

    #[tokio::test]
    async fn lifecycle_is_persistent_and_cross_repo_analysis_uses_a_validated_model() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("project root");
        let registry = ModelPackRegistry::load_directory(&root.join("model-packs")).unwrap();
        let pack = registry.get("flux-t2i-v1").unwrap();
        let path = std::env::temp_dir().join(format!("vidioai-model-lab-{}.json", Uuid::new_v4()));
        let store = ModelLabStore::open(path.clone()).await.unwrap();
        let first_model = catalog_fixture("owner/first", &"a".repeat(40));
        let first = store
            .analyzed(&first_model, Some(pack), "1.0.0", "1", 1)
            .await
            .unwrap();
        assert_eq!(first.lifecycle, LabLifecycle::Analyzed);
        store
            .attach_install_job(first.id, Uuid::new_v4(), "lab-owner-first-a".into(), 2)
            .await
            .unwrap();
        store.mark_experimental(first.id, 3).await.unwrap();
        let validated = store.validate_for_promotion(first.id, 4).await.unwrap();
        assert_eq!(validated.lifecycle, LabLifecycle::Validated);
        let ready = store.mark_ready(first.id, 5).await.unwrap();
        assert_eq!(ready.lifecycle, LabLifecycle::Ready);
        assert_eq!(ready.promotion.as_ref().unwrap().revision, "a".repeat(40));

        let second_model = catalog_fixture("other/compatible", &"b".repeat(40));
        let second = store
            .analyzed(&second_model, Some(pack), "1.0.0", "1", 6)
            .await
            .unwrap();
        assert_eq!(second.closest_model, Some(first.id));
        assert_eq!(second.lifecycle, LabLifecycle::Analyzed);
        assert_ne!(second.lifecycle, LabLifecycle::Ready);

        store
            .note_available_revision("owner/first", &"c".repeat(40), 7)
            .await
            .unwrap();
        let reopened = ModelLabStore::open(path.clone()).await.unwrap();
        let persisted = reopened
            .list()
            .await
            .into_iter()
            .find(|item| item.id == first.id)
            .unwrap();
        assert_eq!(persisted.lifecycle, LabLifecycle::Ready);
        assert_eq!(persisted.revision, "a".repeat(40));
        assert!(persisted.update_available);
        assert_eq!(persisted.available_revision, Some("c".repeat(40)));
        let _ = fs::remove_file(path).await;
    }
}

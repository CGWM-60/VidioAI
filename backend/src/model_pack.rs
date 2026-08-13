//! Contrat ModelPack partagé par le backend et le worker d'exécution.
//!
//! Un pack décrit une famille/architecture. Le dépôt Hugging Face n'entre
//! volontairement pas dans la résolution : il ne sert qu'à localiser les poids
//! après qu'une architecture et une classe de pipeline ont été inspectées.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CatalogModelStatus {
    Ready,
    Experimental,
    Downloadable,
    Unsupported,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    Comfyui,
    Diffusers,
    Procedural,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPackComponents {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vae: Option<String>,
    #[serde(default)]
    pub text_encoders: Vec<String>,
    #[serde(default)]
    pub loras: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPackDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f32>,
    #[serde(default)]
    pub resolution: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPackMemoryPolicy {
    #[serde(default)]
    pub min_vram_bytes: u64,
    #[serde(default)]
    pub safety_reserve_bytes: u64,
    #[serde(default)]
    pub supports_cpu_offload: bool,
    #[serde(default)]
    pub supports_sequential_offload: bool,
    #[serde(default)]
    pub supports_quantization: bool,
    #[serde(default)]
    pub component_placement: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPack {
    pub schema_version: u32,
    pub id: String,
    pub family: String,
    pub status: CatalogModelStatus,
    pub engine: EngineKind,
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub pipeline_classes: Vec<String>,
    pub workflow_by_capability: BTreeMap<String, String>,
    #[serde(default)]
    pub inputs: Value,
    #[serde(default)]
    pub outputs: Value,
    pub components: ModelPackComponents,
    pub defaults: ModelPackDefaults,
    pub memory_policy: ModelPackMemoryPolicy,
    pub presets: BTreeMap<String, Value>,
}

impl ModelPack {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version == 0 {
            return Err("MODEL_PACK_INVALID: schema_version doit être positif".into());
        }
        if self.id.trim().is_empty() || self.family.trim().is_empty() {
            return Err("MODEL_PACK_INVALID: id et family sont requis".into());
        }
        if self.capabilities.is_empty()
            || (self.architectures.is_empty() && self.pipeline_classes.is_empty())
        {
            return Err(
                "MODEL_PACK_INVALID: capacités et sélecteur architecture/pipeline requis".into(),
            );
        }
        if self.family != "generic-diffusers"
            && self
                .components
                .checkpoint
                .as_deref()
                .is_none_or(str::is_empty)
        {
            return Err("MODEL_PACK_INVALID: checkpoint requis".into());
        }
        let video = self
            .capabilities
            .iter()
            .any(|capability| capability.contains("VIDEO"));
        if self.family != "generic-diffusers"
            && (self.defaults.steps.is_none_or(|steps| steps == 0)
                || self.defaults.resolution.is_null()
                || (video
                    && (self.defaults.frames.is_none_or(|frames| frames == 0)
                        || self.defaults.fps.is_none_or(|fps| fps == 0))))
        {
            return Err("MODEL_PACK_INVALID: paramètres d'inférence incomplets".into());
        }
        if self.family != "generic-diffusers" && self.memory_policy.safety_reserve_bytes == 0 {
            return Err("MODEL_PACK_INVALID: safety_reserve_bytes requis".into());
        }
        if self.engine == EngineKind::Comfyui && self.workflow_by_capability.is_empty() {
            return Err("MODEL_PACK_INVALID: workflow ComfyUI requis".into());
        }
        for capability in &self.capabilities {
            if !self.workflow_by_capability.contains_key(capability) {
                return Err(format!(
                    "MODEL_PACK_INVALID: workflow absent pour {capability}"
                ));
            }
        }
        if self
            .workflow_by_capability
            .values()
            .any(|path| path.starts_with('/') || path.split('/').any(|part| part == ".."))
        {
            return Err("MODEL_PACK_INVALID: chemin de workflow non relatif".into());
        }
        for preset in ["FAST", "BALANCED", "QUALITY"] {
            if !self.presets.contains_key(preset) {
                return Err(format!("MODEL_PACK_INVALID: preset {preset} absent"));
            }
        }
        Ok(())
    }

    pub fn engine_name(&self) -> &'static str {
        match self.engine {
            EngineKind::Comfyui => "comfyui",
            EngineKind::Diffusers => "diffusers",
            EngineKind::Procedural => "procedural",
        }
    }

    fn match_score(&self, descriptor: &ModelDescriptor<'_>) -> Option<usize> {
        let architecture_match = descriptor.architectures.iter().any(|candidate| {
            self.architectures
                .iter()
                .any(|known| known.eq_ignore_ascii_case(candidate))
        });
        let pipeline_match = descriptor.pipeline_class.is_some_and(|candidate| {
            self.pipeline_classes
                .iter()
                .any(|known| known.eq_ignore_ascii_case(candidate))
        });
        let wildcard_match = descriptor.pipeline_class.is_some()
            && self.pipeline_classes.iter().any(|known| known == "*");
        if !architecture_match && !pipeline_match && !wildcard_match {
            return None;
        }
        let requested: HashSet<&str> = descriptor.capabilities.iter().copied().collect();
        let capability_matches = self
            .capabilities
            .iter()
            .filter(|capability| requested.contains(capability.as_str()))
            .count();
        if !requested.is_empty() && capability_matches == 0 {
            return None;
        }
        Some(
            usize::from(pipeline_match) * 100
                + usize::from(architecture_match) * 50
                + capability_matches
                + usize::from(wildcard_match),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModelDescriptor<'a> {
    pub architectures: &'a [&'a str],
    pub pipeline_class: Option<&'a str>,
    pub capabilities: &'a [&'a str],
}

#[derive(Debug, Clone, Default)]
pub struct ModelPackRegistry {
    packs: Vec<ModelPack>,
}

impl ModelPackRegistry {
    pub fn new(packs: Vec<ModelPack>) -> Result<Self, String> {
        let mut ids = HashSet::new();
        for pack in &packs {
            pack.validate()?;
            if !ids.insert(pack.id.as_str()) {
                return Err(format!("MODEL_PACK_INVALID: id dupliqué {}", pack.id));
            }
        }
        Ok(Self { packs })
    }

    /// Charge l'autorité JSON versionnée. Une erreur de fichier invalide est
    /// fatale : ignorer silencieusement un pack ferait apparaître un modèle
    /// supporté comme exécutable avec un autre moteur.
    pub fn load_directory(directory: &Path) -> Result<Self, String> {
        let entries = fs::read_dir(directory).map_err(|error| {
            format!(
                "MODEL_PACK_REGISTRY_MISSING: {}: {error}",
                directory.display()
            )
        })?;
        let mut paths = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().and_then(|extension| extension.to_str()) == Some("json")
            })
            .collect::<Vec<_>>();
        paths.sort();
        let mut packs = Vec::with_capacity(paths.len());
        for path in paths {
            let contents = fs::read_to_string(&path)
                .map_err(|error| format!("MODEL_PACK_INVALID: {}: {error}", path.display()))?;
            let pack = serde_json::from_str::<ModelPack>(&contents)
                .map_err(|error| format!("MODEL_PACK_INVALID: {}: {error}", path.display()))?;
            packs.push(pack);
        }
        if packs.is_empty() {
            return Err(format!(
                "MODEL_PACK_REGISTRY_MISSING: aucun manifest dans {}",
                directory.display()
            ));
        }
        Self::new(packs)
    }

    pub fn resolve(&self, descriptor: &ModelDescriptor<'_>) -> Option<&ModelPack> {
        self.packs
            .iter()
            .filter_map(|pack| pack.match_score(descriptor).map(|score| (score, pack)))
            .max_by(|(left_score, left_pack), (right_score, right_pack)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| right_pack.id.cmp(&left_pack.id))
            })
            .map(|(_, pack)| pack)
    }

    pub fn get(&self, pack_id: &str) -> Option<&ModelPack> {
        self.packs.iter().find(|pack| pack.id == pack_id)
    }

    pub fn packs(&self) -> impl Iterator<Item = &ModelPack> {
        self.packs.iter()
    }

    /// Retourne un pack annoncé par un worker uniquement lorsqu'il appartient
    /// bien au registre local et correspond au descripteur inspecté. Le worker
    /// peut attester son état runtime, mais il ne peut pas étendre à lui seul la
    /// matrice de support livrée par le backend.
    pub fn get_matching(
        &self,
        pack_id: &str,
        descriptor: &ModelDescriptor<'_>,
    ) -> Option<&ModelPack> {
        self.get(pack_id)
            .filter(|pack| pack.match_score(descriptor).is_some())
    }

    pub fn validate_workflows(&self, directory: &Path) -> Result<(), String> {
        let mut validated = HashSet::new();
        for template in self
            .packs
            .iter()
            .flat_map(|pack| pack.workflow_by_capability.values())
        {
            if !validated.insert(template.as_str()) {
                continue;
            }
            let path = directory.join(template);
            let contents = fs::read_to_string(&path)
                .map_err(|error| format!("WORKFLOW_INVALID: {}: {error}", path.display()))?;
            let value = serde_json::from_str::<Value>(&contents)
                .map_err(|error| format!("WORKFLOW_INVALID: {}: {error}", path.display()))?;
            if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
                return Err(format!(
                    "WORKFLOW_INVALID: {}: schema_version non supportée",
                    path.display()
                ));
            }
            let nodes = value
                .get("workflow")
                .and_then(Value::as_object)
                .filter(|nodes| !nodes.is_empty())
                .ok_or_else(|| format!("WORKFLOW_INVALID: {}: nodes absents", path.display()))?;
            for (node_id, node) in nodes {
                if node.get("class_type").and_then(Value::as_str).is_none()
                    || node.get("inputs").and_then(Value::as_object).is_none()
                {
                    return Err(format!(
                        "NODE_MISSING: {}: node {node_id} incomplet",
                        path.display()
                    ));
                }
            }
            let bindings = value
                .get("bindings")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("WORKFLOW_INVALID: {}: bindings absents", path.display()))?;
            for binding in bindings.values() {
                let node_id = binding
                    .get("node")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let field = binding
                    .get("field")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !nodes.contains_key(node_id) || field.is_empty() {
                    return Err(format!(
                        "NODE_MISSING: {}: binding vers {node_id:?} invalide",
                        path.display()
                    ));
                }
            }
        }
        Ok(())
    }
}

pub fn public_model_status(
    pack_status: Option<CatalogModelStatus>,
    installed: bool,
    runtime_ready: bool,
) -> CatalogModelStatus {
    match pack_status {
        Some(CatalogModelStatus::Ready) if installed && runtime_ready => CatalogModelStatus::Ready,
        Some(CatalogModelStatus::Experimental) => CatalogModelStatus::Experimental,
        Some(CatalogModelStatus::Ready | CatalogModelStatus::Downloadable) => {
            CatalogModelStatus::Downloadable
        }
        _ => CatalogModelStatus::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pack(id: &str, architecture: &str, pipeline: &str, capability: &str) -> ModelPack {
        ModelPack {
            schema_version: 1,
            id: id.into(),
            family: "fixture".into(),
            status: CatalogModelStatus::Ready,
            engine: EngineKind::Comfyui,
            capabilities: vec![capability.into()],
            architectures: vec![architecture.into()],
            pipeline_classes: vec![pipeline.into()],
            workflow_by_capability: BTreeMap::from([(
                capability.into(),
                "workflows/v1/fixture.json".into(),
            )]),
            inputs: json!({"prompt": {"required": true}}),
            outputs: json!({"kind": "image"}),
            components: ModelPackComponents {
                checkpoint: Some("checkpoint".into()),
                vae: Some("vae".into()),
                text_encoders: vec!["text_encoder".into()],
                loras: vec![],
            },
            defaults: ModelPackDefaults {
                sampler: Some("euler".into()),
                scheduler: Some("simple".into()),
                steps: Some(20),
                cfg: Some(4.0),
                resolution: json!({"width": 1024, "height": 1024}),
                fps: Some(24),
                frames: Some(1),
                dtype: "BF16".into(),
                quantization: None,
            },
            memory_policy: ModelPackMemoryPolicy {
                min_vram_bytes: 8 << 30,
                safety_reserve_bytes: 2 << 30,
                supports_cpu_offload: true,
                supports_sequential_offload: true,
                supports_quantization: true,
                component_placement: BTreeMap::from([("transformer".into(), "cuda".into())]),
            },
            presets: BTreeMap::from([
                ("FAST".into(), json!({"steps": 12})),
                ("BALANCED".into(), json!({"steps": 20})),
                ("QUALITY".into(), json!({"steps": 30})),
            ]),
        }
    }

    #[test]
    fn resolves_by_architecture_and_capability_without_repository_id() {
        let registry = ModelPackRegistry::new(vec![
            pack(
                "flux",
                "FluxTransformer2DModel",
                "FluxPipeline",
                "TEXT_TO_IMAGE",
            ),
            pack(
                "wan",
                "WanTransformer3DModel",
                "WanPipeline",
                "TEXT_TO_VIDEO",
            ),
        ])
        .expect("valid registry");
        let descriptor = ModelDescriptor {
            architectures: &["WanTransformer3DModel"],
            pipeline_class: Some("WanPipeline"),
            capabilities: &["TEXT_TO_VIDEO"],
        };

        assert_eq!(
            registry.resolve(&descriptor).map(|pack| pack.id.as_str()),
            Some("wan")
        );
        assert_eq!(
            registry
                .get_matching("wan", &descriptor)
                .map(|pack| pack.id.as_str()),
            Some("wan")
        );
        assert!(
            registry
                .get_matching("worker-invented-pack", &descriptor)
                .is_none()
        );
    }

    #[test]
    fn strict_public_status_never_marks_a_download_as_ready() {
        assert_eq!(
            public_model_status(Some(CatalogModelStatus::Ready), false, false),
            CatalogModelStatus::Downloadable
        );
        assert_eq!(
            public_model_status(None, true, true),
            CatalogModelStatus::Unsupported
        );
        assert_eq!(
            public_model_status(Some(CatalogModelStatus::Ready), true, true),
            CatalogModelStatus::Ready
        );
    }

    #[test]
    fn invalid_workflow_traversal_is_rejected() {
        let mut invalid = pack("unsafe", "Fixture", "FixturePipeline", "TEXT_TO_IMAGE");
        invalid
            .workflow_by_capability
            .insert("TEXT_TO_IMAGE".into(), "../outside-workflow.json".into());
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn project_registry_and_versioned_workflows_are_valid() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("project root");
        let registry = ModelPackRegistry::load_directory(&root.join("model-packs"))
            .expect("project ModelPacks");
        registry
            .validate_workflows(&root.join("workflows"))
            .expect("versioned workflows");
        assert!(registry.get("flux-t2i-v1").is_some());
        assert!(registry.get("wan22-t2v-v1").is_some());
        assert!(registry.get("ltx-i2v-v1").is_some());
    }
}

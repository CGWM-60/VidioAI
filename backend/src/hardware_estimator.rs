//! Estimation matérielle prudente des modèles Hugging Face.
//!
//! Hugging Face ne publie pas un contrat matériel uniforme. Ce module agrège
//! donc les faits disponibles (Safetensors, configuration, noms et tailles de
//! fichiers), puis sépare strictement la mémoire des poids de la fourchette de
//! VRAM d'inférence. Une valeur calculée reste étiquetée `estimated` ou
//! `partial`; elle n'est jamais présentée comme une mesure réelle.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

const GIB: u64 = 1024 * 1024 * 1024;

/// Origine de l'information, classée dans l'ordre de priorité métier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HardwareSource {
    Official,
    Measured,
    Estimated,
    Partial,
    Unknown,
}

/// La confiance ne transforme jamais une estimation en mesure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EstimateConfidence {
    High,
    Medium,
    Low,
}

/// Une fourchette explicite évite la fausse précision d'un nombre unique.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryRange {
    pub min_bytes: u64,
    pub max_bytes: u64,
}

/// Répartition utile pour les pipelines Diffusers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComponentMemory {
    pub name: String,
    pub bytes: u64,
}

/// Mesure persistée après un vrai chargement ou une vraie inférence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareBenchmark {
    pub model_id: String,
    pub revision: String,
    pub gpu: String,
    pub vram_idle_bytes: u64,
    pub vram_after_load_bytes: u64,
    pub vram_peak_bytes: u64,
    pub ram_peak_bytes: Option<u64>,
    pub runtime: String,
    pub precision: String,
    pub resolution_width: Option<u32>,
    pub resolution_height: Option<u32>,
    pub frames: Option<u32>,
    pub duration_seconds: Option<f64>,
    pub fps: Option<f64>,
    pub batch: u32,
    pub attention_implementation: Option<String>,
    pub vae_tiling: bool,
    pub cpu_offload: bool,
    pub model_offload: bool,
    pub inference_seconds: Option<f64>,
    pub measured_at: u64,
}

/// Contrat JSON normalisé consommé par les cartes et la fiche modèle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareEstimate {
    pub source: HardwareSource,
    pub confidence: EstimateConfidence,
    /// Mémoire nécessaire aux poids seuls. Ce champ n'est pas une estimation
    /// de la consommation complète pendant l'inférence.
    pub weights_memory: Option<MemoryRange>,
    pub estimated_vram_min: Option<u64>,
    pub estimated_vram_recommended: Option<u64>,
    pub estimated_ram: Option<MemoryRange>,
    pub recommended_backend: Option<String>,
    pub recommended_precision: Option<String>,
    pub compatible_with_current_machine: Option<bool>,
    pub optimization_required: bool,
    pub compatibility_level: String,
    pub parameter_count: Option<u64>,
    pub tensor_dtypes: Vec<String>,
    pub components: Vec<ComponentMemory>,
    pub supports_cpu_offload: bool,
    pub notes: Vec<String>,
    pub benchmark: Option<HardwareBenchmark>,
}

impl Default for HardwareEstimate {
    fn default() -> Self {
        Self {
            source: HardwareSource::Unknown,
            confidence: EstimateConfidence::Low,
            weights_memory: None,
            estimated_vram_min: None,
            estimated_vram_recommended: None,
            estimated_ram: None,
            recommended_backend: None,
            recommended_precision: None,
            compatible_with_current_machine: None,
            optimization_required: false,
            compatibility_level: "UNKNOWN".into(),
            parameter_count: None,
            tensor_dtypes: Vec::new(),
            components: Vec::new(),
            supports_cpu_offload: false,
            notes: vec![
                "Les métadonnées disponibles ne permettent pas une estimation fiable.".into(),
            ],
            benchmark: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SafetensorsMetadata {
    pub parameters: BTreeMap<String, u64>,
    pub total: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct HardwareFile {
    pub path: String,
    pub size: Option<u64>,
}

/// Entrée indépendante de la couche HTTP et facile à construire dans les tests.
#[derive(Debug, Clone)]
pub struct HardwareMetadata {
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub architecture: Option<String>,
    pub config: Value,
    pub card_data: Value,
    pub safetensors: Option<SafetensorsMetadata>,
    pub files: Vec<HardwareFile>,
    pub repository_size: Option<u64>,
}

/// Vue minimale de `/api/system` requise par la règle de compatibilité.
#[derive(Debug, Clone, Copy)]
pub struct CurrentMachine {
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub cuda_available: bool,
}

pub struct HardwareEstimator;

impl HardwareEstimator {
    /// Produit d'abord une estimation indépendante de la machine courante.
    pub fn estimate(metadata: &HardwareMetadata) -> HardwareEstimate {
        if let Some(official) = official_requirements(metadata) {
            return official;
        }

        let precision = detect_precision(metadata);
        let parameter_count = metadata
            .safetensors
            .as_ref()
            .and_then(|value| value.total)
            .or_else(|| parameter_count_from_config(&metadata.config));
        let tensor_dtypes = metadata
            .safetensors
            .as_ref()
            .map(|value| value.parameters.keys().cloned().collect())
            .unwrap_or_default();
        let (selected_file_bytes, components, _file_quality) =
            select_weight_files(metadata, &precision);

        // Un repository ModularPipeline peut ne contenir que le manifeste et
        // déléguer ses poids à d'autres repos. Avant matérialisation, ne jamais
        // présenter la petite taille du repo racine comme une estimation VRAM.
        if is_modular(metadata)
            && !metadata.files.iter().any(|file| is_weight(&file.path))
            && has_modular_external_components(&metadata.config)
        {
            return HardwareEstimate {
                source: HardwareSource::Partial,
                confidence: EstimateConfidence::Low,
                recommended_backend: Some("CUDA".into()),
                supports_cpu_offload: true,
                notes: vec![
                    "Repository ModularPipeline détecté.".into(),
                    "Les composants externes doivent être matérialisés avant de pouvoir estimer précisément VRAM et RAM.".into(),
                ],
                ..Default::default()
            };
        }
        let parameter_bytes = parameter_count.map(|count| {
            // Comme Accelerate, on estime ici le stockage des tenseurs selon le
            // dtype. Les surcoûts runtime sont ajoutés séparément ci-dessous.
            bytes_for_parameters(count, &precision)
        });
        let weights = match (parameter_bytes, selected_file_bytes) {
            // Pour un Transformers Safetensors, le compte de paramètres décrit
            // mieux les poids chargés que la taille totale du repository.
            (Some(bytes), _) if metadata.safetensors.is_some() && !is_diffusers(metadata) => {
                Some(bytes)
            }
            // Les dépôts Diffusers contiennent souvent plusieurs variantes. La
            // sélection par composant empêche leur double comptage.
            (_, Some(bytes)) => Some(bytes),
            (Some(bytes), None) => Some(bytes),
            // Dernier repli : la taille du dépôt donne seulement une fourchette
            // basse confiance, car elle peut inclure plusieurs checkpoints.
            (None, None) => metadata.repository_size,
        };

        let Some(weights_bytes) = weights.filter(|value| *value > 0) else {
            return HardwareEstimate::default();
        };

        let has_sized_safetensors = metadata.files.iter().any(|file| {
            file.size.is_some() && file.path.to_ascii_lowercase().ends_with(".safetensors")
        });
        let strong_metadata = metadata.safetensors.is_some() || has_sized_safetensors;
        let source = if strong_metadata {
            HardwareSource::Estimated
        } else {
            HardwareSource::Partial
        };
        let confidence = if metadata.safetensors.is_some() && has_sized_safetensors {
            EstimateConfidence::High
        } else if strong_metadata {
            EstimateConfidence::Medium
        } else {
            EstimateConfidence::Low
        };
        let kind = pipeline_kind(metadata);
        let quantization_overhead = match precision.as_str() {
            "INT4" | "FP4" => 1.18,
            "INT8" => 1.10,
            _ => 1.03,
        };
        let load_min = (weights_bytes as f64 * quantization_overhead) as u64;
        let load_max = (load_min as f64 * 1.08) as u64;

        // Ces profils sont des fourchettes d'inférence prudentes. Ils intègrent
        // activations, workspace et caches, mais restent explicitement estimés.
        let (vram_min, vram_recommended, ram_min, ram_max, profile_note) = match kind {
            PipelineKind::Video => (
                load_min.saturating_mul(13) / 10 + 6 * GIB,
                load_max.saturating_mul(17) / 10 + 12 * GIB,
                load_min.saturating_mul(3) / 2 + 8 * GIB,
                load_max.saturating_mul(2) + 16 * GIB,
                "Profil vidéo de référence : batch 1, 49 images, résolution modérée. La durée, le nombre d'images et la résolution peuvent fortement augmenter la VRAM.",
            ),
            PipelineKind::Image => (
                load_min.saturating_mul(115) / 100 + 1536 * 1024 * 1024,
                load_max.saturating_mul(135) / 100 + 3 * GIB,
                load_min.saturating_mul(3) / 2 + 4 * GIB,
                load_max.saturating_mul(2) + 8 * GIB,
                "Profil image de référence : batch 1, 1024×1024. La résolution, l'attention et le VAE modifient la consommation réelle.",
            ),
            PipelineKind::Transformer => (
                load_min.saturating_mul(6) / 5 + GIB,
                load_max.saturating_mul(7) / 5 + 2 * GIB,
                load_min.saturating_mul(3) / 2 + 2 * GIB,
                load_max.saturating_mul(2) + 4 * GIB,
                "La mémoire des poids est distincte des caches KV et des activations, qui dépendent notamment de la longueur de contexte et du batch.",
            ),
            PipelineKind::Other => (
                load_min.saturating_mul(6) / 5 + GIB,
                load_max.saturating_mul(3) / 2 + 2 * GIB,
                load_min.saturating_mul(3) / 2 + 2 * GIB,
                load_max.saturating_mul(2) + 4 * GIB,
                "Profil générique : la consommation réelle dépend du runtime et des entrées.",
            ),
        };

        let mut notes = vec![
            "Estimation calculée à partir des poids et de la configuration Hugging Face ; ce n'est pas une mesure d'inférence.".into(),
            profile_note.into(),
        ];
        if matches!(source, HardwareSource::Partial) {
            notes.push("Le repository ne fournit pas assez de métadonnées Safetensors ; la fourchette est volontairement large.".into());
        }
        let supports_cpu_offload = is_diffusers(metadata)
            || metadata
                .library_name
                .as_deref()
                .is_some_and(|name| name.eq_ignore_ascii_case("transformers"));

        HardwareEstimate {
            source,
            confidence,
            weights_memory: Some(MemoryRange {
                min_bytes: load_min,
                max_bytes: load_max,
            }),
            estimated_vram_min: Some(vram_min),
            estimated_vram_recommended: Some(vram_recommended.max(vram_min)),
            estimated_ram: Some(MemoryRange {
                min_bytes: ram_min,
                max_bytes: ram_max.max(ram_min),
            }),
            recommended_backend: Some(
                if requires_cuda(metadata) {
                    "CUDA"
                } else {
                    "CPU/CUDA"
                }
                .into(),
            ),
            recommended_precision: Some(precision),
            compatible_with_current_machine: None,
            optimization_required: false,
            compatibility_level: "UNKNOWN".into(),
            parameter_count,
            tensor_dtypes,
            components,
            supports_cpu_offload,
            notes,
            benchmark: None,
        }
    }

    /// Un benchmark réel remplace l'estimation, conformément à l'ordre de
    /// priorité. Les notes gardent le contexte exact de la mesure.
    pub fn with_benchmark(
        mut estimate: HardwareEstimate,
        benchmark: HardwareBenchmark,
    ) -> HardwareEstimate {
        estimate.source = HardwareSource::Measured;
        estimate.confidence = EstimateConfidence::High;
        estimate.estimated_vram_min = Some(benchmark.vram_peak_bytes);
        estimate.estimated_vram_recommended = Some(benchmark.vram_peak_bytes);
        if let Some(ram) = benchmark.ram_peak_bytes {
            estimate.estimated_ram = Some(MemoryRange {
                min_bytes: ram,
                max_bytes: ram,
            });
        }
        estimate.recommended_backend = Some("CUDA".into());
        estimate.recommended_precision = Some(benchmark.precision.clone());
        estimate.notes.insert(
            0,
            format!(
                "Mesuré par VidioAI sur {} avec {} ({}, batch {}).",
                benchmark.gpu, benchmark.runtime, benchmark.precision, benchmark.batch
            ),
        );
        estimate.benchmark = Some(benchmark);
        estimate
    }

    /// Applique les seuils demandés à la machine réellement exposée par
    /// `/api/system`. L'offload n'est jamais supposé disponible sans runtime.
    pub fn with_machine(
        mut estimate: HardwareEstimate,
        machine: Option<CurrentMachine>,
    ) -> HardwareEstimate {
        let Some(machine) = machine else {
            estimate.compatibility_level = "UNKNOWN".into();
            return estimate;
        };
        let (Some(minimum), Some(recommended)) = (
            estimate.estimated_vram_min,
            estimate.estimated_vram_recommended,
        ) else {
            estimate.compatible_with_current_machine = None;
            estimate.compatibility_level = "UNKNOWN".into();
            return estimate;
        };
        let ram_ok = estimate
            .estimated_ram
            .as_ref()
            .is_none_or(|range| machine.ram_bytes >= range.min_bytes);
        let backend_ok =
            estimate.recommended_backend.as_deref() != Some("CUDA") || machine.cuda_available;
        let (level, compatible, optimization) = if !backend_ok {
            ("UNSUPPORTED", false, false)
        } else if machine.vram_bytes >= recommended.saturating_mul(3) / 2 && ram_ok {
            ("EXCELLENT", true, false)
        } else if machine.vram_bytes >= recommended && ram_ok {
            ("COMPATIBLE", true, false)
        } else if machine.vram_bytes >= minimum || (estimate.supports_cpu_offload && ram_ok) {
            ("OPTIMIZATION_REQUIRED", true, true)
        } else {
            // Une fourchette issue des métadonnées du repository n'est pas un
            // plan d'allocation. Elle peut ignorer quantification, tiling et
            // placement CPU. L'installation reste donc autorisée afin que le
            // MemoryPlanner construise ensuite l'ExecutionPlan réel au
            // préflight; seul ce plan peut refuser l'inférence.
            ("PREFLIGHT_REQUIRED", true, true)
        };
        estimate.compatible_with_current_machine = Some(compatible);
        estimate.optimization_required = optimization;
        estimate.compatibility_level = level.into();
        if level == "PREFLIGHT_REQUIRED" {
            estimate.notes.push(
                "Estimation pré-installation non bloquante; la décision runtime appartient au préflight ExecutionPlan."
                    .into(),
            );
        }
        estimate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineKind {
    Transformer,
    Image,
    Video,
    Other,
}

fn pipeline_kind(metadata: &HardwareMetadata) -> PipelineKind {
    let pipeline = metadata.pipeline_tag.as_deref().unwrap_or_default();
    let architecture = metadata
        .architecture
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let modular = metadata
        .config
        .get("_modular_model_index")
        .map(|value| value.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    if pipeline.contains("video") || architecture.contains("video") || modular.contains("video") {
        PipelineKind::Video
    } else if pipeline.contains("image") || is_diffusers(metadata) {
        PipelineKind::Image
    } else if metadata
        .library_name
        .as_deref()
        .is_some_and(|name| name.eq_ignore_ascii_case("transformers"))
        || metadata
            .architecture
            .as_deref()
            .is_some_and(|name| name.contains("CausalLM") || name.contains("ConditionalGeneration"))
    {
        PipelineKind::Transformer
    } else {
        PipelineKind::Other
    }
}

fn is_modular(metadata: &HardwareMetadata) -> bool {
    metadata.config.get("_modular_model_index").is_some()
        || metadata
            .files
            .iter()
            .any(|file| file.path == "modular_model_index.json")
}

fn has_modular_external_components(config: &Value) -> bool {
    let Some(index) = config
        .get("_modular_model_index")
        .and_then(Value::as_object)
    else {
        return false;
    };
    index.values().any(|component| {
        component
            .as_array()
            .and_then(|items| items.get(2))
            .and_then(Value::as_object)
            .and_then(|loading| loading.get("pretrained_model_name_or_path"))
            .and_then(Value::as_str)
            .is_some_and(|value| value.contains('/'))
    })
}

fn is_diffusers(metadata: &HardwareMetadata) -> bool {
    metadata
        .library_name
        .as_deref()
        .is_some_and(|name| name.eq_ignore_ascii_case("diffusers"))
        || metadata.config.get("diffusers").is_some()
        || is_modular(metadata)
        || metadata
            .files
            .iter()
            .any(|file| file.path == "model_index.json")
}

fn requires_cuda(metadata: &HardwareMetadata) -> bool {
    matches!(
        pipeline_kind(metadata),
        PipelineKind::Image | PipelineKind::Video
    )
}

fn parameter_count_from_config(config: &Value) -> Option<u64> {
    ["num_parameters", "parameter_count", "n_params"]
        .iter()
        .find_map(|key| config.get(*key).and_then(Value::as_u64))
}

fn detect_precision(metadata: &HardwareMetadata) -> String {
    let searchable = format!(
        "{} {} {}",
        metadata.tags.join(" "),
        metadata
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        metadata.config
    )
    .to_ascii_lowercase();
    let quantization = metadata
        .config
        .get("quantization_config")
        .or_else(|| metadata.config.pointer("/_root_config/quantization_config"));
    if quantization.is_some_and(|value| {
        value.get("bits").and_then(Value::as_u64) == Some(4)
            || value.get("load_in_4bit").and_then(Value::as_bool) == Some(true)
    }) || ["int4", "4-bit", "4bit", "gptq", "awq", "q4_"]
        .iter()
        .any(|needle| searchable.contains(needle))
    {
        return "INT4".into();
    }
    if quantization.is_some_and(|value| {
        value.get("bits").and_then(Value::as_u64) == Some(8)
            || value.get("load_in_8bit").and_then(Value::as_bool) == Some(true)
    }) || ["int8", "8-bit", "8bit", "q8_"]
        .iter()
        .any(|needle| searchable.contains(needle))
    {
        return "INT8".into();
    }
    if ["nvfp4", "mxfp4", "fp4", "float4"]
        .iter()
        .any(|needle| searchable.contains(needle))
    {
        return "FP4".into();
    }
    if searchable.contains("fp8") || searchable.contains("float8") {
        return "FP8".into();
    }
    if config_dtype(&metadata.config).is_some_and(|dtype| dtype.contains("bfloat16"))
        || metadata
            .safetensors
            .as_ref()
            .is_some_and(|value| value.parameters.contains_key("BF16"))
    {
        return "BF16".into();
    }
    if config_dtype(&metadata.config).is_some_and(|dtype| dtype.contains("float16"))
        || searchable.contains("fp16")
        || metadata
            .safetensors
            .as_ref()
            .is_some_and(|value| value.parameters.contains_key("F16"))
    {
        return "FP16".into();
    }
    "FP32".into()
}

fn config_dtype(config: &Value) -> Option<String> {
    config
        .get("torch_dtype")
        .or_else(|| config.pointer("/_root_config/torch_dtype"))
        .and_then(Value::as_str)
        .map(|value| value.to_ascii_lowercase())
}

fn bytes_for_parameters(parameters: u64, precision: &str) -> u64 {
    match precision {
        "INT4" | "FP4" => parameters.div_ceil(2),
        "INT8" | "FP8" => parameters,
        "FP16" | "BF16" => parameters.saturating_mul(2),
        _ => parameters.saturating_mul(4),
    }
}

fn is_weight(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    [".safetensors", ".bin", ".gguf", ".pt", ".pth"]
        .iter()
        .any(|extension| path.ends_with(extension))
}

fn variant_score(path: &str, precision: &str) -> u8 {
    let path = path.to_ascii_lowercase();
    let wanted = match precision {
        "FP16" => ["fp16", "float16"].as_slice(),
        "BF16" => ["bf16", "bfloat16"].as_slice(),
        "FP8" => ["fp8", "float8"].as_slice(),
        "INT8" => ["int8", "8bit", "q8_"].as_slice(),
        "INT4" => ["int4", "4bit", "q4_", "gptq", "awq"].as_slice(),
        _ => ["fp32", "float32"].as_slice(),
    };
    if wanted.iter().any(|needle| path.contains(needle)) {
        3
    } else if ![
        "fp16", "bf16", "fp8", "int8", "int4", "4bit", "8bit", "q4_", "q8_",
    ]
    .iter()
    .any(|needle| path.contains(needle))
    {
        2
    } else {
        0
    }
}

/// Sélectionne une seule famille de poids. Pour Diffusers, chaque composant est
/// traité séparément; pour Transformers, tous les shards de la même variante
/// sont conservés.
fn select_weight_files(
    metadata: &HardwareMetadata,
    precision: &str,
) -> (Option<u64>, Vec<ComponentMemory>, bool) {
    let known: Vec<_> = metadata
        .files
        .iter()
        .filter(|file| is_weight(&file.path) && file.size.is_some())
        .collect();
    if known.is_empty() {
        return (None, Vec::new(), false);
    }
    if !is_diffusers(metadata) {
        let best_score = known
            .iter()
            .map(|file| variant_score(&file.path, precision))
            .max()
            .unwrap_or_default();
        let selected: Vec<_> = known
            .into_iter()
            .filter(|file| variant_score(&file.path, precision) == best_score)
            .collect();
        let total = selected.iter().filter_map(|file| file.size).sum();
        return (Some(total), Vec::new(), best_score > 0);
    }

    let component_names = [
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "image_encoder",
        "controlnet",
        "safety_checker",
    ];
    let has_components = known.iter().any(|file| {
        file.path
            .split('/')
            .next()
            .is_some_and(|part| component_names.contains(&part))
    });
    let mut groups: BTreeMap<String, Vec<&HardwareFile>> = BTreeMap::new();
    for file in known {
        let first = file.path.split('/').next().unwrap_or("other");
        if has_components && !component_names.contains(&first) {
            // Un checkpoint monolithique à la racine est souvent un doublon du
            // pipeline découpé par composants : il ne doit pas être additionné.
            continue;
        }
        groups.entry(first.to_owned()).or_default().push(file);
    }
    let mut total = 0_u64;
    let mut components = Vec::new();
    for (name, files) in groups {
        let best_score = files
            .iter()
            .map(|file| variant_score(&file.path, precision))
            .max()
            .unwrap_or_default();
        let bytes = files
            .iter()
            .filter(|file| variant_score(&file.path, precision) == best_score)
            .filter_map(|file| file.size)
            .sum::<u64>();
        if bytes > 0 {
            total = total.saturating_add(bytes);
            components.push(ComponentMemory { name, bytes });
        }
    }
    ((total > 0).then_some(total), components, true)
}

fn parse_bytes(value: &Value) -> Option<u64> {
    if let Some(number) = value.as_u64() {
        // Les champs officiels numériques sont interprétés comme des GiB : les
        // auteurs expriment généralement minimum_vram sous cette forme.
        return Some(number.saturating_mul(GIB));
    }
    let text = value
        .as_str()?
        .trim()
        .to_ascii_lowercase()
        .replace(',', ".");
    let numeric = text
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '.')
        .collect::<String>();
    let number: f64 = numeric.parse().ok()?;
    let multiplier = if text.contains("tb") || text.contains("to") {
        1024_f64 * GIB as f64
    } else if text.contains("mb") || text.contains("mo") {
        1024_f64 * 1024_f64
    } else {
        GIB as f64
    };
    Some((number * multiplier) as u64)
}

fn official_requirements(metadata: &HardwareMetadata) -> Option<HardwareEstimate> {
    let minimum = [
        "/hardware/minimum_vram",
        "/hardware/min_vram",
        "/minimum_vram",
        "/min_vram",
    ]
    .iter()
    .find_map(|pointer| metadata.card_data.pointer(pointer).and_then(parse_bytes));
    let recommended = [
        "/hardware/recommended_vram",
        "/recommended_vram",
        "/hardware/vram",
    ]
    .iter()
    .find_map(|pointer| metadata.card_data.pointer(pointer).and_then(parse_bytes));
    let (Some(minimum), Some(recommended)) = (minimum, recommended.or(minimum)) else {
        return None;
    };
    Some(HardwareEstimate {
        source: HardwareSource::Official,
        confidence: EstimateConfidence::High,
        weights_memory: None,
        estimated_vram_min: Some(minimum),
        estimated_vram_recommended: Some(recommended.max(minimum)),
        estimated_ram: None,
        recommended_backend: Some("CUDA".into()),
        recommended_precision: Some(detect_precision(metadata)),
        compatible_with_current_machine: None,
        optimization_required: false,
        compatibility_level: "UNKNOWN".into(),
        parameter_count: metadata.safetensors.as_ref().and_then(|value| value.total),
        tensor_dtypes: metadata
            .safetensors
            .as_ref()
            .map(|value| value.parameters.keys().cloned().collect())
            .unwrap_or_default(),
        components: Vec::new(),
        supports_cpu_offload: is_diffusers(metadata),
        notes: vec![
            "Exigence matérielle structurée publiée par l'auteur dans la Model Card.".into(),
        ],
        benchmark: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn metadata(pipeline: &str, library: &str, files: &[(&str, Option<u64>)]) -> HardwareMetadata {
        HardwareMetadata {
            pipeline_tag: Some(pipeline.into()),
            library_name: Some(library.into()),
            tags: vec![],
            architecture: None,
            config: json!({}),
            card_data: json!({}),
            safetensors: None,
            files: files
                .iter()
                .map(|(path, size)| HardwareFile {
                    path: (*path).into(),
                    size: *size,
                })
                .collect(),
            repository_size: None,
        }
    }

    #[test]
    fn llm_safetensors_uses_parameter_count_and_dtype() {
        let mut model = metadata(
            "text-generation",
            "transformers",
            &[("model.safetensors", Some(GIB))],
        );
        model.architecture = Some("LlamaForCausalLM".into());
        model.safetensors = Some(SafetensorsMetadata {
            parameters: BTreeMap::from([("BF16".into(), 7_000_000_000)]),
            total: Some(7_000_000_000),
        });
        let result = HardwareEstimator::estimate(&model);
        assert_eq!(result.source, HardwareSource::Estimated);
        assert_eq!(result.recommended_precision.as_deref(), Some("BF16"));
        assert!(result.weights_memory.unwrap().min_bytes >= 14_000_000_000);
    }

    #[test]
    fn quantized_model_accounts_for_runtime_overhead() {
        let mut model = metadata(
            "text-generation",
            "transformers",
            &[("model.safetensors", Some(4 * GIB))],
        );
        model.config = json!({"quantization_config": {"bits": 4}});
        model.safetensors = Some(SafetensorsMetadata {
            parameters: BTreeMap::from([("I4".into(), 8_000_000_000)]),
            total: Some(8_000_000_000),
        });
        let result = HardwareEstimator::estimate(&model);
        assert_eq!(result.recommended_precision.as_deref(), Some("INT4"));
        assert!(result.weights_memory.unwrap().min_bytes > 4_000_000_000);
    }

    #[test]
    fn image_diffusers_selects_one_variant_per_component() {
        let model = metadata(
            "text-to-image",
            "diffusers",
            &[
                ("model_index.json", Some(10)),
                ("unet/model.safetensors", Some(4 * GIB)),
                ("unet/model.fp16.safetensors", Some(2 * GIB)),
                ("vae/model.safetensors", Some(GIB)),
                ("vae/model.fp16.safetensors", Some(GIB / 2)),
            ],
        );
        let result = HardwareEstimator::estimate(&model);
        assert_eq!(result.source, HardwareSource::Estimated);
        assert!(result.components.len() >= 2);
        assert!(result.estimated_vram_min.unwrap() < 10 * GIB);
        assert!(result.estimated_vram_min < result.estimated_vram_recommended);
    }

    #[test]
    fn video_pipeline_has_a_wider_range_and_video_note() {
        let mut model = metadata(
            "image-to-video",
            "diffusers",
            &[("transformer/model.safetensors", Some(10 * GIB))],
        );
        model.tags.push("bf16".into());
        let result = HardwareEstimator::estimate(&model);
        assert!(result.estimated_vram_recommended.unwrap() > result.estimated_vram_min.unwrap());
        assert!(result.notes.iter().any(|note| note.contains("49 images")));
    }

    #[test]
    fn incomplete_repository_remains_unknown() {
        let model = metadata(
            "text-generation",
            "transformers",
            &[("README.md", Some(42))],
        );
        assert_eq!(
            HardwareEstimator::estimate(&model).source,
            HardwareSource::Unknown
        );
    }

    #[test]
    fn repository_without_safetensors_is_partial_not_precise() {
        let model = metadata(
            "text-generation",
            "transformers",
            &[("pytorch_model.bin", Some(2 * GIB))],
        );
        let result = HardwareEstimator::estimate(&model);
        assert_eq!(result.source, HardwareSource::Partial);
        assert!(result.estimated_vram_min < result.estimated_vram_recommended);
        assert_eq!(result.confidence, EstimateConfidence::Low);
    }

    #[test]
    fn structured_model_card_requirements_are_official() {
        let mut model = metadata("text-to-image", "diffusers", &[]);
        model.card_data = json!({
            "hardware": {"minimum_vram": "18GB", "recommended_vram": "24 GB"}
        });
        let result = HardwareEstimator::estimate(&model);
        assert_eq!(result.source, HardwareSource::Official);
        assert_eq!(result.estimated_vram_min, Some(18 * GIB));
        assert_eq!(result.estimated_vram_recommended, Some(24 * GIB));
    }

    #[test]
    fn compatibility_uses_the_normalized_machine_profile() {
        let estimate = HardwareEstimate {
            source: HardwareSource::Estimated,
            estimated_vram_min: Some(18 * GIB),
            estimated_vram_recommended: Some(24 * GIB),
            estimated_ram: Some(MemoryRange {
                min_bytes: 16 * GIB,
                max_bytes: 32 * GIB,
            }),
            recommended_backend: Some("CUDA".into()),
            ..HardwareEstimate::default()
        };
        let excellent = HardwareEstimator::with_machine(
            estimate.clone(),
            Some(CurrentMachine {
                ram_bytes: 128 * GIB,
                vram_bytes: 80 * GIB,
                cuda_available: true,
            }),
        );
        assert_eq!(excellent.compatibility_level, "EXCELLENT");
        let compatible = HardwareEstimator::with_machine(
            estimate,
            Some(CurrentMachine {
                ram_bytes: 64 * GIB,
                vram_bytes: 24 * GIB,
                cuda_available: true,
            }),
        );
        assert_eq!(compatible.compatibility_level, "COMPATIBLE");
    }

    #[test]
    fn repository_weight_estimate_never_blocks_h100_install_before_preflight() {
        let estimate = HardwareEstimate {
            source: HardwareSource::Partial,
            weights_memory: Some(MemoryRange {
                min_bytes: 64 * GIB,
                max_bytes: 64 * GIB,
            }),
            estimated_vram_min: Some(90 * GIB),
            estimated_vram_recommended: Some(120 * GIB),
            estimated_ram: Some(MemoryRange {
                min_bytes: 104 * GIB,
                max_bytes: 144 * GIB,
            }),
            recommended_backend: Some("CUDA".into()),
            supports_cpu_offload: true,
            ..HardwareEstimate::default()
        };
        let result = HardwareEstimator::with_machine(
            estimate,
            Some(CurrentMachine {
                ram_bytes: 64 * GIB,
                vram_bytes: 80 * GIB,
                cuda_available: true,
            }),
        );
        assert_eq!(result.compatible_with_current_machine, Some(true));
        assert_eq!(result.compatibility_level, "PREFLIGHT_REQUIRED");
        assert!(result.optimization_required);
    }
}

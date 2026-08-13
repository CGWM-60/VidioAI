//! Contrats de planification et de préflight validés par le backend avant
//! dispatch. Le worker calcule les mesures dépendantes du runtime, puis le
//! backend refuse tout plan incohérent ou susceptible de saturer volontairement
//! la VRAM.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionPlan {
    pub strategy: String,
    pub feasible: bool,
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<String>,
    pub vae_tiling: bool,
    pub vae_slicing: bool,
    pub model_cpu_offload: bool,
    pub sequential_cpu_offload: bool,
    #[serde(default)]
    pub component_placement: BTreeMap<String, String>,
    pub resolution: Resolution,
    pub frames: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    pub batch: u32,
    pub weights_memory_bytes: u64,
    pub runtime_memory_bytes: u64,
    pub latent_memory_bytes: u64,
    pub reserved_memory_bytes: u64,
    pub safety_reserve_bytes: u64,
    pub estimated_peak_vram_bytes: u64,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub ram_required_bytes: u64,
    pub scratch_required_bytes: u64,
    #[serde(default)]
    pub fallbacks: Vec<String>,
    pub reason: String,
}

impl ExecutionPlan {
    pub fn validate(&self) -> Result<(), String> {
        if !self.feasible {
            return Err(format!("MODEL_INCOMPATIBLE: {}", self.reason));
        }
        if self.resolution.width == 0
            || self.resolution.height == 0
            || self.frames == 0
            || self.fps == Some(0)
            || self.batch == 0
        {
            return Err("EXECUTION_PLAN_INVALID: dimensions d'inférence nulles".into());
        }
        if self.vram_total_bytes > 0 && self.safety_reserve_bytes == 0 {
            return Err("EXECUTION_PLAN_INVALID: réserve de sécurité absente".into());
        }
        let protected_peak = self
            .estimated_peak_vram_bytes
            .checked_add(self.safety_reserve_bytes)
            .ok_or("EXECUTION_PLAN_INVALID: dépassement arithmétique VRAM")?;
        if self.vram_total_bytes > 0 && protected_peak > self.vram_total_bytes {
            return Err("GPU_MEMORY_OCCUPIED: pic estimé et réserve dépassent la VRAM".into());
        }
        if self.vram_total_bytes > 0 && protected_peak > self.vram_free_bytes {
            return Err("GPU_MEMORY_OCCUPIED: VRAM libre insuffisante après réserve".into());
        }
        let strategy = self.strategy.to_ascii_uppercase();
        if strategy.contains("OFFLOAD") {
            if !self.model_cpu_offload && !self.sequential_cpu_offload {
                return Err(
                    "EXECUTION_PLAN_NOT_APPLIED: stratégie offload sans option runtime".into(),
                );
            }
            let has_non_cuda_component = self.component_placement.values().any(|placement| {
                !matches!(placement.to_ascii_lowercase().as_str(), "cuda" | "gpu")
            });
            if !has_non_cuda_component {
                return Err(
                    "EXECUTION_PLAN_NOT_APPLIED: tous les composants resteraient sur CUDA".into(),
                );
            }
        }
        if self.model_cpu_offload && self.sequential_cpu_offload {
            return Err("EXECUTION_PLAN_INVALID: modes d'offload mutuellement exclusifs".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightCheck {
    pub name: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredRuntimeError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightResult {
    pub status: String,
    pub ready: bool,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_pack_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_plan: Option<ExecutionPlan>,
    #[serde(default)]
    pub checks: Vec<PreflightCheck>,
    #[serde(default)]
    pub errors: Vec<StructuredRuntimeError>,
    #[serde(default)]
    pub diagnostics: Value,
}

impl PreflightResult {
    pub fn validate_ready(&self) -> Result<(), StructuredRuntimeError> {
        if self.status != "READY_TO_RUN"
            || !self.ready
            || self.checks.iter().any(|check| !check.ok)
            || !self.errors.is_empty()
        {
            return Err(self
                .errors
                .first()
                .cloned()
                .unwrap_or(StructuredRuntimeError {
                    code: "PREFLIGHT_FAILED".into(),
                    message: "Le préflight n'a pas validé toutes les dépendances.".into(),
                    retryable: false,
                }));
        }
        if self.model_pack_id.as_deref().is_none_or(str::is_empty) {
            return Err(StructuredRuntimeError {
                code: "MODEL_PACK_MISSING".into(),
                message: "Aucun ModelPack n'a été résolu.".into(),
                retryable: false,
            });
        }
        if self.workflow.as_deref().is_none_or(str::is_empty) {
            return Err(StructuredRuntimeError {
                code: "WORKFLOW_INVALID".into(),
                message: "Le workflow résolu est vide.".into(),
                retryable: false,
            });
        }
        self.execution_plan
            .as_ref()
            .ok_or_else(|| StructuredRuntimeError {
                code: "EXECUTION_PLAN_INVALID".into(),
                message: "Le worker n'a pas retourné d'ExecutionPlan.".into(),
                retryable: false,
            })?
            .validate()
            .map_err(|message| StructuredRuntimeError {
                code: message
                    .split_once(':')
                    .map(|(code, _)| code)
                    .unwrap_or("EXECUTION_PLAN_INVALID")
                    .into(),
                message,
                retryable: false,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn safe_h100_plan() -> ExecutionPlan {
        ExecutionPlan {
            strategy: "MODEL_CPU_OFFLOAD".into(),
            feasible: true,
            dtype: "BF16".into(),
            quantization: None,
            attention: Some("SDPA".into()),
            vae_tiling: true,
            vae_slicing: true,
            model_cpu_offload: true,
            sequential_cpu_offload: false,
            component_placement: BTreeMap::from([
                ("transformer".into(), "cuda".into()),
                ("text_encoder".into(), "temporary_cpu_offload".into()),
                ("vae".into(), "temporary_cpu_offload".into()),
            ]),
            resolution: Resolution {
                width: 1280,
                height: 720,
            },
            frames: 121,
            fps: Some(24),
            batch: 1,
            weights_memory_bytes: 64 * GIB,
            runtime_memory_bytes: 3 * GIB,
            latent_memory_bytes: GIB,
            reserved_memory_bytes: GIB,
            safety_reserve_bytes: 8 * GIB,
            estimated_peak_vram_bytes: 69 * GIB,
            vram_total_bytes: 80 * GIB,
            vram_free_bytes: 80 * GIB,
            ram_required_bytes: 80 * GIB,
            scratch_required_bytes: 8 * GIB,
            fallbacks: vec!["vae_tiling".into(), "model_cpu_offload".into()],
            reason: "Le transformeur reste résident; VAE et encodeur sont temporaires.".into(),
        }
    }

    #[test]
    fn simulated_h100_accepts_64_gib_model_with_safe_margin() {
        assert!(safe_h100_plan().validate().is_ok());
    }

    #[test]
    fn rejects_a_plan_that_would_fill_the_h100() {
        let mut plan = safe_h100_plan();
        plan.estimated_peak_vram_bytes = 79 * GIB;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn rejects_cpu_offload_when_every_component_is_on_cuda() {
        let mut plan = safe_h100_plan();
        plan.component_placement = BTreeMap::from([
            ("transformer".into(), "cuda".into()),
            ("text_encoder".into(), "cuda".into()),
            ("vae".into(), "cuda".into()),
        ]);
        assert!(plan.validate().is_err());
    }
}

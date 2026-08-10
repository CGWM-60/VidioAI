//! Client HTTP typé du worker GPU.
//!
//! Le backend ne charge jamais PyTorch : il pilote un processus séparé dont le
//! contrat reste testable sans carte NVIDIA. Une absence de worker est une
//! erreur explicite, jamais un signal permettant de simuler une génération IA.

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};

#[derive(Clone)]
pub struct WorkerClient {
    base_url: String,
    token: Option<String>,
    http: reqwest::Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    pub status: String,
    pub service: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerReady {
    pub ready: bool,
    pub profile: String,
    pub runtime_available: bool,
    pub cuda_available: bool,
    pub gpu_required: bool,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerModelStatus {
    pub model_id: String,
    pub state: String,
    #[serde(default)]
    pub repository: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub weights_valid: bool,
    #[serde(default)]
    pub runtime_available: bool,
    #[serde(default)]
    pub runtime_compatible: bool,
    #[serde(default)]
    pub validation_test: bool,
    pub error: Option<String>,
    #[serde(default)]
    pub benchmark: Option<WorkerBenchmarkObservation>,
}

/// Mesure brute du worker. Le backend ajoute l'identifiant public, la révision
/// exacte et l'horodatage avant de la persister.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerBenchmarkObservation {
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
    #[serde(default = "default_batch")]
    pub batch: u32,
    pub attention_implementation: Option<String>,
    #[serde(default)]
    pub vae_tiling: bool,
    #[serde(default)]
    pub cpu_offload: bool,
    #[serde(default)]
    pub model_offload: bool,
    pub inference_seconds: Option<f64>,
}

fn default_batch() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerGpu {
    pub name: String,
    pub backend: String,
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,
    pub utilization_percent: Option<f64>,
    pub temperature_celsius: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResources {
    pub gpu: Option<WorkerGpu>,
    pub gpu_status: String,
    pub worker_status: String,
    pub loaded_models: Vec<serde_json::Value>,
    pub active_jobs: usize,
}

#[derive(Debug, Serialize)]
struct ModelRequest<'a> {
    model_id: &'a str,
}

#[derive(Debug, Serialize)]
struct InstallModelRequest<'a> {
    model_id: &'a str,
    repository: &'a str,
    revision: &'a str,
    capabilities: [&'a str; 1],
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    job_id: &'a str,
    model_id: &'a str,
    prompt: &'a str,
    negative_prompt: Option<&'a str>,
    width: u32,
    height: u32,
    steps: u32,
    guidance_scale: f32,
    seed: Option<u64>,
    output_relative_path: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub job_id: String,
    pub state: String,
    pub output_relative_path: String,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    #[serde(default)]
    pub benchmark: Option<WorkerBenchmarkObservation>,
}

impl WorkerClient {
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("VIDIOAI_WORKER_URL").ok()?;
        if base_url.trim().is_empty() {
            return None;
        }
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(30))
            .build()
            .ok()?;
        Some(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            token: std::env::var("VIDIOAI_WORKER_TOKEN").ok(),
            http,
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let request = self
            .http
            .request(method, format!("{}{}", self.base_url, path));
        match &self.token {
            Some(token) => request.header("X-VidioAI-Worker-Token", token),
            None => request,
        }
    }

    async fn json<T: for<'de> Deserialize<'de>>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, String> {
        let response = request.send().await.map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("worker HTTP {status}: {body}"));
        }
        response.json().await.map_err(|error| error.to_string())
    }

    pub async fn health(&self) -> Result<WorkerHealth, String> {
        self.json(self.request(reqwest::Method::GET, "/health"))
            .await
    }

    pub async fn ready(&self) -> Result<WorkerReady, String> {
        self.json(self.request(reqwest::Method::GET, "/ready"))
            .await
    }

    pub async fn resources(&self) -> Result<WorkerResources, String> {
        self.json(self.request(reqwest::Method::GET, "/v1/resources"))
            .await
    }

    pub async fn model_status(&self, model_id: &str) -> Result<WorkerModelStatus, String> {
        self.json(self.request(
            reqwest::Method::GET,
            &format!("/v1/models/{model_id}/status"),
        ))
        .await
    }

    pub async fn install(
        &self,
        model_id: &str,
        repository: &str,
        revision: &str,
    ) -> Result<WorkerModelStatus, String> {
        self.json(
            self.request(reqwest::Method::POST, "/v1/models/install")
                .timeout(Duration::from_secs(60 * 60))
                .json(&InstallModelRequest {
                    model_id,
                    repository,
                    revision,
                    capabilities: ["TEXT_TO_IMAGE"],
                }),
        )
        .await
    }

    pub async fn load(
        &self,
        model_id: &str,
        _repository: &str,
        _revision: &str,
    ) -> Result<WorkerModelStatus, String> {
        self.json(
            self.request(reqwest::Method::POST, "/v1/models/load")
                .timeout(Duration::from_secs(60 * 30))
                .json(&ModelRequest { model_id }),
        )
        .await
    }

    pub async fn unload(&self, model_id: &str) -> Result<WorkerModelStatus, String> {
        self.json(
            self.request(reqwest::Method::POST, "/v1/models/unload")
                .json(&ModelRequest { model_id }),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn generate_text_to_image(
        &self,
        job_id: &str,
        model_id: &str,
        prompt: &str,
        negative_prompt: Option<&str>,
        output_path: &Path,
    ) -> Result<GenerateResponse, String> {
        let relative = output_path
            .to_str()
            .ok_or_else(|| "Chemin worker non UTF-8".to_owned())?;
        self.json(
            self.request(reqwest::Method::POST, "/v1/generate/text-to-image")
                .timeout(Duration::from_secs(60 * 30))
                .json(&GenerateRequest {
                    job_id,
                    model_id,
                    prompt,
                    negative_prompt,
                    width: 1024,
                    height: 1024,
                    // SD-Turbo est réellement conçu pour 1 à 4 pas et sans
                    // classifier-free guidance. Ces valeurs réduisent aussi le
                    // coût du smoke test matériel.
                    steps: 4,
                    guidance_scale: 0.0,
                    seed: None,
                    output_relative_path: relative,
                }),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn generate_image_to_image(
        &self,
        job_id: &str,
        model_id: &str,
        prompt: &str,
        negative_prompt: Option<&str>,
        output_path: &Path,
        input_path: &str,
    ) -> Result<GenerateResponse, String> {
        let relative = output_path
            .to_str()
            .ok_or_else(|| "Chemin worker non UTF-8".to_owned())?;
        self.json(
            self.request(reqwest::Method::POST, "/v1/generate/image-to-image")
                .timeout(Duration::from_secs(60 * 30))
                .json(&serde_json::json!({
                    "job_id": job_id,
                    "model_id": model_id,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": 1024,
                    "height": 1024,
                    "steps": 4,
                    "guidance_scale": 0.0,
                    "strength": 0.8,
                    "seed": null,
                    "output_relative_path": relative,
                    "input_path": input_path,
                })),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn generate_video(
        &self,
        endpoint: &str,
        job_id: &str,
        model_id: &str,
        prompt: &str,
        negative_prompt: Option<&str>,
        output_path: &Path,
        input_path: Option<&str>,
    ) -> Result<GenerateResponse, String> {
        let relative = output_path
            .to_str()
            .ok_or_else(|| "Chemin worker non UTF-8".to_owned())?;
        let mut payload = serde_json::json!({
            "job_id": job_id,
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": 720,
            "height": 480,
            "steps": 4,
            "guidance_scale": 0.0,
            "duration_seconds": 4,
            "fps": 8,
            "frames": 8,
            "seed": null,
            "output_relative_path": relative,
        });
        if let Some(path) = input_path {
            payload["input_path"] = serde_json::Value::String(path.into());
        }
        self.json(
            self.request(reqwest::Method::POST, endpoint)
                .timeout(Duration::from_secs(60 * 30))
                .json(&payload),
        )
        .await
    }

    pub async fn cancel(&self, job_id: &str) -> Result<(), String> {
        let response = self
            .request(reqwest::Method::POST, "/v1/jobs/cancel")
            .json(&serde_json::json!({ "job_id": job_id }))
            .send()
            .await
            .map_err(|error| error.to_string())?;
        if response.status() == StatusCode::NOT_FOUND || response.status().is_success() {
            Ok(())
        } else {
            Err(format!("annulation worker refusée: {}", response.status()))
        }
    }
}

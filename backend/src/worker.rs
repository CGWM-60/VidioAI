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
    pub scratch_mount_ok: bool,
    #[serde(default)]
    pub scratch_filesystem: Option<String>,
    #[serde(default)]
    pub scratch_total_bytes: u64,
    #[serde(default)]
    pub scratch_available_bytes: u64,
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
    pub loaded: bool,
    #[serde(default)]
    pub ready: bool,
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
    #[serde(default)]
    pub runtime_dependencies: Vec<serde_json::Value>,
    #[serde(default)]
    pub precision_plan: Option<serde_json::Value>,
    #[serde(default)]
    pub memory_plan: Option<serde_json::Value>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub capability: Option<String>,
    #[serde(default)]
    pub pipeline_class: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCompatibility {
    #[serde(default = "default_compatibility_status")]
    pub compatibility_status: String,
    pub runtime_supported: bool,
    #[serde(default)]
    pub runtime_capabilities: Vec<String>,
    pub pipeline_class: Option<String>,
    pub runtime_reason: String,
    pub error_code: Option<String>,
    #[serde(default)]
    pub dependency: Option<String>,
}

fn default_compatibility_status() -> String {
    "UNKNOWN".into()
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
    #[serde(default)]
    pub memory: Option<serde_json::Value>,
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
    capabilities: Vec<&'a str>,
}

#[derive(Debug, Serialize)]
struct CompatibilityRequest<'a> {
    pipeline_class: Option<&'a str>,
    library_name: Option<&'a str>,
    pipeline_tag: Option<&'a str>,
    tags: &'a [String],
    architectures: Vec<&'a str>,
    base_models: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
struct WorkerErrorPayload {
    error: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    retryable: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub job_id: String,
    pub state: String,
    pub output_relative_path: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub requested_quality: Option<String>,
    #[serde(default)]
    pub requested_aspect_ratio: Option<String>,
    #[serde(default)]
    pub actual_width: Option<u32>,
    #[serde(default)]
    pub actual_height: Option<u32>,
    #[serde(default)]
    pub actual_fps: Option<f64>,
    #[serde(default)]
    pub actual_frames: Option<u32>,
    pub sha256: String,
    #[serde(default)]
    pub benchmark: Option<WorkerBenchmarkObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerJobStatus {
    pub job_id: String,
    pub state: String,
    #[serde(default)]
    pub progress: u8,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
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
            if let Ok(payload) = serde_json::from_str::<WorkerErrorPayload>(&body) {
                let prefix = payload.code.unwrap_or_else(|| "WORKER_ERROR".to_owned());
                let retry = payload
                    .retryable
                    .map(|value| if value { "retryable" } else { "non-retryable" })
                    .unwrap_or("unknown");
                return Err(format!(
                    "{prefix}: {} (worker HTTP {status}, {retry})",
                    payload.error
                ));
            }
            return Err(format!("WORKER_HTTP_ERROR: worker HTTP {status}: {body}"));
        }
        response.json().await.map_err(|error| error.to_string())
    }

    pub async fn health(&self) -> Result<WorkerHealth, String> {
        self.json(self.request(reqwest::Method::GET, "/health"))
            .await
    }

    pub async fn ready(&self) -> Result<WorkerReady, String> {
        let response = self
            .request(reqwest::Method::GET, "/ready")
            .send()
            .await
            .map_err(|error| format!("WORKER_READY_REQUEST_FAILED: {error}"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| format!("WORKER_READY_BODY_FAILED: {error}"))?;
        decode_ready(status, &body)
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
        capabilities: &[String],
    ) -> Result<WorkerModelStatus, String> {
        self.json(
            self.request(reqwest::Method::POST, "/v1/models/install")
                .timeout(Duration::from_secs(60 * 60))
                .json(&InstallModelRequest {
                    model_id,
                    repository,
                    revision,
                    capabilities: capabilities.iter().map(String::as_str).collect(),
                }),
        )
        .await
    }

    pub async fn compatibility(
        &self,
        pipeline_class: Option<&str>,
        library_name: Option<&str>,
        pipeline_tag: Option<&str>,
        tags: &[String],
    ) -> Result<WorkerCompatibility, String> {
        self.json(
            self.request(reqwest::Method::POST, "/v1/models/compatibility")
                .timeout(Duration::from_secs(15))
                .json(&CompatibilityRequest {
                    pipeline_class,
                    library_name,
                    pipeline_tag,
                    tags,
                    architectures: pipeline_class.into_iter().collect(),
                    base_models: Vec::new(),
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
    pub async fn generate_image(
        &self,
        endpoint: &str,
        job_id: &str,
        model_id: &str,
        prompt: &str,
        negative_prompt: Option<&str>,
        output_path: &Path,
        input_path: Option<&str>,
        mask_path: Option<&str>,
        control_path: Option<&str>,
        capability: Option<&str>,
    ) -> Result<GenerateResponse, String> {
        let relative = output_path
            .to_str()
            .ok_or_else(|| "Chemin worker non UTF-8".to_owned())?;
        let mut payload = serde_json::json!({
            "job_id": job_id,
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": null,
            "output_relative_path": relative,
        });
        if let Some(path) = input_path {
            payload["input_path"] = serde_json::Value::String(path.to_owned());
        }
        if let Some(path) = mask_path {
            payload["mask_path"] = serde_json::Value::String(path.to_owned());
        }
        if let Some(path) = control_path {
            payload["control_path"] = serde_json::Value::String(path.to_owned());
        }
        if let Some(value) = capability {
            payload["capability"] = serde_json::Value::String(value.to_owned());
        }
        self.json(
            self.request(reqwest::Method::POST, endpoint)
                .timeout(Duration::from_secs(60 * 30))
                .json(&payload),
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
        input_images: Option<serde_json::Value>,
        mask_path: Option<&str>,
        capability: Option<&str>,
        quality: &str,
        aspect_ratio: &str,
        duration_seconds: u32,
        fps: u32,
    ) -> Result<GenerateResponse, String> {
        let relative = output_path
            .to_str()
            .ok_or_else(|| "Chemin worker non UTF-8".to_owned())?;

        let mut payload = serde_json::json!({
            "job_id": job_id,
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "seed": null,
            "output_relative_path": relative,
        });

        if let Some(path) = input_path {
            payload["input_path"] = serde_json::Value::String(path.into());
        }

        if let Some(images) = input_images {
            payload["input_images"] = images;
        }

        if let Some(path) = mask_path {
            payload["mask_path"] = serde_json::Value::String(path.into());
        }

        if let Some(value) = capability {
            payload["capability"] = serde_json::Value::String(value.into());
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

    pub async fn job_status(&self, job_id: &str) -> Result<WorkerJobStatus, String> {
        self.json(self.request(reqwest::Method::GET, &format!("/v1/jobs/{job_id}")))
            .await
    }
}

fn decode_ready(status: StatusCode, body: &str) -> Result<WorkerReady, String> {
    if status.is_success() || status == StatusCode::SERVICE_UNAVAILABLE {
        return serde_json::from_str(body)
            .map_err(|error| format!("WORKER_READY_INVALID_JSON: {error}"));
    }
    Err(format!("WORKER_READY_HTTP_ERROR: worker HTTP {status}"))
}

#[cfg(test)]
mod tests {
    use super::{WorkerClient, WorkerCompatibility, decode_ready};
    use reqwest::StatusCode;
    use std::time::Duration;
    use tokio::{io::AsyncReadExt, net::TcpListener, time::sleep};

    const READY: &str = r#"{
        "ready":true,
        "profile":"GPU_PRODUCTION",
        "runtime_available":true,
        "cuda_available":true,
        "gpu_required":true,
        "scratch_mount_ok":true,
        "scratch_filesystem":"device:contract",
        "scratch_total_bytes":1500000000000,
        "scratch_available_bytes":1400000000000,
        "errors":[]
    }"#;

    #[test]
    fn readiness_contract_accepts_200() {
        let ready = decode_ready(StatusCode::OK, READY).unwrap();
        assert!(ready.ready);
        assert!(ready.cuda_available);
    }

    #[test]
    fn readiness_contract_preserves_structured_503() {
        let payload = READY.replace("\"ready\":true", "\"ready\":false");
        let ready = decode_ready(StatusCode::SERVICE_UNAVAILABLE, &payload).unwrap();
        assert!(!ready.ready);
        assert!(ready.runtime_available);
    }

    #[test]
    fn readiness_contract_rejects_malformed_json() {
        let error = decode_ready(StatusCode::SERVICE_UNAVAILABLE, "not-json").unwrap_err();
        assert!(error.contains("WORKER_READY_INVALID_JSON"));
    }

    #[test]
    fn compatibility_contract_preserves_supported_unknown_and_unsupported() {
        for expected in ["SUPPORTED", "UNKNOWN", "UNSUPPORTED"] {
            let payload = format!(
                r#"{{
                    "compatibility_status":"{expected}",
                    "runtime_supported":{},
                    "runtime_capabilities":[],
                    "pipeline_class":null,
                    "runtime_reason":"contract",
                    "error_code":null
                }}"#,
                expected == "SUPPORTED"
            );
            let parsed: WorkerCompatibility = serde_json::from_str(&payload).unwrap();
            assert_eq!(parsed.compatibility_status, expected);
        }
    }

    #[test]
    fn compatibility_contract_defaults_legacy_payload_to_unknown() {
        let parsed: WorkerCompatibility = serde_json::from_str(
            r#"{
                "runtime_supported":false,
                "runtime_capabilities":[],
                "pipeline_class":null,
                "runtime_reason":"legacy",
                "error_code":null
            }"#,
        )
        .unwrap();
        assert_eq!(parsed.compatibility_status, "UNKNOWN");
    }

    #[tokio::test]
    async fn readiness_contract_reports_timeout_and_sends_authentication() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 4096];
            let read = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..read]).to_ascii_lowercase();
            assert!(request.contains("x-vidioai-worker-token: contract-token"));
            sleep(Duration::from_millis(100)).await;
        });
        let client = WorkerClient {
            base_url: format!("http://{address}"),
            token: Some("contract-token".into()),
            http: reqwest::Client::builder()
                .timeout(Duration::from_millis(20))
                .build()
                .unwrap(),
        };
        let error = client.ready().await.unwrap_err();
        assert!(error.contains("WORKER_READY_REQUEST_FAILED"));
        server.await.unwrap();
    }
}

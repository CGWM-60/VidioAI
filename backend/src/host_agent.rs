//! Client du Host Agent natif et contrat de ressources agrégé.
//!
//! Le backend tourne dans Docker : toute mesure collectée localement décrit le
//! conteneur. Ce module donne donc toujours la priorité au Host Agent et marque
//! explicitement le fallback avec `source = "container"`.

use crate::utils::systeme::{HardwareProfile, StorageVolume as ContainerVolume, check_pc};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResourceSource {
    Host,
    Container,
    Worker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub source: ResourceSource,
    pub os: Option<String>,
    pub os_version: Option<String>,
    pub kernel: Option<String>,
    pub hostname: Option<String>,
    pub architecture: Option<String>,
    pub uptime_seconds: Option<u64>,
    pub measured_at_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub source: ResourceSource,
    pub model: Option<String>,
    pub physical_cores: Option<usize>,
    pub logical_cpus: Option<usize>,
    pub frequency_mhz: Option<u64>,
    pub utilization_percent: Option<f64>,
    pub temperature_celsius: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamMetrics {
    pub source: ResourceSource,
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub memory_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub source: ResourceSource,
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub vram_total_bytes: Option<u64>,
    pub vram_used_bytes: Option<u64>,
    pub vram_available_bytes: Option<u64>,
    pub utilization_percent: Option<f64>,
    pub temperature_celsius: Option<f64>,
    pub driver_version: Option<String>,
    pub runtime_version: Option<String>,
}

impl GpuMetrics {
    pub fn is_nvidia(&self) -> bool {
        self.manufacturer
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("nvidia"))
            || self
                .model
                .as_deref()
                .is_some_and(|value| value.to_ascii_lowercase().contains("nvidia"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageVolume {
    pub name: Option<String>,
    pub mount_point: String,
    pub filesystem: Option<String>,
    pub storage_type: Option<String>,
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    pub source: ResourceSource,
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub volumes: Vec<StorageVolume>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSnapshot {
    pub source: ResourceSource,
    pub agent_version: String,
    pub system: SystemMetrics,
    pub cpu: CpuMetrics,
    pub ram: RamMetrics,
    pub gpus: Vec<GpuMetrics>,
    pub storage: StorageMetrics,
}

impl HostSnapshot {
    pub fn physical_nvidia(&self) -> Option<&GpuMetrics> {
        self.gpus.iter().find(|gpu| gpu.is_nvidia())
    }

    pub fn total_ram_bytes(&self) -> Option<u64> {
        self.ram.total_bytes
    }

    pub fn total_vram_bytes(&self) -> Option<u64> {
        self.gpus
            .iter()
            .filter_map(|gpu| gpu.vram_total_bytes)
            .max()
    }

    fn unavailable_container() -> Self {
        let measured_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or_default();
        Self {
            source: ResourceSource::Container,
            agent_version: env!("CARGO_PKG_VERSION").to_owned(),
            system: SystemMetrics {
                source: ResourceSource::Container,
                os: None,
                os_version: None,
                kernel: None,
                hostname: None,
                architecture: None,
                uptime_seconds: None,
                measured_at_unix,
            },
            cpu: CpuMetrics {
                source: ResourceSource::Container,
                model: None,
                physical_cores: None,
                logical_cpus: None,
                frequency_mhz: None,
                utilization_percent: None,
                temperature_celsius: None,
            },
            ram: RamMetrics {
                source: ResourceSource::Container,
                total_bytes: None,
                used_bytes: None,
                available_bytes: None,
                memory_type: None,
            },
            gpus: Vec::new(),
            storage: StorageMetrics {
                source: ResourceSource::Container,
                total_bytes: None,
                used_bytes: None,
                available_bytes: None,
                volumes: Vec::new(),
            },
        }
    }
}

fn known_string(value: String, unknown: &[&str]) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()
        && !unknown
            .iter()
            .any(|candidate| value.eq_ignore_ascii_case(candidate)))
    .then(|| value.to_owned())
}

fn convert_volume(volume: ContainerVolume) -> StorageVolume {
    StorageVolume {
        name: known_string(volume.nom, &["inconnu"]),
        mount_point: volume.point_montage,
        filesystem: known_string(volume.systeme_fichiers, &["inconnu"]),
        storage_type: known_string(volume.type_stockage, &["inconnu"]),
        total_bytes: (volume.total > 0).then_some(volume.total),
        used_bytes: (volume.total > 0).then_some(volume.utilise),
        available_bytes: (volume.total > 0).then_some(volume.disponible),
    }
}

impl From<HardwareProfile> for HostSnapshot {
    fn from(profile: HardwareProfile) -> Self {
        let gpus = if profile.nbrs_total_gpu > 0 && profile.name_carte != "Aucun GPU" {
            vec![GpuMetrics {
                source: ResourceSource::Container,
                manufacturer: profile.gpu_est_nvidia.then(|| "NVIDIA".to_owned()),
                model: known_string(profile.name_carte, &["aucun gpu", "gpu inconnu"]),
                backend: known_string(profile.gpu_backend, &["cpu", "inconnu"]),
                vram_total_bytes: (profile.vram_total > 0).then_some(profile.vram_total),
                vram_used_bytes: (profile.vram_total > 0).then_some(profile.vram_utilisee),
                vram_available_bytes: (profile.vram_total > 0).then_some(profile.vram_disponible),
                utilization_percent: Some(profile.gpu_utilisation),
                temperature_celsius: profile.gpu_temperature,
                driver_version: profile.gpu_driver_version,
                runtime_version: profile.cuda_version,
            }]
        } else {
            Vec::new()
        };
        let ram_total = (profile.ram_total > 0).then_some(profile.ram_total);
        let ram_available = (profile.ram_total > 0).then_some(profile.ram_disponible);
        let volumes = profile
            .stockage_volumes
            .into_iter()
            .map(convert_volume)
            .collect();

        Self {
            source: ResourceSource::Container,
            agent_version: profile.backend_version,
            system: SystemMetrics {
                source: ResourceSource::Container,
                os: known_string(profile.os, &["inconnu"]),
                os_version: None,
                kernel: known_string(profile.kernel, &["inconnu"]),
                hostname: known_string(profile.hostname, &["inconnu"]),
                architecture: known_string(profile.architecture, &["inconnu"]),
                uptime_seconds: Some(profile.uptime_secondes),
                measured_at_unix: profile.date_mesure_unix,
            },
            cpu: CpuMetrics {
                source: ResourceSource::Container,
                model: known_string(profile.cpu, &["cpu inconnu", "inconnu"]),
                physical_cores: (profile.cpu_coeurs > 0).then_some(profile.cpu_coeurs),
                logical_cpus: (profile.cpu_threads > 0).then_some(profile.cpu_threads),
                frequency_mhz: (profile.cpu_frequence_mhz > 0).then_some(profile.cpu_frequence_mhz),
                utilization_percent: Some(profile.cpu_utilisation),
                temperature_celsius: profile.cpu_temperature,
            },
            ram: RamMetrics {
                source: ResourceSource::Container,
                total_bytes: ram_total,
                used_bytes: ram_total.map(|total| total.saturating_sub(profile.ram_disponible)),
                available_bytes: ram_available,
                memory_type: profile.ram_type,
            },
            gpus,
            storage: StorageMetrics {
                source: ResourceSource::Container,
                total_bytes: (profile.stockage_total > 0).then_some(profile.stockage_total),
                used_bytes: (profile.stockage_total > 0).then_some(
                    profile
                        .stockage_total
                        .saturating_sub(profile.stockage_disponible),
                ),
                available_bytes: (profile.stockage_total > 0)
                    .then_some(profile.stockage_disponible),
                volumes,
            },
        }
    }
}

#[derive(Clone)]
pub struct HostAgentClient {
    base_url: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl HostAgentClient {
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("HOST_AGENT_URL").ok()?;
        if base_url.trim().is_empty() {
            return None;
        }
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(8))
            .build()
            .ok()?;
        Some(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            token: std::env::var("HOST_AGENT_TOKEN").ok(),
            http,
        })
    }

    fn request(&self, path: &str) -> reqwest::RequestBuilder {
        let request = self.http.get(format!("{}{}", self.base_url, path));
        match &self.token {
            Some(token) => request.header("X-VidioAI-Host-Token", token),
            None => request,
        }
    }

    async fn json<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, String> {
        let response = self
            .request(path)
            .send()
            .await
            .map_err(|error| format!("Host Agent inaccessible : {error}"))?;
        let status = response.status();
        if status != StatusCode::OK {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Host Agent HTTP {status}: {body}"));
        }
        response
            .json()
            .await
            .map_err(|error| format!("Réponse Host Agent invalide : {error}"))
    }

    pub async fn health(&self) -> Result<(), String> {
        let value: serde_json::Value = self.json("/health").await?;
        if value.get("status").and_then(|value| value.as_str()) == Some("ok")
            && value.get("service").and_then(|value| value.as_str()) == Some("vidioai-host-agent")
        {
            Ok(())
        } else {
            Err("Le service sur HOST_AGENT_URL n'est pas vidioai-host-agent.".to_owned())
        }
    }

    pub async fn system(&self) -> Result<HostSnapshot, String> {
        let snapshot: HostSnapshot = self.json("/system").await?;
        if snapshot.source != ResourceSource::Host {
            return Err("Le Host Agent n'a pas marqué les données comme hôte.".to_owned());
        }
        Ok(snapshot)
    }
}

/// Résout le meilleur profil disponible et retourne séparément la cause du
/// fallback afin que l'API et l'interface n'affichent jamais un état trompeur.
pub async fn resolve_system(client: Option<&HostAgentClient>) -> (HostSnapshot, Option<String>) {
    if let Some(client) = client {
        match client.system().await {
            Ok(snapshot) => return (snapshot, None),
            Err(error) => {
                let fallback = check_pc()
                    .map(HostSnapshot::from)
                    .unwrap_or_else(|_| HostSnapshot::unavailable_container());
                return (fallback, Some(error));
            }
        }
    }
    let fallback = check_pc()
        .map(HostSnapshot::from)
        .unwrap_or_else(|_| HostSnapshot::unavailable_container());
    (
        fallback,
        Some("HOST_AGENT_URL non configurée : métriques du conteneur affichées.".to_owned()),
    )
}

#[cfg(test)]
mod tests {
    use super::{GpuMetrics, ResourceSource};

    #[test]
    fn recognizes_physical_nvidia_gpu() {
        let gpu = GpuMetrics {
            source: ResourceSource::Host,
            manufacturer: Some("NVIDIA".to_owned()),
            model: Some("H100 PCIe".to_owned()),
            backend: Some("CUDA".to_owned()),
            vram_total_bytes: None,
            vram_used_bytes: None,
            vram_available_bytes: None,
            utilization_percent: None,
            temperature_celsius: None,
            driver_version: None,
            runtime_version: None,
        };
        assert!(gpu.is_nvidia());
    }
}

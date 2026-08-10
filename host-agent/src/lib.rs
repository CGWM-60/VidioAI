//! Collecte matérielle native du Host Agent VidioAI.
//!
//! Ce crate reste volontairement indépendant du backend : il ne connaît ni les
//! modèles, ni les jobs, ni la base de données. Son unique responsabilité est
//! de décrire ce que le système d'exploitation hôte permet réellement de lire.

use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use sysinfo::{Components, Disks, System};

/// Provenance placée sur chaque sous-système. Le backend pourra remplacer
/// `host` par `container` lorsqu'il active son fallback local.
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

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn percentage(value: f64) -> Option<f64> {
    value.is_finite().then(|| value.clamp(0.0, 100.0))
}

fn temperature(value: f64) -> Option<f64> {
    // Zéro est presque toujours la valeur sentinelle d'un capteur inaccessible.
    // Il ne doit jamais devenir « 0 °C » dans l'interface.
    (value.is_finite() && value > 0.0 && value < 200.0).then_some(value)
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn cpu_temperature() -> Option<f64> {
    Components::new_with_refreshed_list()
        .iter()
        .filter(|component| {
            let label = component.label().to_ascii_lowercase();
            label.contains("cpu")
                || label.contains("package")
                || label.contains("tctl")
                || label.contains("soc")
        })
        .filter_map(|component| component.temperature())
        .map(f64::from)
        .filter_map(temperature)
        .max_by(f64::total_cmp)
}

/// Extrait la version CUDA annoncée par le pilote. Cette valeur décrit le
/// runtime supporté par le pilote, pas la disponibilité de CUDA dans le worker.
pub fn parse_cuda_version(output: &str) -> Option<String> {
    let marker = "CUDA Version:";
    let tail = output.split_once(marker)?.1.trim_start();
    non_empty(tail.split_whitespace().next().map(str::to_owned))
}

/// Convertit une ligne CSV produite par `nvidia-smi`. Les nombres invalides ou
/// `N/A` deviennent `None` plutôt qu'une métrique artificielle égale à zéro.
pub fn parse_nvidia_csv_line(line: &str, runtime_version: Option<&str>) -> Option<GpuMetrics> {
    let fields: Vec<_> = line.split(',').map(str::trim).collect();
    if fields.len() < 7 {
        return None;
    }
    let mib = |value: &str| value.parse::<u64>().ok().map(|value| value * 1024 * 1024);
    let number = |value: &str| value.parse::<f64>().ok();
    let total = mib(fields[1]);
    let used = mib(fields[2]);
    let available = match (total, used) {
        (Some(total), Some(used)) => Some(total.saturating_sub(used.min(total))),
        _ => None,
    };

    Some(GpuMetrics {
        source: ResourceSource::Host,
        manufacturer: Some("NVIDIA".to_owned()),
        model: non_empty(Some(fields[0].to_owned())),
        backend: Some("CUDA".to_owned()),
        vram_total_bytes: total,
        vram_used_bytes: used,
        vram_available_bytes: available,
        utilization_percent: number(fields[3]).and_then(percentage),
        temperature_celsius: number(fields[4]).and_then(temperature),
        driver_version: non_empty(Some(fields[5].to_owned())),
        runtime_version: runtime_version
            .map(str::to_owned)
            .and_then(|value| non_empty(Some(value))),
    })
}

fn nvidia_gpus() -> Vec<GpuMetrics> {
    let full_output = command_output("nvidia-smi", &[]).unwrap_or_default();
    let runtime_version = parse_cuda_version(&full_output);
    let Some(csv) = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version,index",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return Vec::new();
    };

    csv.lines()
        .filter_map(|line| parse_nvidia_csv_line(line, runtime_version.as_deref()))
        .collect()
}

fn apple_gpu(architecture: Option<&str>, cpu_model: Option<&str>) -> Option<GpuMetrics> {
    #[cfg(target_os = "macos")]
    {
        let apple_silicon = architecture.is_some_and(|arch| arch == "arm64" || arch == "aarch64")
            && cpu_model.is_some_and(|model| model.to_ascii_lowercase().contains("apple"));
        if apple_silicon {
            return Some(GpuMetrics {
                source: ResourceSource::Host,
                manufacturer: Some("Apple".to_owned()),
                model: cpu_model.map(str::to_owned),
                backend: Some("Metal".to_owned()),
                // Apple Silicon emploie une mémoire unifiée. La présenter comme
                // une VRAM dédiée serait techniquement faux.
                vram_total_bytes: None,
                vram_used_bytes: None,
                vram_available_bytes: None,
                utilization_percent: None,
                temperature_celsius: None,
                driver_version: None,
                runtime_version: None,
            });
        }
    }
    let _ = (architecture, cpu_model);
    None
}

fn collect_storage() -> StorageMetrics {
    let disks = Disks::new_with_refreshed_list();
    let mut volumes = Vec::new();
    let mut physical_pools = HashSet::new();
    let mut total = 0_u64;
    let mut available = 0_u64;

    for disk in &disks {
        if disk.total_space() == 0 {
            continue;
        }
        let name = disk.name().to_string_lossy().into_owned();
        let mount_point = disk.mount_point().to_string_lossy().into_owned();
        let filesystem = disk.file_system().to_string_lossy().into_owned();
        let volume_total = disk.total_space();
        let volume_available = disk.available_space().min(volume_total);
        let volume_used = volume_total.saturating_sub(volume_available);

        volumes.push(StorageVolume {
            name: non_empty(Some(name.clone())),
            mount_point,
            filesystem: non_empty(Some(filesystem)),
            storage_type: non_empty(Some(disk.kind().to_string())),
            total_bytes: Some(volume_total),
            used_bytes: Some(volume_used),
            available_bytes: Some(volume_available),
        });

        // APFS expose plusieurs montages partageant le même pool. Cette clé
        // empêche de doubler le total tout en conservant chaque volume affiché.
        // Les valeurs disponibles de deux montages APFS peuvent différer de
        // quelques blocs pendant la collecte. Le nom du store et sa capacité
        // physique forment une clé plus stable que l'espace libre instantané.
        if physical_pools.insert((name, volume_total)) {
            total = total.saturating_add(volume_total);
            available = available.saturating_add(volume_available);
        }
    }

    StorageMetrics {
        source: ResourceSource::Host,
        total_bytes: (!volumes.is_empty()).then_some(total),
        used_bytes: (!volumes.is_empty()).then_some(total.saturating_sub(available)),
        available_bytes: (!volumes.is_empty()).then_some(available),
        volumes,
    }
}

/// Effectue une photographie synchrone. L'API exécute cette fonction dans une
/// tâche bloquante afin de ne pas immobiliser l'ordonnanceur Tokio.
pub fn collect_snapshot() -> HostSnapshot {
    let mut system = System::new_all();
    system.refresh_all();

    let architecture = non_empty(Some(System::cpu_arch()));
    let sysinfo_cpu = system
        .cpus()
        .first()
        .and_then(|cpu| non_empty(Some(cpu.brand().to_owned())));
    let cpu_model = if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"]).or(sysinfo_cpu)
    } else {
        sysinfo_cpu
    };
    let logical_cpus = (!system.cpus().is_empty()).then_some(system.cpus().len());
    let frequency_mhz = logical_cpus.and_then(|count| {
        let sum = system.cpus().iter().map(|cpu| cpu.frequency()).sum::<u64>();
        (sum > 0).then_some(sum / count as u64)
    });
    let total_ram = system.total_memory();
    let available_ram = system.available_memory().min(total_ram);
    let measured_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();

    let mut gpus = nvidia_gpus();
    if gpus.is_empty()
        && let Some(gpu) = apple_gpu(architecture.as_deref(), cpu_model.as_deref())
    {
        gpus.push(gpu);
    }

    HostSnapshot {
        source: ResourceSource::Host,
        agent_version: env!("CARGO_PKG_VERSION").to_owned(),
        system: SystemMetrics {
            source: ResourceSource::Host,
            // sysinfo expose historiquement le noyau « Darwin » comme nom. Le
            // produit attendu par l'utilisateur est bien macOS.
            os: if cfg!(target_os = "macos") {
                Some("macOS".to_owned())
            } else {
                non_empty(System::name())
            },
            os_version: non_empty(System::os_version()),
            kernel: non_empty(System::kernel_version()),
            hostname: non_empty(System::host_name()),
            architecture,
            uptime_seconds: Some(System::uptime()),
            measured_at_unix,
        },
        cpu: CpuMetrics {
            source: ResourceSource::Host,
            model: cpu_model.clone(),
            physical_cores: System::physical_core_count(),
            logical_cpus,
            frequency_mhz,
            utilization_percent: percentage(f64::from(system.global_cpu_usage())),
            temperature_celsius: cpu_temperature(),
        },
        ram: RamMetrics {
            source: ResourceSource::Host,
            total_bytes: (total_ram > 0).then_some(total_ram),
            used_bytes: (total_ram > 0).then_some(total_ram.saturating_sub(available_ram)),
            available_bytes: (total_ram > 0).then_some(available_ram),
            memory_type: if cfg!(target_os = "macos")
                && cpu_model
                    .as_deref()
                    .is_some_and(|model| model.to_ascii_lowercase().contains("apple"))
            {
                Some("Mémoire unifiée".to_owned())
            } else {
                None
            },
        },
        gpus,
        storage: collect_storage(),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_cuda_version, parse_nvidia_csv_line};

    #[test]
    fn parses_nvidia_metrics_without_inventing_values() {
        let gpu = parse_nvidia_csv_line(
            "NVIDIA H100 PCIe, 81559, 12698, 15, 62, 535.154.05, 0",
            Some("12.4"),
        )
        .expect("valid NVIDIA row");

        assert_eq!(gpu.model.as_deref(), Some("NVIDIA H100 PCIe"));
        assert_eq!(gpu.vram_total_bytes, Some(81_559 * 1024 * 1024));
        assert_eq!(gpu.temperature_celsius, Some(62.0));
        assert_eq!(gpu.runtime_version.as_deref(), Some("12.4"));
    }

    #[test]
    fn zero_temperature_is_unknown() {
        let gpu = parse_nvidia_csv_line("GPU, N/A, N/A, N/A, 0, 550.1, 0", None)
            .expect("structurally valid row");
        assert_eq!(gpu.temperature_celsius, None);
        assert_eq!(gpu.vram_total_bytes, None);
    }

    #[test]
    fn extracts_cuda_version_from_standard_output() {
        assert_eq!(
            parse_cuda_version("Driver Version: 550.54.15    CUDA Version: 12.4     |"),
            Some("12.4".to_owned())
        );
    }
}

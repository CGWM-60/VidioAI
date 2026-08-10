use all_smi::{AllSmi, Result as AllSmiResult};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use sysinfo::{Components, Disks, System};

#[derive(Debug, Serialize)]
pub struct StorageVolume {
    pub nom: String,
    pub point_montage: String,
    pub systeme_fichiers: String,
    pub type_stockage: String,
    pub total: u64,
    pub utilise: u64,
    pub disponible: u64,
}

#[derive(Debug, Serialize)]
pub struct HardwareProfile {
    pub os: String,
    pub kernel: String,
    pub architecture: String,
    pub hostname: String,
    pub date_mesure_unix: u64,
    pub uptime_secondes: u64,
    pub backend_version: String,

    pub cpu: String,
    pub cpu_coeurs: usize,
    pub cpu_threads: usize,
    pub cpu_frequence_mhz: u64,
    pub cpu_utilisation: f64,
    pub cpu_temperature: Option<f64>,

    pub ram_total: u64,
    pub ram_disponible: u64,
    pub ram_type: Option<String>,

    pub nbrs_total_gpu: usize,
    pub name_carte: String,
    pub gpu_est_nvidia: bool,

    pub vram_total: u64,
    pub vram_utilisee: u64,
    pub vram_disponible: u64,
    pub gpu_utilisation: f64,
    pub gpu_temperature: Option<f64>,

    /// Conservé pour compatibilité : correspond à l'espace de stockage disponible.
    pub taille_storage: u64,
    pub stockage_total: u64,
    pub stockage_disponible: u64,
    pub stockage_volumes: Vec<StorageVolume>,

    pub gpu_backend: String,
    pub cuda_disponible: bool,
    pub cuda_version: Option<String>,
    pub gpu_driver_version: Option<String>,
}

fn detail_value(details: &HashMap<String, String>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| details.get(*key).map(|value| value.trim()))
        .find(|value| !value.is_empty() && *value != "0.0")
        .map(str::to_owned)
}

fn is_nvidia_gpu(name: &str, details: &HashMap<String, String>) -> bool {
    name.to_ascii_lowercase().contains("nvidia")
        || details
            .get("lib_name")
            .is_some_and(|backend| backend.eq_ignore_ascii_case("cuda"))
        || details.contains_key("CUDA Version")
        || details.contains_key("cuda_version")
}

fn normalized_percentage(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 100.0)
    } else {
        0.0
    }
}

fn normalized_temperature(value: f64) -> Option<f64> {
    (value.is_finite() && value > 0.0 && value < 200.0).then_some(value)
}

fn component_cpu_temperature() -> Option<f64> {
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
        .filter_map(normalized_temperature)
        .max_by(f64::total_cmp)
}

pub fn check_pc() -> AllSmiResult<HardwareProfile> {
    // =========================
    // ALL-SMI
    // =========================

    let smi = AllSmi::new()?;

    // =========================
    // SYSTEME
    // =========================

    let mut sys = System::new_all();
    sys.refresh_all();

    // =========================
    // OS
    // =========================

    let os = System::long_os_version().unwrap_or_else(|| "Inconnu".to_string());
    let kernel = System::kernel_version().unwrap_or_else(|| "Inconnu".to_string());
    let architecture = System::cpu_arch();
    let hostname = System::host_name().unwrap_or_else(|| "Inconnu".to_string());
    let date_mesure_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let uptime_secondes = System::uptime();
    let backend_version = env!("CARGO_PKG_VERSION").to_string();

    // =========================
    // CPU
    // =========================

    let cpus = smi.get_cpu_info();
    let primary_cpu = cpus.first();

    let cpu_name = primary_cpu
        .map(|cpu| cpu.cpu_model.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            sys.cpus()
                .first()
                .map(|cpu| cpu.brand().trim())
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "CPU inconnu".to_string());

    let cpu_coeurs = System::physical_core_count()
        .or_else(|| primary_cpu.map(|cpu| cpu.total_cores as usize))
        .unwrap_or_else(|| sys.cpus().len());
    let cpu_threads = primary_cpu
        .map(|cpu| cpu.total_threads as usize)
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| sys.cpus().len());
    let sysinfo_frequency = if sys.cpus().is_empty() {
        0
    } else {
        sys.cpus().iter().map(|cpu| cpu.frequency()).sum::<u64>() / sys.cpus().len() as u64
    };
    let cpu_frequence_mhz = primary_cpu
        .map(|cpu| u64::from(cpu.base_frequency_mhz))
        .filter(|frequency| *frequency > 0)
        .unwrap_or(sysinfo_frequency);
    let cpu_utilisation = primary_cpu
        .map(|cpu| normalized_percentage(cpu.utilization))
        .unwrap_or_else(|| normalized_percentage(f64::from(sys.global_cpu_usage())));
    let cpu_temperature = primary_cpu
        .and_then(|cpu| cpu.temperature)
        .and_then(|temperature| normalized_temperature(f64::from(temperature)))
        .or_else(component_cpu_temperature);

    // =========================
    // RAM
    // =========================

    let ram_total = sys.total_memory();
    let ram_disponible = sys.available_memory();
    let ram_type = primary_cpu
        .and_then(|cpu| cpu.apple_silicon_info.as_ref())
        .map(|_| "Mémoire unifiée".to_string());

    // =========================
    // GPU
    // =========================

    let gpus = smi.get_gpu_info();

    let gpu_nbr = gpus.len();

    // Sur une machine multi-GPU, NVIDIA est prioritaire pour que les informations
    // CUDA proviennent bien de la carte qui sera utilisée pour l'inférence.
    let selected_gpu = gpus
        .iter()
        .find(|gpu| is_nvidia_gpu(&gpu.name, &gpu.detail))
        .or_else(|| gpus.first());

    let mut gpu_name = "Aucun GPU".to_string();
    let mut gpu_est_nvidia = false;
    let mut total_vram: u64 = 0;
    let mut used_vram: u64 = 0;
    let mut disponible_vram: u64 = 0;
    let mut gpu_utilisation = 0.0;
    let mut gpu_temperature = None;
    let mut gpu_backend = "CPU".to_string();
    let mut cuda_disponible = false;
    let mut cuda_version = None;
    let mut gpu_driver_version = None;

    if let Some(gpu) = selected_gpu {
        gpu_name = gpu.name.clone();
        gpu_est_nvidia = is_nvidia_gpu(&gpu.name, &gpu.detail);
        total_vram = gpu.total_memory;
        used_vram = gpu.used_memory.min(total_vram);
        disponible_vram = total_vram.saturating_sub(used_vram);
        gpu_utilisation = normalized_percentage(gpu.utilization);
        gpu_temperature = normalized_temperature(f64::from(gpu.temperature));

        if let Some(backend) = detail_value(&gpu.detail, &["lib_name"]) {
            gpu_backend = backend;
        } else {
            gpu_backend = "GPU".to_string();
        }

        if gpu_est_nvidia {
            cuda_version = detail_value(
                &gpu.detail,
                &["CUDA Version", "cuda_version", "lib_version"],
            );
            gpu_driver_version = detail_value(&gpu.detail, &["Driver Version", "driver_version"]);
            cuda_disponible = gpu_backend.eq_ignore_ascii_case("cuda") || cuda_version.is_some();
        }
    }

    // =========================
    // STOCKAGE
    // =========================

    let disks = Disks::new_with_refreshed_list();

    let mut stockage_total: u64 = 0;
    let mut stockage_disponible: u64 = 0;
    let mut stockage_volumes = Vec::new();
    let mut pools_vus = HashSet::new();

    for disk in &disks {
        if disk.total_space() == 0 {
            continue;
        }

        let total = disk.total_space();
        let disponible = disk.available_space().min(total);
        let utilise = total.saturating_sub(disponible);
        let nom = disk.name().to_string_lossy().into_owned();

        // APFS expose notamment les volumes système et données comme deux montages
        // qui partagent exactement le même pool physique. On ne les additionne pas
        // deux fois dans le profil retourné.
        if !pools_vus.insert((nom.clone(), total, disponible)) {
            continue;
        }

        stockage_total = stockage_total.saturating_add(total);
        stockage_disponible = stockage_disponible.saturating_add(disponible);

        stockage_volumes.push(StorageVolume {
            nom,
            point_montage: disk.mount_point().to_string_lossy().into_owned(),
            systeme_fichiers: disk.file_system().to_string_lossy().into_owned(),
            type_stockage: disk.kind().to_string(),
            total,
            utilise,
            disponible,
        });
    }

    // =========================
    // PROFILE FINAL
    // =========================

    let profile = HardwareProfile {
        os,
        kernel,
        architecture,
        hostname,
        date_mesure_unix,
        uptime_secondes,
        backend_version,

        cpu: cpu_name,
        cpu_coeurs,
        cpu_threads,
        cpu_frequence_mhz,
        cpu_utilisation,
        cpu_temperature,

        ram_total,
        ram_disponible,
        ram_type,

        nbrs_total_gpu: gpu_nbr,
        name_carte: gpu_name,
        gpu_est_nvidia,

        vram_total: total_vram,
        vram_utilisee: used_vram,
        vram_disponible: disponible_vram,
        gpu_utilisation,
        gpu_temperature,

        taille_storage: stockage_disponible,
        stockage_total,
        stockage_disponible,
        stockage_volumes,

        gpu_backend,
        cuda_disponible,
        cuda_version,
        gpu_driver_version,
    };

    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::{detail_value, is_nvidia_gpu};
    use std::collections::HashMap;

    #[test]
    fn detects_nvidia_from_cuda_backend() {
        let details = HashMap::from([("lib_name".to_string(), "CUDA".to_string())]);

        assert!(is_nvidia_gpu("GeForce RTX 4090", &details));
    }

    #[test]
    fn reads_the_first_valid_detail_value() {
        let details = HashMap::from([
            ("CUDA Version".to_string(), "0.0".to_string()),
            ("lib_version".to_string(), "12.8".to_string()),
        ]);

        assert_eq!(
            detail_value(&details, &["CUDA Version", "lib_version"]),
            Some("12.8".to_string())
        );
    }
}

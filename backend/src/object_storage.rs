//! Stockage objet S3-compatible pour les assets et snapshots de modèles.
//!
//! Les snapshots n'utilisent volontairement pas `aws s3 sync`: l'uploader
//! contrôle la taille des parties, confirme chaque partie envoyée, reprend les
//! fichiers déjà complets et publie le manifeste de validation en dernier.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{fs, process::Command, task};
use uuid::Uuid;

pub const MIN_MULTIPART_CHUNK_SIZE: u64 = 128 * 1024 * 1024;
pub const MULTIPART_HEADROOM_PARTS: u64 = 900;
pub const MAX_MULTIPART_PARTS: u64 = 1000;
pub const SNAPSHOT_MANIFEST_NAME: &str = "manifest.json";
pub const SNAPSHOT_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotFile {
    pub path: String,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotManifest {
    pub repository: String,
    pub revision: String,
    pub files: Vec<SnapshotFile>,
    pub total_size: u64,
    pub created_at: u64,
    pub schema_version: u32,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UploadProgress {
    pub direction: String,
    pub provider: String,
    pub bytes_total: u64,
    pub bytes_transferred: u64,
    pub files_total: u64,
    pub files_completed: u64,
    pub files_skipped: u64,
    pub current_file: Option<String>,
    pub current_file_size: u64,
    pub current_file_bytes: u64,
    pub bytes_per_second: Option<u64>,
    pub eta_seconds: Option<u64>,
}

impl UploadProgress {
    pub fn percent(&self) -> f64 {
        if self.bytes_total == 0 {
            100.0
        } else {
            (self.bytes_transferred.min(self.bytes_total) as f64 / self.bytes_total as f64) * 100.0
        }
    }
}

fn mark_existing_file(progress: &mut UploadProgress, size: u64) {
    progress.bytes_transferred = progress
        .bytes_transferred
        .saturating_add(size)
        .min(progress.bytes_total);
    progress.files_completed = progress.files_completed.saturating_add(1);
    progress.files_skipped = progress.files_skipped.saturating_add(1);
    progress.current_file_bytes = size;
}

fn mark_transferred_bytes(progress: &mut UploadProgress, transferred: u64, file_size: u64) {
    progress.bytes_transferred = progress
        .bytes_transferred
        .saturating_add(transferred)
        .min(progress.bytes_total);
    progress.current_file_bytes = progress
        .current_file_bytes
        .saturating_add(transferred)
        .min(file_size);
}

fn snapshot_upload_complete(progress: &UploadProgress) -> bool {
    progress.files_completed == progress.files_total
        && progress.bytes_transferred == progress.bytes_total
}

pub type UploadProgressCallback = Arc<dyn Fn(UploadProgress) + Send + Sync>;
pub type TransferProgressCallback = UploadProgressCallback;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UploadOutcome {
    pub manifest: SnapshotManifest,
    pub files_skipped: u64,
}

#[async_trait]
pub trait ObjectStorage: Send + Sync {
    fn enabled(&self) -> bool;
    async fn health(&self) -> Result<(), String>;
    async fn upload_file(&self, local: &Path, key: &str) -> Result<(), String>;
    async fn restore_snapshot(
        &self,
        repository: &str,
        revision: &str,
        local: &Path,
        progress: Option<TransferProgressCallback>,
    ) -> Result<bool, String>;
    async fn list_snapshots(&self) -> Result<Vec<SnapshotManifest>, String>;
    async fn upload_snapshot(
        &self,
        repository: &str,
        revision: &str,
        local: &Path,
        progress: Option<UploadProgressCallback>,
    ) -> Result<UploadOutcome, String>;
}

#[derive(Clone)]
pub struct S3Storage {
    enabled: bool,
    bucket: String,
    endpoint: Option<String>,
    storage_class: String,
}

/// Taille de partie arrondie au MiB supérieur, avec une marge de 900 parties.
pub fn multipart_chunk_size(file_size: u64) -> Result<u64, String> {
    let required = file_size.div_ceil(MULTIPART_HEADROOM_PARTS);
    let mib = 1024 * 1024;
    let chunk = required.max(MIN_MULTIPART_CHUNK_SIZE).div_ceil(mib) * mib;
    if file_size.div_ceil(chunk) > MAX_MULTIPART_PARTS {
        return Err(format!(
            "S3_MULTIPART_CONFIGURATION_INVALID: {file_size} octets exigeraient plus de {MAX_MULTIPART_PARTS} parties"
        ));
    }
    Ok(chunk)
}

/// Source de vérité unique pour les fichiers appartenant réellement au modèle.
pub fn is_snapshot_file(path: &Path) -> bool {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return false;
    }
    let mut saw_component = false;
    for component in path.components() {
        let Component::Normal(value) = component else {
            return false;
        };
        saw_component = true;
        let value = value.to_string_lossy();
        if value == ".cache" || value == "__pycache__" {
            return false;
        }
    }
    let name = path
        .file_name()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    saw_component
        && name != SNAPSHOT_MANIFEST_NAME
        && !name.ends_with(".pyc")
        && !name.ends_with(".tmp")
        && !name.ends_with(".lock")
}

/// Préfixe canonique : `models/<owner>/<repository>/<revision>`.
pub fn model_s3_prefix(repository: &str, revision: &str) -> Result<String, String> {
    let parts = repository.split('/').collect::<Vec<_>>();
    let valid = parts.len() == 2
        && parts.iter().chain(std::iter::once(&revision)).all(|part| {
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && part
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character))
        });
    if !valid {
        return Err("S3_MODEL_KEY_INVALID: repository ou révision invalide".into());
    }
    Ok(format!("models/{repository}/{revision}"))
}

/// Chemin relatif sûr d'un fichier à l'intérieur du snapshot exact.
pub fn relative_snapshot_path(snapshot_root: &Path, file: &Path) -> Result<PathBuf, String> {
    let relative = file
        .strip_prefix(snapshot_root)
        .map_err(|_| "S3_SNAPSHOT_PATH_INVALID: fichier hors snapshot".to_owned())?;
    if !is_snapshot_file(relative) {
        return Err("S3_SNAPSHOT_PATH_EXCLUDED: fichier hors contrat snapshot".into());
    }
    Ok(relative.to_path_buf())
}

impl S3Storage {
    pub fn from_env() -> Self {
        let bucket = std::env::var("AWS_S3_BUCKET").unwrap_or_default();
        Self {
            enabled: std::env::var("VIDIOAI_S3_ENABLED")
                .is_ok_and(|value| value.eq_ignore_ascii_case("true")),
            bucket,
            endpoint: std::env::var("AWS_ENDPOINT_URL_S3").ok(),
            storage_class: std::env::var("AWS_S3_STORAGE_CLASS")
                .unwrap_or_else(|_| "STANDARD".to_owned()),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.bucket.is_empty() {
            return Err("AWS_S3_BUCKET est obligatoire quand S3 est activé".to_owned());
        }
        if self.bucket.contains('/') || self.bucket.starts_with("s3://") {
            return Err(
                "AWS_S3_BUCKET doit être un nom de bucket sans schéma ni préfixe".to_owned(),
            );
        }
        Ok(())
    }

    pub fn snapshot_uri(&self, repository: &str, revision: &str) -> Result<String, String> {
        Ok(format!(
            "s3://{}/{}/{}",
            self.bucket,
            model_s3_prefix(repository, revision)?,
            SNAPSHOT_MANIFEST_NAME
        ))
    }

    async fn read_manifest(&self, key: &str) -> Result<SnapshotManifest, String> {
        let temporary =
            std::env::temp_dir().join(format!("vidioai-s3-manifest-{}.json.tmp", Uuid::new_v4()));
        let result = self
            .aws(&[
                "s3api".into(),
                "get-object".into(),
                "--bucket".into(),
                self.bucket.clone(),
                "--key".into(),
                key.into(),
                temporary.to_string_lossy().into_owned(),
            ])
            .await;
        if let Err(error) = result {
            let _ = fs::remove_file(&temporary).await;
            return Err(error);
        }
        let bytes = fs::read(&temporary)
            .await
            .map_err(|error| format!("S3_MANIFEST_INVALID: {error}"));
        let _ = fs::remove_file(&temporary).await;
        serde_json::from_slice(&bytes?).map_err(|error| format!("S3_MANIFEST_INVALID: {error}"))
    }

    async fn aws_output(&self, args: &[String]) -> Result<Vec<u8>, String> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        self.validate()?;
        let mut command = Command::new("aws");
        command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env("AWS_METADATA_SERVICE_TIMEOUT", "1")
            .env("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1");
        if let Some(endpoint) = &self.endpoint {
            command.arg("--endpoint-url").arg(endpoint);
        }
        let output = command.output().await.map_err(|error| error.to_string())?;
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
        }
    }

    async fn aws(&self, args: &[String]) -> Result<(), String> {
        self.aws_output(args).await.map(|_| ())
    }

    async fn head_size(&self, key: &str) -> Result<Option<u64>, String> {
        let args = vec![
            "s3api".into(),
            "head-object".into(),
            "--bucket".into(),
            self.bucket.clone(),
            "--key".into(),
            key.into(),
            "--query".into(),
            "ContentLength".into(),
            "--output".into(),
            "text".into(),
        ];
        match self.aws_output(&args).await {
            Ok(output) => String::from_utf8_lossy(&output)
                .trim()
                .parse::<u64>()
                .map(Some)
                .map_err(|error| format!("S3_HEAD_INVALID: {error}")),
            Err(error)
                if error.contains("Not Found")
                    || error.contains("404")
                    || error.contains("NoSuchKey") =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    async fn put_object(&self, local: &Path, key: &str) -> Result<(), String> {
        self.aws(&[
            "s3api".into(),
            "put-object".into(),
            "--bucket".into(),
            self.bucket.clone(),
            "--key".into(),
            key.into(),
            "--body".into(),
            local.to_string_lossy().into_owned(),
            "--storage-class".into(),
            self.storage_class.clone(),
        ])
        .await
    }

    async fn upload_large_file<F>(
        &self,
        local: &Path,
        key: &str,
        size: u64,
        mut part_complete: F,
    ) -> Result<(), String>
    where
        F: FnMut(u64),
    {
        let chunk_size = multipart_chunk_size(size)?;
        let part_count = size.div_ceil(chunk_size);
        if part_count > MAX_MULTIPART_PARTS {
            return Err("S3_MULTIPART_CONFIGURATION_INVALID: trop de parties".into());
        }
        let created = self
            .aws_output(&[
                "s3api".into(),
                "create-multipart-upload".into(),
                "--bucket".into(),
                self.bucket.clone(),
                "--key".into(),
                key.into(),
                "--storage-class".into(),
                self.storage_class.clone(),
                "--output".into(),
                "json".into(),
            ])
            .await?;
        let upload_id = serde_json::from_slice::<serde_json::Value>(&created)
            .ok()
            .and_then(|value| value.get("UploadId")?.as_str().map(str::to_owned))
            .ok_or_else(|| "S3_MULTIPART_CREATE_INVALID: UploadId absent".to_owned())?;

        let result: Result<Vec<serde_json::Value>, String> = async {
            let mut completed = Vec::with_capacity(part_count as usize);
            for index in 0..part_count {
                let offset = index * chunk_size;
                let length = chunk_size.min(size - offset);
                let source = local.to_path_buf();
                let temporary = std::env::temp_dir().join(format!(
                    "vidioai-s3-part-{}-{}.tmp",
                    Uuid::new_v4(),
                    index + 1
                ));
                let temporary_for_copy = temporary.clone();
                task::spawn_blocking(move || -> Result<(), String> {
                    let mut input = std::fs::File::open(source).map_err(|e| e.to_string())?;
                    input
                        .seek(SeekFrom::Start(offset))
                        .map_err(|e| e.to_string())?;
                    let mut output =
                        std::fs::File::create(&temporary_for_copy).map_err(|e| e.to_string())?;
                    std::io::copy(&mut input.take(length), &mut output)
                        .map_err(|e| e.to_string())?;
                    output.flush().map_err(|e| e.to_string())
                })
                .await
                .map_err(|error| error.to_string())??;

                let part_number = (index + 1).to_string();
                let uploaded = self
                    .aws_output(&[
                        "s3api".into(),
                        "upload-part".into(),
                        "--bucket".into(),
                        self.bucket.clone(),
                        "--key".into(),
                        key.into(),
                        "--part-number".into(),
                        part_number.clone(),
                        "--upload-id".into(),
                        upload_id.clone(),
                        "--body".into(),
                        temporary.to_string_lossy().into_owned(),
                        "--output".into(),
                        "json".into(),
                    ])
                    .await;
                let _ = fs::remove_file(&temporary).await;
                let uploaded = uploaded?;
                let etag = serde_json::from_slice::<serde_json::Value>(&uploaded)
                    .ok()
                    .and_then(|value| value.get("ETag")?.as_str().map(str::to_owned))
                    .ok_or_else(|| "S3_MULTIPART_PART_INVALID: ETag absent".to_owned())?;
                completed.push(serde_json::json!({
                    "ETag": etag,
                    "PartNumber": index + 1,
                }));
                part_complete(length);
            }
            Ok(completed)
        }
        .await;

        let completed = match result {
            Ok(completed) => completed,
            Err(error) => {
                let _ = self
                    .aws(&[
                        "s3api".into(),
                        "abort-multipart-upload".into(),
                        "--bucket".into(),
                        self.bucket.clone(),
                        "--key".into(),
                        key.into(),
                        "--upload-id".into(),
                        upload_id,
                    ])
                    .await;
                return Err(error);
            }
        };

        let payload = serde_json::to_vec(&serde_json::json!({"Parts": completed}))
            .map_err(|error| error.to_string())?;
        let completion_file =
            std::env::temp_dir().join(format!("vidioai-s3-complete-{}.json.tmp", Uuid::new_v4()));
        fs::write(&completion_file, payload)
            .await
            .map_err(|error| error.to_string())?;
        let completion = self
            .aws(&[
                "s3api".into(),
                "complete-multipart-upload".into(),
                "--bucket".into(),
                self.bucket.clone(),
                "--key".into(),
                key.into(),
                "--upload-id".into(),
                upload_id,
                "--multipart-upload".into(),
                format!("file://{}", completion_file.to_string_lossy()),
            ])
            .await;
        let _ = fs::remove_file(completion_file).await;
        completion
    }

    async fn snapshot_files(local: &Path) -> Result<Vec<(PathBuf, SnapshotFile)>, String> {
        let root = local.to_path_buf();
        task::spawn_blocking(move || {
            let known_hashes = read_known_hashes(&root);
            let mut pending = vec![root.clone()];
            let mut files = Vec::new();
            while let Some(directory) = pending.pop() {
                for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
                    let entry = entry.map_err(|error| error.to_string())?;
                    let path = entry.path();
                    let metadata = entry.metadata().map_err(|error| error.to_string())?;
                    if metadata.is_dir() {
                        let relative = path
                            .strip_prefix(&root)
                            .map_err(|error| error.to_string())?;
                        if is_snapshot_file(relative) {
                            pending.push(path);
                        }
                    } else if metadata.is_file()
                        && let Ok(relative) = relative_snapshot_path(&root, &path)
                    {
                        let relative = relative.to_string_lossy().replace('\\', "/");
                        files.push((
                            path,
                            SnapshotFile {
                                sha256: known_hashes.get(&relative).cloned(),
                                path: relative,
                                size: metadata.len(),
                            },
                        ));
                    }
                }
            }
            files.sort_by(|left, right| left.1.path.cmp(&right.1.path));
            Ok(files)
        })
        .await
        .map_err(|error| error.to_string())?
    }

    fn emit_progress(
        callback: &Option<UploadProgressCallback>,
        progress: &UploadProgress,
        started: Instant,
    ) {
        if let Some(callback) = callback {
            let mut event = progress.clone();
            let elapsed = started.elapsed().as_secs_f64();
            if elapsed > 0.0 && event.bytes_transferred > 0 {
                let rate = (event.bytes_transferred as f64 / elapsed) as u64;
                event.bytes_per_second = Some(rate);
                if rate > 0 {
                    event.eta_seconds = Some(
                        event
                            .bytes_total
                            .saturating_sub(event.bytes_transferred)
                            .div_ceil(rate),
                    );
                }
            }
            callback(event);
        }
    }
}

fn read_known_hashes(root: &Path) -> HashMap<String, String> {
    let Ok(bytes) = std::fs::read(root.join("vidioai-model.json")) else {
        return HashMap::new();
    };
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return HashMap::new();
    };
    payload
        .get("files")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|item| {
            Some((
                item.get("path")?.as_str()?.to_owned(),
                item.get("sha256")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

fn read_snapshot_capabilities(root: &Path) -> Vec<String> {
    let Ok(bytes) = std::fs::read(root.join("vidioai-model.json")) else {
        return Vec::new();
    };
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    payload
        .get("capabilities")
        .or_else(|| payload.get("requested_capabilities"))
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect()
}

async fn sha256_file(path: PathBuf) -> Result<String, String> {
    task::spawn_blocking(move || {
        let mut input = std::fs::File::open(path).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = input.read(&mut buffer).map_err(|error| error.to_string())?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        Ok(format!("{:x}", digest.finalize()))
    })
    .await
    .map_err(|error| error.to_string())?
}

fn safe_manifest_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative);
    if !is_snapshot_file(relative) {
        return Err("S3_MANIFEST_PATH_INVALID: chemin de fichier invalide".into());
    }
    Ok(root.join(relative))
}

#[async_trait]
impl ObjectStorage for S3Storage {
    fn enabled(&self) -> bool {
        self.enabled
    }

    async fn health(&self) -> Result<(), String> {
        self.aws(&[
            "s3api".into(),
            "head-bucket".into(),
            "--bucket".into(),
            self.bucket.clone(),
        ])
        .await
    }

    async fn upload_file(&self, local: &Path, key: &str) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        let size = fs::metadata(local)
            .await
            .map_err(|error| error.to_string())?
            .len();
        if size > MIN_MULTIPART_CHUNK_SIZE {
            self.upload_large_file(local, key, size, |_| {}).await
        } else {
            self.put_object(local, key).await
        }
    }

    async fn restore_snapshot(
        &self,
        repository: &str,
        revision: &str,
        local: &Path,
        callback: Option<TransferProgressCallback>,
    ) -> Result<bool, String> {
        if !self.enabled {
            return Ok(false);
        }
        let prefix = model_s3_prefix(repository, revision)?;
        let manifest_key = format!("{prefix}/{SNAPSHOT_MANIFEST_NAME}");
        if self.head_size(&manifest_key).await?.is_none() {
            return Ok(false);
        }
        fs::create_dir_all(local)
            .await
            .map_err(|error| error.to_string())?;
        let manifest = self.read_manifest(&manifest_key).await?;
        if manifest.schema_version != SNAPSHOT_MANIFEST_SCHEMA_VERSION
            || manifest.repository != repository
            || manifest.revision != revision
        {
            return Err("S3_MANIFEST_INVALID: identité ou schéma inattendu".into());
        }
        let started = Instant::now();
        let mut progress = UploadProgress {
            direction: "download".into(),
            provider: "s3".into(),
            bytes_total: manifest.total_size,
            files_total: manifest.files.len() as u64,
            ..UploadProgress::default()
        };
        Self::emit_progress(&callback, &progress, started);
        for file in &manifest.files {
            progress.current_file = Some(file.path.clone());
            progress.current_file_size = file.size;
            progress.current_file_bytes = 0;
            let destination = safe_manifest_path(local, &file.path)?;
            let mut already_valid = fs::metadata(&destination)
                .await
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() == file.size);
            if already_valid && let Some(expected) = &file.sha256 {
                already_valid = sha256_file(destination.clone()).await? == *expected;
            }
            if !already_valid {
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)
                        .await
                        .map_err(|error| error.to_string())?;
                }
                let temporary = destination.with_extension(format!(
                    "{}.tmp",
                    destination
                        .extension()
                        .and_then(|value| value.to_str())
                        .unwrap_or("")
                ));
                self.aws(&[
                    "s3api".into(),
                    "get-object".into(),
                    "--bucket".into(),
                    self.bucket.clone(),
                    "--key".into(),
                    format!("{prefix}/{}", file.path),
                    temporary.to_string_lossy().into_owned(),
                ])
                .await?;
                let actual = fs::metadata(&temporary)
                    .await
                    .map_err(|error| error.to_string())?
                    .len();
                if actual != file.size {
                    let _ = fs::remove_file(&temporary).await;
                    return Err(format!("S3_SNAPSHOT_SIZE_MISMATCH: {}", file.path));
                }
                if let Some(expected) = &file.sha256
                    && sha256_file(temporary.clone()).await? != *expected
                {
                    let _ = fs::remove_file(&temporary).await;
                    return Err(format!("S3_SNAPSHOT_CHECKSUM_MISMATCH: {}", file.path));
                }
                fs::rename(temporary, destination)
                    .await
                    .map_err(|error| error.to_string())?;
                mark_transferred_bytes(&mut progress, file.size, file.size);
                progress.files_completed += 1;
            } else {
                mark_existing_file(&mut progress, file.size);
            }
            Self::emit_progress(&callback, &progress, started);
        }
        Ok(true)
    }

    async fn list_snapshots(&self) -> Result<Vec<SnapshotManifest>, String> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        let output = self
            .aws_output(&[
                "s3api".into(),
                "list-objects-v2".into(),
                "--bucket".into(),
                self.bucket.clone(),
                "--prefix".into(),
                "models/".into(),
                "--output".into(),
                "json".into(),
            ])
            .await?;
        let payload: serde_json::Value =
            serde_json::from_slice(&output).map_err(|error| format!("S3_LIST_INVALID: {error}"))?;
        let keys = payload
            .get("Contents")
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
            .filter_map(|value| value.get("Key")?.as_str())
            .filter(|key| key.ends_with(&format!("/{SNAPSHOT_MANIFEST_NAME}")))
            .map(str::to_owned)
            .take(200)
            .collect::<Vec<_>>();

        let mut manifests = Vec::new();
        for key in keys {
            let Ok(manifest) = self.read_manifest(&key).await else {
                continue;
            };
            let identity_valid = model_s3_prefix(&manifest.repository, &manifest.revision)
                .is_ok_and(|prefix| key == format!("{prefix}/{SNAPSHOT_MANIFEST_NAME}"));
            if manifest.schema_version != SNAPSHOT_MANIFEST_SCHEMA_VERSION || !identity_valid {
                continue;
            }
            let prefix = model_s3_prefix(&manifest.repository, &manifest.revision)?;
            let mut valid = !manifest.files.is_empty()
                && manifest.total_size == manifest.files.iter().map(|file| file.size).sum::<u64>();
            for file in &manifest.files {
                if safe_manifest_path(Path::new("snapshot"), &file.path).is_err()
                    || self.head_size(&format!("{prefix}/{}", file.path)).await? != Some(file.size)
                {
                    valid = false;
                    break;
                }
            }
            if valid {
                manifests.push(manifest);
            }
        }
        manifests.sort_by_key(|manifest| std::cmp::Reverse(manifest.created_at));
        Ok(manifests)
    }

    async fn upload_snapshot(
        &self,
        repository: &str,
        revision: &str,
        local: &Path,
        callback: Option<UploadProgressCallback>,
    ) -> Result<UploadOutcome, String> {
        let prefix = model_s3_prefix(repository, revision)?;
        let files = Self::snapshot_files(local).await?;
        let manifest = SnapshotManifest {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
            total_size: files.iter().map(|(_, file)| file.size).sum(),
            files: files.iter().map(|(_, file)| file.clone()).collect(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            schema_version: SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            capabilities: read_snapshot_capabilities(local),
        };
        let mut progress = UploadProgress {
            direction: "upload".into(),
            provider: "s3".into(),
            bytes_total: manifest.total_size,
            files_total: manifest.files.len() as u64,
            ..UploadProgress::default()
        };
        let started = Instant::now();
        Self::emit_progress(&callback, &progress, started);

        for (path, file) in &files {
            progress.current_file = Some(file.path.clone());
            progress.current_file_size = file.size;
            progress.current_file_bytes = 0;
            let key = format!("{prefix}/{}", file.path);
            if self.head_size(&key).await? == Some(file.size) {
                mark_existing_file(&mut progress, file.size);
                Self::emit_progress(&callback, &progress, started);
                continue;
            }
            if file.size > MIN_MULTIPART_CHUNK_SIZE {
                self.upload_large_file(path, &key, file.size, |sent| {
                    mark_transferred_bytes(&mut progress, sent, file.size);
                    Self::emit_progress(&callback, &progress, started);
                })
                .await?;
            } else {
                self.put_object(path, &key).await?;
                mark_transferred_bytes(&mut progress, file.size, file.size);
            }
            if self.head_size(&key).await? != Some(file.size) {
                return Err(format!("S3_UPLOAD_VALIDATION_FAILED: {}", file.path));
            }
            progress.files_completed += 1;
            Self::emit_progress(&callback, &progress, started);
        }

        // Le manifeste est la marque de commit du snapshot et part toujours en dernier.
        if !snapshot_upload_complete(&progress) {
            return Err(
                "S3_SNAPSHOT_INCOMPLETE: le manifeste ne peut pas être publié avant les données"
                    .into(),
            );
        }
        let manifest_path = local.join(".vidioai-snapshot-manifest.tmp");
        let manifest_bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
        fs::write(&manifest_path, &manifest_bytes)
            .await
            .map_err(|error| error.to_string())?;
        let manifest_key = format!("{prefix}/{SNAPSHOT_MANIFEST_NAME}");
        let manifest_result = self.put_object(&manifest_path, &manifest_key).await;
        let _ = fs::remove_file(manifest_path).await;
        manifest_result?;
        if self.head_size(&manifest_key).await? != Some(manifest_bytes.len() as u64) {
            return Err(
                "S3_MANIFEST_UPLOAD_VALIDATION_FAILED: manifest absent ou incomplet".into(),
            );
        }
        progress.bytes_transferred = progress.bytes_total;
        progress.current_file = None;
        progress.current_file_size = 0;
        progress.current_file_bytes = 0;
        Self::emit_progress(&callback, &progress, started);
        Ok(UploadOutcome {
            manifest,
            files_skipped: progress.files_skipped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_configuration_handles_500_mib_and_20_gib() {
        let size_500_mib = 500 * 1024 * 1024;
        let size_20_gib = 20 * 1024 * 1024 * 1024;
        assert_eq!(
            multipart_chunk_size(size_500_mib).unwrap(),
            128 * 1024 * 1024
        );
        let chunk = multipart_chunk_size(size_20_gib).unwrap();
        assert!(size_20_gib.div_ceil(chunk) <= MAX_MULTIPART_PARTS);
    }

    #[test]
    fn multipart_chunk_grows_for_very_large_files() {
        let size = 200 * 1024 * 1024 * 1024_u64;
        let chunk = multipart_chunk_size(size).unwrap();
        assert!(chunk > MIN_MULTIPART_CHUNK_SIZE);
        assert!(size.div_ceil(chunk) <= MULTIPART_HEADROOM_PARTS);
    }

    #[test]
    fn snapshot_filter_excludes_caches_and_temporary_files() {
        assert!(!is_snapshot_file(Path::new(
            ".cache/huggingface/trees/a.json"
        )));
        assert!(!is_snapshot_file(Path::new(
            "transformer/__pycache__/x.pyc"
        )));
        assert!(!is_snapshot_file(Path::new("download.tmp")));
        assert!(!is_snapshot_file(Path::new("tokenizer/config.lock")));
        assert!(is_snapshot_file(Path::new("transformer/model.safetensors")));
    }

    #[test]
    fn model_prefix_and_relative_path_never_duplicate_revision() {
        let root = Path::new("/models/storage/revision");
        let file = root.join("transformer/file.safetensors");
        let relative = relative_snapshot_path(root, &file).unwrap();
        let prefix = model_s3_prefix("owner/repo", "revision").unwrap();
        assert_eq!(
            format!("{prefix}/{}", relative.display()),
            "models/owner/repo/revision/transformer/file.safetensors"
        );
        assert!(!format!("{prefix}/{}", relative.display()).contains("revision/revision"));
    }

    #[test]
    fn upload_progress_is_bounded_and_final_is_exact() {
        let mut progress = UploadProgress {
            bytes_total: 42,
            bytes_transferred: 21,
            ..UploadProgress::default()
        };
        assert_eq!(progress.percent(), 50.0);
        progress.bytes_transferred = progress.bytes_total;
        assert_eq!(progress.percent(), 100.0);
        progress.bytes_transferred = 99;
        assert_eq!(progress.percent(), 100.0);
    }

    #[test]
    fn transferred_progress_is_monotone_and_never_exceeds_total() {
        let mut progress = UploadProgress {
            bytes_total: 500,
            current_file_size: 500,
            ..UploadProgress::default()
        };
        let mut observed = vec![progress.bytes_transferred];
        for bytes in [128, 128, 244, 1] {
            mark_transferred_bytes(&mut progress, bytes, 500);
            observed.push(progress.bytes_transferred);
        }
        assert!(observed.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(progress.bytes_transferred, progress.bytes_total);
        assert_eq!(progress.current_file_bytes, progress.current_file_size);
    }

    #[test]
    fn interrupted_upload_cannot_commit_manifest() {
        let progress = UploadProgress {
            bytes_total: 1_000,
            bytes_transferred: 900,
            files_total: 2,
            files_completed: 1,
            ..UploadProgress::default()
        };
        assert!(!snapshot_upload_complete(&progress));
    }

    #[test]
    fn already_valid_remote_file_is_counted_without_upload() {
        let mut progress = UploadProgress {
            bytes_total: 1_000,
            files_total: 2,
            ..UploadProgress::default()
        };
        mark_existing_file(&mut progress, 400);
        assert_eq!(progress.bytes_transferred, 400);
        assert_eq!(progress.files_completed, 1);
        assert_eq!(progress.files_skipped, 1);
        assert!(!snapshot_upload_complete(&progress));
        mark_existing_file(&mut progress, 600);
        assert!(snapshot_upload_complete(&progress));
    }

    #[tokio::test]
    async fn a_bucket_prefixed_with_itself_is_rejected_before_aws_is_called() {
        let storage = S3Storage {
            enabled: true,
            bucket: "vidioai-production/vidioai-production".into(),
            endpoint: None,
            storage_class: "STANDARD".into(),
        };
        let error = storage.health().await.unwrap_err();
        assert!(error.contains("sans schéma ni préfixe"));
    }
}

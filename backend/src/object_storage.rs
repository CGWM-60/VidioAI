//! Stockage objet S3-compatible.
//!
//! Cette implémentation pilote l'AWS CLI avec des arguments séparés (aucun
//! shell, aucun secret dans la commande). Elle fonctionne avec Scaleway Object
//! Storage grâce à `AWS_ENDPOINT_URL_S3` et conserve un backend local lorsque
//! S3 est explicitement désactivé.

use async_trait::async_trait;
use std::{path::Path, process::Stdio};
use tokio::process::Command;

#[async_trait]
pub trait ObjectStorage: Send + Sync {
    fn enabled(&self) -> bool;
    async fn health(&self) -> Result<(), String>;
    async fn upload_file(&self, local: &Path, key: &str) -> Result<(), String>;
    async fn download_prefix(&self, prefix: &str, local: &Path) -> Result<(), String>;
    async fn upload_prefix(&self, local: &Path, prefix: &str) -> Result<(), String>;
}

#[derive(Clone)]
pub struct S3Storage {
    enabled: bool,
    bucket: String,
    endpoint: Option<String>,
    storage_class: String,
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

    async fn aws(&self, args: &[&str]) -> Result<(), String> {
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
        let mut command = Command::new("aws");
        command
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env("AWS_METADATA_SERVICE_TIMEOUT", "1")
            .env("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1");
        if let Some(endpoint) = &self.endpoint {
            command.arg("--endpoint-url").arg(endpoint);
        }
        let output = command.output().await.map_err(|error| error.to_string())?;
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
        }
    }

    fn uri(&self, key: &str) -> String {
        format!("s3://{}/{}", self.bucket, key.trim_start_matches('/'))
    }
}

#[async_trait]
impl ObjectStorage for S3Storage {
    fn enabled(&self) -> bool {
        self.enabled
    }

    async fn health(&self) -> Result<(), String> {
        self.aws(&["s3api", "head-bucket", "--bucket", &self.bucket])
            .await
    }

    async fn upload_file(&self, local: &Path, key: &str) -> Result<(), String> {
        let local = local.to_string_lossy();
        let uri = self.uri(key);
        self.aws(&[
            "s3",
            "cp",
            &local,
            &uri,
            "--storage-class",
            &self.storage_class,
        ])
        .await
    }

    async fn download_prefix(&self, prefix: &str, local: &Path) -> Result<(), String> {
        let uri = self.uri(prefix);
        let local = local.to_string_lossy();
        self.aws(&["s3", "sync", &uri, &local]).await
    }

    async fn upload_prefix(&self, local: &Path, prefix: &str) -> Result<(), String> {
        let local = local.to_string_lossy();
        let uri = self.uri(prefix);
        self.aws(&[
            "s3",
            "sync",
            &local,
            &uri,
            "--storage-class",
            &self.storage_class,
        ])
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::S3Storage;

    #[tokio::test]
    async fn a_bucket_prefixed_with_itself_is_rejected_before_aws_is_called() {
        let storage = S3Storage {
            enabled: true,
            bucket: "vidioai-production/vidioai-production".into(),
            endpoint: None,
            storage_class: "STANDARD".into(),
        };
        let error = storage.aws(&["s3api", "head-bucket"]).await.unwrap_err();
        assert!(error.contains("sans schéma ni préfixe"));
    }
}

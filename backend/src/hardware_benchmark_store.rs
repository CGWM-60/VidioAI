//! Persistance des mesures matérielles réelles.
//!
//! Les benchmarks partagent le fichier SQLite des jobs mais possèdent leur
//! propre table. Une mesure reste attachée au modèle, à sa révision, au runtime
//! et à la précision : elle ne sera jamais réutilisée pour une autre variante.

use crate::hardware_estimator::HardwareBenchmark;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::PathBuf;

#[derive(Clone)]
pub struct HardwareBenchmarkStore {
    path: PathBuf,
}

impl HardwareBenchmarkStore {
    pub async fn open(path: PathBuf) -> Result<Self, String> {
        let store = Self { path };
        store
            .run(|connection| {
                connection.execute_batch(
                    "PRAGMA journal_mode=WAL;
                     PRAGMA synchronous=FULL;
                     CREATE TABLE IF NOT EXISTS hardware_benchmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        revision TEXT NOT NULL,
                        runtime TEXT NOT NULL,
                        precision TEXT NOT NULL,
                        measured_at INTEGER NOT NULL,
                        payload TEXT NOT NULL
                     );
                     CREATE INDEX IF NOT EXISTS idx_hardware_benchmark_model
                     ON hardware_benchmarks(model_id, revision, measured_at DESC);",
                )?;
                Ok(())
            })
            .await?;
        Ok(store)
    }

    async fn run<T, F>(&self, task: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> rusqlite::Result<T> + Send + 'static,
    {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let mut connection = Connection::open(path)?;
            task(&mut connection)
        })
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())
    }

    pub async fn record(&self, benchmark: &HardwareBenchmark) -> Result<(), String> {
        let benchmark = benchmark.clone();
        self.run(move |connection| {
            let payload = serde_json::to_string(&benchmark)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(error.into()))?;
            connection.execute(
                "INSERT INTO hardware_benchmarks(
                    model_id, revision, runtime, precision, measured_at, payload
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    benchmark.model_id,
                    benchmark.revision,
                    benchmark.runtime,
                    benchmark.precision,
                    benchmark.measured_at,
                    payload,
                ],
            )?;
            Ok(())
        })
        .await
    }

    /// La mesure la plus récente de la révision exacte prime. Une ancienne
    /// révision n'est pas extrapolée au nouveau checkpoint.
    pub async fn latest(
        &self,
        model_id: &str,
        revision: &str,
    ) -> Result<Option<HardwareBenchmark>, String> {
        let model_id = model_id.to_owned();
        let revision = revision.to_owned();
        self.run(move |connection| {
            let payload = connection
                .query_row(
                    "SELECT payload FROM hardware_benchmarks
                     WHERE model_id=?1 AND revision=?2
                     ORDER BY measured_at DESC, id DESC LIMIT 1",
                    params![model_id, revision],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            payload
                .map(|json| {
                    serde_json::from_str(&json).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            0,
                            rusqlite::types::Type::Text,
                            error.into(),
                        )
                    })
                })
                .transpose()
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn benchmark(measured_at: u64, peak: u64) -> HardwareBenchmark {
        HardwareBenchmark {
            model_id: "org/model".into(),
            revision: "abc".into(),
            gpu: "H100".into(),
            vram_idle_bytes: 1,
            vram_after_load_bytes: 2,
            vram_peak_bytes: peak,
            ram_peak_bytes: Some(4),
            runtime: "Diffusers".into(),
            precision: "BF16".into(),
            resolution_width: Some(1024),
            resolution_height: Some(1024),
            frames: None,
            duration_seconds: None,
            fps: None,
            batch: 1,
            attention_implementation: None,
            vae_tiling: false,
            cpu_offload: false,
            model_offload: false,
            inference_seconds: Some(3.0),
            measured_at,
        }
    }

    #[tokio::test]
    async fn latest_measurement_replaces_older_estimates() {
        let path = std::env::temp_dir().join(format!("vidioai-hardware-{}.sqlite", Uuid::new_v4()));
        let store = HardwareBenchmarkStore::open(path.clone()).await.unwrap();
        store.record(&benchmark(1, 10)).await.unwrap();
        store.record(&benchmark(2, 20)).await.unwrap();
        assert_eq!(
            store
                .latest("org/model", "abc")
                .await
                .unwrap()
                .unwrap()
                .vram_peak_bytes,
            20
        );
        let _ = std::fs::remove_file(path);
    }
}

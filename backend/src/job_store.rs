//! File de jobs durable SQLite.
//!
//! Une connexion courte est ouverte dans `spawn_blocking` pour ne jamais
//! bloquer l'exécuteur Tokio. Au redémarrage, les jobs actifs deviennent
//! `interrupted`, car leur processus d'inférence n'existe plus.

use crate::platform::{Job, JobStatus};
use rusqlite::{Connection, params};
use std::path::PathBuf;

#[derive(Clone)]
pub struct JobStore {
    path: PathBuf,
}

impl JobStore {
    pub async fn open(path: PathBuf) -> Result<Self, String> {
        let store = Self { path };
        store
            .run(|connection| {
                connection.execute_batch(
                    "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=FULL;
                 CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                 );",
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

    pub async fn upsert(&self, job: &Job) -> Result<(), String> {
        let job = job.clone();
        self.run(move |connection| {
            let payload = serde_json::to_string(&job)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(error.into()))?;
            connection.execute(
                "INSERT INTO jobs(id, payload, status, updated_at) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,
                 status=excluded.status, updated_at=excluded.updated_at",
                params![
                    job.id.to_string(),
                    payload,
                    format!("{:?}", job.status),
                    job.updated_at
                ],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn load_and_interrupt_active(&self) -> Result<Vec<Job>, String> {
        self.run(move |connection| {
            let transaction = connection.transaction()?;
            let mut jobs = Vec::new();
            {
                let mut statement =
                    transaction.prepare("SELECT payload FROM jobs ORDER BY updated_at")?;
                let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
                for row in rows {
                    let payload = row?;
                    if let Ok(mut job) = serde_json::from_str::<Job>(&payload) {
                        if matches!(job.status, JobStatus::Queued | JobStatus::Running) {
                            let was_queued = job.status == JobStatus::Queued;
                            job.status = if was_queued {
                                JobStatus::PendingRetry
                            } else {
                                JobStatus::Interrupted
                            };
                            job.stage = if was_queued {
                                "pending_retry"
                            } else {
                                "interrupted"
                            }
                            .to_owned();
                            job.message = if was_queued {
                                "Job en attente de relance après le redémarrage du backend."
                            } else {
                                "Job interrompu par un redémarrage du backend."
                            }
                            .to_owned();
                            job.updated_at = crate::platform::unix_now();
                            let updated = serde_json::to_string(&job).map_err(|error| {
                                rusqlite::Error::ToSqlConversionFailure(error.into())
                            })?;
                            transaction.execute(
                                "UPDATE jobs SET payload=?1, status=?2, updated_at=?3 WHERE id=?4",
                                params![
                                    updated,
                                    format!("{:?}", job.status),
                                    job.updated_at,
                                    job.id.to_string()
                                ],
                            )?;
                        }
                        jobs.push(job);
                    }
                }
            }
            transaction.commit()?;
            Ok(jobs)
        })
        .await
    }

    pub async fn ping(&self) -> bool {
        self.run(|connection| connection.query_row("SELECT 1", [], |_| Ok(())))
            .await
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::JobStore;
    use crate::platform::{Job, JobKind, JobStatus};
    use uuid::Uuid;

    #[tokio::test]
    async fn active_job_is_durably_marked_interrupted_after_restart() {
        let path = std::env::temp_dir().join(format!("vidioai-jobs-{}.sqlite", Uuid::new_v4()));
        let store = JobStore::open(path.clone()).await.expect("open sqlite");
        let job = Job {
            id: Uuid::new_v4(),
            kind: JobKind::GenerateImage,
            target_id: Uuid::new_v4().to_string(),
            status: JobStatus::Running,
            stage: "generating".into(),
            progress: 42,
            message: "active".into(),
            created_at: 1,
            updated_at: 2,
        };
        store.upsert(&job).await.expect("persist job");

        let reopened = JobStore::open(path.clone()).await.expect("reopen sqlite");
        let jobs = reopened
            .load_and_interrupt_active()
            .await
            .expect("restore jobs");
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, JobStatus::Interrupted);
        assert_eq!(jobs[0].stage, "interrupted");

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn queued_job_becomes_pending_retry_after_restart() {
        let path = std::env::temp_dir().join(format!("vidioai-retry-{}.sqlite", Uuid::new_v4()));
        let store = JobStore::open(path.clone()).await.expect("open sqlite");
        let job = Job {
            id: Uuid::new_v4(),
            kind: JobKind::InstallModel,
            target_id: "stable-image-core".into(),
            status: JobStatus::Queued,
            stage: "queued".into(),
            progress: 0,
            message: "queued".into(),
            created_at: 1,
            updated_at: 1,
        };
        store.upsert(&job).await.expect("persist queued job");
        let jobs = JobStore::open(path.clone())
            .await
            .expect("reopen sqlite")
            .load_and_interrupt_active()
            .await
            .expect("restore queued job");
        assert_eq!(jobs[0].status, JobStatus::PendingRetry);
        assert_eq!(jobs[0].stage, "pending_retry");
        let _ = std::fs::remove_file(path);
    }
}

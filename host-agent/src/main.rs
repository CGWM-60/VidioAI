//! Serveur HTTP natif du Host Agent VidioAI.

use anyhow::Context;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    routing::get,
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use vidioai_host_agent::{HostSnapshot, collect_snapshot};

#[derive(Clone)]
struct AgentState {
    token: Option<String>,
}

fn authorize(state: &AgentState, headers: &HeaderMap) -> Result<(), StatusCode> {
    let Some(expected) = state.token.as_deref() else {
        return Ok(());
    };
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let internal = headers
        .get("x-vidioai-host-token")
        .and_then(|value| value.to_str().ok());
    (bearer == Some(expected) || internal == Some(expected))
        .then_some(())
        .ok_or(StatusCode::UNAUTHORIZED)
}

async fn health(
    State(state): State<Arc<AgentState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    authorize(&state, &headers)?;
    Ok(Json(json!({
        "status": "ok",
        "service": "vidioai-host-agent",
        "version": env!("CARGO_PKG_VERSION")
    })))
}

async fn resources(
    State(state): State<Arc<AgentState>>,
    headers: HeaderMap,
) -> Result<Json<HostSnapshot>, StatusCode> {
    authorize(&state, &headers)?;
    tokio::task::spawn_blocking(collect_snapshot)
        .await
        .map(Json)
        .map_err(|error| {
            eprintln!("Collecte matérielle interrompue : {error}");
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bind_address =
        std::env::var("HOST_AGENT_BIND").unwrap_or_else(|_| "127.0.0.1:8091".to_owned());
    let token = std::env::var("HOST_AGENT_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let state = Arc::new(AgentState { token });
    let app = Router::new()
        .route("/health", get(health))
        .route("/system", get(resources))
        .route("/resources", get(resources))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind_address)
        .await
        .with_context(|| format!("impossible d'écouter sur {bind_address}"))?;

    println!("vidioai-host-agent écoute sur {bind_address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        } else {
            tokio::time::sleep(Duration::from_secs(u64::MAX)).await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
    println!("arrêt gracieux du Host Agent");
}

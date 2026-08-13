mod engine_ia;
mod execution_plan;
mod hardware_benchmark_store;
mod hardware_estimator;
mod host_agent;
mod huggingface_catalog;
mod job_store;
mod model_lab;
mod model_pack;
mod model_pack_registry;
mod object_storage;
mod platform;
mod utils;
mod worker;
use crate::engine_ia::ws_tchat::{charge_modele, text_messages_ia_ws};
use crate::platform::AppState;
use crate::utils::health;
use axum::http::{HeaderValue, Method, header};
use axum::{
    Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use tower_http::cors::{AllowOrigin, CorsLayer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Le premier démarrage charge (ou crée) la configuration persistante avant
    // d'ouvrir le port HTTP. Ainsi aucune route ne peut observer des dossiers
    // partiellement initialisés.
    let state = AppState::initialize()
        .await
        .map_err(|error| anyhow::anyhow!("initialisation VidioAI impossible : {error:?}"))?;
    let allowed_origins = [
        HeaderValue::from_static("http://localhost:3000"),
        HeaderValue::from_static("http://127.0.0.1:3000"),
        HeaderValue::from_static("http://localhost:3001"),
        HeaderValue::from_static("http://127.0.0.1:3001"),
    ];

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(allowed_origins))
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
        ])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    let app = Router::new()
        // Route historique conservée pour les clients existants.
        .route("/ws/tchat", get(ws_handler))
        // Route sous /api afin que le reverse proxy de production relaie aussi
        // correctement l'upgrade WebSocket.
        .route("/api/chat/stream", get(ws_handler))
        .nest("/healthcheck", health::register())
        // Toutes les étapes 4 à 12 sont regroupées dans un routeur à état partagé.
        .nest("/api", platform::router(state))
        .layer(cors);

    // Le port reste 8080 par défaut, mais un bind configurable permet les tests
    // isolés et les environnements PaaS sans modifier le binaire.
    let bind_address =
        std::env::var("VIDIOAI_BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0:8080".to_owned());
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    println!("Serveur VidioAI : {bind_address}");
    println!("WebSocket : ws://localhost:8080/ws/tchat");
    println!("Événements : ws://localhost:8080/api/events");

    // SIGTERM arrête d'abord l'acceptation de nouvelles connexions. Docker
    // laisse ensuite jusqu'à 45 secondes aux jobs et écritures atomiques pour
    // terminer avant de forcer l'arrêt du conteneur.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("gestionnaire Ctrl+C");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("gestionnaire SIGTERM")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
    println!("Arrêt gracieux demandé; fermeture du serveur HTTP.");
}
// ================================
// CONNEXION WEBSOCKET
// ================================

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

// ================================
// CLIENT CONNECTÉ
// ================================

async fn handle_socket(mut socket: WebSocket) {
    let nom = "Qwen/Qwen2.5-0.5B-Instruct".to_string();

    let model = match charge_modele(nom.clone()).await {
        Ok(model) => model,
        Err(error) => {
            // Un échec de chargement ne doit jamais faire paniquer la tâche de
            // connexion. Le client reçoit une erreur exploitable puis le flux
            // se ferme proprement.
            eprintln!("Chargement du modèle de chat impossible : {error}");
            let _ = socket
                .send(Message::Text(
                    "Le modèle de chat n'est pas disponible sur cette machine.".into(),
                ))
                .await;
            // Le retour détruit le socket et envoie la fermeture sans importer
            // une extension de sink uniquement pour ce chemin d'erreur.
            return;
        }
    };

    println!("Client connecté");

    while let Some(message) = socket.recv().await {
        let message = match message {
            Ok(message) => message,

            Err(error) => {
                println!("Erreur websocket : {error}");
                return;
            }
        };

        match message {
            Message::Text(text) => {
                println!("Question : {text}");

                let resultat = text_messages_ia_ws(&mut socket, &model, text.to_string()).await;

                if let Err(error) = resultat {
                    println!("Erreur IA : {error}");
                }
            }

            Message::Close(_) => {
                println!("Client déconnecté");

                break;
            }

            _ => {}
        }
    }
}

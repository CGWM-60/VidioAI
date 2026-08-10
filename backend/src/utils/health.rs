use axum::{Router, http::StatusCode, response::IntoResponse, routing::get};

pub fn register() -> Router {
    Router::new().route("/", get(health_check))
}

pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK).into_response()
}

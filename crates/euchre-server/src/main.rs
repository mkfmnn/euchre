//! Binary entry point: serve one Euchre table over websockets.
//!
//! Listens on `EUCHRE_ADDR` (default `127.0.0.1:8080`) with a single `/ws`
//! route. Connect with the bundled example client:
//!
//! ```text
//! cargo run -p euchre-server
//! cargo run -p euchre-server --example cli_client
//! ```

use euchre_engine::GameConfig;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "euchre_server=info".into()),
        )
        .init();

    let addr = std::env::var("EUCHRE_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!(
        "euchre-server listening on ws://{}/ws",
        listener.local_addr()?
    );

    euchre_server::serve(listener, GameConfig::default()).await
}

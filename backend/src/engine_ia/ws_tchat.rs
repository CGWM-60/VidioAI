use axum::extract::ws::{Message, WebSocket};

use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, Model, ModelBuilder, Response,
    TextMessageRole, TextMessages,
};
// dans le OK on passe la variable que l'on veux envoyer . dans anyhow::Result<Model> on passe l'object du result
pub async fn charge_modele(modele: String) -> anyhow::Result<Model> {
    println!("chargement du modele");

    let model = ModelBuilder::new(modele).build().await?;

    println!("chargement modele termine");

    Ok(model)
}
// &Model est une copie qaund on dois passe un object on fais une copie
pub async fn text_messages_ia_ws(
    socket: &mut WebSocket,
    model: &Model,
    msg: String,
) -> anyhow::Result<()> {
    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "Tu es un assistant francophone.
            Réponds clairement et directement.
            Évite les répétitions.
            Si tu as répondu à la question, termine ta réponse.
            Ne répète jamais une phrase ou une idée plusieurs fois.",
        )
        .add_message(TextMessageRole::User, msg);

    let mut ia_stream = model.stream_chat_request(messages).await?;

    while let Some(item) = ia_stream.next().await {
        if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = item
            && let Some(ChunkChoice {
                delta:
                    Delta {
                        content: Some(text),
                        ..
                    },
                ..
            }) = choices.first()
        {
            print!("{text}");

            socket.send(Message::Text(text.clone().into())).await?;
        }
    }

    println!();

    Ok(())
}

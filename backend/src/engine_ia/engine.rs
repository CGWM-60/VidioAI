use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, ModelBuilder, Response, TextMessageRole,
    TextMessages,
};

pub async fn text_messages_ia(
    modele: String,
    msg: String,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("Chargement du modèle...");

    let model = ModelBuilder::new(modele)
        //.with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    println!("Modèle chargé !");

    let messages = TextMessages::new().add_message(TextMessageRole::User, msg);

    let mut stream = model.stream_chat_request(messages).await?;

    let mut reponse = String::new();

    while let Some(item) = stream.next().await {
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

            reponse.push_str(text);
        }
    }

    // println!();

    Ok(reponse)
}

use agentic_flow_lib::llm_client::{LLMClient, OllamaModel};
use agentic_flow_lib::model::ChatMessage;

#[tokio::test]
async fn test_ollama_chat_completion_gemma() {
    let client = LLMClient::from_ollama(OllamaModel::Gemma2_2b);
    let messages = vec![ChatMessage::user("Hello, who are you?".to_string())];
    let tools = vec![];
    
    let result = client.chat_completions(messages, tools).await;

    assert!(
        result.is_ok(),
        "Ollama chat completion failed: {:?}",
        result
    );
    assert!(!result.unwrap().message().content.is_empty());
}

#[tokio::test]
async fn test_ollama_text_completion_gemma() {
    let client = LLMClient::from_ollama(OllamaModel::Gemma2_2b);
    let prompt = "Say hello in one word.".to_string();

    let result = client.completion(prompt).await;

    assert!(
        result.is_ok(),
        "Ollama text completion failed: {:?}",
        result
    );
    assert!(!result.unwrap().response().is_empty());
}

use std::sync::Arc;

use agentic_flow_lib::{
    errors::AgenticFlowError,
    llm_client::LLMProvider,
    model::{ChatMessage, ChatResponse, OllamaResponse},
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use tokio::sync::Mutex;

pub struct MockLLMProvider {
    response: Arc<Mutex<Option<ChatMessage>>>,
}

impl MockLLMProvider {
    pub fn new() -> Self {
        Self {
            response: Arc::new(Mutex::new(Some(ChatMessage::assistant("".to_string())))),
        }
    }

    pub async fn set_response(&self, resp: Option<ChatMessage>) {
        self.response
            .lock()
            .await
            .replace(resp.unwrap_or_else(|| ChatMessage::assistant("".to_string())));
    }
}

#[async_trait]
impl LLMProvider for MockLLMProvider {
    fn http_client(&self) -> &Client {
        unimplemented!("Mock model does not have an HTTP client")
    }

    fn base_url(&self) -> &str {
        unimplemented!("Mock model does not have a base URL")
    }

    fn model(&self) -> &str {
        unimplemented!("Mock model does not have a model name")
    }

    async fn chat_completions(
        &self,
        _messages: Vec<ChatMessage>,
        _temperature: f32,
        _tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        let response = self.response.lock().await.take();
        let chat_message = response.unwrap_or(ChatMessage::assistant("".to_string()));
        Ok(Box::new(OllamaResponse {
            message: chat_message,
        }) as Box<dyn ChatResponse>)
    }
}
use agentic_flow_lib::{
    errors::AgenticFlowError,
    llm_client::LLMProvider,
    model::{
        ChatMessage, ChatResponse, OllamaCompletionResponse, OllamaResponse,
    },
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

pub struct MockLLMProvider {
    chat_response: OllamaResponse,
    completion_response: OllamaCompletionResponse,
}

impl MockLLMProvider {
    pub fn new() -> Self {
        Self {
            chat_response: OllamaResponse::default(),
            completion_response: OllamaCompletionResponse {
                response: "".to_string(),
            },
        }
    }

    pub async fn with_completion_response(mut self, resp: Option<String>) -> Self {
        self.completion_response = OllamaCompletionResponse {
            response: resp.unwrap_or_else(|| "".to_string()),
        };
        self
    }

    pub async fn with_chat_response(mut self, resp: Option<ChatMessage>) -> Self {
        self.chat_response = OllamaResponse {
            message: resp.unwrap_or_else(|| ChatMessage::assistant("".to_string())),
        };
        self
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

    async fn chat_completions(
        &self,
        _messages: Vec<ChatMessage>,
        _temperature: f32,
        _tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        Ok(Box::new(self.chat_response.clone()))
    }

    async fn completion(
        &self,
        _prompt: String,
        _temperature: f32,
    ) -> Result<Box<dyn agentic_flow_lib::model::CompletionResponse>, AgenticFlowError> {
        Ok(Box::new(self.completion_response.clone()))
    }
}

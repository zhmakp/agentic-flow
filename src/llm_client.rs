use std::sync::Arc;

use async_trait::async_trait;
use reqwest::{Client as HttpClient, Response};
use serde_json::{Value, json};

use crate::{
    errors::AgenticFlowError,
    model::{ChatResponse, OllamaResponse, OpenRouterResponse},
};

#[derive(Debug, Clone)]
pub enum OllamaModel {
    GPToss,
    Gemma,
    Qwen3_8B,
}

impl OllamaModel {
    pub fn to_string(&self) -> &'static str {
        match self {
            OllamaModel::GPToss => "gpt-oss:20b",
            OllamaModel::Gemma => "gemma3:4b",
            OllamaModel::Qwen3_8B => "qwen3:8b",
        }
    }
}

#[derive(Debug, Clone)]
pub enum OpenRouterModel {
    Flash2,
    GPTMini,
}

impl OpenRouterModel {
    pub fn to_string(&self) -> &'static str {
        match self {
            OpenRouterModel::Flash2 => "google/gemini-2.0-flash-001",
            OpenRouterModel::GPTMini => "openai/gpt-4o-mini",
        }
    }
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn http_client(&self) -> &reqwest::Client;

    fn base_url(&self) -> &str;

    fn model(&self) -> &str;

    fn api_key(&self) -> Option<String> {
        None
    }

    async fn chat_completions(
        &self,
        body: Value,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError>;

    async fn send_request(
        &self,
        body: Value,
        tools: Vec<Value>,
        endpoint: &str,
    ) -> Result<Response, AgenticFlowError> {
        let url = format!("{}/{}", self.base_url(), endpoint);
        let response = self
            .http_client()
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key().unwrap_or_default()),
            )
            .json(&json!({
                "model": self.model(),
                "messages": body,
                "stream": false,
                "tools": tools
            }))
            .send()
            .await
            .map_err(|e| {
                AgenticFlowError::NetworkError(format!("Failed to send request: {}", e))
            })?;

        if response.status().is_success() {
            Ok(response)
        } else {
            Err(AgenticFlowError::ApiClientError(format!(
                "API request failed with status: {} {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }
}

struct OllamaProvider {
    client: HttpClient,
    base_url: String,
    model: &'static str,
}

impl OllamaProvider {
    pub fn new(base_url: String, model: OllamaModel) -> Self {
        Self {
            base_url,
            client: HttpClient::new(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn http_client(&self) -> &HttpClient {
        &self.client
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat_completions(
        &self,
        body: Value,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        let response = self.send_request(body, tools, "api/chat").await?;

        let response_text = response.text().await.unwrap();
        serde_json::from_str::<OllamaResponse>(&response_text)
            .map_err(|e| AgenticFlowError::ParseError(format!("Failed to parse response: {}", e)))
            .map(|res| Box::new(res) as Box<dyn ChatResponse>)
    }
}

struct OpenRouterProvider {
    client: HttpClient,
    base_url: &'static str,
    model: &'static str,
}

impl OpenRouterProvider {
    pub fn new(model: OpenRouterModel) -> Self {
        Self {
            client: HttpClient::new(),
            base_url: "https://openrouter.ai/api/v1",
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenRouterProvider {
    fn http_client(&self) -> &HttpClient {
        &self.client
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn api_key(&self) -> Option<String> {
        match std::env::var("OPENROUTER_API_KEY") {
            Ok(key) => Some(key),
            Err(_) => {
                println!("WARNING: OPENROUTER_API_KEY is not set in environment variables.");
                None
            }
        }
    }

    async fn chat_completions(
        &self,
        body: Value,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        let response = self.send_request(body, tools, "chat/completions").await?;

        let response_text = response.text().await.unwrap();
        serde_json::from_str::<OpenRouterResponse>(&response_text)
            .map_err(|e| AgenticFlowError::ParseError(format!("Failed to parse response: {}", e)))
            .map(|res| Box::new(res) as Box<dyn ChatResponse>)
    }
}

#[derive(Clone)]
pub struct LLMClient {
    inner: Arc<dyn LLMProvider>,
}

impl LLMClient {
    pub fn from_ollama(model: OllamaModel) -> Self {
        Self {
            inner: Arc::new(OllamaProvider::new(
                "http://localhost:11434".to_string(),
                model,
            )),
        }
    }

    pub fn from_open_router(model: OpenRouterModel) -> Self {
        Self {
            inner: Arc::new(OpenRouterProvider::new(model)),
        }
    }

    pub async fn chat_completions(
        &self,
        body: Value,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        self.inner.chat_completions(body, tools).await
    }
}

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::{Client as HttpClient, Response};
use serde_json::{Value, json};

use crate::{errors::AgenticFlowError, model::*};

#[derive(Debug, Clone)]
pub enum OllamaModel {
    GPToss,
    Gemma2_2b,
    Gemma3_4b,
    Qwen3_8B,
    Custom(String),
}

impl OllamaModel {
    pub fn to_string(&self) -> String {
        match self {
            OllamaModel::GPToss => "gpt-oss:20b".to_string(),
            OllamaModel::Gemma2_2b => "gemma2:2b".to_string(),
            OllamaModel::Gemma3_4b => "gemma3:4b".to_string(),
            OllamaModel::Qwen3_8B => "qwen3:8b".to_string(),
            OllamaModel::Custom(name) => name.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OpenRouterModel {
    Flash2,
    GPTMini,
    Custom(String),
}

impl OpenRouterModel {
    pub fn to_string(&self) -> String{
        match self {
            OpenRouterModel::Flash2 => "google/gemini-2.0-flash-001".to_string(),
            OpenRouterModel::GPTMini => "openai/gpt-4o-mini".to_string(),
            OpenRouterModel::Custom(name) => name.clone(),
        }
    }
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn http_client(&self) -> &reqwest::Client;

    fn base_url(&self) -> &str;

    fn api_key(&self) -> Option<String> {
        None
    }

    async fn completion(
        &self,
        prompt: String,
        temperature: f32,
    ) -> Result<Box<dyn CompletionResponse>, AgenticFlowError>;

    async fn chat_completions(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError>;

    async fn send_request(
        &self,
        request: Value,
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
            .json(&request)
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
    model: String,
}

impl OllamaProvider {
    pub fn new(model: OllamaModel) -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
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

    async fn chat_completions(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        let req = ChatCompletionRequest {
            model: self.model.to_string(),
            messages,
            temperature,
            stream: false,
            tools,
        };
        let response = self.send_request(json!(req), "api/chat").await?;

        let response_text = response.text().await.unwrap();
        serde_json::from_str::<OllamaResponse>(&response_text)
            .map_err(|e| AgenticFlowError::ParseError(format!("Failed to parse response: {}", e)))
            .map(|res| Box::new(res) as Box<dyn ChatResponse>)
    }

    async fn completion(
        &self,
        prompt: String,
        temperature: f32,
    ) -> Result<Box<dyn CompletionResponse>, AgenticFlowError> {
        let request = CompletionRequest {
            model: self.model.to_string(),
            prompt: prompt,
            max_tokens: None,
            temperature: Some(temperature),
            stream: Some(false),
        };
        let response = self.send_request(json!(request), "api/generate").await?;

        let response_text = response.text().await.unwrap();
        serde_json::from_str::<OllamaCompletionResponse>(&response_text)
            .map_err(|e| AgenticFlowError::ParseError(format!("Failed to parse response: {}", e)))
            .map(|res| Box::new(res) as Box<dyn CompletionResponse>)
    }
}

struct OpenRouterProvider {
    client: HttpClient,
    base_url: &'static str,
    model: String,
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
        messages: Vec<ChatMessage>,
        temperature: f32,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        let req = ChatCompletionRequest {
            model: self.model.to_string(),
            messages,
            temperature,
            stream: false,
            tools,
        };
        let response = self.send_request(json!(req), "chat/completions").await?;

        let response_text = response.text().await.unwrap();
        serde_json::from_str::<OpenRouterResponse>(&response_text)
            .map_err(|e| AgenticFlowError::ParseError(format!("Failed to parse response: {}", e)))
            .map(|res| Box::new(res) as Box<dyn ChatResponse>)
    }

    async fn completion(
        &self,
        prompt: String,
        temperature: f32,
    ) -> Result<Box<dyn CompletionResponse>, AgenticFlowError> {
        let request = CompletionRequest {
            model: self.model.to_string(),
            prompt,
            max_tokens: None,
            temperature: Some(temperature),
            stream: Some(false),
        };
        let response = self.send_request(json!(request), "completions").await?;

        let response_text = response.text().await.unwrap();
        serde_json::from_str::<OpenRouterCompletionResponse>(&response_text)
            .map_err(|e| AgenticFlowError::ParseError(format!("Failed to parse response: {}", e)))
            .map(|res| Box::new(res) as Box<dyn CompletionResponse>)
    }
}

#[derive(Clone)]
pub struct LLMClient {
    inner: Arc<dyn LLMProvider>,
    temperature: f32,
}

impl Default for LLMClient {
    fn default() -> Self {
        Self::from_ollama(OllamaModel::Qwen3_8B)
    }
}

impl LLMClient {
    pub fn from_ollama(model: OllamaModel) -> Self {
        Self {
            inner: Arc::new(OllamaProvider::new(model)),
            temperature: 0.7,
        }
    }

    pub fn from_open_router(model: OpenRouterModel) -> Self {
        Self {
            inner: Arc::new(OpenRouterProvider::new(model)),
            temperature: 0.7,
        }
    }

    pub fn from<T>(provider: T) -> Self
    where
        T: LLMProvider + 'static,
    {
        Self {
            inner: Arc::new(provider),
            temperature: 0.7,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub async fn chat_completions(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        self.inner
            .chat_completions(messages, self.temperature, tools)
            .await
    }

    pub async fn completion(
        &self,
        prompt: String,
    ) -> Result<Box<dyn CompletionResponse>, AgenticFlowError> {
        self.inner.completion(prompt, self.temperature).await
    }
}

use std::fmt::Debug;

use serde::{ Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub stream: bool,
    pub tools: Vec<Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub function: Function,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage{
    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
            thinking: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            thinking: None,
            tool_calls: None,
        }
    }

    pub fn system(content: String) -> Self {
        Self {
            role: "system".to_string(),
            content,
            thinking: None,
            tool_calls: None,
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OllamaResponse {
    pub message: ChatMessage,
}

impl Default for OllamaResponse {
    fn default() -> Self {
        Self {
            message: ChatMessage::assistant("".to_string()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenRouterChoice {
    message: ChatMessage,
    finish_reason: String,
}

pub trait ChatResponse: Send + Sync + Debug {
    fn message(&self) -> &ChatMessage;
}

impl ChatResponse for OpenRouterResponse {
    fn message(&self) -> &ChatMessage {
        &self.choices[0].message
    }
}

impl ChatResponse for OllamaResponse {
    fn message(&self) -> &ChatMessage {
        &self.message
    }
}

// Completions takes a prompt input instead of a series of messages
#[derive(Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OpenRouterCompletionResponse {
    pub id: String,
    pub choices: Vec<CompletionChoice>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OllamaCompletionResponse {
    pub response: String,
}

pub trait CompletionResponse: Send + Sync + Debug {
    fn response(&self) -> &str;
}

impl CompletionResponse for OpenRouterCompletionResponse {
    fn response(&self) -> &str {
        &self.choices[0].text
    }
}

impl CompletionResponse for OllamaCompletionResponse {
    fn response(&self) -> &str {
        &self.response
    }
}

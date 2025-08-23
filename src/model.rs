use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

#[derive(Serialize, Deserialize)]
pub struct ToolCall {
    pub function: Function,
}

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
pub struct OllamaResponse {
    pub message: ChatMessage,
}

#[derive(Serialize, Deserialize)]
pub struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Serialize, Deserialize)]
struct OpenRouterChoice {
    message: ChatMessage,
    finish_reason: String,
}

pub trait ChatResponse: Send + Sync {
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
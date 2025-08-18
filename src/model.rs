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
}

#[derive(Serialize, Deserialize)]
pub struct OllamaResponse {
    pub message: ChatMessage,
}


// {
//   "id": "gen-12345",
//   "choices": [
//     {
//       "message": {
//         "role": "assistant",
//         "content": "The meaning of life is a complex and subjective question...",
//         "refusal": ""
//       },
//       "logprobs": {},
//       "finish_reason": "stop",
//       "index": 0
//     }
//   ],
//   "provider": "OpenAI",
//   "model": "openai/gpt-3.5-turbo",
//   "object": "chat.completion",
//   "created": 1735317796,
//   "system_fingerprint": {},
//   "usage": {
//     "prompt_tokens": 14,
//     "completion_tokens": 163,
//     "total_tokens": 177
//   }
// }

#[derive(Serialize, Deserialize)]
pub struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Serialize, Deserialize)]
struct OpenRouterChoice {
    message: ChatMessage,
    finish_reason: String,
}

pub trait ChatResponse {
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
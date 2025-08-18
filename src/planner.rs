use std::{fmt::Debug, sync::Arc, vec};

use tokio::sync::Mutex;

use serde_json::Value;

use crate::{
    errors::AgenticFlowError,
    llm_client::{LLMClient, OllamaModel},
    model::ChatMessage,
    tool_registry::ToolRegistry,
};

pub struct PlanStep {
    pub tool_name: String,
    pub params: Value,
}

#[async_trait::async_trait]
pub trait Executor: Send + Sync {
    async fn execute(&self, steps: Vec<PlanStep>) -> Result<String, AgenticFlowError>;
}

#[async_trait::async_trait]
pub trait Planner: Send + Sync {
    fn tool_registry(&self) -> Arc<Mutex<ToolRegistry>>;
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError>;
}

pub struct MultiStepPlanner {
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl MultiStepPlanner {
    pub fn new(tool_registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self { tool_registry }
    }
}

#[async_trait::async_trait]
impl Planner for MultiStepPlanner {
    fn tool_registry(&self) -> Arc<Mutex<ToolRegistry>> {
        self.tool_registry.clone()
    }

    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError> {
        let messages = vec![
            ChatMessage::system("Analyze the task and create a multi-step plan.".to_string()),
            ChatMessage::user(task.to_string()),
        ];

        let tools = self.tool_registry.lock().await.get_tools_for_planner();

        LLMClient::from_ollama(OllamaModel::Qwen3_8B)
            .chat_completions(messages, tools)
            .await
            .map(|response| {
                let message = response.message();
                message
                    .tool_calls
                    .iter()
                    .flat_map(|f| {
                        f.iter().map(|t| PlanStep {
                            tool_name: t.function.name.clone(),
                            params: t.function.arguments.clone(),
                        })
                    })
                    .collect()
            })
    }
}

use std::{sync::Arc, vec};

use tokio::sync::Mutex;

use serde_json::Value;

use crate::{
    errors::AgenticFlowError,
    llm_client::LLMClient,
    model::{ChatMessage, ToolCall},
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
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError>;
}

pub struct MultiStepPlanner {
    llm_client: LLMClient,
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl MultiStepPlanner {
    pub fn new(llm_client: LLMClient, tool_registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self {
            llm_client,
            tool_registry,
        }
    }
}

#[async_trait::async_trait]
impl Planner for MultiStepPlanner {
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError> {
        let messages = vec![
            ChatMessage::system("Analyze the task and create a multi-step plan.".to_string()),
            ChatMessage::user(task.to_string()),
        ];

        let tools = self.tool_registry.lock().await.get_tools_for_planner();

        self.llm_client
            .chat_completions(messages, tools)
            .await
            .map(|response| {
                let message = response.message();
                collect_as_plan_steps(&message.tool_calls)
            })
    }
}

impl From<&ToolCall> for PlanStep {
    fn from(tool_call: &ToolCall) -> Self {
        PlanStep {
            tool_name: tool_call.function.name.clone(),
            params: tool_call.function.arguments.clone(),
        }
    }
}

fn collect_as_plan_steps(tool_calls: &Option<Vec<ToolCall>>) -> Vec<PlanStep> {
    tool_calls
        .iter()
        .flat_map(|f| f.into_iter().map(|tool_call| tool_call.into()))
        .collect()
}

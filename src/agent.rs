use serde_json::{json};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::errors::AgenticFlowError;
use crate::llm_client::LLMClient;
use crate::mcp_manager::MCPManager;
use crate::model::{ChatMessage, ChatResponse};
use crate::planner::{Executor, PlanStep};
use crate::tool_registry::{ExecutionContext, ToolRegistry};

pub struct Agent {
    manager: Arc<Mutex<MCPManager>>,
    tool_registry: Arc<Mutex<ToolRegistry>>,
    llm_client: LLMClient,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub timeout_seconds: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            timeout_seconds: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub content: String,
    pub tools_used: Vec<String>,
    pub execution_time_ms: u64,
}

impl Agent {
    pub fn new(
        manager: Arc<Mutex<MCPManager>>,
        tool_registry: Arc<Mutex<ToolRegistry>>,
        llm_client: LLMClient,
    ) -> Self {
        Self {
            manager,
            tool_registry,
            llm_client,
        }
    }

    pub async fn execute_tool(
        &self,
        tool_name: &str,
        params: serde_json::Value,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value, AgenticFlowError> {
        let manager = self.manager.lock().await;
        let tool_registry = self.tool_registry.lock().await;

        tool_registry
            .execute_tool(tool_name, params, &*manager, context)
            .await
    }

    async fn call_llm(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<Box<dyn ChatResponse>, AgenticFlowError> {
        self.llm_client.chat_completions(messages, vec![]).await
    }
}

#[async_trait::async_trait]
impl Executor for Agent {
    async fn execute(&self, steps: Vec<PlanStep>) -> Result<String, AgenticFlowError> {
        let mut context = ExecutionContext::new();
        let mut step = 1;

        for PlanStep { tool_name, params } in steps {
            let result = self
                .execute_tool(&tool_name, params, &mut context)
                .await
                .unwrap();
            context.set(format!("{}: {}", step, tool_name), result);
            step += 1;
        }

        self.call_llm(vec![
            ChatMessage::system("Synthesize the following context into result".to_string()),
            ChatMessage::user(format!("Context: {}", json!(context.data()))),
        ]).await.map(|res| res.message().content.to_string())
    }
}

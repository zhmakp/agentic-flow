use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::errors::AgenticFlowError;
use crate::llm_client::LLMClient;
use crate::mcp_manager::MCPManager;
use crate::model::ChatMessage;
use crate::tool_registry::{ExecutionContext, ToolRegistry};

pub struct TodoAgent {
    manager: Arc<Mutex<MCPManager>>,
    tool_registry: Arc<Mutex<ToolRegistry>>,
    llm_client: LLMClient,
    config: AgentConfig,
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

impl TodoAgent {
    pub fn new(
        manager: Arc<Mutex<MCPManager>>,
        tool_registry: Arc<Mutex<ToolRegistry>>,
        llm_client: LLMClient,
        config: AgentConfig,
    ) -> Self {
        Self {
            manager,
            tool_registry,
            llm_client,
            config: config,
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

    pub async fn process_request(&self, input: &str) -> Result<AgentResponse, AgenticFlowError> {
        let start_time = std::time::Instant::now();
        let mut context = ExecutionContext::new();
        context.set("original_instruction".to_string(), serde_json::json!(input));

        let tool_registry = self.tool_registry.lock().await;
        let messages = vec![
            ChatMessage::system("Process the following instruction and use available tools if necessary.".to_string()),
            ChatMessage::user(input.to_string()),
        ];
        let result = self.llm_client.chat_completions(messages, tool_registry.get_tools_for_planner()).await?;

        // For now, return a simple response
        // This would be expanded to include the full planning loop from the existing agent
        let response = AgentResponse {
            content: format!("Processed: {}", json!(result.message())),
            tools_used: vec![],
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(response)
    }

    pub async fn get_available_tools(&self) -> Vec<String> {
        self.tool_registry.lock().await.get_tools_names()
    }
}

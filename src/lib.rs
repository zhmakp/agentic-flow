pub mod agent;
pub mod errors;
pub mod llm_client;
pub mod mcp_manager;
pub mod model;
pub mod planner;
pub mod tool_registry;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use agent::{AgentConfig, AgentResponse, TodoAgent};
use errors::AgenticFlowError;
use llm_client::LLMClient;
use mcp_manager::{MCPConfig, MCPManager};
use tool_registry::ToolRegistry;

use crate::{llm_client::OllamaModel, tool_registry::LocalTool};

#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub mcp_config: MCPConfig,
    pub enabled_servers: Vec<String>,
    pub llm_config: LLMConfig,
    pub agent_config: AgentConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            mcp_config: MCPConfig {
                servers: HashMap::new(),
            },
            enabled_servers: vec![],
            llm_config: LLMConfig {
                model: OllamaModel::GPToss.to_string(),
            },
            agent_config: AgentConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub model: &'static str,
}

pub struct AgenticSystem {
    manager: Arc<Mutex<MCPManager>>,
    agent: TodoAgent,
    system_config: SystemConfig,
    llm_client: LLMClient,
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl AgenticSystem {
    pub async fn new(
        config: SystemConfig,
        tools: Vec<Box<dyn LocalTool>>,
    ) -> Result<Self, AgenticFlowError> {
        // 1. Initialize MCP Manager
        let mut manager = MCPManager::new(config.mcp_config.clone());

        // 2. Start configured MCP servers
        for server_name in &config.enabled_servers {
            manager.start_server(server_name).await?;
        }

        let manager = Arc::new(Mutex::new(manager));

        // 3. Initialize Tool Registry
        let tool_registry = Arc::new(Mutex::new(ToolRegistry::new()));

        // Note: Local tools would be registered here
        for tool in tools {
            tool_registry.lock().await.register_local_tool(tool);
        }

        // Refresh MCP tools from active servers
        tool_registry
            .lock()
            .await
            .refresh_mcp_tools(&*manager.lock().await)
            .await?;

        // 4. Initialize LLM Client
        let llm_client = LLMClient::from_ollama(OllamaModel::GPToss);

        // 5. Create Agent
        let agent = TodoAgent::new(
            manager.clone(),
            tool_registry.clone(),
            llm_client.clone(),
            config.agent_config.clone(),
        );

        Ok(Self {
            manager,
            agent,
            system_config: config,
            llm_client: llm_client,
            tool_registry: tool_registry,
        })
    }

    pub async fn plan_and_execute(&self, task: &str) -> Result<String, AgenticFlowError> {
        let agent = TodoAgent::new(
            self.manager.clone(),
            self.tool_registry.clone(),
            self.llm_client.clone(),
            self.system_config.agent_config.clone(),
        );
        let planner = planner::Planner::new(agent, self.tool_registry.clone());
        planner.plan_and_execute(task).await
    }

    pub async fn process_user_request(
        &self,
        request: &str,
    ) -> Result<AgentResponse, AgenticFlowError> {
        self.agent.process_request(request).await
    }

    pub async fn get_available_tools(&self) -> Vec<String> {
        self.agent.get_available_tools().await
    }

    pub async fn shutdown(&self) -> Result<(), AgenticFlowError> {
        let mut manager = self.manager.lock().await;
        for server_name in manager.get_active_server_names().clone() {
            manager.stop_server(&server_name).await?;
        }
        Ok(())
    }
}

// Example configuration helper
impl SystemConfig {
    pub fn example() -> Self {
        let mut servers = HashMap::new();
        // servers.insert("web_search".to_string(), mcp_manager::ServerConfig {
        //     server_type: mcp_manager::ServerType::Python,
        //     module_name: Some("mcp_server_brave_search".to_string()),
        //     package_name: None,
        //     auto_install: false,
        //     config: None,
        // });

        Self {
            mcp_config: MCPConfig { servers },
            enabled_servers: vec![],
            llm_config: LLMConfig {
                model: OllamaModel::GPToss.to_string(),
            },
            agent_config: AgentConfig::default(),
        }
    }
}

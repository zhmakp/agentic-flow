pub mod agent;
pub mod config;
pub mod errors;
pub mod llm_client;
pub mod mcp_manager;
pub mod model;
pub mod planner;
pub mod tool_registry;
pub mod worker;

use std::sync::Arc;
use tokio::sync::Mutex;

use agent::Agent;
use errors::AgenticFlowError;
use llm_client::LLMClient;
use mcp_manager::MCPManager;
use tool_registry::ToolRegistry;

use crate::{
    config::SystemConfig,
    planner::{Executor, MultiStepPlanner, Planner},
    tool_registry::LocalTool,
};

pub struct AgenticSystem {
    manager: Arc<Mutex<MCPManager>>,
    agent: Box<dyn Executor>,
    tool_registry: Arc<Mutex<ToolRegistry>>,
    planner: Box<dyn Planner>,
    system_config: SystemConfig,
}

impl AgenticSystem {
    pub async fn new(
        config: SystemConfig,
        tools: Vec<Box<dyn LocalTool>>,
        llm_client: LLMClient,
    ) -> Result<Self, AgenticFlowError> {
        let manager = Self::initialize_mcp_manager(&config).await?;
        let tool_registry = Self::initialize_tool_registry(tools, &manager).await?;

        let agent = Box::new(Agent::new(
            manager.clone(),
            tool_registry.clone(),
            llm_client.clone(),
        ));

        let planner = Box::new(MultiStepPlanner::new(
            llm_client.clone(),
            tool_registry.clone(),
        ));

        Ok(Self {
            manager,
            agent,
            system_config: config,
            tool_registry,
            planner,
        })
    }

    async fn initialize_mcp_manager(
        config: &SystemConfig,
    ) -> Result<Arc<Mutex<MCPManager>>, AgenticFlowError> {
        let mut manager = MCPManager::new(config.mcp_config.clone());

        for server_name in config.mcp_config.servers.keys() {
            manager.start_server(server_name).await?;
        }

        Ok(Arc::new(Mutex::new(manager)))
    }

    async fn initialize_tool_registry(
        tools: Vec<Box<dyn LocalTool>>,
        manager: &Arc<Mutex<MCPManager>>,
    ) -> Result<Arc<Mutex<ToolRegistry>>, AgenticFlowError> {
        let tool_registry = Arc::new(Mutex::new(ToolRegistry::new()));

        for tool in tools {
            tool_registry.lock().await.register_local_tool(tool);
        }

        tool_registry
            .lock()
            .await
            .refresh_mcp_tools(&*manager.lock().await)
            .await?;

        Ok(tool_registry)
    }

    /// Plans and executes a complex task
    pub async fn plan_and_execute(&self, task: &str) -> Result<String, AgenticFlowError> {
        let steps = self.planner.plan(task).await?;
        self.agent.execute(steps).await
    }

    /// Returns available tools
    pub async fn get_available_tools(&self) -> Vec<String> {
        self.tool_registry.lock().await.get_tools_names()
    }

    /// Gracefully shuts down the system
    pub async fn shutdown(self) -> Result<(), AgenticFlowError> {
        let mut manager = self.manager.lock().await;
        for server_name in manager.get_active_server_names().clone() {
            manager.stop_server(&server_name).await?;
        }
        Ok(())
    }
}

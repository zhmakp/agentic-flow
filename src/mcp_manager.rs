use rmcp::{
    RoleClient, ServiceExt,
    service::RunningService,
    transport::{ConfigureCommandExt, TokioChildProcess},
};

use std::collections::HashMap;
use tokio::process::Command;

use crate::{
    config::{MCPConfig, ServerType},
    errors::AgenticFlowError,
};

#[derive(Debug, Clone)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub server_name: String,
}

pub struct MCPManager {
    active_servers: HashMap<String, RunningService<RoleClient, ()>>,
    config: MCPConfig,
}

impl MCPManager {
    pub fn new(config: MCPConfig) -> Self {
        Self {
            active_servers: HashMap::new(),
            config,
        }
    }

    pub async fn start_server(&mut self, server_name: &str) -> Result<(), AgenticFlowError> {
        let server_config = self.config.servers.get(server_name).ok_or_else(|| {
            AgenticFlowError::ToolError(format!("Server config not found: {}", server_name))
        })?;

        let service = match server_config.server_type {
            ServerType::Python => {
                let module_name = server_config.module_name.as_ref().ok_or_else(|| {
                    AgenticFlowError::ToolError("Python module name required".to_string())
                })?;
                ().serve(
                    TokioChildProcess::new(Command::new("python").configure(|cmd| {
                        cmd.arg("-m").arg(module_name);
                    }))
                    .map_err(|e| {
                        AgenticFlowError::ToolError(format!("Failed to start Python server: {}", e))
                    })?,
                )
                .await
            }
            ServerType::Node => {
                let package_name = server_config.package_name.as_ref().ok_or_else(|| {
                    AgenticFlowError::ToolError("Node package name required".to_string())
                })?;
                ().serve(
                    TokioChildProcess::new(Command::new("npx").configure(|cmd| {
                        cmd.arg("-y").arg(package_name);
                    }))
                    .map_err(|e| {
                        AgenticFlowError::ToolError(format!("Failed to start Node server: {}", e))
                    })?,
                )
                .await
            }
        }
        .unwrap();

        self.active_servers.insert(server_name.to_string(), service);

        Ok(())
    }

    pub async fn stop_server(&mut self, server_name: &str) -> Result<(), AgenticFlowError> {
        if let Some(service) = self.active_servers.remove(server_name) {
            service.cancel().await.map_err(|e| {
                AgenticFlowError::ToolError(format!(
                    "Failed to stop server '{}': {}",
                    server_name, e
                ))
            })?;
        }
        Ok(())
    }

    pub async fn get_server_tools(
        &self,
        server_name: &str,
    ) -> Result<Vec<MCPTool>, AgenticFlowError> {
        let service = self
            .active_servers
            .get(server_name)
            .ok_or(AgenticFlowError::ServerNotFound)?;

        if let Ok(tools) = service.list_tools(Default::default()).await {
            Ok(tools
                .tools
                .into_iter()
                .map(|tool| MCPTool {
                    name: tool.name.clone().to_string(),
                    description: tool.description.clone().unwrap_or_default().to_string(),
                    input_schema: tool.schema_as_json_value(),
                    server_name: server_name.to_string(),
                })
                .collect())
        } else {
            Err(AgenticFlowError::ToolError(
                "Failed to list tools".to_string(),
            ))
        }
    }

    pub fn get_active_server_names(&self) -> Vec<String> {
        self.active_servers.keys().cloned().collect()
    }

    pub fn get_server_connection(
        &self,
        server_name: &str,
    ) -> Option<&RunningService<RoleClient, ()>> {
        self.active_servers.get(server_name)
    }
}

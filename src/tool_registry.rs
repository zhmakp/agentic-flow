use async_trait::async_trait;
use rmcp::model::CallToolRequestParam;
use serde_json::Value;
use std::collections::HashMap;

use crate::errors::AgenticFlowError;
use crate::mcp_manager::MCPManager;

#[async_trait]
pub trait LocalTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameter_schema(&self) -> serde_json::Value;
    async fn execute(
        &self,
        params: serde_json::Value,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value, AgenticFlowError>;
}

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    data: HashMap<String, serde_json::Value>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }

    pub fn set(&mut self, key: String, value: serde_json::Value) {
        self.data.insert(key, value);
    }

    pub fn data(&self) -> &HashMap<String, serde_json::Value> {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct MCPToolDescriptor {
    pub server_name: String,
    pub tool_name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ToolDescriptor {
    Local {
        name: String,
        description: String,
        schema: serde_json::Value,
    },
    MCP {
        name: String,
        description: String,
        schema: serde_json::Value,
        #[serde(skip_serializing)]
        server_name: String,
    },
}

pub struct ToolRegistry {
    local_tools: HashMap<String, Box<dyn LocalTool>>,
    mcp_tool_map: HashMap<String, MCPToolDescriptor>,
    available_tools: Vec<ToolDescriptor>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            local_tools: HashMap::new(),
            mcp_tool_map: HashMap::new(),
            available_tools: Vec::new(),
        }
    }

    pub fn register_local_tool(&mut self, tool: Box<dyn LocalTool>) {
        let name = tool.name().to_string();
        let descriptor = ToolDescriptor::Local {
            name: name.clone(),
            description: tool.description().to_string(),
            schema: tool.parameter_schema(),
        };

        self.local_tools.insert(name, tool);
        self.available_tools.push(descriptor);
    }

    pub async fn refresh_mcp_tools(
        &mut self,
        manager: &MCPManager,
    ) -> Result<(), AgenticFlowError> {
        // Clear existing MCP tools
        self.mcp_tool_map.clear();
        self.available_tools
            .retain(|t| matches!(t, ToolDescriptor::Local { .. }));

        // Discover tools from each active server
        for server_name in manager.get_active_server_names() {
            let tools = manager.get_server_tools(&server_name).await?;

            for tool in tools {
                let tool_name = tool.name.clone();

                // Create MCP tool descriptor
                let mcp_descriptor = MCPToolDescriptor {
                    server_name: server_name.clone(),
                    tool_name: tool_name.clone(),
                    description: tool.description.clone(),
                    input_schema: tool.input_schema.clone(),
                };

                // Map tool name to server (handles conflicts)
                let final_tool_name = if self.mcp_tool_map.contains_key(&tool_name) {
                    format!("{}::{}", server_name, tool_name) // Namespace conflicts
                } else {
                    tool_name.clone()
                };

                self.mcp_tool_map
                    .insert(final_tool_name.clone(), mcp_descriptor);

                // Add to available tools for planner
                self.available_tools.push(ToolDescriptor::MCP {
                    name: final_tool_name,
                    description: tool.description,
                    schema: tool.input_schema,
                    server_name: server_name.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn get_tools_names(&self) -> Vec<String> {
        self.available_tools
            .iter()
            .map(|t| match t {
                ToolDescriptor::Local { name, .. } => name.clone(),
                ToolDescriptor::MCP { name, .. } => name.clone(),
            })
            .collect()
    }

    pub fn get_tools_for_planner(&self) -> Vec<Value> {
        self.available_tools
            .iter()
            .map(|t| match t {
                ToolDescriptor::Local {
                    name,
                    description,
                    schema,
                } => (name, description, schema),
                ToolDescriptor::MCP {
                    name,
                    description,
                    schema,
                    ..
                } => (name, description, schema),
            })
            .map(|(name, description, schema)| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": schema
                    }
                })
            })
            .collect()
    }

    pub async fn execute_tool(
        &self,
        tool_name: &str,
        params: serde_json::Value,
        manager: &MCPManager,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value, AgenticFlowError> {
        // 1. Check if it's a local tool
        if let Some(local_tool) = self.local_tools.get(tool_name) {
            return local_tool.execute(params, context).await;
        }

        // 2. Check if it's an MCP tool
        if let Some(mcp_descriptor) = self.mcp_tool_map.get(tool_name) {
            return self.execute_mcp_tool(mcp_descriptor, params, manager).await;
        }

        Err(AgenticFlowError::ToolError(format!(
            "Tool '{}' not found",
            tool_name
        )))
    }

    async fn execute_mcp_tool(
        &self,
        descriptor: &MCPToolDescriptor,
        params: serde_json::Value,
        manager: &MCPManager,
    ) -> Result<serde_json::Value, AgenticFlowError> {
        let connection = manager
            .get_server_connection(&descriptor.server_name)
            .ok_or(AgenticFlowError::ServerNotFound)?;

        let result = connection
            .service
            .call_tool(CallToolRequestParam {
                name: descriptor.tool_name.clone().into(),
                arguments: params.as_object().cloned(),
            })
            .await
            .map_err(|e| {
                AgenticFlowError::ToolError(format!(
                    "Failed to call MCP tool '{}': {}",
                    descriptor.tool_name, e
                ))
            })?;

        Ok(result.structured_content.unwrap_or_default())
    }
}

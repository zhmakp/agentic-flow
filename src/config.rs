use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{agent::AgentConfig, llm_client::OllamaModel};

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
            mcp_config: MCPConfig::default(),
            enabled_servers: vec![],
            llm_config: LLMConfig::default(),
            agent_config: AgentConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MCPConfig {
    pub servers: HashMap<String, ServerConfig>,
}

impl Default for MCPConfig {
    fn default() -> Self {
        Self {
            servers: HashMap::new(),
        }
    }
    
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ServerType {
    Python,
    Node,
    // TODO: Docker or Docker Toolkit
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub server_type: ServerType,
    pub module_name: Option<String>,
    pub package_name: Option<String>,
    pub auto_install: bool,
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub model: &'static str,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model: OllamaModel::GPToss.to_string(),
        }
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

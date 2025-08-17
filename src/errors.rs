#[derive(Debug, Clone)]
pub enum AgenticFlowError {
    PlanningError(String),
    ToolError(String),
    ApiClientError(String),
    ParseError(String),
    NetworkError(String),
    ServerNotFound
}

impl std::fmt::Display for AgenticFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgenticFlowError::PlanningError(msg) => write!(f, "Planning error: {}", msg),
            AgenticFlowError::ToolError(msg) => write!(f, "Tool error: {}", msg),
            AgenticFlowError::ApiClientError(msg) => write!(f, "API client error: {}", msg),
            AgenticFlowError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            AgenticFlowError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            AgenticFlowError::ServerNotFound => write!(f, "Server not found"),
        }
    }
}

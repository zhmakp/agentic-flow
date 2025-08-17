use agentic_flow_lib::{
    errors::AgenticFlowError,
    tool_registry::{ExecutionContext, LocalTool},
};
use serde_json::json;

pub struct MockTool;

#[async_trait::async_trait]
impl LocalTool for MockTool {
    fn name(&self) -> &str {
        "mock_tool"
    }

    fn description(&self) -> &str {
        "Mock tool for testing"
    }

    fn parameter_schema(&self) -> serde_json::Value {
        json!({})
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value, AgenticFlowError> {
        context.set(
            "step_1".to_string(),
            json!({
                "tool": "MockTool",
                "success": true
            }),
        );
        Ok(json!({"result": "Say phrase 'test successful step 1'", "params": params}))
    }
}
pub struct MockToolFollowUp;

#[async_trait::async_trait]
impl LocalTool for MockToolFollowUp {
    fn name(&self) -> &str {
        "mock_tool_follow_up"
    }

    fn description(&self) -> &str {
        "Mock tool follow up for testing"
    }

    fn parameter_schema(&self) -> serde_json::Value {
        json!({})
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value, AgenticFlowError> {
        context.set(
            "step_1".to_string(),
            json!({
                "tool": "MockTool",
                "success": true
            }),
        );
        Ok(json!({"result": "Say phrase 'test successful step 2'", "params": params}))
    }
}

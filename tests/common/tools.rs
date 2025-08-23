use agentic_flow_lib::{
    errors::AgenticFlowError,
    tool_registry::{ExecutionContext, LocalTool},
};
use serde_json::{json, Value};

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
        json!({"type": "object", "properties": {"foo": {"type": "string"}}})
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

pub struct EchoTool;

#[async_trait::async_trait]
impl LocalTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn parameter_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string"
                }
            },
            "required": ["text"]
        })
    }
    fn description(&self) -> &str {
        "Echoes the input text"
    }

    async fn execute(&self, params: Value, context: &mut ExecutionContext) -> Result<Value, AgenticFlowError> {
        let text = params.get("text").and_then(Value::as_str).ok_or_else(|| {
            AgenticFlowError::ToolError("text".to_string())
        })?;
        context.set("echoed_text".to_string(), json!(text));
        Ok(json!({"text": text}))
    }
}

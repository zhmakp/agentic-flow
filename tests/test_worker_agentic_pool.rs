mod common;

use agentic_flow_lib::model::Function;
use agentic_flow_lib::model::ToolCall;
use serde_json::json;
use std::{str, sync::Arc};
use tokio::sync::Mutex;

use agentic_flow_lib::{
    agent::Agent, config::MCPConfig, errors::AgenticFlowError, llm_client::LLMClient,
    mcp_manager::MCPManager, model::ChatMessage, planner::PlanStep, tool_registry::ToolRegistry,
    worker::AgenticTaskPool,
};

use crate::common::llm_provider::MockLLMProvider;
use crate::common::tools::EchoTool;

async fn make_mock_agent(response: Option<ChatMessage>) -> Arc<Mutex<Agent>> {
    // Change these as needed—it assumes your types implement Default.
    let manager = MCPManager::new(MCPConfig::default());
    let dummy_manager = Arc::new(Mutex::new(manager));

    let mut tool_registry = ToolRegistry::new();
    tool_registry.register_local_tool(Box::new(EchoTool));
    let dummy_tool_registry = Arc::new(Mutex::new(tool_registry));

    let provider = MockLLMProvider::new();
    provider.set_response(response).await;

    let dummy_llm_client = LLMClient::from(provider);

    Arc::new(Mutex::new(Agent::new(
        dummy_manager,
        dummy_tool_registry,
        dummy_llm_client,
    )))
}

fn make_tool_call(text: &str) -> ToolCall {
    ToolCall {
        function: Function {
            name: "echo".to_string(),
            arguments: json!({"text": text}),
        },
    }
}

#[tokio::test]
async fn test_agentic_task_pool_new_and_shutdown() -> Result<(), AgenticFlowError> {
    let agent = make_mock_agent(None).await;
    let pool = AgenticTaskPool::new(4, agent);
    assert_eq!(pool.worker_count(), 4);
    assert!(pool.is_active());
    pool.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_agentic_task_pool_execute_step() -> Result<(), AgenticFlowError> {
    let response = ChatMessage::assistant("hello, world!".to_string())
        .with_tool_calls(vec![make_tool_call("hello, world!")]);
    let agent = make_mock_agent(Some(response)).await;
    let pool = AgenticTaskPool::new(2, agent.clone());

    // Create a simple echo step (the mock agent should return the input parameters)
    let step = PlanStep {
        tool_name: "echo".to_string(),
        params: json!({"text": "hello, world!"}),
    };

    let result = pool.execute_step(step).await?;
    // For an echo tool the result should equal the input parameters.
    assert_eq!(result, json!({"text": "hello, world!"}));

    pool.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_agentic_task_pool_execute_parallel() -> Result<(), AgenticFlowError> {
    let response = ChatMessage::assistant("hello, world!".to_string()).with_tool_calls(vec![
        make_tool_call("one"),
        make_tool_call("two"),
        make_tool_call("three"),
    ]);
    let agent = make_mock_agent(Some(response)).await;
    let pool = AgenticTaskPool::new(3, agent.clone());

    let steps = vec![
        PlanStep {
            tool_name: "echo".to_string(),
            params: json!({"text": "one"}),
        },
        PlanStep {
            tool_name: "echo".to_string(),
            params: json!({"text": "two"}),
        },
        PlanStep {
            tool_name: "echo".to_string(),
            params: json!({"text": "three"}),
        },
    ];

    let results = pool.execute_parallel(steps).await?;
    assert_eq!(results.len(), 3);
    // Check that each result equals the corresponding parameters.
    assert_eq!(results[0], json!({"text": "one"}));
    assert_eq!(results[1], json!({"text": "two"}));
    assert_eq!(results[2], json!({"text": "three"}));

    pool.shutdown().await?;
    Ok(())
}

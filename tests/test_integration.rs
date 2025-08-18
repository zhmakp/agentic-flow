mod common;

use agentic_flow_lib::{
    AgenticSystem, 
    config::SystemConfig, 
    llm_client::LLMClient, 
    tool_registry::LocalTool,
};

use common::tools::{MockTool, MockToolFollowUp};

#[tokio::test]
async fn test_available_tools() {
    let tools = vec![Box::new(MockTool) as Box<dyn LocalTool>];
    let agentic_system = AgenticSystem::new(SystemConfig::example(), tools, LLMClient::default())
        .await
        .unwrap();

    let result = agentic_system.get_available_tools().await;

    assert!(result.contains(&"mock_tool".to_string()));
}

#[tokio::test]
async fn test_plan_and_execute() {
    let tools = vec![
        Box::new(MockTool) as Box<dyn LocalTool>,
        Box::new(MockToolFollowUp) as Box<dyn LocalTool>,
    ];
    let agentic_system = AgenticSystem::new(SystemConfig::example(), tools, LLMClient::default())
        .await
        .unwrap();

    let result = agentic_system
        .plan_and_execute("execute mocking tool and follow up")
        .await
        .unwrap();

    assert!(result.contains("test successful step 1"));
    assert!(result.contains("test successful step 2"));
}

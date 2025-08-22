mod common;

use std::sync::Arc;
use tokio::sync::Mutex;

use agentic_flow_lib::llm_client::{
    LLMClient, 
};
use agentic_flow_lib::planner::{
    ChainOfThoughtPlanner, HTNPlanner, MonteCarloTreeSearchPlanner, MultiStepPlanner, PlanStep,
    Planner,
};
use common::tools::{MockTool};
use agentic_flow_lib::tool_registry::ToolRegistry;

fn make_llm_client() -> LLMClient {
    LLMClient::default()
}

fn make_tool_registry() -> Arc<Mutex<ToolRegistry>> {
    let mut registry = ToolRegistry::new();
    registry.register_local_tool(Box::new(MockTool));
    Arc::new(Mutex::new(registry))
}

#[tokio::test]
async fn test_multistep_planner() {
    let planner = MultiStepPlanner::new(make_llm_client(), make_tool_registry());
    let steps = planner.plan("test task with bar param").await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_name, "mock_tool");
    assert_eq!(steps[0].params["foo"], "bar");
}

#[tokio::test]
async fn test_chain_of_thought_planner() {
    let planner = ChainOfThoughtPlanner::new(make_llm_client(), make_tool_registry());
    let steps = planner.plan("test task with bar param").await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_name, "mock_tool");
    assert_eq!(steps[0].params["foo"], "bar");
}

#[tokio::test]
async fn test_htn_planner() {
    let planner = HTNPlanner::new(make_llm_client(), make_tool_registry());
    let steps = planner.plan("test task with bar param").await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_name, "mock_tool");
    assert_eq!(steps[0].params["foo"], "bar");
}

#[tokio::test]
async fn test_mcts_planner() {
    let planner = MonteCarloTreeSearchPlanner::new(make_llm_client(), make_tool_registry(), 3);
    let steps = planner.plan("test task with bar param").await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_name, "mock_tool");
    assert_eq!(steps[0].params["foo"], "bar");
}


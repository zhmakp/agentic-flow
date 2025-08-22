use core::fmt;
use std::{sync::Arc, vec};

use serde::de;
use tokio::sync::Mutex;

use serde_json::Value;

use crate::{
    errors::AgenticFlowError,
    llm_client::LLMClient,
    model::{ChatMessage, ToolCall},
    tool_registry::ToolRegistry,
};

pub struct PlanStep {
    pub tool_name: String,
    pub params: Value,
}

impl fmt::Debug for PlanStep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlanStep {{ tool_name: {}, params: {} }}", self.tool_name, self.params)
    }
}

#[async_trait::async_trait]
pub trait Executor: Send + Sync {
    async fn execute(&self, steps: Vec<PlanStep>) -> Result<String, AgenticFlowError>;
}

#[async_trait::async_trait]
pub trait Planner: Send + Sync {
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError>;
}

pub struct MultiStepPlanner {
    llm_client: LLMClient,
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl MultiStepPlanner {
    pub fn new(llm_client: LLMClient, tool_registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self {
            llm_client,
            tool_registry,
        }
    }
}

#[async_trait::async_trait]
impl Planner for MultiStepPlanner {
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError> {
        let messages = vec![
            ChatMessage::system("Analyze the task and create a multi-step plan.".to_string()),
            ChatMessage::user(task.to_string()),
        ];

        let tools = self.tool_registry.lock().await.get_tools_for_planner();

        self.llm_client
            .chat_completions(messages, tools)
            .await
            .map(|response| {
                let message = response.message();
                collect_as_plan_steps(&message.tool_calls)
            })
    }
}

impl From<&ToolCall> for PlanStep {
    fn from(tool_call: &ToolCall) -> Self {
        PlanStep {
            tool_name: tool_call.function.name.clone(),
            params: tool_call.function.arguments.clone(),
        }
    }
}

fn collect_as_plan_steps(tool_calls: &Option<Vec<ToolCall>>) -> Vec<PlanStep> {
    tool_calls
        .iter()
        .flat_map(|f| f.into_iter().map(|tool_call| tool_call.into()))
        .collect()
}
pub struct ChainOfThoughtPlanner {
    llm_client: LLMClient,
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl ChainOfThoughtPlanner {
    pub fn new(llm_client: LLMClient, tool_registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self {
            llm_client,
            tool_registry,
        }
    }
}

#[async_trait::async_trait]
impl Planner for ChainOfThoughtPlanner {
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError> {
        // Step 1: Ask the LLM for a detailed chain of thought.
        let chain_messages = vec![
            ChatMessage::system("Provide a detailed chain-of-thought analysis before forming a plan.".to_string()),
            ChatMessage::user(format!("Task: {}\nChain-of-Thought:", task)),
        ];
        let chain_response = self.llm_client
            .chat_completions(chain_messages, vec![])
            .await?;
        let chain_thought = &chain_response.message().content;
        
        // Step 2: Use the chain-of-thought to generate a multi-step plan.
        let plan_prompt = format!(
            "Based on the following chain-of-thought, generate a multi-step plan with tool calls in JSON format.\n\nChain-of-Thought:\n{}\n\nPlan:",
            chain_thought
        );
        let plan_messages = vec![
            ChatMessage::system("Generate a multi-step plan using the provided chain-of-thought.".to_string()),
            ChatMessage::user(plan_prompt),
        ];
        let tools = self.tool_registry.lock().await.get_tools_for_planner();
        let plan_response = self.llm_client
            .chat_completions(plan_messages, tools)
            .await?;
        
        let tool_calls = &plan_response.message().tool_calls;
        Ok(collect_as_plan_steps(tool_calls))
    }
}

pub struct HTNPlanner {
    llm_client: LLMClient,
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl HTNPlanner {
    pub fn new(llm_client: LLMClient, tool_registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self {
            llm_client,
            tool_registry,
        }
    }
}

#[async_trait::async_trait]
impl Planner for HTNPlanner {
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError> {
        // Step 1: Decompose the task into high-level subtasks
        let decompose_messages = vec![
            ChatMessage::system("You are an HTN planner. Decompose the high-level task into logical subtasks.".to_string()),
            ChatMessage::user(format!("Task: {}\nDecompose this into a hierarchy of subtasks:", task)),
        ];
        let decompose_response = self.llm_client
            .chat_completions(decompose_messages, vec![])
            .await?;
        let hierarchy = &decompose_response.message().content;
        
        // Step 2: Refine each subtask into primitive actions (tool calls)
        let refine_messages = vec![
            ChatMessage::system("Based on the task hierarchy, generate a concrete execution plan using available tools.".to_string()),
            ChatMessage::user(format!(
                "Task: {}\n\nTask Hierarchy:\n{}\n\nGenerate a detailed plan using tool calls that implements this hierarchy:",
                task, hierarchy
            )),
        ];
        
        let tools = self.tool_registry.lock().await.get_tools_for_planner();
        let plan_response = self.llm_client
            .chat_completions(refine_messages, tools)
            .await?;

        let tool_calls = &plan_response.message().tool_calls;
        Ok(collect_as_plan_steps(tool_calls))
    }
}

#[derive(Clone)]
pub struct MonteCarloTreeSearchPlanner {
    llm_client: LLMClient,
    tool_registry: Arc<Mutex<ToolRegistry>>,
    simulations: usize,
}

impl MonteCarloTreeSearchPlanner {
    pub fn new(
        llm_client: LLMClient,
        tool_registry: Arc<Mutex<ToolRegistry>>,
        simulations: usize,
    ) -> Self {
        Self {
            llm_client,
            tool_registry,
            simulations,
        }
    }
}

#[async_trait::async_trait]
impl Planner for MonteCarloTreeSearchPlanner {
    async fn plan(&self, task: &str) -> Result<Vec<PlanStep>, AgenticFlowError> {
        // Initialize MCTS parameters.
        let mut best_plan = Vec::new();
        let mut best_score = f64::MIN;

        let tools = self.tool_registry.lock().await.get_tools_for_planner();
        let llm_client = self.llm_client.clone().with_temperature(0.9);
        // Perform multiple simulations.
        for _ in 0..self.simulations {
            // Use the LLM to simulate a plan for a given task.
            let simulation_messages = vec![
                ChatMessage::system("Simulate a potential plan for task execution using Monte Carlo Tree Search.".to_string()),
                ChatMessage::user(format!("Task: {}", task)),
            ];

            let simulation_response = llm_client
                .chat_completions(simulation_messages, tools.clone())
                .await?;

            let tool_calls = &simulation_response.message().tool_calls;
            let plan_steps = collect_as_plan_steps(tool_calls);

            // Evaluate the simulated plan using a simple heuristic:
            // Here, a shorter plan is considered more efficient.
            let score = if plan_steps.is_empty() {
                0.0
            } else {
                1.0 / plan_steps.len() as f64
            };

            // Keep the best plan according to the score.
            if score > best_score {
                best_score = score;
                best_plan = plan_steps;
            }
        }

        Ok(best_plan)
    }
}
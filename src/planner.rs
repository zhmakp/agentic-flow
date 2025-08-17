use std::sync::Arc;

use tokio::sync::Mutex;

use serde_json::{Value, json};

use crate::{
    agent::TodoAgent,
    errors::AgenticFlowError,
    llm_client::{LLMClient, OllamaModel},
    tool_registry::{ExecutionContext, ToolRegistry},
};

pub struct Planner {
    agent: TodoAgent,
    tool_registry: Arc<Mutex<ToolRegistry>>,
}

impl Planner {
    pub fn new(agent: TodoAgent, tool_registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self {
            agent,
            tool_registry,
        }
    }

    pub async fn plan_and_execute(&self, task: &str) -> Result<String, AgenticFlowError> {
        let steps = self
            .generate_steps(task)
            .await
            .map_err(|e| {
                println!("Error generating steps: {}", e);
                e
            })
            .unwrap_or(vec![]);

        let mut context = ExecutionContext::new();
        let mut step = 1;
        for (tool, args) in steps {
            println!("Executing step {}: {} with args {:?}", step, tool, args);
            let result = self
                .agent
                .execute_tool(&tool, args, &mut context)
                .await
                .unwrap();
            context.set(format!("{}: {}", step, tool), result);
            step += 1;
        }

        self.synthesize_result(&context).await
    }

    async fn synthesize_result(
        &self,
        context: &ExecutionContext,
    ) -> Result<String, AgenticFlowError> {
        let messages = json!([
            {
                "role": "system",
                "content": "Synthesize the following context into result"
            },
            {
                "role": "user",
                "content": format!(
                    "Context: {}", json!(context.data())
                )
            }
        ]);

        LLMClient::from_ollama(OllamaModel::Qwen3_8B)
            .chat_completions(messages, vec![])
            .await
            .map(|response| response.message().content.to_string())
    }

    async fn generate_steps(&self, task: &str) -> Result<Vec<(String, Value)>, AgenticFlowError> {
        let messages = self.generate_steps_prompt(task);
        let tools = self.tool_registry.lock().await.get_tools_for_planner();

        LLMClient::from_ollama(OllamaModel::Qwen3_8B)
            .chat_completions(messages, tools)
            .await
            .map(|response| {
                let message = response.message();
                message
                    .tool_calls
                    .iter()
                    .flat_map(|f| {
                        f.iter()
                            .map(|t| (t.function.name.clone(), t.function.arguments.clone()))
                    })
                    .collect()
            })
    }

    fn generate_steps_prompt(&self, task: &str) -> Value {
        json!([
            {
                "role": "system",
                "content": "Analyze the task and create a multi-step plan."
            },
            {
                "role": "user",
                "content": task
            }
        ])
    }
}

use serde_json::Value;
use std::{fmt::Debug, sync::Arc};
use tokio::{
    sync::{
        Mutex,
        mpsc::{self, Sender},
    },
    task::JoinHandle,
};

use crate::{
    agent::Agent, 
    errors::AgenticFlowError, 
    planner::PlanStep, 
    tool_registry::ExecutionContext,
};

/// A simple task pool for executing agentic plan steps concurrently.
/// Provides a lightweight worker pattern for distributing tool execution across multiple async workers.
///
/// # Purpose
/// This TaskPool is designed for agentic AI systems that need to:
/// - Execute multiple plan steps concurrently using local tools
/// - Distribute tool execution across worker threads
/// - Handle execution failures gracefully
/// - Maintain execution context across steps
/// - Provide simple parallel execution for independent steps
///
/// # Usage
/// ```rust
/// let tool_registry = Arc::new(Mutex::new(ToolRegistry::new()));
/// let mut pool = AgenticTaskPool::new(4, tool_registry.clone());
/// let steps = vec![PlanStep { tool_name: "echo".to_string(), params: json!({"text": "hello"}) }];
/// let results = pool.execute_parallel(steps).await?;
/// pool.shutdown().await;
/// ```
pub struct AgenticTaskPool {
    /// Collection of worker task handles for managing concurrent execution
    workers: Vec<JoinHandle<()>>,
    /// Channel sender for distributing plan steps to workers
    sender: Option<Sender<WorkerTask>>,
    /// Channel capacity for buffering tasks
    capacity: usize,
}

/// Internal task structure for worker communication
#[derive(Debug)]
struct WorkerTask {
    /// The plan step to execute
    step: PlanStep,
    /// Response channel for sending results back
    response: tokio::sync::oneshot::Sender<Result<Value, AgenticFlowError>>,
}

impl AgenticTaskPool {
    /// Creates a new AgenticTaskPool with the specified number of workers.
    ///
    /// # Arguments
    /// * `worker_count` - Number of concurrent workers to spawn
    /// * `tool_registry` - Shared tool registry for executing plan steps
    ///
    /// # Returns
    /// A new AgenticTaskPool ready to execute plan steps
    pub fn new(worker_count: usize, agent: Arc<Mutex<Agent>>) -> Self {
        Self::new_with_capacity(worker_count, 100, agent)
    }

    /// Creates a new AgenticTaskPool with custom channel capacity.
    ///
    /// # Arguments
    /// * `worker_count` - Number of concurrent workers to spawn
    /// * `capacity` - Channel buffer size for queued tasks
    /// * `tool_registry` - Shared tool registry for executing plan steps
    pub fn new_with_capacity(
        worker_count: usize,
        capacity: usize,
        agent: Arc<Mutex<Agent>>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel::<WorkerTask>(capacity);
        let mut workers = Vec::new();
        let receiver = Arc::new(Mutex::new(receiver));

        // Spawn worker tasks that process incoming plan steps
        for worker_id in 0..worker_count {
            let agent = agent.clone();
            let receiver = receiver.clone();
            let worker = tokio::spawn(async move {
                println!("Agentic worker {} started", worker_id);
                while let Some(worker_task) = receiver.lock().await.recv().await {
                    println!(
                        "Worker {} executing step: {}",
                        worker_id, worker_task.step.tool_name
                    );

                    // Execute the plan step using the tool registry
                    let mut context = ExecutionContext::new();
                    let result = {
                        let agent = agent.lock().await;
                        agent
                            .execute_tool(
                                &worker_task.step.tool_name,
                                worker_task.step.params,
                                &mut context,
                            )
                            .await
                    };

                    // Send result back through the response channel
                    let _ = worker_task.response.send(result);
                }
                println!("Agentic worker {} shutting down", worker_id);
            });
            workers.push(worker);
        }

        Self {
            workers,
            sender: Some(sender),
            capacity,
        }
    }

    /// Executes a single plan step by sending it to an available worker.
    ///
    /// # Arguments
    /// * `step` - The plan step to execute
    ///
    /// # Returns
    /// Result containing the execution output or error
    ///
    /// # Errors
    /// Returns error if the task pool has been shut down or execution fails
    pub async fn execute_step(&self, step: PlanStep) -> Result<Value, AgenticFlowError> {
        match &self.sender {
            Some(sender) => {
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();
                let worker_task = WorkerTask {
                    step,
                    response: response_tx,
                };

                sender.send(worker_task).await.map_err(|_| {
                    AgenticFlowError::ExecutionError("Task pool is shut down".to_string())
                })?;

                response_rx.await.map_err(|_| {
                    AgenticFlowError::ExecutionError("Worker disconnected".to_string())
                })?
            }
            None => Err(AgenticFlowError::ExecutionError(
                "Task pool is shut down".to_string(),
            )),
        }
    }

    /// Executes multiple plan steps in parallel.
    ///
    /// # Arguments
    /// * `steps` - The plan steps to execute concurrently
    ///
    /// # Returns
    /// Vector of results in the same order as input steps
    ///
    /// # Errors
    /// Returns error if any step fails or the pool is shut down
    pub async fn execute_parallel(
        &self,
        steps: Vec<PlanStep>,
    ) -> Result<Vec<Value>, AgenticFlowError> {
        let mut handles = Vec::new();

        for step in steps {
            let handle = self.execute_step(step);
            handles.push(handle);
        }

        // Wait for all steps to complete
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await?);
        }

        Ok(results)
    }

    /// Gracefully shuts down the task pool.
    /// Closes the task channel and waits for all workers to complete.
    ///
    /// # Returns
    /// Result indicating successful shutdown or any worker errors
    pub async fn shutdown(mut self) -> Result<(), AgenticFlowError> {
        // Close the sender to signal workers to shut down
        self.sender.take();

        // Wait for all workers to complete
        for worker in self.workers {
            worker
                .await
                .map_err(|e| AgenticFlowError::ExecutionError(format!("Worker error: {}", e)))?;
        }

        Ok(())
    }

    /// Returns the number of active workers
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Returns the channel capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Checks if the task pool is still accepting tasks
    pub fn is_active(&self) -> bool {
        self.sender.is_some()
    }
}

/// Generic task pool for non-agentic use cases (kept for compatibility)
pub struct TaskPool<T> {
    workers: Vec<JoinHandle<()>>,
    sender: Option<Sender<T>>,
    capacity: usize,
}

impl<T> TaskPool<T>
where
    T: Send + 'static + Debug,
{
    pub async fn new(
        worker_count: usize,
        processor: Arc<Mutex<dyn Fn(T) + Send + 'static>>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(100);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::new();

        for _ in 0..worker_count {
            let receiver = receiver.clone();
            let processor = processor.clone();

            let worker = tokio::spawn(async move {
                loop {
                    let mut rx = receiver.lock().await;
                    match rx.recv().await {
                        Some(task) => processor.lock().await(task),
                        None => break,
                    }
                }
            });
            workers.push(worker);
        }

        Self {
            workers,
            sender: Some(sender),
            capacity: 100,
        }
    }

    pub async fn execute(&self, task: T) -> Result<(), AgenticFlowError> {
        match &self.sender {
            Some(sender) => {
                sender
                    .send(task)
                    .await
                    .map_err(|_| AgenticFlowError::ExecutionError("Pool shutdown".to_string()))?;
                Ok(())
            }
            None => Err(AgenticFlowError::ExecutionError(
                "Pool shutdown".to_string(),
            )),
        }
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub async fn shutdown(mut self) {
        self.sender.take();

        for worker in self.workers {
            worker
                .await
                .map_err(|e| AgenticFlowError::ExecutionError(format!("Worker error: {}", e)))
                .unwrap();
        }
    }
}

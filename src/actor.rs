use std::collections::HashMap;

use crate::errors::AgenticFlowError;
use uuid::Uuid;
use tokio::sync::{mpsc, oneshot, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActorId(Uuid);

impl ActorId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ActorId {
    fn default() -> Self {
        Self::new()
    }
}

/// Actor trait for handling messages
#[async_trait::async_trait]
pub trait Actor: Send + Sync {
    async fn handle_message(&mut self, message: Message) -> Result<(), AgenticFlowError>;
    async fn initialize(&mut self) -> Result<(), AgenticFlowError> { Ok(()) }
    async fn shutdown(&mut self) -> Result<(), AgenticFlowError> { Ok(()) }
}

/// Actor handle for sending messages
#[derive(Clone)]
pub struct ActorHandle {
    id: ActorId,
    sender: mpsc::UnboundedSender<Message>,
}

/// Messages that can be sent between actors
#[derive(Debug)]
pub enum Message {
    ExecuteTool {
        tool_name: String,
        params: serde_json::Value,
        respond_to: oneshot::Sender<Result<serde_json::Value, AgenticFlowError>>,
    },
    HealthCheck {
        respond_to: oneshot::Sender<bool>,
    },
    Shutdown,
}

impl ActorHandle {
    pub fn new(id: ActorId, sender: mpsc::UnboundedSender<Message>) -> Self {
        Self { id, sender }
    }
    
    pub fn id(&self) -> ActorId {
        self.id
    }

    pub async fn send(&self, message: Message) -> Result<(), AgenticFlowError> {
        self.sender.send(message)
            .map_err(|_| AgenticFlowError::NetworkError("Failed to send message".into()))
    }

    pub async fn call<F, T>(&self, f: F) -> Result<T, AgenticFlowError>
    where
        F: FnOnce(oneshot::Sender<T>) -> Message,
    {
        let (tx, rx) = oneshot::channel();
        self.send(f(tx)).await?;
        rx.await.map_err(|_| AgenticFlowError::NetworkError("Failed to receive response".into()))
    }

    pub async fn health_check(&self) -> Result<bool, AgenticFlowError> {
        self.call(|respond_to| Message::HealthCheck { respond_to }).await
    }
}


/// Actor system for managing actors
pub struct ActorSystem {
    actors: HashMap<ActorId, ActorHandle>,
    _shutdown_tx: mpsc::UnboundedSender<()>,
    shutdown_rx: mpsc::UnboundedReceiver<()>,
}


impl ActorSystem {
    pub fn new() -> Self {
        let (_shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();
        
        Self {
            actors: HashMap::new(),
            _shutdown_tx,
            shutdown_rx,
        }
    }
    
    pub async fn spawn_actor<A>(&mut self, mut actor: A) -> ActorHandle
    where
        A: Actor + 'static,
    {
        let id = ActorId::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handle = ActorHandle::new(id, tx);
        
        // Spawn actor task
        let actor_handle = handle.clone();
        tokio::spawn(async move {
            if let Err(e) = actor.initialize().await {
                eprintln!("Actor {} initialization failed: {}", id.0, e);
                return;
            }
            
            while let Some(message) = rx.recv().await {
                match message {
                    Message::Shutdown => {
                        let _ = actor.shutdown().await;
                        break;
                    }
                    msg => {
                        if let Err(e) = actor.handle_message(msg).await {
                            eprintln!("Actor {} error handling message: {}", id.0, e);
                        }
                    }
                }
            }
        });
        
        self.actors.insert(id, handle.clone());
        handle
    }
    
    pub fn get_actor(&self, id: ActorId) -> Option<&ActorHandle> {
        self.actors.get(&id)
    }

    pub async fn shutdown_all(&mut self) -> Result<(), AgenticFlowError> {
        for handle in self.actors.values() {
            let _ = handle.send(Message::Shutdown).await;
        }
        
        // Wait for all actors to shut down
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        self.actors.clear();
        
        Ok(())
    }
}


/// Coordinator Actor that orchestrates the system
pub struct CoordinatorActor {
    mcp_manager: ActorHandle,
    tool_registry: ActorHandle,
    executor_handle: Option<ActorHandle>,
    planner_handle: Option<ActorHandle>,
}

impl CoordinatorActor {
    pub fn new(mcp_manager: ActorHandle, tool_registry: ActorHandle) -> Self {
        Self {
            mcp_manager,
            tool_registry,
            executor_handle: None,
            planner_handle: None,
        }
    }

    pub fn set_executor(&mut self, executor: ActorHandle) {
        self.executor_handle = Some(executor);
    }
    
    pub fn set_planner(&mut self, planner: ActorHandle) {
        self.planner_handle = Some(planner);
    }
}

mod examples {
    use serde_json::json;

    use super::*;

    struct EchoActor;

    #[async_trait::async_trait]
    impl Actor for EchoActor {
        async fn handle_message(&mut self, message: Message) -> Result<(), AgenticFlowError> {
            match message {
                Message::ExecuteTool { tool_name, params, respond_to } => {
                    let _ = respond_to.send(Ok(json!({
                        "tool": tool_name,
                        "params": params
                    })));
                }
                Message::HealthCheck { respond_to } => {
                    let _ = respond_to.send(true);
                }
                Message::Shutdown => {}
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_actor_system() {
        let mut system = ActorSystem::new();
        let echo_handle = system.spawn_actor(EchoActor).await;

        // Health check
        let healthy = echo_handle.health_check().await.unwrap();
        assert!(healthy);

        // ExecuteTool
        let (tx, rx) = tokio::sync::oneshot::channel();
        let msg = Message::ExecuteTool {
            tool_name: "echo".to_string(),
            params: json!({"foo": "bar"}),
            respond_to: tx,
        };
        echo_handle.send(msg).await.unwrap();
        let response = rx.await.unwrap().unwrap();
        assert_eq!(response["tool"], "echo");
        assert_eq!(response["params"]["foo"], "bar");

        // Shutdown all
        system.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_execute_tool_messages() {
        let mut system = ActorSystem::new();
        let echo_handle = system.spawn_actor(EchoActor).await;

        let mut handles = vec![];
        for i in 0..10 {
            let handle = echo_handle.clone();
            handles.push(tokio::spawn(async move {
                let (tx, rx) = tokio::sync::oneshot::channel();
                let msg = Message::ExecuteTool {
                    tool_name: format!("tool_{}", i),
                    params: json!({ "index": i }),
                    respond_to: tx,
                };
                handle.send(msg).await.unwrap();
                let response = rx.await.unwrap().unwrap();
                assert_eq!(response["tool"], format!("tool_{}", i));
                assert_eq!(response["params"]["index"], i);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        system.shutdown_all().await.unwrap();
    }

    #[tokio::test]
    async fn test_shutdown_message() {
        let mut system = ActorSystem::new();
        let echo_handle = system.spawn_actor(EchoActor).await;

        // Send shutdown message directly
        echo_handle.send(Message::Shutdown).await.unwrap();

        // Give some time for shutdown
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Further messages should fail (actor task is dropped)
        let result = echo_handle.health_check().await;
        assert!(result.is_err());

        system.shutdown_all().await.unwrap();
    }
}
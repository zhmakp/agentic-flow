
# agentic-flow

A lightweight Rust library for building agentic workflows, planning, and LLM-powered automation.

## Features

- Agent orchestration and planning
- LLM client integrations (Ollama, OpenRouter)
- Tool registry supporting local and MCP tools
- Extensible agent system with async support

## Add to your project

Add this crate directly from GitHub using cargo:

```powershell
cargo add agentic-flow --git https://github.com/zhmakp/agentic-flow
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
agentic-flow = { git = "https://github.com/zhmakp/agentic-flow" }
```

**Note:** In Rust code, use underscores: `agentic_flow`.

## Usage Examples

### 1. LLM Client (Minimal Example)

```rust
use agentic_flow::llm_client::{LLMClient, OllamaModel, OpenRouterModel};
use agentic_flow::model::ChatMessage;
use agentic_flow::errors::AgenticFlowError;
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), AgenticFlowError> {
    // Create a client for Ollama (local instance)
    let client = LLMClient::from_ollama(OllamaModel::Gemma);
    // Or use OpenRouter:
    // let client = LLMClient::from_open_router(OpenRouterModel::Flash2);

    // Build a minimal `messages` payload
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Summarize the following text: 'Rust is fast.'"),
    ];

    let tools: Vec<Value> = Vec::new();
    let response = client.chat_completions(messages, tools).await?;
    println!("Got response: {:?}", response);
    Ok(())
}
```

#### LLMClient Usage

- Use `LLMClient::from_ollama(model)` for local Ollama models (e.g., `OllamaModel::Gemma`, `OllamaModel::Qwen3_8B`).
- Use `LLMClient::from_open_router(model)` for OpenRouter models (e.g., `OpenRouterModel::Flash2`).
- The `chat_completions` method expects a `Vec<ChatMessage>` and a `Vec<Value>` for tools (can be empty).
- The response is a boxed trait object implementing `ChatResponse` (see `model.rs`).

**Environment:**
- For OpenRouter, set the `OPENROUTER_API_KEY` environment variable.

### 2. Agentic System (Planning & Tool Use)

```rust
use agentic_flow::{AgenticSystem, SystemConfig, tool_registry::LocalTool, llm_client::{LLMClient, OllamaModel}};
use agentic_flow::errors::AgenticFlowError;

#[tokio::main]
async fn main() -> Result<(), AgenticFlowError> {
    // Prepare your tools (implement LocalTool for your custom tools)
    let tools: Vec<Box<dyn LocalTool>> = vec![];
    let config = SystemConfig::default();
    let llm_client = LLMClient::from_ollama(OllamaModel::Gemma);
    let agentic_system = AgenticSystem::new(config, tools, llm_client).await?;

    // Plan and execute a task
    let result = agentic_system.plan_and_execute("your task here").await?;
    println!("Result: {}", result);
    Ok(())
}
```

### 3. Testing (see `tests/test_integration.rs`)

Integration tests demonstrate how to use the agentic system with mock tools:

```rust
#[tokio::test]
async fn test_local_tool_calling() {
    let tools = vec![Box::new(MockTool) as Box<dyn LocalTool>];
    let llm_client = LLMClient::from_ollama(OllamaModel::Gemma); // or a mock LLM client
    let agentic_system = AgenticSystem::new(SystemConfig::example(), tools, llm_client).await.unwrap();
    let result = agentic_system.plan_and_execute("execute testing tool").await.unwrap();
    assert!(result.contains("mock_tool"));
}
```

## API Notes

- `AgenticSystem` is the main entry point for agent orchestration and planning.
- `SystemConfig` provides configuration for MCP servers, LLMs, and agent behavior.
- Tools must implement the `LocalTool` trait and are registered asynchronously at system startup.
- LLM integration is via the `LLMClient` abstraction, which must be provided to `AgenticSystem::new`.

## Contributing

- Please open issues for feature requests and bugs.
- Ensure secrets are not committed (use `.env` or CI secrets).

## License

[MIT License](LICENSE)

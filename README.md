
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
use serde_json::json;
use agentic_flow::llm_client::{LLMClient, OllamaModel};
use agentic_flow::errors::AgenticFlowError;

#[tokio::main]
async fn main() -> Result<(), AgenticFlowError> {
    // Create a client for Ollama (local instance)
    let client = LLMClient::from_ollama(OllamaModel::Gemma);

    // Build a minimal `messages` payload
    let messages = json!([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the following text: 'Rust is fast.'"}
    ]);

    let tools = Vec::new();
    let response = client.chat_completions(messages, tools).await?;
    println!("Got response: {:?}", response);
    Ok(())
}
```

### 2. Agentic System (Planning & Tool Use)

```rust
use agentic_flow::{AgenticSystem, SystemConfig, tool_registry::LocalTool};
use agentic_flow::errors::AgenticFlowError;

#[tokio::main]
async fn main() -> Result<(), AgenticFlowError> {
    // Prepare your tools (implement LocalTool for your custom tools)
    let tools: Vec<Box<dyn LocalTool>> = vec![];
    let config = SystemConfig::default();
    let agentic_system = AgenticSystem::new(config, tools).await?;

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
    let agentic_system = AgenticSystem::new(SystemConfig::example(), tools).await.unwrap();
    let result = agentic_system.process_user_request("execute testing tool").await.unwrap();
    assert!(result.content.contains("mock_tool"));
}
```

## API Notes

- `AgenticSystem` is the main entry point for agent orchestration and planning.
- `SystemConfig` provides configuration for MCP servers, LLMs, and agent behavior.
- Tools must implement the `LocalTool` trait and can be registered at system startup.
- LLM integration is via the `LLMClient` abstraction.

## Contributing

- Please open issues for feature requests and bugs.
- Ensure secrets are not committed (use `.env` or CI secrets).

## License

[MIT License](LICENSE)

# Local LLM Backend Feasibility Analysis

## Summary

Adding local LLM support to llm9p is **highly feasible** and architecturally straightforward. The existing `Backend` interface already provides the right abstraction, and the ecosystem has converged on OpenAI-compatible APIs as the standard interface for local model servers. A new `OpenAIClient` backend (~400-500 lines of Go) would enable llm9p to work with Ollama, llama.cpp, vLLM, LocalAI, LM Studio, and any other server exposing `/v1/chat/completions`.

## Current Architecture

llm9p uses a clean `Backend` interface (`internal/llm/backend.go`) with two implementations:

| | API Backend (`Client`) | CLI Backend (`CLIClient`) |
|---|---|---|
| Transport | Anthropic HTTP API | `claude` subprocess |
| Auth | `ANTHROPIC_API_KEY` | Claude Max subscription |
| Token counting | Exact (from API) | Estimated (chars/4) |
| Streaming | SSE via SDK | stdout pipe |
| Tool support | Native `tool_use` protocol | Text-only |

The `SessionManager` wraps any `Backend` and provides per-session isolation via the stateless `AskWithRequest` method. This means a new backend only needs to implement the `Backend` interface -- everything above it (sessions, filesystem, 9P protocol) works unchanged.

## The OpenAI Compatibility Ecosystem

Every major local LLM server now exposes an OpenAI-compatible `/v1/chat/completions` endpoint:

### Ollama
- **Endpoint**: `http://localhost:11434/v1/chat/completions`
- **Streaming**: Yes (SSE)
- **Tool calling**: Yes (Llama 3.1+, Mistral, Qwen 2.5)
- **Models**: Llama 3.x, Mistral, Phi, Qwen, Gemma, DeepSeek, etc.
- **Setup**: `ollama run llama3.1:8b` (auto-downloads)
- **Key advantage**: Simplest user experience; single binary, auto-manages models

### llama.cpp (llama-server)
- **Endpoint**: `http://localhost:8080/v1/chat/completions`
- **Streaming**: Yes (SSE)
- **Tool calling**: Yes (with `--jinja` flag)
- **Models**: Any GGUF model (quantized, runs on CPU or GPU)
- **Setup**: `llama-server -m model.gguf --port 8080`
- **Key advantage**: Also supports Anthropic Messages API at `/v1/messages`

### vLLM
- **Endpoint**: `http://localhost:8000/v1/chat/completions`
- **Streaming**: Yes (SSE)
- **Tool calling**: Yes (with `--enable-auto-tool-choice`)
- **Models**: HuggingFace models (full-precision or quantized)
- **Setup**: `vllm serve meta-llama/Llama-3.1-8B-Instruct`
- **Key advantage**: Highest throughput; PagedAttention, continuous batching

### LocalAI
- **Endpoint**: `http://localhost:8080/v1/chat/completions`
- **Streaming**: Yes (SSE)
- **Tool calling**: Yes (mature, production-ready)
- **Models**: GGUF, Safetensors, PyTorch, GPTQ, AWQ
- **Setup**: Docker: `docker run -p 8080:8080 localai/localai:latest`
- **Key advantage**: Also supports Anthropic API; most format-flexible

### LM Studio
- **Endpoint**: `http://localhost:1234/v1/chat/completions`
- **Streaming**: Yes (SSE)
- **Tool calling**: Yes (function calling support)
- **Models**: GGUF models via GUI
- **Setup**: Desktop app with built-in model browser
- **Key advantage**: GUI for non-technical users

## Implementation Strategy

### Recommended Approach: OpenAI-Compatible Backend

Create a new `OpenAIClient` in `internal/llm/openai_client.go` that implements the `Backend` interface using the OpenAI Chat Completions API. This is the right abstraction because:

1. **Universal compatibility** -- works with all 5+ local servers listed above
2. **Well-defined API** -- the OpenAI chat completions format is the de facto standard
3. **Go library available** -- `github.com/sashabaranov/go-openai` provides a mature, well-maintained Go client with streaming, tool calling, and custom base URL support
4. **No vendor lock-in** -- the same backend works for local models AND any OpenAI-compatible cloud provider

### Alternative Considered: Reuse Anthropic Client with Custom Base URL

Since llama.cpp and LocalAI now support the Anthropic Messages API, we *could* point the existing `Client` at a local server using `option.WithBaseURL()`. However, this is worse because:

- Only works with llama.cpp and LocalAI, not Ollama/vLLM/LM Studio
- The Anthropic Messages API is less universally supported than OpenAI
- Ties local LLM support to Anthropic-specific SDK behavior

### Implementation Outline

```go
// internal/llm/openai_client.go
type OpenAIClient struct {
    client         *openai.Client
    mu             sync.RWMutex
    model          string        // e.g., "llama3.1:8b", "mistral"
    temperature    float64
    systemPrompt   string
    prefill        string
    messages       []Message
    lastTokens     int
    totalTokens    int
    thinkingTokens int
    streaming      bool
    streamChan     chan string
    streamDone     chan struct{}
}

func NewOpenAIClient(baseURL, apiKey, model string) *OpenAIClient {
    config := openai.DefaultConfig(apiKey)
    config.BaseURL = baseURL
    return &OpenAIClient{
        client:      openai.NewClientWithConfig(config),
        model:       model,
        temperature: 0.7,
        messages:    make([]Message, 0),
    }
}
```

The implementation follows the same pattern as the existing `CLIClient`:
- Conversation history managed as `[]Message`
- Token counting estimated from response usage fields (most servers report these)
- Streaming via the SDK's `CreateChatCompletionStream`
- Tool calling via OpenAI function calling format

### Backend Interface Compatibility

All 18 methods of the `Backend` interface map cleanly:

| Method | OpenAI Implementation |
|---|---|
| `Model()` / `SetModel()` | Local field; model name passed to API |
| `Temperature()` / `SetTemperature()` | Local field; sent in request |
| `SystemPrompt()` / `SetSystemPrompt()` | Sent as `system` role message |
| `ThinkingTokens()` / `SetThinkingTokens()` | Ignored (local models don't support this) |
| `Prefill()` / `SetPrefill()` | Simulated (prepend to response) |
| `LastTokens()` / `TotalTokens()` | From API response `Usage` field |
| `ContextLimit()` | Configurable per model (default 8K or 32K) |
| `Compact()` | Use self (local model) for summarization |
| `Messages()` / `MessagesJSON()` | Same as existing backends |
| `AddSystemMessage()` / `Reset()` | Same as existing backends |
| `Ask()` | `CreateChatCompletion` |
| `AskWithHistory()` | Same with explicit history |
| `AskWithRequest()` | Full stateless call with tools |
| `StartStream()` / `ReadStreamChunk()` / `IsStreaming()` / `WaitStream()` | `CreateChatCompletionStream` |

### CLI Flag Design

```bash
# Ollama (default port)
./llm9p -backend openai -openai-url http://localhost:11434/v1 -model llama3.1:8b

# llama-server
./llm9p -backend openai -openai-url http://localhost:8080/v1 -model default

# vLLM
./llm9p -backend openai -openai-url http://localhost:8000/v1 -model meta-llama/Llama-3.1-8B-Instruct

# LM Studio
./llm9p -backend openai -openai-url http://localhost:1234/v1 -model local-model

# With API key (for cloud OpenAI-compatible providers)
OPENAI_API_KEY=sk-... ./llm9p -backend openai -openai-url https://api.openai.com/v1 -model gpt-4o
```

### New Dependency

```
github.com/sashabaranov/go-openai  (MIT license, ~8.6k stars)
```

This library supports:
- Custom base URLs (essential for local servers)
- Chat completions with streaming
- Function/tool calling
- Token usage reporting
- All model parameters (temperature, max_tokens, etc.)

## Effort Estimate

| Component | Scope |
|---|---|
| `openai_client.go` | ~400-500 lines (following CLIClient patterns) |
| `main.go` changes | ~15 lines (new flag case + validation) |
| `backend.go` | Add `var _ Backend = (*OpenAIClient)(nil)` |
| Tests | ~200 lines (unit tests with mock server) |
| `go.mod` | Add `sashabaranov/go-openai` dependency |
| Documentation | Update README, CLAUDE.md |

Total: **~700 lines of new code**, mostly mechanical since it follows the existing `CLIClient` structure closely.

## Feature Gaps and Limitations

### Features that degrade gracefully with local models:
- **Extended thinking**: Not supported by local models. `ThinkingTokens` would be ignored (same as current API backend behavior).
- **Prefill**: Not natively supported by OpenAI API format. Would be simulated by prepending to response text (same as CLIClient).
- **Tool calling**: Depends on model capability. Works well with Llama 3.1+, Mistral, Qwen 2.5. Smaller models may not support it.
- **Token counting**: Most servers report usage, but some (older Ollama versions) may not. Fallback to character-based estimation.

### Features that work fully:
- Conversation history / multi-turn
- System prompts
- Temperature control
- Model switching (within what the server has loaded)
- Streaming
- Session isolation (handled by SessionManager, not the backend)

## Recommended Models for Testing

| Model | Size | Context | Tool Calling | Notes |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | 4-8 GB | 128K | Yes | Best balance of quality and speed |
| Qwen 2.5 7B Instruct | 4-8 GB | 32K | Yes | Strong multilingual, good at tools |
| Mistral 7B Instruct | 4-8 GB | 32K | Yes | Fast, good instruction following |
| Phi-3 Mini 3.8B | 2-4 GB | 128K | Limited | Smallest usable model |
| DeepSeek-R1 7B | 4-8 GB | 64K | No | Strong reasoning, no tool calling |

## Conclusion

Adding local LLM support is a well-scoped, low-risk enhancement. The `Backend` interface is already designed for exactly this kind of extension. The OpenAI-compatible API is the clear integration point since the entire ecosystem has standardized on it. The `sashabaranov/go-openai` Go library provides everything needed. Implementation would take roughly a day of development, following the established patterns in `cli_client.go`.

The result would make llm9p usable in fully offline/air-gapped environments, eliminate API costs for development and experimentation, and open the door to any model ecosystem (not just Claude).

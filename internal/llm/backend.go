// Package llm provides LLM backends for the 9P filesystem.
package llm

import "context"

// Backend defines the interface for LLM backends.
// Both API and CLI clients implement this interface.
type Backend interface {
	// Model returns the current model name
	Model() string
	// SetModel sets the model for subsequent requests
	SetModel(model string)
	// Temperature returns the current temperature
	Temperature() float64
	// SetTemperature sets the temperature (0.0-2.0)
	SetTemperature(temp float64) error
	// SystemPrompt returns the current system prompt
	SystemPrompt() string
	// SetSystemPrompt sets the system prompt
	SetSystemPrompt(prompt string)
	// ThinkingTokens returns the thinking token budget (-1=max, 0=disabled, >0=budget)
	ThinkingTokens() int
	// SetThinkingTokens sets the thinking token budget
	SetThinkingTokens(tokens int)
	// Prefill returns the assistant response prefill string
	Prefill() string
	// SetPrefill sets a string to prefill the assistant response
	// This helps keep the model in character (e.g., "[Veltro] ")
	SetPrefill(prefill string)
	// LastTokens returns token count from last response
	LastTokens() int
	// TotalTokens returns cumulative token count for this conversation
	TotalTokens() int
	// ContextLimit returns the model's context window limit
	ContextLimit() int
	// Compact summarizes the conversation to reduce token usage
	// The conversation history is replaced with a summary
	Compact(ctx context.Context) error
	// Messages returns conversation history
	Messages() []Message
	// MessagesJSON returns conversation history as JSON
	MessagesJSON() ([]byte, error)
	// AddSystemMessage adds a system message to conversation history
	AddSystemMessage(content string)
	// Reset clears conversation history (but preserves system prompt)
	Reset()
	// Ask sends a prompt and returns the response (blocking)
	Ask(ctx context.Context, prompt string) (string, error)
	// AskWithHistory sends a prompt with explicit message history (for per-fid isolation)
	// Returns response text and token count
	AskWithHistory(ctx context.Context, history []Message, prompt string) (string, int, error)
	// AskWithRequest sends a prompt with all settings from the request (CSP - no client state)
	// This is the primary method for the clone-based session architecture.
	AskWithRequest(ctx context.Context, req AskRequest) (string, int, error)
	// StartStream begins streaming a response
	StartStream(ctx context.Context, prompt string) error
	// ReadStreamChunk reads the next streaming chunk
	ReadStreamChunk() (string, bool)
	// IsStreaming returns whether a stream is in progress
	IsStreaming() bool
	// WaitStream waits for stream to complete
	WaitStream()
}

// Verify that both clients implement Backend
var _ Backend = (*Client)(nil)
var _ Backend = (*CLIClient)(nil)

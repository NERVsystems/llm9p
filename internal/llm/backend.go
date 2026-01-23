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
	// LastTokens returns token count from last response
	LastTokens() int
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

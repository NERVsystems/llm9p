// Package llm provides a wrapper around the Anthropic API for use with the 9P filesystem.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Message represents a single message in a conversation
type Message struct {
	Role    string `json:"role"`    // "user" or "assistant"
	Content string `json:"content"` // message content
}

// Client wraps the Anthropic API client with conversation state
type Client struct {
	client       anthropic.Client
	mu           sync.RWMutex
	model        string
	temperature  float64
	systemPrompt string
	messages     []Message
	lastTokens   int
	streaming    bool
	streamChan   chan string
	streamDone   chan struct{}
}

// NewClient creates a new LLM client
func NewClient(apiKey string) *Client {
	client := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &Client{
		client:      client,
		model:       "claude-sonnet-4-20250514",
		temperature: 0.7,
		messages:    make([]Message, 0),
	}
}

// Model returns the current model name
func (c *Client) Model() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.model
}

// SetModel sets the model for subsequent requests
func (c *Client) SetModel(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.model = model
}

// Temperature returns the current temperature
func (c *Client) Temperature() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.temperature
}

// SetTemperature sets the temperature for subsequent requests
func (c *Client) SetTemperature(temp float64) error {
	if temp < 0.0 || temp > 2.0 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.temperature = temp
	return nil
}

// SystemPrompt returns the current system prompt
func (c *Client) SystemPrompt() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.systemPrompt
}

// SetSystemPrompt sets the system prompt for subsequent requests
func (c *Client) SetSystemPrompt(prompt string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.systemPrompt = prompt
}

// LastTokens returns the token count from the last response
func (c *Client) LastTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lastTokens
}

// Messages returns a copy of the conversation history
func (c *Client) Messages() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make([]Message, len(c.messages))
	copy(result, c.messages)
	return result
}

// MessagesJSON returns the conversation history as JSON
func (c *Client) MessagesJSON() ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return json.MarshalIndent(c.messages, "", "  ")
}

// AddSystemMessage adds a system message to the context
func (c *Client) AddSystemMessage(content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// System messages are prepended to conversations
	c.messages = append([]Message{{Role: "system", Content: content}}, c.messages...)
}

// Reset clears the conversation history
func (c *Client) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = make([]Message, 0)
	c.lastTokens = 0
}

// Ask sends a prompt to the LLM and returns the response
func (c *Client) Ask(ctx context.Context, prompt string) (string, error) {
	c.mu.Lock()
	// Add user message to history
	c.messages = append(c.messages, Message{Role: "user", Content: prompt})

	// Build the API messages from conversation history
	apiMessages := make([]anthropic.MessageParam, 0, len(c.messages))
	var systemBlocks []anthropic.TextBlockParam

	// Add dedicated system prompt first
	if c.systemPrompt != "" {
		systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
			Text: c.systemPrompt,
		})
	}

	for _, msg := range c.messages {
		switch msg.Role {
		case "system":
			// Also include system messages from conversation history
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
				Text: msg.Content,
			})
		case "user":
			apiMessages = append(apiMessages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		case "assistant":
			apiMessages = append(apiMessages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		}
	}

	model := c.model
	temp := c.temperature
	c.mu.Unlock()

	// Build request params
	params := anthropic.MessageNewParams{
		Model:       anthropic.Model(model),
		MaxTokens:   4096,
		Messages:    apiMessages,
		Temperature: anthropic.Float(temp),
	}

	// Add system prompt if present
	if len(systemBlocks) > 0 {
		params.System = systemBlocks
	}

	// Make the API call
	response, err := c.client.Messages.New(ctx, params)
	if err != nil {
		// Remove the user message on error
		c.mu.Lock()
		if len(c.messages) > 0 {
			c.messages = c.messages[:len(c.messages)-1]
		}
		c.mu.Unlock()
		return "", fmt.Errorf("API error: %w", err)
	}

	// Extract response text
	var responseText string
	for _, block := range response.Content {
		if block.Type == "text" {
			responseText += block.Text
		}
	}

	// Update state
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: "assistant", Content: responseText})
	c.lastTokens = int(response.Usage.InputTokens + response.Usage.OutputTokens)
	c.mu.Unlock()

	return responseText, nil
}

// StartStream begins streaming a response for the given prompt
func (c *Client) StartStream(ctx context.Context, prompt string) error {
	c.mu.Lock()
	if c.streaming {
		c.mu.Unlock()
		return fmt.Errorf("stream already in progress")
	}

	// Add user message to history
	c.messages = append(c.messages, Message{Role: "user", Content: prompt})

	// Build the API messages from conversation history
	apiMessages := make([]anthropic.MessageParam, 0, len(c.messages))
	var systemBlocks []anthropic.TextBlockParam

	// Add dedicated system prompt first
	if c.systemPrompt != "" {
		systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
			Text: c.systemPrompt,
		})
	}

	for _, msg := range c.messages {
		switch msg.Role {
		case "system":
			// Also include system messages from conversation history
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
				Text: msg.Content,
			})
		case "user":
			apiMessages = append(apiMessages, anthropic.NewUserMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		case "assistant":
			apiMessages = append(apiMessages, anthropic.NewAssistantMessage(
				anthropic.NewTextBlock(msg.Content),
			))
		}
	}

	model := c.model
	temp := c.temperature

	c.streaming = true
	c.streamChan = make(chan string, 100)
	c.streamDone = make(chan struct{})
	c.mu.Unlock()

	// Start streaming in a goroutine
	go func() {
		defer func() {
			c.mu.Lock()
			c.streaming = false
			close(c.streamChan)
			close(c.streamDone)
			c.mu.Unlock()
		}()

		// Build request params
		params := anthropic.MessageNewParams{
			Model:       anthropic.Model(model),
			MaxTokens:   4096,
			Messages:    apiMessages,
			Temperature: anthropic.Float(temp),
		}

		if len(systemBlocks) > 0 {
			params.System = systemBlocks
		}

		// Use streaming
		stream := c.client.Messages.NewStreaming(ctx, params)

		var fullResponse string
		var inputTokens, outputTokens int64

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "content_block_delta":
				delta := event.Delta
				if delta.Type == "text_delta" {
					chunk := delta.Text
					fullResponse += chunk
					select {
					case c.streamChan <- chunk:
					case <-ctx.Done():
						return
					}
				}
			case "message_delta":
				outputTokens = event.Usage.OutputTokens
			case "message_start":
				inputTokens = event.Message.Usage.InputTokens
			}
		}

		if err := stream.Err(); err != nil {
			// Send error as chunk
			select {
			case c.streamChan <- fmt.Sprintf("\n[Error: %v]", err):
			case <-ctx.Done():
			}
			// Remove user message on error
			c.mu.Lock()
			if len(c.messages) > 0 {
				c.messages = c.messages[:len(c.messages)-1]
			}
			c.mu.Unlock()
			return
		}

		// Update state with complete response
		c.mu.Lock()
		c.messages = append(c.messages, Message{Role: "assistant", Content: fullResponse})
		c.lastTokens = int(inputTokens + outputTokens)
		c.mu.Unlock()
	}()

	return nil
}

// ReadStreamChunk reads the next chunk from the stream, blocking until available
// Returns empty string and false when stream is complete
func (c *Client) ReadStreamChunk() (string, bool) {
	c.mu.RLock()
	streamChan := c.streamChan
	c.mu.RUnlock()

	if streamChan == nil {
		return "", false
	}

	chunk, ok := <-streamChan
	return chunk, ok
}

// IsStreaming returns whether a stream is currently in progress
func (c *Client) IsStreaming() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.streaming
}

// WaitStream waits for the current stream to complete
func (c *Client) WaitStream() {
	c.mu.RLock()
	done := c.streamDone
	c.mu.RUnlock()

	if done != nil {
		<-done
	}
}

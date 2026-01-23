// CLI backend for Claude Max subscription via Claude Code CLI.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"sync"
)

// CLIClient uses the Claude Code CLI for LLM requests.
// This allows using a Claude Max subscription instead of API tokens.
type CLIClient struct {
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

// cliResponse represents the JSON response from claude CLI
type cliResponse struct {
	Type   string `json:"type"`
	Result string `json:"result"`
}

// NewCLIClient creates a new CLI-based LLM client
func NewCLIClient() *CLIClient {
	return &CLIClient{
		model:       "sonnet", // CLI uses short model names
		temperature: 0.7,
		messages:    make([]Message, 0),
	}
}

// normalizeModel converts full model names to CLI aliases
func normalizeModel(model string) string {
	model = strings.ToLower(model)
	switch {
	case strings.Contains(model, "opus"):
		return "opus"
	case strings.Contains(model, "haiku"):
		return "haiku"
	default:
		return "sonnet"
	}
}

// Model returns the current model name
func (c *CLIClient) Model() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.model
}

// SetModel sets the model for subsequent requests
func (c *CLIClient) SetModel(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.model = normalizeModel(model)
}

// Temperature returns the current temperature
func (c *CLIClient) Temperature() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.temperature
}

// SetTemperature sets the temperature for subsequent requests
func (c *CLIClient) SetTemperature(temp float64) error {
	if temp < 0.0 || temp > 2.0 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.temperature = temp
	return nil
}

// SystemPrompt returns the current system prompt
func (c *CLIClient) SystemPrompt() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.systemPrompt
}

// SetSystemPrompt sets the system prompt for subsequent requests
func (c *CLIClient) SetSystemPrompt(prompt string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.systemPrompt = prompt
}

// LastTokens returns the token count from the last response
// Note: CLI doesn't provide token counts, so this is always 0
func (c *CLIClient) LastTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lastTokens
}

// Messages returns a copy of the conversation history
func (c *CLIClient) Messages() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make([]Message, len(c.messages))
	copy(result, c.messages)
	return result
}

// MessagesJSON returns the conversation history as JSON
func (c *CLIClient) MessagesJSON() ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return json.MarshalIndent(c.messages, "", "  ")
}

// AddSystemMessage adds a system message to the context
func (c *CLIClient) AddSystemMessage(content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append([]Message{{Role: "system", Content: content}}, c.messages...)
}

// Reset clears the conversation history
func (c *CLIClient) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = make([]Message, 0)
	c.lastTokens = 0
}

// buildPrompt builds a full prompt string from conversation history
func (c *CLIClient) buildPrompt() string {
	var parts []string
	for _, msg := range c.messages {
		switch msg.Role {
		case "user":
			parts = append(parts, fmt.Sprintf("Human: %s", msg.Content))
		case "assistant":
			parts = append(parts, fmt.Sprintf("Assistant: %s", msg.Content))
		}
	}
	return strings.Join(parts, "\n\n")
}

// getSystemPrompt builds the full system prompt from dedicated prompt and history
func (c *CLIClient) getSystemPrompt() string {
	var systems []string
	// Add dedicated system prompt first
	if c.systemPrompt != "" {
		systems = append(systems, c.systemPrompt)
	}
	// Also include system messages from conversation history
	for _, msg := range c.messages {
		if msg.Role == "system" {
			systems = append(systems, msg.Content)
		}
	}
	return strings.Join(systems, "\n\n")
}

// Ask sends a prompt to the LLM via CLI and returns the response
func (c *CLIClient) Ask(ctx context.Context, prompt string) (string, error) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: "user", Content: prompt})
	fullPrompt := c.buildPrompt()
	systemPrompt := c.getSystemPrompt()
	model := c.model
	c.mu.Unlock()

	// Build claude CLI command.
	// --print: non-interactive mode, output to stdout
	// --output-format json: structured output we can parse
	// --allowedTools "": disable all tools (text-only, no Bash/Edit/etc.)
	//
	// Note: --dangerously-skip-permissions is NOT needed when tools are disabled
	// and --print mode is used. The CLI only prompts for permission when tools
	// might take actions. With tools disabled, it's purely text-in/text-out.
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", model,
		"--allowedTools", "",
	}

	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}

	args = append(args, "-") // Read from stdin

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Stdin = bytes.NewBufferString(fullPrompt)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		// Remove user message on error
		c.mu.Lock()
		if len(c.messages) > 0 {
			c.messages = c.messages[:len(c.messages)-1]
		}
		c.mu.Unlock()
		return "", fmt.Errorf("claude CLI error: %w (stderr: %s)", err, stderr.String())
	}

	// Parse JSON response
	responseText, err := parseJSONResponse(stdout.String())
	if err != nil {
		// Remove user message on error
		c.mu.Lock()
		if len(c.messages) > 0 {
			c.messages = c.messages[:len(c.messages)-1]
		}
		c.mu.Unlock()
		return "", fmt.Errorf("failed to parse CLI response: %w", err)
	}

	// Update state
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: "assistant", Content: responseText})
	c.lastTokens = 0 // CLI doesn't provide token counts
	c.mu.Unlock()

	return responseText, nil
}

// parseJSONResponse extracts the result from claude CLI JSON output
func parseJSONResponse(output string) (string, error) {
	// Try parsing each line as JSON (CLI may output multiple JSON objects)
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var resp cliResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			continue // Not valid JSON, try next line
		}

		if resp.Type == "result" && resp.Result != "" {
			return resp.Result, nil
		}
	}

	// Fallback: return raw output if no JSON result found
	output = strings.TrimSpace(output)
	if output != "" {
		return output, nil
	}

	return "", fmt.Errorf("no result in CLI output")
}

// StartStream begins streaming a response for the given prompt
// Note: CLI streaming is simulated - we run the command and feed output progressively
func (c *CLIClient) StartStream(ctx context.Context, prompt string) error {
	c.mu.Lock()
	if c.streaming {
		c.mu.Unlock()
		return fmt.Errorf("stream already in progress")
	}

	c.messages = append(c.messages, Message{Role: "user", Content: prompt})
	fullPrompt := c.buildPrompt()
	systemPrompt := c.getSystemPrompt()
	model := c.model

	c.streaming = true
	c.streamChan = make(chan string, 100)
	c.streamDone = make(chan struct{})
	c.mu.Unlock()

	go func() {
		defer func() {
			c.mu.Lock()
			c.streaming = false
			close(c.streamChan)
			close(c.streamDone)
			c.mu.Unlock()
		}()

		// Build command (same flags as Ask - see comments there)
		args := []string{
			"--print",
			"--output-format", "json",
			"--model", model,
			"--allowedTools", "",
		}

		if systemPrompt != "" {
			args = append(args, "--system-prompt", systemPrompt)
		}

		args = append(args, "-")

		cmd := exec.CommandContext(ctx, "claude", args...)
		cmd.Stdin = bytes.NewBufferString(fullPrompt)

		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr

		if err := cmd.Run(); err != nil {
			select {
			case c.streamChan <- fmt.Sprintf("[Error: %v]", err):
			case <-ctx.Done():
			}
			c.mu.Lock()
			if len(c.messages) > 0 {
				c.messages = c.messages[:len(c.messages)-1]
			}
			c.mu.Unlock()
			return
		}

		// Parse and send response
		responseText, err := parseJSONResponse(stdout.String())
		if err != nil {
			select {
			case c.streamChan <- fmt.Sprintf("[Error: %v]", err):
			case <-ctx.Done():
			}
			c.mu.Lock()
			if len(c.messages) > 0 {
				c.messages = c.messages[:len(c.messages)-1]
			}
			c.mu.Unlock()
			return
		}

		// Send response as a single chunk (CLI doesn't truly stream)
		select {
		case c.streamChan <- responseText:
		case <-ctx.Done():
			return
		}

		// Update state
		c.mu.Lock()
		c.messages = append(c.messages, Message{Role: "assistant", Content: responseText})
		c.lastTokens = 0
		c.mu.Unlock()
	}()

	return nil
}

// ReadStreamChunk reads the next chunk from the stream
func (c *CLIClient) ReadStreamChunk() (string, bool) {
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
func (c *CLIClient) IsStreaming() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.streaming
}

// WaitStream waits for the current stream to complete
func (c *CLIClient) WaitStream() {
	c.mu.RLock()
	done := c.streamDone
	c.mu.RUnlock()

	if done != nil {
		<-done
	}
}

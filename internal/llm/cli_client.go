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
	mu             sync.RWMutex
	model          string
	temperature    float64
	systemPrompt   string
	prefill        string // assistant response prefill for keeping model in character
	messages       []Message
	lastTokens     int
	totalTokens    int // cumulative estimated token count
	thinkingTokens int // 0 = disabled, >0 = budget, -1 = max (default)
	streaming      bool
	streamChan     chan string
	streamDone     chan struct{}
}

// cliResponse represents the JSON response from claude CLI
type cliResponse struct {
	Type   string `json:"type"`
	Result string `json:"result"`
}

// NewCLIClient creates a new CLI-based LLM client
func NewCLIClient() *CLIClient {
	return &CLIClient{
		model:          "sonnet", // CLI uses short model names
		temperature:    0.7,
		messages:       make([]Message, 0),
		thinkingTokens: -1, // -1 = max thinking (31999 tokens) enabled by default
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

// ThinkingTokens returns the current thinking token budget
// -1 = max (31999), 0 = disabled, >0 = specific budget
func (c *CLIClient) ThinkingTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.thinkingTokens
}

// SetThinkingTokens sets the thinking token budget
// -1 = max (31999), 0 = disabled, >0 = specific budget
func (c *CLIClient) SetThinkingTokens(tokens int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.thinkingTokens = tokens
}

// Prefill returns the assistant response prefill string
func (c *CLIClient) Prefill() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.prefill
}

// SetPrefill sets a string to prefill the assistant response
func (c *CLIClient) SetPrefill(prefill string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.prefill = prefill
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
	c.totalTokens = 0
}

// TotalTokens returns cumulative estimated token count for this conversation
func (c *CLIClient) TotalTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalTokens
}

// ContextLimit returns the model's context window limit
func (c *CLIClient) ContextLimit() int {
	c.mu.RLock()
	model := c.model
	c.mu.RUnlock()
	return contextLimitForModel(model)
}

// Compact summarizes the conversation to reduce token usage
func (c *CLIClient) Compact(ctx context.Context) error {
	c.mu.Lock()
	if len(c.messages) < 4 {
		c.mu.Unlock()
		return nil // Not enough to compact
	}

	// Build conversation text for summarization
	var conversationText string
	for _, msg := range c.messages {
		if msg.Role == "system" {
			continue // Don't include system messages in summary
		}
		conversationText += fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content)
	}

	model := c.model
	thinkingTokens := c.thinkingTokens
	c.mu.Unlock()

	// Use a compact summarization prompt
	summaryPrompt := "Summarize this conversation concisely, preserving key facts, decisions, and context needed to continue:\n\n" + conversationText

	// Build CLI command for summarization
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", model,
		"--allowedTools", "",
		"--dangerously-skip-permissions",
		"-",
	}

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Stdin = bytes.NewBufferString(summaryPrompt)

	// Set thinking token budget
	cmd.Env = append(cmd.Environ(), func() string {
		if thinkingTokens < 0 {
			return "MAX_THINKING_TOKENS=31999"
		}
		return fmt.Sprintf("MAX_THINKING_TOKENS=%d", thinkingTokens)
	}())

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("compaction failed: %w (stderr: %s)", err, stderr.String())
	}

	summary, err := parseJSONResponse(stdout.String())
	if err != nil {
		return fmt.Errorf("compaction parse failed: %w", err)
	}

	// Replace conversation with summary
	c.mu.Lock()
	c.messages = []Message{{Role: "system", Content: "Previous conversation summary: " + summary}}
	// Estimate tokens for the new conversation state (chars * 0.25)
	c.totalTokens = len(summary) / 4
	c.mu.Unlock()

	return nil
}

// estimateTokens estimates token count from character count
// Uses rough approximation of 4 chars per token
func estimateTokens(s string) int {
	return (len(s) + 3) / 4 // Round up
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
	thinkingTokens := c.thinkingTokens
	c.mu.Unlock()

	// Build claude CLI command.
	// --print: non-interactive mode, output to stdout
	// --output-format json: structured output we can parse
	// --allowedTools "": disable all tools (text-only, no Bash/Edit/etc.)
	// --dangerously-skip-permissions: prevents macOS permission dialogs from blocking
	//   (Photo Library, Audio, etc. that Claude CLI initializes even with tools disabled)
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", model,
		"--allowedTools", "",
		"--dangerously-skip-permissions",
	}

	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}

	args = append(args, "-") // Read from stdin

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Stdin = bytes.NewBufferString(fullPrompt)

	// Set thinking token budget via environment variable
	// -1 = max (31999), 0 = disabled, >0 = specific budget
	cmd.Env = append(cmd.Environ(), func() string {
		if thinkingTokens < 0 {
			return "MAX_THINKING_TOKENS=31999"
		}
		return fmt.Sprintf("MAX_THINKING_TOKENS=%d", thinkingTokens)
	}())

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
	// Estimate tokens: prompt + response (chars / 4)
	c.lastTokens = estimateTokens(fullPrompt) + estimateTokens(responseText)
	c.totalTokens += c.lastTokens
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
// Uses text output mode and reads stdout progressively for real streaming
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
	thinkingTokens := c.thinkingTokens

	c.streaming = true
	c.streamChan = make(chan string, 100)
	c.streamDone = make(chan struct{})
	c.mu.Unlock()

	go func() {
		var fullResponse string

		defer func() {
			// Update conversation history with full response
			c.mu.Lock()
			if fullResponse != "" {
				c.messages = append(c.messages, Message{Role: "assistant", Content: fullResponse})
				c.lastTokens = estimateTokens(fullPrompt) + estimateTokens(fullResponse)
				c.totalTokens += c.lastTokens
			}
			c.streaming = false
			close(c.streamChan)
			close(c.streamDone)
			c.mu.Unlock()
		}()

		// Build command for streaming - use text output, not JSON
		// --output-format text gives us raw text we can stream
		args := []string{
			"--print",
			"--output-format", "text",
			"--model", model,
			"--allowedTools", "",
			"--dangerously-skip-permissions",
		}

		if systemPrompt != "" {
			args = append(args, "--system-prompt", systemPrompt)
		}

		args = append(args, "-")

		cmd := exec.CommandContext(ctx, "claude", args...)
		cmd.Stdin = bytes.NewBufferString(fullPrompt)

		// Set thinking token budget via environment variable
		cmd.Env = append(cmd.Environ(), func() string {
			if thinkingTokens < 0 {
				return "MAX_THINKING_TOKENS=31999"
			}
			return fmt.Sprintf("MAX_THINKING_TOKENS=%d", thinkingTokens)
		}())

		// Get stdout pipe for streaming reads
		stdout, err := cmd.StdoutPipe()
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

		// Start the command
		if err := cmd.Start(); err != nil {
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

		// Read stdout in chunks and send to channel
		buf := make([]byte, 256) // Small buffer for responsive streaming
		for {
			n, err := stdout.Read(buf)
			if n > 0 {
				chunk := string(buf[:n])
				fullResponse += chunk
				select {
				case c.streamChan <- chunk:
				case <-ctx.Done():
					cmd.Process.Kill()
					return
				}
			}
			if err != nil {
				break // EOF or error
			}
		}

		// Wait for command to finish
		if err := cmd.Wait(); err != nil {
			// Only report error if we got no response
			if fullResponse == "" {
				select {
				case c.streamChan <- fmt.Sprintf("[Error: %v]", err):
				case <-ctx.Done():
				}
				c.mu.Lock()
				if len(c.messages) > 0 {
					c.messages = c.messages[:len(c.messages)-1]
				}
				c.mu.Unlock()
			}
		}
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

// AskWithHistory sends a prompt with explicit message history for per-fid isolation.
// Unlike Ask(), this does not modify the client's internal messages state.
// Returns response text and estimated token count.
func (c *CLIClient) AskWithHistory(ctx context.Context, history []Message, prompt string) (string, int, error) {
	// Get settings with lock
	c.mu.RLock()
	model := c.model
	thinkingTokens := c.thinkingTokens
	systemPromptSetting := c.systemPrompt
	prefill := c.prefill
	c.mu.RUnlock()

	// Build prompt from provided history
	var parts []string
	var systemParts []string

	// Add dedicated system prompt first
	if systemPromptSetting != "" {
		systemParts = append(systemParts, systemPromptSetting)
	}

	for _, msg := range history {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, msg.Content)
		case "user":
			parts = append(parts, fmt.Sprintf("Human: %s", msg.Content))
		case "assistant":
			parts = append(parts, fmt.Sprintf("Assistant: %s", msg.Content))
		}
	}

	// Add the new user prompt
	parts = append(parts, fmt.Sprintf("Human: %s", prompt))

	fullPrompt := strings.Join(parts, "\n\n")
	systemPrompt := strings.Join(systemParts, "\n\n")

	// Build claude CLI command
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", model,
		"--allowedTools", "",
		"--dangerously-skip-permissions",
	}

	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}

	args = append(args, "-") // Read from stdin

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Stdin = bytes.NewBufferString(fullPrompt)

	// Set thinking token budget via environment variable
	cmd.Env = append(cmd.Environ(), func() string {
		if thinkingTokens < 0 {
			return "MAX_THINKING_TOKENS=31999"
		}
		return fmt.Sprintf("MAX_THINKING_TOKENS=%d", thinkingTokens)
	}())

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", 0, fmt.Errorf("claude CLI error: %w (stderr: %s)", err, stderr.String())
	}

	// Parse JSON response
	responseText, err := parseJSONResponse(stdout.String())
	if err != nil {
		return "", 0, fmt.Errorf("failed to parse CLI response: %w", err)
	}

	// Prepend prefill to response to keep model in character
	// Note: CLI doesn't support true prefill (partial assistant message),
	// so we prepend it to the response for consistent behavior with API client
	// Only add if response doesn't already start with it (model may echo from history)
	if prefill != "" && !strings.HasPrefix(responseText, prefill) {
		responseText = prefill + responseText
	}

	// Estimate tokens: prompt + response (chars / 4)
	tokens := estimateTokens(fullPrompt) + estimateTokens(responseText)

	return responseText, tokens, nil
}

// AskWithRequest sends a prompt with all settings from the request (CSP - no client state).
// This is the primary method for the clone-based session architecture.
// All settings come from the request parameter, making this a stateless API call.
func (c *CLIClient) AskWithRequest(ctx context.Context, req AskRequest) (string, int, error) {
	// Build prompt from provided history
	var parts []string
	var systemParts []string

	// Add system prompt from request
	if req.SystemPrompt != "" {
		systemParts = append(systemParts, req.SystemPrompt)
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, msg.Content)
		case "user":
			parts = append(parts, fmt.Sprintf("Human: %s", msg.Content))
		case "assistant":
			parts = append(parts, fmt.Sprintf("Assistant: %s", msg.Content))
		}
	}

	// Add the new user prompt
	parts = append(parts, fmt.Sprintf("Human: %s", req.Prompt))

	fullPrompt := strings.Join(parts, "\n\n")
	systemPrompt := strings.Join(systemParts, "\n\n")

	// Use model from request, normalize to CLI alias
	model := normalizeModel(req.Model)
	if req.Model == "" {
		c.mu.RLock()
		model = c.model
		c.mu.RUnlock()
	}

	// Use thinking tokens from request
	thinkingTokens := req.ThinkingTokens

	// Build claude CLI command
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", model,
		"--allowedTools", "",
		"--dangerously-skip-permissions",
	}

	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}

	args = append(args, "-") // Read from stdin

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Stdin = bytes.NewBufferString(fullPrompt)

	// Set thinking token budget via environment variable
	cmd.Env = append(cmd.Environ(), func() string {
		if thinkingTokens < 0 {
			return "MAX_THINKING_TOKENS=31999"
		}
		return fmt.Sprintf("MAX_THINKING_TOKENS=%d", thinkingTokens)
	}())

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", 0, fmt.Errorf("claude CLI error: %w (stderr: %s)", err, stderr.String())
	}

	// Parse JSON response
	responseText, err := parseJSONResponse(stdout.String())
	if err != nil {
		return "", 0, fmt.Errorf("failed to parse CLI response: %w", err)
	}

	// Prepend prefill to response to keep model in character
	// Only add if response doesn't already start with it (model may echo from history)
	if req.Prefill != "" && !strings.HasPrefix(responseText, req.Prefill) {
		responseText = req.Prefill + responseText
	}

	// Estimate tokens: prompt + response (chars / 4)
	tokens := estimateTokens(fullPrompt) + estimateTokens(responseText)

	return responseText, tokens, nil
}

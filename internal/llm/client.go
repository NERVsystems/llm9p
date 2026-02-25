// Package llm provides a wrapper around the Anthropic API for use with the 9P filesystem.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Message represents a single message in a conversation.
// StructuredContent, when non-empty, holds a JSON array of content blocks
// for proper API replay of tool-use turns. Plain-text turns leave it empty.
type Message struct {
	Role              string `json:"role"`          // "user" or "assistant"
	Content           string `json:"content"`       // text content (always set)
	StructuredContent string `json:"sc,omitempty"`  // JSON content blocks (tool turns only)
}

// MetricsCallback is called after each LLM request with performance data
type MetricsCallback func(inputTokens, outputTokens int, latencyMs int64)

// Global metrics callback - set by llmfs to record metrics
var metricsCallback MetricsCallback

// SetMetricsCallback registers a callback for recording metrics
func SetMetricsCallback(cb MetricsCallback) {
	metricsCallback = cb
}

// RecordMetrics calls the registered callback if set
func RecordMetrics(inputTokens, outputTokens int, latencyMs int64) {
	if metricsCallback != nil {
		metricsCallback(inputTokens, outputTokens, latencyMs)
	}
}

// Client wraps the Anthropic API client with conversation state
type Client struct {
	client         anthropic.Client
	mu             sync.RWMutex
	model          string
	temperature    float64
	systemPrompt   string
	prefill        string // assistant response prefill for keeping model in character
	messages       []Message
	lastTokens     int
	totalTokens    int // cumulative token count for context tracking
	thinkingTokens int // 0 = disabled, >0 = budget, -1 = max (default for CLI, not used for API yet)
	streaming      bool
	streamChan     chan string
	streamDone     chan struct{}
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

// ThinkingTokens returns the current thinking token budget
// Note: API backend does not currently use extended thinking
func (c *Client) ThinkingTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.thinkingTokens
}

// SetThinkingTokens sets the thinking token budget
// Note: API backend does not currently use extended thinking
func (c *Client) SetThinkingTokens(tokens int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.thinkingTokens = tokens
}

// Prefill returns the assistant response prefill string
func (c *Client) Prefill() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.prefill
}

// SetPrefill sets a string to prefill the assistant response
func (c *Client) SetPrefill(prefill string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.prefill = prefill
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
	c.totalTokens = 0
}

// TotalTokens returns cumulative token count for this conversation
func (c *Client) TotalTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalTokens
}

// ContextLimit returns the model's context window limit
func (c *Client) ContextLimit() int {
	c.mu.RLock()
	model := c.model
	c.mu.RUnlock()
	return contextLimitForModel(model)
}

// Compact summarizes the conversation to reduce token usage
func (c *Client) Compact(ctx context.Context) error {
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
	c.mu.Unlock()

	// Use a compact summarization prompt
	summaryPrompt := "Summarize this conversation concisely, preserving key facts, decisions, and context needed to continue:\n\n" + conversationText

	// Build API request for summarization
	apiMessages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(summaryPrompt)),
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 2048,
		Messages:  apiMessages,
	}

	response, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return fmt.Errorf("compaction failed: %w", err)
	}

	// Extract summary
	var summary string
	for _, block := range response.Content {
		if block.Type == "text" {
			summary += block.Text
		}
	}

	// Replace conversation with summary
	c.mu.Lock()
	c.messages = []Message{{Role: "system", Content: "Previous conversation summary: " + summary}}
	c.totalTokens = int(response.Usage.InputTokens + response.Usage.OutputTokens)
	c.mu.Unlock()

	return nil
}

// contextLimitForModel returns the context window size for a model
func contextLimitForModel(model string) int {
	model = strings.ToLower(model)
	// Claude models and their context limits
	switch {
	case strings.Contains(model, "opus"):
		return 200000
	case strings.Contains(model, "sonnet"):
		return 200000
	case strings.Contains(model, "haiku"):
		return 200000
	default:
		return 200000 // Default to 200K for newer Claude models
	}
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

	// Make the API call with timing
	startTime := time.Now()
	response, err := c.client.Messages.New(ctx, params)
	latencyMs := time.Since(startTime).Milliseconds()

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
	c.totalTokens += c.lastTokens
	c.mu.Unlock()

	// Record metrics (input and output tokens separately for analysis)
	inputToks := int(response.Usage.InputTokens)
	outputToks := int(response.Usage.OutputTokens)
	RecordMetrics(inputToks, outputToks, latencyMs)

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
		c.totalTokens += c.lastTokens
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

// AskWithHistory sends a prompt with explicit message history for per-fid isolation.
// Unlike Ask(), this does not modify the client's internal messages state.
// Returns response text and token count.
func (c *Client) AskWithHistory(ctx context.Context, history []Message, prompt string) (string, int, error) {
	// Get settings with lock
	c.mu.RLock()
	model := c.model
	temp := c.temperature
	systemPrompt := c.systemPrompt
	prefill := c.prefill
	c.mu.RUnlock()

	// Build API messages from provided history plus the new prompt
	apiMessages := make([]anthropic.MessageParam, 0, len(history)+2)
	var systemBlocks []anthropic.TextBlockParam

	// Add dedicated system prompt first
	if systemPrompt != "" {
		systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
			Text: systemPrompt,
		})
	}

	for _, msg := range history {
		switch msg.Role {
		case "system":
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

	// Add the new user prompt
	apiMessages = append(apiMessages, anthropic.NewUserMessage(
		anthropic.NewTextBlock(prompt),
	))

	// Add prefill as partial assistant message to keep model in character
	// The model will continue from this point
	if prefill != "" {
		apiMessages = append(apiMessages, anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(prefill),
		))
	}

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

	// Make the API call with timing
	startTime := time.Now()
	response, err := c.client.Messages.New(ctx, params)
	latencyMs := time.Since(startTime).Milliseconds()

	if err != nil {
		return "", 0, fmt.Errorf("API error: %w", err)
	}

	// Extract response text
	var responseText string
	for _, block := range response.Content {
		if block.Type == "text" {
			responseText += block.Text
		}
	}

	// Prepend prefill to response (it was used as partial assistant message)
	if prefill != "" {
		responseText = prefill + responseText
	}

	tokens := int(response.Usage.InputTokens + response.Usage.OutputTokens)

	// Record metrics
	inputToks := int(response.Usage.InputTokens)
	outputToks := int(response.Usage.OutputTokens)
	RecordMetrics(inputToks, outputToks, latencyMs)

	return responseText, tokens, nil
}

// AskWithRequest sends a prompt with all settings from the request (CSP - no client state).
// This is the primary method for the clone-based session architecture.
// When req.ToolDefs is non-nil, uses the Anthropic native tool_use protocol and
// returns a STOP:-prefixed response. Otherwise returns plain text (backward-compatible).
func (c *Client) AskWithRequest(ctx context.Context, req AskRequest) (AskResponse, error) {
	// Build API messages from provided history
	apiMessages := make([]anthropic.MessageParam, 0, len(req.Messages)+2)
	var systemBlocks []anthropic.TextBlockParam

	if req.SystemPrompt != "" {
		systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: req.SystemPrompt})
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: msg.Content})
		case "user", "assistant":
			param := buildMessageParam(msg)
			apiMessages = append(apiMessages, param)
		}
	}

	// Add new user turn: either a text prompt or tool results.
	if len(req.ToolResults) > 0 {
		// Tool results ARE the new user turn — build tool_result content blocks.
		var content []anthropic.ContentBlockParamUnion
		for _, r := range req.ToolResults {
			content = append(content, anthropic.NewToolResultBlock(r.ToolUseID, r.Content, false))
		}
		apiMessages = append(apiMessages, anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleUser,
			Content: content,
		})
	} else if req.Prompt != "" {
		apiMessages = append(apiMessages, anthropic.NewUserMessage(anthropic.NewTextBlock(req.Prompt)))
	}

	// Prefill only when not in tool mode (prefill is inappropriate mid-tool-loop).
	if req.Prefill != "" && len(req.ToolDefs) == 0 {
		apiMessages = append(apiMessages, anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(req.Prefill),
		))
	}

	model := req.Model
	if model == "" {
		c.mu.RLock()
		model = c.model
		c.mu.RUnlock()
	}

	params := anthropic.MessageNewParams{
		Model:       anthropic.Model(model),
		MaxTokens:   4096,
		Messages:    apiMessages,
		Temperature: anthropic.Float(req.Temperature),
	}
	if len(systemBlocks) > 0 {
		params.System = systemBlocks
	}

	// Attach tool definitions when present.
	if len(req.ToolDefs) > 0 {
		params.Tools = buildToolParams(req.ToolDefs)
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfToolChoiceAuto: &anthropic.ToolChoiceAutoParam{},
		}
	}

	var (
		resp      *anthropic.Message
		latencyMs int64
	)
	startTime := time.Now()

	if req.StreamFunc != nil {
		// Streaming path: use SSE, call StreamFunc for each text_delta chunk.
		// Accumulate the full message for STOP:/TOOL: formatting after streaming ends.
		stream := c.client.Messages.NewStreaming(ctx, params)
		var acc anthropic.Message
		for stream.Next() {
			event := stream.Current()
			if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
				req.StreamFunc(event.Delta.Text)
			}
			_ = acc.Accumulate(event)
		}
		latencyMs = time.Since(startTime).Milliseconds()
		if err := stream.Err(); err != nil {
			return AskResponse{}, fmt.Errorf("API streaming error: %w", err)
		}
		resp = &acc
	} else {
		// Blocking path: single HTTP request, no streaming.
		r, err := c.client.Messages.New(ctx, params)
		latencyMs = time.Since(startTime).Milliseconds()
		if err != nil {
			return AskResponse{}, fmt.Errorf("API error: %w", err)
		}
		resp = r
	}

	tokens := int(resp.Usage.InputTokens + resp.Usage.OutputTokens)
	RecordMetrics(int(resp.Usage.InputTokens), int(resp.Usage.OutputTokens), latencyMs)

	// Plain-text mode (no tools): return text as before.
	if len(req.ToolDefs) == 0 {
		var text string
		for _, block := range resp.Content {
			if block.Type == "text" {
				text += block.Text
			}
		}
		if req.Prefill != "" && !strings.HasPrefix(text, req.Prefill) {
			text = req.Prefill + text
		}
		return AskResponse{Response: text, Tokens: tokens}, nil
	}

	// Tool mode: format STOP: response and build structured JSON for history.
	var textParts []string
	var toolCalls []struct{ id, name, args string }
	var structBlocks []string // JSON content blocks for history

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				textParts = append(textParts, block.Text)
				escaped := jsonEscapeString(block.Text)
				structBlocks = append(structBlocks, fmt.Sprintf(`{"type":"text","text":"%s"}`, escaped))
			}
		case "tool_use":
			tb := block.AsResponseToolUseBlock()
			args := extractToolArgs(tb.Input)
			toolCalls = append(toolCalls, struct{ id, name, args string }{tb.ID, tb.Name, args})
			inputJSON := string(tb.Input)
			if inputJSON == "" {
				inputJSON = "{}"
			}
			idEsc := jsonEscapeString(tb.ID)
			nameEsc := jsonEscapeString(tb.Name)
			structBlocks = append(structBlocks,
				fmt.Sprintf(`{"type":"tool_use","id":"%s","name":"%s","input":%s}`, idEsc, nameEsc, inputJSON))
		}
	}

	structuredJSON := ""
	if len(structBlocks) > 0 {
		structuredJSON = "[" + strings.Join(structBlocks, ",") + "]"
	}

	// Build formatted response.
	var sb strings.Builder
	if resp.StopReason == anthropic.MessageStopReasonToolUse {
		sb.WriteString("STOP:tool_use\n")
		for _, tc := range toolCalls {
			// Escape newlines in args so the TOOL: line stays single-line.
			safeArgs := strings.ReplaceAll(tc.args, "\n", `\n`)
			sb.WriteString(fmt.Sprintf("TOOL:%s:%s:%s\n", tc.id, tc.name, safeArgs))
		}
	} else {
		sb.WriteString("STOP:end_turn\n")
	}
	sb.WriteString(strings.Join(textParts, ""))

	return AskResponse{Response: sb.String(), StructuredJSON: structuredJSON, Tokens: tokens}, nil
}

// buildMessageParam converts a Message to an anthropic.MessageParam.
// When StructuredContent is set, the full content blocks are rebuilt for API replay.
func buildMessageParam(msg Message) anthropic.MessageParam {
	role := anthropic.MessageParamRoleUser
	if msg.Role == "assistant" {
		role = anthropic.MessageParamRoleAssistant
	}

	if msg.StructuredContent == "" {
		// Plain text message. Guard against empty text blocks — the Anthropic API
		// rejects any content block with text:"". This can happen when an assistant
		// end_turn after tool results has no text (Claude acknowledged silently).
		content := msg.Content
		if content == "" {
			content = "..."
		}
		if role == anthropic.MessageParamRoleUser {
			return anthropic.NewUserMessage(anthropic.NewTextBlock(content))
		}
		return anthropic.NewAssistantMessage(anthropic.NewTextBlock(content))
	}

	// Structured content: unmarshal and rebuild content blocks.
	type rawBlock struct {
		Type      string          `json:"type"`
		Text      string          `json:"text"`
		ID        string          `json:"id"`
		Name      string          `json:"name"`
		Input     json.RawMessage `json:"input"`
		ToolUseID string          `json:"tool_use_id"`
		Content   string          `json:"content"`
		IsError   bool            `json:"is_error"`
	}
	var blocks []rawBlock
	if err := json.Unmarshal([]byte(msg.StructuredContent), &blocks); err != nil {
		// Fallback to plain text on parse error.
		if role == anthropic.MessageParamRoleUser {
			return anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content))
		}
		return anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content))
	}

	var content []anthropic.ContentBlockParamUnion
	for _, b := range blocks {
		switch b.Type {
		case "text":
			content = append(content, anthropic.ContentBlockParamOfRequestTextBlock(b.Text))
		case "tool_use":
			var input interface{} = b.Input
			if len(b.Input) == 0 {
				input = map[string]interface{}{}
			}
			content = append(content, anthropic.ContentBlockParamOfRequestToolUseBlock(b.ID, input, b.Name))
		case "tool_result":
			content = append(content, anthropic.NewToolResultBlock(b.ToolUseID, b.Content, b.IsError))
		}
	}

	return anthropic.MessageParam{Role: role, Content: content}
}

// buildToolParams converts ToolDef slice to anthropic SDK tool params.
func buildToolParams(defs []ToolDef) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(defs))
	for _, d := range defs {
		schema := anthropic.ToolInputSchemaParam{
			Properties: d.InputSchema["properties"],
		}
		desc := d.Description
		out = append(out, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        d.Name,
				Description: anthropic.String(desc),
				InputSchema: schema,
			},
		})
	}
	return out
}

// extractToolArgs extracts the "args" string from a tool_use input JSON.
// Falls back to the raw JSON string if "args" is not present.
func extractToolArgs(input json.RawMessage) string {
	var m map[string]interface{}
	if err := json.Unmarshal(input, &m); err != nil {
		return string(input)
	}
	if args, ok := m["args"].(string); ok {
		return args
	}
	// No "args" key — join all string values as fallback.
	var parts []string
	for _, v := range m {
		if s, ok := v.(string); ok {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, " ")
}

// jsonEscapeString escapes a string for embedding in a JSON string literal.
func jsonEscapeString(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	s = strings.ReplaceAll(s, "\r", `\r`)
	s = strings.ReplaceAll(s, "\t", `\t`)
	return s
}

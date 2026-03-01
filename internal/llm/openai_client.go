// OpenAI-compatible backend for local LLM servers (Ollama, llama.cpp, vLLM, etc.).
//
// This backend speaks the OpenAI Chat Completions API (/v1/chat/completions),
// which is the de facto standard for local model serving. Primary target is
// GPT-OSS, but any model served by Ollama, vLLM, llama-server, LocalAI, or
// LM Studio works.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// OpenAIClient uses an OpenAI-compatible API for LLM requests.
// Works with any server that implements /v1/chat/completions:
// Ollama, vLLM, llama-server, LocalAI, LM Studio, etc.
type OpenAIClient struct {
	client         *openai.Client
	mu             sync.RWMutex
	model          string
	temperature    float64
	systemPrompt   string
	prefill        string
	messages       []Message
	lastTokens     int
	totalTokens    int
	thinkingTokens int
	contextLimit   int
	streaming      bool
	streamChan     chan string
	streamDone     chan struct{}
}

// NewOpenAIClient creates a new OpenAI-compatible LLM client.
// baseURL should include /v1 (e.g. "http://localhost:11434/v1" for Ollama).
// apiKey can be empty or a dummy value for local servers that don't require auth.
// model is the model name as the server expects it (e.g. "gpt-oss:20b").
func NewOpenAIClient(baseURL, apiKey, model string) *OpenAIClient {
	if apiKey == "" {
		apiKey = "not-needed"
	}
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseURL
	return &OpenAIClient{
		client:       openai.NewClientWithConfig(config),
		model:        model,
		temperature:  0.7,
		messages:     make([]Message, 0),
		contextLimit: 128000, // Default for GPT-OSS; overridable
	}
}

// Model returns the current model name.
func (c *OpenAIClient) Model() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.model
}

// SetModel sets the model for subsequent requests.
func (c *OpenAIClient) SetModel(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.model = model
}

// Temperature returns the current temperature.
func (c *OpenAIClient) Temperature() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.temperature
}

// SetTemperature sets the temperature for subsequent requests.
func (c *OpenAIClient) SetTemperature(temp float64) error {
	if temp < 0.0 || temp > 2.0 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.temperature = temp
	return nil
}

// ThinkingTokens returns the thinking token budget.
// Not used by OpenAI-compatible backends; stored for interface compliance.
func (c *OpenAIClient) ThinkingTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.thinkingTokens
}

// SetThinkingTokens sets the thinking token budget.
// Not used by OpenAI-compatible backends.
func (c *OpenAIClient) SetThinkingTokens(tokens int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.thinkingTokens = tokens
}

// Prefill returns the assistant response prefill string.
func (c *OpenAIClient) Prefill() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.prefill
}

// SetPrefill sets a string to prefill the assistant response.
// OpenAI API doesn't support native prefill, so we prepend to the response.
func (c *OpenAIClient) SetPrefill(prefill string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.prefill = prefill
}

// SystemPrompt returns the current system prompt.
func (c *OpenAIClient) SystemPrompt() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.systemPrompt
}

// SetSystemPrompt sets the system prompt for subsequent requests.
func (c *OpenAIClient) SetSystemPrompt(prompt string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.systemPrompt = prompt
}

// LastTokens returns the token count from the last response.
func (c *OpenAIClient) LastTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lastTokens
}

// TotalTokens returns cumulative token count for this conversation.
func (c *OpenAIClient) TotalTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalTokens
}

// ContextLimit returns the model's context window limit.
func (c *OpenAIClient) ContextLimit() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.contextLimit
}

// Messages returns a copy of the conversation history.
func (c *OpenAIClient) Messages() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make([]Message, len(c.messages))
	copy(result, c.messages)
	return result
}

// MessagesJSON returns the conversation history as JSON.
func (c *OpenAIClient) MessagesJSON() ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return json.MarshalIndent(c.messages, "", "  ")
}

// AddSystemMessage adds a system message to the context.
func (c *OpenAIClient) AddSystemMessage(content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append([]Message{{Role: "system", Content: content}}, c.messages...)
}

// Reset clears the conversation history.
func (c *OpenAIClient) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = make([]Message, 0)
	c.lastTokens = 0
	c.totalTokens = 0
}

// Compact summarizes the conversation to reduce token usage.
func (c *OpenAIClient) Compact(ctx context.Context) error {
	c.mu.Lock()
	if len(c.messages) < 4 {
		c.mu.Unlock()
		return nil
	}

	var conversationText string
	for _, msg := range c.messages {
		if msg.Role == "system" {
			continue
		}
		conversationText += fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content)
	}

	model := c.model
	c.mu.Unlock()

	summaryPrompt := "Summarize this conversation concisely, preserving key facts, decisions, and context needed to continue:\n\n" + conversationText

	req := openai.ChatCompletionRequest{
		Model:       model,
		MaxTokens:   2048,
		Temperature: 0.3,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: summaryPrompt},
		},
	}

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("compaction failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return fmt.Errorf("compaction returned no choices")
	}

	summary := resp.Choices[0].Message.Content
	tokens := resp.Usage.TotalTokens

	c.mu.Lock()
	c.messages = []Message{{Role: "system", Content: "Previous conversation summary: " + summary}}
	c.totalTokens = tokens
	c.mu.Unlock()

	return nil
}

// buildChatMessages converts internal Message history to OpenAI API format.
// Returns system messages separately (as a single system message) and the
// conversation messages.
func buildChatMessages(systemPrompt string, msgs []Message) []openai.ChatCompletionMessage {
	apiMsgs := make([]openai.ChatCompletionMessage, 0, len(msgs)+2)

	// Collect all system content into a single system message
	var systemParts []string
	if systemPrompt != "" {
		systemParts = append(systemParts, systemPrompt)
	}
	for _, msg := range msgs {
		if msg.Role == "system" {
			systemParts = append(systemParts, msg.Content)
		}
	}
	if len(systemParts) > 0 {
		apiMsgs = append(apiMsgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: strings.Join(systemParts, "\n\n"),
		})
	}

	// Add conversation messages
	for _, msg := range msgs {
		switch msg.Role {
		case "user":
			apiMsgs = append(apiMsgs, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: msg.Content,
			})
		case "assistant":
			// Check if this message has structured content with tool calls
			if msg.StructuredContent != "" {
				apiMsg := rebuildAssistantToolMessage(msg)
				apiMsgs = append(apiMsgs, apiMsg)
			} else {
				apiMsgs = append(apiMsgs, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: msg.Content,
				})
			}
		}
	}

	return apiMsgs
}

// rebuildAssistantToolMessage reconstructs an assistant message with tool calls
// from the stored StructuredContent JSON.
func rebuildAssistantToolMessage(msg Message) openai.ChatCompletionMessage {
	type rawBlock struct {
		Type      string          `json:"type"`
		Text      string          `json:"text"`
		ID        string          `json:"id"`
		Name      string          `json:"name"`
		Input     json.RawMessage `json:"input"`
		ToolUseID string          `json:"tool_use_id"`
		Content   string          `json:"content"`
	}

	var blocks []rawBlock
	if err := json.Unmarshal([]byte(msg.StructuredContent), &blocks); err != nil {
		// Fallback to plain text
		return openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: msg.Content,
		}
	}

	apiMsg := openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleAssistant,
	}

	for _, b := range blocks {
		switch b.Type {
		case "text":
			apiMsg.Content += b.Text
		case "tool_use":
			idx := len(apiMsg.ToolCalls)
			apiMsg.ToolCalls = append(apiMsg.ToolCalls, openai.ToolCall{
				Index: &idx,
				ID:    b.ID,
				Type:  openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      b.Name,
					Arguments: string(b.Input),
				},
			})
		}
	}

	return apiMsg
}

// Ask sends a prompt to the LLM and returns the response.
func (c *OpenAIClient) Ask(ctx context.Context, prompt string) (string, error) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: "user", Content: prompt})
	apiMsgs := buildChatMessages(c.systemPrompt, c.messages)
	model := c.model
	temp := c.temperature
	c.mu.Unlock()

	req := openai.ChatCompletionRequest{
		Model:       model,
		MaxTokens:   4096,
		Temperature: float32(temp),
		Messages:    apiMsgs,
	}

	startTime := time.Now()
	resp, err := c.client.CreateChatCompletion(ctx, req)
	latencyMs := time.Since(startTime).Milliseconds()

	if err != nil {
		c.mu.Lock()
		if len(c.messages) > 0 {
			c.messages = c.messages[:len(c.messages)-1]
		}
		c.mu.Unlock()
		return "", fmt.Errorf("OpenAI API error: %w", err)
	}

	if len(resp.Choices) == 0 {
		c.mu.Lock()
		if len(c.messages) > 0 {
			c.messages = c.messages[:len(c.messages)-1]
		}
		c.mu.Unlock()
		return "", fmt.Errorf("OpenAI API returned no choices")
	}

	responseText := resp.Choices[0].Message.Content
	tokens := resp.Usage.TotalTokens
	if tokens == 0 {
		tokens = estimateTokens(prompt) + estimateTokens(responseText)
	}

	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: "assistant", Content: responseText})
	c.lastTokens = tokens
	c.totalTokens += tokens
	c.mu.Unlock()

	RecordMetrics(resp.Usage.PromptTokens, resp.Usage.CompletionTokens, latencyMs)

	return responseText, nil
}

// AskWithHistory sends a prompt with explicit message history for per-fid isolation.
func (c *OpenAIClient) AskWithHistory(ctx context.Context, history []Message, prompt string) (string, int, error) {
	c.mu.RLock()
	model := c.model
	temp := c.temperature
	systemPrompt := c.systemPrompt
	prefill := c.prefill
	c.mu.RUnlock()

	// Build messages from history + new prompt
	combined := make([]Message, len(history))
	copy(combined, history)
	combined = append(combined, Message{Role: "user", Content: prompt})

	apiMsgs := buildChatMessages(systemPrompt, combined)

	req := openai.ChatCompletionRequest{
		Model:       model,
		MaxTokens:   4096,
		Temperature: float32(temp),
		Messages:    apiMsgs,
	}

	startTime := time.Now()
	resp, err := c.client.CreateChatCompletion(ctx, req)
	latencyMs := time.Since(startTime).Milliseconds()

	if err != nil {
		return "", 0, fmt.Errorf("OpenAI API error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", 0, fmt.Errorf("OpenAI API returned no choices")
	}

	responseText := resp.Choices[0].Message.Content

	if prefill != "" && !strings.HasPrefix(responseText, prefill) {
		responseText = prefill + responseText
	}

	tokens := resp.Usage.TotalTokens
	if tokens == 0 {
		tokens = estimateTokens(prompt) + estimateTokens(responseText)
	}

	RecordMetrics(resp.Usage.PromptTokens, resp.Usage.CompletionTokens, latencyMs)

	return responseText, tokens, nil
}

// AskWithRequest sends a prompt with all settings from the request (CSP - no client state).
// This is the primary method for the clone-based session architecture.
// When req.ToolDefs is non-nil, uses OpenAI function calling and returns a
// STOP:-prefixed response. Otherwise returns plain text (backward-compatible).
func (c *OpenAIClient) AskWithRequest(ctx context.Context, req AskRequest) (AskResponse, error) {
	// Build messages from request history
	combined := make([]Message, len(req.Messages))
	copy(combined, req.Messages)

	// Add tool results as tool-role messages
	if len(req.ToolResults) > 0 {
		for _, r := range req.ToolResults {
			combined = append(combined, Message{
				Role:    "user",
				Content: fmt.Sprintf("Tool result for %s: %s", r.ToolUseID, r.Content),
			})
		}
	} else if req.Prompt != "" {
		combined = append(combined, Message{Role: "user", Content: req.Prompt})
	}

	apiMsgs := buildChatMessages(req.SystemPrompt, combined)

	// Handle tool results: convert to OpenAI tool message format
	// We need to rebuild the last messages if we have tool results
	if len(req.ToolResults) > 0 {
		// Remove the placeholder user messages we added above
		apiMsgs = apiMsgs[:len(apiMsgs)-len(req.ToolResults)]
		// Add proper tool result messages
		for _, r := range req.ToolResults {
			apiMsgs = append(apiMsgs, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    r.Content,
				ToolCallID: r.ToolUseID,
			})
		}
	}

	model := req.Model
	if model == "" {
		c.mu.RLock()
		model = c.model
		c.mu.RUnlock()
	}

	temp := req.Temperature

	chatReq := openai.ChatCompletionRequest{
		Model:       model,
		MaxTokens:   4096,
		Temperature: float32(temp),
		Messages:    apiMsgs,
	}

	// Attach tool definitions when present
	if len(req.ToolDefs) > 0 {
		chatReq.Tools = buildOpenAITools(req.ToolDefs)
		chatReq.ToolChoice = "auto"
	}

	var (
		responseText   string
		toolCalls      []openai.ToolCall
		finishReason   openai.FinishReason
		promptTokens   int
		completionToks int
		totalTokens    int
		latencyMs      int64
	)

	startTime := time.Now()

	if req.StreamFunc != nil {
		// Streaming path
		chatReq.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
		stream, err := c.client.CreateChatCompletionStream(ctx, chatReq)
		if err != nil {
			return AskResponse{}, fmt.Errorf("OpenAI streaming error: %w", err)
		}
		defer stream.Close()

		var textParts []string
		toolCallMap := make(map[int]*openai.ToolCall)

		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return AskResponse{}, fmt.Errorf("OpenAI streaming error: %w", err)
			}

			// Capture usage from the final chunk
			if chunk.Usage != nil {
				promptTokens = chunk.Usage.PromptTokens
				completionToks = chunk.Usage.CompletionTokens
				totalTokens = chunk.Usage.TotalTokens
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			finishReason = choice.FinishReason

			// Text delta
			if choice.Delta.Content != "" {
				textParts = append(textParts, choice.Delta.Content)
				req.StreamFunc(choice.Delta.Content)
			}

			// Tool call deltas
			for _, tc := range choice.Delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				existing, ok := toolCallMap[idx]
				if !ok {
					toolCallMap[idx] = &openai.ToolCall{
						ID:   tc.ID,
						Type: tc.Type,
						Function: openai.FunctionCall{
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
					}
				} else {
					if tc.ID != "" {
						existing.ID = tc.ID
					}
					if tc.Function.Name != "" {
						existing.Function.Name += tc.Function.Name
					}
					existing.Function.Arguments += tc.Function.Arguments
				}
			}
		}

		latencyMs = time.Since(startTime).Milliseconds()
		responseText = strings.Join(textParts, "")

		// Collect tool calls in order
		for i := 0; i < len(toolCallMap); i++ {
			if tc, ok := toolCallMap[i]; ok {
				toolCalls = append(toolCalls, *tc)
			}
		}
	} else {
		// Blocking path
		resp, err := c.client.CreateChatCompletion(ctx, chatReq)
		latencyMs = time.Since(startTime).Milliseconds()
		if err != nil {
			return AskResponse{}, fmt.Errorf("OpenAI API error: %w", err)
		}

		if len(resp.Choices) == 0 {
			return AskResponse{}, fmt.Errorf("OpenAI API returned no choices")
		}

		responseText = resp.Choices[0].Message.Content
		toolCalls = resp.Choices[0].Message.ToolCalls
		finishReason = resp.Choices[0].FinishReason
		promptTokens = resp.Usage.PromptTokens
		completionToks = resp.Usage.CompletionTokens
		totalTokens = resp.Usage.TotalTokens
	}

	if totalTokens == 0 {
		totalTokens = estimateTokens(responseText)
	}

	RecordMetrics(promptTokens, completionToks, latencyMs)

	// Plain-text mode (no tools): return text as before
	if len(req.ToolDefs) == 0 {
		if req.Prefill != "" && !strings.HasPrefix(responseText, req.Prefill) {
			responseText = req.Prefill + responseText
		}
		return AskResponse{Response: responseText, Tokens: totalTokens}, nil
	}

	// Tool mode: format STOP: response and build structured JSON for history
	var textParts []string
	var toolCallEntries []struct{ id, name, args string }
	var structBlocks []string

	if responseText != "" {
		textParts = append(textParts, responseText)
		escaped := jsonEscapeString(responseText)
		structBlocks = append(structBlocks, fmt.Sprintf(`{"type":"text","text":"%s"}`, escaped))
	}

	for _, tc := range toolCalls {
		toolCallEntries = append(toolCallEntries, struct{ id, name, args string }{
			tc.ID, tc.Function.Name, tc.Function.Arguments,
		})
		inputJSON := tc.Function.Arguments
		if inputJSON == "" {
			inputJSON = "{}"
		}
		idEsc := jsonEscapeString(tc.ID)
		nameEsc := jsonEscapeString(tc.Function.Name)
		structBlocks = append(structBlocks,
			fmt.Sprintf(`{"type":"tool_use","id":"%s","name":"%s","input":%s}`, idEsc, nameEsc, inputJSON))
	}

	structuredJSON := ""
	if len(structBlocks) > 0 {
		structuredJSON = "[" + strings.Join(structBlocks, ",") + "]"
	}

	var sb strings.Builder
	if finishReason == openai.FinishReasonToolCalls {
		sb.WriteString("STOP:tool_use\n")
		for _, tc := range toolCallEntries {
			safeArgs := strings.ReplaceAll(tc.args, "\n", `\n`)
			sb.WriteString(fmt.Sprintf("TOOL:%s:%s:%s\n", tc.id, tc.name, safeArgs))
		}
	} else {
		sb.WriteString("STOP:end_turn\n")
	}
	sb.WriteString(strings.Join(textParts, ""))

	return AskResponse{Response: sb.String(), StructuredJSON: structuredJSON, Tokens: totalTokens}, nil
}

// buildOpenAITools converts ToolDef slice to OpenAI SDK tool params.
func buildOpenAITools(defs []ToolDef) []openai.Tool {
	tools := make([]openai.Tool, 0, len(defs))
	for _, d := range defs {
		tools = append(tools, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        d.Name,
				Description: d.Description,
				Parameters:  d.InputSchema,
			},
		})
	}
	return tools
}

// StartStream begins streaming a response for the given prompt.
func (c *OpenAIClient) StartStream(ctx context.Context, prompt string) error {
	c.mu.Lock()
	if c.streaming {
		c.mu.Unlock()
		return fmt.Errorf("stream already in progress")
	}

	c.messages = append(c.messages, Message{Role: "user", Content: prompt})
	apiMsgs := buildChatMessages(c.systemPrompt, c.messages)
	model := c.model
	temp := c.temperature

	c.streaming = true
	c.streamChan = make(chan string, 100)
	c.streamDone = make(chan struct{})
	c.mu.Unlock()

	go func() {
		var fullResponse string

		defer func() {
			c.mu.Lock()
			if fullResponse != "" {
				c.messages = append(c.messages, Message{Role: "assistant", Content: fullResponse})
			}
			c.streaming = false
			close(c.streamChan)
			close(c.streamDone)
			c.mu.Unlock()
		}()

		chatReq := openai.ChatCompletionRequest{
			Model:         model,
			MaxTokens:     4096,
			Temperature:   float32(temp),
			Messages:      apiMsgs,
			StreamOptions: &openai.StreamOptions{IncludeUsage: true},
		}

		stream, err := c.client.CreateChatCompletionStream(ctx, chatReq)
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
		defer stream.Close()

		var inputTokens, outputTokens int

		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				select {
				case c.streamChan <- fmt.Sprintf("\n[Error: %v]", err):
				case <-ctx.Done():
				}
				if fullResponse == "" {
					c.mu.Lock()
					if len(c.messages) > 0 {
						c.messages = c.messages[:len(c.messages)-1]
					}
					c.mu.Unlock()
				}
				return
			}

			if chunk.Usage != nil {
				inputTokens = chunk.Usage.PromptTokens
				outputTokens = chunk.Usage.CompletionTokens
			}

			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				text := chunk.Choices[0].Delta.Content
				fullResponse += text
				select {
				case c.streamChan <- text:
				case <-ctx.Done():
					return
				}
			}
		}

		tokens := inputTokens + outputTokens
		if tokens == 0 {
			tokens = estimateTokens(fullResponse)
		}

		// Update token counts (fullResponse and messages handled in defer)
		c.mu.Lock()
		c.lastTokens = tokens
		c.totalTokens += tokens
		c.mu.Unlock()
	}()

	return nil
}

// ReadStreamChunk reads the next chunk from the stream.
func (c *OpenAIClient) ReadStreamChunk() (string, bool) {
	c.mu.RLock()
	streamChan := c.streamChan
	c.mu.RUnlock()

	if streamChan == nil {
		return "", false
	}

	chunk, ok := <-streamChan
	return chunk, ok
}

// IsStreaming returns whether a stream is currently in progress.
func (c *OpenAIClient) IsStreaming() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.streaming
}

// WaitStream waits for the current stream to complete.
func (c *OpenAIClient) WaitStream() {
	c.mu.RLock()
	done := c.streamDone
	c.mu.RUnlock()

	if done != nil {
		<-done
	}
}

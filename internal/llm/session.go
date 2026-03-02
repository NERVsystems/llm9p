// Package llm provides LLM backends for the 9P filesystem.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// SessionDefaults are copied to new sessions at creation time.
type SessionDefaults struct {
	Model          string
	Temperature    float64
	SystemPrompt   string
	ThinkingTokens int
	Prefill        string
}

// DefaultSessionDefaults returns sensible defaults for new sessions.
func DefaultSessionDefaults() SessionDefaults {
	return SessionDefaults{
		Model:          "claude-sonnet-4-5-20250929",
		Temperature:    0.7,
		SystemPrompt:   "",
		ThinkingTokens: 0,
		Prefill:        "",
	}
}

// Session holds ALL state for one session - fully independent (CSP).
// Each session is a complete, isolated unit with no shared mutable state.
type Session struct {
	ID           int
	messages     []Message
	lastResponse string
	lastTokens   int
	totalTokens  int

	// Per-session settings (no globals - CSP compliant)
	model          string
	temperature    float64
	systemPrompt   string
	thinkingTokens int
	prefill        string
	tools          []ToolDef // native tool definitions (nil = text-only mode)

	mu     sync.RWMutex
	closed bool
	refs   int32 // atomic reference count; session closed when it drops to 0

	// Async generation + streaming support
	streamCh chan string   // raw text chunks during generation; nil when idle
	doneCh   chan struct{} // closed when generation completes; nil when idle
	streamMu sync.Mutex    // guards streamCh and doneCh
}

// NewSession creates a new session with the given ID and defaults.
func NewSession(id int, defaults SessionDefaults) *Session {
	return &Session{
		ID:             id,
		messages:       make([]Message, 0),
		model:          defaults.Model,
		temperature:    defaults.Temperature,
		systemPrompt:   defaults.SystemPrompt,
		thinkingTokens: defaults.ThinkingTokens,
		prefill:        defaults.Prefill,
	}
}

// Messages returns a copy of the session's conversation history.
func (s *Session) Messages() []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]Message, len(s.messages))
	copy(result, s.messages)
	return result
}

// MessagesJSON returns the session's conversation history as JSON.
func (s *Session) MessagesJSON() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return json.MarshalIndent(s.messages, "", "  ")
}

// AddMessage adds a message to the session's history.
func (s *Session) AddMessage(role, content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.messages = append(s.messages, Message{Role: role, Content: content})
}

// SetLastResponse sets the last response for this session.
func (s *Session) SetLastResponse(response string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastResponse = response
}

// LastResponse returns the last response for this session.
func (s *Session) LastResponse() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastResponse
}

// TotalTokens returns cumulative token count for this session.
func (s *Session) TotalTokens() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.totalTokens
}

// AddTokens adds to the token counts for this session.
func (s *Session) AddTokens(tokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastTokens = tokens
	s.totalTokens += tokens
}

// EstimatedContextTokens returns a rough token estimate for context window usage.
// Uses 4 chars/token heuristic across all current messages.
// More accurate than totalTokens (which grows quadratically) for threshold decisions.
func (s *Session) EstimatedContextTokens() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	total := 0
	for _, msg := range s.messages {
		total += len(msg.Content) / 4
	}
	return total
}

// Reset clears the session's conversation history but keeps settings.
func (s *Session) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.messages = make([]Message, 0)
	s.lastResponse = ""
	s.lastTokens = 0
	s.totalTokens = 0
}

// Model returns the session's model setting.
func (s *Session) Model() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.model
}

// SetModel sets the session's model.
func (s *Session) SetModel(model string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.model = model
}

// Temperature returns the session's temperature setting.
func (s *Session) Temperature() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.temperature
}

// SetTemperature sets the session's temperature.
func (s *Session) SetTemperature(temp float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.temperature = temp
}

// SystemPrompt returns the session's system prompt.
func (s *Session) SystemPrompt() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.systemPrompt
}

// SetSystemPrompt sets the session's system prompt.
func (s *Session) SetSystemPrompt(prompt string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.systemPrompt = prompt
}

// ThinkingTokens returns the session's thinking token budget.
func (s *Session) ThinkingTokens() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.thinkingTokens
}

// SetThinkingTokens sets the session's thinking token budget.
func (s *Session) SetThinkingTokens(tokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.thinkingTokens = tokens
}

// Prefill returns the session's prefill string.
func (s *Session) Prefill() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.prefill
}

// SetPrefill sets the session's prefill string.
func (s *Session) SetPrefill(prefill string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prefill = prefill
}

// Tools returns the session's tool definitions (nil = text-only mode).
func (s *Session) Tools() []ToolDef {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.tools == nil {
		return nil
	}
	out := make([]ToolDef, len(s.tools))
	copy(out, s.tools)
	return out
}

// SetTools sets the tool definitions for this session.
// After setting, subsequent Ask calls will use native tool_use protocol.
func (s *Session) SetTools(tools []ToolDef) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools = tools
}

// AddStructuredMessage appends a message with optional structured content.
func (s *Session) AddStructuredMessage(role, content, structuredJSON string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.messages = append(s.messages, Message{
		Role:              role,
		Content:           content,
		StructuredContent: structuredJSON,
	})
}

// IsClosed returns whether the session has been closed.
func (s *Session) IsClosed() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.closed
}

// BeginGeneration allocates the stream channel and completion signal.
// Must be called before starting an async LLM generation.
func (s *Session) BeginGeneration() {
	s.streamMu.Lock()
	defer s.streamMu.Unlock()
	s.streamCh = make(chan string, 256)
	s.doneCh = make(chan struct{})
}

// EndGeneration closes the stream channel and completion signal.
// Called (via defer) when the async generation goroutine finishes.
//
// streamCh is closed but NOT set to nil: any buffered chunks remain readable
// from the closed channel even if the reader hasn't opened the file yet.
// BeginGeneration() will overwrite streamCh with a fresh channel next time.
func (s *Session) EndGeneration() {
	s.streamMu.Lock()
	ch := s.streamCh
	done := s.doneCh
	// Do NOT nil streamCh — leave closed channel readable for late-opening readers.
	s.doneCh = nil
	s.streamMu.Unlock()
	if ch != nil {
		close(ch)
	}
	if done != nil {
		close(done)
	}
}

// SendChunk sends a text chunk to the stream channel (non-blocking; drops if buffer full).
func (s *Session) SendChunk(text string) {
	s.streamMu.Lock()
	ch := s.streamCh
	s.streamMu.Unlock()
	if ch == nil {
		return
	}
	select {
	case ch <- text:
	default: // drop if buffer full
	}
}

// GetStreamCh returns the current stream channel, or nil if no generation is active.
func (s *Session) GetStreamCh() chan string {
	s.streamMu.Lock()
	defer s.streamMu.Unlock()
	return s.streamCh
}

// WaitDone blocks until the current generation completes, or returns immediately
// if no generation is in progress.
func (s *Session) WaitDone() {
	s.streamMu.Lock()
	done := s.doneCh
	s.streamMu.Unlock()
	if done != nil {
		<-done
	}
}

// SessionManager manages sessions and provides API access.
// The APIClient is stateless - all conversation state is in sessions.
type SessionManager struct {
	sessions  map[int]*Session
	nextID    int
	apiClient Backend         // Stateless API caller
	defaults  SessionDefaults // Defaults for new sessions
	mu        sync.RWMutex
}

// NewSessionManager creates a new session manager.
func NewSessionManager(apiClient Backend) *SessionManager {
	return &SessionManager{
		sessions:  make(map[int]*Session),
		nextID:    0,
		apiClient: apiClient,
		defaults:  DefaultSessionDefaults(),
	}
}

// SetDefaults sets the defaults for new sessions.
func (sm *SessionManager) SetDefaults(defaults SessionDefaults) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.defaults = defaults
}

// Create creates a new session and returns its ID.
func (sm *SessionManager) Create() int {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	id := sm.nextID
	sm.nextID++

	sm.sessions[id] = NewSession(id, sm.defaults)
	return id
}

// Get returns the session with the given ID, or nil if not found.
func (sm *SessionManager) Get(id int) *Session {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.sessions[id]
}

// Close closes and removes the session with the given ID.
func (sm *SessionManager) Close(id int) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	session, ok := sm.sessions[id]
	if !ok {
		return nil // Already closed
	}

	session.mu.Lock()
	session.closed = true
	session.mu.Unlock()

	delete(sm.sessions, id)
	return nil
}

// IncRef increments the reference count for the session with the given ID.
// Called when a 9P client opens a file inside the session directory.
func (sm *SessionManager) IncRef(id int) {
	session := sm.Get(id)
	if session != nil {
		atomic.AddInt32(&session.refs, 1)
	}
}

// DecRef decrements the reference count for the session with the given ID.
// When the count reaches zero the session is closed and removed, freeing
// all conversation history from memory.  Called when a 9P client clunks
// (closes) a file inside the session directory.
func (sm *SessionManager) DecRef(id int) {
	session := sm.Get(id)
	if session == nil {
		return
	}
	if atomic.AddInt32(&session.refs, -1) <= 0 {
		sm.Close(id) //nolint:errcheck
	}
}

// Reset clears the conversation history for the given session.
func (sm *SessionManager) Reset(id int) error {
	session := sm.Get(id)
	if session == nil {
		return nil
	}
	session.Reset()
	return nil
}

// Ask sends a prompt using the session's conversation history and settings.
// The response is stored in the session and returned.
func (sm *SessionManager) Ask(ctx context.Context, id int, prompt string) (string, error) {
	session := sm.Get(id)
	if session == nil {
		return "", ErrSessionNotFound
	}

	if session.IsClosed() {
		return "", ErrSessionClosed
	}

	// Get session settings
	session.mu.RLock()
	history := make([]Message, len(session.messages))
	copy(history, session.messages)
	model := session.model
	temperature := session.temperature
	systemPrompt := session.systemPrompt
	thinkingTokens := session.thinkingTokens
	prefill := session.prefill
	session.mu.RUnlock()

	// Build request with session's settings.
	// Enable streaming when a generation is already in progress (BeginGeneration was called).
	req := AskRequest{
		Messages:       history,
		Prompt:         prompt,
		Model:          model,
		Temperature:    temperature,
		SystemPrompt:   systemPrompt,
		ThinkingTokens: thinkingTokens,
		Prefill:        prefill,
		ToolDefs:       session.Tools(),
	}
	if session.GetStreamCh() != nil {
		req.StreamFunc = session.SendChunk
	}

	// Make API call (stateless)
	ar, err := sm.apiClient.AskWithRequest(ctx, req)
	if err != nil {
		if isContentFilterError(err) {
			// Content filtering: reset history and retry once with a clean slate.
			session.Reset()
			req.Messages = nil
			ar2, err2 := sm.apiClient.AskWithRequest(ctx, req)
			if err2 != nil {
				session.SetLastResponse("Error: " + err2.Error())
				return "", err2
			}
			ar = ar2
		} else {
			session.SetLastResponse("Error: " + err.Error())
			return "", err
		}
	}

	// Update session state — store structured content for tool turns
	textContent := extractTextContent(ar.Response)
	session.AddMessage("user", prompt)
	session.AddStructuredMessage("assistant", textContent, ar.StructuredJSON)
	session.AddTokens(ar.Tokens)
	session.SetLastResponse(ar.Response)

	return ar.Response, nil
}

// AskWithToolResults submits tool execution results as a user turn and gets
// the next assistant response. Called after parsing TOOL_RESULTS from the ask file.
func (sm *SessionManager) AskWithToolResults(ctx context.Context, id int, results []ToolResult) (string, error) {
	session := sm.Get(id)
	if session == nil {
		return "", ErrSessionNotFound
	}
	if session.IsClosed() {
		return "", ErrSessionClosed
	}

	session.mu.RLock()
	history := make([]Message, len(session.messages))
	copy(history, session.messages)
	model := session.model
	temperature := session.temperature
	systemPrompt := session.systemPrompt
	thinkingTokens := session.thinkingTokens
	session.mu.RUnlock()

	req := AskRequest{
		Messages:       history,
		Model:          model,
		Temperature:    temperature,
		SystemPrompt:   systemPrompt,
		ThinkingTokens: thinkingTokens,
		ToolDefs:       session.Tools(),
		ToolResults:    results,
		// Prompt is intentionally empty — tool results ARE the new user turn.
		// Prefill is intentionally empty — prefill is inappropriate mid-tool-loop.
	}
	if session.GetStreamCh() != nil {
		req.StreamFunc = session.SendChunk
	}

	ar, err := sm.apiClient.AskWithRequest(ctx, req)
	if err != nil {
		if isContentFilterError(err) {
			// Content filtering on a tool-result turn: the offending content is
			// in the tool results. Reset history and return a synthetic end_turn
			// so the agent can surface a message rather than hard-crashing.
			session.Reset()
			synthetic := "STOP:end_turn\nContent filtering policy blocked a tool result. The session history has been reset. Please try a different approach or rephrase your request."
			session.SetLastResponse(synthetic)
			return synthetic, nil
		}
		session.SetLastResponse("Error: " + err.Error())
		return "", err
	}

	// Store tool_result user turn + assistant response in history
	toolResultsText := fmt.Sprintf("tool results: %d results submitted", len(results))
	toolResultsJSON := buildToolResultsJSON(results)
	session.AddStructuredMessage("user", toolResultsText, toolResultsJSON)
	textContent := extractTextContent(ar.Response)
	session.AddStructuredMessage("assistant", textContent, ar.StructuredJSON)
	session.AddTokens(ar.Tokens)
	session.SetLastResponse(ar.Response)

	return ar.Response, nil
}

// extractTextContent extracts the plain text from a STOP:-formatted response.
// For plain-text (no-tools) responses, returns the response as-is.
func extractTextContent(response string) string {
	if !strings.HasPrefix(response, "STOP:") {
		return response
	}
	// Skip STOP: line and TOOL: lines, return the text portion
	lines := strings.SplitN(response, "\n", -1)
	var text []string
	for _, line := range lines {
		if strings.HasPrefix(line, "STOP:") || strings.HasPrefix(line, "TOOL:") {
			continue
		}
		text = append(text, line)
	}
	return strings.Join(text, "\n")
}

// buildToolResultsJSON builds the JSON content blocks for a tool_results user turn.
// Stored in Message.StructuredContent for proper API replay.
func buildToolResultsJSON(results []ToolResult) string {
	if len(results) == 0 {
		return ""
	}
	var parts []string
	for _, r := range results {
		content := strings.ReplaceAll(r.Content, `\`, `\\`)
		content = strings.ReplaceAll(content, `"`, `\"`)
		content = strings.ReplaceAll(content, "\n", `\n`)
		content = strings.ReplaceAll(content, "\r", `\r`)
		toolUseID := strings.ReplaceAll(r.ToolUseID, `"`, `\"`)
		parts = append(parts, fmt.Sprintf(`{"type":"tool_result","tool_use_id":"%s","content":"%s"}`, toolUseID, content))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// ListSessions returns the IDs of all active sessions.
func (sm *SessionManager) ListSessions() []int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	ids := make([]int, 0, len(sm.sessions))
	for id := range sm.sessions {
		ids = append(ids, id)
	}
	return ids
}

// EstimatedContextTokens returns the estimated token count for a session.
// Delegates to Session.EstimatedContextTokens().
func (sm *SessionManager) EstimatedContextTokens(id int) int {
	session := sm.Get(id)
	if session == nil {
		return 0
	}
	return session.EstimatedContextTokens()
}

// ContextLimit returns the context window limit (200K for all Claude models).
func (sm *SessionManager) ContextLimit() int {
	return 200000
}

// Compact summarizes a session's conversation to reduce context window usage.
// The conversation history is replaced with a compact summary exchange.
// No-op if the session has fewer than 4 messages (nothing meaningful to compact).
func (sm *SessionManager) Compact(ctx context.Context, id int) error {
	session := sm.Get(id)
	if session == nil {
		return ErrSessionNotFound
	}

	session.mu.RLock()
	msgs := make([]Message, len(session.messages))
	copy(msgs, session.messages)
	model := session.model
	session.mu.RUnlock()

	if len(msgs) < 4 {
		return nil
	}

	// Build conversation text for the summarization prompt
	var sb strings.Builder
	for _, msg := range msgs {
		if msg.Role == "system" {
			continue
		}
		sb.WriteString(msg.Role)
		sb.WriteString(": ")
		sb.WriteString(msg.Content)
		sb.WriteString("\n\n")
	}

	req := AskRequest{
		Prompt:      "Summarize this conversation concisely, preserving key facts, decisions, file paths, code snippets, and all context needed to continue the work:\n\n" + sb.String(),
		Model:       model,
		Temperature: 0.3,
	}

	ar, err := sm.apiClient.AskWithRequest(ctx, req)
	if err != nil {
		return fmt.Errorf("compaction LLM call failed: %w", err)
	}

	// Replace history with a minimal exchange conveying the summary
	session.mu.Lock()
	session.messages = []Message{
		{Role: "user", Content: "Context from earlier in this session:\n" + ar.Response},
		{Role: "assistant", Content: "Understood. I have the context from our previous work and will continue from there."},
	}
	session.totalTokens = ar.Tokens
	session.mu.Unlock()

	return nil
}

// AskRequest contains all parameters for an API call.
type AskRequest struct {
	Messages       []Message
	Prompt         string // empty when ToolResults is set (tool_results IS the new user turn)
	Model          string
	Temperature    float64
	SystemPrompt   string
	ThinkingTokens int
	Prefill        string
	ToolDefs       []ToolDef    // non-nil enables native tool_use protocol
	ToolResults    []ToolResult // non-nil: submit tool results as a new user turn
	StreamFunc     func(string) // optional; called for each text_delta chunk during streaming
}

// isContentFilterError returns true when the API rejected the request due to
// Anthropic's content filtering policy. The session history should be reset
// before retrying in this case.
func isContentFilterError(err error) bool {
	return strings.Contains(err.Error(), "content filtering policy")
}

// Errors
type SessionError string

func (e SessionError) Error() string { return string(e) }

const (
	ErrSessionNotFound SessionError = "session not found"
	ErrSessionClosed   SessionError = "session closed"
)

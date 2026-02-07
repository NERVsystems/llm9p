// Package llm provides LLM backends for the 9P filesystem.
package llm

import (
	"context"
	"encoding/json"
	"sync"
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
		Model:          "claude-sonnet-4-20250514",
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

	mu     sync.RWMutex
	closed bool
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

// IsClosed returns whether the session has been closed.
func (s *Session) IsClosed() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.closed
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

	// Build request with session's settings
	req := AskRequest{
		Messages:       history,
		Prompt:         prompt,
		Model:          model,
		Temperature:    temperature,
		SystemPrompt:   systemPrompt,
		ThinkingTokens: thinkingTokens,
		Prefill:        prefill,
	}

	// Make API call (stateless)
	response, tokens, err := sm.apiClient.AskWithRequest(ctx, req)
	if err != nil {
		session.SetLastResponse("Error: " + err.Error())
		return "", err
	}

	// Update session state
	session.AddMessage("user", prompt)
	session.AddMessage("assistant", response)
	session.AddTokens(tokens)
	session.SetLastResponse(response)

	return response, nil
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

// AskRequest contains all parameters for an API call.
type AskRequest struct {
	Messages       []Message
	Prompt         string
	Model          string
	Temperature    float64
	SystemPrompt   string
	ThinkingTokens int
	Prefill        string
}

// Errors
type SessionError string

func (e SessionError) Error() string { return string(e) }

const (
	ErrSessionNotFound SessionError = "session not found"
	ErrSessionClosed   SessionError = "session closed"
)

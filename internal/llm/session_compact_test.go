package llm

import (
	"context"
	"testing"
)

// mockAPIClient is a minimal Backend for testing SessionManager.Compact.
type mockAPIClient struct {
	askResponse string
	askTokens   int
	askError    error
}

func (m *mockAPIClient) Model() string                 { return "claude-sonnet-4-20250514" }
func (m *mockAPIClient) SetModel(string)               {}
func (m *mockAPIClient) Temperature() float64          { return 0.7 }
func (m *mockAPIClient) SetTemperature(float64) error  { return nil }
func (m *mockAPIClient) SystemPrompt() string          { return "" }
func (m *mockAPIClient) SetSystemPrompt(string)        {}
func (m *mockAPIClient) ThinkingTokens() int           { return 0 }
func (m *mockAPIClient) SetThinkingTokens(int)         {}
func (m *mockAPIClient) Prefill() string               { return "" }
func (m *mockAPIClient) SetPrefill(string)             {}
func (m *mockAPIClient) LastTokens() int               { return m.askTokens }
func (m *mockAPIClient) TotalTokens() int              { return 0 }
func (m *mockAPIClient) ContextLimit() int             { return 200000 }
func (m *mockAPIClient) Compact(context.Context) error { return nil }
func (m *mockAPIClient) Messages() []Message           { return nil }
func (m *mockAPIClient) MessagesJSON() ([]byte, error) { return []byte("[]"), nil }
func (m *mockAPIClient) AddSystemMessage(string)       {}
func (m *mockAPIClient) Reset()                        {}
func (m *mockAPIClient) Ask(_ context.Context, _ string) (string, error) {
	return m.askResponse, m.askError
}
func (m *mockAPIClient) AskWithHistory(_ context.Context, _ []Message, _ string) (string, int, error) {
	return m.askResponse, m.askTokens, m.askError
}
func (m *mockAPIClient) AskWithRequest(_ context.Context, req AskRequest) (AskResponse, error) {
	if m.askError != nil {
		return AskResponse{}, m.askError
	}
	return AskResponse{Response: m.askResponse, Tokens: m.askTokens}, nil
}
func (m *mockAPIClient) StartStream(context.Context, string) error { return nil }
func (m *mockAPIClient) ReadStreamChunk() (string, bool)           { return "", false }
func (m *mockAPIClient) IsStreaming() bool                         { return false }
func (m *mockAPIClient) WaitStream()                               {}

var _ Backend = (*mockAPIClient)(nil)

// ---- EstimatedContextTokens ----

func TestSession_EstimatedContextTokens_Empty(t *testing.T) {
	s := NewSession(0, DefaultSessionDefaults())
	if got := s.EstimatedContextTokens(); got != 0 {
		t.Errorf("empty session: got %d, want 0", got)
	}
}

func TestSession_EstimatedContextTokens_Counts(t *testing.T) {
	s := NewSession(0, DefaultSessionDefaults())
	// 400 chars user + 200 chars assistant = 600 chars → 150 tokens
	s.AddMessage("user", string(make([]byte, 400)))
	s.AddMessage("assistant", string(make([]byte, 200)))
	got := s.EstimatedContextTokens()
	want := 150 // (400 + 200) / 4
	if got != want {
		t.Errorf("EstimatedContextTokens() = %d, want %d", got, want)
	}
}

// ---- SessionManager.Compact ----

func TestSessionManager_Compact_NotFound(t *testing.T) {
	sm := NewSessionManager(&mockAPIClient{askResponse: "summary"})
	err := sm.Compact(context.Background(), 999)
	if err != ErrSessionNotFound {
		t.Errorf("Compact(unknown) = %v, want ErrSessionNotFound", err)
	}
}

func TestSessionManager_Compact_TooShort(t *testing.T) {
	api := &mockAPIClient{askResponse: "summary", askTokens: 50}
	sm := NewSessionManager(api)
	id := sm.Create()

	// 2 messages — below the "< 4" threshold
	session := sm.Get(id)
	session.AddMessage("user", "hello")
	session.AddMessage("assistant", "hi")

	if err := sm.Compact(context.Background(), id); err != nil {
		t.Fatalf("Compact() error: %v", err)
	}
	// Messages should be unchanged (compaction skipped)
	msgs := session.Messages()
	if len(msgs) != 2 {
		t.Errorf("short session: messages = %d, want 2 (no compaction)", len(msgs))
	}
}

func TestSessionManager_Compact_ReplacesMessages(t *testing.T) {
	api := &mockAPIClient{askResponse: "This is a summary.", askTokens: 300}
	sm := NewSessionManager(api)
	id := sm.Create()

	session := sm.Get(id)
	for i := 0; i < 3; i++ {
		session.AddMessage("user", "question "+string(rune('A'+i)))
		session.AddMessage("assistant", "answer "+string(rune('A'+i)))
	}
	session.AddTokens(10000)

	if err := sm.Compact(context.Background(), id); err != nil {
		t.Fatalf("Compact() error: %v", err)
	}

	msgs := session.Messages()
	// Should be exactly 2 messages: context exchange
	if len(msgs) != 2 {
		t.Errorf("after compact: messages = %d, want 2", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("msgs[0].Role = %q, want 'user'", msgs[0].Role)
	}
	if msgs[1].Role != "assistant" {
		t.Errorf("msgs[1].Role = %q, want 'assistant'", msgs[1].Role)
	}
	// Summary should appear in first message
	if got := msgs[0].Content; len(got) == 0 {
		t.Error("msgs[0].Content is empty after compaction")
	}
}

func TestSessionManager_Compact_ResetsTokens(t *testing.T) {
	api := &mockAPIClient{askResponse: "summary", askTokens: 400}
	sm := NewSessionManager(api)
	id := sm.Create()

	session := sm.Get(id)
	for i := 0; i < 3; i++ {
		session.AddMessage("user", "msg")
		session.AddMessage("assistant", "reply")
	}
	session.AddTokens(50000)

	if err := sm.Compact(context.Background(), id); err != nil {
		t.Fatalf("Compact() error: %v", err)
	}

	// totalTokens should be reset to what the compaction LLM call returned
	if got := session.TotalTokens(); got != 400 {
		t.Errorf("TotalTokens after compact = %d, want 400", got)
	}
}

func TestSessionManager_ContextLimit(t *testing.T) {
	sm := NewSessionManager(&mockAPIClient{})
	if got := sm.ContextLimit(); got != 200000 {
		t.Errorf("ContextLimit() = %d, want 200000", got)
	}
}

func TestSessionManager_EstimatedContextTokens(t *testing.T) {
	sm := NewSessionManager(&mockAPIClient{})
	id := sm.Create()
	session := sm.Get(id)

	// 800 chars / 4 = 200 tokens
	session.AddMessage("user", string(make([]byte, 800)))

	got := sm.EstimatedContextTokens(id)
	if got != 200 {
		t.Errorf("EstimatedContextTokens() = %d, want 200", got)
	}
}

func TestSessionManager_EstimatedContextTokens_NotFound(t *testing.T) {
	sm := NewSessionManager(&mockAPIClient{})
	if got := sm.EstimatedContextTokens(999); got != 0 {
		t.Errorf("EstimatedContextTokens(unknown) = %d, want 0", got)
	}
}

package llmfs

import (
	"context"
	"fmt"

	"github.com/NERVsystems/llm9p/internal/llm"
)

// MockBackend implements llm.Backend for testing
type MockBackend struct {
	model          string
	temperature    float64
	systemPrompt   string
	prefill        string
	messages       []llm.Message
	lastTokens     int
	totalTokens    int
	contextLimit   int
	thinkingTokens int
	compactCalled  bool
	compactError   error
	askResponse    string
	askError       error
}

func NewMockBackend() *MockBackend {
	return &MockBackend{
		model:        "mock-model",
		temperature:  0.7,
		contextLimit: 200000,
		messages:     make([]llm.Message, 0),
	}
}

func (m *MockBackend) Model() string                  { return m.model }
func (m *MockBackend) SetModel(model string)          { m.model = model }
func (m *MockBackend) Temperature() float64           { return m.temperature }
func (m *MockBackend) SetTemperature(temp float64) error {
	if temp < 0 || temp > 2 {
		return fmt.Errorf("invalid temperature")
	}
	m.temperature = temp
	return nil
}
func (m *MockBackend) SystemPrompt() string          { return m.systemPrompt }
func (m *MockBackend) SetSystemPrompt(prompt string) { m.systemPrompt = prompt }
func (m *MockBackend) ThinkingTokens() int           { return m.thinkingTokens }
func (m *MockBackend) SetThinkingTokens(tokens int)  { m.thinkingTokens = tokens }
func (m *MockBackend) Prefill() string               { return m.prefill }
func (m *MockBackend) SetPrefill(prefill string)     { m.prefill = prefill }
func (m *MockBackend) LastTokens() int               { return m.lastTokens }
func (m *MockBackend) TotalTokens() int              { return m.totalTokens }
func (m *MockBackend) ContextLimit() int             { return m.contextLimit }

func (m *MockBackend) Compact(ctx context.Context) error {
	m.compactCalled = true
	if m.compactError != nil {
		return m.compactError
	}
	// Simulate compaction - reduce tokens
	m.totalTokens = m.totalTokens / 4
	m.messages = []llm.Message{{Role: "system", Content: "compacted summary"}}
	return nil
}

func (m *MockBackend) Messages() []llm.Message {
	result := make([]llm.Message, len(m.messages))
	copy(result, m.messages)
	return result
}

func (m *MockBackend) MessagesJSON() ([]byte, error) {
	return []byte("[]"), nil
}

func (m *MockBackend) AddSystemMessage(content string) {
	m.messages = append([]llm.Message{{Role: "system", Content: content}}, m.messages...)
}

func (m *MockBackend) Reset() {
	m.messages = make([]llm.Message, 0)
	m.lastTokens = 0
	m.totalTokens = 0
}

func (m *MockBackend) Ask(ctx context.Context, prompt string) (string, error) {
	if m.askError != nil {
		return "", m.askError
	}
	m.messages = append(m.messages, llm.Message{Role: "user", Content: prompt})
	m.messages = append(m.messages, llm.Message{Role: "assistant", Content: m.askResponse})
	m.lastTokens = len(prompt) + len(m.askResponse)
	m.totalTokens += m.lastTokens
	return m.askResponse, nil
}

func (m *MockBackend) AskWithHistory(ctx context.Context, history []llm.Message, prompt string) (string, int, error) {
	if m.askError != nil {
		return "", 0, m.askError
	}
	tokens := len(prompt) + len(m.askResponse)
	return m.askResponse, tokens, nil
}

func (m *MockBackend) AskWithRequest(ctx context.Context, req llm.AskRequest) (string, int, error) {
	if m.askError != nil {
		return "", 0, m.askError
	}
	tokens := len(req.Prompt) + len(m.askResponse)
	return m.askResponse, tokens, nil
}

func (m *MockBackend) StartStream(ctx context.Context, prompt string) error {
	return fmt.Errorf("streaming not implemented in mock")
}

func (m *MockBackend) ReadStreamChunk() (string, bool) {
	return "", false
}

func (m *MockBackend) IsStreaming() bool {
	return false
}

func (m *MockBackend) WaitStream() {}

// Verify MockBackend implements Backend
var _ llm.Backend = (*MockBackend)(nil)

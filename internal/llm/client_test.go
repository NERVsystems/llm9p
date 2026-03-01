package llm

import (
	"testing"
)

func TestContextLimitForModel(t *testing.T) {
	tests := []struct {
		model    string
		expected int
	}{
		{"claude-3-opus-20240229", 200000},
		{"claude-3-sonnet-20240229", 200000},
		{"claude-3-haiku-20240307", 200000},
		{"claude-sonnet-4-20250514", 200000},
		{"CLAUDE-3-OPUS", 200000},     // case insensitive
		{"some-sonnet-model", 200000}, // substring match
		{"unknown-model", 200000},     // default
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			got := contextLimitForModel(tc.model)
			if got != tc.expected {
				t.Errorf("contextLimitForModel(%q) = %d, want %d", tc.model, got, tc.expected)
			}
		})
	}
}

func TestClientReset(t *testing.T) {
	// Create a client with dummy API key (won't make real calls)
	c := NewClient("dummy-key")

	// Manually set some state
	c.messages = []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	c.lastTokens = 100
	c.totalTokens = 500

	// Reset should clear everything
	c.Reset()

	if len(c.messages) != 0 {
		t.Errorf("Reset() should clear messages, got %d", len(c.messages))
	}
	if c.lastTokens != 0 {
		t.Errorf("Reset() should clear lastTokens, got %d", c.lastTokens)
	}
	if c.totalTokens != 0 {
		t.Errorf("Reset() should clear totalTokens, got %d", c.totalTokens)
	}
}

func TestClientTotalTokens(t *testing.T) {
	c := NewClient("dummy-key")

	// Initially should be 0
	if got := c.TotalTokens(); got != 0 {
		t.Errorf("TotalTokens() = %d, want 0", got)
	}

	// Manually set for testing
	c.totalTokens = 12345
	if got := c.TotalTokens(); got != 12345 {
		t.Errorf("TotalTokens() = %d, want 12345", got)
	}
}

func TestClientContextLimit(t *testing.T) {
	c := NewClient("dummy-key")

	// Default model should have 200K limit
	got := c.ContextLimit()
	if got != 200000 {
		t.Errorf("ContextLimit() = %d, want 200000", got)
	}

	// Change model and verify
	c.SetModel("claude-3-haiku-20240307")
	got = c.ContextLimit()
	if got != 200000 {
		t.Errorf("ContextLimit() for haiku = %d, want 200000", got)
	}
}

func TestClientTemperature(t *testing.T) {
	c := NewClient("dummy-key")

	// Default temperature
	if got := c.Temperature(); got != 0.7 {
		t.Errorf("Temperature() = %f, want 0.7", got)
	}

	// Valid temperature
	if err := c.SetTemperature(1.5); err != nil {
		t.Errorf("SetTemperature(1.5) error: %v", err)
	}
	if got := c.Temperature(); got != 1.5 {
		t.Errorf("Temperature() = %f, want 1.5", got)
	}

	// Invalid temperature - too low
	if err := c.SetTemperature(-0.1); err == nil {
		t.Error("SetTemperature(-0.1) should return error")
	}

	// Invalid temperature - too high
	if err := c.SetTemperature(2.1); err == nil {
		t.Error("SetTemperature(2.1) should return error")
	}
}

func TestClientSystemPrompt(t *testing.T) {
	c := NewClient("dummy-key")

	// Initially empty
	if got := c.SystemPrompt(); got != "" {
		t.Errorf("SystemPrompt() = %q, want empty", got)
	}

	// Set and verify
	c.SetSystemPrompt("You are a helpful assistant")
	if got := c.SystemPrompt(); got != "You are a helpful assistant" {
		t.Errorf("SystemPrompt() = %q, want 'You are a helpful assistant'", got)
	}
}

func TestClientMessages(t *testing.T) {
	c := NewClient("dummy-key")

	// Initially empty
	if msgs := c.Messages(); len(msgs) != 0 {
		t.Errorf("Messages() should be empty, got %d", len(msgs))
	}

	// Add messages
	c.messages = []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}

	msgs := c.Messages()
	if len(msgs) != 2 {
		t.Errorf("Messages() = %d messages, want 2", len(msgs))
	}

	// Verify it's a copy (modification doesn't affect original)
	msgs[0].Content = "modified"
	if c.messages[0].Content == "modified" {
		t.Error("Messages() should return a copy, not the original")
	}
}

func TestClientAddSystemMessage(t *testing.T) {
	c := NewClient("dummy-key")

	// Add system message
	c.AddSystemMessage("Context info here")

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("Messages() = %d messages, want 1", len(msgs))
	}

	if msgs[0].Role != "system" {
		t.Errorf("Message role = %q, want 'system'", msgs[0].Role)
	}
	if msgs[0].Content != "Context info here" {
		t.Errorf("Message content = %q, want 'Context info here'", msgs[0].Content)
	}
}

func TestClientModel(t *testing.T) {
	c := NewClient("dummy-key")

	// Default model
	if got := c.Model(); got != "claude-sonnet-4-20250514" {
		t.Errorf("Model() = %q, want 'claude-sonnet-4-20250514'", got)
	}

	// Change model
	c.SetModel("claude-3-haiku-20240307")
	if got := c.Model(); got != "claude-3-haiku-20240307" {
		t.Errorf("Model() = %q, want 'claude-3-haiku-20240307'", got)
	}
}

func TestClientMessagesJSON(t *testing.T) {
	c := NewClient("dummy-key")

	c.messages = []Message{
		{Role: "user", Content: "hello"},
	}

	data, err := c.MessagesJSON()
	if err != nil {
		t.Fatalf("MessagesJSON() error: %v", err)
	}

	// Should contain the message content
	json := string(data)
	if !contains(json, "hello") || !contains(json, "user") {
		t.Errorf("MessagesJSON() = %s, should contain 'hello' and 'user'", json)
	}
}

func TestClientIsStreaming(t *testing.T) {
	c := NewClient("dummy-key")

	// Initially not streaming
	if c.IsStreaming() {
		t.Error("IsStreaming() should be false initially")
	}
}

// Helper function
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

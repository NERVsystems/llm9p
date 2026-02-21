package llmfs

import (
	"io"
	"strings"
	"testing"

	"github.com/NERVsystems/llm9p/internal/llm"
)

// newTestSession creates a SessionManager with a mock backend and a fresh session.
// Returns the manager and the session ID.
func newTestSession() (*llm.SessionManager, int) {
	mock := NewMockBackend()
	mock.askResponse = "This is a compact summary of the conversation."
	sm := llm.NewSessionManager(mock)
	id := sm.Create()
	return sm, id
}

// ---- SessionCompactFile ----

func TestSessionCompactFile_Read(t *testing.T) {
	sm, id := newTestSession()
	f := NewSessionCompactFile(sm, id)

	buf := make([]byte, 100)
	n, err := f.Read(buf, 0)
	if err != nil {
		t.Fatalf("Read() error: %v", err)
	}
	content := string(buf[:n])
	if content == "" {
		t.Error("Read() returned empty content")
	}
}

func TestSessionCompactFile_Read_EOF(t *testing.T) {
	sm, id := newTestSession()
	f := NewSessionCompactFile(sm, id)

	buf := make([]byte, 100)
	n, err := f.Read(buf, 10000)
	if err != io.EOF {
		t.Errorf("Read(large offset) = %v, want io.EOF", err)
	}
	if n != 0 {
		t.Errorf("Read(large offset) n = %d, want 0", n)
	}
}

func TestSessionCompactFile_Write_NoOp_ShortHistory(t *testing.T) {
	sm, id := newTestSession()
	session := sm.Get(id)
	session.AddMessage("user", "hello")
	session.AddMessage("assistant", "hi")

	f := NewSessionCompactFile(sm, id)
	n, err := f.Write([]byte("compact"), 0)
	if err != nil {
		t.Fatalf("Write() error: %v", err)
	}
	if n != 7 {
		t.Errorf("Write() n = %d, want 7", n)
	}
	// Session with < 4 messages: compaction is a no-op (messages unchanged)
	msgs := session.Messages()
	if len(msgs) != 2 {
		t.Errorf("after no-op compact: messages = %d, want 2", len(msgs))
	}
}

func TestSessionCompactFile_Write_Compacts(t *testing.T) {
	sm, id := newTestSession()
	session := sm.Get(id)
	// Add 3 turns (6 messages) to pass the threshold
	for i := 0; i < 3; i++ {
		session.AddMessage("user", "question")
		session.AddMessage("assistant", "answer")
	}

	f := NewSessionCompactFile(sm, id)
	_, err := f.Write([]byte("compact"), 0)
	if err != nil {
		t.Fatalf("Write() error: %v", err)
	}

	// After compaction messages should be condensed
	msgs := session.Messages()
	if len(msgs) >= 6 {
		t.Errorf("after compact: messages = %d, expected fewer than 6", len(msgs))
	}
}

func TestSessionCompactFile_Stat(t *testing.T) {
	sm, id := newTestSession()
	f := NewSessionCompactFile(sm, id)
	stat := f.Stat()
	if stat.Length == 0 {
		t.Error("Stat().Length should be non-zero")
	}
}

// ---- SessionUsageFile ----

func TestSessionUsageFile_Read_Format(t *testing.T) {
	sm, id := newTestSession()
	session := sm.Get(id)
	// 400 chars → 100 estimated tokens
	session.AddMessage("user", strings.Repeat("x", 400))

	f := NewSessionUsageFile(sm, id)
	buf := make([]byte, 64)
	n, err := f.Read(buf, 0)
	if err != nil {
		t.Fatalf("Read() error: %v", err)
	}

	content := string(buf[:n])
	// Should be "100/200000\n"
	if content != "100/200000\n" {
		t.Errorf("Read() = %q, want %q", content, "100/200000\n")
	}
}

func TestSessionUsageFile_Read_EOF(t *testing.T) {
	sm, id := newTestSession()
	f := NewSessionUsageFile(sm, id)

	buf := make([]byte, 64)
	n, err := f.Read(buf, 10000)
	if err != io.EOF {
		t.Errorf("Read(large offset) = %v, want io.EOF", err)
	}
	if n != 0 {
		t.Errorf("Read(large offset) n = %d, want 0", n)
	}
}

func TestSessionUsageFile_Write_ReadOnly(t *testing.T) {
	sm, id := newTestSession()
	f := NewSessionUsageFile(sm, id)

	n, err := f.Write([]byte("anything"), 0)
	if err == nil {
		t.Error("Write() should return error (read-only)")
	}
	if n != 0 {
		t.Errorf("Write() n = %d, want 0", n)
	}
}

func TestSessionUsageFile_Stat_Length(t *testing.T) {
	sm, id := newTestSession()
	f := NewSessionUsageFile(sm, id)
	stat := f.Stat()
	// "0/200000\n" = 9 chars
	if stat.Length != 9 {
		t.Errorf("Stat().Length = %d, want 9 (for '0/200000\\n')", stat.Length)
	}
}

func TestSessionUsageFile_Reflects_Content(t *testing.T) {
	sm, id := newTestSession()
	session := sm.Get(id)
	f := NewSessionUsageFile(sm, id)

	buf := make([]byte, 64)

	// Initially empty
	n, _ := f.Read(buf, 0)
	if string(buf[:n]) != "0/200000\n" {
		t.Errorf("initial: got %q, want '0/200000\\n'", string(buf[:n]))
	}

	// Add 800 chars → 200 estimated tokens
	session.AddMessage("user", strings.Repeat("a", 800))

	n, _ = f.Read(buf, 0)
	if string(buf[:n]) != "200/200000\n" {
		t.Errorf("after add: got %q, want '200/200000\\n'", string(buf[:n]))
	}
}

// ---- SessionDir includes compact and usage ----

func TestSessionDir_HasCompactAndUsage(t *testing.T) {
	sm, id := newTestSession()
	dir := NewSessionDir(sm, id)

	children := dir.Children()
	names := make(map[string]bool)
	for _, f := range children {
		names[f.Stat().Name] = true
	}

	for _, want := range []string{"compact", "usage"} {
		if !names[want] {
			t.Errorf("SessionDir.Children() missing %q", want)
		}
	}
}

func TestSessionDir_LookupCompact(t *testing.T) {
	sm, id := newTestSession()
	dir := NewSessionDir(sm, id)

	f, err := dir.Lookup("compact")
	if err != nil {
		t.Fatalf("Lookup('compact') error: %v", err)
	}
	if _, ok := f.(*SessionCompactFile); !ok {
		t.Errorf("Lookup('compact') returned %T, want *SessionCompactFile", f)
	}
}

func TestSessionDir_LookupUsage(t *testing.T) {
	sm, id := newTestSession()
	dir := NewSessionDir(sm, id)

	f, err := dir.Lookup("usage")
	if err != nil {
		t.Fatalf("Lookup('usage') error: %v", err)
	}
	if _, ok := f.(*SessionUsageFile); !ok {
		t.Errorf("Lookup('usage') returned %T, want *SessionUsageFile", f)
	}
}

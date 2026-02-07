package llmfs

import (
	"context"
	"io"
	"log"
	"strings"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionAskFile is the ask file for a specific session: /n/llm/N/ask
// Write a prompt, read the response.
type SessionAskFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionAskFile creates an ask file for the given session.
func NewSessionAskFile(sm *llm.SessionManager, id int) *SessionAskFile {
	return &SessionAskFile{
		BaseFile: protocol.NewBaseFile("ask", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the last response from this session.
func (f *SessionAskFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	content := session.LastResponse()
	if content != "" && !strings.HasSuffix(content, "\n") {
		content += "\n"
	}

	if offset >= int64(len(content)) {
		return 0, io.EOF
	}

	n := copy(p, content[offset:])
	return n, nil
}

// Write sends a prompt to the LLM using this session's settings.
func (f *SessionAskFile) Write(p []byte, offset int64) (int, error) {
	log.Printf("llm9p: SessionAskFile.Write session=%d len=%d", f.id, len(p))

	prompt := strings.TrimSpace(string(p))
	if prompt == "" {
		return len(p), nil // Empty write is a no-op
	}

	log.Printf("llm9p: SessionAskFile.Write prompt: %s", prompt[:min(len(prompt), 50)])

	ctx := context.Background()
	response, err := f.sm.Ask(ctx, f.id, prompt)
	if err != nil {
		log.Printf("llm9p: SessionAskFile.Write error: %v", err)
		// Error is stored in session.LastResponse by SessionManager
		return len(p), nil // Return success so client knows write completed
	}

	log.Printf("llm9p: SessionAskFile.Write success, response len=%d", len(response))
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionAskFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	// Length is dynamic based on last response
	session := f.sm.Get(f.id)
	if session != nil {
		s.Length = uint64(len(session.LastResponse()))
	}
	return s
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

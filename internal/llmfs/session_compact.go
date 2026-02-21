package llmfs

import (
	"context"
	"io"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionCompactFile triggers conversation compaction for a session: /n/llm/N/compact
// Write any content to trigger; read returns "ok\n" or an error message.
type SessionCompactFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionCompactFile creates a compact trigger file for the given session.
func NewSessionCompactFile(sm *llm.SessionManager, id int) *SessionCompactFile {
	return &SessionCompactFile{
		BaseFile: protocol.NewBaseFile("compact", 0644),
		sm:       sm,
		id:       id,
	}
}

// Read returns a status line.
func (f *SessionCompactFile) Read(p []byte, offset int64) (int, error) {
	content := "write to compact conversation\n"
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write triggers compaction of the session's conversation history.
func (f *SessionCompactFile) Write(p []byte, offset int64) (int, error) {
	if err := f.sm.Compact(context.Background(), f.id); err != nil {
		return 0, protocol.Error("compact: " + err.Error())
	}
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionCompactFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	s.Length = uint64(len("write to compact conversation\n"))
	return s
}

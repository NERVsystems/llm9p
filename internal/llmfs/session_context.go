package llmfs

import (
	"io"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionContextFile exposes the conversation history: /n/llm/N/context
// Read returns JSON of the conversation history.
type SessionContextFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionContextFile creates a context file for the given session.
func NewSessionContextFile(sm *llm.SessionManager, id int) *SessionContextFile {
	return &SessionContextFile{
		BaseFile: protocol.NewBaseFile("context", 0444),
		sm:       sm,
		id:       id,
	}
}

// Read returns the conversation history as JSON.
func (f *SessionContextFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	content, err := session.MessagesJSON()
	if err != nil {
		return 0, err
	}
	// Add newline
	content = append(content, '\n')

	if offset >= int64(len(content)) {
		return 0, io.EOF
	}

	n := copy(p, content[offset:])
	return n, nil
}

// Stat returns the file's metadata.
func (f *SessionContextFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	session := f.sm.Get(f.id)
	if session != nil {
		content, err := session.MessagesJSON()
		if err == nil {
			s.Length = uint64(len(content) + 1)
		}
	}
	return s
}

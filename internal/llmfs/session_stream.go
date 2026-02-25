package llmfs

import (
	"io"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionStreamFile is the stream file for a specific session: /n/llm/N/stream
// Read blocks until the next chunk is available, returning chunks as they arrive
// and io.EOF when generation completes (or no generation is active).
type SessionStreamFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionStreamFile creates a stream file for the given session.
func NewSessionStreamFile(sm *llm.SessionManager, id int) *SessionStreamFile {
	return &SessionStreamFile{
		BaseFile: protocol.NewBaseFile("stream", 0444),
		sm:       sm,
		id:       id,
	}
}

// Read blocks until the next text chunk is available and copies it into p.
// Returns io.EOF when generation completes or no generation is active.
// The offset parameter is ignored — this is a streaming file, not seekable.
func (f *SessionStreamFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	ch := session.GetStreamCh()
	if ch == nil {
		// No active generation.
		return 0, io.EOF
	}

	chunk, ok := <-ch
	if !ok {
		// Channel closed: generation complete.
		return 0, io.EOF
	}

	n := copy(p, chunk)
	return n, nil
}

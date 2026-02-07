package llmfs

import (
	"fmt"
	"io"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// NewFile is the session factory: /n/llm/new
// Read creates a new session and returns its ID.
// This follows the Plan 9 clone pattern (like /net/tcp/clone, Acme windows).
type NewFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
}

// NewNewFile creates the new file (session factory).
func NewNewFile(sm *llm.SessionManager) *NewFile {
	return &NewFile{
		BaseFile: protocol.NewBaseFile("new", 0444),
		sm:       sm,
	}
}

// Read creates a new session and returns its ID.
// Each read creates a fresh session with default settings.
func (f *NewFile) Read(p []byte, offset int64) (int, error) {
	// Only create session on first read (offset 0)
	// Subsequent reads at higher offsets just return the remaining data
	if offset > 0 {
		return 0, io.EOF
	}

	// Create new session
	id := f.sm.Create()

	// Return session ID
	content := fmt.Sprintf("%d\n", id)
	n := copy(p, content)
	return n, nil
}

// Write is not supported - this is a read-only factory.
func (f *NewFile) Write(p []byte, offset int64) (int, error) {
	return 0, protocol.ErrPermission
}

// Stat returns the file's metadata.
func (f *NewFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	// Length unknown until read
	s.Length = 0
	return s
}

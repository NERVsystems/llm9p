package llmfs

import (
	"fmt"
	"io"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionUsageFile provides per-session token usage: /n/llm/N/usage
// Read returns "estimated_tokens/context_limit\n" (e.g., "42000/200000")
// Estimated tokens use a 4 chars/token heuristic over current message content.
type SessionUsageFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionUsageFile creates a usage file for the given session.
func NewSessionUsageFile(sm *llm.SessionManager, id int) *SessionUsageFile {
	return &SessionUsageFile{
		BaseFile: protocol.NewBaseFile("usage", 0444),
		sm:       sm,
		id:       id,
	}
}

// Read returns estimated token usage over context limit.
func (f *SessionUsageFile) Read(p []byte, offset int64) (int, error) {
	estimated := f.sm.EstimatedContextTokens(f.id)
	limit := f.sm.ContextLimit()
	content := fmt.Sprintf("%d/%d\n", estimated, limit)

	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write is not allowed.
func (f *SessionUsageFile) Write(p []byte, offset int64) (int, error) {
	return 0, fmt.Errorf("usage is read-only")
}

// Stat returns the file's metadata.
func (f *SessionUsageFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	estimated := f.sm.EstimatedContextTokens(f.id)
	limit := f.sm.ContextLimit()
	s.Length = uint64(len(fmt.Sprintf("%d/%d\n", estimated, limit)))
	return s
}

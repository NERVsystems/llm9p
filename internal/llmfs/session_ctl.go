package llmfs

import (
	"io"
	"strings"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionCtlFile is the control file for a session: /n/llm/N/ctl
// Supports commands: "reset" (clear history), "close" (remove session)
type SessionCtlFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionCtlFile creates a ctl file for the given session.
func NewSessionCtlFile(sm *llm.SessionManager, id int) *SessionCtlFile {
	return &SessionCtlFile{
		BaseFile: protocol.NewBaseFile("ctl", 0222),
		sm:       sm,
		id:       id,
	}
}

// Read returns empty for the control file.
func (f *SessionCtlFile) Read(p []byte, offset int64) (int, error) {
	return 0, io.EOF
}

// Write processes control commands.
func (f *SessionCtlFile) Write(p []byte, offset int64) (int, error) {
	cmd := strings.TrimSpace(string(p))

	switch cmd {
	case "reset":
		f.sm.Reset(f.id)
	case "close":
		f.sm.Close(f.id)
	default:
		return 0, protocol.Error("unknown command: " + cmd)
	}

	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionCtlFile) Stat() protocol.Stat {
	return f.BaseFile.Stat()
}

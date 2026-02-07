package llmfs

import (
	"fmt"
	"io"
	"strconv"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionDir represents a single session directory: /n/llm/N/
// Contains: ask, context, ctl, model, temperature, system, thinking, prefill
type SessionDir struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionDir creates a session directory for the given session ID.
func NewSessionDir(sm *llm.SessionManager, id int) *SessionDir {
	return &SessionDir{
		BaseFile: protocol.NewBaseFile(strconv.Itoa(id), protocol.DMDIR|0555),
		sm:       sm,
		id:       id,
	}
}

// Children returns the files in this session directory.
func (d *SessionDir) Children() []protocol.File {
	session := d.sm.Get(d.id)
	if session == nil {
		return nil
	}

	return []protocol.File{
		NewSessionAskFile(d.sm, d.id),
		NewSessionContextFile(d.sm, d.id),
		NewSessionCtlFile(d.sm, d.id),
		NewSessionModelFile(d.sm, d.id),
		NewSessionTemperatureFile(d.sm, d.id),
		NewSessionSystemFile(d.sm, d.id),
		NewSessionThinkingFile(d.sm, d.id),
		NewSessionPrefillFile(d.sm, d.id),
	}
}

// Lookup finds a child file by name.
func (d *SessionDir) Lookup(name string) (protocol.File, error) {
	session := d.sm.Get(d.id)
	if session == nil {
		return nil, protocol.ErrNotFound
	}

	switch name {
	case "ask":
		return NewSessionAskFile(d.sm, d.id), nil
	case "context":
		return NewSessionContextFile(d.sm, d.id), nil
	case "ctl":
		return NewSessionCtlFile(d.sm, d.id), nil
	case "model":
		return NewSessionModelFile(d.sm, d.id), nil
	case "temperature":
		return NewSessionTemperatureFile(d.sm, d.id), nil
	case "system":
		return NewSessionSystemFile(d.sm, d.id), nil
	case "thinking":
		return NewSessionThinkingFile(d.sm, d.id), nil
	case "prefill":
		return NewSessionPrefillFile(d.sm, d.id), nil
	default:
		return nil, protocol.ErrNotFound
	}
}

// Read returns directory listing as packed stat entries.
func (d *SessionDir) Read(p []byte, offset int64) (int, error) {
	var buf []byte
	for _, f := range d.Children() {
		stat := f.Stat()
		entry := make([]byte, 256)
		n := stat.Encode(entry)
		buf = append(buf, entry[:n]...)
	}

	if offset >= int64(len(buf)) {
		return 0, io.EOF
	}

	n := copy(p, buf[offset:])
	return n, nil
}

// Stat returns the directory's metadata.
func (d *SessionDir) Stat() protocol.Stat {
	s := d.BaseFile.Stat()
	s.Qid.Type = protocol.QTDIR
	return s
}

// SessionsDir is the root /n/llm directory.
// Contains only the "new" file plus dynamically created session directories.
type SessionsDir struct {
	*protocol.BaseFile
	sm      *llm.SessionManager
	newFile *NewFile
}

// NewSessionsDir creates the root LLM directory.
func NewSessionsDir(sm *llm.SessionManager) *SessionsDir {
	return &SessionsDir{
		BaseFile: protocol.NewBaseFile("llm", protocol.DMDIR|0555),
		sm:       sm,
		newFile:  NewNewFile(sm),
	}
}

// Children returns the files in the root directory.
// This includes "new" plus all active session directories.
func (d *SessionsDir) Children() []protocol.File {
	children := []protocol.File{d.newFile}

	// Add session directories for all active sessions
	for _, id := range d.sm.ListSessions() {
		children = append(children, NewSessionDir(d.sm, id))
	}

	return children
}

// Lookup finds a child by name.
func (d *SessionsDir) Lookup(name string) (protocol.File, error) {
	// Check for "new" file
	if name == "new" {
		return d.newFile, nil
	}

	// Try to parse as session ID
	id, err := strconv.Atoi(name)
	if err != nil {
		return nil, protocol.ErrNotFound
	}

	// Check if session exists
	session := d.sm.Get(id)
	if session == nil {
		return nil, protocol.ErrNotFound
	}

	return NewSessionDir(d.sm, id), nil
}

// Read returns directory listing as packed stat entries.
func (d *SessionsDir) Read(p []byte, offset int64) (int, error) {
	var buf []byte
	for _, f := range d.Children() {
		stat := f.Stat()
		entry := make([]byte, 256)
		n := stat.Encode(entry)
		buf = append(buf, entry[:n]...)
	}

	if offset >= int64(len(buf)) {
		return 0, io.EOF
	}

	n := copy(p, buf[offset:])
	return n, nil
}

// Stat returns the directory's metadata.
func (d *SessionsDir) Stat() protocol.Stat {
	s := d.BaseFile.Stat()
	s.Qid.Type = protocol.QTDIR
	return s
}

// Silence unused import warning
var _ = fmt.Sprint

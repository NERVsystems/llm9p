package llmfs

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionModelFile controls the model for a session: /n/llm/N/model
type SessionModelFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionModelFile creates a model file for the given session.
func NewSessionModelFile(sm *llm.SessionManager, id int) *SessionModelFile {
	return &SessionModelFile{
		BaseFile: protocol.NewBaseFile("model", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the current model name.
func (f *SessionModelFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	content := session.Model() + "\n"
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write sets the model name.
func (f *SessionModelFile) Write(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	model := strings.TrimSpace(string(p))
	if model != "" {
		session.SetModel(model)
	}
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionModelFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	session := f.sm.Get(f.id)
	if session != nil {
		s.Length = uint64(len(session.Model()) + 1)
	}
	return s
}

// SessionTemperatureFile controls the temperature for a session: /n/llm/N/temperature
type SessionTemperatureFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionTemperatureFile creates a temperature file for the given session.
func NewSessionTemperatureFile(sm *llm.SessionManager, id int) *SessionTemperatureFile {
	return &SessionTemperatureFile{
		BaseFile: protocol.NewBaseFile("temperature", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the current temperature.
func (f *SessionTemperatureFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	content := fmt.Sprintf("%.2f\n", session.Temperature())
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write sets the temperature.
func (f *SessionTemperatureFile) Write(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	temp, err := strconv.ParseFloat(strings.TrimSpace(string(p)), 64)
	if err != nil {
		return 0, protocol.Error("invalid temperature: " + err.Error())
	}
	if temp < 0.0 || temp > 2.0 {
		return 0, protocol.Error("temperature must be between 0.0 and 2.0")
	}
	session.SetTemperature(temp)
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionTemperatureFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	session := f.sm.Get(f.id)
	if session != nil {
		s.Length = uint64(len(fmt.Sprintf("%.2f\n", session.Temperature())))
	}
	return s
}

// SessionSystemFile controls the system prompt for a session: /n/llm/N/system
type SessionSystemFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionSystemFile creates a system file for the given session.
func NewSessionSystemFile(sm *llm.SessionManager, id int) *SessionSystemFile {
	return &SessionSystemFile{
		BaseFile: protocol.NewBaseFile("system", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the current system prompt.
func (f *SessionSystemFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	content := session.SystemPrompt()
	if content != "" && !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write sets the system prompt.
func (f *SessionSystemFile) Write(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	prompt := strings.TrimSpace(string(p))
	session.SetSystemPrompt(prompt)
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionSystemFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	session := f.sm.Get(f.id)
	if session != nil {
		content := session.SystemPrompt()
		if content != "" {
			s.Length = uint64(len(content) + 1)
		}
	}
	return s
}

// SessionThinkingFile controls the thinking token budget: /n/llm/N/thinking
type SessionThinkingFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionThinkingFile creates a thinking file for the given session.
func NewSessionThinkingFile(sm *llm.SessionManager, id int) *SessionThinkingFile {
	return &SessionThinkingFile{
		BaseFile: protocol.NewBaseFile("thinking", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the current thinking token budget.
func (f *SessionThinkingFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	tokens := session.ThinkingTokens()
	var content string
	switch {
	case tokens < 0:
		content = "max\n"
	case tokens == 0:
		content = "disabled\n"
	default:
		content = fmt.Sprintf("%d\n", tokens)
	}

	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write sets the thinking token budget.
func (f *SessionThinkingFile) Write(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	value := strings.TrimSpace(string(p))
	switch value {
	case "max", "-1":
		session.SetThinkingTokens(-1)
	case "disabled", "off", "0":
		session.SetThinkingTokens(0)
	default:
		tokens, err := strconv.Atoi(value)
		if err != nil {
			return 0, protocol.Error("invalid thinking budget: " + err.Error())
		}
		session.SetThinkingTokens(tokens)
	}
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionThinkingFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	// Estimate length
	s.Length = 16
	return s
}

// SessionPrefillFile controls the response prefill: /n/llm/N/prefill
type SessionPrefillFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionPrefillFile creates a prefill file for the given session.
func NewSessionPrefillFile(sm *llm.SessionManager, id int) *SessionPrefillFile {
	return &SessionPrefillFile{
		BaseFile: protocol.NewBaseFile("prefill", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the current prefill string.
func (f *SessionPrefillFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	content := session.Prefill()
	if content != "" && !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	return copy(p, content[offset:]), nil
}

// Write sets the prefill string.
func (f *SessionPrefillFile) Write(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	// Don't trim - prefill may have intentional trailing space
	prefill := string(p)
	// But do remove trailing newline since shell adds it
	prefill = strings.TrimSuffix(prefill, "\n")
	session.SetPrefill(prefill)
	return len(p), nil
}

// Stat returns the file's metadata.
func (f *SessionPrefillFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	session := f.sm.Get(f.id)
	if session != nil {
		content := session.Prefill()
		if content != "" {
			s.Length = uint64(len(content) + 1)
		}
	}
	return s
}

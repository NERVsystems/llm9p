package llmfs

import (
	"io"
	"strings"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SystemFile exposes the system prompt (read/write)
type SystemFile struct {
	*protocol.BaseFile
	client llm.Backend
}

// NewSystemFile creates the system file
func NewSystemFile(client llm.Backend) *SystemFile {
	return &SystemFile{
		BaseFile: protocol.NewBaseFile("system", 0666),
		client:   client,
	}
}

func (f *SystemFile) Read(p []byte, offset int64) (int, error) {
	content := f.client.SystemPrompt()
	if content != "" {
		content += "\n"
	}
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	n := copy(p, content[offset:])
	return n, nil
}

func (f *SystemFile) Write(p []byte, offset int64) (int, error) {
	prompt := strings.TrimSpace(string(p))
	f.client.SetSystemPrompt(prompt)
	return len(p), nil
}

func (f *SystemFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	content := f.client.SystemPrompt()
	if content != "" {
		s.Length = uint64(len(content) + 1) // +1 for newline
	} else {
		s.Length = 0
	}
	return s
}

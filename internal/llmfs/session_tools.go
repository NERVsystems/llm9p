package llmfs

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionToolsFile is the tools file for a specific session: /n/llm/N/tools
// Write a JSON array of tool definitions to enable native tool_use protocol.
// Write an empty string to clear tools and return to text-only mode.
//
// Tool definition format (JSON array):
//
//	[{"name":"toolname","description":"what it does","input_schema":{"type":"object","properties":{"args":{"type":"string"}},"required":["args"]}}]
type SessionToolsFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionToolsFile creates a tools file for the given session.
func NewSessionToolsFile(sm *llm.SessionManager, id int) *SessionToolsFile {
	return &SessionToolsFile{
		BaseFile: protocol.NewBaseFile("tools", 0222), // write-only
		sm:       sm,
		id:       id,
	}
}

// Write accepts a JSON array of tool definitions and installs them on the session.
// An empty write clears all tools, returning the session to text-only mode.
func (f *SessionToolsFile) Write(p []byte, offset int64) (int, error) {
	content := strings.TrimSpace(string(p))

	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	if content == "" {
		// Empty write → clear tools (text-only mode)
		session.SetTools(nil)
		log.Printf("llm9p: SessionToolsFile.Write session=%d cleared tools", f.id)
		return len(p), nil
	}

	var tools []llm.ToolDef
	if err := json.Unmarshal([]byte(content), &tools); err != nil {
		log.Printf("llm9p: SessionToolsFile.Write parse error: %v", err)
		return 0, fmt.Errorf("tools: invalid JSON: %v", err)
	}

	session.SetTools(tools)
	log.Printf("llm9p: SessionToolsFile.Write session=%d installed %d tools", f.id, len(tools))
	return len(p), nil
}

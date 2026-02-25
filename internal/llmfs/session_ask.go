package llmfs

import (
	"context"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// SessionAskFile is the ask file for a specific session: /n/llm/N/ask
// Write a prompt, read the response.
type SessionAskFile struct {
	*protocol.BaseFile
	sm *llm.SessionManager
	id int
}

// NewSessionAskFile creates an ask file for the given session.
func NewSessionAskFile(sm *llm.SessionManager, id int) *SessionAskFile {
	return &SessionAskFile{
		BaseFile: protocol.NewBaseFile("ask", 0666),
		sm:       sm,
		id:       id,
	}
}

// Read returns the last response from this session.
// Blocks until any in-progress generation completes before returning content.
func (f *SessionAskFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

	// Wait for the background generation goroutine (if any) to finish.
	session.WaitDone()

	content := session.LastResponse()
	if content != "" && !strings.HasSuffix(content, "\n") {
		content += "\n"
	}

	if offset >= int64(len(content)) {
		return 0, io.EOF
	}

	n := copy(p, content[offset:])
	return n, nil
}

// Write sends a prompt to the LLM using this session's settings.
// Returns immediately after starting the generation in a background goroutine.
// The response is available via Read (pread) once generation completes.
//
// If the write begins with "TOOL_RESULTS\n", it is parsed as tool execution
// results and submitted via AskWithToolResults instead of a plain Ask.
//
// TOOL_RESULTS format:
//
//	TOOL_RESULTS
//	<tool_use_id>
//	<result content (may be multi-line)>
//	---
//	<tool_use_id2>
//	<result content>
//	---
func (f *SessionAskFile) Write(p []byte, offset int64) (int, error) {
	log.Printf("llm9p: SessionAskFile.Write session=%d len=%d", f.id, len(p))

	prompt := strings.TrimSpace(string(p))
	if prompt == "" {
		return len(p), nil // Empty write is a no-op
	}

	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}
	if session.IsClosed() {
		return 0, protocol.ErrPermission
	}

	// Allocate streaming channel and done signal before launching goroutine.
	// This ensures the stream file can observe a non-nil channel immediately
	// after the write returns (no race between Write and stream Read).
	session.BeginGeneration()

	// Launch generation goroutine — Write returns to the caller immediately.
	sm := f.sm
	id := f.id
	go func() {
		defer session.EndGeneration()
		ctx := context.Background()

		if strings.HasPrefix(prompt, "TOOL_RESULTS\n") {
			results, err := parseToolResults(prompt)
			if err != nil {
				log.Printf("llm9p: async gen TOOL_RESULTS parse error: %v", err)
				session.SetLastResponse("Error: " + err.Error())
				return
			}
			log.Printf("llm9p: async gen submitting %d tool results", len(results))
			if _, err = sm.AskWithToolResults(ctx, id, results); err != nil {
				log.Printf("llm9p: async gen tool results error: %v", err)
			}
			return
		}

		log.Printf("llm9p: async gen prompt: %s", prompt[:min(len(prompt), 50)])
		if _, err := sm.Ask(ctx, id, prompt); err != nil {
			log.Printf("llm9p: async gen error: %v", err)
		}
	}()

	return len(p), nil
}

// parseToolResults parses the TOOL_RESULTS wire format into a slice of ToolResult.
// Each result block starts with a tool_use_id line, followed by content lines,
// terminated by "---" (or end of input).
func parseToolResults(text string) ([]llm.ToolResult, error) {
	lines := strings.Split(text, "\n")
	if len(lines) < 2 || lines[0] != "TOOL_RESULTS" {
		return nil, fmt.Errorf("missing TOOL_RESULTS header")
	}

	var results []llm.ToolResult
	i := 1 // Skip "TOOL_RESULTS" header

	for i < len(lines) {
		// Skip blank lines between blocks
		if strings.TrimSpace(lines[i]) == "" {
			i++
			continue
		}

		// Next non-empty line is the tool_use_id
		toolUseID := strings.TrimSpace(lines[i])
		i++

		// Collect content lines until "---" separator or end
		var contentLines []string
		for i < len(lines) && lines[i] != "---" {
			contentLines = append(contentLines, lines[i])
			i++
		}
		// Skip "---" separator
		if i < len(lines) && lines[i] == "---" {
			i++
		}

		content := strings.TrimRight(strings.Join(contentLines, "\n"), "\n")
		results = append(results, llm.ToolResult{
			ToolUseID: toolUseID,
			Content:   content,
		})
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("TOOL_RESULTS contained no results")
	}

	return results, nil
}

// Stat returns the file's metadata.
func (f *SessionAskFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	// Length is dynamic based on last response
	session := f.sm.Get(f.id)
	if session != nil {
		s.Length = uint64(len(session.LastResponse()))
	}
	return s
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

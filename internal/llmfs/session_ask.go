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
func (f *SessionAskFile) Read(p []byte, offset int64) (int, error) {
	session := f.sm.Get(f.id)
	if session == nil {
		return 0, protocol.ErrNotFound
	}

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

	ctx := context.Background()

	// Detect TOOL_RESULTS prefix → submit tool results, not a plain prompt
	if strings.HasPrefix(prompt, "TOOL_RESULTS\n") {
		results, err := parseToolResults(prompt)
		if err != nil {
			log.Printf("llm9p: SessionAskFile.Write TOOL_RESULTS parse error: %v", err)
			// Store error in session so client can read it back
			session := f.sm.Get(f.id)
			if session != nil {
				session.SetLastResponse("Error: " + err.Error())
			}
			return len(p), nil
		}
		log.Printf("llm9p: SessionAskFile.Write submitting %d tool results", len(results))
		_, err = f.sm.AskWithToolResults(ctx, f.id, results)
		if err != nil {
			log.Printf("llm9p: SessionAskFile.Write tool results error: %v", err)
		}
		return len(p), nil
	}

	// Regular text prompt
	log.Printf("llm9p: SessionAskFile.Write prompt: %s", prompt[:min(len(prompt), 50)])
	response, err := f.sm.Ask(ctx, f.id, prompt)
	if err != nil {
		log.Printf("llm9p: SessionAskFile.Write error: %v", err)
		return len(p), nil
	}

	log.Printf("llm9p: SessionAskFile.Write success, response len=%d", len(response))
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

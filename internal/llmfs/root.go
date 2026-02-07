// Package llmfs implements the LLM filesystem exposed via 9P.
package llmfs

import (
	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// NewRoot creates the root directory of the LLM filesystem.
// This implements the clone-based session architecture (CSP compliant):
//
//	/n/llm/
//	├── new              # Read to create session, returns ID
//	├── 0/               # Session 0 (fully independent)
//	│   ├── ask
//	│   ├── context
//	│   ├── ctl
//	│   ├── model
//	│   ├── temperature
//	│   ├── system
//	│   ├── thinking
//	│   └── prefill
//	├── 1/               # Session 1 (fully independent)
//	└── ...
//
// No global files. Each session is isolated with its own settings.
func NewRoot(sm *llm.SessionManager) protocol.Dir {
	return NewSessionsDir(sm)
}

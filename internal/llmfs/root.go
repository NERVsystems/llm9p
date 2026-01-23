// Package llmfs implements the LLM filesystem exposed via 9P.
package llmfs

import (
	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// NewRoot creates the root directory of the LLM filesystem
func NewRoot(client llm.Backend) protocol.Dir {
	root := protocol.NewStaticDir("llm")

	// Add all files
	root.AddChild(NewAskFile(client))
	root.AddChild(NewModelFile(client))
	root.AddChild(NewTemperatureFile(client))
	root.AddChild(NewSystemFile(client))
	root.AddChild(NewTokensFile(client))
	root.AddChild(NewNewFile(client))
	root.AddChild(NewContextFile(client))
	root.AddChild(NewExampleFile())

	// Add stream directory
	streamDir := protocol.NewStaticDir("stream")
	streamDir.AddChild(NewStreamAskFile(client))
	streamDir.AddChild(NewChunkFile(client))
	root.AddChild(streamDir)

	return root
}

package llmfs

import (
	"github.com/NERVsystems/llm9p/internal/protocol"
)

const exampleContent = `LLM 9P Filesystem Usage Examples
=================================

Basic Interaction:
  echo "What is 2+2?" > ask     # Send prompt to LLM
  cat ask                        # Read response

Configuration:
  cat model                      # View current model
  echo "claude-3-haiku-20240307" > model   # Change model
  cat temperature                # View current temperature (0.0-2.0)
  echo "0.5" > temperature       # Set temperature
  cat system                     # View current system prompt
  echo "You are a helpful coding assistant." > system  # Set system prompt

Conversation Management:
  cat context                    # View conversation history (JSON)
  echo "Additional context..." > context  # Add system message to history
  echo "" > new                  # Reset conversation (keeps system prompt)

Token Usage:
  cat tokens                     # View tokens from last response

Streaming:
  echo "Tell me a story" > stream/ask  # Start streaming request
  cat stream/chunk                      # Read next chunk (blocks until available)
  # Keep reading stream/chunk until EOF for full response
  # Note: Read chunks immediately after writing to stream/ask

Shell Scripting:
  #!/bin/sh
  # Ask the LLM and get response
  echo "$1" > /mnt/llm/ask
  cat /mnt/llm/ask

Mounting (Linux/macOS):
  # Using 9pfuse (Plan 9 from User Space)
  9pfuse localhost:5640 /mnt/llm

  # Using mount_9p (macOS with plan9port)
  mount_9p localhost:5640 /mnt/llm

Environment:
  ANTHROPIC_API_KEY must be set when starting the server

Files:
  ask          Read/write: prompt goes in, response comes out (sync)
  model        Read/write: current model name
  temperature  Read/write: sampling temperature (0.0-2.0)
  system       Read/write: system prompt (persists across resets)
  tokens       Read-only: token count from last response
  new          Write-only: any write resets conversation (keeps system prompt)
  context      Read: JSON history; Write: add system message to history
  _example     Read-only: this help text
  stream/ask   Write-only: starts a streaming request
  stream/chunk Read-only: returns next chunk (blocks), EOF when done
`

// NewExampleFile creates the _example file with usage examples
func NewExampleFile() *protocol.StaticFile {
	return protocol.NewStaticFile("_example", []byte(exampleContent))
}

# llm9p - Development Guide

This guide is for Claude Code and developers working on the llm9p codebase.

## Quick Reference

### Build and Run

```bash
# Build
go build -o llm9p ./cmd/llm9p

# Run
ANTHROPIC_API_KEY=sk-... ./llm9p -addr :5640

# Run with debug logging
ANTHROPIC_API_KEY=sk-... ./llm9p -addr :5640 -debug
```

### Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Test specific package
go test ./internal/protocol/...
go test ./internal/llmfs/...
```

### Development Workflow

```bash
# Install dependencies
go mod tidy

# Format code
go fmt ./...

# Vet code
go vet ./...

# Build
go build -o llm9p ./cmd/llm9p
```

## Architecture Overview

### Project Structure

```
llm9p/
├── cmd/
│   └── llm9p/
│       └── main.go           # Entry point, CLI flags, server setup
├── internal/
│   ├── protocol/             # 9P2000 protocol implementation
│   │   ├── protocol.go       # Message types, constants, encoding
│   │   ├── message.go        # Individual message types
│   │   ├── server.go         # Connection handling
│   │   └── fs.go             # File/Dir interfaces, base implementations
│   ├── llm/                  # LLM client wrapper
│   │   └── client.go         # Anthropic API integration
│   └── llmfs/                # LLM filesystem implementation
│       ├── root.go           # Root directory construction
│       ├── ask.go            # Ask file (shim pattern)
│       ├── state.go          # Model, temperature files
│       ├── tokens.go         # Read-only token counter
│       ├── new.go            # Conversation reset trigger
│       ├── context.go        # Conversation history
│       ├── example.go        # Usage examples
│       └── stream.go         # Streaming interface
├── go.mod
├── go.sum
├── README.md
└── CLAUDE.md
```

### Key Components

1. **Protocol Layer (`internal/protocol/`)**
   - Implements 9P2000 protocol
   - No external dependencies (stdlib only)
   - `File` and `Dir` interfaces define the filesystem abstraction

2. **LLM Client (`internal/llm/client.go`)**
   - Wraps Anthropic SDK
   - Manages conversation state
   - Supports both sync and streaming responses
   - Tracks token usage

3. **LLM Filesystem (`internal/llmfs/`)**
   - Implements each file in the LLM filesystem
   - `AskFile` is the core interaction point
   - State files (`model`, `temperature`) modify client settings
   - `ChunkFile` provides streaming access

## Adding a New File

1. Create a new file in `internal/llmfs/`:

```go
package llmfs

import (
    "github.com/NERVsystems/llm9p/internal/llm"
    "github.com/NERVsystems/llm9p/internal/protocol"
)

type MyFile struct {
    *protocol.BaseFile
    client *llm.Client
}

func NewMyFile(client *llm.Client) *MyFile {
    return &MyFile{
        BaseFile: protocol.NewBaseFile("myfile", 0666),
        client:   client,
    }
}

func (f *MyFile) Read(p []byte, offset int64) (int, error) {
    // Implement read
}

func (f *MyFile) Write(p []byte, offset int64) (int, error) {
    // Implement write
}

func (f *MyFile) Stat() protocol.Stat {
    s := f.BaseFile.Stat()
    // Update s.Length if dynamic
    return s
}
```

2. Add to root directory in `internal/llmfs/root.go`:

```go
root.AddChild(NewMyFile(client))
```

## Protocol Implementation Notes

### Message Flow

1. Client sends T-message (request)
2. Server responds with R-message (response)
3. Each message has a tag for matching requests/responses

### Key 9P Operations

- `Tversion/Rversion` - Protocol negotiation
- `Tattach/Rattach` - Connect to filesystem
- `Twalk/Rwalk` - Navigate directory tree
- `Topen/Ropen` - Open a file
- `Tread/Rread` - Read from file
- `Twrite/Rwrite` - Write to file
- `Tclunk/Rclunk` - Close a fid

### File Interfaces

```go
// File is the interface that files must implement
type File interface {
    Stat() Stat
    Open(mode uint8) error
    Read(p []byte, offset int64) (int, error)
    Write(p []byte, offset int64) (int, error)
    Close() error
}

// Dir extends File with directory operations
type Dir interface {
    File
    Children() []File
    Lookup(name string) (File, error)
}
```

## Debugging

### Enable Debug Logging

```bash
ANTHROPIC_API_KEY=sk-... ./llm9p -debug
```

This logs all 9P messages sent and received.

### Test with 9p Client (plan9port)

```bash
# Using 9pfuse
9pfuse localhost:5640 /mnt/llm

# Using Plan 9's 9p tool (no mount needed)
9p -a localhost:5640 ls llm
9p -a localhost:5640 read llm/model
9p -a localhost:5640 read llm/temperature
9p -a localhost:5640 write llm/ask "What is 2+2?"
9p -a localhost:5640 read llm/ask        # Returns "4"
9p -a localhost:5640 read llm/tokens     # Returns token count

# Multi-turn conversation
9p -a localhost:5640 write llm/ask "Remember the number 42"
9p -a localhost:5640 write llm/ask "What number did I just mention?"
9p -a localhost:5640 read llm/ask        # Returns "42"

# Add system message (e.g., persona)
9p -a localhost:5640 write llm/context "Respond like a pirate"
9p -a localhost:5640 write llm/ask "Hello"
9p -a localhost:5640 read llm/ask        # Pirate-style response

# Reset conversation
9p -a localhost:5640 write llm/new "reset"
```

### Test with Infernode (Inferno OS)

```bash
# Start infernode (from infernode directory)
./emu

# Inside infernode shell:
mkdir /n/llm
mount -A tcp!127.0.0.1!5640 /n/llm
ls -l /n/llm
cat /n/llm/model
echo 'What is the capital of France?' > /n/llm/ask
cat /n/llm/ask
```

**Infernode Notes:**
- Use `127.0.0.1` not `localhost` (DNS resolution differs)
- Create mount point before mounting: `mkdir /n/llm`
- The `-A` flag enables anonymous auth

### Common Issues

**"file not found"**
- Check file name spelling
- Ensure file is added to root directory

**"permission denied"**
- Check file mode (read-only files have mode 0444)
- Write-only files have mode 0222

**Connection refused**
- Ensure server is running
- Check address/port

**API errors**
- Check ANTHROPIC_API_KEY is set
- Check API key is valid
- Check rate limits

## Code Style

### Error Handling

Files should handle errors gracefully and expose them to the user:

```go
func (f *AskFile) Write(p []byte, offset int64) (int, error) {
    response, err := f.client.Ask(ctx, prompt)
    if err != nil {
        // Store error so it can be read back
        f.lastResponse = "Error: " + err.Error()
        return len(p), nil
    }
    f.lastResponse = response
    return len(p), nil
}
```

### Stat Implementation

Always implement `Stat()` to return accurate `Length`:

```go
func (f *MyFile) Stat() protocol.Stat {
    s := f.BaseFile.Stat()
    s.Length = uint64(len(f.content))
    return s
}
```

## Verified Test Cases

The following scenarios have been tested and verified working:

### plan9port (9p tool)
- [x] `ls llm` - List filesystem root
- [x] `read llm/model` - Returns model name
- [x] `read llm/temperature` - Returns temperature
- [x] `read llm/tokens` - Returns 0 initially, updates after queries
- [x] `read llm/_example` - Returns usage examples
- [x] `write llm/temperature "0.5"` - Updates temperature setting
- [x] `write llm/ask "What is 2+2?"` followed by `read llm/ask` - Returns "4"
- [x] Multi-turn conversation maintains context
- [x] `write llm/context "Respond like a pirate"` - System message works
- [x] `write llm/new "reset"` - Clears conversation history

### Infernode (Inferno OS)
- [x] `mount -A tcp!127.0.0.1!5640 /n/llm` - Mounts successfully
- [x] `ls -l /n/llm` - Lists all files with correct permissions
- [x] `cat /n/llm/model` - Returns model name
- [x] `cat /n/llm/temperature` - Returns temperature
- [x] `echo 'prompt' > /n/llm/ask` followed by `cat /n/llm/ask` - Full LLM interaction works
- [x] LLM correctly identifies client as Inferno OS when asked

## Future Enhancements

- [ ] Multiple conversation support (via subdirectories)
- [ ] Prompt templates
- [ ] Response caching
- [ ] Rate limiting
- [ ] Authentication
- [ ] Unix socket support
- [ ] Integration tests

## Resources

- [9P Protocol Specification](http://man.cat-v.org/plan_9/5/intro)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Plan 9 from User Space](https://9fans.github.io/plan9port/)
- [Infernode](https://github.com/NERVsystems/infernode) - Hosted Inferno OS with native 9P support

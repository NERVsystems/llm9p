# llm9p

An LLM (Claude) exposed as a 9P filesystem.

llm9p enables users, scripts, and AI agents to interact with an LLM through standard filesystem operations. Write a prompt to a file, read the response from the same file.

## What is 9P?

9P is a simple, lightweight network filesystem protocol originally developed for Plan 9 from Bell Labs. It lets you access remote resources as if they were local files. This means you can interact with Claude using basic file operations (`cat`, `echo`, `>`, `<`) instead of SDKs or HTTP APIs.

**Why use 9P for LLM access?**
- **Universal**: Any language or tool that can read/write files can use it
- **Scriptable**: Chain LLM calls with standard Unix pipes and shell scripts
- **Composable**: Mount multiple 9P services and combine them
- **Simple**: No libraries, no dependencies, just files

## Installation

```bash
go install github.com/NERVsystems/llm9p/cmd/llm9p@latest
```

Or build from source:

```bash
git clone https://github.com/NERVsystems/llm9p
cd llm9p
go build -o llm9p ./cmd/llm9p
```

## Usage

### Start the Server

**Option A: Using Anthropic API** (requires API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./llm9p -addr :5640
```

**Option B: Using Claude Max Subscription** (via Claude Code CLI)

If you have a Claude Max subscription and the Claude Code CLI installed:

```bash
./llm9p -addr :5640 -backend cli
```

This uses your Claude Max subscription instead of API tokens. No API key required.

**Requirements for CLI backend:**
- Claude Code CLI installed and authenticated (`claude` command available)
- Active Claude Max subscription

### Mount the Filesystem

There are several ways to mount the filesystem depending on your environment.

#### Option 1: Plan 9 from User Space (plan9port)

[plan9port](https://9fans.github.io/plan9port/) provides Plan 9 tools for Unix systems.

```bash
# Install on macOS
brew install plan9port

# Install on Debian/Ubuntu
sudo apt-get install 9base

# Mount using 9pfuse
mkdir -p /mnt/llm
9pfuse localhost:5640 /mnt/llm

# Or use the 9p tool directly (no mount needed)
9p -a localhost:5640 ls llm
9p -a localhost:5640 read llm/model
9p -a localhost:5640 write llm/ask "What is 2+2?"
9p -a localhost:5640 read llm/ask
```

#### Option 2: Infernode (Inferno OS)

[Infernode](https://github.com/NERVsystems/infernode) is a hosted Inferno OS environment with native 9P support. This is an excellent way to explore 9P filesystems.

```sh
# Start infernode
cd /path/to/infernode
./emu

# Inside infernode shell, mount llm9p
# Note: Use IP address 127.0.0.1, not "localhost"
mkdir /n/llm
mount -A tcp!127.0.0.1!5640 /n/llm

# List available files
ls -l /n/llm

# Check current model
cat /n/llm/model

# Ask a question
echo 'What is the capital of France?' > /n/llm/ask
cat /n/llm/ask

# View conversation history
cat /n/llm/context

# Reset conversation
echo reset > /n/llm/new
```

**Infernode Tips:**
- Always use `127.0.0.1` instead of `localhost` for the server address
- Create the mount point with `mkdir /n/llm` before mounting
- The `-A` flag to mount enables anonymous authentication (no auth required)
- Use single quotes around prompts to avoid shell interpretation issues

#### Option 3: Linux 9P Mount

Linux has built-in 9P filesystem support via the `9p` kernel module.

```bash
# Mount via kernel 9p module
sudo mount -t 9p -o port=5640,version=9p2000 localhost /mnt/llm
```

### Interact with the LLM

```bash
# Ask a question
echo "What is 2+2?" > /mnt/llm/ask
cat /mnt/llm/ask

# View token usage
cat /mnt/llm/tokens

# Change model
echo "claude-3-haiku-20240307" > /mnt/llm/model

# Adjust temperature
echo "0.5" > /mnt/llm/temperature

# Set a system prompt (persists across conversation resets)
echo "You are a helpful coding assistant." > /mnt/llm/system

# View current system prompt
cat /mnt/llm/system

# View conversation history
cat /mnt/llm/context

# Add a system message
echo "You are a helpful coding assistant." > /mnt/llm/context

# Reset conversation
echo "" > /mnt/llm/new

# View help
cat /mnt/llm/_example
```

## Filesystem Schema

```
/llm/
├── ask              # Write prompt, read response (same file)
├── model            # Read/write: current model name
├── temperature      # Read/write: temperature float (0.0-2.0)
├── system           # Read/write: system prompt (persists across resets)
├── tokens           # Read-only: last response token count
├── new              # Write anything to start fresh conversation
├── context          # Read: conversation history; Write: add system message
├── _example         # Read-only: usage examples
└── stream/          # Streaming interface
    ├── ask          # Write-only: starts a streaming request
    └── chunk        # Read-only: blocks until next chunk, EOF on completion
```

### File Behaviors

| File | Read | Write |
|------|------|-------|
| `ask` | Returns last LLM response | Sends prompt to LLM (sync), stores response |
| `model` | Returns current model name | Sets model for subsequent requests |
| `temperature` | Returns current temperature | Sets temperature (0.0-2.0) |
| `system` | Returns current system prompt | Sets system prompt (persists across resets) |
| `tokens` | Returns last response token count | Permission denied |
| `new` | Permission denied | Any write resets conversation (keeps system prompt) |
| `context` | Returns JSON conversation history | Appends system message to history |
| `_example` | Returns usage examples | Permission denied |
| `stream/ask` | Permission denied | Starts a streaming request |
| `stream/chunk` | Blocks until next chunk, returns it | Permission denied |

## Streaming

For long responses, use the streaming interface to see output as it's generated:

```bash
# Start a streaming request (using 9p tool)
echo "Write a poem about the moon" | 9p -a localhost:5640 write stream/ask &

# Read chunks as they arrive
while chunk=$(9p -a localhost:5640 read stream/chunk 2>/dev/null); do
  [ -z "$chunk" ] && break
  printf "%s" "$chunk"
done
```

With a mounted filesystem:

```bash
# Start streaming in background
echo "Explain quantum computing" > /mnt/llm/stream/ask &

# Read chunks
while read -r chunk < /mnt/llm/stream/chunk 2>/dev/null; do
  printf "%s" "$chunk"
done
```

**Note:** Start reading chunks immediately after writing to `stream/ask`. If you wait too long, the stream may complete and you'll get EOF.

## Shell Scripting

```bash
#!/bin/sh
# ask.sh - Simple LLM query script

if [ -z "$1" ]; then
    echo "Usage: $0 <question>"
    exit 1
fi

echo "$1" > /mnt/llm/ask
cat /mnt/llm/ask
```

## Configuration

### Command Line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-addr` | `:5640` | Address to listen on |
| `-backend` | `api` | Backend: `api` (Anthropic API) or `cli` (Claude Code CLI) |
| `-debug` | `false` | Enable debug logging |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

## Default Settings

- **Model**: `claude-sonnet-4-20250514` (API) or `sonnet` (CLI)
- **Temperature**: `0.7`
- **Max Tokens**: `4096`

### Backend Differences

| Feature | API Backend | CLI Backend |
|---------|-------------|-------------|
| Authentication | API key required | Claude Max subscription |
| Token counting | Accurate | Not available (always 0) |
| Model names | Full names | Aliases (opus, sonnet, haiku) |
| Streaming | True streaming | Simulated (full response) |
| Rate limits | API limits apply | Subscription limits apply |

## Requirements

- Go 1.21+
- Anthropic API key
- 9P client (one of the following):
  - [plan9port](https://9fans.github.io/plan9port/) - Plan 9 tools for Unix (macOS, Linux)
  - [Infernode](https://github.com/NERVsystems/infernode) - Hosted Inferno OS with native 9P
  - Linux 9P kernel module - Built into Linux kernel
  - Native Plan 9 or Inferno OS

## Troubleshooting

### Port already in use

```bash
# Check what's using the port
lsof -i :5640

# Use a different port
./llm9p -addr :5641
```

### Connection refused

Ensure the server is running and listening on the expected port:

```bash
# Start with debug logging to see connections
./llm9p -addr :5640 -debug
```

### Infernode: "localhost" not resolving

Inferno's DNS resolution works differently. Use the IP address directly:

```sh
# Wrong
mount -A tcp!localhost!5640 /n/llm

# Correct
mount -A tcp!127.0.0.1!5640 /n/llm
```

### Infernode: Mount point doesn't exist

Create the mount point before mounting:

```sh
mkdir /n/llm
mount -A tcp!127.0.0.1!5640 /n/llm
```

### Permission denied on write

Some files are read-only by design:
- `tokens` - Read-only (token count from last response)
- `_example` - Read-only (usage examples)
- `stream/chunk` - Read-only (streaming output)

### Empty response from `ask`

The `ask` file only contains content after you write a prompt to it:

```bash
# First write a prompt
echo "Hello" > /mnt/llm/ask

# Then read the response
cat /mnt/llm/ask
```

### API errors

Check that your API key is set and valid:

```bash
# Ensure the key is exported
export ANTHROPIC_API_KEY=sk-ant-...

# Verify it's set
echo $ANTHROPIC_API_KEY
```

## How It Works

1. **Server starts**: llm9p listens for 9P connections on the specified port
2. **Client connects**: A 9P client (9pfuse, Infernode, etc.) connects and negotiates the protocol
3. **Filesystem exposed**: The client sees a virtual filesystem with files like `ask`, `model`, `tokens`
4. **Write prompt**: Writing to `ask` sends the text to Claude via the Anthropic API
5. **Read response**: Reading from `ask` returns Claude's response
6. **State persists**: Conversation history is maintained until you write to `new`

The 9P protocol handles all the complexity of making this look like a regular filesystem, so any tool that can read and write files can interact with the LLM.

## Related Projects

- [Infernode](https://github.com/NERVsystems/infernode) - Hosted Inferno OS with native 9P support
- [plan9port](https://9fans.github.io/plan9port/) - Plan 9 from User Space
- [u9fs](https://github.com/9fans/plan9port/tree/master/src/cmd/9pserve) - 9P file server

## License

MIT

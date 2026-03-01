// llm9p exposes an LLM (Claude) as a 9P filesystem.
//
// Usage:
//
//	ANTHROPIC_API_KEY=sk-... llm9p -addr :5640
//
// Or with Claude Max subscription (via Claude Code CLI):
//
//	llm9p -addr :5640 -backend cli
//
// Mount with:
//
//	9pfuse localhost:5640 /mnt/llm
//
// Interact:
//
//	echo "What is 2+2?" > /mnt/llm/ask
//	cat /mnt/llm/ask
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"syscall"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/llmfs"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

func main() {
	addr := flag.String("addr", ":5640", "Address to listen on")
	debug := flag.Bool("debug", false, "Enable debug logging")
	backend := flag.String("backend", "api", "Backend to use: 'api' (Anthropic API), 'cli' (Claude Code CLI), or 'openai' (OpenAI-compatible local server)")
	openaiURL := flag.String("openai-url", "http://localhost:11434/v1", "Base URL for OpenAI-compatible server (e.g. http://localhost:11434/v1 for Ollama)")
	model := flag.String("model", "", "Model name for OpenAI-compatible backend (e.g. gpt-oss:20b)")
	flag.Parse()

	var client llm.Backend

	switch *backend {
	case "cli":
		// Check that claude CLI is available
		if _, err := exec.LookPath("claude"); err != nil {
			fmt.Fprintln(os.Stderr, "Error: 'claude' CLI not found in PATH")
			fmt.Fprintln(os.Stderr, "Install Claude Code CLI or use -backend api with ANTHROPIC_API_KEY")
			os.Exit(1)
		}
		client = llm.NewCLIClient()
		log.Println("Using Claude Code CLI backend (Claude Max subscription)")

	case "api":
		// Get API key from environment
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			fmt.Fprintln(os.Stderr, "Error: ANTHROPIC_API_KEY environment variable not set")
			fmt.Fprintln(os.Stderr, "Set ANTHROPIC_API_KEY or use -backend cli for Claude Max subscription")
			os.Exit(1)
		}
		client = llm.NewClient(apiKey)
		log.Println("Using Anthropic API backend")

	case "openai":
		if *model == "" {
			fmt.Fprintln(os.Stderr, "Error: -model flag is required for OpenAI-compatible backend")
			fmt.Fprintln(os.Stderr, "Example: -backend openai -model gpt-oss:20b")
			os.Exit(1)
		}
		apiKey := os.Getenv("OPENAI_API_KEY")
		client = llm.NewOpenAIClient(*openaiURL, apiKey, *model)
		log.Printf("Using OpenAI-compatible backend at %s (model: %s)", *openaiURL, *model)

	default:
		fmt.Fprintf(os.Stderr, "Error: unknown backend '%s' (use 'api', 'cli', or 'openai')\n", *backend)
		os.Exit(1)
	}

	// Create session manager for per-fid isolation
	sm := llm.NewSessionManager(client)

	// Create filesystem
	root := llmfs.NewRoot(sm)

	// Create 9P server
	server := protocol.NewServer(root)
	server.SetDebug(*debug)

	// Listen
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", *addr, err)
	}

	log.Printf("llm9p listening on %s", *addr)
	log.Printf("Mount with: 9pfuse %s /mnt/llm", listener.Addr())

	// Handle shutdown gracefully
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		cancel()
		listener.Close()
	}()

	// Serve
	if err := server.Serve(ctx, listener); err != nil && ctx.Err() == nil {
		log.Fatalf("Server error: %v", err)
	}
}

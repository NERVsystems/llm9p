package llmfs

import (
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// Metrics tracks LLM performance statistics
type Metrics struct {
	mu              sync.RWMutex
	requestCount    int64
	totalInputToks  int64
	totalOutputToks int64
	totalLatencyMs  int64
	lastLatencyMs   int64
	minLatencyMs    int64
	maxLatencyMs    int64
	lastRequestTime time.Time
}

// Global metrics instance
var GlobalMetrics = &Metrics{
	minLatencyMs: 999999,
}

// RecordRequest records a completed LLM request
func (m *Metrics) RecordRequest(inputTokens, outputTokens int, latencyMs int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.requestCount++
	m.totalInputToks += int64(inputTokens)
	m.totalOutputToks += int64(outputTokens)
	m.totalLatencyMs += latencyMs
	m.lastLatencyMs = latencyMs
	m.lastRequestTime = time.Now()

	if latencyMs < m.minLatencyMs {
		m.minLatencyMs = latencyMs
	}
	if latencyMs > m.maxLatencyMs {
		m.maxLatencyMs = latencyMs
	}
}

// Report returns a formatted metrics report
func (m *Metrics) Report() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.requestCount == 0 {
		return "requests: 0\n"
	}

	avgLatencyMs := m.totalLatencyMs / m.requestCount
	avgToksPerReq := (m.totalInputToks + m.totalOutputToks) / m.requestCount

	return fmt.Sprintf(`requests: %d
input_tokens: %d
output_tokens: %d
total_tokens: %d
avg_tokens_per_request: %d
last_latency_ms: %d
avg_latency_ms: %d
min_latency_ms: %d
max_latency_ms: %d
last_request: %s
`,
		m.requestCount,
		m.totalInputToks,
		m.totalOutputToks,
		m.totalInputToks+m.totalOutputToks,
		avgToksPerReq,
		m.lastLatencyMs,
		avgLatencyMs,
		m.minLatencyMs,
		m.maxLatencyMs,
		m.lastRequestTime.Format(time.RFC3339),
	)
}

// MetricsFile exposes performance metrics via 9P
type MetricsFile struct {
	*protocol.BaseFile
	client llm.Backend
}

// NewMetricsFile creates the metrics file and registers the metrics callback
func NewMetricsFile(client llm.Backend) *MetricsFile {
	// Register our metrics callback with the llm package
	llm.SetMetricsCallback(func(inputTokens, outputTokens int, latencyMs int64) {
		GlobalMetrics.RecordRequest(inputTokens, outputTokens, latencyMs)
	})

	return &MetricsFile{
		BaseFile: protocol.NewBaseFile("metrics", 0444),
		client:   client,
	}
}

func (f *MetricsFile) Read(p []byte, offset int64) (int, error) {
	content := GlobalMetrics.Report()
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	n := copy(p, content[offset:])
	return n, nil
}

func (f *MetricsFile) Write(p []byte, offset int64) (int, error) {
	return 0, protocol.ErrPermission
}

func (f *MetricsFile) Stat() protocol.Stat {
	s := f.BaseFile.Stat()
	s.Length = uint64(len(GlobalMetrics.Report()))
	return s
}

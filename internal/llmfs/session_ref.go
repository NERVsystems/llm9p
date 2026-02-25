package llmfs

import (
	"sync"

	"github.com/NERVsystems/llm9p/internal/llm"
	"github.com/NERVsystems/llm9p/internal/protocol"
)

// sessionRefFile wraps a protocol.File and tracks a reference to its session.
// It calls sm.IncRef(id) on Open and sm.DecRef(id) on the first Close, so
// that sessions are automatically cleaned up when all their files are clunked.
type sessionRefFile struct {
	protocol.File
	sm     *llm.SessionManager
	id     int
	once   sync.Once
	opened bool
}

func (r *sessionRefFile) Open(mode uint8) error {
	err := r.File.Open(mode)
	if err == nil {
		r.opened = true
		r.sm.IncRef(r.id)
	}
	return err
}

func (r *sessionRefFile) Close() error {
	err := r.File.Close()
	if r.opened {
		r.once.Do(func() {
			r.sm.DecRef(r.id)
		})
	}
	return err
}

// sessionRefDir wraps a protocol.Dir (SessionDir) with the same ref counting.
// It additionally overrides Lookup so that child files are also ref-counted.
type sessionRefDir struct {
	protocol.Dir
	sm     *llm.SessionManager
	id     int
	once   sync.Once
	opened bool
}

func (r *sessionRefDir) Open(mode uint8) error {
	err := r.Dir.Open(mode)
	if err == nil {
		r.opened = true
		r.sm.IncRef(r.id)
	}
	return err
}

func (r *sessionRefDir) Close() error {
	err := r.Dir.Close()
	if r.opened {
		r.once.Do(func() {
			r.sm.DecRef(r.id)
		})
	}
	return err
}

// Lookup wraps child files in sessionRefFile so each open fid contributes
// its own reference to the session.
func (r *sessionRefDir) Lookup(name string) (protocol.File, error) {
	f, err := r.Dir.Lookup(name)
	if err != nil {
		return nil, err
	}
	return &sessionRefFile{File: f, sm: r.sm, id: r.id}, nil
}

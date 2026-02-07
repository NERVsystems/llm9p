package protocol

import (
	"io"
	"sync/atomic"
	"time"
)

// File is the interface that files must implement.
// This is the core abstraction for anything exposed via 9P.
type File interface {
	// Stat returns the file's metadata
	Stat() Stat

	// Open prepares the file for reading/writing
	Open(mode uint8) error

	// Read reads up to len(p) bytes starting at offset
	Read(p []byte, offset int64) (n int, err error)

	// Write writes len(p) bytes starting at offset
	Write(p []byte, offset int64) (n int, err error)

	// Close releases any resources
	Close() error
}

// Dir is the interface that directories must implement
type Dir interface {
	File

	// Children returns the directory's children
	Children() []File

	// Lookup finds a child by name
	Lookup(name string) (File, error)
}

// pathCounter generates unique path IDs for qids
var pathCounter uint64

func NextPath() uint64 {
	return atomic.AddUint64(&pathCounter, 1)
}

// BaseFile provides a default implementation of common File methods
type BaseFile struct {
	Name_   string
	Mode_   uint32
	Uid_    string
	Gid_    string
	Qid_    Qid
	Mtime_  time.Time
	Length_ uint64
}

// NewBaseFile creates a new base file
func NewBaseFile(name string, mode uint32) *BaseFile {
	now := time.Now()
	qtype := QTFILE
	if mode&DMDIR != 0 {
		qtype = QTDIR
	}
	return &BaseFile{
		Name_:  name,
		Mode_:  mode,
		Uid_:   "llm",
		Gid_:   "llm",
		Mtime_: now,
		Qid_: Qid{
			Type:    qtype,
			Version: 0,
			Path:    NextPath(),
		},
	}
}

func (f *BaseFile) Stat() Stat {
	return Stat{
		Type:   0,
		Dev:    0,
		Qid:    f.Qid_,
		Mode:   f.Mode_,
		Atime:  uint32(f.Mtime_.Unix()),
		Mtime:  uint32(f.Mtime_.Unix()),
		Length: f.Length_,
		Name:   f.Name_,
		Uid:    f.Uid_,
		Gid:    f.Gid_,
		Muid:   f.Uid_,
	}
}

func (f *BaseFile) Open(mode uint8) error                   { return nil }
func (f *BaseFile) Close() error                            { return nil }
func (f *BaseFile) Read(p []byte, offset int64) (int, error)  { return 0, io.EOF }
func (f *BaseFile) Write(p []byte, offset int64) (int, error) { return 0, ErrPermission }

// SetLength updates the file length
func (f *BaseFile) SetLength(n uint64) {
	f.Length_ = n
	f.Mtime_ = time.Now()
	f.Qid_.Version++
}

// StaticFile is a file with static content
type StaticFile struct {
	*BaseFile
	Content []byte
}

// NewStaticFile creates a file with static content
func NewStaticFile(name string, content []byte) *StaticFile {
	f := &StaticFile{
		BaseFile: NewBaseFile(name, 0444),
		Content:  content,
	}
	f.Length_ = uint64(len(content))
	return f
}

func (f *StaticFile) Read(p []byte, offset int64) (int, error) {
	if offset >= int64(len(f.Content)) {
		return 0, io.EOF
	}
	n := copy(p, f.Content[offset:])
	return n, nil
}

// StaticDir is a directory with static children
type StaticDir struct {
	*BaseFile
	children map[string]File
	order    []string // preserve order for listing
}

// NewStaticDir creates a new static directory
func NewStaticDir(name string) *StaticDir {
	return &StaticDir{
		BaseFile: NewBaseFile(name, DMDIR|0555),
		children: make(map[string]File),
		order:    make([]string, 0),
	}
}

// AddChild adds a child to the directory
func (d *StaticDir) AddChild(f File) {
	name := f.Stat().Name
	if _, exists := d.children[name]; !exists {
		d.order = append(d.order, name)
	}
	d.children[name] = f
}

func (d *StaticDir) Children() []File {
	result := make([]File, len(d.order))
	for i, name := range d.order {
		result[i] = d.children[name]
	}
	return result
}

func (d *StaticDir) Lookup(name string) (File, error) {
	if f, ok := d.children[name]; ok {
		return f, nil
	}
	return nil, ErrNotFound
}

func (d *StaticDir) Read(p []byte, offset int64) (int, error) {
	// Directory read returns packed stat entries
	var buf []byte
	for _, f := range d.Children() {
		stat := f.Stat()
		entry := make([]byte, 256)
		n := stat.Encode(entry)
		buf = append(buf, entry[:n]...)
	}

	if offset >= int64(len(buf)) {
		return 0, io.EOF
	}

	n := copy(p, buf[offset:])
	return n, nil
}

// DynamicFile is a file whose content is generated on read
type DynamicFile struct {
	*BaseFile
	Generator func() []byte
}

// NewDynamicFile creates a file with dynamic content
func NewDynamicFile(name string, generator func() []byte) *DynamicFile {
	return &DynamicFile{
		BaseFile:  NewBaseFile(name, 0444),
		Generator: generator,
	}
}

func (f *DynamicFile) Read(p []byte, offset int64) (int, error) {
	content := f.Generator()
	if offset >= int64(len(content)) {
		return 0, io.EOF
	}
	n := copy(p, content[offset:])
	return n, nil
}

func (f *DynamicFile) Stat() Stat {
	s := f.BaseFile.Stat()
	s.Length = uint64(len(f.Generator()))
	return s
}

// Errors
type Error string

func (e Error) Error() string { return string(e) }

const (
	ErrNotFound   Error = "file not found"
	ErrPermission Error = "permission denied"
	ErrNotDir     Error = "not a directory"
	ErrIsDir      Error = "is a directory"
	ErrBadFid     Error = "bad fid"
	ErrFidInUse   Error = "fid already in use"
	ErrBadOffset  Error = "bad offset"
)

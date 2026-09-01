// Package protocol implements the 9P2000 protocol for the LLM filesystem.
//
// This is a minimal, clean implementation focused on the subset of 9P
// needed for LLM interaction. It is designed to be:
//   - Zero external dependencies (stdlib only)
//   - LLM-friendly (self-describing, good errors)
//   - Simple to understand and maintain
//
// The 9P protocol uses a simple request-response model over a bidirectional
// stream. Each message has a 4-byte size, 1-byte type, and 2-byte tag,
// followed by type-specific payload.
package protocol

import (
	"encoding/binary"
	"fmt"
	"io"
)

// Protocol constants
const (
	// Version is the protocol version we implement
	Version = "9P2000"

	// MaxMessageSize is the maximum size of a 9P message.
	// Must be large enough for system prompts (which can be 10-15KB with
	// tool documentation and reminders). Inferno's mount() proposes its
	// own msize; the negotiated size is min(client, server).
	MaxMessageSize = 65536

	// MaxWElem is 9P2000's MAXWELEM: the most path elements one Twalk may
	// carry. Clients needing to walk deeper issue successive walks.
	MaxWElem = 16

	// NoTag is used for Tversion/Rversion which don't use tags
	NoTag uint16 = 0xFFFF

	// NoFid represents an invalid fid
	NoFid uint32 = 0xFFFFFFFF
)

// Message types (T = request from client, R = response from server)
const (
	Tversion uint8 = 100
	Rversion uint8 = 101
	Tauth    uint8 = 102
	Rauth    uint8 = 103
	Tattach  uint8 = 104
	Rattach  uint8 = 105
	Terror   uint8 = 106 // never sent
	Rerror   uint8 = 107
	Tflush   uint8 = 108
	Rflush   uint8 = 109
	Twalk    uint8 = 110
	Rwalk    uint8 = 111
	Topen    uint8 = 112
	Ropen    uint8 = 113
	Tcreate  uint8 = 114
	Rcreate  uint8 = 115
	Tread    uint8 = 116
	Rread    uint8 = 117
	Twrite   uint8 = 118
	Rwrite   uint8 = 119
	Tclunk   uint8 = 120
	Rclunk   uint8 = 121
	Tremove  uint8 = 122
	Rremove  uint8 = 123
	Tstat    uint8 = 124
	Rstat    uint8 = 125
	Twstat   uint8 = 126
	Rwstat   uint8 = 127
)

// Open modes
const (
	OREAD  uint8 = 0  // open for read
	OWRITE uint8 = 1  // open for write
	ORDWR  uint8 = 2  // open for read/write
	OEXEC  uint8 = 3  // execute (unused in our context)
	OTRUNC uint8 = 16 // truncate file first
)

// File modes (high bits of Stat.Mode)
const (
	DMDIR    uint32 = 0x80000000 // directory
	DMAPPEND uint32 = 0x40000000 // append only
	DMEXCL   uint32 = 0x20000000 // exclusive use
	DMTMP    uint32 = 0x04000000 // temporary file
)

// Qid represents a unique file identifier
type Qid struct {
	Type    uint8  // QTDIR, QTFILE, etc.
	Version uint32 // version number for cache coherence
	Path    uint64 // unique path identifier
}

// Qid types
const (
	QTDIR    uint8 = 0x80 // directory
	QTAPPEND uint8 = 0x40 // append-only
	QTEXCL   uint8 = 0x20 // exclusive use
	QTTMP    uint8 = 0x04 // temporary
	QTFILE   uint8 = 0x00 // regular file
)

// Stat represents file metadata
type Stat struct {
	Size   uint16 // size of this stat structure (for wire format)
	Type   uint16 // server type
	Dev    uint32 // server device
	Qid    Qid    // unique id
	Mode   uint32 // permissions and flags
	Atime  uint32 // last access time
	Mtime  uint32 // last modification time
	Length uint64 // file length
	Name   string // file name
	Uid    string // owner
	Gid    string // group
	Muid   string // last modifier
}

// Encoder handles encoding messages to the wire format
type Encoder struct {
	w   io.Writer
	buf []byte
}

// NewEncoder creates a new encoder
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		w:   w,
		buf: make([]byte, MaxMessageSize),
	}
}

// Decoder handles decoding messages from the wire format
type Decoder struct {
	r   io.Reader
	buf []byte
}

// NewDecoder creates a new decoder
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{
		r:   r,
		buf: make([]byte, MaxMessageSize),
	}
}

// ReadMessage reads a complete 9P message from the stream
func (d *Decoder) ReadMessage() (msgType uint8, tag uint16, payload []byte, err error) {
	// Read 4-byte size
	if _, err := io.ReadFull(d.r, d.buf[:4]); err != nil {
		return 0, 0, nil, fmt.Errorf("reading size: %w", err)
	}
	size := binary.LittleEndian.Uint32(d.buf[:4])

	if size < 7 {
		return 0, 0, nil, fmt.Errorf("message too small: %d", size)
	}
	if size > MaxMessageSize {
		return 0, 0, nil, fmt.Errorf("message too large: %d", size)
	}

	// Read rest of message
	remaining := size - 4
	if _, err := io.ReadFull(d.r, d.buf[:remaining]); err != nil {
		return 0, 0, nil, fmt.Errorf("reading message: %w", err)
	}

	msgType = d.buf[0]
	tag = binary.LittleEndian.Uint16(d.buf[1:3])
	payload = d.buf[3:remaining]

	return msgType, tag, payload, nil
}

// WriteMessage writes a complete 9P message to the stream
func (e *Encoder) WriteMessage(msgType uint8, tag uint16, payload []byte) error {
	size := uint32(4 + 1 + 2 + len(payload))
	if size > MaxMessageSize {
		return fmt.Errorf("message too large: %d", size)
	}

	binary.LittleEndian.PutUint32(e.buf[0:4], size)
	e.buf[4] = msgType
	binary.LittleEndian.PutUint16(e.buf[5:7], tag)
	copy(e.buf[7:], payload)

	_, err := e.w.Write(e.buf[:size])
	return err
}

// String encoding helpers

func EncodeString(buf []byte, s string) int {
	binary.LittleEndian.PutUint16(buf[0:2], uint16(len(s)))
	copy(buf[2:], s)
	return 2 + len(s)
}

func DecodeString(buf []byte) (string, int) {
	if len(buf) < 2 {
		return "", 0
	}
	size := binary.LittleEndian.Uint16(buf[0:2])
	if len(buf) < int(2+size) {
		return "", 0
	}
	return string(buf[2 : 2+size]), int(2 + size)
}

// Qid encoding

func (q *Qid) Encode(buf []byte) int {
	buf[0] = q.Type
	binary.LittleEndian.PutUint32(buf[1:5], q.Version)
	binary.LittleEndian.PutUint64(buf[5:13], q.Path)
	return 13
}

func DecodeQid(buf []byte) (Qid, int) {
	if len(buf) < 13 {
		return Qid{}, 0
	}
	return Qid{
		Type:    buf[0],
		Version: binary.LittleEndian.Uint32(buf[1:5]),
		Path:    binary.LittleEndian.Uint64(buf[5:13]),
	}, 13
}

// Stat encoding

func (s *Stat) Encode(buf []byte) int {
	// Skip size field, we'll fill it at the end
	n := 2

	// Fixed fields
	binary.LittleEndian.PutUint16(buf[n:n+2], s.Type)
	n += 2
	binary.LittleEndian.PutUint32(buf[n:n+4], s.Dev)
	n += 4
	n += s.Qid.Encode(buf[n:])
	binary.LittleEndian.PutUint32(buf[n:n+4], s.Mode)
	n += 4
	binary.LittleEndian.PutUint32(buf[n:n+4], s.Atime)
	n += 4
	binary.LittleEndian.PutUint32(buf[n:n+4], s.Mtime)
	n += 4
	binary.LittleEndian.PutUint64(buf[n:n+8], s.Length)
	n += 8

	// Variable fields
	n += EncodeString(buf[n:], s.Name)
	n += EncodeString(buf[n:], s.Uid)
	n += EncodeString(buf[n:], s.Gid)
	n += EncodeString(buf[n:], s.Muid)

	// Fill in size (total - 2 for size field itself)
	s.Size = uint16(n - 2)
	binary.LittleEndian.PutUint16(buf[0:2], s.Size)

	return n
}

func DecodeStat(buf []byte) (Stat, int) {
	if len(buf) < 2 {
		return Stat{}, 0
	}

	s := Stat{}
	s.Size = binary.LittleEndian.Uint16(buf[0:2])

	if len(buf) < int(s.Size)+2 {
		return Stat{}, 0
	}

	n := 2
	s.Type = binary.LittleEndian.Uint16(buf[n : n+2])
	n += 2
	s.Dev = binary.LittleEndian.Uint32(buf[n : n+4])
	n += 4

	var qn int
	s.Qid, qn = DecodeQid(buf[n:])
	n += qn

	s.Mode = binary.LittleEndian.Uint32(buf[n : n+4])
	n += 4
	s.Atime = binary.LittleEndian.Uint32(buf[n : n+4])
	n += 4
	s.Mtime = binary.LittleEndian.Uint32(buf[n : n+4])
	n += 4
	s.Length = binary.LittleEndian.Uint64(buf[n : n+8])
	n += 8

	var sn int
	s.Name, sn = DecodeString(buf[n:])
	n += sn
	s.Uid, sn = DecodeString(buf[n:])
	n += sn
	s.Gid, sn = DecodeString(buf[n:])
	n += sn
	s.Muid, sn = DecodeString(buf[n:])
	n += sn

	return s, int(s.Size) + 2
}

// MessageName returns the human-readable name of a message type
func MessageName(t uint8) string {
	names := map[uint8]string{
		Tversion: "Tversion", Rversion: "Rversion",
		Tauth: "Tauth", Rauth: "Rauth",
		Tattach: "Tattach", Rattach: "Rattach",
		Rerror: "Rerror",
		Tflush: "Tflush", Rflush: "Rflush",
		Twalk: "Twalk", Rwalk: "Rwalk",
		Topen: "Topen", Ropen: "Ropen",
		Tcreate: "Tcreate", Rcreate: "Rcreate",
		Tread: "Tread", Rread: "Rread",
		Twrite: "Twrite", Rwrite: "Rwrite",
		Tclunk: "Tclunk", Rclunk: "Rclunk",
		Tremove: "Tremove", Rremove: "Rremove",
		Tstat: "Tstat", Rstat: "Rstat",
		Twstat: "Twstat", Rwstat: "Rwstat",
	}
	if name, ok := names[t]; ok {
		return name
	}
	return fmt.Sprintf("unknown(%d)", t)
}

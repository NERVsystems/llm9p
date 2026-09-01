package protocol

import (
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"
)

// The Linux 9P client turns O_TRUNC into a Twstat setting length 0, sent
// before any Twrite. A plain shell redirect opens with O_TRUNC, so without a
// Twstat handler the server answers "unknown message type" and
//
//	echo hello > /mnt/llm/ask
//
// fails with a bare errno. These tests pin that behaviour.

type session struct {
	t    *testing.T
	conn net.Conn
	tag  uint16
}

func newSession(t *testing.T, root Dir) *session {
	t.Helper()
	client, server := net.Pipe()
	srv := NewServer(root)
	go srv.ServeConn(server)
	t.Cleanup(func() { client.Close() })
	_ = client.SetDeadline(time.Now().Add(5 * time.Second))
	return &session{t: t, conn: client}
}

func (s *session) rpc(typ uint8, payload []byte) (uint8, []byte) {
	s.t.Helper()
	s.tag++
	msg := binary.LittleEndian.AppendUint32(nil, uint32(7+len(payload)))
	msg = append(msg, typ)
	msg = binary.LittleEndian.AppendUint16(msg, s.tag)
	msg = append(msg, payload...)
	if _, err := s.conn.Write(msg); err != nil {
		s.t.Fatalf("write: %v", err)
	}

	var sz [4]byte
	if _, err := io.ReadFull(s.conn, sz[:]); err != nil {
		s.t.Fatalf("read size: %v", err)
	}
	n := binary.LittleEndian.Uint32(sz[:])
	body := make([]byte, n-4)
	if _, err := io.ReadFull(s.conn, body); err != nil {
		s.t.Fatalf("read body: %v", err)
	}
	return body[0], body[3:]
}

func str(b []byte, s string) []byte {
	b = binary.LittleEndian.AppendUint16(b, uint16(len(s)))
	return append(b, s...)
}

func (s *session) handshake() {
	s.t.Helper()
	p := binary.LittleEndian.AppendUint32(nil, 8192)
	if rt, _ := s.rpc(Tversion, str(p, "9P2000")); rt != Rversion {
		s.t.Fatalf("version: got %s", MessageName(rt))
	}
	p = binary.LittleEndian.AppendUint32(nil, 0)        // fid
	p = binary.LittleEndian.AppendUint32(p, ^uint32(0)) // afid = NOFID
	p = str(p, "tester")
	if rt, _ := s.rpc(Tattach, str(p, "")); rt != Rattach {
		s.t.Fatalf("attach: got %s", MessageName(rt))
	}
}

// walkTo clones fid 0 to fid 1 at name, and opens it for writing.
func (s *session) walkTo(name string) {
	s.t.Helper()
	p := binary.LittleEndian.AppendUint32(nil, 0)
	p = binary.LittleEndian.AppendUint32(p, 1)
	p = binary.LittleEndian.AppendUint16(p, 1)
	if rt, _ := s.rpc(Twalk, str(p, name)); rt != Rwalk {
		s.t.Fatalf("walk %s: got %s", name, MessageName(rt))
	}
	p = binary.LittleEndian.AppendUint32(nil, 1)
	p = append(p, 1) // OWRITE
	if rt, _ := s.rpc(Topen, p); rt != Ropen {
		s.t.Fatalf("open %s: got %s", name, MessageName(rt))
	}
}

// wstat builds a Twstat carrying a "don't touch" stat with length 0, which is
// what the kernel sends for O_TRUNC.
func wstatPayload(fid uint32) []byte {
	stat := make([]byte, 0, 64)
	stat = binary.LittleEndian.AppendUint16(stat, 0xFFFF)     // type: unchanged
	stat = binary.LittleEndian.AppendUint32(stat, 0xFFFFFFFF) // dev
	stat = append(stat, 0xFF)                                 // qid.type
	stat = binary.LittleEndian.AppendUint32(stat, 0xFFFFFFFF) // qid.vers
	stat = binary.LittleEndian.AppendUint64(stat, ^uint64(0)) // qid.path
	stat = binary.LittleEndian.AppendUint32(stat, 0xFFFFFFFF) // mode
	stat = binary.LittleEndian.AppendUint32(stat, 0xFFFFFFFF) // atime
	stat = binary.LittleEndian.AppendUint32(stat, 0xFFFFFFFF) // mtime
	stat = binary.LittleEndian.AppendUint64(stat, 0)          // length 0 = truncate
	for i := 0; i < 4; i++ {                                  // name, uid, gid, muid
		stat = binary.LittleEndian.AppendUint16(stat, 0)
	}
	sized := binary.LittleEndian.AppendUint16(nil, uint16(len(stat)))
	sized = append(sized, stat...)

	p := binary.LittleEndian.AppendUint32(nil, fid)
	p = binary.LittleEndian.AppendUint16(p, uint16(len(sized)))
	return append(p, sized...)
}

func testRoot() Dir {
	root := NewStaticDir("/")
	root.AddChild(NewStaticFile("ask", []byte("")))
	return root
}

// TestWstatAcceptedForTruncate is the regression: before the handler existed
// this returned Rerror("unknown message type: 126").
func TestWstatAcceptedForTruncate(t *testing.T) {
	s := newSession(t, testRoot())
	s.handshake()
	s.walkTo("ask")

	rt, payload := s.rpc(Twstat, wstatPayload(1))
	if rt == Rerror {
		msg, _ := DecodeString(payload)
		t.Fatalf("Twstat rejected: %q -- `echo x > file` will fail", msg)
	}
	if rt != Rwstat {
		t.Fatalf("got %s, want Rwstat", MessageName(rt))
	}
}

// An unknown fid must still be refused rather than silently accepted.
func TestWstatUnknownFidRejected(t *testing.T) {
	s := newSession(t, testRoot())
	s.handshake()

	if rt, _ := s.rpc(Twstat, wstatPayload(999)); rt != Rerror {
		t.Fatalf("Twstat on unknown fid returned %s, want Rerror", MessageName(rt))
	}
}

// A truncated Twstat must be refused, not panic. handleConn runs in a bare
// goroutine, so a panic here would be a remote crash.
func TestWstatShortMessageRejected(t *testing.T) {
	s := newSession(t, testRoot())
	s.handshake()

	if rt, _ := s.rpc(Twstat, []byte{1, 2}); rt != Rerror {
		t.Fatalf("short Twstat returned %s, want Rerror", MessageName(rt))
	}
}

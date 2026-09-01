package protocol

import (
	"bytes"
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"
)

// Input here arrives on an untrusted socket. The property under test is that
// the server rejects rather than guesses, and never panics: handleConn runs in
// a bare goroutine, so a panic is a remotely triggerable process kill.

// TestTwalkNwnameBounded is the regression for an allocation amplification:
// a 10-byte message claiming 65535 path elements made the server allocate for
// all of them. 9P2000 caps a walk at MAXWELEM (16).
func TestTwalkNwnameBounded(t *testing.T) {
	p := make([]byte, 10)
	binary.LittleEndian.PutUint32(p[0:4], 1)      // fid
	binary.LittleEndian.PutUint32(p[4:8], 2)      // newfid
	binary.LittleEndian.PutUint16(p[8:10], 65535) // nwname, far over spec

	m, err := DecodeTwalk(p)
	if err != nil {
		return // outright rejection is the ideal outcome
	}
	if len(m.Names) > MaxWElem {
		t.Errorf("allocated %d names from a %d-byte message; MAXWELEM is %d",
			len(m.Names), len(p), MaxWElem)
	}
}

func TestTwalkAcceptsLegalWalk(t *testing.T) {
	p := binary.LittleEndian.AppendUint32(nil, 1)
	p = binary.LittleEndian.AppendUint32(p, 2)
	p = binary.LittleEndian.AppendUint16(p, 2)
	for _, n := range []string{"a", "bb"} {
		p = binary.LittleEndian.AppendUint16(p, uint16(len(n)))
		p = append(p, n...)
	}
	m, err := DecodeTwalk(p)
	if err != nil {
		t.Fatalf("legal 2-element walk rejected: %v", err)
	}
	if len(m.Names) != 2 || m.Names[0] != "a" || m.Names[1] != "bb" {
		t.Errorf("names = %v, want [a bb]", m.Names)
	}
}

// TestDecodersNeverPanic walks every payload decoder across every truncation
// of a plausible message.
func TestDecodersNeverPanic(t *testing.T) {
	full := make([]byte, 64)
	for i := range full {
		full[i] = byte(i)
	}
	decoders := map[string]func([]byte) error{
		"Tversion": func(b []byte) error { _, err := DecodeTversion(b); return err },
		"Tattach":  func(b []byte) error { _, err := DecodeTattach(b); return err },
		"Twalk":    func(b []byte) error { _, err := DecodeTwalk(b); return err },
		"Topen":    func(b []byte) error { _, err := DecodeTopen(b); return err },
		"Tread":    func(b []byte) error { _, err := DecodeTread(b); return err },
		"Twrite":   func(b []byte) error { _, err := DecodeTwrite(b); return err },
		"Tclunk":   func(b []byte) error { _, err := DecodeTclunk(b); return err },
		"Tstat":    func(b []byte) error { _, err := DecodeTstat(b); return err },
		"Tflush":   func(b []byte) error { _, err := DecodeTflush(b); return err },
	}
	for name, dec := range decoders {
		for n := 0; n <= len(full); n++ {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("%s panicked on %d-byte input: %v", name, n, r)
					}
				}()
				_ = dec(full[:n])
			}()
		}
	}
}

func TestReadMessageRejectsBadSizes(t *testing.T) {
	for _, size := range []uint32{0, 6, MaxMessageSize + 1, 0xFFFFFFFF} {
		var buf bytes.Buffer
		_ = binary.Write(&buf, binary.LittleEndian, size)
		buf.Write(make([]byte, 64))
		if _, _, _, err := NewDecoder(&buf).ReadMessage(); err == nil {
			t.Errorf("size %d accepted; want rejection", size)
		}
	}
}

// TestServerSurvivesGarbage throws structurally invalid traffic at the real
// server loop; it must drop the connection, not the process.
func TestServerSurvivesGarbage(t *testing.T) {
	root := NewStaticDir("/")
	root.AddChild(NewStaticFile("f", []byte("x")))
	srv := NewServer(root)

	frame := func(typ uint8, payload []byte) []byte {
		b := binary.LittleEndian.AppendUint32(nil, uint32(7+len(payload)))
		b = append(b, typ)
		b = binary.LittleEndian.AppendUint16(b, 1)
		return append(b, payload...)
	}
	inputs := [][]byte{
		{0x00},
		{0xFF, 0xFF, 0xFF, 0xFF},
		frame(Twalk, []byte{1, 0, 0, 0, 2, 0, 0, 0, 0xFF, 0xFF}),
		frame(Twrite, []byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF}),
		frame(200, []byte{1, 2, 3}), // nonexistent message type
		frame(Tstat, []byte{}),
	}
	for i, in := range inputs {
		client, server := net.Pipe()
		done := make(chan struct{})
		go func() { srv.ServeConn(server); close(done) }()

		_ = client.SetDeadline(time.Now().Add(2 * time.Second))
		_, _ = client.Write(in)
		_, _ = io.ReadAll(io.LimitReader(client, 1024))
		client.Close()

		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatalf("input %d: connection not released", i)
		}
	}
}

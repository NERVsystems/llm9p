package protocol

import (
	"encoding/binary"
	"fmt"
)

// Message is the interface implemented by all 9P messages
type Message interface {
	Type() uint8
	Encode(buf []byte) int
}

// Request messages (T-messages)

// TversionMsg negotiates protocol version
type TversionMsg struct {
	Msize   uint32 // maximum message size
	Version string // protocol version string
}

func (m *TversionMsg) Type() uint8 { return Tversion }

func (m *TversionMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Msize)
	return 4 + EncodeString(buf[4:], m.Version)
}

func DecodeTversion(buf []byte) (*TversionMsg, error) {
	if len(buf) < 6 {
		return nil, fmt.Errorf("Tversion too short")
	}
	m := &TversionMsg{
		Msize: binary.LittleEndian.Uint32(buf[0:4]),
	}
	m.Version, _ = DecodeString(buf[4:])
	return m, nil
}

// RversionMsg is the response to Tversion
type RversionMsg struct {
	Msize   uint32
	Version string
}

func (m *RversionMsg) Type() uint8 { return Rversion }

func (m *RversionMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Msize)
	return 4 + EncodeString(buf[4:], m.Version)
}

// TattachMsg attaches to a filesystem
type TattachMsg struct {
	Fid   uint32 // fid to use for this connection
	Afid  uint32 // auth fid (NoFid if no auth)
	Uname string // user name
	Aname string // attach name (filesystem to attach)
}

func (m *TattachMsg) Type() uint8 { return Tattach }

func (m *TattachMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	binary.LittleEndian.PutUint32(buf[4:8], m.Afid)
	n := 8
	n += EncodeString(buf[n:], m.Uname)
	n += EncodeString(buf[n:], m.Aname)
	return n
}

func DecodeTattach(buf []byte) (*TattachMsg, error) {
	if len(buf) < 12 {
		return nil, fmt.Errorf("Tattach too short")
	}
	m := &TattachMsg{
		Fid:  binary.LittleEndian.Uint32(buf[0:4]),
		Afid: binary.LittleEndian.Uint32(buf[4:8]),
	}
	n := 8
	var sn int
	m.Uname, sn = DecodeString(buf[n:])
	n += sn
	m.Aname, _ = DecodeString(buf[n:])
	return m, nil
}

// RattachMsg is the response to Tattach
type RattachMsg struct {
	Qid Qid
}

func (m *RattachMsg) Type() uint8 { return Rattach }

func (m *RattachMsg) Encode(buf []byte) int {
	return m.Qid.Encode(buf)
}

// TwalkMsg walks a path
type TwalkMsg struct {
	Fid    uint32   // starting fid
	Newfid uint32   // fid for the result
	Names  []string // path components to walk
}

func (m *TwalkMsg) Type() uint8 { return Twalk }

func (m *TwalkMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	binary.LittleEndian.PutUint32(buf[4:8], m.Newfid)
	binary.LittleEndian.PutUint16(buf[8:10], uint16(len(m.Names)))
	n := 10
	for _, name := range m.Names {
		n += EncodeString(buf[n:], name)
	}
	return n
}

func DecodeTwalk(buf []byte) (*TwalkMsg, error) {
	if len(buf) < 10 {
		return nil, fmt.Errorf("Twalk too short")
	}
	m := &TwalkMsg{
		Fid:    binary.LittleEndian.Uint32(buf[0:4]),
		Newfid: binary.LittleEndian.Uint32(buf[4:8]),
	}
	nwname := binary.LittleEndian.Uint16(buf[8:10])
	// 9P2000 caps a walk at MAXWELEM elements. Without this a client can
	// claim 65535 names in a 10-byte message and make the server allocate
	// for all of them.
	if int(nwname) > MaxWElem {
		return nil, fmt.Errorf("Twalk: nwname %d exceeds MAXWELEM %d", nwname, MaxWElem)
	}
	m.Names = make([]string, nwname)
	n := 10
	for i := range m.Names {
		var sn int
		m.Names[i], sn = DecodeString(buf[n:])
		n += sn
	}
	return m, nil
}

// RwalkMsg is the response to Twalk
type RwalkMsg struct {
	Qids []Qid // qids for each successfully walked element
}

func (m *RwalkMsg) Type() uint8 { return Rwalk }

func (m *RwalkMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint16(buf[0:2], uint16(len(m.Qids)))
	n := 2
	for i := range m.Qids {
		n += m.Qids[i].Encode(buf[n:])
	}
	return n
}

// TopenMsg opens a file
type TopenMsg struct {
	Fid  uint32
	Mode uint8
}

func (m *TopenMsg) Type() uint8 { return Topen }

func (m *TopenMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	buf[4] = m.Mode
	return 5
}

func DecodeTopen(buf []byte) (*TopenMsg, error) {
	if len(buf) < 5 {
		return nil, fmt.Errorf("Topen too short")
	}
	return &TopenMsg{
		Fid:  binary.LittleEndian.Uint32(buf[0:4]),
		Mode: buf[4],
	}, nil
}

// RopenMsg is the response to Topen
type RopenMsg struct {
	Qid    Qid
	Iounit uint32
}

func (m *RopenMsg) Type() uint8 { return Ropen }

func (m *RopenMsg) Encode(buf []byte) int {
	n := m.Qid.Encode(buf)
	binary.LittleEndian.PutUint32(buf[n:n+4], m.Iounit)
	return n + 4
}

// TreadMsg reads from a file
type TreadMsg struct {
	Fid    uint32
	Offset uint64
	Count  uint32
}

func (m *TreadMsg) Type() uint8 { return Tread }

func (m *TreadMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	binary.LittleEndian.PutUint64(buf[4:12], m.Offset)
	binary.LittleEndian.PutUint32(buf[12:16], m.Count)
	return 16
}

func DecodeTread(buf []byte) (*TreadMsg, error) {
	if len(buf) < 16 {
		return nil, fmt.Errorf("Tread too short")
	}
	return &TreadMsg{
		Fid:    binary.LittleEndian.Uint32(buf[0:4]),
		Offset: binary.LittleEndian.Uint64(buf[4:12]),
		Count:  binary.LittleEndian.Uint32(buf[12:16]),
	}, nil
}

// RreadMsg is the response to Tread
type RreadMsg struct {
	Data []byte
}

func (m *RreadMsg) Type() uint8 { return Rread }

func (m *RreadMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], uint32(len(m.Data)))
	copy(buf[4:], m.Data)
	return 4 + len(m.Data)
}

// TwriteMsg writes to a file
type TwriteMsg struct {
	Fid    uint32
	Offset uint64
	Data   []byte
}

func (m *TwriteMsg) Type() uint8 { return Twrite }

func (m *TwriteMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	binary.LittleEndian.PutUint64(buf[4:12], m.Offset)
	binary.LittleEndian.PutUint32(buf[12:16], uint32(len(m.Data)))
	copy(buf[16:], m.Data)
	return 16 + len(m.Data)
}

func DecodeTwrite(buf []byte) (*TwriteMsg, error) {
	if len(buf) < 16 {
		return nil, fmt.Errorf("Twrite too short")
	}
	count := binary.LittleEndian.Uint32(buf[12:16])
	if len(buf) < int(16+count) {
		return nil, fmt.Errorf("Twrite data truncated")
	}
	return &TwriteMsg{
		Fid:    binary.LittleEndian.Uint32(buf[0:4]),
		Offset: binary.LittleEndian.Uint64(buf[4:12]),
		Data:   buf[16 : 16+count],
	}, nil
}

// RwriteMsg is the response to Twrite
type RwriteMsg struct {
	Count uint32
}

func (m *RwriteMsg) Type() uint8 { return Rwrite }

func (m *RwriteMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Count)
	return 4
}

// TclunkMsg closes a fid
type TclunkMsg struct {
	Fid uint32
}

func (m *TclunkMsg) Type() uint8 { return Tclunk }

func (m *TclunkMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	return 4
}

func DecodeTclunk(buf []byte) (*TclunkMsg, error) {
	if len(buf) < 4 {
		return nil, fmt.Errorf("Tclunk too short")
	}
	return &TclunkMsg{
		Fid: binary.LittleEndian.Uint32(buf[0:4]),
	}, nil
}

// RclunkMsg is the response to Tclunk
type RclunkMsg struct{}

func (m *RclunkMsg) Type() uint8 { return Rclunk }

func (m *RclunkMsg) Encode(buf []byte) int {
	return 0
}

// TstatMsg requests file stats
type TstatMsg struct {
	Fid uint32
}

func (m *TstatMsg) Type() uint8 { return Tstat }

func (m *TstatMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint32(buf[0:4], m.Fid)
	return 4
}

func DecodeTstat(buf []byte) (*TstatMsg, error) {
	if len(buf) < 4 {
		return nil, fmt.Errorf("Tstat too short")
	}
	return &TstatMsg{
		Fid: binary.LittleEndian.Uint32(buf[0:4]),
	}, nil
}

// RstatMsg is the response to Tstat
type RstatMsg struct {
	Stat Stat
}

func (m *RstatMsg) Type() uint8 { return Rstat }

func (m *RstatMsg) Encode(buf []byte) int {
	// Rstat has an extra 2-byte length prefix for the stat
	statBuf := buf[2:]
	n := m.Stat.Encode(statBuf)
	binary.LittleEndian.PutUint16(buf[0:2], uint16(n))
	return 2 + n
}

// RerrorMsg indicates an error
type RerrorMsg struct {
	Ename string
}

func (m *RerrorMsg) Type() uint8 { return Rerror }

func (m *RerrorMsg) Encode(buf []byte) int {
	return EncodeString(buf, m.Ename)
}

// TflushMsg cancels a pending request
type TflushMsg struct {
	Oldtag uint16
}

func (m *TflushMsg) Type() uint8 { return Tflush }

func (m *TflushMsg) Encode(buf []byte) int {
	binary.LittleEndian.PutUint16(buf[0:2], m.Oldtag)
	return 2
}

func DecodeTflush(buf []byte) (*TflushMsg, error) {
	if len(buf) < 2 {
		return nil, fmt.Errorf("Tflush too short")
	}
	return &TflushMsg{
		Oldtag: binary.LittleEndian.Uint16(buf[0:2]),
	}, nil
}

// RflushMsg is the response to Tflush
type RflushMsg struct{}

func (m *RflushMsg) Type() uint8 { return Rflush }

func (m *RflushMsg) Encode(buf []byte) int {
	return 0
}

package protocol

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
)

// Server is a 9P file server
type Server struct {
	root    Dir
	debug   bool
	mu      sync.Mutex
	clients map[net.Conn]*clientState
}

// clientState tracks state for a single client connection
type clientState struct {
	fids  map[uint32]File
	msize uint32
}

// NewServer creates a new 9P server with the given root directory
func NewServer(root Dir) *Server {
	return &Server{
		root:    root,
		clients: make(map[net.Conn]*clientState),
	}
}

// SetDebug enables debug logging
func (s *Server) SetDebug(debug bool) {
	s.debug = debug
}

// Serve handles incoming connections on the listener
func (s *Server) Serve(ctx context.Context, listener net.Listener) error {
	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				log.Printf("accept error: %v", err)
				continue
			}
		}

		go s.handleConn(conn)
	}
}

// ServeConn handles a single connection (useful for testing)
func (s *Server) ServeConn(conn net.Conn) {
	s.handleConn(conn)
}

func (s *Server) handleConn(conn net.Conn) {
	defer conn.Close()

	state := &clientState{
		fids:  make(map[uint32]File),
		msize: MaxMessageSize,
	}

	s.mu.Lock()
	s.clients[conn] = state
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		delete(s.clients, conn)
		s.mu.Unlock()
	}()

	dec := NewDecoder(conn)
	enc := NewEncoder(conn)
	buf := make([]byte, MaxMessageSize)

	for {
		msgType, tag, payload, err := dec.ReadMessage()
		if err != nil {
			if err != io.EOF {
				log.Printf("read error: %v", err)
			}
			return
		}

		if s.debug {
			log.Printf("< %s tag=%d len=%d", MessageName(msgType), tag, len(payload))
		}

		resp, respType := s.handleMessage(state, msgType, payload, buf)

		if s.debug {
			log.Printf("> %s tag=%d len=%d", MessageName(respType), tag, len(resp))
		}

		if err := enc.WriteMessage(respType, tag, resp); err != nil {
			log.Printf("write error: %v", err)
			return
		}
	}
}

func (s *Server) handleMessage(state *clientState, msgType uint8, payload []byte, buf []byte) ([]byte, uint8) {
	switch msgType {
	case Tversion:
		return s.handleVersion(state, payload, buf)
	case Tattach:
		return s.handleAttach(state, payload, buf)
	case Twalk:
		return s.handleWalk(state, payload, buf)
	case Topen:
		return s.handleOpen(state, payload, buf)
	case Tread:
		return s.handleRead(state, payload, buf)
	case Twrite:
		return s.handleWrite(state, payload, buf)
	case Tclunk:
		return s.handleClunk(state, payload, buf)
	case Tstat:
		return s.handleStat(state, payload, buf)
	case Tflush:
		return s.handleFlush(state, payload, buf)
	default:
		return s.errorResponse(buf, fmt.Sprintf("unknown message type: %d", msgType))
	}
}

func (s *Server) errorResponse(buf []byte, msg string) ([]byte, uint8) {
	resp := &RerrorMsg{Ename: msg}
	n := resp.Encode(buf)
	return buf[:n], Rerror
}

func (s *Server) handleVersion(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTversion(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	// Negotiate message size
	msize := msg.Msize
	if msize > MaxMessageSize {
		msize = MaxMessageSize
	}
	state.msize = msize

	// Check version - accept both 9P2000 and Styx (Inferno's name)
	version := msg.Version
	if msg.Version != Version && msg.Version != "Styx" {
		version = "unknown"
	}

	if s.debug {
		log.Printf("Version negotiation: client=%q responding=%q msize=%d", msg.Version, version, msize)
	}

	resp := &RversionMsg{Msize: msize, Version: version}
	n := resp.Encode(buf)
	return buf[:n], Rversion
}

func (s *Server) handleAttach(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTattach(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	if _, exists := state.fids[msg.Fid]; exists {
		return s.errorResponse(buf, ErrFidInUse.Error())
	}

	state.fids[msg.Fid] = s.root

	resp := &RattachMsg{Qid: s.root.Stat().Qid}
	n := resp.Encode(buf)
	return buf[:n], Rattach
}

func (s *Server) handleWalk(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTwalk(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	file, exists := state.fids[msg.Fid]
	if !exists {
		return s.errorResponse(buf, ErrBadFid.Error())
	}

	if msg.Fid != msg.Newfid {
		if _, exists := state.fids[msg.Newfid]; exists {
			return s.errorResponse(buf, ErrFidInUse.Error())
		}
	}

	// Walk the path
	qids := make([]Qid, 0, len(msg.Names))
	current := file

	for _, name := range msg.Names {
		dir, ok := current.(Dir)
		if !ok {
			return s.errorResponse(buf, ErrNotDir.Error())
		}

		next, err := dir.Lookup(name)
		if err != nil {
			// Return partial walk
			break
		}

		qids = append(qids, next.Stat().Qid)
		current = next
	}

	// Only update fid if we walked at least one element (or no elements requested)
	if len(qids) == len(msg.Names) {
		state.fids[msg.Newfid] = current
	}

	resp := &RwalkMsg{Qids: qids}
	n := resp.Encode(buf)
	return buf[:n], Rwalk
}

func (s *Server) handleOpen(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTopen(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	file, exists := state.fids[msg.Fid]
	if !exists {
		return s.errorResponse(buf, ErrBadFid.Error())
	}

	if err := file.Open(msg.Mode); err != nil {
		return s.errorResponse(buf, err.Error())
	}

	resp := &RopenMsg{
		Qid:    file.Stat().Qid,
		Iounit: 0, // 0 means use msize - overhead
	}
	n := resp.Encode(buf)
	return buf[:n], Ropen
}

func (s *Server) handleRead(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTread(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	file, exists := state.fids[msg.Fid]
	if !exists {
		return s.errorResponse(buf, ErrBadFid.Error())
	}

	// Limit read size to available buffer
	count := msg.Count
	maxData := state.msize - 4 - 1 - 2 - 4 // size, type, tag, count
	if count > maxData {
		count = maxData
	}

	data := make([]byte, count)
	n, err := file.Read(data, int64(msg.Offset))
	if err != nil && err != io.EOF {
		return s.errorResponse(buf, err.Error())
	}

	resp := &RreadMsg{Data: data[:n]}
	rn := resp.Encode(buf)
	return buf[:rn], Rread
}

func (s *Server) handleWrite(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTwrite(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	file, exists := state.fids[msg.Fid]
	if !exists {
		return s.errorResponse(buf, ErrBadFid.Error())
	}

	n, err := file.Write(msg.Data, int64(msg.Offset))
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	resp := &RwriteMsg{Count: uint32(n)}
	rn := resp.Encode(buf)
	return buf[:rn], Rwrite
}

func (s *Server) handleClunk(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTclunk(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	file, exists := state.fids[msg.Fid]
	if !exists {
		return s.errorResponse(buf, ErrBadFid.Error())
	}

	file.Close()
	delete(state.fids, msg.Fid)

	resp := &RclunkMsg{}
	n := resp.Encode(buf)
	return buf[:n], Rclunk
}

func (s *Server) handleStat(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	msg, err := DecodeTstat(payload)
	if err != nil {
		return s.errorResponse(buf, err.Error())
	}

	file, exists := state.fids[msg.Fid]
	if !exists {
		return s.errorResponse(buf, ErrBadFid.Error())
	}

	resp := &RstatMsg{Stat: file.Stat()}
	n := resp.Encode(buf)
	return buf[:n], Rstat
}

func (s *Server) handleFlush(state *clientState, payload []byte, buf []byte) ([]byte, uint8) {
	// We don't have async operations to cancel, so just respond OK
	resp := &RflushMsg{}
	n := resp.Encode(buf)
	return buf[:n], Rflush
}

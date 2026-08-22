// SPDX-License-Identifier: Apache-2.0

package sentencepiece

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// modelProto is the slice of sentencepiece_model.proto rembed needs:
//
//	message ModelProto {
//	  repeated SentencePiece pieces = 1;   // piece=1 string, score=2 float, type=3 enum
//	  optional TrainerSpec trainer_spec = 2;      // unused
//	  optional NormalizerSpec normalizer_spec = 3;
//	}
//	message NormalizerSpec {
//	  optional string name = 1;
//	  optional bytes precompiled_charsmap = 2;
//	  optional bool add_dummy_prefix = 3 [default = true];
//	  optional bool remove_extra_whitespaces = 4 [default = true];
//	  optional bool escape_whitespaces = 5 [default = true];
//	}
//
// Field numbers were read from the installed sentencepiece proto
// descriptor, not guessed. The parser below is a minimal proto2 wire
// reader — varint, fixed32, and length-delimited are the only wire types
// these messages use — so rembed needs no protobuf dependency.
type modelProto struct {
	pieces []piece
	norm   normalizerSpec
}

type piece struct {
	text  string
	score float32
	kind  int // 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 6=BYTE
}

type normalizerSpec struct {
	name                   string
	precompiledCharsmap    []byte
	addDummyPrefix         bool
	removeExtraWhitespaces bool
	escapeWhitespaces      bool
}

// parseModel reads a .model file (SentencePiece ModelProto).
func parseModel(path string) (*modelProto, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m := &modelProto{
		// proto2 defaults: the three bools default to TRUE when absent.
		norm: normalizerSpec{addDummyPrefix: true, removeExtraWhitespaces: true, escapeWhitespaces: true},
	}
	err = walkFields(raw, func(field int, wire int, data []byte, varint uint64) error {
		switch field {
		case 1: // pieces
			if wire != 2 {
				return fmt.Errorf("pieces: wire type %d", wire)
			}
			p := piece{kind: 1} // proto2 default type = NORMAL
			if err := walkFields(data, func(f, w int, d []byte, v uint64) error {
				switch f {
				case 1:
					p.text = string(d)
				case 2:
					p.score = math.Float32frombits(uint32(v))
				case 3:
					p.kind = int(v)
				}
				return nil
			}); err != nil {
				return err
			}
			m.pieces = append(m.pieces, p)
		case 3: // normalizer_spec
			if wire != 2 {
				return fmt.Errorf("normalizer_spec: wire type %d", wire)
			}
			return walkFields(data, func(f, w int, d []byte, v uint64) error {
				switch f {
				case 1:
					m.norm.name = string(d)
				case 2:
					m.norm.precompiledCharsmap = d
				case 3:
					m.norm.addDummyPrefix = v != 0
				case 4:
					m.norm.removeExtraWhitespaces = v != 0
				case 5:
					m.norm.escapeWhitespaces = v != 0
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("sentencepiece: %s: %w", path, err)
	}
	if len(m.pieces) == 0 {
		return nil, fmt.Errorf("sentencepiece: %s: no pieces (not a SentencePiece model?)", path)
	}
	return m, nil
}

// walkFields iterates a proto2 wire-format message, invoking fn per
// field. Length-delimited fields pass data; varint and fixed32 pass the
// numeric value (fixed32 arrives in varint for uniformity).
func walkFields(buf []byte, fn func(field, wire int, data []byte, varint uint64) error) error {
	for len(buf) > 0 {
		tag, n := binary.Uvarint(buf)
		if n <= 0 {
			return fmt.Errorf("bad varint tag")
		}
		buf = buf[n:]
		field := int(tag >> 3)
		wire := int(tag & 7)
		switch wire {
		case 0: // varint
			v, n := binary.Uvarint(buf)
			if n <= 0 {
				return fmt.Errorf("bad varint value (field %d)", field)
			}
			buf = buf[n:]
			if err := fn(field, wire, nil, v); err != nil {
				return err
			}
		case 5: // fixed32 (float)
			if len(buf) < 4 {
				return fmt.Errorf("short fixed32 (field %d)", field)
			}
			v := binary.LittleEndian.Uint32(buf)
			buf = buf[4:]
			if err := fn(field, wire, nil, uint64(v)); err != nil {
				return err
			}
		case 1: // fixed64
			if len(buf) < 8 {
				return fmt.Errorf("short fixed64 (field %d)", field)
			}
			v := binary.LittleEndian.Uint64(buf)
			buf = buf[8:]
			if err := fn(field, wire, nil, v); err != nil {
				return err
			}
		case 2: // length-delimited
			l, n := binary.Uvarint(buf)
			if n <= 0 || uint64(len(buf)-n) < l {
				return fmt.Errorf("bad length-delimited field %d", field)
			}
			data := buf[n : n+int(l)]
			buf = buf[n+int(l):]
			if err := fn(field, wire, data, 0); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unsupported wire type %d (field %d)", wire, field)
		}
	}
	return nil
}

/* pivco_huffman_wire.h — single source of truth for the per-node wire format.
 *
 * All backends MUST consume/produce the per-non-flat-internal-node wire
 * record through these helpers.  Previously each backend hand-rolled
 * the read/write of K_right header + FSE marker byte + bitmap bytes,
 * which led to silent drift (scalar+NEON added the FSE marker byte in
 * 2026-05-13, x86+AVX-512 didn't — broke scalar↔SSE cross-decoding).
 *
 * Wire format (v0.6 — post-order records, pre-order splits; branch
 * postorder-bitmaps, 2026-07-02):
 *
 * Strictly, the layout is an Euler-tour hybrid, not pure post-order:
 * each node's bytes land in two places — the K_right varint at its
 * PRE-order position (on the way down; forward parsing of
 * variable-size regions requires sizing information in prefix
 * position), and the marker+bitmap record (the bulk) at its POST-order
 * position (on the way up, where the merge consumes it).  A pure
 * post-order stream would need every subtree region to be
 * self-delimiting, i.e. a length at every terminal instead of one per
 * recursion site — more slots, plus explicit zero-lengths for symbols
 * absent from a block.
 *
 * Per-block header (once, at the very start of each encoded block):
 *   [block_N: uint16 LE, 2 bytes]                  symbol count N for this
 *                                                  block; the decoder reads
 *                                                  it before starting the
 *                                                  tree walk.  Lets the
 *                                                  codec encode any N up to
 *                                                  65535 — no longer pinned
 *                                                  to PIVCO_BLOCK_SIZE.
 *
 * Per non-flat internal node — the node's record comes AFTER its
 * children's regions:
 *
 *   [optional K_right: LEB128 varint, 1-3 bytes]   if kr_header_needed();
 *                                                  emitted BEFORE the child
 *                                                  regions (at node entry)
 *   [left child region][right child region]        recursively, same layout
 *   [FSE marker byte:        uint8,    1 byte]    always
 *   [bitmap body]                                  marker == 0: raw n-bit
 *                                                  bitmap, ceil(n/8) bytes
 *                                                  marker != 0: 2-byte LE
 *                                                  fse_len + fse_len bytes
 *                                                  of FSE-compressed bytes
 *
 * Rationale: the BU decoder CONSUMES each bitmap only at merge time,
 * after both children are decoded.  v0.5's pre-order layout therefore
 * loaded bitmaps backwards (each merge reached behind the cursor to a
 * region it skipped earlier), keeping the whole compressed block live
 * in cache.  Post-order puts every record exactly where the decode
 * consumes it: the stream is read strictly forward, each byte touched
 * once, so the L1 working set is a moving window instead of the block.
 *
 * The K_right varint is consumed at node entry (before the children)
 * because the recursion needs both child counts up front to carve the
 * scratch arena.  Same slot set as v0.5's u16 headers (one per
 * recursion site into a non-leaf child, kr_header_needed()); LEB128
 * (7 bits/byte, LSB-first) instead of fixed u16 shaves the common
 * K_right < 128 case to 1 byte.  Leaf-only nodes (BOTH_LEAVES, HALF_*
 * with a leaf child) carry no length in either scheme, and empty
 * subtrees (n == 0) emit nothing at all.
 *
 * Flat-subtree nodes do NOT use the per-node record — they emit n·D
 * packed bits directly (post-order position, but they have no children
 * so it is the same place).  See pivco_huffman.h:flat_depth.
 *
 * Internal header, not part of the public API.
 */

#ifndef PIVCO_HUFFMAN_WIRE_H
#define PIVCO_HUFFMAN_WIRE_H

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_prof.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif

#include <stdint.h>
#include <string.h>

#define PIVCO_BLOCK_N_BYTES 2  /* per-block N header: uint16 little-endian */

/* ---------- Per-block N header ---------- */

/* Encode: write the block's symbol count N as the first 2 bytes of the
 * encoded stream.  N <= 65535 (the existing PIVCO_BLOCK_SIZE of 8192/4096
 * leaves plenty of headroom; uint16 caps any future variable-block work
 * at the same 65535 limit). */
static inline void wire_write_block_n(uint8_t *out_ptr, int n)
{
    out_ptr[0] = (uint8_t)(n & 0xFF);
    out_ptr[1] = (uint8_t)((n >> 8) & 0xFF);
}

/* Decode: read the block's symbol count N from the first 2 bytes and
 * advance *in_ptr. */
static inline int wire_read_block_n(const uint8_t **in_ptr)
{
    uint16_t v;
    memcpy(&v, *in_ptr, 2);
    *in_ptr += PIVCO_BLOCK_N_BYTES;
    return (int)v;
}

/* ---------- K_right varint (LEB128, LSB-first 7-bit groups) ----------
 *
 * Unlike v0.5's reserve/commit u16 (the value wasn't known until the
 * partition ran), the post-order encoder writes the varint directly:
 * the partition runs before anything is emitted for the node.
 * K_right <= n <= PIVCO_WIRE_MAX_N (65535) => at most 3 bytes. */
static inline void wire_write_kr_varint(uint8_t **out_ptr, int v)
{
    while (v >= 0x80) {
        *(*out_ptr)++ = (uint8_t)(v | 0x80);
        v >>= 7;
    }
    *(*out_ptr)++ = (uint8_t)v;
}

static inline int wire_read_kr_varint(const uint8_t **in_ptr)
{
    PROF_TIC();
    int v = 0, shift = 0;
    uint8_t b;
    do {
        b = *(*in_ptr)++;
        v |= (int)(b & 0x7F) << shift;
        shift += 7;
    } while (b & 0x80);
    PROF_TOC(PROF_WIRE_KR, 1);
    return v;
}

/* Note: the FSE marker byte + bitmap (or FSE payload) is emitted by
 * the backend's `prim_encode_node` primitive, not by a helper here.
 * Backends that attempt FSE-coding of the bitmap need to make that
 * decision after building the raw bitmap, which is intrinsically
 * backend-specific; threading a wire-helper through that flow would
 * be more complexity than win.  The wire FORMAT — 1 byte marker
 * followed by raw bitmap (marker == 0) or [fse_len:u16][fse_payload]
 * (marker != 0) — is still authoritative here in the header doc, and
 * `wire_read_bitmap` below is the corresponding decoder. */

/* ---------- Decode side ---------- */

/* Read the per-node bitmap body (marker + payload).  Returns a pointer
 * to the usable n-bit bitmap (either pointing into the input stream
 * for marker==0, or into the caller-provided `scratch` for the FSE
 * path).  Advances *in_ptr past the whole record.
 *
 * scratch must hold at least bitmap_bytes(n) bytes and stay live for
 * the entire span where the returned pointer is dereferenced. */
static inline const uint8_t *wire_read_bitmap(const uint8_t **in_ptr,
                                                int n,
                                                uint8_t *scratch)
{
    PROF_TIC();
    int nbytes = bitmap_bytes(n);
    uint8_t marker = **in_ptr;
    *in_ptr += 1;
    if (marker == 0) {
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        PROF_TOC(PROF_WIRE_BITMAP_RAW, n);
        return bm;
    }
#ifdef PIVCO_HAS_FSE
    int t_id = marker & 0x7F;
    int xor_flag = (marker >> 7) & 1;
    uint16_t fse_len;
    memcpy(&fse_len, *in_ptr, 2);
    *in_ptr += 2;
    size_t out_len = 0;
    (void)pivco_fse_decompress(t_id, *in_ptr, fse_len,
                                scratch, (size_t)nbytes,
                                (size_t)nbytes, &out_len);
    *in_ptr += fse_len;
    if (xor_flag) pivco_fse_flip_bits(scratch, (size_t)nbytes);
    PROF_TOC(PROF_WIRE_BITMAP_FSE, n);
    return scratch;
#else
    /* FSE not built but stream uses it — best-effort fallback.  The
     * caller will produce wrong output; the file codec will catch the
     * mismatch.  We don't fault, just advance and return zeros. */
    (void)scratch;
    *in_ptr += nbytes;
    PROF_TOC(PROF_WIRE_BITMAP_FSE, n);
    return *in_ptr - nbytes;
#endif
}

#endif  /* PIVCO_HUFFMAN_WIRE_H */

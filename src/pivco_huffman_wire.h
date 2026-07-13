/* pivco_huffman_wire.h — single source of truth for the per-node wire format.
 *
 * All backends MUST consume/produce the per-non-flat-internal-node wire
 * record through these helpers.  Previously each backend hand-rolled
 * the read/write of K_right header + FSE marker byte + bitmap bytes,
 * which led to silent drift (scalar+NEON added the FSE marker byte in
 * 2026-05-13, x86+AVX-512 didn't — broke scalar↔SSE cross-decoding).
 *
 * Wire format (v0.7+):
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
 * The block body is an Euler walk of the tree in DECOMPRESSION ORDER
 * (v0.6): a node's pieces land exactly where the BU decoder consumes
 * them, so the input cursor moves strictly forward, single-touch.  Per
 * non-flat internal node (FULL / LEAF_LEFT; the former both-leaves PAIR
 * record is gone — sibling pairs are flat D=1 regions):
 *
 *   [K_right_header: uint16 LE, 2 bytes]   at the node's PRE-order
 *                                          position — the decoder sizes
 *                                          both children before their
 *                                          regions arrive
 *   ... the children's regions, LARGER-K child first (v0.7; strict >,
 *       ties left-first — a leaf child emits nothing) ...
 *   [FSE marker byte: uint8, 1 byte]       at the node's POST-order
 *   [bitmap body]                          position, right where the
 *                                          decoder merges.
 *                                          marker == 0: raw n-bit
 *                                          bitmap, ceil(n/8) bytes
 *                                          marker != 0: 2-byte LE
 *                                          fse_len + fse_len bytes
 *                                          of FSE-compressed bytes
 *
 * Flat-subtree nodes do NOT use this header — they emit n·D packed bits
 * directly at their (single) visit position.  See
 * pivco_huffman.h:flat_depth.
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

/* ---------- Encode side ----------
 *
 * The encoder partitions BEFORE emitting anything for the node (the
 * bitmap is staged across the child recursion), so K_right is known up
 * front and written directly — the old reserve/commit pair is gone.
 *
 * Every non-flat schedule record (FULL / LEAF_LEFT) carries a K_right
 * header — the pair records that had none are flat D=1 regions now —
 * so the production codec calls the unconditional wire_write_kr /
 * wire_read_kr; the read *_header variant keyed on the explicit tree
 * remains for legacy decoders. */
static inline void wire_write_kr(uint8_t **out_ptr, int n_right)
{
    uint8_t *slot = *out_ptr;
    slot[0] = (uint8_t)(n_right & 0xFF);
    slot[1] = (uint8_t)((n_right >> 8) & 0xFF);
    *out_ptr += KR_HEADER_BYTES;
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

/* Read the K_right header (unconditional form; see the reserve-side
 * note for when a header is present). */
static inline int wire_read_kr(const uint8_t **in_ptr)
{
    PROF_TIC();
    uint16_t v;
    memcpy(&v, *in_ptr, 2);
    *in_ptr += KR_HEADER_BYTES;
    PROF_TOC(PROF_WIRE_KR, 1);
    return (int)v;
}

/* Skip the K_right header bytes, returning the value as an int.  If no
 * header is present for this node, returns -1.  (Top-down decoders
 * don't use the value; bottom-up ones do.) */
static inline int wire_read_kr_header(const pivco_huffman_table_t *table,
                                       int16_t node_id,
                                       const uint8_t **in_ptr)
{
    if (!kr_header_needed(table, node_id)) return -1;
    return wire_read_kr(in_ptr);
}

/* Bounds-checked reads for the production decoder: the stream is
 * untrusted, so every record read is validated against `end` before
 * dereferencing.  wire_read_kr_checked returns -1 on truncation;
 * wire_read_bitmap_checked returns NULL on truncation, a bad FSE
 * marker, or an FSE payload that fails to decode to exactly the bitmap
 * size.  Costs a couple of predictable compares per NODE record —
 * nothing per symbol. */
static inline int wire_read_kr_checked(const uint8_t **in_ptr,
                                       const uint8_t *end)
{
    if (end - *in_ptr < KR_HEADER_BYTES) return -1;
    return wire_read_kr(in_ptr);
}

static inline const uint8_t *wire_read_bitmap_checked(const uint8_t **in_ptr,
                                                      const uint8_t *end,
                                                      int n, uint8_t *scratch)
{
    PROF_TIC();
    int nbytes = bitmap_bytes(n);
    if (*in_ptr >= end) return NULL;
    uint8_t marker = **in_ptr;
    *in_ptr += 1;
    if (marker == 0) {
        if (end - *in_ptr < nbytes) return NULL;
        const uint8_t *bm = *in_ptr;
        *in_ptr += nbytes;
        PROF_TOC(PROF_WIRE_BITMAP_RAW, n);
        return bm;
    }
#ifdef PIVCO_HAS_FSE
    int t_id = marker & 0x7F;
    int xor_flag = (marker >> 7) & 1;
    if (end - *in_ptr < 2) return NULL;
    uint16_t fse_len;
    memcpy(&fse_len, *in_ptr, 2);
    *in_ptr += 2;
    if (end - *in_ptr < (ptrdiff_t)fse_len) return NULL;
    size_t out_len = 0;
    if (pivco_fse_decompress(t_id, *in_ptr, fse_len,
                             scratch, (size_t)nbytes,
                             (size_t)nbytes, &out_len) != PIVCO_FSE_OK
        || out_len != (size_t)nbytes)
        return NULL;    /* bad table id / malformed payload */
    *in_ptr += fse_len;
    if (xor_flag) pivco_fse_flip_bits(scratch, (size_t)nbytes);
    PROF_TOC(PROF_WIRE_BITMAP_FSE, n);
    return scratch;
#else
    /* FSE not built but the stream claims an FSE record: reject. */
    (void)scratch;
    return NULL;
#endif
}

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

/* pivco_huffman_codec.c — unified encode + bottom-up decode.
 *
 * One source file, compiled once per backend tier (CMake passes
 * -DPIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}).  The tree walk + dispatch
 * + wire format are identical across backends; the per-node SIMD work
 * lives in pivco_huffman_primitives_<backend>.h, selected via the
 * router pivco_huffman_primitives.h.
 *
 * Two responsibilities only:
 *
 *   1. Walk the Huffman tree (encode recursion + BU decode recursion).
 *   2. Read/write per-node wire records via pivco_huffman_wire.h.
 *
 * Everything backend-shaped (bitmap build, partition, flat-decode,
 * merge, etc.) is a `prim_*` call.  No vector types here.
 *
 * The bottom-up decoder is the production path (top-down has been
 * parked).  Encode is shared.
 */

#include "pivco_huffman.h"
#include "pivco_huffman_common.h"
#include "pivco_huffman_primitives.h"   /* before wire.h: defines PIVCO_PRIM_DEC_* */
#include "pivco_huffman_wire.h"
#include "pivco_prof.h"
#ifdef PIVCO_HAS_FSE
#include "pivco_fse.h"
#endif

#include <stdlib.h>
#include <string.h>
#include "pivco_check.h"

/* Thread-local growable decode scratch arena.  Replaces the former fixed
 * `static __thread` array sized at PIVCO_BLOCK_SIZE so the block size is a
 * pure runtime parameter (the wire N header carries the per-block count).
 * Grows on demand and is reused across blocks; never shrinks.  One instance
 * per backend translation unit, which is exactly what we want. */
static __thread uint8_t *g_decode_scratch     = NULL;
static __thread size_t   g_decode_scratch_cap = 0;

static uint8_t *decode_scratch_ensure(size_t need)
{
    if (need > g_decode_scratch_cap) {
        uint8_t *p = (uint8_t *)realloc(g_decode_scratch, need);
        if (!p) return NULL;
        g_decode_scratch     = p;
        g_decode_scratch_cap = need;
    }
    return g_decode_scratch;
}

/* MERGE_OVERREAD: the SIMD merges load their source buffers in
 * full-vector chunks, so they may READ (never write) up to this many
 * bytes past a source's end.  Every buffer the decode walk hands to a
 * merge therefore needs this much trailing slack inside the arena; the
 * caller's `symbols` buffer, which guarantees none, is only ever a
 * merge DESTINATION (writes are exact). */
#define MERGE_OVERREAD ((size_t)PIVCO_PRIM_MERGE_OVERREAD)

/* Thread-local growable encode scratch arena.  Mirrors the decode arena
 * above: holds the per-block ranks buffer + the tree-walk's right-half
 * recursion scratch, grown on demand and reused across blocks (never shrinks),
 * so a block-loop encode doesn't malloc/free per block.
 * @todo stopgap: this per-thread global should become part of an explicit
 * encoder API context (a pivco_huffman_encoder_t handle) so the scratch's
 * ownership and lifetime are caller-controlled rather than a hidden
 * thread_local.  See IDEAS.md. */
static __thread uint8_t *g_encode_scratch     = NULL;
static __thread size_t   g_encode_scratch_cap = 0;

static uint8_t *encode_scratch_ensure(size_t need)
{
    if (need > g_encode_scratch_cap) {
        uint8_t *p = (uint8_t *)realloc(g_encode_scratch, need);
        if (!p) return NULL;
        g_encode_scratch     = p;
        g_encode_scratch_cap = need;
    }
    return g_encode_scratch;
}

/* ---------- FSE dispatch parameters ----------
 *
 * The thresholds match the NEON encoder's settings so the wire format
 * is byte-identical across backends: same skew threshold, same per-
 * codeword cost gate, same minimum bitmap size.  See docs/FSE-V0.md for the
 * derivation of each value.  Overridable at build time via -D... . */
#ifndef PIVCO_FSE_MIN_THRESHOLD
#define PIVCO_FSE_MIN_THRESHOLD 0.625
#endif
#ifndef PIVCO_FSE_MIN_RATIO
#define PIVCO_FSE_MIN_RATIO     0.95
#endif
#ifndef PIVCO_FSE_MIN_BITMAP_BYTES
#define PIVCO_FSE_MIN_BITMAP_BYTES 32
#endif

/* FSE per-table-id stats live in src/pivco_huffman.c (backend-neutral
 * TU) so the symbols resolve regardless of backend.  codec.c writes
 * the counters every time it commits or rejects an FSE attempt. */
extern uint64_t g_pivco_fse_commit  [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_attempt [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_bytes_in [PIVCO_FSE_STATS_SLOTS];
extern uint64_t g_pivco_fse_bytes_out[PIVCO_FSE_STATS_SLOTS];

/* ---------- Backend → entry-point name ---------- */

#if defined(PIVCO_BACKEND_SCALAR)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_scalar
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_scalar
#elif defined(PIVCO_BACKEND_NEON)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_neon
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_neon
#elif defined(PIVCO_BACKEND_X86)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_x86
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_x86
#elif defined(PIVCO_BACKEND_AVX512)
#  define CODEC_ENCODE_ENTRY pivco_huffman_encode_avx512
#  define CODEC_DECODE_ENTRY pivco_huffman_decode_bu_avx512
#else
#  error "pivco_huffman_codec.c needs PIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}"
#endif

/* ---------- Encode tree walk ---------- *
 *
 * DFS, emitting records in decompression order (an Euler walk): the
 * partition runs at node entry (it routes the ranks the recursion
 * needs) and the K_right header is written there too, but the node's
 * marker+bitmap record is emitted AFTER the children's regions —
 * exactly where the decoder's merge consumes it, so the decoder reads
 * the stream strictly forward.  At each non-flat internal node,
 * `ranks[0..n)` holds the surviving leaves' in-order ranks; partition
 * routes each by `rank > split_rank[node]`, leaving the left half in
 * place in `ranks[0..n_left)` and compacting the right half into
 * `tmp[0..n_right)`.  The recursion descends left on `ranks`, right on
 * `tmp`.  The bitmap is staged in a stack buffer across the recursion
 * (its final stream position depends on the children's — FSE-variable —
 * encoded sizes, so it can't be written in place up front);
 * ≤ bitmap_bytes(N)+64 per level, tree height ≤ PIVCO_MAX_CODE_LEN
 * levels. */

/* Arch-agnostic FSE attempt on a freshly-built raw bitmap.
 *
 * Inputs:
 *   marker_slot  — points at the 1-byte FSE marker (currently 0 = raw)
 *   bm           — points at the ceil(n/8)-byte raw bitmap region
 *                  immediately after the marker
 *   nbytes       — bitmap_bytes(n)
 *   n / n_left / n_right — partition counts (for the skew test)
 *   depth        — for the codeword-cost gate
 *   out_ptr      — cursor; advanced past the FSE payload on commit
 *
 * On commit: rewrites *marker_slot, replaces bm with [fse_len:u16
 * LE][fse_payload], advances *out_ptr to one past the payload.  Stats
 * (g_pivco_fse_*) are bumped.
 *
 * On no-commit / no-attempt: stream and stats untouched.
 *
 * No-op when PIVCO_HAS_FSE is not defined. */
static inline void codec_maybe_fse_attempt(uint8_t *marker_slot,
                                            uint8_t *bm, int nbytes,
                                            int n, int n_left, int n_right,
                                            int depth, uint8_t **out_ptr)
{
#ifdef PIVCO_HAS_FSE
    if (!pivco_huffman_get_fse_enabled()) return;
    if (nbytes < PIVCO_FSE_MIN_BITMAP_BYTES) return;

    int n_major = (n_left >= n_right) ? n_left : n_right;
    double p_major = (n > 0) ? (double)n_major / (double)n : 0.0;
    if (p_major < PIVCO_FSE_MIN_THRESHOLD) return;

    int t_id = pivco_fse_select_table(p_major);
    if (t_id < 1) return;
    PIVCO_CHECK(t_id < PIVCO_FSE_STATS_SLOTS);  /* guards the g_pivco_fse_* indexing */

    int xor_flag = (n_right > n_left);
    uint8_t scratch[(size_t)nbytes + 16];
    if (xor_flag) {
        for (int i = 0; i < nbytes; i++) scratch[i] = (uint8_t)~bm[i];
    } else {
        memcpy(scratch, bm, (size_t)nbytes);
    }

    uint8_t fse_out[(size_t)nbytes + 64];
    size_t fse_len = 0;
    pivco_fse_status_t rc = pivco_fse_compress(t_id, scratch, (size_t)nbytes,
                                                fse_out, sizeof(fse_out),
                                                &fse_len);
    g_pivco_fse_attempt[t_id]++;
    if (rc != PIVCO_FSE_OK) {
        g_pivco_fse_commit[0]++;     /* slot 0 = attempted, rejected */
        return;
    }
    /* Per-codeword commit gate (see docs/FSE-V0.md):
     *   raw: every codeword through this node costs (depth + 1) bits
     *   fse: (depth + (fse_len + 2 wire-prefix) * 8 / n) bits
     * Commit iff (depth + fse_frac) <= MIN_RATIO * (depth + 1). */
    double fse_frac = (double)(fse_len + 2) * 8.0 / (double)n;
    double codeword_ratio = ((double)depth + fse_frac) /
                              ((double)depth + 1.0);
    if (codeword_ratio > (double)PIVCO_FSE_MIN_RATIO) {
        g_pivco_fse_commit[0]++;
        return;
    }

    /* Commit: rewrite marker + bitmap region with [fse_len][payload],
     * adjust the wire cursor to one past the payload. */
    *marker_slot = (uint8_t)((xor_flag ? 0x80 : 0) | t_id);
    uint8_t *p = bm;
    *p++ = (uint8_t)( fse_len       & 0xFF);
    *p++ = (uint8_t)((fse_len >> 8) & 0xFF);
    memcpy(p, fse_out, fse_len);
    *out_ptr = p + fse_len;

    g_pivco_fse_commit  [t_id]++;
    g_pivco_fse_bytes_in [t_id] += (uint64_t)nbytes;
    g_pivco_fse_bytes_out[t_id] += (uint64_t)(fse_len + 3);
#else
    (void)marker_slot; (void)bm; (void)nbytes;
    (void)n; (void)n_left; (void)n_right; (void)depth; (void)out_ptr;
#endif
}

static void codec_encode_node(const pivco_huffman_table_t *table,
                               int16_t node_id,
                               uint8_t *ranks, int n,
                               int depth,
                               uint8_t **out_ptr,
                               uint8_t *tmp)
{
    if (n == 0) return;
    PROF_COUNT_ONLY(PROF_ENC_NODE_VISIT, n);

    const pivco_tree_node_t *node = &table->tree[node_id];
    if (node->symbol >= 0) return;  /* leaf — nothing to emit */

    /* Flat-subtree fast path: pack n*D bits, no marker, no K_right. */
    if (table->flat_depth[node_id] >= 2) {
        int D = table->flat_depth[node_id];
        int total_bytes = (n * D + 7) >> 3;
        PROF_TIC();
        prim_enc_pack_dN(ranks, n, D, table->flat_base_rank[node_id], *out_ptr);
        PROF_TOC(PROF_ENC_FLAT, n);
        *out_ptr += total_bytes;
        return;
    }

    /* Non-flat internal node.  codec.c owns: K_right header, FSE marker
     * byte, optional FSE-attempt on the raw bitmap.  The arch-specific
     * primitive does only the SIMD-bound work: build the raw bitmap and
     * partition the ranks.
     *
     * The bitmap is built into a stack staging buffer (+64 slack
     * absorbs the SIMD partitions' over-wide tail stores) and copied
     * into the stream after the children's regions. */
    int nbytes = bitmap_bytes(n);
    uint8_t bm_stage[(size_t)nbytes + 64];

    /* Pick the partition variant by node_type, mirroring the decode-side
     * dispatch.  The bitmap (and thus the wire bytes) is identical across
     * variants; only the encode-internal scatter work differs — a leaf child
     * never reads its scattered side, so that side's scatter is skipped:
     * BOTH_LEAVES stores nothing, LEAF_LEFT only the right (compacted into
     * tmp), FULL both. */
    uint8_t thr = table->split_rank[node_id];
    int n_right;
    PROF_TIC();
    switch ((pivco_node_type_t)table->node_type[node_id]) {
    case PIVCO_NODE_BOTH_LEAVES:
        n_right = prim_enc_partition_none(ranks, n, thr, bm_stage);        break;
    case PIVCO_NODE_LEAF_LEFT:
        n_right = prim_enc_partition_right(ranks, n, thr, bm_stage, tmp);  break;
    default:
        n_right = prim_enc_partition_full(ranks, n, thr, bm_stage, tmp);   break;
    }
    PROF_TOC(PROF_ENC_NODE_FULL, n);
    int n_left  = n - n_right;

    /* One K_right header per recursion site, consumed by the decoder
     * at node entry so it can size both children before their regions
     * arrive. */
    wire_write_kr_header(table, node_id, out_ptr, n_right);

    /* Emit the larger-K child's region first: the decoder can then
     * decode it into scratch that the smaller, not-yet-decoded
     * sibling's buffer overlaps (hole-reuse), shrinking the arena
     * high-water.  The two rank buffers (ranks=left, tmp=right) and
     * the shared deeper scratch tmp+n_right are mutually disjoint, so
     * the call order is free.  A leaf child emits nothing, so this
     * only changes the stream at INTERNAL_FULL nodes — exactly where
     * the decoder reorders. */
    if (n_right > n_left) {
        codec_encode_node(table, node->right, tmp,   n_right, depth + 1,
                           out_ptr, tmp + n_right);
        codec_encode_node(table, node->left,  ranks, n_left,  depth + 1,
                           out_ptr, tmp + n_right);
    } else {
        codec_encode_node(table, node->left,  ranks, n_left,  depth + 1,
                           out_ptr, tmp + n_right);
        codec_encode_node(table, node->right, tmp,   n_right, depth + 1,
                           out_ptr, tmp + n_right);
    }

    /* Emit this node's record: marker + staged bitmap.  The FSE attempt
     * may rewrite marker+bm in place with [fse_len][payload] and pull
     * *out_ptr back to the payload end.  No-op otherwise. */
    uint8_t *marker_slot = *out_ptr;
    *marker_slot = 0;
    *out_ptr += 1;
    uint8_t *bm = *out_ptr;
    memcpy(bm, bm_stage, (size_t)nbytes);
    *out_ptr += nbytes;
    codec_maybe_fse_attempt(marker_slot, bm, nbytes,
                             n, n_left, n_right, depth, out_ptr);
}

int CODEC_ENCODE_ENTRY(const uint8_t *symbols, size_t n,
                       const pivco_huffman_table_t *table,
                       uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;
    if (n == 0 || n > PIVCO_WIRE_MAX_N) return PIVCO_ERR_OVERFLOW;
    prim_codec_init();

    const int N = (int)n;

    /* Block header: write N as the first 2 bytes so the decoder can
     * recover it without an out-of-band channel. */
    uint8_t *ptr = out;
    wire_write_block_n(ptr, N);
    ptr += PIVCO_BLOCK_N_BYTES;

    /* One heap block: the per-block ranks buffer + the recursion's right-half
     * scratch (see the tree-walk note above).  +64 slack on ranks absorbs the
     * SIMD partition's over-wide (16/64-byte) tail store at end-of-buffer; the
     * scratch holds one right-half per recursion level, hence (MAX_CODE_LEN+2)*N. */
    const size_t ranks_capacity = (size_t)N + 64;
    const size_t tmp_capacity   = (size_t)N * (PIVCO_MAX_CODE_LEN + 2);
    uint8_t *ranks = encode_scratch_ensure(ranks_capacity + tmp_capacity);
    if (!ranks) return PIVCO_ERR_NULL;
    uint8_t *tmp = ranks + ranks_capacity;

    /* ranks[i] = in-order rank of symbols[i] (gather table->sym_to_rank). */
    PROF_COUNT_ONLY(PROF_ENC_ENTRY, N);
    PROF_TIC();
    prim_enc_init(ranks, N, symbols, table->sym_to_rank, &table->enc_init_aux);
    PROF_TOC(PROF_ENC_INIT, N);

    codec_encode_node(table, table->tree_root, ranks, N, 0, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

/* ---------- Bottom-up decode tree walk (ping-pong scratch) ---------- *
 *
 * Each call decodes a subtree's K symbols into out[0,K).  Internal
 * nodes recurse into their children, then merge per the node's bitmap.
 * The flat-subtree fast path bypasses recursion entirely.
 *
 * The wire is in decompression order — larger-K child first — so the
 * input cursor is consumed strictly forward and each record is loaded
 * exactly where it is used: the K_right header at node entry, the
 * children's regions during their recursion, the node's bitmap right
 * before its merge.
 *
 * Scratch placement is a two-buffer ping-pong (out, tmp), made
 * tail-free-safe by carrying explicit WRITE LIMITS:
 *
 *   - the LARGER child decodes in place into out's tail, shifted
 *     DEC_SHIFT bytes right of the classic K_small position: any
 *     rightward shift preserves the in-place invariant (by the time
 *     the merge writes out[i] it has consumed at least i - K_small
 *     tail bytes, so its read frontier leads the write cursor by the
 *     shift), and the shift is what gives the out-prefix partner its
 *     slack (below);
 *   - the SMALLER child decodes into tmp[0, K_small), and its own
 *     recursion uses out's still-empty prefix out[0, K_small +
 *     DEC_SHIFT) as ITS partner — the pair (tmp, out-prefix)
 *     ping-pongs down the smaller-child spine instead of growing an
 *     arena.
 *
 * Tail-free backends (PIVCO_PRIM_DEC_STORE_QUANTUM = Q > 1) write a
 * K-symbol region as exactly ceilq(K) = ceil(K/Q)*Q bytes of stores,
 * so every placement carries an explicit end: the INVARIANT for a
 * (buf, K, buf_end) triple is buf + ceilq(K) <= buf_end (checked at
 * placement).  A child whose preferred slot fails the invariant —
 * possible only in tight prefix partners hosting deep spines / odd
 * sizes — falls back to a bump slice from the arena's spill region
 * (cursor threaded by value, so a subtree's slices are reclaimed when
 * it completes).  Exact backends (Q == 1, DEC_SHIFT == 0) satisfy
 * every check by construction and reproduce the classic placement
 * bit-for-bit.
 *
 * Merges may additionally READ up to MERGE_OVERREAD past any source's
 * end — reads are unconstrained inside the arena (its tail slack
 * covers the last region); only writes obey buf_end.
 *
 * Dispatch on node_type, computed at build-table time (by children's
 * leafness — a leaf child's symbol goes straight into the parent's
 * merge, so the walk never recurses into a leaf):
 *
 *   INTERNAL_FLAT   — packed-bits flat decode into out
 *   BOTH_LEAVES     — both children leaves, merge_cst_cst directly
 *   LEAF_LEFT       — left child leaf, recurse right (in place, into
 *                     out's tail), merge_cst_vec
 *   INTERNAL_FULL   — both children internal: larger child in place,
 *                     smaller via the ping-pong partner, merge_vec_vec */

#define DEC_Q ((size_t)PIVCO_PRIM_DEC_STORE_QUANTUM)
#define DEC_SHIFT (PIVCO_PRIM_DEC_STORE_QUANTUM > 1 ? 16 : 0)
/* Per-slice spill headroom: enough for a full spine of shifts below a
 * bumped child, so a bump slice never re-bumps for geometry alone. */
#define DEC_SLOP ((size_t)DEC_SHIFT * (PIVCO_MAX_CODE_LEN + 2))

static inline size_t dec_ceilq(size_t k)
{
    return (k + DEC_Q - 1) & ~(DEC_Q - 1);
}

/* Decode-walk I/O context, constant across a block's walk. */
typedef struct {
    const uint8_t *in_end;      /* one past the last readable input byte */
    uint8_t       *in_bounce;   /* N+16 slab: end-of-input flat tails    */
    uint8_t       *bump_limit;  /* end of the bump spill region          */
} codec_dec_io_t;

/* Flat region decode with end-of-input protection.  The packed-flat
 * kernels read up to SRC_SLACK bytes past the region (except D == 8,
 * which is an exact memcpy).  When a region ends within SRC_SLACK of
 * in_end — in practice the block's final region against a tight
 * buffer — decode the longest 16-code-aligned prefix straight from
 * the stream (16 codes = 2*D bytes, so the split is byte-aligned, and
 * the prefix call's reads stay at or under in_end) and only the
 * remaining tail codes from a copy in `slab`: a <= 48-byte memcpy
 * instead of bouncing the whole region. */
static inline void codec_flat_region(uint8_t *out, int K,
                                     const uint8_t *bm, int D,
                                     const uint8_t *c2s,
                                     const uint8_t *in_end,
                                     uint8_t *slab)
{
    int total_bytes = (K * D + 7) >> 3;
    if (PIVCO_PRIM_DEC_SRC_SLACK == 0 || D == 8
        || bm + total_bytes + PIVCO_PRIM_DEC_SRC_SLACK <= in_end) {
        prim_merge_flat(out, K, bm, D, c2s);
        return;
    }
    size_t avail = (size_t)(in_end - bm);
    size_t safe  = avail > (size_t)PIVCO_PRIM_DEC_SRC_SLACK
                 ? avail - (size_t)PIVCO_PRIM_DEC_SRC_SLACK : 0;
    int n1 = (int)((safe * 8 / (size_t)D) & ~(size_t)15);
    if (n1 > K) n1 = K & ~15;
    if (n1 > 0) prim_merge_flat(out, n1, bm, D, c2s);
    int n2 = K - n1;
    if (n2 > 0) {
        size_t b1 = (size_t)n1 * (size_t)D / 8;
        memcpy(slab, bm + b1, (size_t)total_bytes - b1);
        prim_merge_flat(out + n1, n2, slab, D, c2s);
    }
}

/* ---------- Exact-tail root epilogue (tail-free backends) ----------
 *
 * A tail-free merge stores ceilq(K) bytes, so for K not a quantum
 * multiple it cannot target the caller's `symbols` directly.  Instead
 * of bouncing the whole block through scratch, split at n1 = K & ~15:
 * the prefix merge's stores are exact (n1 is chunk-aligned) and bit
 * n1 is byte-aligned in the bitmap, so the remaining n2 = K - n1
 * (1..15) symbols decode as an independent sub-merge into a 16-byte
 * temp, memcpy'd out.  Cursors are recovered O(1) from the wire's
 * K_right and a popcount of the <=15 tail bits.  Only the entry's
 * root merges (the only ones targeting `symbols`) use these. */

/* popcount of bitmap bits [n1, n1+n2), n1 chunk-aligned, n2 <= 15.
 * The 2-byte read is within the merges' 2*ceil(K/16) bitmap read
 * bound, so it is covered by wire_read_bitmap's bounce decision. */
static inline int codec_bm_tail_pop(const uint8_t *bm, int n1, int n2)
{
    uint16_t w;
    memcpy(&w, bm + (n1 >> 3), 2);
    return __builtin_popcount((unsigned)w & ((1u << n2) - 1u));
}

static void codec_merge_vec_vec_exact(const uint8_t *bm, int K, int K_right,
                                      const uint8_t *left,
                                      const uint8_t *right,
                                      uint8_t *out)
{
    int n1 = K & ~15, n2 = K - n1;
    prim_merge_vec_vec(bm, n1, left, right, out);
    int r  = codec_bm_tail_pop(bm, n1, n2);
    int rc = K_right - r;
    int lc = (K - K_right) - (n2 - r);
    uint8_t t[16];
    prim_merge_vec_vec(bm + (n1 >> 3), n2, left + lc, right + rc, t);
    memcpy(out + n1, t, (size_t)n2);
}

static void codec_merge_cst_vec_exact(const uint8_t *bm, int K, int K_right,
                                      uint8_t left_sym,
                                      const uint8_t *right,
                                      uint8_t *out)
{
    int n1 = K & ~15, n2 = K - n1;
    prim_merge_cst_vec(bm, n1, left_sym, right, out);
    int rc = K_right - codec_bm_tail_pop(bm, n1, n2);
    uint8_t t[16];
    prim_merge_cst_vec(bm + (n1 >> 3), n2, left_sym, right + rc, t);
    memcpy(out + n1, t, (size_t)n2);
}

static void codec_merge_cst_cst_exact(const uint8_t *bm, int K,
                                      uint8_t left_sym, uint8_t right_sym,
                                      uint8_t *out)
{
    int n1 = K & ~15, n2 = K - n1;
    prim_merge_cst_cst(bm, n1, left_sym, right_sym, out);
    uint8_t t[16];
    prim_merge_cst_cst(bm + (n1 >> 3), n2, left_sym, right_sym, t);
    memcpy(out + n1, t, (size_t)n2);
}

/* codec_flat_region with EXACT stores for the flat-root-into-symbols
 * case: chunk-aligned prefix direct, <=15 tail codes from a small
 * stack copy into a 16-byte temp.  D == 8 is a memcpy kernel — exact
 * both ways. */
static void codec_flat_region_exact(uint8_t *out, int K,
                                    const uint8_t *bm, int D,
                                    const uint8_t *c2s,
                                    const uint8_t *in_end,
                                    uint8_t *slab)
{
    if (D == 8) {
        prim_merge_flat(out, K, bm, D, c2s);
        return;
    }
    int n1 = K & ~15, n2 = K - n1;
    codec_flat_region(out, n1, bm, D, c2s, in_end, slab);
    size_t b1 = (size_t)n1 * (size_t)D / 8;
    size_t tail_len = (size_t)((K * D + 7) >> 3) - b1;
    uint8_t src[32];   /* <=15 codes * <=7 bits: 14 B data + 16 B kernel lookahead */
    memcpy(src, bm + b1, tail_len);
    uint8_t t[16];
    prim_merge_flat(t, n2, src, D, c2s);
    memcpy(out + n1, t, (size_t)n2);
}

static void codec_decode_subtree(const pivco_huffman_table_t *table,
                                   int16_t node_id, int K,
                                   uint8_t *out, uint8_t *out_end,
                                   uint8_t *tmp, uint8_t *tmp_end,
                                   uint8_t *bump,
                                   const uint8_t **in_ptr,
                                   const codec_dec_io_t *io)
{
    if (K == 0) return;

    PIVCO_CHECK_DEBUG(out + dec_ceilq((size_t)K) <= out_end);

    const pivco_tree_node_t *node = &table->tree[node_id];

    switch ((pivco_node_type_t)table->node_type[node_id]) {

    case PIVCO_NODE_LEAF:
        /* Unreachable: every parent consumes a leaf child via its
         * cst_* merge instead of recursing into it. */
        pivco_check_fail("codec_decode_subtree dispatched on a leaf",
                         __FILE__, __LINE__);

    case PIVCO_NODE_INTERNAL_FLAT: {
        int D = table->flat_depth[node_id];
        int total_bytes = (K * D + 7) >> 3;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[node_id]];
        codec_flat_region(out, K, bm, D, c2s, io->in_end, io->in_bounce);
        return;
    }

    case PIVCO_NODE_BOTH_LEAVES: {
        /* No K_right header (kr_header_needed returns false). */
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch,
                                             io->in_end);
        prim_merge_cst_cst(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           (uint8_t)table->tree[node->right].symbol,
                           out);
        return;
    }

    case PIVCO_NODE_LEAF_LEFT: {
        /* One internal child (right); the leaf contributes the K_left
         * symbols the merge fills into out's prefix.  The right child
         * decodes in place into out's tail (shifted), or into a bump
         * slice when the shifted slot's stores would pass out_end. */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        uint8_t *right_buf = out + (K - K_right) + DEC_SHIFT;
        if (right_buf + dec_ceilq((size_t)K_right) <= out_end) {
            codec_decode_subtree(table, node->right, K_right,
                                  right_buf, out_end, tmp, tmp_end,
                                  bump, in_ptr, io);
        } else {
            right_buf = bump;
            uint8_t *slice_end = bump + dec_ceilq((size_t)K_right) + DEC_SLOP;
            PIVCO_CHECK(slice_end <= io->bump_limit);
            codec_decode_subtree(table, node->right, K_right,
                                  right_buf, slice_end, tmp, tmp_end,
                                  slice_end, in_ptr, io);
        }

        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch,
                                             io->in_end);
        prim_merge_cst_vec(bm, K,
                           (uint8_t)table->tree[node->left].symbol,
                           right_buf, out);
        return;
    }

    case PIVCO_NODE_INTERNAL_FULL:
    default: {
        /* Both children internal.  Stream order == decode order ==
         * larger first (strict >, ties left-first — must match the
         * encoder).  The larger child sits in place at out + K_small +
         * DEC_SHIFT (or a bump slice if that overruns out_end); the
         * smaller child goes to tmp (or a bump slice), its partner
         * being out's still-empty prefix up to the larger child's
         * data. */
        int K_right = wire_read_kr_header(table, node_id, in_ptr);
        int K_left  = K - K_right;
        int K_large = K_right > K_left ? K_right : K_left;
        int K_small = K - K_large;
        int16_t child_large = K_right > K_left ? node->right : node->left;
        int16_t child_small = K_right > K_left ? node->left  : node->right;

        /* Larger child: in place (shifted) or bump slice. */
        uint8_t *large_buf = out + K_small + DEC_SHIFT;
        uint8_t *prefix_end = large_buf;   /* live boundary in out */
        uint8_t *bump_after = bump;
        if (large_buf + dec_ceilq((size_t)K_large) <= out_end) {
            codec_decode_subtree(table, child_large, K_large,
                                  large_buf, out_end, tmp, tmp_end,
                                  bump, in_ptr, io);
        } else {
            large_buf = bump;
            uint8_t *slice_end = bump + dec_ceilq((size_t)K_large) + DEC_SLOP;
            PIVCO_CHECK(slice_end <= io->bump_limit);
            bump_after = slice_end;
            prefix_end = out_end;          /* out is entirely free */
            codec_decode_subtree(table, child_large, K_large,
                                  large_buf, slice_end, tmp, tmp_end,
                                  slice_end, in_ptr, io);
        }

        /* Smaller child: tmp (partner = out's prefix) or bump slice. */
        uint8_t *small_buf = tmp;
        if (small_buf + dec_ceilq((size_t)K_small) <= tmp_end) {
            codec_decode_subtree(table, child_small, K_small,
                                  small_buf, tmp_end, out, prefix_end,
                                  bump_after, in_ptr, io);
        } else {
            small_buf = bump_after;
            uint8_t *slice_end = bump_after + dec_ceilq((size_t)K_small) + DEC_SLOP;
            PIVCO_CHECK(slice_end <= io->bump_limit);
            codec_decode_subtree(table, child_small, K_small,
                                  small_buf, slice_end, out, prefix_end,
                                  slice_end, in_ptr, io);
        }

        uint8_t *left_buf  = K_right > K_left ? small_buf : large_buf;
        uint8_t *right_buf = K_right > K_left ? large_buf : small_buf;

        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap(in_ptr, K, bm_scratch,
                                             io->in_end);
        prim_merge_vec_vec(bm, K, left_buf, right_buf, out);
        return;
    }
    }
}

/* Exact caller contract (same as the classic codec): `symbols`
 * receives exactly N bytes and no read passes in + in_len.  Tail-free
 * backends confine their over-wide stores/loads to the arena: the
 * root's own merge targets `symbols` directly only when N is a
 * multiple of the store quantum (always true for production block
 * sizes); otherwise it goes through the arena's dst slab + exact
 * memcpy.  Input regions ending within SRC_SLACK of in + in_len are
 * bounced into padded scratch before the kernels read them. */
int CODEC_DECODE_ENTRY(const uint8_t *in, size_t in_len,
                       const pivco_huffman_table_t *table,
                       uint8_t *symbols, size_t *consumed)
{
    if (!in || !table || !symbols || !consumed) return PIVCO_ERR_NULL;
    if (in_len < PIVCO_BLOCK_N_BYTES) return PIVCO_ERR_CORRUPT;
    prim_codec_init();

    /* Block header: first 2 bytes are N (symbol count for this block). */
    const uint8_t *ptr = in;
    const uint8_t *in_end = in + in_len;
    const int N = wire_read_block_n(&ptr);
    if (N <= 0 || N > PIVCO_WIRE_MAX_N) return PIVCO_ERR_CORRUPT;
    const pivco_tree_node_t *root = &table->tree[table->tree_root];

    /* Root-is-leaf: fill everything with the single symbol. */
    if (root->symbol >= 0) {
        memset(symbols, (uint8_t)root->symbol, (size_t)N);
        *consumed = 0;
        return PIVCO_OK;
    }

    /* Arena geometry (all offsets from the arena base):
     *   [0, walk_off)            ping-pong walk region: root slots +
     *                            partner regions (true footprint
     *                            ~1.5N + slops; 2N + slop is safe)
     *   [walk_off, bump_off)     bump spill region for placements that
     *                            fail the ceilq end-invariant
     *   [bump_off, +N+16)        in_bounce slab (end-of-input regions)
     * plus MERGE_OVERREAD trailing read slack. */
    const size_t slab_sz  = (size_t)N + 16;
    const size_t walk_off = 2 * (size_t)N + 2 * DEC_SLOP + 512;
    /* Bump region: at most two slices live per recursion level (larger
     * + smaller child), each ceilq(K) data + DEC_SLOP headroom, K sums
     * to <= N per level — so N*(L+2) for the data plus a constant for
     * the per-slice headroom (dominant for tiny N). */
    const size_t bump_off = walk_off + (size_t)N * (PIVCO_MAX_CODE_LEN + 2)
                          + 2 * (PIVCO_MAX_CODE_LEN + 2) * (DEC_SLOP + 64);
    const size_t need     = bump_off + slab_sz + MERGE_OVERREAD;

    /* Root merge stores are exact iff N is a store-quantum multiple;
     * otherwise the entry's root merges run the exact-tail epilogue
     * (chunk-aligned prefix direct + <=15-symbol sub-merge via a
     * 16-byte temp) — no block-sized bounce, no unaligned-N cliff. */
    const int root_tail =
        (PIVCO_PRIM_DEC_STORE_QUANTUM > 1)
        && ((size_t)N % DEC_Q) != 0;

    /* Fast path: BOTH_LEAVES at root — a 2-symbol (or single-symbol)
     * tree, where the whole block collapses to "read the K-bit
     * partition, blend two symbols".  Skips the recursive
     * codec_decode_subtree machinery (switch dispatch + bm_scratch
     * stack frame + scratch TLS reference / arena ensure).  Worth −26%
     * on two_sym decode on older narrow x86 (IvyBridge), noise on
     * modern hosts.  (Still no arena reference: the unaligned-N
     * epilogue works on stack temps.) */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_BOTH_LEAVES) {
        uint8_t bm_scratch[(size_t)bitmap_bytes(N) + 16];
        const uint8_t *bm = wire_read_bitmap(&ptr, N, bm_scratch, in_end);
        const pivco_tree_node_t *left_child  = &table->tree[root->left];
        const pivco_tree_node_t *right_child = &table->tree[root->right];
        if (root_tail)
            codec_merge_cst_cst_exact(bm, N,
                               (uint8_t)left_child->symbol,
                               (uint8_t)right_child->symbol,
                               symbols);
        else
            prim_merge_cst_cst(bm, N,
                               (uint8_t)left_child->symbol,
                               (uint8_t)right_child->symbol,
                               symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Flat root: the whole tree is one packed-bits region, decoded
     * straight into symbols (no merge, no walk).  The arena is needed
     * only when the region abuts in_end (src split); the unaligned-N
     * epilogue works on stack temps. */
    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_INTERNAL_FLAT) {
        int D = table->flat_depth[table->tree_root];
        int total_bytes = (N * D + 7) >> 3;
        const uint8_t *bm = ptr;
        ptr += total_bytes;
        const uint8_t *c2s =
            &table->flat_code_to_sym[table->flat_offset[table->tree_root]];
        const int src_split_needed =
            PIVCO_PRIM_DEC_SRC_SLACK > 0 && D != 8
            && bm + total_bytes + PIVCO_PRIM_DEC_SRC_SLACK > in_end;
        uint8_t *slab = NULL;
        if (src_split_needed) {
            uint8_t *scratch = decode_scratch_ensure(need);
            if (!scratch) return PIVCO_ERR_NULL;
            slab = scratch + bump_off;
        }
        if (root_tail)
            codec_flat_region_exact(symbols, N, bm, D, c2s, in_end, slab);
        else
            codec_flat_region(symbols, N, bm, D, c2s, in_end, slab);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Remaining root shapes recurse.  The ping-pong walk parks children
     * in its out buffer's tail and prefix and the merges READ their
     * sources with up-to-MERGE_OVERREAD slack past the end — guarantees
     * the caller's `symbols` doesn't offer.  So the root's children
     * decode into the thread-local arena (grown on demand, reused
     * across blocks) and only the root's own merge targets `symbols`
     * (via the dst slab when its stores wouldn't be exact). */
    uint8_t *scratch = decode_scratch_ensure(need);
    if (!scratch) return PIVCO_ERR_NULL;

    codec_dec_io_t io = { in_end,
                          scratch + bump_off,
                          scratch + bump_off };
    uint8_t *bump_base = scratch + walk_off;

    if ((pivco_node_type_t)table->node_type[table->tree_root]
        == PIVCO_NODE_LEAF_LEFT) {
        /* One internal child: it decodes at the arena base (its spine
         * may extend DEC_SLOP past its ceilq size) with the region
         * after it as ping-pong partner; the cst_vec merge fills
         * symbols. */
        int K_right = wire_read_kr_header(table, table->tree_root, &ptr);
        uint8_t *slot_end = scratch + dec_ceilq((size_t)K_right) + DEC_SLOP;
        codec_decode_subtree(table, root->right, K_right,
                              scratch, slot_end,
                              slot_end, scratch + walk_off,
                              bump_base, &ptr, &io);

        uint8_t bm_scratch[(size_t)bitmap_bytes(N) + 16];
        const uint8_t *bm = wire_read_bitmap(&ptr, N, bm_scratch, in_end);
        if (root_tail)
            codec_merge_cst_vec_exact(bm, N, K_right,
                           (uint8_t)table->tree[root->left].symbol,
                           scratch, symbols);
        else
            prim_merge_cst_vec(bm, N,
                           (uint8_t)table->tree[root->left].symbol,
                           scratch, symbols);
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* INTERNAL_FULL root — hybrid hole-reuse placement.  Both children
     * decode into the arena's walk region, [larger | smaller], and the
     * final merge writes symbols.  The larger child (first on the
     * wire) uses the smaller sibling's still-empty slot as its
     * ping-pong partner — hole-reuse; the smaller child then decodes
     * into its slot with the fresh region beyond as partner. */
    int K_right = wire_read_kr_header(table, table->tree_root, &ptr);
    int K_left  = N - K_right;
    int K_large = K_right > K_left ? K_right : K_left;
    int K_small = N - K_large;

    uint8_t *slotA     = scratch;
    uint8_t *slotA_end = slotA + dec_ceilq((size_t)K_large) + DEC_SLOP;
    uint8_t *slotB     = slotA_end;
    uint8_t *slotB_end = slotB + dec_ceilq((size_t)K_small) + DEC_SLOP;

    if (K_right > K_left) {                  /* right larger -> first on the wire */
        codec_decode_subtree(table, root->right, K_right,
                              slotA, slotA_end, slotB, slotB_end,
                              bump_base, &ptr, &io);
        codec_decode_subtree(table, root->left,  K_left,
                              slotB, slotB_end, slotB_end, scratch + walk_off,
                              bump_base, &ptr, &io);
    } else {
        codec_decode_subtree(table, root->left,  K_left,
                              slotA, slotA_end, slotB, slotB_end,
                              bump_base, &ptr, &io);
        codec_decode_subtree(table, root->right, K_right,
                              slotB, slotB_end, slotB_end, scratch + walk_off,
                              bump_base, &ptr, &io);
    }
    uint8_t *buf_left  = K_right > K_left ? slotB : slotA;
    uint8_t *buf_right = K_right > K_left ? slotA : slotB;

    uint8_t bm_scratch[(size_t)bitmap_bytes(N) + 16];
    const uint8_t *bm = wire_read_bitmap(&ptr, N, bm_scratch, in_end);
    if (root_tail)
        codec_merge_vec_vec_exact(bm, N, K_right,
                                  buf_left, buf_right, symbols);
    else
        prim_merge_vec_vec(bm, N, buf_left, buf_right, symbols);

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

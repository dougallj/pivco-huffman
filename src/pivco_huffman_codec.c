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
 *      The tree is IMPLICIT (issue #7): the walk streams the decode
 *      table's sched[] (embedded in the full table as table->dec),
 *      the pre-order program rendered from the rank-range form at
 *      build-table time -- see the walk note below.  table->tree[] is
 *      never read here.
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
#include "pivco_huffman_wire.h"
#include "pivco_huffman_primitives.h"
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

/* ---------- Runtime placement flags (issue #10 integration) ----------
 *
 * PIVCO_DEC_CARVE=1    page-hazard-aware scratch carving: child slices
 *                      are nudged off page boundaries (issue #8's
 *                      split-load penalty on a parked merge cursor).
 * PIVCO_DEC_INPLACE=1  in-place merge: an internal node's longer child
 *                      decodes into out_buf's tail instead of scratch,
 *                      halving the merge working set.
 *
 * Both default OFF: placement is then byte-identical to the plain bump
 * allocator.  Read once per process (per backend TU). */
static int g_dec_carve   = -1;
static int g_dec_inplace = -1;
static inline int dec_flag(const char *name, int *cache)
{
    int v = *cache;
    if (v < 0) {
        const char *e = getenv(name);
        v = (e && e[0] == '1');
        *cache = v;
    }
    return v;
}

/* SCRATCH_PAGE is the page-split-load hazard granularity used by
 * scratch_carve below: the hardware page size (16 KB Apple Silicon,
 * 4 KB elsewhere in the fleet; a 64 KB-page Linux arm64 kernel gets
 * needless-but-harmless 4 KB padding). */
#if defined(__APPLE__) && defined(__aarch64__)
#define SCRATCH_PAGE   ((uintptr_t)16384)
#else
#define SCRATCH_PAGE   ((uintptr_t)4096)
#endif
#define MERGE_OVERREAD ((uintptr_t)PIVCO_PRIM_MERGE_OVERREAD)
/* Keep the first ~cache line of every slice straddle-free: a merge
 * cursor lingers NEAR offset 0 (not just at it) while the other side
 * drains. */
#define START_GUARD    ((uintptr_t)64)

/* ALL padding draws from a per-block budget (reset in the decode
 * entry): when it runs out, slices are placed unpadded -- a perf
 * hazard reachable only by adversarial page phases, never a safety
 * issue.  Absolute (not per-page) so the arena bound is identical on
 * every platform. */
#define PIVCO_SCRATCH_PAD_BUDGET ((size_t)16384)
static __thread size_t g_scratch_pad_left;

/* Carve a child slice off the arena.  With PIVCO_DEC_CARVE off this is
 * a plain bump.  With it on, page-hazard-aware placement (issue #8):
 *   - sub-page slice: never crosses a page boundary (its cursor dwells
 *     everywhere inside it, so a mid-slice straddle is the hammer);
 *   - page-crossing slice: interior straddles are unavoidable (and its
 *     cursor moves fast anyway); small nudges keep the START zone and
 *     the exhausted-cursor END position straddle-free (the end lands ON
 *     a boundary, so the end-position load reads the next page without
 *     straddling). */
static inline uint8_t *scratch_carve(uint8_t **top, int size)
{
    uintptr_t p = (uintptr_t)*top;
    if (g_dec_carve) {
        uintptr_t rem = SCRATCH_PAGE - (p & (SCRATCH_PAGE - 1));  /* 1..PAGE */
        uintptr_t pad = 0;
        if ((uintptr_t)size + MERGE_OVERREAD <= SCRATCH_PAGE) {
            if ((uintptr_t)size + MERGE_OVERREAD > rem)
                pad = rem;                              /* sub-page: never cross */
        } else {
            if (rem < START_GUARD) pad = rem;           /* clean start */
            uintptr_t end  = p + pad + (uintptr_t)size;
            uintptr_t erem = SCRATCH_PAGE - (end & (SCRATCH_PAGE - 1));
            if (erem < MERGE_OVERREAD) {
                pad += erem;                            /* end onto boundary */
                uintptr_t srem = SCRATCH_PAGE - ((p + pad) & (SCRATCH_PAGE - 1));
                if (srem < START_GUARD) pad += srem;    /* re-clean start */
            }
        }
        if (pad <= g_scratch_pad_left) g_scratch_pad_left -= pad; else pad = 0;
        p += pad;
    }
    *top = (uint8_t *)(p + (uintptr_t)size);
    return (uint8_t *)p;
}

/* Place an internal node's tail child at out_buf + offset (in-place
 * merge; only called when tail_ok && PIVCO_DEC_INPLACE).  A tail's
 * position is fixed by the merge invariant, so it cannot be nudged;
 * the one dangerous placement is a SUB-PAGE tail that would cross a
 * page boundary (a small slice's cursor is parked everywhere in it) --
 * that one falls back to a no-cross carve, charging the extra scratch
 * content to the same pad budget. */
static inline uint8_t *place_tail(uint8_t *out_buf, int offset, int size,
                                  uint8_t **top)
{
    uint8_t *p = out_buf + offset;
    if (g_dec_carve && (uintptr_t)size + MERGE_OVERREAD <= SCRATCH_PAGE) {
        uintptr_t rem = SCRATCH_PAGE - ((uintptr_t)p & (SCRATCH_PAGE - 1));
        if ((uintptr_t)size + MERGE_OVERREAD > rem &&
            (size_t)size <= g_scratch_pad_left) {
            g_scratch_pad_left -= (size_t)size;   /* charge the content... */
            return scratch_carve(top, size);      /* ...the carve pads itself */
        }
    }
    return p;
}

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
#  define CODEC_ENCODE_ENTRY    pivco_huffman_encode_scalar
#  define CODEC_ENCODE_CT_ENTRY pivco_huffman_encode_scalar_ct
#  define CODEC_DECODE_ENTRY    pivco_huffman_decode_scalar
#  define CODEC_DECODE_DT_ENTRY pivco_huffman_decode_scalar_dt
#elif defined(PIVCO_BACKEND_NEON)
#  define CODEC_ENCODE_ENTRY    pivco_huffman_encode_neon
#  define CODEC_ENCODE_CT_ENTRY pivco_huffman_encode_neon_ct
#  define CODEC_DECODE_ENTRY    pivco_huffman_decode_bu_neon
#  define CODEC_DECODE_DT_ENTRY pivco_huffman_decode_bu_neon_dt
#elif defined(PIVCO_BACKEND_X86)
#  define CODEC_ENCODE_ENTRY    pivco_huffman_encode_x86
#  define CODEC_ENCODE_CT_ENTRY pivco_huffman_encode_x86_ct
#  define CODEC_DECODE_ENTRY    pivco_huffman_decode_bu_x86
#  define CODEC_DECODE_DT_ENTRY pivco_huffman_decode_bu_x86_dt
#elif defined(PIVCO_BACKEND_AVX512)
#  define CODEC_ENCODE_ENTRY    pivco_huffman_encode_avx512
#  define CODEC_ENCODE_CT_ENTRY pivco_huffman_encode_avx512_ct
#  define CODEC_DECODE_ENTRY    pivco_huffman_decode_bu_avx512
#  define CODEC_DECODE_DT_ENTRY pivco_huffman_decode_bu_avx512_dt
#else
#  error "pivco_huffman_codec.c needs PIVCO_BACKEND_{SCALAR,NEON,X86,AVX512}"
#endif

/* ---------- Schedule-driven tree walk (issue #7) ---------- *
 *
 * Both walks (encode + BU decode) traverse table->sched[]: the implicit
 * rank-range tree rendered at build-table time into one 3-byte pre-order
 * record per visible internal node (see pivco_sched_rec_t).  Record
 * indices are passed BY VALUE — the left child is idx + 1, the right
 * child idx + rec->right — mirroring how the retired tree walk passed
 * node ids (a by-reference cursor variant measurably fed x86 register
 * pressure; see results/sweep_2026-07-10_aws_SUMMARY.md).  The wire
 * supplies all child sizes (K_right), leaf symbols are
 * rank_to_sym[rec->param(+1)], and a flat subtree's code_to_sym table
 * is the rank_to_sym slice at rec->param.  Per-node hot metadata is
 * the ~100-byte program plus the 256-byte rank_to_sym array, replacing
 * ~5 KB of tree[] + node_type[] + flat_*[] loads.
 *
 * Empty subtrees: a child that receives 0 elements (K_right == 0 or
 * K_left == 0) emits/reads NOTHING on the wire — the caller simply
 * does not visit it (child indices are static, so nothing to skip). */

/* ---------- Encode tree walk ---------- *
 *
 * DFS in wire order (v0.7 decode order): emit the node's K_right header,
 * recurse into the children larger-K first, then emit the node's
 * marker+bitmap.  At each non-flat internal node, `ranks[0..n)` holds the
 * surviving leaves' in-order ranks; partition routes each by `rank > thr`
 * (thr = the max rank of the left subtree, from the schedule record),
 * leaving the left half in place in `ranks[0..n_left)` and compacting the
 * right half into `tmp[0..n_right)`.  The recursion descends left on
 * `ranks`, right on `tmp`. */

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

/* Callers guarantee n > 0 and do not visit empty (n == 0) child
 * subtrees; record indices ride by value.  Takes the bare decode table —
 * the walk reads only sched[], so the full-table and codec-table encode
 * entries share it. */
static void codec_encode_node(const pivco_huffman_decode_table_t *dt,
                               int idx,
                               uint8_t *ranks, int n,
                               int depth,
                               uint8_t **out_ptr,
                               uint8_t *tmp)
{
    PROF_COUNT_ONLY(PROF_ENC_NODE_VISIT, n);

    const pivco_sched_rec_t *rec = &dt->sched[idx];
    const unsigned kind = rec->kd & 3u;

    /* Flat-subtree fast path: pack n*D bits, no marker, no K_right.
     * D=1 (the former PAIR kind) packs the same bits the old pair
     * bitmap held, minus the marker byte and the FSE option. */
    if (kind == PIVCO_SCHED_FLAT) {
        int D = rec->kd >> 2;
        int total_bytes = (n * D + 7) >> 3;
        PROF_TIC();
        prim_enc_pack_dN(ranks, n, D, rec->param, *out_ptr);
        PROF_TOC(PROF_ENC_FLAT, n);
        *out_ptr += total_bytes;
        return;
    }

    /* Non-flat internal node, decode-order layout (wire v0.7): the
     * K_right header goes at the node's PRE-order position (the decoder
     * sizes both children before their regions arrive), the marker +
     * bitmap at its POST-order position (right where the decoder
     * merges).  The bitmap is built into a stack staging buffer (+64
     * slack absorbs the SIMD partitions' over-wide tail stores) and
     * copied into the stream after the children's regions, whose
     * (FSE-variable) encoded sizes fix its final position. */
    int nbytes = bitmap_bytes(n);
    uint8_t bm_stage[(size_t)nbytes + 64];

    /* Partition against thr (== rec->param for every non-flat kind; for
     * LEAF_LEFT it doubles as rank_begin).  The variant choice
     * mirrors the decode-side dispatch.  The bitmap (and thus the wire
     * bytes) is identical across variants; only the encode-internal
     * scatter work differs — a leaf child never reads its scattered side,
     * so that side's scatter is skipped: leaf-left only stores the right
     * (compacted into tmp), full both.  (The former PAIR/partition_none
     * case is now the flat D=1 path above.) */
    const uint8_t thr = rec->param;
    int n_right;
    PROF_TIC();
    if (kind == PIVCO_SCHED_LEAF_LEFT)
        n_right = prim_enc_partition_right(ranks, n, thr, bm_stage, tmp);
    else
        n_right = prim_enc_partition_full(ranks, n, thr, bm_stage, tmp);
    PROF_TOC(PROF_ENC_NODE_FULL, n);
    int n_left  = n - n_right;

    wire_write_kr(out_ptr, n_right);

    /* Children's regions, larger-K first (strict >, ties left-first —
     * must match the decoder).  The recursion order is free: ranks
     * (left half, in place), tmp (right half) and the shared deeper
     * scratch tmp + n_right are mutually disjoint.  LEAF_LEFT's left
     * child is a leaf (no record, nothing on the wire); an empty child
     * is simply not visited. */
    if (kind == PIVCO_SCHED_FULL && n_right > n_left) {
        codec_encode_node(dt, idx + rec->right, tmp, n_right, depth + 1,
                           out_ptr, tmp + n_right);
        if (n_left > 0)
            codec_encode_node(dt, idx + 1, ranks, n_left, depth + 1,
                               out_ptr, tmp + n_right);
    } else {
        if (kind == PIVCO_SCHED_FULL && n_left > 0)
            codec_encode_node(dt, idx + 1, ranks, n_left, depth + 1,
                               out_ptr, tmp + n_right);
        if (n_right > 0)
            codec_encode_node(dt, idx + rec->right, tmp, n_right, depth + 1,
                               out_ptr, tmp + n_right);
    }

    /* This node's record: marker + staged bitmap.  The FSE attempt may
     * rewrite both in place with [fse_len][payload] and pull *out_ptr
     * back to the payload end.  No-op otherwise. */
    uint8_t *marker_slot = *out_ptr;
    *marker_slot = 0;
    *out_ptr += 1;
    uint8_t *bm = *out_ptr;
    memcpy(bm, bm_stage, (size_t)nbytes);
    *out_ptr += nbytes;
    codec_maybe_fse_attempt(marker_slot, bm, nbytes,
                             n, n_left, n_right, depth, out_ptr);
}

/* Shared encode body; the two public entries below differ only in where
 * the three table pieces live. */
static int codec_encode_core(const uint8_t *symbols, size_t n,
                             const pivco_huffman_decode_table_t *dt,
                             const uint8_t *sym_to_rank,
                             const pivco_huffman_enc_init_aux_t *aux,
                             uint8_t *out, size_t *out_len)
{
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
    prim_enc_init(ranks, N, symbols, sym_to_rank, aux);
    PROF_TOC(PROF_ENC_INIT, N);

    codec_encode_node(dt, 0, ranks, N, 0, &ptr, tmp);

    *out_len = (size_t)(ptr - out);
    return PIVCO_OK;
}

int CODEC_ENCODE_ENTRY(const uint8_t *symbols, size_t n,
                       const pivco_huffman_table_t *table,
                       uint8_t *out, size_t *out_len)
{
    if (!symbols || !table || !out || !out_len) return PIVCO_ERR_NULL;
    return codec_encode_core(symbols, n, &table->dec, table->sym_to_rank,
                             &table->enc_init_aux, out, out_len);
}

int CODEC_ENCODE_CT_ENTRY(const uint8_t *symbols, size_t n,
                          const pivco_huffman_codec_table_t *ct,
                          uint8_t *out, size_t *out_len)
{
    if (!symbols || !ct || !out || !out_len) return PIVCO_ERR_NULL;
    return codec_encode_core(symbols, n, &ct->dec, ct->sym_to_rank,
                             &ct->enc_init_aux, out, out_len);
}

/* ---------- Bottom-up decode tree walk ---------- *
 *
 * Bottom-up: each call decodes a subtree into a contiguous K-byte
 * output buffer.  Internal nodes recurse into their children (which
 * write to scratch arenas), then merge per the bitmap.  The flat-
 * subtree fast path bypasses recursion entirely.
 *
 * Dispatch comes straight off the schedule record (see the walk note
 * above); a leaf child's symbol goes straight into the parent's merge,
 * so leaves have no records and the walk never visits them:
 *
 *   FLAT      — packed-bits flat decode into out_buf (D=1 is the former
 *               PAIR kind; merge_flat routes it to the cst_cst kernel)
 *   LEAF_LEFT — left child lone leaf, recurse right, merge_cst_vec
 *   FULL      — both children internal: recurse both, merge_vec_vec
 *
 * Callers guarantee K > 0; empty (K == 0) child subtrees are simply
 * not visited (child record indices are static).
 *
 * `scratch_top` is the arena pointer for child output buffers; each
 * caller carves its children's slices off it (see scratch_carve) when
 * calling further down.
 *
 * `tail_ok` is nonzero when out_buf is arena-backed (a carve or a
 * nested tail) and so has MERGE_OVERREAD trailing slack: a child may
 * then be placed in its tail (see place_tail) when PIVCO_DEC_INPLACE
 * is on.  The root call passes 0 -- the caller's output buffer has no
 * over-read slack. */

/* Returns 0, or -1 when the (untrusted) stream is truncated or carries
 * an impossible K_right / bad FSE record.  All stream reads are checked
 * against in_end BEFORE dereferencing; K_right is clamped to [0, K] so
 * child sizes, scratch carves and merge extents stay within the bounds
 * the entry sized for N.  (A bitmap whose popcount disagrees with
 * K_right yields garbage output but touches only in-bounds memory.)
 *
 * Dispatch is a SWITCH on the record's 2-bit kind, mirroring the old
 * explicit-tree walk's single indirect jump on the precomputed
 * node_type[] (the compiler emits the same jump-table shape; verified
 * on x86).  NB: measured NEUTRAL vs the earlier if-chain on c8i/gcc-13
 * — dispatch shape is NOT the source of the ~3% x86 decode delta vs
 * the pre-schedule codec (nor is loop alignment; both tested and
 * rejected — see results/sweep_2026-07-10_aws_SUMMARY.md).  Kept as
 * the better idiom for a precomputed kind. */
static int codec_decode_subtree(const pivco_huffman_decode_table_t *dt,
                                   int idx, int K,
                                   uint8_t *out_buf,
                                   const uint8_t **in_ptr,
                                   const uint8_t *in_end,
                                   uint8_t *scratch_top, int tail_ok)
{
    const pivco_sched_rec_t *rec = &dt->sched[idx];

    switch ((pivco_sched_kind_t)(rec->kd & 3u)) {

    case PIVCO_SCHED_FLAT: {
        /* Flat subtree: the 2^D code_to_sym entries are the subtree's own
         * slice of rank_to_sym (ranks are in code order). */
        int D = rec->kd >> 2;
        int total_bytes = (K * D + 7) >> 3;
        if (in_end - *in_ptr < (ptrdiff_t)total_bytes) return -1;
        const uint8_t *bm = *in_ptr;
        *in_ptr += total_bytes;
        prim_merge_flat(out_buf, K, bm, D, &dt->rank_to_sym[rec->param]);
        return 0;
    }

    case PIVCO_SCHED_LEAF_LEFT: {
        /* Left child is a lone leaf (a lone leaf child is always left). */
        int K_right = wire_read_kr_checked(in_ptr, in_end);
        if (K_right < 0 || K_right > K) return -1;
        uint8_t *right_buf = tail_ok
            ? place_tail(out_buf, K - K_right, K_right, &scratch_top)
            : scratch_carve(&scratch_top, K_right);
        /* right child record is idx + 1 (the bare left leaf has none) */
        if (K_right > 0) {
            if (codec_decode_subtree(dt, idx + 1, K_right,
                                     right_buf, in_ptr, in_end, scratch_top,
                                     g_dec_inplace) != 0)
                return -1;
        }
        /* This node's marker + bitmap sit at its post-order position —
         * read right at merge time (the VLA no longer lives across the
         * child recursion). */
        uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
        const uint8_t *bm = wire_read_bitmap_checked(in_ptr, in_end, K,
                                                     bm_scratch);
        if (!bm) return -1;
        prim_merge_cst_vec(bm, K,
                           dt->rank_to_sym[rec->param],
                           right_buf, out_buf);
        return 0;
    }

    case PIVCO_SCHED_FULL:
    default:
        break;      /* falls through to the FULL body below */
    }

    /* FULL: both children internal.  Recurse into both with disjoint
     * scratch slices — in stream order, larger-K child first (strict >,
     * ties left-first; must match the encoder) — then merge. */
    int K_right = wire_read_kr_checked(in_ptr, in_end);
    if (K_right < 0 || K_right > K) return -1;
    int K_left = K - K_right;
    uint8_t *left_buf, *right_buf;
    if (tail_ok && K_left >= K_right) {
        /* Longer child decodes into out_buf's tail (free -- see
         * place_tail), the shorter into a scratch carve.  Placement is
         * position-based, so it is independent of the decode order. */
        left_buf  = place_tail(out_buf, K_right, K_left, &scratch_top);
        right_buf = scratch_carve(&scratch_top, K_right);
    } else if (tail_ok) {
        right_buf = place_tail(out_buf, K_left, K_right, &scratch_top);
        left_buf  = scratch_carve(&scratch_top, K_left);
    } else {
        left_buf  = scratch_carve(&scratch_top, K_left);
        right_buf = scratch_carve(&scratch_top, K_right);
    }

    if (K_right > K_left) {
        if (codec_decode_subtree(dt, idx + rec->right, K_right,
                                 right_buf, in_ptr, in_end, scratch_top,
                                 g_dec_inplace) != 0)
            return -1;
        if (K_left > 0 &&
            codec_decode_subtree(dt, idx + 1, K_left,
                                 left_buf,  in_ptr, in_end, scratch_top,
                                 g_dec_inplace) != 0)
            return -1;
    } else {
        if (K_left > 0 &&
            codec_decode_subtree(dt, idx + 1, K_left,
                                 left_buf,  in_ptr, in_end, scratch_top,
                                 g_dec_inplace) != 0)
            return -1;
        if (K_right > 0 &&
            codec_decode_subtree(dt, idx + rec->right, K_right,
                                 right_buf, in_ptr, in_end, scratch_top,
                                 g_dec_inplace) != 0)
            return -1;
    }
    /* Post-order record: marker + bitmap, read right at merge time. */
    uint8_t bm_scratch[(size_t)bitmap_bytes(K) + 16];
    const uint8_t *bm = wire_read_bitmap_checked(in_ptr, in_end, K,
                                                 bm_scratch);
    if (!bm) return -1;
    prim_merge_vec_vec(bm, K, left_buf, right_buf, out_buf);
    return 0;
}

int CODEC_DECODE_DT_ENTRY(const uint8_t *in, size_t in_len,
                          const pivco_huffman_decode_table_t *dt,
                          uint8_t *symbols, size_t *consumed)
{
    if (!in || !dt || !symbols || !consumed) return PIVCO_ERR_NULL;
    prim_codec_init();

    /* Block header: first 2 bytes are N (symbol count for this block).
     * The stream is untrusted: every read below is bounds-checked
     * against in_end. */
    if (in_len < PIVCO_BLOCK_N_BYTES) return PIVCO_ERR_CORRUPT;
    const uint8_t *ptr = in;
    const uint8_t *in_end = in + in_len;
    const int N = wire_read_block_n(&ptr);
    if (N <= 0 || N > PIVCO_WIRE_MAX_N) return PIVCO_ERR_CORRUPT;

    /* Fast path: flat root — the whole block is one packed N·D-bit
     * region decoded straight into the output buffer; no recursion, no
     * scratch.  Skips the codec_decode_subtree machinery (dispatch +
     * bm_scratch stack frame + scratch TLS reference / arena ensure).
     * D=1 is the former 2-rank/PAIR fast path (worth −26% on two_sym
     * decode on older narrow x86); D>=2 covers e.g. the uniform-dist
     * full-alphabet flat tree, which never needed the arena either. */
    if ((pivco_sched_kind_t)(dt->sched[0].kd & 3u) == PIVCO_SCHED_FLAT) {
        int D = dt->sched[0].kd >> 2;
        int total_bytes = (N * D + 7) >> 3;
        if (in_end - ptr < (ptrdiff_t)total_bytes) return PIVCO_ERR_CORRUPT;
        prim_merge_flat(symbols, N, ptr, D, &dt->rank_to_sym[dt->sched[0].param]);
        ptr += total_bytes;
        *consumed = (size_t)(ptr - in);
        return PIVCO_OK;
    }

    /* Scratch arena.  Worst case at a heavily-skewed node, the
     * partition is one-sided so a single recursion can consume up to
     * N bytes.  Bounded by (MAX_CODE_LEN+2) * N.  Grown on demand from a
     * thread-local heap buffer so block size is a runtime parameter.
     * With carve or in-place placement on, double it for carve-bump
     * waste and add page + over-read slack (the exact 2N + budget
     * bound comes with the arena-shrink step). */
    dec_flag("PIVCO_DEC_CARVE",   &g_dec_carve);
    dec_flag("PIVCO_DEC_INPLACE", &g_dec_inplace);
    size_t need = (size_t)N * (PIVCO_MAX_CODE_LEN + 2);
    if (g_dec_carve || g_dec_inplace)
        need = 2 * need + SCRATCH_PAGE + MERGE_OVERREAD;
    uint8_t *scratch = decode_scratch_ensure(need);
    if (!scratch) return PIVCO_ERR_NULL;
    g_scratch_pad_left = PIVCO_SCRATCH_PAD_BUDGET;

    if (codec_decode_subtree(dt, 0, N,
                             symbols, &ptr, in_end, scratch,
                             /*tail_ok=*/0) != 0)
        return PIVCO_ERR_CORRUPT;

    *consumed = (size_t)(ptr - in);
    return PIVCO_OK;
}

/* Full-table shim: the decode core is embedded as table->dec. */
int CODEC_DECODE_ENTRY(const uint8_t *in, size_t in_len,
                       const pivco_huffman_table_t *table,
                       uint8_t *symbols, size_t *consumed)
{
    if (!table) return PIVCO_ERR_NULL;
    return CODEC_DECODE_DT_ENTRY(in, in_len, &table->dec, symbols, consumed);
}

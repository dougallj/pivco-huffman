/* pivco_decode_table_ref.c — standalone reference: code lengths -> the
 * ~1 KB pivco decode table (issue #7 construction).
 *
 * The production build (src/huffman_table.c) shares one build_core
 * between the full encode-side table and the decode table, and supports
 * the research tree modes.  This file is the same algorithm SPECIALIZED
 * to the production case — OPTIMIZED tree mode, decode table only — as
 * a compact illustration of the whole construction (written for easy
 * lifting into other codebases, e.g. the OpenZL port).  It is
 * cross-checked against the production build on every distribution by
 * extras/bench/bench_rank_range_table.c, so it cannot silently rot.
 *
 * Pipeline (input: 256 code lengths, output: rank_to_sym + sched):
 *
 *   1. Histogram the lengths and counting-sort the symbols by length
 *      (fused into one NEON sweep, or two scalar passes portably).
 *      Validation is by bin accounting: any byte outside 0..MAX makes
 *      the bins + zeros sum fall short of 256.
 *   2. Decompose each length's count c_L by its binary representation
 *      into "chunks": every set bit b gives 2^b same-length leaves
 *      grouped under one ancestor at depth L-b (a flat subtree for
 *      b >= 2, a sibling pair for b == 1, a bare leaf for b == 0).
 *      Kraft: the chunk depths tile the code space exactly.
 *   3. Stable-sort chunks by root depth (insertion sort on purpose:
 *      the generation order is already nearly sorted).
 *   4. The depth-sorted chunk list is the tree's left-to-right
 *      leaf-depth sequence, which uniquely determines the tree; a
 *      greedy pre-order reconstruction (next chunk at my depth = that
 *      leaf, else recurse twice one level down) emits the walk
 *      schedule, copies each chunk's symbols into rank order, and
 *      reads each internal node's partition threshold off the running
 *      rank cursor.  Incomplete or over-full codes fail here.
 *
 * Build knobs (mirroring src/huffman_table.c):
 *   PIVCO_HISTO_PORTABLE   force the portable histogram + counting sort
 *   PIVCO_REF_SCHED_ITER   explicit-stack schedule generation instead of
 *                          the recursive form (measured equivalent; the
 *                          recursion is the readable default)
 */
#include "pivco_huffman.h"
#include <string.h>

#ifndef PIVCO_HISTO_PORTABLE
#  if defined(__aarch64__) || defined(__ARM_NEON)
#    define REF_HISTO_SIMD 1
#    define REF_HISTO_SIMD_NEON 1
#    include <arm_neon.h>
#  elif defined(__x86_64__) || defined(__i386__)
#    define REF_HISTO_SIMD 1
#    define REF_HISTO_SIMD_SSE 1
#    include <emmintrin.h>
#  endif
#endif

/* A chunk: 2^bit same-length leaves under one root at `depth`.
 * sym_idx points into the length-sorted symbol array. */
typedef struct {
    uint8_t depth;
    uint8_t bit;
    uint8_t sym_idx;
} ref_chunk_t;

/* ---- 1. histogram + counting sort ---- */

#ifdef REF_HISTO_SIMD
/* One cmeq sweep per bin counts it AND extracts its symbols (movemask +
 * bit loop, register output cursor) — no scatter, no store-forwarding
 * chains.  Returns n_used or -1 on invalid lengths. */
static int ref_histo_sort(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                          uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1],
                          uint8_t items[PIVCO_MAX_SYMBOLS],
                          int per_len_start[PIVCO_MAX_CODE_LEN + 2])
{
    int seen;
#ifdef REF_HISTO_SIMD_NEON
    {
        const uint8x16_t vzero = vdupq_n_u8(0);
        uint8x16_t zacc = vzero;
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 16)
            zacc = vsubq_u8(zacc, vceqq_u8(vld1q_u8(lengths + i), vzero));
        seen = (int)vaddlvq_u8(zacc);
    }
#else
    {
        const __m128i vzero = _mm_setzero_si128();
        __m128i zacc = vzero;
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 16)
            zacc = _mm_sub_epi8(zacc, _mm_cmpeq_epi8(
                       _mm_loadu_si128((const __m128i *)(lengths + i)), vzero));
        __m128i s = _mm_sad_epu8(zacc, vzero);
        seen = _mm_cvtsi128_si32(s)
             + _mm_cvtsi128_si32(_mm_srli_si128(s, 8));
    }
#endif

    int out = 0;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        per_len_start[L] = out;
#ifdef REF_HISTO_SIMD_NEON
        const uint8x16_t target = vdupq_n_u8((uint8_t)L);
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 16) {
            uint8x16_t eq = vceqq_u8(vld1q_u8(lengths + i), target);
            uint64_t m = vget_lane_u64(vreinterpret_u64_u8(
                             vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
            m &= 0x1111111111111111ull;
            while (m) {
                items[out++] = (uint8_t)(i + (__builtin_ctzll(m) >> 2));
                m &= m - 1;
            }
        }
#else
        const __m128i target = _mm_set1_epi8((char)L);
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 16) {
            unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_loadu_si128((const __m128i *)(lengths + i)), target));
            while (m) {
                items[out++] = (uint8_t)(i + (int)__builtin_ctz(m));
                m &= m - 1;
            }
        }
#endif
        sym_count[L] = (uint16_t)(out - per_len_start[L]);
    }
    per_len_start[PIVCO_MAX_CODE_LEN + 1] = out;
    return (seen + out == PIVCO_MAX_SYMBOLS) ? out : -1;
}
#else
/* Portable: four interleaved sub-histograms (splits the same-bin
 * store-forwarding chains four ways), then a cursor placement pass. */
static int ref_histo_sort(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                          uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1],
                          uint8_t items[PIVCO_MAX_SYMBOLS],
                          int per_len_start[PIVCO_MAX_CODE_LEN + 2])
{
    uint16_t h0[16] = {0}, h1[16] = {0}, h2[16] = {0}, h3[16] = {0};
    int n_zero = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 4) {
        uint8_t a = lengths[i],     b = lengths[i + 1];
        uint8_t c = lengths[i + 2], d = lengths[i + 3];
        if ((a | b | c | d) > 15) return -1;
        n_zero += (a == 0) + (b == 0) + (c == 0) + (d == 0);
        if (a) h0[a]++;
        if (b) h1[b]++;
        if (c) h2[c]++;
        if (d) h3[d]++;
    }
    int acc = 0;
    int cursor[PIVCO_MAX_CODE_LEN + 2];
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        sym_count[L] = (uint16_t)(h0[L] + h1[L] + h2[L] + h3[L]);
        per_len_start[L] = acc;
        cursor[L] = acc;
        acc += sym_count[L];
    }
    per_len_start[PIVCO_MAX_CODE_LEN + 1] = acc;
    if (acc + n_zero != PIVCO_MAX_SYMBOLS) return -1;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        uint8_t L = lengths[s];
        if (L) items[cursor[L]++] = (uint8_t)s;
    }
    return acc;
}
#endif

/* ---- 4. schedule generation ---- */

#ifndef PIVCO_REF_SCHED_ITER
typedef struct {
    pivco_huffman_decode_table_t *dt;
    const ref_chunk_t *chunks;
    const uint8_t *items;
    int      n_chunks;
    int      ci;
    unsigned rank;
    int      err;
} ref_gen_t;

/* Recursive form: returns the record count of the subtree at `depth`. */
static uint8_t ref_sched_gen(ref_gen_t *g, int depth)
{
    if (g->err) return 0;
    if (g->ci >= g->n_chunks || depth > PIVCO_MAX_CODE_LEN) {
        g->err = 1;
        return 0;
    }

    const ref_chunk_t *c = &g->chunks[g->ci];
    if ((int)c->depth == depth) {
        g->ci++;
        unsigned b = c->bit;
        unsigned rank0 = g->rank;
        uint8_t *dst = &g->dt->rank_to_sym[rank0];
        const uint8_t *s = &g->items[c->sym_idx];
        if (b == 0)      dst[0] = s[0];
        else if (b == 1) { dst[0] = s[0]; dst[1] = s[1]; }
        else             memcpy(dst, s, (size_t)1 << b);
        g->rank += 1u << b;
        if (b == 0) return 0;
        pivco_sched_rec_t *rec = &g->dt->sched[g->dt->sched_len++];
        rec->kd    = (b == 1) ? (uint8_t)PIVCO_SCHED_PAIR
                              : (uint8_t)(PIVCO_SCHED_FLAT | (b << 2));
        rec->param = (uint8_t)rank0;
        rec->skip  = 1;
        return 1;
    }

    uint16_t my = g->dt->sched_len++;
    unsigned rank0 = g->rank;
    uint8_t nl = ref_sched_gen(g, depth + 1);
    unsigned thr = g->rank - 1;
    int left_lone = (nl == 0 && g->rank == rank0 + 1);
    uint8_t nr = ref_sched_gen(g, depth + 1);
    if (g->err) return 0;
    int right_lone = (nr == 0 && g->rank == thr + 2);

    pivco_sched_rec_t *rec = &g->dt->sched[my];
    if (left_lone && right_lone)  rec->kd = (uint8_t)PIVCO_SCHED_PAIR;
    else if (left_lone)           rec->kd = (uint8_t)PIVCO_SCHED_LEAF_LEFT;
    else if (right_lone)          { g->err = 1; return 0; }
    else                          rec->kd = (uint8_t)PIVCO_SCHED_FULL;
    rec->param = (uint8_t)thr;
    rec->skip  = (uint8_t)(1 + nl + nr);
    return rec->skip;
}

static int ref_run_sched(pivco_huffman_decode_table_t *dt,
                         const ref_chunk_t *chunks, int n_chunks,
                         const uint8_t *items)
{
    ref_gen_t g = { dt, chunks, items, n_chunks, 0, 0, 0 };
    ref_sched_gen(&g, 0);
    if (g.err || g.ci != n_chunks) return -1;
    dt->num_ranks = (uint16_t)g.rank;
    return 0;
}
#else
/* Explicit-stack form.  The stack depth IS the node depth, and a
 * subtree's records are contiguous (pre-order), so a node's skip is
 * just sched_len - my at completion. */
typedef struct {
    uint16_t my;         /* this node's record index */
    uint16_t mid_sched;  /* sched_len when the right child started */
    uint8_t  rank0;      /* rank on entry */
    uint8_t  mid_rank;   /* rank when the right child started */
    uint8_t  state;      /* 0 = doing left child, 1 = doing right */
} ref_frame_t;

static int ref_run_sched(pivco_huffman_decode_table_t *dt,
                         const ref_chunk_t *chunks, int n_chunks,
                         const uint8_t *items)
{
    ref_frame_t stk[PIVCO_MAX_CODE_LEN + 1];
    int sp = 0;
    int ci = 0;
    unsigned rank = 0;

    for (;;) {
        if (ci >= n_chunks) return -1;      /* under-subscribed lengths */

        /* Descend the left spine until a chunk sits at this depth. */
        while ((int)chunks[ci].depth != sp) {
            if (sp > PIVCO_MAX_CODE_LEN) return -1;
            ref_frame_t *f = &stk[sp++];
            f->my    = dt->sched_len++;
            f->rank0 = (uint8_t)rank;
            f->state = 0;
        }

        /* Consume the chunk-leaf. */
        {
            const ref_chunk_t *c = &chunks[ci++];
            unsigned b = c->bit;
            unsigned rank0 = rank;
            uint8_t *dst = &dt->rank_to_sym[rank0];
            const uint8_t *s = &items[c->sym_idx];
            if (b == 0)      dst[0] = s[0];
            else if (b == 1) { dst[0] = s[0]; dst[1] = s[1]; }
            else             memcpy(dst, s, (size_t)1 << b);
            rank += 1u << b;
            if (b != 0) {
                pivco_sched_rec_t *rec = &dt->sched[dt->sched_len++];
                rec->kd    = (b == 1) ? (uint8_t)PIVCO_SCHED_PAIR
                                      : (uint8_t)(PIVCO_SCHED_FLAT | (b << 2));
                rec->param = (uint8_t)rank0;
                rec->skip  = 1;
            }
        }

        /* Ascend, completing parents whose right child just finished. */
        for (;;) {
            if (sp == 0) {                  /* root subtree complete */
                if (ci != n_chunks) return -1;
                dt->num_ranks = (uint16_t)rank;
                return 0;
            }
            ref_frame_t *f = &stk[sp - 1];
            if (f->state == 0) {            /* left done; do the right child */
                f->state     = 1;
                f->mid_sched = dt->sched_len;
                f->mid_rank  = (uint8_t)rank;
                break;
            }
            /* Right done: finalize this internal node's record. */
            int left_lone  = (f->mid_sched == f->my + 1) &&
                             (f->mid_rank == (uint8_t)(f->rank0 + 1));
            int right_lone = (dt->sched_len == f->mid_sched) &&
                             (rank == (unsigned)f->mid_rank + 1);
            pivco_sched_rec_t *rec = &dt->sched[f->my];
            if (left_lone && right_lone)  rec->kd = (uint8_t)PIVCO_SCHED_PAIR;
            else if (left_lone)           rec->kd = (uint8_t)PIVCO_SCHED_LEAF_LEFT;
            else if (right_lone)          return -1;
            else                          rec->kd = (uint8_t)PIVCO_SCHED_FULL;
            rec->param = (uint8_t)(f->mid_rank - 1);
            rec->skip  = (uint8_t)(dt->sched_len - f->my);
            sp--;
        }
    }
}
#endif

/* ---- entry point ---- */

int pivco_huffman_build_decode_table_ref(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_huffman_decode_table_t *dt)
{
    if (!code_lens || !dt) return PIVCO_ERR_NULL;

    uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1] = {0};
    uint8_t items[PIVCO_MAX_SYMBOLS];
    int per_len_start[PIVCO_MAX_CODE_LEN + 2];
    int n_used = ref_histo_sort(code_lens, sym_count, items, per_len_start);
    if (n_used < 0) return PIVCO_ERR_CORRUPT;
    if (n_used == 0) return PIVCO_ERR_EMPTY;

    if (n_used == 1) {
        /* Degenerate table: two ranks of the same symbol, one PAIR record,
         * so the walk needs no special case. */
        uint8_t sym = items[0];
        dt->rank_to_sym[0] = sym;
        dt->rank_to_sym[1] = sym;
        dt->num_ranks = 2;
        dt->sched[0].kd    = (uint8_t)PIVCO_SCHED_PAIR;
        dt->sched[0].param = 0;
        dt->sched[0].skip  = 1;
        dt->sched_len = 1;
        return PIVCO_OK;
    }

    uint8_t max_len = 0;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++)
        if (sym_count[L]) max_len = (uint8_t)L;

    /* 2. binary chunk decomposition, larger chunks first within a length */
    ref_chunk_t chunks[PIVCO_MAX_SYMBOLS];
    int n_chunks = 0;
    for (int L = 1; L <= max_len; L++) {
        int c = sym_count[L];
        int cur = per_len_start[L];
        for (int bit = PIVCO_MAX_CODE_LEN; bit >= 0; bit--) {
            if (c & (1 << bit)) {
                int depth;
                if      (bit >= 2) depth = L - bit;
                else if (bit == 1) depth = L - 1;
                else               depth = L;
                chunks[n_chunks].bit     = (uint8_t)bit;
                chunks[n_chunks].depth   = (uint8_t)depth;
                chunks[n_chunks].sym_idx = (uint8_t)cur;
                cur += 1 << bit;
                n_chunks++;
            }
        }
    }

    /* 3. stable depth sort (insertion: input is already nearly sorted) */
    for (int i = 1; i < n_chunks; i++) {
        ref_chunk_t cur = chunks[i];
        int j = i - 1;
        while (j >= 0 && chunks[j].depth > cur.depth) {
            chunks[j + 1] = chunks[j];
            j--;
        }
        chunks[j + 1] = cur;
    }

    /* 4. schedule + ranks from the chunk list */
    dt->sched_len = 0;
    if (ref_run_sched(dt, chunks, n_chunks, items) != 0)
        return PIVCO_ERR_CORRUPT;
    return PIVCO_OK;
}

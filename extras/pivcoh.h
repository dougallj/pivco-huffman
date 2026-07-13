/* pivcoh.h - v1.5 - minimal single-file PIVCO-Huffman block codec
 *
 * A tiny, allocation-free-capable implementation of the
 * PIVCO-Huffman wire format (https://github.com/MarcinZukowski/pivco-huffman).
 * Scalar throughout, except that on AArch64 the hot decode merge uses a
 * NEON kernel (define PIVCOH_NO_NEON for pure scalar); the first decode
 * then lazily builds an 8 KiB static shuffle table — the writes are
 * idempotent, so concurrent first decodes are benign.
 * Streams are byte-identical to the full library's PH-only mode (raw
 * bitmaps — pivco_huffman_set_fse_enabled(0), which is also what the
 * pivcohuf tool's default non-ANS format uses; "optimized" tree shaping,
 * max code length 11).  FSE/ANS-coded blocks are not supported and are
 * rejected on decode.
 *
 * Do this in ONE C file to create the implementation:
 *     #define PIVCOH_IMPLEMENTATION
 *     #include "pivcoh.h"
 *
 * Usage — encoder side:
 *     pivcoh_table t;
 *     if (!pivcoh_table_from_freqs(&t, freq)) ...;   // freq: uint64_t[256]
 *     // transmit t.code_len (256 values, all <= 11: nibble-packable)
 *     unsigned char out[PIVCOH_ENCODE_BOUND(4096)];
 *     ptrdiff_t len = pivcoh_encode(&t, in, n, out, sizeof out, NULL);
 *
 * Usage — decoder side:
 *     pivcoh_table t;
 *     if (!pivcoh_table_from_lens(&t, code_len)) ...;  // validates lengths
 *     ptrdiff_t n = pivcoh_decode(&t, buf, buf_len, sym, sym_cap, NULL, NULL);
 *
 * One block covers 1..65535 symbols (bytes).  Blocks are self-delimiting
 * (a 2-byte symbol-count header leads the stream), so blocks can be
 * concatenated; pivcoh_decode reports the consumed byte count.
 *
 * The trailing `scratch` parameter of encode/decode may be NULL (malloc
 * is used internally) or a caller buffer of PIVCOH_SCRATCH_SIZE(max_n)
 * bytes for allocation-free operation.  Tables and scratch are plain
 * memory: no cleanup calls, safe to copy, const tables are shareable
 * across threads (encode/decode themselves touch only their arguments).
 *
 * Encoding a symbol whose frequency/length was zero produces a valid but
 * meaningless stream (never memory-unsafe).  Decode is safe on hostile
 * input: it never reads past `in + in_len`, never writes past N symbols,
 * and returns -1 on any malformed stream.
 *
 * License: Apache-2.0, same as the pivco-huffman repository.
 */
#ifndef PIVCOH_H
#define PIVCOH_H

#include <stddef.h>
#include <stdint.h>

#ifdef PIVCOH_STATIC
#define PIVCOHDEF static
#else
#define PIVCOHDEF extern
#endif

/* Worst-case encoded size of one n-symbol block (payload is at most 11
 * bits/symbol plus per-node headers and byte rounding). */
#define PIVCOH_ENCODE_BOUND(n)  ((11 * (size_t)(n) + 7) / 8 + 1024)

/* Scratch bytes for encode or decode of blocks up to n symbols (2n: the
 * walk places each node's larger child in place, so a K-element subtree
 * touches at most K scratch bytes — n for the decoder's partner buffer
 * or the encoder's partition pool, plus n for the encoder's rank
 * buffer; decode uses only the first n). */
#define PIVCOH_SCRATCH_SIZE(n)  (2 * (size_t)(n))

typedef struct { uint8_t kd, param, right; } pivcoh__rec;

typedef struct {
    uint8_t code_len[256];   /* per-symbol code length, 0 = absent, max 11.
                                Filled by pivcoh_table_from_freqs; this is
                                what the encoder transmits to the decoder. */
    /* internals */
    uint16_t num_ranks, sched_len;
    uint8_t rank_to_sym[256], sym_to_rank[256];
    pivcoh__rec sched[60];   /* Kraft-complete max is 59 records (33 chunks,
                                27 with bit >= 1); +1 so gen's overflow guard
                                can't fire mid-walk on a maximal table */
} pivcoh_table;

/* Build a table from symbol frequencies (encoder side).  Derives optimal
 * length-limited code lengths into t->code_len.  Returns 1, or 0 if every
 * frequency is zero. */
PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256]);

/* Build a table from received code lengths (decoder side).  Returns 1, or
 * 0 on invalid lengths (any > 11, all zero, or not Kraft-complete).  Both
 * sides build identical tables from identical lengths. */
PIVCOHDEF int pivcoh_table_from_lens(pivcoh_table *t, const uint8_t code_len[256]);

/* Encode n symbols (1..65535) into out.  Requires out_cap >=
 * PIVCOH_ENCODE_BOUND(n).  Returns the encoded byte count, or -1 on bad
 * arguments / malloc failure. */
PIVCOHDEF ptrdiff_t pivcoh_encode(const pivcoh_table *t,
                                  const uint8_t *in, size_t n,
                                  uint8_t *out, size_t out_cap, void *scratch);

/* Decode one block from in[0..in_len).  Writes the block's N symbols to
 * out (fails if N > out_cap).  Returns N, or -1 on malformed/truncated
 * input or malloc failure.  If consumed is non-NULL it receives the
 * block's byte length (for concatenated blocks). */
PIVCOHDEF ptrdiff_t pivcoh_decode(const pivcoh_table *t,
                                  const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap,
                                  size_t *consumed, void *scratch);

#endif /* PIVCOH_H */

#ifdef PIVCOH_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && !defined(PIVCOH_NO_NEON)
#define PIVCOH__NEON 1
#include <arm_neon.h>
#endif

#define PIVCOH__MAXLEN 11
enum { PIVCOH__FULL = 0, PIVCOH__FLAT = 1, PIVCOH__LEAFL = 3 };

/* ---- table build ---- */

typedef struct { uint8_t depth, bit, sym_idx; } pivcoh__chunk;

/* Pre-order schedule from the depth-sorted chunk list: chunks are the
 * tree's left-to-right leaves, and a leaf-depth sequence determines the
 * tree.  Returns 0 on non-Kraft-complete lengths. */
static int pivcoh__gen(pivcoh_table *t, const pivcoh__chunk *ch, int nch,
                       int *ci, int *rank, int depth, const uint8_t *items)
{
    if (*ci >= nch || depth > PIVCOH__MAXLEN ||
        t->sched_len >= (int)(sizeof t->sched / sizeof *t->sched)) return 0;
    if (ch[*ci].depth == depth) {              /* chunk leaf */
        const pivcoh__chunk *c = &ch[(*ci)++];
        int n = 1 << c->bit, r0 = *rank;
        memcpy(t->rank_to_sym + r0, items + c->sym_idx, (size_t)n);
        *rank += n;
        if (c->bit) {
            pivcoh__rec *r = &t->sched[t->sched_len++];
            r->kd = (uint8_t)(PIVCOH__FLAT | c->bit << 2);
            r->param = (uint8_t)r0;
            r->right = 0;
        }
        return 1;
    }
    int my = t->sched_len++, r0 = *rank;
    if (!pivcoh__gen(t, ch, nch, ci, rank, depth + 1, items)) return 0;
    int mid_s = t->sched_len, mid_r = *rank;
    if (!pivcoh__gen(t, ch, nch, ci, rank, depth + 1, items)) return 0;
    int ll = (mid_s == my + 1 && mid_r == r0 + 1);
    pivcoh__rec *r = &t->sched[my];
    r->kd    = (uint8_t)(ll ? PIVCOH__LEAFL : PIVCOH__FULL);
    r->param = (uint8_t)(mid_r - 1);           /* thr / rank_begin */
    r->right = (uint8_t)(mid_s - my);
    return 1;
}

PIVCOHDEF int pivcoh_table_from_lens(pivcoh_table *t, const uint8_t code_len[256])
{
    int cnt[PIVCOH__MAXLEN + 1] = {0}, n_used = 0, s, L;
    for (s = 0; s < 256; s++) {
        if (code_len[s] > PIVCOH__MAXLEN) return 0;
        if (code_len[s]) { cnt[code_len[s]]++; n_used++; }
    }
    if (n_used == 0) return 0;
    memcpy(t->code_len, code_len, 256);
    t->sched_len = 0;

    if (n_used == 1) {                         /* degenerate: 1-bit code */
        for (s = 0; !code_len[s]; s++) {}
        memset(t->code_len, 0, 256);
        t->code_len[s] = 1;
        t->rank_to_sym[0] = t->rank_to_sym[1] = (uint8_t)s;
        t->num_ranks = 2;
        t->sched[0].kd = PIVCOH__FLAT | 1 << 2;
        t->sched[0].param = t->sched[0].right = 0;
        t->sched_len = 1;
    } else {
        /* symbols counting-sorted by length (symbol order within a length) */
        uint8_t items[256];
        int cur[PIVCOH__MAXLEN + 2], acc = 0;
        for (L = 1; L <= PIVCOH__MAXLEN; L++) { cur[L] = acc; acc += cnt[L]; }
        for (s = 0; s < 256; s++)
            if (code_len[s]) items[cur[code_len[s]]++] = (uint8_t)s;

        /* "optimized" chunking: split each length's count by its set bits
         * (largest first), a 2^b chunk rooted at depth L-b; then stable
         * depth-sort so canonical assignment fills the tree left-to-right */
        pivcoh__chunk ch[49];   /* max sum popcount(cnt[L]): 11 classes, sum <= 256 */
        int nch = 0, i, j;
        for (L = 1, acc = 0; L <= PIVCOH__MAXLEN; acc += cnt[L], L++)
            for (i = 8, j = acc; i >= 0; i--)
                if (cnt[L] & (1 << i)) {
                    ch[nch].bit = (uint8_t)i;
                    ch[nch].depth = (uint8_t)(i ? L - i : L);
                    ch[nch].sym_idx = (uint8_t)j;
                    j += 1 << i;
                    nch++;
                }
        for (i = 1; i < nch; i++) {
            pivcoh__chunk c = ch[i];
            for (j = i - 1; j >= 0 && ch[j].depth > c.depth; j--) ch[j + 1] = ch[j];
            ch[j + 1] = c;
        }

        int ci = 0, rank = 0;
        if (!pivcoh__gen(t, ch, nch, &ci, &rank, 0, items) || ci != nch) return 0;
        t->num_ranks = (uint16_t)rank;
    }

    memset(t->sym_to_rank, 0, 256);
    for (s = t->num_ranks - 1; s >= 0; s--)
        t->sym_to_rank[t->rank_to_sym[s]] = (uint8_t)s;
    return 1;
}

PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256])
{
    /* leaves sorted ascending by (freq, symbol) */
    uint64_t f[256];
    uint16_t sym[256];
    int n = 0, i, j;
    for (i = 0; i < 256; i++)
        if (freq[i]) {
            uint64_t v = freq[i];
            for (j = n - 1; j >= 0 && f[j] > v; j--) { f[j + 1] = f[j]; sym[j + 1] = sym[j]; }
            f[j + 1] = v;
            sym[j + 1] = (uint16_t)i;
            n++;
        }
    if (n == 0) return 0;

    uint8_t lens[256] = {0};
    if (n == 1) {
        lens[sym[0]] = 1;
    } else {
        /* van Leeuwen two-queue: sorted leaves + FIFO of made internals */
        uint64_t nf[512];
        int parent[512], li = 0, ih = n, ni = n, rem;
        memcpy(nf, f, (size_t)n * sizeof(uint64_t));
        for (rem = n; rem > 1; rem--) {
            int a, b;
            if (li < n && (ih == ni || nf[li] <= nf[ih])) a = li++; else a = ih++;
            if (li < n && (ih == ni || nf[li] <= nf[ih])) b = li++; else b = ih++;
            nf[ni] = nf[a] + nf[b];
            parent[a] = parent[b] = ni++;
        }
        uint8_t depth[512];
        int maxd = 1;
        depth[ni - 1] = 0;
        for (i = ni - 2; i >= 0; i--) depth[i] = (uint8_t)(depth[parent[i]] + 1);
        for (i = 0; i < n; i++) {
            lens[sym[i]] = depth[i] ? depth[i] : 1;
            if (lens[sym[i]] > maxd) maxd = lens[sym[i]];
        }

        if (maxd > PIVCOH__MAXLEN) {           /* DEFLATE-style length limiting */
            /* Clamp to MAXLEN, then repair the Kraft sum: in units of
             * 2^-MAXLEN a clamped symbol weighs 1 instead of < 1, so
             * 0 < debt < cnt[MAXLEN].  Each step re-homes one MAXLEN
             * symbol as the sibling of a symbol demoted from the deepest
             * shorter level: net -1 unit, so the loop lands on Kraft == 1
             * exactly.  A non-empty b always exists: 256 symbols all at
             * MAXLEN would be under-subscribed. */
            int cnt[PIVCOH__MAXLEN + 2] = {0}, b;
            for (i = 0; i < 256; i++)
                if (lens[i]) cnt[lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN]++;
            long kraft = 0;
            for (i = 1; i <= PIVCOH__MAXLEN; i++) kraft += (long)cnt[i] << (PIVCOH__MAXLEN - i);
            for (; kraft > 1L << PIVCOH__MAXLEN; kraft--) {
                for (b = PIVCOH__MAXLEN - 1; !cnt[b]; b--) {}
                cnt[b]--;
                cnt[b + 1] += 2;
                cnt[PIVCOH__MAXLEN]--;
            }
            /* reassign: stable by (capped old length, symbol) — snapshot the
             * order first, the assignments overwrite the sort keys */
            uint8_t order[256];
            int no = 0, cl = 1, left = cnt[1], cap;
            for (cap = 1; cap <= PIVCOH__MAXLEN; cap++)
                for (i = 0; i < 256; i++)
                    if (lens[i] && (lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN) == cap)
                        order[no++] = (uint8_t)i;
            for (i = 0; i < no; i++) {
                while (!left) left = cnt[++cl];
                lens[order[i]] = (uint8_t)cl;
                left--;
            }
        }
    }
    return pivcoh_table_from_lens(t, lens);    /* lengths -> schedule (shared) */
}

/* ---- encode ---- */

static void pivcoh__enc(const pivcoh_table *t, int idx, uint8_t *ranks, int K,
                        uint8_t **pp, uint8_t *pool)
{
    const pivcoh__rec *rec = &t->sched[idx];
    int kind = rec->kd & 3, j;
    uint8_t *p = *pp;

    if (kind == PIVCOH__FLAT) {                /* K local codes, D bits each */
        int D = rec->kd >> 2, nb = 0;
        uint64_t buf = 0;
        for (j = 0; j < K; j++) {
            buf |= (uint64_t)(uint8_t)(ranks[j] - rec->param) << nb;
            nb += D;
            while (nb >= 8) { *p++ = (uint8_t)buf; buf >>= 8; nb -= 8; }
        }
        if (nb) *p++ = (uint8_t)(buf & ((1u << nb) - 1));
        *pp = p;
        return;
    }
    uint8_t *kr = p;                                /* K_right, u16 LE */
    p += 2;
    *p++ = 0;                                       /* marker: raw bitmap */
    uint8_t *bm = p;
    p += (K + 7) >> 3;
    memset(bm, 0, (size_t)((K + 7) >> 3));
    int KR = 0;                                /* bit j = 1: rank > thr (right) */
    for (j = 0; j < K; j++)
        if (ranks[j] > rec->param) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); KR++; }
    kr[0] = (uint8_t)KR;
    kr[1] = (uint8_t)(KR >> 8);
    *pp = p;
    /* Mirror of the decode placement: the larger side compacts IN PLACE in
     * the rank slab (stable: write cursor <= read cursor), the smaller side
     * is extracted to the pool, and the first child's slab — dead once it
     * returns — serves as the second child's pool.  A K-subtree touches at
     * most K pool bytes (exact by induction).  LEAF_LEFT drops its left
     * side entirely and needs no pool at this level. */
    int KL = K - KR, wl = 0, wr = 0;
    if (kind == PIVCOH__LEAFL) {
        for (j = 0; j < K; j++)
            if (ranks[j] > rec->param) ranks[wr++] = ranks[j];
        if (KR > 0) pivcoh__enc(t, idx + rec->right, ranks, KR, pp, pool);
    } else if (KL >= KR) {
        for (j = 0; j < K; j++) {
            uint8_t v = ranks[j];
            if (v > rec->param) pool[wr++] = v; else ranks[wl++] = v;
        }
        if (KL > 0) pivcoh__enc(t, idx + 1, ranks, KL, pp, pool + KR);
        if (KR > 0) pivcoh__enc(t, idx + rec->right, pool, KR, pp, ranks);
    } else {
        for (j = 0; j < K; j++) {
            uint8_t v = ranks[j];
            if (v > rec->param) ranks[wr++] = v; else pool[wl++] = v;
        }
        if (KL > 0) pivcoh__enc(t, idx + 1, pool, KL, pp, pool + KL);
        if (KR > 0) pivcoh__enc(t, idx + rec->right, ranks, KR, pp, pool);
    }
}

PIVCOHDEF ptrdiff_t pivcoh_encode(const pivcoh_table *t,
                                  const uint8_t *in, size_t n,
                                  uint8_t *out, size_t out_cap, void *scratch)
{
    if (!t || !in || !out || n < 1 || n > 65535 || !t->num_ranks) return -1;
    if (out_cap < PIVCOH_ENCODE_BOUND(n)) return -1;
    uint8_t *sc = scratch ? (uint8_t *)scratch : (uint8_t *)malloc(PIVCOH_SCRATCH_SIZE(n));
    if (!sc) return -1;
    size_t i;
    for (i = 0; i < n; i++) sc[i] = t->sym_to_rank[in[i]];
    uint8_t *p = out;
    *p++ = (uint8_t)n;
    *p++ = (uint8_t)(n >> 8);
    pivcoh__enc(t, 0, sc, (int)n, &p, sc + n);
    if (!scratch) free(sc);
    return p - out;
}

/* ---- decode ---- */

#ifdef PIVCOH__NEON
/* Port of the production merge_vec_vec_neon 16-byte kernel: one 2-source
 * vqtbl2q over {R16, L16} per 16 outputs, cross-half cursor offset
 * folded into the index by SABD (|shuf0 - shuf1|).  The two 256x16 index
 * tables (8 KiB) build lazily on first decode; the writes are idempotent,
 * so concurrent first decodes are benign. */
static int8_t pivcoh__shuf0[256 * 16] __attribute__((aligned(16)));
static int8_t pivcoh__shuf1[256 * 16] __attribute__((aligned(16)));
static void pivcoh__merge_tabs(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
        int8_t pop = 0;
        int8_t *o0 = &pivcoh__shuf0[m * 16], *o1 = &pivcoh__shuf1[m * 16];
        for (int k = 0; k < 8; k++) {
            if ((m >> k) & 1) {
                o0[k] = pop; o1[k + 8] = (int8_t)(-pop); pop++;
            } else {
                int8_t v = (int8_t)(-16 - k + pop);
                o0[k] = v; o1[k + 8] = (int8_t)(8 - v);
            }
        }
        for (int k = 0; k < 8; k++) { o0[k + 8] = pop; o1[k] = 0; }
    }
    built = 1;
}
static inline void pivcoh__merge16(uint8_t *dest, const uint8_t *l,
                                   const uint8_t *r, uint64_t mask)
{
    int8x16_t s0 = vld1q_s8(&pivcoh__shuf0[(mask << 4) & 0xff0]);
    int8x16_t s1 = vld1q_s8(&pivcoh__shuf1[(mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r);
    src.val[1] = vld1q_u8(l);
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}
#endif

/* SWAR byte popcount (keeps the scalar core free of compiler builtins). */
static int pivcoh__pc8(unsigned m)
{
    m = m - ((m >> 1) & 0x55);
    m = (m & 0x33) + ((m >> 2) & 0x33);
    return (int)((m + (m >> 4)) & 0x0f);
}

/* Returns the advanced input pointer, or NULL on malformed input.  out
 * receives exactly K symbols; partner must have capacity K (see the
 * placement note at the merge below). */
static const uint8_t *pivcoh__dec(const pivcoh_table *t, int idx, int K, uint8_t *out,
                                  const uint8_t *p, const uint8_t *end, uint8_t *partner)
{
    const pivcoh__rec *rec = &t->sched[idx];
    int kind = rec->kd & 3, j;

    if (kind == PIVCOH__FLAT) {
        int D = rec->kd >> 2;
        size_t nb = ((size_t)K * (size_t)D + 7) >> 3;
        if ((size_t)(end - p) < nb) return NULL;
        const uint8_t *c2s = t->rank_to_sym + rec->param;
        /* 8 codes consume exactly D whole bytes, so the bit phase repeats
         * per group: one unaligned 64-bit load covers all 8 (8D <= 64)
         * and the cursor steps D bytes — no per-D dispatch.  The load
         * reads 8 bytes but consumes D, so run while 8 bytes remain in
         * the INPUT (not the region); the bit-cursor loop finishes. */
        const uint8_t *q = p;
        unsigned msk = (1u << D) - 1;
        for (j = 0; j + 8 <= K && end - q >= 8; j += 8, q += D) {
            uint64_t v;
            memcpy(&v, q, 8);
            out[j]     = c2s[ v            & msk];
            out[j + 1] = c2s[(v >> D)      & msk];
            out[j + 2] = c2s[(v >> (2*D))  & msk];
            out[j + 3] = c2s[(v >> (3*D))  & msk];
            out[j + 4] = c2s[(v >> (4*D))  & msk];
            out[j + 5] = c2s[(v >> (5*D))  & msk];
            out[j + 6] = c2s[(v >> (6*D))  & msk];
            out[j + 7] = c2s[(v >> (7*D))  & msk];
        }
        for (; j < K; j++) {
            int bit = j * D, off = bit & 7;
            unsigned v = p[bit >> 3];
            if (off + D > 8) v |= (unsigned)p[(bit >> 3) + 1] << 8;
            out[j] = c2s[(v >> off) & msk];
        }
        return p + nb;
    }
    if (end - p < 2) return NULL;
    int KR = p[0] | p[1] << 8;
    p += 2;
    if (KR > K) return NULL;
    if (end - p < 1 || *p++ != 0) return NULL; /* raw-bitmap marker only (no FSE) */
    size_t nb = (size_t)((K + 7) >> 3);
    if ((size_t)(end - p) < nb) return NULL;
    const uint8_t *bm = p;
    p += nb;
#define PIVCOH__BIT(j) ((bm[(j) >> 3] >> ((j) & 7)) & 1)

    /* The larger child decodes IN PLACE into out's tail; the smaller into
     * partner[0..KS), its own recursion ping-ponging into out's still-empty
     * prefix (right-larger children take partner+KL, capacity K-KL = KR).
     * So a K-subtree touches at most K partner bytes — exact by induction,
     * no allocator — and the forward merge is safe because its write cursor
     * can never pass its tail-side read cursor (j = li+ri <= tail0+ti, with
     * equality a self-copy; the li/ri guards keep hostile bitmaps inside). */
    int KL = K - KR;
    if (kind == PIVCOH__LEAFL) {               /* left = KL copies of one leaf */
        if (KR > 0 && !(p = pivcoh__dec(t, idx + rec->right, KR, out + KL, p, end, partner)))
            return NULL;
        uint8_t ls = t->rank_to_sym[rec->param];
        int r = 0;
        j = 0;
        /* Shift-register groups, one-sided: the left "register" is the
         * constant.  The register snapshot makes the in-place overlap
         * with out's tail harmless within a group; r + 8 <= KR bounds
         * the load, and the checked tail + final r == KR keep hostile
         * bitmaps memory-safe and rejected. */
        while (j + 8 <= K && r + 8 <= KR) {
            unsigned m = bm[j >> 3];
            uint64_t rv;
            memcpy(&rv, out + KL + r, 8);
#define PIVCOH__STEP(k) do { unsigned b_ = (m >> (k)) & 1; \
            out[j + (k)] = b_ ? (uint8_t)rv : ls;          \
            rv >>= b_ << 3; } while (0)
            PIVCOH__STEP(0); PIVCOH__STEP(1); PIVCOH__STEP(2); PIVCOH__STEP(3);
            PIVCOH__STEP(4); PIVCOH__STEP(5); PIVCOH__STEP(6); PIVCOH__STEP(7);
#undef PIVCOH__STEP
            r += pivcoh__pc8(m);
            j += 8;
        }
        for (; j < K; j++) {
            if (PIVCOH__BIT(j)) { if (r == KR) return NULL; out[j] = out[KL + r]; r++; }
            else out[j] = ls;
        }
        return r == KR ? p : NULL;             /* bitmap must match K_right */
    }
    const uint8_t *L, *R;
    if (KL >= KR) {
        if (KL > 0 && !(p = pivcoh__dec(t, idx + 1, KL, out + KR, p, end, partner)))
            return NULL;
        if (KR > 0 && !(p = pivcoh__dec(t, idx + rec->right, KR, partner, p, end, out)))
            return NULL;
        L = out + KR; R = partner;
    } else {
        if (KL > 0 && !(p = pivcoh__dec(t, idx + 1, KL, partner, p, end, out)))
            return NULL;
        if (KR > 0 && !(p = pivcoh__dec(t, idx + rec->right, KR, out + KL, p, end, partner + KL)))
            return NULL;
        L = partner; R = out + KL;
    }
    int li = 0, ri = 0;
    j = 0;
#ifdef PIVCOH__NEON
    /* The entry guard bounds the two 16-byte loads to the cursors' next
     * 16 bytes, covers the in-place overlap the same way the scalar
     * argument does (the store never passes the still-unread tail while
     * cursor + 16 <= side count), and keeps li/ri <= KL/KR so hostile-
     * bitmap validation still lands in the scalar tail.  No slack bytes
     * needed anywhere.  16 bits/iter measured FASTER than 32/64-bit
     * unrolls of the same kernel on M4 (the finer guard keeps more
     * skewed and small nodes on the vector path), and smaller. */
    while (j + 16 <= K && li + 16 <= KL && ri + 16 <= KR) {
        uint16_t mask;
        memcpy(&mask, bm + (j >> 3), 2);
        int pt = __builtin_popcount(mask);
        pivcoh__merge16(out + j, L + li, R + ri, mask);
        li += 16 - pt;
        ri += pt;
        j += 16;
    }
#endif
    /* Branch-free scalar groups: 8 elements off one mask byte.  Both
     * sides load 8 bytes into shift registers; each element selects the
     * low byte of one and shifts the consumed side by 8 — conditional
     * moves, no data-dependent branches (8x the branchy loop on random
     * bitmaps, and faster than it even on fully predictable ones).
     * Same guard/tail contract as the NEON loop, at 8-byte grain. */
    while (j + 8 <= K && li + 8 <= KL && ri + 8 <= KR) {
        unsigned m = bm[j >> 3];
        int pc = pivcoh__pc8(m);
        uint64_t lv, rv;
        memcpy(&lv, L + li, 8);
        memcpy(&rv, R + ri, 8);
#define PIVCOH__STEP(k) do { unsigned b_ = (m >> (k)) & 1; \
        out[j + (k)] = (uint8_t)(b_ ? rv : lv);            \
        rv >>= b_ << 3; lv >>= (b_ ^ 1) << 3; } while (0)
        PIVCOH__STEP(0); PIVCOH__STEP(1); PIVCOH__STEP(2); PIVCOH__STEP(3);
        PIVCOH__STEP(4); PIVCOH__STEP(5); PIVCOH__STEP(6); PIVCOH__STEP(7);
#undef PIVCOH__STEP
        li += 8 - pc;
        ri += pc;
        j += 8;
    }
    for (; j < K; j++) {
        if (PIVCOH__BIT(j)) { if (ri == KR) return NULL; out[j] = R[ri++]; }
        else               { if (li == KL) return NULL; out[j] = L[li++]; }
    }
    return p;
#undef PIVCOH__BIT
}

PIVCOHDEF ptrdiff_t pivcoh_decode(const pivcoh_table *t,
                                  const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap,
                                  size_t *consumed, void *scratch)
{
    if (!t || !in || !out || !t->num_ranks || in_len < 2) return -1;
#ifdef PIVCOH__NEON
    pivcoh__merge_tabs();
#endif
    int N = in[0] | in[1] << 8;
    if (N < 1 || (size_t)N > out_cap) return -1;
    uint8_t *sc = scratch ? (uint8_t *)scratch : (uint8_t *)malloc((size_t)N);
    if (!sc) return -1;
    const uint8_t *p = pivcoh__dec(t, 0, N, out, in + 2, in + in_len, sc);
    if (!scratch) free(sc);
    if (!p) return -1;
    if (consumed) *consumed = (size_t)(p - in);
    return N;
}

#endif /* PIVCOH_IMPLEMENTATION */

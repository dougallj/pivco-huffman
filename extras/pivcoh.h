/* pivcoh.h - v1.2 - minimal single-file PIVCO-Huffman block codec
 *
 * A tiny, scalar, allocation-free-capable implementation of the
 * PIVCO-Huffman wire format (https://github.com/MarcinZukowski/pivco-huffman).
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
 * is used internally) or a caller buffer for allocation-free operation:
 * PIVCOH_SCRATCH_SIZE(max_n) bytes covers either call, and a pure
 * decoder needs only PIVCOH_DECODE_SCRATCH_SIZE(max_n).  Tables and scratch are plain
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

/* Scratch bytes for encode (or either call) on blocks up to n symbols:
 * the encoder's rank buffer plus its partition pool (each at most n —
 * the walk keeps each node's larger side in place). */
#define PIVCOH_SCRATCH_SIZE(n)  (2 * (size_t)(n))

/* Decode-only scratch: the decoder skips ahead in the (in-memory) wire
 * to decode each node's larger child first, in place in out's tail, so
 * its partner buffer needs only floor(n/2) bytes. */
#define PIVCOH_DECODE_SCRATCH_SIZE(n)  ((size_t)(n) / 2 + 1)

typedef struct { uint8_t kd, param, right; } pivcoh__rec;

typedef struct {
    uint8_t code_len[256];   /* per-symbol code length, 0 = absent, max 11.
                                Filled by pivcoh_table_from_freqs; this is
                                what the encoder transmits to the decoder. */
    /* internals */
    uint16_t num_ranks, sched_len;
    uint8_t rank_to_sym[256], sym_to_rank[256];
    pivcoh__rec sched[255];
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

#define PIVCOH__MAXLEN 11
enum { PIVCOH__FULL = 0, PIVCOH__FLAT = 1, PIVCOH__PAIR = 2, PIVCOH__LEAFL = 3 };

/* ---- table build ---- */

typedef struct { uint8_t depth, bit, sym_idx; } pivcoh__chunk;

/* Pre-order schedule from the depth-sorted chunk list: chunks are the
 * tree's left-to-right leaves, and a leaf-depth sequence determines the
 * tree.  Returns 0 on non-Kraft-complete lengths. */
static int pivcoh__gen(pivcoh_table *t, const pivcoh__chunk *ch, int nch,
                       int *ci, int *rank, int depth, const uint8_t *items)
{
    if (*ci >= nch || depth > PIVCOH__MAXLEN || t->sched_len >= 255) return 0;
    if (ch[*ci].depth == depth) {              /* chunk leaf */
        const pivcoh__chunk *c = &ch[(*ci)++];
        int n = 1 << c->bit, r0 = *rank;
        memcpy(t->rank_to_sym + r0, items + c->sym_idx, (size_t)n);
        *rank += n;
        if (c->bit) {
            pivcoh__rec *r = &t->sched[t->sched_len++];
            r->kd = c->bit == 1 ? PIVCOH__PAIR
                                : (uint8_t)(PIVCOH__FLAT | c->bit << 2);
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
    int rl = (t->sched_len == mid_s && *rank == mid_r + 1);
    pivcoh__rec *r = &t->sched[my];
    if (rl && !ll) return 0;                   /* lone leaf is always LEFT */
    r->kd    = (uint8_t)(ll ? (rl ? PIVCOH__PAIR : PIVCOH__LEAFL) : PIVCOH__FULL);
    r->param = (uint8_t)(mid_r - 1);           /* thr / rank_begin */
    r->right = (uint8_t)(r->kd == PIVCOH__PAIR ? 0 : mid_s - my);
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
        t->sched[0].kd = PIVCOH__PAIR;
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
        pivcoh__chunk ch[256];
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

        if (maxd > PIVCOH__MAXLEN) {           /* RFC1951-style length limiting */
            int cnt[PIVCOH__MAXLEN + 2] = {0}, b;
            for (i = 0; i < 256; i++)
                if (lens[i]) cnt[lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN]++;
            long kraft = 0, target = 1L << PIVCOH__MAXLEN;
            for (i = 1; i <= PIVCOH__MAXLEN; i++) kraft += (long)cnt[i] << (PIVCOH__MAXLEN - i);
            while (kraft > target) {           /* over-full: lengthen the longest */
                for (b = PIVCOH__MAXLEN - 1; b >= 1 && !cnt[b]; b--) {}
                if (b < 1) break;
                cnt[b]--; cnt[b + 1]++;
                kraft -= 1L << (PIVCOH__MAXLEN - b - 1);
            }
            while (kraft < target && cnt[PIVCOH__MAXLEN]) {  /* under-full: shorten */
                for (b = PIVCOH__MAXLEN - 1; b >= 1; b--) {
                    long d = (1L << (PIVCOH__MAXLEN - b)) - 1;
                    if (kraft + d <= target) { cnt[PIVCOH__MAXLEN]--; cnt[b]++; kraft += d; break; }
                }
                if (kraft < target) {
                    if (cnt[PIVCOH__MAXLEN] < 2) break;
                    cnt[PIVCOH__MAXLEN]--; cnt[PIVCOH__MAXLEN - 1]++; kraft++;
                }
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
    uint8_t *kr = NULL;
    if (kind != PIVCOH__PAIR) { kr = p; p += 2; }   /* K_right, u16 LE */
    *p++ = 0;                                       /* marker: raw bitmap */
    uint8_t *bm = p;
    p += (K + 7) >> 3;
    memset(bm, 0, (size_t)((K + 7) >> 3));
    int KR = 0;                                /* bit j = 1: rank > thr (right) */
    for (j = 0; j < K; j++)
        if (ranks[j] > rec->param) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); KR++; }
    if (kr) { kr[0] = (uint8_t)KR; kr[1] = (uint8_t)(KR >> 8); }
    *pp = p;
    if (kind == PIVCOH__PAIR) return;
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

/* Advance past a K-element subtree's wire bytes without decoding it —
 * the wire layout is fully determined by the schedule plus the K_right
 * headers, so this reads ~3 header bytes per node.  Used to reach a
 * larger right child so it can decode first; the real decode of the
 * skipped bytes follows and re-validates them. */
static const uint8_t *pivcoh__skip(const pivcoh_table *t, int idx, int K,
                                   const uint8_t *p, const uint8_t *end)
{
    const pivcoh__rec *rec = &t->sched[idx];
    int kind = rec->kd & 3;
    if (kind == PIVCOH__FLAT) {
        size_t nb = ((size_t)K * (size_t)(rec->kd >> 2) + 7) >> 3;
        return (size_t)(end - p) < nb ? NULL : p + nb;
    }
    int KR = 0;
    if (kind != PIVCOH__PAIR) {
        if (end - p < 2) return NULL;
        KR = p[0] | p[1] << 8;
        p += 2;
        if (KR > K) return NULL;
    }
    size_t nb = 1 + (size_t)((K + 7) >> 3);    /* marker + bitmap */
    if ((size_t)(end - p) < nb || *p != 0) return NULL;
    p += nb;
    if (kind == PIVCOH__FULL && K - KR > 0 &&
        !(p = pivcoh__skip(t, idx + 1, K - KR, p, end))) return NULL;
    return (kind != PIVCOH__PAIR && KR > 0)
               ? pivcoh__skip(t, idx + rec->right, KR, p, end) : p;
}

/* Returns the advanced input pointer, or NULL on malformed input.  out
 * receives exactly K symbols; partner must have capacity floor(K/2)
 * (see the placement note at the merge below). */
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
        for (j = 0; j < K; j++) {
            int bit = j * D, off = bit & 7;
            unsigned v = p[bit >> 3];
            if (off + D > 8) v |= (unsigned)p[(bit >> 3) + 1] << 8;
            out[j] = c2s[(v >> off) & ((1u << D) - 1)];
        }
        return p + nb;
    }
    int KR = 0;
    if (kind != PIVCOH__PAIR) {
        if (end - p < 2) return NULL;
        KR = p[0] | p[1] << 8;
        p += 2;
        if (KR > K) return NULL;
    }
    if (end - p < 1 || *p++ != 0) return NULL; /* raw-bitmap marker only (no FSE) */
    size_t nb = (size_t)((K + 7) >> 3);
    if ((size_t)(end - p) < nb) return NULL;
    const uint8_t *bm = p;
    p += nb;
#define PIVCOH__BIT(j) ((bm[(j) >> 3] >> ((j) & 7)) & 1)

    if (kind == PIVCOH__PAIR) {
        for (j = 0; j < K; j++) out[j] = t->rank_to_sym[rec->param + PIVCOH__BIT(j)];
        return p;
    }
    /* The larger child always decodes FIRST, IN PLACE into out's tail —
     * when that child is the right one, pivcoh__skip jumps the wire cursor
     * over the left subtree's bytes and the left decodes second.  The
     * smaller child goes to partner[0..KS), its own recursion ping-ponging
     * into out's still-empty prefix.  Every child's partner need is thus
     * <= floor(K/2) (exact by induction, no allocator), and the forward
     * merge is safe because its write cursor can never pass its tail-side
     * read cursor (j = li+ri <= tail read index, equality a self-copy; the
     * li/ri guards keep hostile bitmaps inside). */
    int KL = K - KR;
    if (kind == PIVCOH__LEAFL) {               /* left = KL copies of one leaf */
        if (KR > 0 && !(p = pivcoh__dec(t, idx + rec->right, KR, out + KL, p, end, partner)))
            return NULL;
        uint8_t ls = t->rank_to_sym[rec->param];
        int r = 0;
        for (j = 0; j < K; j++) {
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
        const uint8_t *pl = p;
        if (KL > 0 && !(p = pivcoh__skip(t, idx + 1, KL, p, end))) return NULL;
        const uint8_t *pr = p;
        if (!(p = pivcoh__dec(t, idx + rec->right, KR, out + KL, p, end, partner)))
            return NULL;
        if (KL > 0 && !pivcoh__dec(t, idx + 1, KL, partner, pl, pr, out))
            return NULL;
        L = partner; R = out + KL;
    }
    int li = 0, ri = 0;
    for (j = 0; j < K; j++) {
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
    int N = in[0] | in[1] << 8;
    if (N < 1 || (size_t)N > out_cap) return -1;
    uint8_t *sc = scratch ? (uint8_t *)scratch : (uint8_t *)malloc(PIVCOH_DECODE_SCRATCH_SIZE(N));
    if (!sc) return -1;
    const uint8_t *p = pivcoh__dec(t, 0, N, out, in + 2, in + in_len, sc);
    if (!scratch) free(sc);
    if (!p) return -1;
    if (consumed) *consumed = (size_t)(p - in);
    return N;
}

#endif /* PIVCOH_IMPLEMENTATION */

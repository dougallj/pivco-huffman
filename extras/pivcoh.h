/* pivcoh.h - v2.0 - single-file PIVCO-Huffman block codec, NEON edition
 *
 * A minimal, allocation-free-capable implementation of the
 * PIVCO-Huffman wire format (https://github.com/MarcinZukowski/pivco-huffman).
 * This edition is aarch64-only: it carries straight ports of the
 * production library's NEON kernels (SABD two-table merge, per-D flat
 * decode, p16rev partition, ryg flat pack) and the production decoder's
 * bump-arena buffer placement, so decode/encode speed tracks the real
 * library rather than a portable-scalar floor.  ~53 KiB of static
 * shuffle tables build lazily on first use — the writes are idempotent,
 * so concurrent first calls are benign.
 * Streams are byte-identical to the full library's PH-only mode (wire
 * v0.7 decode-order layout; raw bitmaps —
 * pivco_huffman_set_fse_enabled(0), which is also what the pivcohuf
 * tool's default non-ANS format uses; "optimized" tree shaping, max
 * code length 11).  FSE/ANS-coded blocks are not supported and are
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
 * PIVCOH_SCRATCH_SIZE(max_n) bytes always suffices, and decode-only
 * callers can pass the smaller PIVCOH_DECODE_SCRATCH_SIZE(max_n).
 * Tables and scratch are plain memory: no cleanup calls, safe to copy,
 * const tables are shareable across threads (encode/decode themselves
 * touch only their arguments).
 *
 * Encoding a symbol whose frequency/length was zero produces a valid but
 * meaningless stream (never memory-unsafe).  Encode's SIMD packers may
 * scribble up to 16 junk bytes past the returned length — always inside
 * out_cap (>= PIVCOH_ENCODE_BOUND(n) is required).  Decode is safe on
 * hostile input: it never reads past `in + in_len`, never writes past N
 * symbols, and returns -1 on any malformed stream — truncation,
 * structural errors, and bitmaps whose popcount contradicts their
 * K_right header (each merge checks its final cursor position against
 * the header; monotone cursors make that one compare an exact
 * full-bitmap validation).
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

/* Scratch bytes for encode or decode of blocks up to n symbols.  The
 * encoder dominates: a ranks buffer (n + 64) plus one right-half per
 * recursion level (max 11 levels), like the production arena. */
#define PIVCOH_SCRATCH_SIZE(n)  (13 * (size_t)(n) + 128)

/* Scratch bytes for DECODE ONLY of blocks up to n symbols.  The decode
 * walk ping-pongs children through (out, partner) buffer pairs instead
 * of growing an arena, so a valid stream touches at most 1.5n + 128
 * bytes (the root's children plus the largest partner).  The other
 * 0.5n is hostile-input headroom: a bitmap that lies about its K_right
 * can walk a merge cursor up to K bytes past its side before the
 * end-of-merge check rejects the stream, and keeping that in-bounds by
 * padding is free where in-loop cursor guards would cost ~2% decode
 * speed.  PIVCOH_SCRATCH_SIZE also always suffices. */
#define PIVCOH_DECODE_SCRATCH_SIZE(n)  (2 * (size_t)(n) + 128)

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

#if !defined(__aarch64__)
#error "pivcoh.h v2.x is the NEON edition and requires aarch64 (the scalar codec lives on the main branch)"
#endif

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#define PIVCOH__MAXLEN 11
enum { PIVCOH__FULL = 0, PIVCOH__FLAT = 1, PIVCOH__LEAFL = 3 };

/* ---- table build ---- */

typedef struct { uint8_t depth, bit, sym_idx; } pivcoh__chunk;

/* Pre-order schedule from the depth-sorted chunk list: chunks are the
 * tree's left-to-right leaves, and the leaf-depth sequence determines
 * the tree.  Iterative, with an explicit stack of open internal nodes
 * (port of the production build_schedule; the recursion this replaces
 * was the dominant per-window table-build cost on ragged deep trees).
 * The walk doubles as Kraft-completeness validation.  Returns 0, or -1
 * on non-Kraft-complete lengths. */
typedef struct { int my, rank0, mid_sched, mid_rank, state; } pivcoh__frame;

static int pivcoh__sched(pivcoh_table *t, const pivcoh__chunk *ch, int nch,
                         const uint8_t *items)
{
    pivcoh__frame stk[PIVCOH__MAXLEN + 1];
    const int cap = (int)(sizeof t->sched / sizeof *t->sched);
    int sp = 0, ci = 0, rank = 0;

    for (;;) {
        if (ci >= nch) return -1;              /* under-subscribed lengths */

        /* Descend the left spine until a chunk sits at this depth. */
        while (ch[ci].depth != sp) {
            if (sp > PIVCOH__MAXLEN || t->sched_len >= cap) return -1;
            pivcoh__frame *f = &stk[sp++];
            f->my    = t->sched_len++;
            f->rank0 = rank;
            f->state = 0;
        }

        /* Consume the chunk-leaf.  Width-1/2 chunks dominate skewed
         * alphabets; keep their copies inline (a variable-size memcpy
         * is a libc dispatch per chunk). */
        {
            const pivcoh__chunk *c = &ch[ci++];
            int b = c->bit, r0 = rank;
            uint8_t *dst = t->rank_to_sym + r0;
            const uint8_t *s = items + c->sym_idx;
            if (b == 0)      dst[0] = s[0];
            else if (b == 1) { dst[0] = s[0]; dst[1] = s[1]; }
            else             memcpy(dst, s, (size_t)1 << b);
            rank += 1 << b;
            if (b != 0) {
                if (t->sched_len >= cap) return -1;
                pivcoh__rec *r = &t->sched[t->sched_len++];
                r->kd    = (uint8_t)(PIVCOH__FLAT | b << 2);
                r->param = (uint8_t)r0;
                r->right = 0;
            }
        }

        /* Ascend, completing parents whose right child just finished. */
        for (;;) {
            if (sp == 0) {                     /* root subtree complete */
                if (ci != nch) return -1;      /* over-subscribed */
                t->num_ranks = (uint16_t)rank;
                return 0;
            }
            pivcoh__frame *f = &stk[sp - 1];
            if (f->state == 0) {               /* left done; do the right */
                f->state     = 1;
                f->mid_sched = t->sched_len;
                f->mid_rank  = rank;
                break;
            }
            /* Right done: finalize this internal node's record.  A lone
             * leaf beside an internal sibling is always LEFT (chunk
             * depths never decrease left-to-right under a node), and the
             * optimized chunking emits at most one width-1 chunk per
             * length so two lone siblings cannot meet — right_lone only
             * flags malformed inputs. */
            int left_lone  = f->mid_sched == f->my + 1 &&
                             f->mid_rank == f->rank0 + 1;
            int right_lone = t->sched_len == f->mid_sched &&
                             rank == f->mid_rank + 1;
            if (right_lone) return -1;
            pivcoh__rec *r = &t->sched[f->my];
            r->kd    = (uint8_t)(left_lone ? PIVCOH__LEAFL : PIVCOH__FULL);
            r->param = (uint8_t)(f->mid_rank - 1); /* thr / rank_begin */
            r->right = (uint8_t)(f->mid_sched - f->my);
            sp--;
        }
    }
}

PIVCOHDEF int pivcoh_table_from_lens(pivcoh_table *t, const uint8_t code_len[256])
{
    /* Fused validation + length histogram + counting sort (port of the
     * production length_histogram_sort): one vceq sweep per length L
     * counts bin L and extracts its symbols — in symbol order, so
     * items[] comes out sorted by (length, symbol) — with a REGISTER
     * output cursor.  No bin-increment or cursor[L]++ store-to-load
     * forwarding chains, and validation falls out of bin accounting:
     * any byte outside {0} u [1, MAXLEN] is never counted, so the zero
     * count plus the extracted total falls short of 256.  (NEON has no
     * byte movemask; vshrn narrows the compare to 4 bits per lane.) */
    uint8_t items[256];
    int cnt[PIVCOH__MAXLEN + 1], n_used = 0, n_zero, s, L;
    {
        const uint8x16_t vzero = vdupq_n_u8(0);
        uint8x16_t zacc = vzero;
        for (s = 0; s < 256; s += 16)
            zacc = vsubq_u8(zacc, vceqq_u8(vld1q_u8(code_len + s), vzero));
        n_zero = (int)vaddlvq_u8(zacc);
    }
    for (L = 1; L <= PIVCOH__MAXLEN; L++) {
        int start = n_used;
        const uint8x16_t target = vdupq_n_u8((uint8_t)L);
        for (s = 0; s < 256; s += 16) {
            uint8x16_t eq = vceqq_u8(vld1q_u8(code_len + s), target);
            uint64_t m = vget_lane_u64(vreinterpret_u64_u8(
                             vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
            m &= 0x1111111111111111ull;
            while (m) {
                items[n_used++] = (uint8_t)(s + (__builtin_ctzll(m) >> 2));
                m &= m - 1;
            }
        }
        cnt[L] = n_used - start;
    }
    if (n_used == 0 || n_zero + n_used != 256) return 0;
    memcpy(t->code_len, code_len, 256);
    t->sched_len = 0;

    if (n_used == 1) {                         /* degenerate: 1-bit code */
        s = items[0];
        memset(t->code_len, 0, 256);
        t->code_len[s] = 1;
        t->rank_to_sym[0] = t->rank_to_sym[1] = (uint8_t)s;
        t->num_ranks = 2;
        t->sched[0].kd = PIVCOH__FLAT | 1 << 2;
        t->sched[0].param = t->sched[0].right = 0;
        t->sched_len = 1;
    } else {
        /* "optimized" chunking: split each length's count by its set bits
         * (largest first), a 2^b chunk rooted at depth L-b; then stable
         * depth-sort so canonical assignment fills the tree left-to-right */
        pivcoh__chunk ch[49];   /* max sum popcount(cnt[L]): 11 classes, sum <= 256 */
        int nch = 0, i, j, acc;
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

        if (pivcoh__sched(t, ch, nch, items) != 0) return 0;
    }

    memset(t->sym_to_rank, 0, 256);
    for (s = t->num_ranks - 1; s >= 0; s--)
        t->sym_to_rank[t->rank_to_sym[s]] = (uint8_t)s;
    return 1;
}

typedef struct { uint64_t freq; uint16_t sym; } pivcoh__leaf;

/* Stable ascending (freq, sym) sort of the leaves.  They arrive in
 * symbol order (the stable seed), so a stable freq sort IS the (freq,
 * sym) order.  Small alphabets insertion-sort; larger ones take an LSD
 * radix over only the frequency bytes that VARY across the set (vary =
 * OR ^ AND of all freqs, a free by-product of the caller's scan) — a
 * constant byte is an identity pass, so it is skipped outright.  Port
 * of the production sort_leaves_by_freq (same n <= 40 crossover, minus
 * its dominant-bin scatter specialization); the O(n^2) insertion sort
 * this replaces was 3-5x the whole production table build on
 * fresh-tables-every-4K workloads over near-full alphabets. */
static void pivcoh__sort_leaves(pivcoh__leaf *leaf, int n, uint64_t vary)
{
    int i, j;
    if (n <= 40) {
        for (i = 1; i < n; i++) {
            pivcoh__leaf cur = leaf[i];
            for (j = i - 1; j >= 0 && leaf[j].freq > cur.freq; j--)
                leaf[j + 1] = leaf[j];
            leaf[j + 1] = cur;
        }
        return;
    }
    int shift[8], npass = 0;
    for (int b = 0; b < 64; b += 8)
        if ((vary >> b) & 0xFF) shift[npass++] = b;
    if (npass == 0) return;                    /* all frequencies equal */
    uint16_t cnt[8][256];
    memset(cnt, 0, (size_t)npass * sizeof(cnt[0]));
    for (i = 0; i < n; i++)                    /* all planes in one pass */
        for (int p = 0; p < npass; p++)
            cnt[p][(leaf[i].freq >> shift[p]) & 0xFF]++;
    pivcoh__leaf tmp[256], *src = leaf, *dst = tmp;
    for (int p = 0; p < npass; p++) {
        unsigned sum = 0;
        for (int k = 0; k < 256; k++) {
            unsigned c = cnt[p][k];
            cnt[p][k] = (uint16_t)sum;
            sum += c;
        }
        for (i = 0; i < n; i++)
            dst[cnt[p][(src[i].freq >> shift[p]) & 0xFF]++] = src[i];
        pivcoh__leaf *t = src; src = dst; dst = t;
    }
    if (src != leaf) memcpy(leaf, src, (size_t)n * sizeof(*leaf));
}

PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256])
{
    pivcoh__leaf leaf[256];
    uint64_t orv = 0, andv = ~(uint64_t)0;
    int n = 0, i;
    for (i = 0; i < 256; i++)
        if (freq[i]) {
            leaf[n].freq = freq[i];
            leaf[n].sym  = (uint16_t)i;
            n++;
            orv |= freq[i];
            andv &= freq[i];
        }
    if (n == 0) return 0;
    pivcoh__sort_leaves(leaf, n, orv ^ andv);

    uint8_t lens[256] = {0};
    if (n == 1) {
        lens[leaf[0].sym] = 1;
    } else {
        /* van Leeuwen two-queue: sorted leaves + FIFO of made internals */
        uint64_t nf[512];
        int parent[512], li = 0, ih = n, ni = n, rem;
        for (i = 0; i < n; i++) nf[i] = leaf[i].freq;
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
            uint8_t L = depth[i] ? depth[i] : 1;
            lens[leaf[i].sym] = L;
            if (L > maxd) maxd = L;
        }

        if (maxd > PIVCOH__MAXLEN) {           /* DEFLATE-style length limiting */
            /* Clamp to MAXLEN, then repair the Kraft sum: in units of
             * 2^-MAXLEN a clamped symbol weighs 1 instead of < 1, so
             * 0 < debt < cnt[MAXLEN].  Each step re-homes one MAXLEN
             * symbol as the sibling of a symbol demoted from the deepest
             * shorter level: net -1 unit, so the loop lands on Kraft == 1
             * exactly.  A non-empty b always exists: 256 symbols all at
             * MAXLEN would be under-subscribed. */
            int cnt[PIVCOH__MAXLEN + 2] = {0}, b, L;
            for (i = 0; i < 256; i++)
                if (lens[i]) cnt[lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN]++;
            /* reassign order: stable by (capped old length, symbol) — one
             * counting-sort scatter off the pre-repair histogram.  Comes
             * first: the repair below rewrites the histogram and the
             * assignments rewrite the sort keys.  (This replaces an
             * 11x256 order-building sweep that dominated the build on
             * exactly the windows deep enough to need limiting.) */
            uint8_t order[256];
            int cur[PIVCOH__MAXLEN + 1], no = 0;
            for (L = 1; L <= PIVCOH__MAXLEN; L++) { cur[L] = no; no += cnt[L]; }
            for (i = 0; i < 256; i++)
                if (lens[i])
                    order[cur[lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN]++] = (uint8_t)i;
            long kraft = 0;
            for (i = 1; i <= PIVCOH__MAXLEN; i++) kraft += (long)cnt[i] << (PIVCOH__MAXLEN - i);
            for (; kraft > 1L << PIVCOH__MAXLEN; kraft--) {
                for (b = PIVCOH__MAXLEN - 1; !cnt[b]; b--) {}
                cnt[b]--;
                cnt[b + 1] += 2;
                cnt[PIVCOH__MAXLEN]--;
            }
            int cl = 1, left = cnt[1];
            for (i = 0; i < no; i++) {
                while (!left) left = cnt[++cl];
                lens[order[i]] = (uint8_t)cl;
                left--;
            }
        }
    }
    return pivcoh_table_from_lens(t, lens);    /* lengths -> schedule (shared) */
}

/* ================= decode kernels (ports of primitives_neon) ================
 *
 * Buffer contract: a merge kernel may read up to 16 bytes past a source
 * cursor and — interior (non-EXACT) merges only — overwrite up to 15
 * bytes past out+K, saved and restored around the merge.  Validation is
 * the end-of-merge r_end equality: cursors are monotone, so final
 * r == r_end proves no prefix of the bitmap ever overdrew either side
 * (each output consumes exactly one input, so the l side is implied) —
 * an exact "bitmap popcount == K_right" test for one compare per merge,
 * nothing per iteration (in-loop guards were tried and cost ~2%).  On a
 * stream that fails it the cursors strayed mid-merge first: by at most
 * K bytes past a side plus the 64-byte iteration window, absorbed by
 * the decode arena's pad.  Writes of EXACT merges (the root, targeting
 * the caller's buffer) are exactly bounded to out[0,K).  Bitmap reads
 * never pass ceil(K/8) bytes, flat-region reads never pass ceil(K*D/8)
 * bytes.
 */

/* Two-table SABD merge shuffles (8 KiB — the only merge tables: every
 * merge tail is a 16-byte SABD chunk too, so the old stride-16/8
 * expand-tab ladder and its ~21 KiB of tables are gone). */
static int8_t  pivcoh__mshuf0[256 * 16]     __attribute__((aligned(16)));
static int8_t  pivcoh__mshuf1[256 * 16]     __attribute__((aligned(16)));

static void pivcoh__init_dec(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
        int8_t pop = 0;
        int8_t *o0 = &pivcoh__mshuf0[m * 16], *o1 = &pivcoh__mshuf1[m * 16];
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

/* One 16-byte merge: 2-source TBL over {R,L}, cross-half cursor offset
 * folded into the index by SABD (|shuf0 - shuf1|). */
static inline void pivcoh__merge16(uint8_t *dest, const uint8_t *l_list,
                                   const uint8_t *r_list, uint64_t mask)
{
    int8x16_t s0 = vld1q_s8(&pivcoh__mshuf0[(mask << 4) & 0xff0]);
    int8x16_t s1 = vld1q_s8(&pivcoh__mshuf1[(mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r_list);
    src.val[1] = vld1q_u8(l_list);
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}

/* Same, with the L source in a broadcast register (LEAF_LEFT). */
static inline void pivcoh__merge16cst(uint8_t *dest, uint8x16_t Lb,
                                      const uint8_t *r_list, uint64_t mask)
{
    int8x16_t s0 = vld1q_s8(&pivcoh__mshuf0[(mask << 4) & 0xff0]);
    int8x16_t s1 = vld1q_s8(&pivcoh__mshuf1[(mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r_list);
    src.val[1] = Lb;
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}

/* Load the final partial chunk's mask (rem = K mod 16 bits, 1..15),
 * trimmed to the real bits: trimming keeps the cursor advance exact
 * (the r_end equality check depends on it), makes the phantom lanes
 * select harmless left/register bytes, and never touches a bitmap
 * byte past ceil(K/8). */
static inline unsigned pivcoh__tailmask(const uint8_t *bm, int j, unsigned rem)
{
    unsigned mask = bm[j >> 3];
    if (rem > 8) mask |= (unsigned)bm[(j >> 3) + 1] << 8;
    return mask & ((1u << rem) - 1);
}

/* merge_vec_vec: 64 bytes/iter main loop — four 16-byte SABD chunks
 * share one vcnt + 64-bit-multiply prefix sum for the per-chunk cursor
 * splits and the L/R advance — then 16-byte chunks.
 *
 * EXACT=0 (interior nodes, arena-backed out): entirely SIMD, no scalar
 * tail.  The final partial chunk runs mask-trimmed at full 16-byte
 * width, overwriting up to 15 bytes past out+K; the 16 bytes there are
 * saved up front and restored after.  EXACT=1 (the root merge, which
 * targets the caller's buffer): whole chunks while they fit, then a
 * plain scalar tail (<= 15 elements, once per block).
 *
 * Both variants validate at the end — see the section comment.  Returns
 * 0, or -1 when the bitmap contradicts the K_right header. */
__attribute__((always_inline)) static inline
int pivcoh__mvv(const uint8_t *bm, int K, const uint8_t *l, int KL,
                const uint8_t *r, int KR, uint8_t *out, int EXACT)
{
    const uint8_t *l_end = l + KL, *r_end = r + KR;
    uint8x16_t keep = vdupq_n_u8(0);
    if (!EXACT) keep = vld1q_u8(out + K);
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        /* Byte k of pfx = sum of the mask's byte-popcounts 0..k: bytes
         * 1/3/5 are the 16-bit chunk boundaries, byte 7 the total. */
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff;
        pivcoh__merge16(out + i,      l,           r,      mask);
        pivcoh__merge16(out + i + 16, l + 16 - p0, r + p0, mask >> 16);
        pivcoh__merge16(out + i + 32, l + 32 - p1, r + p1, mask >> 32);
        pivcoh__merge16(out + i + 48, l + 48 - p2, r + p2, mask >> 48);
        intptr_t p3 = pfx >> 56;
        r += p3; l += 64 - p3;
    }
    int j = (int)i;
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        pivcoh__merge16(out + j, l, r, m16);
        int pt = __builtin_popcount(m16);
        r += pt; l += 16 - pt;
    }
    if (!EXACT) {
        if (j < K) {
            unsigned mask = pivcoh__tailmask(bm, j, (unsigned)(K - j));
            pivcoh__merge16(out + j, l, r, mask);
            r += __builtin_popcount(mask);
        }
        vst1q_u8(out + K, keep);
    } else {
        for (; j < K; j++) {
            if ((bm[j >> 3] >> (j & 7)) & 1) {
                if (r == r_end) return -1;
                out[j] = *r++;
            } else {
                if (l == l_end) return -1;
                out[j] = *l++;
            }
        }
    }
    return r == r_end ? 0 : -1;
}
static int pivcoh__merge_vec_vec(const uint8_t *bm, int K,
                                 const uint8_t *l, int KL,
                                 const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mvv(bm, K, l, KL, r, KR, out, 0); }
static int pivcoh__merge_vec_vec_x(const uint8_t *bm, int K,
                                   const uint8_t *l, int KL,
                                   const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mvv(bm, K, l, KL, r, KR, out, 1); }

/* merge_cst_vec: L is a broadcast constant (LEAF_LEFT) — no L load or
 * cursor; only the R cursor advances (and is guarded/validated).  Same
 * EXACT/tail-free split as pivcoh__mvv. */
__attribute__((always_inline)) static inline
int pivcoh__mcv(const uint8_t *bm, int K, uint8_t left_sym,
                const uint8_t *r, int KR, uint8_t *out, int EXACT)
{
    const uint8_t *r_end = r + KR;
    uint8x16_t Lb = vdupq_n_u8(left_sym);
    uint8x16_t keep = vdupq_n_u8(0);
    if (!EXACT) keep = vld1q_u8(out + K);
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff;
        pivcoh__merge16cst(out + i,      Lb, r,      mask);
        pivcoh__merge16cst(out + i + 16, Lb, r + p0, mask >> 16);
        pivcoh__merge16cst(out + i + 32, Lb, r + p1, mask >> 32);
        pivcoh__merge16cst(out + i + 48, Lb, r + p2, mask >> 48);
        r += pfx >> 56;
    }
    int j = (int)i;
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        pivcoh__merge16cst(out + j, Lb, r, m16);
        r += __builtin_popcount(m16);
    }
    if (!EXACT) {
        if (j < K) {
            unsigned mask = pivcoh__tailmask(bm, j, (unsigned)(K - j));
            pivcoh__merge16cst(out + j, Lb, r, mask);
            r += __builtin_popcount(mask);
        }
        vst1q_u8(out + K, keep);
    } else {
        for (; j < K; j++) {
            if ((bm[j >> 3] >> (j & 7)) & 1) {
                if (r == r_end) return -1;
                out[j] = *r++;
            } else out[j] = left_sym;
        }
    }
    return r == r_end ? 0 : -1;
}
static int pivcoh__merge_cst_vec(const uint8_t *bm, int K, uint8_t left_sym,
                                 const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mcv(bm, K, left_sym, r, KR, out, 0); }
static int pivcoh__merge_cst_vec_x(const uint8_t *bm, int K, uint8_t left_sym,
                                   const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mcv(bm, K, left_sym, r, KR, out, 1); }

/* merge_cst_cst: both sides constant — a D=1 flat decode.  A 2-byte
 * (left, right) c2s replicated across 16 lanes, indexed by the bm bit
 * via the same dup-shuffle + per-lane shift as the D=2 unpack.  Two
 * tails like the flat kernels: tf runs 16-wide chunks to K with
 * save/restore (the last chunk reads <= 1 byte past the region, inside
 * the stream); tf=0 (a 2-symbol flat root) finishes scalar. */
static const uint8_t pivcoh__two_dup_tab[16]   = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
static const int8_t  pivcoh__two_shift_tab[16] = {0,-1,-2,-3,-4,-5,-6,-7,
                                                  0,-1,-2,-3,-4,-5,-6,-7};
static void pivcoh__merge_cst_cst(const uint8_t *bm, int K,
                                  uint8_t left_sym, uint8_t right_sym,
                                  uint8_t *out, int tf)
{
    uint16_t lr_word = (uint16_t)left_sym | ((uint16_t)right_sym << 8);
    uint8x16_t c2s_vec = vreinterpretq_u8_u16(vdupq_n_u16(lr_word));
    uint8x16_t dup_v   = vld1q_u8(pivcoh__two_dup_tab);
    int8x16_t  shift_v = vld1q_s8(pivcoh__two_shift_tab);
    uint8x16_t one_v   = vdupq_n_u8(1);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + K);

    int j = 0, lim = tf ? K : K - 15;
    for (; j < lim; j += 16) {
        uint16_t bm_word; memcpy(&bm_word, bm + (j >> 3), 2);
        uint8x16_t bm_lo = vreinterpretq_u8_u16(
            vsetq_lane_u16(bm_word, vdupq_n_u16(0), 0));
        uint8x16_t dup     = vqtbl1q_u8(bm_lo, dup_v);
        uint8x16_t shifted = vshlq_u8(dup, shift_v);
        uint8x16_t idx     = vandq_u8(shifted, one_v);
        vst1q_u8(out + j, vqtbl1q_u8(c2s_vec, idx));
    }
    if (tf) { vst1q_u8(out + K, keep); return; }
    for (; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right_sym : left_sym;
    }
}

/* ---- flat-subtree D-bit decode helpers ----
 *
 * Every flat kernel below has two tails, chosen by `tf` (tail-free):
 * the walk passes tf=1 when the input holds >= 16 readable bytes past
 * the flat region AND out is arena-backed — which is (almost) every
 * interior flat region: in decode order the last bytes of a block are
 * the ROOT's merge bitmap, so an interior region is followed by its
 * ancestors' records (a parent's bitmap alone exceeds 16 bytes once
 * the region holds ~120 symbols).  tf kernels simply run their MAIN
 * loop past n — same shape, same constants, no separate tail pipeline —
 * reading at most 16 bytes past the region (inside the stream) and
 * scribbling up to one iteration's width minus one (15..63 bytes,
 * kernel-dependent) past out+n, saved and restored at loop width — so
 * the byte-wise "safe" unpacks and per-code scalar tails are gone.
 * tf=0 (a flat ROOT decoding into the caller's buffer — where the
 * region really can end the stream — or a rare end-of-block interior
 * region) keeps the region-bounded vector loops and finishes the last
 * few codes with the scalar extractor below. */

static inline uint32_t pivcoh__extract_bits(const uint8_t *in, int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3, bit_off = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8) val |= ((uint32_t)in[byte_idx + 1]) << 8;
    return (val >> bit_off) & ((1u << D) - 1);
}

static const uint8_t pivcoh__d7_shuf_tab[16] = {0,1, 0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,6};
static const int16_t pivcoh__d7_shift_tab[8] = {0, -7, -6, -5, -4, -3, -2, -1};
static inline uint8x8_t pivcoh__d7_unpack(const uint8_t *bm_ptr)
{
    uint8x16_t bm_lo = vld1q_u8(bm_ptr);
    uint16x8_t w = vreinterpretq_u16_u8(vqtbl1q_u8(bm_lo, vld1q_u8(pivcoh__d7_shuf_tab)));
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(pivcoh__d7_shift_tab));
    return vmovn_u16(vandq_u16(shifted, vdupq_n_u16(0x7F)));
}

/* One 16-output pair-gather chunk for the byte-crossing depths: a
 * 16-byte load, one vqtbl1 placing two adjacent codes in each u16 lane,
 * a u16 shift aligning the pair, a u8 shift + mask isolating each code,
 * then the c2s scatter.  Consumes 2D input bytes of the 16 loaded.
 * These ARE the D=5/6 main-loop bodies — main loop and tf tail share
 * one shape and one constant set per kernel. */
static inline void pivcoh__d5_chunk16(uint8_t *dst, const uint8_t *src,
                                      uint8x16x2_t c2s_vec)
{
    static const uint8_t pair_shuf_t[16] = { 0,1, 1,2, 2,3, 3,4, 5,6, 6,7, 7,8, 8,9 };
    static const int16_t hshift_t[8]     = { 3, 1, -1, -3, 3, 1, -1, -3 };
    static const int8_t  bshr_t[16]      = { -3,0, -3,0, -3,0, -3,0, -3,0, -3,0, -3,0, -3,0 };
    uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(src), vld1q_u8(pair_shuf_t)));
    x = vshlq_u16(x, vld1q_s16(hshift_t));
    uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), vld1q_s8(bshr_t));
    vst1q_u8(dst, vqtbl2q_u8(c2s_vec, vandq_u8(y, vdupq_n_u8(0x1f))));
}

static inline void pivcoh__d6_chunk16(uint8_t *dst, const uint8_t *src,
                                      uint8x16x4_t c2s_vec)
{
    static const uint8_t pair_shuf_t[16] = { 0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11 };
    static const int16_t hshift_t[8]     = { 2,-2, 2,-2, 2,-2, 2,-2 };
    static const int8_t  bshr_t[16]      = { -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0 };
    uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(src), vld1q_u8(pair_shuf_t)));
    x = vshlq_u16(x, vld1q_s16(hshift_t));
    uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), vld1q_s8(bshr_t));
    vst1q_u8(dst, vqtbl4q_u8(c2s_vec, vandq_u8(y, vdupq_n_u8(0x3f))));
}

/* ---- per-D flat decodes (contiguous output) ---- */

/* D=2 (4 codes/byte): 64/iter maps each input nibble straight to a
 * symbol pair via two prepped tables, interleaved back with vst4q.
 * The tf tail is the same loop run past n (scribbling < 64 bytes past
 * out+n, saved and restored) — one shape, one constant set. */
static void pivcoh__flat_d2(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    int i = 0;
    if (tf || n >= 64) {
        static const uint8_t th_idx[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        uint32_t w; memcpy(&w, c2s, 4);
        const uint8x16_t TL = vreinterpretq_u8_u32(vdupq_n_u32(w));   /* c2s[n&3] */
        const uint8x16_t TH = vqtbl1q_u8(TL, vld1q_u8(th_idx));       /* c2s[(n>>2)&3] */
        const uint8x16_t m  = vdupq_n_u8(0x0F);
        uint8x16_t k0 = vdupq_n_u8(0), k1 = k0, k2 = k0, k3 = k0;
        if (tf) {
            k0 = vld1q_u8(out + n);      k1 = vld1q_u8(out + n + 16);
            k2 = vld1q_u8(out + n + 32); k3 = vld1q_u8(out + n + 48);
        }
        int lim = tf ? n : n - 63;
        for (; i < lim; i += 64) {
            uint8x16_t v  = vld1q_u8(bm + (i >> 2));
            uint8x16_t lo = vandq_u8(v, m), hi = vshrq_n_u8(v, 4);
            uint8x16x4_t o = {{ vqtbl1q_u8(TL, lo), vqtbl1q_u8(TH, lo),
                                vqtbl1q_u8(TL, hi), vqtbl1q_u8(TH, hi) }};
            vst4q_u8(out + i, o);
        }
        if (tf) {
            vst1q_u8(out + n, k0);      vst1q_u8(out + n + 16, k1);
            vst1q_u8(out + n + 32, k2); vst1q_u8(out + n + 48, k3);
            return;
        }
    }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 2, 2)];
}

/* D=3: 32 codes/iter via the D=6 pair-gather (two D=3 codes per byte),
 * split lo/hi and interleave with vst2q.  The tf tail is the same loop
 * run past n (scribbling < 32 bytes past out+n, saved and restored;
 * the tf=0 main-loop bound i+48 <= n keeps its 16-byte loads inside
 * ceil(3n/8) — production guards only against the output and can read
 * 4 bytes past the region). */
static void pivcoh__flat_d3(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    int i = 0;
    if (tf || n >= 48) {
        const uint8x8_t  c2s8  = vld1_u8(c2s);
        const uint8x16_t c2s16 = vcombine_u8(c2s8, c2s8);
        const uint8x16_t m7    = vdupq_n_u8(7);
        uint8x16x2_t c2s32; c2s32.val[0] = c2s16; c2s32.val[1] = c2s16;
        static const uint8_t pair6_shuf_t[16] = { 0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11 };
        static const int16_t hshift6_t[8]     = { 2,-2, 2,-2, 2,-2, 2,-2 };
        static const int8_t  bshr6_t[16]      = { -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0 };
        const uint8x16_t pair6_shuf = vld1q_u8(pair6_shuf_t);
        const int16x8_t  hshift6    = vld1q_s16(hshift6_t);
        const int8x16_t  bshr6      = vld1q_s8(bshr6_t);
        uint8x16_t k0 = vdupq_n_u8(0), k1 = k0;
        if (tf) { k0 = vld1q_u8(out + n); k1 = vld1q_u8(out + n + 16); }
        int lim = tf ? n : n - 47;
        const uint8_t *bp = bm;
        for (; i < lim; i += 32, bp += 12) {
            uint8x16_t packed = vld1q_u8(bp);
            uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(packed, pair6_shuf));
            x = vshlq_u16(x, hshift6);
            uint8x16_t pair6 = vshlq_u8(vreinterpretq_u8_u16(x), bshr6);
            uint8x16x2_t o;
            o.val[0] = vqtbl1q_u8(c2s16, vandq_u8(pair6, m7));
            o.val[1] = vqtbl2q_u8(c2s32, vshrq_n_u8(pair6, 3));
            vst2q_u8(out + i, o);
        }
        if (tf) {
            vst1q_u8(out + n, k0);
            vst1q_u8(out + n + 16, k1);
            return;
        }
    }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 3, 3)];
}

/* D=4: nibbles index c2s directly; 32/iter via vzip.  The tf tail is
 * the same loop run past n (scribbling < 32 bytes past out+n, saved
 * and restored) — one shape, one constant set. */
static void pivcoh__flat_d4(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    const uint8x16_t m = vdupq_n_u8(0x0F);
    uint8x16_t k0 = vdupq_n_u8(0), k1 = k0;
    if (tf) { k0 = vld1q_u8(out + n); k1 = vld1q_u8(out + n + 16); }
    int i = 0, lim = tf ? n : n - 31;
    for (; i < lim; i += 32) {
        uint8x16_t v  = vld1q_u8(bm + (i >> 1));
        uint8x16_t lo = vandq_u8(v, m), hi = vshrq_n_u8(v, 4);
        uint8x16_t a = vqtbl1q_u8(c2s_vec, lo), b = vqtbl1q_u8(c2s_vec, hi);
        vst1q_u8(out + i,      vzip1q_u8(a, b));
        vst1q_u8(out + i + 16, vzip2q_u8(a, b));
    }
    if (tf) {
        vst1q_u8(out + n, k0);
        vst1q_u8(out + n + 16, k1);
        return;
    }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 4, 4)];
}

/* D=5: pair-gather chunks (two adjacent codes per u16 lane, positioned
 * so the byte reinterpret interleaves even/odd for free); vqtbl2
 * scatter.  tf=0 keeps the region-safe block count. */
static void pivcoh__flat_d5(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16x2_t c2s_vec = vld1q_u8_x2(c2s);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);
    int i = 0, lim = tf ? n : (n >= 25 ? ((n - 9) >> 4) << 4 : 0);
    for (; i < lim; i += 16)
        pivcoh__d5_chunk16(out + i, bm + ((i * 5) >> 3), c2s_vec);
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 5, 5)];
}

/* D=6: same pair-gather as D=5 (12-bit pairs); 64-byte c2s => vqtbl4q. */
static void pivcoh__flat_d6(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16x4_t c2s_vec = vld1q_u8_x4(c2s);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);
    int i = 0, lim = tf ? n : (n >= 24 ? ((n - 8) >> 4) << 4 : 0);
    for (; i < lim; i += 16)
        pivcoh__d6_chunk16(out + i, bm + ((i * 6) >> 3), c2s_vec);
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 6, 6)];
}

/* D=7: 128-entry c2s = vqtbl4 low half + vqtbx4 high half (vqtbx keeps
 * the first result for out-of-range lanes — no OR-merge).  16-wide
 * while whole chunks fit, 8-wide finish. */
static void pivcoh__flat_d7(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16x4_t lo = vld1q_u8_x4(c2s), hi = vld1q_u8_x4(c2s + 64);
    uint8x16_t sub64q = vdupq_n_u8(64);
    uint8x8_t  sub64  = vdup_n_u8(64);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);
    int i = 0, fast_end = tf ? n : (n >= 24 ? n - 24 : 0);
    for (; i < fast_end - 15; i += 16) {
        uint8x8_t cl = pivcoh__d7_unpack(bm + ((i      * 7) >> 3));
        uint8x8_t ch = pivcoh__d7_unpack(bm + (((i + 8) * 7) >> 3));
        uint8x16_t codes = vcombine_u8(cl, ch);
        uint8x16_t s = vqtbl4q_u8(lo, codes);
        s = vqtbx4q_u8(s, hi, vsubq_u8(codes, sub64q));
        vst1q_u8(out + i, s);
    }
    for (; i < (tf ? n : fast_end - 7); i += 8) {
        uint8x8_t codes = pivcoh__d7_unpack(bm + ((i * 7) >> 3));
        uint8x8_t s = vqtbl4_u8(lo, codes);
        s = vqtbx4_u8(s, hi, vsub_u8(codes, sub64));
        vst1_u8(out + i, s);
    }
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 7, 7)];
}

/* Dispatcher.  D=8 is a full-alphabet equal-length code: c2s is the
 * identity, so the byte-aligned codes ARE the symbols — memcpy (exact
 * either way; tf is moot). */
static void pivcoh__merge_flat(uint8_t *out, int n, const uint8_t *bm, int D,
                               const uint8_t *c2s, int tf)
{
    switch (D) {
    case 1: pivcoh__merge_cst_cst(bm, n, c2s[0], c2s[1], out, tf); break;
    case 2: pivcoh__flat_d2(out, n, bm, c2s, tf); break;
    case 3: pivcoh__flat_d3(out, n, bm, c2s, tf); break;
    case 4: pivcoh__flat_d4(out, n, bm, c2s, tf); break;
    case 5: pivcoh__flat_d5(out, n, bm, c2s, tf); break;
    case 6: pivcoh__flat_d6(out, n, bm, c2s, tf); break;
    case 7: pivcoh__flat_d7(out, n, bm, c2s, tf); break;
    case 8: memcpy(out, bm, (size_t)n); break;
    default:
        for (int i = 0; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * D, D)];
        break;
    }
}

/* ---- decode tree walk ---- */

/* Read a node's post-order marker + raw bitmap: sets *bm and returns the
 * input pointer advanced past it, or NULL on truncation / a non-raw
 * (FSE) marker. */
static inline const uint8_t *pivcoh__read_bm(const uint8_t **bm, int K,
                                             const uint8_t *p, const uint8_t *end)
{
    size_t nb = (size_t)((K + 7) >> 3);
    if ((size_t)(end - p) < nb + 1) return NULL;
    if (*p++ != 0) return NULL;                /* raw-bitmap marker only (no FSE) */
    *bm = p;
    return p + nb;
}

/* Interior walk: decodes a subtree's K symbols into out[0,K), returning
 * the advanced input pointer or NULL on invalid input.  The wire is in
 * decode order (K_right at node entry, children larger-K first, the
 * marker+bitmap at the node's post-order position, read right at merge
 * time), so the input cursor moves strictly forward, single-touch.
 *
 * Scratch placement is a two-buffer ping-pong (out, tmp):
 *
 *   - the LARGER child decodes IN PLACE into out's tail out[K_small, K):
 *     safe under the merge, whose write cursor cannot overtake its
 *     tail-side read cursor — by the time it writes out[i] it has
 *     consumed at least i - K_small tail bytes (on a bitmap that lies,
 *     the merge's memory use stays inside the arena bound and the r_end
 *     check reports -1; whatever garbage was written is discarded);
 *   - the SMALLER child decodes into tmp[0, K_small), and its own
 *     recursion uses out's still-empty prefix out[0, K_small) as ITS
 *     partner — the pair (tmp, out-prefix) ping-pongs down the
 *     smaller-child spine instead of growing an arena.
 *
 * The caller guarantees tmp capacity floor(K/2): a smaller child is at
 * most floor(K/2), and everything its subtree puts in ITS partner stays
 * inside out[0, K_small), by induction.  A subtree's whole valid-stream
 * footprint is out[0,K) plus at most floor(K/2) partner bytes plus the
 * kernels' read/scribble slack; PIVCOH_DECODE_SCRATCH_SIZE's extra pad
 * absorbs a lying bitmap's bounded pre-detection strays (kernel section
 * comment).  A FLAT node never touches tmp (so a flat root runs
 * scratch-free). */
static const uint8_t *pivcoh__dec(const pivcoh_table *t, int idx, int K,
                                  uint8_t *out, uint8_t *tmp,
                                  const uint8_t *p, const uint8_t *end)
{
    const pivcoh__rec *rec = &t->sched[idx];
    int kind = rec->kd & 3;

    if (kind == PIVCOH__FLAT) {
        int D = rec->kd >> 2;
        size_t nb = ((size_t)K * (size_t)D + 7) >> 3;
        if ((size_t)(end - p) < nb) return NULL;
        /* tf whenever the stream holds 16 readable bytes past the
         * region — true for essentially every interior flat (the walk's
         * out is always arena-backed; a parent's bitmap alone covers it
         * once K is non-tiny, since the root's bitmap ends the block). */
        pivcoh__merge_flat(out, K, p, D, t->rank_to_sym + rec->param,
                           (size_t)(end - p) >= nb + 16);
        return p + nb;
    }
    if (end - p < 2) return NULL;
    int KR = p[0] | p[1] << 8;
    if (KR > K) return NULL;
    p += 2;
    int KL = K - KR;

    if (kind == PIVCOH__LEAFL) {               /* left = lone leaf */
        uint8_t *rbuf = out + KL;              /* the one child, in place */
        if (KR > 0 && !(p = pivcoh__dec(t, idx + rec->right, KR, rbuf, tmp, p, end)))
            return NULL;
        const uint8_t *bm;
        if (!(p = pivcoh__read_bm(&bm, K, p, end))) return NULL;
        if (pivcoh__merge_cst_vec(bm, K, t->rank_to_sym[rec->param], rbuf, KR, out))
            return NULL;
        return p;
    }
    /* FULL: both children internal, larger first on the wire. */
    if (KR > KL) {
        if (!(p = pivcoh__dec(t, idx + rec->right, KR, out + KL, tmp, p, end)))
            return NULL;
        if (KL > 0 && !(p = pivcoh__dec(t, idx + 1, KL, tmp, out, p, end)))
            return NULL;
        const uint8_t *bm;
        if (!(p = pivcoh__read_bm(&bm, K, p, end))) return NULL;
        if (pivcoh__merge_vec_vec(bm, K, tmp, KL, out + KL, KR, out)) return NULL;
    } else {
        if (!(p = pivcoh__dec(t, idx + 1, KL, out + KR, tmp, p, end)))
            return NULL;
        if (KR > 0 && !(p = pivcoh__dec(t, idx + rec->right, KR, tmp, out, p, end)))
            return NULL;
        const uint8_t *bm;
        if (!(p = pivcoh__read_bm(&bm, K, p, end))) return NULL;
        if (pivcoh__merge_vec_vec(bm, K, out + KR, KL, tmp, KR, out)) return NULL;
    }
    return p;
}

PIVCOHDEF ptrdiff_t pivcoh_decode(const pivcoh_table *t,
                                  const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap,
                                  size_t *consumed, void *scratch)
{
    if (!t || !in || !out || !t->num_ranks || in_len < 2) return -1;
    int N = in[0] | in[1] << 8;
    if (N < 1 || (size_t)N > out_cap) return -1;
    const uint8_t *end = in + in_len;
    const uint8_t *p = in + 2;
    const pivcoh__rec *root = &t->sched[0];
    int kind = root->kd & 3;

    if (kind == PIVCOH__FLAT) {                /* flat root: no scratch at all */
        int D = root->kd >> 2;
        size_t nb = ((size_t)N * (size_t)D + 7) >> 3;
        if ((size_t)(end - p) < nb) return -1;
        /* tf=0: out is the caller's buffer (no scribble allowed), and a
         * flat root's region really can end the stream. */
        pivcoh__merge_flat(out, N, p, D, t->rank_to_sym + root->param, 0);
        if (consumed) *consumed = (size_t)(p + nb - in);
        return N;
    }

    /* The interior walk's merges save/restore-scribble past out+K and
     * read their sources with 16 bytes of slack — guarantees the
     * caller's `out` doesn't offer.  So the root's children decode into
     * the arena, and only the root's own merge, whose writes are exact
     * (the EXACT `_x` kernels), targets `out`. */
    pivcoh__init_dec();
    if (end - p < 2) return -1;
    int KR = p[0] | p[1] << 8;
    if (KR > N) return -1;
    p += 2;
    int KL = N - KR;
    uint8_t *sc = scratch ? (uint8_t *)scratch
                          : (uint8_t *)malloc(PIVCOH_DECODE_SCRATCH_SIZE(N));
    if (!sc) return -1;
    const uint8_t *bm;

    if (kind == PIVCOH__LEAFL) {
        /* One internal child: it decodes at the arena base with the
         * space after it as ping-pong partner. */
        if (KR > 0) p = pivcoh__dec(t, root->right, KR, sc, sc + KR, p, end);
        if (p && (p = pivcoh__read_bm(&bm, N, p, end)) != NULL &&
            pivcoh__merge_cst_vec_x(bm, N, t->rank_to_sym[root->param], sc, KR, out))
            p = NULL;
    } else {
        /* FULL root, hybrid hole-reuse: both children decode into the
         * arena's first N bytes, [larger | smaller].  The larger child
         * (first on the wire) borrows the smaller sibling's still-empty
         * slot as its partner — spilling past N only when a spine
         * smaller-child outgrows it — and the smaller child follows
         * with a fresh partner beyond N. */
        uint8_t *lbuf, *rbuf;
        if (KR > KL) {
            rbuf = sc; lbuf = sc + KR;
            p = pivcoh__dec(t, root->right, KR, rbuf, lbuf, p, end);
            if (p && KL > 0) p = pivcoh__dec(t, 1, KL, lbuf, sc + N, p, end);
        } else {
            lbuf = sc; rbuf = sc + KL;
            p = pivcoh__dec(t, 1, KL, lbuf, rbuf, p, end);
            if (p && KR > 0) p = pivcoh__dec(t, root->right, KR, rbuf, sc + N, p, end);
        }
        if (p && (p = pivcoh__read_bm(&bm, N, p, end)) != NULL &&
            pivcoh__merge_vec_vec_x(bm, N, lbuf, KL, rbuf, KR, out))
            p = NULL;
    }
    if (!scratch) free(sc);
    if (!p) return -1;
    if (consumed) *consumed = (size_t)(p - in);
    return N;
}

/* ================= encode kernels (ports of primitives_neon) ================ */

/* Per-mask LUTs:
 *   pc8[m]         popcount of mask byte m
 *   ctab8[m][0:8]  right source lanes packed at [0,n_right), 0xff fill
 *                  (vtbl1 returns 0 for out-of-range indices)
 * p16rev partition LUTs (part_full): one combined index per 16-lane group
 * packs {left, forward, front} | {right, reversed, back}; left+right tile
 * the 16 lanes so the OR of the two disjoint-support tables is exact.
 *   ptabA[m0]      low byte: left -> front, right -> back reversed
 *   ptabB0[m1]     high byte, pc0=0 layout; pc0>0 is the same table
 *                  loaded at byte offset pc0 (padded to 32 B/entry). */
static uint8_t pivcoh__pc8[256];
static uint8_t pivcoh__ctab8[256][16]   __attribute__((aligned(16)));
static uint8_t pivcoh__ptabA[256][16]   __attribute__((aligned(16)));
static uint8_t pivcoh__ptabB0[256][32]  __attribute__((aligned(32)));

static void pivcoh__init_enc(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
        pivcoh__pc8[m] = (uint8_t)__builtin_popcount(m);
        memset(pivcoh__ctab8[m], 0xff, 16);
        int qr = 0, ql = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) pivcoh__ctab8[m][qr++]     = (uint8_t)k;
            else              pivcoh__ctab8[m][8 + ql++] = (uint8_t)k;
        }
    }
    for (int m0 = 0; m0 < 256; m0++) {
        memset(pivcoh__ptabA[m0], 0, 16);
        int lp = 0, rp = 15;
        for (int k = 0; k < 8; k++) {
            if ((m0 >> k) & 1) pivcoh__ptabA[m0][rp--] = (uint8_t)k;
            else               pivcoh__ptabA[m0][lp++] = (uint8_t)k;
        }
    }
    for (int m1 = 0; m1 < 256; m1++) {
        memset(pivcoh__ptabB0[m1], 0, 32);
        int lp = 8, rp = 15;   /* pc0 = 0 layout; pc0 > 0 via the load offset */
        for (int k = 0; k < 8; k++) {
            if ((m1 >> k) & 1) pivcoh__ptabB0[m1][rp--] = (uint8_t)(8 + k);
            else               pivcoh__ptabB0[m1][lp++] = (uint8_t)(8 + k);
        }
    }
    built = 1;
}

static const uint8_t pivcoh__bw8[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* 8-bit mask of (ids > thr) over 8 ranks. */
static inline uint8_t pivcoh__nmask8(uint8x8_t ids, uint8x8_t thr)
{
    return vaddv_u8(vand_u8(vcgt_u8(ids, thr), vld1_u8(pivcoh__bw8)));
}

/* 8 partition mask bytes for 64 ranks, packed LE into a u64 via one
 * vpaddq reduction tree (each lane holds its bit-weight after the
 * vcgt+and, so 4 pairwise adds collapse every chunk to one byte). */
static inline uint64_t pivcoh__masks64(uint8x16_t v0, uint8x16_t v1,
                                       uint8x16_t v2, uint8x16_t v3,
                                       uint8x16_t vt, uint8x16_t bw)
{
    uint8x16_t w0 = vandq_u8(vcgtq_u8(v0, vt), bw);
    uint8x16_t w1 = vandq_u8(vcgtq_u8(v1, vt), bw);
    uint8x16_t w2 = vandq_u8(vcgtq_u8(v2, vt), bw);
    uint8x16_t w3 = vandq_u8(vcgtq_u8(v3, vt), bw);
    uint8x16_t t0 = vpaddq_u8(w0, w1);
    uint8x16_t t1 = vpaddq_u8(w2, w3);
    uint8x16_t u0 = vpaddq_u8(t0, t1);
    uint8x16_t r  = vpaddq_u8(u0, u0);
    return vget_lane_u64(vreinterpret_u64_u8(vget_low_u8(r)), 0);
}

/* ranks[i] = s2r[sym[i]]: the s2r table lives in 16 NEON regs; each
 * 16-lane input does one vqtbl4 + three vqtbx4, with 4 extra GPR
 * gathers interleaved into the idle scalar slots (20 sym/iter). */
static void pivcoh__enc_init(uint8_t *ranks, int n,
                             const uint8_t *sym, const uint8_t *s2r)
{
    int i = 0;
    if (n >= 20) {
        uint8x16x4_t t0 = vld1q_u8_x4(s2r), t1 = vld1q_u8_x4(s2r + 64);
        uint8x16x4_t t2 = vld1q_u8_x4(s2r + 128), t3 = vld1q_u8_x4(s2r + 192);
        const uint8x16_t s64  = vdupq_n_u8(64);
        const uint8x16_t s128 = vdupq_n_u8(128);
        const uint8x16_t s192 = vdupq_n_u8(192);
        for (; i + 20 <= n; i += 20) {
            uint8x16_t c = vld1q_u8(sym + i);
            uint32_t a; memcpy(&a, sym + i + 16, 4);
            uint8x16_t r = vqtbl4q_u8(t0, c);
            unsigned r0 = s2r[(uint8_t)a];
            r = vqtbx4q_u8(r, t1, vsubq_u8(c, s64));
            unsigned r1 = s2r[(uint8_t)(a >> 8)];
            r = vqtbx4q_u8(r, t2, vsubq_u8(c, s128));
            unsigned r2 = s2r[(uint8_t)(a >> 16)];
            r = vqtbx4q_u8(r, t3, vsubq_u8(c, s192));
            unsigned r3 = s2r[(uint8_t)(a >> 24)];
            vst1q_u8(ranks + i, r);
            uint32_t h = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
            memcpy(ranks + i + 16, &h, 4);
        }
    }
    for (; i < n; i++) ranks[i] = s2r[sym[i]];
}

/* Full partition: bitmap + both sides compacted (left in place in ranks,
 * right into tmp).  Per 16-lane group, ONE combined shuffle index (the
 * OR of ptabA[m0] | ptabB0[m1]+pc0) yields both sides at once: the
 * register IS the left output, and a loop-invariant full-reverse vqtbl1
 * recovers the right.  Tail stores overstore up to 16 B past the valid
 * counts — absorbed by the ranks/tmp slack (see pivcoh_encode). */
static int pivcoh__part_full(uint8_t *ranks, int n, uint8_t thr,
                             uint8_t *bm, uint8_t *tmp)
{
    int n_left = 0, n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = pivcoh__masks64(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                           vcnt_u8(vcreate_u8(mask_word))), 0);
        uint64_t pfx = pcw * 0x0101010101010101ULL;
        uint8x16_t vg[4] = { v0, v1, v2, v3 };
#define PIVCOH__PART(g) do {                                                 \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t ri = vorrq_u8(vld1q_u8(pivcoh__ptabA[m0]),               \
                                 vld1q_u8(&pivcoh__ptabB0[m1][pc0]));       \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                            \
        vst1q_u8(ranks + n_left + (16*(g) - cr), comb);                     \
        vst1q_u8(tmp + n_right + cr, vqtbl1q_u8(comb, rev16));              \
    } while (0)
        PIVCOH__PART(0); PIVCOH__PART(1); PIVCOH__PART(2); PIVCOH__PART(3);
#undef PIVCOH__PART
        uint32_t total_r = (uint32_t)(pfx >> 56);
        n_right += (int)total_r;
        n_left  += 64 - (int)total_r;
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7)); tmp[n_right++] = r; }
        else         { ranks[n_left++] = r; }
    }
    return n_right;
}

/* One-sided (right/none) partition: bitmap always, right side compacted
 * into tmp when EMIT_RIGHT (the left side of a LEAF_LEFT node is dead).
 * EMIT_RIGHT=0 folds to a pure bitmap build (the D=1 flat pack). */
__attribute__((always_inline)) static inline
int pivcoh__part_core(uint8_t *ranks, int n, uint8_t thr,
                      uint8_t *bm, uint8_t *tmp, int EMIT_RIGHT)
{
    int n_right = 0;
    int j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    uint8x8_t  vt8 = vdup_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t mask_word = pivcoh__masks64(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &mask_word, 8);
        uint64_t pc_word = vget_lane_u64(vreinterpret_u64_u8(
                               vcnt_u8(vcreate_u8(mask_word))), 0);
        uint64_t pfx = pc_word * 0x0101010101010101ULL;
        uint8x8_t cv[8] = {
            vget_low_u8(v0), vget_high_u8(v0),
            vget_low_u8(v1), vget_high_u8(v1),
            vget_low_u8(v2), vget_high_u8(v2),
            vget_low_u8(v3), vget_high_u8(v3),
        };
#define PIVCOH__PART1(K_) do {                                               \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF); \
        if (EMIT_RIGHT) {                                                    \
            const uint8_t *tab = pivcoh__ctab8[(uint8_t)(mask_word >> (8*(K_)))]; \
            vst1_u8(tmp + n_right + cr, vtbl1_u8(cv[K_], vld1_u8(tab)));     \
        }                                                                    \
    } while (0)
        PIVCOH__PART1(0); PIVCOH__PART1(1); PIVCOH__PART1(2); PIVCOH__PART1(3);
        PIVCOH__PART1(4); PIVCOH__PART1(5); PIVCOH__PART1(6); PIVCOH__PART1(7);
#undef PIVCOH__PART1
        n_right += (int)(uint32_t)(pfx >> 56);
    }
    for (; j + 8 <= n; j += 8) {
        uint8x8_t v = vld1_u8(ranks + j);
        uint8_t mask = pivcoh__nmask8(v, vt8);
        bm[j >> 3] = mask;
        if (EMIT_RIGHT) vst1_u8(tmp + n_right, vtbl1_u8(v, vld1_u8(pivcoh__ctab8[mask])));
        n_right += pivcoh__pc8[mask];
    }
    for (; j < n; j++) {
        if ((j & 7) == 0) bm[j >> 3] = 0;
        uint8_t r = ranks[j];
        if (r > thr) { bm[j >> 3] |= (uint8_t)(1u << (j & 7));
                       if (EMIT_RIGHT) tmp[n_right] = r; n_right++; }
    }
    return n_right;
}

/* ---- flat pack: (rank - base) is already the D-bit local code ---- */

/* D=2: 16 ranks -> 4 bytes. */
static inline int pivcoh__pack_d2(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d2[16] = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    uint8x16_t vb = vdupq_n_u8(base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t b = vsubq_u8(vld1q_u8(ranks + i), vb);
        b = vshlq_u8(b, vld1q_s8(shifts_d2));
        uint8x16_t s1 = vpaddq_u8(b, b);
        uint8x16_t s2 = vpaddq_u8(s1, s1);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i * 2 / 8), &packed4, 4);
    }
    return i;
}

/* D=3: 8 ranks -> 24 bits, u32 horizontal accumulator. */
static inline int pivcoh__pack_d3(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int32_t shifts_lo[4] = { 0, 3, 6, 9 };
    static const int32_t shifts_hi[4] = { 12, 15, 18, 21 };
    uint8x8_t vb = vdup_n_u8(base);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vmovl_u8(vsub_u8(vld1_u8(ranks + i), vb));
        uint32x4_t lo = vshlq_u32(vmovl_u16(vget_low_u16(v)),  vld1q_s32(shifts_lo));
        uint32x4_t hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), vld1q_s32(shifts_hi));
        uint32_t packed = vaddvq_u32(vaddq_u32(lo, hi));
        int bi = i * 3 / 8;
        out[bi]     = (uint8_t)(packed        & 0xff);
        out[bi + 1] = (uint8_t)((packed >> 8 ) & 0xff);
        out[bi + 2] = (uint8_t)((packed >> 16) & 0xff);
    }
    return i;
}

/* D=4: 16 ranks -> 8 bytes. */
static inline int pivcoh__pack_d4(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d4[16] = { 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4 };
    uint8x16_t vb = vdupq_n_u8(base);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t b = vsubq_u8(vld1q_u8(ranks + i), vb);
        b = vshlq_u8(b, vld1q_s8(shifts_d4));
        vst1_u8(out + (i * 4 / 8), vget_low_u8(vpaddq_u8(b, b)));
    }
    return i;
}

/* D=5/6/7: shift-insert pack, 16 codes/iter.  A u16 lane of the code
 * bytes ALREADY holds code[2i] | code[2i+1] << 8, so pairing is just
 * re-basing the high half from << 8 to << D: one USHR + one SLI
 * (shift-left-insert keeps the low code's bits [0, D)), repeated at
 * u32 (<< 16 -> << 2D) and u64 (<< 32 -> << 4D) width — 6 ops, no
 * constants beyond the compact shuffle.  This replaces the ryg
 * multiply-as-shift pyramid (vmull with {1, 2^D} multipliers +
 * vpaddq): the multiply only existed because ARM's per-lane dynamic
 * shift has no widening form, and no widening is actually needed.
 * Each 16-byte store carries 16-2D trailing junk bytes, overwritten by
 * the next iter / next record (the caller's out_cap >=
 * PIVCOH_ENCODE_BOUND keeps even the last one in bounds). */
static const uint8_t pivcoh__pack_compact_d5[16] = {
    0, 1, 2, 3, 4,   8, 9, 10, 11, 12,  0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivcoh__pack_compact_d6[16] = {
    0, 1, 2, 3, 4, 5,   8, 9, 10, 11, 12, 13,  0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivcoh__pack_compact_d7[16] = {
    0, 1, 2, 3, 4, 5, 6,   8, 9, 10, 11, 12, 13, 14,  0xff, 0xff
};

#define PIVCOH__PACK_DN(NAME, D_VAL, COMPACT_TAB)                                \
static inline int NAME(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)  \
{                                                                                \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                            \
    int i = 0;                                                                   \
    for (; i + 16 <= n; i += 16) {                                               \
        uint8x16_t cb = vsubq_u8(vld1q_u8(ranks + i), vdupq_n_u8(base));         \
        uint16x8_t w16 = vreinterpretq_u16_u8(cb);                               \
        w16 = vsliq_n_u16(w16, vshrq_n_u16(w16, 8), D_VAL);                      \
        uint32x4_t w32 = vreinterpretq_u32_u16(w16);                             \
        w32 = vsliq_n_u32(w32, vshrq_n_u32(w32, 16), 2 * (D_VAL));               \
        uint64x2_t w64 = vreinterpretq_u64_u32(w32);                             \
        w64 = vsliq_n_u64(w64, vshrq_n_u64(w64, 32), 4 * (D_VAL));               \
        uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact);      \
        vst1q_u8(out + ((i * (D_VAL)) >> 3), packed);                            \
    }                                                                            \
    return i;                                                                    \
}
PIVCOH__PACK_DN(pivcoh__pack_d5, 5, pivcoh__pack_compact_d5)
PIVCOH__PACK_DN(pivcoh__pack_d6, 6, pivcoh__pack_compact_d6)
PIVCOH__PACK_DN(pivcoh__pack_d7, 7, pivcoh__pack_compact_d7)
#undef PIVCOH__PACK_DN

/* D=8: byte-aligned. */
static inline int pivcoh__pack_d8(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    uint8x16_t vb = vdupq_n_u8(base);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        vst1q_u8(out + i, vsubq_u8(vld1q_u8(ranks + i), vb));
    return i;
}

/* Dispatcher: SIMD per-D path + scalar tail (packs (rank - base) LSB-first). */
static void pivcoh__pack_dN(uint8_t *out, const uint8_t *ranks,
                            int n, int D, uint8_t base)
{
    int total_bytes = (n * D + 7) >> 3;
    if (total_bytes > 0) out[total_bytes - 1] = 0;

    int i = 0;
    switch (D) {
    case 1: /* the D=1 bit IS the partition bit: reuse the bitmap build
             * (EMIT_RIGHT=0 never writes ranks; the cast is sound) */
            (void)pivcoh__part_core((uint8_t *)(uintptr_t)ranks, n, base,
                                    out, NULL, 0);
            return;
    case 2: i = pivcoh__pack_d2(out, ranks, n, base); break;
    case 3: i = pivcoh__pack_d3(out, ranks, n, base); break;
    case 4: i = pivcoh__pack_d4(out, ranks, n, base); break;
    case 5: i = pivcoh__pack_d5(out, ranks, n, base); break;
    case 6: i = pivcoh__pack_d6(out, ranks, n, base); break;
    case 7: i = pivcoh__pack_d7(out, ranks, n, base); break;
    case 8: i = pivcoh__pack_d8(out, ranks, n, base); break;
    default: break;
    }
    if (i >= n) return;

    int bit_pos = i * D;
    int byte_idx = bit_pos >> 3;
    int bits_in_buf = bit_pos & 7;
    uint64_t buf = bits_in_buf > 0
        ? (uint64_t)out[byte_idx] & ((1u << bits_in_buf) - 1)
        : 0;
    for (; i < n; i++) {
        uint32_t local = (uint32_t)(uint8_t)(ranks[i] - base);
        buf |= (uint64_t)local << bits_in_buf;
        bits_in_buf += D;
        while (bits_in_buf >= 8) {
            out[byte_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if (bits_in_buf > 0) out[byte_idx] = (uint8_t)(buf & ((1u << bits_in_buf) - 1));
}

/* ---- encode tree walk ----
 * Production placement: partition leaves the left half in place in
 * `ranks` and compacts the right half into `tmp`; recursion descends
 * left on ranks, right on tmp, both children using tmp + n_right as
 * their scratch.  Per level tmp grows by n_right <= n, depth <= 10,
 * plus the partition's 16 B overstore — inside PIVCOH_SCRATCH_SIZE. */
static void pivcoh__enc_node(const pivcoh_table *t, int idx,
                             uint8_t *ranks, int n,
                             uint8_t **pp, uint8_t *tmp)
{
    const pivcoh__rec *rec = &t->sched[idx];
    int kind = rec->kd & 3;
    uint8_t *p = *pp;

    if (kind == PIVCOH__FLAT) {                /* n local codes, D bits each */
        int D = rec->kd >> 2;
        pivcoh__pack_dN(p, ranks, n, D, rec->param);
        *pp = p + ((n * D + 7) >> 3);
        return;
    }
    /* Decode-order record (wire v0.7): the K_right header goes at the
     * node's PRE-order position, the marker+bitmap at its POST-order
     * position, the children's regions between, larger-K child first.
     * The bitmap is staged on the stack across the child recursion (its
     * stream position depends on the children's encoded sizes). */
    int nbytes = (n + 7) >> 3;
    uint8_t bm_stage[(size_t)nbytes];
    int n_right = (kind == PIVCOH__LEAFL)
        ? pivcoh__part_core(ranks, n, rec->param, bm_stage, tmp, 1)
        : pivcoh__part_full(ranks, n, rec->param, bm_stage, tmp);
    int n_left = n - n_right;
    *p++ = (uint8_t)n_right;                   /* K_right, u16 LE */
    *p++ = (uint8_t)(n_right >> 8);
    *pp = p;
    if (kind == PIVCOH__FULL && n_right > n_left) {
        pivcoh__enc_node(t, idx + rec->right, tmp, n_right, pp, tmp + n_right);
        if (n_left > 0)
            pivcoh__enc_node(t, idx + 1, ranks, n_left, pp, tmp + n_right);
    } else {
        if (kind == PIVCOH__FULL && n_left > 0)
            pivcoh__enc_node(t, idx + 1, ranks, n_left, pp, tmp + n_right);
        if (n_right > 0)
            pivcoh__enc_node(t, idx + rec->right, tmp, n_right, pp, tmp + n_right);
    }
    p = *pp;
    *p++ = 0;                                  /* marker: raw bitmap */
    memcpy(p, bm_stage, (size_t)nbytes);
    *pp = p + nbytes;
}

PIVCOHDEF ptrdiff_t pivcoh_encode(const pivcoh_table *t,
                                  const uint8_t *in, size_t n,
                                  uint8_t *out, size_t out_cap, void *scratch)
{
    if (!t || !in || !out || n < 1 || n > 65535 || !t->num_ranks) return -1;
    if (out_cap < PIVCOH_ENCODE_BOUND(n)) return -1;
    pivcoh__init_enc();
    uint8_t *sc = scratch ? (uint8_t *)scratch : (uint8_t *)malloc(PIVCOH_SCRATCH_SIZE(n));
    if (!sc) return -1;
    uint8_t *ranks = sc, *tmp = sc + n + 64;   /* +64: partition tail overstore */
    pivcoh__enc_init(ranks, (int)n, in, t->sym_to_rank);
    uint8_t *p = out;
    *p++ = (uint8_t)n;
    *p++ = (uint8_t)(n >> 8);
    pivcoh__enc_node(t, 0, ranks, (int)n, &p, tmp);
    if (!scratch) free(sc);
    return p - out;
}

#endif /* PIVCOH_IMPLEMENTATION */

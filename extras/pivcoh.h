/* pivcoh.h - v3.0 - single-file PIVCO-Huffman block codec, NEON edition
 *
 * A minimal, allocation-free-capable, VLA-free implementation of the
 * PIVCO-Huffman wire format (https://github.com/MarcinZukowski/pivco-huffman).
 * This edition is aarch64-only: it began as straight ports of the
 * production library's NEON kernels (SABD two-table merge, per-D flat
 * decode, p16rev partition) and has since evolved past them — every
 * kernel is fully tail-free (whole vectors end to end; the only scalar
 * remnants are the root merge's exact stores into the caller's buffer
 * and the flat-root extractor), decode ping-pongs two buffers instead
 * of growing an arena, and the flat pack runs a converging-shift
 * pyramid.  Encode and decode meet or beat the production codec on
 * every measured distribution.  ~24 KiB of static shuffle tables build
 * lazily on first use — the writes are idempotent, so concurrent first
 * calls are benign.
 * Streams are byte-identical to the full library's PH-only mode (wire
 * v0.7 decode-order layout; raw bitmaps —
 * pivco_huffman_set_fse_enabled(0), which is also what the pivcohuf
 * tool's default non-ANS format uses; "optimized" tree shaping, max
 * code length 11) for histograms totalling < 4 GiB — see
 * pivcoh_table_from_freqs.  FSE/ANS-coded blocks are not supported and
 * are rejected on decode.
 *
 * Do this in ONE C file to create the implementation:
 *     #define PIVCOH_IMPLEMENTATION
 *     #include "pivcoh.h"
 *
 * Usage — encoder side:
 *     pivcoh_table t;
 *     if (!pivcoh_table_from_freqs(&t, freq)) ...;   // freq: uint64_t[256]
 *     // or _joint(&t, freq, &j, NULL) to trade a guarded sliver of
 *     // compressed size for a much faster-to-decode tree shape
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
 * const tables are shareable across threads (decode never writes them;
 * the FIRST encode on a table completes its encoder view in place with
 * idempotent writes, so concurrent first encodes are also benign).
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
 * encoder dominates: a ranks buffer (n + 64) plus, per recursion level
 * (max 11), one staged bitmap, one compacted right-half and a 64-byte
 * overshoot gap (~9n/8 + 73 per level) for the tail-free partition's
 * scatter, which strays up to 63 bytes past a ranks region. */
#define PIVCOH_SCRATCH_SIZE(n)  (14 * (size_t)(n) + 1024)

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

/* ---- optional joint length/shape optimization (encoder side) ----
 *
 * pivcoh_table_from_freqs picks lengths that minimize compressed bits.
 * pivcoh_table_from_freqs_joint additionally bends them — at an
 * explicitly priced, guard-bounded cost in bits — so the class counts
 * land on round binary numbers and the decoder's counts-to-chunks rule
 * yields fewer, larger flat blocks and fewer merge passes.  Measured on
 * windowed LZ-literal workloads this buys +25%..+130% decode speed at
 * a compression delta within ±0.3 pp (better at small windows).  The
 * wire carries only lengths, so ANY decoder reads the output.
 * Port of the production joint optimizer (joint-cost-model @ c6073f5). */
typedef struct {
    float lambda;      /* bits one merge pass is worth; <= 0 disables the
                          pass entirely (plain Huffman lengths) */
    int   gran;        /* solve tier: 0 auto (exact DP to 64 symbols, then
                          grouped; always ~<= 10 us), 1 exact DP (~100 us
                          worst case), 2/4/8 fixed grouping, -1 greedy
                          nudger (~2 us, no DP, about half the win).
                          Other values behave as 0. */
    float guard_bits;  /* adopt only if modeled bits <= guard_bits * baseline */
    float guard_time;  /* ... and modeled decode time <= guard_time * baseline;
                          otherwise the plain Huffman lengths are kept */
    float gamma;       /* fixed decode cost per schedule record per block, in
                          merge element-pass units (dispatch + wire header).
                          Blocks are modeled at 16K symbols; gamma and block
                          size enter the cost only as gamma/block, so scale
                          gamma if your decode granularity differs */
    float kappa[9];    /* flat-kernel decode cost per symbol at depth b, in
                          merge-pass units; all-zero models kernels free */
    float mu_cst;      /* lone-leaf merge cost relative to a full merge */
    float prefill;     /* fraction of the prefilled top-symbol leaf its
                          parent merge skips (pivcoh does not prefill: 0) */
} pivcoh_joint;
#define PIVCOH_JOINT_DEFAULTS \
    { 0.1f, 0, 1.015f, 0.90f, 170.0f, {0}, 1.0f, 0.0f }

/* Scratch for the joint DP tiers: the exact solve at a full 256-symbol
 * alphabet needs a (257 diagonals x 132 cells) plane of f32 costs plus
 * 11 u16 backtrack planes (+8 alignment).  Tiers gran 0/-1 use < 64 KiB
 * of this.  As with encode/decode, scratch may be NULL (malloc). */
#define PIVCOH_JOINT_SCRATCH_SIZE  (257 * 132 * 26 + 8)

typedef struct { uint8_t kd, param, right; } pivcoh__rec;

typedef struct {
    uint8_t code_len[256];   /* per-symbol code length, 0 = absent, max 11.
                                Filled by pivcoh_table_from_freqs; this is
                                what the encoder transmits to the decoder. */
    /* internals */
    uint16_t num_ranks, sched_len;
    uint8_t enc_ready;       /* sym_to_rank valid (filled lazily on first
                                encode; decode never needs it) */
    uint8_t rank_to_sym[256], sym_to_rank[256];
    pivcoh__rec sched[60];   /* Kraft-complete max is 59 records (33 chunks,
                                27 with bit >= 1); +1 so the schedule render's
                                cap check can't fire mid-walk on a maximal
                                table */
} pivcoh_table;

/* Build a table from symbol frequencies (encoder side).  Derives optimal
 * length-limited code lengths into t->code_len.  Frequencies are treated
 * mod 2^32 internally: a histogram totalling >= 4 GiB may derive
 * different — still valid — lengths.  Returns 1, or 0 if every
 * frequency is zero. */
PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256]);

/* As pivcoh_table_from_freqs, plus the joint length/shape pass when
 * j && j->lambda > 0 (start from PIVCOH_JOINT_DEFAULTS).  On any
 * internal reject — the adoption guard, an out-of-contract lambda, or
 * malloc failure with scratch == NULL — the plain Huffman lengths are
 * kept, so the return value means exactly what it does above. */
PIVCOHDEF int pivcoh_table_from_freqs_joint(pivcoh_table *t,
                                            const uint64_t freq[256],
                                            const pivcoh_joint *j,
                                            void *scratch);

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
#include <math.h>
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

    t->enc_ready = 0;        /* sym_to_rank fills lazily on first encode:
                                this builder is also the decode-side table
                                build, which never reads it */
    return 1;
}

typedef struct { uint32_t freq; uint16_t sym; } pivcoh__leaf;

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
static void pivcoh__sort_leaves(pivcoh__leaf *leaf, int n, uint32_t vary)
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
    int shift[4], npass = 0;
    for (int b = 0; b < 32; b += 8)
        if ((vary >> b) & 0xFF) shift[npass++] = b;
    if (npass == 0) return;                    /* all frequencies equal */
    /* u8 bins cannot go wrong at n <= 256: a bin could only reach 256
     * if every leaf shared that byte, but such a plane does not vary
     * and is skipped, so varying-plane bins are <= 255.  A prefix that
     * wraps to 0 is only stored for an empty bin (never indexed), and
     * the final in-scatter increment that wraps is never read again. */
    uint8_t cnt[4][256];
    memset(cnt, 0, (size_t)npass * sizeof(cnt[0]));
    for (i = 0; i < n; i++)                    /* all planes in one pass */
        for (int p = 0; p < npass; p++)
            cnt[p][(leaf[i].freq >> shift[p]) & 0xFF]++;
    pivcoh__leaf tmp[256], *src = leaf, *dst = tmp;
    for (int p = 0; p < npass; p++) {
        unsigned sum = 0;
        for (int k = 0; k < 256; k++) {
            unsigned c = cnt[p][k];
            cnt[p][k] = (uint8_t)sum;
            sum += c;
        }
        for (i = 0; i < n; i++)
            dst[cnt[p][(src[i].freq >> shift[p]) & 0xFF]++] = src[i];
        pivcoh__leaf *t = src; src = dst; dst = t;
    }
    if (src != leaf) memcpy(leaf, src, (size_t)n * sizeof(*leaf));
}

/* ============ joint length/shape optimization (encoder side) ============
 *
 * Port of the production joint_lengths.c (joint-cost-model @ c6073f5).
 * Chunk model: choosing lengths IS choosing at most one chunk per
 * (level L <= 11, flat depth b <= min(8, L)) — a chunk holds 2^b
 * symbols at length L inside a depth-b flat, so each of its symbols'
 * occurrences costs L bits and L - b merge passes.  Objective:
 *     J = sum_s n_s * (L_s + lambda*(L_s - b_s + kappa[b_s]))
 *         + lambda * gamma * blocks * records
 * subject to chunk-root Kraft equality.  For a fixed chunk multiset the
 * optimal symbol assignment deals freq-sorted symbols into cost-sorted
 * chunks (rearrangement inequality), which turns the solve into a DP
 * over (symbols placed, open slots); lambda = 0 degenerates to the
 * Huffman baseline, so the result can only improve in-model, and a
 * kind-aware time model guards against out-of-model regressions.
 *
 * Deliberately not ported: the FSE decode-tax term (pivcoh speaks the
 * raw-bitmap subset — no bitmap is ever FSE-coded) and the ~10 MB
 * mass-DP fallback for lambda > 1/7 — when the slot DP's validity
 * condition fails, the baseline is kept, the same contract as a guard
 * reject. */

/* ---- kind-aware decode-time model (the adoption guard) ----
 *
 * The per-occurrence model above prices every merge alike, but the
 * decoder's merges differ by KIND: a merge with a lone-leaf child uses
 * the cheap cst kernels, a merge of two internal streams pays the full
 * partition.  Tree arrangement is deterministic from the chunk
 * multiset (roots sorted by depth, canonical prefixes), so for <= 33
 * chunks we simulate the skeleton exactly and price each node by kind.
 * Used on BOTH sides of the guard's comparison; the DP keeps its
 * separable search cost (the guard is where mispricing must not
 * survive). */
typedef struct { uint8_t r, D; double W; } pivcoh__jl_ch;

/* Subtree at depth d spanning chunks ch[*i..): consumes them, returns
 * the subtree's decode-time units and its weight; *kind reports what
 * the parent sees (0 = lone leaf, 1 = internal).  pre marks the chunk
 * index holding the prefilled top symbol (-1 = none). */
static double pivcoh__jl_sim(const pivcoh__jl_ch *ch, int n, int *i, int d,
                             int pre, const pivcoh_joint *jp,
                             const double *kap, int *recs,
                             double *Wout, int *kind)
{
    if (d > PIVCOH__MAXLEN) {   /* non-tiling multiset: cut the recursion;
                                 * the caller's i != n check reports -1.
                                 * Unreachable from the in-header callers
                                 * (their multisets are Kraft-exact by
                                 * construction) — pure stack-safety.
                                 * Upstream fix 93b5a7e. */
        *Wout = 0; *kind = 1;
        return 0.0;
    }
    if (*i < n && ch[*i].r == d) {
        const pivcoh__jl_ch *c = &ch[(*i)++];
        *Wout = c->W;
        if (c->D == 0) { *kind = 0; return 0.0; }
        *kind = 1;
        (*recs)++;                                 /* pair/flat record */
        return c->W * kap[c->D];                   /* D=1 pair: kap[1] */
    }
    double Wl = 0, Wr = 0, tl, tr;
    int kl, kr;
    const int il = *i;
    (*recs)++;                                     /* merge record */
    tl = pivcoh__jl_sim(ch, n, i, d + 1, pre, jp, kap, recs, &Wl, &kl);
    tr = pivcoh__jl_sim(ch, n, i, d + 1, pre, jp, kap, recs, &Wr, &kr);
    double W = Wl + Wr;
    double t;
    if (kl == 0 || kr == 0) {
        t = W * jp->mu_cst;                   /* one lone leaf: cst_vec */
        /* prefilled leaf: its side is memset ahead; the merge only
         * moves the internal side */
        if (pre >= 0 && ((kl == 0 && il == pre) ||
                         (kr == 0 && *i - 1 == pre)))
            t -= (kl == 0 ? Wl : Wr) * (double)jp->prefill * jp->mu_cst;
    } else
        t = W;                                /* full partition */
    *Wout = W;
    *kind = 1;
    return t + tl + tr;
}

/* Kind-aware decode time for a chunk list (any order; sorted here by
 * root depth asc, D desc, W desc).  The prefill chunk is the heaviest-
 * per-symbol chunk, if it is a lone leaf — exact under the deal's
 * sorted order. */
static double pivcoh__jl_time(pivcoh__jl_ch *ch, int n,
                              const pivcoh_joint *jp, const double *kap,
                              double total_weight)
{
    int i, j;
    for (i = 1; i < n; i++) {
        pivcoh__jl_ch c = ch[i];
        for (j = i - 1; j >= 0 && (ch[j].r > c.r ||
                 (ch[j].r == c.r && (ch[j].D < c.D ||
                  (ch[j].D == c.D && ch[j].W < c.W)))); j--)
            ch[j + 1] = ch[j];
        ch[j + 1] = c;
    }
    int pre = -1;
    double best = -1;
    for (i = 0; i < n; i++) {
        double per = ch[i].W / (double)(1 << ch[i].D);
        if (per > best) { best = per; pre = ch[i].D == 0 ? i : -1; }
    }
    int ii = 0, kind, recs = 0;
    double W;
    double t = pivcoh__jl_sim(ch, n, &ii, 0, pre, jp, kap, &recs, &W, &kind);
    if (ii != n) return -1.0;   /* malformed multiset (cannot happen) */
    if (jp->gamma > 0) {        /* per-record fixed cost x blocks/window */
        double blocks = ceil(total_weight / 16384.0);
        if (blocks < 1) blocks = 1;
        t += (double)jp->gamma * (double)recs * blocks;
    }
    return t;
}

/* ---- slot-ledger DP (exact for lambda <= 1/7) ----
 *
 * A state is (k symbols placed, s open slots at the current level);
 * Kraft EQUALITY forces s <= sigma - k at every level.  Levels are
 * processed ascending, chunk types within a level in cost order; that
 * equals GLOBAL chunk-cost order — the sorted-matching exactness
 * requirement — iff pivcoh__jl_order's spread bound holds (kappa = 0
 * recovers the classic lambda <= 1/7).  Three structural facts make the
 * walk L1-resident:
 *
 * DIAGONALS.  A take (k, s) -> (k + 2^b, s - 2^b) preserves t = k + s,
 * so within a level the DP decomposes into independent diagonals.
 * Stored diagonal-major, all take sweeps of a level run over one
 * <~0.5 KB row; the plane is traversed once per level (the doubling).
 *
 * PARITY.  Level-entry states have even s (they come from the doubling
 * s' = 2s) and takes with b >= 1 preserve s-parity, so the live lattice
 * is k == t (mod 2): compact index j = (k - (t&1))/2 halves each row.
 * b = 0 — the only parity flip, always last in the level's cost order —
 * is folded into the doubling (an odd-s cell's unique source is its
 * even-lattice predecessor plus one lone leaf) and reconstructed from
 * s-parity at backtrack.  The deepest level never takes b = 0: entry s
 * is even and the terminal needs takes summing to s exactly.
 *
 * CAPACITY BAND.  A state at level L can place at most s * 2^h more
 * symbols (h = levels below), so sigma - k <= (t - k) << h is necessary
 * — and met by every completing trajectory, making the prune exact.
 * Feasibility is preserved cell-to-cell by takes and by the doubling,
 * so pruned — hence stale — cells are never read.
 *
 * Terminal: (k = sigma, s = 0) after the deepest level.  Per-level u16
 * pick rows (bits 1..8; bit 0 is implicit in parity) are archived per
 * diagonal for backtrack. */
#define PIVCOH__JL_WMAX 132     /* max compact row: j <= 128, padded to x4 */

/* Largest compact index j on diagonal t whose k = 2j + (t&1) can still
 * feed sigma - k leaves through (t - k) slots h levels above the
 * bottom; -1 if the whole row is infeasible. */
static inline int pivcoh__jl_jcap(int t, int h, int sigma)
{
    const int p = t & 1;
    int kcap;
    if (h == 0) {
        kcap = t;                     /* t == sigma: all k feasible */
    } else {
        const int num = (t << h) - sigma;
        if (num < 0) return -1;
        kcap = num / ((1 << h) - 1);
        if (kcap > t) kcap = t;
    }
    if (kcap < p) return -1;
    return (kcap - p) >> 1;
}

/* Within-level sweep/deal order under kernel costs.  cost(L, b) =
 * L(1+lam) + g(b) with g(b) = lam*(kap[b] - b): the within-level cost
 * order is L-independent, so one sorted order serves every level.
 * Exactness of the slot DP needs (a) cross-level monotonicity:
 * spread(g) <= 1 + lam (kappa = 0 recovers lam <= 1/7), and (b) b = 0
 * dearest within the level (the parity fold runs it last).  Fills
 * border[0..*nb) with b = 1..bcap by ascending g; returns 1 iff both
 * hold (on 0 the caller keeps the baseline). */
static int pivcoh__jl_order(double lam, const double *kap, int bcap,
                            int border[8], int *nb)
{
    double g[9];
    double gmin = 0, gmax = 0;
    for (int b = 0; b <= bcap; b++) {
        g[b] = lam * (kap[b] - (double)b);
        if (b == 0 || g[b] < gmin) gmin = g[b];
        if (b == 0 || g[b] > gmax) gmax = g[b];
    }
    if (gmax - gmin > (1.0 + lam) * (1.0 - 1e-9)) return 0;
    int n = 0;
    for (int b = 1; b <= bcap; b++) {
        if (g[b] > g[0] + 1e-12) return 0;   /* b0 must stay dearest */
        int i = n++;
        while (i > 0 && (g[border[i - 1]] > g[b]
                         || (g[border[i - 1]] == g[b] && border[i - 1] < b))) {
            border[i] = border[i - 1];
            i--;
        }
        border[i] = b;                       /* ties: larger b first */
    }
    *nb = n;
    return 1;
}

/* lmax/bcap parameterize the level range and flat-depth cap so the
 * same solver runs the exact problem (11, 8) and the 2^G-grouped
 * coarse problem (11-G, 8-G): a group of 2^G sorted symbols at real
 * level L is a depth-G flat, so the coarse problem is this problem
 * shifted by G with an identical cost form.  tc0/tc1: per-take J
 * constants (lambda * gamma * blocks * records added) for b = 0 and
 * b >= 1 takes.  scratch: PIVCOH_JOINT_SCRATCH_SIZE or NULL. */
static double pivcoh__jl_slots(const double *P, int sigma, double lam,
                               int lmax, int bcap, const double *kap,
                               double tc0, double tc1,
                               uint16_t out_BL[PIVCOH__MAXLEN + 1],
                               void *scratch)
{
    int border[8], nb;
    if (!pivcoh__jl_order(lam, kap, bcap, border, &nb))
        return -1.0;
    const int W = (((sigma >> 1) + 2) + 3) & ~3;   /* compact row width */
    const size_t plane = (size_t)(sigma + 1) * (size_t)W;
    uint8_t *own = scratch ? NULL :
        (uint8_t *)malloc(plane * (4 + 2 * (size_t)lmax) + 8);
    if (!own && !scratch) return -1.0;
    float *cost = (float *)(((uintptr_t)(own ? own : (uint8_t *)scratch) + 3)
                            & ~(uintptr_t)3);
    uint16_t *arch = (uint16_t *)(cost + plane);
    float     dPt[2][9][PIVCOH__JL_WMAX];   /* [t&1][b][j]: P[k+2^b]-P[k] */
    float     dP0[257];                     /* P[k] - P[k-1] */

    for (int p = 0; p < 2; p++)
        for (int b = 1; b <= bcap; b++) {
            const int cnk = 1 << b;
            for (int j = 0; j < W; j++) {
                const int k = 2 * j + p;
                dPt[p][b][j] = k + cnk <= sigma
                             ? (float)(P[k + cnk] - P[k]) : 0.0f;
            }
        }
    dP0[0] = 0.0f;
    for (int k = 1; k <= sigma; k++) dP0[k] = (float)(P[k] - P[k - 1]);

    int tlo[PIVCOH__MAXLEN + 1], thi[PIVCOH__MAXLEN + 1];
    for (int L = 1; L <= lmax; L++) {
        const int h = lmax - L;
        thi[L] = (1 << L) > sigma ? sigma : (1 << L);
        tlo[L] = (sigma + (1 << h) - 1) >> h;
        if (tlo[L] < 1) tlo[L] = 1;
    }

    /* Lazy init: every row is fully written by the doubling that
     * produces its level, so only the level-1 band rows need priming. */
    for (int t = tlo[1]; t <= thi[1]; t++)
        for (int j = 0; j < W; j++) cost[(size_t)t * W + j] = INFINITY;
    cost[2 * W + 0] = 0.0f;      /* level-1 entry: k = 0, s = 2, t = 2 */

    for (int L = 1; L <= lmax; L++) {
        const int h = lmax - L;
        const int bmax = L < bcap ? L : bcap;
        uint16_t *archL = arch + (size_t)(L - 1) * plane;
        for (int t = tlo[L]; t <= thi[L]; t++) {
            const int p = t & 1;
            const int jcap = pivcoh__jl_jcap(t, h, sigma);
            if (jcap < 0) continue;
            float *row = cost + (size_t)t * W;
            uint16_t *prow = archL + (size_t)t * W;   /* picks, archived
                                                       * in place */
            memset(prow, 0, (size_t)(jcap + 1) * sizeof(uint16_t));
            for (int oi = 0; oi < nb; oi++) {
                const int b = border[oi];
                if (b > bmax) continue;
                const int jstep = 1 << (b - 1);       /* = 2^b slots / 2 */
                const int jhi = jcap - jstep;         /* dest j <= jcap  */
                if (jhi < 0) continue;
                const float a = (float)((double)L
                                         + lam * ((double)(L - b) + kap[b]));
                const float tc = (float)tc1;
                const float *dpb = dPt[p][b];
                int j = jhi;
                /* 0/1 in-place: dest j + jstep > src j, so iterate j
                 * descending — a written dest is never re-read as a
                 * source for the same chunk.  Stores are unconditional:
                 * everything is L1-resident, so blending beats the
                 * data-dependent branch of an "improved?" early-out. */
                const float32x4_t va = vdupq_n_f32(a);
                const uint16x4_t vbit = vdup_n_u16((uint16_t)(1u << b));
                for (; j >= 3; j -= 4) {
                    const int base = j - 3;
                    float32x4_t src = vld1q_f32(row + base);
                    float32x4_t cand = vaddq_f32(
                        vfmaq_f32(src, vld1q_f32(dpb + base), va),
                        vdupq_n_f32(tc));
                    float32x4_t dst = vld1q_f32(row + base + jstep);
                    uint32x4_t m = vcltq_f32(cand, dst);
                    vst1q_f32(row + base + jstep, vbslq_f32(m, cand, dst));
                    uint16x4_t pm = vmovn_u32(m);
                    uint16x4_t pv = vorr_u16(vld1_u16(prow + base), vbit);
                    uint16x4_t qv = vld1_u16(prow + base + jstep);
                    vst1_u16(prow + base + jstep, vbsl_u16(pm, pv, qv));
                }
                for (; j >= 0; j--) {
                    const float v = row[j];
                    if (!(v < INFINITY)) continue;
                    const float cand = v + a * dpb[j] + tc;
                    if (cand < row[j + jstep]) {
                        row[j + jstep] = cand;
                        prow[j + jstep] = (uint16_t)(prow[j] | (1u << b));
                    }
                }
            }
        }
        if (L == lmax) break;
        /* Doubling s' = 2s with the b = 0 take folded in.  Dest cell
         * (t', k) has the unique source (t = (t'+k)/2, k): even-lattice
         * there if k == t (mod 2), else the odd-s product of a lone
         * leaf taken at level L from (t, k-1).  In place, t' and j'
         * descending: sources live on rows <= t', and the single
         * same-row read (t = t', only at k = t') happens before its
         * cell is overwritten.
         *
         * Branchless: on dest row t' the source diagonal is t = t0 + j'
         * (t0 = (t'+p')/2), so the level-L band check hoists to a
         * j'-range, and the source parity d = (t^k)&1 alternates with
         * j' — two constant-stride subloops with the unified source
         * index (k - d - (t&1))/2.  The subloop containing the top cell
         * runs first (it holds the only same-row read). */
        /* NB the production source declares this constant and then never
         * adds it — its DP under-prices b = 0 takes by the per-record
         * gamma surcharge (the nudge scorer and the guard both charge
         * it).  Fixed here: the fold's take cost carries + tcz. */
        const float a0 = (float)((double)L * (1.0 + lam) + lam * kap[0]);
        const float tcz = (float)tc0;
        for (int tp = thi[L + 1]; tp >= tlo[L + 1]; tp--) {
            const int pp = tp & 1;
            const int jcap2 = pivcoh__jl_jcap(tp, h - 1, sigma);
            if (jcap2 < 0) continue;
            float *nrow = cost + (size_t)tp * W;
            const int t0 = (tp + pp) >> 1;
            int jlo = tlo[L] - t0; if (jlo < 0) jlo = 0;
            int jhi2 = thi[L] - t0; if (jhi2 > jcap2) jhi2 = jcap2;
            for (int jp2 = jcap2; jp2 > jhi2; jp2--) nrow[jp2] = INFINITY;
            for (int jp2 = jlo - 1; jp2 >= 0; jp2--) nrow[jp2] = INFINITY;
            for (int half = 0; half < 2; half++) {
                int jp2 = jhi2 - half;
                if (jp2 < jlo) continue;
                const int k1 = 2 * jp2 + pp;
                const int t1 = t0 + jp2;
                const int d = (t1 ^ k1) & 1;
                const float *src = cost + (size_t)t1 * W
                                 + (size_t)((k1 - d - (t1 & 1)) >> 1);
                if (d == 0) {
                    for (; jp2 >= jlo; jp2 -= 2, src -= 2 * W + 2)
                        nrow[jp2] = *src;
                } else {
                    /* k = 0 has no lone-leaf predecessor: if this
                     * chain reaches cell (jp2 = 0, k = 0), stop above
                     * it and mark it unreachable. */
                    int floor2 = jlo, patch0 = 0;
                    if (pp == 0 && (jp2 & 1) == 0 && jlo == 0) {
                        floor2 = 2;
                        patch0 = 1;
                    }
                    const float *dp0 = dP0 + k1;
                    for (; jp2 >= floor2; jp2 -= 2, src -= 2 * W + 2, dp0 -= 4)
                        nrow[jp2] = *src + a0 * *dp0 + tcz;
                    if (patch0)
                        nrow[0] = INFINITY;
                }
            }
        }
    }

    double J = cost[(size_t)sigma * W + (size_t)((sigma - (sigma & 1)) >> 1)];
    if (J < INFINITY) {
        /* Backtrack: invert each level's transition; odd end-of-level
         * s means the folded b0 was taken there — recover its bits
         * from the even-lattice predecessor and set bit 0. */
        int k = sigma, s = 0;
        for (int L = lmax; L >= 1; L--) {
            const int t = k + s;
            const int p = t & 1;
            const uint16_t *arow = arch + (size_t)(L - 1) * plane
                                        + (size_t)t * W;
            uint16_t BL;
            if ((k ^ t) & 1)
                BL = (uint16_t)(arow[(k - 1 - p) >> 1] | 1u);
            else
                BL = arow[(k - p) >> 1];
            out_BL[L] = BL;
            int cL = 0;
            for (int b = 0; b <= 8; b++) if (BL & (1 << b)) cL += 1 << b;
            k -= cL;
            s += cL;                  /* slots at level L entry (even) */
            if (L > 1) s >>= 1;       /* pre-doubling slots left       */
        }
    } else {
        J = -1.0;
    }
    free(own);
    return J;
}

/* ---- greedy boundary nudger (gran -1): no DP ----
 *
 * The DP acts as a boundary nudger that rounds class counts to few
 * powers of two on a nearly degenerate objective.  Do that directly:
 * one shallow-to-deep walk with the slot ledger, at each level choosing
 * among a handful of low-popcount roundings of the baseline class
 * count inside the feasibility window, scored by the exact chunk cost
 * of this level plus a clamped-baseline rollout of the rest (a
 * one-level lookahead systematically walks into corners).  ~2 us; the
 * adoption guard rejects any bad pick.
 *
 * Feasibility window at level L (h levels below, s open slots, rest
 * symbols unplaced): capacity below needs c <= (s*2^h - rest)/(2^h - 1)
 * (leaves taken now eat slots the remainder needs); completeness
 * (every open slot must eventually host >= 1 leaf) needs c >= 2s - rest.
 * Given the invariants s <= rest <= s*2^h the window is provably
 * nonempty at every level, and any in-window choice preserves them, so
 * the walk cannot die.  At h = 0 it collapses to c = rest = s. */
static inline int pivcoh__jl_clamp(int c, int s, int rest, int h)
{
    int hi = s < rest ? s : rest;
    int lo = 0;
    if (h == 0)
        return rest <= hi ? rest : -1;
    const long cap = (((long)s << h) - rest) / ((1l << h) - 1);
    if (cap < hi) hi = (int)cap;
    const long l2 = 2l * s - rest;
    if (l2 > 0) lo = (int)l2;
    if (lo > hi) return -1;
    return c < lo ? lo : c > hi ? hi : c;
}

/* Exact chunk cost of placing count c at level L on prefix [k, k+c);
 * bits are dealt in the within-level cost order ord[0..nord). */
static inline double pivcoh__jl_ccost(const double *P, int k, int c, int L,
                                      double lam, const double *kap,
                                      const int *ord, int nord,
                                      double tc0, double tc1)
{
    double sc = 0;
    int off = 0;
    for (int oi = 0; oi < nord; oi++) {
        const int b = ord[oi];
        if (c & (1 << b)) {
            sc += (P[k + off + (1 << b)] - P[k + off])
                * ((double)L + lam * ((double)(L - b) + kap[b]))
                + (b == 0 ? tc0 : tc1);
            off += 1 << b;
        }
    }
    return sc;
}

/* Complete the walk from (k, s) at level L0 following the clamped
 * baseline counts; the exact modeled cost of that completion is the
 * lookahead score for candidate choices. */
static double pivcoh__jl_rollout(const double *P, int sigma, double lam,
                                 const double *kap, const int *ord, int nord,
                                 double tc0, double tc1,
                                 const int cls_n[PIVCOH__MAXLEN + 1],
                                 int k, int s, int L0)
{
    double cost = 0;
    for (int L = L0; L <= PIVCOH__MAXLEN; L++) {
        const int c = pivcoh__jl_clamp(cls_n[L], s, sigma - k,
                                       PIVCOH__MAXLEN - L);
        if (c < 0) return INFINITY;
        cost += pivcoh__jl_ccost(P, k, c, L, lam, kap, ord, nord, tc0, tc1);
        k += c;
        s = 2 * (s - c);
    }
    return cost;
}

static double pivcoh__jl_nudge(const double *P, int sigma, double lam,
                               const double *kap, double tc0, double tc1,
                               const int base[PIVCOH__MAXLEN + 1],
                               uint16_t out_BL[PIVCOH__MAXLEN + 1])
{
    /* Within-level deal order under kappa: all b in 0..8 by ascending
     * g(b) = lam*(kap[b] - b) — no validity condition; the nudger is a
     * heuristic and the guard re-scores its output. */
    int ord[9];
    int n = 0;
    for (int b = 0; b <= 8; b++) {
        double gb = lam * (kap[b] - (double)b);
        int i = n++;
        while (i > 0) {
            double gp = lam * (kap[ord[i - 1]] - (double)ord[i - 1]);
            if (gp > gb || (gp == gb && ord[i - 1] < b)) {
                ord[i] = ord[i - 1];
                i--;
            } else break;
        }
        ord[i] = b;
    }
    /* Score with lambda inflated 1.5x: greedy under-commits to
     * flattening relative to the DP, and the guard judges with the
     * REAL lambda anyway, so biasing the search toward flatter shapes
     * raises adoption without risking quality.  (tc0/tc1 stay honest.) */
    lam *= 1.5;
    double total = 0;
    int k = 0, s = 2;
    for (int L = 1; L <= PIVCOH__MAXLEN; L++) {
        const int c0 = pivcoh__jl_clamp(base[L], s, sigma - k,
                                        PIVCOH__MAXLEN - L);
        if (c0 < 0) return -1.0;             /* cannot happen from a
                                              * valid baseline */
        /* candidates: clamped baseline, its 1- and 2-bit
         * down-roundings, the next power of two up, and 0 (kill the
         * level), each re-clamped */
        int cand[5], nc = 0;
        cand[nc++] = c0;
        if (c0 > 0) {
            const int top = 1 << (31 - __builtin_clz((unsigned)c0));
            const int lowmask = c0 & ~top;
            cand[nc++] = top;
            if (lowmask)
                cand[nc++] = top | (1 << (31 - __builtin_clz((unsigned)lowmask)));
            if (top != c0)
                cand[nc++] = top << 1;
            cand[nc++] = 0;
        }
        double bestsc = INFINITY;
        int bestc = c0;
        int prev = -1;
        for (int i = 0; i < nc; i++) {
            int c = pivcoh__jl_clamp(cand[i], s, sigma - k, PIVCOH__MAXLEN - L);
            if (c < 0 || c == prev) continue;
            prev = c;
            const double sc = pivcoh__jl_ccost(P, k, c, L, lam, kap,
                                               ord, n, tc0, tc1)
                + pivcoh__jl_rollout(P, sigma, lam, kap, ord, n, tc0, tc1,
                                     base, k + c, 2 * (s - c), L + 1);
            if (sc < bestsc) { bestsc = sc; bestc = c; }
        }
        out_BL[L] = (uint16_t)bestc;   /* binary decomposition == bits */
        total += pivcoh__jl_ccost(P, k, bestc, L, lam, kap, ord, n, tc0, tc1);
        k += bestc;
        s = 2 * (s - bestc);
    }
    return total;
}

/* Core over the build's leaf array (ascending — reversed in place
 * here; ghost-padding may append).  Overwrites lens[] on adoption;
 * any reject leaves them untouched. */
static int pivcoh__jl_core(pivcoh__leaf *sf, int sigma,
                           const uint64_t freq[256], uint8_t lens[256],
                           const pivcoh_joint *jp, void *scratch)
{
    const double lam = (double)jp->lambda;
    if (sigma < 2) return -1;
    double kap[9];
    for (int b = 0; b <= 8; b++)
        kap[b] = (jp->kappa[b] >= 0.0f && jp->kappa[b] < 100.0f)
               ? (double)jp->kappa[b] : 0.0;
    pivcoh_joint jv = *jp;      /* sanitized model knobs for the guard */
    if (!(jv.mu_cst > 0 && jv.mu_cst < 100)) jv.mu_cst = 1.0f;
    if (!(jv.prefill >= 0 && jv.prefill <= 1)) jv.prefill = 0.0f;
    if (!(jv.gamma >= 0 && jv.gamma < 1e6f)) jv.gamma = 0.0f;
    const double gbits = jp->guard_bits > 0 ? (double)jp->guard_bits : 1.015;
    const double gtime = jp->guard_time > 0 ? (double)jp->guard_time : 0.90;

    for (int i = 0; i < sigma / 2; i++) {  /* ascending -> descending */
        pivcoh__leaf tmp = sf[i];
        sf[i] = sf[sigma - 1 - i];
        sf[sigma - 1 - i] = tmp;
    }
    double P[257];
    P[0] = 0.0;
    for (int i = 0; i < sigma; i++) P[i + 1] = P[i] + (double)sf[i].freq;

    /* Baseline model for the adoption guard: bits + kind-aware decode
     * time of the INCOMING lengths (exchangeable per-class weights,
     * exact canonical skeleton). */
    double base_bits = 0, base_time;
    int    cls_n[PIVCOH__MAXLEN + 1] = {0};
    {
        double cls_w[PIVCOH__MAXLEN + 1] = {0};
        for (int i = 0; i < sigma; i++) {
            int L = lens[sf[i].sym];
            if (L < 1 || L > PIVCOH__MAXLEN) L = PIVCOH__MAXLEN;
            cls_n[L]++; cls_w[L] += (double)sf[i].freq;
        }
        pivcoh__jl_ch pch[40];
        int npc = 0;
        for (int L = 1; L <= PIVCOH__MAXLEN; L++) {
            if (!cls_n[L]) continue;
            base_bits += cls_w[L] * L;
            const double wbar = cls_w[L] / (double)cls_n[L];
            for (int b = 0; b <= 8; b++)
                if (cls_n[L] & (1 << b)) {
                    pch[npc].r = (uint8_t)(L - b);
                    pch[npc].D = (uint8_t)b;
                    pch[npc].W = wbar * (double)(1 << b);
                    npc++;
                }
        }
        base_time = pivcoh__jl_time(pch, npc, &jv, kap, P[sigma]);
        if (base_time < 0) return -1;
    }

    /* Per-take fixed-cost constants: lambda * gamma * blocks, one
     * record for D0 takes (the skeleton merge above the leaf), two for
     * deeper chunks (the flat record + its stitch merge). */
    double blocks = ceil(P[sigma] / 16384.0);
    if (blocks < 1) blocks = 1;
    const double tc0 = lam * (double)jv.gamma * blocks;
    const double tc1 = 2.0 * tc0;

    /* Tier resolve.  Granularity g = 2^G groups the freq-sorted symbols
     * by g and solves the identical problem G levels shallower (a group
     * of g sorted symbols at real level L is a depth-G flat), 4^G fewer
     * states; near-optimal solutions are dense enough that g = 2 loses
     * ~0.13 % of J on average, g = 4 ~0.25 % (measured on lits data),
     * and the guard still rejects any bad case per window.  sigma is
     * ghost-padded to a multiple of g with zero-frequency unused byte
     * values — real leaves the encoder never emits; there are always
     * enough since sigma % g != 0 implies sigma < 256. */
    int gran = jp->gran;
    if (gran != -1 && gran != 1 && gran != 2 && gran != 4 && gran != 8)
        gran = 0;
    if (gran == 0)   /* auto: keep the solve ~<= 10 us at every sigma */
        gran = sigma <= 64 ? 1 : sigma <= 128 ? 2 : 4;
    int obuf[8], on;
    if (gran > 1 && (sigma < 8 * gran
                     || !pivcoh__jl_order(lam,
                                          kap + (gran == 8 ? 3 : gran == 4 ? 2 : 1),
                                          8 - (gran == 8 ? 3 : gran == 4 ? 2 : 1),
                                          obuf, &on)))
        gran = 1;
    const int glog = gran == 8 ? 3 : gran == 4 ? 2 : gran == 2 ? 1 : 0;
    int sigma_pad = sigma;
    if (glog) {
        const int pad = (gran - (sigma % gran)) % gran;
        int added = 0;
        for (int s = 0; s < 256 && added < pad; s++)
            if (!freq[s]) {
                sf[sigma_pad].freq = 0; sf[sigma_pad].sym = (uint16_t)s;
                P[sigma_pad + 1] = P[sigma];
                sigma_pad++; added++;
            }
        if (added < pad) return -1;      /* unreachable: pad <= 256-sigma */
    }

    uint16_t BL[PIVCOH__MAXLEN + 1] = {0};
    if (gran == -1) {
        if (pivcoh__jl_nudge(P, sigma, lam, kap, tc0, tc1, cls_n, BL) < 0)
            return -1;
    } else if (glog) {
        double Pg[130];
        const int sp = sigma_pad / gran;
        for (int i = 0; i <= sp; i++) Pg[i] = P[i * gran];
        uint16_t BLc[PIVCOH__MAXLEN + 1] = {0};
        /* kap + glog: local b' prices the real depth b' + glog; a
         * grouped b' = 0 take is a real 2^glog flat, hence tc1 twice */
        if (pivcoh__jl_slots(Pg, sp, lam, PIVCOH__MAXLEN - glog, 8 - glog,
                             kap + glog, tc1, tc1, BLc, scratch) < 0)
            return -1;
        for (int L = 1; L <= PIVCOH__MAXLEN - glog; L++)
            BL[L + glog] = (uint16_t)(BLc[L] << glog);
    } else if (pivcoh__jl_slots(P, sigma, lam, PIVCOH__MAXLEN, 8, kap,
                                tc0, tc1, BL, scratch) < 0) {
        return -1;      /* order condition failed (lambda > 1/7) or OOM */
    }

    /* Collect the chosen chunks in GLOBAL per-occurrence cost order —
     * under kappa the plain "L ascending, b descending" deal is no
     * longer the cost order, and the sorted matching the solvers assume
     * must be the assignment we actually realize. */
    struct { double cost; uint8_t L, b; uint16_t size; } chunks[40];
    int nchunks = 0;
    for (int L = 1; L <= PIVCOH__MAXLEN; L++)
        for (int b = 0; b <= 8; b++)
            if (BL[L] & (1 << b)) {
                double c = (double)L + lam * ((double)(L - b) + kap[b]);
                int i = nchunks++;
                while (i > 0 && (chunks[i - 1].cost > c
                                 || (chunks[i - 1].cost == c
                                     && (chunks[i - 1].L > L
                                         || (chunks[i - 1].L == L
                                             && chunks[i - 1].b < b))))) {
                    chunks[i] = chunks[i - 1];
                    i--;
                }
                chunks[i].cost = c;
                chunks[i].L    = (uint8_t)L;
                chunks[i].b    = (uint8_t)b;
                chunks[i].size = (uint16_t)(1 << b);
            }

    /* Model the result and apply the adoption guard.  Ghost chunks
     * carry zero weight, so the model scores real symbols exactly. */
    double dp_bits = 0, dp_time;
    {
        pivcoh__jl_ch dch[40];
        int cur = 0;
        for (int i = 0; i < nchunks; i++) {
            const int L = chunks[i].L, b = chunks[i].b;
            double w = P[cur + chunks[i].size] - P[cur];
            dp_bits += w * L;
            dch[i].r = (uint8_t)(L - b);
            dch[i].D = (uint8_t)b;
            dch[i].W = w;
            cur += chunks[i].size;
        }
        if (cur != sigma_pad) return -1;
        dp_time = pivcoh__jl_time(dch, nchunks, &jv, kap, P[sigma]);
        if (dp_time < 0) return -1;
    }
    /* Near-incompressible windows: the RELATIVE bits cap vetoes the
     * store-collapse tree (one full-depth flat: a single record, zero
     * merge passes, ~memcpy decode) exactly where it shines — costing
     * ~+2pp on 97%-ratio data that lambda's objective happily pays.
     * Above 95% baseline ratio the bits cap is waived (exposure is
     * structurally < +5pp: the proposal can't exceed 8 bits/sym) and
     * lambda plus the time cap decide.  Upstream IDEAS 2026-07-12,
     * "guard bits-cap misfires on near-incompressible windows". */
    const int incompressible = base_bits >= 0.95 * 8.0 * P[sigma];
    if (!(dp_time <= gtime * base_time
          && (incompressible || dp_bits <= gbits * base_bits)))
        return -1;

    /* Deal freq-sorted symbols to the chunks in that same order.
     * Ghosts (sorted last) land in the final, dearest chunk: unused
     * byte values receive real codes the encoder never emits. */
    {
        int cur = 0;
        for (int i = 0; i < nchunks; i++)
            for (int j = 0; j < chunks[i].size; j++)
                lens[sf[cur++].sym] = chunks[i].L;
    }
    return 0;
}

static int pivcoh__from_freqs(pivcoh_table *t, const uint64_t freq[256],
                              const pivcoh_joint *jp, void *scratch)
{
    /* Frequencies narrow to u32 (and internal sums wrap mod 2^32): a
     * histogram totalling >= 4 GiB may derive different -- still valid,
     * still Kraft-exact, but no longer production-identical -- code
     * lengths.  Correctness-only: every index below is bounded
     * structurally, never by frequency values. */
    pivcoh__leaf leaf[256];
    uint32_t orv = 0, andv = ~(uint32_t)0;
    int n = 0, i;
    for (i = 0; i < 256; i++)
        if (freq[i]) {
            uint32_t f = (uint32_t)freq[i];
            leaf[n].freq = f;
            leaf[n].sym  = (uint16_t)i;
            n++;
            orv |= f;
            andv &= f;
        }
    if (n == 0) return 0;
    pivcoh__sort_leaves(leaf, n, orv ^ andv);

    uint8_t lens[256] = {0};
    if (n == 1) {
        lens[leaf[0].sym] = 1;
    } else {
        /* van Leeuwen two-queue: sorted leaves + FIFO of made internals */
        uint32_t nf[512];
        int16_t parent[512];
        int li = 0, ih = n, ni = n, rem;
        for (i = 0; i < n; i++) nf[i] = leaf[i].freq;
        for (rem = n; rem > 1; rem--) {
            int a, b;
            if (li < n && (ih == ni || nf[li] <= nf[ih])) a = li++; else a = ih++;
            if (li < n && (ih == ni || nf[li] <= nf[ih])) b = li++; else b = ih++;
            nf[ni] = nf[a] + nf[b];
            parent[a] = parent[b] = (int16_t)ni++;
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
        /* Optional joint length/shape pass (encoder side only; the
         * decoder rebuilds identically from the transmitted lengths).
         * Any internal reject keeps the Huffman lengths above.  leaf[]
         * is free again after the two-queue — the pass reverses it in
         * place and may append zero-frequency ghosts. */
        if (jp && jp->lambda > 0.0f)
            (void)pivcoh__jl_core(leaf, n, freq, lens, jp, scratch);
    }
    return pivcoh_table_from_lens(t, lens);    /* lengths -> schedule (shared) */
}

PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256])
{
    return pivcoh__from_freqs(t, freq, NULL, NULL);
}

PIVCOHDEF int pivcoh_table_from_freqs_joint(pivcoh_table *t,
                                            const uint64_t freq[256],
                                            const pivcoh_joint *j,
                                            void *scratch)
{
    return pivcoh__from_freqs(t, freq, j, scratch);
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
 * The loop is software-pipelined one iteration deep: the carried chain
 * (bitmap load -> vcnt -> lane move -> multiply -> cursor advance) is
 * ~12 cycles of latency against ~16 cycles of work, so each iteration
 * starts the NEXT mask/prefix up front and the chain resolves under
 * the current merges.  The popcount path loads its own copy of the
 * bitmap straight into SIMD (a GPR->SIMD fmov costs a load-port uop on
 * Apple and would sit mid-chain).  Byte k of pfx = sum of the mask's
 * byte-popcounts 0..k: bytes 1/3/5 are the 16-bit chunk boundaries,
 * byte 7 the total.  Store cadence is unchanged — the 128B-unroll
 * shape that won microbenches but lost e2e to streaming effects is
 * deliberately avoided (micro: +7% L1, +4% streaming as-is).
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
#define PIVCOH__PFX8(p) (vget_lane_u64(vreinterpret_u64_u8(                \
                             vcnt_u8(vld1_u8(p))), 0) * 0x0101010101010101ull)
__attribute__((always_inline)) static inline
int pivcoh__mvv(const uint8_t *bm, int K, const uint8_t *l, int KL,
                const uint8_t *r, int KR, uint8_t *out, int EXACT)
{
    const uint8_t *l_end = l + KL, *r_end = r + KR;
    uint8x16_t keep = vdupq_n_u8(0);
    if (!EXACT) keep = vld1q_u8(out + K);
    intptr_t i = 0;
#define PIVCOH__MVV4(mask, pfx) do {                                       \
        intptr_t p0 = ((pfx) >> 8) & 0xff, p1 = ((pfx) >> 24) & 0xff,      \
                 p2 = ((pfx) >> 40) & 0xff, p3 = (pfx) >> 56;              \
        pivcoh__merge16(out + i,      l,           r,      (mask));        \
        pivcoh__merge16(out + i + 16, l + 16 - p0, r + p0, (mask) >> 16);  \
        pivcoh__merge16(out + i + 32, l + 32 - p1, r + p1, (mask) >> 32);  \
        pivcoh__merge16(out + i + 48, l + 48 - p2, r + p2, (mask) >> 48);  \
        r += p3; l += 64 - p3;                                             \
    } while (0)
    if (i + 64 <= K) {
        uint64_t mask; memcpy(&mask, bm, 8);
        uint64_t pfx = PIVCOH__PFX8(bm);
        for (; i + 128 <= K; i += 64) {
            uint64_t nmask; memcpy(&nmask, bm + ((i + 64) >> 3), 8);
            uint64_t npfx = PIVCOH__PFX8(bm + ((i + 64) >> 3));
            PIVCOH__MVV4(mask, pfx);
            mask = nmask; pfx = npfx;
        }
        PIVCOH__MVV4(mask, pfx);
        i += 64;
    }
#undef PIVCOH__MVV4
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
#undef PIVCOH__PFX8

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

/* D=1 (8 codes/byte): the two symbols replicated across 16 lanes,
 * indexed by the bm bit via the same dup-shuffle + per-lane shift as
 * the D=2 unpack.  The tf chunks read <= 1 byte past the region
 * (inside the stream); tf=0 (a 2-symbol flat root) finishes scalar. */
static const uint8_t pivcoh__d1_dup_tab[16]   = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
static const int8_t  pivcoh__d1_shift_tab[16] = {0,-1,-2,-3,-4,-5,-6,-7,
                                                 0,-1,-2,-3,-4,-5,-6,-7};
static void pivcoh__flat_d1(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint16_t lr_word; memcpy(&lr_word, c2s, 2);
    uint8x16_t c2s_vec = vreinterpretq_u8_u16(vdupq_n_u16(lr_word));
    uint8x16_t dup_v   = vld1q_u8(pivcoh__d1_dup_tab);
    int8x16_t  shift_v = vld1q_s8(pivcoh__d1_shift_tab);
    uint8x16_t one_v   = vdupq_n_u8(1);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);

    int j = 0, lim = tf ? n : n - 15;
    for (; j < lim; j += 16) {
        uint16_t bm_word; memcpy(&bm_word, bm + (j >> 3), 2);
        uint8x16_t bm_lo = vreinterpretq_u8_u16(
            vsetq_lane_u16(bm_word, vdupq_n_u16(0), 0));
        uint8x16_t dup     = vqtbl1q_u8(bm_lo, dup_v);
        uint8x16_t shifted = vshlq_u8(dup, shift_v);
        uint8x16_t idx     = vandq_u8(shifted, one_v);
        vst1q_u8(out + j, vqtbl1q_u8(c2s_vec, idx));
    }
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; j < n; j++)
        out[j] = c2s[(bm[j >> 3] >> (j & 7)) & 1];
}

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
    case 1: pivcoh__flat_d1(out, n, bm, c2s, tf); break;
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
 *   ctab8[m][0:8]  right source lanes packed at [0,n_right), 0xff fill
 *                  (vtbl1 returns 0 for out-of-range indices)
 * p16rev partition LUTs (part_full): one combined index per 16-lane group
 * packs {left, forward, front} | {right, reversed, back}; left+right tile
 * the 16 lanes so the OR of the two disjoint-support tables is exact.
 *   ptabA[m0]      low byte: left -> front, right -> back reversed
 *   ptabB0[m1]     high byte, pc0=0 layout; pc0>0 is the same table
 *                  loaded at byte offset pc0 (padded to 32 B/entry). */
static uint8_t pivcoh__ctab8[256][16]   __attribute__((aligned(16)));
static uint8_t pivcoh__ptabA[256][16]   __attribute__((aligned(16)));
static uint8_t pivcoh__ptabB0[256][32]  __attribute__((aligned(32)));

static void pivcoh__init_enc(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
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

/* 16-lane movemask via two GPR magic multiplies: for 0x00/0xFF compare
 * bytes, x * 0x103070F1F3F80 accumulates each u64 half's byte-MSBs into
 * its top byte (all 256 patterns verified per half).  Cheaper than the
 * masks64 reduction tree at narrow-tail widths. */
static inline uint32_t pivcoh__movemask16(uint8x16_t cm)
{
    const uint64_t magic = 0x103070F1F3F80ull;
    uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(cm), 0);
    uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(cm), 1);
    return (uint32_t)((lo * magic) >> 56)
         | ((uint32_t)((hi * magic) >> 48) & 0xFF00u);
}

/* Scatter one 64-rank group-set of a full partition: per 16-lane
 * group, ONE combined shuffle index (the OR of ptabA[m0] |
 * ptabB0[m1]+pc0) yields both sides at once — the register IS the
 * left output, and a loop-invariant full-reverse vqtbl1 recovers the
 * right.  mask_word may be tail-masked: zero bits scatter their lanes
 * as (phantom) lefts AFTER the real ones, so the left prefix stays
 * exact and only dead bytes past it take garbage.  Stores run 16 wide
 * (up to +64/+16 past the valid counts, into dead/slack space).
 * Returns the group-set's right count. */
__attribute__((always_inline)) static inline
int pivcoh__part64_full(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2,
                        uint8x16_t v3, uint64_t mask_word,
                        uint8_t *ldst, uint8_t *rdst)
{
    uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                       vcnt_u8(vcreate_u8(mask_word))), 0);
    uint64_t pfx = pcw * 0x0101010101010101ULL;
    uint8x16_t vg[4] = { v0, v1, v2, v3 };
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
#define PIVCOH__PART(g) do {                                                 \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t ri = vorrq_u8(vld1q_u8(pivcoh__ptabA[m0]),               \
                                 vld1q_u8(&pivcoh__ptabB0[m1][pc0]));       \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                            \
        vst1q_u8(ldst + (16*(g) - cr), comb);                               \
        vst1q_u8(rdst + cr, vqtbl1q_u8(comb, rev16));                       \
    } while (0)
    PIVCOH__PART(0); PIVCOH__PART(1); PIVCOH__PART(2); PIVCOH__PART(3);
#undef PIVCOH__PART
    return (int)(pfx >> 56);
}

/* Full partition: bitmap + both sides compacted (left in place in
 * ranks, right into tmp).  Tail-free at 16-rank granularity: each tail
 * step is one movemask + one p16rev group scatter, the final step's
 * mask trimmed to the real ranks — the wire bytes stay exact, loads
 * overread into the ranks slack/gaps, and the phantom-left scatter
 * lands in dead bytes.  No scalar tail: the per-element rank>thr
 * branch is the bitmap itself, i.e. maximally unpredictable. */
static int pivcoh__part_full(uint8_t *ranks, int n, uint8_t thr,
                             uint8_t *bm, uint8_t *tmp)
{
    int n_left = 0, n_right = 0, j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t w = pivcoh__masks64(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &w, 8);
        int tr = pivcoh__part64_full(v0, v1, v2, v3, w,
                                     ranks + n_left, tmp + n_right);
        n_right += tr;
        n_left  += 64 - tr;
    }
    /* Narrow tail: one 16-rank group per step (the 64-wide group-set is
     * heavy for the 1..20-rank tails deep trees are made of).  The
     * final step's mask is trimmed to the real ranks; phantom lefts
     * land after the real ones in dead bytes, as in the main loop. */
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    for (; j < n; j += 16) {
        uint8x16_t v = vld1q_u8(ranks + j);
        int keep = n - j < 16 ? n - j : 16;
        uint32_t m = pivcoh__movemask16(vcgtq_u8(v, vt)) & ((1u << keep) - 1);
        uint16_t m16 = (uint16_t)m;
        memcpy(bm + (j >> 3), &m16, 2);
        uint32_t pc0 = (uint32_t)__builtin_popcount(m & 0xFF);
        uint8x16_t ri = vorrq_u8(vld1q_u8(pivcoh__ptabA[m & 0xFF]),
                                 vld1q_u8(&pivcoh__ptabB0[m >> 8][pc0]));
        uint8x16_t comb = vqtbl1q_u8(v, ri);
        vst1q_u8(ranks + n_left, comb);
        vst1q_u8(tmp + n_right, vqtbl1q_u8(comb, rev16));
        int tr = __builtin_popcount(m);
        n_right += tr;
        n_left  += 16 - tr;
    }
    return n_right;
}

/* Scatter one 64-rank group-set's right side via ctab8 (8-lane
 * chunks); same phantom-left masking argument as part64_full.
 * Returns the group-set's right count. */
__attribute__((always_inline)) static inline
int pivcoh__part64_right(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2,
                         uint8x16_t v3, uint64_t mask_word, uint8_t *rdst)
{
    uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(
                       vcnt_u8(vcreate_u8(mask_word))), 0);
    uint64_t pfx = pcw * 0x0101010101010101ULL;
    uint8x8_t cv[8] = {
        vget_low_u8(v0), vget_high_u8(v0),
        vget_low_u8(v1), vget_high_u8(v1),
        vget_low_u8(v2), vget_high_u8(v2),
        vget_low_u8(v3), vget_high_u8(v3),
    };
#define PIVCOH__PART1(K_) do {                                               \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF); \
        const uint8_t *tab = pivcoh__ctab8[(uint8_t)(mask_word >> (8*(K_)))]; \
        vst1_u8(rdst + cr, vtbl1_u8(cv[K_], vld1_u8(tab)));                  \
    } while (0)
    PIVCOH__PART1(0); PIVCOH__PART1(1); PIVCOH__PART1(2); PIVCOH__PART1(3);
    PIVCOH__PART1(4); PIVCOH__PART1(5); PIVCOH__PART1(6); PIVCOH__PART1(7);
#undef PIVCOH__PART1
    return (int)(pfx >> 56);
}

/* One-sided (right/none) partition: bitmap always, right side compacted
 * into tmp when EMIT_RIGHT (the left side of a LEAF_LEFT node is dead).
 * EMIT_RIGHT=0 folds to a pure bitmap build (the D=1 flat pack, whose
 * `bm` is the STREAM: the tail's whole-word store rides the packs'
 * junk-byte contract there).  Tail-free like part_full. */
__attribute__((always_inline)) static inline
int pivcoh__part_core(uint8_t *ranks, int n, uint8_t thr,
                      uint8_t *bm, uint8_t *tmp, int EMIT_RIGHT)
{
    int n_right = 0, j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    for (; j + 64 <= n; j += 64) {
        uint8x16_t v0 = vld1q_u8(ranks + j);
        uint8x16_t v1 = vld1q_u8(ranks + j + 16);
        uint8x16_t v2 = vld1q_u8(ranks + j + 32);
        uint8x16_t v3 = vld1q_u8(ranks + j + 48);
        uint64_t w = pivcoh__masks64(v0, v1, v2, v3, vt, bw);
        memcpy(bm + (j >> 3), &w, 8);
        if (EMIT_RIGHT)
            n_right += pivcoh__part64_right(v0, v1, v2, v3, w, tmp + n_right);
        else
            n_right += __builtin_popcountll(w);
    }
    /* Narrow tail, as in part_full. */
    for (; j < n; j += 16) {
        uint8x16_t v = vld1q_u8(ranks + j);
        int keep = n - j < 16 ? n - j : 16;
        uint32_t m = pivcoh__movemask16(vcgtq_u8(v, vt)) & ((1u << keep) - 1);
        uint16_t m16 = (uint16_t)m;
        memcpy(bm + (j >> 3), &m16, 2);
        if (EMIT_RIGHT) {
            vst1_u8(tmp + n_right,
                    vtbl1_u8(vget_low_u8(v), vld1_u8(pivcoh__ctab8[m & 0xFF])));
            n_right += __builtin_popcount(m & 0xFF);
            vst1_u8(tmp + n_right,
                    vtbl1_u8(vget_high_u8(v), vld1_u8(pivcoh__ctab8[m >> 8])));
            n_right += __builtin_popcount(m >> 8);
        } else {
            n_right += __builtin_popcount(m);
        }
    }
    return n_right;
}

/* ---- flat pack: (rank - base) is already the D-bit local code ----
 *
 * Every kernel packs ALL n codes by running its vector loop past n:
 * the final vector loads up to 15 garbage ranks past the region
 * (inside the partition gaps/slack) and packs garbage bits, which land
 * only where they don't matter — bytes past ceil(n*D/8) are junk under
 * the usual contract (overwritten by the next record; inside out_cap
 * at stream end), and the padding bits inside the last partial byte
 * are zeroed by one byte RMW in the dispatcher, store-forwarded from
 * the final vector store. */

/* D=2: 64 ranks -> 16 bytes (4 ranks per byte, no byte crossings) —
 * unrolled x4 so both vpaddq_u8 reduction levels pair full vectors and
 * the result is a whole 16-byte store.  The pipeline is linear mod 256
 * (shifts multiply, vpaddq adds, u8 wrap IS the target modulus), so
 * the base subtract distributes to one op at the end:
 * sum (r_i - b) 4^i = r0 + 4r1 + 16r2 + 64r3 - 85b (mod 256, exact
 * since the true byte is in range).  The 16-rank remainder keeps the
 * self-pairing quarter-width form. */
static inline void pivcoh__pack_d2(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d2[16] = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const int8x16_t sh = vld1q_s8(shifts_d2);
    const uint8x16_t b85 = vdupq_n_u8((uint8_t)(85 * base));
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      sh);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), sh);
        uint8x16_t b2 = vshlq_u8(vld1q_u8(ranks + i + 32), sh);
        uint8x16_t b3 = vshlq_u8(vld1q_u8(ranks + i + 48), sh);
        uint8x16_t r  = vpaddq_u8(vpaddq_u8(b0, b1), vpaddq_u8(b2, b3));
        vst1q_u8(out + (i >> 2), vsubq_u8(r, b85));
    }
    for (; i < n; i += 16) {
        uint8x16_t b  = vshlq_u8(vld1q_u8(ranks + i), sh);
        uint8x16_t s1 = vpaddq_u8(b, b);
        uint8x16_t s2 = vsubq_u8(vpaddq_u8(s1, s1), b85);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i >> 2), &packed4, 4);
    }
}

/* D=4: 32 ranks -> 16 bytes, pairing (r[2k], r[2k+1]) into one byte —
 * unrolled once so the vpaddq_u8 pairs two full input vectors, with
 * the base subtract distributed like D=2's:
 * (r0 - b) + 16(r1 - b) = r0 + 16r1 - 17b (mod 256, exact). */
static inline void pivcoh__pack_d4(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d4[16] = { 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4 };
    const int8x16_t sh = vld1q_s8(shifts_d4);
    const uint8x16_t b17 = vdupq_n_u8((uint8_t)(17 * base));
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      sh);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), sh);
        vst1q_u8(out + (i >> 1), vsubq_u8(vpaddq_u8(b0, b1), b17));
    }
    for (; i < n; i += 16) {
        uint8x16_t b = vshlq_u8(vld1q_u8(ranks + i), sh);
        vst1_u8(out + (i >> 1),
                vget_low_u8(vsubq_u8(vpaddq_u8(b, b), b17)));
    }
}

/* D=5/6/7: variable-shift pack, 16 codes/iter (D=3 pairs itself into
 * D=6 below and rides the same pyramid).  At each width the two
 * halves of a lane pair are shifted TOWARD each other with one USHL of
 * {+s, -s} per-lane counts — the even half's top bit and the odd
 * half's bottom bit meet at the lane boundary — so each pairing level
 * is a single instruction and the packed field rides mid-lane until
 * one final immediate right shift re-bases it:
 *   L1 u8  {8-D, 0}:            u16 = pair  << (8-D)
 *   L2 u16 {8-D, -(8-D)}:       u32 = quad  << (16-2D)
 *   L3 u32 {16-2D, -(16-2D)}:   u64 = octet << (32-4D)
 * The compact shuffle absorbs the whole bytes of the final (32-4D)
 * re-basing shift (its tables start at byte 1 for D=5/6), leaving a
 * residual >> 4 for D=5/7 and NO final shift for D=6 -- 3-4 shift ops,
 * count vectors are vdups of computed constants.  (Byte-aligning the
 * fields EARLY so the tbl can also do the u64 level -- e.g. D=6's
 * 24-bit quad at [0,24) -- costs a shr+sli pair per level, one op
 * more: a sub-lane field can't cross its own byte/lane boundary with
 * a single per-lane shift, which is exactly what the converging
 * meet-at-the-boundary placement avoids.)
 * (History: ryg's multiply-as-shift vmull pyramid, then a 6-op
 * USHR+SLI ladder, each replaced in turn.)  Each 16-byte store carries
 * 16-2D trailing junk bytes, overwritten by the next iter / next
 * record (the caller's out_cap >= PIVCOH_ENCODE_BOUND keeps even the
 * last one in bounds). */
static const uint8_t pivcoh__pack_compact_d5[16] = {
    1, 2, 3, 4, 5,   9, 10, 11, 12, 13,  0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivcoh__pack_compact_d6[16] = {
    1, 2, 3, 4, 5, 6,   9, 10, 11, 12, 13, 14,  0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivcoh__pack_compact_d7[16] = {
    0, 1, 2, 3, 4, 5, 6,   8, 9, 10, 11, 12, 13, 14,  0xff, 0xff
};

#define PIVCOH__PACK_DN(NAME, D_VAL, BITSHR, COMPACT_TAB)                        \
static inline void NAME(uint8_t *out, const uint8_t *ranks, int n, uint8_t base) \
{                                                                                \
    const int8x16_t s1 = vreinterpretq_s8_u16(vdupq_n_u16(8 - (D_VAL)));         \
    const int16x8_t s2 = vreinterpretq_s16_u32(vdupq_n_u32(                      \
        (uint32_t)(uint16_t)(8 - (D_VAL)) |                                      \
        ((uint32_t)(uint16_t)-(8 - (D_VAL)) << 16)));                            \
    const int32x4_t s3 = vreinterpretq_s32_u64(vdupq_n_u64(                      \
        (uint64_t)(uint32_t)(16 - 2 * (D_VAL)) |                                 \
        ((uint64_t)(uint32_t)-(16 - 2 * (D_VAL)) << 32)));                       \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                            \
    for (int i = 0; i < n; i += 16) {                                            \
        uint8x16_t cb = vsubq_u8(vld1q_u8(ranks + i), vdupq_n_u8(base));         \
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(cb, s1));                 \
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));              \
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));              \
        if (BITSHR) w64 = vshrq_n_u64(w64, (BITSHR) ? (BITSHR) : 1);             \
        vst1q_u8(out + ((i * (D_VAL)) >> 3),                                     \
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));                \
    }                                                                            \
}
PIVCOH__PACK_DN(pivcoh__pack_d5, 5, 4, pivcoh__pack_compact_d5)
PIVCOH__PACK_DN(pivcoh__pack_d6, 6, 0, pivcoh__pack_compact_d6)
PIVCOH__PACK_DN(pivcoh__pack_d7, 7, 4, pivcoh__pack_compact_d7)
#undef PIVCOH__PACK_DN

/* D=3: pair adjacent codes into 6-bit values the D=4 way — per-lane
 * {0,3} shifts + one vpaddq (pair = c_even + 8 c_odd, one pair per
 * byte), with the base subtract distributed through the mod-256
 * pairing (- 9b, exact since the true pair < 64) — then run the D=6
 * pyramid on the pairs: a 3-bit LSB-first stream IS the 6-bit
 * LSB-first stream of its pairs, so the wire is unchanged, and D=6
 * needs no final shift.  32 codes in 11 uops; a 16-code remainder
 * runs the same body self-paired (6 valid output bytes, junk store
 * contract as everywhere). */
static inline void pivcoh__pack_d3(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_p[16] = { 0,3, 0,3, 0,3, 0,3, 0,3, 0,3, 0,3, 0,3 };
    const int8x16_t shp = vld1q_s8(shifts_p);
    const uint8x16_t b9 = vdupq_n_u8((uint8_t)(9 * base));
    const int8x16_t s1 = vreinterpretq_s8_u16(vdupq_n_u16(2));
    const int16x8_t s2 = vreinterpretq_s16_u32(vdupq_n_u32(
        (uint32_t)(uint16_t)2 | ((uint32_t)(uint16_t)-2 << 16)));
    const int32x4_t s3 = vreinterpretq_s32_u64(vdupq_n_u64(
        (uint64_t)(uint32_t)4 | ((uint64_t)(uint32_t)-4 << 32)));
    const uint8x16_t compact = vld1q_u8(pivcoh__pack_compact_d6);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      shp);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), shp);
        uint8x16_t pair = vsubq_u8(vpaddq_u8(b0, b1), b9);
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
        vst1q_u8(out + ((i * 3) >> 3),
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));
    }
    for (; i < n; i += 16) {           /* <= 2 self-paired half-blocks */
        uint8x16_t b = vshlq_u8(vld1q_u8(ranks + i), shp);
        uint8x16_t pair = vsubq_u8(vpaddq_u8(b, b), b9);
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
        vst1q_u8(out + ((i * 3) >> 3),
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));
    }
}

/* D=8: byte-aligned; no blend needed — the junk bytes past n are all
 * beyond the region, and there is no partial byte to zero-pad. */
static inline void pivcoh__pack_d8(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    uint8x16_t vb = vdupq_n_u8(base);
    for (int i = 0; i < n; i += 16)
        vst1q_u8(out + i, vsubq_u8(vld1q_u8(ranks + i), vb));
}

/* Dispatcher: D is structural (1..8); every kernel packs all n codes. */
static void pivcoh__pack_dN(uint8_t *out, const uint8_t *ranks,
                            int n, int D, uint8_t base)
{
    switch (D) {
    case 1: /* the D=1 bit IS the partition bit: reuse the bitmap build
             * (EMIT_RIGHT=0 never writes ranks; the cast is sound) */
            (void)pivcoh__part_core((uint8_t *)(uintptr_t)ranks, n, base,
                                    out, NULL, 0);
            break;
    case 2: pivcoh__pack_d2(out, ranks, n, base); break;
    case 3: pivcoh__pack_d3(out, ranks, n, base); break;
    case 4: pivcoh__pack_d4(out, ranks, n, base); break;
    case 5: pivcoh__pack_d5(out, ranks, n, base); break;
    case 6: pivcoh__pack_d6(out, ranks, n, base); break;
    case 7: pivcoh__pack_d7(out, ranks, n, base); break;
    case 8: pivcoh__pack_d8(out, ranks, n, base); break;
    }
    /* Zero the padding bits of the last partial byte (the kernels'
     * final vector packed garbage there); one store-forwarded RMW,
     * idempotent for D=1/8 whose padding is already exact.
     * Unconditional: at rem_bits == 0 the mask is 0 and the target is
     * the first junk byte PAST the region, zeroed harmlessly under the
     * usual contract — cheaper than a per-node data-dependent branch. */
    int rem_bits = (n * D) & 7;
    out[(n * D) >> 3] &= (uint8_t)((1u << rem_bits) - 1);
}

/* ---- encode tree walk ----
 * Production placement, arena-staged: each node's scratch starts with
 * its staged bitmap (nbytes+8), then the compacted right half, then
 * the children's deeper scratch.  Per level that is n/8 + 9 + n_right
 * bytes, depth <= 11, plus the tail-free partition's bounded
 * overstores — inside PIVCOH_SCRATCH_SIZE.  The header is VLA-free. */
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
     * The bitmap is staged across the child recursion (its stream
     * position depends on the children's encoded sizes) at the base of
     * this node's scratch, NOT the stack: as a VLA it was live across
     * the recursion, ~90KB of stack on a worst-case 64K block.  +8 pads
     * the partition tail's 2-byte mask stores (<= 1 byte past nbytes)
     * with margin.  The
     * children's scratch starts 64 bytes past the right ranks: a
     * node's tail-free left scatter overshoots up to 63 bytes past its
     * OWN ranks region, and a right child's ranks end exactly where
     * its scratch (holding its live stage) would otherwise begin. */
    int nbytes = (n + 7) >> 3;
    uint8_t *bm_stage = tmp;
    uint8_t *rout = tmp + nbytes + 8;
    int n_right = (kind == PIVCOH__LEAFL)
        ? pivcoh__part_core(ranks, n, rec->param, bm_stage, rout, 1)
        : pivcoh__part_full(ranks, n, rec->param, bm_stage, rout);
    int n_left = n - n_right;
    *p++ = (uint8_t)n_right;                   /* K_right, u16 LE */
    *p++ = (uint8_t)(n_right >> 8);
    *pp = p;
    if (kind == PIVCOH__FULL && n_right > n_left) {
        pivcoh__enc_node(t, idx + rec->right, rout, n_right, pp, rout + n_right + 64);
        if (n_left > 0)
            pivcoh__enc_node(t, idx + 1, ranks, n_left, pp, rout + n_right + 64);
    } else {
        if (kind == PIVCOH__FULL && n_left > 0)
            pivcoh__enc_node(t, idx + 1, ranks, n_left, pp, rout + n_right + 64);
        if (n_right > 0)
            pivcoh__enc_node(t, idx + rec->right, rout, n_right, pp, rout + n_right + 64);
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
    if (!t->enc_ready) {
        /* Encoder view of the table, built on first use so decode-side
         * builds skip it.  The writes are a pure function of
         * rank_to_sym and idempotent, and enc_ready is set last, so
         * concurrent first encodes on a shared table are benign — the
         * same contract as the lazy static tables above. */
        pivcoh_table *tw = (pivcoh_table *)t;
        memset(tw->sym_to_rank, 0, 256);
        for (int s = t->num_ranks - 1; s >= 0; s--)
            tw->sym_to_rank[t->rank_to_sym[s]] = (uint8_t)s;
        tw->enc_ready = 1;
    }
    uint8_t *sc = scratch ? (uint8_t *)scratch : (uint8_t *)malloc(PIVCOH_SCRATCH_SIZE(n));
    if (!sc) return -1;
    uint8_t *ranks = sc, *tmp = sc + n + 64;   /* +64: the root ranks' overshoot
                                                  gap (the tail-free partition
                                                  strays <= 63 B past a ranks
                                                  region; children get the same
                                                  gap in enc_node) */
    pivcoh__enc_init(ranks, (int)n, in, t->sym_to_rank);
    uint8_t *p = out;
    *p++ = (uint8_t)n;
    *p++ = (uint8_t)(n >> 8);
    pivcoh__enc_node(t, 0, ranks, (int)n, &p, tmp);
    if (!scratch) free(sc);
    return p - out;
}

#endif /* PIVCOH_IMPLEMENTATION */

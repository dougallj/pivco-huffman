#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include "pivco_check.h"

/* The length histogram + counting sort is the hottest part of the
 * decode-table build and is SIMD-friendly (256 bytes, <= 12 bins), so it
 * gets NEON and SSE2 paths here — a deliberate exception to the
 * "intrinsics live in the primitives headers" rule, since this file is
 * backend-agnostic and compiled once.  SSE2 is x86-64 baseline, so no
 * dispatch is needed.  -DPIVCO_HISTO_PORTABLE forces the portable path
 * (for A/B). */
#ifndef PIVCO_HISTO_PORTABLE
#  if defined(__aarch64__) || defined(__ARM_NEON)
#    define PIVCO_HISTO_SIMD 1
#    define PIVCO_HISTO_SIMD_NEON 1
#    include <arm_neon.h>
#  elif defined(__x86_64__) || defined(__i386__)
#    define PIVCO_HISTO_SIMD 1
#    define PIVCO_HISTO_SIMD_SSE 1
#    include <emmintrin.h>
#  endif
#endif

/* ---------- Code lengths via van Leeuwen's two-queue method ----------
 * Replaces the index-indirected binary min-heap.  The heap spent ~⅔ of the
 * whole table build pointer-chasing (nodes[indices[i]].freq is two dependent
 * loads per compare) for a trivial <=256-leaf tree.  The two-queue method is
 * O(n) after one sort: leaves pre-sorted ascending by frequency go in one
 * queue, internal nodes (whose frequencies are generated monotonically) in a
 * second FIFO, so each of the two minima per merge is an O(1) front compare.
 *
 * Tie discipline reproduces the heap's exactly, giving byte-identical lengths:
 * the heap broke ties by node index, and leaves (indices 0..n-1) always sort
 * before internals (indices >=n), so on an equal frequency a leaf wins -- the
 * `<=` below -- and within a queue the front already holds the lowest index
 * (leaves sorted by (freq,sym); internals in creation order). */
typedef pivco_huffman_leaf_t leaf_t;   /* { u64 freq; u16 sym } */

/* Stable sort of leaf[0..n) by frequency ascending.  Stability over the
 * symbol-ordered seed keeps the (freq,sym) tie discipline the heap relied on.
 *
 * Small alphabets take an insertion sort: a radix pass costs 256 bins of
 * zero + prefix regardless of n.  Measured crossover on M4: insertion
 * wins through n = 40 even on its worst case (freqs descending in symbol
 * order — a ranked alphabet), breaks even at ~48, loses past that; on
 * randomly-ordered freqs it wins through ~64.  40 never loses either way.
 *
 * Larger ones take an LSD radix over only the bytes that VARY across the
 * frequencies (`vary` = OR ^ AND of all freqs, a free by-product of the
 * caller's scan): a constant byte is an identity (stable) pass, so it is
 * skipped outright.  Near-uniform frequency sets — whose same-value runs
 * serialize hardest on a bin's store-to-load forwarding — are exactly the
 * sets with few varying bytes.  The remaining planes' histograms are all
 * gathered in ONE pass over the leaves (u16 bins suffice for n <= 256), so
 * same-byte runs split across the planes' independent forwarding chains
 * instead of hammering one bin per pass. */
#define PIVCO_LEAF_SORT_INSERTION_MAX 40

static void sort_leaves_by_freq(leaf_t *leaf, int n, uint64_t vary)
{
    if (n <= PIVCO_LEAF_SORT_INSERTION_MAX) {
        for (int i = 1; i < n; i++) {
            leaf_t cur = leaf[i];
            int j = i - 1;
            while (j >= 0 && leaf[j].freq > cur.freq) { leaf[j + 1] = leaf[j]; j--; }
            leaf[j + 1] = cur;
        }
        return;
    }

    uint8_t pass_shift[8];
    int npass = 0;
    for (int b = 0; b < 8; b++)
        if ((vary >> (8 * b)) & 0xFF)
            pass_shift[npass++] = (uint8_t)(8 * b);
    if (npass == 0) return;              /* all frequencies equal */

    uint16_t cnt[8][256];
    memset(cnt, 0, (size_t)npass * sizeof(cnt[0]));
    /* Constant plane count so the inner loop unrolls (2-3 planes is the
     * per-window norm: counts <= 100K with a zero high-vary byte or two). */
#define PIVCO_HIST_PLANES(NP)                                   \
        for (int i = 0; i < n; i++) {                           \
            uint64_t f = leaf[i].freq;                          \
            for (int p = 0; p < (NP); p++)                      \
                cnt[p][(f >> pass_shift[p]) & 0xFF]++;          \
        }
    switch (npass) {
    case 1:  PIVCO_HIST_PLANES(1); break;
    case 2:  PIVCO_HIST_PLANES(2); break;
    case 3:  PIVCO_HIST_PLANES(3); break;
    default: PIVCO_HIST_PLANES(npass); break;
    }
#undef PIVCO_HIST_PLANES

    leaf_t tmp[PIVCO_MAX_SYMBOLS];
    leaf_t *src = leaf, *dst = tmp;
    for (int p = 0; p < npass; p++) {
        int shift = pass_shift[p];
        uint16_t *c = cnt[p];
        unsigned sum = 0, maxc = 0;
        for (int k = 0; k < 256; k++) {
            unsigned t = c[k]; c[k] = (uint16_t)sum; sum += t;
            if (t > maxc) maxc = t;
        }
        if (maxc >= (unsigned)n / 2) {
            /* Dominant-bin pass (e.g. the top freq byte, where every count
             * below 64K lands in bin 0): the plain scatter's c[k]++ becomes
             * a serial store-to-load-forward chain on that bin.  Cache the
             * ACTIVE bin's cursor in a register across the run; a dominant
             * bin >= n/2 bounds the same-bin transition rate at >= ~50%, so
             * the k!=prev branch predicts well too. */
            unsigned prev_k = (src[0].freq >> shift) & 0xFF;
            unsigned cur = c[prev_k];
            for (int i = 0; i < n; i++) {
                unsigned k = (src[i].freq >> shift) & 0xFF;
                if (k != prev_k) {
                    c[prev_k] = (uint16_t)cur;
                    cur = c[k];
                    prev_k = k;
                }
                dst[cur++] = src[i];
            }
            c[prev_k] = (uint16_t)cur;
        } else {
            for (int i = 0; i < n; i++) { unsigned k = (src[i].freq >> shift) & 0xFF; dst[c[k]++] = src[i]; }
        }
        leaf_t *t = src; src = dst; dst = t;
    }
    if (src != leaf) memcpy(leaf, src, (size_t)n * sizeof(leaf_t));
}

/* Scan the nonzero frequencies into leaf[] in symbol order (the sort's
 * stable seed) and report which freq bytes vary (OR ^ AND — see
 * sort_leaves_by_freq).  Returns the leaf count. */
static int scan_nonzero_freqs(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              leaf_t leaf[PIVCO_MAX_SYMBOLS],
                              uint64_t *vary)
{
    int n_used = 0;
    uint64_t orv = 0, andv = ~(uint64_t)0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        uint64_t f = freq[i];
        if (f > 0) {
            leaf[n_used].freq = f;
            leaf[n_used].sym  = (uint16_t)i;
            n_used++;
            orv |= f;
            andv &= f;
        }
    }
    *vary = orv ^ andv;
    return n_used;
}

/* Derives code lengths for the n_used (>=2) leaves into lengths[] (indexed
 * by symbol; untouched entries stay 0).  leaf[] arrives in symbol order and
 * is sorted here; `vary` is the OR ^ AND of all frequencies (see
 * sort_leaves_by_freq).  Returns the max derived length -- the caller runs
 * limit_code_lengths only when it exceeds PIVCO_MAX_CODE_LEN. */
static int build_lengths_twoqueue(leaf_t leaf[PIVCO_MAX_SYMBOLS],
                                  int n_used, uint64_t vary,
                                  uint8_t lengths[PIVCO_MAX_SYMBOLS])
{
    sort_leaves_by_freq(leaf, n_used, vary);

    /* nodes 0..n_used-1 = leaves (sorted order); n_used.. = internals.
     * The internal queue is the contiguous index range [ih, it). */
    const int N = n_used;
    uint64_t nfreq[PIVCO_MAX_SYMBOLS * 2];
    int      parent[PIVCO_MAX_SYMBOLS * 2];
    for (int i = 0; i < N; i++) nfreq[i] = leaf[i].freq;

    int li = 0;          /* next unconsumed leaf */
    int ih = N;          /* internal-queue head (oldest) */
    int ni = N;          /* next internal node index to create (== queue tail) */
    for (int remaining = N; remaining > 1; remaining--) {
        int a, b;
        if (li < N && (ih == ni || nfreq[li] <= nfreq[ih])) a = li++; else a = ih++;
        if (li < N && (ih == ni || nfreq[li] <= nfreq[ih])) b = li++; else b = ih++;
        nfreq[ni] = nfreq[a] + nfreq[b];
        parent[a] = ni;
        parent[b] = ni;
        ni++;            /* extends the internal queue tail */
    }

    const int root = ni - 1;            /* == 2N-2 */
    uint8_t depth[PIVCO_MAX_SYMBOLS * 2];
    depth[root] = 0;
    for (int i = root - 1; i >= 0; i--) /* parent index always > child index */
        depth[i] = (uint8_t)(depth[parent[i]] + 1);
    /* Leaf depths are non-increasing along the freq-sorted leaves (sibling
     * property), so the max is leaf 0's -- but track it explicitly; the
     * branch updates once and then predicts perfectly. */
    int max_len = 1;
    for (int i = 0; i < N; i++) {
        uint8_t d = depth[i] > 0 ? depth[i] : 1;
        lengths[leaf[i].sym] = d;
        if (d > max_len) max_len = d;
    }
    return max_len;
}

/* ---------- Code length limiting (DEFLATE-style, RFC 1951) ---------- */

static void limit_code_lengths(uint8_t *lengths, int n_symbols, int max_len)
{
    /* Count symbols at each length, folding everything past max_len into
     * the max_len bin as we go (unlimited two-queue depths reach ~90 for
     * adversarial u64 freqs — the previous count[64] indexed by RAW
     * length was an out-of-bounds write for those).  Four striped
     * sub-histograms break the store-to-load-forward chain that
     * same-length runs otherwise serialize on (same fix as
     * length_histogram / the file-level input histogram). */
    PIVCO_CHECK(max_len <= PIVCO_MAX_CODE_LEN && (n_symbols & 3) == 0);
    uint16_t h[4][PIVCO_MAX_CODE_LEN + 1];
    memset(h, 0, sizeof(h));
    int over = 0;   /* any length beyond the cap? */
    for (int i = 0; i < n_symbols; i += 4) {
        for (int j = 0; j < 4; j++) {
            unsigned L = lengths[i + j];
            if (L == 0) continue;
            if (L > (unsigned)max_len) { L = (unsigned)max_len; over = 1; }
            h[j][L]++;
        }
    }
    if (!over) return; /* nothing to do */

    int count[PIVCO_MAX_CODE_LEN + 2] = {0};
    for (int L = 1; L <= max_len; L++)
        count[L] = h[0][L] + h[1][L] + h[2][L] + h[3][L];

    /* Kraft sum may exceed 1.0. Fix by moving symbols from max_len
       to shorter lengths. Each time we move one symbol from length L
       to length L-1, the Kraft delta is: 2^(max-L+1) - 2^(max-L) = 2^(max-L).
       But that creates a "debt" at length L-1 which may also overflow.

       Work bottom-up: for each length from max_len down, if we have
       overflow, push pairs up to parent (length-1). */

    /* Compute Kraft sum in units of 2^(-max_len) */
    uint64_t kraft = 0;
    for (int i = 1; i <= max_len; i++) {
        kraft += (uint64_t)count[i] << (max_len - i);
    }
    uint64_t target = (uint64_t)1 << max_len;

    /* While over-full, increase the longest codes */
    while (kraft > target) {
        /* Find a symbol at a length < max_len and increase it by 1.
           This reduces Kraft by 2^(max_len - len) - 2^(max_len - len - 1)
           = 2^(max_len - len - 1). Pick the longest such length to
           minimize Kraft reduction per step. */
        int best = -1;
        for (int len = max_len - 1; len >= 1; len--) {
            if (count[len] > 0) {
                best = len;
                break;
            }
        }
        if (best < 0) break; /* shouldn't happen */

        count[best]--;
        count[best + 1]++;
        kraft -= (uint64_t)1 << (max_len - best - 1);
    }

    /* While under-full, decrease some max_len codes to shorter lengths.
       This fills unused Kraft capacity. */
    while (kraft < target && count[max_len] > 0) {
        /* Find the shortest length where we can add capacity */
        for (int len = max_len - 1; len >= 1; len--) {
            /* Moving one code from max_len to len changes kraft by:
               +2^(max_len-len) - 2^(max_len-max_len) = 2^(max_len-len) - 1 */
            uint64_t delta = ((uint64_t)1 << (max_len - len)) - 1;
            if (kraft + delta <= target && count[max_len] > 0) {
                count[max_len]--;
                count[len]++;
                kraft += delta;
                break;
            }
        }
        /* If we couldn't shorten anything, done */
        if (kraft < target) {
            /* Try filling one slot at max_len-1 at a time */
            uint64_t delta = ((uint64_t)1 << 1) - 1; /* moving max_len to max_len-1 */
            if (kraft + delta <= target && count[max_len] >= 2) {
                /* Move one from max_len to max_len-1: net = +2 - 1 = +1 */
                count[max_len]--;
                count[max_len - 1]++;
                kraft += 1;
            } else {
                break;
            }
        }
    }

    /* Reassign lengths based on new counts.
       Sort symbols by original length (as proxy for frequency),
       assign shortest new lengths to the most frequent symbols. */
    /* Build sorted list of (original_length, symbol_index) */
    typedef struct { uint8_t len; uint8_t sym; } ls_t;
    ls_t sorted[PIVCO_MAX_SYMBOLS];
    int ns = 0;
    for (int i = 0; i < n_symbols; i++) {
        if (lengths[i] > 0) {
            sorted[ns].len = lengths[i] > max_len ? (uint8_t)max_len : lengths[i];
            sorted[ns].sym = (uint8_t)i;
            ns++;
        }
    }
    /* Sort by original length (shorter = more frequent = should get shorter code) */
    for (int i = 1; i < ns; i++) {
        ls_t tmp = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j].len > tmp.len) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = tmp;
    }

    /* Assign new lengths from count array */
    int si = 0;
    for (int len = 1; len <= max_len && si < ns; len++) {
        for (int c = 0; c < count[len] && si < ns; c++) {
            lengths[sorted[si].sym] = (uint8_t)len;
            si++;
        }
    }
}

/* ---------- Canonical Huffman code assignment ---------- */

/* Builds everything downstream of the code lengths (canonical assignment,
 * rank arrays, walk schedule, aux tables).  Shared by the encode path
 * (after the two-queue derives lengths from frequencies) and the decode path
 * (lengths come straight off the wire).  The explicit tree is NOT built --
 * see pivco_huffman_build_explicit_tree.  Assumes `table` is already zeroed
 * and table->num_symbols is set; caller handles n_used <= 1. */
/* Full build downstream of the code lengths -- defined after build_core
 * below; forward-declared for pivco_huffman_build_table. */
static int build_table_finish(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table);

/* Fill enc_init_aux from sym_to_rank: on x86 the 2tab merge hi table (rank<<8)
 * and point the aux at it; elsewhere leave the aux NULL.  Called by every build
 * path that finalizes sym_to_rank (build_table_finish and the single-symbol fast
 * path), so the x86 prim_enc_init never sees a NULL aux. */
static void fill_enc_init_aux(pivco_huffman_table_t *table)
{
#if defined(__x86_64__) || defined(__i386__)
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++)
        table->enc_init_hi[s] = (uint16_t)((unsigned)table->sym_to_rank[s] << 8);
    table->enc_init_aux.s2r_hi = table->enc_init_hi;
#else
    table->enc_init_aux.s2r_hi = NULL;
#endif
}

/* ---------- Code-length histogram ----------
 *
 * The scalar bin-increment histogram is a store-to-load-forwarding trap:
 * a run of same-length symbols (uniform256 = 256 increments of ONE bin)
 * serializes on the ~6-cycle forward latency.  The variants below break
 * that chain.  Returns n_used (the sum of bins 1..PIVCO_MAX_CODE_LEN),
 * or -1 if any length is invalid (> PIVCO_MAX_CODE_LEN — the lengths
 * come off the wire).  Validation is by bin ACCOUNTING, not a max
 * sweep: zeros are counted too, and any byte outside 0..MAX makes
 * n_used + n_zero fall short of 256.
 * sym_count[1..PIVCO_MAX_CODE_LEN] must be zeroed by the caller. */
#ifdef PIVCO_HISTO_SIMD
/* Per-bin equality sweeps, fused with the counting sort: the same cmeq
 * pass that counts bin L extracts its symbol indices (movemask + bit
 * loop) into items[], in symbol order, with a REGISTER output cursor.
 * That kills the second forwarding chain too — the scalar counting
 * sort's cursor[L]++ serializes same-length runs exactly like the
 * histogram's bins did.  All lanes independent, no scatter, inherently
 * in-bounds for ANY input byte.
 *
 * NEON has no byte movemask, so the compare narrows to a 4-bit-per-lane
 * nibble mask (vshrn) reduced to 1 bit per lane; SSE2's pmovmskb gives
 * the bitmask directly. */
static int length_histogram_sort(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                                 uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1],
                                 uint8_t items[PIVCO_MAX_SYMBOLS],
                                 int per_len_start[PIVCO_MAX_CODE_LEN + 2])
{
    /* Zero-bin count (validation accounting only — nothing to extract). */
    int seen;
#ifdef PIVCO_HISTO_SIMD_NEON
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
        __m128i s = _mm_sad_epu8(zacc, vzero);   /* horizontal u8 sum */
        seen = _mm_cvtsi128_si32(s)
             + _mm_cvtsi128_si32(_mm_srli_si128(s, 8));
    }
#endif

    int out = 0;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        per_len_start[L] = out;
#ifdef PIVCO_HISTO_SIMD_NEON
        const uint8x16_t target = vdupq_n_u8((uint8_t)L);
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 16) {
            uint8x16_t eq = vceqq_u8(vld1q_u8(lengths + i), target);
            /* narrowing shift = 4-bit-per-lane movemask; reduce to 1 bit */
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
    seen += out;
    return (seen == PIVCO_MAX_SYMBOLS) ? out : -1;
}
#else
/* Four interleaved sub-histograms: same-bin runs split across four
 * independent forwarding chains.  Bytes >= 16 are rejected before
 * indexing (they'd store past the 16-wide bins); semantic garbage in
 * (MAX, 16) lands in bins the accounting sum exposes.  The len>0 guard
 * stays — it skips the zero runs of sparse alphabets outright (a bin-0
 * pileup was measured 5x slower than the well-predicted branch, back
 * when this was a single scalar loop). */
static int length_histogram(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                            uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1])
{
    uint16_t h0[16] = {0};
    uint16_t h1[16] = {0};
    uint16_t h2[16] = {0};
    uint16_t h3[16] = {0};
    int n_zero = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i += 4) {
        uint8_t a = lengths[i],     b = lengths[i + 1];
        uint8_t c = lengths[i + 2], d = lengths[i + 3];
        if ((a | b | c | d) > 15) return -1;   /* any byte >= 16 sets a high bit */
        n_zero += (a == 0) + (b == 0) + (c == 0) + (d == 0);
        if (a) h0[a]++;
        if (b) h1[b]++;
        if (c) h2[c]++;
        if (d) h3[d]++;
    }
    int n_used = 0;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        sym_count[L] = (uint16_t)(h0[L] + h1[L] + h2[L] + h3[L]);
        n_used += sym_count[L];
    }
    return (n_used + n_zero == PIVCO_MAX_SYMBOLS) ? n_used : -1;
}
#endif

/* Single-symbol degenerate decode table: a fabricated pair of two ranks
 * holding the same symbol, so the walk needs no degenerate special case —
 * root range [0,2) is a plain both-leaves node.  The schedule is a single
 * PAIR record (thr == rank_begin == 0). */
static void build_single_symbol_decode(int sym,
                                       pivco_huffman_decode_table_t *dt)
{
    dt->rank_to_sym[0] = (uint8_t)sym;
    dt->rank_to_sym[1] = (uint8_t)sym;
    dt->num_ranks = 2;
    dt->sched[0].kd    = (uint8_t)PIVCO_SCHED_PAIR;
    dt->sched[0].param = 0;
    dt->sched[0].right = 0;    /* PAIR: no child records */
    dt->sched_len = 1;
}

/* Full-table single-symbol build (codewords 0/1 at length 1).
 * Assumes `table`'s head is zeroed. */
static void build_single_symbol_table(int sym, pivco_huffman_table_t *table)
{
    table->code[sym] = 0;
    table->code_len[sym] = 1;
    table->max_len = 1;
    table->min_len = 1;
    table->sym_count[1] = 1;
    build_single_symbol_decode(sym, &table->dec);
    fill_enc_init_aux(table);   /* sym_to_rank is all-zero (rank 0) here; aux must not stay NULL */
}

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    if (!freq || !table) return PIVCO_ERR_NULL;

    /* Clear only the build-owned head; the on-demand tail (decode tables
     * + explicit tree) is left undefined -- see the table struct doc. */
    memset(table, 0, offsetof(pivco_huffman_table_t, decode_sym));

    /* Scan the nonzero frequencies straight into the leaf array (symbol
     * order = the sort's stable seed). */
    leaf_t leaf[PIVCO_MAX_SYMBOLS];
    uint64_t vary;
    int n_used = scan_nonzero_freqs(freq, leaf, &vary);

    if (n_used == 0) return PIVCO_ERR_EMPTY;

    table->num_symbols = (uint16_t)n_used;

    if (n_used == 1) {
        build_single_symbol_table(leaf[0].sym, table);
        return PIVCO_OK;
    }

    /* Derive code lengths from frequencies (two-queue, no heap) */
    uint8_t lengths[PIVCO_MAX_SYMBOLS];
    memset(lengths, 0, sizeof(lengths));
    int max_len = build_lengths_twoqueue(leaf, n_used, vary, lengths);

    /* Limit code lengths to PIVCO_MAX_CODE_LEN (rarely needed: per-window
     * counts must be quite skewed to push a depth past 11) */
    if (max_len > PIVCO_MAX_CODE_LEN)
        limit_code_lengths(lengths, PIVCO_MAX_SYMBOLS, PIVCO_MAX_CODE_LEN);

    /* Optional joint length/flat-shape optimization (encoder-side only;
     * the decoder rebuilds identically from the transmitted lengths).
     * No-op unless pivco_huffman_set_joint_lambda(>0) was called; on
     * any internal failure the Huffman lengths above are kept. */
    if (pivco_huffman_get_joint_lambda() > 0.0
        || pivco_huffman_get_joint_time_target() > 0.0)
        (void)pivco_joint_optimize_lengths_leaves(leaf, n_used, lengths);

    return build_table_finish(lengths, table);
}

/* ---------- Pre-order walk schedule (issue #7 execution form) ----------
 *
 * Renders the tree into sched[]: one 3-byte record per visible internal
 * node in pre-order, which is all the codec walk reads at runtime (see
 * pivco_sched_rec_t in pivco_huffman.h).
 *
 * The tree is generated DIRECTLY from the chunk list: chunks in rank
 * order are the left-to-right leaves of the chunk-level tree, and a
 * left-to-right leaf-depth sequence uniquely determines a binary tree.
 * The classic greedy reconstruction recovers it with no codewords at
 * all — at a node at depth d, if the next unconsumed chunk sits at
 * depth d it IS this node (a chunk-leaf: FLAT for width >= 4, PAIR for
 * width 2, a bare leaf for width 1); otherwise the node is internal and
 * both children recurse at d+1.  (An earlier version assigned every
 * rank its MSB-aligned codeword and re-derived the splits by scanning
 * for bit boundaries — a round-trip through the codes that this
 * generation makes unnecessary.)
 *
 * Alongside the records, each consumed chunk memcpy's its symbols into
 * rank_to_sym (chunk-consumption order IS rank order), and the running
 * rank cursor yields each internal node's thr (max rank of the left
 * subtree) for free.
 *
 * Kraft validation falls out naturally: lengths that over- or
 * under-subscribe the code space leave chunks unconsumed or exhaust
 * them mid-tree, reported as PIVCO_ERR_CORRUPT by the caller (the
 * lengths typically come off the wire). */

/* 6 bytes/chunk: the three fields every consumer reads fit a byte each
 * (depth <= PIVCO_MAX_CODE_LEN, bit <= 8, sym_idx <= 255); the chunk's
 * width is always 1 << bit (recomputed, not stored), and its length L
 * (= depth + bit) has no consumer since codeword generation was retired.
 * root_code is written/read by CANONICAL_FLAT's full build only. */
typedef struct {
    uint8_t  depth;     /* tree-depth of chunk root */
    uint8_t  bit;       /* log2 of chunk width, 0..8 */
    uint8_t  sym_idx;   /* index into the length-sorted symbol array */
    uint16_t root_code; /* canonical code of the chunk root (CANONICAL_FLAT) */
} chunk_t;

/* Explicit-stack pre-order generation (a recursion is the natural
 * shape, and extras/pivco_decode_table_ref.c keeps the readable
 * recursive twin, but the flattened form measured ~5% faster on
 * M4/M1 Max — it inlines into build_core and drops the call frames).
 * The stack depth IS the node depth, and a subtree's records are
 * contiguous in pre-order, so a node's skip is sched_len - my at
 * completion.  Returns 0, or -1 on non-Kraft-complete lengths (chunks
 * exhausted mid-tree, left over at the end, or a lone-leaf right
 * child, which no valid chunk sequence produces). */
typedef struct {
    uint16_t my;         /* this node's record index */
    uint16_t mid_sched;  /* sched_len when the right child started */
    uint8_t  rank0;      /* rank on entry */
    uint8_t  mid_rank;   /* rank when the right child started */
    uint8_t  state;      /* 0 = doing left child, 1 = doing right */
} sched_frame_t;

static int build_schedule(pivco_huffman_decode_table_t *dt,
                          const chunk_t *chunks, int n_chunks,
                          const uint8_t *items)
{
    sched_frame_t stk[PIVCO_MAX_CODE_LEN + 1];
    int sp = 0;
    int ci = 0;
    unsigned rank = 0;

    for (;;) {
        if (ci >= n_chunks) return -1;      /* under-subscribed lengths */

        /* Descend the left spine until a chunk sits at this depth. */
        while ((int)chunks[ci].depth != sp) {
            if (sp > PIVCO_MAX_CODE_LEN) return -1;
            PIVCO_CHECK(dt->sched_len < PIVCO_MAX_SYMBOLS - 1);
            sched_frame_t *f = &stk[sp++];
            f->my    = dt->sched_len++;
            f->rank0 = (uint8_t)rank;
            f->state = 0;
        }

        /* Consume the chunk-leaf. */
        {
            const chunk_t *c = &chunks[ci++];
            unsigned b = c->bit;
            unsigned rank0 = rank;
            /* Width-1/2 chunks dominate skewed alphabets; keep their copies
             * inline (a variable-size memcpy is a libc dispatch per chunk). */
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
                rec->param = (uint8_t)rank0;    /* == thr for PAIR */
                rec->right = 0;                 /* no child records */
            }
        }

        /* Ascend, completing parents whose right child just finished. */
        for (;;) {
            if (sp == 0) {                  /* root subtree complete */
                if (ci != n_chunks) return -1;
                dt->num_ranks = (uint16_t)rank;
                return 0;
            }
            sched_frame_t *f = &stk[sp - 1];
            if (f->state == 0) {            /* left done; do the right child */
                f->state     = 1;
                f->mid_sched = dt->sched_len;
                f->mid_rank  = (uint8_t)rank;
                break;
            }
            /* Right done: finalize this internal node's record.  A lone
             * leaf next to an internal sibling is always LEFT (chunk
             * depths never decrease left-to-right under a node); the
             * both-lone case is a sibling singleton pair (NAIVE mode —
             * OPTIMIZED emits at most one singleton per length). */
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
            /* right child's record starts where the left subtree's
             * records ended (1 for LEAF_LEFT: bare left leaf, no record;
             * 0 for the PAIR case: no child records at all). */
            rec->right = (rec->kd == (uint8_t)PIVCO_SCHED_PAIR)
                             ? 0
                             : (uint8_t)(f->mid_sched - f->my);
            sp--;
        }
    }
}

/* Core build: code lengths (n_used >= 2, all <= PIVCO_MAX_CODE_LEN) into a
 * decode table, plus — when `full` is non-NULL — the full table's per-symbol
 * codes, sym_to_rank, sym_count and min/max lengths.  The decode-only path
 * (`full` == NULL) writes nothing but dt, and every dt byte it leaves
 * untouched is beyond num_ranks / sched_len, so callers need no memset. */
static int build_core(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                      pivco_huffman_decode_table_t *dt,
                      pivco_huffman_table_t *full)
{
    /* Histogram code lengths + validate + count used symbols (n_used is
       the bin sum, so no caller needs its own pre-count pass).  The
       0/1-symbol dispatch lives here too.  On NEON/SSE2 the histogram
       sweep also performs the counting sort (items / per_len_start
       filled here); the portable path fills them in its own pass
       below. */
    uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1] = {0};
    uint8_t items[PIVCO_MAX_SYMBOLS];   /* symbols, counting-sorted by length */
    int per_len_start[PIVCO_MAX_CODE_LEN + 2];
#ifdef PIVCO_HISTO_SIMD
    int n_used = length_histogram_sort(lengths, sym_count, items, per_len_start);
#else
    int n_used = length_histogram(lengths, sym_count);
#endif
    if (n_used < 0) return PIVCO_ERR_CORRUPT;
    if (n_used == 0) return PIVCO_ERR_EMPTY;
    if (full) full->num_symbols = (uint16_t)n_used;
    if (n_used == 1) {
        int sym = 0;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++)
            if (lengths[s] && lengths[s] <= PIVCO_MAX_CODE_LEN) { sym = s; break; }
        if (full) {
            /* Degenerate convention: the lone symbol codes as one bit,
               whatever length the input claimed. */
            full->code[sym] = 0;
            full->code_len[sym] = 1;
            full->max_len = 1;
            full->min_len = 1;
            memset(full->sym_count, 0, sizeof(full->sym_count));
            full->sym_count[1] = 1;
        }
        build_single_symbol_decode(sym, dt);
        return PIVCO_OK;
    }

    /* Derive min/max code length from the (<=11) length bins. */
    uint8_t max_len = 0, min_len = PIVCO_MAX_CODE_LEN + 1;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        if (sym_count[L]) {
            if (L < min_len) min_len = (uint8_t)L;
            max_len = (uint8_t)L;
        }
    }

    /* ---------- Flat-aware code assignment ----------
     *
     * Goal: give each symbol a code of its assigned length such that the
     * resulting binary tree has as many large flat-D>=2 subtrees as
     * possible (consolidates the partition path during tree-walk decode).
     * Compression is unaffected — code lengths match the Huffman result.
     *
     * Algorithm: per length L, decompose c_L by its binary representation
     * into "chunks": bits >= 2 form D>=2 flat subtrees of size 2^D rooted
     * at depth L-D; bit 1 forms a D=1 sibling pair (handled by stage
     * fusion at decode); bit 0 is a singleton.  Sort chunks by their
     * tree-depth asc (depth = L-D for D>=2 chunks, L-1 for D=1, L for
     * singletons), then canonical-assign codes to chunks.  Within each
     * chunk, top-freq-first symbols of length L are assigned to its
     * 2^bit suffix slots (highest freqs go to the largest-D chunk per
     * length, where the partition-path savings are deepest).
     *
     * See IDEAS.md "Flat-aware Huffman tree restructurer" for the gap
     * analysis (extras/bench/bench_flat_optimal.c).
     */

    /* Per-length: collect symbols in symbol-value order.
     *
     * We used to sort within a tier by frequency-desc so the heaviest
     * symbols landed in the largest flat chunk.  That was dropped: it is
     * the only thing that made the tree depend on within-tier frequency
     * order, which in turn forced a within-tier ordering onto the wire
     * (the v0.3 ORDERING section + rank_within_tier) so the decoder could
     * reproduce it.  On FSE-coded blocks the freq-order "win" only *masked*
     * a bad FSE commit policy (the gate ignores FSE decode cost).  Plain
     * symbol-value order is deterministic from the code lengths alone, so
     * encoder and decoder agree with no rank info transmitted. */
#ifndef PIVCO_HISTO_SIMD
    {
        /* Counting sort by length: prefix-sum the per-length counts, then a
           single symbol-order pass places each symbol.  (The SIMD build
           did this inside the histogram sweep — see length_histogram_sort;
           this cursor loop's cursor[L]++ chain serializes same-length runs
           just like the scalar histogram's bins did.) */
        int acc = 0;
        int cursor[PIVCO_MAX_CODE_LEN + 2];
        for (int L = 1; L <= max_len; L++) {
            per_len_start[L] = acc;
            cursor[L] = acc;
            acc += sym_count[L];
        }
        per_len_start[max_len + 1] = acc;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            uint8_t L = lengths[s];
            if (L) items[cursor[L]++] = (uint8_t)s;
        }
    }
#endif

    /* Decompose each c_L into chunks.  Strategy depends on tree mode --
       see pivco_huffman_set_tree_mode().  Default OPTIMIZED matches the
       original production behavior (decompose c_L by its set bits). */
    chunk_t chunks[PIVCO_MAX_SYMBOLS];   /* upper bound: one chunk per symbol */
    int n_chunks = 0;
    pivco_tree_mode_t tree_mode = pivco_huffman_get_tree_mode();

    if (tree_mode == PIVCO_TREE_MODE_NAIVE) {
        /* Every symbol is its own D=0 chunk at depth L. */
        for (int L = 1; L <= max_len; L++) {
            int c = sym_count[L];
            int cur = per_len_start[L];
            for (int i = 0; i < c; i++) {
                chunks[n_chunks].bit     = 0;
                chunks[n_chunks].depth   = (uint8_t)L;
                chunks[n_chunks].sym_idx = (uint8_t)(cur + i);
                n_chunks++;
            }
        }
    } else if (tree_mode == PIVCO_TREE_MODE_FUSED) {
        /* D=1 sibling pairs first within each length, then a D=0 singleton
           for the odd-tail symbol.  Sequential reassign in the standard
           depth-sort step gives canonical Huffman codes. */
        for (int L = 1; L <= max_len; L++) {
            int c = sym_count[L];
            int cur = per_len_start[L];
            int n_pairs = c / 2;
            int n_singletons = c & 1;
            for (int i = 0; i < n_pairs; i++) {
                chunks[n_chunks].bit     = 1;
                chunks[n_chunks].depth   = (uint8_t)(L - 1);
                chunks[n_chunks].sym_idx = (uint8_t)cur;
                cur += 2;
                n_chunks++;
            }
            for (int i = 0; i < n_singletons; i++) {
                chunks[n_chunks].bit     = 0;
                chunks[n_chunks].depth   = (uint8_t)L;
                chunks[n_chunks].sym_idx = (uint8_t)cur;
                cur++;
                n_chunks++;
            }
        }
    } else if (tree_mode == PIVCO_TREE_MODE_CANONICAL_FLAT) {
        /* Compute canonical first_code[L] = (first_code[L-1] + c_{L-1}) << 1
           starting from min_len.  For each length, greedy-peel the largest
           2^k chunk such that the canonical start code C is 2^k-aligned and
           2^k <= remaining.  root_code = C >> k. */
        uint32_t fc[PIVCO_MAX_CODE_LEN + 2] = {0};
        uint32_t code = 0;
        int last_L = 0;
        for (int L = 1; L <= max_len; L++) {
            if (sym_count[L]) {
                if (last_L) code = (code + (uint32_t)sym_count[last_L]) << (L - last_L);
                fc[L] = code;
                last_L = L;
            }
        }
        for (int L = 1; L <= max_len; L++) {
            int c = sym_count[L];
            if (c == 0) continue;
            int cur = per_len_start[L];
            uint32_t C = fc[L];
            int remaining = c;
            while (remaining > 0) {
                int max_k_align = (C == 0) ? PIVCO_MAX_CODE_LEN : __builtin_ctz(C);
                int max_k_count = (remaining > 1) ? (31 - __builtin_clz((unsigned)remaining)) : 0;
                int k = max_k_align < max_k_count ? max_k_align : max_k_count;
                /* Safety: chunk depth = L-k must be >= 0; since k <= log2(remaining) <= log2(c) <= L-1
                   under any valid Kraft length distribution, this is always true. */
                int n = 1 << k;
                chunks[n_chunks].bit       = (uint8_t)k;
                chunks[n_chunks].depth     = (uint8_t)(L - k);
                chunks[n_chunks].root_code = (uint16_t)(C >> k);
                chunks[n_chunks].sym_idx   = (uint8_t)cur;
                cur += n;
                n_chunks++;
                C += (uint32_t)n;
                remaining -= n;
            }
        }
    } else {
        /* OPTIMIZED (default): original bit-decomposition of c_L. */
        for (int L = 1; L <= max_len; L++) {
            int c = sym_count[L];
            int cur = per_len_start[L];
            /* Iterate set bits high-to-low so larger chunks come first
               within the length (matters only for top-freq-first symbol
               assignment within the length). */
            for (int bit = PIVCO_MAX_CODE_LEN; bit >= 0; bit--) {
                if (c & (1 << bit)) {
                    int n = 1 << bit;
                    int depth;
                    if      (bit >= 2) depth = L - bit;
                    else if (bit == 1) depth = L - 1;
                    else               depth = L;
                    chunks[n_chunks].bit     = (uint8_t)bit;
                    chunks[n_chunks].depth   = (uint8_t)depth;
                    chunks[n_chunks].sym_idx = (uint8_t)cur;
                    cur += n;
                    n_chunks++;
                }
            }
        }
    }


    /* Chunk order below == rank order (the tree's left-to-right leaf
       order): CANONICAL_FLAT chunks are generated in canonical code
       order; every other mode depth-sorts, and canonical assignment
       over sorted depths fills the tree leftmost-first.  build_schedule
       consumes the chunks in exactly this order.

       Per-symbol CODES are needed only by the full table (wire header
       serialization, trad codecs, tools) — the decode-only path skips
       the assignment loops entirely. */
    if (tree_mode == PIVCO_TREE_MODE_CANONICAL_FLAT) {
        /* Chunks already carry canonical root_codes; assign symbol codes
           directly from them (sequential reassign would clobber the
           canonical prefixes when chunks span multiple depths). */
        if (full) {
            unsigned rank = 0;
            for (int ci = 0; ci < n_chunks; ci++) {
                int bit = chunks[ci].bit;
                int n   = 1 << bit;
                uint16_t root = chunks[ci].root_code;
                for (int i = 0; i < n; i++) {
                    uint8_t sym = items[chunks[ci].sym_idx + i];
                    full->code[sym] =
                        (uint16_t)(((uint32_t)root << bit) | (uint32_t)i);
                    full->sym_to_rank[sym] = (uint8_t)rank;
                    rank++;
                }
            }
        }
    } else {
        /* Sort chunks by depth asc (stable; ties keep their natural order
           which is L asc by length, larger-bit-first within length).
           Insertion sort on purpose: the generation order above is
           already nearly depth-sorted (exactly sorted in NAIVE mode,
           where n_chunks can reach 256), so it runs near-linear, and
           the worst case is bounded by n_chunks <= ~50 in OPTIMIZED.
           A stable counting sort by depth measured SLOWER here (fixed
           bin overhead + a cold 3 KB scratch array). */
        for (int i = 1; i < n_chunks; i++) {
            chunk_t cur = chunks[i];
            int j = i - 1;
            while (j >= 0 && chunks[j].depth > cur.depth) {
                chunks[j + 1] = chunks[j];
                j--;
            }
            chunks[j + 1] = cur;
        }

        /* Canonical-assign codes to chunks (chunk-level Kraft sum = 1).
           Each chunk gets a code prefix of length `chunk.depth`.  Within
           the chunk, symbol i takes suffix i for i in [0, 2^bit). */
        if (full) {
            uint32_t code = 0;
            int prev_depth = 0;
            unsigned rank = 0;
            for (int ci = 0; ci < n_chunks; ci++) {
                int d = chunks[ci].depth;
                if (d > prev_depth) code <<= (d - prev_depth);
                int bit = chunks[ci].bit;
                int n   = 1 << bit;
                for (int i = 0; i < n; i++) {
                    uint8_t sym = items[chunks[ci].sym_idx + i];
                    full->code[sym] = (uint16_t)((code << bit) | (uint32_t)i);
                    full->sym_to_rank[sym] = (uint8_t)rank;
                    rank++;
                }
                code += 1;
                prev_depth = d;
            }
        }
    }

    /* On-demand extras NOT built here: the 2^MAX_CODE_LEN flat decode
     * tables (pivco_huffman_build_traditional_table) and the explicit
     * tree (pivco_huffman_build_explicit_tree).  The codec streams the
     * schedule below and reads neither. */

    /* Schedule + rank_to_sym, straight from the chunk list
     * (build_schedule above).  Also validates Kraft-completeness. */
    dt->sched_len = 0;
    if (build_schedule(dt, chunks, n_chunks, items) != 0)
        return PIVCO_ERR_CORRUPT;

    if (full) {
        memcpy(full->sym_count, sym_count, sizeof(sym_count));
        full->max_len = max_len;
        full->min_len = min_len;
    }
    return PIVCO_OK;
}

/* Full build downstream of the code lengths: decode core (into
 * table->dec) plus per-symbol codes, sym_to_rank and the encode aux
 * tables.  Assumes `table`'s head is zeroed and num_symbols is set;
 * caller handles n_used <= 1. */
static int build_table_finish(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    memcpy(table->code_len, lengths, PIVCO_MAX_SYMBOLS);

    int rc = build_core(lengths, &table->dec, table);
    if (rc != PIVCO_OK) return rc;

    fill_enc_init_aux(table);   /* x86 2tab/4tab gather tables (or NULL elsewhere) */
    return PIVCO_OK;
}

/* Public API: build ONLY the ~1 KB decode table from code lengths — the
 * minimal decode-side setup.  No memset: every dt byte beyond num_ranks /
 * sched_len is simply never read. */
int pivco_huffman_build_decode_table(const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
                                     pivco_huffman_decode_table_t *dt)
{
    if (!code_lens || !dt) return PIVCO_ERR_NULL;
    return build_core(code_lens, dt, NULL);
}

/* Encode-side fields of the minimal codec table, from its finished decode
 * core: sym_to_rank is one inversion of rank_to_sym (descending, so the
 * degenerate single-symbol table's duplicate pair resolves to rank 0,
 * matching the full build's zero seed), zeroed first so off-table symbols
 * map to rank 0 like the full table's memset head.  Plus the x86 gather
 * aux (NULL elsewhere) — same convention as fill_enc_init_aux. */
static void ct_fill_encode_side(pivco_huffman_codec_table_t *ct)
{
    memset(ct->sym_to_rank, 0, sizeof(ct->sym_to_rank));
    for (int r = ct->dec.num_ranks - 1; r >= 0; r--)
        ct->sym_to_rank[ct->dec.rank_to_sym[r]] = (uint8_t)r;
#if defined(__x86_64__) || defined(__i386__)
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++)
        ct->enc_init_hi[s] = (uint16_t)((unsigned)ct->sym_to_rank[s] << 8);
    ct->enc_init_aux.s2r_hi = ct->enc_init_hi;
#else
    ct->enc_init_aux.s2r_hi = NULL;
#endif
}

int pivco_huffman_build_codec_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                    pivco_huffman_codec_table_t *ct)
{
    if (!freq || !ct) return PIVCO_ERR_NULL;

    leaf_t leaf[PIVCO_MAX_SYMBOLS];
    uint64_t vary;
    int n_used = scan_nonzero_freqs(freq, leaf, &vary);
    if (n_used == 0) return PIVCO_ERR_EMPTY;

    /* Lengths derive straight into ct->code_len — they double as the
     * header lengths the caller serializes. */
    memset(ct->code_len, 0, sizeof(ct->code_len));
    if (n_used == 1) {
        ct->code_len[leaf[0].sym] = 1;   /* degenerate convention */
        build_single_symbol_decode(leaf[0].sym, &ct->dec);
        ct_fill_encode_side(ct);
        return PIVCO_OK;
    }
    int max_len = build_lengths_twoqueue(leaf, n_used, vary, ct->code_len);
    if (max_len > PIVCO_MAX_CODE_LEN)
        limit_code_lengths(ct->code_len, PIVCO_MAX_SYMBOLS, PIVCO_MAX_CODE_LEN);

    /* Optional joint length/flat-shape optimization (encoder-side only;
     * the decoder rebuilds identically from the transmitted lengths).
     * No-op unless pivco_huffman_set_joint_lambda(>0) was called; on
     * any internal failure the Huffman lengths above are kept. */
    if (pivco_huffman_get_joint_lambda() > 0.0
        || pivco_huffman_get_joint_time_target() > 0.0)
        (void)pivco_joint_optimize_lengths_leaves(leaf, n_used, ct->code_len);

    int rc = build_core(ct->code_len, &ct->dec, NULL);
    if (rc != PIVCO_OK) return rc;   /* unreachable: lengths are Kraft-exact */
    ct_fill_encode_side(ct);
    return PIVCO_OK;
}

int pivco_huffman_build_codec_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_huffman_codec_table_t *ct)
{
    if (!code_lens || !ct) return PIVCO_ERR_NULL;
    int rc = build_core(code_lens, &ct->dec, NULL);
    if (rc != PIVCO_OK) return rc;
    memcpy(ct->code_len, code_lens, PIVCO_MAX_SYMBOLS);
    /* Degenerate single-symbol table: the lone symbol codes as one bit
     * whatever length the input claimed (full-build convention; the
     * fabricated rank pair identifies the case). */
    if (ct->dec.num_ranks == 2 &&
        ct->dec.rank_to_sym[0] == ct->dec.rank_to_sym[1]) {
        memset(ct->code_len, 0, sizeof(ct->code_len));
        ct->code_len[ct->dec.rank_to_sym[0]] = 1;
    }
    ct_fill_encode_side(ct);
    return PIVCO_OK;
}

/* Public API: build a table from code lengths alone.  The tree is fully
 * determined by the lengths (within-tier order is symbol-value), so encoder
 * and decoder reconstruct identical tables with no extra wire info.  Goes
 * straight to build_table_finish -- no synthetic frequencies, no Huffman
 * heap (the lengths are already final). */
int pivco_huffman_build_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_huffman_table_t *table)
{
    if (!code_lens || !table) return PIVCO_ERR_NULL;
    /* Clear only the build-owned head; the on-demand tail (decode tables
     * + explicit tree, ~11 KB) is filled by its builders when a tool asks
     * for it, so zeroing it here is wasted work -- see the struct doc. */
    memset(table, 0, offsetof(pivco_huffman_table_t, decode_sym));

    /* n_used counting and the 0/1-symbol dispatch live in build_core
       (they fall out of its histogram); num_symbols is set there too. */
    return build_table_finish(code_lens, table);
}

/* Fill the 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) read by
 * the traditional flat-table decoder (trad_huffman_decode*).  Call once after
 * the table is built; the production tree-walk decoder does not need it, so
 * pivco_huffman_build_table no longer fills it automatically. */
void pivco_huffman_build_traditional_table(pivco_huffman_table_t *table)
{
    if (!table) return;
    /* Defensive base fill covers any gap for incomplete codes (single sym);
     * rank 0's symbol is always a valid table symbol. */
    memset(table->decode_sym, table->dec.rank_to_sym[0], sizeof(table->decode_sym));
    memset(table->decode_len, 1, sizeof(table->decode_len));
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        int len = table->code_len[s];
        if (len <= 0) continue;
        int shift = PIVCO_MAX_CODE_LEN - len;
        uint32_t base  = (uint32_t)table->code[s] << shift;
        uint32_t count = (uint32_t)1 << shift;
        memset(&table->decode_sym[base], s, count);
        memset(&table->decode_len[base], len, count);
    }
}

/* ---------- On-demand explicit tree (analysis / debug) ----------
 *
 * The production codec streams sched[] and never reads tree[] /
 * node_type[] / flat_*[] / split_rank[] / flat_base_rank[] /
 * max_leaf_depth[], so the normal build no longer fills them.  Tools
 * that inspect the tree (bench stats, viz, dumps) call
 * pivco_huffman_build_explicit_tree() after building the table; it
 * reconstructs the identical topology from the schedule + rank arrays.
 * Nodes are numbered in pre-order (root = 0); the retired build
 * numbered them in chunk-spine creation order, so raw indices differ
 * while every structural property (node set, counts, types, ranks,
 * depths) is unchanged. */

typedef struct {
    pivco_huffman_table_t *t;
    unsigned cursor;      /* schedule cursor */
    int16_t  node_count;
    uint16_t pool;        /* flat_code_to_sym fill cursor */
} etree_ctx_t;

static int16_t etree_leaf(etree_ctx_t *c, unsigned rank)
{
    pivco_huffman_table_t *t = c->t;
    int16_t id = c->node_count++;
    t->tree[id].symbol = (int16_t)t->dec.rank_to_sym[rank];
    t->tree[id].left   = -1;
    t->tree[id].right  = -1;
    t->node_type[id] = (uint8_t)PIVCO_NODE_LEAF;
    return id;
}

static int16_t etree_walk(etree_ctx_t *c, unsigned rank_begin, unsigned rank_end)
{
    pivco_huffman_table_t *t = c->t;
    if (rank_end - rank_begin == 1) return etree_leaf(c, rank_begin);

    const pivco_sched_rec_t *rec = &t->dec.sched[c->cursor++];
    const unsigned kind = rec->kd & 3u;
    int16_t id = c->node_count++;
    t->tree[id].symbol = -1;
    t->tree[id].left   = -1;
    t->tree[id].right  = -1;

    switch (kind) {
    case PIVCO_SCHED_FLAT: {
        unsigned D = rec->kd >> 2;
        t->flat_depth[id]     = (uint8_t)D;
        t->flat_offset[id]    = c->pool;
        t->flat_base_rank[id] = rec->param;
        memcpy(&t->flat_code_to_sym[c->pool], &t->dec.rank_to_sym[rec->param],
               (size_t)1 << D);
        c->pool = (uint16_t)(c->pool + (1u << D));
        t->node_type[id] = (uint8_t)PIVCO_NODE_INTERNAL_FLAT;
        t->max_leaf_depth[id] = (uint8_t)D;
        break;
    }
    case PIVCO_SCHED_PAIR:
        t->tree[id].left  = etree_leaf(c, rank_begin);
        t->tree[id].right = etree_leaf(c, rank_begin + 1);
        t->split_rank[id] = rec->param;
        t->node_type[id] = (uint8_t)PIVCO_NODE_BOTH_LEAVES;
        t->max_leaf_depth[id] = 1;
        break;
    case PIVCO_SCHED_LEAF_LEFT: {
        t->tree[id].left = etree_leaf(c, rank_begin);
        int16_t r = etree_walk(c, rank_begin + 1, rank_end);
        t->tree[id].right = r;
        t->split_rank[id] = rec->param;
        t->node_type[id] = (uint8_t)PIVCO_NODE_LEAF_LEFT;
        t->max_leaf_depth[id] = (uint8_t)(1 + t->max_leaf_depth[r]);
        break;
    }
    default: {  /* PIVCO_SCHED_FULL */
        unsigned split = (unsigned)rec->param + 1;
        int16_t l = etree_walk(c, rank_begin, split);
        int16_t r = etree_walk(c, split, rank_end);
        t->tree[id].left  = l;
        t->tree[id].right = r;
        t->split_rank[id] = rec->param;
        t->node_type[id] = (uint8_t)PIVCO_NODE_INTERNAL_FULL;
        uint8_t lm = t->max_leaf_depth[l], rm = t->max_leaf_depth[r];
        t->max_leaf_depth[id] = (uint8_t)(1 + (lm > rm ? lm : rm));
        break;
    }
    }
    return id;
}

void pivco_huffman_build_explicit_tree(pivco_huffman_table_t *table)
{
    if (!table || table->dec.num_ranks < 2) return;
    /* Zero the per-node arrays: the flat test (flat_depth >= 2), the
     * flat-root split_rank convention (stays 0) and max_leaf_depth leaf
     * entries all rely on zeros, and the table may have been built
     * without the memset covering a previous explicit-tree fill. */
    memset(table->tree,           0, sizeof(table->tree));
    memset(table->node_type,      0, sizeof(table->node_type));
    memset(table->flat_depth,     0, sizeof(table->flat_depth));
    memset(table->flat_offset,    0, sizeof(table->flat_offset));
    memset(table->flat_code_to_sym, 0, sizeof(table->flat_code_to_sym));
    memset(table->split_rank,     0, sizeof(table->split_rank));
    memset(table->flat_base_rank, 0, sizeof(table->flat_base_rank));
    memset(table->max_leaf_depth, 0, sizeof(table->max_leaf_depth));

    etree_ctx_t c = { table, 0, 0, 0 };
    table->tree_root = etree_walk(&c, 0, table->dec.num_ranks);
    table->tree_node_count = c.node_count;
    PIVCO_CHECK(table->tree_root == 0);
    PIVCO_CHECK(c.cursor == table->dec.sched_len);
}

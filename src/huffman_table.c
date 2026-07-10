#include "pivco_huffman.h"
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include "pivco_check.h"

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
typedef struct { uint64_t freq; uint16_t sym; } leaf_t;

/* Stable LSD radix sort of leaf[0..n) by frequency ascending, over only the
 * bytes the max frequency needs (typically 2-3 for per-window counts).  Beats
 * qsort here: no indirect compare per element, and stability over the
 * symbol-ordered seed keeps the (freq,sym) tie discipline the heap relied on. */
static void sort_leaves_by_freq(leaf_t *leaf, int n)
{
    uint64_t mx = 0;
    for (int i = 0; i < n; i++) if (leaf[i].freq > mx) mx = leaf[i].freq;
    int nbytes = 0;
    while (mx) { nbytes++; mx >>= 8; }   /* freq>0 for every leaf => nbytes>=1 */

    leaf_t tmp[PIVCO_MAX_SYMBOLS];
    leaf_t *src = leaf, *dst = tmp;
    for (int b = 0; b < nbytes; b++) {
        int shift = b * 8;
        int cnt[256] = {0};
        for (int i = 0; i < n; i++) cnt[(src[i].freq >> shift) & 0xFF]++;
        int sum = 0;
        for (int c = 0; c < 256; c++) { int t = cnt[c]; cnt[c] = sum; sum += t; }
        for (int i = 0; i < n; i++) { int k = (src[i].freq >> shift) & 0xFF; dst[cnt[k]++] = src[i]; }
        leaf_t *t = src; src = dst; dst = t;
    }
    if (src != leaf) memcpy(leaf, src, (size_t)n * sizeof(leaf_t));
}

/* Derives code lengths for n_used (>=2) symbols into lengths[] (indexed by
 * symbol; untouched entries stay 0).  No length limiting -- the caller applies
 * limit_code_lengths afterwards, same as the heap path did. */
static void build_lengths_twoqueue(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                   int n_used, const int used[PIVCO_MAX_SYMBOLS],
                                   uint8_t lengths[PIVCO_MAX_SYMBOLS])
{
    leaf_t leaf[PIVCO_MAX_SYMBOLS];
    for (int i = 0; i < n_used; i++) {
        leaf[i].freq = freq[used[i]];
        leaf[i].sym  = (uint16_t)used[i];
    }
    sort_leaves_by_freq(leaf, n_used);

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
    for (int i = 0; i < N; i++)
        lengths[leaf[i].sym] = depth[i] > 0 ? depth[i] : 1;
}

/* ---------- Code length limiting (DEFLATE-style, RFC 1951) ---------- */

static void limit_code_lengths(uint8_t *lengths, int n_symbols, int max_len)
{
    /* Count symbols at each length */
    int count[64] = {0}; /* support original lengths up to 63 */
    int max_orig = 0;
    for (int i = 0; i < n_symbols; i++) {
        if (lengths[i] > 0) {
            count[lengths[i]]++;
            if (lengths[i] > max_orig) max_orig = lengths[i];
        }
    }
    if (max_orig <= max_len) return; /* nothing to do */

    /* Move all symbols longer than max_len down to max_len */
    for (int i = max_orig; i > max_len; i--) {
        count[max_len] += count[i];
        count[i] = 0;
    }

    /* Now Kraft sum may exceed 1.0. Fix by moving symbols from max_len
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

/* Single-symbol degenerate table: a fabricated pair of two ranks holding
 * the same symbol (codewords 0/1 at length 1), so the walk needs no
 * degenerate special case — root range [0,2) is a plain both-leaves node.
 * Assumes `table` is zeroed. */
static void build_single_symbol_table(int sym, pivco_huffman_table_t *table)
{
    table->code[sym] = 0;
    table->code_len[sym] = 1;
    table->max_len = 1;
    table->min_len = 1;
    table->sym_count[1] = 1;
    table->first_code[1] = 0;
    table->first_sym_idx[1] = 0;
    table->sorted_symbols[0] = (uint8_t)sym;
    table->rank_to_sym[0] = (uint8_t)sym;
    table->rank_to_sym[1] = (uint8_t)sym;
    table->rank_to_codeword[0] = 0x0000;
    table->rank_to_codeword[1] = 0x8000;
    table->num_ranks = 2;
    /* Walk schedule: a single PAIR record (thr == rank_begin == 0). */
    table->sched[0].kd    = (uint8_t)PIVCO_SCHED_PAIR;
    table->sched[0].param = 0;
    table->sched[0].skip  = 1;
    table->sched_len = 1;
    fill_enc_init_aux(table);   /* sym_to_rank is all-zero (rank 0) here; aux must not stay NULL */
}

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    if (!freq || !table) return PIVCO_ERR_NULL;

    /* Clear only the build-owned head; the on-demand tail (decode tables
     * + explicit tree) is left undefined -- see the table struct doc. */
    memset(table, 0, offsetof(pivco_huffman_table_t, decode_sym));

    /* Count symbols with nonzero frequency */
    int n_used = 0;
    int used[PIVCO_MAX_SYMBOLS];
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (freq[i] > 0) {
            used[n_used++] = i;
        }
    }

    if (n_used == 0) return PIVCO_ERR_EMPTY;

    table->num_symbols = (uint16_t)n_used;

    if (n_used == 1) {
        build_single_symbol_table(used[0], table);
        return PIVCO_OK;
    }

    /* Derive code lengths from frequencies (two-queue, no heap) */
    uint8_t lengths[PIVCO_MAX_SYMBOLS];
    memset(lengths, 0, sizeof(lengths));
    build_lengths_twoqueue(freq, n_used, used, lengths);

    /* Limit code lengths to PIVCO_MAX_CODE_LEN */
    limit_code_lengths(lengths, PIVCO_MAX_SYMBOLS, PIVCO_MAX_CODE_LEN);

    return build_table_finish(lengths, table);
}

/* ---------- Pre-order walk schedule (issue #7 execution form) ----------
 *
 * Renders the implicit rank-range tree into sched[]: one 3-byte record
 * per visible internal node in pre-order, which is all the codec walk
 * reads at runtime (see pivco_sched_rec_t in pivco_huffman.h).  Built by
 * one recursion over the rank ranges, evaluating each node's split
 * exactly once — the splits are a pure function of the table, so
 * computing them here (instead of lazily per block, as the canonical
 * form allows) amortizes them across every block the table encodes or
 * decodes.  Returns the record count of the subtree (== the root
 * record's skip). */

/* First rank of the right child: linear scan for the first codeword with
 * bit `level` set.  Runs once per node per table build — total work is
 * O(sum of leaf depths) <= 256 * PIVCO_MAX_CODE_LEN — so plain scalar. */
static unsigned sched_split_rank(const pivco_huffman_table_t *table,
                                 int level,
                                 unsigned rank_begin, unsigned rank_end)
{
    const uint16_t mask = (uint16_t)(0x8000u >> level);
    unsigned r = rank_begin + 1;
    while (r < rank_end && (table->rank_to_codeword[r] & mask) == 0)
        r++;
    PIVCO_CHECK(r < rank_end);
    return r;
}

static uint8_t build_walk_schedule(pivco_huffman_table_t *table,
                                   unsigned rank_begin, unsigned rank_end,
                                   int level)
{
    unsigned range_len = rank_end - rank_begin;
    if (range_len == 1) return 0;               /* leaf — no record */

    PIVCO_CHECK(table->sched_len < PIVCO_MAX_SYMBOLS - 1);
    pivco_sched_rec_t *rec = &table->sched[table->sched_len++];
    unsigned cnt = 1;

    unsigned D = table->rank_to_flat_depth[rank_begin];
    if ((1u << D) == range_len) {               /* flat subtree (D >= 2) */
        rec->kd    = (uint8_t)(PIVCO_SCHED_FLAT | (D << 2));
        rec->param = (uint8_t)rank_begin;
    } else if (range_len == 2) {                /* both children leaves */
        rec->kd    = (uint8_t)PIVCO_SCHED_PAIR;
        rec->param = (uint8_t)rank_begin;       /* == thr */
    } else {
        unsigned split = sched_split_rank(table, level, rank_begin, rank_end);
        if (split == rank_begin + 1) {          /* lone left leaf */
            rec->kd    = (uint8_t)PIVCO_SCHED_LEAF_LEFT;
            rec->param = (uint8_t)rank_begin;   /* == thr; sym = rank_to_sym[param] */
            cnt += build_walk_schedule(table, split, rank_end, level + 1);
        } else {                                /* both children internal */
            rec->kd    = (uint8_t)PIVCO_SCHED_FULL;
            rec->param = (uint8_t)(split - 1);  /* thr */
            cnt += build_walk_schedule(table, rank_begin, split, level + 1);
            cnt += build_walk_schedule(table, split, rank_end, level + 1);
        }
    }
    rec->skip = (uint8_t)cnt;
    return (uint8_t)cnt;
}

static int build_table_finish(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table)
{
    /* Copy lengths to table */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        table->code_len[i] = lengths[i];
    }

    /* Histogram code lengths.  The len>0 guard isn't about correctness
       (sym_count[0] is an unread scratch bin) -- it keeps the unused symbols
       from all piling onto bin 0, whose serial store-to-load-forward chain
       was 5x slower than the (well-predicted) branch on sparse alphabets.
       min/max are derived from the bins below, not inline here. */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (lengths[i] > 0)
            table->sym_count[lengths[i]]++;
    }

    /* Derive min/max code length from the (<=11) length bins. */
    uint8_t max_len = 0, min_len = PIVCO_MAX_CODE_LEN + 1;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        if (table->sym_count[L]) {
            if (L < min_len) min_len = (uint8_t)L;
            max_len = (uint8_t)L;
        }
    }
    table->max_len = max_len;
    table->min_len = min_len;

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
    typedef struct {
        uint8_t  sym;
    } sf_t;
    sf_t  flat_items[PIVCO_MAX_SYMBOLS];
    int   per_len_start[PIVCO_MAX_CODE_LEN + 2];
    {
        /* Counting sort by length: prefix-sum the per-length counts, then a
           single symbol-order pass places each symbol.  Equivalent to the
           old nested for-L/for-s scan but O(256) instead of O(max_len*256). */
        int acc = 0;
        int cursor[PIVCO_MAX_CODE_LEN + 2];
        for (int L = 1; L <= max_len; L++) {
            per_len_start[L] = acc;
            cursor[L] = acc;
            acc += table->sym_count[L];
        }
        per_len_start[max_len + 1] = acc;
        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
            uint8_t L = lengths[s];
            if (L) flat_items[cursor[L]++].sym = (uint8_t)s;
        }
    }

    /* Decompose each c_L into chunks.  Strategy depends on tree mode --
       see pivco_huffman_set_tree_mode().  Default OPTIMIZED matches the
       original production behavior (decompose c_L by its set bits). */
    typedef struct {
        uint16_t L;
        uint16_t bit;       /* 0..PIVCO_MAX_CODE_LEN */
        uint16_t depth;     /* tree-depth of chunk root */
        uint16_t n_syms;    /* 1 << bit */
        uint16_t root_code; /* canonical code of the chunk root (depth bits) */
        int      sym_idx;   /* index into flat_items */
    } chunk_t;
    chunk_t chunks[PIVCO_MAX_SYMBOLS];   /* upper bound: one chunk per symbol */
    int n_chunks = 0;
    pivco_tree_mode_t tree_mode = pivco_huffman_get_tree_mode();

    if (tree_mode == PIVCO_TREE_MODE_NAIVE) {
        /* Every symbol is its own D=0 chunk at depth L. */
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
            int cur = per_len_start[L];
            for (int i = 0; i < c; i++) {
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = 0;
                chunks[n_chunks].depth   = (uint16_t)L;
                chunks[n_chunks].n_syms  = 1;
                chunks[n_chunks].sym_idx = cur + i;
                n_chunks++;
            }
        }
    } else if (tree_mode == PIVCO_TREE_MODE_FUSED) {
        /* D=1 sibling pairs first within each length, then a D=0 singleton
           for the odd-tail symbol.  Sequential reassign in the standard
           depth-sort step gives canonical Huffman codes. */
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
            int cur = per_len_start[L];
            int n_pairs = c / 2;
            int n_singletons = c & 1;
            for (int i = 0; i < n_pairs; i++) {
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = 1;
                chunks[n_chunks].depth   = (uint16_t)(L - 1);
                chunks[n_chunks].n_syms  = 2;
                chunks[n_chunks].sym_idx = cur;
                cur += 2;
                n_chunks++;
            }
            for (int i = 0; i < n_singletons; i++) {
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = 0;
                chunks[n_chunks].depth   = (uint16_t)L;
                chunks[n_chunks].n_syms  = 1;
                chunks[n_chunks].sym_idx = cur;
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
            if (table->sym_count[L]) {
                if (last_L) code = (code + (uint32_t)table->sym_count[last_L]) << (L - last_L);
                fc[L] = code;
                last_L = L;
            }
        }
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
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
                chunks[n_chunks].L        = (uint16_t)L;
                chunks[n_chunks].bit      = (uint16_t)k;
                chunks[n_chunks].depth    = (uint16_t)(L - k);
                chunks[n_chunks].n_syms   = (uint16_t)n;
                chunks[n_chunks].root_code = (uint16_t)(C >> k);
                chunks[n_chunks].sym_idx  = cur;
                cur += n;
                n_chunks++;
                C += (uint32_t)n;
                remaining -= n;
            }
        }
    } else {
        /* OPTIMIZED (default): original bit-decomposition of c_L. */
        for (int L = 1; L <= max_len; L++) {
            int c = table->sym_count[L];
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
                    chunks[n_chunks].L      = (uint16_t)L;
                    chunks[n_chunks].bit    = (uint16_t)bit;
                    chunks[n_chunks].depth  = (uint16_t)depth;
                    chunks[n_chunks].n_syms = (uint16_t)n;
                    chunks[n_chunks].sym_idx = cur;
                    cur += n;
                    n_chunks++;
                }
            }
        }
    }


    /* The code-assignment loops below also fill the rank-range arrays
       (rank_to_sym / rank_to_flat_depth / rank_to_codeword) that the
       production codec walks: once chunks are in code-assignment order,
       their MSB-aligned codewords are strictly increasing, so chunk
       iteration order IS rank order -- the in-order leaf order of the
       (implicit) tree. */

    /* For CANONICAL_FLAT, chunks already carry canonical root_codes; assign
       symbol codes directly from them and skip the depth-sort + sequential
       reassign step (sequential reassign would clobber the canonical
       prefixes when chunks span multiple depths).  All other modes use
       the standard pipeline. */
    if (tree_mode == PIVCO_TREE_MODE_CANONICAL_FLAT) {
        unsigned rank = 0;
        for (int ci = 0; ci < n_chunks; ci++) {
            int bit = chunks[ci].bit;
            int n   = chunks[ci].n_syms;
            int L   = chunks[ci].L;
            uint16_t root = chunks[ci].root_code;
            for (int i = 0; i < n; i++) {
                uint8_t sym = flat_items[chunks[ci].sym_idx + i].sym;
                uint16_t c16 = (uint16_t)(((uint32_t)root << bit) | (uint32_t)i);
                table->code[sym] = c16;
                table->sym_to_rank[sym]         = (uint8_t)rank;
                table->rank_to_sym[rank]        = sym;
                table->rank_to_flat_depth[rank] = (uint8_t)(bit >= 2 ? bit : 0);
                table->rank_to_codeword[rank]   = (uint16_t)(c16 << (16 - L));
                rank++;
            }
        }
        table->num_ranks = (uint16_t)rank;
    } else {
        /* Sort chunks by depth asc (stable; ties keep their natural order
           which is L asc by length, larger-bit-first within length). */
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
        {
            uint32_t code = 0;
            int prev_depth = 0;
            unsigned rank = 0;
            for (int ci = 0; ci < n_chunks; ci++) {
                int d = chunks[ci].depth;
                if (d > prev_depth) code <<= (d - prev_depth);
                chunks[ci].root_code = (uint16_t)code;
                int bit = chunks[ci].bit;
                int n   = chunks[ci].n_syms;
                int L   = chunks[ci].L;
                for (int i = 0; i < n; i++) {
                    uint8_t sym = flat_items[chunks[ci].sym_idx + i].sym;
                    uint16_t c16 = (uint16_t)((code << bit) | (uint32_t)i);
                    table->code[sym] = c16;
                    table->sym_to_rank[sym]         = (uint8_t)rank;
                    table->rank_to_sym[rank]        = sym;
                    table->rank_to_flat_depth[rank] = (uint8_t)(bit >= 2 ? bit : 0);
                    table->rank_to_codeword[rank]   = (uint16_t)(c16 << (16 - L));
                    rank++;
                }
                code += 1;
                prev_depth = d;
            }
            table->num_ranks = (uint16_t)rank;
        }

    }

    /* Populate sorted_symbols / first_sym_idx / first_code from the
       new code assignment.  These fields are not used by runtime
       decoders (only by the tree-walk pass below), but we keep them
       in length-asc order for compatibility with anyone inspecting
       the table. */
    int sorted_idx = per_len_start[max_len + 1];
    for (int i = 0; i < sorted_idx; i++)
        table->sorted_symbols[i] = flat_items[i].sym;
    for (int len = 1; len <= max_len; len++) {
        table->first_sym_idx[len] = (uint16_t)per_len_start[len];
        uint16_t min_code = 0xFFFF;
        for (int i = per_len_start[len]; i < per_len_start[len + 1]; i++) {
            uint16_t c = table->code[flat_items[i].sym];
            if (c < min_code) min_code = c;
        }
        table->first_code[len] = (min_code == 0xFFFF) ? 0 : min_code;
    }

    /* The 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) is used
     * ONLY by the traditional flat-table decoder (trad_huffman_decode*), never
     * by the production tree-walk path.  It is built on demand via
     * pivco_huffman_build_traditional_table() so the normal build -- and the
     * decode-side rebuild from code lengths -- don't pay for the 2 KB fill. */


    /* The explicit tree (tree[] / node_type[] / flat_*[] / split_rank[] /
     * flat_base_rank[] / max_leaf_depth[]) is NOT built here: the codec
     * walks the schedule below and never reads it.  Analysis tools that
     * want it call pivco_huffman_build_explicit_tree() — its
     * materialization (a spine walk per chunk plus classify,
     * max_leaf_depth and in-order-rank passes over up-to-511-node
     * arrays) was the bulk of small-input table-build time (issue #7). */

    /* Pre-order walk schedule: render the rank-range tree into the
     * program the codec streams (see build_walk_schedule above). */
    table->sched_len = 0;
    build_walk_schedule(table, 0, table->num_ranks, 0);

    fill_enc_init_aux(table);   /* x86 2tab/4tab gather tables (or NULL elsewhere) */

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

    int n_used = 0, last = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++)
        if (code_lens[i] > 0) { n_used++; last = i; }
    if (n_used == 0) return PIVCO_ERR_EMPTY;
    table->num_symbols = (uint16_t)n_used;

    if (n_used == 1) {
        build_single_symbol_table(last, table);
        return PIVCO_OK;
    }
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
     * sorted_symbols[0] = shortest-code (most frequent) symbol. */
    memset(table->decode_sym, table->sorted_symbols[0], sizeof(table->decode_sym));
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
    t->tree[id].symbol = (int16_t)t->rank_to_sym[rank];
    t->tree[id].left   = -1;
    t->tree[id].right  = -1;
    t->node_type[id] = (uint8_t)PIVCO_NODE_LEAF;
    return id;
}

static int16_t etree_walk(etree_ctx_t *c, unsigned rank_begin, unsigned rank_end)
{
    pivco_huffman_table_t *t = c->t;
    if (rank_end - rank_begin == 1) return etree_leaf(c, rank_begin);

    const pivco_sched_rec_t *rec = &t->sched[c->cursor++];
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
        memcpy(&t->flat_code_to_sym[c->pool], &t->rank_to_sym[rec->param],
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
    if (!table || table->num_ranks < 2) return;
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
    table->tree_root = etree_walk(&c, 0, table->num_ranks);
    table->tree_node_count = c.node_count;
    PIVCO_CHECK(table->tree_root == 0);
    PIVCO_CHECK(c.cursor == table->sched_len);
}

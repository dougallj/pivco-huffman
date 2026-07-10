/* bench_rank_range_table: prototype of the issue-#7 rank-range table build.
 *
 * terrelln's OpenZL port (github.com/MarcinZukowski/pivco-huffman/issues/7)
 * replaces the explicit pivco tree with an implicit one: leaves sorted by
 * MSB-aligned codeword ("rank" order) make every subtree a contiguous rank
 * range [rank_begin, rank_end), so the whole tree is three arrays --
 * rank_to_codeword, rank_to_flat_depth, rank_to_sym -- and two operations:
 *
 *   leaf test    (1 << rank_to_flat_depth[rank_begin]) == rank_end - rank_begin
 *                (flat depth 0 = single-symbol leaf, D>=2 = flat subtree,
 *                 matching table->flat_depth semantics)
 *   split        at tree level L the range's codewords read 0..0 1..1 at bit
 *                L, so a lazy linear scan for the first set bit finds the
 *                left/right boundary; each node needs its split exactly once
 *                per walk, so nothing is precomputed.
 *
 * This file contains (a) rr_build_from_code_lens -- the rank-range analog of
 * pivco_huffman_build_table_from_code_lens, sharing the chunk decomposition
 * that defines the (non-canonical) OPTIMIZED tree shape but skipping the tree
 * materialization, node classification, max_leaf_depth DFS, and recursive
 * rank assignment -- and (b) a verifier that recursively walks the rank
 * ranges in lockstep with the production tree and checks:
 *
 *   - per-symbol code / code_len identical to the production build
 *   - rank order identical (sym_to_rank matches the production build)
 *   - every split matches split_rank[], every flat root matches
 *     flat_depth[] / flat_base_rank[] / flat_code_to_sym[]
 *   - node_type[] is recoverable from the range shape alone (the decode
 *     dispatch needs no side table), and max_leaf_depth[] is recoverable as
 *     max code_len in range minus level (the encoder's u8-repack test)
 *
 * OPTIMIZED tree mode only (the production default).  Not benchmarked here;
 * this is a correctness prototype for the representation.
 */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void        bench_init(void);
extern int         bench_num_distributions(void);
extern const char *bench_dist_name(int idx);
extern void        bench_generate_symbols(int dist_idx, uint8_t *symbols,
                                          int n_symbols, unsigned seed);

#define N_SYMBOLS  (1 << 16)
#define SEED       0xC0FFEEu

/* ---------------- rank-range table (the entire "tree") ---------------- */

typedef struct {
    uint8_t  sym_to_rank[PIVCO_MAX_SYMBOLS];
    uint8_t  rank_to_sym[PIVCO_MAX_SYMBOLS];
    uint8_t  rank_to_flat_depth[PIVCO_MAX_SYMBOLS]; /* 0 = plain leaf; D>=2 = flat */
    uint16_t rank_to_codeword[PIVCO_MAX_SYMBOLS];   /* MSB-aligned into 16 bits */
    uint16_t code[PIVCO_MAX_SYMBOLS];               /* encode side, same as table->code */
    uint8_t  code_len[PIVCO_MAX_SYMBOLS];
    uint16_t num_ranks;
    uint8_t  max_len;
    uint8_t  min_len;
} rr_table_t;

/* Root range is [0, num_ranks) at level 0. */

static int rr_range_is_leaf(const rr_table_t *rt, unsigned rb, unsigned re)
{
    return (1u << rt->rank_to_flat_depth[rb]) == re - rb;
}

/* First rank of the right child: lazy scan for the first codeword with bit
 * `level` set.  Precondition: internal-node range (rb + 1 < re), so both a
 * clear-bit prefix and a set-bit suffix exist. */
static unsigned rr_split(const rr_table_t *rt, unsigned level,
                         unsigned rb, unsigned re)
{
    uint16_t mask = (uint16_t)(0x8000u >> level);
    unsigned r = rb + 1;
    while (r < re && (rt->rank_to_codeword[r] & mask) == 0)
        r++;
    return r;
}

/* Rank-range analog of pivco_huffman_build_table_from_code_lens, OPTIMIZED
 * mode.  Everything through chunk code assignment is the same pipeline the
 * production build runs (it defines the tree shape / codes and must match
 * bit-for-bit); the difference is what happens after: one flat pass over the
 * depth-sorted chunks fills codes AND all rank arrays -- chunks in canonical
 * depth order have strictly increasing MSB-aligned codes, so chunk iteration
 * order IS rank order, the in-order leaf order of the tree. */
static int rr_build_from_code_lens(const uint8_t lengths[PIVCO_MAX_SYMBOLS],
                                   rr_table_t *rt)
{
    memset(rt, 0, sizeof(*rt));

    uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1] = {0};
    int n_used = 0, last = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (lengths[s] > 0) {
            if (lengths[s] > PIVCO_MAX_CODE_LEN) return PIVCO_ERR_CORRUPT;
            sym_count[lengths[s]]++;
            n_used++;
            last = s;
        }
    }
    if (n_used == 0) return PIVCO_ERR_EMPTY;

    if (n_used == 1) {
        /* Degenerate 1-bit code; a single rank whose range is the root. */
        rt->code[last] = 0;
        rt->code_len[last] = 1;
        rt->rank_to_sym[0] = (uint8_t)last;
        rt->rank_to_codeword[0] = 0;
        rt->num_ranks = 1;
        rt->max_len = rt->min_len = 1;
        return PIVCO_OK;
    }

    uint8_t max_len = 0, min_len = PIVCO_MAX_CODE_LEN + 1;
    for (int L = 1; L <= PIVCO_MAX_CODE_LEN; L++) {
        if (sym_count[L]) {
            if (L < min_len) min_len = (uint8_t)L;
            max_len = (uint8_t)L;
        }
    }
    rt->max_len = max_len;
    rt->min_len = min_len;

    /* Counting-sort symbols by length (symbol-value order within a length),
     * same as build_table_finish. */
    uint8_t items[PIVCO_MAX_SYMBOLS];
    int per_len_start[PIVCO_MAX_CODE_LEN + 2];
    {
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

    /* OPTIMIZED chunk decomposition: per length L, one chunk per set bit of
     * c_L, larger bits first. */
    typedef struct {
        uint16_t L;
        uint16_t bit;
        uint16_t depth;
        int      sym_idx;
    } chunk_t;
    chunk_t chunks[PIVCO_MAX_SYMBOLS];
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
                chunks[n_chunks].L       = (uint16_t)L;
                chunks[n_chunks].bit     = (uint16_t)bit;
                chunks[n_chunks].depth   = (uint16_t)depth;
                chunks[n_chunks].sym_idx = cur;
                cur += 1 << bit;
                n_chunks++;
            }
        }
    }

    /* Stable sort by chunk-root depth asc (same insertion sort as the
     * production build; ties keep L-asc / larger-bit-first order). */
    for (int i = 1; i < n_chunks; i++) {
        chunk_t cur = chunks[i];
        int j = i - 1;
        while (j >= 0 && chunks[j].depth > cur.depth) {
            chunks[j + 1] = chunks[j];
            j--;
        }
        chunks[j + 1] = cur;
    }

    /* Canonical chunk-code assignment fused with rank filling.  This one
     * loop replaces the production build's tree construction, node
     * classification, max_leaf_depth DFS and in-order rank recursion. */
    {
        uint32_t code = 0;
        int prev_depth = 0;
        unsigned rank = 0;
        for (int ci = 0; ci < n_chunks; ci++) {
            int d   = chunks[ci].depth;
            int bit = chunks[ci].bit;
            int L   = chunks[ci].L;
            int n   = 1 << bit;
            if (d > prev_depth) code <<= (d - prev_depth);
            for (int i = 0; i < n; i++) {
                uint8_t sym = items[chunks[ci].sym_idx + i];
                uint16_t c16 = (uint16_t)((code << bit) | (uint32_t)i);
                rt->code[sym]     = c16;
                rt->code_len[sym] = (uint8_t)L;
                rt->sym_to_rank[sym]  = (uint8_t)rank;
                rt->rank_to_sym[rank] = sym;
                rt->rank_to_codeword[rank]   = (uint16_t)(c16 << (16 - L));
                rt->rank_to_flat_depth[rank] = (uint8_t)(bit >= 2 ? bit : 0);
                rank++;
            }
            code += 1;
            prev_depth = d;
        }
        rt->num_ranks = (uint16_t)rank;
    }
    return PIVCO_OK;
}

/* ---------------- structural-equivalence verifier ---------------- */

static int g_fails;

#define FAIL(...) do { \
    fprintf(stderr, "  FAIL: " __VA_ARGS__); \
    fputc('\n', stderr); \
    g_fails++; \
} while (0)

/* Max code_len over a rank range == deepest leaf below the node; minus the
 * node's level it must reproduce table->max_leaf_depth[]. */
static int rr_range_max_len(const rr_table_t *rt, unsigned rb, unsigned re)
{
    int mx = 0;
    for (unsigned r = rb; r < re; r++) {
        int L = rt->code_len[rt->rank_to_sym[r]];
        if (L > mx) mx = L;
    }
    return mx;
}

/* Walk the production tree and the rank ranges in lockstep. */
static void verify_walk(const pivco_huffman_table_t *t, const rr_table_t *rt,
                        int16_t node, unsigned rb, unsigned re, unsigned level)
{
    const pivco_tree_node_t *n = &t->tree[node];

    if (n->symbol >= 0) {                                  /* plain leaf */
        if (re - rb != 1)
            FAIL("leaf node %d: range [%u,%u) not a single rank", node, rb, re);
        if (rt->rank_to_flat_depth[rb] != 0)
            FAIL("leaf rank %u: flat_depth %u != 0", rb, rt->rank_to_flat_depth[rb]);
        if (rt->rank_to_sym[rb] != n->symbol)
            FAIL("leaf rank %u: sym %u != tree sym %d",
                 rb, rt->rank_to_sym[rb], n->symbol);
        if (!rr_range_is_leaf(rt, rb, re))
            FAIL("leaf rank %u: rr leaf test says internal", rb);
        return;
    }

    if (t->flat_depth[node] >= 2) {                        /* flat subtree */
        unsigned D = t->flat_depth[node];
        if (rt->rank_to_flat_depth[rb] != D)
            FAIL("flat node %d: flat_depth %u != %u",
                 node, rt->rank_to_flat_depth[rb], D);
        if (re - rb != (1u << D))
            FAIL("flat node %d: range len %u != 2^%u", node, re - rb, D);
        if (t->flat_base_rank[node] != rb)
            FAIL("flat node %d: base rank %u != %u",
                 node, t->flat_base_rank[node], rb);
        if (!rr_range_is_leaf(rt, rb, re))
            FAIL("flat node %d: rr leaf test says internal", node);
        for (unsigned i = 0; i < (1u << D) && rb + i < re; i++) {
            uint8_t want = t->flat_code_to_sym[t->flat_offset[node] + i];
            if (rt->rank_to_sym[rb + i] != want)
                FAIL("flat node %d code %u: sym %u != %u",
                     node, i, rt->rank_to_sym[rb + i], want);
        }
        if (t->node_type[node] != PIVCO_NODE_INTERNAL_FLAT)
            FAIL("flat node %d: node_type %u", node, t->node_type[node]);
        int want_mld = rr_range_max_len(rt, rb, re) - (int)level;
        if (t->max_leaf_depth[node] != want_mld)
            FAIL("flat node %d: max_leaf_depth %u != range-derived %d",
                 node, t->max_leaf_depth[node], want_mld);
        return;
    }

    /* Internal node: the range must NOT test as a leaf, the lazy split must
     * land exactly on split_rank[]+1, and node_type / max_leaf_depth must be
     * recoverable from the range shape alone. */
    if (rr_range_is_leaf(rt, rb, re))
        FAIL("internal node %d: rr leaf test says leaf on [%u,%u)", node, rb, re);

    unsigned split = rr_split(rt, level, rb, re);
    if (split <= rb || split >= re) {
        FAIL("internal node %d: split %u outside (%u,%u)", node, split, rb, re);
        return;
    }
    if (t->split_rank[node] != split - 1)
        FAIL("internal node %d: split_rank %u != rr split-1 %u",
             node, t->split_rank[node], split - 1);

    int left_leaf  = rr_range_is_leaf(rt, rb, split) &&
                     rt->rank_to_flat_depth[rb] == 0;
    int right_leaf = rr_range_is_leaf(rt, split, re) &&
                     rt->rank_to_flat_depth[split] == 0;
    uint8_t want_type = (left_leaf && right_leaf) ? PIVCO_NODE_BOTH_LEAVES
                       : left_leaf                ? PIVCO_NODE_LEAF_LEFT
                                                  : PIVCO_NODE_INTERNAL_FULL;
    if (t->node_type[node] != want_type)
        FAIL("internal node %d: node_type %u != range-derived %u",
             node, t->node_type[node], want_type);

    int want_mld = rr_range_max_len(rt, rb, re) - (int)level;
    if (t->max_leaf_depth[node] != want_mld)
        FAIL("internal node %d: max_leaf_depth %u != range-derived %d",
             node, t->max_leaf_depth[node], want_mld);

    verify_walk(t, rt, n->left,  rb,    split, level + 1);
    verify_walk(t, rt, n->right, split, re,    level + 1);
}

static void verify_case(const char *name, const uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    int fails_before = g_fails;

    pivco_huffman_table_t *t = calloc(1, sizeof(*t));
    if (pivco_huffman_build_table(freq, t) != PIVCO_OK) {
        FAIL("%s: reference build_table failed", name);
        free(t);
        return;
    }
    pivco_huffman_build_explicit_tree(t);  /* reference tree is on-demand now */

    rr_table_t rt;
    if (rr_build_from_code_lens(t->code_len, &rt) != PIVCO_OK) {
        FAIL("%s: rr build failed", name);
        free(t);
        return;
    }

    if (rt.num_ranks != t->num_symbols)
        FAIL("%s: num_ranks %u != num_symbols %u",
             name, rt.num_ranks, t->num_symbols);

    if (t->num_symbols == 1) {
        /* Reference builds a degenerate two-leaf tree for the 1-bit code;
         * the rank table is just one rank, so skip the lockstep walk. */
        int sym = rt.rank_to_sym[0];
        if (rt.code_len[sym] != 1 || rt.code[sym] != 0)
            FAIL("%s: single-symbol code/len mismatch", name);
        goto done;
    }

    /* Per-symbol codes and ranks must be byte-identical to production. */
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) {
        if (t->code_len[s] == 0) continue;
        if (rt.code_len[s] != t->code_len[s] || rt.code[s] != t->code[s])
            FAIL("%s: sym %d code %u/%u != %u/%u", name, s,
                 rt.code[s], rt.code_len[s], t->code[s], t->code_len[s]);
        if (rt.sym_to_rank[s] != t->sym_to_rank[s])
            FAIL("%s: sym %d rank %u != %u", name, s,
                 rt.sym_to_rank[s], t->sym_to_rank[s]);
    }
    if (rt.max_len != t->max_len || rt.min_len != t->min_len)
        FAIL("%s: min/max len %u/%u != %u/%u", name,
             rt.min_len, rt.max_len, t->min_len, t->max_len);

    /* Rank order must be strictly increasing MSB-aligned codewords (the
     * invariant the lazy split scan relies on). */
    for (unsigned r = 1; r < rt.num_ranks; r++)
        if (rt.rank_to_codeword[r] <= rt.rank_to_codeword[r - 1])
            FAIL("%s: rank_to_codeword not strictly increasing at %u", name, r);

    verify_walk(t, &rt, t->tree_root, 0, rt.num_ranks, 0);

done:
    printf("  %-24s %s (%u ranks, max_len %u)\n", name,
           g_fails == fails_before ? "OK" : "FAIL", rt.num_ranks, rt.max_len);
    free(t);
}

int main(void)
{
    bench_init();
    pivco_huffman_set_tree_mode(PIVCO_TREE_MODE_OPTIMIZED);

    printf("rank-range table build prototype (issue #7) -- equivalence check\n");

    printf("synthetic edge cases:\n");
    {
        uint64_t freq[PIVCO_MAX_SYMBOLS];

        memset(freq, 0, sizeof(freq));
        freq['A'] = 1000;
        verify_case("single_symbol", freq);

        memset(freq, 0, sizeof(freq));
        freq['A'] = 1000; freq['B'] = 1;
        verify_case("two_symbols", freq);

        memset(freq, 0, sizeof(freq));
        freq[0] = 1; freq[1] = 1; freq[2] = 1000;
        verify_case("three_skewed", freq);

        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) freq[s] = 1;
        verify_case("uniform_256", freq);

        for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++) freq[s] = (uint64_t)s + 1;
        verify_case("ramp_256", freq);

        /* Geometric: unlimited Huffman would exceed PIVCO_MAX_CODE_LEN,
         * exercising limit_code_lengths upstream of both builds. */
        memset(freq, 0, sizeof(freq));
        uint64_t f = 1;
        for (int s = 0; s < 40; s++) { freq[s] = f; if (f < (1ull << 40)) f *= 2; }
        verify_case("geometric_40", freq);
    }

    printf("bench distributions (n=%d, seed 0x%X):\n", N_SYMBOLS, SEED);
    uint8_t *sym = malloc(N_SYMBOLS);
    if (!sym) return 1;
    for (int d = 0; d < bench_num_distributions(); d++) {
        uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
        bench_generate_symbols(d, sym, N_SYMBOLS, SEED);
        for (int i = 0; i < N_SYMBOLS; i++) freq[sym[i]]++;
        verify_case(bench_dist_name(d), freq);
    }
    free(sym);

    if (g_fails) {
        printf("%d FAILURES\n", g_fails);
        return 1;
    }
    printf("all cases OK: rank-range build reproduces the production tree\n");
    return 0;
}

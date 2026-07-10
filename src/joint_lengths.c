/* joint_lengths.c — joint code-length / flat-shape optimization.
 *
 * Replaces the Huffman-optimal code lengths with lengths that trade a
 * little compressed size for a flatter decode tree (fewer merge
 * passes), per docs/JOINT-LENGTHS.md.  Encoder-side only: the wire
 * carries lengths, and the decoder derives the (identical) chunk
 * structure from them, so any decoder — including pre-branch ones —
 * reads the output.
 *
 * Model (variant B of the doc): choose at most one chunk (L, b) per
 * (length L <= 11, flat depth b <= min(8, L)); a chunk holds 2^b
 * symbols at length L inside a depth-b flat, so each of its symbols'
 * occurrences costs L bits and (L - b) merge passes.  Constraints:
 * chunk-root Kraft equality and total symbol count.  Objective:
 * minimize  sum n_s * (l_s + lambda * (l_s - D_s)).  For a fixed chunk
 * multiset the optimal symbol assignment is sorted-to-sorted
 * (rearrangement inequality), so processing chunk types in global
 * per-occurrence-cost order makes a 0/1 knapsack DP over
 * (symbols placed, Kraft mass used) exact.  lambda = 0 degenerates to
 * an optimal length-limited code (== the Huffman + limit baseline), so
 * the knob's off state simply skips this pass.
 *
 * DP size: <= ~70 items x 257 x 2049 states; ~10 MB transient scratch,
 * a few ms — runs once per table build.
 */

#include "pivco_huffman.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define JL_LMAX  PIVCO_MAX_CODE_LEN            /* 11 */
#define JL_MASS  (1 << JL_LMAX)                /* Kraft units of 2^-LMAX */
#define JL_MAXITEMS 128                        /* (L,b) pairs; 75 actual */

static double g_joint_lambda = 0.0;

void pivco_huffman_set_joint_lambda(double lam)
{
    g_joint_lambda = (lam > 0.0 && lam < 100.0) ? lam : 0.0;
}

double pivco_huffman_get_joint_lambda(void)
{
    return g_joint_lambda;
}

typedef struct {
    double   cost;      /* per-occurrence: L + lambda * (L - b) */
    uint8_t  L, b;
    uint16_t size;      /* 1 << b symbols */
    uint16_t mass;      /* 1 << (LMAX - L + b) Kraft units */
} jl_item_t;

static int jl_cmp_item(const void *a, const void *b)
{
    double d = ((const jl_item_t *)a)->cost - ((const jl_item_t *)b)->cost;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}

typedef struct { uint64_t freq; uint8_t sym; } jl_sf_t;

static int jl_cmp_sf(const void *a, const void *b)
{
    const jl_sf_t *x = a, *y = b;
    if (x->freq != y->freq) return x->freq > y->freq ? -1 : 1;
    return (int)x->sym - (int)y->sym;   /* deterministic ties */
}

/* Overwrites lengths[] with the joint-optimal assignment.  Returns 0 on
 * success, -1 on any failure (caller keeps the Huffman lengths). */
int pivco_joint_optimize_lengths(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                 uint8_t lengths[PIVCO_MAX_SYMBOLS])
{
    const double lam = g_joint_lambda;
    if (lam <= 0.0) return -1;

    /* Symbols sorted by frequency desc + prefix sums. */
    jl_sf_t sf[PIVCO_MAX_SYMBOLS];
    int sigma = 0;
    for (int s = 0; s < PIVCO_MAX_SYMBOLS; s++)
        if (freq[s]) { sf[sigma].freq = freq[s]; sf[sigma].sym = (uint8_t)s; sigma++; }
    if (sigma < 2 || sigma > (1 << JL_LMAX)) return -1;
    qsort(sf, (size_t)sigma, sizeof(jl_sf_t), jl_cmp_sf);
    double P[PIVCO_MAX_SYMBOLS + 1];
    P[0] = 0.0;
    for (int i = 0; i < sigma; i++) P[i + 1] = P[i] + (double)sf[i].freq;

    /* Chunk items in per-occurrence cost order. */
    jl_item_t items[JL_MAXITEMS];
    int n_items = 0;
    for (int L = 1; L <= JL_LMAX; L++) {
        int bmax = L < 8 ? L : 8;
        for (int b = 0; b <= bmax; b++) {
            if ((1 << b) > sigma) break;
            items[n_items].cost = (double)L + lam * (double)(L - b);
            items[n_items].L    = (uint8_t)L;
            items[n_items].b    = (uint8_t)b;
            items[n_items].size = (uint16_t)(1 << b);
            items[n_items].mass = (uint16_t)(1 << (JL_LMAX - L + b));
            n_items++;
        }
    }
    qsort(items, (size_t)n_items, sizeof(jl_item_t), jl_cmp_item);

    /* 0/1 knapsack over (k symbols placed, m Kraft units used); chosen
     * set tracked as a 128-bit mask per state.  Descending (k, m)
     * iteration gives 0/1 semantics with in-place update. */
    const size_t nstates = (size_t)(sigma + 1) * (JL_MASS + 1);
    float    *dp   = malloc(nstates * sizeof(float));
    uint64_t *msk0 = malloc(nstates * sizeof(uint64_t));
    uint64_t *msk1 = malloc(nstates * sizeof(uint64_t));
    if (!dp || !msk0 || !msk1) { free(dp); free(msk0); free(msk1); return -1; }
    for (size_t i = 0; i < nstates; i++) dp[i] = INFINITY;
    memset(msk0, 0, nstates * sizeof(uint64_t));
    memset(msk1, 0, nstates * sizeof(uint64_t));
    dp[0] = 0.0f;

#define JL_IX(k, m) ((size_t)(k) * (JL_MASS + 1) + (size_t)(m))
    for (int it = 0; it < n_items; it++) {
        const int   size = items[it].size, mass = items[it].mass;
        const uint64_t bit_lo = it < 64 ? (1ull << it) : 0;
        const uint64_t bit_hi = it >= 64 ? (1ull << (it - 64)) : 0;
        for (int k = sigma - size; k >= 0; k--) {
            const float add = (float)(items[it].cost * (P[k + size] - P[k]));
            const size_t src_row = JL_IX(k, 0), dst_row = JL_IX(k + size, mass);
            for (int m = JL_MASS - mass; m >= 0; m--) {
                const float v = dp[src_row + (size_t)m];
                if (!(v < INFINITY)) continue;
                const float cand = v + add;
                if (cand < dp[dst_row + (size_t)m]) {
                    dp[dst_row + (size_t)m]   = cand;
                    msk0[dst_row + (size_t)m] = msk0[src_row + (size_t)m] | bit_lo;
                    msk1[dst_row + (size_t)m] = msk1[src_row + (size_t)m] | bit_hi;
                }
            }
        }
    }

    int rc = -1;
    if (dp[JL_IX(sigma, JL_MASS)] < INFINITY) {
        /* Deal freq-sorted symbols to chosen chunks in cost order. */
        const uint64_t m0 = msk0[JL_IX(sigma, JL_MASS)];
        const uint64_t m1 = msk1[JL_IX(sigma, JL_MASS)];
        int cur = 0;
        for (int it = 0; it < n_items; it++) {
            const int on = it < 64 ? (int)((m0 >> it) & 1)
                                   : (int)((m1 >> (it - 64)) & 1);
            if (!on) continue;
            for (int j = 0; j < items[it].size; j++)
                lengths[sf[cur++].sym] = items[it].L;
        }
        rc = (cur == sigma) ? 0 : -1;
    }
    free(dp); free(msk0); free(msk1);
    return rc;
}

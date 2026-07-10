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
 * DP size: the production path (lambda <= 1/7) runs the slot-ledger DP
 * below on a diagonal-major, parity-compact, capacity-banded lattice —
 * ~1 MB transient scratch, tens of microseconds.  Larger lambda falls
 * back to the original mass DP (~10 MB, a few ms).
 */

#include "pivco_huffman.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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

/* ---------- Fast slot-ledger DP (exact for lambda <= 1/7) ----------
 *
 * The mass DP below tracks Kraft usage in 2^-11 units (state
 * (k, m <= 2048), ~10 MB, ~6 ms at sigma=256).  Expressed instead in
 * OPEN SLOTS at the current level, the ledger is tiny: a state is
 * (k symbols placed, s open slots), and Kraft EQUALITY forces
 * s <= sigma - k at every level (each remaining symbol fills at most
 * one current-level slot's worth of mass).
 *
 * Levels are processed ascending, bits within a level descending.
 * That order equals GLOBAL slot-cost order -- the requirement for the
 * sorted-matching exactness argument (see docs/JOINT-LENGTHS.md and
 * RESPONSE.md #1) -- precisely when cost(L, b) = L + lam*(L - b) is
 * monotone across levels for every b, i.e. max_b cost(L, b) <=
 * min_b cost(L+1, b):  L(1+lam) <= (L+1)(1+lam) - 8*lam  <=>
 * lam <= 1/7.  For larger lambda the caller falls back to the exact
 * mass DP.
 *
 * Three structural facts turn the plane walk into an L1-resident
 * kernel:
 *
 * DIAGONALS.  A take (k, s) -> (k + 2^b, s - 2^b) preserves t = k + s,
 * so within a level the DP decomposes into independent diagonals.
 * Stored diagonal-major ([t] rows, k along the row), all take sweeps
 * of a level run over one <~0.5 KB row that stays in L1; the plane is
 * traversed once per level (the doubling), not once per sweep.
 *
 * PARITY.  Level-entry states have even s (they come from the
 * doubling s' = 2s) and takes with b >= 1 preserve s-parity, so the
 * live lattice is k == t (mod 2): compact index j = (k - (t&1))/2
 * halves each row and makes it stride-1 dense.  b = 0 — the only
 * parity flip, always last in the level's cost order — is folded into
 * the doubling (an odd-s cell's unique source is its even-lattice
 * predecessor plus one lone leaf) and reconstructed from s-parity at
 * backtrack.  Level LMAX never takes b = 0: entry s is even and the
 * terminal needs its takes to sum to s exactly, so bit 0 of c_LMAX
 * is always clear (deepest-level leaf counts are even).
 *
 * CAPACITY BAND.  A state at level L can place at most s * 2^h more
 * symbols (h = LMAX - L; push everything to depth LMAX), so
 * sigma - k <= (t - k) << h is necessary — and met by every
 * completing trajectory, making the prune exact.  In diagonal
 * coordinates: rows t >= ceil(sigma / 2^h) with per-row cap
 * k <= (t*2^h - sigma)/(2^h - 1).  Level 11 collapses to the single
 * diagonal t = sigma, level 10 loses half its rows, level 9 three
 * quarters.  Feasibility is preserved cell-to-cell by takes (same
 * condition) and by the doubling (dest feasible at L+1 <=> even
 * source feasible at L; the folded-b0 predecessor is implied with
 * 2^h >= 2 slack), so pruned — hence stale — cells are never read.
 *
 * Terminal: (k = sigma, s = 0), i.e. cell k = sigma of diagonal
 * t = sigma after level 11.  Per-level u16 pick rows (bits 1..8; bit
 * 0 is implicit in parity) are archived per diagonal for backtrack.
 */
#define JL_WMAX 132     /* max compact row: j <= 128, padded to x4 */

/* Largest compact index j on diagonal t whose k = 2j + (t&1) can still
 * feed sigma - k leaves through (t - k) slots h levels above the
 * bottom; -1 if the whole row is infeasible. */
static inline int jl_row_jcap(int t, int h, int sigma)
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

static double jl_solve_slots(const double *P, int sigma, double lam,
                             uint16_t out_BL[JL_LMAX + 1])
{
    const int W = (((sigma >> 1) + 2) + 3) & ~3;   /* compact row width */
    const size_t plane = (size_t)(sigma + 1) * (size_t)W;
    float    *cost = malloc(plane * sizeof(float));
    uint16_t *arch = malloc((size_t)JL_LMAX * plane * sizeof(uint16_t));
    float     dPt[2][9][JL_WMAX];   /* [t&1][b][j]: P[k + 2^b] - P[k] */
    float     dP0[PIVCO_MAX_SYMBOLS + 1];          /* P[k] - P[k-1]   */
    if (!cost || !arch) { free(cost); free(arch); return -1.0; }

    for (int p = 0; p < 2; p++)
        for (int b = 1; b <= 8; b++) {
            const int cnk = 1 << b;
            for (int j = 0; j < W; j++) {
                const int k = 2 * j + p;
                dPt[p][b][j] = k + cnk <= sigma
                             ? (float)(P[k + cnk] - P[k]) : 0.0f;
            }
        }
    dP0[0] = 0.0f;
    for (int k = 1; k <= sigma; k++) dP0[k] = (float)(P[k] - P[k - 1]);

    int tlo[JL_LMAX + 1], thi[JL_LMAX + 1];
    for (int L = 1; L <= JL_LMAX; L++) {
        const int h = JL_LMAX - L;
        thi[L] = (1 << L) > sigma ? sigma : (1 << L);
        tlo[L] = (sigma + (1 << h) - 1) >> h;
        if (tlo[L] < 1) tlo[L] = 1;
    }

    /* Lazy init: every row is fully written by the doubling that
     * produces its level (INF outside the feasible range), so only
     * the level-1 band rows need priming. */
    for (int t = tlo[1]; t <= thi[1]; t++)
        for (int j = 0; j < W; j++) cost[(size_t)t * W + j] = INFINITY;
    cost[2 * W + 0] = 0.0f;      /* level-1 entry: k = 0, s = 2, t = 2 */

    for (int L = 1; L <= JL_LMAX; L++) {
        const int h = JL_LMAX - L;
        const int bmax = L < 8 ? L : 8;
        uint16_t *archL = arch + (size_t)(L - 1) * plane;
        for (int t = tlo[L]; t <= thi[L]; t++) {
            const int p = t & 1;
            const int jcap = jl_row_jcap(t, h, sigma);
            if (jcap < 0) continue;
            float *row = cost + (size_t)t * W;
            uint16_t *prow = archL + (size_t)t * W;   /* pick, archived
                                                       * in place      */
            memset(prow, 0, (size_t)(jcap + 1) * sizeof(uint16_t));
            for (int b = bmax; b >= 1; b--) {
                const int jstep = 1 << (b - 1);       /* = 2^b slots / 2 */
                const int jhi = jcap - jstep;         /* dest j <= jcap  */
                if (jhi < 0) continue;
                const float a = (float)((double)L + lam * (double)(L - b));
                const float *dpb = dPt[p][b];
                int j = jhi;
                /* 0/1 in-place: dest j + jstep > src j, so iterate j
                 * descending — a written dest is never re-read as a
                 * source for the same chunk (within a 4-block, loads
                 * precede stores, which is the same pre-update read).
                 * Stores are unconditional: everything is L1-resident,
                 * so blending beats the data-dependent branch of a
                 * "did anything improve" early-out. */
#if defined(__aarch64__)
                const float32x4_t va = vdupq_n_f32(a);
                const uint16x4_t vbit = vdup_n_u16((uint16_t)(1u << b));
                for (; j >= 3; j -= 4) {
                    const int base = j - 3;
                    float32x4_t src = vld1q_f32(row + base);
                    float32x4_t cand = vfmaq_f32(src, vld1q_f32(dpb + base), va);
                    float32x4_t dst = vld1q_f32(row + base + jstep);
                    uint32x4_t m = vcltq_f32(cand, dst);
                    vst1q_f32(row + base + jstep, vbslq_f32(m, cand, dst));
                    uint16x4_t pm = vmovn_u32(m);
                    uint16x4_t pv = vorr_u16(vld1_u16(prow + base), vbit);
                    uint16x4_t qv = vld1_u16(prow + base + jstep);
                    vst1_u16(prow + base + jstep, vbsl_u16(pm, pv, qv));
                }
#endif
                for (; j >= 0; j--) {
                    const float v = row[j];
                    if (!(v < INFINITY)) continue;
                    const float cand = v + a * dpb[j];
                    if (cand < row[j + jstep]) {
                        row[j + jstep] = cand;
                        prow[j + jstep] = (uint16_t)(prow[j] | (1u << b));
                    }
                }
            }
        }
        if (L == JL_LMAX) break;
        /* Doubling s' = 2s with the b = 0 take folded in.  Dest cell
         * (t', k) has the unique source (t = (t'+k)/2, k): even-lattice
         * there if k == t (mod 2), else the odd-s product of a lone
         * leaf taken at level L from (t, k-1).  In place, t' and j'
         * descending: sources live on rows <= t', and the single
         * same-row read (t = t', only at k = t') happens before its
         * cell is overwritten.
         *
         * Branchless: on dest row t' the source diagonal is
         * t = t0 + j' (t0 = (t'+p')/2), so the level-L band check
         * hoists to a j'-range, and the source parity d = (t^k)&1
         * alternates with j' — two constant-stride subloops with the
         * unified source index (k - d - (t&1))/2.  The subloop that
         * contains the top cell runs first (it holds the only
         * same-row read). */
        const float a0 = (float)((double)L * (1.0 + lam));
        for (int tp = thi[L + 1]; tp >= tlo[L + 1]; tp--) {
            const int pp = tp & 1;
            const int jcap2 = jl_row_jcap(tp, h - 1, sigma);
            if (jcap2 < 0) continue;
            float *nrow = cost + (size_t)tp * W;
            const int t0 = (tp + pp) >> 1;
            int jlo = tlo[L] - t0; if (jlo < 0) jlo = 0;
            int jhi2 = thi[L] - t0; if (jhi2 > jcap2) jhi2 = jcap2;
            for (int jp = jcap2; jp > jhi2; jp--) nrow[jp] = INFINITY;
            for (int jp = jlo - 1; jp >= 0; jp--) nrow[jp] = INFINITY;
            for (int half = 0; half < 2; half++) {
                int jp = jhi2 - half;
                if (jp < jlo) continue;
                const int k1 = 2 * jp + pp;
                const int t1 = t0 + jp;
                const int d = (t1 ^ k1) & 1;
                const float *src = cost + (size_t)t1 * W
                                 + (size_t)((k1 - d - (t1 & 1)) >> 1);
                if (d == 0) {
                    for (; jp >= jlo; jp -= 2, src -= 2 * W + 2)
                        nrow[jp] = *src;
                } else {
                    /* k = 0 has no lone-leaf predecessor: if this
                     * chain reaches cell (jp = 0, k = 0), stop above
                     * it and mark it unreachable. */
                    int floor2 = jlo, patch0 = 0;
                    if (pp == 0 && (jp & 1) == 0 && jlo == 0) {
                        floor2 = 2;
                        patch0 = 1;
                    }
                    const float *dp0 = dP0 + k1;
                    for (; jp >= floor2; jp -= 2, src -= 2 * W + 2, dp0 -= 4)
                        nrow[jp] = *src + a0 * *dp0;
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
        for (int L = JL_LMAX; L >= 1; L--) {
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
    free(cost); free(arch);
    return J;
}

/* Overwrites lengths[] with the joint-optimal assignment.  Returns 0 on
 * success, -1 on any failure (caller keeps the Huffman lengths). */
/* ---------- Exact mass DP (fallback for lambda > 1/7) ----------
 * The original 0/1 knapsack over (k symbols, Kraft mass in 2^-11
 * units) with items in global cost order; ~10 MB transient, ~2-8 ms.
 * Used only when lambda is large enough that level order != cost
 * order (lam > 1/7), where the slot-ledger DP's exactness argument
 * does not apply. */
static double jl_solve_mass(const double *P, int sigma, double lam,
                            uint16_t out_BL[JL_LMAX + 1])
{
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

    const size_t nstates = (size_t)(sigma + 1) * (JL_MASS + 1);
    float    *dp   = malloc(nstates * sizeof(float));
    uint64_t *msk0 = malloc(nstates * sizeof(uint64_t));
    uint64_t *msk1 = malloc(nstates * sizeof(uint64_t));
    if (!dp || !msk0 || !msk1) { free(dp); free(msk0); free(msk1); return -1.0; }
    for (size_t i = 0; i < nstates; i++) dp[i] = INFINITY;
    memset(msk0, 0, nstates * sizeof(uint64_t));
    memset(msk1, 0, nstates * sizeof(uint64_t));
    dp[0] = 0.0f;
#define JL_IX(k, m) ((size_t)(k) * (JL_MASS + 1) + (size_t)(m))
    for (int it = 0; it < n_items; it++) {
        const int size = items[it].size, mass = items[it].mass;
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
    double J = dp[JL_IX(sigma, JL_MASS)];
    if (J < INFINITY) {
        const uint64_t m0 = msk0[JL_IX(sigma, JL_MASS)];
        const uint64_t m1 = msk1[JL_IX(sigma, JL_MASS)];
        for (int it = 0; it < n_items; it++) {
            const int on = it < 64 ? (int)((m0 >> it) & 1)
                                   : (int)((m1 >> (it - 64)) & 1);
            if (on) out_BL[items[it].L] |= (uint16_t)(1 << items[it].b);
        }
    } else {
        J = -1.0;
    }
    free(dp); free(msk0); free(msk1);
    return J;
}

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

    /* Production model for the adoption guard: bits + exchangeable-model
     * merge passes of the INCOMING lengths. */
    double prod_bits = 0, prod_passes = 0;
    {
        int    cls_n[JL_LMAX + 1] = {0};
        double cls_w[JL_LMAX + 1] = {0};
        for (int i = 0; i < sigma; i++) {
            int L = lengths[sf[i].sym];
            if (L < 1 || L > JL_LMAX) L = JL_LMAX;
            cls_n[L]++; cls_w[L] += (double)sf[i].freq;
        }
        for (int L = 1; L <= JL_LMAX; L++) {
            if (!cls_n[L]) continue;
            prod_bits += cls_w[L] * L;
            double dbar = 0;
            for (int b = 0; b <= JL_LMAX; b++)
                if (cls_n[L] & (1 << b)) dbar += (double)b * (1 << b);
            dbar /= (double)cls_n[L];
            prod_passes += cls_w[L] * ((double)L - dbar);
        }
    }

    /* Solve: slot-ledger DP (exact and fast) whenever the level-order
     * == cost-order condition lam <= 1/7 holds — always true for the
     * production lambda range; exact mass DP otherwise. */
    uint16_t BL[JL_LMAX + 1] = {0};
    if (lam <= (1.0 / 7.0) + 1e-9) {
        if (jl_solve_slots(P, sigma, lam, BL) < 0) return -1;
    } else {
        if (jl_solve_mass(P, sigma, lam, BL) < 0) return -1;
    }

    /* Model the DP result and apply the adoption guard. */
    double dp_bits = 0, dp_passes = 0;
    {
        int cur = 0;
        for (int L = 1; L <= JL_LMAX; L++)
            for (int b = 8; b >= 0; b--)
                if (BL[L] & (1 << b)) {
                    double w = P[cur + (1 << b)] - P[cur];
                    dp_bits   += w * L;
                    dp_passes += w * (L - b);
                    cur += 1 << b;
                }
        if (cur != sigma) return -1;
    }
    if (!(dp_passes <= 0.90 * prod_passes && dp_bits <= 1.015 * prod_bits))
        return -1;

    /* Deal freq-sorted symbols to chunks in cost order (L asc, b desc). */
    {
        int cur = 0;
        for (int L = 1; L <= JL_LMAX; L++)
            for (int b = 8; b >= 0; b--)
                if (BL[L] & (1 << b))
                    for (int j = 0; j < (1 << b); j++)
                        lengths[sf[cur++].sym] = (uint8_t)L;
    }
    return 0;
}

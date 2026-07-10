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
 * one current-level slot's worth of mass), so s <= 256 and the whole
 * plane is 257 x 257 -- cache resident.
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
 * Transitions: at level L with plane cost[k][s], for b = min(8,L)..0,
 * optionally take chunk (L, b): (k, s) -> (k + 2^b, s - 2^b) adding
 * cost(L,b) * (P[k+2^b] - P[k]).  In-place 0/1 semantics hold with
 * k-descending / s-ascending iteration.  Between levels, s -> 2*s
 * with the s <= sigma - k feasibility prune.  Terminal: (sigma, 0)
 * after level 11.  Per-state u16 planes record B_L for backtracking.
 */
#define JL_SCAP 256

static double jl_solve_slots(const double *P, int sigma, double lam,
                             uint16_t out_BL[JL_LMAX + 1])
{
    enum { W = JL_SCAP + 1 };
    const size_t plane = (size_t)(sigma + 1) * W;
    float    *cost = malloc(plane * sizeof(float));
    uint16_t *pick = malloc(plane * sizeof(uint16_t));
    uint16_t *arch = malloc((size_t)(JL_LMAX + 1) * plane * sizeof(uint16_t));
    if (!cost || !pick || !arch) { free(cost); free(pick); free(arch); return -1.0; }
    for (size_t i = 0; i < plane; i++) cost[i] = INFINITY;
#define SIX(k, s) ((size_t)(k) * W + (size_t)(s))
    cost[SIX(0, 2)] = 0.0f;               /* level 1 opens with 2 slots */

    for (int L = 1; L <= JL_LMAX; L++) {
        memset(pick, 0, plane * sizeof(uint16_t));
        int bmax = L < 8 ? L : 8;
        /* Reachability triangle: entering level L, the slot identity
         * gives k + s <= 2^L, and Kraft-equality feasibility gives
         * s <= sigma - k at entry (enforced by the doubling prune);
         * takes preserve k + s, so the whole level lives inside
         * k + s <= min(2^L, sigma) — early levels are tiny, late ones
         * halve to the sigma-triangle. */
        const int cap = (L >= 9 || (1 << L) > sigma) ? sigma : (1 << L);
        for (int b = bmax; b >= 0; b--) {
            const int cnk = 1 << b;
            if (cnk > sigma) continue;
            const float a = (float)((double)L + lam * (double)(L - b));
            int khi = sigma - cnk;
            if (khi > cap - cnk) khi = cap - cnk;
            for (int k = khi; k >= 0; k--) {
                const float add = a * (float)(P[k + cnk] - P[k]);
                float    *crow = cost + SIX(k, 0), *drow = cost + SIX(k + cnk, 0);
                uint16_t *prow = pick + SIX(k, 0), *qrow = pick + SIX(k + cnk, 0);
                int shi = cap - k;
                if (shi > JL_SCAP) shi = JL_SCAP;
                /* Parity: level-entry states live only at even s (they
                 * come from the doubling), and takes with b >= 1 keep
                 * s-parity, so odd cells can only appear after the
                 * (last-processed) b == 0 sweep.  Bits b >= 1 therefore
                 * step s by 2 — half the lattice for 8 of 9 sweeps. */
                const int step = b >= 1 ? 2 : 1;
#if defined(__aarch64__)
                if (step == 1) {
                    const float32x4_t vadd = vdupq_n_f32(add);
                    const uint16x4_t  vbit = vdup_n_u16((uint16_t)(1u << b));
                    int s = cnk;
                    for (; s + 4 <= shi + 1; s += 4) {
                        float32x4_t v    = vld1q_f32(crow + s);
                        float32x4_t dstv = vld1q_f32(drow + s - cnk);
                        float32x4_t cand = vaddq_f32(v, vadd);
                        uint32x4_t  m    = vcltq_f32(cand, dstv);
                        if (!vmaxvq_u32(m)) continue;
                        vst1q_f32(drow + s - cnk, vbslq_f32(m, cand, dstv));
                        uint16x4_t  pm   = vmovn_u32(m);
                        uint16x4_t  pv   = vorr_u16(vld1_u16(prow + s), vbit);
                        uint16x4_t  qv   = vld1_u16(qrow + s - cnk);
                        vst1_u16(qrow + s - cnk, vbsl_u16(pm, pv, qv));
                    }
                    for (; s <= shi; s++) {
                        const float v = crow[s];
                        if (!(v < INFINITY)) continue;
                        const float cand = v + add;
                        if (cand < drow[s - cnk]) {
                            drow[s - cnk] = cand;
                            qrow[s - cnk] = (uint16_t)(prow[s] | (1u << b));
                        }
                    }
                } else {
                    /* even-s sub-lattice: gather stride-2 via vld2 */
                    const float32x4_t vadd = vdupq_n_f32(add);
                    const uint16x4_t  vbit = vdup_n_u16((uint16_t)(1u << b));
                    int s = cnk;                    /* cnk even here */
                    for (; s + 8 <= shi + 1; s += 8) {
                        float32x4x2_t v2 = vld2q_f32(crow + s);
                        float32x4x2_t d2 = vld2q_f32(drow + s - cnk);
                        float32x4_t cand = vaddq_f32(v2.val[0], vadd);
                        uint32x4_t  m    = vcltq_f32(cand, d2.val[0]);
                        if (!vmaxvq_u32(m)) continue;
                        d2.val[0] = vbslq_f32(m, cand, d2.val[0]);
                        vst2q_f32(drow + s - cnk, d2);
                        uint16x4x2_t p2 = vld2_u16(prow + s);
                        uint16x4x2_t q2 = vld2_u16(qrow + s - cnk);
                        uint16x4_t  pm  = vmovn_u32(m);
                        q2.val[0] = vbsl_u16(pm, vorr_u16(p2.val[0], vbit), q2.val[0]);
                        vst2_u16(qrow + s - cnk, q2);
                    }
                    for (; s <= shi; s += 2) {
                        const float v = crow[s];
                        if (!(v < INFINITY)) continue;
                        const float cand = v + add;
                        if (cand < drow[s - cnk]) {
                            drow[s - cnk] = cand;
                            qrow[s - cnk] = (uint16_t)(prow[s] | (1u << b));
                        }
                    }
                }
#else
                for (int s = cnk; s <= shi; s += step) {
                    const float v = crow[s];
                    if (!(v < INFINITY)) continue;
                    const float cand = v + add;
                    if (cand < drow[s - cnk]) {
                        drow[s - cnk] = cand;
                        qrow[s - cnk] = (uint16_t)(prow[s] | (1u << b));
                    }
                }
#endif
            }
        }
        memcpy(arch + (size_t)L * plane, pick, plane * sizeof(uint16_t));
        if (L == JL_LMAX) break;
        /* Between levels: s' = 2*s, pruned by s' <= sigma - k (Kraft
         * equality feasibility).  In place, s' descending: every read
         * (at s'/2 < s') hits a not-yet-written slot. */
        for (int k = 0; k <= sigma; k++) {
            float *row = cost + SIX(k, 0);
            const int spmax = sigma - k;        /* max legal s' */
            for (int t = JL_SCAP; t >= 1; t--)
                row[t] = ((t & 1) == 0 && t <= spmax) ? row[t / 2]
                                                      : INFINITY;
            /* t == 0: s'=0 from s=0 — row[0] unchanged. */
        }
    }

    double J = cost[SIX(sigma, 0)];
    if (J < INFINITY) {
        /* Backtrack: invert each level's transition. */
        int k = sigma, sleft = 0;
        for (int L = JL_LMAX; L >= 1; L--) {
            uint16_t BL = arch[(size_t)L * plane + SIX(k, sleft)];
            out_BL[L] = BL;
            int cL = 0;
            for (int b = 0; b <= 8; b++) if (BL & (1 << b)) cL += 1 << b;
            k -= cL;
            int s_entry = sleft + cL;         /* slots at level L entry  */
            sleft = s_entry / 2;              /* pre-doubling slots left */
        }
    } else {
        J = -1.0;
    }
    free(cost); free(pick); free(arch);
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

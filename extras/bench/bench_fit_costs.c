/* bench_fit_costs — fit the joint cost model's per-arch constants on
 * this host: mu_full (ns per element-pass), mu_cst (relative), prefill,
 * kappa[1..8] (flat-kernel cost in merge-pass units), gamma (per-record
 * fixed cost in pass units).
 *
 * Method: ~21 controlled tree designs (pure flats per D, cst ladders,
 * full-merge mixes, a record-count sweep, geometric archetypes) are
 * decoded with FRESH data per block (single-block repetition lets the
 * branch predictor memorize the walk — measured +6% vs -27% on the
 * same tree pair) into a streamed output buffer.  Each design's
 * feature vector (full-merge weight, cst weight, prefill weight,
 * per-D flat weights, records/element) is extracted by probing
 * pivco_huffman_joint_model_time with one-hot cost settings, so the
 * features are BY CONSTRUCTION the ones the production guard uses —
 * the fit cannot drift from the model.  Ordinary least squares (tiny
 * ridge for conditioning) then solves for the constants.
 *
 * Output: per-design measured vs fitted ns/sym, the constants, and a
 * ready-to-paste jl_arch_costs_t initializer.
 *
 * After the kernel fit, a tau section measures the FSE decode tax:
 * skewed two-symbol trees whose single root bitmap commits under the
 * bytes-shrink rule (lambda stays 0 here so the lambda-aware gate
 * cannot suppress) are decoded PH vs PHA; tau = extra merge passes
 * per element of committed coverage = (ns_PHA - ns_PH) / mu_full.
 *
 * cc -O2 -Iinclude extras/bench/bench_fit_costs.c \
 *    build/libpivco_huffman.a -o bfc -lm && ./bfc
 */
#include "pivco_huffman.h"
#include "pivco_prof.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define N     16384            /* elements per block */
#define NBLK  128              /* fresh data per block (2 MB total) */
#define RUNS  5
#define REPS  8                /* block-set passes per timed run */
#define NF    12               /* feature count */
#define MAXDSG 32

static uint64_t rs;
static uint64_t xr(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }

typedef struct {
    char   name[8];
    double f[NF];              /* w_full, w_cst, -w_pre, w_flat[1..8], recs/elem */
    double ns;                 /* measured ns/sym */
} dsg_t;
static dsg_t g_dsg[MAXDSG];
static int g_ndsg = 0;

/* Feature extraction: probe the library model with one-hot settings.
 * Model units: mu_full == 1, so model(kappa=0, gamma=0, mu_cst=eps,
 * prefill=0) isolates the full-merge weight, and each one-hot knob
 * adds exactly its feature. */
static void extract_features(const uint64_t freq[256],
                             const uint8_t lens[256], double f[NF])
{
    static const double kz[9] = {0};
    double k1[9];
    pivco_huffman_set_joint_gamma(0.0);
    pivco_huffman_set_joint_kappa(kz);
    pivco_huffman_set_joint_merge_costs(1e-9, 0.0);
    const double w_full = pivco_huffman_joint_model_time(freq, lens);
    pivco_huffman_set_joint_merge_costs(1.0, 0.0);
    const double m_cst = pivco_huffman_joint_model_time(freq, lens);
    pivco_huffman_set_joint_merge_costs(1.0, 1.0);
    const double m_pre = pivco_huffman_joint_model_time(freq, lens);
    f[0] = w_full;
    f[1] = m_cst - w_full;      /* w_cst */
    f[2] = -(m_cst - m_pre);    /* -w_pre (prefill term is a discount) */
    pivco_huffman_set_joint_merge_costs(1e-9, 0.0);
    for (int D = 1; D <= 8; D++) {
        memcpy(k1, kz, sizeof k1); k1[D] = 1.0;
        pivco_huffman_set_joint_kappa(k1);
        f[2 + D] = pivco_huffman_joint_model_time(freq, lens) - w_full;
    }
    pivco_huffman_set_joint_kappa(kz);
    pivco_huffman_set_joint_gamma(1.0);
    f[11] = pivco_huffman_joint_model_time(freq, lens) - w_full;
    pivco_huffman_set_joint_gamma(0.0);
}

/* one design: nsym symbols; lens[i]; prob[i] (sums ~1).  Returns the
 * measured ns/sym (also recorded for the LSQ unless record == 0). */
static double run2(const char *name, int nsym, const uint8_t *lens,
                   const double *prob, int record)
{
    uint8_t code_lens[256] = {0};
    uint64_t freq[256] = {0};
    for (int i = 0; i < nsym; i++) code_lens[i] = lens[i];
    static pivco_huffman_codec_table_t ct;
    if (pivco_huffman_build_codec_table_from_code_lens(code_lens, &ct) != PIVCO_OK) {
        printf("%-6s BUILD FAIL\n", name); return -1;
    }
    /* NBLK blocks of exact per-block counts, each freshly shuffled */
    static uint8_t sym[NBLK][N];
    static uint8_t dec[(size_t)NBLK * N + 64];
    int cnt[256], tot = 0;
    for (int i = 0; i < nsym; i++) { cnt[i] = (int)(prob[i] * N + 0.5); tot += cnt[i]; }
    cnt[0] += N - tot;
    rs = 0x9E3779B97F4A7C15ull;
    for (int b = 0; b < NBLK; b++) {
        int pos = 0;
        for (int i = 0; i < nsym; i++) {
            freq[i] += (uint64_t)cnt[i];
            for (int j = 0; j < cnt[i]; j++) sym[b][pos++] = (uint8_t)i;
        }
        for (int i = N - 1; i > 0; i--) {       /* Fisher-Yates */
            int j = (int)(xr() % (uint64_t)(i + 1));
            uint8_t t = sym[b][i]; sym[b][i] = sym[b][j]; sym[b][j] = t;
        }
    }
    if (!record) for (int i = 0; i < 256; i++) freq[i] = 0;  /* unused */
    static uint8_t enc[NBLK][N * 2 + 64];
    static size_t  el[NBLK];
    for (int b = 0; b < NBLK; b++)
        pivco_huffman_encode_ct(sym[b], N, &ct, enc[b], &el[b]);

    double best = 1e18;
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int rep = 0; rep < REPS; rep++)
            for (int b = 0; b < NBLK; b++) {
                size_t consumed;
                pivco_huffman_decode_dt(enc[b], el[b] + 16, &ct.dec,
                                        dec + (size_t)b * N, &consumed);
            }
        double ns = (now_sec() - t0) / REPS / ((double)NBLK * N) * 1e9;
        if (ns < best) best = ns;
    }
    for (int b = 0; b < NBLK; b++)
        if (memcmp(dec + (size_t)b * N, sym[b], N) != 0) {
            printf("%-6s VERIFY FAIL\n", name); return -1;
        }
    if (!record) return best;

    dsg_t *d = &g_dsg[g_ndsg++];
    snprintf(d->name, sizeof d->name, "%s", name);
    d->ns = best;
    /* features from the INTENDED probabilities (floored at 1 so rare
     * symbols keep their lengths in the model's class histogram; the
     * scale keeps blocks ~= W/16K so the gamma feature matches the
     * measured N=16K blocks) */
    uint64_t ffreq[256] = {0};
    for (int i = 0; i < nsym; i++)
        ffreq[i] = (uint64_t)(prob[i] * 16384.0 * 65536.0) + 1;
    extract_features(ffreq, code_lens, d->f);
    return best;
}
#define run(name, nsym, lens, prob) (void)run2(name, nsym, lens, prob, 1)

/* Solve (A^T A + ridge) x = A^T y for x[NF] by Gaussian elimination. */
static void lsq(double x[NF])
{
    double ata[NF][NF] = {{0}}, aty[NF] = {0};
    for (int d = 0; d < g_ndsg; d++)
        for (int i = 0; i < NF; i++) {
            aty[i] += g_dsg[d].f[i] * g_dsg[d].ns;
            for (int j = 0; j < NF; j++)
                ata[i][j] += g_dsg[d].f[i] * g_dsg[d].f[j];
        }
    for (int i = 0; i < NF; i++) ata[i][i] += 1e-9;
    for (int c = 0; c < NF; c++) {              /* partial-pivot GE */
        int piv = c;
        for (int r = c + 1; r < NF; r++)
            if (fabs(ata[r][c]) > fabs(ata[piv][c])) piv = r;
        for (int j = 0; j < NF; j++) {
            double t = ata[c][j]; ata[c][j] = ata[piv][j]; ata[piv][j] = t;
        }
        double t = aty[c]; aty[c] = aty[piv]; aty[piv] = t;
        for (int r = 0; r < NF; r++) {
            if (r == c || ata[c][c] == 0) continue;
            const double m = ata[r][c] / ata[c][c];
            for (int j = 0; j < NF; j++) ata[r][j] -= m * ata[c][j];
            aty[r] -= m * aty[c];
        }
    }
    for (int i = 0; i < NF; i++) x[i] = ata[i][i] != 0 ? aty[i] / ata[i][i] : 0;
}

int main(void)
{
    pivco_prof_pin_cpu(0);   /* macOS: QoS bump so ssh-spawned runs
                              * hold a P-core (the M4 E-core trap) */
    pivco_huffman_set_fse_enabled(0);
    /* clock warmup */
    { volatile uint64_t s = 0; double t0 = now_sec();
      while (now_sec() - t0 < 1.0) for (int i = 0; i < 1 << 20; i++) s += (uint64_t)i; }

    /* pure pair */
    { uint8_t l[2] = {1,1}; double p[2] = {.5,.5}; run("P1", 2, l, p); }
    /* pure flats D=2..8, uniform */
    for (int D = 2; D <= 8; D++) {
        static uint8_t l[256]; static double p[256];
        int n = 1 << D;
        for (int i = 0; i < n; i++) { l[i] = (uint8_t)D; p[i] = 1.0 / n; }
        char nm[8]; snprintf(nm, 8, "F%d", D);
        run(nm, n, l, p);
    }
    /* cst ladders, dyadic head */
    for (int k = 4; k <= 8; k += 2) {
        static uint8_t l[64]; static double p[64];
        for (int i = 0; i < k - 1; i++) { l[i] = (uint8_t)(i + 1); p[i] = 1.0 / (1 << (i + 1)); }
        l[k-1] = l[k] = (uint8_t)k; p[k-1] = p[k] = 1.0 / (1 << k);
        char nm[8]; snprintf(nm, 8, "L%d", k);
        run(nm, k + 1, l, p);
    }
    /* full-merge mixes */
    { uint8_t l[6] = {2,2,3,3,3,3}; double p[6] = {.25,.25,.125,.125,.125,.125};
      run("U1", 6, l, p); }
    { uint8_t l[8] = {2,2,3,3,4,4,4,4};
      double p[8] = {.25,.25,.125,.125,.0625,.0625,.0625,.0625};
      run("U2", 8, l, p); }
    { uint8_t l[6] = {1,2,4,4,4,4};
      double p[6] = {.5,.25,.0625,.0625,.0625,.0625};
      run("M1", 6, l, p); }
    /* record-count sweep: pair cascades, near-constant merge volume */
    for (int m = 4; m <= 10; m += 2) {
        static uint8_t l[64]; static double p[64];
        int n = 0;
        for (int L = 2; L <= m; L++) {
            l[n] = (uint8_t)L; p[n] = pow(0.5, L); n++;
            l[n] = (uint8_t)L; p[n] = pow(0.5, L); n++;
        }
        for (int j = 0; j < 4; j++) {
            l[n] = (uint8_t)(m + 1); p[n] = pow(0.5, m + 1) / 2; n++;
        }
        char nm[8]; snprintf(nm, 8, "C%d", m);
        run(nm, n, l, p);
    }
    /* geometric archetypes (deep flats + head leaves) */
    { static uint8_t l[256]; static double p[256];
      double rem = 1.0;
      for (int i = 0; i < 256; i++) {
          p[i] = i < 30 ? pow(0.5, i + 1) : rem / (256 - 30);
          if (i < 30) rem -= p[i];
      }
      l[0] = 1; l[1] = 2; l[2] = 3; l[3] = 9;
      for (int i = 4; i < 256; i++) l[i] = 11;
      run("GEOb", 256, l, p);
      l[0] = 1; l[1] = 3; l[2] = 3; l[3] = 4; l[4] = 5; l[5] = 6;
      l[6] = 6; l[7] = 8;
      for (int i = 8; i < 256; i++) l[i] = 11;
      run("GEOj", 256, l, p); }
    { static uint8_t l[130]; static double p[130];
      l[0] = 1; p[0] = .5; l[1] = 2; p[1] = .25;
      for (int i = 2; i < 130; i++) { l[i] = 9; p[i] = .25/128; }
      run("G1", 130, l, p); }

    /* ---- fit ---- */
    double x[NF];
    lsq(x);
    const double muf = x[0];
    printf("\n%-6s %9s %9s %7s\n", "design", "meas", "fit", "resid");
    double worst = 0;
    for (int d = 0; d < g_ndsg; d++) {
        double fit = 0;
        for (int i = 0; i < NF; i++) fit += g_dsg[d].f[i] * x[i];
        double r = g_dsg[d].ns - fit;
        if (fabs(r) > worst) worst = fabs(r);
        printf("%-6s %9.4f %9.4f %+7.4f\n", g_dsg[d].name, g_dsg[d].ns, fit, r);
    }
    printf("\nfitted (worst residual %.4f ns/sym):\n", worst);
    printf("  mu_full  %.4f ns/elem-pass\n", muf);
    printf("  mu_cst   %.3f (rel)\n", x[1] / muf);
    printf("  prefill  %.3f\n", x[1] != 0 ? x[2] / x[1] : 0);
    printf("  gamma    %.0f passes (%.2f ns/record/block)\n", x[11] / muf, x[11]);
    printf("  kappa   ");
    for (int D = 1; D <= 8; D++) printf(" b%d %.2f", D, x[2 + D] / muf);
    printf("\n\n/* jl_arch_costs_t initializer (paste into joint_lengths.c): */\n");
    printf("{ \"HOST\", %.3f, %.3f, %.0f,\n  { 0, ", x[1] / muf,
           x[1] != 0 ? x[2] / x[1] : 0, x[11] / muf);
    for (int D = 1; D <= 8; D++)
        printf("%.2f%s", x[2 + D] / muf, D < 8 ? ", " : " } },");
    printf("  /* mu_full %.4f ns */\n", muf);

    /* ---- tau: FSE decode tax (PH vs PHA on committed root bitmaps).
     * lambda is 0 throughout, so commits follow the plain bytes-shrink
     * rule regardless of the lambda-aware gate. ---- */
    printf("\ntau (FSE decode tax, extra passes/elem):\n");
    double taus[8]; int ntau = 0;
    double probs[4] = { 0.85, 0.90, 0.95, 0.97 };
    for (int k = 0; k < 4; k++) {
        uint8_t l[2] = { 1, 1 };
        double p[2] = { probs[k], 1.0 - probs[k] };
        char nm[8]; snprintf(nm, 8, "T%02d", (int)(probs[k] * 100));
        pivco_huffman_set_fse_enabled(0);
        double ph = run2(nm, 2, l, p, 0);
        pivco_huffman_set_fse_enabled(1);
        double pha = run2(nm, 2, l, p, 0);
        pivco_huffman_set_fse_enabled(0);
        if (ph < 0 || pha < 0) continue;
        double tau = (pha - ph) / muf;
        if (pha <= ph * 1.02) {
            printf("  p=%.2f  PH %.4f  PHA %.4f  (no commit / no tax)\n",
                   probs[k], ph, pha);
            continue;
        }
        printf("  p=%.2f  PH %.4f  PHA %.4f  tau %.2f\n", probs[k], ph, pha, tau);
        taus[ntau++] = tau;
    }
    if (ntau) {
        double lo = 1e18, hi = 0, mean = 0;
        for (int i = 0; i < ntau; i++) {
            mean += taus[i];
            if (taus[i] < lo) lo = taus[i];
            if (taus[i] > hi) hi = taus[i];
        }
        mean /= ntau;
        printf("  => tau mean %.2f (range %.2f..%.2f; table-dependent"
               " -- profile ships the mean)\n", mean, lo, hi);
    } else printf("  => no committed designs; tau not fitted\n");
    return 0;
}

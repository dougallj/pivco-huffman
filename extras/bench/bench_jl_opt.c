/* bench_jl_opt — timing harness for pivcoh__jl_slots / pivcoh__jl_core
 * optimization work.  Per 16K window of each .lits file:
 *
 *   slots    exact pivcoh__jl_slots alone (P built once, uncounted)
 *   core g1  pivcoh__jl_core end-to-end at gran 1 (exact)
 *   core g0  ... gran 0 (auto tier)
 *   core g-1 ... gran -1 (coarse tier)
 *
 * best-of --reps per window, summed.  An FNV-1a hash over the adopted
 * lens (and adoption flags) of every window x tier is printed: any
 * optimization patch must reproduce it bit-exactly (the DP's FP
 * expressions are untouched), or say loudly why not.
 *
 * Usage: bench_jl_opt [--G=KB] [--reps=N] [--lambda=F] file...
 * Defaults: G=16, reps=7, lambda=0.1, m4 kappa profile, the 12 Silesia
 * .lits (run from the repo root). */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint8_t *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = (uint8_t *)malloc((size_t)sz);
    if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(p); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return p;
}

static uint64_t fnv1a(uint64_t h, const void *buf, size_t n)
{
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

static int cmp_leaf_asc(const void *a, const void *b)
{
    const pivcoh__leaf *x = (const pivcoh__leaf *)a, *y = (const pivcoh__leaf *)b;
    if (x->freq != y->freq) return x->freq < y->freq ? -1 : 1;
    return x->sym < y->sym ? -1 : 1;
}

int main(int argc, char **argv)
{
    size_t G = 16 * 1024;
    int reps = 7;
    pivcoh_joint jp = PIVCOH_JOINT_DEFAULTS;
    const float m4kap[9] = { 0, 0.64f, 0.49f, 0.63f, 0.53f, 0.70f, 0.91f, 1.80f, 0.41f };
    memcpy(jp.kappa, m4kap, sizeof m4kap);
    jp.gamma = 210.0f;
    const char *deffiles[] = {
        "testdata/silesia-lits/dickens.lits", "testdata/silesia-lits/mozilla.lits",
        "testdata/silesia-lits/mr.lits",      "testdata/silesia-lits/nci.lits",
        "testdata/silesia-lits/ooffice.lits", "testdata/silesia-lits/osdb.lits",
        "testdata/silesia-lits/reymont.lits", "testdata/silesia-lits/samba.lits",
        "testdata/silesia-lits/sao.lits",     "testdata/silesia-lits/webster.lits",
        "testdata/silesia-lits/x-ray.lits",   "testdata/silesia-lits/xml.lits" };
    const char *files[64];
    int nfiles = 0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--G=", 4)) G = (size_t)atoi(argv[i] + 4) * 1024;
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--lambda=", 9)) jp.lambda = (float)atof(argv[i] + 9);
        else if (nfiles < 64) files[nfiles++] = argv[i];
    }
    if (!nfiles) { memcpy(files, deffiles, sizeof deffiles); nfiles = 12; }

    void *scratch = malloc(PIVCOH_JOINT_SCRATCH_SIZE);
    static const int grans[3] = { 1, 0, -1 };
    double t_slots = 0, t_core[3] = { 0, 0, 0 };
    long windows = 0, adopts[3] = { 0, 0, 0 };
    uint64_t hash = 0xcbf29ce484222325ull;
    double sig_sum = 0;

    for (int fi = 0; fi < nfiles; fi++) {
        size_t n;
        uint8_t *buf = slurp(files[fi], &n);
        if (!buf) { fprintf(stderr, "cannot read %s\n", files[fi]); return 1; }
        for (size_t off = 0; off + G <= n || (off < n && off == 0); off += G) {
            const size_t wlen = off + G <= n ? G : n - off;
            uint64_t freq[256] = { 0 };
            for (size_t i = 0; i < wlen; i++) freq[buf[off + i]]++;
            pivcoh__leaf leaf0[256];
            int sigma = 0;
            for (int s = 0; s < 256; s++)
                if (freq[s]) { leaf0[sigma].freq = (uint32_t)freq[s]; leaf0[sigma].sym = (uint16_t)s; sigma++; }
            if (sigma < 2) continue;
            qsort(leaf0, (size_t)sigma, sizeof *leaf0, cmp_leaf_asc);
            /* baseline Huffman lens via the production builder (uncounted) */
            static pivcoh_table tbl;
            if (!pivcoh_table_from_freqs(&tbl, freq)) continue;
            windows++; sig_sum += sigma;

            /* exact jl_slots alone */
            {
                double P[257];
                P[0] = 0;
                for (int i = 0; i < sigma; i++) P[i + 1] = P[i] + (double)leaf0[sigma - 1 - i].freq;
                double kapd[9];
                for (int b = 0; b <= 8; b++) kapd[b] = (double)jp.kappa[b];
                double blocks = ceil(P[sigma] / 16384.0);
                if (blocks < 1) blocks = 1;
                const double tc0 = (double)jp.lambda * (double)jp.gamma * blocks, tc1 = 2 * tc0;
                uint16_t BL[PIVCOH__MAXLEN + 1];
                double best = 1e9;
                for (int r = 0; r < reps; r++) {
                    memset(BL, 0, sizeof BL);
                    double t0 = now_sec();
                    double J = pivcoh__jl_slots(P, sigma, (double)jp.lambda, PIVCOH__MAXLEN, 8,
                                                kapd, tc0, tc1, BL, scratch);
                    double t1 = now_sec();
                    if (J < 0) { fprintf(stderr, "slots failed\n"); return 1; }
                    if (t1 - t0 < best) best = t1 - t0;
                }
                t_slots += best;
                hash = fnv1a(hash, BL, sizeof BL);
            }
            /* jl_core end-to-end per tier */
            for (int gi = 0; gi < 3; gi++) {
                pivcoh_joint j2 = jp;
                j2.gran = grans[gi];
                pivcoh__leaf leaf[256];
                uint8_t lens[256];
                double best = 1e9;
                int rc = -1;
                for (int r = 0; r < reps; r++) {
                    memcpy(leaf, leaf0, sizeof leaf);
                    memcpy(lens, tbl.code_len, 256);
                    double t0 = now_sec();
                    rc = pivcoh__jl_core(leaf, sigma, freq, lens, &j2, scratch);
                    double t1 = now_sec();
                    if (t1 - t0 < best) best = t1 - t0;
                }
                t_core[gi] += best;
                adopts[gi] += rc == 0;
                hash = fnv1a(hash, &rc, sizeof rc);
                hash = fnv1a(hash, lens, 256);
            }
        }
        free(buf);
    }
    printf("windows %ld · mean sigma %.0f · lambda %.2f\n", windows, sig_sum / (double)windows,
           (double)jp.lambda);
    printf("us/solve mean: slots-exact %.2f · core g1 %.2f · core g0 %.2f · core g-1 %.2f\n",
           1e6 * t_slots / (double)windows, 1e6 * t_core[0] / (double)windows,
           1e6 * t_core[1] / (double)windows, 1e6 * t_core[2] / (double)windows);
    printf("adopt: g1 %ld · g0 %ld · g-1 %ld  (of %ld)\n", adopts[0], adopts[1], adopts[2], windows);
    printf("identity hash: %016llx\n", (unsigned long long)hash);
    free(scratch);
    return 0;
}

/* bench_time_target — target-speed mode demo on the .lits corpus:
 * per window, minimize bits subject to MODELED decode time <= target,
 * then verify against MEASURED per-window decode e2e (decode-table
 * rebuild + decode kernels).  The metric that matters is the target
 * HIT RATE: the fraction of windows whose measured speed meets the
 * target the model promised — i.e. whether the cost model is
 * contract-grade on this host.
 *
 * Phase 1 calibrates the host's ns-per-model-pass (median over all
 * baseline windows of measured ns/B divided by modeled passes/B) and
 * records baseline speeds/sizes.  Phase 2, per target T (GB/s), sets
 * pivco_huffman_set_joint_time_target((1/T)/mu_hat/(1+margin)) and
 * re-encodes + re-measures every window.
 *
 * PH (per-node FSE off).  Usage:
 *   bench_time_target [--G=KB] [--targets=T1,T2,..] [--margin=M]
 *                     [--reps=N] file...
 *
 * cc -O2 -Iinclude extras/bench/bench_time_target.c \
 *    build/libpivco_huffman.a -o btt -lm
 */
#include "pivco_huffman.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)sz + 16);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return NULL; }
    fclose(f); *n = (size_t)sz; return b;
}
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Measured e2e decode ns/byte of one window's stream: rebuild the
 * decode table from lengths, decode all blocks.  Region sized to
 * >= ~1.2 ms; best of `reps`. */
static double measure_window(const uint8_t *enc, size_t elen,
                             const uint8_t *lens, uint8_t *dec,
                             size_t wlen, int reps)
{
    pivco_huffman_decode_table_t dt;
    /* calibration pass (also warms stream + predictor) */
    double t0 = now_sec();
    pivco_huffman_build_decode_table(lens, &dt);
    size_t o = 0, dof = 0;
    while (dof < wlen) {
        size_t consumed;
        pivco_huffman_decode_dt(enc + o, elen - o + 16, &dt, dec + dof, &consumed);
        o += consumed;
        dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
    }
    double t1 = now_sec() - t0;
    int inner = (int)(0.0012 / (t1 > 1e-9 ? t1 : 1e-9)) + 1;
    if (inner > 4096) inner = 4096;
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        t0 = now_sec();
        for (int ii = 0; ii < inner; ii++) {
            pivco_huffman_build_decode_table(lens, &dt);
            o = 0; dof = 0;
            while (dof < wlen) {
                size_t consumed;
                pivco_huffman_decode_dt(enc + o, elen - o + 16, &dt, dec + dof, &consumed);
                o += consumed;
                dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
            }
        }
        double nsb = (now_sec() - t0) * 1e9 / ((double)wlen * inner);
        if (nsb < best) best = nsb;
    }
    return best;
}

#define MAXW 16384

typedef struct {
    int      file, widx;
    size_t   n;             /* window length */
    size_t   base_enc;      /* baseline encoded bytes */
    double   base_nsb;      /* baseline measured ns/byte */
    double   base_ppe;      /* baseline modeled passes/elem */
    uint8_t  lens[256];     /* baseline lengths */
} win_t;

int main(int argc, char **argv)
{
    size_t G = 64 * 1024;
    double targets[16]; int ntgt = 0;
    double margin = 0.0;
    int reps = 4;
    int argi = 1;
    for (; argi < argc && argv[argi][0] == '-'; argi++) {
        if (!strncmp(argv[argi], "--G=", 4)) G = (size_t)atoi(argv[argi] + 4) * 1024;
        else if (!strncmp(argv[argi], "--margin=", 9)) margin = atof(argv[argi] + 9);
        else if (!strncmp(argv[argi], "--reps=", 7)) reps = atoi(argv[argi] + 7);
        else if (!strncmp(argv[argi], "--targets=", 10))
            for (const char *p = argv[argi] + 10; *p && ntgt < 16; ) {
                targets[ntgt++] = strtod(p, NULL);
                p = strchr(p, ','); if (!p) break; p++;
            }
    }
    if (!ntgt) { targets[0] = 8; targets[1] = 10; targets[2] = 12; ntgt = 3; }
    pivco_huffman_set_fse_enabled(0);

    uint8_t *data[64]; size_t dlen[64]; const char *dname[64];
    int nfiles = 0;
    for (; argi < argc && nfiles < 64; argi++) {
        data[nfiles] = slurp(argv[argi], &dlen[nfiles]);
        if (!data[nfiles]) { fprintf(stderr, "skip %s\n", argv[argi]); continue; }
        const char *b = strrchr(argv[argi], '/');
        dname[nfiles] = b ? b + 1 : argv[argi];
        nfiles++;
    }

    static win_t win[MAXW];
    int nwin = 0;
    uint8_t *enc = malloc(2 * G + 4096), *dec = malloc(G + 64);

    /* ---- phase 1: baseline build/measure + mu calibration ---- */
    static double ratio_samples[MAXW];
    pivco_huffman_set_joint_lambda(0.0);
    pivco_huffman_set_joint_time_target(0.0);
    for (int f = 0; f < nfiles; f++)
        for (size_t w = 0; w * G < dlen[f] && nwin < MAXW; w++) {
            size_t wlen = (w + 1) * G <= dlen[f] ? G : dlen[f] - w * G;
            const uint8_t *p = data[f] + w * G;
            uint64_t freq[256] = {0};
            for (size_t i = 0; i < wlen; i++) freq[p[i]]++;
            pivco_huffman_codec_table_t ct;
            pivco_huffman_build_codec_table(freq, &ct);
            size_t off = 0;
            for (size_t b = 0; b < wlen; b += PIVCO_BLOCK_SIZE) {
                size_t bn = wlen - b < PIVCO_BLOCK_SIZE ? wlen - b : PIVCO_BLOCK_SIZE;
                size_t el;
                pivco_huffman_encode_ct(p + b, bn, &ct, enc + off, &el);
                off += el;
            }
            win_t *W = &win[nwin];
            W->file = f; W->widx = (int)w; W->n = wlen; W->base_enc = off;
            memcpy(W->lens, ct.code_len, 256);
            W->base_ppe = pivco_huffman_joint_model_time(freq, ct.code_len);
            W->base_nsb = measure_window(enc, off, ct.code_len, dec, wlen, reps);
            if (memcmp(dec, p, wlen)) { fprintf(stderr, "RT FAIL\n"); return 1; }
            if (W->base_ppe > 0) ratio_samples[nwin] = W->base_nsb / W->base_ppe;
            nwin++;
        }
    qsort(ratio_samples, (size_t)nwin, sizeof(double), cmp_dbl);
    const double mu = ratio_samples[nwin / 2];   /* ns per model pass */
    printf("windows %d  G=%zuK  PH  margin=%.0f%%  calibrated mu = %.4f ns/pass "
           "(p10 %.4f  p90 %.4f)\n",
           nwin, G / 1024, margin * 100, mu,
           ratio_samples[nwin / 10], ratio_samples[nwin - 1 - nwin / 10]);

    /* ---- phase 2: per target ---- */
    printf("%6s | %8s %8s | %7s %7s | %8s | %9s %9s\n",
           "target", "base-hit", "tgt-hit", "adopted", "capped", "dRatio",
           "p5 marg", "med marg");
    for (int t = 0; t < ntgt; t++) {
        const double T = targets[t];
        const double tgt_ppe = (1.0 / T) / mu / (1.0 + margin);
        pivco_huffman_set_joint_time_target(tgt_ppe);
        long base_hit = 0, hit = 0, adopted = 0, capped = 0;
        double dbits = 0, dn = 0;
        static double margins[MAXW];
        for (int i = 0; i < nwin; i++) {
            win_t *W = &win[i];
            const uint8_t *p = data[W->file] + (size_t)W->widx * G;
            uint64_t freq[256] = {0};
            for (size_t k = 0; k < W->n; k++) freq[p[k]]++;
            pivco_huffman_codec_table_t ct;
            pivco_huffman_build_codec_table(freq, &ct);
            int ad = memcmp(ct.code_len, W->lens, 256) != 0;
            double nsb, encb;
            if (ad) {
                size_t off = 0;
                for (size_t b = 0; b < W->n; b += PIVCO_BLOCK_SIZE) {
                    size_t bn = W->n - b < PIVCO_BLOCK_SIZE ? W->n - b : PIVCO_BLOCK_SIZE;
                    size_t el;
                    pivco_huffman_encode_ct(p + b, bn, &ct, enc + off, &el);
                    off += el;
                }
                encb = (double)off;
                nsb = measure_window(enc, off, ct.code_len, dec, W->n, reps);
                if (memcmp(dec, p, W->n)) { fprintf(stderr, "RT FAIL\n"); return 1; }
                double m = pivco_huffman_joint_model_time(freq, ct.code_len);
                if (m > tgt_ppe * 1.0000001) capped++;
            } else {
                encb = (double)W->base_enc;
                nsb = W->base_nsb;
            }
            adopted += ad;
            hit += (1.0 / nsb) >= T;
            margins[i] = (1.0 / nsb) / T;
            dbits += encb - (double)W->base_enc;
            dn += (double)W->n;
        }
        for (int i = 0; i < nwin; i++) base_hit += (1.0 / win[i].base_nsb) >= T;
        qsort(margins, (size_t)nwin, sizeof(double), cmp_dbl);
        printf("%4.0fGB | %6.1f%% %7.1f%% | %6.1f%% %6.1f%% | %+7.3fpp | %8.3fx %8.3fx\n",
               T, 100.0 * base_hit / nwin, 100.0 * hit / nwin,
               100.0 * adopted / nwin, 100.0 * capped / nwin,
               100.0 * dbits / dn,
               margins[nwin / 20], margins[nwin / 2]);
    }
    pivco_huffman_set_joint_time_target(0.0);
    return 0;
}

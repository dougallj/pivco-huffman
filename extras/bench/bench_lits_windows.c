/* bench_lits_windows — the "tables every 5..100 KB" workload on real
 * LZ-preprocessed literal streams (e.g. Silesia .lits from
 * extract_zstd_lits), on the minimal codec-table APIs.
 *
 * For each input file and window size G: slice the stream into G-byte
 * windows; per window build a fresh codec table from the window's own
 * histogram, encode in PIVCO_BLOCK_SIZE blocks, decode, verify.
 *
 * Reported per (file, config):
 *   - ratio INCLUDING the 128-byte nibble-packed lengths header per
 *     window (the real cost of shipping a table at this cadence)
 *   - enc-e2e MB/s: histogram + pivco_huffman_build_codec_table (incl.
 *     the joint pass when enabled) + encode_ct kernels
 *   - dec-e2e MB/s: per-window pivco_huffman_build_decode_table (the
 *     decoder's true per-window cost; the 128-byte nibble unpack is
 *     noise) + decode_dt kernels
 *   - enc/dec kernel-only MB/s, table-build us/window, adopted windows
 * plus a geomean summary over all files.
 *
 * --ladder runs the full solve ladder per file, back-to-back so the
 * rungs share thermal conditions: off (lambda 0), nudge (gran -1),
 * auto DP (gran 0), exact DP (gran 1), with deltas vs off.
 * Otherwise --joint=L runs the lambda-0 / lambda-L pair as before.
 *
 * Usage: bench_lits_windows [--G=KB] [--joint=L] [--ladder] [--reps=N]
 *        [--fse=0|1] [--gran=N] file...
 * --fse=0 benches PH (no per-node FSE attempt); default 1 = PHA.
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

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)sz + 16);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f); *n = (size_t)sz;
    return b;
}

typedef struct {
    double ratio, enc_k, dec_k, enc_e2e, dec_e2e, build_us;
    int adopted, nwin;
} res_t;

/* One full measurement of (file, lambda, granularity). */
static int run_file(const uint8_t *data, size_t n, size_t G, int reps,
                    double lam, int gran, res_t *r)
{
    pivco_huffman_set_joint_lambda(lam);
    pivco_huffman_set_joint_granularity(gran);
    size_t nwin = (n + G - 1) / G;
    pivco_huffman_codec_table_t *tabs = malloc(nwin * sizeof(*tabs));
    pivco_huffman_decode_table_t *dt = malloc(sizeof(*dt));
    uint8_t *enc = malloc(n * 2 + nwin * 1024 + 4096);
    uint8_t *dec = malloc(n + 64);
    size_t  *woff = malloc((nwin + 1) * sizeof(size_t));
    if (!tabs || !dt || !enc || !dec || !woff) return -1;

    /* ---- encoder-side table builds: histogram + build (timed) ---- */
    double t0 = now_sec();
    for (size_t w = 0; w < nwin; w++) {
        size_t wlen = (w + 1) * G <= n ? G : n - w * G;
        uint64_t freq[256] = {0};
        const uint8_t *p = data + w * G;
        for (size_t i = 0; i < wlen; i++) freq[p[i]]++;
        pivco_huffman_build_codec_table(freq, &tabs[w]);
    }
    double build_s = now_sec() - t0;

    /* adoption count (untimed): rebuild with lambda off, compare lens */
    int adopted = 0;
    if (lam > 0) {
        pivco_huffman_set_joint_lambda(0.0);
        for (size_t w = 0; w < nwin; w++) {
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            uint64_t freq[256] = {0};
            const uint8_t *p = data + w * G;
            for (size_t i = 0; i < wlen; i++) freq[p[i]]++;
            pivco_huffman_codec_table_t t;
            pivco_huffman_build_codec_table(freq, &t);
            if (memcmp(t.code_len, tabs[w].code_len, 256) != 0) adopted++;
        }
        pivco_huffman_set_joint_lambda(lam);
    }

    /* ---- encode kernels (timed) ---- */
    /* Tiny files (Calgary-scale) finish a whole-file pass below the
     * timer tick; loop enough passes inside the timed region that it
     * spans ~8 MB of work. */
    const int inner = (int)(1 + ((size_t)8 << 20) / (n ? n : 1));
    size_t total_enc = 0;
    double enc_best = 0;
    for (int rep = 0; rep < reps; rep++) {
        double t1 = now_sec();
        size_t off = 0;
        for (int ii = 0; ii < inner; ii++) {
        off = 0;
        for (size_t w = 0; w < nwin; w++) {
            woff[w] = off;
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            for (size_t b = 0; b < wlen; b += PIVCO_BLOCK_SIZE) {
                size_t bn = wlen - b < PIVCO_BLOCK_SIZE ? wlen - b : PIVCO_BLOCK_SIZE;
                size_t el;
                pivco_huffman_encode_ct(data + w * G + b, bn, &tabs[w], enc + off, &el);
                off += el;
            }
        }
        }
        woff[nwin] = off; total_enc = off;
        double mbs = (double)n * inner / (now_sec() - t1) / 1e6;
        if (mbs > enc_best) enc_best = mbs;
    }

    /* ---- decode, kernel only (prebuilt decode cores) ---- */
    double dec_best = 0;
    for (int rep = 0; rep < reps; rep++) {
        double t1 = now_sec();
        for (int ii = 0; ii < inner; ii++)
        for (size_t w = 0; w < nwin; w++) {
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            size_t off = woff[w], dof = 0;
            while (dof < wlen) {
                size_t consumed;
                pivco_huffman_decode_dt(enc + off, woff[w + 1] - off + 16,
                                        &tabs[w].dec, dec + w * G + dof, &consumed);
                off += consumed;
                dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
            }
        }
        double mbs = (double)n * inner / (now_sec() - t1) / 1e6;
        if (mbs > dec_best) dec_best = mbs;
    }
    if (memcmp(dec, data, n) != 0) return -2;

    /* ---- decode end-to-end: rebuild the decode table per window ---- */
    double dec_e2e = 0;
    for (int rep = 0; rep < reps; rep++) {
        double t1 = now_sec();
        for (int ii = 0; ii < inner; ii++)
        for (size_t w = 0; w < nwin; w++) {
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            pivco_huffman_build_decode_table(tabs[w].code_len, dt);
            size_t off = woff[w], dof = 0;
            while (dof < wlen) {
                size_t consumed;
                pivco_huffman_decode_dt(enc + off, woff[w + 1] - off + 16,
                                        dt, dec + w * G + dof, &consumed);
                off += consumed;
                dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
            }
        }
        double mbs = (double)n * inner / (now_sec() - t1) / 1e6;
        if (mbs > dec_e2e) dec_e2e = mbs;
    }
    if (memcmp(dec, data, n) != 0) return -3;

    r->ratio    = (double)(total_enc + nwin * 128) / (double)n;
    r->enc_k    = enc_best;
    r->dec_k    = dec_best;
    r->enc_e2e  = (double)n / (build_s + (double)n / (enc_best * 1e6)) / 1e6;
    r->dec_e2e  = dec_e2e;
    r->build_us = build_s / (double)nwin * 1e6;
    r->adopted  = adopted;
    r->nwin     = (int)nwin;
    free(tabs); free(dt); free(enc); free(dec); free(woff);
    return 0;
}

typedef struct { const char *name; double lam; int gran; } cfg_t;

int main(int argc, char **argv)
{
    size_t G = 64 * 1024;
    double lam = 0.0;
    int reps = 5, fse = 1, gran = 1, ladder = 0;
    int argi = 1;
    for (; argi < argc && argv[argi][0] == '-'; argi++) {
        if (!strncmp(argv[argi], "--G=", 4)) G = (size_t)atoi(argv[argi] + 4) * 1024;
        else if (!strncmp(argv[argi], "--joint=", 8)) lam = atof(argv[argi] + 8);
        else if (!strncmp(argv[argi], "--reps=", 7)) reps = atoi(argv[argi] + 7);
        else if (!strncmp(argv[argi], "--fse=", 6)) fse = atoi(argv[argi] + 6);
        else if (!strncmp(argv[argi], "--gran=", 7)) gran = atoi(argv[argi] + 7);
        else if (!strcmp(argv[argi], "--ladder")) ladder = 1;
    }
    pivco_huffman_set_fse_enabled(fse);

    cfg_t cfgs[4];
    int ncfg;
    if (ladder) {
        cfgs[0] = (cfg_t){ "off",   0.0, 1 };
        cfgs[1] = (cfg_t){ "nudge", 0.1, -1 };
        cfgs[2] = (cfg_t){ "auto",  0.1, 0 };
        cfgs[3] = (cfg_t){ "exact", 0.1, 1 };
        ncfg = 4;
        printf("LADDER G=%zuK reps=%d fse=%d (%s)\n", G / 1024, reps,
               fse, fse ? "PHA" : "PH");
    } else {
        cfgs[0] = (cfg_t){ "base", 0.0, gran };
        cfgs[1] = (cfg_t){ "joint", lam, gran };
        ncfg = lam > 0 ? 2 : 1;
        printf("G=%zuK lambda=%.2f reps=%d fse=%d (%s) gran=%d\n", G / 1024,
               lam, reps, fse, fse ? "PHA" : "PH", gran);
    }
    printf("%-12s %6s %8s %9s %9s %9s %9s %9s %9s\n",
           "file", "cfg", "ratio", "enc-e2e", "dec-e2e", "enc-k", "dec-k",
           "build u/w", "adopted");

    double gm[4][4] = {{0}};      /* [cfg][enc_e2e dec_e2e enc_k dec_k] */
    double ratio_pp[4] = {0};
    int nfiles = 0;
    for (; argi < argc; argi++) {
        size_t n;
        uint8_t *data = slurp(argv[argi], &n);
        if (!data) { fprintf(stderr, "skip %s\n", argv[argi]); continue; }
        const char *base = strrchr(argv[argi], '/');
        base = base ? base + 1 : argv[argi];

        res_t r[4];
        int ok = 1;
        for (int j = 0; j < ncfg; j++) {
            int rc = run_file(data, n, G, reps, cfgs[j].lam, cfgs[j].gran, &r[j]);
            if (rc != 0) { printf("%-12s %s FAIL rc=%d\n", base, cfgs[j].name, rc); ok = 0; break; }
            char ad[16] = "-";
            if (cfgs[j].lam > 0)
                snprintf(ad, sizeof ad, "%d/%d", r[j].adopted, r[j].nwin);
            printf("%-12s %6s %8.4f %9.0f %9.0f %9.0f %9.0f %9.1f %9s\n",
                   base, cfgs[j].name, r[j].ratio, r[j].enc_e2e, r[j].dec_e2e,
                   r[j].enc_k, r[j].dec_k, r[j].build_us, ad);
        }
        if (ok && ncfg > 1) {
            for (int j = 0; j < ncfg; j++) {
                gm[j][0] += log(r[j].enc_e2e); gm[j][1] += log(r[j].dec_e2e);
                gm[j][2] += log(r[j].enc_k);   gm[j][3] += log(r[j].dec_k);
                ratio_pp[j] += 100 * (r[j].ratio - r[0].ratio);
            }
            nfiles++;
        }
        free(data);
    }
    if (nfiles > 1) {
        printf("== geomean over %d files (G=%zuK)\n", nfiles, G / 1024);
        for (int j = 0; j < ncfg; j++)
            printf("%6s: enc-e2e %6.0f (%+5.1f%%)  dec-e2e %6.0f (%+5.1f%%)  "
                   "enc-k %6.0f  dec-k %6.0f  ratio %+6.3fpp\n",
                   cfgs[j].name,
                   exp(gm[j][0] / nfiles),
                   100 * (exp((gm[j][0] - gm[0][0]) / nfiles) - 1),
                   exp(gm[j][1] / nfiles),
                   100 * (exp((gm[j][1] - gm[0][1]) / nfiles) - 1),
                   exp(gm[j][2] / nfiles), exp(gm[j][3] / nfiles),
                   ratio_pp[j] / nfiles);
    }
    pivco_huffman_set_joint_lambda(0.0);
    pivco_huffman_set_joint_granularity(1);
    return 0;
}

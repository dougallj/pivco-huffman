/* bench_lits_windows — the "tables every 5..100 KB" workload on real
 * LZ-preprocessed literal streams (e.g. Silesia .lits from
 * extract_zstd_lits).
 *
 * For each input file and window size G: slice the stream into G-byte
 * windows; per window build a fresh table from the window's own
 * histogram, encode in PIVCO_BLOCK_SIZE blocks, decode, verify.
 *
 * With --joint=L each file is measured twice back-to-back — lambda = 0
 * (plain Huffman + limit) and lambda = L — so the pair shares thermal
 * conditions and the per-file delta is fair even when a long batch
 * drifts.  Reported per (file, lambda):
 *   - ratio INCLUDING the 128-byte nibble-packed lengths header per
 *     window (the real cost of shipping a table at this cadence)
 *   - enc-e2e MB/s: histogram + table build (incl. the joint DP) +
 *     encode kernels — the encoder's true all-in speed
 *   - dec-e2e MB/s: per-window pivco_huffman_build_table_from_code_lens
 *     (the decoder's true per-window cost; the 128-byte nibble unpack
 *     itself is noise) + decode kernels
 *   - enc/dec kernel-only MB/s, table-build us/window, adopted windows
 * plus a geomean summary over all files.
 *
 * Usage: bench_lits_windows [--G=KB] [--joint=L] [--reps=N] [--fse=0|1] file...
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

/* One full measurement of (file, lambda).  Returns 0 on success. */
static int run_file(const uint8_t *data, size_t n, size_t G, int reps,
                    double lam, res_t *r)
{
    pivco_huffman_set_joint_lambda(lam);
    size_t nwin = (n + G - 1) / G;
    pivco_huffman_table_t *tabs = malloc(nwin * sizeof(*tabs));
    pivco_huffman_table_t *dtab = malloc(sizeof(*dtab));
    uint8_t *enc = malloc(n * 2 + nwin * 1024 + 4096);
    uint8_t *dec = malloc(n + 64);
    size_t  *woff = malloc((nwin + 1) * sizeof(size_t));
    if (!tabs || !dtab || !enc || !dec || !woff) return -1;

    /* ---- encoder-side table builds: histogram + build (timed) ---- */
    double t0 = now_sec();
    for (size_t w = 0; w < nwin; w++) {
        size_t wlen = (w + 1) * G <= n ? G : n - w * G;
        uint64_t freq[256] = {0};
        const uint8_t *p = data + w * G;
        for (size_t i = 0; i < wlen; i++) freq[p[i]]++;
        pivco_huffman_build_table(freq, &tabs[w]);
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
            pivco_huffman_table_t t;
            pivco_huffman_build_table(freq, &t);
            if (memcmp(t.code_len, tabs[w].code_len, 256) != 0) adopted++;
        }
        pivco_huffman_set_joint_lambda(lam);
    }

    /* ---- encode kernels (timed) ---- */
    size_t total_enc = 0;
    double enc_best = 0;
    for (int rep = 0; rep < reps; rep++) {
        double t1 = now_sec();
        size_t off = 0;
        for (size_t w = 0; w < nwin; w++) {
            woff[w] = off;
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            for (size_t b = 0; b < wlen; b += PIVCO_BLOCK_SIZE) {
                size_t bn = wlen - b < PIVCO_BLOCK_SIZE ? wlen - b : PIVCO_BLOCK_SIZE;
                size_t el;
                pivco_huffman_encode(data + w * G + b, bn, &tabs[w], enc + off, &el);
                off += el;
            }
        }
        woff[nwin] = off; total_enc = off;
        double mbs = (double)n / (now_sec() - t1) / 1e6;
        if (mbs > enc_best) enc_best = mbs;
    }

    /* ---- decode, kernel only (prebuilt tables) ---- */
    double dec_best = 0;
    for (int rep = 0; rep < reps; rep++) {
        double t1 = now_sec();
        for (size_t w = 0; w < nwin; w++) {
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            size_t off = woff[w], dof = 0;
            while (dof < wlen) {
                size_t consumed;
                pivco_huffman_decode(enc + off, woff[w + 1] - off + 16,
                                     &tabs[w], dec + w * G + dof, &consumed);
                off += consumed;
                dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
            }
        }
        double mbs = (double)n / (now_sec() - t1) / 1e6;
        if (mbs > dec_best) dec_best = mbs;
    }
    if (memcmp(dec, data, n) != 0) return -2;

    /* ---- decode end-to-end: rebuild the decode table per window ---- */
    double dec_e2e = 0;
    for (int rep = 0; rep < reps; rep++) {
        double t1 = now_sec();
        for (size_t w = 0; w < nwin; w++) {
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            pivco_huffman_build_table_from_code_lens(tabs[w].code_len, dtab);
            size_t off = woff[w], dof = 0;
            while (dof < wlen) {
                size_t consumed;
                pivco_huffman_decode(enc + off, woff[w + 1] - off + 16,
                                     dtab, dec + w * G + dof, &consumed);
                off += consumed;
                dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
            }
        }
        double mbs = (double)n / (now_sec() - t1) / 1e6;
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
    free(tabs); free(dtab); free(enc); free(dec); free(woff);
    return 0;
}

int main(int argc, char **argv)
{
    size_t G = 64 * 1024;
    double lam = 0.0;
    int reps = 5, fse = 1;
    int argi = 1;
    for (; argi < argc && argv[argi][0] == '-'; argi++) {
        if (!strncmp(argv[argi], "--G=", 4)) G = (size_t)atoi(argv[argi] + 4) * 1024;
        else if (!strncmp(argv[argi], "--joint=", 8)) lam = atof(argv[argi] + 8);
        else if (!strncmp(argv[argi], "--reps=", 7)) reps = atoi(argv[argi] + 7);
        else if (!strncmp(argv[argi], "--fse=", 6)) fse = atoi(argv[argi] + 6);
    }
    pivco_huffman_set_fse_enabled(fse);
    printf("G=%zuK lambda=%.2f reps=%d fse=%d (%s)\n", G / 1024, lam, reps,
           fse, fse ? "PHA" : "PH");
    printf("%-12s %5s %8s %9s %9s %9s %9s %9s %9s\n",
           "file", "lam", "ratio", "enc-e2e", "dec-e2e", "enc-k", "dec-k",
           "build u/w", "adopted");

    double gm[2][4] = {{0}};        /* [lam][enc_e2e dec_e2e enc_k dec_k] */
    double ratio_pp = 0;
    int nfiles = 0;
    for (; argi < argc; argi++) {
        size_t n;
        uint8_t *data = slurp(argv[argi], &n);
        if (!data) { fprintf(stderr, "skip %s\n", argv[argi]); continue; }
        const char *base = strrchr(argv[argi], '/');
        base = base ? base + 1 : argv[argi];

        res_t r[2];
        int nlam = lam > 0 ? 2 : 1;
        int ok = 1;
        for (int j = 0; j < nlam; j++) {
            double l = j ? lam : 0.0;
            int rc = run_file(data, n, G, reps, l, &r[j]);
            if (rc != 0) { printf("%-12s FAIL rc=%d\n", base, rc); ok = 0; break; }
            char ad[16] = "-";
            if (j) snprintf(ad, sizeof ad, "%d/%d", r[j].adopted, r[j].nwin);
            printf("%-12s %5.2f %8.4f %9.0f %9.0f %9.0f %9.0f %9.1f %9s\n",
                   base, l, r[j].ratio, r[j].enc_e2e, r[j].dec_e2e,
                   r[j].enc_k, r[j].dec_k, r[j].build_us, ad);
        }
        if (ok && nlam == 2) {
            printf("%-12s delta %+7.3fpp %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%%\n",
                   base, 100 * (r[1].ratio - r[0].ratio),
                   100 * (r[1].enc_e2e / r[0].enc_e2e - 1),
                   100 * (r[1].dec_e2e / r[0].dec_e2e - 1),
                   100 * (r[1].enc_k / r[0].enc_k - 1),
                   100 * (r[1].dec_k / r[0].dec_k - 1));
            for (int j = 0; j < 2; j++) {
                gm[j][0] += log(r[j].enc_e2e); gm[j][1] += log(r[j].dec_e2e);
                gm[j][2] += log(r[j].enc_k);   gm[j][3] += log(r[j].dec_k);
            }
            ratio_pp += 100 * (r[1].ratio - r[0].ratio);
            nfiles++;
        }
        free(data);
    }
    if (nfiles > 1) {
        printf("== geomean over %d files (G=%zuK)\n", nfiles, G / 1024);
        for (int j = 0; j < 2; j++)
            printf("lam=%.2f: enc-e2e %6.0f  dec-e2e %6.0f  enc-k %6.0f  dec-k %6.0f\n",
                   j ? lam : 0.0, exp(gm[j][0] / nfiles), exp(gm[j][1] / nfiles),
                   exp(gm[j][2] / nfiles), exp(gm[j][3] / nfiles));
        printf("delta   : enc-e2e %+5.1f%%  dec-e2e %+5.1f%%  enc-k %+5.1f%%  "
               "dec-k %+5.1f%%  ratio %+6.3fpp avg\n",
               100 * (exp((gm[1][0] - gm[0][0]) / nfiles) - 1),
               100 * (exp((gm[1][1] - gm[0][1]) / nfiles) - 1),
               100 * (exp((gm[1][2] - gm[0][2]) / nfiles) - 1),
               100 * (exp((gm[1][3] - gm[0][3]) / nfiles) - 1),
               ratio_pp / nfiles);
    }
    pivco_huffman_set_joint_lambda(0.0);
    return 0;
}

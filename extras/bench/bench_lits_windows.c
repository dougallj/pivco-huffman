/* bench_lits_windows — the "tables every 5..100 KB" workload on real
 * LZ-preprocessed literal streams (e.g. Silesia .lits from
 * extract_zstd_lits).
 *
 * For each input file and window size G: slice the stream into G-byte
 * windows; per window build a fresh table from the window's own
 * histogram, encode in PIVCO_BLOCK_SIZE blocks, decode, verify.
 * Reports, per (file, lambda):
 *   - ratio INCLUDING the 128-byte nibble-packed lengths header per
 *     window (the real cost of shipping a table at this cadence)
 *   - decode MB/s and encode MB/s (kernel only)
 *   - table-build us/window and its share of total encode wall
 *   - windows where the joint DP's result was adopted vs kept
 *
 * Usage: bench_lits_windows [--G=KB] [--joint=L] [--reps=N] [--fse=0|1] file...
 * --fse=0 benches PH (no per-node FSE attempt); default 1 = PHA.
 */
#include "pivco_huffman.h"
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

int main(int argc, char **argv)
{
    size_t G = 64 * 1024;
    double lam = 0.0;
    int reps = 6, fse = 1;
    int argi = 1;
    for (; argi < argc && argv[argi][0] == '-'; argi++) {
        if (!strncmp(argv[argi], "--G=", 4)) G = (size_t)atoi(argv[argi] + 4) * 1024;
        else if (!strncmp(argv[argi], "--joint=", 8)) lam = atof(argv[argi] + 8);
        else if (!strncmp(argv[argi], "--reps=", 7)) reps = atoi(argv[argi] + 7);
        else if (!strncmp(argv[argi], "--fse=", 6)) fse = atoi(argv[argi] + 6);
    }
    pivco_huffman_set_joint_lambda(lam);
    pivco_huffman_set_fse_enabled(fse);
    printf("G=%zuK lambda=%.2f reps=%d fse=%d (%s)\n", G / 1024, lam, reps,
           fse, fse ? "PHA" : "PH");
    printf("%-12s %9s %9s %9s %10s %10s %9s\n",
           "file", "ratio", "dec MB/s", "enc MB/s", "build us/w", "build:enc", "adopted");

    for (; argi < argc; argi++) {
        size_t n;
        uint8_t *data = slurp(argv[argi], &n);
        if (!data) { fprintf(stderr, "skip %s\n", argv[argi]); continue; }
        size_t nwin = (n + G - 1) / G;
        pivco_huffman_table_t *tabs = malloc(nwin * sizeof(*tabs));
        uint8_t *enc = malloc(n * 2 + nwin * 1024 + 4096);
        uint8_t *dec = malloc(n + 64);
        size_t  *woff = malloc((nwin + 1) * sizeof(size_t));   /* enc offset per window */
        size_t  *blen = malloc(nwin * 64 * sizeof(size_t));    /* per-block enc lens */
        (void)blen;

        /* ---- table builds (timed separately) ---- */
        int adopted = 0;
        double t0 = now_sec();
        for (size_t w = 0; w < nwin; w++) {
            size_t wlen = (w + 1) * G <= n ? G : n - w * G;
            uint64_t freq[256] = {0};
            const uint8_t *p = data + w * G;
            for (size_t i = 0; i < wlen; i++) freq[p[i]]++;
            pivco_huffman_build_table(freq, &tabs[w]);
        }
        double build_s = now_sec() - t0;

        /* adoption count: rebuild once with lambda off to compare lens */
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

        /* ---- encode (timed, kernel only) ---- */
        size_t total_enc = 0;
        double enc_best = 0;
        for (int r = 0; r < reps; r++) {
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
            double mbs = (double)n * 1 / (now_sec() - t1) / 1e6;
            if (mbs > enc_best) enc_best = mbs;
        }

        /* ---- decode (timed) ---- */
        double dec_best = 0;
        for (int r = 0; r < reps; r++) {
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
        if (memcmp(dec, data, n) != 0) { printf("%-12s VERIFY FAIL\n", argv[argi]); continue; }

        double ratio = (double)(total_enc + nwin * 128) / (double)n;
        double build_per_w = build_s / (double)nwin * 1e6;
        double enc_wall = (double)n / (enc_best * 1e6);
        const char *base = strrchr(argv[argi], '/');
        printf("%-12s %8.4f %9.0f %9.0f %10.1f %9.2fx %6d/%zu\n",
               base ? base + 1 : argv[argi], ratio, dec_best, enc_best,
               build_per_w, build_s / enc_wall, adopted, nwin);
        free(data); free(tabs); free(enc); free(dec); free(woff); free(blen);
    }
    pivco_huffman_set_joint_lambda(0.0);
    return 0;
}

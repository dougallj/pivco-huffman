/* bench_prof_shares -- decode-kernel time decomposition on the .lits
 * corpus: share of wall spent in flat decode vs each merge kind vs
 * wire reads, per (file, joint config), PH, G=64K, exact solver.
 * Configs: off / lambda=0.1+guard (the production candidate) /
 * lambda=0.25 guard-off.  Produced results/m4-20260712-prof-shares-*.
 *
 * Build (needs the library compiled with profiling):
 *   cmake -B build-prof -DCMAKE_BUILD_TYPE=Release \
 *         -DCMAKE_C_FLAGS="-DPIVCO_PROF=1"
 *   cmake --build build-prof --target pivco_huffman
 *   cc -O2 -DPIVCO_PROF=1 -Iinclude -Isrc extras/bench/bench_prof_shares.c \
 *      build-prof/libpivco_huffman.a -o bench_prof_shares -lm
 * Run:
 *   ./bench_prof_shares testdata/silesia-lits/*.lits
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
static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)sz + 16);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return NULL; }
    fclose(f); *n = (size_t)sz; return b;
}

typedef struct { double wall, flat, vv, cv, cc, popk, wire, memset_t; uint64_t flat_el, merge_el, n; } shares_t;

static void run(const uint8_t *data, size_t n, size_t G, double lam, int guard_off,
                int reps, shares_t *s)
{
    pivco_huffman_set_joint_lambda(lam);
    pivco_huffman_set_joint_granularity(1);
    pivco_huffman_set_joint_guard(guard_off ? 1e9 : 0, guard_off ? 1e9 : 0);
    size_t nwin = (n + G - 1) / G;
    pivco_huffman_codec_table_t *tabs = malloc(nwin * sizeof(*tabs));
    uint8_t *enc = malloc(n * 2 + nwin * 1024 + 4096);
    uint8_t *dec = malloc(n + 64);
    size_t *woff = malloc((nwin + 1) * sizeof(size_t));
    for (size_t w = 0; w < nwin; w++) {
        size_t wlen = (w + 1) * G <= n ? G : n - w * G;
        uint64_t freq[256] = {0};
        const uint8_t *p = data + w * G;
        for (size_t i = 0; i < wlen; i++) freq[p[i]]++;
        pivco_huffman_build_codec_table(freq, &tabs[w]);
    }
    size_t off = 0;
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
    woff[nwin] = off;

    const int inner = (int)(1 + ((size_t)32 << 20) / (n ? n : 1));
    pivco_prof_reset();
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        for (int ii = 0; ii < inner; ii++)
            for (size_t w = 0; w < nwin; w++) {
                size_t wlen = (w + 1) * G <= n ? G : n - w * G;
                size_t o = woff[w], dof = 0;
                while (dof < wlen) {
                    size_t consumed;
                    pivco_huffman_decode_dt(enc + o, woff[w + 1] - o + 16,
                                            &tabs[w].dec, dec + w * G + dof, &consumed);
                    o += consumed;
                    dof += (wlen - dof < PIVCO_BLOCK_SIZE) ? wlen - dof : PIVCO_BLOCK_SIZE;
                }
            }
    s->wall = now_sec() - t0;
    if (memcmp(dec, data, n) != 0) { fprintf(stderr, "ROUNDTRIP FAIL\n"); exit(1); }
    #define T(id) (double)pivco_prof_counters[id].ticks
    s->flat = T(PROF_BU_MERGE_FLAT);   s->vv = T(PROF_BU_MERGE_VEC_VEC);
    s->cv = T(PROF_BU_MERGE_CST_VEC);  s->cc = T(PROF_BU_MERGE_CST_CST);
    s->popk = T(PROF_BU_POPCOUNT_K);   s->memset_t = T(PROF_BU_LEAF_MEMSET);
    s->wire = T(PROF_WIRE_KR) + T(PROF_WIRE_BITMAP_RAW) + T(PROF_WIRE_BITMAP_FSE) + T(PROF_FSE_DEC);
    s->flat_el = pivco_prof_counters[PROF_BU_MERGE_FLAT].elements;
    s->merge_el = pivco_prof_counters[PROF_BU_MERGE_VEC_VEC].elements
                + pivco_prof_counters[PROF_BU_MERGE_CST_VEC].elements
                + pivco_prof_counters[PROF_BU_MERGE_CST_CST].elements;
    s->n = (uint64_t)n * (uint64_t)reps * (uint64_t)inner;
    free(tabs); free(enc); free(dec); free(woff);
}

int main(int argc, char **argv)
{
    pivco_prof_pin_cpu(0);
    pivco_huffman_set_fse_enabled(0);
    double hz = pivco_prof_probe_tick_freq();
    printf("tick freq %.3f GHz\n", hz / 1e9);
    struct { const char *name; double lam; int goff; } cfgs[] = {
        { "off",       0.0,  0 },
        { "L0.1+g",    0.1,  0 },
        { "L0.25-g",   0.25, 1 },
    };
    printf("%-10s %-8s %7s | %5s %5s %5s %5s %4s %4s %5s | %6s %6s\n",
           "file", "cfg", "MB/s", "flat", "vv", "cv", "cc", "popK", "wire", "other",
           "fl-cov", "mrg/el");
    double agg[3][8] = {{0}}; uint64_t aggel[3][3] = {{0}};
    for (int a = 1; a < argc; a++) {
        size_t n; uint8_t *d = slurp(argv[a], &n);
        if (!d) continue;
        const char *base = strrchr(argv[a], '/'); base = base ? base + 1 : argv[a];
        for (int c = 0; c < 3; c++) {
            shares_t s; run(d, n, 64 * 1024, cfgs[c].lam, cfgs[c].goff, 3, &s);
            double wt = s.wall * hz;   /* wall in ticks */
            double known = s.flat + s.vv + s.cv + s.cc + s.popk + s.wire + s.memset_t;
            printf("%-10.10s %-8s %7.0f | %4.1f%% %4.1f%% %4.1f%% %4.1f%% %3.1f%% %3.1f%% %4.1f%% | %5.1f%% %6.3f\n",
                   base, cfgs[c].name, (double)s.n / s.wall / 1e6,
                   100 * s.flat / wt, 100 * s.vv / wt, 100 * s.cv / wt,
                   100 * s.cc / wt, 100 * s.popk / wt, 100 * s.wire / wt,
                   100 * (wt - known) / wt,
                   100.0 * (double)s.flat_el / (double)s.n,
                   (double)s.merge_el / (double)s.n);
            agg[c][0] += s.flat; agg[c][1] += s.vv; agg[c][2] += s.cv; agg[c][3] += s.cc;
            agg[c][4] += s.popk; agg[c][5] += s.wire; agg[c][6] += s.memset_t; agg[c][7] += wt;
            aggel[c][0] += s.flat_el; aggel[c][1] += s.merge_el; aggel[c][2] += s.n;
        }
        free(d);
    }
    printf("== aggregate (time-weighted over all files)\n");
    for (int c = 0; c < 3; c++) {
        double wt = agg[c][7];
        double known = agg[c][0]+agg[c][1]+agg[c][2]+agg[c][3]+agg[c][4]+agg[c][5]+agg[c][6];
        printf("%-8s flat %4.1f%%  vec_vec %4.1f%%  cst_vec %4.1f%%  cst_cst %4.1f%%  popK %3.1f%%  wire %3.1f%%  other %4.1f%%  | flat-cov %5.1f%%  merge-passes/el %5.3f\n",
               cfgs[c].name, 100*agg[c][0]/wt, 100*agg[c][1]/wt, 100*agg[c][2]/wt,
               100*agg[c][3]/wt, 100*agg[c][4]/wt, 100*agg[c][5]/wt, 100*(wt-known)/wt,
               100.0*(double)aggel[c][0]/(double)aggel[c][2],
               (double)aggel[c][1]/(double)aggel[c][2]);
    }
    pivco_huffman_set_joint_lambda(0.0);
    pivco_huffman_set_joint_guard(0, 0);
    return 0;
}

/* bench_pivcoh_speed: decode throughput of extras/pivcoh.h (the stb-style
 * single-header codec) vs the production dispatch decoder, on the MAIN
 * bench distributions.  Streams are encoded once with the production
 * encoder (FSE off — the shared raw-bitmap subset), then each decoder
 * sweeps all blocks.  Methodology: >=100 ms hot-loop DVFS warmup per
 * engine, rep count auto-calibrated to ~120 ms per run, median of 7. */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void bench_init(void);
extern int  bench_num_distributions(void);
extern const char *bench_dist_name(int);
extern int  bench_dist_is_main(int);
extern void bench_generate_symbols(int, uint8_t *, int, uint64_t);
extern int  bench_dist_size(int, int, int);

#define BLK   16384
#define TOTAL (4 * 1024 * 1024)
#define RUNS  7

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint8_t buf[TOTAL], dec[TOTAL + 64];
static uint8_t enc[(TOTAL / BLK) * (2 * BLK)];
static size_t  off[TOTAL / BLK + 1];
static uint8_t mscratch[PIVCOH_SCRATCH_SIZE(BLK)];

typedef void (*sweep_fn)(const pivco_huffman_table_t *, const pivcoh_table *,
                         int nblk);

static void sweep_prod(const pivco_huffman_table_t *T, const pivcoh_table *M,
                       int nblk)
{
    (void)M;
    for (int b = 0; b < nblk; b++) {
        size_t consumed;
        if (pivco_huffman_decode_dt(enc + off[b], off[b + 1] - off[b],
                                    &T->dec, dec + (size_t)b * BLK,
                                    &consumed) != PIVCO_OK)
            exit(fprintf(stderr, "prod decode failed blk %d\n", b));
    }
}

static void sweep_mini(const pivco_huffman_table_t *T, const pivcoh_table *M,
                       int nblk)
{
    (void)T;
    for (int b = 0; b < nblk; b++)
        if (pivcoh_decode(M, enc + off[b], off[b + 1] - off[b],
                          dec + (size_t)b * BLK, BLK, NULL, mscratch) != BLK)
            exit(fprintf(stderr, "pivcoh decode failed blk %d\n", b));
}

/* >=100 ms hot loop (DVFS ramp), calibrate reps to ~120 ms/run, median-of-7. */
static double bench_engine(sweep_fn fn, const pivco_huffman_table_t *T,
                           const pivcoh_table *M, int nblk, size_t bytes)
{
    double t0 = now();
    do { fn(T, M, nblk); } while (now() - t0 < 0.1);

    double t1 = now();
    fn(T, M, nblk);
    double per_sweep = now() - t1;
    int reps = per_sweep > 0 ? (int)(0.12 / per_sweep) + 1 : 1;

    double runs[RUNS];
    for (int r = 0; r < RUNS; r++) {
        double s = now();
        for (int i = 0; i < reps; i++) fn(T, M, nblk);
        runs[r] = (now() - s) / (double)reps;
    }
    qsort(runs, RUNS, sizeof(double), cmp_d);
    return (double)bytes / runs[RUNS / 2] / 1e6;   /* MB/s, median */
}

int main(void)
{
    bench_init();
    pivco_huffman_set_fse_enabled(0);
    static pivco_huffman_table_t T;
    static pivcoh_table M;

    printf("%-14s | %9s %9s | %6s\n", "DIST", "pivcoh", "prod", "ratio");

    for (int d = 0; d < bench_num_distributions(); d++) {
        if (!bench_dist_is_main(d)) continue;
        int total = bench_dist_size(d, TOTAL, BLK);
        if (total > TOTAL) total = TOTAL;
        int nblk = total / BLK;
        bench_generate_symbols(d, buf, total, 42);

        uint64_t freq[256] = {0};
        for (int i = 0; i < total; i++) freq[buf[i]]++;
        if (pivco_huffman_build_table(freq, &T) != PIVCO_OK)
            return fprintf(stderr, "build_table failed\n");
        if (!pivcoh_table_from_lens(&M, T.code_len))
            return fprintf(stderr, "pivcoh table failed\n");

        off[0] = 0;
        for (int b = 0; b < nblk; b++) {
            size_t elen = 0;
            if (pivco_huffman_encode(buf + (size_t)b * BLK, BLK, &T,
                                     enc + off[b], &elen) != PIVCO_OK)
                return fprintf(stderr, "encode failed blk %d\n", b);
            off[b + 1] = off[b] + elen;
        }

        /* correctness gate before timing */
        memset(dec, 0, (size_t)nblk * BLK);
        sweep_mini(&T, &M, nblk);
        if (memcmp(dec, buf, (size_t)nblk * BLK) != 0)
            return fprintf(stderr, "%s: pivcoh output differs\n",
                           bench_dist_name(d));

        double mini = bench_engine(sweep_mini, &T, &M, nblk, (size_t)nblk * BLK);
        double prod = bench_engine(sweep_prod, &T, &M, nblk, (size_t)nblk * BLK);
        printf("%-14s | %9.0f %9.0f | %5.2fx\n",
               bench_dist_name(d), mini, prod, mini / prod);
    }
    return 0;
}

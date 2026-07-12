/* Classic-distribution decode-throughput ladder on joint-cost-model:
 * one global table per distribution (no windowing), pure decode-kernel
 * GB/s per mode (off / nudge / auto / exact at lambda=0.1, production
 * guard + cost-model defaults: gamma=170, FSE tax tau=4, kappa=0).
 * '*' = joint lengths adopted (guard passed); no mark = vetoed/no-op.
 * PHA (per-node FSE on). */
#include "pivco_huffman.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void bench_init(void);
extern int bench_num_distributions(void);
extern const char *bench_dist_name(int);
extern void bench_generate_symbols(int, uint8_t *, int, unsigned);

#define N (4 << 20)
#define REPS 5
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    bench_init();
    pivco_huffman_set_fse_enabled(1);
    struct { const char *name; double lam; int gran; } modes[] = {
        { "off", 0.0, 1 }, { "nudge", 0.1, -1 }, { "auto", 0.1, 0 }, { "exact", 0.1, 1 },
    };
    uint8_t *sym = malloc(N), *enc = malloc((size_t)N * 2 + 4096), *dec = malloc(N + 64);
    /* ~1 s of real work up front so the first measured mode isn't on a
     * cold/ramping clock (first-row "off" measured 12-25 % low without) */
    bench_generate_symbols(0, sym, N, 1);
    double tw = now_sec();
    volatile uint64_t sink = 0;
    while (now_sec() - tw < 1.0)
        for (int i = 0; i < N; i += 64) sink += sym[i];
    printf("%-14s %9s | %-16s %-16s %-16s   (dRatio pp for adopted)\n",
           "dist", "off GB/s", "nudge", "auto", "exact");
    for (int d = 0; d < bench_num_distributions(); d++) {
        bench_generate_symbols(d, sym, N, 0xC0FFEE);
        uint64_t freq[256] = {0};
        for (int i = 0; i < N; i++) freq[sym[i]]++;
        double gbs[4], ratio[4]; int adopted[4]; uint8_t base_lens[256];
        for (int m = 0; m < 4; m++) {
            pivco_huffman_set_joint_lambda(modes[m].lam);
            pivco_huffman_set_joint_granularity(modes[m].gran);
            pivco_huffman_codec_table_t ct;
            pivco_huffman_build_codec_table(freq, &ct);
            if (m == 0) memcpy(base_lens, ct.code_len, 256);
            adopted[m] = m > 0 && memcmp(base_lens, ct.code_len, 256) != 0;
            size_t off = 0, nblk = 0;
            for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                size_t el;
                pivco_huffman_encode_ct(sym + b, PIVCO_BLOCK_SIZE, &ct, enc + off, &el);
                off += el; nblk++;
            }
            ratio[m] = (double)off / N;
            /* warmup + calibration pass (untimed result), then size the
             * timed region to ~12 ms so the ~1 us timer quantum is noise */
            double t0 = now_sec();
            size_t o = 0;
            for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                size_t consumed;
                pivco_huffman_decode_dt(enc + o, off - o + 16, &ct.dec,
                                        dec + b, &consumed);
                o += consumed;
            }
            double t1 = now_sec() - t0;
            int inner = (int)(0.012 / t1) + 1;
            if (inner > 1024) inner = 1024;
            double best = 0;
            for (int r = 0; r < REPS; r++) {
                t0 = now_sec();
                for (int ii = 0; ii < inner; ii++) {
                    o = 0;
                    for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                        size_t consumed;
                        pivco_huffman_decode_dt(enc + o, off - o + 16, &ct.dec,
                                                dec + b, &consumed);
                        o += consumed;
                    }
                }
                double g = (double)N * inner / (now_sec() - t0) / 1e9;
                if (g > best) best = g;
            }
            if (memcmp(dec, sym, N) != 0) { printf("FAIL %s %s\n", bench_dist_name(d), modes[m].name); return 1; }
            gbs[m] = best;
        }
        printf("%-14s %9.2f |", bench_dist_name(d), gbs[0]);
        for (int m = 1; m < 4; m++)
            printf(" %6.2f (%+5.1f%%)%c", gbs[m], 100 * (gbs[m] / gbs[0] - 1),
                   adopted[m] ? '*' : ' ');
        printf("  ");
        for (int m = 1; m < 4; m++)
            if (adopted[m]) printf(" %s%+.2f", m == 1 ? "n" : m == 2 ? "a" : "e",
                                   100 * (ratio[m] - ratio[0]));
        printf("\n");
    }
    pivco_huffman_set_joint_lambda(0.0);
    pivco_huffman_set_joint_granularity(1);
    return 0;
}

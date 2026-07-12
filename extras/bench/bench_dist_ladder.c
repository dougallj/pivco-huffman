/* bench_dist_ladder -- classic-distribution decode-throughput ladder on
 * the cost-model branch: one global table per distribution (no
 * windowing), pure decode-kernel GB/s at off / nudge / auto / exact,
 * lambda = 0.1, production guard unless --guard overrides, PHA unless
 * --fse=0.  '*' = joint lengths adopted; unmarked cells are VERIFIED
 * byte-identical encoded streams (any mismatch is reported loudly), so
 * their deltas are pure measurement noise.
 *
 * Methodology (each guards against a measured artifact):
 *   - timed regions sized to ~12 ms (a single 4 MB pass sits at the
 *     ~1 us timer quantum and best-of picks fiction)
 *   - ~1 s warmup before the first measurement (cold-clock first cell
 *     otherwise reads 12-25 % low)
 *   - timing rounds INTERLEAVED across modes (sequential mode timing
 *     puts the last column a few % low from thermal drift)
 */
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

static pivco_huffman_codec_table_t g_ct[4];

int main(int argc, char **argv)
{
    bench_init();
    int fse = 1;
    for (int a = 1; a < argc; a++) {
        if (!strncmp(argv[a], "--fse=", 6)) fse = atoi(argv[a] + 6);
        else if (!strncmp(argv[a], "--guard=", 8)) {
            double b = strtod(argv[a] + 8, NULL), pc = 0;
            const char *c = strchr(argv[a] + 8, ',');
            if (c) pc = strtod(c + 1, NULL);
            pivco_huffman_set_joint_guard(b, pc);
            printf("guard: bits %.3f pass %.2f\n", b, pc);
        }
    }
    pivco_huffman_set_fse_enabled(fse);
    printf("mode: %s\n", fse ? "PHA" : "PH");
    struct { const char *name; double lam; int gran; } modes[] = {
        { "off", 0.0, 1 }, { "nudge", 0.1, -1 }, { "auto", 0.1, 0 }, { "exact", 0.1, 1 },
    };
    uint8_t *sym = malloc(N), *dec = malloc(N + 64), *enc[4];
    for (int m = 0; m < 4; m++) enc[m] = malloc((size_t)N * 2 + 4096);

    /* ~1 s of real work so the first measurement isn't on a cold clock */
    bench_generate_symbols(0, sym, N, 1);
    double tw = now_sec();
    volatile uint64_t sink = 0;
    while (now_sec() - tw < 1.0)
        for (int i = 0; i < N; i += 64) sink += sym[i];

    long ident_cells = 0, ident_bad = 0;
    printf("%-14s %9s | %-16s %-16s %-16s   (dRatio pp for adopted)\n",
           "dist", "off GB/s", "nudge", "auto", "exact");
    for (int d = 0; d < bench_num_distributions(); d++) {
        bench_generate_symbols(d, sym, N, 0xC0FFEE);
        uint64_t freq[256] = {0};
        for (int i = 0; i < N; i++) freq[sym[i]]++;

        size_t elen[4]; int adopted[4]; double ratio[4];
        for (int m = 0; m < 4; m++) {
            pivco_huffman_set_joint_lambda(modes[m].lam);
            pivco_huffman_set_joint_granularity(modes[m].gran);
            pivco_huffman_build_codec_table(freq, &g_ct[m]);
            adopted[m] = m > 0 && memcmp(g_ct[0].code_len, g_ct[m].code_len, 256) != 0;
            size_t off = 0;
            for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                size_t el;
                pivco_huffman_encode_ct(sym + b, PIVCO_BLOCK_SIZE, &g_ct[m], enc[m] + off, &el);
                off += el;
            }
            elen[m] = off; ratio[m] = (double)off / N;
            if (m > 0 && !adopted[m]) {
                ident_cells++;
                if (elen[m] != elen[0] || memcmp(enc[m], enc[0], elen[0]) != 0) {
                    printf("!! %s %s: lengths equal but STREAM DIFFERS\n",
                           bench_dist_name(d), modes[m].name);
                    ident_bad++;
                }
            }
        }

        /* warmup + region calibration per mode, then interleaved rounds */
        int inner[4]; double best[4] = {0};
        for (int m = 0; m < 4; m++) {
            double t0 = now_sec();
            size_t o = 0;
            for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                size_t consumed;
                pivco_huffman_decode_dt(enc[m] + o, elen[m] - o + 16, &g_ct[m].dec,
                                        dec + b, &consumed);
                o += consumed;
            }
            double t1 = now_sec() - t0;
            if (memcmp(dec, sym, N) != 0) { printf("FAIL %s %s\n", bench_dist_name(d), modes[m].name); return 1; }
            inner[m] = (int)(0.012 / t1) + 1;
            if (inner[m] > 1024) inner[m] = 1024;
        }
        for (int r = 0; r < REPS; r++)
            for (int m = 0; m < 4; m++) {
                double t0 = now_sec();
                for (int ii = 0; ii < inner[m]; ii++) {
                    size_t o = 0;
                    for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                        size_t consumed;
                        pivco_huffman_decode_dt(enc[m] + o, elen[m] - o + 16, &g_ct[m].dec,
                                                dec + b, &consumed);
                        o += consumed;
                    }
                }
                double g = (double)N * inner[m] / (now_sec() - t0) / 1e9;
                if (g > best[m]) best[m] = g;
            }

        printf("%-14s %9.2f |", bench_dist_name(d), best[0]);
        for (int m = 1; m < 4; m++)
            printf(" %6.2f (%+5.1f%%)%c", best[m], 100 * (best[m] / best[0] - 1),
                   adopted[m] ? '*' : ' ');
        printf("  ");
        for (int m = 1; m < 4; m++)
            if (adopted[m]) printf(" %s%+.2f", m == 1 ? "n" : m == 2 ? "a" : "e",
                                   100 * (ratio[m] - ratio[0]));
        printf("\n");
    }
    printf("unadopted cells: %ld, streams byte-identical: %ld%s\n",
           ident_cells, ident_cells - ident_bad, ident_bad ? "  << MISMATCHES ABOVE" : "");
    pivco_huffman_set_joint_lambda(0.0);
    pivco_huffman_set_joint_granularity(1);
    return 0;
}

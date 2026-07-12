/* Null ladder: four IDENTICAL lambda=0 configs through the exact same
 * plumbing as bench_dist_ladder (per-mode table slot + per-mode stream
 * copy), then shared-object variants, to attribute any spread:
 *   A: enc[m], table[m]   (= what bench_dist_ladder does)
 *   B: enc[0], table[0]   (everything shared -- pure timing noise)
 *   C: enc[m], table[0]   (isolates stream-buffer placement)
 *   D: enc[0], table[m]   (isolates table placement)
 * All four "modes" are byte-identical streams from byte-identical
 * tables; any spread beyond B is object-placement effect. */
#include "pivco_huffman.h"
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
int main(void)
{
    bench_init();
    pivco_huffman_set_fse_enabled(1);
    const char *want[] = { "source_c", "csv_numeric", "english" };
    uint8_t *sym = malloc(N), *dec = malloc(N + 64), *enc[4];
    for (int m = 0; m < 4; m++) enc[m] = malloc((size_t)N * 2 + 4096);
    /* warmup */
    bench_generate_symbols(0, sym, N, 1);
    double tw = now_sec(); volatile uint64_t sink = 0;
    while (now_sec() - tw < 1.0) for (int i = 0; i < N; i += 64) sink += sym[i];

    for (int d = 0; d < bench_num_distributions(); d++) {
        int hit = 0;
        for (unsigned k = 0; k < 3; k++) if (!strcmp(bench_dist_name(d), want[k])) hit = 1;
        if (!hit) continue;
        bench_generate_symbols(d, sym, N, 0xC0FFEE);
        uint64_t freq[256] = {0};
        for (int i = 0; i < N; i++) freq[sym[i]]++;
        pivco_huffman_set_joint_lambda(0.0);
        pivco_huffman_build_codec_table(freq, &g_ct[0]);
        for (int m = 1; m < 4; m++) memcpy(&g_ct[m], &g_ct[0], sizeof g_ct[0]);
        size_t elen = 0;
        for (int m = 0; m < 4; m++) {
            size_t off = 0;
            for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                size_t el;
                pivco_huffman_encode_ct(sym + b, PIVCO_BLOCK_SIZE, &g_ct[m], enc[m] + off, &el);
                off += el;
            }
            elen = off;
            if (m && memcmp(enc[m], enc[0], elen)) { printf("STREAM DIFFERS?!\n"); return 1; }
        }
        double t0 = now_sec(); size_t o = 0;
        for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
            size_t consumed;
            pivco_huffman_decode_dt(enc[0] + o, elen - o + 16, &g_ct[0].dec, dec + b, &consumed);
            o += consumed;
        }
        int inner = (int)(0.012 / (now_sec() - t0)) + 1;
        if (memcmp(dec, sym, N)) { printf("FAIL\n"); return 1; }
        printf("%s (inner=%d):\n", bench_dist_name(d), inner);
        for (int ph = 0; ph < 4; ph++) {
            double best[4] = {0};
            for (int r = 0; r < REPS; r++)
                for (int m = 0; m < 4; m++) {
                    const uint8_t *src = (ph == 0 || ph == 2) ? enc[m] : enc[0];
                    const pivco_huffman_decode_table_t *dt =
                        (ph == 0 || ph == 3) ? &g_ct[m].dec : &g_ct[0].dec;
                    double t1 = now_sec();
                    for (int ii = 0; ii < inner; ii++) {
                        size_t oo = 0;
                        for (int b = 0; b < N; b += PIVCO_BLOCK_SIZE) {
                            size_t consumed;
                            pivco_huffman_decode_dt(src + oo, elen - oo + 16, dt, dec + b, &consumed);
                            oo += consumed;
                        }
                    }
                    double g = (double)N * inner / (now_sec() - t1) / 1e9;
                    if (g > best[m]) best[m] = g;
                }
            double mn = 1e9, mx = 0;
            for (int m = 0; m < 4; m++) { if (best[m] < mn) mn = best[m]; if (best[m] > mx) mx = best[m]; }
            const char *pn[] = { "A enc[m]+tab[m]", "B shared both  ", "C enc[m] only  ", "D tab[m] only  " };
            printf("  %s: %.3f %.3f %.3f %.3f  spread %+.1f%%\n",
                   pn[ph], best[0], best[1], best[2], best[3], 100 * (mx / mn - 1));
        }
    }
    return 0;
}

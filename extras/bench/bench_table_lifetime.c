/* bench_table_lifetime: decode throughput of an LZ-literal stream when
 * the Huffman table is rebuilt every G bytes (ryg: "a typical Huffman
 * table will last somewhere between 5 and 100 kilobytes of data").
 *
 * For each granularity G: the stream is split into G-byte segments,
 * each encoded with its own table (code lengths on the "wire"); the
 * timed decode loop rebuilds the decode table per segment and decodes
 * its blocks, exactly like a decoder tracking an adaptive encoder.
 *
 * Feed it .lits files from extras/lz4_lits.c over e.g. the Silesia
 * corpus.  Compile with -DOLD_API against a pre-decode-table checkout
 * (full 13 KB table + build_table_from_code_lens) to reproduce the
 * before/after comparison; the default build uses the ~1 KB
 * pivco_huffman_decode_table_t path.
 *
 * Headline (M4, LZ4HC Silesia literals, geomean end-to-end decode
 * speedup of the 2026-07-10 table-build work vs main):
 *   G=5K 1.52x   10K 1.44x   20K 1.26x   50K 1.12x   100K 1.05x
 * (old build spent 48-64% of decode time in table builds at G=5K;
 * new spends 24-50%). */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef OLD_API
typedef pivco_huffman_table_t dt_t;
#define BUILD_DT(lens, dt)      pivco_huffman_build_table_from_code_lens(lens, dt)
#define DECODE(in, len, dt, out, cons) pivco_huffman_decode(in, len, dt, out, cons)
#else
typedef pivco_huffman_decode_table_t dt_t;
#define BUILD_DT(lens, dt)      pivco_huffman_build_decode_table(lens, dt)
#define DECODE(in, len, dt, out, cons) pivco_huffman_decode_dt(in, len, dt, out, cons)
#endif

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

typedef struct {
    uint8_t  code_lens[256];
    size_t   enc_off;        /* into the big encoded buffer */
    int      n_blocks;
    uint32_t blk_len[8];     /* encoded length per block (<= 8 x 16K per 100K seg) */
    uint32_t blk_raw[8];     /* raw symbol count per block */
} seg_t;

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s FILE.lits\n", argv[0]); return 1; }
    const char *base = strrchr(argv[1], '/');
    base = base ? base + 1 : argv[1];
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long fn = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = (size_t)fn;
    uint8_t *in = malloc(n);
    if (fread(in, 1, n, f) != n) return 1;
    fclose(f);

    static const size_t GRAN[] = { 5120, 10240, 20480, 51200, 102400 };
    enum { NG = sizeof(GRAN) / sizeof(GRAN[0]) };

    uint8_t *out = malloc(n);
    uint8_t *enc = malloc(2 * n + (n / 4096 + 16) * 64);
    dt_t *dt = malloc(sizeof(dt_t));

    printf("%-14s %8s %8s %10s %12s %12s %10s\n",
           "file", "G", "tables", "ns/table", "dec MB/s", "d+b MB/s", "build %");

    for (int gi = 0; gi < NG; gi++) {
        size_t G = GRAN[gi];
        size_t n_segs = (n + G - 1) / G;
        seg_t *segs = malloc(n_segs * sizeof(seg_t));

        /* ---- encode (untimed): one table per G-byte segment ---- */
        size_t eoff = 0;
        for (size_t si = 0; si < n_segs; si++) {
            size_t s0 = si * G;
            size_t slen = (s0 + G <= n) ? G : n - s0;
            uint64_t freq[256] = {0};
            for (size_t i = 0; i < slen; i++) freq[in[s0 + i]]++;
            pivco_huffman_table_t *t = malloc(sizeof(*t));
            if (pivco_huffman_build_table(freq, t) != PIVCO_OK) return 2;
            memcpy(segs[si].code_lens, t->code_len, 256);
            segs[si].enc_off = eoff;
            segs[si].n_blocks = 0;
            for (size_t b0 = 0; b0 < slen; b0 += 16384) {
                size_t blen = (b0 + 16384 <= slen) ? 16384 : slen - b0;
                size_t elen = 0;
                if (pivco_huffman_encode(in + s0 + b0, blen, t,
                                         enc + eoff, &elen) != PIVCO_OK) return 3;
                int bi = segs[si].n_blocks++;
                segs[si].blk_len[bi] = (uint32_t)elen;
                segs[si].blk_raw[bi] = (uint32_t)blen;
                eoff += elen;
            }
            free(t);
        }

        /* ---- verify one full decode pass ---- */
        memset(out, 0, n);
        for (size_t si = 0; si < n_segs; si++) {
            if (BUILD_DT(segs[si].code_lens, dt) != PIVCO_OK) return 4;
            size_t o = si * G, e = segs[si].enc_off;
            for (int b = 0; b < segs[si].n_blocks; b++) {
                size_t cons = 0;
                if (DECODE(enc + e, segs[si].blk_len[b], dt, out + o, &cons)
                        != PIVCO_OK || cons != segs[si].blk_len[b]) return 5;
                o += segs[si].blk_raw[b];
                e += segs[si].blk_len[b];
            }
        }
        if (memcmp(in, out, n) != 0) { fprintf(stderr, "MISMATCH G=%zu\n", G); return 6; }

        /* ---- warm-up ~150ms ---- */
        for (double t0 = now_ns(); now_ns() - t0 < 150e6; )
            for (size_t si = 0; si < n_segs; si += 7)
                BUILD_DT(segs[si].code_lens, dt);

        /* ---- timed: decode incl. per-segment table build ---- */
        double best_comb = 1e30, best_build = 1e30;
        for (int rep = 0; rep < 7; rep++) {
            double t0 = now_ns();
            for (size_t si = 0; si < n_segs; si++) {
                BUILD_DT(segs[si].code_lens, dt);
                size_t o = si * G, e = segs[si].enc_off;
                for (int b = 0; b < segs[si].n_blocks; b++) {
                    size_t cons = 0;
                    DECODE(enc + e, segs[si].blk_len[b], dt, out + o, &cons);
                    o += segs[si].blk_raw[b];
                    e += segs[si].blk_len[b];
                }
            }
            double t1 = now_ns();
            int K = (int)(256 / n_segs) + 1;   /* beat timer granularity */
            for (int k = 0; k < K; k++)
                for (size_t si = 0; si < n_segs; si++)
                    BUILD_DT(segs[si].code_lens, dt);
            double t2 = now_ns();
            if (t1 - t0 < best_comb)  best_comb  = t1 - t0;
            if ((t2 - t1) / K < best_build) best_build = (t2 - t1) / K;
        }

        double mbps_comb = (double)n / best_comb * 1e9 / 1e6;
        double mbps_dec  = (double)n / (best_comb - best_build) * 1e9 / 1e6;
        printf("%-14s %7zuK %8zu %10.0f %12.0f %12.0f %9.1f%%\n",
               base, G / 1024, n_segs, best_build / (double)n_segs,
               mbps_dec, mbps_comb, 100.0 * best_build / best_comb);
        free(segs);
    }
    return 0;
}

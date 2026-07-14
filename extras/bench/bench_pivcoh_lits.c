/* bench_pivcoh_lits — pivcoh.h vs the production codec on the
 * "fresh table every G bytes" workload, on real LZ-preprocessed literal
 * streams (the Silesia .lits corpus; see testdata/silesia-lits/README.md
 * on the silesia-lits branch — `git checkout silesia-lits -- testdata`
 * fetches it without merging the binaries).
 *
 * For each input file and window size G: slice the stream into G-byte
 * windows; per window build a fresh table from the window's own
 * histogram, encode in PIVCO_BLOCK_SIZE blocks, decode.  Both engines
 * are gated byte-identical (code lengths and wire) before any timing,
 * then timed separately on:
 *
 *   enc-e2e MB/s: histogram + table build + encode kernels
 *                 (pivcoh_table_from_freqs+pivcoh_encode vs
 *                  build_codec_table+encode_ct)
 *   dec-e2e MB/s: per-window decode-table build + decode kernels
 *                 (pivcoh_table_from_lens+pivcoh_decode vs
 *                  build_decode_table+decode_dt)
 *   ebuild/dbuild us/window: the table builds alone (histograms
 *                 precomputed), for the "how much is table build"
 *                 question.  NB the decode-side builders differ in
 *                 scope: pivcoh_table_from_lens also derives the
 *                 encoder fields (there is only one pivcoh builder),
 *                 build_decode_table is decode-only.
 *
 * Usage: bench_pivcoh_lits [--G=KB] [--reps=N] [--joint=TIER]
 *                          [--lambda=F] [--gamma=F] file...
 *   --G given: that window size only.  Default: sweep G = 4/16/64/128K
 *   with per-file rows and a geomean summary per G.
 *   --joint: off (default) | nudge | auto | exact | 2 | 4 | 8 — run the
 *   pivcoh side with the joint length/shape pass at that tier.  Joint
 *   lengths diverge from production's by design, so the byte-identity
 *   gate is replaced by cross-decode both ways, and each row gains the
 *   compression delta (pp, header included) and the adoption count.
 * FSE is forced off (pivcoh speaks the raw-bitmap subset). */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include "pivco_huffman.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLK PIVCO_BLOCK_SIZE

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint8_t *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)sz + 16);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

/* Per-file state shared by the pass functions (house style: file-scope,
 * like the other extras benches). */
static const uint8_t *D;            /* input stream */
static size_t   N, G, NW;           /* file size, window size, #windows */
static uint64_t (*FR)[256];         /* per-window histogram (for *build) */
static uint8_t  (*LEN)[256];        /* per-window code lengths */
static uint8_t  *ENC, *DEC;         /* ciphertext (+ WOFF), decode sink */
static uint8_t  *ENC2;              /* joint mode: production's encode sink —
                                       its streams differ from the joint
                                       ciphertext ENC/WOFF/LEN describe, so
                                       the timed pass must not overwrite it */
static size_t   *WOFF;              /* per-window ciphertext offsets */
static uint8_t  escratch[PIVCOH_SCRATCH_SIZE(BLK)];
static uint8_t  dscratch[PIVCOH_DECODE_SCRATCH_SIZE(BLK)];
static uint8_t  gatebuf[PIVCOH_ENCODE_BOUND(BLK)];
static volatile unsigned g_sink;    /* defeats DCE of the header-inlined builders */

static int          JOINT;          /* 0 = plain; else joint tier active */
static pivcoh_joint JP;             /* tier/lambda/gamma from the CLI */
static uint8_t      jscratch[PIVCOH_JOINT_SCRATCH_SIZE];
static size_t       j_adopt, j_bytes, p_bytes;   /* per-file, from setup */

static size_t wlen_of(size_t w) { return (w + 1) * G <= N ? G : N - w * G; }

/* ---- whole-file passes, one per (engine, metric) ---- */

static void pass_enc_prod(void)
{
    uint8_t *sink = JOINT ? ENC2 : ENC;
    size_t off = 0;
    for (size_t w = 0; w < NW; w++) {
        size_t wlen = wlen_of(w);
        uint64_t f[256] = {0};
        const uint8_t *p = D + w * G;
        for (size_t i = 0; i < wlen; i++) f[p[i]]++;
        pivco_huffman_codec_table_t ct;
        pivco_huffman_build_codec_table(f, &ct);
        for (size_t b = 0; b < wlen; b += BLK) {
            size_t bn = wlen - b < BLK ? wlen - b : BLK, el;
            pivco_huffman_encode_ct(p + b, bn, &ct, sink + off, &el);
            off += el;
        }
    }
}

static void pass_enc_mini(void)
{
    size_t off = 0;
    for (size_t w = 0; w < NW; w++) {
        size_t wlen = wlen_of(w);
        uint64_t f[256] = {0};
        const uint8_t *p = D + w * G;
        for (size_t i = 0; i < wlen; i++) f[p[i]]++;
        pivcoh_table t;
        if (JOINT) pivcoh_table_from_freqs_joint(&t, f, &JP, jscratch);
        else       pivcoh_table_from_freqs(&t, f);
        for (size_t b = 0; b < wlen; b += BLK) {
            size_t bn = wlen - b < BLK ? wlen - b : BLK;
            off += (size_t)pivcoh_encode(&t, p + b, bn, ENC + off,
                                         PIVCOH_ENCODE_BOUND(bn), escratch);
        }
    }
}

static void pass_dec_prod(void)
{
    for (size_t w = 0; w < NW; w++) {
        size_t wlen = wlen_of(w), off = WOFF[w], dof = 0;
        pivco_huffman_decode_table_t dt;
        pivco_huffman_build_decode_table(LEN[w], &dt);
        while (dof < wlen) {
            size_t consumed;
            pivco_huffman_decode_dt(ENC + off, WOFF[w + 1] - off, &dt,
                                    DEC + w * G + dof, &consumed);
            off += consumed;
            dof += wlen - dof < BLK ? wlen - dof : BLK;
        }
    }
}

static void pass_dec_mini(void)
{
    for (size_t w = 0; w < NW; w++) {
        size_t wlen = wlen_of(w), off = WOFF[w], dof = 0;
        pivcoh_table t;
        pivcoh_table_from_lens(&t, LEN[w]);
        while (dof < wlen) {
            size_t consumed;
            ptrdiff_t dn = pivcoh_decode(&t, ENC + off, WOFF[w + 1] - off,
                                         DEC + w * G + dof, wlen - dof,
                                         &consumed, dscratch);
            if (dn < 0) exit(fprintf(stderr, "pivcoh decode failed\n"));
            off += consumed;
            dof += (size_t)dn;
        }
    }
}

static void pass_ebuild_prod(void)
{
    for (size_t w = 0; w < NW; w++) {
        pivco_huffman_codec_table_t ct;
        pivco_huffman_build_codec_table(FR[w], &ct);
    }
}

static void pass_ebuild_mini(void)
{
    for (size_t w = 0; w < NW; w++) {
        pivcoh_table t;
        if (JOINT) pivcoh_table_from_freqs_joint(&t, FR[w], &JP, jscratch);
        else       pivcoh_table_from_freqs(&t, FR[w]);
        g_sink += t.num_ranks;
    }
}

static void pass_dbuild_prod(void)
{
    for (size_t w = 0; w < NW; w++) {
        pivco_huffman_decode_table_t dt;
        pivco_huffman_build_decode_table(LEN[w], &dt);
    }
}

static void pass_dbuild_mini(void)
{
    for (size_t w = 0; w < NW; w++) {
        pivcoh_table t;
        pivcoh_table_from_lens(&t, LEN[w]);
        g_sink += t.num_ranks;
    }
}

/* Best-of-reps seconds for one whole-file pass; small files loop enough
 * passes inside the timed region to span ~8 MB. */
static double timeit(void (*fn)(void), int reps)
{
    const int inner = (int)(1 + ((size_t)8 << 20) / (N ? N : 1));
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < inner; i++) fn();
        double dt = (now_sec() - t0) / inner;
        if (dt < best) best = dt;
    }
    return best;
}

/* Untimed setup + gates: reference ciphertext with production, pivcoh
 * gated to identical code lengths and wire bytes, both decoders
 * round-trip.  Returns 0 on success. */
static int setup_and_gate(const char *base)
{
    j_adopt = j_bytes = p_bytes = 0;
    NW   = (N + G - 1) / G;
    FR   = malloc(NW * sizeof(*FR));
    LEN  = malloc(NW * sizeof(*LEN));
    WOFF = malloc((NW + 1) * sizeof(*WOFF));
    ENC  = malloc(2 * N + NW * 1024 + 65536);
    ENC2 = JOINT ? malloc(2 * N + NW * 1024 + 65536) : NULL;
    DEC  = malloc(N + 64);
    if (!FR || !LEN || !WOFF || !ENC || !DEC || (JOINT && !ENC2)) return -1;

    size_t off = 0;
    for (size_t w = 0; w < NW; w++) {
        WOFF[w] = off;
        size_t wlen = wlen_of(w);
        const uint8_t *p = D + w * G;
        memset(FR[w], 0, sizeof(FR[w]));
        for (size_t i = 0; i < wlen; i++) FR[w][p[i]]++;

        pivco_huffman_codec_table_t ct;
        if (pivco_huffman_build_codec_table(FR[w], &ct) != PIVCO_OK)
            return fprintf(stderr, "%s: prod build failed w=%zu\n", base, w);
        pivcoh_table t;
        if (JOINT) {
            /* Joint lengths diverge from production's by design; the
             * timed passes and the ciphertext use them for BOTH engines
             * (production decodes from LEN too — "any decoder reads the
             * output").  Production's plain encode still runs, untimed,
             * for the compression-delta column. */
            if (!pivcoh_table_from_freqs_joint(&t, FR[w], &JP, jscratch))
                return fprintf(stderr, "%s: joint build failed w=%zu\n", base, w);
            memcpy(LEN[w], t.code_len, 256);
            if (memcmp(t.code_len, ct.code_len, 256) != 0) j_adopt++;
            for (size_t b = 0; b < wlen; b += BLK) {
                size_t bn = wlen - b < BLK ? wlen - b : BLK, el;
                if (pivco_huffman_encode_ct(p + b, bn, &ct, gatebuf, &el) != PIVCO_OK)
                    return fprintf(stderr, "%s: prod encode failed w=%zu\n", base, w);
                p_bytes += el;
                ptrdiff_t ml = pivcoh_encode(&t, p + b, bn, ENC + off,
                                             PIVCOH_ENCODE_BOUND(bn), escratch);
                if (ml < 0)
                    return fprintf(stderr, "%s: joint encode failed w=%zu\n", base, w);
                off += (size_t)ml;
            }
        } else {
            memcpy(LEN[w], ct.code_len, 256);
            if (!pivcoh_table_from_freqs(&t, FR[w]) ||
                memcmp(t.code_len, LEN[w], 256) != 0)
                return fprintf(stderr, "%s: pivcoh lens differ w=%zu\n", base, w);

            for (size_t b = 0; b < wlen; b += BLK) {
                size_t bn = wlen - b < BLK ? wlen - b : BLK, el;
                if (pivco_huffman_encode_ct(p + b, bn, &ct, ENC + off, &el) != PIVCO_OK)
                    return fprintf(stderr, "%s: prod encode failed w=%zu\n", base, w);
                ptrdiff_t ml = pivcoh_encode(&t, p + b, bn, gatebuf,
                                             sizeof(gatebuf), escratch);
                if (ml != (ptrdiff_t)el || memcmp(gatebuf, ENC + off, el) != 0)
                    return fprintf(stderr, "%s: pivcoh wire differs w=%zu\n", base, w);
                off += el;
            }
        }
    }
    WOFF[NW] = off;
    j_bytes += off;

    memset(DEC, 0, N);
    pass_dec_mini();
    if (memcmp(DEC, D, N) != 0)
        return fprintf(stderr, "%s: pivcoh decode differs\n", base);
    memset(DEC, 0, N);
    pass_dec_prod();
    if (memcmp(DEC, D, N) != 0)
        return fprintf(stderr, "%s: prod decode differs\n", base);
    return 0;
}

int main(int argc, char **argv)
{
    size_t gs[4] = { 4096, 16384, 65536, 131072 };
    int ngs = 4, reps = 5, argi = 1;
    const char *jname = "off";
    pivcoh_joint jp = PIVCOH_JOINT_DEFAULTS;
    JP = jp;
    for (; argi < argc && argv[argi][0] == '-'; argi++) {
        if (!strncmp(argv[argi], "--G=", 4)) {
            gs[0] = (size_t)atoi(argv[argi] + 4) * 1024;
            ngs = 1;
        } else if (!strncmp(argv[argi], "--reps=", 7)) {
            reps = atoi(argv[argi] + 7);
        } else if (!strncmp(argv[argi], "--joint=", 8)) {
            jname = argv[argi] + 8;
            JOINT = 1;
            if      (!strcmp(jname, "off"))   JOINT = 0;
            else if (!strcmp(jname, "nudge")) JP.gran = -1;
            else if (!strcmp(jname, "auto"))  JP.gran = 0;
            else if (!strcmp(jname, "exact")) JP.gran = 1;
            else if (!strcmp(jname, "2"))     JP.gran = 2;
            else if (!strcmp(jname, "4"))     JP.gran = 4;
            else if (!strcmp(jname, "8"))     JP.gran = 8;
            else { fprintf(stderr, "bad --joint tier\n"); return 1; }
        } else if (!strncmp(argv[argi], "--lambda=", 9)) {
            JP.lambda = (float)atof(argv[argi] + 9);
        } else if (!strncmp(argv[argi], "--gamma=", 8)) {
            JP.gamma = (float)atof(argv[argi] + 8);
        } else {
            fprintf(stderr, "usage: %s [--G=KB] [--reps=N] [--joint=TIER]"
                            " [--lambda=F] [--gamma=F] file...\n", argv[0]);
            return 1;
        }
    }
    pivco_huffman_set_fse_enabled(0);

    printf("blocks=%d fse=0 reps=%d joint=%s", (int)BLK, reps, jname);
    if (JOINT)
        printf(" lambda=%.3f gamma=%.0f", (double)JP.lambda, (double)JP.gamma);
    printf("\n");
    printf("%-13s %5s | %-22s | %-22s | %-11s | %-11s\n",
           "", "", "enc-e2e MB/s", "dec-e2e MB/s",
           "ebuild us/w", "dbuild us/w");
    printf("%-13s %5s | %6s %6s %6s | %6s %6s %6s | %5s %5s | %5s %5s\n",
           "file", "G", "pivcoh", "prod", "ratio", "pivcoh", "prod", "ratio",
           "pvch", "prod", "pvch", "prod");

    for (int gi = 0; gi < ngs; gi++) {
        G = gs[gi];
        double gm[4] = {0};     /* log-sums: enc m/p, dec m/p */
        size_t tj = 0, tp = 0, tn = 0, ta = 0, tw = 0;  /* joint ratio totals */
        int nfiles = 0;
        for (int ai = argi; ai < argc; ai++) {
            uint8_t *data = slurp(argv[ai], &N);
            if (!data) { fprintf(stderr, "skip %s\n", argv[ai]); continue; }
            D = data;
            const char *base = strrchr(argv[ai], '/');
            base = base ? base + 1 : argv[ai];

            if (setup_and_gate(base) != 0) return 1;

            /* >=100 ms hot loop first — DVFS ramp (same discipline as
             * bench_pivcoh_speed); best-of-reps alone doesn't guarantee
             * the first-timed engine runs at full clocks. */
            double w0 = now_sec();
            do { pass_enc_prod(); pass_enc_mini(); } while (now_sec() - w0 < 0.1);

            double em = (double)N / timeit(pass_enc_mini, reps) / 1e6;
            double ep = (double)N / timeit(pass_enc_prod, reps) / 1e6;
            double dm = (double)N / timeit(pass_dec_mini, reps) / 1e6;
            double dp = (double)N / timeit(pass_dec_prod, reps) / 1e6;
            double ebm = timeit(pass_ebuild_mini, reps) / (double)NW * 1e6;
            double ebp = timeit(pass_ebuild_prod, reps) / (double)NW * 1e6;
            double dbm = timeit(pass_dbuild_mini, reps) / (double)NW * 1e6;
            double dbp = timeit(pass_dbuild_prod, reps) / (double)NW * 1e6;

            printf("%-13s %4zuK | %6.0f %6.0f %5.2fx | %6.0f %6.0f %5.2fx |"
                   " %5.2f %5.2f | %5.2f %5.2f",
                   base, G / 1024, em, ep, em / ep, dm, dp, dm / dp,
                   ebm, ebp, dbm, dbp);
            if (JOINT)
                printf(" | %+.3fpp %zu/%zu",
                       ((double)j_bytes - (double)p_bytes) / (double)N * 100.0,
                       j_adopt, NW);
            printf("\n");
            gm[0] += log(em); gm[1] += log(ep);
            gm[2] += log(dm); gm[3] += log(dp);
            tj += j_bytes; tp += p_bytes; tn += N; ta += j_adopt; tw += NW;
            nfiles++;
            free(data);
            free(FR); free(LEN); free(WOFF); free(ENC); free(ENC2); free(DEC);
        }
        if (nfiles > 1) {
            printf("%-13s %4zuK | %6.0f %6.0f %5.2fx | %6.0f %6.0f %5.2fx |"
                   " geomean over %d files",
                   "== geomean", G / 1024,
                   exp(gm[0] / nfiles), exp(gm[1] / nfiles),
                   exp((gm[0] - gm[1]) / nfiles),
                   exp(gm[2] / nfiles), exp(gm[3] / nfiles),
                   exp((gm[2] - gm[3]) / nfiles), nfiles);
            if (JOINT)
                printf(" | %+.3fpp %zu/%zu",
                       ((double)tj - (double)tp) / (double)tn * 100.0,
                       ta, tw);
            printf("\n");
        }
    }
    return 0;
}

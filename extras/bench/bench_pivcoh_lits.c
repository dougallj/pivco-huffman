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
 *                          [--lambda=F] [--gamma=F] [--profile=m1|m4] file...
 *   --G given: that window size only.  Default: sweep G = 4/16/64/128K
 *   with per-file rows and a geomean summary per G.
 *   --joint: off (default) | coarse | auto | exact | 2 | 4 | 8 — run the
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
static uint8_t  (*LEN)[256];        /* per-window pivcoh code lengths */
static uint8_t  (*LENP)[256];       /* per-window production code lengths */
static uint8_t  *ENC, *DEC;         /* pivcoh ciphertext (+ WOFF), decode sink */
static uint8_t  *ENC2;              /* production ciphertext (+ WOFFP): the v4
                                       pivcoh wire is not production-decodable
                                       (or vice versa), so each engine keeps
                                       its own streams */
static size_t   *WOFF;              /* per-window ciphertext offsets */
static size_t   *WOFFP;             /* production per-window offsets */
static size_t   *WSTART;            /* per-window byte offset into D (NW+1) */
static int      BLOCKS;             /* --blocks: one window per zstd block */
static int      TABLES;             /* --tables: merge blocks per zstd HUF-table lifetime */
static const char *CURPATH;         /* current input path (for the sidecars) */
static uint8_t  escratch[PIVCOH_SCRATCH_SIZE(BLK)];
static uint8_t  dscratch[PIVCOH_DECODE_SCRATCH_SIZE(BLK)];
static volatile unsigned g_sink;    /* defeats DCE of the header-inlined builders */

static int          JOINT;          /* 0 = plain; else joint tier active */
static pivcoh_joint JP;             /* tier/lambda/gamma from the CLI */
static uint8_t      jscratch[PIVCOH_JOINT_SCRATCH_SIZE];
static size_t       j_adopt, j_bytes, p_bytes;   /* per-file, from setup */

static size_t wlen_of(size_t w) { return WSTART[w + 1] - WSTART[w]; }

/* Partition the file into windows.  Default: fixed G bytes.  --blocks:
 * one window per zstd block (boundaries from the .litblk sidecar, u32 LE
 * per-block litSize).  --tables: merge consecutive blocks that share a
 * zstd HUF table into one window — a new window starts only where zstd
 * built a fresh table (.lithdr type==2 Compressed); Raw/RLE/Treeless
 * blocks (0/1/3) extend the current window, mirroring zstd's real
 * table-build cadence.  Sets NW and WSTART[0..NW]. */
static int build_windows(const char *base)
{
    if (BLOCKS || TABLES) {
        size_t plen = strlen(CURPATH);
        if (plen < 5 || strcmp(CURPATH + plen - 5, ".lits") != 0)
            return fprintf(stderr, "%s: --blocks/--tables needs a .lits input\n", base);
        char path[4096];
        snprintf(path, sizeof path, "%.*s.litblk", (int)(plen - 5), CURPATH);
        size_t bn;
        uint8_t *b = slurp(path, &bn);
        if (!b)     return fprintf(stderr, "%s: cannot read %s\n", base, path);
        if (bn % 4) return fprintf(stderr, "%s: %s not u32-aligned\n", base, path);
        size_t m = bn / 4;                       /* zstd block count */
        uint8_t *hdr = NULL;
        if (TABLES) {
            snprintf(path, sizeof path, "%.*s.lithdr", (int)(plen - 5), CURPATH);
            size_t hn;
            hdr = slurp(path, &hn);
            if (!hdr)     { free(b); return fprintf(stderr, "%s: cannot read %s\n", base, path); }
            if (hn != m)  { free(b); free(hdr);
                return fprintf(stderr, "%s: lithdr %zu != litblk %zu\n", base, hn, m); }
        }
        /* a block starts a window iff it's the first, or (--tables) it
         * built a fresh HUF table; --blocks alone => every block starts one */
        #define WIN_START(i) (!TABLES || (i) == 0 || hdr[i] == 2)
        NW = 0;
        for (size_t i = 0; i < m; i++) if (WIN_START(i)) NW++;
        WSTART = malloc((NW + 1) * sizeof(*WSTART));
        if (!WSTART) { free(b); free(hdr); return -1; }
        size_t off = 0, w = 0;
        for (size_t i = 0; i < m; i++) {
            uint32_t e = (uint32_t)b[4 * i]       | (uint32_t)b[4 * i + 1] << 8 |
                         (uint32_t)b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
            if (WIN_START(i)) WSTART[w++] = off;
            off += e;
        }
        #undef WIN_START
        WSTART[NW] = off;
        free(b); free(hdr);
        if (off != N)
            return fprintf(stderr, "%s: litblk sum %zu != file size %zu\n",
                           base, off, N);
    } else {
        NW = (N + G - 1) / G;
        WSTART = malloc((NW + 1) * sizeof(*WSTART));
        if (!WSTART) return -1;
        for (size_t w = 0; w < NW; w++) WSTART[w] = w * G;
        WSTART[NW] = N;
    }
    return 0;
}

/* ---- whole-file passes, one per (engine, metric) ---- */

static void pass_enc_prod(void)
{
    uint8_t *sink = ENC2;
    size_t off = 0;
    for (size_t w = 0; w < NW; w++) {
        size_t wlen = wlen_of(w);
        uint64_t f[256] = {0};
        const uint8_t *p = D + WSTART[w];
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
        const uint8_t *p = D + WSTART[w];
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
        size_t wlen = wlen_of(w), off = WOFFP[w], dof = 0;
        pivco_huffman_decode_table_t dt;
        pivco_huffman_build_decode_table(LENP[w], &dt);
        while (dof < wlen) {
            size_t consumed;
            pivco_huffman_decode_dt(ENC2 + off, WOFFP[w + 1] - off, &dt,
                                    DEC + WSTART[w] + dof, &consumed);
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
                                         DEC + WSTART[w] + dof, wlen - dof,
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
        pivco_huffman_build_decode_table(LENP[w], &dt);
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

/* Untimed setup + gates: each engine encodes its own ciphertext (the
 * v4 pivcoh wire is not production-decodable), both round-trip, and in
 * plain mode the code LENGTHS are still gated identical — the tree
 * logic is unchanged, only the wire around it moved.  Sizes include
 * each side's own per-window table header (production: 128 B nibbles;
 * pivcoh: the v4 lens wire).  Returns 0 on success. */
static int setup_and_gate(const char *base)
{
    j_adopt = j_bytes = p_bytes = 0;
    if (build_windows(base) != 0) return -1;
    FR    = malloc(NW * sizeof(*FR));
    LEN   = malloc(NW * sizeof(*LEN));
    LENP  = malloc(NW * sizeof(*LENP));
    WOFF  = malloc((NW + 1) * sizeof(*WOFF));
    WOFFP = malloc((NW + 1) * sizeof(*WOFFP));
    ENC  = malloc(2 * N + NW * 1024 + 65536);
    ENC2 = malloc(2 * N + NW * 1024 + 65536);
    DEC  = malloc(N + 64);
    if (!FR || !LEN || !LENP || !WOFF || !WOFFP || !ENC || !ENC2 || !DEC)
        return -1;

    size_t off = 0, offp = 0;
    for (size_t w = 0; w < NW; w++) {
        WOFF[w] = off;
        WOFFP[w] = offp;
        size_t wlen = wlen_of(w);
        const uint8_t *p = D + WSTART[w];
        memset(FR[w], 0, sizeof(FR[w]));
        for (size_t i = 0; i < wlen; i++) FR[w][p[i]]++;

        pivco_huffman_codec_table_t ct;
        if (pivco_huffman_build_codec_table(FR[w], &ct) != PIVCO_OK)
            return fprintf(stderr, "%s: prod build failed w=%zu\n", base, w);
        memcpy(LENP[w], ct.code_len, 256);
        pivcoh_table t;
        if (JOINT) {
            if (!pivcoh_table_from_freqs_joint(&t, FR[w], &JP, jscratch))
                return fprintf(stderr, "%s: joint build failed w=%zu\n", base, w);
            if (memcmp(t.code_len, ct.code_len, 256) != 0) j_adopt++;
        } else {
            if (!pivcoh_table_from_freqs(&t, FR[w]) ||
                memcmp(t.code_len, ct.code_len, 256) != 0)
                return fprintf(stderr, "%s: pivcoh lens differ w=%zu\n", base, w);
        }
        memcpy(LEN[w], t.code_len, 256);
        uint8_t lwire[PIVCOH_LENS_WIRE_BOUND];
        j_bytes += (size_t)pivcoh_lens_wire_write(lwire, t.code_len);
        p_bytes += 128;                        /* production nibble header */
        for (size_t b = 0; b < wlen; b += BLK) {
            size_t bn = wlen - b < BLK ? wlen - b : BLK, el;
            if (pivco_huffman_encode_ct(p + b, bn, &ct, ENC2 + offp, &el) != PIVCO_OK)
                return fprintf(stderr, "%s: prod encode failed w=%zu\n", base, w);
            offp += el;
            ptrdiff_t ml = pivcoh_encode(&t, p + b, bn, ENC + off,
                                         PIVCOH_ENCODE_BOUND(bn), escratch);
            if (ml < 0)
                return fprintf(stderr, "%s: pivcoh encode failed w=%zu\n", base, w);
            off += (size_t)ml;
        }
    }
    WOFF[NW] = off;
    WOFFP[NW] = offp;
    j_bytes += off;
    p_bytes += offp;

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
        } else if (!strcmp(argv[argi], "--blocks")) {
            BLOCKS = 1;
        } else if (!strcmp(argv[argi], "--tables")) {
            TABLES = 1;
        } else if (!strncmp(argv[argi], "--reps=", 7)) {
            reps = atoi(argv[argi] + 7);
        } else if (!strncmp(argv[argi], "--joint=", 8)) {
            jname = argv[argi] + 8;
            JOINT = 1;
            if      (!strcmp(jname, "off"))   JOINT = 0;
            else if (!strcmp(jname, "coarse") ||
                     !strcmp(jname, "nudge"))  JP.gran = -1;
            else if (!strcmp(jname, "auto"))  JP.gran = 0;
            else if (!strcmp(jname, "exact")) JP.gran = 1;
            else if (!strcmp(jname, "2"))     JP.gran = 2;
            else if (!strcmp(jname, "4"))     JP.gran = 4;
            else if (!strcmp(jname, "8"))     JP.gran = 8;
            else { fprintf(stderr, "bad --joint tier\n"); return 1; }
        } else if (!strncmp(argv[argi], "--profile=", 10)) {
            /* Upstream fitted cost profiles (93b5a7e).  NB both are
             * fits of the PRODUCTION kernels on that host; pivcoh has
             * no prefill pass and slightly faster kernels, so these
             * are transfer tests, not native fits. */
            static const float m1kap[9] = { 0, 0.63f, 0.46f, 0.52f,
                                            0.42f, 0.60f, 0.77f, 1.55f, 0.70f };
            static const float m4kap[9] = { 0, 0.64f, 0.49f, 0.63f,
                                            0.53f, 0.70f, 0.91f, 1.80f, 0.41f };
            if (!strcmp(argv[argi] + 10, "m1")) {
                memcpy(JP.kappa, m1kap, sizeof m1kap);
                JP.mu_cst = 0.898f; JP.prefill = 0.244f; JP.gamma = 163.0f;
            } else if (!strcmp(argv[argi] + 10, "m4")) {
                memcpy(JP.kappa, m4kap, sizeof m4kap);
                JP.mu_cst = 0.897f; JP.prefill = 0.235f; JP.gamma = 210.0f;
            } else { fprintf(stderr, "bad --profile\n"); return 1; }
        } else if (!strncmp(argv[argi], "--lambda=", 9)) {
            JP.lambda = (float)atof(argv[argi] + 9);
        } else if (!strncmp(argv[argi], "--gamma=", 8)) {
            JP.gamma = (float)atof(argv[argi] + 8);
        } else {
            fprintf(stderr, "usage: %s [--G=KB] [--blocks] [--tables] [--reps=N] [--joint=TIER]"
                            " [--lambda=F] [--gamma=F] file...\n", argv[0]);
            return 1;
        }
    }
    if (BLOCKS || TABLES) ngs = 1;
    pivco_huffman_set_fse_enabled(0);

    printf("codec_blk=%d fse=0 reps=%d joint=%s win=%s", (int)BLK, reps, jname,
           TABLES ? "tables" : BLOCKS ? "litblk" : "fixedG");
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
        char glab[8];
        if (TABLES)      snprintf(glab, sizeof glab, "tbl");
        else if (BLOCKS) snprintf(glab, sizeof glab, "blk");
        else             snprintf(glab, sizeof glab, "%zuK", G / 1024);
        double gm[4] = {0};     /* log-sums: enc m/p, dec m/p */
        size_t tj = 0, tp = 0, tn = 0, ta = 0, tw = 0;  /* joint ratio totals */
        int nfiles = 0;
        for (int ai = argi; ai < argc; ai++) {
            uint8_t *data = slurp(argv[ai], &N);
            if (!data) { fprintf(stderr, "skip %s\n", argv[ai]); continue; }
            D = data;
            CURPATH = argv[ai];
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

            printf("%-13s %5s | %6.0f %6.0f %5.2fx | %6.0f %6.0f %5.2fx |"
                   " %5.2f %5.2f | %5.2f %5.2f",
                   base, glab, em, ep, em / ep, dm, dp, dm / dp,
                   ebm, ebp, dbm, dbp);
            printf(" | %+.3fpp %zu/%zu",
                   ((double)j_bytes - (double)p_bytes) / (double)N * 100.0,
                   j_adopt, NW);
            printf("\n");
            gm[0] += log(em); gm[1] += log(ep);
            gm[2] += log(dm); gm[3] += log(dp);
            tj += j_bytes; tp += p_bytes; tn += N; ta += j_adopt; tw += NW;
            nfiles++;
            free(data);
            free(FR); free(LEN); free(LENP); free(WOFF); free(WOFFP);
            free(WSTART); free(ENC); free(ENC2); free(DEC);
        }
        if (nfiles > 1) {
            printf("%-13s %5s | %6.0f %6.0f %5.2fx | %6.0f %6.0f %5.2fx |"
                   " geomean over %d files",
                   "== geomean", glab,
                   exp(gm[0] / nfiles), exp(gm[1] / nfiles),
                   exp((gm[0] - gm[1]) / nfiles),
                   exp(gm[2] / nfiles), exp(gm[3] / nfiles),
                   exp((gm[2] - gm[3]) / nfiles), nfiles);
            printf(" | %+.3fpp %zu/%zu",
                   ((double)tj - (double)tp) / (double)tn * 100.0,
                   ta, tw);
            printf("\n");
        }
    }
    return 0;
}

/* bench_huf0_lits — pivcoh (SIMPLEST + BALANCED) vs huf0 (zstd's Huffman,
 * HUF_compress/HUF_decompress from ext/fse) on the zstd-HUF literal corpus,
 * windowed at zstd's real per-block boundaries (the .litblk sidecar).
 *
 * Per window (one fresh table each): time encode-e2e (histogram + table
 * build + encode kernels) and decode-e2e (decode-table build + decode),
 * and record compressed size for the ratio.  huf0 caps a single table at
 * HUF_BLOCKSIZE_MAX = 128 KiB, which every individual zstd block's
 * literals fit under — so this is the per-block cadence (--blocks), the
 * only one huf0 can express in one table.
 *
 * pivcoh size = encoded payload + the v4 code-lengths wire per window
 * (pivcoh_lens_wire_*, typically 30..60 B), with a raw-store fallback on
 * windows coding does not shrink — the same two moves huf0 makes (its
 * FSE-packed weights header and raw fallback).  The decode pass parses
 * the lens wire and rebuilds the table per window, so header costs are
 * in the timings for both engines.
 *
 * Usage: bench_huf0_lits [--reps=N] <file.lits>...   (needs sibling .litblk) */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include "huf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLK 16384               /* pivcoh sub-block (matches bench_pivcoh_lits) */
#define LWSTRIDE 130            /* per-window lens wire: [len | 0=raw][bytes] */

enum { M_HUF0 = 0, M_SIMPLE = 1, M_BAL = 2, NM = 3 };
static const char *MNAME[NM] = { "huf0", "pvSMPL", "pvBAL" };

static double now_sec(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static uint8_t *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)sz + 16);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f); *n = (size_t)sz; return b;
}

/* per-file state */
static const uint8_t *D;
static size_t   N, NW;
static size_t  *WSTART;                 /* window byte offsets (NW+1) */
static uint8_t *ENCS[NM];               /* per-method encoded stream */
static size_t  *EOFF[NM];               /* per-method per-window offsets (NW+1) */
static uint8_t *LENS[NM];               /* per-window 256 code_len (pivcoh only) */
static size_t   COMP[NM];               /* total compressed bytes (for ratio) */
static uint8_t *DEC;                    /* decode sink (N) */
static uint8_t *SCR;                    /* throwaway encode scratch */
static uint8_t  escr[PIVCOH_SCRATCH_SIZE(BLK)];
static uint8_t  dscr[PIVCOH_DECODE_SCRATCH_SIZE(BLK)];
static uint8_t  jscr[PIVCOH_JOINT_SCRATCH_SIZE];
static pivcoh_joint JP;                 /* BALANCED = coarse (gran -1) */
static size_t   SCRCAP;

static size_t wlen_of(size_t w) { return WSTART[w + 1] - WSTART[w]; }

/* Build a pivcoh table for window w into t; SIMPLE = plain lengths,
 * BAL = coarse joint shaping.  Returns 0 on failure. */
static int pv_build(pivcoh_table *t, const uint8_t *p, size_t wlen, int bal)
{
    uint64_t f[256] = {0};
    for (size_t i = 0; i < wlen; i++) f[p[i]]++;
    return bal ? pivcoh_table_from_freqs_joint(t, f, &JP, jscr)
               : pivcoh_table_from_freqs(t, f);
}
/* Encode window into dst (cap large); returns payload bytes or -1. */
static ptrdiff_t pv_encode(const pivcoh_table *t, const uint8_t *p, size_t wlen, uint8_t *dst)
{
    size_t off = 0;
    for (size_t b = 0; b < wlen; b += BLK) {
        size_t bn = wlen - b < BLK ? wlen - b : BLK;
        ptrdiff_t r = pivcoh_encode(t, p + b, bn, dst + off, PIVCOH_ENCODE_BOUND(bn), escr);
        if (r < 0) return -1;
        off += (size_t)r;
    }
    return (ptrdiff_t)off;
}
/* Decode window from src into out (wlen bytes); returns 0 on success. */
static int pv_decode(const pivcoh_table *t, const uint8_t *src, size_t slen,
                     uint8_t *out, size_t wlen)
{
    size_t off = 0, dof = 0;
    while (dof < wlen) {
        size_t consumed;
        ptrdiff_t dn = pivcoh_decode(t, src + off, slen - off, out + dof, wlen - dof, &consumed, dscr);
        if (dn < 0) return -1;
        off += consumed; dof += (size_t)dn;
    }
    return 0;
}

/* Untimed: encode every window with every method, store streams + sizes,
 * and gate each method's round-trip.  Returns 0 on success. */
static int setup_and_gate(const char *base)
{
    for (int m = 0; m < NM; m++) {
        ENCS[m] = malloc(2 * N + NW * 256 + 65536);
        EOFF[m] = malloc((NW + 1) * sizeof(size_t));
        LENS[m] = (m == M_HUF0) ? NULL : malloc(NW * LWSTRIDE);
        if (!ENCS[m] || !EOFF[m] || (m != M_HUF0 && !LENS[m])) return -1;
        COMP[m] = 0; EOFF[m][0] = 0;
    }
    size_t off[NM] = {0};
    for (size_t w = 0; w < NW; w++) {
        const uint8_t *p = D + WSTART[w];
        size_t wlen = wlen_of(w);
        if (wlen > HUF_BLOCKSIZE_MAX)
            return fprintf(stderr, "%s: window %zu = %zu > huf0 128KB cap\n", base, w, wlen);

        /* huf0: one table for the whole window (raw fallback if r==0) */
        size_t r = HUF_compress(ENCS[M_HUF0] + off[M_HUF0], HUF_compressBound(wlen), p, wlen);
        if (HUF_isError(r)) return fprintf(stderr, "%s: HUF_compress err w=%zu\n", base, w);
        if (r == 0) { memcpy(ENCS[M_HUF0] + off[M_HUF0], p, wlen); r = wlen; }
        off[M_HUF0] += r; EOFF[M_HUF0][w + 1] = off[M_HUF0]; COMP[M_HUF0] += r;

        /* pivcoh SIMPLEST + BALANCED: payload + lens wire, raw fallback */
        for (int m = M_SIMPLE; m <= M_BAL; m++) {
            pivcoh_table t;
            if (!pv_build(&t, p, wlen, m == M_BAL))
                return fprintf(stderr, "%s: %s build failed w=%zu\n", base, MNAME[m], w);
            uint8_t *lw = LENS[m] + w * LWSTRIDE;
            int lwn = pivcoh_lens_wire_write(lw + 1, t.code_len);
            lw[0] = (uint8_t)lwn;
            ptrdiff_t pl = pv_encode(&t, p, wlen, ENCS[m] + off[m]);
            if (pl < 0) return fprintf(stderr, "%s: %s encode failed w=%zu\n", base, MNAME[m], w);
            if ((size_t)pl + (size_t)lwn >= wlen) {   /* raw store */
                memcpy(ENCS[m] + off[m], p, wlen);
                pl = (ptrdiff_t)wlen;
                lw[0] = 0;
                COMP[m] += wlen;
            } else
                COMP[m] += (size_t)pl + (size_t)lwn;
            off[m] += (size_t)pl; EOFF[m][w + 1] = off[m];
        }
    }
    /* gate: every method round-trips to D */
    for (int m = 0; m < NM; m++) {
        memset(DEC, 0, N);
        for (size_t w = 0; w < NW; w++) {
            const uint8_t *src = ENCS[m] + EOFF[m][w];
            size_t slen = EOFF[m][w + 1] - EOFF[m][w], wlen = wlen_of(w);
            uint8_t *out = DEC + WSTART[w];
            if (m == M_HUF0) {
                size_t d = HUF_decompress(out, wlen, src, slen);
                if (HUF_isError(d) || d != wlen)
                    return fprintf(stderr, "%s: huf0 decode err w=%zu\n", base, w);
            } else {
                const uint8_t *lw = LENS[m] + w * LWSTRIDE;
                if (lw[0] == 0) {                      /* raw window */
                    if (slen != wlen) return fprintf(stderr, "%s: raw len w=%zu\n", base, w);
                    memcpy(out, src, wlen);
                } else {
                    uint8_t cl[256];
                    pivcoh_table t;
                    if (pivcoh_lens_wire_read(cl, lw + 1, lw[0]) != (int)lw[0] ||
                        !pivcoh_table_from_lens(&t, cl) ||
                        pv_decode(&t, src, slen, out, wlen) != 0)
                        return fprintf(stderr, "%s: %s decode err w=%zu\n", base, MNAME[m], w);
                }
            }
        }
        if (memcmp(DEC, D, N) != 0)
            return fprintf(stderr, "%s: %s round-trip mismatch\n", base, MNAME[m]);
    }
    return 0;
}

/* timed encode-e2e (rebuild + encode into throwaway SCR) */
static void enc_pass(int m)
{
    for (size_t w = 0; w < NW; w++) {
        const uint8_t *p = D + WSTART[w];
        size_t wlen = wlen_of(w);
        if (m == M_HUF0) {
            HUF_compress(SCR, SCRCAP, p, wlen);
        } else {
            pivcoh_table t;
            pv_build(&t, p, wlen, m == M_BAL);
            pivcoh_lens_wire_write(SCR + SCRCAP - 256, t.code_len);
            pv_encode(&t, p, wlen, SCR);
        }
    }
}
/* timed decode-e2e (rebuild decode table + decode from stored stream) */
static void dec_pass(int m)
{
    for (size_t w = 0; w < NW; w++) {
        const uint8_t *src = ENCS[m] + EOFF[m][w];
        size_t slen = EOFF[m][w + 1] - EOFF[m][w], wlen = wlen_of(w);
        uint8_t *out = DEC + WSTART[w];
        if (m == M_HUF0) {
            HUF_decompress(out, wlen, src, slen);
        } else {
            const uint8_t *lw = LENS[m] + w * LWSTRIDE;
            if (lw[0] == 0) {
                memcpy(out, src, wlen);
            } else {
                uint8_t cl[256];
                pivcoh_table t;
                pivcoh_lens_wire_read(cl, lw + 1, lw[0]);
                pivcoh_table_from_lens(&t, cl);
                pv_decode(&t, src, slen, out, wlen);
            }
        }
    }
}

static double timeit(void (*fn)(int), int m, int reps)
{
    int inner = (int)(1 + ((size_t)8 << 20) / (N ? N : 1));
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < inner; i++) fn(m);
        double dt = (now_sec() - t0) / inner;
        if (dt < best) best = dt;
    }
    return best;
}

static int build_windows(const char *base, const char *path)
{
    size_t plen = strlen(path);
    if (plen < 5 || strcmp(path + plen - 5, ".lits") != 0)
        return fprintf(stderr, "%s: need a .lits input\n", base);
    char bp[4096];
    snprintf(bp, sizeof bp, "%.*s.litblk", (int)(plen - 5), path);
    size_t bn; uint8_t *b = slurp(bp, &bn);
    if (!b || bn % 4) { free(b); return fprintf(stderr, "%s: bad %s\n", base, bp); }
    NW = bn / 4;
    WSTART = malloc((NW + 1) * sizeof(size_t));
    size_t off = 0;
    for (size_t w = 0; w < NW; w++) {
        uint32_t e = (uint32_t)b[4*w] | (uint32_t)b[4*w+1] << 8 |
                     (uint32_t)b[4*w+2] << 16 | (uint32_t)b[4*w+3] << 24;
        WSTART[w] = off; off += e;
    }
    WSTART[NW] = off; free(b);
    if (off != N) return fprintf(stderr, "%s: litblk sum %zu != %zu\n", base, off, N);
    return 0;
}

int main(int argc, char **argv)
{
    int reps = 12, argi = 1;
    for (; argi < argc && !strncmp(argv[argi], "--", 2); argi++) {
        if (!strncmp(argv[argi], "--reps=", 7)) reps = atoi(argv[argi] + 7);
        else { fprintf(stderr, "usage: %s [--reps=N] file.lits...\n", argv[0]); return 1; }
    }
    pivcoh_joint jp = PIVCOH_JOINT_DEFAULTS; JP = jp; JP.gran = -1;   /* coarse = BALANCED */
    SCRCAP = 2 * (size_t)HUF_BLOCKSIZE_MAX + 65536;   /* fits huf0 + pivcoh worst-case window */
    SCR = malloc(SCRCAP);

    printf("per-block windows, fse=0, reps=%d  (enc/dec MB/s, ratio=orig/comp)\n", reps);
    printf("%-12s %6s | %-20s | %-20s | %-18s\n", "", "wins",
           "huf0  enc  dec ratio", "pvSMPL enc  dec ratio", "pvBAL enc dec ratio");

    double gm[NM][3] = {{0}};        /* log-sums: enc, dec, ratio */
    int nf = 0;
    for (int ai = argi; ai < argc; ai++) {
        uint8_t *data = slurp(argv[ai], &N);
        if (!data) { fprintf(stderr, "skip %s\n", argv[ai]); continue; }
        D = data;
        const char *base = strrchr(argv[ai], '/'); base = base ? base + 1 : argv[ai];
        if (build_windows(base, argv[ai]) != 0) return 1;
        DEC = malloc(N + 64);
        if (setup_and_gate(base) != 0) return 1;

        /* DVFS warmup */
        double w0 = now_sec();
        do { enc_pass(M_HUF0); enc_pass(M_SIMPLE); } while (now_sec() - w0 < 0.1);

        double en[NM], de[NM], ra[NM];
        for (int m = 0; m < NM; m++) {
            en[m] = (double)N / timeit(enc_pass, m, reps) / 1e6;
            de[m] = (double)N / timeit(dec_pass, m, reps) / 1e6;
            ra[m] = (double)N / (double)COMP[m];
            gm[m][0] += log(en[m]); gm[m][1] += log(de[m]); gm[m][2] += log(ra[m]);
        }
        printf("%-12s %6zu | %5.0f %5.0f %5.2f | %5.0f %5.0f %5.2f | %5.0f %5.0f %5.2f\n",
               base, NW, en[0], de[0], ra[0], en[1], de[1], ra[1], en[2], de[2], ra[2]);
        nf++;
        for (int m = 0; m < NM; m++) { free(ENCS[m]); free(EOFF[m]); free(LENS[m]); }
        free(WSTART); free(DEC); free(data);
    }
    if (nf > 1)
        printf("%-12s %6s | %5.0f %5.0f %5.2f | %5.0f %5.0f %5.2f | %5.0f %5.0f %5.2f\n",
               "== geomean", "",
               exp(gm[0][0]/nf), exp(gm[0][1]/nf), exp(gm[0][2]/nf),
               exp(gm[1][0]/nf), exp(gm[1][1]/nf), exp(gm[1][2]/nf),
               exp(gm[2][0]/nf), exp(gm[2][1]/nf), exp(gm[2][2]/nf));
    free(SCR);
    return 0;
}

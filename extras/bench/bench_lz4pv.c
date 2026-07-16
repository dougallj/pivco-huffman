/* bench_lz4pv — end-to-end "lz4pv" codec: LZ4HC parse split into four
 * streams (extras/lz4hc_split), literals + tokens + offset-high-bytes
 * entropy-coded with pivcoh v4, offset-low-bytes + overflow raw, and
 * the plane-aware split decoder (lz4_split_decompress_planes) for
 * execution.  Reference columns: plain LZ4HC wire (same parse) through
 * LZ4_decompress_safe.
 *
 * Wire per stream: [u8 mode] mode 0 = raw bytes; mode 1 = [pivcoh lens
 * wire][pivcoh blocks of <= 32767 syms].  Container: u32 sizes.  Build
 * with -DPIVCOH_LENS_WIRE_NO_MODE2 (decode-speed-priority lens wire).
 *
 * Env:  PIVCOH_EFFORT = simple | balanced | faster | fastest  (table
 *       shaping tier; default balanced)
 *       LZ4_LEVEL = 1..12 (default 9, matching the May analysis)
 *
 * Usage: bench_lz4pv [--reps=N] file...      (default reps 7) */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include "../lz4_split.h"
#include "lz4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PVB 32767
#define PAD 64

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
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)sz + PAD);
    if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(p); return NULL; }
    fclose(f); *n = (size_t)sz; return p;
}

static int EFFORT = -1;                 /* gran; -2 = SIMPLEST (no joint) */
static const char *EFFNAME = "balanced";
static uint8_t escratch[PIVCOH_SCRATCH_SIZE(PVB)];
static uint8_t jscratch[PIVCOH_JOINT_SCRATCH_SIZE];
static uint8_t dscratch[PIVCOH_DECODE_SCRATCH_SIZE(PVB)];

/* [u8 mode][payload]; returns wire bytes written. */
static size_t pv_stream_encode(uint8_t *dst, size_t dcap, const uint8_t *src, size_t n)
{
    if (n == 0) { dst[0] = 0; return 1; }
    uint64_t freq[256];
    pivcoh_histogram(freq, src, n);
    pivcoh_table t;
    int ok;
    if (EFFORT == -2) ok = pivcoh_table_from_freqs(&t, freq);
    else {
        pivcoh_joint jp = PIVCOH_JOINT_DEFAULTS;
        jp.gran = EFFORT;
        ok = pivcoh_table_from_freqs_joint(&t, freq, &jp, jscratch);
    }
    size_t cur = 1;
    int bad = !ok;
    if (!bad) {
        int lwn = pivcoh_lens_wire_write(dst + cur, t.code_len);
        cur += (size_t)lwn;
        for (size_t b = 0; !bad && b < n; b += PVB) {
            size_t bn = n - b < PVB ? n - b : PVB;
            if (cur + PIVCOH_ENCODE_BOUND(bn) > dcap) { bad = 1; break; }
            ptrdiff_t el = pivcoh_encode(&t, src + b, bn, dst + cur,
                                         PIVCOH_ENCODE_BOUND(bn), escratch);
            if (el < 0) { bad = 1; break; }
            cur += (size_t)el;
        }
    }
    if (bad || cur >= n + 1) {          /* raw store */
        dst[0] = 0;
        memcpy(dst + 1, src, n);
        return 1 + n;
    }
    dst[0] = 1;
    return cur;
}

/* Decode a stream wire into dst (capacity n + PAD); returns 0/-1. */
static int pv_stream_decode(uint8_t *dst, size_t n, const uint8_t *src, size_t cn)
{
    if (n == 0) return cn == 1 ? 0 : -1;
    if (src[0] == 0) {
        if (cn != 1 + n) return -1;
        memcpy(dst, src + 1, n);
        return 0;
    }
    if (src[0] != 1) return -1;
    uint8_t lens[256];
    const uint8_t *p = src + 1;
    size_t rem = cn - 1;
    int lb = pivcoh_lens_wire_read(lens, p, rem);
    pivcoh_table t;
    if (lb < 0 || !pivcoh_table_from_lens(&t, lens)) return -1;
    p += lb; rem -= (size_t)lb;
    size_t dof = 0;
    while (dof < n) {
        size_t consumed;
        ptrdiff_t dn = pivcoh_decode(&t, p, rem, dst + dof, n - dof, &consumed, dscratch);
        if (dn <= 0) return -1;
        p += consumed; rem -= consumed; dof += (size_t)dn;
    }
    return rem == 0 ? 0 : -1;
}

typedef struct {                        /* one file's staged codec state */
    lz4_split_ctx_t sp;                 /* split streams (encoder output) */
    uint8_t *lz4wire;                   /* standard LZ4 wire (reference) */
    int lz4len;
    uint8_t *olo, *ohi;                 /* deinterleaved offset planes */
    size_t n_off;
    uint8_t *wire;                      /* lz4pv container */
    size_t wlen;
    uint8_t *dlit, *dtok, *dlo, *dhi, *dovf;   /* decode-side buffers */
    uint8_t *out;
} state_t;

static size_t wr32(uint8_t *p, size_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24); return 4; }
static size_t rd32(const uint8_t *p) { return (size_t)p[0] | (size_t)p[1]<<8 | (size_t)p[2]<<16 | (size_t)p[3]<<24; }

/* full encode: parse + entropy.  Returns container size. */
static size_t enc_pass(state_t *st, const uint8_t *src, size_t n, int level)
{
    lz4_split_ctx_t *sp = &st->sp;
    sp->lit_pos = sp->tok_pos = sp->off_pos = sp->ovf_pos = 0;
    st->lz4len = phsplit_LZ4_compress_HC_split((const char *)src, (int)n,
                                               st->lz4wire, LZ4_compressBound((int)n),
                                               level, sp);
    if (!sp->ok || st->lz4len <= 0) return 0;
    st->n_off = sp->off_pos / 2;
    for (size_t i = 0; i < st->n_off; i++) {         /* deinterleave */
        st->olo[i] = sp->offsets[2 * i];
        st->ohi[i] = sp->offsets[2 * i + 1];
    }
    uint8_t *w = st->wire;
    size_t h = 4 + 4 + 5 * 8;                         /* raw, nseq, 5 pairs */
    size_t cur = h;
    size_t lens[5] = { sp->lit_pos, sp->tok_pos, st->n_off, st->n_off, sp->ovf_pos };
    const uint8_t *srcs[5] = { sp->literals, sp->tokens, st->olo, st->ohi, sp->overflow };
    const int coded[5] = { 1, 1, 0, 1, 0 };           /* off_lo + ovf raw */
    size_t hoff = 8;
    for (int s = 0; s < 5; s++) {
        size_t cn;
        if (coded[s]) cn = pv_stream_encode(w + cur, (size_t)-1, srcs[s], lens[s]);
        else { w[cur] = 0; memcpy(w + cur + 1, srcs[s], lens[s]); cn = 1 + lens[s]; }
        wr32(w + hoff, lens[s]); wr32(w + hoff + 4, cn);
        hoff += 8; cur += cn;
    }
    wr32(w, n); wr32(w + 4, st->n_off);
    st->wlen = cur;
    return cur;
}

/* full decode: entropy + execution into st->out.  Returns 0/-1. */
static int dec_pass(state_t *st, size_t n)
{
    const uint8_t *w = st->wire;
    size_t raw = rd32(w), n_off = rd32(w + 4);
    if (raw != n) return -1;
    uint8_t *bufs[5] = { st->dlit, st->dtok, st->dlo, st->dhi, st->dovf };
    size_t lens[5];
    size_t hoff = 8, cur = 8 + 5 * 8;
    for (int s = 0; s < 5; s++) {
        lens[s] = rd32(w + hoff);
        size_t cn = rd32(w + hoff + 4);
        if (pv_stream_decode(bufs[s], lens[s], w + cur, cn) != 0) return -1;
        hoff += 8; cur += cn;
    }
    if (lens[2] != n_off || lens[3] != n_off) return -1;
    return lz4_split_decompress_planes(st->dlit, lens[0], st->dtok, lens[1],
                                       st->dlo, st->dhi, n_off,
                                       st->dovf, lens[4], st->out, n);
}

int main(int argc, char **argv)
{
    int reps = 7, argi = 1, level = 9;
    const char *ev = getenv("PIVCOH_EFFORT");
    const char *lv = getenv("LZ4_LEVEL");
    if (lv) level = atoi(lv);
    if (ev) {
        if (!strcasecmp(ev, "simple") || !strcmp(ev, "0"))        { EFFORT = -2; EFFNAME = "simple"; }
        else if (!strcasecmp(ev, "balanced") || !strcmp(ev, "1")) { EFFORT = -1; EFFNAME = "balanced"; }
        else if (!strcasecmp(ev, "faster") || !strcmp(ev, "2"))   { EFFORT = 0;  EFFNAME = "faster"; }
        else if (!strcasecmp(ev, "fastest") || !strcmp(ev, "3"))  { EFFORT = 1;  EFFNAME = "fastest"; }
        else { fprintf(stderr, "bad PIVCOH_EFFORT\n"); return 1; }
    }
    for (; argi < argc && !strncmp(argv[argi], "--", 2); argi++)
        if (!strncmp(argv[argi], "--reps=", 7)) reps = atoi(argv[argi] + 7);
    printf("lz4pv: LZ4HC-%d parse + pivcoh(%s) lits/tokens/off_hi · reps=%d\n",
           level, EFFNAME, reps);
    printf("%-14s %9s | %7s %7s | %8s %8s | %8s %8s | %7s\n",
           "file", "bytes", "lz4rat", "pvrat", "enc MB/s", "  (lz4)", "dec MB/s", "  (lz4)", "litMB/s");
    double g[6] = {0}; int nf = 0;
    for (; argi < argc; argi++) {
        size_t n;
        uint8_t *src = slurp(argv[argi], &n);
        if (!src) { fprintf(stderr, "cannot read %s\n", argv[argi]); continue; }
        state_t st;
        memset(&st, 0, sizeof st);
        st.sp.literals = malloc(n + PAD); st.sp.lit_cap = n;
        st.sp.tokens   = malloc(n / 4 + 1024); st.sp.tok_cap = n / 4 + 1024;
        st.sp.offsets  = malloc(n + PAD); st.sp.off_cap = n;
        st.sp.overflow = malloc(n / 2 + 1024); st.sp.ovf_cap = n / 2 + 1024;
        st.lz4wire = malloc((size_t)LZ4_compressBound((int)n) + PAD);
        st.olo = malloc(n / 2 + PAD); st.ohi = malloc(n / 2 + PAD);
        st.wire = malloc(3 * n + 65536);
        st.dlit = malloc(n + PAD); st.dtok = malloc(n / 4 + 1024 + PAD);
        st.dlo = malloc(n / 2 + PAD); st.dhi = malloc(n / 2 + PAD);
        st.dovf = malloc(n / 2 + 1024 + PAD);
        st.out = malloc(n + PAD);

        /* correctness first */
        size_t comp = enc_pass(&st, src, n, level);
        if (!comp) { fprintf(stderr, "%s: encode failed\n", argv[argi]); return 1; }
        memset(st.out, 0xAA, n);
        if (dec_pass(&st, n) != 0 || memcmp(st.out, src, n) != 0) {
            fprintf(stderr, "%s: ROUNDTRIP MISMATCH\n", argv[argi]);
            return 1;
        }
        memset(st.out, 0, n);
        if (LZ4_decompress_safe((const char *)st.lz4wire, (char *)st.out,
                                st.lz4len, (int)n) != (int)n
            || memcmp(st.out, src, n) != 0) {
            fprintf(stderr, "%s: lz4 reference mismatch\n", argv[argi]);
            return 1;
        }

        /* >= 100 ms hot loop before any timing — DVFS ramp (house
         * discipline from bench_pivcoh_lits; without it small-file
         * decode ratios are unreliable). */
        {   double w0 = now_sec();
            do {
                if (dec_pass(&st, n) != 0) return 1;
                LZ4_decompress_safe((const char *)st.lz4wire, (char *)st.out,
                                    st.lz4len, (int)n);
            } while (now_sec() - w0 < 0.1);
        }
        double te = 1e30, td = 1e30, tld = 1e30, tle = 1e30, tlit = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_sec();
            enc_pass(&st, src, n, level);
            double t1 = now_sec();
            if (t1 - t0 < te) te = t1 - t0;

            t0 = now_sec();
            if (dec_pass(&st, n) != 0) return 1;
            t1 = now_sec();
            if (t1 - t0 < td) td = t1 - t0;

            t0 = now_sec();
            LZ4_decompress_safe((const char *)st.lz4wire, (char *)st.out, st.lz4len, (int)n);
            t1 = now_sec();
            if (t1 - t0 < tld) tld = t1 - t0;

            t0 = now_sec();                        /* lits entropy alone */
            {   size_t cn = rd32(st.wire + 8 + 4);
                pv_stream_decode(st.dlit, st.sp.lit_pos, st.wire + 8 + 40, cn);
            }
            t1 = now_sec();
            if (t1 - t0 < tlit) tlit = t1 - t0;
        }
        tle = te;                                  /* enc incl. parse */
        double rat_lz4 = (double)n / (double)st.lz4len;
        double rat_pv  = (double)n / (double)st.wlen;
        const char *base = strrchr(argv[argi], '/');
        base = base ? base + 1 : argv[argi];
        printf("%-14s %9zu | %7.3f %7.3f | %8.0f %8s | %8.0f %8.0f | %7.0f\n",
               base, n, rat_lz4, rat_pv,
               n / te / 1e6, "-", n / td / 1e6, n / tld / 1e6,
               st.sp.lit_pos / tlit / 1e6);
        g[0] += log(rat_lz4); g[1] += log(rat_pv);
        g[2] += log(n / te / 1e6); g[3] += log(n / td / 1e6);
        g[4] += log(n / tld / 1e6); g[5] += log(st.sp.lit_pos / tlit / 1e6);
        nf++;
        (void)tle;
        free(src); free(st.sp.literals); free(st.sp.tokens); free(st.sp.offsets);
        free(st.sp.overflow); free(st.lz4wire); free(st.olo); free(st.ohi);
        free(st.wire); free(st.dlit); free(st.dtok); free(st.dlo); free(st.dhi);
        free(st.dovf); free(st.out);
    }
    if (nf > 1)
        printf("%-14s %9s | %7.3f %7.3f | %8.0f %8s | %8.0f %8.0f | %7.0f  geomean/%d\n",
               "==", "", exp(g[0]/nf), exp(g[1]/nf), exp(g[2]/nf), "-",
               exp(g[3]/nf), exp(g[4]/nf), exp(g[5]/nf), nf);
    return 0;
}

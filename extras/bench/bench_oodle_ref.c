/* Oodle 2.9.3 reference bench: Selkie / Mermaid / Kraken at
 * HyperFast1 / VeryFast / Normal / Optimal2, silesia per-file +
 * geomean, house methodology (>=100ms DVFS warmup, best-of-reps,
 * preallocated decoder memory so decode is malloc-free).
 *
 * Needs the (non-redistributable) Oodle SDK; not in the cmake build:
 *   cc -O2 -std=c11 -I <sdk>/include bench_oodle_ref.c \
 *      <sdk>/liboo2coremac64.a -lm -lc++ -o bench_oodle_ref
 * Results: results/oodle_ref-m4mini-20260717.{md,txt}.
 * NB stdout is fully buffered when piped (~11KB total output can sit
 * unflushed to the end); Optimal2 encodes at 5-7 MB/s, so the full
 * sweep is ~10 min on silesia. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "oodle2.h"

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint8_t *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)sz);
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(p); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return p;
}

int main(int argc, char **argv)
{
    int enc_reps = 3, dec_reps = 8;
    OodleLZ_Compressor comps[3] = { OodleLZ_Compressor_Selkie,
                                    OodleLZ_Compressor_Mermaid,
                                    OodleLZ_Compressor_Kraken };
    const char *cnames[3] = { "Selkie", "Mermaid", "Kraken" };
    OodleLZ_CompressionLevel lvls[4] = { OodleLZ_CompressionLevel_HyperFast1,
                                         OodleLZ_CompressionLevel_VeryFast,
                                         OodleLZ_CompressionLevel_Normal,
                                         OodleLZ_CompressionLevel_Optimal2 };
    const char *lnames[4] = { "HyperFast1", "VeryFast", "Normal", "Optimal2" };

    for (int ci = 0; ci < 3; ci++)
        for (int li = 0; li < 4; li++) {
            printf("--- %s %s ---\n", cnames[ci], lnames[li]);
            printf("%-14s %9s | %6s | %8s | %8s |\n",
                   "file", "bytes", "ratio", "enc MB/s", "dec MB/s");
            double g[3] = {0};
            int nf = 0;
            for (int a = 1; a < argc; a++) {
                size_t n;
                uint8_t *src = slurp(argv[a], &n);
                if (!src) { fprintf(stderr, "cannot read %s\n", argv[a]); continue; }
                OO_SINTa cap = OodleLZ_GetCompressedBufferSizeNeeded(comps[ci], (OO_SINTa)n);
                uint8_t *comp = malloc((size_t)cap);
                uint8_t *out = malloc(n);
                OO_S32 dmsz = OodleLZDecoder_MemorySizeNeeded(comps[ci], -1);
                void *dmem = malloc((size_t)dmsz);

                OO_SINTa clen = OodleLZ_Compress(comps[ci], src, (OO_SINTa)n, comp,
                                                 lvls[li], NULL, NULL, NULL, NULL, 0);
                if (clen <= 0) { fprintf(stderr, "%s: compress failed\n", argv[a]); return 1; }
                memset(out, 0xAA, n);
                OO_SINTa dn = OodleLZ_Decompress(comp, clen, out, (OO_SINTa)n,
                                                 OodleLZ_FuzzSafe_Yes, OodleLZ_CheckCRC_No,
                                                 OodleLZ_Verbosity_None, NULL, 0, NULL, NULL,
                                                 dmem, dmsz, OodleLZ_Decode_Unthreaded);
                if (dn != (OO_SINTa)n || memcmp(out, src, n) != 0) {
                    fprintf(stderr, "%s: ROUNDTRIP MISMATCH\n", argv[a]);
                    return 1;
                }

                {   double w0 = now_sec();
                    do {
                        OodleLZ_Decompress(comp, clen, out, (OO_SINTa)n,
                                           OodleLZ_FuzzSafe_Yes, OodleLZ_CheckCRC_No,
                                           OodleLZ_Verbosity_None, NULL, 0, NULL, NULL,
                                           dmem, dmsz, OodleLZ_Decode_Unthreaded);
                    } while (now_sec() - w0 < 0.1);
                }
                double te = 1e30, td = 1e30;
                for (int r = 0; r < enc_reps; r++) {
                    double t0 = now_sec();
                    OodleLZ_Compress(comps[ci], src, (OO_SINTa)n, comp,
                                     lvls[li], NULL, NULL, NULL, NULL, 0);
                    double t1 = now_sec();
                    if (t1 - t0 < te) te = t1 - t0;
                }
                for (int r = 0; r < dec_reps; r++) {
                    double t0 = now_sec();
                    OodleLZ_Decompress(comp, clen, out, (OO_SINTa)n,
                                       OodleLZ_FuzzSafe_Yes, OodleLZ_CheckCRC_No,
                                       OodleLZ_Verbosity_None, NULL, 0, NULL, NULL,
                                       dmem, dmsz, OodleLZ_Decode_Unthreaded);
                    double t1 = now_sec();
                    if (t1 - t0 < td) td = t1 - t0;
                }
                double rat = (double)n / (double)clen;
                const char *base = strrchr(argv[a], '/');
                base = base ? base + 1 : argv[a];
                printf("%-14s %9zu | %6.3f | %8.0f | %8.0f |\n",
                       base, n, rat, n / te / 1e6, n / td / 1e6);
                g[0] += log(rat); g[1] += log(n / te / 1e6); g[2] += log(n / td / 1e6);
                nf++;
                free(src); free(comp); free(out); free(dmem);
            }
            if (nf > 1)
                printf("%-14s %9s | %6.3f | %8.0f | %8.0f |  geomean/%d\n",
                       "==", "", exp(g[0] / nf), exp(g[1] / nf), exp(g[2] / nf), nf);
        }
    return 0;
}

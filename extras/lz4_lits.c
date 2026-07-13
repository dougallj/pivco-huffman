/* lz4_lits: LZ-preprocess a file and dump the literal stream.
 * Compresses with LZ4HC (level 9), then walks the LZ4 block format
 * (token / literals / offset / matchlen) collecting the literal runs —
 * i.e. exactly the bytes an LZ-front-ended entropy coder would see.
 *
 * Typical use (feeds extras/bench/bench_table_lifetime.c):
 *   curl -O http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip && unzip silesia.zip
 *   for f in dickens mozilla ...; do pivco_lz4_lits $f $f.lits; done
 * Needs liblz4 (homebrew); target is gated on it in extras/bench. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <lz4hc.h>

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: %s IN OUT.lits\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *in = malloc((size_t)n);
    if (fread(in, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);

    int cap = LZ4_compressBound((int)n);
    uint8_t *comp = malloc((size_t)cap);
    int clen = LZ4_compress_HC((const char *)in, (char *)comp, (int)n, cap, 9);
    if (clen <= 0) { fprintf(stderr, "compress failed\n"); return 1; }

    uint8_t *lits = malloc((size_t)n);
    size_t lout = 0;
    const uint8_t *ip = comp, *end = comp + clen;
    while (ip < end) {
        unsigned token = *ip++;
        size_t ll = token >> 4;
        if (ll == 15) { unsigned b; do { b = *ip++; ll += b; } while (b == 255); }
        memcpy(lits + lout, ip, ll);
        lout += ll;
        ip += ll;
        if (ip >= end) break;              /* final literal run */
        ip += 2;                           /* match offset */
        size_t ml = token & 15;
        if (ml == 15) { unsigned b; do { b = *ip++; ml += b; } while (b == 255); }
    }

    uint64_t cnt[256] = {0};
    for (size_t i = 0; i < lout; i++) cnt[lits[i]]++;
    double H = 0.0;
    for (int s = 0; s < 256; s++)
        if (cnt[s]) { double p = (double)cnt[s] / (double)lout; H -= p * log2(p); }

    FILE *o = fopen(argv[2], "wb");
    fwrite(lits, 1, lout, o);
    fclose(o);
    printf("%-10s %9ld -> lz4hc %9d B; literals %9zu (%.1f%%), H=%.2f bits\n",
           argv[1], n, clen, lout, 100.0 * (double)lout / (double)n, H);
    return 0;
}

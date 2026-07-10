/* fuzz_decode: corrupt valid pivco streams (byte flips, truncations,
 * garbage) and check the decoder returns OK/CORRUPT without crashing or
 * writing outside symbols[0..N).  Canary bands around the output catch
 * overwrites without ASan. */
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N     16384
#define GUARD 4096

static unsigned rng = 0x1234567;
static unsigned rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

int main(int argc, char **argv)
{
    long iters = argc > 1 ? atol(argv[1]) : 200000;
    static uint8_t in[N];
    static uint8_t enc[3 * N];
    static uint8_t outbuf[N + 2 * GUARD];
    uint8_t *out = outbuf + GUARD;

    /* skewed-ish data with a fat alphabet so trees are deep + flat-rich */
    for (int i = 0; i < N; i++) {
        unsigned r = rnd() % 1000;
        in[i] = (uint8_t)(r < 400 ? r % 3 : (r < 800 ? 3 + r % 24 : 27 + r % 200));
    }
    uint64_t freq[256] = {0};
    for (int i = 0; i < N; i++) freq[in[i]]++;
    pivco_huffman_table_t *t = calloc(1, sizeof(*t));
    if (pivco_huffman_build_table(freq, t) != PIVCO_OK) return 1;
    size_t elen = 0;
    if (pivco_huffman_encode(in, N, t, enc, &elen) != PIVCO_OK) return 1;

    pivco_huffman_decode_table_t dt;
    if (pivco_huffman_build_decode_table(t->code_len, &dt) != PIVCO_OK) return 1;

    static uint8_t buf[3 * N];
    long ok = 0, corrupt = 0;
    for (long it = 0; it < iters; it++) {
        size_t len = elen;
        memcpy(buf, enc, elen);
        switch (it % 4) {
        case 0: /* 1-4 byte flips */
            for (int k = 0; k < 1 + (int)(rnd() % 4); k++)
                buf[rnd() % elen] ^= (uint8_t)(1 + rnd() % 255);
            break;
        case 1: /* truncation */
            len = rnd() % elen;
            break;
        case 2: /* flip + truncate */
            buf[rnd() % elen] ^= (uint8_t)(1 + rnd() % 255);
            len = 1 + rnd() % elen;
            break;
        case 3: /* pure garbage, random length */
            len = 1 + rnd() % elen;
            for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)rnd();
            /* keep a plausible N header half the time */
            if (rnd() & 1) { buf[0] = enc[0]; buf[1] = enc[1]; }
            break;
        }
        memset(outbuf, 0xCD, sizeof(outbuf));
        size_t cons = 0;
        int rc = pivco_huffman_decode_dt(buf, len, &dt, out, &cons);
        if (rc == PIVCO_OK) ok++; else corrupt++;
        for (int g = 0; g < GUARD; g++)
            if (outbuf[g] != 0xCD || outbuf[GUARD + N + g] != 0xCD) {
                fprintf(stderr, "GUARD SMASH iter %ld rc=%d\n", it, rc);
                return 2;
            }
        if (rc == PIVCO_OK && cons > len) {
            fprintf(stderr, "consumed %zu > len %zu iter %ld\n", cons, len, it);
            return 3;
        }
    }
    printf("fuzz: %ld iters, %ld decoded, %ld rejected, guards intact\n",
           iters, ok, corrupt);
    return 0;
}

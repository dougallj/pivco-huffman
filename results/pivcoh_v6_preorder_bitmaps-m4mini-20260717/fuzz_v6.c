/* fuzz_v6 — hostile-stream fuzz for pivcoh.h wire v6.  Random tables x
 * random blocks; each block's stream is attacked with multi-byte
 * garbles, truncations and pure-noise buffers, decoded with EXACT
 * malloc'd out/scratch buffers so ASan redzones enforce the
 * PIVCOH_DECODE_SCRATCH_SIZE(n) = 2n + 128 contract byte-for-byte.
 * Build with -fsanitize=address,undefined; a clean exit is the pass.
 *
 * Usage: fuzz_v6 [n_decodes]   (default 200000)
 */
#define PIVCOH_IMPLEMENTATION
#include "pivcoh.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

#define MAXN 8192

int main(int argc, char **argv)
{
    long target = argc > 1 ? atol(argv[1]) : 200000;
    long done = 0, valid = 0;
    uint8_t *blk = malloc(MAXN);
    uint8_t *enc = malloc(PIVCOH_ENCODE_BOUND(MAXN));
    uint8_t *mut = malloc(PIVCOH_ENCODE_BOUND(MAXN));
    uint8_t *escratch = malloc(PIVCOH_SCRATCH_SIZE(MAXN));
    pivcoh_table t;

    while (done < target) {
        /* random table: alphabet size + skew profile */
        int nsym = 2 + (int)(rnd() % 255);
        uint64_t freq[256] = {0};
        int shift = (int)(rnd() % 6);           /* 0 flat .. 5 very skewed */
        for (int s = 0; s < nsym; s++)
            freq[s] = 1 + (rnd() % 1000 >> (uint32_t)((s * shift) / 40 > 30
                                                      ? 30 : (s * shift) / 40));
        if (!pivcoh_table_from_freqs(&t, freq)) continue;

        for (int b = 0; b < 4 && done < target; b++) {
            int N = 1 + (int)(rnd() % MAXN);
            for (int i = 0; i < N; i++)
                blk[i] = (uint8_t)(rnd() % (uint32_t)nsym);
            ptrdiff_t el = pivcoh_encode(&t, blk, (size_t)N, enc,
                                         PIVCOH_ENCODE_BOUND(MAXN), escratch);
            if (el < 0) return fprintf(stderr, "encode failed\n"), 1;

            /* exact-size decode buffers: fresh per block so redzones sit
             * flush against the contract */
            uint8_t *out = malloc((size_t)N);
            uint8_t *dsc = malloc(PIVCOH_DECODE_SCRATCH_SIZE(N));

            /* clean roundtrip must hold */
            if (pivcoh_decode(&t, enc, (size_t)el, out, (size_t)N,
                              NULL, dsc) != N || memcmp(out, blk, (size_t)N))
                return fprintf(stderr, "roundtrip failed N=%d\n", N), 1;
            done++; valid++;

            for (int a = 0; a < 25 && done < target; a++) {
                size_t len = (size_t)el;
                memcpy(mut, enc, len);
                switch (rnd() % 4) {
                case 0:                       /* garble 1..8 bytes */
                    for (int g = 1 + (int)(rnd() % 8); g > 0; g--)
                        mut[rnd() % len] = (uint8_t)rnd();
                    break;
                case 1:                       /* truncate */
                    len = rnd() % len;
                    break;
                case 2:                       /* garble + truncate */
                    for (int g = 1 + (int)(rnd() % 4); g > 0; g--)
                        mut[rnd() % len] = (uint8_t)rnd();
                    len = 1 + rnd() % len;
                    break;
                default:                      /* pure noise, plausible N */
                    len = 2 + rnd() % 512;
                    for (size_t i = 0; i < len; i++)
                        mut[i] = (uint8_t)rnd();
                    if (rnd() & 1) {          /* half the time: valid header */
                        mut[0] = (uint8_t)N; mut[1] = (uint8_t)(N >> 8);
                    }
                    break;
                }
                (void)pivcoh_decode(&t, mut, len, out, (size_t)N, NULL, dsc);
                done++;
            }
            free(out);
            free(dsc);
        }
    }
    printf("fuzz_v6 PASS: %ld hostile decodes (%ld clean roundtrips) survived\n",
           done - valid, valid);
    return 0;
}

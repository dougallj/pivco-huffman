/* bench_pivcoh_check: verifies extras/pivcoh.h (the stb-style single-header
 * codec) against the production library:
 *   - table_from_freqs code_len == pivco_huffman_build_table code_len
 *   - mini encode wire == pivco_huffman_encode wire, byte for byte
 *   - cross-decode both ways + mini roundtrip (malloc and caller-scratch)
 *   - invalid lengths rejected; hostile-stream fuzz (mutations/truncations)
 *     never crashes (run under ASan for the real assurance) */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = 1234;
static uint64_t rng(void)
{
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static pivco_huffman_table_t ref_table;
static pivcoh_table mini, mini2;
static uint8_t blk[8192], enc_ref[65536], enc_mini[65536], dec_buf[8192];
static uint8_t scratch[PIVCOH_SCRATCH_SIZE(8192)];         /* encode */
static uint8_t dscratch[PIVCOH_DECODE_SCRATCH_SIZE(8192)]; /* decode: the tight
                                                              bound, so ASan
                                                              enforces it */
static int n_tables, n_blocks, n_fuzz;

#define FAIL(...) do { fprintf(stderr, "FAIL %s/%d: ", tag, id); \
                       fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

static void one_case(const uint64_t freq[256], const char *tag, int id)
{
    int rc_ref = pivco_huffman_build_table(freq, &ref_table);
    int rc_min = pivcoh_table_from_freqs(&mini, freq);
    if ((rc_ref == PIVCO_OK) != rc_min) FAIL("build rc %d vs %d", rc_ref, rc_min);
    if (rc_min == 0) return;
    if (memcmp(ref_table.code_len, mini.code_len, 256)) FAIL("code_len");
    if (!pivcoh_table_from_lens(&mini2, ref_table.code_len)) FAIL("from_lens rc");
    if (memcmp(&mini, &mini2, sizeof(mini))) FAIL("from_freqs vs from_lens table");
    if (mini.num_ranks != ref_table.dec.num_ranks) FAIL("num_ranks %d vs %d", mini.num_ranks, ref_table.dec.num_ranks);
    if (mini.sched_len != ref_table.dec.sched_len) FAIL("sched_len %d vs %d", mini.sched_len, ref_table.dec.sched_len);
    if (memcmp(mini.rank_to_sym, ref_table.dec.rank_to_sym, mini.num_ranks)) FAIL("rank_to_sym");
    for (int k = 0; k < mini.sched_len; k++)
        if (mini.sched[k].kd != ref_table.dec.sched[k].kd ||
            mini.sched[k].param != ref_table.dec.sched[k].param ||
            mini.sched[k].right != ref_table.dec.sched[k].right)
            FAIL("sched[%d]: kd %02x/%02x param %d/%d right %d/%d", k,
                 mini.sched[k].kd, ref_table.dec.sched[k].kd,
                 mini.sched[k].param, ref_table.dec.sched[k].param,
                 mini.sched[k].right, ref_table.dec.sched[k].right);
    n_tables++;

    int used[256], n_used = 0;
    for (int s = 0; s < 256; s++) if (freq[s]) used[n_used++] = s;
    static const size_t BN[] = { 1, 2, 100, 8192 };
    for (size_t bi = 0; bi < sizeof(BN) / sizeof(*BN); bi++) {
        size_t N = BN[bi];
        for (size_t i = 0; i < N; i++) blk[i] = (uint8_t)used[rng() % (uint64_t)n_used];

        size_t ref_len = 0;
        if (pivco_huffman_encode(blk, N, &ref_table, enc_ref, &ref_len) != PIVCO_OK)
            FAIL("ref encode N=%zu", N);
        ptrdiff_t mini_len = pivcoh_encode(&mini, blk, N, enc_mini, sizeof(enc_mini), scratch);
        if (mini_len < 0 || (size_t)mini_len != ref_len ||
            memcmp(enc_ref, enc_mini, ref_len)) FAIL("wire differs N=%zu", N);

        ptrdiff_t m2 = pivcoh_encode(&mini, blk, N, enc_mini, sizeof(enc_mini), NULL);
        if (m2 != mini_len || memcmp(enc_ref, enc_mini, ref_len)) FAIL("malloc-path wire N=%zu", N);

        /* mini decodes ref stream; ref decodes mini stream; consumed exact */
        size_t cons = 0;
        memset(dec_buf, 0xAA, N);
        ptrdiff_t dn = pivcoh_decode(&mini, enc_ref, ref_len, dec_buf, sizeof(dec_buf), &cons, dscratch);
        if (dn != (ptrdiff_t)N || cons != ref_len || memcmp(dec_buf, blk, N)) {
            size_t mm = 0;
            while (mm < N && dec_buf[mm] == blk[mm]) mm++;
            FAIL("mini decode N=%zu: dn=%td cons=%zu/%zu first-mismatch@%zu (got %02x want %02x)",
                 N, dn, cons, ref_len, mm, dec_buf[mm < N ? mm : 0], blk[mm < N ? mm : 0]);
        }
        memset(dec_buf, 0xAA, N);
        if (pivco_huffman_decode_dt(enc_mini, mini_len, &ref_table.dec, dec_buf, &cons)
                != PIVCO_OK || cons != ref_len || memcmp(dec_buf, blk, N))
            FAIL("ref decode of mini stream N=%zu", N);
        n_blocks++;

        /* hostile: single-byte mutations + truncations must never crash */
        for (int f = 0; f < 40; f++) {
            memcpy(enc_mini, enc_ref, ref_len);
            enc_mini[rng() % ref_len] ^= (uint8_t)(1u << (rng() & 7));
            (void)pivcoh_decode(&mini, enc_mini, ref_len, dec_buf, sizeof(dec_buf), NULL, dscratch);
            (void)pivcoh_decode(&mini, enc_ref, rng() % (ref_len + 1), dec_buf,
                                sizeof(dec_buf), NULL, dscratch);
            n_fuzz += 2;
        }
    }
}

int main(void)
{
    uint64_t freq[256];
    char tag[32];

    /* compare against pure-PH streams (the library FSE-codes bitmaps by
     * default; pivcoh speaks only the raw-bitmap subset) */
    pivco_huffman_set_fse_enabled(0);

    static const uint64_t SCALE[] = { 2, 100, 65535, 1u << 20 };
    static const int NSYM[] = { 1, 2, 3, 7, 30, 64, 200, 256 };
    for (size_t sc = 0; sc < sizeof(SCALE) / sizeof(*SCALE); sc++)
        for (size_t ns = 0; ns < sizeof(NSYM) / sizeof(*NSYM); ns++)
            for (int rep = 0; rep < 5; rep++) {
                memset(freq, 0, sizeof(freq));
                for (int k = 0; k < NSYM[ns]; k++) {
                    int s;
                    do { s = (int)(rng() & 255); } while (freq[s]);
                    freq[s] = 1 + rng() % SCALE[sc];
                }
                snprintf(tag, sizeof(tag), "rand%zu.%d", sc, NSYM[ns]);
                one_case(freq, tag, rep);
            }

    /* uniform 256: full-alphabet flat tree (D=8 root) */
    for (int s = 0; s < 256; s++) freq[s] = 7;
    one_case(freq, "uniform256", 0);
    /* deep skew: exercises length limiting */
    memset(freq, 0, sizeof(freq));
    { uint64_t a = 1, b = 1;
      for (int s = 0; s < 90; s++) { freq[s] = a; uint64_t t = a + b; a = b; b = t; } }
    one_case(freq, "fib90", 0);

    /* invalid lengths must be rejected */
    { const char *tag = "badlens"; int id = 0;
      uint8_t lens[256];
      memset(lens, 0, 256);
      if (pivcoh_table_from_lens(&mini, lens)) FAIL("accepted all-zero");
      lens[0] = 12;
      if (pivcoh_table_from_lens(&mini, lens)) FAIL("accepted len 12");
      memset(lens, 0, 256); lens[0] = lens[1] = 1; lens[2] = 1;   /* over-subscribed */
      if (pivcoh_table_from_lens(&mini, lens)) FAIL("accepted kraft > 1");
      memset(lens, 0, 256); lens[0] = 2; lens[1] = 2;             /* under-subscribed */
      if (pivcoh_table_from_lens(&mini, lens)) FAIL("accepted kraft < 1"); }

    printf("pivcoh check PASS: %d tables, %d blocks wire-identical + cross-decoded, "
           "%d hostile decodes survived\n", n_tables, n_blocks, n_fuzz);
    return 0;
}

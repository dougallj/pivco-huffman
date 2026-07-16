/* bench_pivcoh_check: verifies extras/pivcoh.h (the stb-style single-header
 * codec) against the production library:
 *   - table_from_freqs code_len == pivco_huffman_build_table code_len,
 *     and the derived schedule/rank tables match (the TREE is still
 *     production-identical; only the wire around it is v4's own)
 *   - mini roundtrips its own v4 wire (malloc and caller-scratch paths,
 *     deterministic re-encode), production roundtrips its own; outputs
 *     equal the input on both sides.  Byte-identity of the streams died
 *     with wire v4 (no FSE marker, 1-byte K_right, lens wire).
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
    /* fresh-zero the pivcoh tables: sym_to_rank now fills lazily on
     * first encode, so whole-struct memcmps are only meaningful from a
     * common (zero) starting state */
    memset(&mini, 0, sizeof(mini));
    memset(&mini2, 0, sizeof(mini2));
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
        if (mini_len < 0) FAIL("mini encode N=%zu", N);

        /* deterministic re-encode through the malloc path */
        static uint8_t enc_mini2[65536];
        ptrdiff_t m2 = pivcoh_encode(&mini, blk, N, enc_mini2, sizeof(enc_mini2), NULL);
        if (m2 != mini_len || memcmp(enc_mini2, enc_mini, (size_t)mini_len))
            FAIL("malloc-path wire N=%zu", N);

        /* mini roundtrips its own v4 stream; production its own */
        size_t cons = 0;
        memset(dec_buf, 0xAA, N);
        ptrdiff_t dn = pivcoh_decode(&mini, enc_mini, (size_t)mini_len, dec_buf, sizeof(dec_buf), &cons, dscratch);
        if (dn != (ptrdiff_t)N || cons != (size_t)mini_len || memcmp(dec_buf, blk, N)) {
            size_t mm = 0;
            while (mm < N && dec_buf[mm] == blk[mm]) mm++;
            FAIL("mini decode N=%zu: dn=%td cons=%zu/%td first-mismatch@%zu (got %02x want %02x)",
                 N, dn, cons, mini_len, mm, dec_buf[mm < N ? mm : 0], blk[mm < N ? mm : 0]);
        }
        memset(dec_buf, 0xAA, N);
        if (pivco_huffman_decode_dt(enc_ref, ref_len, &ref_table.dec, dec_buf, &cons)
                != PIVCO_OK || cons != ref_len || memcmp(dec_buf, blk, N))
            FAIL("ref roundtrip N=%zu", N);
        n_blocks++;

        /* hostile: single-byte mutations + truncations must never crash */
        size_t fz_len = (size_t)mini_len;
        memcpy(enc_ref, enc_mini, fz_len);     /* fuzz base: mini's own v4 stream */
        for (int f = 0; f < 40; f++) {
            memcpy(enc_mini, enc_ref, fz_len);
            enc_mini[rng() % fz_len] ^= (uint8_t)(1u << (rng() & 7));
            (void)pivcoh_decode(&mini, enc_mini, fz_len, dec_buf, sizeof(dec_buf), NULL, dscratch);
            (void)pivcoh_decode(&mini, enc_ref, rng() % (fz_len + 1), dec_buf,
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
    /* deep skew: exercises length limiting.  45 Fibonacci frequencies
     * total fib(47)-1 < 2^32: pivcoh sums frequencies mod 2^32
     * (documented), so production parity is only promised below 4 GiB
     * histogram totals. */
    memset(freq, 0, sizeof(freq));
    { uint64_t a = 1, b = 1;
      for (int s = 0; s < 45; s++) { freq[s] = a; uint64_t t = a + b; a = b; b = t; } }
    one_case(freq, "fib45", 0);

    /* huge-histogram wrap: >= 4 GiB totals derive different-but-valid
     * tables; verify build + self-roundtrip only (no parity). */
    { const char *tag = "hugefreq"; int id = 0;
      memset(freq, 0, sizeof(freq));
      uint64_t a = 1, b = 1;
      for (int s = 0; s < 90; s++) { freq[s] = a; uint64_t t = a + b; a = b; b = t; }
      if (!pivcoh_table_from_freqs(&mini, freq)) FAIL("build failed");
      for (int i = 0; i < 100; i++) blk[i] = (uint8_t)(i % 90);
      ptrdiff_t el = pivcoh_encode(&mini, blk, 100, enc_mini, sizeof(enc_mini), scratch);
      if (el < 0) FAIL("encode failed");
      memset(dec_buf, 0, 100);
      if (pivcoh_decode(&mini, enc_mini, (size_t)el, dec_buf, sizeof(dec_buf),
                        NULL, dscratch) != 100 || memcmp(dec_buf, blk, 100))
          FAIL("roundtrip"); }

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

    /* High-window alphabets (every used symbol > 128): before the
     * enc_umin clamp, enc_init's half-size path read past sym_to_rank.
     * Full wire/roundtrip parity via one_case, then the out-of-window
     * contract: a frequency-zero symbol must encode as rank 0, i.e.
     * byte-identical to a stream with rank_to_sym[0] in its place. */
    { memset(freq, 0, sizeof(freq));
      for (int s = 200; s <= 220; s++) freq[s] = 1 + rng() % 65535;
      one_case(freq, "hiwin.200-220", 0);
      memset(freq, 0, sizeof(freq));
      for (int s = 129; s <= 255; s++) freq[s] = 1 + rng() % 65535;
      one_case(freq, "hiwin.129-255", 0);            /* base clamps to 128 */
      memset(freq, 0, sizeof(freq));
      freq[255] = 3;                                 /* umin would be 255 */
      one_case(freq, "hiwin.255", 0);

      const char *tag = "hiwin.oow"; int id = 0;
      memset(freq, 0, sizeof(freq));
      for (int s = 200; s <= 220; s++) freq[s] = 1 + rng() % 65535;
      memset(&mini, 0, sizeof(mini));
      if (!pivcoh_table_from_freqs(&mini, freq)) FAIL("build");
      for (int i = 0; i < 4096; i++) blk[i] = (uint8_t)(200 + rng() % 21);
      static uint8_t blk2[4096], enc2[65536];
      memcpy(blk2, blk, 4096);
      for (int i = 0; i < 4096; i += 9) {            /* sprinkle both OOW
                                                        classes: below the
                                                        window (wraps) and
                                                        inside it, unused
                                                        (zero s2r bytes) */
          blk[i] = (i & 16) ? (uint8_t)(i % 200)     /* 0..199: below     */
                            : (uint8_t)(221 + i % 35); /* 221..255: inside */
          blk2[i] = mini.rank_to_sym[0];
      }
      ptrdiff_t e1 = pivcoh_encode(&mini, blk, 4096, enc_mini,
                                   sizeof(enc_mini), scratch);
      ptrdiff_t e2 = pivcoh_encode(&mini, blk2, 4096, enc2,
                                   sizeof(enc2), scratch);
      if (e1 < 0 || e1 != e2 || memcmp(enc_mini, enc2, (size_t)e1))
          FAIL("out-of-window symbol not rank 0 (%td vs %td)", e1, e2); }

    /* Joint length/shape tiers.  Production on this branch has no joint
     * pass (defaults keep exact parity above), so these are consistency
     * checks: every tier's lengths form a table both engines accept
     * (production still validates and builds from them — the TREE
     * remains interchangeable even though the v4 wire is not), the
     * scratch and malloc paths agree byte-for-byte, and joint streams
     * roundtrip through a lengths-only pivcoh rebuild. */
    { const char *tag = "joint"; int id = 0;
      static uint8_t jscratch[PIVCOH_JOINT_SCRATCH_SIZE];
      static pivcoh_table jt, jt2;
      static pivco_huffman_decode_table_t dt;
      static const int GR[] = { -1, 0, 1, 2, 4, 8 };
      int n_solve = 0, n_adopt = 0;
      for (id = 0; id < 40; id++) {
          memset(freq, 0, sizeof(freq));
          int ns = 2 + (int)(rng() % 255);
          for (int k = 0; k < ns; k++) {
              int s;
              do { s = (int)(rng() & 255); } while (freq[s]);
              freq[s] = 1 + rng() % (id & 1 ? 65535 : 1u << 20);
          }
          if (!pivcoh_table_from_freqs(&mini, freq)) FAIL("baseline build");
          int used[256], n_used = 0;
          for (int s = 0; s < 256; s++) if (freq[s]) used[n_used++] = s;
          const size_t N = 1 + rng() % 8192;
          for (size_t i = 0; i < N; i++)
              blk[i] = (uint8_t)used[rng() % (uint64_t)n_used];
          for (size_t gi = 0; gi < sizeof(GR) / sizeof(*GR); gi++) {
              pivcoh_joint j = PIVCOH_JOINT_DEFAULTS;
              j.gran = GR[gi];
              memset(&jt, 0, sizeof(jt));
              memset(&jt2, 0, sizeof(jt2));
              if (!pivcoh_table_from_freqs_joint(&jt, freq, &j, jscratch))
                  FAIL("joint build gran=%d", GR[gi]);
              if (!pivcoh_table_from_freqs_joint(&jt2, freq, &j, NULL))
                  FAIL("joint build (malloc) gran=%d", GR[gi]);
              if (memcmp(&jt, &jt2, sizeof(jt)))
                  FAIL("scratch/malloc mismatch gran=%d", GR[gi]);
              n_solve++;
              if (memcmp(jt.code_len, mini.code_len, 256)) n_adopt++;
              ptrdiff_t el = pivcoh_encode(&jt, blk, N, enc_mini,
                                           sizeof(enc_mini), scratch);
              if (el < 0) FAIL("joint encode gran=%d", GR[gi]);
              if (!pivcoh_table_from_lens(&jt2, jt.code_len))
                  FAIL("joint lens rejected by from_lens gran=%d", GR[gi]);
              memset(dec_buf, 0xAA, N);
              if (pivcoh_decode(&jt2, enc_mini, (size_t)el, dec_buf,
                                sizeof(dec_buf), NULL, dscratch)
                      != (ptrdiff_t)N || memcmp(dec_buf, blk, N))
                  FAIL("joint roundtrip gran=%d", GR[gi]);
              if (pivco_huffman_build_decode_table(jt.code_len, &dt)
                      != PIVCO_OK)
                  FAIL("joint lens rejected by production gran=%d", GR[gi]);
          }
          /* lambda > 1/7 breaks the slot DP's order condition; the
           * mass-DP fallback is deliberately not ported, so the reject
           * must gracefully keep the plain Huffman lengths.  (The
           * condition depends only on lambda/kappa, never the data, so
           * in-contract knobs can never reach it.) */
          pivcoh_joint jbig = PIVCOH_JOINT_DEFAULTS;
          jbig.lambda = 0.2f;
          memset(&jt, 0, sizeof(jt));
          if (!pivcoh_table_from_freqs_joint(&jt, freq, &jbig, jscratch)
                  || memcmp(jt.code_len, mini.code_len, 256))
              FAIL("lambda>1/7 did not keep the baseline");

          /* lambda <= 0 must be byte-identical to the plain build
           * (fresh-zeroed structs: bytes past the live region are
           * residue of prior builds, meaningless to compare) */
          pivcoh_joint j0 = PIVCOH_JOINT_DEFAULTS;
          j0.lambda = 0.0f;
          memset(&jt, 0, sizeof(jt));
          memset(&jt2, 0, sizeof(jt2));
          if (!pivcoh_table_from_freqs_joint(&jt, freq, &j0, jscratch)
                  || !pivcoh_table_from_freqs(&jt2, freq)
                  || memcmp(&jt, &jt2, sizeof(jt)))
              FAIL("lambda=0 not identical to plain build");
      }
      printf("pivcoh joint: %d solves consistent (all tiers), %d adopted\n",
             n_solve, n_adopt);
    }

    /* One-shot frame API: roundtrip across sizes and efforts, scratch/
     * malloc determinism, packed-lens equivalence, histogram vs naive,
     * hostile frame fuzz. */
    { const char *tag = "frame"; int id = 0;
      enum { FN_MAX = 1 << 20 };
      static uint8_t fsrc[FN_MAX], fdst[PIVCOH_COMPRESS_BOUND(FN_MAX)];
      static uint8_t fdst2[PIVCOH_COMPRESS_BOUND(FN_MAX)], fout[FN_MAX];
      static uint8_t cscratch[PIVCOH_COMPRESS_SCRATCH_SIZE];
      static uint8_t dscratch2[PIVCOH_DECOMPRESS_SCRATCH_SIZE];
      static const size_t FNS[] = { 0, 1, 2, 100, 4096, 32768, 32769,
                                    100000, FN_MAX };
      static const pivcoh_effort EF[] = { PIVCOH_SIMPLEST_COMPRESS,
                                          PIVCOH_BALANCED,
                                          PIVCOH_FASTER_DECOMPRESS,
                                          PIVCOH_FASTEST_DECOMPRESS,
                                          PIVCOH_FASTEST_COMPRESS };
      int n_frames = 0;
      for (size_t fi = 0; fi < sizeof(FNS) / sizeof(*FNS); fi++) {
          const size_t N = FNS[fi];
          id = (int)N;
          int nsym = 1 + (int)(rng() % 255);
          for (size_t i = 0; i < N; i++)
              fsrc[i] = (uint8_t)(rng() % (uint64_t)nsym);
          for (size_t ei = 0; ei < sizeof(EF) / sizeof(*EF); ei++) {
              ptrdiff_t c = pivcoh_compress(fdst, sizeof(fdst), fsrc, N,
                                            EF[ei], cscratch);
              if (c < 0) FAIL("compress effort=%d", (int)EF[ei]);
              ptrdiff_t c2 = pivcoh_compress(fdst2, sizeof(fdst2), fsrc, N,
                                             EF[ei], NULL);
              if (c2 != c || memcmp(fdst, fdst2, (size_t)c))
                  FAIL("compress scratch/malloc differ effort=%d", (int)EF[ei]);
              if (pivcoh_decompressed_size(fdst, (size_t)c) != N)
                  FAIL("decompressed_size");
              memset(fout, 0xAA, N ? N : 1);
              ptrdiff_t r = pivcoh_decompress(fout, N, fdst, (size_t)c,
                                              dscratch2);
              if (r != (ptrdiff_t)N || memcmp(fout, fsrc, N))
                  FAIL("frame roundtrip effort=%d", (int)EF[ei]);
              if (pivcoh_decompress(fout, N, fdst, (size_t)c, NULL)
                      != (ptrdiff_t)N)
                  FAIL("frame roundtrip (malloc)");
              size_t msz = 12345;
              uint8_t *mbuf = pivcoh_decompress_malloc(fdst, (size_t)c, &msz);
              if (!mbuf || msz != N || memcmp(mbuf, fsrc, N))
                  FAIL("decompress_malloc");
              free(mbuf);
              n_frames++;
              /* hostile: bit flips + truncations must never crash and
               * never write past raw_size (ASan enforces both) */
              for (int f = 0; f < 30; f++) {
                  memcpy(fdst2, fdst, (size_t)c);
                  fdst2[rng() % (uint64_t)c] ^= (uint8_t)(1u << (rng() & 7));
                  (void)pivcoh_decompress(fout, N, fdst2, (size_t)c, dscratch2);
                  (void)pivcoh_decompress(fout, N, fdst,
                                          rng() % ((uint64_t)c + 1), dscratch2);
              }
          }
      }
      /* the BLOCK decoder accepts any u16 symbol count (65535), even
       * though v4 frame coded segments cap at 32767 (the u16's top bit
       * is the raw-store discriminator): roundtrip a max-size block
       * directly, then hand-build a v4 frame from mixed segments
       * (coded + raw) and decompress it */
      { id = 65535;
        pivcoh_table bt;
        uint64_t bfreq[256] = {0};
        for (size_t i = 0; i < 65535; i++) {
            fsrc[i] = (uint8_t)(rng() & 63);
            bfreq[fsrc[i]]++;
        }
        if (!pivcoh_table_from_freqs(&bt, bfreq)) FAIL("big-block table");
        static uint8_t bscratch[PIVCOH_SCRATCH_SIZE(65535)];
        static uint8_t bdscratch[PIVCOH_DECODE_SCRATCH_SIZE(65535)];
        ptrdiff_t el = pivcoh_encode(&bt, fsrc, 65535, fdst,
                                     PIVCOH_ENCODE_BOUND(65535), bscratch);
        if (el < 0) FAIL("big-block encode");
        size_t cons = 0;
        memset(fout, 0xAA, 65535);
        if (pivcoh_decode(&bt, fdst, (size_t)el, fout, 65535, &cons, bdscratch)
                != 65535 || cons != (size_t)el || memcmp(fout, fsrc, 65535))
            FAIL("65535-symbol block rejected or wrong");

        /* hand-built v4 frame: [varint 40000][lens wire]
         * [coded seg 32767][raw seg 7233] */
        size_t fo = 0;
        fdst[fo++] = (uint8_t)(40000 & 127) | 128;
        fdst[fo++] = (uint8_t)(40000 >> 7 & 127) | 128;
        fdst[fo++] = (uint8_t)(40000 >> 14);
        int lw = pivcoh_lens_wire_write(fdst + fo, bt.code_len);
        if (lw < 1) FAIL("frame lens wire");
        fo += (size_t)lw;
        el = pivcoh_encode(&bt, fsrc, 32767, fdst + fo,
                           PIVCOH_ENCODE_BOUND(32767), bscratch);
        if (el < 0) FAIL("frame seg encode");
        fo += (size_t)el;
        fdst[fo++] = (uint8_t)(7233 & 255);
        fdst[fo++] = (uint8_t)(7233 >> 8) | 0x80;
        memcpy(fdst + fo, fsrc + 32767, 7233);
        fo += 7233;
        memset(fout, 0xAA, 40000);
        if (pivcoh_decompress(fout, 40000, fdst, fo, dscratch2) != 40000
                || memcmp(fout, fsrc, 40000))
            FAIL("hand-built v4 frame rejected or wrong");
      }

      /* utilities: packed lens roundtrip + fused build; histogram */
      { id = 0;
        uint8_t packed[128], lens[256];
        pivcoh_lens_pack(packed, mini.code_len);
        pivcoh_lens_unpack(lens, packed);
        if (memcmp(lens, mini.code_len, 256)) FAIL("lens pack/unpack");
        pivcoh_table pt, lt;
        memset(&pt, 0, sizeof(pt)); memset(&lt, 0, sizeof(lt));
        if (!pivcoh_table_from_packed_lens(&pt, packed) ||
            !pivcoh_table_from_lens(&lt, mini.code_len) ||
            memcmp(&pt, &lt, sizeof(pt)))
            FAIL("from_packed_lens != from_lens");
        uint64_t h1[256] = {0}, h4[256];
        for (size_t i = 0; i < 100000; i++) h1[fsrc[i]]++;
        pivcoh_histogram(h4, fsrc, 100000);
        if (memcmp(h1, h4, sizeof(h1))) FAIL("histogram mismatch");
      }
      printf("pivcoh frame: %d frames round-tripped (all efforts), "
             "utilities consistent\n", n_frames);
    }

    printf("pivcoh check PASS: %d tables, %d blocks round-tripped (v4 wire, trees production-identical), "
           "%d hostile decodes survived\n", n_tables, n_blocks, n_fuzz);
    return 0;
}

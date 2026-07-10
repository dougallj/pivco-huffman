#include "pivco_huffman.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Utilities ---------- */

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

#define FAIL(msg, ...) do { \
    printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
    return 1; \
} while (0)

/* ---------- Test: build table and verify canonical codes ---------- */

static int test_table_build(void)
{
    printf("[test_table_build] ");

    uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
    freq[0] = 100;
    freq[1] = 50;
    freq[2] = 25;
    freq[3] = 12;
    freq[4] = 6;

    pivco_huffman_table_t table;
    int rc = pivco_huffman_build_table(freq, &table);
    if (rc != PIVCO_OK) FAIL("build_table returned %d", rc);

    /* Verify prefix-free property: no code is a prefix of another */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (table.code_len[i] == 0) continue;
        for (int j = i + 1; j < PIVCO_MAX_SYMBOLS; j++) {
            if (table.code_len[j] == 0) continue;
            int shorter = table.code_len[i] < table.code_len[j] ? i : j;
            int longer  = shorter == i ? j : i;
            int slen = table.code_len[shorter];
            int llen = table.code_len[longer];
            uint16_t prefix = table.code[longer] >> (llen - slen);
            if (prefix == table.code[shorter]) {
                FAIL("code[%d]=%u/%d is prefix of code[%d]=%u/%d",
                     shorter, table.code[shorter], slen,
                     longer, table.code[longer], llen);
            }
        }
    }

    pivco_huffman_build_traditional_table(&table);
    /* Verify decode table round-trips */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        if (table.code_len[i] == 0) continue;
        uint16_t code = table.code[i];
        int len = table.code_len[i];
        uint32_t idx = (uint32_t)code << (PIVCO_MAX_CODE_LEN - len);
        if (table.decode_sym[idx] != (uint8_t)i) {
            FAIL("decode_sym[%u] = %d, expected %d", idx, table.decode_sym[idx], i);
        }
        if (table.decode_len[idx] != len) {
            FAIL("decode_len[%u] = %d, expected %d", idx, table.decode_len[idx], len);
        }
    }

    printf("PASS\n");
    return 0;
}

/* ---------- Test: single-symbol alphabet ---------- */

static int test_single_symbol(void)
{
    printf("[test_single_symbol] ");

    uint64_t freq[PIVCO_MAX_SYMBOLS] = {0};
    freq[42] = 100;

    pivco_huffman_table_t table;
    pivco_huffman_build_table(freq, &table);

    uint8_t symbols[PIVCO_BLOCK_SIZE];
    memset(symbols, 42, sizeof(symbols));

    /* PIVCO roundtrip */
    uint8_t encoded[PIVCO_MAX_ENCODED_SIZE];
    size_t enc_len;
    int rc = pivco_huffman_encode_scalar(symbols, PIVCO_BLOCK_SIZE, &table, encoded, &enc_len);
    if (rc != PIVCO_OK) FAIL("encode returned %d", rc);

    uint8_t decoded[PIVCO_BLOCK_SIZE];
    size_t consumed;
    rc = pivco_huffman_decode_scalar(encoded, enc_len, &table, decoded, &consumed);
    if (rc != PIVCO_OK) FAIL("decode returned %d", rc);

    if (memcmp(symbols, decoded, PIVCO_BLOCK_SIZE) != 0) {
        FAIL("PIVCO roundtrip mismatch");
    }

    /* Traditional roundtrip */
    pivco_huffman_build_traditional_table(&table);
    uint8_t trad_enc[PIVCO_BLOCK_SIZE * 2];
    size_t trad_len, trad_bits;
    rc = trad_huffman_encode(symbols, PIVCO_BLOCK_SIZE, &table,
                             trad_enc, &trad_len, &trad_bits);
    if (rc != PIVCO_OK) FAIL("trad encode returned %d", rc);

    uint8_t trad_dec[PIVCO_BLOCK_SIZE];
    rc = trad_huffman_decode(trad_enc, trad_bits, &table,
                             trad_dec, PIVCO_BLOCK_SIZE);
    if (rc != PIVCO_OK) FAIL("trad decode returned %d", rc);

    if (memcmp(symbols, trad_dec, PIVCO_BLOCK_SIZE) != 0) {
        FAIL("trad roundtrip mismatch");
    }

    printf("PASS (encoded %zu bytes pivco, %zu bytes trad)\n", enc_len, trad_len);
    return 0;
}

/* ---------- Helper: roundtrip test with a given frequency distribution ---------- */

static int test_roundtrip_dist(const char *name, const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                uint64_t seed)
{
    printf("[test_roundtrip_%s] ", name);

    pivco_huffman_table_t table;
    int rc = pivco_huffman_build_table(freq, &table);
    if (rc != PIVCO_OK) FAIL("build_table returned %d", rc);

    /* Build CDF for sampling */
    uint64_t total = 0;
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) total += freq[i];
    if (total == 0) FAIL("empty frequency table");

    /* Generate random symbols from the distribution */
    uint8_t symbols[PIVCO_BLOCK_SIZE];
    uint64_t rng = seed;
    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        uint64_t r = xorshift64(&rng) % total;
        uint64_t cum = 0;
        int sym = 0;
        for (sym = 0; sym < PIVCO_MAX_SYMBOLS; sym++) {
            cum += freq[sym];
            if (r < cum) break;
        }
        symbols[i] = (uint8_t)sym;
    }

    /* PIVCO scalar roundtrip */
    uint8_t encoded[PIVCO_MAX_ENCODED_SIZE];
    size_t enc_len;
    rc = pivco_huffman_encode_scalar(symbols, PIVCO_BLOCK_SIZE, &table, encoded, &enc_len);
    if (rc != PIVCO_OK) FAIL("pivco encode returned %d", rc);

    uint8_t decoded[PIVCO_BLOCK_SIZE];
    size_t consumed;
    rc = pivco_huffman_decode_scalar(encoded, enc_len, &table, decoded, &consumed);
    if (rc != PIVCO_OK) FAIL("pivco decode returned %d", rc);

    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        if (symbols[i] != decoded[i]) {
            FAIL("pivco mismatch at position %d: expected %d, got %d",
                 i, symbols[i], decoded[i]);
        }
    }

    if (consumed != enc_len) {
        FAIL("pivco consumed %zu bytes, expected %zu", consumed, enc_len);
    }

    /* Traditional roundtrip */
    pivco_huffman_build_traditional_table(&table);
    uint8_t trad_enc[PIVCO_BLOCK_SIZE * 4];
    size_t trad_len, trad_bits;
    rc = trad_huffman_encode(symbols, PIVCO_BLOCK_SIZE, &table,
                             trad_enc, &trad_len, &trad_bits);
    if (rc != PIVCO_OK) FAIL("trad encode returned %d", rc);

    uint8_t trad_dec[PIVCO_BLOCK_SIZE];
    rc = trad_huffman_decode(trad_enc, trad_bits, &table,
                             trad_dec, PIVCO_BLOCK_SIZE);
    if (rc != PIVCO_OK) FAIL("trad decode returned %d", rc);

    for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
        if (symbols[i] != trad_dec[i]) {
            FAIL("trad mismatch at position %d: expected %d, got %d",
                 i, symbols[i], trad_dec[i]);
        }
    }

#ifdef PIVCO_HAS_NEON
    /* NEON encode + BU NEON decode roundtrip (the production path). */
    uint8_t neon_enc[PIVCO_MAX_ENCODED_SIZE];
    size_t neon_len;
    rc = pivco_huffman_encode_neon(symbols, PIVCO_BLOCK_SIZE, &table, neon_enc, &neon_len);
    if (rc != PIVCO_OK) FAIL("neon encode returned %d", rc);

    {
        uint8_t bu_dec[PIVCO_BLOCK_SIZE];
        size_t bu_consumed;
        rc = pivco_huffman_decode_bu_neon(neon_enc, neon_len, &table,
                                           bu_dec, &bu_consumed);
        if (rc != PIVCO_OK) FAIL("bu_neon decode returned %d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != bu_dec[i]) {
                FAIL("bu_neon mismatch at position %d: expected %d, got %d",
                     i, symbols[i], bu_dec[i]);
            }
        }
        if (bu_consumed != neon_len) {
            FAIL("bu_neon consumed %zu bytes, expected %zu",
                 bu_consumed, neon_len);
        }
    }

    /* Cross-check: NEON-encoded stream against scalar decoder.
     * Catches encoder bugs that NEON decode reads symmetrically. */
    {
        uint8_t cross_dec[PIVCO_BLOCK_SIZE];
        size_t cross_consumed;
        rc = pivco_huffman_decode_scalar(neon_enc, neon_len, &table,
                                          cross_dec, &cross_consumed);
        if (rc != PIVCO_OK) FAIL("neon-enc -> scalar-dec rc=%d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != cross_dec[i]) {
                FAIL("neon-enc / scalar-dec mismatch at %d: "
                     "expected %d, got %d",
                     i, symbols[i], cross_dec[i]);
            }
        }
    }
#endif

#ifdef PIVCO_HAS_SSE4
    /* SSE encode + BU SSE/AVX-512 decode roundtrip. */
    {
        uint8_t sse_enc[PIVCO_MAX_ENCODED_SIZE];
        size_t sse_len;
        rc = pivco_huffman_encode_x86(symbols, PIVCO_BLOCK_SIZE, &table, sse_enc, &sse_len);
        if (rc != PIVCO_OK) FAIL("sse encode returned %d", rc);

        uint8_t bu_dec[PIVCO_BLOCK_SIZE];
        size_t bu_consumed;
        rc = pivco_huffman_decode_bu_x86(sse_enc, sse_len, &table,
                                          bu_dec, &bu_consumed);
        if (rc != PIVCO_OK) FAIL("bu_x86 decode returned %d", rc);
        for (int i = 0; i < PIVCO_BLOCK_SIZE; i++) {
            if (symbols[i] != bu_dec[i]) {
                FAIL("bu_x86 mismatch at position %d: expected %d, got %d",
                     i, symbols[i], bu_dec[i]);
            }
        }
        if (bu_consumed != sse_len) {
            FAIL("bu_x86 consumed %zu bytes, expected %zu",
                 bu_consumed, sse_len);
        }
    }
#endif

    printf("PASS (pivco=%zu B, trad=%zu B, ratio=%.2fx)\n",
           enc_len, trad_len, (double)enc_len / (double)trad_len);
    return 0;
}

/* ---------- Distribution generators ---------- */

static void make_uniform(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) freq[i] = 100;
}

static void make_english(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    /* Approximate English character frequencies */
    freq[' '] = 1830; freq['e'] = 1270; freq['t'] = 910;
    freq['a'] = 820;  freq['o'] = 750;  freq['i'] = 700;
    freq['n'] = 670;  freq['s'] = 630;  freq['h'] = 610;
    freq['r'] = 600;  freq['d'] = 430;  freq['l'] = 400;
    freq['c'] = 280;  freq['u'] = 280;  freq['m'] = 240;
    freq['w'] = 240;  freq['f'] = 220;  freq['g'] = 200;
    freq['y'] = 200;  freq['p'] = 190;  freq['b'] = 150;
    freq['v'] = 100;  freq['k'] = 80;   freq['j'] = 15;
    freq['x'] = 15;   freq['q'] = 10;   freq['z'] = 7;
    freq['.'] = 65;   freq[','] = 61;   freq['\n'] = 50;
}

static void make_zipfian(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        freq[i] = (uint64_t)(10000.0 / (double)(i + 1));
        if (freq[i] == 0) freq[i] = 1;
    }
}

static void make_sparse_4(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    freq[0] = 100; freq[1] = 100; freq[2] = 100; freq[3] = 100;
}

static void make_sparse_16(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    for (int i = 0; i < 16; i++) freq[i] = 100;
}

static void make_geometric(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    /* Steep geometric: freq[i] ~= 2^(15-i), capped to ensure 15-bit codes */
    for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) {
        int shift = i < 30 ? 30 - i : 0;
        freq[i] = (uint64_t)1 << shift;
        if (freq[i] == 0) freq[i] = 1;
    }
}

static void make_two_symbol_equal(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    freq[0] = 500; freq[1] = 500;
}

static void make_two_symbol_skewed(uint64_t freq[PIVCO_MAX_SYMBOLS])
{
    memset(freq, 0, PIVCO_MAX_SYMBOLS * sizeof(uint64_t));
    freq[0] = 900; freq[1] = 100;
}

#ifdef PIVCO_HAS_NEON
/* ---------- Exact-contract stress (tail-free decode) ----------
 *
 * The NEON decode kernels are tail-free (full-width loops straight
 * past every region end) and the codec confines the over-wide
 * stores/loads to internal scratch.  Verify the EXACT caller contract
 * survives that: decode blocks of many sizes under five tree shapes
 * with a dst buffer of exactly n bytes (32-byte 0xCB canary directly
 * at n — any spill fails) and a src buffer of exactly enc_len bytes
 * (heap-allocated, so a GuardMalloc run faults on any overread; the
 * codec's in_end bounce is what keeps reads inside).  Results are
 * cross-checked against the scalar decoder. */
static int test_exact_contract_sizes(void)
{
    printf("[test_exact_contract_sizes] ");
    static const int sizes[] = { 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32,
                                 33, 48, 63, 64, 65, 100, 127, 128, 129, 255,
                                 257, 1000, 4095, 4096, 4097, 8191, 8192 };
    enum { NSIZES = (int)(sizeof(sizes) / sizeof(sizes[0])),
           MAXN = 8192, CANARY = 32 };
    static uint8_t sym[MAXN], ref[MAXN];
    static uint8_t enc[PIVCO_MAX_ENCODED_SIZE];
    uint64_t freq[PIVCO_MAX_SYMBOLS];
    uint64_t rng = 0x1234abcd5678ULL;

    for (int d = 0; d < 5; d++) {
        switch (d) {
        case 0: make_english(freq);           break;
        case 1: make_uniform(freq);           break;
        case 2: make_two_symbol_skewed(freq); break;
        case 3: make_sparse_16(freq);         break;
        default: make_geometric(freq);        break;
        }
        pivco_huffman_table_t table;
        if (pivco_huffman_build_table(freq, &table) != PIVCO_OK)
            FAIL("build_table failed (dist %d)", d);
        uint64_t total = 0;
        for (int i = 0; i < PIVCO_MAX_SYMBOLS; i++) total += freq[i];

        for (int s = 0; s < NSIZES; s++) {
            int n = sizes[s];
            for (int i = 0; i < n; i++) {
                uint64_t r = xorshift64(&rng) % total;
                uint64_t cum = 0;
                int v;
                for (v = 0; v < PIVCO_MAX_SYMBOLS; v++) {
                    cum += freq[v];
                    if (r < cum) break;
                }
                sym[i] = (uint8_t)v;
            }
            size_t enc_len, c1, c2;
            if (pivco_huffman_encode_neon(sym, (size_t)n, &table,
                                          enc, &enc_len) != PIVCO_OK)
                FAIL("encode n=%d dist=%d", n, d);
            if (pivco_huffman_decode_scalar(enc, enc_len, &table,
                                            ref, &c1) != PIVCO_OK)
                FAIL("scalar decode n=%d dist=%d", n, d);
            if (memcmp(sym, ref, (size_t)n) != 0)
                FAIL("scalar roundtrip mismatch n=%d dist=%d", n, d);

            /* Exact-size src: any kernel read past enc_len is out of
             * this allocation (GuardMalloc-fatal). */
            uint8_t *tight_src = malloc(enc_len);
            if (!tight_src) FAIL("oom src n=%d", n);
            memcpy(tight_src, enc, enc_len);

            /* Exact-size dst + canary directly at n. */
            uint8_t *dec = malloc((size_t)n + CANARY);
            if (!dec) { free(tight_src); FAIL("oom dst n=%d", n); }
            memset(dec, 0xCB, (size_t)n + CANARY);

            int rc = pivco_huffman_decode_bu_neon(tight_src, enc_len,
                                                  &table, dec, &c2);
            if (rc != PIVCO_OK) {
                free(tight_src); free(dec);
                FAIL("neon decode n=%d dist=%d rc=%d", n, d, rc);
            }
            int bad = memcmp(dec, ref, (size_t)n) != 0;
            int spilled = 0;
            for (int i = 0; i < CANARY; i++)
                if (dec[n + i] != 0xCB) spilled = i + 1;
            free(tight_src); free(dec);
            if (bad)
                FAIL("neon/scalar decode mismatch n=%d dist=%d", n, d);
            if (spilled)
                FAIL("write past dst end: n=%d dist=%d byte +%d",
                     n, d, spilled - 1);
            if (c2 != enc_len)
                FAIL("consumed %zu != enc_len %zu (n=%d dist=%d)",
                     c2, enc_len, n, d);
        }
    }
    printf("PASS\n");
    return 0;
}
#endif  /* PIVCO_HAS_NEON */

/* ---------- Main test runner ---------- */

int test_roundtrip_all(void)
{
    int failures = 0;

    failures += test_table_build();
    failures += test_single_symbol();

    uint64_t freq[PIVCO_MAX_SYMBOLS];
    uint64_t seed = 0xDEADBEEFCAFE1234ULL;

    make_uniform(freq);
    failures += test_roundtrip_dist("uniform", freq, seed++);

    make_english(freq);
    failures += test_roundtrip_dist("english", freq, seed++);

    make_zipfian(freq);
    failures += test_roundtrip_dist("zipfian", freq, seed++);

    make_sparse_4(freq);
    failures += test_roundtrip_dist("sparse_4", freq, seed++);

    make_sparse_16(freq);
    failures += test_roundtrip_dist("sparse_16", freq, seed++);

    make_geometric(freq);
    failures += test_roundtrip_dist("geometric", freq, seed++);

    make_two_symbol_equal(freq);
    failures += test_roundtrip_dist("two_sym_eq", freq, seed++);

    make_two_symbol_skewed(freq);
    failures += test_roundtrip_dist("two_sym_skew", freq, seed++);

#ifdef PIVCO_HAS_NEON
    failures += test_exact_contract_sizes();
#endif

    /* Multiple blocks with different seeds */
    make_zipfian(freq);
    for (int b = 0; b < 10; b++) {
        char name[32];
        snprintf(name, sizeof(name), "zipf_block_%d", b);
        failures += test_roundtrip_dist(name, freq, seed + (uint64_t)b * 12345);
    }

    return failures;
}

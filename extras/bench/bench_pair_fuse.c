/* bench_pair_fuse: fused merge(bm1, pair(bm2,a,b), stream) vs the
 * production two-pass (merge_cst_cst -> scratch; merge_vec_vec).
 * Kernels copied from pivco_huffman_primitives_neon.h idioms.
 * K multiple of 64; fresh data per buffer. */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static uint64_t rs = 0x243F6A8885A308D3ull;
static uint64_t xr(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }

/* ---- production merge tables (verbatim) ---- */
static int8_t g_merge_shuf0[256 * 16] __attribute__((aligned(16)));
static int8_t g_merge_shuf1[256 * 16] __attribute__((aligned(16)));
static void init_merge_tables(void)
{
    for (int i = 0; i < 256; i++) {
        int8_t pop = 0;
        int8_t *o0 = &g_merge_shuf0[i * 16];
        int8_t *o1 = &g_merge_shuf1[i * 16];
        for (int j = 0; j < 8; j++) {
            if ((i >> j) & 1) { o0[j] = pop; o1[j + 8] = (int8_t)(-pop); pop++; }
            else { int8_t v = (int8_t)(-16 - j + pop); o0[j] = v; o1[j + 8] = (int8_t)(8 - v); }
        }
        for (int j = 0; j < 8; j++) { o0[j + 8] = pop; o1[j] = 0; }
    }
}
static inline void merge_neon_16B(uint8_t *dest, const uint8_t *l_list,
                                  const uint8_t *r_list, intptr_t mask)
{
    int8x16_t shuf0 = vld1q_s8(&g_merge_shuf0[((uintptr_t)mask << 4) & 0xff0]);
    int8x16_t shuf1 = vld1q_s8(&g_merge_shuf1[((uintptr_t)mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(shuf0, shuf1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r_list);
    src.val[1] = vld1q_u8(l_list);
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}
static void merge_vec_vec(const uint8_t *bm, int K, const uint8_t *left,
                          const uint8_t *right, uint8_t *out)
{
    const uint8_t *l_list = left, *r_list = right;
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff, p3 = pfx >> 56;
        merge_neon_16B(out + i,      l_list,           r_list,      mask);
        merge_neon_16B(out + i + 16, l_list + 16 - p0, r_list + p0, mask >> 16);
        merge_neon_16B(out + i + 32, l_list + 32 - p1, r_list + p1, mask >> 32);
        merge_neon_16B(out + i + 48, l_list + 48 - p2, r_list + p2, mask >> 48);
        r_list += p3; l_list += 64 - p3;
    }
    int lc = (int)(l_list - left), rc = (int)(r_list - right);
    for (int j = (int)i; j < K; j++) {
        int mb = (bm[j >> 3] >> (j & 7)) & 1;
        out[j] = mb ? right[rc++] : left[lc++];
    }
}
/* ---- production pair kernel (merge_cst_cst idiom) ---- */
static const uint8_t dup16_tab[16]  = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
static const uint8_t bit16_tab[16]  = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
static void merge_cst_cst(const uint8_t *bm, int K, uint8_t a, uint8_t b,
                          uint8_t *out)
{
    uint8x16_t va = vdupq_n_u8(a), vb = vdupq_n_u8(b);
    uint8x16_t dt = vld1q_u8(dup16_tab), bt = vld1q_u8(bit16_tab);
    int i = 0;
    for (; i + 16 <= K; i += 16) {
        uint16_t bits; memcpy(&bits, bm + (i >> 3), 2);
        uint8x16_t v = vreinterpretq_u8_u16(vdupq_n_u16(bits));
        uint8x16_t sel = vqtbl1q_u8(v, dt);
        uint8x16_t hit = vtstq_u8(sel, bt);
        vst1q_u8(out + i, vbslq_u8(hit, vb, va));
    }
    for (; i < K; i++) out[i] = ((bm[i >> 3] >> (i & 7)) & 1) ? b : a;
}
/* ---- FUSED: pair-side generated in-register at a bit cursor ---- */
static inline uint8x16_t pair16(const uint8_t *bm2, intptr_t bitpos,
                                uint8x16_t va, uint8x16_t vb,
                                uint8x16_t dt, uint8x16_t bt)
{
    uint32_t w; memcpy(&w, bm2 + (bitpos >> 3), 4);
    uint16_t bits = (uint16_t)(w >> (bitpos & 7));
    uint8x16_t v = vreinterpretq_u8_u16(vdupq_n_u16(bits));
    uint8x16_t sel = vqtbl1q_u8(v, dt);
    uint8x16_t hit = vtstq_u8(sel, bt);
    return vbslq_u8(hit, vb, va);
}
static void merge_pair_vec(const uint8_t *bm1, const uint8_t *bm2, int K,
                           uint8_t a, uint8_t b, const uint8_t *left,
                           uint8_t *out)
{
    uint8x16_t va = vdupq_n_u8(a), vb = vdupq_n_u8(b);
    uint8x16_t dt = vld1q_u8(dup16_tab), bt = vld1q_u8(bit16_tab);
    const uint8_t *l_list = left;
    intptr_t pc = 0;            /* bit cursor into bm2 (pair side) */
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm1 + (i >> 3), 8);
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff, p3 = pfx >> 56;
#define _MPV(off, lofs, pofs, mk) do {                                         \
        int8x16_t s0 = vld1q_s8(&g_merge_shuf0[(((uintptr_t)(mk)) << 4) & 0xff0]); \
        int8x16_t s1 = vld1q_s8(&g_merge_shuf1[(((uintptr_t)(mk)) >> 4) & 0xff0]); \
        uint8x16_t sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));                 \
        uint8x16x2_t src;                                                      \
        src.val[0] = pair16(bm2, pc + (pofs), va, vb, dt, bt);                 \
        src.val[1] = vld1q_u8(l_list + (lofs));                                \
        vst1q_u8(out + i + (off), vqtbl2q_u8(src, sh));                        \
    } while (0)
        _MPV(0,  0,       0,  mask);
        _MPV(16, 16 - p0, p0, mask >> 16);
        _MPV(32, 32 - p1, p1, mask >> 32);
        _MPV(48, 48 - p2, p2, mask >> 48);
#undef _MPV
        pc += p3; l_list += 64 - p3;
    }
    int lc = (int)(l_list - left);
    for (int j = (int)i; j < K; j++) {
        int mb = (bm1[j >> 3] >> (j & 7)) & 1;
        if (mb) { out[j] = ((bm2[pc >> 3] >> (pc & 7)) & 1) ? b : a; pc++; }
        else out[j] = left[lc++];
    }
}

#define K 16384
#define NBUF 64
int main(int argc, char **argv)
{
    init_merge_tables();
    double dens[] = {0.125, 0.25, 0.5, 0.75};
    printf("%6s | %8s %8s %8s | %8s %8s\n",
           "pair%%", "2pass", "fused", "saving", "pairpass", "mergeonly");
    for (int di = 0; di < 4; di++) {
        double q = dens[di];
        static uint8_t bm1[NBUF][K / 8 + 16], bm2[NBUF][K / 8 + 16];
        static uint8_t stream[NBUF][K + 16], scratch[K + 64];
        static uint8_t out1[K + 64], out2[K + 64];
        static int npair[NBUF];
        for (int n = 0; n < NBUF; n++) {
            int np = 0;
            for (int i = 0; i < K; i++) {
                int bit = (xr() % 1000) < (uint64_t)(q * 1000);
                if (bit) { bm1[n][i >> 3] |= 1 << (i & 7); np++; }
                else bm1[n][i >> 3] &= ~(1 << (i & 7));
            }
            npair[n] = np;
            for (int i = 0; i < K / 8 + 8; i++) bm2[n][i] = (uint8_t)xr();
            for (int i = 0; i < K; i++) stream[n][i] = (uint8_t)xr();
        }
        /* verify */
        for (int n = 0; n < 4; n++) {
            merge_cst_cst(bm2[n], npair[n], 0xAA, 0x55, scratch);
            merge_vec_vec(bm1[n], K, stream[n], scratch, out1);
            merge_pair_vec(bm1[n], bm2[n], K, 0xAA, 0x55, stream[n], out2);
            if (memcmp(out1, out2, K) != 0) { printf("VERIFY FAIL q=%.2f n=%d\n", q, n); return 1; }
        }
        /* bench */
        double t2p = 1e18, tf = 1e18, tpp = 1e18, tm = 1e18;
        for (int r = 0; r < 5; r++) {
            double t0 = now_sec();
            for (int rep = 0; rep < 200; rep++)
                for (int n = 0; n < NBUF; n++) {
                    merge_cst_cst(bm2[n], npair[n], 0xAA, 0x55, scratch);
                    merge_vec_vec(bm1[n], K, stream[n], scratch, out1);
                }
            double d = (now_sec() - t0) / 200 / NBUF / K * 1e9;
            if (d < t2p) t2p = d;
            t0 = now_sec();
            for (int rep = 0; rep < 200; rep++)
                for (int n = 0; n < NBUF; n++)
                    merge_pair_vec(bm1[n], bm2[n], K, 0xAA, 0x55, stream[n], out2);
            d = (now_sec() - t0) / 200 / NBUF / K * 1e9;
            if (d < tf) tf = d;
            t0 = now_sec();
            for (int rep = 0; rep < 200; rep++)
                for (int n = 0; n < NBUF; n++)
                    merge_cst_cst(bm2[n], npair[n], 0xAA, 0x55, scratch);
            d = (now_sec() - t0) / 200 / NBUF / K * 1e9;
            if (d < tpp) tpp = d;
            t0 = now_sec();
            for (int rep = 0; rep < 200; rep++)
                for (int n = 0; n < NBUF; n++)
                    merge_vec_vec(bm1[n], K, stream[n], scratch, out1);
            d = (now_sec() - t0) / 200 / NBUF / K * 1e9;
            if (d < tm) tm = d;
        }
        printf("%5.0f%% | %8.4f %8.4f %+7.1f%% | %8.4f %8.4f\n",
               q * 100, t2p, tf, 100 * (1 - tf / t2p), tpp, tm);
    }
    return 0;
}

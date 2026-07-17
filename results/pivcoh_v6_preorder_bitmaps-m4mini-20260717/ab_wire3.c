/* ab_wire3 — single-binary 3-way A/B of pivcoh wire v6 vs v4 on
 * fixed-G windows of real literal streams.  One pivcoh block per
 * window (G <= 32767).
 *
 *   A = wire v6, larger-K-first child regions (extras/pivcoh.h as-is)
 *   R = wire v6, right-child-first regions   (pivcohR.h, sed-namespaced,
 *       compiled with PIVCOHR_RIGHT_FIRST=1)
 *   B = wire v4 baseline                     (pivcohB.h, sed-namespaced
 *       from pivcoh-neon @ 0efba3b)
 *
 * Per window all engines build tables from the same histogram (gated
 * identical code_len), encode their own ciphertext (gated: A and R
 * equal SIZE — same bytes, permuted regions; B >= A — v4 adds 1-2
 * header bytes per non-flat internal node) and round-trip.  Timing:
 * >=100 ms DVFS warmup, reps calibrated to the target ms per timed
 * run, ROUNDS rounds with the 3-engine order rotating per round,
 * medians reported.  dec = *_decode with prebuilt tables (caller
 * scratch); enc = *_encode with prebuilt tables.
 *
 * Usage: ab_wire3 [--G=BYTES] [--reps=MS] [--joint] file...
 */
#define PIVCOH_IMPLEMENTATION
#include "pivcohA.h"
#define PIVCOHR_IMPLEMENTATION
#include "pivcohR.h"
#define PIVCOHB_IMPLEMENTATION
#include "pivcohB.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

#define ROUNDS 12
#define MAXWIN 32768

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint8_t escratch[PIVCOH_SCRATCH_SIZE(MAXWIN)];
static uint8_t dscratchA[PIVCOH_DECODE_SCRATCH_SIZE(MAXWIN)];
static uint8_t dscratchR[PIVCOHR_DECODE_SCRATCH_SIZE(MAXWIN)];
static uint8_t dscratchB[PIVCOHB_DECODE_SCRATCH_SIZE(MAXWIN)];
static uint8_t encout[PIVCOH_ENCODE_BOUND(MAXWIN)];
static uint8_t jscratch[PIVCOH_JOINT_SCRATCH_SIZE];

typedef struct {
    uint32_t off, n;            /* window bytes in src */
    uint32_t eoffA, elenA;      /* ciphertext in per-engine arena */
    uint32_t eoffR, elenR;
    uint32_t eoffB, elenB;
} win_t;

static uint8_t *src, *out, *encA, *encR, *encB;
static win_t *wins;
static pivcoh_table  *tabA;
static pivcohR_table *tabR;
static pivcohB_table *tabB;
static int nwin;

static void sweep_decA(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcoh_decode(&tabA[w], encA + wins[w].eoffA, wins[w].elenA,
                          out + wins[w].off, wins[w].n, NULL, dscratchA) < 0)
            exit(fprintf(stderr, "A decode failed w%d\n", w));
}

static void sweep_decR(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcohR_decode(&tabR[w], encR + wins[w].eoffR, wins[w].elenR,
                           out + wins[w].off, wins[w].n, NULL, dscratchR) < 0)
            exit(fprintf(stderr, "R decode failed w%d\n", w));
}

static void sweep_decB(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcohB_decode(&tabB[w], encB + wins[w].eoffB, wins[w].elenB,
                           out + wins[w].off, wins[w].n, NULL, dscratchB) < 0)
            exit(fprintf(stderr, "B decode failed w%d\n", w));
}

static void sweep_encA(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcoh_encode(&tabA[w], src + wins[w].off, wins[w].n,
                          encout, sizeof encout, escratch) < 0)
            exit(fprintf(stderr, "A encode failed w%d\n", w));
}

static void sweep_encR(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcohR_encode(&tabR[w], src + wins[w].off, wins[w].n,
                           encout, sizeof encout, escratch) < 0)
            exit(fprintf(stderr, "R encode failed w%d\n", w));
}

static void sweep_encB(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcohB_encode(&tabB[w], src + wins[w].off, wins[w].n,
                           encout, sizeof encout, escratch) < 0)
            exit(fprintf(stderr, "B encode failed w%d\n", w));
}

/* one interleaved 3-way measurement: median seconds per sweep, per
 * engine, with the in-round order rotating so each engine sees every
 * position ROUNDS/3 times */
static void ab_time3(void (*f[3])(void), double target_ms, double m[3])
{
    double t0 = now_sec();                       /* DVFS ramp */
    do { f[0](); f[1](); f[2](); } while (now_sec() - t0 < 0.1);
    double t1 = now_sec();
    f[0]();
    double per = now_sec() - t1;
    int reps = per > 0 ? (int)(target_ms / 1e3 / per) + 1 : 1;

    double r[3][ROUNDS];
    for (int rd = 0; rd < ROUNDS; rd++) {
        for (int k = 0; k < 3; k++) {
            int e = (rd + k) % 3;
            double s = now_sec();
            for (int i = 0; i < reps; i++) f[e]();
            r[e][rd] = (now_sec() - s) / reps;
        }
    }
    for (int e = 0; e < 3; e++) {
        qsort(r[e], ROUNDS, sizeof(double), cmp_d);
        m[e] = (r[e][ROUNDS / 2 - 1] + r[e][ROUNDS / 2]) / 2;
    }
}

int main(int argc, char **argv)
{
    size_t G = 4096;
    double target_ms = 30;
    int joint = 0, argi = 1;
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    for (; argi < argc && argv[argi][0] == '-'; argi++) {
        if (!strncmp(argv[argi], "--G=", 4)) G = (size_t)atoi(argv[argi] + 4);
        else if (!strncmp(argv[argi], "--reps=", 7)) target_ms = atof(argv[argi] + 7);
        else if (!strcmp(argv[argi], "--joint")) joint = 1;
        else return fprintf(stderr, "unknown arg %s\n", argv[argi]);
    }
    if (G > MAXWIN) return fprintf(stderr, "G too large\n");
    printf("ab_wire3  G=%zu  joint=%s  (A = v6 larger-first, R = v6 right-first,"
           " B = v4; ratio>1 = v6 faster)\n", G, joint ? "coarse" : "off");
    printf("%-14s %8s | %7s %7s %7s %6s %6s | %7s %7s %7s %6s %6s | %8s %6s\n",
           "file", "windows",
           "decA", "decR", "decB", "dA/B", "dR/B",
           "encA", "encR", "encB", "eA/B", "eR/B",
           "v4bytes", "sv%");

    double gdA = 0, gdR = 0, geA = 0, geR = 0, gsv = 0;
    int nfiles = 0;
    for (; argi < argc; argi++) {
        FILE *fp = fopen(argv[argi], "rb");
        if (!fp) return fprintf(stderr, "can't open %s\n", argv[argi]);
        fseek(fp, 0, SEEK_END);
        size_t fsz = (size_t)ftell(fp);
        fseek(fp, 0, SEEK_SET);
        if (fsz > (size_t)48 << 20) fsz = (size_t)48 << 20;
        src = realloc(src, fsz);
        out = realloc(out, fsz);
        if (fread(src, 1, fsz, fp) != fsz) return 1;
        fclose(fp);

        nwin = (int)(fsz / G);                 /* drop the ragged tail */
        if (!nwin) continue;
        wins = realloc(wins, (size_t)nwin * sizeof *wins);
        tabA = realloc(tabA, (size_t)nwin * sizeof *tabA);
        tabR = realloc(tabR, (size_t)nwin * sizeof *tabR);
        tabB = realloc(tabB, (size_t)nwin * sizeof *tabB);
        encA = realloc(encA, (size_t)nwin * PIVCOH_ENCODE_BOUND(G));
        encR = realloc(encR, (size_t)nwin * PIVCOH_ENCODE_BOUND(G));
        encB = realloc(encB, (size_t)nwin * PIVCOH_ENCODE_BOUND(G));

        /* build + encode + gates (untimed) */
        size_t ca = 0, cr = 0, cb = 0;
        double sizA = 0, sizB = 0;
        for (int w = 0; w < nwin; w++) {
            wins[w].off = (uint32_t)(w * G);
            wins[w].n = (uint32_t)G;
            uint64_t freq[256];
            pivcoh_histogram(freq, src + wins[w].off, G);
            memset(&tabA[w], 0, sizeof tabA[w]);
            memset(&tabR[w], 0, sizeof tabR[w]);
            memset(&tabB[w], 0, sizeof tabB[w]);
            int ra, rr, rb;
            if (joint) {
                pivcoh_joint  ja = PIVCOH_JOINT_DEFAULTS;
                pivcohR_joint jr = PIVCOHR_JOINT_DEFAULTS;
                pivcohB_joint jb = PIVCOHB_JOINT_DEFAULTS;
                ja.gran = jr.gran = jb.gran = -1;
                ra = pivcoh_table_from_freqs_joint(&tabA[w], freq, &ja, jscratch);
                rr = pivcohR_table_from_freqs_joint(&tabR[w], freq, &jr, jscratch);
                rb = pivcohB_table_from_freqs_joint(&tabB[w], freq, &jb, jscratch);
            } else {
                ra = pivcoh_table_from_freqs(&tabA[w], freq);
                rr = pivcohR_table_from_freqs(&tabR[w], freq);
                rb = pivcohB_table_from_freqs(&tabB[w], freq);
            }
            if (!ra || !rr || !rb ||
                memcmp(tabA[w].code_len, tabR[w].code_len, 256) ||
                memcmp(tabA[w].code_len, tabB[w].code_len, 256))
                return fprintf(stderr, "table gate failed w%d\n", w);
            wins[w].eoffA = (uint32_t)ca;
            wins[w].eoffR = (uint32_t)cr;
            wins[w].eoffB = (uint32_t)cb;
            ptrdiff_t ea = pivcoh_encode(&tabA[w], src + wins[w].off, G,
                                         encA + ca, PIVCOH_ENCODE_BOUND(G),
                                         escratch);
            ptrdiff_t er = pivcohR_encode(&tabR[w], src + wins[w].off, G,
                                          encR + cr, PIVCOHR_ENCODE_BOUND(G),
                                          escratch);
            ptrdiff_t eb = pivcohB_encode(&tabB[w], src + wins[w].off, G,
                                          encB + cb, PIVCOHB_ENCODE_BOUND(G),
                                          escratch);
            /* R is byte-count-equal to A (order permutation) — except in
             * the DIAG_STORE_KR decomposition build, where the R slot
             * carries v4's headers and must match B instead. */
            if (ea < 0 || (er != ea && er != eb) || eb < ea)
                return fprintf(stderr, "size gate failed w%d (A %td R %td B %td)\n",
                               w, ea, er, eb);
            wins[w].elenA = (uint32_t)ea;
            wins[w].elenR = (uint32_t)er;
            wins[w].elenB = (uint32_t)eb;
            sizA += (double)ea;
            sizB += (double)eb;
            ca += PIVCOH_ENCODE_BOUND(G);      /* bound-strided: encode may
                                                  scribble to its bound */
            cr += PIVCOHR_ENCODE_BOUND(G);
            cb += PIVCOHB_ENCODE_BOUND(G);
        }
        memset(out, 0, fsz);
        sweep_decA();
        if (memcmp(out, src, (size_t)nwin * G))
            return fprintf(stderr, "A roundtrip failed\n");
        memset(out, 0, fsz);
        sweep_decR();
        if (memcmp(out, src, (size_t)nwin * G))
            return fprintf(stderr, "R roundtrip failed\n");
        memset(out, 0, fsz);
        sweep_decB();
        if (memcmp(out, src, (size_t)nwin * G))
            return fprintf(stderr, "B roundtrip failed\n");

        double md[3], me[3];
        void (*fd[3])(void) = { sweep_decA, sweep_decR, sweep_decB };
        void (*fe[3])(void) = { sweep_encA, sweep_encR, sweep_encB };
        ab_time3(fd, target_ms, md);
        ab_time3(fe, target_ms, me);
        double bytes = (double)nwin * (double)G;
        double sv = 100.0 * (sizB - sizA) / sizB;
        const char *base = strrchr(argv[argi], '/');
        printf("%-14s %8d | %7.0f %7.0f %7.0f %6.3f %6.3f |"
               " %7.0f %7.0f %7.0f %6.3f %6.3f | %8.0f %6.3f\n",
               base ? base + 1 : argv[argi], nwin,
               bytes / md[0] / 1e6, bytes / md[1] / 1e6, bytes / md[2] / 1e6,
               md[2] / md[0], md[2] / md[1],
               bytes / me[0] / 1e6, bytes / me[1] / 1e6, bytes / me[2] / 1e6,
               me[2] / me[0], me[2] / me[1],
               sizB, sv);
        gdA += log(md[2] / md[0]);
        gdR += log(md[2] / md[1]);
        geA += log(me[2] / me[0]);
        geR += log(me[2] / me[1]);
        gsv += sv;
        nfiles++;
    }
    if (nfiles)
        printf("geomean dec A/B %.3f  R/B %.3f | enc A/B %.3f  R/B %.3f |"
               " mean saving %.3f%%  (%d files)\n",
               exp(gdA / nfiles), exp(gdR / nfiles),
               exp(geA / nfiles), exp(geR / nfiles),
               gsv / nfiles, nfiles);
    return 0;
}

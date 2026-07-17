/* ab_wire — single-binary A/B of pivcoh wire v5 (A, extras/pivcoh.h)
 * vs wire v4 (B, sed-namespaced baseline) on fixed-G windows of real
 * literal streams.  One pivcoh block per window (G <= 32767).
 *
 * Per window both engines build tables from the same histogram (gated
 * identical code_len), encode their own ciphertext (gated equal SIZE —
 * v5 is a pure permutation of v4 — and round-trip).  Timing: >=100 ms
 * DVFS warmup, reps calibrated to >=30 ms per timed run, ROUNDS
 * interleaved rounds with the engine order alternating per round,
 * medians reported.  dec = pivcoh_decode with prebuilt tables (caller
 * scratch); enc = pivcoh_encode with prebuilt tables.
 *
 * Usage: ab_wire [--G=BYTES] [--reps=MS] [--joint] file...
 */
#define PIVCOH_IMPLEMENTATION
#include "pivcohA.h"
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
static uint8_t dscratchB[PIVCOHB_DECODE_SCRATCH_SIZE(MAXWIN)];
static uint8_t encout[PIVCOH_ENCODE_BOUND(MAXWIN)];
static uint8_t jscratch[PIVCOH_JOINT_SCRATCH_SIZE];

typedef struct {
    uint32_t off, n;            /* window bytes in src */
    uint32_t eoffA, elenA;      /* ciphertext in encA arena */
    uint32_t eoffB, elenB;
} win_t;

static uint8_t *src, *out, *encA, *encB;
static win_t *wins;
static pivcoh_table *tabA;
static pivcohB_table *tabB;
static int nwin;

static void sweep_decA(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcoh_decode(&tabA[w], encA + wins[w].eoffA, wins[w].elenA,
                          out + wins[w].off, wins[w].n, NULL, dscratchA) < 0)
            exit(fprintf(stderr, "A decode failed w%d\n", w));
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

static void sweep_encB(void)
{
    for (int w = 0; w < nwin; w++)
        if (pivcohB_encode(&tabB[w], src + wins[w].off, wins[w].n,
                           encout, sizeof encout, escratch) < 0)
            exit(fprintf(stderr, "B encode failed w%d\n", w));
}

/* one interleaved A/B measurement: returns median seconds per sweep */
static void ab_time(void (*fa)(void), void (*fb)(void), double target_ms,
                    double *ma, double *mb)
{
    double t0 = now_sec();                       /* DVFS ramp */
    do { fa(); fb(); } while (now_sec() - t0 < 0.1);
    double t1 = now_sec();
    fa();
    double per = now_sec() - t1;
    int reps = per > 0 ? (int)(target_ms / 1e3 / per) + 1 : 1;

    double ra[ROUNDS], rb[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        void (*first)(void)  = (r & 1) ? fb : fa;
        void (*second)(void) = (r & 1) ? fa : fb;
        double *tf = (r & 1) ? &rb[r] : &ra[r];
        double *ts = (r & 1) ? &ra[r] : &rb[r];
        double s = now_sec();
        for (int i = 0; i < reps; i++) first();
        *tf = (now_sec() - s) / reps;
        s = now_sec();
        for (int i = 0; i < reps; i++) second();
        *ts = (now_sec() - s) / reps;
    }
    qsort(ra, ROUNDS, sizeof(double), cmp_d);
    qsort(rb, ROUNDS, sizeof(double), cmp_d);
    *ma = (ra[ROUNDS / 2 - 1] + ra[ROUNDS / 2]) / 2;
    *mb = (rb[ROUNDS / 2 - 1] + rb[ROUNDS / 2]) / 2;
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
    printf("ab_wire  G=%zu  joint=%s  (A = wire v5, B = wire v4; ratio>1 = v5 faster)\n",
           G, joint ? "coarse" : "off");
    printf("%-14s %10s | %8s %8s %6s | %8s %8s %6s\n", "file", "windows",
           "decA", "decB", "d-A/B", "encA", "encB", "e-A/B");

    double gdec = 0, genc = 0;
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
        tabB = realloc(tabB, (size_t)nwin * sizeof *tabB);
        encA = realloc(encA, (size_t)nwin * PIVCOH_ENCODE_BOUND(G));
        encB = realloc(encB, (size_t)nwin * PIVCOH_ENCODE_BOUND(G));

        /* build + encode + gates (untimed) */
        size_t ca = 0, cb = 0;
        for (int w = 0; w < nwin; w++) {
            wins[w].off = (uint32_t)(w * G);
            wins[w].n = (uint32_t)G;
            uint64_t freq[256];
            pivcoh_histogram(freq, src + wins[w].off, G);
            memset(&tabA[w], 0, sizeof tabA[w]);
            memset(&tabB[w], 0, sizeof tabB[w]);
            int ra, rb;
            if (joint) {
                pivcoh_joint ja = PIVCOH_JOINT_DEFAULTS;
                pivcohB_joint jb = PIVCOHB_JOINT_DEFAULTS;
                ja.gran = jb.gran = -1;
                ra = pivcoh_table_from_freqs_joint(&tabA[w], freq, &ja, jscratch);
                rb = pivcohB_table_from_freqs_joint(&tabB[w], freq, &jb, jscratch);
            } else {
                ra = pivcoh_table_from_freqs(&tabA[w], freq);
                rb = pivcohB_table_from_freqs(&tabB[w], freq);
            }
            if (!ra || !rb || memcmp(tabA[w].code_len, tabB[w].code_len, 256))
                return fprintf(stderr, "table gate failed w%d\n", w);
            wins[w].eoffA = (uint32_t)ca;
            wins[w].eoffB = (uint32_t)cb;
            ptrdiff_t ea = pivcoh_encode(&tabA[w], src + wins[w].off, G,
                                         encA + ca, PIVCOH_ENCODE_BOUND(G),
                                         escratch);
            ptrdiff_t eb = pivcohB_encode(&tabB[w], src + wins[w].off, G,
                                          encB + cb, PIVCOHB_ENCODE_BOUND(G),
                                          escratch);
            if (ea < 0 || ea != eb)
                return fprintf(stderr, "size gate failed w%d (%td vs %td)\n",
                               w, ea, eb);
            wins[w].elenA = (uint32_t)ea;
            wins[w].elenB = (uint32_t)eb;
            ca += PIVCOH_ENCODE_BOUND(G);      /* bound-strided: encode may
                                                  scribble to its bound */
            cb += PIVCOHB_ENCODE_BOUND(G);
        }
        memset(out, 0, fsz);
        sweep_decA();
        if (memcmp(out, src, (size_t)nwin * G))
            return fprintf(stderr, "A roundtrip failed\n");
        memset(out, 0, fsz);
        sweep_decB();
        if (memcmp(out, src, (size_t)nwin * G))
            return fprintf(stderr, "B roundtrip failed\n");

        double da, db, ea, eb;
        ab_time(sweep_decA, sweep_decB, target_ms, &da, &db);
        ab_time(sweep_encA, sweep_encB, target_ms, &ea, &eb);
        double bytes = (double)nwin * (double)G;
        const char *base = strrchr(argv[argi], '/');
        printf("%-14s %10d | %8.0f %8.0f %5.3f | %8.0f %8.0f %5.3f\n",
               base ? base + 1 : argv[argi], nwin,
               bytes / da / 1e6, bytes / db / 1e6, db / da,
               bytes / ea / 1e6, bytes / eb / 1e6, eb / ea);
        gdec += log(db / da);
        genc += log(eb / ea);
        nfiles++;
    }
    if (nfiles)
        printf("geomean dec %.3f  enc %.3f  (%d files)\n",
               exp(gdec / nfiles), exp(genc / nfiles), nfiles);
    return 0;
}

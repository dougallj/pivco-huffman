/* bench_pivcoh_prof — decode-time decomposition for pivcoh on the zstd-lits
 * corpus, per table-lifetime SEGMENT (one tree rebuild + all its blocks),
 * matching zstd's real HUF-table cadence (.lithdr type==2 starts a segment).
 *
 * Reports, so kernel-tail ROI can be estimated:
 *   - segments, sub-blocks, tree-nodes (sched records = kernel calls) and
 *     their kind mix (flat / lone-leaf-cst / vec-vec merge);
 *   - decode-table rebuild time vs kernel time (rebuild excluded), and the
 *     resulting kernel-only GB/s;
 *   - ns and cycles per segment, and ns/cycles per kernel call.
 * BALANCED shape (coarse joint) by default; --simplest for plain lengths.
 * FSE is n/a (pivcoh is the raw-bitmap subset).  Pass --mhz=F for the cycle
 * columns (default 4400, M4 P-core; measure with the companion loop). */
#define PIVCOH_IMPLEMENTATION
#include "../pivcoh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLK 16384

static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }
static uint8_t *slurp(const char *path, size_t *n){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *b=malloc((size_t)sz+16);
    if(!b||fread(b,1,(size_t)sz,f)!=(size_t)sz){fclose(f);free(b);return NULL;}
    fclose(f); *n=(size_t)sz; return b; }

static const uint8_t *D;
static size_t N, NW;
static size_t *WSTART;
static uint8_t *ENC; static size_t *EOFF;      /* encoded stream + per-seg offsets */
static uint8_t (*LEN)[256];                     /* per-seg code lengths */
static uint8_t *DEC;
static uint8_t escr[PIVCOH_SCRATCH_SIZE(BLK)];
static uint8_t dscr[PIVCOH_DECODE_SCRATCH_SIZE(BLK)];
static uint8_t jscr[PIVCOH_JOINT_SCRATCH_SIZE];
static pivcoh_joint JP; static int SIMPLEST;
/* structural accumulators */
static double s_ranks, s_sched;                 /* summed over segments */
static uint64_t tot_sub, tot_kcalls, kind_cnt[4];  /* kind: flat/leafl/full per SUBBLOCK */

static size_t seglen(size_t w){ return WSTART[w+1]-WSTART[w]; }
static size_t nsub_of(size_t wlen){ return (wlen+BLK-1)/BLK; }

static int build_windows(const char *base, const char *path)
{
    size_t plen=strlen(path);
    char bp[4096]; snprintf(bp,sizeof bp,"%.*s.litblk",(int)(plen-5),path);
    char hp[4096]; snprintf(hp,sizeof hp,"%.*s.lithdr",(int)(plen-5),path);
    size_t bn,hn; uint8_t *b=slurp(bp,&bn), *h=slurp(hp,&hn);
    if(!b||!h||bn%4||hn!=bn/4){free(b);free(h);return fprintf(stderr,"%s: bad sidecars\n",base);}
    size_t m=bn/4;
    NW=0; for(size_t i=0;i<m;i++) if(i==0||h[i]==2) NW++;
    WSTART=malloc((NW+1)*sizeof(size_t));
    size_t off=0,w=0;
    for(size_t i=0;i<m;i++){
        uint32_t e=(uint32_t)b[4*i]|(uint32_t)b[4*i+1]<<8|(uint32_t)b[4*i+2]<<16|(uint32_t)b[4*i+3]<<24;
        if(i==0||h[i]==2) WSTART[w++]=off;
        off+=e;
    }
    WSTART[NW]=off; free(b); free(h);
    return off==N?0:fprintf(stderr,"%s: litblk sum %zu != %zu\n",base,off,N);
}

/* build the per-segment tables + encoded stream; tally the schedules */
static int setup(const char *base)
{
    EOFF=malloc((NW+1)*sizeof(size_t)); EOFF[0]=0;
    LEN=malloc(NW*256); ENC=malloc(2*N+NW*512+65536);
    size_t off=0;
    for(size_t w=0;w<NW;w++){
        const uint8_t *p=D+WSTART[w]; size_t wlen=seglen(w);
        uint64_t f[256]={0}; for(size_t i=0;i<wlen;i++) f[p[i]]++;
        pivcoh_table t;
        int ok = SIMPLEST ? pivcoh_table_from_freqs(&t,f)
                          : pivcoh_table_from_freqs_joint(&t,f,&JP,jscr);
        if(!ok) return fprintf(stderr,"%s: build failed w=%zu\n",base,w);
        memcpy(LEN[w],t.code_len,256);
        s_ranks+=t.num_ranks; s_sched+=t.sched_len;
        size_t nsub=nsub_of(wlen); tot_sub+=nsub;
        for(int r=0;r<t.sched_len;r++) kind_cnt[t.sched[r].kd & 3]+=nsub;
        tot_kcalls += (uint64_t)t.sched_len * nsub;
        for(size_t b=0;b<wlen;b+=BLK){
            size_t bn=wlen-b<BLK?wlen-b:BLK;
            ptrdiff_t r=pivcoh_encode(&t,p+b,bn,ENC+off,PIVCOH_ENCODE_BOUND(bn),escr);
            if(r<0) return fprintf(stderr,"%s: encode failed\n",base);
            off+=(size_t)r;
        }
        EOFF[w+1]=off;
    }
    /* gate */
    memset(DEC,0,N);
    for(size_t w=0;w<NW;w++){
        pivcoh_table t; pivcoh_table_from_lens(&t,LEN[w]);
        size_t o=EOFF[w],dof=0,wlen=seglen(w);
        while(dof<wlen){ size_t c; ptrdiff_t dn=pivcoh_decode(&t,ENC+o,EOFF[w+1]-o,DEC+WSTART[w]+dof,wlen-dof,&c,dscr);
            if(dn<0) return fprintf(stderr,"%s: decode failed\n",base); o+=c; dof+=(size_t)dn; }
    }
    return memcmp(DEC,D,N)? fprintf(stderr,"%s: mismatch\n",base):0;
}

static volatile unsigned g_sink;
static void pass_dbuild(void){ for(size_t w=0;w<NW;w++){ pivcoh_table t; pivcoh_table_from_lens(&t,LEN[w]); g_sink += t.num_ranks + t.sched_len; } }
static void pass_dec(void){
    for(size_t w=0;w<NW;w++){
        pivcoh_table t; pivcoh_table_from_lens(&t,LEN[w]);
        size_t o=EOFF[w],dof=0,wlen=seglen(w);
        while(dof<wlen){ size_t c; ptrdiff_t dn=pivcoh_decode(&t,ENC+o,EOFF[w+1]-o,DEC+WSTART[w]+dof,wlen-dof,&c,dscr);
            o+=c; dof+=(size_t)dn; }
    }
}
static double timeit(void(*fn)(void),int reps){
    int inner=(int)(1+((size_t)8<<20)/(N?N:1)); double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now_sec(); for(int i=0;i<inner;i++) fn(); double dt=(now_sec()-t0)/inner; if(dt<best) best=dt; }
    return best;
}

int main(int argc,char**argv){
    int reps=20, argi=1; double mhz=4400.0;
    for(;argi<argc && !strncmp(argv[argi],"--",2);argi++){
        if(!strncmp(argv[argi],"--reps=",7)) reps=atoi(argv[argi]+7);
        else if(!strncmp(argv[argi],"--mhz=",6)) mhz=atof(argv[argi]+6);
        else if(!strcmp(argv[argi],"--simplest")) SIMPLEST=1;
        else { fprintf(stderr,"usage: %s [--reps=N] [--mhz=F] [--simplest] file.lits...\n",argv[0]); return 1; }
    }
    pivcoh_joint jp=PIVCOH_JOINT_DEFAULTS; JP=jp; JP.gran=-1;   /* coarse=BALANCED */

    double tot_dec=0, tot_bld=0; size_t segs=0, symbols=0; int nf=0;
    for(int ai=argi;ai<argc;ai++){
        uint8_t *data=slurp(argv[ai],&N); if(!data){fprintf(stderr,"skip %s\n",argv[ai]);continue;}
        D=data; const char *base=strrchr(argv[ai],'/'); base=base?base+1:argv[ai];
        if(build_windows(base,argv[ai])!=0) return 1;
        DEC=malloc(N+64);
        if(setup(base)!=0) return 1;
        double w0=now_sec(); do{ pass_dec(); }while(now_sec()-w0<0.1);   /* warmup */
        double dec=timeit(pass_dec,reps), bld=timeit(pass_dbuild,reps);
        tot_dec+=dec; tot_bld+=bld; segs+=NW; symbols+=N; nf++;
        free(data); free(DEC); free(WSTART); free(EOFF); free(LEN); free(ENC);
    }
    double kern = tot_dec - tot_bld;                         /* decode excl. rebuild */
    double ns_seg = tot_dec/segs*1e9, ns_bld=tot_bld/segs*1e9, ns_kern=kern/segs*1e9;
    double ns_kcall = kern/(double)tot_kcalls*1e9;
    printf("== pivcoh decode profile (%s, table-lifetime segments, %d files) ==\n",
           SIMPLEST?"SIMPLEST":"BALANCED", nf);
    printf("segments (tree rebuilds) : %zu\n", segs);
    printf("symbols total            : %zu (%.2f MB)\n", symbols, symbols/1e6);
    printf("symbols / segment        : %.0f\n", (double)symbols/segs);
    printf("sub-blocks (%dB) total    : %llu  (%.2f / segment)\n", BLK,(unsigned long long)tot_sub,(double)tot_sub/segs);
    printf("tree nodes = sched_len   : %.2f / sub-block (avg)\n", s_sched/segs);
    printf("num_ranks (leaves)       : %.2f / segment (avg)\n", s_ranks/segs);
    printf("kernel calls total       : %llu  (%.1f / segment)\n",(unsigned long long)tot_kcalls,(double)tot_kcalls/segs);
    /* kd&3: 0=FULL vec-vec merge, 1=FLAT, 3=LEAF_LEFT (lone-leaf cst) */
    printf("  kind mix  flat=%.1f%%  cst(lone-leaf)=%.1f%%  vec-vec merge=%.1f%%\n",
           100.0*kind_cnt[1]/tot_kcalls, 100.0*kind_cnt[3]/tot_kcalls, 100.0*kind_cnt[0]/tot_kcalls);
    printf("  (kd&3 raw: 0=%llu 1=%llu 2=%llu 3=%llu)\n",
           (unsigned long long)kind_cnt[0],(unsigned long long)kind_cnt[1],(unsigned long long)kind_cnt[2],(unsigned long long)kind_cnt[3]);
    printf("--- timing (best of %d, %.0f MHz) ---\n", reps, mhz);
    printf("decode e2e (build+kern)  : %.2f GB/s | %.0f ns/seg | %.0f cyc/seg\n",
           symbols/tot_dec/1e9, ns_seg, ns_seg*mhz/1000.0);
    printf("  table rebuild alone    : %.2f%% of e2e | %.0f ns/seg | %.0f cyc/seg\n",
           100.0*tot_bld/tot_dec, ns_bld, ns_bld*mhz/1000.0);
    printf("KERNELS only (excl bld)  : %.2f GB/s | %.0f ns/seg | %.0f cyc/seg\n",
           symbols/kern/1e9, ns_kern, ns_kern*mhz/1000.0);
    printf("  per kernel call        : %.2f ns | %.1f cyc\n", ns_kcall, ns_kcall*mhz/1000.0);
    return 0;
}

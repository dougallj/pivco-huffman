/* pivcoh.h - v4.0 - single-file PIVCO-Huffman block codec
 *
 * Based on https://github.com/MarcinZukowski/pivco-huffman
 *
 * AArch64/NEON-only.
 *
 * v4 WIRE: this codec's stream format diverges from the production
 * pivco-huffman wire (v0.7) to stop paying its small-chunk taxes:
 * per-node FSE marker bytes are gone (there is no FSE mode to mark),
 * K_right headers shrink to one byte wherever the node's symbol count
 * fits (the decoder always knows it), code lengths travel as a
 * run-length nibble wire (typically 30..60 bytes instead of a flat
 * 128), and the one-shot frame uses a varint size plus raw-store
 * segment escapes for incompressible spans.  v3.x streams (production
 * wire) are NOT decodable by v4 or vice versa.
 *
 * Do this in ONE C file to create the implementation:
 *     #define PIVCOH_IMPLEMENTATION
 *     #include "pivcoh.h"
 *
 * Usage — one-shot (whole buffer, self-describing frame; see the
 * frame API section for the format and the effort knob):
 *     unsigned char *dst = malloc(PIVCOH_COMPRESS_BOUND(n));
 *     ptrdiff_t c = pivcoh_compress(dst, PIVCOH_COMPRESS_BOUND(n),
 *                                   src, n, PIVCOH_BALANCED, NULL);
 *     ...
 *     size_t raw;
 *     unsigned char *out = pivcoh_decompress_malloc(dst, (size_t)c, &raw);
 *     // or size the buffer yourself: pivcoh_decompressed_size + pivcoh_decompress
 *
 * Usage — encoder side:
 *     pivcoh_table t;
 *     if (!pivcoh_table_from_freqs(&t, freq)) ...;   // freq: uint64_t[256]
 *     // or _joint(&t, freq, &j, NULL) to trade a guarded sliver of
 *     // compressed size for a much faster-to-decode tree shape
 *     // transmit t.code_len (256 values, all <= 11: nibble-packable)
 *     unsigned char out[PIVCOH_ENCODE_BOUND(4096)];
 *     ptrdiff_t len = pivcoh_encode(&t, in, n, out, sizeof out, NULL);
 *
 * Usage — decoder side:
 *     pivcoh_table t;
 *     if (!pivcoh_table_from_lens(&t, code_len)) ...;  // validates lengths
 *     ptrdiff_t n = pivcoh_decode(&t, buf, buf_len, sym, sym_cap, NULL, NULL);
 *
 * One block covers 1..65535 symbols (bytes).  Blocks are self-delimiting
 * (a 2-byte symbol-count header leads the stream), so blocks can be
 * concatenated; pivcoh_decode reports the consumed byte count.
 *
 * The trailing `scratch` parameter of encode/decode may be NULL (malloc
 * is used internally) or a caller buffer for allocation-free operation:
 * PIVCOH_SCRATCH_SIZE(max_n) bytes always suffices, and decode-only
 * callers can pass the smaller PIVCOH_DECODE_SCRATCH_SIZE(max_n).
 * Tables and scratch are plain memory: no cleanup calls, safe to copy,
 * const tables are shareable across threads (decode never writes them;
 * the FIRST encode on a table completes its encoder view in place with
 * idempotent writes, so concurrent first encodes are also benign).
 *
 * Encoding a symbol whose frequency/length was zero produces a valid but
 * meaningless stream (never memory-unsafe).  Encode's SIMD packers may
 * scribble up to 16 junk bytes past the returned length — always inside
 * out_cap (>= PIVCOH_ENCODE_BOUND(n) is required).  Decode is safe on
 * hostile input: it never reads past `in + in_len`, never writes past N
 * symbols, and returns -1 on any malformed stream — truncation,
 * structural errors, and bitmaps whose popcount contradicts their
 * K_right header (each merge checks its final cursor position against
 * the header; monotone cursors make that one compare an exact
 * full-bitmap validation).
 *
 * License: Apache-2.0, same as the pivco-huffman repository.
 */
#ifndef PIVCOH_H
#define PIVCOH_H

#include <stddef.h>
#include <stdint.h>

#ifdef PIVCOH_STATIC
#define PIVCOHDEF static
#else
#define PIVCOHDEF extern
#endif

/* Worst-case encoded size of one n-symbol block (payload is at most 11
 * bits/symbol plus per-node headers and byte rounding). */
#define PIVCOH_ENCODE_BOUND(n)  ((11 * (size_t)(n) + 7) / 8 + 1024)

/* Scratch bytes for encode or decode of blocks up to n symbols.  The
 * encoder dominates: a ranks buffer (n + 64) plus, per recursion level
 * (max 11), one staged bitmap, one compacted right-half and a 64-byte
 * overshoot gap (~9n/8 + 73 per level) for the tail-free partition's
 * scatter, which strays up to 63 bytes past a ranks region. */
#define PIVCOH_SCRATCH_SIZE(n)  (14 * (size_t)(n) + 1024)

/* Scratch bytes for DECODE ONLY of blocks up to n symbols.  The decode
 * walk ping-pongs children through (out, partner) buffer pairs instead
 * of growing an arena, so a valid stream touches at most 1.5n + 128
 * bytes (the root's children plus the largest partner).  The other
 * 0.5n is hostile-input headroom: a bitmap that lies about its K_right
 * can walk a merge cursor up to K bytes past its side before the
 * end-of-merge check rejects the stream, and keeping that in-bounds by
 * padding is free where in-loop cursor guards would cost ~2% decode
 * speed.  PIVCOH_SCRATCH_SIZE also always suffices. */
#define PIVCOH_DECODE_SCRATCH_SIZE(n)  (2 * (size_t)(n) + 128)

/* ---- optional joint length/shape optimization (encoder side) ----
 *
 * pivcoh_table_from_freqs picks lengths that minimize compressed bits.
 * pivcoh_table_from_freqs_joint additionally bends them — at an
 * explicitly priced, guard-bounded cost in bits — so the class counts
 * land on round binary numbers and the decoder's counts-to-chunks rule
 * yields fewer, larger flat blocks and fewer merge passes.  Measured on
 * windowed LZ-literal workloads this buys +25%..+130% decode speed at
 * a compression delta within ±0.3 pp (better at small windows).  The
 * wire carries only lengths, so ANY decoder reads the output.
 * Port of the production joint optimizer (joint-cost-model @ c6073f5). */
typedef struct {
    float lambda;      /* bits one merge pass is worth; <= 0 disables the
                          pass entirely (plain Huffman lengths) */
    int   gran;        /* solve tier: 0 auto (exact DP to 64 symbols, then
                          grouped; always ~<= 10 us), 1 exact DP (~100 us
                          worst case), 2/4/8 fixed grouping, -1 coarse auto
                          (one grouping step chunkier, ~2-4 us: most of
                          auto's decode win at a fraction of its encode
                          cost -- it replaced a greedy nudger that the
                          coarse DP dominated once the guard's near-
                          incompressible waiver landed).  Other values
                          behave as 0. */
    float guard_bits;  /* adopt only if modeled bits <= guard_bits * baseline */
    float guard_time;  /* ... and modeled decode time <= guard_time * baseline;
                          otherwise the plain Huffman lengths are kept */
    float gamma;       /* fixed decode cost per schedule record per block, in
                          merge element-pass units (dispatch + wire header).
                          Blocks are modeled at 16K symbols; gamma and block
                          size enter the cost only as gamma/block, so scale
                          gamma if your decode granularity differs */
    float kappa[9];    /* flat-kernel decode cost per symbol at depth b, in
                          merge-pass units; all-zero models kernels free */
    float mu_cst;      /* lone-leaf merge cost relative to a full merge */
    float prefill;     /* fraction of the prefilled top-symbol leaf its
                          parent merge skips (pivcoh does not prefill: 0) */
} pivcoh_joint;
#define PIVCOH_JOINT_DEFAULTS \
    { 0.1f, 0, 1.015f, 0.90f, 170.0f, {0}, 1.0f, 0.0f }

/* Scratch for the joint DP tiers: the exact solve at a full 256-symbol
 * alphabet needs a (257 diagonals x 132 cells) plane of f32 costs plus
 * 11 u16 backtrack planes (+8 alignment).  Tiers gran 0/-1 use < 64 KiB
 * of this.  As with encode/decode, scratch may be NULL (malloc). */
#define PIVCOH_JOINT_SCRATCH_SIZE  (257 * 132 * 26 + 8)

typedef struct { uint8_t kd, param, right; } pivcoh__rec;

typedef struct {
    uint8_t code_len[256];   /* per-symbol code length, 0 = absent, max 11.
                                Filled by pivcoh_table_from_freqs; this is
                                what the encoder transmits to the decoder. */
    /* internals */
    uint16_t num_ranks, sched_len;
    uint8_t enc_ready;       /* encoder view valid (filled lazily on first
                                encode; decode never needs it) */
    uint8_t enc_umin;        /* used-symbol window: min symbol and span-1;
                                a span < 128 halves enc_init's lookup */
    uint8_t enc_span1;
    uint8_t rank_to_sym[256], sym_to_rank[256];
    pivcoh__rec sched[60];   /* Kraft-complete max is 59 records (33 chunks,
                                27 with bit >= 1); +1 so the schedule render's
                                cap check can't fire mid-walk on a maximal
                                table */
} pivcoh_table;

/* Build a table from symbol frequencies (encoder side).  Derives optimal
 * length-limited code lengths into t->code_len.  Frequencies are treated
 * mod 2^32 internally: a histogram totalling >= 4 GiB may derive
 * different — still valid — lengths.  Returns 1, or 0 if every
 * frequency is zero. */
PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256]);

/* As pivcoh_table_from_freqs, plus the joint length/shape pass when
 * j && j->lambda > 0 (start from PIVCOH_JOINT_DEFAULTS).  On any
 * internal reject — the adoption guard, an out-of-contract lambda, or
 * malloc failure with scratch == NULL — the plain Huffman lengths are
 * kept, so the return value means exactly what it does above. */
PIVCOHDEF int pivcoh_table_from_freqs_joint(pivcoh_table *t,
                                            const uint64_t freq[256],
                                            const pivcoh_joint *j,
                                            void *scratch);

/* Build a table from received code lengths (decoder side).  Returns 1, or
 * 0 on invalid lengths (any > 11, all zero, or not Kraft-complete).  Both
 * sides build identical tables from identical lengths. */
PIVCOHDEF int pivcoh_table_from_lens(pivcoh_table *t, const uint8_t code_len[256]);

/* Encode n symbols (1..65535) into out.  Requires out_cap >=
 * PIVCOH_ENCODE_BOUND(n).  Returns the encoded byte count, or -1 on bad
 * arguments / malloc failure. */
PIVCOHDEF ptrdiff_t pivcoh_encode(const pivcoh_table *t,
                                  const uint8_t *in, size_t n,
                                  uint8_t *out, size_t out_cap, void *scratch);

/* Decode one block from in[0..in_len).  Writes the block's N symbols to
 * out (fails if N > out_cap).  Returns N, or -1 on malformed/truncated
 * input or malloc failure.  If consumed is non-NULL it receives the
 * block's byte length (for concatenated blocks). */
PIVCOHDEF ptrdiff_t pivcoh_decode(const pivcoh_table *t,
                                  const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap,
                                  size_t *consumed, void *scratch);

/* ---- one-shot frame API ----
 *
 * FRAME = [varint decompressed_size]     (LEB128, 1..10 bytes)
 *         [code-lengths wire]            (pivcoh_lens_wire_*, <= 129 B)
 *         [SEG...]                       (decompressed_size = 0: no
 *                                         lens, no segments)
 * SEG:  the leading u16 (LE) is the discriminator.
 *         high bit set:   raw store — (u16 & 0x7fff) verbatim bytes
 *                         follow (1..32767)
 *         high bit clear: one coded pivcoh block, starting with that
 *                         same u16 as its symbol count (1..32767);
 *                         self-delimiting via decode
 *
 * One table serves the whole frame (for windowed retraining use the
 * low-level API).  This encoder writes <= 32767-symbol segments (the
 * measured decode-throughput plateau on Apple Silicon is 24..40 K) and
 * stores a segment raw when coding would not shrink it — the huf0-
 * style incompressible fallback.  Segments are not length-prefixed:
 * v4 trades v3's skippable u32-framed blocks for 4 fewer bytes per
 * segment; streaming readers must decode to find boundaries. */

/* Compression effort: how much table-build time pivcoh_compress spends
 * shaping the code for DECOMPRESSION speed (the joint pass below).
 * More shaping: slower compression, faster decompression, ~same size (a
 * guard bounds the growth).  The cost is per compress CALL — one table
 * covers the frame — so it only matters for small inputs or high call
 * rates.  The superlatives are the extremes; most callers want the
 * middle. */
typedef enum {
    PIVCOH_SIMPLEST_COMPRESS  = 0,  /* plain Huffman lengths: never any
                                       shaping time, but decompression
                                       leaves 25..50% speed unclaimed */
    PIVCOH_BALANCED           = 1,  /* the default: ~2-4 us of shaping
                                       buys most of the decompress win */
    PIVCOH_FASTER_DECOMPRESS  = 2,  /* <= ~10 us: nearly all of the win */
    PIVCOH_FASTEST_DECOMPRESS = 3,  /* ~100 us, provably optimal shape:
                                       encode-once-decode-forever data */
    PIVCOH_FASTEST_COMPRESS   = 4,  /* SIMPLEST below 256 KiB of input,
                                       BALANCED above: past that point a
                                       flatter tree ENCODES faster than
                                       the shaping solve costs (+10..30%
                                       measured; and when shaping can't
                                       help, its <3% cost keeps
                                       shrinking as 1/n) */
} pivcoh_effort;

/* Worst-case frame size (frame header + per-segment headers/rounding;
 * the per-segment 1032 also covers the last block's ENCODE_BOUND slack
 * so pivcoh_encode can target the frame buffer directly). */
#define PIVCOH_COMPRESS_BOUND(n) \
    (140 + (11 * (size_t)(n) + 7) / 8 + ((size_t)(n) / 32767 + 1) * 1032)

/* Worst-case bytes of the code-lengths wire (mode byte + 128). */
#define PIVCOH_LENS_WIRE_BOUND 129

/* Fixed scratch sizes for the frame API (blocks cap at 32768 symbols,
 * so these are input-size-independent). */
#define PIVCOH_COMPRESS_SCRATCH_SIZE \
    (PIVCOH_SCRATCH_SIZE(32768) + PIVCOH_JOINT_SCRATCH_SIZE)
#define PIVCOH_DECOMPRESS_SCRATCH_SIZE  PIVCOH_DECODE_SCRATCH_SIZE(65535)

/* Compress src[0..n) into a self-describing frame.  Requires dst_cap >=
 * PIVCOH_COMPRESS_BOUND(n).  Returns the frame's byte length, or -1 on
 * bad arguments / malloc failure.  scratch: NULL (malloc) or
 * PIVCOH_COMPRESS_SCRATCH_SIZE bytes. */
PIVCOHDEF ptrdiff_t pivcoh_compress(uint8_t *dst, size_t dst_cap,
                                    const uint8_t *src, size_t n,
                                    pivcoh_effort effort, void *scratch);

/* Full-control form: shaping configured by a pivcoh_joint (NULL = no
 * shaping, exactly PIVCOH_FASTEST_COMPRESS). */
PIVCOHDEF ptrdiff_t pivcoh_compress_joint(uint8_t *dst, size_t dst_cap,
                                          const uint8_t *src, size_t n,
                                          const pivcoh_joint *j,
                                          void *scratch);

/* Decompress a whole frame.  Returns the byte count written to dst, or
 * -1 on malformed input, dst_cap too small, or malloc failure.  Strict:
 * the frame must parse exactly to its declared decompressed size with
 * no trailing bytes.  scratch: NULL or PIVCOH_DECOMPRESS_SCRATCH_SIZE. */
PIVCOHDEF ptrdiff_t pivcoh_decompress(uint8_t *dst, size_t dst_cap,
                                      const uint8_t *src, size_t n,
                                      void *scratch);

/* Read a frame's declared decompressed size (for sizing dst).
 * Returns UINT64_MAX if src is too short to hold a frame header. */
PIVCOHDEF uint64_t pivcoh_decompressed_size(const uint8_t *src, size_t n);

/* pivcoh_decompress into a malloc'd buffer of exactly the frame's
 * decompressed size (an empty frame yields a freeable 1-byte buffer).
 * Returns the buffer, or NULL on malformed input / malloc failure; on
 * success *size receives the byte count (size may be NULL). */
PIVCOHDEF uint8_t *pivcoh_decompress_malloc(const uint8_t *src, size_t n,
                                            size_t *size);

/* ---- utilities (also used by the frame API) ---- */

/* Byte histogram, exact for any n.  Runs 4 interleaved count tables to
 * break the store-to-load chains that stall the naive loop on repeated
 * bytes (1.6x on typical data, ~4x on skewed). */
PIVCOHDEF void pivcoh_histogram(uint64_t freq[256], const uint8_t *p, size_t n);

/* Code lengths <-> the raw 128-byte nibble packing (pivcohuf layout;
 * also lens-wire mode 0's body). */
PIVCOHDEF void pivcoh_lens_pack(uint8_t packed[128], const uint8_t code_len[256]);
PIVCOHDEF void pivcoh_lens_unpack(uint8_t code_len[256], const uint8_t packed[128]);

/* Compact code-lengths wire (the frame's table header; usable
 * standalone for windowed formats).  _write emits into dst (capacity
 * >= PIVCOH_LENS_WIRE_BOUND) and returns the byte count; _read parses,
 * fills code_len[256], and returns the bytes consumed, or -1 on a
 * malformed/truncated wire. */
PIVCOHDEF int pivcoh_lens_wire_write(uint8_t *dst, const uint8_t code_len[256]);
PIVCOHDEF int pivcoh_lens_wire_read(uint8_t code_len[256],
                                    const uint8_t *src, size_t n);

/* pivcoh_table_from_lens on nibble-packed lengths (returns 0 on invalid
 * lengths, exactly like it). */
PIVCOHDEF int pivcoh_table_from_packed_lens(pivcoh_table *t,
                                            const uint8_t packed[128]);

#endif /* PIVCOH_H */

#ifdef PIVCOH_IMPLEMENTATION

#if !defined(__aarch64__)
#error "pivcoh.h v2.x is the NEON edition and requires aarch64 (the scalar codec lives on the main branch)"
#endif

#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PIVCOH__MAXLEN 11
enum { PIVCOH__FULL = 0, PIVCOH__FLAT = 1, PIVCOH__LEAFL = 3 };

/* ---- table build ---- */

typedef struct { uint8_t depth, bit, sym_idx; } pivcoh__chunk;

/* Pre-order schedule from the depth-sorted chunk list: chunks are the
 * tree's left-to-right leaves, and the leaf-depth sequence determines
 * the tree.  Iterative, with an explicit stack of open internal nodes
 * (port of the production build_schedule; the recursion this replaces
 * was the dominant per-window table-build cost on ragged deep trees).
 * The walk doubles as Kraft-completeness validation.  Returns 0, or -1
 * on non-Kraft-complete lengths. */
typedef struct { int my, rank0, mid_sched, mid_rank, state; } pivcoh__frame;

static int pivcoh__sched(pivcoh_table *t, const pivcoh__chunk *ch, int nch,
                         const uint8_t *items)
{
    pivcoh__frame stk[PIVCOH__MAXLEN + 1];
    const int cap = (int)(sizeof t->sched / sizeof *t->sched);
    int sp = 0, ci = 0, rank = 0;

    for (;;) {
        if (ci >= nch) return -1;              /* under-subscribed lengths */

        /* Descend the left spine until a chunk sits at this depth. */
        while (ch[ci].depth != sp) {
            if (sp > PIVCOH__MAXLEN || t->sched_len >= cap) return -1;
            pivcoh__frame *f = &stk[sp++];
            f->my    = t->sched_len++;
            f->rank0 = rank;
            f->state = 0;
        }

        /* Consume the chunk-leaf.  Width-1/2 chunks dominate skewed
         * alphabets; keep their copies inline (a variable-size memcpy
         * is a libc dispatch per chunk). */
        {
            const pivcoh__chunk *c = &ch[ci++];
            int b = c->bit, r0 = rank;
            uint8_t *dst = t->rank_to_sym + r0;
            const uint8_t *s = items + c->sym_idx;
            if (b == 0)      dst[0] = s[0];
            else if (b == 1) { dst[0] = s[0]; dst[1] = s[1]; }
            else             memcpy(dst, s, (size_t)1 << b);
            rank += 1 << b;
            if (b != 0) {
                if (t->sched_len >= cap) return -1;
                pivcoh__rec *r = &t->sched[t->sched_len++];
                r->kd    = (uint8_t)(PIVCOH__FLAT | b << 2);
                r->param = (uint8_t)r0;
                r->right = 0;
            }
        }

        /* Ascend, completing parents whose right child just finished. */
        for (;;) {
            if (sp == 0) {                     /* root subtree complete */
                if (ci != nch) return -1;      /* over-subscribed */
                t->num_ranks = (uint16_t)rank;
                return 0;
            }
            pivcoh__frame *f = &stk[sp - 1];
            if (f->state == 0) {               /* left done; do the right */
                f->state     = 1;
                f->mid_sched = t->sched_len;
                f->mid_rank  = rank;
                break;
            }
            /* Right done: finalize this internal node's record.  Only the
             * LEFT child can be a lone leaf.  A lone right child would be a
             * b=0 (singleton) chunk still unconsumed after the left subtree,
             * but the depth-sorted list keeps equal-depth chunks
             * singleton-first (the sort is STABLE, classes generate in
             * ascending L, and depth d's only possible singleton is length
             * class d's b=0 chunk), so that singleton is always consumed as
             * the left child instead.  Corrupt lengths only perturb the
             * counts, not this ordering -- so a lone right child cannot occur
             * (verified: 0 hits over 85M valid + malformed length vectors,
             * incl. exhaustive small shapes).  NB the production builder's
             * naive research-tree mode makes every symbol a singleton and can
             * reach both; pivcoh's power-of-two chunking never does. */
            int left_lone = f->mid_sched == f->my + 1 &&
                            f->mid_rank == f->rank0 + 1;
            pivcoh__rec *r = &t->sched[f->my];
            r->kd    = (uint8_t)(left_lone ? PIVCOH__LEAFL : PIVCOH__FULL);
            r->param = (uint8_t)(f->mid_rank - 1); /* thr / rank_begin */
            r->right = (uint8_t)(f->mid_sched - f->my);
            sp--;
        }
    }
}

/* select8[m][j] = position (0..7) of the (j+1)-th set bit of m, 0xff
 * past the popcount.  Used as a TBL index into a group's [base..base+7]
 * symbol ids so a length class's members scatter into items[] in one
 * store per non-empty byte — replacing the per-symbol ctz loop, whose
 * data-dependent branch is unpredictable.  Compile-time (2 KiB, cold) so
 * the decode table build never triggers the lazy encoder init; the
 * packed 8-byte rows are also denser in L1 than the encoder's 16-byte
 * ctab8 for this hot loop. */
static const uint8_t pivcoh__select8[256][8] = {
    {255,255,255,255,255,255,255,255}, {0,255,255,255,255,255,255,255}, {1,255,255,255,255,255,255,255}, {0,1,255,255,255,255,255,255},
    {2,255,255,255,255,255,255,255}, {0,2,255,255,255,255,255,255}, {1,2,255,255,255,255,255,255}, {0,1,2,255,255,255,255,255},
    {3,255,255,255,255,255,255,255}, {0,3,255,255,255,255,255,255}, {1,3,255,255,255,255,255,255}, {0,1,3,255,255,255,255,255},
    {2,3,255,255,255,255,255,255}, {0,2,3,255,255,255,255,255}, {1,2,3,255,255,255,255,255}, {0,1,2,3,255,255,255,255},
    {4,255,255,255,255,255,255,255}, {0,4,255,255,255,255,255,255}, {1,4,255,255,255,255,255,255}, {0,1,4,255,255,255,255,255},
    {2,4,255,255,255,255,255,255}, {0,2,4,255,255,255,255,255}, {1,2,4,255,255,255,255,255}, {0,1,2,4,255,255,255,255},
    {3,4,255,255,255,255,255,255}, {0,3,4,255,255,255,255,255}, {1,3,4,255,255,255,255,255}, {0,1,3,4,255,255,255,255},
    {2,3,4,255,255,255,255,255}, {0,2,3,4,255,255,255,255}, {1,2,3,4,255,255,255,255}, {0,1,2,3,4,255,255,255},
    {5,255,255,255,255,255,255,255}, {0,5,255,255,255,255,255,255}, {1,5,255,255,255,255,255,255}, {0,1,5,255,255,255,255,255},
    {2,5,255,255,255,255,255,255}, {0,2,5,255,255,255,255,255}, {1,2,5,255,255,255,255,255}, {0,1,2,5,255,255,255,255},
    {3,5,255,255,255,255,255,255}, {0,3,5,255,255,255,255,255}, {1,3,5,255,255,255,255,255}, {0,1,3,5,255,255,255,255},
    {2,3,5,255,255,255,255,255}, {0,2,3,5,255,255,255,255}, {1,2,3,5,255,255,255,255}, {0,1,2,3,5,255,255,255},
    {4,5,255,255,255,255,255,255}, {0,4,5,255,255,255,255,255}, {1,4,5,255,255,255,255,255}, {0,1,4,5,255,255,255,255},
    {2,4,5,255,255,255,255,255}, {0,2,4,5,255,255,255,255}, {1,2,4,5,255,255,255,255}, {0,1,2,4,5,255,255,255},
    {3,4,5,255,255,255,255,255}, {0,3,4,5,255,255,255,255}, {1,3,4,5,255,255,255,255}, {0,1,3,4,5,255,255,255},
    {2,3,4,5,255,255,255,255}, {0,2,3,4,5,255,255,255}, {1,2,3,4,5,255,255,255}, {0,1,2,3,4,5,255,255},
    {6,255,255,255,255,255,255,255}, {0,6,255,255,255,255,255,255}, {1,6,255,255,255,255,255,255}, {0,1,6,255,255,255,255,255},
    {2,6,255,255,255,255,255,255}, {0,2,6,255,255,255,255,255}, {1,2,6,255,255,255,255,255}, {0,1,2,6,255,255,255,255},
    {3,6,255,255,255,255,255,255}, {0,3,6,255,255,255,255,255}, {1,3,6,255,255,255,255,255}, {0,1,3,6,255,255,255,255},
    {2,3,6,255,255,255,255,255}, {0,2,3,6,255,255,255,255}, {1,2,3,6,255,255,255,255}, {0,1,2,3,6,255,255,255},
    {4,6,255,255,255,255,255,255}, {0,4,6,255,255,255,255,255}, {1,4,6,255,255,255,255,255}, {0,1,4,6,255,255,255,255},
    {2,4,6,255,255,255,255,255}, {0,2,4,6,255,255,255,255}, {1,2,4,6,255,255,255,255}, {0,1,2,4,6,255,255,255},
    {3,4,6,255,255,255,255,255}, {0,3,4,6,255,255,255,255}, {1,3,4,6,255,255,255,255}, {0,1,3,4,6,255,255,255},
    {2,3,4,6,255,255,255,255}, {0,2,3,4,6,255,255,255}, {1,2,3,4,6,255,255,255}, {0,1,2,3,4,6,255,255},
    {5,6,255,255,255,255,255,255}, {0,5,6,255,255,255,255,255}, {1,5,6,255,255,255,255,255}, {0,1,5,6,255,255,255,255},
    {2,5,6,255,255,255,255,255}, {0,2,5,6,255,255,255,255}, {1,2,5,6,255,255,255,255}, {0,1,2,5,6,255,255,255},
    {3,5,6,255,255,255,255,255}, {0,3,5,6,255,255,255,255}, {1,3,5,6,255,255,255,255}, {0,1,3,5,6,255,255,255},
    {2,3,5,6,255,255,255,255}, {0,2,3,5,6,255,255,255}, {1,2,3,5,6,255,255,255}, {0,1,2,3,5,6,255,255},
    {4,5,6,255,255,255,255,255}, {0,4,5,6,255,255,255,255}, {1,4,5,6,255,255,255,255}, {0,1,4,5,6,255,255,255},
    {2,4,5,6,255,255,255,255}, {0,2,4,5,6,255,255,255}, {1,2,4,5,6,255,255,255}, {0,1,2,4,5,6,255,255},
    {3,4,5,6,255,255,255,255}, {0,3,4,5,6,255,255,255}, {1,3,4,5,6,255,255,255}, {0,1,3,4,5,6,255,255},
    {2,3,4,5,6,255,255,255}, {0,2,3,4,5,6,255,255}, {1,2,3,4,5,6,255,255}, {0,1,2,3,4,5,6,255},
    {7,255,255,255,255,255,255,255}, {0,7,255,255,255,255,255,255}, {1,7,255,255,255,255,255,255}, {0,1,7,255,255,255,255,255},
    {2,7,255,255,255,255,255,255}, {0,2,7,255,255,255,255,255}, {1,2,7,255,255,255,255,255}, {0,1,2,7,255,255,255,255},
    {3,7,255,255,255,255,255,255}, {0,3,7,255,255,255,255,255}, {1,3,7,255,255,255,255,255}, {0,1,3,7,255,255,255,255},
    {2,3,7,255,255,255,255,255}, {0,2,3,7,255,255,255,255}, {1,2,3,7,255,255,255,255}, {0,1,2,3,7,255,255,255},
    {4,7,255,255,255,255,255,255}, {0,4,7,255,255,255,255,255}, {1,4,7,255,255,255,255,255}, {0,1,4,7,255,255,255,255},
    {2,4,7,255,255,255,255,255}, {0,2,4,7,255,255,255,255}, {1,2,4,7,255,255,255,255}, {0,1,2,4,7,255,255,255},
    {3,4,7,255,255,255,255,255}, {0,3,4,7,255,255,255,255}, {1,3,4,7,255,255,255,255}, {0,1,3,4,7,255,255,255},
    {2,3,4,7,255,255,255,255}, {0,2,3,4,7,255,255,255}, {1,2,3,4,7,255,255,255}, {0,1,2,3,4,7,255,255},
    {5,7,255,255,255,255,255,255}, {0,5,7,255,255,255,255,255}, {1,5,7,255,255,255,255,255}, {0,1,5,7,255,255,255,255},
    {2,5,7,255,255,255,255,255}, {0,2,5,7,255,255,255,255}, {1,2,5,7,255,255,255,255}, {0,1,2,5,7,255,255,255},
    {3,5,7,255,255,255,255,255}, {0,3,5,7,255,255,255,255}, {1,3,5,7,255,255,255,255}, {0,1,3,5,7,255,255,255},
    {2,3,5,7,255,255,255,255}, {0,2,3,5,7,255,255,255}, {1,2,3,5,7,255,255,255}, {0,1,2,3,5,7,255,255},
    {4,5,7,255,255,255,255,255}, {0,4,5,7,255,255,255,255}, {1,4,5,7,255,255,255,255}, {0,1,4,5,7,255,255,255},
    {2,4,5,7,255,255,255,255}, {0,2,4,5,7,255,255,255}, {1,2,4,5,7,255,255,255}, {0,1,2,4,5,7,255,255},
    {3,4,5,7,255,255,255,255}, {0,3,4,5,7,255,255,255}, {1,3,4,5,7,255,255,255}, {0,1,3,4,5,7,255,255},
    {2,3,4,5,7,255,255,255}, {0,2,3,4,5,7,255,255}, {1,2,3,4,5,7,255,255}, {0,1,2,3,4,5,7,255},
    {6,7,255,255,255,255,255,255}, {0,6,7,255,255,255,255,255}, {1,6,7,255,255,255,255,255}, {0,1,6,7,255,255,255,255},
    {2,6,7,255,255,255,255,255}, {0,2,6,7,255,255,255,255}, {1,2,6,7,255,255,255,255}, {0,1,2,6,7,255,255,255},
    {3,6,7,255,255,255,255,255}, {0,3,6,7,255,255,255,255}, {1,3,6,7,255,255,255,255}, {0,1,3,6,7,255,255,255},
    {2,3,6,7,255,255,255,255}, {0,2,3,6,7,255,255,255}, {1,2,3,6,7,255,255,255}, {0,1,2,3,6,7,255,255},
    {4,6,7,255,255,255,255,255}, {0,4,6,7,255,255,255,255}, {1,4,6,7,255,255,255,255}, {0,1,4,6,7,255,255,255},
    {2,4,6,7,255,255,255,255}, {0,2,4,6,7,255,255,255}, {1,2,4,6,7,255,255,255}, {0,1,2,4,6,7,255,255},
    {3,4,6,7,255,255,255,255}, {0,3,4,6,7,255,255,255}, {1,3,4,6,7,255,255,255}, {0,1,3,4,6,7,255,255},
    {2,3,4,6,7,255,255,255}, {0,2,3,4,6,7,255,255}, {1,2,3,4,6,7,255,255}, {0,1,2,3,4,6,7,255},
    {5,6,7,255,255,255,255,255}, {0,5,6,7,255,255,255,255}, {1,5,6,7,255,255,255,255}, {0,1,5,6,7,255,255,255},
    {2,5,6,7,255,255,255,255}, {0,2,5,6,7,255,255,255}, {1,2,5,6,7,255,255,255}, {0,1,2,5,6,7,255,255},
    {3,5,6,7,255,255,255,255}, {0,3,5,6,7,255,255,255}, {1,3,5,6,7,255,255,255}, {0,1,3,5,6,7,255,255},
    {2,3,5,6,7,255,255,255}, {0,2,3,5,6,7,255,255}, {1,2,3,5,6,7,255,255}, {0,1,2,3,5,6,7,255},
    {4,5,6,7,255,255,255,255}, {0,4,5,6,7,255,255,255}, {1,4,5,6,7,255,255,255}, {0,1,4,5,6,7,255,255},
    {2,4,5,6,7,255,255,255}, {0,2,4,5,6,7,255,255}, {1,2,4,5,6,7,255,255}, {0,1,2,4,5,6,7,255},
    {3,4,5,6,7,255,255,255}, {0,3,4,5,6,7,255,255}, {1,3,4,5,6,7,255,255}, {0,1,3,4,5,6,7,255},
    {2,3,4,5,6,7,255,255}, {0,2,3,4,5,6,7,255}, {1,2,3,4,5,6,7,255}, {0,1,2,3,4,5,6,7},
};

/* 8 length-class bitmap bytes for 64 symbols: masks64v's vpaddq tree with
 * the compare flipped to equality (bit s of the returned u64 = symbol s
 * has this exact length).  Local to the table build; the encode-side
 * masks64v (rank > thr) lives with the partition kernels. */
static inline uint8x8_t pivcoh__eqmasks64(uint8x16_t v0, uint8x16_t v1,
                                          uint8x16_t v2, uint8x16_t v3,
                                          uint8x16_t vt, uint8x16_t bw)
{
    uint8x16_t w0 = vandq_u8(vceqq_u8(v0, vt), bw);
    uint8x16_t w1 = vandq_u8(vceqq_u8(v1, vt), bw);
    uint8x16_t w2 = vandq_u8(vceqq_u8(v2, vt), bw);
    uint8x16_t w3 = vandq_u8(vceqq_u8(v3, vt), bw);
    uint8x16_t t0 = vpaddq_u8(w0, w1);
    uint8x16_t t1 = vpaddq_u8(w2, w3);
    uint8x16_t u0 = vpaddq_u8(t0, t1);
    return vget_low_u8(vpaddq_u8(u0, u0));
}

PIVCOHDEF int pivcoh_table_from_lens(pivcoh_table *t, const uint8_t code_len[256])
{
    /* Transposed classify in two passes.  Pass 1 scans code_len once for the
     * shortest and longest code present (vmin/vmax); pass 2 re-loads each
     * 64-symbol group (still hot in L1) and builds the length bitmaps only
     * for the live classes [minlen, maxlen] -- the empty bottom and top
     * classes never cost an eqmasks fold.  Fusing the scan and the classify
     * (the former single pass) would force all 11 classes, since min/max
     * aren't known mid-scan; on real code lengths minlen sits at ~3-4 and
     * maxlen at ~9-11, so the two passes run ~4*(range) ~= 28 folds vs 44,
     * and the reload is far cheaper than the eqmasks it removes.  minlen is
     * the max of the byte-wise two's-complement negations of the lengths:
     * an absent symbol (length 0) negates to 0 and drops out of the max,
     * while length L maps to 256-L, so the largest negation is the shortest
     * length -- no compare, and the all-zero case gives minlen=256 > maxlen,
     * caught by the used count.  (NEON has no byte movemask; the eqmasks
     * vpaddq tree folds four 16-lane equals into 8 bitmap bytes per class.) */
    static const uint8_t bw_a[16] =
        {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t bw = vld1q_u8(bw_a);
    uint64_t cmask[PIVCOH__MAXLEN][4];        /* [L-1][group]: bit s set iff
                                                 symbol 64g+s has length L */
    uint64_t popc[PIVCOH__MAXLEN][4];         /* [L-1][group]: byte j = popcount
                                                 of cmask byte j, precomputed by
                                                 vcnt so the extract loop's
                                                 n_used chain needs only an AND */
    const uint8x16_t vz = vdupq_n_u8(0);
    uint8x16_t vmax = vz, vlo = vz;           /* vlo = max of negated lengths */
    for (int g = 0; g < 4; g++) {             /* pass 1: shortest/longest only */
        const uint8_t *b = code_len + 64 * g;
        uint8x16_t x0 = vld1q_u8(b),      x1 = vld1q_u8(b + 16),
                   x2 = vld1q_u8(b + 32), x3 = vld1q_u8(b + 48);
        /* min-of-present first: its negate->max path is a step longer than
         * vmax's, so issue it before the plain vmax to give it slack. */
        uint8x16_t g0 = vsubq_u8(vz, x0), g1 = vsubq_u8(vz, x1),
                   g2 = vsubq_u8(vz, x2), g3 = vsubq_u8(vz, x3);
        vlo  = vmaxq_u8(vlo,  vmaxq_u8(vmaxq_u8(g0, g1), vmaxq_u8(g2, g3)));
        vmax = vmaxq_u8(vmax, vmaxq_u8(vmaxq_u8(x0, x1), vmaxq_u8(x2, x3)));
    }
    int minlen = 256 - vmaxvq_u8(vlo);               /* shortest; 256 if all-zero */
    int maxlen = vmaxvq_u8(vmax);                     /* longest code present */
    if (maxlen > PIVCOH__MAXLEN) return 0;            /* invalid length */
    for (int g = 0; g < 4; g++) {             /* pass 2: classify live range only */
        const uint8_t *b = code_len + 64 * g;
        uint8x16_t x0 = vld1q_u8(b),      x1 = vld1q_u8(b + 16),
                   x2 = vld1q_u8(b + 32), x3 = vld1q_u8(b + 48);
        for (int L = minlen; L <= maxlen; L++) {
            uint8x8_t em = pivcoh__eqmasks64(x0, x1, x2, x3,
                                             vdupq_n_u8((uint8_t)L), bw);
            cmask[L - 1][g] = vget_lane_u64(vreinterpret_u64_u8(em), 0);
            popc[L - 1][g]  = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(em)), 0);
        }
    }

    /* Extract items[] in (length, symbol) order + per-length counts.  Every
     * sub-group byte (up to the highest set one) scatters through select8 +
     * one 8-byte store, branchlessly: an empty byte (mm==0) looks up
     * select8[0]=all-0xFF, so vtbl yields 8 zeros that land at items[n_used]
     * with n_used unchanged and are overwritten by the next store (or absorbed
     * by items's 8-byte pad).  Dropping the per-sub-group "if (!mm) continue"
     * removed an unpredictable branch worth ~1.3x on these tables.  n_used
     * advances by the precomputed popc byte (an AND), keeping the scalar
     * popcount's GPR<->SIMD round-trip off the store's serial address chain. */
    uint8_t items[256 + 8];
    int cnt[PIVCOH__MAXLEN + 1], n_used = 0, s;
    const uint8x8_t iota8 = vcreate_u8(0x0706050403020100ull);
    for (int L = minlen; L <= maxlen; L++) {         /* only live length classes */
        int start = n_used;
        for (int g = 0; g < 4; g++) {
            uint64_t m = cmask[L - 1][g], pc = popc[L - 1][g];
            for (int gb = 64 * g; m; gb += 8, m >>= 8, pc >>= 8) {
                unsigned mm = (unsigned)(m & 0xff);
                uint8x8_t ids = vadd_u8(iota8, vdup_n_u8((uint8_t)gb));
                vst1_u8(items + n_used,
                        vtbl1_u8(ids, vld1_u8(pivcoh__select8[mm])));
                n_used += (unsigned)(pc & 0xff);
            }
        }
        cnt[L] = n_used - start;
    }
    if (n_used == 0) return 0;
    memcpy(t->code_len, code_len, 256);
    t->sched_len = 0;

    if (n_used == 1) {                         /* degenerate: 1-bit code */
        s = items[0];
        memset(t->code_len, 0, 256);
        t->code_len[s] = 1;
        t->rank_to_sym[0] = t->rank_to_sym[1] = (uint8_t)s;
        t->num_ranks = 2;
        t->sched[0].kd = PIVCOH__FLAT | 1 << 2;
        t->sched[0].param = t->sched[0].right = 0;
        t->sched_len = 1;
    } else {
        /* "optimized" chunking: split each length's count by its set bits
         * (largest first), a 2^b chunk rooted at depth L-b, ordered so
         * canonical assignment fills the tree left-to-right. */
        pivcoh__chunk ch[49];   /* max sum popcount(cnt[L]): 11 classes, sum <= 256 */
        int nch = 0;
        /* Generate the chunks (length asc, bit desc), then stable depth-sort.
         * Walk only the set bits of cnt[L] via clz (highest first) rather than
         * testing all 9 positions -- fewer stores and fewer data-dependent
         * branches than the unrolled bit-test loop, ~1.1-1.18x on the M4. */
        int i, j, n, acc, L;
        for (L = minlen, acc = 0; L <= maxlen; acc += cnt[L], L++) {
            uint32_t tmp = (uint32_t)cnt[L];
            for (j = acc; tmp; tmp &= ~(0x80000000u >> n)) {
                n = __builtin_clz(tmp);
                i = 31 - n;
                ch[nch].bit = (uint8_t)i;
                ch[nch].depth = (uint8_t)(L - i);
                ch[nch].sym_idx = (uint8_t)j;
                j += 1 << i;
                nch++;
            }
        }
        for (i = 1; i < nch; i++) {
            pivcoh__chunk c = ch[i];
            for (j = i - 1; j >= 0 && ch[j].depth > c.depth; j--) ch[j + 1] = ch[j];
            ch[j + 1] = c;
        }
        if (pivcoh__sched(t, ch, nch, items) != 0) return 0;
    }

    t->enc_ready = 0;        /* sym_to_rank fills lazily on first encode:
                                this builder is also the decode-side table
                                build, which never reads it */
    return 1;
}

typedef struct { uint32_t freq; uint16_t sym; } pivcoh__leaf;

/* Stable ascending (freq, sym) sort of the leaves.  They arrive in
 * symbol order (the stable seed), so a stable freq sort IS the (freq,
 * sym) order.  Small alphabets insertion-sort; larger ones take an LSD
 * radix over only the frequency bytes that VARY across the set (vary =
 * OR ^ AND of all freqs, a free by-product of the caller's scan) — a
 * constant byte is an identity pass, so it is skipped outright.  Port
 * of the production sort_leaves_by_freq (same n <= 40 crossover, minus
 * its dominant-bin scatter specialization); the O(n^2) insertion sort
 * this replaces was 3-5x the whole production table build on
 * fresh-tables-every-4K workloads over near-full alphabets. */
static void pivcoh__sort_leaves(pivcoh__leaf *leaf, int n, uint32_t vary)
{
    int i, j;
    if (n <= 40) {
        for (i = 1; i < n; i++) {
            pivcoh__leaf cur = leaf[i];
            for (j = i - 1; j >= 0 && leaf[j].freq > cur.freq; j--)
                leaf[j + 1] = leaf[j];
            leaf[j + 1] = cur;
        }
        return;
    }
    int shift[4], npass = 0;
    for (int b = 0; b < 32; b += 8)
        if ((vary >> b) & 0xFF) shift[npass++] = b;
    if (npass == 0) return;                    /* all frequencies equal */
    /* u8 bins cannot go wrong at n <= 256: a bin could only reach 256
     * if every leaf shared that byte, but such a plane does not vary
     * and is skipped, so varying-plane bins are <= 255.  A prefix that
     * wraps to 0 is only stored for an empty bin (never indexed), and
     * the final in-scatter increment that wraps is never read again. */
    uint8_t cnt[4][256];
    memset(cnt, 0, (size_t)npass * sizeof(cnt[0]));
    for (i = 0; i < n; i++)                    /* all planes in one pass */
        for (int p = 0; p < npass; p++)
            cnt[p][(leaf[i].freq >> shift[p]) & 0xFF]++;
    pivcoh__leaf tmp[256], *src = leaf, *dst = tmp;
    for (int p = 0; p < npass; p++) {
        unsigned sum = 0;
        for (int k = 0; k < 256; k++) {
            unsigned c = cnt[p][k];
            cnt[p][k] = (uint8_t)sum;
            sum += c;
        }
        for (i = 0; i < n; i++)
            dst[cnt[p][(src[i].freq >> shift[p]) & 0xFF]++] = src[i];
        pivcoh__leaf *t = src; src = dst; dst = t;
    }
    if (src != leaf) memcpy(leaf, src, (size_t)n * sizeof(*leaf));
}

/* ============ joint length/shape optimization (encoder side) ============
 *
 * Port of the production joint_lengths.c (joint-cost-model @ c6073f5).
 * Chunk model: choosing lengths IS choosing at most one chunk per
 * (level L <= 11, flat depth b <= min(8, L)) — a chunk holds 2^b
 * symbols at length L inside a depth-b flat, so each of its symbols'
 * occurrences costs L bits and L - b merge passes.  Objective:
 *     J = sum_s n_s * (L_s + lambda*(L_s - b_s + kappa[b_s]))
 *         + lambda * gamma * blocks * records
 * subject to chunk-root Kraft equality.  For a fixed chunk multiset the
 * optimal symbol assignment deals freq-sorted symbols into cost-sorted
 * chunks (rearrangement inequality), which turns the solve into a DP
 * over (symbols placed, open slots); lambda = 0 degenerates to the
 * Huffman baseline, so the result can only improve in-model, and a
 * kind-aware time model guards against out-of-model regressions.
 *
 * Deliberately not ported: the FSE decode-tax term (pivcoh speaks the
 * raw-bitmap subset — no bitmap is ever FSE-coded) and the ~10 MB
 * mass-DP fallback for lambda > 1/7 — when the slot DP's validity
 * condition fails, the baseline is kept, the same contract as a guard
 * reject. */

/* ---- kind-aware decode-time model (the adoption guard) ----
 *
 * The per-occurrence model above prices every merge alike, but the
 * decoder's merges differ by KIND: a merge with a lone-leaf child uses
 * the cheap cst kernels, a merge of two internal streams pays the full
 * partition.  Tree arrangement is deterministic from the chunk
 * multiset (roots sorted by depth, canonical prefixes), so for <= 33
 * chunks we simulate the skeleton exactly and price each node by kind.
 * Used on BOTH sides of the guard's comparison; the DP keeps its
 * separable search cost (the guard is where mispricing must not
 * survive). */
typedef struct { uint8_t r, D; double W; } pivcoh__jl_ch;

/* Subtree at depth d spanning chunks ch[*i..): consumes them, returns
 * the subtree's decode-time units and its weight; *kind reports what
 * the parent sees (0 = lone leaf, 1 = internal).  pre marks the chunk
 * index holding the prefilled top symbol (-1 = none).
 *
 * Iterative: an explicit frame stack replaces the recursion, matching
 * the decode walk's style — the tree is only <= MAXLEN + 1 deep, but
 * the recursive form carried ten arguments per call and the guard runs
 * the walk twice per window.  phase 0 frames are waiting on their left
 * child, phase 1 on their right; combine order, cursor state at the
 * prefill test, and the ((t + tl) + tr) association all match the
 * recursive form exactly, so results are bit-identical. */
static double pivcoh__jl_sim(const pivcoh__jl_ch *ch, int n, int *i, int d,
                             int pre, const pivcoh_joint *jp,
                             const double *kap, int *recs,
                             double *Wout, int *kind)
{
    struct {
        double tl, Wl;
        int il, kl;
        uint8_t d, phase;
    } stk[PIVCOH__MAXLEN + 2];
    int sp = 0;
    double rt, rW;
    int rkind;

enter:
    if (d > PIVCOH__MAXLEN) {   /* non-tiling multiset: cut the walk;
                                 * the caller's i != n check reports -1.
                                 * Unreachable from the in-header callers
                                 * (their multisets are Kraft-exact by
                                 * construction) — pure stack-safety.
                                 * Upstream fix 93b5a7e. */
        rt = 0.0; rW = 0; rkind = 1;
        goto unwind;
    }
    if (*i < n && ch[*i].r == d) {
        const pivcoh__jl_ch *c = &ch[(*i)++];
        rW = c->W;
        if (c->D == 0) { rkind = 0; rt = 0.0; goto unwind; }
        rkind = 1;
        (*recs)++;                                 /* pair/flat record */
        rt = c->W * kap[c->D];                     /* D=1 pair: kap[1] */
        goto unwind;
    }
    stk[sp].il = *i;
    stk[sp].d = (uint8_t)d;
    stk[sp].phase = 0;
    sp++;
    (*recs)++;                                     /* merge record */
    d++;
    goto enter;

unwind:
    if (sp == 0) {
        *Wout = rW;
        *kind = rkind;
        return rt;
    }
    if (stk[sp - 1].phase == 0) {                  /* left child done */
        stk[sp - 1].tl = rt;
        stk[sp - 1].Wl = rW;
        stk[sp - 1].kl = rkind;
        stk[sp - 1].phase = 1;
        d = stk[sp - 1].d + 1;
        goto enter;                                /* right child */
    }
    {                                              /* right child done */
        const double Wl = stk[sp - 1].Wl, Wr = rW;
        const double tl = stk[sp - 1].tl, tr = rt;
        const int kl = stk[sp - 1].kl, kr = rkind;
        const int il = stk[sp - 1].il;
        const double W = Wl + Wr;
        double t;
        if (kl == 0 || kr == 0) {
            t = W * jp->mu_cst;               /* one lone leaf: cst_vec */
            /* prefilled leaf: its side is memset ahead; the merge only
             * moves the internal side */
            if (pre >= 0 && ((kl == 0 && il == pre) ||
                             (kr == 0 && *i - 1 == pre)))
                t -= (kl == 0 ? Wl : Wr) * (double)jp->prefill * jp->mu_cst;
        } else
            t = W;                            /* full partition */
        rt = t + tl + tr;
        rW = W;
        rkind = 1;
        sp--;
        goto unwind;
    }
}

/* Kind-aware decode time for a chunk list (any order; sorted here into
 * the order the table builder realizes: root depth asc, D asc.  The
 * builder stable-sorts its L-ascending generation by depth only, and
 * equal-depth chunks from lower classes have smaller D, so depth-then-D
 * ascending IS that order — and since a class emits each D at most
 * once, (r, D) is unique and the sort is total).  The prefill chunk is
 * the heaviest-per-symbol chunk, if it is a lone leaf. */
static double pivcoh__jl_time(pivcoh__jl_ch *ch, int n,
                              const pivcoh_joint *jp, const double *kap,
                              double total_weight)
{
    int i, j;
    for (i = 1; i < n; i++) {
        pivcoh__jl_ch c = ch[i];
        for (j = i - 1; j >= 0 && (ch[j].r > c.r ||
                 (ch[j].r == c.r && ch[j].D > c.D)); j--)
            ch[j + 1] = ch[j];
        ch[j + 1] = c;
    }
    int pre = -1;
    double best = -1;
    for (i = 0; i < n; i++) {
        double per = ch[i].W / (double)(1 << ch[i].D);
        if (per > best) { best = per; pre = ch[i].D == 0 ? i : -1; }
    }
    int ii = 0, kind, recs = 0;
    double W;
    double t = pivcoh__jl_sim(ch, n, &ii, 0, pre, jp, kap, &recs, &W, &kind);
    if (ii != n) return -1.0;   /* malformed multiset (cannot happen) */
    if (jp->gamma > 0) {        /* per-record fixed cost x blocks/window */
        double blocks = ceil(total_weight / 16384.0);
        if (blocks < 1) blocks = 1;
        t += (double)jp->gamma * (double)recs * blocks;
    }
    return t;
}

/* ---- slot-ledger DP (exact for lambda <= 1/7) ----
 *
 * A state is (k symbols placed, s open slots at the current level);
 * Kraft EQUALITY forces s <= sigma - k at every level.  Levels are
 * processed ascending, chunk types within a level in cost order; that
 * equals GLOBAL chunk-cost order — the sorted-matching exactness
 * requirement — iff pivcoh__jl_order's spread bound holds (kappa = 0
 * recovers the classic lambda <= 1/7).  Three structural facts make the
 * walk L1-resident:
 *
 * DIAGONALS.  A take (k, s) -> (k + 2^b, s - 2^b) preserves t = k + s,
 * so within a level the DP decomposes into independent diagonals.
 * Stored diagonal-major, all take sweeps of a level run over one
 * <~0.5 KB row; the plane is traversed once per level (the doubling).
 *
 * PARITY.  Level-entry states have even s (they come from the doubling
 * s' = 2s) and takes with b >= 1 preserve s-parity, so the live lattice
 * is k == t (mod 2): compact index j = (k - (t&1))/2 halves each row.
 * b = 0 — the only parity flip, always last in the level's cost order —
 * is folded into the doubling (an odd-s cell's unique source is its
 * even-lattice predecessor plus one lone leaf) and reconstructed from
 * s-parity at backtrack.  The deepest level never takes b = 0: entry s
 * is even and the terminal needs takes summing to s exactly.
 *
 * CAPACITY BAND.  A state at level L can place at most s * 2^h more
 * symbols (h = levels below), so sigma - k <= (t - k) << h is necessary
 * — and met by every completing trajectory, making the prune exact.
 * Feasibility is preserved cell-to-cell by takes and by the doubling,
 * so pruned — hence stale — cells are never read.
 *
 * Terminal: (k = sigma, s = 0) after the deepest level.  Per-level u16
 * pick rows (bits 1..8; bit 0 is implicit in parity) are archived per
 * diagonal for backtrack. */
#define PIVCOH__JL_WMAX 132     /* max compact row: j <= 128, padded to x4 */

/* Largest compact index j on diagonal t whose k = 2j + (t&1) can still
 * feed sigma - k leaves through (t - k) slots h levels above the
 * bottom; -1 if the whole row is infeasible. */
static inline int pivcoh__jl_jcap(int t, int h, int sigma)
{
    const int p = t & 1;
    int kcap;
    if (h == 0) {
        kcap = t;                     /* t == sigma: all k feasible */
    } else {
        const int num = (t << h) - sigma;
        if (num < 0) return -1;
        kcap = num / ((1 << h) - 1);
        if (kcap > t) kcap = t;
    }
    if (kcap < p) return -1;
    return (kcap - p) >> 1;
}

/* Within-level sweep/deal order under kernel costs.  cost(L, b) =
 * L(1+lam) + g(b) with g(b) = lam*(kap[b] - b): the within-level cost
 * order is L-independent, so one sorted order serves every level.
 * Exactness of the slot DP needs (a) cross-level monotonicity:
 * spread(g) <= 1 + lam (kappa = 0 recovers lam <= 1/7), and (b) b = 0
 * dearest within the level (the parity fold runs it last).  Fills
 * border[0..*nb) with b = 1..bcap by ascending g; returns 1 iff both
 * hold (on 0 the caller keeps the baseline). */
static int pivcoh__jl_order(double lam, const double *kap, int bcap,
                            int border[8], int *nb)
{
    double g[9];
    double gmin = 0, gmax = 0;
    for (int b = 0; b <= bcap; b++) {
        g[b] = lam * (kap[b] - (double)b);
        if (b == 0 || g[b] < gmin) gmin = g[b];
        if (b == 0 || g[b] > gmax) gmax = g[b];
    }
    if (gmax - gmin > (1.0 + lam) * (1.0 - 1e-9)) return 0;
    int n = 0;
    for (int b = 1; b <= bcap; b++) {
        if (g[b] > g[0] + 1e-12) return 0;   /* b0 must stay dearest */
        int i = n++;
        while (i > 0 && (g[border[i - 1]] > g[b]
                         || (g[border[i - 1]] == g[b] && border[i - 1] < b))) {
            border[i] = border[i - 1];
            i--;
        }
        border[i] = b;                       /* ties: larger b first */
    }
    *nb = n;
    return 1;
}

/* lmax/bcap parameterize the level range and flat-depth cap so the
 * same solver runs the exact problem (11, 8) and the 2^G-grouped
 * coarse problem (11-G, 8-G): a group of 2^G sorted symbols at real
 * level L is a depth-G flat, so the coarse problem is this problem
 * shifted by G with an identical cost form.  tc0/tc1: per-take J
 * constants (lambda * gamma * blocks * records added) for b = 0 and
 * b >= 1 takes.  scratch: PIVCOH_JOINT_SCRATCH_SIZE or NULL. */
static double pivcoh__jl_slots(const double *P, int sigma, double lam,
                               int lmax, int bcap, const double *kap,
                               double tc0, double tc1,
                               uint16_t out_BL[PIVCOH__MAXLEN + 1],
                               void *scratch)
{
    int border[8], nb;
    if (!pivcoh__jl_order(lam, kap, bcap, border, &nb))
        return -1.0;
    /* sigma <= 32 rows fit five q-registers at a fixed W = 20, and
     * sigma <= 64 rows nine at W = 36: the take sweeps then run
     * register-resident per diagonal (loads/stores once per row
     * instead of per item), which is where the grouped tiers' time
     * lives.  The 9-group form keeps only a two-register candidate
     * window live: dest groups are processed descending, so a
     * candidate's source group is always still pre-item. */
    const int sm32 = sigma <= 32;
    const int sm64 = !sm32 && sigma <= 64;
    const int W = sm32 ? 20 : sm64 ? 36 : (((sigma >> 1) + 2) + 3) & ~3;
    const size_t plane = (size_t)(sigma + 1) * (size_t)W;
    uint8_t *own = scratch ? NULL :
        (uint8_t *)malloc(plane * (4 + 2 * (size_t)lmax) + 8);
    if (!own && !scratch) return -1.0;
    float *cost = (float *)(((uintptr_t)(own ? own : (uint8_t *)scratch) + 3)
                            & ~(uintptr_t)3);
    uint16_t *arch = (uint16_t *)(cost + plane);
    float     dPt[2][9][PIVCOH__JL_WMAX];   /* [t&1][b][j]: P[k+2^b]-P[k] */
    float     dP0[257];                     /* P[k] - P[k-1] */

    for (int p = 0; p < 2; p++)
        for (int b = 1; b <= bcap; b++) {
            const int cnk = 1 << b;
            for (int j = 0; j < W; j++) {
                const int k = 2 * j + p;
                dPt[p][b][j] = k + cnk <= sigma
                             ? (float)(P[k + cnk] - P[k]) : 0.0f;
            }
        }
    dP0[0] = 0.0f;
    for (int k = 1; k <= sigma; k++) dP0[k] = (float)(P[k] - P[k - 1]);

    int tlo[PIVCOH__MAXLEN + 1], thi[PIVCOH__MAXLEN + 1];
    for (int L = 1; L <= lmax; L++) {
        const int h = lmax - L;
        thi[L] = (1 << L) > sigma ? sigma : (1 << L);
        tlo[L] = (sigma + (1 << h) - 1) >> h;
        if (tlo[L] < 1) tlo[L] = 1;
    }

    /* Lazy init: every row is fully written by the doubling that
     * produces its level, so only the level-1 band rows need priming. */
    for (int t = tlo[1]; t <= thi[1]; t++)
        for (int j = 0; j < W; j++) cost[(size_t)t * W + j] = INFINITY;
    cost[2 * W + 0] = 0.0f;      /* level-1 entry: k = 0, s = 2, t = 2 */

    for (int L = 1; L <= lmax; L++) {
        const int h = lmax - L;
        const int bmax = L < bcap ? L : bcap;
        uint16_t *archL = arch + (size_t)(L - 1) * plane;
        for (int t = tlo[L]; t <= thi[L]; t++) {
            const int p = t & 1;
            const int jcap = pivcoh__jl_jcap(t, h, sigma);
            if (jcap < 0) continue;
            float *row = cost + (size_t)t * W;
            uint16_t *prow = archL + (size_t)t * W;   /* picks, archived
                                                       * in place */
            if (jcap >= 20 && jcap < 36) {
                /* Nine-group register-resident sweep for rows whose
                 * dests all fit lanes 0..35: every wide sm64 row (the
                 * grouped auto tier's bulk) and the mid-band rows of
                 * full-width exact solves.  Same junk-propagation
                 * safety as below: dest > source always, so beyond-jcap
                 * lanes never contaminate the band.  Candidates are
                 * computed on the fly per dest group, descending, so
                 * sources are pre-item.  Narrower rows fall through to
                 * the five-group body — it only touches lanes 0..19,
                 * which cover every dest, and processing 9 groups for
                 * a 2-group band costs more than it saves. */
                float32x4_t r[9];
                uint16x4_t pk[9];
                const float32x4_t vinf = vdupq_n_f32(INFINITY);
                const uint16x4_t z16 = vdup_n_u16(0);
#pragma clang loop unroll(full)
                for (int g = 0; g < 9; g++) {
                    r[g] = vld1q_f32(row + 4 * g);
                    pk[g] = vdup_n_u16(0);
                }
                for (int oi = 0; oi < nb; oi++) {
                    const int b = border[oi];
                    if (b > bmax) continue;
                    const int jstep = 1 << (b - 1);
                    if (jcap - jstep < 0) continue;
                    const float a = (float)((double)L
                                            + lam * ((double)(L - b) + kap[b]));
                    const float32x4_t va = vdupq_n_f32(a);
                    const float32x4_t vtc = vdupq_n_f32((float)tc1);
                    const float *dpb = dPt[p][b];
                    const uint16x4_t vbit = vdup_n_u16((uint16_t)(1u << b));
#define PIVCOH__JL9_CAND(g) \
    vaddq_f32(vfmaq_f32(r[g], vld1q_f32(dpb + 4 * (g)), va), vtc)
#define PIVCOH__JL9_TAKE(g, s, kq) do { \
    const uint32x4_t m_ = vcltq_f32((s), r[g]); \
    r[g] = vbslq_f32(m_, (s), r[g]); \
    pk[g] = vbsl_u16(vmovn_u32(m_), (kq), pk[g]); } while (0)
#define PIVCOH__JL9_SHIFTK(K) do { \
    _Pragma("clang loop unroll(full)") \
    for (int g = 8; g >= (K); g--) { \
        const float32x4_t c_ = PIVCOH__JL9_CAND(g - (K)); \
        const uint16x4_t kq_ = vorr_u16(pk[g - (K)], vbit); \
        PIVCOH__JL9_TAKE(g, c_, kq_); \
    } } while (0)
#define PIVCOH__JL9_EXT(N) do { \
    float32x4_t chi_ = PIVCOH__JL9_CAND(8); \
    uint16x4_t khi_ = vorr_u16(pk[8], vbit); \
    _Pragma("clang loop unroll(full)") \
    for (int g = 8; g >= 1; g--) { \
        const float32x4_t clo_ = PIVCOH__JL9_CAND(g - 1); \
        const uint16x4_t klo_ = vorr_u16(pk[g - 1], vbit); \
        PIVCOH__JL9_TAKE(g, vextq_f32(clo_, chi_, N), \
                         vext_u16(klo_, khi_, N)); \
        chi_ = clo_; khi_ = klo_; \
    } \
    PIVCOH__JL9_TAKE(0, vextq_f32(vinf, chi_, N), \
                     vext_u16(z16, khi_, N)); } while (0)
                    switch (jstep) {
                    case 1:  PIVCOH__JL9_EXT(3);    break;
                    case 2:  PIVCOH__JL9_EXT(2);    break;
                    case 4:  PIVCOH__JL9_SHIFTK(1); break;
                    case 8:  PIVCOH__JL9_SHIFTK(2); break;
                    case 16: PIVCOH__JL9_SHIFTK(4); break;
                    default: PIVCOH__JL9_SHIFTK(8); break;   /* 32 */
                    }
#undef PIVCOH__JL9_CAND
#undef PIVCOH__JL9_TAKE
#undef PIVCOH__JL9_SHIFTK
#undef PIVCOH__JL9_EXT
                }
#pragma clang loop unroll(full)
                for (int g = 0; g < 9; g++) {
                    vst1q_f32(row + 4 * g, r[g]);
                    vst1_u16(prow + 4 * g, pk[g]);
                }
                continue;
            }
            if (sm32 || sm64 || jcap < 20) {
                /* Whole row in five registers across every item — used
                 * whenever every dest fits lanes 0..19: all of
                 * sigma <= 32, narrow sm64 rows, and the narrow-band
                 * rows of full-width exact solves (deep levels, band
                 * edges); wider lanes are simply left untouched.  No
                 * lane masking: a candidate's dest is always above its
                 * source, so lanes beyond jcap only ever contaminate
                 * lanes beyond jcap, and nothing in band ever reads
                 * them (same argument the in-place generic sweep
                 * relies on).  Shift-ins at the low edge are +inf. */
                float32x4_t r0 = vld1q_f32(row),      r1 = vld1q_f32(row + 4),
                            r2 = vld1q_f32(row + 8),  r3 = vld1q_f32(row + 12),
                            r4 = vld1q_f32(row + 16);
                uint16x4_t p0 = vdup_n_u16(0), p1 = p0, p2 = p0, p3 = p0,
                           p4 = p0;
                const float32x4_t vinf = vdupq_n_f32(INFINITY);
                const uint16x4_t z16 = vdup_n_u16(0);
                for (int oi = 0; oi < nb; oi++) {
                    const int b = border[oi];
                    if (b > bmax) continue;
                    const int jstep = 1 << (b - 1);
                    if (jcap - jstep < 0) continue;
                    const float a = (float)((double)L
                                            + lam * ((double)(L - b) + kap[b]));
                    const float32x4_t va = vdupq_n_f32(a);
                    const float32x4_t vtc = vdupq_n_f32((float)tc1);
                    const float *dpb = dPt[p][b];
                    float32x4_t c0 = vaddq_f32(vfmaq_f32(r0, vld1q_f32(dpb), va), vtc);
                    float32x4_t c1 = vaddq_f32(vfmaq_f32(r1, vld1q_f32(dpb + 4), va), vtc);
                    float32x4_t c2 = vaddq_f32(vfmaq_f32(r2, vld1q_f32(dpb + 8), va), vtc);
                    float32x4_t c3 = vaddq_f32(vfmaq_f32(r3, vld1q_f32(dpb + 12), va), vtc);
                    float32x4_t c4 = vaddq_f32(vfmaq_f32(r4, vld1q_f32(dpb + 16), va), vtc);
                    const uint16x4_t vbit = vdup_n_u16((uint16_t)(1u << b));
                    uint16x4_t q0 = vorr_u16(p0, vbit), q1 = vorr_u16(p1, vbit),
                               q2 = vorr_u16(p2, vbit), q3 = vorr_u16(p3, vbit),
                               q4 = vorr_u16(p4, vbit);
                    float32x4_t s0, s1, s2, s3, s4;
                    uint16x4_t k0, k1, k2, k3, k4;
                    switch (jstep) {
                    case 1:
                        s0 = vextq_f32(vinf, c0, 3); s1 = vextq_f32(c0, c1, 3);
                        s2 = vextq_f32(c1, c2, 3);   s3 = vextq_f32(c2, c3, 3);
                        s4 = vextq_f32(c3, c4, 3);
                        k0 = vext_u16(z16, q0, 3);   k1 = vext_u16(q0, q1, 3);
                        k2 = vext_u16(q1, q2, 3);    k3 = vext_u16(q2, q3, 3);
                        k4 = vext_u16(q3, q4, 3);
                        break;
                    case 2:
                        s0 = vextq_f32(vinf, c0, 2); s1 = vextq_f32(c0, c1, 2);
                        s2 = vextq_f32(c1, c2, 2);   s3 = vextq_f32(c2, c3, 2);
                        s4 = vextq_f32(c3, c4, 2);
                        k0 = vext_u16(z16, q0, 2);   k1 = vext_u16(q0, q1, 2);
                        k2 = vext_u16(q1, q2, 2);    k3 = vext_u16(q2, q3, 2);
                        k4 = vext_u16(q3, q4, 2);
                        break;
                    case 4:
                        s0 = vinf; s1 = c0; s2 = c1; s3 = c2; s4 = c3;
                        k0 = z16;  k1 = q0; k2 = q1; k3 = q2; k4 = q3;
                        break;
                    case 8:
                        s0 = vinf; s1 = vinf; s2 = c0; s3 = c1; s4 = c2;
                        k0 = z16;  k1 = z16;  k2 = q0; k3 = q1; k4 = q2;
                        break;
                    default: /* 16 */
                        s0 = vinf; s1 = vinf; s2 = vinf; s3 = vinf; s4 = c0;
                        k0 = z16;  k1 = z16;  k2 = z16;  k3 = z16;  k4 = q0;
                        break;
                    }
                    uint32x4_t m;
                    m = vcltq_f32(s0, r0); r0 = vbslq_f32(m, s0, r0);
                    p0 = vbsl_u16(vmovn_u32(m), k0, p0);
                    m = vcltq_f32(s1, r1); r1 = vbslq_f32(m, s1, r1);
                    p1 = vbsl_u16(vmovn_u32(m), k1, p1);
                    m = vcltq_f32(s2, r2); r2 = vbslq_f32(m, s2, r2);
                    p2 = vbsl_u16(vmovn_u32(m), k2, p2);
                    m = vcltq_f32(s3, r3); r3 = vbslq_f32(m, s3, r3);
                    p3 = vbsl_u16(vmovn_u32(m), k3, p3);
                    m = vcltq_f32(s4, r4); r4 = vbslq_f32(m, s4, r4);
                    p4 = vbsl_u16(vmovn_u32(m), k4, p4);
                }
                vst1q_f32(row, r0);      vst1q_f32(row + 4, r1);
                vst1q_f32(row + 8, r2);  vst1q_f32(row + 12, r3);
                vst1q_f32(row + 16, r4);
                vst1_u16(prow, p0);      vst1_u16(prow + 4, p1);
                vst1_u16(prow + 8, p2);  vst1_u16(prow + 12, p3);
                vst1_u16(prow + 16, p4);
                continue;
            }
            memset(prow, 0, (size_t)(jcap + 1) * sizeof(uint16_t));
            for (int oi = 0; oi < nb; oi++) {
                const int b = border[oi];
                if (b > bmax) continue;
                const int jstep = 1 << (b - 1);       /* = 2^b slots / 2 */
                const int jhi = jcap - jstep;         /* dest j <= jcap  */
                if (jhi < 0) continue;
                const float a = (float)((double)L
                                         + lam * ((double)(L - b) + kap[b]));
                const float tc = (float)tc1;
                const float *dpb = dPt[p][b];
                int j = jhi;
                /* 0/1 in-place: dest j + jstep > src j, so iterate j
                 * descending — a written dest is never re-read as a
                 * source for the same chunk.  Stores are unconditional:
                 * everything is L1-resident, so blending beats the
                 * data-dependent branch of an "improved?" early-out. */
                const float32x4_t va = vdupq_n_f32(a);
                const float32x4_t vtc = vdupq_n_f32(tc);
                const uint16x4_t vbit = vdup_n_u16((uint16_t)(1u << b));
                for (; j >= 7; j -= 8) {
                    const int b1 = j - 3, b2 = j - 7;
                    float32x4_t s1 = vld1q_f32(row + b1);
                    float32x4_t s2 = vld1q_f32(row + b2);
                    float32x4_t c1 = vaddq_f32(
                        vfmaq_f32(s1, vld1q_f32(dpb + b1), va), vtc);
                    float32x4_t c2 = vaddq_f32(
                        vfmaq_f32(s2, vld1q_f32(dpb + b2), va), vtc);
                    float32x4_t d1 = vld1q_f32(row + b1 + jstep);
                    float32x4_t d2 = vld1q_f32(row + b2 + jstep);
                    uint32x4_t m1 = vcltq_f32(c1, d1);
                    uint32x4_t m2 = vcltq_f32(c2, d2);
                    vst1q_f32(row + b1 + jstep, vbslq_f32(m1, c1, d1));
                    vst1q_f32(row + b2 + jstep, vbslq_f32(m2, c2, d2));
                    uint16x4_t pv1 = vorr_u16(vld1_u16(prow + b1), vbit);
                    uint16x4_t pv2 = vorr_u16(vld1_u16(prow + b2), vbit);
                    uint16x4_t qv1 = vld1_u16(prow + b1 + jstep);
                    uint16x4_t qv2 = vld1_u16(prow + b2 + jstep);
                    vst1_u16(prow + b1 + jstep,
                             vbsl_u16(vmovn_u32(m1), pv1, qv1));
                    vst1_u16(prow + b2 + jstep,
                             vbsl_u16(vmovn_u32(m2), pv2, qv2));
                }
                for (; j >= 3; j -= 4) {
                    const int base = j - 3;
                    float32x4_t src = vld1q_f32(row + base);
                    float32x4_t cand = vaddq_f32(
                        vfmaq_f32(src, vld1q_f32(dpb + base), va), vtc);
                    float32x4_t dst = vld1q_f32(row + base + jstep);
                    uint32x4_t m = vcltq_f32(cand, dst);
                    vst1q_f32(row + base + jstep, vbslq_f32(m, cand, dst));
                    uint16x4_t pm = vmovn_u32(m);
                    uint16x4_t pv = vorr_u16(vld1_u16(prow + base), vbit);
                    uint16x4_t qv = vld1_u16(prow + base + jstep);
                    vst1_u16(prow + base + jstep, vbsl_u16(pm, pv, qv));
                }
                for (; j >= 0; j--) {
                    const float v = row[j];
                    if (!(v < INFINITY)) continue;
                    const float cand = v + a * dpb[j] + tc;
                    if (cand < row[j + jstep]) {
                        row[j + jstep] = cand;
                        prow[j + jstep] = (uint16_t)(prow[j] | (1u << b));
                    }
                }
            }
        }
        if (L == lmax) break;
        /* Doubling s' = 2s with the b = 0 take folded in.  Dest cell
         * (t', k) has the unique source (t = (t'+k)/2, k): even-lattice
         * there if k == t (mod 2), else the odd-s product of a lone
         * leaf taken at level L from (t, k-1).  In place, t' and j'
         * descending: sources live on rows <= t', and the single
         * same-row read (t = t', only at k = t') happens before its
         * cell is overwritten.
         *
         * Branchless: on dest row t' the source diagonal is t = t0 + j'
         * (t0 = (t'+p')/2), so the level-L band check hoists to a
         * j'-range, and the source parity d = (t^k)&1 alternates with
         * j' — two constant-stride subloops with the unified source
         * index (k - d - (t&1))/2.  The subloop containing the top cell
         * runs first (it holds the only same-row read). */
        /* NB the production source declares this constant and then never
         * adds it — its DP under-prices b = 0 takes by the per-record
         * gamma surcharge, which the guard then charges.  Fixed here:
         * the fold's take cost carries + tcz. */
        const float a0 = (float)((double)L * (1.0 + lam) + lam * kap[0]);
        const float tcz = (float)tc0;
        for (int tp = thi[L + 1]; tp >= tlo[L + 1]; tp--) {
            const int pp = tp & 1;
            const int jcap2 = pivcoh__jl_jcap(tp, h - 1, sigma);
            if (jcap2 < 0) continue;
            float *nrow = cost + (size_t)tp * W;
            const int t0 = (tp + pp) >> 1;
            int jlo = tlo[L] - t0; if (jlo < 0) jlo = 0;
            int jhi2 = thi[L] - t0; if (jhi2 > jcap2) jhi2 = jcap2;
            for (int jp2 = jcap2; jp2 > jhi2; jp2--) nrow[jp2] = INFINITY;
            for (int jp2 = jlo - 1; jp2 >= 0; jp2--) nrow[jp2] = INFINITY;
            for (int half = 0; half < 2; half++) {
                int jp2 = jhi2 - half;
                if (jp2 < jlo) continue;
                const int k1 = 2 * jp2 + pp;
                const int t1 = t0 + jp2;
                const int d = (t1 ^ k1) & 1;
                /* Signed source index into cost[]: at the odd branch's
                 * degenerate k = 0 cell the offset (k1 - d - (t1 & 1)) >> 1
                 * is negative.  The net cell is in-bounds and never read
                 * there, but casting that lone offset to size_t would wrap
                 * the pointer backwards (UB); keeping the whole index
                 * signed and indexing cost[]/dP0[] avoids forming any
                 * out-of-array pointer (a negative sentinel index is just
                 * an integer). */
                ptrdiff_t si = (ptrdiff_t)t1 * W + ((k1 - d - (t1 & 1)) >> 1);
                if (d == 0) {
                    for (; jp2 >= jlo; jp2 -= 2, si -= 2 * W + 2)
                        nrow[jp2] = cost[si];
                } else {
                    /* k = 0 has no lone-leaf predecessor: if this
                     * chain reaches cell (jp2 = 0, k = 0), stop above
                     * it and mark it unreachable. */
                    int floor2 = jlo, patch0 = 0;
                    if (pp == 0 && (jp2 & 1) == 0 && jlo == 0) {
                        floor2 = 2;
                        patch0 = 1;
                    }
                    ptrdiff_t di = k1;
                    for (; jp2 >= floor2; jp2 -= 2, si -= 2 * W + 2, di -= 4)
                        nrow[jp2] = cost[si] + a0 * dP0[di] + tcz;
                    if (patch0)
                        nrow[0] = INFINITY;
                }
            }
        }
    }

    double J = cost[(size_t)sigma * W + (size_t)((sigma - (sigma & 1)) >> 1)];
    if (J < INFINITY) {
        /* Backtrack: invert each level's transition; odd end-of-level
         * s means the folded b0 was taken there — recover its bits
         * from the even-lattice predecessor and set bit 0. */
        int k = sigma, s = 0;
        for (int L = lmax; L >= 1; L--) {
            const int t = k + s;
            const int p = t & 1;
            const uint16_t *arow = arch + (size_t)(L - 1) * plane
                                        + (size_t)t * W;
            uint16_t BL;
            if ((k ^ t) & 1)
                BL = (uint16_t)(arow[(k - 1 - p) >> 1] | 1u);
            else
                BL = arow[(k - p) >> 1];
            out_BL[L] = BL;
            int cL = 0;
            for (int b = 0; b <= 8; b++) if (BL & (1 << b)) cL += 1 << b;
            k -= cL;
            s += cL;                  /* slots at level L entry (even) */
            if (L > 1) s >>= 1;       /* pre-doubling slots left       */
        }
    } else {
        J = -1.0;
    }
    free(own);
    return J;
}

/* Realized chunk lists (r, D, W) for BOTH lens images — the incoming
 * baseline lb and the deal's candidate lc — priced as the tables
 * pivcoh_table_from_lens will build: within a class the builder takes
 * symbols in ascending symbol order and splits the count largest-set-
 * bit first, so chunk membership — and with it each chunk's weight —
 * is a function of lens alone, NOT of the deal that chose them.
 * Pricing the guard on realized weights (both sides) keeps it honest:
 * the DP's sorted matching of heavy symbols to cheap chunks is a
 * search relaxation the canonical rebuild does not reproduce.
 *
 * cnt* are per-class symbol counts, supplied by the caller (they fall
 * out of data already at hand); bins 0 and 12..15 are trash (absent /
 * garbage lengths — internal lens never exceed MAXLEN).  One fused
 * ascending 256-symbol sweep then deals every class's chunk cursor on
 * both sides simultaneously — ascending symbol order IS the builder's
 * membership order — with absent symbols draining into a zero-weight
 * dummy chunk via the &15 trash bins, branchlessly (their weight
 * contribution is 0.0).  ~0.25 us/solve; a per-class-sweep form cost
 * ~2.5 us, 40-60% of coarse-tier ebuild.  Frequencies narrow to u32
 * as in the leaf sort. */
static void pivcoh__jl_realized2(const uint8_t *lb, const uint8_t *lc,
                                 const int cntb[16], const int cntc[16],
                                 const uint64_t freq[256],
                                 pivcoh__jl_ch chb[40], int *nb_out,
                                 pivcoh__jl_ch chc[40], int *nc_out)
{
    int curb[16], curc[16], leftb[16], leftc[16], endb[16], endc[16];
    int nb = 0, nc = 0;
    for (int L = 1; L <= PIVCOH__MAXLEN; L++) {
        curb[L] = nb;
        curc[L] = nc;
        for (int b = 8; b >= 0; b--) {
            if (cntb[L] & (1 << b)) {
                chb[nb].r = (uint8_t)(b ? L - b : L);
                chb[nb].D = (uint8_t)b;
                nb++;
            }
            if (cntc[L] & (1 << b)) {
                chc[nc].r = (uint8_t)(b ? L - b : L);
                chc[nc].D = (uint8_t)b;
                nc++;
            }
        }
        endb[L]  = nb;
        endc[L]  = nc;
        leftb[L] = curb[L] < nb ? 1 << chb[curb[L]].D : 1;
        leftc[L] = curc[L] < nc ? 1 << chc[curc[L]].D : 1;
    }
    for (int t = 0; t < 16; t++)
        if (t == 0 || t > PIVCOH__MAXLEN) {
            curb[t] = nb; endb[t] = nb; leftb[t] = 0x7fffffff;
            curc[t] = nc; endc[t] = nc; leftc[t] = 0x7fffffff;
        }
    /* u64 accumulators: chunk weights are sums of <= 256 u32 values
     * (< 2^40), exact in either u64 or double — bit-identical prices,
     * but integer adds dodge the ucvtf + FP-latency chain per symbol.
     * Index nb/nc is the trash-bin dummy sink. */
    uint64_t wb[41] = {0}, wc[41] = {0};
    for (int s = 0; s < 256; s++) {
        const uint32_t f = (uint32_t)freq[s];
        const int Lb = lb[s] & 15, Lc = lc[s] & 15;
        wb[curb[Lb]] += f;
        if (--leftb[Lb] == 0 && ++curb[Lb] < endb[Lb])
            leftb[Lb] = 1 << chb[curb[Lb]].D;
        wc[curc[Lc]] += f;
        if (--leftc[Lc] == 0 && ++curc[Lc] < endc[Lc])
            leftc[Lc] = 1 << chc[curc[Lc]].D;
    }
    for (int i = 0; i < nb; i++) chb[i].W = (double)wb[i];
    for (int i = 0; i < nc; i++) chc[i].W = (double)wc[i];
    *nb_out = nb;
    *nc_out = nc;
}

/* Core over the build's leaf array (ascending — reversed in place
 * here; ghost-padding may append).  Overwrites lens[] on adoption;
 * any reject leaves them untouched. */
static int pivcoh__jl_core(pivcoh__leaf *sf, int sigma,
                           const uint64_t freq[256], uint8_t lens[256],
                           const pivcoh_joint *jp, void *scratch)
{
    const double lam = (double)jp->lambda;
    if (sigma < 2) return -1;
    double kap[9];
    for (int b = 0; b <= 8; b++)
        kap[b] = (jp->kappa[b] >= 0.0f && jp->kappa[b] < 100.0f)
               ? (double)jp->kappa[b] : 0.0;
    pivcoh_joint jv = *jp;      /* sanitized model knobs for the guard */
    if (!(jv.mu_cst > 0 && jv.mu_cst < 100)) jv.mu_cst = 1.0f;
    if (!(jv.prefill >= 0 && jv.prefill <= 1)) jv.prefill = 0.0f;
    if (!(jv.gamma >= 0 && jv.gamma < 1e6f)) jv.gamma = 0.0f;
    const double gbits = jp->guard_bits > 0 ? (double)jp->guard_bits : 1.015;
    const double gtime = jp->guard_time > 0 ? (double)jp->guard_time : 0.90;

    for (int i = 0; i < sigma / 2; i++) {  /* ascending -> descending */
        pivcoh__leaf tmp = sf[i];
        sf[i] = sf[sigma - 1 - i];
        sf[sigma - 1 - i] = tmp;
    }
    double P[257];
    P[0] = 0.0;
    for (int i = 0; i < sigma; i++) P[i + 1] = P[i] + (double)sf[i].freq;

    /* Baseline class counts (the guard itself is priced after the deal,
     * one fused pass covering both sides; internal lens are <= MAXLEN
     * so the &15 bins are exact, with 0 collecting absent symbols). */
    int cntb[16] = {0};
    for (int i = 0; i < sigma; i++)
        cntb[lens[sf[i].sym] & 15]++;

    /* Per-take fixed-cost constants: lambda * gamma * blocks, one
     * record for D0 takes (the skeleton merge above the leaf), two for
     * deeper chunks (the flat record + its stitch merge). */
    double blocks = ceil(P[sigma] / 16384.0);
    if (blocks < 1) blocks = 1;
    const double tc0 = lam * (double)jv.gamma * blocks;
    const double tc1 = 2.0 * tc0;

    /* Tier resolve.  Granularity g = 2^G groups the freq-sorted symbols
     * by g and solves the identical problem G levels shallower (a group
     * of g sorted symbols at real level L is a depth-G flat), 4^G fewer
     * states; near-optimal solutions are dense enough that g = 2 loses
     * ~0.13 % of J on average, g = 4 ~0.25 % (measured on lits data),
     * and the guard still rejects any bad case per window.  sigma is
     * ghost-padded to a multiple of g with zero-frequency unused byte
     * values — real leaves the encoder never emits; there are always
     * enough since sigma % g != 0 implies sigma < 256. */
    int gran = jp->gran;
    if (gran != -1 && gran != 1 && gran != 2 && gran != 4 && gran != 8)
        gran = 0;
    if (gran == 0)        /* auto: keep the solve ~<= 10 us at every sigma */
        gran = sigma <= 64 ? 1 : sigma <= 128 ? 2 : 4;
    else if (gran == -1)  /* coarse auto: one granularity step chunkier */
        gran = sigma <= 64 ? 2 : sigma <= 128 ? 4 : 8;
    int obuf[8], on;
    if (gran > 1 && (sigma < 8 * gran
                     || !pivcoh__jl_order(lam,
                                          kap + (gran == 8 ? 3 : gran == 4 ? 2 : 1),
                                          8 - (gran == 8 ? 3 : gran == 4 ? 2 : 1),
                                          obuf, &on)))
        gran = 1;
    const int glog = gran == 8 ? 3 : gran == 4 ? 2 : gran == 2 ? 1 : 0;
    int sigma_pad = sigma;
    if (glog) {
        const int pad = (gran - (sigma % gran)) % gran;
        int added = 0;
        for (int s = 0; s < 256 && added < pad; s++)
            if (!freq[s]) {
                sf[sigma_pad].freq = 0; sf[sigma_pad].sym = (uint16_t)s;
                P[sigma_pad + 1] = P[sigma];
                sigma_pad++; added++;
            }
        if (added < pad) return -1;      /* unreachable: pad <= 256-sigma */
    }

    uint16_t BL[PIVCOH__MAXLEN + 1] = {0};
    if (glog) {
        double Pg[130];
        const int sp = sigma_pad / gran;
        for (int i = 0; i <= sp; i++) Pg[i] = P[i * gran];
        uint16_t BLc[PIVCOH__MAXLEN + 1] = {0};
        /* kap + glog: local b' prices the real depth b' + glog; a
         * grouped b' = 0 take is a real 2^glog flat, hence tc1 twice */
        if (pivcoh__jl_slots(Pg, sp, lam, PIVCOH__MAXLEN - glog, 8 - glog,
                             kap + glog, tc1, tc1, BLc, scratch) < 0)
            return -1;
        for (int L = 1; L <= PIVCOH__MAXLEN - glog; L++)
            BL[L + glog] = (uint16_t)(BLc[L] << glog);
    } else if (pivcoh__jl_slots(P, sigma, lam, PIVCOH__MAXLEN, 8, kap,
                                tc0, tc1, BL, scratch) < 0) {
        return -1;      /* order condition failed (lambda > 1/7) or OOM */
    }

    /* Collect the chosen chunks in GLOBAL per-occurrence cost order —
     * under kappa the plain "L ascending, b descending" deal is no
     * longer the cost order, and the sorted matching the solvers assume
     * must be the assignment we actually realize. */
    struct { double cost; uint8_t L, b; uint16_t size; } chunks[40];
    int nchunks = 0;
    for (int L = 1; L <= PIVCOH__MAXLEN; L++)
        for (int b = 0; b <= 8; b++)
            if (BL[L] & (1 << b)) {
                double c = (double)L + lam * ((double)(L - b) + kap[b]);
                int i = nchunks++;
                while (i > 0 && (chunks[i - 1].cost > c
                                 || (chunks[i - 1].cost == c
                                     && (chunks[i - 1].L > L
                                         || (chunks[i - 1].L == L
                                             && chunks[i - 1].b < b))))) {
                    chunks[i] = chunks[i - 1];
                    i--;
                }
                chunks[i].cost = c;
                chunks[i].L    = (uint8_t)L;
                chunks[i].b    = (uint8_t)b;
                chunks[i].size = (uint16_t)(1 << b);
            }

    /* Deal freq-sorted symbols to the chunks in that same order, into
     * a CANDIDATE lens image (the caller's lens hold the baseline until
     * the guard passes).  Ghosts (sorted last) take the dearest chunks:
     * unused byte values receive real codes the encoder never emits.
     * dp_bits is exact off the deal — bits depend only on per-symbol
     * length, which the rebuild preserves. */
    uint8_t cand[256];
    double dp_bits = 0, dp_time;
    memcpy(cand, lens, 256);
    {
        int cur = 0;
        for (int i = 0; i < nchunks; i++) {
            dp_bits += (P[cur + chunks[i].size] - P[cur]) * chunks[i].L;
            for (int j = 0; j < chunks[i].size; j++)
                cand[sf[cur++].sym] = chunks[i].L;
        }
        if (cur != sigma_pad) return -1;
    }
    /* Apply the adoption guard on the tables the decoder will actually
     * build: the deal's heavy-to-cheap matching is not realizable (the
     * rebuild redistributes a class's symbols over its chunks in symbol
     * order), so both sides price the realized weights.  Ghost chunks
     * carry zero weight, so real symbols are scored exactly. */
    double base_bits = 0, base_time;
    {
        pivcoh__jl_ch chb[40], chc[40];
        int cntc[16] = {0}, nb, nc;
        for (int i = 0; i < nchunks; i++)
            cntc[chunks[i].L] += chunks[i].size;
        pivcoh__jl_realized2(lens, cand, cntb, cntc, freq,
                             chb, &nb, chc, &nc);
        for (int i = 0; i < nb; i++)
            base_bits += chb[i].W * (double)(chb[i].r + chb[i].D);
        base_time = pivcoh__jl_time(chb, nb, &jv, kap, P[sigma]);
        dp_time   = pivcoh__jl_time(chc, nc, &jv, kap, P[sigma]);
        if (base_time < 0 || dp_time < 0) return -1;
    }
    if (!(dp_time <= gtime * base_time && dp_bits <= gbits * base_bits))
        return -1;
    memcpy(lens, cand, 256);
    return 0;
}

static int pivcoh__from_freqs(pivcoh_table *t, const uint64_t freq[256],
                              const pivcoh_joint *jp, void *scratch)
{
    /* Frequencies narrow to u32 (and internal sums wrap mod 2^32): a
     * histogram totalling >= 4 GiB may derive different -- still valid,
     * still Kraft-exact, but no longer production-identical -- code
     * lengths.  Correctness-only: every index below is bounded
     * structurally, never by frequency values. */
    pivcoh__leaf leaf[256];
    uint32_t orv = 0, andv = ~(uint32_t)0;
    int n = 0, i;
    for (i = 0; i < 256; i++)
        if (freq[i]) {
            uint32_t f = (uint32_t)freq[i];
            leaf[n].freq = f;
            leaf[n].sym  = (uint16_t)i;
            n++;
            orv |= f;
            andv &= f;
        }
    if (n == 0) return 0;
    pivcoh__sort_leaves(leaf, n, orv ^ andv);

    uint8_t lens[256] = {0};
    if (n == 1) {
        lens[leaf[0].sym] = 1;
    } else {
        /* van Leeuwen two-queue: sorted leaves + FIFO of made internals */
        uint32_t nf[512];
        int16_t parent[512];
        int li = 0, ih = n, ni = n, rem;
        for (i = 0; i < n; i++) nf[i] = leaf[i].freq;
        for (rem = n; rem > 1; rem--) {
            int a, b;
            if (li < n && (ih == ni || nf[li] <= nf[ih])) a = li++; else a = ih++;
            if (li < n && (ih == ni || nf[li] <= nf[ih])) b = li++; else b = ih++;
            nf[ni] = nf[a] + nf[b];
            parent[a] = parent[b] = (int16_t)ni++;
        }
        uint8_t depth[512];
        int maxd = 1;
        depth[ni - 1] = 0;
        for (i = ni - 2; i >= 0; i--) depth[i] = (uint8_t)(depth[parent[i]] + 1);
        for (i = 0; i < n; i++) {
            uint8_t L = depth[i] ? depth[i] : 1;
            lens[leaf[i].sym] = L;
            if (L > maxd) maxd = L;
        }

        if (maxd > PIVCOH__MAXLEN) {           /* DEFLATE-style length limiting */
            /* Clamp to MAXLEN, then repair the Kraft sum: in units of
             * 2^-MAXLEN a clamped symbol weighs 1 instead of < 1, so
             * 0 < debt < cnt[MAXLEN].  Each step re-homes one MAXLEN
             * symbol as the sibling of a symbol demoted from the deepest
             * shorter level: net -1 unit, so the loop lands on Kraft == 1
             * exactly.  A non-empty b always exists: 256 symbols all at
             * MAXLEN would be under-subscribed. */
            int cnt[PIVCOH__MAXLEN + 2] = {0}, b, L;
            for (i = 0; i < 256; i++)
                if (lens[i]) cnt[lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN]++;
            /* reassign order: stable by (capped old length, symbol) — one
             * counting-sort scatter off the pre-repair histogram.  Comes
             * first: the repair below rewrites the histogram and the
             * assignments rewrite the sort keys.  (This replaces an
             * 11x256 order-building sweep that dominated the build on
             * exactly the windows deep enough to need limiting.) */
            uint8_t order[256];
            int cur[PIVCOH__MAXLEN + 1], no = 0;
            for (L = 1; L <= PIVCOH__MAXLEN; L++) { cur[L] = no; no += cnt[L]; }
            for (i = 0; i < 256; i++)
                if (lens[i])
                    order[cur[lens[i] < PIVCOH__MAXLEN ? lens[i] : PIVCOH__MAXLEN]++] = (uint8_t)i;
            long kraft = 0;
            for (i = 1; i <= PIVCOH__MAXLEN; i++) kraft += (long)cnt[i] << (PIVCOH__MAXLEN - i);
            for (; kraft > 1L << PIVCOH__MAXLEN; kraft--) {
                for (b = PIVCOH__MAXLEN - 1; !cnt[b]; b--) {}
                cnt[b]--;
                cnt[b + 1] += 2;
                cnt[PIVCOH__MAXLEN]--;
            }
            int cl = 1, left = cnt[1];
            for (i = 0; i < no; i++) {
                while (!left) left = cnt[++cl];
                lens[order[i]] = (uint8_t)cl;
                left--;
            }
        }
        /* Optional joint length/shape pass (encoder side only; the
         * decoder rebuilds identically from the transmitted lengths).
         * Any internal reject keeps the Huffman lengths above.  leaf[]
         * is free again after the two-queue — the pass reverses it in
         * place and may append zero-frequency ghosts. */
        if (jp && jp->lambda > 0.0f)
            (void)pivcoh__jl_core(leaf, n, freq, lens, jp, scratch);
    }
    return pivcoh_table_from_lens(t, lens);    /* lengths -> schedule (shared) */
}

PIVCOHDEF int pivcoh_table_from_freqs(pivcoh_table *t, const uint64_t freq[256])
{
    return pivcoh__from_freqs(t, freq, NULL, NULL);
}

PIVCOHDEF int pivcoh_table_from_freqs_joint(pivcoh_table *t,
                                            const uint64_t freq[256],
                                            const pivcoh_joint *j,
                                            void *scratch)
{
    return pivcoh__from_freqs(t, freq, j, scratch);
}

/* ================= decode kernels (ports of primitives_neon) ================
 *
 * Buffer contract: a merge kernel may read up to 16 bytes past a source
 * cursor and — interior (non-EXACT) merges only — overwrite up to 15
 * bytes past out+K, saved and restored around the merge.  Validation is
 * the end-of-merge r_end equality: cursors are monotone, so final
 * r == r_end proves no prefix of the bitmap ever overdrew either side
 * (each output consumes exactly one input, so the l side is implied) —
 * an exact "bitmap popcount == K_right" test for one compare per merge,
 * nothing per iteration (in-loop guards were tried and cost ~2%).  On a
 * stream that fails it the cursors strayed mid-merge first: by at most
 * K bytes past a side plus the 64-byte iteration window, absorbed by
 * the decode arena's pad.  Writes of EXACT merges (the root, targeting
 * the caller's buffer) are exactly bounded to out[0,K).  Bitmap reads
 * never pass ceil(K/8) bytes, flat-region reads never pass ceil(K*D/8)
 * bytes.
 */

/* Two-table SABD merge shuffles (8 KiB — the only merge tables: every
 * merge tail is a 16-byte SABD chunk too, so the old stride-16/8
 * expand-tab ladder and its ~21 KiB of tables are gone). */
static int8_t  pivcoh__mshuf0[256 * 16]     __attribute__((aligned(16)));
static int8_t  pivcoh__mshuf1[256 * 16]     __attribute__((aligned(16)));

static void pivcoh__init_dec(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
        int8_t pop = 0;
        int8_t *o0 = &pivcoh__mshuf0[m * 16], *o1 = &pivcoh__mshuf1[m * 16];
        for (int k = 0; k < 8; k++) {
            if ((m >> k) & 1) {
                o0[k] = pop; o1[k + 8] = (int8_t)(-pop); pop++;
            } else {
                int8_t v = (int8_t)(-16 - k + pop);
                o0[k] = v; o1[k + 8] = (int8_t)(8 - v);
            }
        }
        for (int k = 0; k < 8; k++) { o0[k + 8] = pop; o1[k] = 0; }
    }
    built = 1;
}

/* One 16-byte merge: 2-source TBL over {R,L}, cross-half cursor offset
 * folded into the index by SABD (|shuf0 - shuf1|). */
static inline void pivcoh__merge16(uint8_t *dest, const uint8_t *l_list,
                                   const uint8_t *r_list, uint64_t mask)
{
    int8x16_t s0 = vld1q_s8(&pivcoh__mshuf0[(mask << 4) & 0xff0]);
    int8x16_t s1 = vld1q_s8(&pivcoh__mshuf1[(mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r_list);
    src.val[1] = vld1q_u8(l_list);
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}

/* Same, with the L source in a broadcast register (LEAF_LEFT). */
static inline void pivcoh__merge16cst(uint8_t *dest, uint8x16_t Lb,
                                      const uint8_t *r_list, uint64_t mask)
{
    int8x16_t s0 = vld1q_s8(&pivcoh__mshuf0[(mask << 4) & 0xff0]);
    int8x16_t s1 = vld1q_s8(&pivcoh__mshuf1[(mask >> 4) & 0xff0]);
    uint8x16_t shuf = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    uint8x16x2_t src;
    src.val[0] = vld1q_u8(r_list);
    src.val[1] = Lb;
    vst1q_u8(dest, vqtbl2q_u8(src, shuf));
}

/* Load the final partial chunk's mask (rem = K mod 16 bits, 1..15),
 * trimmed to the real bits: trimming keeps the cursor advance exact
 * (the r_end equality check depends on it), makes the phantom lanes
 * select harmless left/register bytes, and never touches a bitmap
 * byte past ceil(K/8). */
static inline unsigned pivcoh__tailmask(const uint8_t *bm, int j, unsigned rem)
{
    unsigned mask = bm[j >> 3];
    if (rem > 8) mask |= (unsigned)bm[(j >> 3) + 1] << 8;
    return mask & ((1u << rem) - 1);
}

/* merge_vec_vec: 64 bytes/iter main loop — four 16-byte SABD chunks
 * share one vcnt + 64-bit-multiply prefix sum for the per-chunk cursor
 * splits and the L/R advance — then 16-byte chunks.
 *
 * The loop is software-pipelined one iteration deep: the carried chain
 * (bitmap load -> vcnt -> lane move -> multiply -> cursor advance) is
 * ~12 cycles of latency against ~16 cycles of work, so each iteration
 * starts the NEXT mask/prefix up front and the chain resolves under
 * the current merges.  The popcount path loads its own copy of the
 * bitmap straight into SIMD (a GPR->SIMD fmov costs a load-port uop on
 * Apple and would sit mid-chain).  Byte k of pfx = sum of the mask's
 * byte-popcounts 0..k: bytes 1/3/5 are the 16-bit chunk boundaries,
 * byte 7 the total.  Store cadence is unchanged — the 128B-unroll
 * shape that won microbenches but lost e2e to streaming effects is
 * deliberately avoided (micro: +7% L1, +4% streaming as-is).
 *
 * EXACT=0 (interior nodes, arena-backed out): entirely SIMD, no scalar
 * tail.  The final partial chunk runs mask-trimmed at full 16-byte
 * width, overwriting up to 15 bytes past out+K; the 16 bytes there are
 * saved up front and restored after.  EXACT=1 (the root merge, which
 * targets the caller's buffer): whole chunks while they fit, then a
 * plain scalar tail (<= 15 elements, once per block).
 *
 * Both variants validate at the end — see the section comment.  Returns
 * 0, or -1 when the bitmap contradicts the K_right header. */
#define PIVCOH__PFX8(p) (vget_lane_u64(vreinterpret_u64_u8(                \
                             vcnt_u8(vld1_u8(p))), 0) * 0x0101010101010101ull)
__attribute__((always_inline)) static inline
int pivcoh__mvv(const uint8_t *bm, int K, const uint8_t *l, int KL,
                const uint8_t *r, int KR, uint8_t *out, int EXACT)
{
    const uint8_t *l_end = l + KL, *r_end = r + KR;
    uint8x16_t keep = vdupq_n_u8(0);
    if (!EXACT) keep = vld1q_u8(out + K);
    intptr_t i = 0;
#define PIVCOH__MVV4(mask, pfx) do {                                       \
        intptr_t p0 = ((pfx) >> 8) & 0xff, p1 = ((pfx) >> 24) & 0xff,      \
                 p2 = ((pfx) >> 40) & 0xff, p3 = (pfx) >> 56;              \
        pivcoh__merge16(out + i,      l,           r,      (mask));        \
        pivcoh__merge16(out + i + 16, l + 16 - p0, r + p0, (mask) >> 16);  \
        pivcoh__merge16(out + i + 32, l + 32 - p1, r + p1, (mask) >> 32);  \
        pivcoh__merge16(out + i + 48, l + 48 - p2, r + p2, (mask) >> 48);  \
        r += p3; l += 64 - p3;                                             \
    } while (0)
    if (i + 64 <= K) {
        uint64_t mask; memcpy(&mask, bm, 8);
        uint64_t pfx = PIVCOH__PFX8(bm);
        for (; i + 128 <= K; i += 64) {
            uint64_t nmask; memcpy(&nmask, bm + ((i + 64) >> 3), 8);
            uint64_t npfx = PIVCOH__PFX8(bm + ((i + 64) >> 3));
            PIVCOH__MVV4(mask, pfx);
            mask = nmask; pfx = npfx;
        }
        PIVCOH__MVV4(mask, pfx);
        i += 64;
    }
#undef PIVCOH__MVV4
    int j = (int)i;
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        pivcoh__merge16(out + j, l, r, m16);
        int pt = __builtin_popcount(m16);
        r += pt; l += 16 - pt;
    }
    if (!EXACT) {
        if (j < K) {
            unsigned mask = pivcoh__tailmask(bm, j, (unsigned)(K - j));
            pivcoh__merge16(out + j, l, r, mask);
            r += __builtin_popcount(mask);
        }
        vst1q_u8(out + K, keep);
    } else {
        for (; j < K; j++) {
            if ((bm[j >> 3] >> (j & 7)) & 1) {
                if (r == r_end) return -1;
                out[j] = *r++;
            } else {
                if (l == l_end) return -1;
                out[j] = *l++;
            }
        }
    }
    return r == r_end ? 0 : -1;
}
static int pivcoh__merge_vec_vec(const uint8_t *bm, int K,
                                 const uint8_t *l, int KL,
                                 const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mvv(bm, K, l, KL, r, KR, out, 0); }
static int pivcoh__merge_vec_vec_x(const uint8_t *bm, int K,
                                   const uint8_t *l, int KL,
                                   const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mvv(bm, K, l, KL, r, KR, out, 1); }
#undef PIVCOH__PFX8

/* merge_cst_vec: L is a broadcast constant (LEAF_LEFT) — no L load or
 * cursor; only the R cursor advances (and is guarded/validated).  Same
 * EXACT/tail-free split as pivcoh__mvv. */
__attribute__((always_inline)) static inline
int pivcoh__mcv(const uint8_t *bm, int K, uint8_t left_sym,
                const uint8_t *r, int KR, uint8_t *out, int EXACT)
{
    const uint8_t *r_end = r + KR;
    uint8x16_t Lb = vdupq_n_u8(left_sym);
    uint8x16_t keep = vdupq_n_u8(0);
    if (!EXACT) keep = vld1q_u8(out + K);
    intptr_t i = 0;
    for (; i + 64 <= K; i += 64) {
        uint64_t mask; memcpy(&mask, bm + (i >> 3), 8);
        uint8x8_t pop8 = vcnt_u8(vcreate_u8(mask));
        uint64_t pfx = vget_lane_u64(vreinterpret_u64_u8(pop8), 0) * 0x0101010101010101ull;
        intptr_t p0 = (pfx >> 8) & 0xff, p1 = (pfx >> 24) & 0xff, p2 = (pfx >> 40) & 0xff;
        pivcoh__merge16cst(out + i,      Lb, r,      mask);
        pivcoh__merge16cst(out + i + 16, Lb, r + p0, mask >> 16);
        pivcoh__merge16cst(out + i + 32, Lb, r + p1, mask >> 32);
        pivcoh__merge16cst(out + i + 48, Lb, r + p2, mask >> 48);
        r += pfx >> 56;
    }
    int j = (int)i;
    for (; j + 16 <= K; j += 16) {
        uint16_t m16; memcpy(&m16, bm + (j >> 3), 2);
        pivcoh__merge16cst(out + j, Lb, r, m16);
        r += __builtin_popcount(m16);
    }
    if (!EXACT) {
        if (j < K) {
            unsigned mask = pivcoh__tailmask(bm, j, (unsigned)(K - j));
            pivcoh__merge16cst(out + j, Lb, r, mask);
            r += __builtin_popcount(mask);
        }
        vst1q_u8(out + K, keep);
    } else {
        for (; j < K; j++) {
            if ((bm[j >> 3] >> (j & 7)) & 1) {
                if (r == r_end) return -1;
                out[j] = *r++;
            } else out[j] = left_sym;
        }
    }
    return r == r_end ? 0 : -1;
}
static int pivcoh__merge_cst_vec(const uint8_t *bm, int K, uint8_t left_sym,
                                 const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mcv(bm, K, left_sym, r, KR, out, 0); }
static int pivcoh__merge_cst_vec_x(const uint8_t *bm, int K, uint8_t left_sym,
                                   const uint8_t *r, int KR, uint8_t *out)
{ return pivcoh__mcv(bm, K, left_sym, r, KR, out, 1); }

/* ---- flat-subtree D-bit decode helpers ----
 *
 * Every flat kernel below has two tails, chosen by `tf` (tail-free):
 * the walk passes tf=1 when the input holds >= 16 readable bytes past
 * the flat region AND out is arena-backed — which is (almost) every
 * interior flat region: in decode order the last bytes of a block are
 * the ROOT's merge bitmap, so an interior region is followed by its
 * ancestors' records (a parent's bitmap alone exceeds 16 bytes once
 * the region holds ~120 symbols).  tf kernels simply run their MAIN
 * loop past n — same shape, same constants, no separate tail pipeline —
 * reading at most 16 bytes past the region (inside the stream) and
 * scribbling up to one iteration's width minus one (15..63 bytes,
 * kernel-dependent) past out+n, saved and restored at loop width — so
 * the byte-wise "safe" unpacks and per-code scalar tails are gone.
 * tf=0 (a flat ROOT decoding into the caller's buffer — where the
 * region really can end the stream — or a rare end-of-block interior
 * region) keeps the region-bounded vector loops and finishes the last
 * few codes with the scalar extractor below. */

static inline uint32_t pivcoh__extract_bits(const uint8_t *in, int bit_pos, int D)
{
    int byte_idx = bit_pos >> 3, bit_off = bit_pos & 7;
    uint32_t val = (uint32_t)in[byte_idx];
    if (bit_off + D > 8) val |= ((uint32_t)in[byte_idx + 1]) << 8;
    return (val >> bit_off) & ((1u << D) - 1);
}

static const uint8_t pivcoh__d7_shuf_tab[16] = {0,1, 0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,6};
static const int16_t pivcoh__d7_shift_tab[8] = {0, -7, -6, -5, -4, -3, -2, -1};
static inline uint8x8_t pivcoh__d7_unpack(const uint8_t *bm_ptr)
{
    uint8x16_t bm_lo = vld1q_u8(bm_ptr);
    uint16x8_t w = vreinterpretq_u16_u8(vqtbl1q_u8(bm_lo, vld1q_u8(pivcoh__d7_shuf_tab)));
    uint16x8_t shifted = vshlq_u16(w, vld1q_s16(pivcoh__d7_shift_tab));
    return vmovn_u16(vandq_u16(shifted, vdupq_n_u16(0x7F)));
}

/* One 16-output pair-gather chunk for the byte-crossing depths: a
 * 16-byte load, one vqtbl1 placing two adjacent codes in each u16 lane,
 * a u16 shift aligning the pair, a u8 shift + mask isolating each code,
 * then the c2s scatter.  Consumes 2D input bytes of the 16 loaded.
 * These ARE the D=5/6 main-loop bodies — main loop and tf tail share
 * one shape and one constant set per kernel. */
static inline void pivcoh__d5_chunk16(uint8_t *dst, const uint8_t *src,
                                      uint8x16x2_t c2s_vec)
{
    static const uint8_t pair_shuf_t[16] = { 0,1, 1,2, 2,3, 3,4, 5,6, 6,7, 7,8, 8,9 };
    static const int16_t hshift_t[8]     = { 3, 1, -1, -3, 3, 1, -1, -3 };
    static const int8_t  bshr_t[16]      = { -3,0, -3,0, -3,0, -3,0, -3,0, -3,0, -3,0, -3,0 };
    uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(src), vld1q_u8(pair_shuf_t)));
    x = vshlq_u16(x, vld1q_s16(hshift_t));
    uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), vld1q_s8(bshr_t));
    vst1q_u8(dst, vqtbl2q_u8(c2s_vec, vandq_u8(y, vdupq_n_u8(0x1f))));
}

static inline void pivcoh__d6_chunk16(uint8_t *dst, const uint8_t *src,
                                      uint8x16x4_t c2s_vec)
{
    static const uint8_t pair_shuf_t[16] = { 0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11 };
    static const int16_t hshift_t[8]     = { 2,-2, 2,-2, 2,-2, 2,-2 };
    static const int8_t  bshr_t[16]      = { -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0 };
    uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(vld1q_u8(src), vld1q_u8(pair_shuf_t)));
    x = vshlq_u16(x, vld1q_s16(hshift_t));
    uint8x16_t y = vshlq_u8(vreinterpretq_u8_u16(x), vld1q_s8(bshr_t));
    vst1q_u8(dst, vqtbl4q_u8(c2s_vec, vandq_u8(y, vdupq_n_u8(0x3f))));
}

/* ---- per-D flat decodes (contiguous output) ---- */

/* D=1 (8 codes/byte): each output lane bit-selects between the two
 * broadcast symbols — dup-shuffle the 2 bitmap bytes across the lanes,
 * CMTST each lane's own bit, BSL the symbols (one op less than the
 * old shift+mask+lookup, though the kernel is store-bound either way).
 * The tf chunks read <= 1 byte past the region (inside the stream);
 * tf=0 (a 2-symbol flat root) finishes scalar. */
static const uint8_t pivcoh__d1_dup_tab[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
static const uint8_t pivcoh__d1_bit_tab[16] = {1,2,4,8,16,32,64,128,
                                               1,2,4,8,16,32,64,128};
static void pivcoh__flat_d1(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16_t Lv    = vdupq_n_u8(c2s[0]);
    uint8x16_t Rv    = vdupq_n_u8(c2s[1]);
    uint8x16_t dup_v = vld1q_u8(pivcoh__d1_dup_tab);
    uint8x16_t bit_v = vld1q_u8(pivcoh__d1_bit_tab);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);

    int j = 0, lim = tf ? n : n - 15;
    for (; j < lim; j += 16) {
        uint16_t bm_word; memcpy(&bm_word, bm + (j >> 3), 2);
        uint8x16_t bm_lo = vreinterpretq_u8_u16(
            vsetq_lane_u16(bm_word, vdupq_n_u16(0), 0));
        uint8x16_t dup = vqtbl1q_u8(bm_lo, dup_v);
        vst1q_u8(out + j, vbslq_u8(vtstq_u8(dup, bit_v), Rv, Lv));
    }
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; j < n; j++)
        out[j] = c2s[(bm[j >> 3] >> (j & 7)) & 1];
}

/* D=2 (4 codes/byte): 64/iter maps each input nibble straight to a
 * symbol pair via two prepped tables, interleaved back with vst4q.
 * The tf tail is the same loop run past n (scribbling < 64 bytes past
 * out+n, saved and restored) — one shape, one constant set. */
static void pivcoh__flat_d2(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    int i = 0;
    if (tf || n >= 64) {
        static const uint8_t th_idx[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        uint32_t w; memcpy(&w, c2s, 4);
        const uint8x16_t TL = vreinterpretq_u8_u32(vdupq_n_u32(w));   /* c2s[n&3] */
        const uint8x16_t TH = vqtbl1q_u8(TL, vld1q_u8(th_idx));       /* c2s[(n>>2)&3] */
        const uint8x16_t m  = vdupq_n_u8(0x0F);
        uint8x16_t k0 = vdupq_n_u8(0), k1 = k0, k2 = k0, k3 = k0;
        if (tf) {
            k0 = vld1q_u8(out + n);      k1 = vld1q_u8(out + n + 16);
            k2 = vld1q_u8(out + n + 32); k3 = vld1q_u8(out + n + 48);
        }
        int lim = tf ? n : n - 63;
        for (; i < lim; i += 64) {
            uint8x16_t v  = vld1q_u8(bm + (i >> 2));
            uint8x16_t lo = vandq_u8(v, m), hi = vshrq_n_u8(v, 4);
            uint8x16x4_t o = {{ vqtbl1q_u8(TL, lo), vqtbl1q_u8(TH, lo),
                                vqtbl1q_u8(TL, hi), vqtbl1q_u8(TH, hi) }};
            vst4q_u8(out + i, o);
        }
        if (tf) {
            vst1q_u8(out + n, k0);      vst1q_u8(out + n + 16, k1);
            vst1q_u8(out + n + 32, k2); vst1q_u8(out + n + 48, k3);
            return;
        }
    }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 2, 2)];
}

/* D=3: 32 codes/iter via the D=6 pair-gather (two D=3 codes per byte),
 * split lo/hi and interleave with vst2q.  The tf tail is the same loop
 * run past n (scribbling < 32 bytes past out+n, saved and restored;
 * the tf=0 main-loop bound i+48 <= n keeps its 16-byte loads inside
 * ceil(3n/8) — production guards only against the output and can read
 * 4 bytes past the region). */
static void pivcoh__flat_d3(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    int i = 0;
    if (tf || n >= 48) {
        const uint8x8_t  c2s8  = vld1_u8(c2s);
        const uint8x16_t c2s16 = vcombine_u8(c2s8, c2s8);
        const uint8x16_t m7    = vdupq_n_u8(7);
        uint8x16x2_t c2s32; c2s32.val[0] = c2s16; c2s32.val[1] = c2s16;
        static const uint8_t pair6_shuf_t[16] = { 0,1, 1,2, 3,4, 4,5, 6,7, 7,8, 9,10, 10,11 };
        static const int16_t hshift6_t[8]     = { 2,-2, 2,-2, 2,-2, 2,-2 };
        static const int8_t  bshr6_t[16]      = { -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0, -2,0 };
        const uint8x16_t pair6_shuf = vld1q_u8(pair6_shuf_t);
        const int16x8_t  hshift6    = vld1q_s16(hshift6_t);
        const int8x16_t  bshr6      = vld1q_s8(bshr6_t);
        uint8x16_t k0 = vdupq_n_u8(0), k1 = k0;
        if (tf) { k0 = vld1q_u8(out + n); k1 = vld1q_u8(out + n + 16); }
        int lim = tf ? n : n - 47;
        const uint8_t *bp = bm;
        for (; i < lim; i += 32, bp += 12) {
            uint8x16_t packed = vld1q_u8(bp);
            uint16x8_t x = vreinterpretq_u16_u8(vqtbl1q_u8(packed, pair6_shuf));
            x = vshlq_u16(x, hshift6);
            uint8x16_t pair6 = vshlq_u8(vreinterpretq_u8_u16(x), bshr6);
            uint8x16x2_t o;
            o.val[0] = vqtbl1q_u8(c2s16, vandq_u8(pair6, m7));
            o.val[1] = vqtbl2q_u8(c2s32, vshrq_n_u8(pair6, 3));
            vst2q_u8(out + i, o);
        }
        if (tf) {
            vst1q_u8(out + n, k0);
            vst1q_u8(out + n + 16, k1);
            return;
        }
    }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 3, 3)];
}

/* D=4: nibbles index c2s directly; 32/iter via vzip.  The tf tail is
 * the same loop run past n (scribbling < 32 bytes past out+n, saved
 * and restored) — one shape, one constant set. */
static void pivcoh__flat_d4(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16_t c2s_vec = vld1q_u8(c2s);
    const uint8x16_t m = vdupq_n_u8(0x0F);
    uint8x16_t k0 = vdupq_n_u8(0), k1 = k0;
    if (tf) { k0 = vld1q_u8(out + n); k1 = vld1q_u8(out + n + 16); }
    int i = 0, lim = tf ? n : n - 31;
    for (; i < lim; i += 32) {
        uint8x16_t v  = vld1q_u8(bm + (i >> 1));
        uint8x16_t lo = vandq_u8(v, m), hi = vshrq_n_u8(v, 4);
        uint8x16_t a = vqtbl1q_u8(c2s_vec, lo), b = vqtbl1q_u8(c2s_vec, hi);
        vst1q_u8(out + i,      vzip1q_u8(a, b));
        vst1q_u8(out + i + 16, vzip2q_u8(a, b));
    }
    if (tf) {
        vst1q_u8(out + n, k0);
        vst1q_u8(out + n + 16, k1);
        return;
    }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 4, 4)];
}

/* D=5: pair-gather chunks (two adjacent codes per u16 lane, positioned
 * so the byte reinterpret interleaves even/odd for free); vqtbl2
 * scatter.  tf=0 keeps the region-safe block count. */
static void pivcoh__flat_d5(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16x2_t c2s_vec = vld1q_u8_x2(c2s);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);
    int i = 0, lim = tf ? n : (n >= 25 ? ((n - 9) >> 4) << 4 : 0);
    for (; i < lim; i += 16)
        pivcoh__d5_chunk16(out + i, bm + ((i * 5) >> 3), c2s_vec);
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 5, 5)];
}

/* D=6: same pair-gather as D=5 (12-bit pairs); 64-byte c2s => vqtbl4q. */
static void pivcoh__flat_d6(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16x4_t c2s_vec = vld1q_u8_x4(c2s);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);
    int i = 0, lim = tf ? n : (n >= 24 ? ((n - 8) >> 4) << 4 : 0);
    for (; i < lim; i += 16)
        pivcoh__d6_chunk16(out + i, bm + ((i * 6) >> 3), c2s_vec);
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 6, 6)];
}

/* D=7: 128-entry c2s = vqtbl4 low half + vqtbx4 high half (vqtbx keeps
 * the first result for out-of-range lanes — no OR-merge).  16-wide
 * while whole chunks fit, 8-wide finish. */
static void pivcoh__flat_d7(uint8_t *out, int n, const uint8_t *bm,
                            const uint8_t *c2s, int tf)
{
    uint8x16x4_t lo = vld1q_u8_x4(c2s), hi = vld1q_u8_x4(c2s + 64);
    uint8x16_t sub64q = vdupq_n_u8(64);
    uint8x8_t  sub64  = vdup_n_u8(64);
    uint8x16_t keep = vdupq_n_u8(0);
    if (tf) keep = vld1q_u8(out + n);
    int i = 0, fast_end = tf ? n : (n >= 24 ? n - 24 : 0);
    for (; i < fast_end - 15; i += 16) {
        uint8x8_t cl = pivcoh__d7_unpack(bm + ((i      * 7) >> 3));
        uint8x8_t ch = pivcoh__d7_unpack(bm + (((i + 8) * 7) >> 3));
        uint8x16_t codes = vcombine_u8(cl, ch);
        uint8x16_t s = vqtbl4q_u8(lo, codes);
        s = vqtbx4q_u8(s, hi, vsubq_u8(codes, sub64q));
        vst1q_u8(out + i, s);
    }
    for (; i < (tf ? n : fast_end - 7); i += 8) {
        uint8x8_t codes = pivcoh__d7_unpack(bm + ((i * 7) >> 3));
        uint8x8_t s = vqtbl4_u8(lo, codes);
        s = vqtbx4_u8(s, hi, vsub_u8(codes, sub64));
        vst1_u8(out + i, s);
    }
    if (tf) { vst1q_u8(out + n, keep); return; }
    for (; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * 7, 7)];
}

/* Dispatcher.  D=8 is a full-alphabet equal-length code: c2s is the
 * identity, so the byte-aligned codes ARE the symbols — memcpy (exact
 * either way; tf is moot). */
static void pivcoh__merge_flat(uint8_t *out, int n, const uint8_t *bm, int D,
                               const uint8_t *c2s, int tf)
{
    switch (D) {
    case 1: pivcoh__flat_d1(out, n, bm, c2s, tf); break;
    case 2: pivcoh__flat_d2(out, n, bm, c2s, tf); break;
    case 3: pivcoh__flat_d3(out, n, bm, c2s, tf); break;
    case 4: pivcoh__flat_d4(out, n, bm, c2s, tf); break;
    case 5: pivcoh__flat_d5(out, n, bm, c2s, tf); break;
    case 6: pivcoh__flat_d6(out, n, bm, c2s, tf); break;
    case 7: pivcoh__flat_d7(out, n, bm, c2s, tf); break;
    case 8: memcpy(out, bm, (size_t)n); break;
    default:
        for (int i = 0; i < n; i++) out[i] = c2s[pivcoh__extract_bits(bm, i * D, D)];
        break;
    }
}

/* ---- decode tree walk ---- */

/* Read a node's post-order raw bitmap: sets *bm and returns the input
 * pointer advanced past it, or NULL on truncation.  (v4: the v3 per-
 * node FSE marker byte is gone — there is no FSE mode to mark.) */
static inline const uint8_t *pivcoh__read_bm(const uint8_t **bm, int K,
                                             const uint8_t *p, const uint8_t *end)
{
    size_t nb = (size_t)((K + 7) >> 3);
    if ((size_t)(end - p) < nb) return NULL;
    *bm = p;
    return p + nb;
}

/* Read a node's K_right header: one byte when the node's K fits one,
 * two (LE) otherwise — the decoder always knows K at node entry.
 * Returns the advanced pointer or NULL on truncation / KR > K. */
static inline const uint8_t *pivcoh__read_kr(int *KR, int K,
                                             const uint8_t *p, const uint8_t *end)
{
    int kr;
    if (K > 255) {
        if (end - p < 2) return NULL;
        kr = p[0] | p[1] << 8;
        p += 2;
    } else {
        if (end - p < 1) return NULL;
        kr = p[0];
        p += 1;
    }
    if (kr > K) return NULL;
    *KR = kr;
    return p;
}

/* Interior walk: decodes a subtree's K symbols into out[0,K), returning
 * the advanced input pointer or NULL on invalid input.  The wire is in
 * decode order (K_right at node entry, children larger-K first, the
 * marker+bitmap at the node's post-order position, read right at merge
 * time), so the input cursor moves strictly forward, single-touch.
 *
 * Scratch placement is a two-buffer ping-pong (out, tmp):
 *
 *   - the LARGER child decodes IN PLACE into out's tail out[K_small, K):
 *     safe under the merge, whose write cursor cannot overtake its
 *     tail-side read cursor — by the time it writes out[i] it has
 *     consumed at least i - K_small tail bytes (on a bitmap that lies,
 *     the merge's memory use stays inside the arena bound and the r_end
 *     check reports -1; whatever garbage was written is discarded);
 *   - the SMALLER child decodes into tmp[0, K_small), and its own
 *     recursion uses out's still-empty prefix out[0, K_small) as ITS
 *     partner — the pair (tmp, out-prefix) ping-pongs down the
 *     smaller-child spine instead of growing an arena.
 *
 * The caller guarantees tmp capacity floor(K/2): a smaller child is at
 * most floor(K/2), and everything its subtree puts in ITS partner stays
 * inside out[0, K_small), by induction.  A subtree's whole valid-stream
 * footprint is out[0,K) plus at most floor(K/2) partner bytes plus the
 * kernels' read/scribble slack; PIVCOH_DECODE_SCRATCH_SIZE's extra pad
 * absorbs a lying bitmap's bounded pre-detection strays (kernel section
 * comment).  A FLAT node never touches tmp (so a flat root runs
 * scratch-free). */
static const uint8_t *pivcoh__dec_internal(const pivcoh_table *t, int idx, int K,
                                           uint8_t *out, uint8_t *tmp,
                                           const uint8_t *p, const uint8_t *end);

/* Decode one child subtree.  Flat children are dispatched here in the
 * parent (no recursive entry).  Only the root's children come through
 * here; dec_internal handles its own descendants iteratively. */
static inline const uint8_t *pivcoh__dec_child(const pivcoh_table *t, int idx,
                                               int K, uint8_t *out, uint8_t *tmp,
                                               const uint8_t *p, const uint8_t *end)
{
    if (K == 0) return p;
    const pivcoh__rec *rec = &t->sched[idx];
    unsigned kd = rec->kd;
    if ((kd & 3) == PIVCOH__FLAT) {
        unsigned D = kd >> 2;
        size_t nb = ((size_t)K * (size_t)D + 7) >> 3;
        size_t avail = (size_t)(end - p);
        if (avail < nb) return NULL;
        pivcoh__merge_flat(out, K, p, (int)D, t->rank_to_sym + rec->param,
                           avail - nb >= 16);
        return p + nb;
    }
    return pivcoh__dec_internal(t, idx, K, out, tmp, p, end);
}

/* Iterative interior walk (exp 5): an explicit continuation stack replaces
 * recursion.  t, p, end and the rank map stay live in registers across the
 * loop instead of being reshuffled through call arguments; flat children
 * are decoded inline.  Two states — descend (open a node, start its big
 * child) and complete (a child finished: start the sibling, else read the
 * post-order bitmap, merge, pop).  Depth is bounded by the tree height. */
static const uint8_t *pivcoh__dec_internal(const pivcoh_table *t, int idx, int K,
                                           uint8_t *out, uint8_t *tmp,
                                           const uint8_t *p, const uint8_t *end)
{
    struct cont { int K, KR, KL, small_idx, small_K; uint8_t *out, *tmp;
                  uint8_t kind, phase, right_big, sym; } stk[PIVCOH__MAXLEN + 2];
    int sp = 0;
    int cidx = 0, cK = 0; uint8_t *cout = out, *ctmp = tmp;

descend:            /* (idx, K, out, tmp) is an INTERNAL subtree */
    if (sp >= (int)(sizeof stk / sizeof *stk)) return NULL;   /* defensive */
    {
        const pivcoh__rec *rec = &t->sched[idx];
        int KR;
        if (!(p = pivcoh__read_kr(&KR, K, p, end))) return NULL;
        int KL = K - KR;
        int kind = rec->kd & 3;
        struct cont *f = &stk[sp++];
        f->K = K; f->KR = KR; f->KL = KL; f->out = out; f->tmp = tmp;
        f->kind = (uint8_t)kind; f->phase = 0;
        if (kind == PIVCOH__LEAFL) {
            f->sym = t->rank_to_sym[rec->param];   /* the one child, in place */
            cidx = idx + rec->right; cK = KR; cout = out + KL; ctmp = tmp;
        } else {                                   /* FULL: larger child first */
            int rb = (KR > KL);
            f->right_big = (uint8_t)rb;
            f->small_idx = rb ? idx + 1 : idx + rec->right;
            f->small_K   = rb ? KL : KR;
            cidx = rb ? idx + rec->right : idx + 1;
            cK   = rb ? KR : KL;
            cout = out + (rb ? KL : KR);           /* big child in out's tail */
            ctmp = tmp;
        }
    }
try_child:          /* decode child (cidx, cK, cout, ctmp) */
    if (cK != 0) {
        const pivcoh__rec *crec = &t->sched[cidx];
        unsigned ckd = crec->kd;
        if ((ckd & 3) == PIVCOH__FLAT) {
            unsigned D = ckd >> 2;
            size_t nb = ((size_t)cK * (size_t)D + 7) >> 3;
            size_t avail = (size_t)(end - p);
            if (avail < nb) return NULL;
            pivcoh__merge_flat(cout, cK, p, (int)D, t->rank_to_sym + crec->param,
                               avail - nb >= 16);
            p += nb;
        } else {                                   /* internal: descend */
            idx = cidx; K = cK; out = cout; tmp = ctmp;
            goto descend;
        }
    }
    /* child finished: advance the frames that are now complete */
    while (sp > 0) {
        struct cont *f = &stk[sp - 1];
        if (f->kind == PIVCOH__FULL && f->phase == 0) {   /* big done; do small */
            f->phase = 1;
            cidx = f->small_idx; cK = f->small_K; cout = f->tmp; ctmp = f->out;
            goto try_child;
        }
        const uint8_t *bm;
        if (!(p = pivcoh__read_bm(&bm, f->K, p, end))) return NULL;
        if (f->kind == PIVCOH__LEAFL) {
            if (pivcoh__merge_cst_vec(bm, f->K, f->sym, f->out + f->KL, f->KR, f->out))
                return NULL;
        } else {
            int smallK = f->right_big ? f->KL : f->KR;
            uint8_t *big_out = f->out + smallK;
            uint8_t *lbuf = f->right_big ? f->tmp : big_out;
            uint8_t *rbuf = f->right_big ? big_out : f->tmp;
            if (pivcoh__merge_vec_vec(bm, f->K, lbuf, f->KL, rbuf, f->KR, f->out))
                return NULL;
        }
        sp--;
    }
    return p;
}

PIVCOHDEF ptrdiff_t pivcoh_decode(const pivcoh_table *t,
                                  const uint8_t *in, size_t in_len,
                                  uint8_t *out, size_t out_cap,
                                  size_t *consumed, void *scratch)
{
    if (!t || !in || !out || !t->num_ranks || in_len < 2) return -1;
    int N = in[0] | in[1] << 8;
    if (N < 1 || (size_t)N > out_cap) return -1;
    const uint8_t *end = in + in_len;
    const uint8_t *p = in + 2;
    const pivcoh__rec *root = &t->sched[0];
    int kind = root->kd & 3;

    if (kind == PIVCOH__FLAT) {                /* flat root: no scratch at all */
        int D = root->kd >> 2;
        size_t nb = ((size_t)N * (size_t)D + 7) >> 3;
        if ((size_t)(end - p) < nb) return -1;
        /* tf=0: out is the caller's buffer (no scribble allowed), and a
         * flat root's region really can end the stream. */
        pivcoh__merge_flat(out, N, p, D, t->rank_to_sym + root->param, 0);
        if (consumed) *consumed = (size_t)(p + nb - in);
        return N;
    }

    /* The interior walk's merges save/restore-scribble past out+K and
     * read their sources with 16 bytes of slack — guarantees the
     * caller's `out` doesn't offer.  So the root's children decode into
     * the arena, and only the root's own merge, whose writes are exact
     * (the EXACT `_x` kernels), targets `out`. */
    pivcoh__init_dec();
    int KR;
    if (!(p = pivcoh__read_kr(&KR, N, p, end))) return -1;
    int KL = N - KR;
    uint8_t *sc = scratch ? (uint8_t *)scratch
                          : (uint8_t *)malloc(PIVCOH_DECODE_SCRATCH_SIZE(N));
    if (!sc) return -1;
    const uint8_t *bm;

    if (kind == PIVCOH__LEAFL) {
        /* One internal child: it decodes at the arena base with the
         * space after it as ping-pong partner. */
        if (KR > 0) p = pivcoh__dec_child(t, root->right, KR, sc, sc + KR, p, end);
        if (p && (p = pivcoh__read_bm(&bm, N, p, end)) != NULL &&
            pivcoh__merge_cst_vec_x(bm, N, t->rank_to_sym[root->param], sc, KR, out))
            p = NULL;
    } else {
        /* FULL root, hybrid hole-reuse: both children decode into the
         * arena's first N bytes, [larger | smaller].  The larger child
         * (first on the wire) borrows the smaller sibling's still-empty
         * slot as its partner — spilling past N only when a spine
         * smaller-child outgrows it — and the smaller child follows
         * with a fresh partner beyond N. */
        uint8_t *lbuf, *rbuf;
        if (KR > KL) {
            rbuf = sc; lbuf = sc + KR;
            p = pivcoh__dec_child(t, root->right, KR, rbuf, lbuf, p, end);
            if (p && KL > 0) p = pivcoh__dec_child(t, 1, KL, lbuf, sc + N, p, end);
        } else {
            lbuf = sc; rbuf = sc + KL;
            p = pivcoh__dec_child(t, 1, KL, lbuf, rbuf, p, end);
            if (p && KR > 0) p = pivcoh__dec_child(t, root->right, KR, rbuf, sc + N, p, end);
        }
        if (p && (p = pivcoh__read_bm(&bm, N, p, end)) != NULL &&
            pivcoh__merge_vec_vec_x(bm, N, lbuf, KL, rbuf, KR, out))
            p = NULL;
    }
    if (!scratch) free(sc);
    if (!p) return -1;
    if (consumed) *consumed = (size_t)(p - in);
    return N;
}

/* ================= encode kernels (ports of primitives_neon) ================ */

/* Per-mask LUTs:
 *   ctab8[m][0:8]  right source lanes packed at [0,n_right), 0xff fill
 *                  (vtbl1 returns 0 for out-of-range indices)
 * p16rev partition LUTs (part_full): one combined index per 16-lane group
 * packs {left, forward, front} | {right, reversed, back}; left+right tile
 * the 16 lanes so the OR of the two disjoint-support tables is exact.
 *   ptabA[m0]      low byte: left -> front, right -> back reversed
 *   ptabB0[m1]     high byte, pc0=0 layout; pc0>0 is the same table
 *                  loaded at byte offset pc0 (padded to 32 B/entry). */
static uint8_t pivcoh__ctab8[256][16]   __attribute__((aligned(16)));
static uint8_t pivcoh__ptabA[256][16]   __attribute__((aligned(16)));
static uint8_t pivcoh__ptabB0[256][32]  __attribute__((aligned(32)));

static void pivcoh__init_enc(void)
{
    static int built = 0;
    if (built) return;
    for (int m = 0; m < 256; m++) {
        memset(pivcoh__ctab8[m], 0xff, 16);
        int qr = 0, ql = 0;
        for (int k = 0; k < 8; k++) {
            if (m & (1 << k)) pivcoh__ctab8[m][qr++]     = (uint8_t)k;
            else              pivcoh__ctab8[m][8 + ql++] = (uint8_t)k;
        }
    }
    for (int m0 = 0; m0 < 256; m0++) {
        memset(pivcoh__ptabA[m0], 0, 16);
        int lp = 0, rp = 15;
        for (int k = 0; k < 8; k++) {
            if ((m0 >> k) & 1) pivcoh__ptabA[m0][rp--] = (uint8_t)k;
            else               pivcoh__ptabA[m0][lp++] = (uint8_t)k;
        }
    }
    for (int m1 = 0; m1 < 256; m1++) {
        memset(pivcoh__ptabB0[m1], 0, 32);
        int lp = 8, rp = 15;   /* pc0 = 0 layout; pc0 > 0 via the load offset */
        for (int k = 0; k < 8; k++) {
            if ((m1 >> k) & 1) pivcoh__ptabB0[m1][rp--] = (uint8_t)(8 + k);
            else               pivcoh__ptabB0[m1][lp++] = (uint8_t)(8 + k);
        }
    }
    built = 1;
}

/* 8 partition mask bytes for 64 ranks via one vpaddq reduction tree
 * (each lane holds its bit-weight after the vcgt+and, so 4 pairwise
 * adds collapse every chunk to one byte).  Returned SIMD-side: the
 * callers vcnt the byte popcounts BEFORE the lane move, so the cursor
 * chain never round-trips GPR->SIMD (that fmov costs a load-port uop
 * and ~6 cycles mid-chain). */
static inline uint8x8_t pivcoh__masks64v(uint8x16_t v0, uint8x16_t v1,
                                         uint8x16_t v2, uint8x16_t v3,
                                         uint8x16_t vt, uint8x16_t bw)
{
    uint8x16_t w0 = vandq_u8(vcgtq_u8(v0, vt), bw);
    uint8x16_t w1 = vandq_u8(vcgtq_u8(v1, vt), bw);
    uint8x16_t w2 = vandq_u8(vcgtq_u8(v2, vt), bw);
    uint8x16_t w3 = vandq_u8(vcgtq_u8(v3, vt), bw);
    uint8x16_t t0 = vpaddq_u8(w0, w1);
    uint8x16_t t1 = vpaddq_u8(w2, w3);
    uint8x16_t u0 = vpaddq_u8(t0, t1);
    return vget_low_u8(vpaddq_u8(u0, u0));
}

static inline uint64_t pivcoh__masks64(uint8x16_t v0, uint8x16_t v1,
                                       uint8x16_t v2, uint8x16_t v3,
                                       uint8x16_t vt, uint8x16_t bw)
{
    return vget_lane_u64(vreinterpret_u64_u8(
               pivcoh__masks64v(v0, v1, v2, v3, vt, bw)), 0);
}

/* ranks[i] = s2r[sym[i]]: the s2r table lives in 16 NEON regs; each
 * 16-lane input does one vqtbl4 + three vqtbx4, with 4 extra GPR
 * gathers interleaved into the idle scalar slots (20 sym/iter).
 * Tables whose used symbols span < 128 values (statically known --
 * every ASCII-ish window qualifies) take a half-size path: two tables
 * indexed by sym - umin, one vqtbl4 + one vqtbx4.  Out-of-window
 * symbols (frequency zero) index past both tables and map to rank 0 --
 * the same valid-but-meaningless-stream contract as the full path. */
static void pivcoh__enc_init(uint8_t *ranks, int n, const uint8_t *sym,
                             const uint8_t *s2r, unsigned umin, unsigned span1)
{
    int i = 0;
    if (n >= 20 && span1 < 128) {
        const uint8_t *w = s2r + umin;         /* umin <= 128 (clamped at
                                                  enc-view build), so the
                                                  128-byte window stays
                                                  inside sym_to_rank */
        uint8x16x4_t t0 = vld1q_u8_x4(w), t1 = vld1q_u8_x4(w + 64);
        const uint8x16_t vmin = vdupq_n_u8((uint8_t)umin);
        const uint8x16_t s64  = vdupq_n_u8(64);
        for (; i + 20 <= n; i += 20) {
            uint8x16_t c = vsubq_u8(vld1q_u8(sym + i), vmin);
            uint32_t a; memcpy(&a, sym + i + 16, 4);
            uint8x16_t r = vqtbl4q_u8(t0, c);
            unsigned r0 = s2r[(uint8_t)a];
            unsigned r1 = s2r[(uint8_t)(a >> 8)];
            r = vqtbx4q_u8(r, t1, vsubq_u8(c, s64));
            unsigned r2 = s2r[(uint8_t)(a >> 16)];
            unsigned r3 = s2r[(uint8_t)(a >> 24)];
            vst1q_u8(ranks + i, r);
            uint32_t h = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
            memcpy(ranks + i + 16, &h, 4);
        }
    } else if (n >= 20) {
        uint8x16x4_t t0 = vld1q_u8_x4(s2r), t1 = vld1q_u8_x4(s2r + 64);
        uint8x16x4_t t2 = vld1q_u8_x4(s2r + 128), t3 = vld1q_u8_x4(s2r + 192);
        const uint8x16_t s64  = vdupq_n_u8(64);
        const uint8x16_t s128 = vdupq_n_u8(128);
        const uint8x16_t s192 = vdupq_n_u8(192);
        for (; i + 20 <= n; i += 20) {
            uint8x16_t c = vld1q_u8(sym + i);
            uint32_t a; memcpy(&a, sym + i + 16, 4);
            uint8x16_t r = vqtbl4q_u8(t0, c);
            unsigned r0 = s2r[(uint8_t)a];
            r = vqtbx4q_u8(r, t1, vsubq_u8(c, s64));
            unsigned r1 = s2r[(uint8_t)(a >> 8)];
            r = vqtbx4q_u8(r, t2, vsubq_u8(c, s128));
            unsigned r2 = s2r[(uint8_t)(a >> 16)];
            r = vqtbx4q_u8(r, t3, vsubq_u8(c, s192));
            unsigned r3 = s2r[(uint8_t)(a >> 24)];
            vst1q_u8(ranks + i, r);
            uint32_t h = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
            memcpy(ranks + i + 16, &h, 4);
        }
    }
    for (; i < n; i++) ranks[i] = s2r[sym[i]];
}

/* 16-lane movemask via two GPR magic multiplies: for 0x00/0xFF compare
 * bytes, x * 0x103070F1F3F80 accumulates each u64 half's byte-MSBs into
 * its top byte (all 256 patterns verified per half).  Cheaper than the
 * masks64 reduction tree at narrow-tail widths. */
static inline uint32_t pivcoh__movemask16(uint8x16_t cm)
{
    const uint64_t magic = 0x103070F1F3F80ull;
    uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(cm), 0);
    uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(cm), 1);
    return (uint32_t)((lo * magic) >> 56)
         | ((uint32_t)((hi * magic) >> 48) & 0xFF00u);
}

/* Scatter one 64-rank group-set of a full partition: per 16-lane
 * group, ONE combined shuffle index (the OR of ptabA[m0] |
 * ptabB0[m1]+pc0) yields both sides at once — the register IS the
 * left output, and a loop-invariant full-reverse vqtbl1 recovers the
 * right.  mask_word may be tail-masked: zero bits scatter their lanes
 * as (phantom) lefts AFTER the real ones, so the left prefix stays
 * exact and only dead bytes past it take garbage.  Stores run 16 wide
 * (up to +64/+16 past the valid counts, into dead/slack space).
 * Returns the group-set's right count. */
__attribute__((always_inline)) static inline
int pivcoh__part64_full(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2,
                        uint8x16_t v3, uint64_t mask_word, uint64_t pcw,
                        uint8_t *ldst, uint8_t *rdst)
{
    uint64_t pfx = pcw * 0x0101010101010101ULL;
    uint8x16_t vg[4] = { v0, v1, v2, v3 };
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
#define PIVCOH__PART(g) do {                                                 \
        uint8_t  m0 = (uint8_t)(mask_word >> (16*(g)));                     \
        uint8_t  m1 = (uint8_t)(mask_word >> (16*(g) + 8));                 \
        uint32_t pc0 = (uint32_t)((pcw >> (16*(g)))     & 0xFF);            \
        uint32_t cr  = (g) == 0 ? 0u                                        \
                     : (uint32_t)((pfx >> (8*(2*(g) - 1))) & 0xFF);         \
        uint8x16_t ri = vorrq_u8(vld1q_u8(pivcoh__ptabA[m0]),               \
                                 vld1q_u8(&pivcoh__ptabB0[m1][pc0]));       \
        uint8x16_t comb = vqtbl1q_u8(vg[g], ri);                            \
        vst1q_u8(ldst + (16*(g) - cr), comb);                               \
        vst1q_u8(rdst + cr, vqtbl1q_u8(comb, rev16));                       \
    } while (0)
    PIVCOH__PART(0); PIVCOH__PART(1); PIVCOH__PART(2); PIVCOH__PART(3);
#undef PIVCOH__PART
    return (int)(pfx >> 56);
}

/* Full partition: bitmap + both sides compacted (left in place in
 * ranks, right into tmp).  Tail-free at 16-rank granularity: each tail
 * step is one movemask + one p16rev group scatter, the final step's
 * mask trimmed to the real ranks — the wire bytes stay exact, loads
 * overread into the ranks slack/gaps, and the phantom-left scatter
 * lands in dead bytes.  No scalar tail: the per-element rank>thr
 * branch is the bitmap itself, i.e. maximally unpredictable. */
static int pivcoh__part_full(uint8_t *ranks, int n, uint8_t thr,
                             uint8_t *bm, uint8_t *tmp)
{
    int n_left = 0, n_right = 0, j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    /* Software-pipelined one iteration deep, like the decode merges:
     * the carried chain (loads -> cgt -> three serial vpaddq -> lane
     * moves -> multiply -> cursors) is ~2x the loop's port work, so
     * the NEXT group-set's masks and popcounts start under the current
     * scatter. */
    if (j + 64 <= n) {
        uint8x16_t c0 = vld1q_u8(ranks + j),      c1 = vld1q_u8(ranks + j + 16);
        uint8x16_t c2 = vld1q_u8(ranks + j + 32), c3 = vld1q_u8(ranks + j + 48);
        uint8x8_t mv = pivcoh__masks64v(c0, c1, c2, c3, vt, bw);
        uint64_t w   = vget_lane_u64(vreinterpret_u64_u8(mv), 0);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(mv)), 0);
        for (; j + 128 <= n; j += 64) {
            uint8x16_t n0 = vld1q_u8(ranks + j + 64), n1 = vld1q_u8(ranks + j + 80);
            uint8x16_t n2 = vld1q_u8(ranks + j + 96), n3 = vld1q_u8(ranks + j + 112);
            uint8x8_t nmv = pivcoh__masks64v(n0, n1, n2, n3, vt, bw);
            uint64_t nw   = vget_lane_u64(vreinterpret_u64_u8(nmv), 0);
            uint64_t npcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(nmv)), 0);
            memcpy(bm + (j >> 3), &w, 8);
            int tr = pivcoh__part64_full(c0, c1, c2, c3, w, pcw,
                                         ranks + n_left, tmp + n_right);
            n_right += tr;
            n_left  += 64 - tr;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            w = nw; pcw = npcw;
        }
        memcpy(bm + (j >> 3), &w, 8);
        int tr = pivcoh__part64_full(c0, c1, c2, c3, w, pcw,
                                     ranks + n_left, tmp + n_right);
        n_right += tr;
        n_left  += 64 - tr;
        j += 64;
    }
    /* Narrow tail: one 16-rank group per step (the 64-wide group-set is
     * heavy for the 1..20-rank tails deep trees are made of).  The
     * final step's mask is trimmed to the real ranks; phantom lefts
     * land after the real ones in dead bytes, as in the main loop. */
    static const uint8_t rev16_a[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    uint8x16_t rev16 = vld1q_u8(rev16_a);
    for (; j < n; j += 16) {
        uint8x16_t v = vld1q_u8(ranks + j);
        int keep = n - j < 16 ? n - j : 16;
        uint32_t m = pivcoh__movemask16(vcgtq_u8(v, vt)) & ((1u << keep) - 1);
        uint16_t m16 = (uint16_t)m;
        memcpy(bm + (j >> 3), &m16, 2);
        uint32_t pc0 = (uint32_t)__builtin_popcount(m & 0xFF);
        uint8x16_t ri = vorrq_u8(vld1q_u8(pivcoh__ptabA[m & 0xFF]),
                                 vld1q_u8(&pivcoh__ptabB0[m >> 8][pc0]));
        uint8x16_t comb = vqtbl1q_u8(v, ri);
        vst1q_u8(ranks + n_left, comb);
        vst1q_u8(tmp + n_right, vqtbl1q_u8(comb, rev16));
        int tr = __builtin_popcount(m);
        n_right += tr;
        n_left  += 16 - tr;
    }
    return n_right;
}

/* Scatter one 64-rank group-set's right side via ctab8 (8-lane
 * chunks); same phantom-left masking argument as part64_full.
 * Returns the group-set's right count. */
__attribute__((always_inline)) static inline
int pivcoh__part64_right(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2,
                         uint8x16_t v3, uint64_t mask_word, uint64_t pcw,
                         uint8_t *rdst)
{
    uint64_t pfx = pcw * 0x0101010101010101ULL;
    uint8x8_t cv[8] = {
        vget_low_u8(v0), vget_high_u8(v0),
        vget_low_u8(v1), vget_high_u8(v1),
        vget_low_u8(v2), vget_high_u8(v2),
        vget_low_u8(v3), vget_high_u8(v3),
    };
#define PIVCOH__PART1(K_) do {                                               \
        uint32_t cr = (K_)==0 ? 0u : (uint32_t)((pfx >> (8*((K_)-1))) & 0xFF); \
        const uint8_t *tab = pivcoh__ctab8[(uint8_t)(mask_word >> (8*(K_)))]; \
        vst1_u8(rdst + cr, vtbl1_u8(cv[K_], vld1_u8(tab)));                  \
    } while (0)
    PIVCOH__PART1(0); PIVCOH__PART1(1); PIVCOH__PART1(2); PIVCOH__PART1(3);
    PIVCOH__PART1(4); PIVCOH__PART1(5); PIVCOH__PART1(6); PIVCOH__PART1(7);
#undef PIVCOH__PART1
    return (int)(pfx >> 56);
}

/* One-sided partition: bitmap + the right side compacted into tmp (a
 * LEAF_LEFT node's left side is dead).  Tail-free and pipelined like
 * part_full.  (The old EMIT_RIGHT=0 pure-bitmap mode moved into the
 * dedicated pack_d1.) */
static int pivcoh__part_core(uint8_t *ranks, int n, uint8_t thr,
                             uint8_t *bm, uint8_t *tmp)
{
    int n_right = 0, j = 0;
    uint8x16_t vt = vdupq_n_u8(thr);
    static const uint8_t bw_a[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x16_t bw = vld1q_u8(bw_a);
    if (j + 64 <= n) {
        uint8x16_t c0 = vld1q_u8(ranks + j),      c1 = vld1q_u8(ranks + j + 16);
        uint8x16_t c2 = vld1q_u8(ranks + j + 32), c3 = vld1q_u8(ranks + j + 48);
        uint8x8_t mv = pivcoh__masks64v(c0, c1, c2, c3, vt, bw);
        uint64_t w   = vget_lane_u64(vreinterpret_u64_u8(mv), 0);
        uint64_t pcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(mv)), 0);
        for (; j + 128 <= n; j += 64) {
            uint8x16_t n0 = vld1q_u8(ranks + j + 64), n1 = vld1q_u8(ranks + j + 80);
            uint8x16_t n2 = vld1q_u8(ranks + j + 96), n3 = vld1q_u8(ranks + j + 112);
            uint8x8_t nmv = pivcoh__masks64v(n0, n1, n2, n3, vt, bw);
            uint64_t nw   = vget_lane_u64(vreinterpret_u64_u8(nmv), 0);
            uint64_t npcw = vget_lane_u64(vreinterpret_u64_u8(vcnt_u8(nmv)), 0);
            memcpy(bm + (j >> 3), &w, 8);
            n_right += pivcoh__part64_right(c0, c1, c2, c3, w, pcw, tmp + n_right);
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            w = nw; pcw = npcw;
        }
        memcpy(bm + (j >> 3), &w, 8);
        n_right += pivcoh__part64_right(c0, c1, c2, c3, w, pcw, tmp + n_right);
        j += 64;
    }
    /* Narrow tail, as in part_full. */
    for (; j < n; j += 16) {
        uint8x16_t v = vld1q_u8(ranks + j);
        int keep = n - j < 16 ? n - j : 16;
        uint32_t m = pivcoh__movemask16(vcgtq_u8(v, vt)) & ((1u << keep) - 1);
        uint16_t m16 = (uint16_t)m;
        memcpy(bm + (j >> 3), &m16, 2);
        vst1_u8(tmp + n_right,
                vtbl1_u8(vget_low_u8(v), vld1_u8(pivcoh__ctab8[m & 0xFF])));
        n_right += __builtin_popcount(m & 0xFF);
        vst1_u8(tmp + n_right,
                vtbl1_u8(vget_high_u8(v), vld1_u8(pivcoh__ctab8[m >> 8])));
        n_right += __builtin_popcount(m >> 8);
    }
    return n_right;
}

/* ---- flat pack: (rank - base) is already the D-bit local code ----
 *
 * Every kernel packs ALL n codes by running its vector loop past n:
 * the final vector loads up to 15 garbage ranks past the region
 * (inside the partition gaps/slack) and packs garbage bits, which land
 * only where they don't matter — bytes past ceil(n*D/8) are junk under
 * the usual contract (overwritten by the next record; inside out_cap
 * at stream end), and the padding bits inside the last partial byte
 * are zeroed by one byte RMW in the dispatcher, store-forwarded from
 * the final vector store. */

/* D=1: the bit IS (rank > base) — the partition's masks64 bitmap
 * build (rank == base + 1 exactly when rank > base on a D=1 region)
 * without the compaction, popcount accumulation, or tail trimming:
 * the 16-rank remainder stores junk bits past n and rides the packs'
 * junk-byte contract like every other kernel (the dispatcher's RMW
 * zeroes the last partial byte). */
static inline void pivcoh__pack_d1(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    uint8x16_t vt = vdupq_n_u8(base);
    uint8x16_t bw = vld1q_u8(pivcoh__d1_bit_tab);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint64_t w = pivcoh__masks64(vld1q_u8(ranks + i),
                                     vld1q_u8(ranks + i + 16),
                                     vld1q_u8(ranks + i + 32),
                                     vld1q_u8(ranks + i + 48), vt, bw);
        memcpy(out + (i >> 3), &w, 8);
    }
    for (; i < n; i += 16) {
        uint16_t m16 = (uint16_t)pivcoh__movemask16(
                           vcgtq_u8(vld1q_u8(ranks + i), vt));
        memcpy(out + (i >> 3), &m16, 2);
    }
}

/* D=2: 64 ranks -> 16 bytes (4 ranks per byte, no byte crossings) —
 * unrolled x4 so both vpaddq_u8 reduction levels pair full vectors and
 * the result is a whole 16-byte store.  The pipeline is linear mod 256
 * (shifts multiply, vpaddq adds, u8 wrap IS the target modulus), so
 * the base subtract distributes to one op at the end:
 * sum (r_i - b) 4^i = r0 + 4r1 + 16r2 + 64r3 - 85b (mod 256, exact
 * since the true byte is in range).  The 16-rank remainder keeps the
 * self-pairing quarter-width form. */
static inline void pivcoh__pack_d2(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d2[16] = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const int8x16_t sh = vld1q_s8(shifts_d2);
    const uint8x16_t b85 = vdupq_n_u8((uint8_t)(85 * base));
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      sh);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), sh);
        uint8x16_t b2 = vshlq_u8(vld1q_u8(ranks + i + 32), sh);
        uint8x16_t b3 = vshlq_u8(vld1q_u8(ranks + i + 48), sh);
        uint8x16_t r  = vpaddq_u8(vpaddq_u8(b0, b1), vpaddq_u8(b2, b3));
        vst1q_u8(out + (i >> 2), vsubq_u8(r, b85));
    }
    for (; i < n; i += 16) {
        uint8x16_t b  = vshlq_u8(vld1q_u8(ranks + i), sh);
        uint8x16_t s1 = vpaddq_u8(b, b);
        uint8x16_t s2 = vsubq_u8(vpaddq_u8(s1, s1), b85);
        uint32_t packed4 = vgetq_lane_u32(vreinterpretq_u32_u8(s2), 0);
        memcpy(out + (i >> 2), &packed4, 4);
    }
}

/* D=4: 32 ranks -> 16 bytes, pairing (r[2k], r[2k+1]) into one byte —
 * unrolled once so the vpaddq_u8 pairs two full input vectors, with
 * the base subtract distributed like D=2's:
 * (r0 - b) + 16(r1 - b) = r0 + 16r1 - 17b (mod 256, exact). */
static inline void pivcoh__pack_d4(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t shifts_d4[16] = { 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4, 0,4 };
    const int8x16_t sh = vld1q_s8(shifts_d4);
    const uint8x16_t b17 = vdupq_n_u8((uint8_t)(17 * base));
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      sh);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), sh);
        vst1q_u8(out + (i >> 1), vsubq_u8(vpaddq_u8(b0, b1), b17));
    }
    for (; i < n; i += 16) {
        uint8x16_t b = vshlq_u8(vld1q_u8(ranks + i), sh);
        vst1_u8(out + (i >> 1),
                vget_low_u8(vsubq_u8(vpaddq_u8(b, b), b17)));
    }
}

/* D=5/6/7: variable-shift pack, 16 codes/iter (D=3 pairs itself into
 * D=6 below and rides the same pyramid).  At each width the two
 * halves of a lane pair are shifted TOWARD each other with one USHL of
 * {+s, -s} per-lane counts — the even half's top bit and the odd
 * half's bottom bit meet at the lane boundary — so each pairing level
 * is a single instruction and the packed field rides mid-lane until
 * one final immediate right shift re-bases it:
 *   L1 u8  {8-D, 0}:            u16 = pair  << (8-D)
 *   L2 u16 {8-D, -(8-D)}:       u32 = quad  << (16-2D)
 *   L3 u32 {16-2D, -(16-2D)}:   u64 = octet << (32-4D)
 * The compact shuffle absorbs the whole bytes of the final (32-4D)
 * re-basing shift (its tables start at byte 1 for D=5/6), leaving a
 * residual >> 4 for D=5/7 and NO final shift for D=6 -- 3-4 shift ops
 * off tiny static count tables.  (Byte-aligning the
 * fields EARLY so the tbl can also do the u64 level -- e.g. D=6's
 * 24-bit quad at [0,24) -- costs a shr+sli pair per level, one op
 * more: a sub-lane field can't cross its own byte/lane boundary with
 * a single per-lane shift, which is exactly what the converging
 * meet-at-the-boundary placement avoids.)
 * (History: ryg's multiply-as-shift vmull pyramid, then a 6-op
 * USHR+SLI ladder, each replaced in turn.)  Each 16-byte store carries
 * 16-2D trailing junk bytes, overwritten by the next iter / next
 * record (the caller's out_cap >= PIVCOH_ENCODE_BOUND keeps even the
 * last one in bounds). */
static const uint8_t pivcoh__pack_compact_d5[16] = {
    1, 2, 3, 4, 5,   9, 10, 11, 12, 13,  0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivcoh__pack_compact_d6[16] = {
    1, 2, 3, 4, 5, 6,   9, 10, 11, 12, 13, 14,  0xff, 0xff, 0xff, 0xff
};
static const uint8_t pivcoh__pack_compact_d7[16] = {
    0, 1, 2, 3, 4, 5, 6,   8, 9, 10, 11, 12, 13, 14,  0xff, 0xff
};

#define PIVCOH__PACK_DN(NAME, D_VAL, BITSHR, COMPACT_TAB)                        \
static inline void NAME(uint8_t *out, const uint8_t *ranks, int n, uint8_t base) \
{                                                                                \
    static const int8_t  sh1[16] = { 8-(D_VAL),0, 8-(D_VAL),0, 8-(D_VAL),0,      \
                                     8-(D_VAL),0, 8-(D_VAL),0, 8-(D_VAL),0,      \
                                     8-(D_VAL),0, 8-(D_VAL),0 };                 \
    static const int16_t sh2[8]  = { 8-(D_VAL), -(8-(D_VAL)),                    \
                                     8-(D_VAL), -(8-(D_VAL)),                    \
                                     8-(D_VAL), -(8-(D_VAL)),                    \
                                     8-(D_VAL), -(8-(D_VAL)) };                  \
    static const int32_t sh3[4]  = { 16-2*(D_VAL), -(16-2*(D_VAL)),              \
                                     16-2*(D_VAL), -(16-2*(D_VAL)) };            \
    const int8x16_t s1 = vld1q_s8(sh1);                                          \
    const int16x8_t s2 = vld1q_s16(sh2);                                         \
    const int32x4_t s3 = vld1q_s32(sh3);                                         \
    const uint8x16_t compact = vld1q_u8(COMPACT_TAB);                            \
    for (int i = 0; i < n; i += 16) {                                            \
        uint8x16_t cb = vsubq_u8(vld1q_u8(ranks + i), vdupq_n_u8(base));         \
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(cb, s1));                 \
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));              \
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));              \
        if (BITSHR) w64 = vshrq_n_u64(w64, (BITSHR) ? (BITSHR) : 1);             \
        vst1q_u8(out + ((i * (D_VAL)) >> 3),                                     \
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));                \
    }                                                                            \
}
PIVCOH__PACK_DN(pivcoh__pack_d5, 5, 4, pivcoh__pack_compact_d5)
PIVCOH__PACK_DN(pivcoh__pack_d6, 6, 0, pivcoh__pack_compact_d6)
PIVCOH__PACK_DN(pivcoh__pack_d7, 7, 4, pivcoh__pack_compact_d7)
#undef PIVCOH__PACK_DN

/* D=3: pair adjacent codes into 6-bit values the D=4 way — per-lane
 * {0,3} shifts + one vpaddq (pair = c_even + 8 c_odd, one pair per
 * byte), with the base subtract distributed through the mod-256
 * pairing (- 9b, exact since the true pair < 64) — then run the D=6
 * pyramid on the pairs: a 3-bit LSB-first stream IS the 6-bit
 * LSB-first stream of its pairs, so the wire is unchanged, and D=6
 * needs no final shift.  32 codes in 11 uops; a 16-code remainder
 * runs the same body self-paired (6 valid output bytes, junk store
 * contract as everywhere). */
static inline void pivcoh__pack_d3(uint8_t *out, const uint8_t *ranks, int n, uint8_t base)
{
    static const int8_t  shifts_p[16] = { 0,3, 0,3, 0,3, 0,3, 0,3, 0,3, 0,3, 0,3 };
    static const int8_t  shifts_1[16] = { 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0 };
    static const int16_t shifts_2[8]  = { 2,-2, 2,-2, 2,-2, 2,-2 };
    static const int32_t shifts_4[4]  = { 4,-4, 4,-4 };
    const int8x16_t shp = vld1q_s8(shifts_p);
    const uint8x16_t b9 = vdupq_n_u8((uint8_t)(9 * base));
    const int8x16_t s1 = vld1q_s8(shifts_1);
    const int16x8_t s2 = vld1q_s16(shifts_2);
    const int32x4_t s3 = vld1q_s32(shifts_4);
    const uint8x16_t compact = vld1q_u8(pivcoh__pack_compact_d6);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        uint8x16_t b0 = vshlq_u8(vld1q_u8(ranks + i),      shp);
        uint8x16_t b1 = vshlq_u8(vld1q_u8(ranks + i + 16), shp);
        uint8x16_t pair = vsubq_u8(vpaddq_u8(b0, b1), b9);
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
        vst1q_u8(out + ((i * 3) >> 3),
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));
    }
    for (; i < n; i += 16) {           /* <= 2 self-paired half-blocks */
        uint8x16_t b = vshlq_u8(vld1q_u8(ranks + i), shp);
        uint8x16_t pair = vsubq_u8(vpaddq_u8(b, b), b9);
        uint16x8_t w16 = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
        uint32x4_t w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
        uint64x2_t w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
        vst1q_u8(out + ((i * 3) >> 3),
                 vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact));
    }
}

/* Dispatcher: D is structural (1..8); every kernel packs all n codes.
 * D=8 is a full-alphabet equal-length code (256 ranks at one depth is
 * only Kraft-exact as the whole tree), so base == 0 and the byte-
 * aligned ranks ARE the local codes — memcpy, mirroring the decode
 * side, with no read or write past the region. */
static void pivcoh__pack_dN(uint8_t *out, const uint8_t *ranks,
                            int n, int D, uint8_t base)
{
    switch (D) {
    case 1: pivcoh__pack_d1(out, ranks, n, base); break;
    case 2: pivcoh__pack_d2(out, ranks, n, base); break;
    case 3: pivcoh__pack_d3(out, ranks, n, base); break;
    case 4: pivcoh__pack_d4(out, ranks, n, base); break;
    case 5: pivcoh__pack_d5(out, ranks, n, base); break;
    case 6: pivcoh__pack_d6(out, ranks, n, base); break;
    case 7: pivcoh__pack_d7(out, ranks, n, base); break;
    case 8: memcpy(out, ranks, (size_t)n); break;
    }
    /* Zero the padding bits of the last partial byte (the kernels'
     * final vector packed garbage there); one store-forwarded RMW,
     * idempotent for D=8 whose padding is already exact.
     * Unconditional: at rem_bits == 0 the mask is 0 and the target is
     * the first junk byte PAST the region, zeroed harmlessly under the
     * usual contract — cheaper than a per-node data-dependent branch. */
    int rem_bits = (n * D) & 7;
    out[(n * D) >> 3] &= (uint8_t)((1u << rem_bits) - 1);
}

/* ---- encode tree walk ----
 * Production placement, arena-staged: each node's scratch starts with
 * its staged bitmap (nbytes+8), then the compacted right half, then
 * the children's deeper scratch.  Per level that is n/8 + 9 + n_right
 * bytes, depth <= 11, plus the tail-free partition's bounded
 * overstores — inside PIVCOH_SCRATCH_SIZE.  The header is VLA-free. */
static void pivcoh__enc_node(const pivcoh_table *t, int idx,
                             uint8_t *ranks, int n,
                             uint8_t **pp, uint8_t *tmp)
{
    const pivcoh__rec *rec = &t->sched[idx];
    int kind = rec->kd & 3;
    uint8_t *p = *pp;

    if (kind == PIVCOH__FLAT) {                /* n local codes, D bits each */
        int D = rec->kd >> 2;
        pivcoh__pack_dN(p, ranks, n, D, rec->param);
        *pp = p + ((n * D + 7) >> 3);
        return;
    }
    /* Decode-order record (wire v0.7): the K_right header goes at the
     * node's PRE-order position, the marker+bitmap at its POST-order
     * position, the children's regions between, larger-K child first.
     * The bitmap is staged across the child recursion (its stream
     * position depends on the children's encoded sizes) at the base of
     * this node's scratch, NOT the stack: as a VLA it was live across
     * the recursion, ~90KB of stack on a worst-case 64K block.  +8 pads
     * the partition tail's 2-byte mask stores (<= 1 byte past nbytes)
     * with margin.  The
     * children's scratch starts 64 bytes past the right ranks: a
     * node's tail-free left scatter overshoots up to 63 bytes past its
     * OWN ranks region, and a right child's ranks end exactly where
     * its scratch (holding its live stage) would otherwise begin. */
    int nbytes = (n + 7) >> 3;
    uint8_t *bm_stage = tmp;
    uint8_t *rout = tmp + nbytes + 8;
    int n_right = (kind == PIVCOH__LEAFL)
        ? pivcoh__part_core(ranks, n, rec->param, bm_stage, rout)
        : pivcoh__part_full(ranks, n, rec->param, bm_stage, rout);
    /* Define the 16 bytes past the fresh right half (inside its 64-byte
     * gap): the child's tail-free partition/pack loads read up to 15
     * bytes past its ranks region.  Phantom lanes never reach the wire
     * (masks are trimmed, the last partial byte is RMW-zeroed), but
     * genuinely UNINITIALIZED ones would make MSan/valgrind flag the
     * movemask -> popcount -> cursor/table-index chain as a use of
     * uninitialized data (their multiply shadow models can't see that
     * the movemask magic isolates lanes).  One zero store per fresh
     * region keeps every tail load fully defined; left regions need
     * nothing — bytes past them are stale parent ranks or scatter junk,
     * defined once the root window (pivcoh_encode) is. */
    vst1q_u8(rout + n_right, vdupq_n_u8(0));
    int n_left = n - n_right;
    *p++ = (uint8_t)n_right;                   /* K_right: 1 byte when the
                                                * node's n fits one (the
                                                * decoder knows n) */
    if (n > 255)
        *p++ = (uint8_t)(n_right >> 8);
    *pp = p;
    if (kind == PIVCOH__FULL && n_right > n_left) {
        pivcoh__enc_node(t, idx + rec->right, rout, n_right, pp, rout + n_right + 64);
        if (n_left > 0)
            pivcoh__enc_node(t, idx + 1, ranks, n_left, pp, rout + n_right + 64);
    } else {
        if (kind == PIVCOH__FULL && n_left > 0)
            pivcoh__enc_node(t, idx + 1, ranks, n_left, pp, rout + n_right + 64);
        if (n_right > 0)
            pivcoh__enc_node(t, idx + rec->right, rout, n_right, pp, rout + n_right + 64);
    }
    p = *pp;
    memcpy(p, bm_stage, (size_t)nbytes);       /* post-order raw bitmap */
    *pp = p + nbytes;
}

PIVCOHDEF ptrdiff_t pivcoh_encode(const pivcoh_table *t,
                                  const uint8_t *in, size_t n,
                                  uint8_t *out, size_t out_cap, void *scratch)
{
    if (!t || !in || !out || n < 1 || n > 65535 || !t->num_ranks) return -1;
    if (out_cap < PIVCOH_ENCODE_BOUND(n)) return -1;
    pivcoh__init_enc();
    if (!t->enc_ready) {
        /* Encoder view of the table, built on first use so decode-side
         * builds skip it.  The writes are a pure function of
         * rank_to_sym and idempotent, and enc_ready is set last, so
         * concurrent first encodes on a shared table are benign — the
         * same contract as the lazy static tables above. */
        pivcoh_table *tw = (pivcoh_table *)t;
        memset(tw->sym_to_rank, 0, 256);
        for (int s = t->num_ranks - 1; s >= 0; s--)
            tw->sym_to_rank[t->rank_to_sym[s]] = (uint8_t)s;
        int lo = 255, hi = 0;
        for (int s = 0; s < 256; s++)
            if (t->code_len[s]) { if (s < lo) lo = s; if (s > hi) hi = s; }
        if (lo > 128) lo = 128; /* keep enc_init's 128-byte window inside
                                   sym_to_rank; span1 stays < 128 (hi <=
                                   255) so all-high alphabets keep the
                                   half-size path, and symbols below the
                                   window still wrap to >= 128 -> both
                                   lookups miss -> rank 0 as documented */
        tw->enc_umin  = (uint8_t)lo;
        tw->enc_span1 = (uint8_t)(hi - lo);
        tw->enc_ready = 1;
    }
    uint8_t *sc = scratch ? (uint8_t *)scratch : (uint8_t *)malloc(PIVCOH_SCRATCH_SIZE(n));
    if (!sc) return -1;
    uint8_t *ranks = sc, *tmp = sc + n + 64;   /* +64: the root ranks' overshoot
                                                  gap (the tail-free partition
                                                  strays <= 63 B past a ranks
                                                  region; children get the same
                                                  gap in enc_node) */
    pivcoh__enc_init(ranks, (int)n, in, t->sym_to_rank,
                     t->enc_umin, t->enc_span1);
    vst1q_u8(ranks + n, vdupq_n_u8(0));        /* root tail-read window:
                                                  with enc_node's
                                                  per-partition twin,
                                                  every tail-free 16-byte
                                                  load reads fully
                                                  defined bytes (see
                                                  enc_node) */
    uint8_t *p = out;
    *p++ = (uint8_t)n;
    *p++ = (uint8_t)(n >> 8);
    pivcoh__enc_node(t, 0, ranks, (int)n, &p, tmp);
    if (!scratch) free(sc);
    return p - out;
}

/* ================= one-shot frame API + utilities ================= */

PIVCOHDEF void pivcoh_histogram(uint64_t freq[256], const uint8_t *p, size_t n)
{
    memset(freq, 0, 256 * sizeof(uint64_t));
    while (n) {   /* u32 lanes flush to the u64 totals every GiB: exact
                   * for any n (a lane sees at most chunk/4 hits) */
        size_t chunk = n < ((size_t)1 << 30) ? n : ((size_t)1 << 30);
        uint32_t c[4][256];
        memset(c, 0, sizeof(c));
        size_t i = 0;
        for (; i + 8 <= chunk; i += 8) {
            uint64_t w; memcpy(&w, p + i, 8);
            c[0][w & 255]++;         c[1][(w >> 8) & 255]++;
            c[2][(w >> 16) & 255]++; c[3][(w >> 24) & 255]++;
            c[0][(w >> 32) & 255]++; c[1][(w >> 40) & 255]++;
            c[2][(w >> 48) & 255]++; c[3][w >> 56]++;
        }
        for (; i < chunk; i++) c[0][p[i]]++;
        for (int s = 0; s < 256; s++)
            freq[s] += (uint64_t)c[0][s] + c[1][s] + c[2][s] + c[3][s];
        p += chunk; n -= chunk;
    }
}

PIVCOHDEF void pivcoh_lens_pack(uint8_t packed[128], const uint8_t code_len[256])
{
    for (int i = 0; i < 128; i++)
        packed[i] = (uint8_t)(code_len[2 * i] | code_len[2 * i + 1] << 4);
}

PIVCOHDEF void pivcoh_lens_unpack(uint8_t code_len[256], const uint8_t packed[128])
{
    for (int i = 0; i < 128; i++) {
        code_len[2 * i]     = packed[i] & 15;
        code_len[2 * i + 1] = packed[i] >> 4;
    }
}

PIVCOHDEF int pivcoh_table_from_packed_lens(pivcoh_table *t,
                                            const uint8_t packed[128])
{
    uint8_t lens[256];   /* nibbles 12..15 are invalid lengths and are
                          * rejected by from_lens's bin accounting */
    pivcoh_lens_unpack(lens, packed);
    return pivcoh_table_from_lens(t, lens);
}

/* ---- compact code-lengths wire ----
 *
 * [u8 mode]; mode 0: + the raw 128-nibble packing (129 bytes total).
 * mode 1: a 4-bit token stream (LSB-first nibble order) describing the
 * 256 lengths in symbol order:
 *   1..11       literal length
 *   0           one absent symbol
 *   12, e       run of 3+e absent symbols            (3..18)
 *   13, lo, hi  run of 19 + (lo | hi<<4) absents     (19..274)
 *   14, e       repeat the last literal 3+e MORE times
 *   15, lo, hi  repeat it 19 + (lo | hi<<4) MORE times
 * Run tokens never expand (>= 3 symbols in <= 3 nibbles), so mode 1 is
 * at worst 256 literal nibbles and the writer picks whichever mode is
 * smaller: the wire never exceeds PIVCOH_LENS_WIRE_BOUND.  Zero runs
 * carry text alphabets (absent bytes cluster); repeat runs carry
 * shaped trees (the joint pass makes big equal-length classes). */
PIVCOHDEF int pivcoh_lens_wire_write(uint8_t *dst, const uint8_t code_len[256])
{
    uint8_t nib[512];
    int nn = 0, i = 0;
    while (i < 256) {
        const uint8_t v = code_len[i];
        int j = i + 1;
        while (j < 256 && code_len[j] == v) j++;
        int run = j - i;
        i = j;
        if (v != 0) {
            nib[nn++] = v;
            run--;
        }
        while (run >= 19) {
            const int r = run > 274 ? 274 : run;
            nib[nn++] = (uint8_t)(v ? 15 : 13);
            nib[nn++] = (uint8_t)((r - 19) & 15);
            nib[nn++] = (uint8_t)((r - 19) >> 4);
            run -= r;
        }
        if (run >= 3) {
            nib[nn++] = (uint8_t)(v ? 14 : 12);
            nib[nn++] = (uint8_t)(run - 3);
        } else {
            while (run-- > 0) nib[nn++] = v;
        }
    }
    const int rb = (nn + 1) >> 1;
    if (rb < 128) {
        dst[0] = 1;
        for (int k = 0; k < rb; k++) {
            const uint8_t lo = nib[2 * k];
            const uint8_t hi = (uint8_t)(2 * k + 1 < nn ? nib[2 * k + 1] : 0);
            dst[1 + k] = (uint8_t)(lo | hi << 4);
        }
        return 1 + rb;
    }
    dst[0] = 0;
    pivcoh_lens_pack(dst + 1, code_len);
    return 129;
}

PIVCOHDEF int pivcoh_lens_wire_read(uint8_t code_len[256],
                                    const uint8_t *src, size_t n)
{
    if (!src || n < 1) return -1;
    if (src[0] == 0) {
        if (n < 129) return -1;
        pivcoh_lens_unpack(code_len, src + 1);
        return 129;
    }
    if (src[0] != 1) return -1;
    const size_t maxnib = 2 * (n - 1) < 512 ? 2 * (n - 1) : 512;
    size_t ni = 0;
    int idx = 0, prev = 0;
#define PIVCOH__LWNIB(out_)  do { \
        if (ni >= maxnib) return -1; \
        (out_) = (src[1 + (ni >> 1)] >> ((ni & 1) * 4)) & 15; \
        ni++; } while (0)
    while (idx < 256) {
        int v, run, fill;
        PIVCOH__LWNIB(v);
        if (v <= 11) {
            if (v) prev = v;
            code_len[idx++] = (uint8_t)v;
            continue;
        }
        if (v == 12 || v == 14) {
            int e;
            PIVCOH__LWNIB(e);
            run = 3 + e;
        } else {
            int lo, hi;
            PIVCOH__LWNIB(lo);
            PIVCOH__LWNIB(hi);
            run = 19 + lo + (hi << 4);
        }
        if (v >= 14) {
            if (prev == 0) return -1;      /* repeat before any literal */
            fill = prev;
        } else {
            fill = 0;
        }
        if (idx + run > 256) return -1;
        memset(code_len + idx, fill, (size_t)run);
        idx += run;
    }
#undef PIVCOH__LWNIB
    return 1 + (int)((ni + 1) >> 1);
}

/* LEB128 varint (the frame's size field). */
static inline int pivcoh__varint_put(uint8_t *p, uint64_t v)
{
    int i = 0;
    while (v >= 128) { p[i++] = (uint8_t)(v | 128); v >>= 7; }
    p[i++] = (uint8_t)v;
    return i;
}
static inline int pivcoh__varint_get(const uint8_t *p, size_t n, uint64_t *v)
{
    uint64_t r = 0;
    for (int i = 0; i < 10; i++) {
        if ((size_t)i >= n) return -1;
        r |= (uint64_t)(p[i] & 127) << (7 * i);
        if (!(p[i] & 128)) { *v = r; return i + 1; }
    }
    return -1;
}

PIVCOHDEF ptrdiff_t pivcoh_compress_joint(uint8_t *dst, size_t dst_cap,
                                          const uint8_t *src, size_t n,
                                          const pivcoh_joint *j, void *scratch)
{
    if (!dst || (!src && n) || dst_cap < PIVCOH_COMPRESS_BOUND(n)) return -1;
    size_t off = (size_t)pivcoh__varint_put(dst, (uint64_t)n);
    if (n == 0) return (ptrdiff_t)off;
    uint8_t *sc = scratch ? (uint8_t *)scratch
                          : (uint8_t *)malloc(PIVCOH_COMPRESS_SCRATCH_SIZE);
    if (!sc) return -1;
    uint64_t freq[256];
    pivcoh_histogram(freq, src, n);
    pivcoh_table t;
    pivcoh__from_freqs(&t, freq, j, sc + PIVCOH_SCRATCH_SIZE(32768));
    off += (size_t)pivcoh_lens_wire_write(dst + off, t.code_len);
    for (size_t p = 0; p < n; p += 32767) {
        size_t bn = n - p < 32767 ? n - p : 32767;
        ptrdiff_t el = pivcoh_encode(&t, src + p, bn, dst + off,
                                     PIVCOH_ENCODE_BOUND(bn), sc);
        if (el < 0) { if (!scratch) free(sc); return -1; }   /* unreachable */
        if ((size_t)el >= bn + 2) {
            /* raw store: coding did not shrink this segment */
            dst[off]     = (uint8_t)bn;
            dst[off + 1] = (uint8_t)(bn >> 8 | 0x80);
            memcpy(dst + off + 2, src + p, bn);
            off += 2 + bn;
        } else {
            off += (size_t)el;
        }
    }
    if (!scratch) free(sc);
    return (ptrdiff_t)off;
}

PIVCOHDEF ptrdiff_t pivcoh_compress(uint8_t *dst, size_t dst_cap,
                                    const uint8_t *src, size_t n,
                                    pivcoh_effort effort, void *scratch)
{
    if (effort == PIVCOH_FASTEST_COMPRESS)
        effort = n < (size_t)262144 ? PIVCOH_SIMPLEST_COMPRESS
                                    : PIVCOH_BALANCED;
    pivcoh_joint j = PIVCOH_JOINT_DEFAULTS;
    j.gran = effort == PIVCOH_FASTER_DECOMPRESS  ? 0
           : effort == PIVCOH_FASTEST_DECOMPRESS ? 1
           : -1;                     /* anything else: PIVCOH_BALANCED */
    return pivcoh_compress_joint(dst, dst_cap, src, n,
                                 effort == PIVCOH_SIMPLEST_COMPRESS ? NULL : &j,
                                 scratch);
}

PIVCOHDEF uint64_t pivcoh_decompressed_size(const uint8_t *src, size_t n)
{
    uint64_t raw;
    if (!src || pivcoh__varint_get(src, n, &raw) < 0) return (uint64_t)-1;
    return raw;
}

PIVCOHDEF ptrdiff_t pivcoh_decompress(uint8_t *dst, size_t dst_cap,
                                      const uint8_t *src, size_t n,
                                      void *scratch)
{
    uint64_t raw;
    if (!src) return -1;
    int vb = pivcoh__varint_get(src, n, &raw);
    if (vb < 0) return -1;
    if (raw == 0) return n == (size_t)vb ? 0 : -1;
    if (!dst || raw > dst_cap || raw > (uint64_t)(PTRDIFF_MAX - 1)) return -1;
    uint8_t lens[256];
    int lb = pivcoh_lens_wire_read(lens, src + vb, n - (size_t)vb);
    if (lb < 0) return -1;
    pivcoh_table t;
    if (!pivcoh_table_from_lens(&t, lens)) return -1;
    uint8_t *sc = scratch ? (uint8_t *)scratch
                          : (uint8_t *)malloc(PIVCOH_DECOMPRESS_SCRATCH_SIZE);
    if (!sc) return -1;
    size_t off = (size_t)vb + (size_t)lb, dof = 0;
    int ok = 1;
    while (dof < raw) {
        if (n - off < 2) { ok = 0; break; }
        size_t h = (size_t)src[off] | (size_t)src[off + 1] << 8;
        if (h & 0x8000) {                      /* raw-store segment */
            size_t rn = h & 0x7fff;
            if (rn == 0 || rn > raw - dof || rn > n - off - 2) { ok = 0; break; }
            memcpy(dst + dof, src + off + 2, rn);
            dof += rn;
            off += 2 + rn;
        } else {                               /* coded block, self-delimiting */
            if (h == 0 || h > raw - dof) { ok = 0; break; }
            size_t consumed = 0;
            ptrdiff_t dn = pivcoh_decode(&t, src + off, n - off, dst + dof,
                                         (size_t)raw - dof, &consumed, sc);
            if (dn != (ptrdiff_t)h) { ok = 0; break; }
            dof += (size_t)dn;
            off += consumed;
        }
    }
    if (dof != raw || off != n) ok = 0;       /* strict: no trailing bytes */
    if (!scratch) free(sc);
    return ok ? (ptrdiff_t)raw : -1;
}

PIVCOHDEF uint8_t *pivcoh_decompress_malloc(const uint8_t *src, size_t n,
                                            size_t *size)
{
    uint64_t raw = pivcoh_decompressed_size(src, n);
    if (raw > (uint64_t)(PTRDIFF_MAX - 1)) return NULL;   /* incl. UINT64_MAX */
    uint8_t *buf = (uint8_t *)malloc(raw ? (size_t)raw : 1);
    if (!buf) return NULL;
    if (pivcoh_decompress(buf, (size_t)raw, src, n, NULL) != (ptrdiff_t)raw) {
        free(buf);
        return NULL;
    }
    if (size) *size = (size_t)raw;
    return buf;
}

#endif /* PIVCOH_IMPLEMENTATION */

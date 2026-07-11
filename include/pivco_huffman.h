#ifndef PIVCO_HUFFMAN_H
#define PIVCO_HUFFMAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Constants ---------- */

/* PIVCO_BLOCK_SIZE is the *default* per-block symbol count chosen by the
 * file codec / CLI / benchmarks.  It is no longer a hard codec limit: the
 * codec sizes its scratch dynamically off the runtime N (carried in the
 * per-block uint16 wire header), so any block size in [1, PIVCO_WIRE_MAX_N]
 * works without a recompile.  Defaults are the per-arch sweet spots measured
 * across M4 / Granite Rapids / Zen5 (see issue #2): bigger blocks amortise
 * per-block table/tree reload, which dominates on the smaller-L1 x86 parts. */
#ifndef PIVCO_BLOCK_SIZE
/* 32K is the cross-arch sweet spot measured across the full fleet (12 EC2
 * parts + M4): every uarch peaks at or near 32K, and the fast modern AVX-512
 * parts regress past it (cache cliff).  See docs/BLOCK_SIZE.md.
 *
 * Apple M-series is the exception: 32K regresses its text dists (its wide
 * L1/L2 already absorbs the per-block cost at 16K, after which the larger
 * working set only hurts), so it defaults to 16K.  Gated compile-time on
 * macOS/arm64 — a macOS arm64 binary is always Apple Silicon, and a macOS
 * binary's ISA is fixed at build time, so the gate is exact.  An explicit
 * -DPIVCO_BLOCK_SIZE still wins (this whole block is #ifndef-guarded).
 * Only M4 was measured; M1–M3 are assumed to share the wide-L1 behaviour.
 * A principled runtime gate keyed on cache size (which is the real cause)
 * could supersede this later. */
#if defined(__APPLE__) && defined(__aarch64__)
#define PIVCO_BLOCK_SIZE 16384
#else
#define PIVCO_BLOCK_SIZE 32768
#endif
#endif

/* Hard upper bound on a block's symbol count: the per-block wire header
 * stores N as a uint16 little-endian field, so N must fit in 16 bits. */
#define PIVCO_WIRE_MAX_N 65535

#define PIVCO_MAX_SYMBOLS   256

/* Maximum Huffman code length (length-limited Huffman, like huf0).
 * Capping at 11 matches zstd's huf0 max and bounds the canonical
 * decode_sym/decode_len tables at 2KB each (fits L1 cleanly).  Trees
 * for distributions with naturally deeper Huffman (e.g. prose_pride at
 * natural max 16) get reshaped: rare deep leaves are pulled up and
 * other leaves get longer codes per Kraft.  Compression cost is small
 * (~0.5-1% on text-like data, 0% on most distributions whose natural
 * max is <=11 anyway).  Decode is slightly faster on text-like
 * distributions, slightly slower on geometric — net wash to small win
 * on real workloads.  Override at build time if you need
 * different trade-off: -DPIVCO_MAX_CODE_LEN=15. */
#ifndef PIVCO_MAX_CODE_LEN
#define PIVCO_MAX_CODE_LEN  11
#endif

/* Maximum encoded size for one block (generous upper bound):
   Sum of code bits across all symbols. Worst case: all 8-bit codes
   => N bytes. Plus rounding overhead per tree node. */
#define PIVCO_MAX_ENCODED_SIZE (PIVCO_BLOCK_SIZE * 2)

/* ---------- Error codes ---------- */

#define PIVCO_OK            0
#define PIVCO_ERR_NULL      (-1)
#define PIVCO_ERR_OVERFLOW  (-2)
#define PIVCO_ERR_CORRUPT   (-3)
#define PIVCO_ERR_EMPTY     (-4)

/* ---------- Huffman tree node (for PIVCO tree-walk) ---------- */

/* Compact tree: nodes stored in array, indexed by node ID.
   Max nodes = 2 * MAX_SYMBOLS - 1 = 511.
   Leaf: symbol >= 0.  Internal: symbol = -1, left/right are children. */
#define PIVCO_MAX_TREE_NODES (2 * PIVCO_MAX_SYMBOLS - 1)

typedef struct {
    int16_t symbol;   /* >= 0 for leaf, -1 for internal */
    int16_t left;     /* child node index (bit=0) */
    int16_t right;    /* child node index (bit=1) */
} pivco_tree_node_t;

/* ---------- Per-node decode dispatch ----------
 *
 * Classifies each tree node at build_table time so the decoder can
 * dispatch via a single switch on table->node_type[node_id] instead of
 * per-call conditional chains.  Classification is by "leafness" of the
 * children alone: bottom-up merges consume a leaf child's symbol
 * directly, so a leaf node itself is never dispatched — the parent's
 * merge materializes it.
 *
 * Same classification applies to all backends (scalar, NEON, AVX-512,
 * SSE).
 */
typedef enum {
    PIVCO_NODE_INTERNAL_FULL = 0,  /* both children internal — general partition/merge */
    PIVCO_NODE_INTERNAL_FLAT,      /* flat_depth[i] >= 2 — flat-subtree fast path */
    PIVCO_NODE_BOTH_LEAVES,        /* both children leaves — merge_cst_cst, partition_none */
    PIVCO_NODE_LEAF_LEFT,          /* left child leaf, right internal — merge_cst_vec, partition_right */
    PIVCO_NODE_LEAF,               /* leaf — consumed by the parent merge, never dispatched */
} pivco_node_type_t;

/* ---------- Pre-order walk schedule ----------
 *
 * The execution form of the implicit rank-range tree (issue #7): one
 * record per VISIBLE internal node of the codec walk, in pre-order.
 * Leaves emit no record (parents consume their symbols via cst merges),
 * and nodes buried inside flat subtrees don't exist.  Child links are
 * implicit in the pre-order layout: a node's LEFT child (when it has a
 * record) is the next record, and its RIGHT child sits at the
 * precomputed `right` offset — so the walk passes record indices BY
 * VALUE, exactly like the retired tree walk passed node ids.  (An
 * earlier form streamed a by-reference cursor instead; the extra
 * live-across-calls state measurably fed x86 register pressure — see
 * results/sweep_2026-07-10_aws_SUMMARY.md.)  A child subtree that
 * receives 0 elements is simply not visited.
 *
 *   kd     kind in the low 2 bits; for FLAT, the subtree depth D in the
 *          high bits (kd >> 2).
 *   param  FULL: thr (max rank of the left subtree, the partition
 *          threshold).  Otherwise rank_begin — the flat c2s slice base
 *          for FLAT (symbols are rank_to_sym[param..]), the leaf
 *          symbol(s) rank_to_sym[param] (+ [param+1] for PAIR), and
 *          numerically == thr for PAIR / LEAF_LEFT.
 *   right  offset from this record to the right child's record:
 *          1 + the left subtree's record count for FULL, 1 for
 *          LEAF_LEFT (the lone left leaf has no record), 0 (unused)
 *          for FLAT / PAIR.
 */
typedef enum {
    PIVCO_SCHED_FULL      = 0,  /* both children internal — K_right header */
    PIVCO_SCHED_FLAT      = 1,  /* flat subtree, D = kd >> 2 — no header  */
    PIVCO_SCHED_PAIR      = 2,  /* both children leaves — no K_right      */
    PIVCO_SCHED_LEAF_LEFT = 3,  /* left child lone leaf — K_right header  */
} pivco_sched_kind_t;

typedef struct {
    uint8_t kd;     /* kind | (flat D << 2) */
    uint8_t param;  /* rank_begin or thr — see kind */
    uint8_t right;  /* offset to the right child's record — see above */
} pivco_sched_rec_t;

/* ---------- Decode table ----------
 *
 * EVERYTHING the production decoder reads: the walk program plus the
 * rank -> symbol permutation (~1 KB total).  Ranks are leaves in
 * MSB-aligned-codeword order, so every subtree is a contiguous rank
 * range; sched[] is that implicit tree rendered into pre-order records
 * (issue #7).  Degenerate single-symbol tables get TWO ranks of the
 * same symbol, so num_ranks == 2 there and the walk needs no special
 * case (num_ranks == the used-symbol count otherwise).
 *
 * Build with pivco_huffman_build_decode_table() — no memset, no explicit
 * tree, sized for the "rebuild per small input" decode path.  The full
 * pivco_huffman_table_t embeds one as `dec`, so encode-side tables can
 * also decode. */
typedef struct {
    uint16_t num_ranks;
    uint16_t sched_len;
    uint8_t  rank_to_sym[PIVCO_MAX_SYMBOLS];        /* rank -> symbol (in-order) */
    pivco_sched_rec_t sched[PIVCO_MAX_SYMBOLS - 1]; /* pre-order walk program */
} pivco_huffman_decode_table_t;

/* ---------- Huffman table ---------- */

/* Arch-specific precomputed gather tables for prim_enc_init.  Every pointer is
 * NULL unless the host arch fills it (only x86 SSE/AVX2 today, for the 4tab
 * no-shift merge).  Backends that don't need it ignore the struct; a backend
 * that does asserts the fields it uses are non-NULL.  The struct type is
 * arch-invariant (always these fields) — only the backing storage in the table
 * is arch-gated — so it never degenerates to an empty struct. */
typedef struct {
    const uint16_t *s2r_hi;      /* sym_to_rank[s] << 8 (u16) — x86 2tab merge */
} pivco_huffman_enc_init_aux_t;

/* ---------- Codec table (modern encoder + decoder, minimal) ----------
 *
 * EVERYTHING the production codec reads on BOTH sides: the decode core
 * (walk program + rank -> symbol), plus the encoder's symbol -> rank
 * gather (with its x86 aux) and the code lengths for wire-header
 * serialization.  ~1.5 KB (~2 KB on x86) vs the 13 KB full table —
 * sized, like the decode table, for rebuild-per-window adaptive use.
 * sym_to_rank is the inverse of dec.rank_to_sym (0 for symbols not in
 * the table); code_len of the degenerate single-symbol table is 1
 * whatever length the input claimed, matching the full build.
 *
 * Build with pivco_huffman_build_codec_table() (from frequencies — the
 * encoder side owns counts) or .._from_code_lens() (predefined /
 * static lengths).  Encode via pivco_huffman_encode_ct(); decode via
 * pivco_huffman_decode_dt(&ct->dec).  enc_init_aux is self-referential:
 * rebuild, don't bitwise-copy. */
typedef struct {
    pivco_huffman_decode_table_t dec;
    uint8_t code_len[PIVCO_MAX_SYMBOLS];
    uint8_t sym_to_rank[PIVCO_MAX_SYMBOLS];
#if defined(__x86_64__) || defined(__i386__)
    uint16_t enc_init_hi[PIVCO_MAX_SYMBOLS];   /* backing for enc_init_aux.s2r_hi */
#endif
    pivco_huffman_enc_init_aux_t enc_init_aux;
} pivco_huffman_codec_table_t;

typedef struct {
    /* Decode core — the codec's decode path reads ONLY this (encode also
     * streams dec.sched).  See pivco_huffman_decode_table_t.  The
     * canonical rank-range form (per-rank flat depths + MSB-aligned
     * codewords) is a build-time intermediate, not stored: the schedule
     * is its rendering, and pivco_huffman_build_explicit_tree can
     * reconstruct the full tree from it. */
    pivco_huffman_decode_table_t dec;

    /* "partbyrank" encode: a subtree's leaves are a contiguous rank range,
     * so per-node routing is `rank > thr` (8-bit, vs a 16-bit code
     * bit-test) and a flat subtree's local code is `rank - rank_begin`.
     * Filled by pivco_huffman_build_table; byte-identical wire output. */
    uint8_t  sym_to_rank[PIVCO_MAX_SYMBOLS];        /* in-order leaf rank per symbol */
#if defined(__x86_64__) || defined(__i386__)
    /* Backing storage for enc_init_aux — the x86 2tab merge hi table (sym_to_rank
     * << 8).  Other arches don't allocate it.  Filled by pivco_huffman_build_table. */
    uint16_t enc_init_hi[PIVCO_MAX_SYMBOLS];
#endif
    /* Aux gather tables: pointers into the arch-gated storage above (x86) or all
     * NULL (other arches).  Self-referential — rebuild, don't bitwise-copy, a
     * table after pivco_huffman_build_table. */
    pivco_huffman_enc_init_aux_t enc_init_aux;

    /* Per-symbol code info (wire header serialization, trad codecs, tools) */
    uint16_t code[PIVCO_MAX_SYMBOLS];       /* canonical Huffman code */
    uint8_t  code_len[PIVCO_MAX_SYMBOLS];   /* code length (0 = unused) */
    uint16_t sym_count[PIVCO_MAX_CODE_LEN + 1]; /* per-length histogram */

    uint8_t  max_len;
    uint8_t  min_len;
    uint16_t num_symbols;

    /* ================= ON-DEMAND TAIL =================
     *
     * Everything from decode_sym down is NOT touched by the normal build
     * (not even zeroed — the build clears the struct only up to
     * offsetof(decode_sym), which is most of its small-input cost).
     * Contents are undefined until the matching on-demand builder runs:
     *
     *   decode_sym / decode_len       pivco_huffman_build_traditional_table
     *   tree / node_type / flat_* /
     *   split_rank / flat_base_rank /
     *   max_leaf_depth                pivco_huffman_build_explicit_tree
     *
     * The production codec reads none of these. */

    /* Flat decode table: 2^MAX_CODE_LEN entries (for traditional decoder) */
    uint8_t  decode_sym[1 << PIVCO_MAX_CODE_LEN];
    uint8_t  decode_len[1 << PIVCO_MAX_CODE_LEN];

    /* Explicit tree (analysis / debug; see pivco_huffman_build_explicit_tree) */
    pivco_tree_node_t tree[PIVCO_MAX_TREE_NODES];
    int16_t tree_root;
    int16_t tree_node_count;
    uint8_t  split_rank[PIVCO_MAX_TREE_NODES];      /* max rank in node's left subtree */
    uint8_t  flat_base_rank[PIVCO_MAX_TREE_NODES];  /* min rank in a flat subtree */

    /* Flat-subtree fast path: per-node, if flat_depth[i] >= 2 then node i
       is the root of a MAXIMAL flat subtree of depth D = flat_depth[i]
       (all 2^D leaves at the same relative depth).  Encoder emits N*D
       packed bits at this node instead of D levels of bitmaps; decoder
       reads N*D bits and uses flat_code_to_sym[flat_offset[i] + code]
       per element.  Pool sum of 2^D across flat subtrees <= num_symbols. */
    uint8_t  flat_depth[PIVCO_MAX_TREE_NODES];
    uint16_t flat_offset[PIVCO_MAX_TREE_NODES];
    uint8_t  flat_code_to_sym[PIVCO_MAX_SYMBOLS];

    /* Max leaf depth in the subtree rooted at this node, relative to
     * the global tree.  Historically the encoder's u8-repack test;
     * analysis-only now. */
    uint8_t  max_leaf_depth[PIVCO_MAX_TREE_NODES];

    /* Decode dispatch type per node — see pivco_node_type_t.
     * Analysis-only now (the codec dispatches on sched[] kinds). */
    uint8_t  node_type[PIVCO_MAX_TREE_NODES];
} pivco_huffman_table_t;

/* ---------- Implementation selection ---------- */

typedef enum {
    PIVCO_IMPL_AUTO = 0,
    PIVCO_IMPL_SCALAR,
    PIVCO_IMPL_NEON
} pivco_impl_t;

void         pivco_huffman_set_impl(pivco_impl_t impl);
pivco_impl_t pivco_huffman_get_impl(void);

/* Runtime toggle for the encoder's FSE-dispatch path (v0.2+ wire
 * format).  Default is enabled.  When set to 0, encode_node_* skip
 * the FSE-compress attempt and always emit raw bitmaps with marker=0.
 * Useful for benchmarking the no-FSE codec path without rebuilding,
 * and for cases where the FSE overhead isn't worth its ratio gain on
 * a specific dataset (e.g. proba80-like distributions where the
 * partition bitmaps are too small for FSE to beat marker overhead).
 * The decoder always supports both marker=0 (raw) and marker!=0
 * (FSE) so files produced with FSE enabled can be decoded with FSE
 * disabled and vice versa. */
void pivco_huffman_set_fse_enabled(int enabled);
int  pivco_huffman_get_fse_enabled(void);

/* ---------- Tree-shape mode (build-time) ----------
 *
 * Experimental knob for paper-style ablations.  Changes how the chunks
 * are decomposed inside pivco_huffman_build_table; the codec downstream
 * picks up the resulting table->flat_depth/flat_offset/code[] uniformly.
 *
 *   OPTIMIZED      production: per length L, decompose c_L by its set
 *                  bits.  Produces non-canonical codes that maximize
 *                  flat-D>=2 subtree coverage.
 *   NAIVE          every symbol is a D=0 singleton.  Tree shape ==
 *                  pure canonical Huffman; no leaf fusion, no flat
 *                  subtrees.  Slowest decode; best baseline for "ph
 *                  without any tree optimizations vs Huff0".
 *   FUSED          allow D=1 sibling pairs but no D>=2 flats.  Tree
 *                  shape == canonical with `scatter_two` / `merge_two`
 *                  leaf fusion only.
 *   CANONICAL_FLAT chunks are derived from canonical code positions:
 *                  greedy peel the largest 2^k chunk such that the
 *                  canonical start code is 2^k-aligned and 2^k <=
 *                  remaining.  Produces canonical codes that happen
 *                  to contain flat subtrees; isolates the gain from
 *                  the OPTIMIZED non-canonical reorganization.
 *
 * The flag must be set BEFORE pivco_huffman_build_table() runs; both
 * encoder and decoder side must use the same value (the wire format
 * carries only code lengths, not tree shape).  Default = OPTIMIZED. */
typedef enum {
    PIVCO_TREE_MODE_OPTIMIZED       = 0,
    PIVCO_TREE_MODE_NAIVE           = 1,
    PIVCO_TREE_MODE_FUSED           = 2,
    PIVCO_TREE_MODE_CANONICAL_FLAT  = 3,
} pivco_tree_mode_t;
void pivco_huffman_set_tree_mode(pivco_tree_mode_t mode);
pivco_tree_mode_t pivco_huffman_get_tree_mode(void);

/* ---------- FSE table-usage stats (debug instrumentation) ----------
 *
 * Per-table-id counters incremented inside the encoder every time an
 * FSE-coded bitmap is committed.  Slot 0 = "FSE attempted but did not
 * commit"; slots 1..PIVCO_FSE_NUM_TABLES = pivco_fse_freq[] table picked.
 * MUST be >= PIVCO_FSE_NUM_TABLES + 1 (static-asserted in pivco_fse.c).
 * Not thread-safe; intended for single-threaded analysis runs. */
#define PIVCO_FSE_STATS_SLOTS 51
void pivco_huffman_fse_stats_reset(void);
void pivco_huffman_fse_stats_get(uint64_t commit_count[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t attempt_count[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_in[PIVCO_FSE_STATS_SLOTS],
                                 uint64_t bytes_out[PIVCO_FSE_STATS_SLOTS]);

/* Per-root-event log: one entry per block's root-node visit.
 * Captures table_id chosen (0 if below threshold / no table), the
 * observed p_major, whether the FSE commit succeeded, and byte counts.
 * Useful for showing how a single tree position (the root) adapts
 * across blocks of the same file. */
typedef struct {
    int    table_id;     /* 0 if below MIN_THRESHOLD / no FSE attempt */
    double p_major;      /* observed max(n_left,n_right)/n */
    int    committed;    /* 1 if FSE emitted, 0 otherwise */
    int    nbytes_in;    /* raw bitmap byte count */
    int    nbytes_out;   /* fse_len if committed; nbytes_in if not */
} pivco_huffman_fse_root_event_t;

int  pivco_huffman_fse_root_count(void);
void pivco_huffman_fse_root_get(int idx, pivco_huffman_fse_root_event_t *out);

/* ---------- Joint length/flat-shape optimization (experimental) ----------
 *
 * When lambda > 0, pivco_huffman_build_table deliberately distorts the
 * code-length histogram away from the Huffman optimum, trading
 * compressed bits for a flatter decode tree: lambda is the price, in
 * bits per symbol-occurrence, of one merge pass (the objective is
 * sum n_s * (len_s + lambda * (len_s - flat_depth_s)); see
 * docs/JOINT-LENGTHS.md for the model and the exact DP).  0 (default)
 * = off.  Encoder-side only: the wire still carries plain lengths and
 * ANY decoder reads the output.  Set before build_table. */
void   pivco_huffman_set_joint_lambda(double lam);
double pivco_huffman_get_joint_lambda(void);
int    pivco_joint_optimize_lengths(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                    uint8_t lengths[PIVCO_MAX_SYMBOLS]);

/* Internal fast path for the table builds: identical to
 * pivco_joint_optimize_lengths, but takes the build's already-sorted
 * leaf array (frequency ascending, the two-queue input) instead of
 * re-scanning and re-sorting the 256-entry freq table — that sort was
 * most of the joint pass's fixed overhead.  Layout matches the
 * builder's internal leaf_t. */
typedef struct { uint64_t freq; uint16_t sym; } pivco_huffman_leaf_t;
int pivco_joint_optimize_lengths_leaves(const pivco_huffman_leaf_t *leaf_asc,
                                        int n_used,
                                        uint8_t lengths[PIVCO_MAX_SYMBOLS]);

/* Solve granularity: 1 (default) = exact DP; 2/4/8 = solve on
 * freq-sorted symbol groups of that size — 4x/16x/64x fewer DP states
 * for a ~0.13 %/0.25 %/0.4 % mean objective loss on lits-style data
 * (the per-window adoption guard still applies); 0 = auto, which
 * picks by alphabet size so the solve stays roughly <= 10 us;
 * -1 = greedy boundary nudger, no DP at all (~0.2 us solve). */
void pivco_huffman_set_joint_granularity(int g);
int  pivco_huffman_get_joint_granularity(void);

/* Adoption-guard thresholds (defaults 1.015, 0.90): a joint result is
 * adopted only if modeled bits <= bits_cap * baseline and modeled
 * merge passes <= pass_cap * baseline; otherwise the window keeps its
 * plain Huffman lengths.  Pass big values to disable (experiments). */
void pivco_huffman_set_joint_guard(double bits_cap, double pass_cap);

/* Per-flat-depth kernel costs kappa[b], b = 0..8, in merge-pass units
 * (measured flat-kernel time per symbol at depth b, divided by merge
 * time per symbol).  Default all zeros = kernels modeled free, the
 * historical objective.  Real tables are not monotone in b; all solve
 * paths (exact slot DP via per-level cost-ordered sweeps, mass DP,
 * coarse, nudge) and the adoption guard price them consistently.
 * NULL resets to zeros. */
void pivco_huffman_set_joint_kappa(const double kappa[9]);

/* Merge-kind costs for the guard's kind-aware time model, in units of
 * a full (two-internal-children) merge pass: mu_cst = merge with a
 * lone-leaf child (cheap cst kernels); prefill in [0,1] = fraction of
 * the prefilled top symbol's weight its parent merge skips.  The D=1
 * pair kernel is priced by kappa[1].  Defaults (1, 0) with kappa = 0
 * reproduce the kind-blind pass model exactly. */
void pivco_huffman_set_joint_merge_costs(double mu_cst, double prefill);

/* ---------- Table construction ---------- */

int pivco_huffman_build_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                              pivco_huffman_table_t *table);

/* Build ONLY the ~1 KB decode table from code lengths — the minimal
 * decode-side setup (no explicit tree, no encode/trad fields, no memset).
 * Deterministic from the lengths, so it matches any encoder-side
 * pivco_huffman_build_table over the same lengths.
 *
 * Rejects any length over PIVCO_MAX_CODE_LEN with PIVCO_ERR_CORRUPT
 * (by bin accounting — no separate validation pass), and
 * non-Kraft-complete lengths likewise via the schedule generation. */
int pivco_huffman_build_decode_table(const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
                                     pivco_huffman_decode_table_t *dt);

/* Build the minimal codec table (see pivco_huffman_codec_table_t) from
 * frequencies: derive code lengths (two-queue + limiting), then the decode
 * core plus the encoder's sym_to_rank (one inversion of rank_to_sym) and
 * aux.  This is the encoder-side twin of build_decode_table; the header
 * lengths to transmit are left in ct->code_len. */
int pivco_huffman_build_codec_table(const uint64_t freq[PIVCO_MAX_SYMBOLS],
                                    pivco_huffman_codec_table_t *ct);

/* Same, from predefined code lengths (static tables, table reuse).  Costs
 * build_decode_table plus the sym_to_rank inversion.  Rejects invalid /
 * non-Kraft-complete lengths exactly like build_decode_table. */
int pivco_huffman_build_codec_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_huffman_codec_table_t *ct);

/* Build a Huffman table from already-known code lengths (the path used by
 * decoders that recovered code_lens from a wire format).  The tree is fully
 * determined by the lengths -- within-tier order is symbol-value ascending --
 * so encoder and decoder reconstruct identical tables with no extra wire info.
 *
 * Internally synthesises power-of-two frequencies that reproduce the lengths
 * and runs the same build pipeline as pivco_huffman_build_table.
 *
 * (Through wire v0.3 this also took a rank_within_tier array to reproduce a
 * frequency-based within-tier order; that ordering was dropped in v0.4 -- it
 * required extra wire bytes and only masked a frequency-blind FSE commit
 * policy.  See the reshape note in huffman_table.c.) */
int pivco_huffman_build_table_from_code_lens(
    const uint8_t code_lens[PIVCO_MAX_SYMBOLS],
    pivco_huffman_table_t *table);

/* Fill the 2^MAX_CODE_LEN flat decode table (decode_sym/decode_len) used only
 * by the traditional flat-table decoder (trad_huffman_decode*).  Call after
 * building the table; pivco_huffman_build_table no longer fills it (the
 * production tree-walk decoder does not need it). */
void pivco_huffman_build_traditional_table(pivco_huffman_table_t *table);

/* Materialize the explicit tree (tree[] / node_type[] / flat_depth[] /
 * flat_offset[] / flat_code_to_sym[] / split_rank[] / flat_base_rank[] /
 * max_leaf_depth[]) for analysis / debug tools.  Call after building the
 * table; the production codec streams sched[] and never reads these, so
 * the normal build skips them (the bulk of small-input table-build time,
 * issue #7).  Reconstructed from the schedule + rank arrays with nodes
 * numbered in pre-order (root = 0) — same topology as the retired inline
 * build, different raw node indices. */
void pivco_huffman_build_explicit_tree(pivco_huffman_table_t *table);

/* ---------- PIVCO Huffman encode/decode (variable-N blocks, N ≤ PIVCO_BLOCK_SIZE) ----------
 *
 * Encode takes the symbol count `n` and writes a 2-byte LE N header at the
 * start of the encoded stream (see pivco_huffman_wire.h).  Decode reads N
 * from the wire — no `n` parameter — and writes N symbols to `symbols`,
 * which must have room for at least N bytes (caller must size for the
 * worst case, typically PIVCO_BLOCK_SIZE).  N must satisfy
 * 1 ≤ N ≤ PIVCO_BLOCK_SIZE; values outside that range return error.
 */

int pivco_huffman_encode(const uint8_t *symbols, size_t n,
                         const pivco_huffman_table_t *table,
                         uint8_t *out, size_t *out_len);

int pivco_huffman_decode(const uint8_t *in, size_t in_len,
                         const pivco_huffman_table_t *table,
                         uint8_t *symbols, size_t *consumed);

/* Decode against a bare decode table (see pivco_huffman_build_decode_table)
 * — the minimal-setup path.  pivco_huffman_decode(table, ...) is exactly
 * this over &table->dec.  The per-backend *_dt entries below are the real
 * implementations; the table-taking forms are thin shims. */
int pivco_huffman_decode_dt(const uint8_t *in, size_t in_len,
                            const pivco_huffman_decode_table_t *dt,
                            uint8_t *symbols, size_t *consumed);

/* Encode against the minimal codec table — same walk, same wire bytes as
 * pivco_huffman_encode over a full table built from the same lengths.
 * The per-backend *_ct entries below are the real implementations; both
 * table-taking forms are thin shims over a shared core. */
int pivco_huffman_encode_ct(const uint8_t *symbols, size_t n,
                            const pivco_huffman_codec_table_t *ct,
                            uint8_t *out, size_t *out_len);

int pivco_huffman_encode_scalar(const uint8_t *symbols, size_t n,
                                const pivco_huffman_table_t *table,
                                uint8_t *out, size_t *out_len);
int pivco_huffman_encode_scalar_ct(const uint8_t *symbols, size_t n,
                                   const pivco_huffman_codec_table_t *ct,
                                   uint8_t *out, size_t *out_len);

int pivco_huffman_decode_scalar(const uint8_t *in, size_t in_len,
                                const pivco_huffman_table_t *table,
                                uint8_t *symbols, size_t *consumed);
int pivco_huffman_decode_scalar_dt(const uint8_t *in, size_t in_len,
                                   const pivco_huffman_decode_table_t *dt,
                                   uint8_t *symbols, size_t *consumed);

#ifdef PIVCO_HAS_NEON
int pivco_huffman_encode_neon(const uint8_t *symbols, size_t n,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);
int pivco_huffman_encode_neon_ct(const uint8_t *symbols, size_t n,
                              const pivco_huffman_codec_table_t *ct,
                              uint8_t *out, size_t *out_len);

/* Bottom-up merge decode (NEON). */
int pivco_huffman_decode_bu_neon(const uint8_t *in, size_t in_len,
                                  const pivco_huffman_table_t *table,
                                  uint8_t *symbols, size_t *consumed);
int pivco_huffman_decode_bu_neon_dt(const uint8_t *in, size_t in_len,
                                    const pivco_huffman_decode_table_t *dt,
                                    uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_SSE4
int pivco_huffman_encode_x86(const uint8_t *symbols, size_t n,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);
int pivco_huffman_encode_x86_ct(const uint8_t *symbols, size_t n,
                              const pivco_huffman_codec_table_t *ct,
                              uint8_t *out, size_t *out_len);

/* Bottom-up merge decode (x86 SSE4.1 / AVX-512 VBMI2). */
int pivco_huffman_decode_bu_x86(const uint8_t *in, size_t in_len,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *symbols, size_t *consumed);
int pivco_huffman_decode_bu_x86_dt(const uint8_t *in, size_t in_len,
                                   const pivco_huffman_decode_table_t *dt,
                                   uint8_t *symbols, size_t *consumed);
#endif

/* Prior experimental NEON variants (neon2, neon2b, neon_fused_1leaf)
 * are preserved under extras/ as negative results. See extras/README_*
 * files for writeups. */

/* Prefix-radix research backend retired to extras/pivco_huffman_neon_prefix.c
 * (alongside its bench_prefix_profile.c and pivco_huffman_neon_common.h).
 * BU on the standard 2-way wire format beats it on all 29 distributions
 * across all 7 EC2 test hosts; no production caller remained.  See
 * docs/PREFIX_RADIX.md for the historical design record. */

#ifdef PIVCO_HAS_SVE
int pivco_huffman_encode_sve(const uint8_t *symbols, size_t n,
                              const pivco_huffman_table_t *table,
                              uint8_t *out, size_t *out_len);

int pivco_huffman_decode_sve(const uint8_t *in, size_t in_len,
                              const pivco_huffman_table_t *table,
                              uint8_t *symbols, size_t *consumed);
#endif

#ifdef PIVCO_HAS_AVX512
int pivco_huffman_encode_avx512(const uint8_t *symbols, size_t n,
                                 const pivco_huffman_table_t *table,
                                 uint8_t *out, size_t *out_len);
int pivco_huffman_encode_avx512_ct(const uint8_t *symbols, size_t n,
                                 const pivco_huffman_codec_table_t *ct,
                                 uint8_t *out, size_t *out_len);

/* Bottom-up merge decode (AVX-512 VBMI2). */
int pivco_huffman_decode_bu_avx512(const uint8_t *in, size_t in_len,
                                    const pivco_huffman_table_t *table,
                                    uint8_t *symbols, size_t *consumed);
int pivco_huffman_decode_bu_avx512_dt(const uint8_t *in, size_t in_len,
                                      const pivco_huffman_decode_table_t *dt,
                                      uint8_t *symbols, size_t *consumed);
#endif

/* Top-down (TD) decode entry points have been retired (2026-05-14).
 * BU is the production decoder on every platform.  TD implementations
 * still live in the legacy .c files as now-unreachable static functions;
 * step 3.8 of the unify-framework refactor retires them along with the
 * legacy .c files when codec.c takes over the encode/decode entries.
 * See extras/legacy_td/README.md for the git-archaeology pointer. */

/* ---------- Traditional Huffman encode/decode (for comparison) ---------- */

int trad_huffman_encode(const uint8_t *symbols, size_t n_symbols,
                        const pivco_huffman_table_t *table,
                        uint8_t *out, size_t *out_len, size_t *out_bits);

int trad_huffman_decode(const uint8_t *in, size_t in_bits,
                        const pivco_huffman_table_t *table,
                        uint8_t *symbols, size_t n_symbols);

/* SotA 4-stream encode/decode (huff0-style) */
int trad_huffman_encode_4s(const uint8_t *symbols, size_t n_symbols,
                           const pivco_huffman_table_t *table,
                           uint8_t *out, size_t *out_len);

int trad_huffman_decode_4s(const uint8_t *in, size_t in_len,
                           const pivco_huffman_table_t *table,
                           uint8_t *symbols, size_t n_symbols);

/* ---------- Instrumentation ---------- */
void pivco_instrument_node_size(int n);
void pivco_dump_node_size_hist(void);

#ifdef __cplusplus
}
#endif

#endif /* PIVCO_HUFFMAN_H */

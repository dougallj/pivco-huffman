# pivcoh wire v6 — pre-order bitmaps, popcounted splits: M4-mini A/B

2026-07-17, M4 Mac mini (quiet box, Apple clang 17.0.0, macOS 25.2.0),
single-binary 3-way A/B (v4 baseline sed-namespaced to `pivcohB_*`,
right-first v6 to `pivcohR_*`, all engines in one harness, 12 rounds
with rotating engine order, medians).  Harness: `ab_wire3.c` (staged at
`~/claude/work/pivco-huffma-e721f1d7/ab-preorder/` on the mini); raw
outputs in `pivcoh_v6_preorder_bitmaps-m4mini-20260717/`.  Branch:
`pivcoh-neon-preorder-bitmaps` off `pivcoh-neon` @ 0efba3b.

## What v6 is

Drop every per-node K_right header and move each internal node's raw
bitmap from its v4 POST-order slot (after the children, read at merge
time) to the node's PRE-order slot (at entry).  The decoder popcounts
the bitmap at node entry to derive the split — K_right was pure
redundancy — and keeps the pointer for the merge, so each bitmap is
touched twice.  The fused v4 walk, kernels, trees and lens wire are
untouched; the encoder now streams bitmaps straight to the wire at
partition time, deleting the per-node staged-bitmap memcpy (the v4
stage existed only because a post-order bitmap's stream position
depends on the children's encoded sizes).

Child-region order became a free variable (the split is known before
the children either way), so both orders were benched:

- **A** = larger-K child first, ties left-first (v4's order)
- **R** = right child always first (`PIVCOH_RIGHT_FIRST=1`).  When the
  right child is the smaller this order parks the small result in the
  partner while the big sibling ping-pongs beyond it, growing partner
  capacity from floor(K/2) toward K — paid for exactly by v4's 0.5n
  hostile-stray headroom, which a derived split makes dead capacity
  (a popcounted bitmap cannot contradict itself), so
  `PIVCOH_DECODE_SCRATCH_SIZE` stays 2n + 128.

Verification (final 4-accumulator code): bench_pivcoh_check (165
tables, 660 blocks, 52,800 hostile decodes, trees
production-identical) on the M1 Max and under ASan/UBSan on the mini,
for all three wire configs (larger-first, right-first,
DIAG_STORE_KR); plus a 200 K hostile-stream fuzz per config on the
mini (`fuzz_v6.c` — multi-byte garbles, truncations, noise streams,
decoded into exact malloc'd out/2n+128 scratch so redzones enforce
the contract byte-for-byte).  Harness gates per window: identical
code_len across engines, |A| = |R| (order is a pure region
permutation), |B| >= |A|, and full roundtrips for all three engines.
(ASan hangs on the M1 Max laptop — sanitizer runs live on the mini.)

## Headline

geomean over the 12 silesia-lits files (dec/enc: engine/v4, >1 = v6
faster; sv% = compressed bytes saved vs v4).  Raw rows: `ab2_G*.txt`.

| regime            | dec A/B | dec R/B | enc A/B | enc R/B |   sv% |
|-------------------|---------|---------|---------|---------|-------|
| G=4 K   SIMPLEST  |   0.859 |   0.891 |   1.053 |   1.065 | 0.736 |
| G=4 K   joint -1  |   0.882 |   0.893 |   1.037 |   1.047 | 0.382 |
| G=16 K  SIMPLEST  |   0.931 |   0.936 |   1.040 |   1.047 | 0.208 |
| G=16 K  joint -1  |   0.946 |   0.944 |   1.028 |   1.032 | 0.106 |
| G=32 K  SIMPLEST  |   0.950 |   0.949 |   1.024 |   1.028 | 0.104 |
| G=32 K  joint -1  |   0.959 |   0.954 |   1.019 |   1.022 | 0.055 |

- **Ratio**: helps, but only by the header bytes — 0.74 % of
  compressed size at 4 K SIMPLEST, fading to 0.06 % at 32 K joint
  (headers are 1-2 B per non-flat internal node; joint trees have few,
  large windows amortize them).
- **Decode**: loses everywhere, −4 % to −14 %, worst exactly where
  the ratio win is best (deep ragged 4 K SIMPLEST trees = most nodes
  per byte).  No regime came out ahead.
- **Encode**: wins everywhere, +2 % to +6.5 % — the staged-bitmap
  memcpy deletion, biggest at small G.
- **Child order**: right-first wins decode at 4 K (+1.1 to +3.2 pp
  over larger-first) and encode in every regime; the two orders tie
  at 16-32 K.  With the split on a popcount chain instead of a 4-5 c
  byte load, the data-dependent "which child first" branch resolves
  late — a fixed order removes it from the sequencing path.  (On v4,
  where the split is a byte load, larger-first won instead.)

## The split-popcount ladder (implementation lesson)

The first cut popcounted K > 64 bitmaps with a single vpadalq
accumulator — a ~3 c/16 B serial chain, ~190 c on a 32 K root's 4 KB
bitmap, all on the walk's entry path.  Unrolling to 4 accumulators
(`a2_G*` vs original `ab_G*`) recovered 3-6 pp of the decode ratio —
over half the total loss at 32 K:

| dec, geomean      | 1-acc A/B | 4-acc A/B | 1-acc R/B | 4-acc R/B |
|-------------------|-----------|-----------|-----------|-----------|
| G=4 K   SIMPLEST  |     0.831 |     0.859 |     0.840 |     0.891 |
| G=4 K   joint -1  |     0.848 |     0.882 |     0.850 |     0.893 |
| G=16 K  SIMPLEST  |     0.882 |     0.931 |     0.884 |     0.936 |
| G=16 K  joint -1  |     0.894 |     0.946 |     0.897 |     0.944 |
| G=32 K  SIMPLEST  |     0.886 |     0.950 |     0.899 |     0.949 |
| G=32 K  joint -1  |     0.896 |     0.959 |     0.906 |     0.954 |

6 accumulators (the latency x throughput ideal: ~3 c vpadal x 2
chunks/c) was benched head-to-head in one binary (A = 4-acc, R =
6-acc, identical streams; `acc46_G*.txt`): 4 K SIMPLEST 0.854 vs
0.858, 32 K 0.951 vs 0.948 — no measurable difference, keep 4.  The
other two tiers: 256-B byte-table for K <= 16 (measured ~nothing vs
builtin — those chains are short either way) and an end-guarded u64 +
builtin popcount to K = 64.

## Decomposition (DIAG_STORE_KR)

Runs with the R slot rebuilt as `PIVCOH_DIAG_STORE_KR=1`: pre-order
bitmaps but v4's K_right header still stored ahead of each one and
read instead of popcounted (larger-first order; raw rows in
`diag2_G*.txt`, `diag3_G4096_simplest.txt`).  Caveat discovered while
re-running: measured A/B shifts by up to ~3 pp at 4 K depending on
which sibling engines share the binary (diag binary A/B 0.824/0.831
vs 3-way binary 0.857/0.859 for the same source) — code-layout
effects — so the split below uses WITHIN-binary ratios only, which
reproduced across runs (diag2 vs diag3: 0.878 vs 0.884 popcount
share at 4 K SIMPLEST).

| dec, geomean      | position only (R/B) | popcount only (A/R) |
|-------------------|---------------------|---------------------|
| G=4 K   SIMPLEST  |               0.940 |               0.881 |
| G=4 K   joint -1  |               0.925 |               0.935 |
| G=32 K  SIMPLEST  |               0.957 |               0.991 |
| G=32 K  joint -1  |               0.961 |               0.995 |

With the popcount properly unrolled, the two ingredients split
cleanly by regime:

1. **At 4 K the popcount substitution still dominates (−7 to −12 %)**.
   ~20-25 non-flat internal nodes per block (read off the header
   savings) x a ~2,300-cycle v4 block budget: the split sits on the
   walk's only serial descend chain — between two successive
   popcounts there is only ~10 instructions of walk bookkeeping, all
   kernel work happens at the leaves and on the way up — so each
   node's popcount adds its full ~5-13 c latency, hidden ~zero.
   v4's chain link was a 4-5 c byte load that pipelined with the
   bookkeeping.  Production's "stored K_right = +0 % on M4" (IDEAS.md
   2026-05-12) does NOT transfer: production BU ran popcounts as a
   batch with 4-wide cross-node ILP; the fused walk needs each split
   before it can descend, pure latency.
2. **At 32 K the position move is nearly the whole residual (−4 %;
   popcount ~−1 %)**.  Big blocks have few nodes per byte, and wide
   bitmaps popcount at streaming throughput; what remains is
   entry-side read_bm + bm-pointer bookkeeping and the merge reading
   its bitmap across the whole subtree's decode span instead of right
   where the cursor stopped (at 32 K the line can be long evicted).

## Verdict

Negative for decode, and the experiment prices the format's coupling
precisely: the K_right bytes cost 0.055-0.736 % of compressed size
and buy 4-14 % of M4 decode (after giving the popcount its best
implementation); the post-order bitmap position costs one staged
memcpy per node on the encoder (+2-6.5 % encode when removed) and
buys ~4-6 % of decode.  Keep wire v4 for real use.

Salvageable / notable:

- **Right-child-first is the better order for THIS wire** (decode
  +1-3 pp at 4 K, encode +0.3-1.2 pp everywhere, never worse beyond
  noise) — and it is affordable at all only because the derived split
  cannot lie, which turns v4's 0.5n hostile-stray pad into the
  partner capacity the order needs; scratch macro unchanged at
  2n + 128.
- **The encode win is real and portable in principle** (+2-6.5 %,
  biggest at small G): any wire that fixes the bitmap's position at
  node entry lets the partition stream it directly.  v5 (kernel-order
  sections) saw the same effect for the same reason.
- Multi-engine single-binary A/Bs carry a ~1-3 pp code-layout
  systematic between binaries; trust within-binary ratios.
- The end-of-merge r_end check survives as a free always-true
  invariant; hostile streams now decode to garbage instead of -1 when
  the garble is non-structural (truncation still detected) — the
  format has strictly less redundancy, which is exactly what was
  removed.

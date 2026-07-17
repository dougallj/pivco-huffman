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

Verification: bench_pivcoh_check (165 tables, 660 blocks, 52,800
hostile decodes, trees production-identical) on the M1 Max, both
orders; on the mini, ASan/UBSan check builds for all three wire
configs (larger-first, right-first, DIAG_STORE_KR) plus a 200 K
hostile-stream fuzz per config (`fuzz_v6.c` — multi-byte garbles,
truncations, noise streams, decoded into exact malloc'd out/2n+128
scratch so redzones enforce the contract byte-for-byte).  Harness
gates per window: identical code_len across engines, |A| = |R| (order
is a pure region permutation), |B| >= |A|, and full roundtrips for all
three engines on the mini.

## Headline

geomean over the 12 silesia-lits files (dec/enc: engine/v4, >1 = v6
faster; sv% = compressed bytes saved vs v4):

| regime            | dec A/B | dec R/B | enc A/B | enc R/B |   sv% |
|-------------------|---------|---------|---------|---------|-------|
| G=4 K   SIMPLEST  |   0.831 |   0.840 |   1.053 |   1.064 | 0.736 |
| G=4 K   joint -1  |   0.848 |   0.850 |   1.038 |   1.045 | 0.382 |
| G=16 K  SIMPLEST  |   0.882 |   0.884 |   1.046 |   1.043 | 0.208 |
| G=16 K  joint -1  |   0.894 |   0.897 |   1.033 |   1.028 | 0.106 |
| G=32 K  SIMPLEST  |   0.886 |   0.899 |   1.026 |   1.024 | 0.104 |
| G=32 K  joint -1  |   0.896 |   0.906 |   1.019 |   1.018 | 0.055 |

- **Ratio**: helps, but only by the header bytes — 0.74 % of
  compressed size at 4 K SIMPLEST, fading to 0.06 % at 32 K joint
  (headers are 1-2 B per non-flat internal node; joint trees have few,
  large windows amortize them).
- **Decode**: loses everywhere, −10 % to −17 %, worst exactly where
  the ratio win is best (deep ragged 4 K SIMPLEST trees = most nodes
  per byte).  No file in any regime came out ahead.
- **Encode**: wins everywhere, +2 % to +6 % — the staged-bitmap
  memcpy deletion, biggest at small G.
- **Child order**: right-first ≥ larger-first on decode in every
  regime (+0.2 to +1.5 pp) and on encode at 4 K.  With the split on a
  ~10-15 c popcount chain instead of a 4-5 c byte load, the
  data-dependent "which child first" branch resolves late — a fixed
  order removes it from the sequencing path entirely.

## Decomposition (DIAG_STORE_KR)

Second run with the R slot rebuilt as `PIVCOH_DIAG_STORE_KR=1`:
pre-order bitmaps but v4's K_right header still stored ahead of each
one and read instead of popcounted (larger-first order; raw rows in
`diag_G*.txt`).  That splits the decode loss into its two ingredients
(dec, geomean over the 12 files; A/B reproduced the main sweep within
noise in all four cells — 0.834/0.852/0.885/0.895):

| regime            | full v6 | position only | popcount only (implied) |
|-------------------|---------|---------------|--------------------------|
| G=4 K   SIMPLEST  |   0.834 |         0.946 |                    0.882 |
| G=4 K   joint -1  |   0.852 |         0.932 |                    0.914 |
| G=32 K  SIMPLEST  |   0.885 |         0.961 |                    0.921 |
| G=32 K  joint -1  |   0.895 |         0.963 |                    0.929 |

Both ingredients cost, and they compose multiplicatively:

1. **Popcount replacing the header read is the bigger half (−7 to
   −12 %)**.  The split sits on the walk's only serial chain — a node
   cannot place, size or start its children before its split is known
   — and a popcount is ~10-15 c of latency (u64 load + GPR->SIMD cnt
   round trip, or vcnt loop at large K) where v4's header was one
   4-5 c byte load.  A 256-B byte-popcount table for K <= 16 nodes
   (popc8, kept in the code) bought nothing measurable: the chain, not
   the op, is the cost.  Notable contrast: production measured stored
   K_right at "+0 % on M4" (IDEAS.md 2026-05-12) — but that was the
   production BU walk, where the popcount ran off the critical path.
   In pivcoh's fused descend the header bytes are NOT redundancy; they
   are buying ~7-12 % of M4 decode.
2. **The pre-order position itself costs −3.7 to −6.8 %** even with
   headers still stored.  Entry-side read_bm + bm-pointer bookkeeping
   lengthens the descend hot path, and the merge now reads its bitmap
   across the whole subtree's decode span instead of right where the
   cursor stopped (at 32 K the line can be long evicted).

## Verdict

Negative for decode, and the experiment prices the format's coupling
precisely: the K_right bytes cost 0.06-0.74 % of compressed size and
buy 10-17 % of M4 decode; the post-order bitmap position costs one
staged memcpy per node on the encoder (+2-6 % encode when removed) and
buys ~4-7 % of decode.  Keep wire v4 for real use.

Salvageable / notable:

- **Right-child-first is the better order for THIS wire** — decode
  +0.2 to +1.5 pp over larger-first in every regime, encode +1 pp at
  4 K (fixed order = no data-dependent sequencing branch waiting on
  the popcount; on v4, where the split resolves in a 1-byte load,
  larger-first won instead).  It is affordable at all only because the
  derived split cannot lie, which turns v4's 0.5n hostile-stray pad
  into the partner capacity the order needs — scratch macro unchanged
  at 2n + 128.
- **The encode win is real and portable in principle** (+2-6 %,
  biggest at small G): any wire that fixes the bitmap's position at
  node entry lets the partition stream it directly.  v5 (kernel-order
  sections) saw the same effect for the same reason.
- The end-of-merge r_end check survives as a free always-true
  invariant; hostile streams now decode to garbage instead of -1 when
  the garble is non-structural (truncation still detected) — the
  format has strictly less redundancy, which is exactly what was
  removed.


# pivcoh wire v5 — kernel-order / leaf-first sections: M4-mini A/B

2026-07-17, M4 Mac mini (quiet box), single-binary A/B (v4 baseline
sed-namespaced to `pivcohB_*`, both wires compiled into one harness,
12 interleaved rounds alternating engine order, medians).  Harness:
`ab_wire.c` (staged at `~/claude/work/pivco-huffma-e721f1d7/ab/` on the
mini); raw outputs in `pivcoh_v5_kernel_order-m4mini-20260717/`.
Working-tree state: wire v5 in `extras/pivcoh.h` on top of `pivcoh-neon`
@ 0efba3b (uncommitted at sweep time).

## What v5 is

Pure permutation of the v4 block body into kernel-order sections —
`[N][K_right headers, pre-order][flat regions, walk order][bitmaps,
reverse pre-order]` — so decode can run as phases: pass A plans the
whole block from the header section (program of slim 12-16 B ops),
phase B unpacks every flat region back-to-back, phase C runs every
merge back-to-back (the tight mvv/mcv loop).  Placement: two
depth-parity arenas (scratch macro unchanged at 2n + 128); rare spine
merges whose lying-bitmap stray could exit the pad run with an in-loop
r-cursor guard.  Encoder: headers stream direct, flats pack at a
pessimistic base, bitmaps write straight to wire positions through a
backward cursor (per-node staged-bitmap memcpy deleted); two end-of-walk
memmoves close the gaps.  Compressed size is byte-count-identical to v4
(verified block- and frame-level); trees stay production-identical;
check + 200 K-decode hostile fuzz + ASan/UBSan (mini) all pass.

## Headline: decode LOSES everywhere; encode is a wash

geomean v5/v4 over the 12 silesia-lits files (>1 = v5 faster):

| regime            | dec   | enc   |
|-------------------|-------|-------|
| G=4 K   SIMPLEST  | 0.934 | 1.009 |
| G=4 K   joint -1  | 0.955 | 0.997 |
| G=16 K  SIMPLEST  | 0.966 | 1.004 |
| G=16 K  joint -1  | 0.985 | 1.003 |
| G=32 K  SIMPLEST  | 0.954 | 0.984 |
| G=32 K  joint -1  | 0.974 | 0.992 |

No file and no regime came out ahead on decode (best single cell
reymont 16 K joint 1.003).  Encode: small wins at 4 K (staging memcpy
deleted > memmove cost), −1..−2 % at 32 K (the two section memmoves
scale with payload).

## Why (measured decomposition)

The naive first cut was −21 % dec at 4 K; three fixes recovered it to
−6.6 %, each verified on dickens 4 K/laptop then confirmed by the mini
sweep:

1. Bitmap-popcount pre-validation of stray-risky merges cost ~9-11 %
   (ragged SIMPLEST trees make most spine merges "risky") → replaced
   with a compiled-in r-cursor guard variant of mvv/mcv used only on
   those merges (l side provably cannot exit the pad; guard cost lands
   on the guarded merges alone).
2. Per-D-grouped flat phase (8 passes over the op list) cost more than
   the kernel-constant reloads it saved → single back-to-back loop.
3. Fat 48 B pointer-carrying ops made pass A a fifth of decode →
   slimmed to 12-16 B span records; pointers/flags derived inside the
   phase loops under the kernels' OOO shadow.

The irreducible residue, per cntvct phase counters: pass A (header
parse + placement walk + program stores) is ~13 % of decode at 4 K and
runs SERIAL — the v4 fused walk did the same bookkeeping interleaved
between kernel calls, where the OOO window hid it almost entirely (the
whole interpreter tax was previously measured at ~2-3 %).  Phase B + C
together run ~6 % faster than v4's entire decode (the tight loop does
pay), but not enough to buy back the serialized plan.  At 32 K a second
term appears: phase separation keeps every leaf buffer live at the B/C
boundary, so the columns touch 2n scratch + n output ≈ 128 KB = the
whole M4 L1d, where v4's ping-pong touched 1.5n + n ≈ 112 KB — hence
32 K regressing more than 16 K despite 8× better pass-A amortization.

## Verdict + parked follow-ups

Keep wire v4 + the fused walk for real use; this branch stands as the
measured record of the experiment.  The hypothesis "amortize per-node
overhead into a tight kernel loop" inverts on Apple OOO cores: the
fused walk's overhead was already amortized (by the hardware), and
phasing converts hidden work into serial work.  Parked ideas if ever
revisited:

- Fused-walk-on-sectioned-wire needs the flat/bitmap section bases
  known at walk start: +3 B/block of section-size prefixes (hsize u8 +
  fsize u16/u24) and a pre-order bitmap section (encoder then needs a
  bitmap arena, fits inside PIVCOH_SCRATCH_SIZE).  Predicted ≈ 1.00 vs
  v4 — a wire cleanup, not a speedup — so not built.
- The guarded-mvv trick (in-loop cursor guard compiled only into
  placement-risky merges) is independently useful anywhere the 2n+128
  scratch bound must survive an execution-order change.

## Addendum: chunk-size sweep at fixed table cadence (2026-07-17)

Follow-up question: does shrinking the chunk size to stay inside L1
rescue v5?  The first sweep's G conflated table-rebuild interval and
block size, so the harness was reworked: one table per T = 65535-byte
window (both engines, gated identical), the window's data coded as
B-sized blocks all reusing that table — a --B sweep varies only the
decode/encode chunk footprint, never the trees.  Raw rows in
`bsweep/`; absolute geomeans over the 12 lits files:

SIMPLEST                          joint (coarse)
B      dec v5  dec v4  v5/v4     dec v5  dec v4  v5/v4
2048     5558    5880  0.945       8036    8185  0.982
4096     7303    7466  0.978       9891    9999  0.989
8192     8528    8936  0.954      11304   11669  0.969
16384    9250    9502  0.973      12208   12296  0.993
24576    9238    9626  0.960      12154   12421  0.979
32768    9280    9686  0.958      12280   12504  0.982
49152    8948    9435  0.948      11658   12025  0.969
65535    8787    9314  0.943      11260   11716  0.961

Answer: NO inversion anywhere.  Both wires plateau at B = 16-32 K and
fall past ~48 K (the known L1 plateau); v5's falloff is steeper
(32K->64K: −5.3 % vs v4's −3.8 % SIMPLEST, −8.3 % vs −6.3 % joint),
confirming the 2n-columns+out working-set analysis, and shrinking B
does narrow the deficit (closest cell: 16 K joint, 0.993) — but v4
wins every cell, both wires peak at the SAME B = 32768, and
v5-at-its-best / v4-at-its-best = 0.958 SIMPLEST / 0.982 joint.  At
tiny B the ratio worsens again (0.945 at 2 K): every block carries the
full tree's headers and walks the full schedule in pass A, so the
serial plan tax per byte grows as B shrinks.  (Unexplained wobble: the
8 K SIMPLEST cell dips to 0.954 between better neighbors; not chased —
doesn't affect the envelope.)

Encode side find: v5 encode WINS at small blocks (+7.6 % at 2 K, +1.2 %
at 4 K SIMPLEST; +4.4 % at 2 K joint) — the deleted per-node
staged-bitmap memcpy dominates there — and loses ~1.5 % at 24-32 K
where the two section memmoves dominate.  If v5 ever gets revisited,
the small-block encode path is the part worth salvaging.

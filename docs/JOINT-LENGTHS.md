# Joint code-length / tree-shape optimization — problem statement

Self-contained statement of the problem, intended to be readable without
knowledge of this codebase.  Status: proposed; a claimed exact algorithm
is sketched at the end and this document exists to get that claim
checked.

## 1. Setting

We entropy-code blocks of `N` symbols drawn from an alphabet of
`σ ≤ 256` distinct symbols with a binary prefix code.  Symbol `s` occurs
`n_s` times in the block (`Σ n_s = N`; in practice the `n_s` are
file-global estimates and blocks jitter around them).

The decoder is not a bit-by-bit tree walker.  It decodes **bottom-up
over the code tree**:

* A **flat subtree** — an internal node whose 2^D descendant leaves all
  sit exactly D levels below it, 2 ≤ D ≤ 8 — is decoded by a fast
  table-lookup kernel over a packed D-bit-per-symbol region.  Cost:
  roughly `κ_D` time units per symbol occurrence, with κ small
  (0.02–0.13 ns/sym depending on D) and NOT monotone in D (D=4 and D=8
  are the cheapest kernels, D=5/7 the dearest).
* Every internal node **not inside** a flat subtree performs a
  **merge**: one pass over all symbol occurrences routed through it
  (interleaving its two children's outputs by a transmitted bitmap).
  Cost: `μ` time units per occurrence per node, with μ ≈ 0.1 ns/sym —
  roughly 3–5× any κ_D.  There is also a small per-node constant `γ`
  (wire record + dispatch, ~2–5 bytes + ~10 ns).
* A depth-1 pair (two sibling leaves) is "fused": its two symbols
  materialize during the parent's merge; model it as a flat with D=1.
  A lone leaf is D=0.

So if symbol `s` gets code length `ℓ_s` and belongs to a flat/fused
chunk of depth `D_s` (D_s = 0 for a lone leaf), each of its occurrences
passes through exactly `ℓ_s − D_s` merges.

## 2. The wire constraint (this is the crux)

Only the 256 code lengths are transmitted (4-bit nibbles).  The decoder
rebuilds the entire tree — including the flat-chunk structure —
**deterministically from the lengths alone**, as follows:

1. Let `c_L = #{s : ℓ_s = L}` be the length histogram.
2. Each class L is decomposed into chunks by the **binary
   representation of c_L**: each set bit b contributes one chunk of
   2^b symbols, which is a flat subtree of depth `b` rooted at tree
   depth `L − b` (b=1 → fused pair, b=0 → lone leaf).
3. Within a class, symbols are ordered by **symbol value** and dealt
   into chunks largest-first.
4. Chunk roots are sorted by root depth `d = L − b` ascending and
   assigned canonical prefixes (this always succeeds: root Kraft masses
   are dyadic and sum to exactly 1).

Consequences: the optimizer may choose ANY valid length assignment
(class membership is fully free), but the chunk structure is a
*function* of the histogram — e.g. two same-size chunks within one
class are inexpressible — and which symbol of a class lands in which of
its chunks is fixed by value order, not choosable.

## 3. Known results for FIXED lengths (proved; sanity-checked by brute force)

With the histogram given, over all ways to regroup each class into
power-of-two chunks:

* **(K1)** Binary decomposition maximizes the number of leaves in
  D≥2 chunks: any D≥2 chunk has size ≡ 0 (mod 4), so covered ≤
  c_L − (c_L mod 4), which binary attains.
* **(K2)** It simultaneously minimizes the chunk count (popcount is the
  least number of powers of two summing to c_L), hence the merge count
  (#merges = #chunks − 1 in every arrangement).
* **(K3)** It also maximizes the frequency-weighted flat-depth sum
  Σ_chunks W_c·D_c for ANY weights (two equal-size chunks can always be
  merged — same total Kraft mass — strictly increasing the sum), i.e.
  minimizes merged-bytes = Σ_s n_s(ℓ_s − D_s), PROVIDED heavy symbols
  go to big chunks; the value-order rule of §2.3 forgoes that
  assignment freedom (deliberately: freq order would need extra wire).
* **(K4)** Total merged-bytes is independent of the arrangement of
  chunks into a tree (every chunk root at depth d has exactly d merge
  ancestors), so the depth-sort affects only merge *types* and bitmap
  skew, not volume.
* **(K5)** Packing is never a constraint: any chunk multiset arising
  from a Kraft-equality histogram packs greedily (dyadic sizes =
  divisible item sizes; the classic polynomial case of bin packing).

## 4. The joint problem

**Choose the lengths themselves** — deliberately deviating from the
Huffman histogram — to trade a little compressed size for fewer
merged bytes.

Decision variable, in its natural parameterization: for each level
`L ∈ [1, 11]` a set of bit positions `B_L ⊆ {0, …, min(8, L)}`
(so `c_L = Σ_{b∈B_L} 2^b`; the b ≤ 8 cap is the flat-kernel limit;
b = L means the whole tree is one flat).  Constraints:

* chunk-level Kraft equality: `Σ_L Σ_{b∈B_L} 2^{−(L−b)} = 1`
* symbol count: `Σ_L c_L = σ`
* (nothing else — see K5)

Symbols are then assigned to classes (free choice) and to chunks within
classes (value order, per §2.3).

**Objective** (per block, tunable λ ≥ 0 in bits per merge-pass):

```
J(lengths) = Σ_s n_s·ℓ_s                 (compressed bits)
           + λ·[ Σ_s n_s·(ℓ_s − D_s)     (merge passes)
               + Σ_s n_s·κ̂(D_s)          (flat kernel, in pass-units)
               + γ̂·(#chunks − 1) ]       (per-merge constant)
```

Sweeping λ traces the ratio-vs-decode-speed frontier; alternatively fix
a ratio budget ε (e.g. ≤ 0.5 % larger than Huffman) and pick the best
J-decode point under it.  Secondary, ignorable-at-first refinements:
flats also shrink the wire slightly (no per-level node records), and
per-block K-jitter around n_s is ± a few % — the objective is an
expectation.

## 5. Claimed algorithm (please poke holes)

Under assignment variant **B** — heavy symbols allowed into big chunks,
i.e. within-class assignment by frequency (this either needs the old
per-tier rank bytes back on the wire, or serves as a lower bound /
approximation for variant A = value order):

Sort symbols by frequency descending (prefix sums P[·]).  Define for
each candidate chunk item `(L, b)` a **per-occurrence slot cost**
`cost(L, b) = L + λ·(L − b + κ̂_b) [+ amortized γ̂]` — constant across
the chunk's 2^b slots.  For a FIXED chunk multiset, the optimal
symbol→slot assignment is sorted-to-sorted (rearrangement inequality),
i.e. process chunks in increasing cost and hand each the next-heaviest
2^b symbols.

Therefore: order the ≤ ~70 items (L ∈ 1..11, b ∈ 0..min(8, L)) by
cost and run a **0/1 knapsack DP** with state
`(symbols placed k ≤ σ, Kraft mass used m ≤ 2^11 in 2^−11 units)`;
taking item (L, b) at state k adds `cost(L,b)·(P[k+2^b] − P[k])` and
advances the state.  Feasible terminal: `k = σ, m = 2^11`.  Each (L,b)
is 0/1, which automatically enforces "B_L is a set" (§2).  ~70 × 257 ×
2049 ≈ 37 M cells → well under a millisecond in C at table-build time.

* Claim 1: this is EXACT for variant B.  (Contiguity of each chunk's
  symbols in the frequency order follows from the exchange argument;
  items are processed in global cost order so the DP's runs realize the
  sorted matching for every multiset it explores.)
* Claim 2: λ = 0 degenerates to optimal length-limited coding (the DP
  is then a package-merge equivalent), which gives the baseline for
  free and a built-in correctness check.
* For variant A (production wire, value-order tiers): replace the exact
  within-class cost by its exchangeable-model expectation
  (per-occurrence cost uses the class's mean flat depth
  D̄ = Σ b·2^b / c_L).  The DP is then exact-in-expectation but not
  worst-case; a final exact re-scoring of the DP's top candidates under
  true value-order costs (O(σ) each) closes most of the gap.  Whether
  variant A admits an exact polynomial algorithm is OPEN (the coupling
  is between chosen class membership and the fixed value order).

## 6. Questions for review

1. Is Claim 1 airtight?  In particular the step from "sorted matching
   is optimal per multiset" to "processing items in one global cost
   order inside the DP loses nothing".  Ties in cost?
2. Is Kraft EQUALITY the right constraint, or can deficit codes
   (incomplete trees) ever help this objective?  (The decoder currently
   assumes complete.)
3. Variant A: exact algorithm, hardness, or a smarter canonical
   within-tier order that both sides can derive from lengths alone and
   that correlates usefully with frequency?
4. Is per-occurrence cost the right granularity, or do per-merge
   effects (merge kernel types differing by leaf-adjacency, bitmap
   skew → FSE-ability of the transmitted bitmaps) move the optimum in
   practice?
5. Robustness: lengths are chosen from file-global n_s but decode cost
   is realized per block.  Does optimizing the expectation suffice?

---

## Post-review addendum (2026-07-10, after RESPONSE.md)

Review verdicts folded in: Claim 1 is exact for the FREE-assignment
variant (what the implementation computes); with kernel terms
kappa_b it needs monotonicity condition (M): kappa_{b+1} - kappa_b
<= 1, which real kernel timings plausibly violate at b=4->5 — the
likely mechanism behind the geometric regression, since the current
implementation sets kappa = 0.  gamma must be a fixed take-charge per
item, not amortized into slot cost.  Kraft equality stays for the
production decoder (the deficit counterexample requires nonmonotone
kappa).  The lambda sweep finds only SUPPORTED Pareto points; hard
ratio budgets need epsilon-constraint search.  Variant A has an exact
route: histogram enumeration in B_free lower-bound order + per-
histogram chain-shuffle A* with the free relaxation as admissible
heuristic.  Bonus identity (review §8): ideal merge-bitmap size is
tree-arrangement INVARIANT (multinomial cancellation), strengthening
K4.  Better within-class orders without full rank bytes: permutation
codebook (4-8 bit profile per tree) or hash order.

## Realistic-workload results (Silesia zstd literals, tables per window)

M4, 12 .lits files, per-window table build + 16K-block encode/decode,
ratio INCLUDES the 128-byte lengths header per window, lambda = 0.1:

| G    | decode      | encode      | ratio          | adopted    |
|------|-------------|-------------|----------------|------------|
| 16 K | +63.4 % gm  | +70.1 % gm  | +0.08 % (worst +0.41 %) | 91 % of windows |
| 64 K | +37.6 % gm  | +34.5 % gm  | +0.14 % (worst +0.54 %) | 79 % of windows |

No real-data regressions (min −1.8 % at 16 K, +1.1 % at 64 K); some
files get SMALLER (nci −0.67 %).

## Table-build cost history (the productization blocker)

Original mass DP: ~6 ms/window (~230-2900x the window's own encode
time) — data-bake grade only.  Two rewrites of the exact DP:

1. **Slot-ledger DP** (state (k symbols, s open slots); Kraft equality
   gives s <= sigma - k): ~10x, 0.02-0.9 ms.
2. **Diagonal / parity / capacity-band form** (this document's routes
   1+2, fused): takes preserve t = k + s, so levels decompose into
   L1-resident diagonals swept in place; live cells have k == t
   (mod 2) — compact stride-1 rows with b = 0 folded into the
   inter-level doubling; and sigma - k <= (t - k)*2^(11-L) collapses
   level 11 to one diagonal, level 10 to half.  Plus branch-free
   sweeps/doubling (blend beats early-out once everything is L1).
   Another ~6-9x on M1: sigma = 256 worst case 810 -> 127 us,
   english 77 -> 8.4 us, proba80 17 -> 2.4 us.  Exactness re-verified
   against the mass DP on 10k+ cases incl. structural (Kraft/count/
   J-recompute) validation.  Density check: 89 % of swept cells are
   live, so this is near the floor for the exact DP.

M4 per-window cost on the Silesia-lits workload (build:enc = table
build time / the window's own encode time):

| G    | build us/w      | build:enc     | was (slot-ledger) |
|------|-----------------|---------------|-------------------|
| 16 K | 16-105          | 4.1-37x       | ~10-56x           |
| 64 K | 38-179          | 2.4-12.1x     | 6-52x             |

The joint DP's incremental cost over the plain Huffman build is now
8-124 us/window on M4 (0.3-2.3x the baseline build itself).  Decode /
encode / ratio are unchanged from the table above — the DP returns
identical optima, only faster.

## End-to-end sweep, 4-128 K windows (M4, PHA, geomean of 12 files)

enc-e2e = histogram + table build (incl. joint DP) + encode kernels;
dec-e2e = per-window build_table_from_code_lens + decode kernels (the
decoder's true all-in speed at this cadence; the 128-byte header
unpack is noise).  Baseline (lam = 0) and joint (lam = 0.1) measured
back-to-back per file, so deltas are thermal-fair.  MB/s:

| G     | enc-e2e 0 -> J     | dec-e2e 0 -> J      | ratio avg |
|-------|--------------------|---------------------|-----------|
| 4 K   | 446 -> 95 (-79 %)  | 2058 -> 2867 (+39 %)| -0.42 pp  |
| 8 K   | 735 -> 171 (-77 %) | 3524 -> 4737 (+34 %)| -0.07 pp  |
| 16 K  | 1053 -> 302 (-71 %)| 5248 -> 6724 (+28 %)| +0.08 pp  |
| 32 K  | 1226 -> 483 (-61 %)| 6402 -> 8086 (+26 %)| +0.08 pp  |
| 64 K  | 1279 -> 718 (-44 %)| 6765 -> 8885 (+31 %)| +0.13 pp  |
| 128 K | 1356 -> 956 (-30 %)| 6917 -> 8820 (+28 %)| +0.09 pp  |

Notable: at G <= 8 K the joint result also COMPRESSES better on
average (the lam = 0 baseline is the production limit_code_lengths
heuristic, and flats drop node records), and kernel decode gains grow
to +70 % at 4 K.  Decoder-side table build is not free even for the
baseline: dec-k vs dec-e2e differ by ~25-60 % at small G — which the
old per-file table regime never exposed.  The encode-side e2e cost is
the joint DP (95 MB/s at 4 K, 956 at 128 K); with the original mass
DP this column read ~1-10 MB/s, i.e. the sweep only became meaningful
after the fast DP.

Same sweep as PH (--fse=0):

| G     | enc-e2e 0 -> J     | dec-e2e 0 -> J       | ratio avg |
|-------|--------------------|----------------------|-----------|
| 4 K   | 473 -> 95 (-80 %)  | 2114 -> 2932 (+39 %) | -0.41 pp  |
| 8 K   | 745 -> 171 (-77 %) | 3526 -> 4743 (+35 %) | -0.05 pp  |
| 16 K  | 1048 -> 300 (-71 %)| 5176 -> 6771 (+31 %) | +0.11 pp  |
| 32 K  | 1242 -> 491 (-61 %)| 6602 -> 8724 (+32 %) | +0.13 pp  |
| 64 K  | 1327 -> 731 (-45 %)| 7203 -> 9719 (+35 %) | +0.12 pp  |
| 128 K | 1454 -> 993 (-32 %)| 7969 -> 10678 (+34 %)| +0.10 pp  |

The joint deltas are the same story as PHA; PH's absolute decode is
what grows — dec-e2e with joint reaches 9.7-10.7 GB/s at 64-128 K
(+9 % / +21 % over PHA joint), because per-node FSE decode is pure
overhead wherever it fired.

FSE's ABSOLUTE ratio contribution at this cadence (PH minus PHA,
same lambda, positive = FSE helps), from the paired sweeps:

| G     | lam=0 avg (max)      | lam=0.1 avg (max)      |
|-------|----------------------|------------------------|
| 4 K   | +0.000 (all files 0) | +0.013 (+0.16 nci)     |
| 8 K   | +0.000 (all files 0) | +0.026 (+0.30 nci)     |
| 16 K  | +0.000 (all files 0) | +0.030 (+0.35 nci)     |
| 32 K  | +0.051 (+0.17 samba) | +0.102 (+0.37 nci)     |
| 64 K  | +0.166 (+0.52 samba) | +0.157 (+0.48 samba)   |
| 128 K | +0.390 (+1.78 reymont)| +0.402 (+1.79 reymont)|

So: at G <= 16 K baseline FSE literally never fires (no node bitmap
is big enough to beat raw + marker) and PH strictly dominates —
same ratio, faster decode.  It starts paying around 32-64 K and
reaches ~0.4 pp average / 1.8 pp max at 128 K, converging toward the
per-file regime where PHA earns its keep.  Curious side effect:
under joint lengths FSE fires MORE at small G than baseline (nci
+0.35 pp at 16 K vs 0) — flats absorb the balanced structure, and
the surviving merges pair very unequal subtrees, whose skewed
bitmaps are exactly what FSE compresses.

## The coarse-granularity heuristic (shipped)

An earlier evaluation deferred the heuristic (free-solver ceiling
1.2-3.0x on all-in encode, nothing on decode/ratio).  Studying how
the DP transforms trees on lits windows (scratch tool
study_transform) revived it by killing one idea and producing a
better one:

* 98.4 % of moved symbols move exactly +-1 level, but in RUNS —
  per-level count deltas reach the tens, popcount(c_L) drops
  2.4 -> 1.5, levels-in-use 6.5 -> 5.4, flat coverage 84 -> 92 %,
  merge passes x0.60.  The DP is a boundary nudger that rounds class
  counts to few powers of two.
* The optimal trajectory is NOT near the baseline's (max |dk| up to
  144), so warm-start banding fails.  But near-optimal solutions are
  everywhere: the J landscape is nearly degenerate.

That degeneracy is the heuristic: solve on GROUPS of g freq-sorted
symbols.  A group of g = 2^G at real level L is a depth-G flat, so
the coarse problem is the SAME solver with sigma' = sigma/g,
lmax' = 11-G, bcap' = 8-G (identical cost form up to a constant),
4^G fewer states, ghost-padded with <= g-1 zero-frequency unused
byte values when g does not divide sigma.  Measured against exact
on all lits windows (study_coarse):

| g | solve us (sigma=256) | mean/max J gap | adoption agreement |
|---|------|----------------|--------|
| 2 | 25   | 0.13 % / 1.0 % | 94-96 %|
| 4 | 8.3  | 0.21 % / 2.2 % | 80-88 %|
| 8 | 3.2  | 0.40 % / 2.0 % | 73 %   |

pivco_huffman_set_joint_granularity: 1 exact (default), 2/4/8 fixed,
0 auto (exact to sigma 64, g=2 to 128, g=4 above — solve stays
~<= 10 us at every sigma).  The per-window adoption guard applies
unchanged, so the heuristic is never-worse-than-baseline by
construction (modulo the shared model).

End-to-end effect (M4 sweep, PHA, auto granularity, geomean):

| G     | enc-e2e 0 -> J (exact was) | dec-e2e | ratio avg |
|-------|----------------------------|---------|-----------|
| 4 K   | -51.7 % (-79 %)            | +39.5 % | -0.28 pp  |
| 8 K   | -47.5 % (-77 %)            | +30.8 % | +0.04 pp  |
| 16 K  | -38.4 % (-71 %)            | +27.7 % | +0.18 pp  |
| 32 K  | -24.8 % (-61 %)            | +26.4 % | +0.17 pp  |
| 64 K  | -9.6 %  (-44 %)            | +32.5 % | +0.22 pp  |
| 128 K | +0.3 %  (-30 %)            | +29.0 % | +0.18 pp  |

Decode wins are indistinguishable from the exact DP's; encode-side
break-even arrives at ~128 K windows, and at 4 K the joint result
still nets a ratio IMPROVEMENT over baseline.  The remaining encode
gap at small G is mostly qsort + model + residual solve; the next
lever there is amortizing the joint solve across windows (re-run on
histogram drift), not a faster solver.

## Granularity -1: the greedy boundary nudger (no DP at all)

The dumbest thing that works: one shallow-to-deep walk with the slot
ledger.  At each level, clamp the baseline class count into the
feasibility window (capacity c <= (s*2^h - rest)/(2^h - 1) — note an
UPPER bound, leaves taken now eat slots the remainder needs;
completeness c >= 2s - rest; provably nonempty), then choose among
five candidates — clamped baseline, its 1- and 2-bit down-roundings,
the next power of two up, and 0 (kill the level) — scored by the
exact chunk cost at this level plus a clamped-baseline ROLLOUT of
the remainder (a one-level lookahead proxy was badly biased toward
displacement: 9-14 % payoff; the rollout fixed it).  The scorer runs
with lambda x1.5: greedy under-flattens relative to the DP, and the
guard judges with the real lambda, so the bias raises adoption for
free.  ~2 us per window, ~45x faster than the exact solve.

vs exact on all lits windows: adopts 65-70 % (exact 87-92 %),
mean J gap 0.32-0.35 % on co-adopted windows, and captures 43-50 %
of the exact DP's deployed objective improvement.  M4 e2e sweep
(gran = -1, PHA):

| G     | enc-e2e (auto-DP was) | dec-e2e (auto-DP was) | ratio    |
|-------|-----------------------|-----------------------|----------|
| 4 K   | -35.7 % (-51.7 %)     | +23.9 % (+39.5 %)     | -0.26 pp |
| 8 K   | -33.4 % (-47.5 %)     | +18.2 % (+30.8 %)     | -0.02 pp |
| 16 K  | -26.4 % (-38.4 %)     | +16.6 % (+27.7 %)     | +0.09 pp |
| 32 K  | -13.1 % (-24.8 %)     | +13.6 % (+26.4 %)     | +0.09 pp |
| 64 K  | -4.3 %  (-9.6 %)      | +15.4 % (+32.5 %)     | +0.12 pp |
| 128 K | +1.5 %  (+0.3 %)      | +16.0 % (+29.0 %)     | +0.12 pp |

Roughly half the decode payoff at roughly half the encode cost of
the auto DP — a clean third rung on the ladder (nudge 2 us / auto DP
8-10 us / exact ~100 us).  The residual joint overhead at small G is
now dominated by the shared plumbing (qsort of the alphabet, the
guard model, the deal), not the solve.

## PH vs PHA on this workload

The per-window tables above are PHA (per-node FSE on).
`bench_lits_windows --fse=0` benches PH: on skewed/low-entropy files
FSE-decode of the bitmap regions is pure overhead, so PH is
substantially faster — samba +45 %, nci +29 %, mr +18 %, xml +15 %
decode at G = 64 K — for 0-0.4 pp of ratio; english-like files
(dickens, webster) show parity.  Caveat on M4-mini batch numbers:
files late in a hot batch throttle-drift by up to ~2x (dickens
standalone: 6.2-6.4 GB/s; in-batch: 2.3-4.3 GB/s), so cross-file
comparisons within one batch are indicative only — the M1 solve-time
A/Bs and same-binary deltas are the load-bearing measurements.

## On fast tables: the ladder goes positive (branch joint-flat-lengths-plus-fast-tables)

Rebased onto rank-range-codec (minimal ~1.5 KB codec table, <1 us
decode-table build, two-queue length derivation).  Two consequences:

1. The decoder-side table build stopped masking kernel gains: at
   G = 4 K the auto-DP dec-e2e delta grew from +40 % to +67 %.
2. The builds hand their already-sorted leaf array to the joint pass
   (pivco_joint_optimize_lengths_leaves), killing its scan + qsort:
   the nudge rung's build increment fell to ~+1.8 us/window.

M4 ladder (PH, geomean of 12 lits files, deltas vs off, all four
configs measured back-to-back per file):

| G     | rung  | enc-e2e   | dec-e2e   | ratio     |
|-------|-------|-----------|-----------|-----------|
| 4 K   | nudge | -8.6 %    | +39.9 %   | -0.26 pp  |
|       | auto  | -42.5 %   | +67.0 %   | -0.27 pp  |
|       | exact | -79.6 %   | +67.6 %   | -0.41 pp  |
| 16 K  | nudge | -2.0 %    | +24.2 %   | +0.09 pp  |
|       | auto  | -24.7 %   | +44.2 %   | +0.21 pp  |
|       | exact | -70.4 %   | +45.8 %   | +0.11 pp  |
| 32 K  | nudge | +1.8 %    | +17.0 %   | +0.09 pp  |
| 64 K  | nudge | +7.4 %    | +17.8 %   | +0.10 pp  |
|       | auto  | -2.1 %    | +41.6 %   | +0.20 pp  |
| 128 K | nudge | +7.1 %    | +19.4 %   | +0.10 pp  |
|       | auto  | +3.8 %    | +37.7 %   | +0.18 pp  |
|       | exact | -28.8 %   | +39.4 %   | +0.10 pp  |

The nudge rung is free-and-always-helps from ~32 K up: ENCODES AND
DECODES both get faster (+2..+7 % / +17..+19 %) at +0.1 pp ratio —
and at 4-8 K it still nets a ratio improvement.  Auto DP crosses to
encode-positive at 128 K while carrying ~2x the decode win.  Absolute
dec-e2e with auto/exact: 11.0-11.3 GB/s at 64-128 K windows.

## What the price buys: the lambda sweep

Exact-DP sweep over all 2011 G=16K lits windows (results/
m1-20260711-lambda-sweep-g16.txt).  The RAW frontier is convex and
never saturates: passes remaining (bits premium) go 0.81 (+0.00%) at
lam=0.005, 0.59 (+0.4%) at 0.1, 0.51 (+0.75%) at 1/7, 0.40 (+1.7%)
at 0.3, and 0.09 (+8.9%) at 1.0 — the last being "the tree is nearly
one flat", i.e. converging on fixed 8-bit coding.  You can always buy
more decode speed; the exchange rate just worsens.

DEPLOYED (post-guard) is what matters, and there the optimum is set
by the guard, not the frontier: under bits <= 1.015 the best price is
lam = 0.10-0.14 (passes 0.607 -> 0.595; adoption peaks 92%) and
larger lambda gets vetoed into uselessness (adoption 15% at lam=1).
Loosening the cap to 1.03 moves the optimum only to lam ~ 0.2
(passes 0.572) with lam = 1/7 nearly matching it (0.578).  Happy
coincidence: the slot DP's exactness ceiling lam <= 1/7 covers the
deployed-optimal region under any sane guard — the mass-DP fallback
is effectively dead code for production prices.  lam = 0.1 (current)
is within ~2% of deployed-optimal; 1/7 is the better default if the
guard stays at 1.015.

Smoothness: per window the solution path is a STAIRCASE — at each
sweep step 35-50% of windows swap trees, so a typical window has
several breakpoints across the range, and (per the trajectory study)
a swap can jump to a distant corner of state space.  The aggregate
curve is smooth only because 2011 windows break in different places.
J*(lambda) itself is concave piecewise-linear per window (lower
envelope of one line per candidate tree).  Caveat at large lambda:
the model runs kappa = 0, and deep-flat regimes are exactly where
per-D kernel differences bite, so the far end of the frontier is the
least trustworthy part of it.

## Skipping the guard: what it actually protects (and what it can't)

In-model the guard is REDUNDANT for the exact DP: the baseline is a
feasible point, so bits + lambda*passes <= baseline always — the
price alone guarantees every adopted trade is favorable as the model
prices it.  The guard is therefore purely (a) a floor that skips
not-worth-shipping wins and (b) a firewall against model-vs-reality
gaps.  Measured A/B (results/m4-20260711-guard-ab.txt):

* Lits-style data: guard on == guard off (92 % adoption already; the
  extra guard-off adoptions are washes).
* bell_s10: guard off ships a MEASURED -14.3 % decode regression
  (with -3.4 % size); the guard catches it.
* geometric: -28 % decode ships WITH THE GUARD ON (alongside -10.6 %
  size).  The model prices this tree as better on both axes — the
  baseline limit_code_lengths heuristic is ~10 % off optimal bits on
  deep-natural-depth shapes, so the DP legitimately grabs the bits —
  and the guard judges with the same kappa = 0 model, so it cannot
  veto what the model itself mispriced.  This is the long-standing
  open item: per-D kappa terms (+ condition (M)) in solver AND guard.

So: the price is the right mechanism; the failure mode lives in the
cost model on deep-tree shapes (geometric / narrow-bell), where the
"win" ships as much-smaller-but-slower.  pivco_huffman_set_joint_guard
exposes the thresholds (defaults 1.015 / 0.90).

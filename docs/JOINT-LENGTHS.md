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

What any heuristic could still buy is bounded by the all-in ENCODE
throughput (G / (encode + table build) per window; decode and ratio
gain nothing).  At G = 64 K on M4, joint all-in currently runs at
0.42-0.90x of baseline all-in; replacing the DP with a FREE solver of
equal quality would reach 1.06-1.19x of baseline (joint lengths speed
encoding up too), i.e. a ceiling of 1.2-3.0x over today — at which
point the plain Huffman table build (8-55 us/window) is the equal
bottleneck.  Against that ceiling a greedy-carry heuristic brings
approximation risk exactly where the model is known to be thin
(kappa/condition (M), the geometric case) and a second solver to
maintain.  Deferred; better first moves if encode-side cost matters:
shave the generic build_table (shared by baseline), or amortize the
joint solve across windows by re-running it only on histogram drift.
The sweeps themselves are compute-bound at ~0.4 ns/cell NEON with
89 % state density, so only an inexact search could go much below
the current cost.

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

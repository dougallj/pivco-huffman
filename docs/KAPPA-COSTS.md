# Kernel-cost-aware joint optimization — problem statement

Third installment (after docs/JOINT-LENGTHS.md and
docs/FLAT-ROUTING-ENTROPY.md; reviews RESPONSE.md, RESPONSE2.md).
This one is half solved-and-verified, half open — we want the solved
half checked and the open half attacked.

## 1. The failure this is about

The joint optimizer prices decode time as merge passes only: a symbol
occurrence in a depth-D flat at root depth r costs `r` passes and the
flat kernel costs NOTHING (kappa = 0).  Real kernels are not free and
not monotone in D (D = 4/8 cheapest, D = 5/7 dearest; all below the
merge cost mu).  Measured consequence: on the synthetic geometric
distribution the optimizer ships a tree that is 10.6 % SMALLER and
28 % SLOWER to decode — and the adoption guard cannot veto it, because
the guard prices with the same kappa = 0 model that made the mistake.
(Real corpora show only miniature versions: worst measured -2.8 % on
raw Calgary obj1.  But the contract "the knob makes decode faster"
deserves a model that cannot be fooled by construction.)

## 2. Cost model

Chunk type (L, b): 2^b symbols at code length L inside a depth-b flat
rooted at depth r = L - b.  Per-occurrence cost

```
cost(L, b)  =  L  +  lambda * ( (L - b) + kappa_b )
            =  L * (1 + lambda)  +  g(b),        g(b) = lambda * (kappa_b - b)
```

with kappa_b >= 0 the depth-b kernel time in merge-pass units
(kappa_b = measured kernel ns/sym divided by merge ns/sym; b = 0 is
the lone-leaf scatter, b = 1 the fused pair, both cheap).  Everything
else — Kraft equality over chunk roots, symbol count, the wire's
binary-decomposition constraint — is as in JOINT-LENGTHS.md.

## 3. What is now SOLVED (implemented and machine-verified)

**Claim (replaces condition (M)).**  The slot-ledger DP remains exact
for arbitrary kappa tables provided:

* (a) *spread bound*:  max_b g(b) - min_b g(b)  <=  1 + lambda,
* (b) *b0-dearest*:    g(0) >= g(b) for all b   (i.e. kappa_b <= kappa_0 + b).

**Why.**  The DP's exactness argument needs the realized
symbol-to-chunk assignment order to be a global per-occurrence-cost
order (sorted matching).  The DP processes levels ascending; RESPONSE.md's
condition (M) (kappa monotone-ish in b) arose from assuming the
within-level sweep order is fixed at b-descending.  But the
within-level order is a FREE CHOICE: since cost(L, b) = L(1+lambda) +
g(b), the within-level cost order is the SAME permutation of b at
every level — sweep (and deal) each level's chunk types in ascending
g and the within-level requirement disappears entirely.  What remains
is the cross-level requirement

```
max_b cost(L, b) <= min_b cost(L+1, b)
   <=>   L(1+lambda) + max g  <=  (L+1)(1+lambda) + min g
   <=>   spread(g) <= 1 + lambda,
```

which at kappa = 0 (spread(g) = 8*lambda) recovers the classic
lambda <= 1/7.  Condition (b) exists only because the DP's parity
compaction folds the b = 0 take into the inter-level doubling, which
requires it to be the level's last (dearest) type; every plausible
real kernel table satisfies kappa_b <= kappa_0 + b trivially.

For real tables (kappa in [0, ~1.3], kappa_0 - kappa_8 small) the
bound is lambda <~ 1/(7 + kappa_0 - kappa_8) — production lambda = 0.1
remains comfortably inside.  Outside the bound the mass DP takes over:
its global item qsort makes it exact for ANY lambda and kappa at ~60x
the cost.  The final dealing was likewise generalized to sort the
chosen chunks by global cost (this also fixed a latent quirk: the old
lambda > 1/7 mass path dealt in level order, not the sorted matching
its optimum assumed).

**Verification.**  Slot DP == mass DP on 16k instances (two seeds),
half with random nonmonotone kappa tables in [0, 1.5), wherever the
spread bound admits the slot DP; structural validation (Kraft,
counts, J recomputed from the cost-ordered dealing); full roundtrip
suite including a nonmonotone-kappa pass; GuardMalloc clean.  The
guard now prices kappa identically on both sides of its comparison.

Status: mechanism is in (pivco_huffman_set_joint_kappa); the default
table is all-zero until per-platform measurement lands (§5).

## 4. What is OPEN (the questions)

1. **Fast exact beyond the spread bound.**  For lambda above the
   bound the only exact solver is the mass-DP grid (sigma x 2^11
   states).  At lambda = 0 the problem degenerates to length-limited
   Huffman, solved by package-merge in O(sigma * L).  Is there a
   package-merge / coin-collector analogue for the full objective —
   chunk types as "coins" of dyadic Kraft mass with per-occurrence
   cost cost(L, b) — giving O(sigma * L * |B|) at any lambda?  (The
   segmentation DP of RESPONSE2 §4 is exact for the H-model objective
   with arbitrary kappa and NO order condition — contiguity replaces
   sorted matching — but it lives on the same big grid.  Can its mass
   axis be compressed the way the slot ledger compressed ours?)

2. **Non-separable merge costs.**  mu is not really constant: a merge
   whose child is a leaf/pair uses a cheaper kernel (cst-merges) than
   one merging two internal streams, and bitmap skew changes both the
   merge cost and (under PHA) the coder's take.  These costs attach
   to tree ADJACENCIES, not to chunks — K4 guarantees merge VOLUME is
   arrangement-invariant, but kinds are not.  With the wire's
   canonical arrangement fixed, kind counts are a deterministic
   function of the chunk multiset: is the resulting objective still
   per-chunk separable enough for an exact DP, or does it become
   genuinely combinatorial?  An exchangeable-model approximation with
   a bounded error term would already be useful — how large can its
   error be?

3. **Robust adoption under model uncertainty.**  kappa and mu are
   measured with error and vary by microarchitecture; a table fitted
   on M4 will be wrong on Zen 5.  The guard currently does a point
   comparison.  Formulate the adoption test as robust optimization:
   accept only if the modeled win holds for every kappa in an
   uncertainty box [kappa_lo, kappa_hi].  Because the model is linear
   in kappa, the worst case is attained at a box corner — is
   per-corner evaluation (2 corners suffice? which?) exact for the
   guard's one-sided test, and how much adoption does robustness
   cost on real corpora?

4. **Identifiability / experiment design.**  kappa and mu enter
   decode time only through per-tree aggregates (sum over chunks of
   W_c * kappa_{D_c}, passes * mu, kind counts).  Given the freedom
   to construct arbitrary tables and inputs (we control the tree!),
   design the minimal experiment set that identifies (mu_kinds,
   kappa_0..8) per platform with tight confidence — e.g. same input
   decoded under trees differing in exactly one contraction, the
   local (W, q) surface of RESPONSE2 Stage 4/5.

## 5. Practical plan

(i) Measure kappa_b and mu per platform from bench_micro's flat/merge
throughput probes (one-contraction A/B pairs per §4.4 for
confirmation); (ii) set the table via pivco_huffman_set_joint_kappa
and re-run the geometric family — the misfit witness should flip from
"adopted, -28 %" to "rejected" or "adopted with a genuinely faster
tree"; (iii) re-run the lits ladder to check the kappa-aware optimum
doesn't give back the measured wins (expected: mild D-mix shifts
toward 4/8-depth flats, slightly fewer deep flats); (iv) fold the
same table into the guard's robust form once §4.3 has an answer.

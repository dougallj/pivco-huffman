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

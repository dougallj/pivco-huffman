## Overall verdict

The central idea is strong, and the two-dimensional DP really is exact for a cleanly defined relaxation. I would revise the claims as follows:

| Statement                                                | Verdict                                                  |
| -------------------------------------------------------- | -------------------------------------------------------- |
| Claim 1 with completely free symbol-to-chunk assignment  | **Exact**                                                |
| Claim 1 with “frequency order, then largest chunk first” | **Exact only under a kernel-monotonicity condition**     |
| Ties in slot cost                                        | Harmless, with a suitable deterministic tie-break        |
| Amortized (\hat\gamma) inside the slot cost              | **Not exact**; (\gamma) must be a fixed take-charge      |
| (\lambda=0)                                              | Yes: it searches all complete length-limited histograms  |
| Kraft equality for the current decoder/model             | Yes                                                      |
| Kraft equality without the complete-tree requirement     | Not automatically without an additional kernel condition |
| Variant-A mean-depth substitution in the same item DP    | **Not exact as written**                                 |
| File-global means for expected block cost                | Exactly right for the stated linear model                |

The most important hole is a subtle mismatch between “heavy symbols go to big chunks” and “symbols are sorted by actual scalar slot cost.” Because (\hat\kappa_b) is nonmonotone, those need not be the same order.

---

## 1. Claim 1: the exact theorem and its caveat

Let an item (i=(L_i,b_i)) have

[
w_i=2^{b_i},\qquad
r_i=2^{11-L_i+b_i},
]

where (w_i) is its number of symbol slots and (r_i) its Kraft mass in (2^{-11}) units. Define its per-occurrence scalar cost

[
a_i=L_i+\lambda\bigl(L_i-b_i+\hat\kappa_{b_i}\bigr).
]

Let the frequencies be sorted as

[
n_1\ge n_2\ge\cdots\ge n_\sigma,\qquad
P[k]=\sum_{j=1}^{k}n_j.
]

### Free-slot variant

Suppose that, after choosing the chunks, any symbol may be assigned to any chunk slot. Expand every selected item into (w_i) identical slots of cost (a_i). For a fixed selected set, the rearrangement inequality says that the minimum is obtained by pairing decreasing frequencies with increasing slot costs.

Sort all 71 candidate items by nondecreasing (a_i). Then the exact recurrence is

[
F_i(k,m)=\min\left{
\begin{aligned}
&F_{i-1}(k,m),\
&F_{i-1}(k-w_i,m-r_i)
+a_i\bigl(P[k]-P[k-w_i]\bigr)+g
\end{aligned}
\right.,
]

where

[
g=\lambda\hat\gamma.
]

The terminal value is

[
F_{71}(\sigma,2^{11})-g.
]

The subtraction accounts for the fact that (q) selected chunks create (q-1) merge nodes.

This is an airtight DP because, after processing the first (i) cost-ordered items, a state ((k,m)) always means:

* the first (k) frequencies have been consumed;
* Kraft mass (m) has been used;
* every future contribution depends only on (k,m), not on which earlier subset produced the state.

That gives the required optimal substructure.

### Ties

Equal (a_i) values cause no objective ambiguity: exchanging two equal-cost blocks changes nothing. Use a deterministic tie-break for reproducibility.

If the production rule requires larger chunks first within a class, then ties within the same (L) should be broken by decreasing (b).

### Do not amortize (\gamma) into (a_i)

The per-node constant is a fixed charge per selected item, not an occurrence-weighted cost. Expressions such as

[
a_i+\frac{\gamma}{w_i}
]

are wrong because the DP then multiplies the alleged amortization by the total frequency assigned to the item. Add (g) once on the take transition and subtract it once at the terminal.

The same treatment applies to any fixed flat-kernel startup cost: add an item-specific fixed charge, but do not put it in the slot-ordering coefficient.

---

## 2. The important Claim-1 counterexample: “big first” need not mean “cheap first”

Within one length class (L), the decode part of a slot cost is

[
q_{L,b}=L-b+\hat\kappa_b.
]

Larger chunks are ordered before smaller chunks by the proposed frequency-tier rule. That agrees with increasing actual cost only if

[
q_{L,b+1}\le q_{L,b},
]

or equivalently

[
\boxed{\hat\kappa_{b+1}-\hat\kappa_b\le 1.}\tag{M}
]

Condition (M) is independent of (L) and (\lambda>0). It says that moving to a larger flat may cost more in its flat kernel, but not by more than the merge pass it eliminates.

Your rough measurements make violation plausible: if (D=4) costs about (0.02) ns/symbol and (D=5) about (0.13), while one merge is (0.1), the normalized jump is about (1.1).

For example, consider the complete valid chunk set

[
(2,0),\quad(6,5),\quad(6,4).
]

Its Kraft mass is

[
\frac14+\frac12+\frac14=1.
]

With (\hat\kappa_4=0.2) and (\hat\kappa_5=1.3),

[
q_{6,4}=2.2,\qquad q_{6,5}=2.3.
]

So the 16-slot (b=4) chunk is cheaper than the 32-slot (b=5) chunk. The global cost-sorted assignment gives the heavier symbols to (b=4), while “sort by frequency and deal largest-first” gives them to (b=5).

Therefore:

> **Claim 1 is exact for a free-chunk-assignment variant, but not necessarily for the stated frequency-largest-first variant.**

There are three clean fixes:

1. **Define variant (B_{\rm free}):** rank bytes identify arbitrary chunk membership, so the encoder may assign by actual (a_i).
2. **Order tiers by actual decode cost**, (L-b+\hat\kappa_b), rather than by chunk size.
3. **Assume and verify condition (M)** for the calibrated kernel table.

If largest-first must remain and (M) fails, the chunks of each length class form an ordering chain. The fixed-multiset assignment is then no longer plain sorted matching; it becomes a minimum-cost interleaving of up to 11 chains.

K3 remains correct for merge-volume alone, but it does not prove optimality once the nonmonotone flat-kernel term is included.

---

## 3. The (\lambda=0) claim

This part is correct, apart from one edge case.

At (\lambda=0),

[
a_{L,b}=L.
]

For a fixed histogram, assigning higher frequencies to shorter lengths is optimal. The item selection also spans every valid histogram:

* Kraft feasibility implies (c_L\le 2^L), so the binary expansion of (c_L) never needs a bit (b>L).
* Since (\sigma\le256), it never needs (b>8).
* Every integer (c_L) has a unique binary expansion, so every feasible histogram corresponds to exactly one subset (B_L).

Thus the DP searches all complete binary length histograms with (L\le11), and its (\lambda=0) optimum equals the optimal length-limited Huffman value. Package-Merge is the classical (O(\sigma L_{\max})) algorithm for that problem, so I would say “same optimum as Package-Merge,” rather than that this particular DP degenerates algorithmically into Package-Merge. ([ics.uci.edu][1])

The edge case is (\sigma=1): the natural complete code has one codeword of length zero. Since the candidate domain starts at (L=1), single-symbol alphabets need a special case or an (L=0,b=0) item.

Also, the complexity statement is optimistic. There are exactly

[
71\cdot257\cdot2049=37{,}388{,}103
]

dense transitions. That is perfectly reasonable at table-build time, but a single-core sub-millisecond claim would require over 37 billion state transitions per second. Benchmark it rather than promising that latency. Sparse reachable-state iteration and feasibility bounds should cut it substantially.

---

## 4. Kraft equality

### For the current decoder and current objective: equality is required

Several identities in the problem rely on completeness:

[
#\text{merge nodes}=#\text{chunks}-1,
]

and

[
\text{merge passes for chunk }c=\text{root depth }d_c=L_c-b_c.
]

With Kraft deficit, the minimal prefix trie contains unary paths. A unary path contributes code bits but is not naturally a merge between two nonempty streams. Consequently:

* (\ell_s-D_s) need not equal the number of actual merge passes;
* K4 no longer follows from root depth;
* the decoder must specify whether unary nodes are elided, materialized as empty-child operations, or completed with dummy leaves.

So merely replacing

[
\sum 2^{-d}=1
]

by

[
\sum 2^{-d}\le1
]

while retaining the same objective is not a coherent relaxation.

### When equality is without loss of optimality

Condition (M),

[
\hat\kappa_{b+1}-\hat\kappa_b\le1,
]

is also a useful sufficient condition here.

Starting from an incomplete chunk-root tree:

1. Suppress unary root-tree edges. This shortens every affected codeword and never increases its flat-kernel cost.
2. If suppression creates two identical ((L,b)) chunks, perform the binary carry
   [
   2\times(L,b)\longrightarrow(L,b+1).
   ]
3. The carry changes the per-occurrence decode term by
   [
   -1+\hat\kappa_{b+1}-\hat\kappa_b,
   ]
   which is nonpositive under (M), and it reduces the chunk count.

With (\sigma\le256), a carry from (b=8) cannot arise because it would require at least 512 symbol slots.

So under (M), a complete representation weakly dominates an incomplete one. Without (M), that theorem fails.

### A reproducible deficit counterexample

Under the naïve extension that retains the stated (\ell-D) cost, take

[
\hat\kappa=(0,0,0.3,0.3,0.2,1.3,0.4,1.3,0.2),
\quad \lambda=10,\quad \hat\gamma=0,
]

with 16 symbol frequencies equal to 1000 and 16 equal to 1.

Running the same DP with an equality terminal versus a (\le) terminal gives:

| Terminal | Selected chunks           | Kraft mass |     (J) |
| -------- | ------------------------- | ---------: | ------: |
| Equality | ((5,4),(4,2),(5,2),(6,3)) |        (1) | 272,572 |
| Deficit  | ((5,4),(6,4))             |      (3/4) | 272,448 |

The deficit solution avoids forcing the heavy 16-symbol group through the expensive (D=5) kernel. If unary paths are elided rather than charged as merges, the deficit solution becomes cheaper still.

My recommendation is therefore:

> Keep Kraft equality for the production decoder. If incomplete codes are explored, treat them as a different decoder/tree model, not a one-character change to the constraint.

An alternative format design is to permit implicit zero-frequency dummy symbols and retain equality. Dummies then have to participate in the histogram/chunk optimization because they can trigger binary carries and change (\kappa) and (\gamma).

---

## 5. Variant A has an exact fixed-histogram formulation

For a fixed histogram, each nonempty length class (L) defines an ordered chain of slots:

[
A_{L,1},A_{L,2},\ldots,A_{L,c_L},
]

where chunks appear in decreasing (b), and each chunk contributes (2^b) repeated slot costs.

Let symbols be in increasing symbol-value order, with frequencies

[
n_1,n_2,\ldots,n_\sigma.
]

A valid assignment is precisely a shuffle, or linear extension, of the length-class chains. If (r_L) slots of class (L) have already been consumed and

[
t=\sum_L r_L,
]

then the exact recurrence is

[
G(r+e_L)
========

\min\left{
G(r)+n_{t+1}A_{L,r_L+1}
\right},
]

over classes with (r_L<c_L).

This has

[
\prod_{L:c_L>0}(c_L+1)
]

states. It is practical for two or three substantial classes, but can be enormous for 11 balanced classes. In parameterized-complexity language, it is an XP algorithm in the number of nonempty classes. Related sequence-interleaving problems become hard when the number of chains is variable, although that does not by itself prove hardness for your highly structured costs. ([arXiv][2])

### A practical, globally exact branch-and-bound algorithm

Variant (B_{\rm free}) gives a lower bound for variant A for every histogram. That turns the proposed heuristic into a certifying exact search:

1. Treat the Claim-1 DP as a layered shortest-path DAG.
2. Enumerate feasible histograms in nondecreasing (B_{\rm free}) lower-bound order using a (k)-shortest-path or A* traversal.
3. For each emitted histogram, solve its exact chain-shuffle assignment using the grid DP above, or A*.
4. Stop as soon as the next unprocessed lower bound is at least the best exact A score found.

For the fixed-histogram A* search, an excellent admissible heuristic is:

* take all remaining symbol frequencies;
* ignore their value order;
* take all remaining slot costs;
* sort-to-sort them freely.

That is exactly the relaxation you already know how to evaluate.

This algorithm is exponential in the worst case, but it is exact and has a clean stopping certificate. “Rescore the top (K)” is the same idea without the stopping certificate.

---

## 6. The exchangeable mean-depth approximation needs a different selection object

For a class pattern (B_L), define

[
c_L=\sum_{b\in B_L}2^b,
]

[
\bar b_L
========

\frac{\sum_{b\in B_L}b,2^b}{c_L},
\qquad
\overline{\kappa}_L
===================

\frac{\sum_{b\in B_L}\hat\kappa_b,2^b}{c_L}.
]

Under a genuinely exchangeable random ordering of the frequencies within the class, the expected per-occurrence coefficient is

[
\bar a(L,B_L)
=============

L+\lambda\left(L-\bar b_L+\overline{\kappa}_L\right).
]

Two corrections follow.

First, average (\kappa_b) itself; do not evaluate (\kappa) at the mean depth, because (\kappa) is not linear or monotone.

Second, this coefficient belongs to the **whole class pattern**, not independently to its constituent chunk items. Selecting another bit in (B_L) changes (c_L), (\bar b_L), and the expected cost of every slot in the class.

Therefore the original 71-item DP is not exact-in-expectation after simply substituting a mean depth. An exact exchangeable-model solver would select class patterns ((L,c_L)), where the binary expansion of (c_L) determines (B_L), with an at-most-one-pattern-per-(L) constraint.

Also, exchangeability is a modeling assumption, not a consequence of value order. It is exact only if the value ranking of the selected class members is independent of their frequencies.

---

## 7. Better deterministic within-class orders

No rule derived solely from the length vector can recover arbitrary within-class frequency order in the worst case. A class containing (c) symbols has (c!) possible frequency rankings compatible with exactly the same lengths. An adversary can always reverse any fixed canonical rule.

There are still useful zero- or low-wire choices:

**Static learned permutation.** Replace numeric symbol order with a fixed permutation of the 256 byte values trained on representative data. Both sides know it, and it costs no wire bits.

**Previous-block or cumulative-frequency order.** Both sides maintain counts from already decoded data. This provides file-specific correlation with no per-block rank bytes, but weakens random access and parallel decoding.

**A small permutation codebook.** Store 16 or 256 fixed permutations and transmit a 4- or 8-bit profile index per tree. The encoder chooses the profile with the best exact value-order score. This is likely a much better wire/performance trade than restoring a full rank list.

**Hash order.** A deterministic or lightly seeded hash of symbol value removes systematic numeric-order bias. It does not correlate with frequency, but makes the exchangeability approximation much more defensible.

---

## 8. Per-merge effects and bitmap skew

The first-order model is appropriate as long as merge cost is approximately linear and type-independent. A more faithful exact rescore would be

[
T=
\sum_{\text{chunks }c}\hat\kappa_{b_c}W_c
+
\sum_{\text{merge nodes }v}
\left[
\hat\mu_{\tau(v)}(p_v,W_v),W_v
+
\hat\gamma_{\tau(v)}
\right],
]

where

* (W_c) is a chunk’s total frequency;
* (W_v) is the traffic through merge node (v);
* (p_v=W_{v,0}/W_v) is bitmap skew;
* (\tau(v)) identifies merge/child types.

This is easily evaluated in (O(\sigma)) or (O(#\text{nodes})) once a candidate is reconstructed.

### Ideal bitmap compression is tree-arrangement invariant

There is a useful exact identity here. Treat chunks as super-symbols with counts (W_c). If every merge bitmap is encoded enumeratively, then

[
\sum_v
\log_2 {W_v\choose W_{v,0}}
===========================

\log_2\frac{N!}{\prod_c W_c!}.
]

All intermediate factorials cancel. Thus ideal merge-bitmap size depends on the chunk partition and chunk weights, but not on how the chunk roots are paired into a binary tree.

The entropy approximation says the same thing:

[
\sum_v W_v h_2(p_v)
===================

N,H!\left(\frac{W_1}{N},\ldots,\frac{W_q}{N}\right).
]

Consequently, bitmap skew alone does not create a first-order reason to prefer one root-tree arrangement over another. What can matter is:

* FSE normalization and table overhead per bitmap;
* small-bitmap inefficiency;
* temporal/context structure not captured by counts;
* different merge kernels for leaf/flat/merge children;
* cache and allocation thresholds;
* raw (D)-bit flat regions, whose internal decisions are not entropy-coded.

Once these refinements matter, the root-depth sort needs a fully specified tie-break among equal-depth roots, because sibling pairings can affect finite overhead and kernel types even though merged-byte volume is unchanged.

A robust engineering approach is to use the simple DP as a generator/lower bound, retain multiple near-optimal trees, and exact-rescore them with whole-tree measurements or a fitted node model.

---

## 9. Robustness to block jitter

For the objective as currently written, optimizing the expectation is not merely an approximation.

For a fixed tree (x), write

[
J_x(K)=\sum_s K_s,c_s(x)+G(x),
]

where (K_s) is the realized block count and (G(x)) contains the fixed node charges. Then

[
\mathbb E[J_x(K)]
=================

# \sum_s \mathbb E[K_s],c_s(x)+G(x)

J_x(\mathbb E[K]).
]

No independence assumption is needed. Therefore, if the file-global (n_s) are the true expected block counts, minimizing the objective using those (n_s) exactly minimizes expected modeled block cost. The variance and the “few percent” jitter do not change that conclusion.

Expectation stops being sufficient when the goal or implementation is nonlinear, for example:

* a per-block ratio limit rather than an average limit;
* p95 or worst-case decode latency;
* FSE size as a nonlinear function of realized skew;
* skipping startup/dispatch for empty nodes;
* cache thresholds;
* uncertain estimates of the mean, rather than random counts with a known mean.

For those cases, exact-rescore candidate trees over the actual block histograms and optimize the empirical mean, quantile, or chance constraint. Because the candidate set near the frontier should be small, scenario evaluation is inexpensive.

---

## 10. One more frontier caveat

Sweeping (\lambda) does not necessarily recover every nondominated discrete tree. Weighted scalarization

[
B+\lambda T
]

returns the **supported** Pareto points on the lower convex hull. A nondominated point can lie above a chord between two other points and never be optimal for any (\lambda), yet still be the best tree under a particular hard bit budget.

So the precise statement is:

> Sweeping (\lambda) traces the supported ratio/speed frontier.

For an exact (0.5%) ratio budget, use an (\varepsilon)-constraint search, Pareto labels, or the certifying candidate-enumeration procedure rather than assuming a (\lambda) sweep is complete.

---

## Recommended formulation

I would write the production algorithm around these definitions:

1. **Variant (B_{\rm free}):** arbitrary symbol-to-chunk assignment, with Claim 1 exactly as the two-resource DP above.
2. **Variant (B_{\rm big}):** frequency order and largest-first chunks; Claim 1 is exact only if condition (M) holds. Otherwise use chain interleaving.
3. **Variant A:** fixed canonical symbol order; solve approximately with (B_{\rm free}), or exactly by lower-bound-ordered histogram enumeration plus chain A*.
4. Add (\gamma) and any flat startup as fixed item charges.
5. Keep Kraft equality for the current decoder.
6. Reconstruct and exact-rescore whole trees, including node types and bitmap effects.
7. Validate the few surviving trees across real block histograms.
8. For a hard compression budget, do not rely solely on the (\lambda) sweep.

With those changes, the (B_{\rm free}) result is a clean exact theorem, the A solver has a practical route to an optimality certificate, and the nonmonotone kernel measurements are handled rather than silently assumed away.

[1]: https://ics.uci.edu/~dan/pubs/LenLimHuff.pdf "https://ics.uci.edu/~dan/pubs/LenLimHuff.pdf"
[2]: https://arxiv.org/abs/1605.00901 "https://arxiv.org/abs/1605.00901"

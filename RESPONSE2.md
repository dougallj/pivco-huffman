# Overall verdict

The sequel’s central conclusion is correct:

[
\boxed{\text{ideal routing rate}
=\text{source entropy}
+\text{flat penalties}}
]

so ordinary code-length distortion ceases to be a rate penalty once the surviving routing decisions are entropy-coded well.

However, four distinctions are essential:

1. The multinomial identity is exact, but (N H(p)) and the proposed flat gaps are the **empirical-entropy approximation**, not the exact enumerative bit count.
2. The free partition in §4 is a relaxation of the original lengths-only wire. The original “one chunk per ((L,D))” and value-order rules still matter.
3. The nonlinear gap objective does **not** destroy contiguity. In fact, concavity gives a clean exchange proof, leading to a new exact segmentation/Kraft DP for the relaxed problem.
4. The proposed skew theorem is false. There is a stronger invariant showing exactly what is and is not shape-dependent.

My answers to the six questions are:

| Question             | Verdict                                                                                                                   |            |       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- |
| 1. Contiguity        | **Yes**, for free frequency assignment; proof below                                                                       |            |       |
| 2. Exact algorithm   | **Yes** for the free-partition reformulation; (O(\sigma 2^R                                                               | \mathcal T | )) DP |
| 3. Frontier          | Zero-gap face is exactly uniform-within-flat grouping; its knee is highly source-dependent                                |            |       |
| 4. Coder dial        | No universal (\eta^*); locally, a better coder makes nonuniform flattening more expensive in bits                         |            |       |
| 5. Skew theorem      | **False**; aggregate skew redundancy is fixed by partition and weighted path length                                       |            |       |
| 6. Enumerative coder | Theoretically excellent; naïve decoding is probably too serial, but an 8/16-bit block-hypergeometric version is promising |            |       |

---

# 1. First repair: there are two different “ideal” models

The statement currently blends an exact type-class count with a first-order entropy codelength. Both are useful, but they give slightly different flat penalties.

Let the flats be (c), with

[
m_c=2^{D_c},\qquad W_c=\sum_{s\in c}n_s.
]

## 1.1 Empirical-entropy ideal

For the skeleton whose leaves are the flat chunks,

[
\sum_{v\in\text{skeleton}}W_vH_2(q_v)
=====================================

N H(P_1,\ldots,P_C),
\qquad P_c=W_c/N.
]

The raw flat streams cost (\sum_c D_cW_c). Consequently,

[
B_H
===

N H(P_1,\ldots,P_C)
+
\sum_cD_cW_c.
]

Using the entropy chain rule,

[
H(p)
====

H(P_1,\ldots,P_C)
+
\sum_cP_cH(p\mid c),
]

so

[
B_H
===

N H(p)+G_H,
]

where

[
G_H
===

\sum_cW_c\bigl(D_c-H(p\mid c)\bigr).
]

This can be written especially cleanly as a KL divergence. Define the piecewise-uniform approximation

[
\bar p_s=\frac{P_c}{m_c},
\qquad s\in c.
]

Then

[
\boxed{
G_H=N,D_{\rm KL}(p\Vert \bar p).
}
]

That is, flat placement is a constrained probability-quantization problem: approximate the source distribution by a distribution that is constant on dyadic-sized groups.

This is the right idealization for a near-arithmetic/FSE coder whose per-node rate approaches (W_vH(q_v)).

## 1.2 Exact type-conditioned enumerative ideal

The exact skeleton arrangement count is

[
\prod_{v\in\text{skeleton}}
\binom{W_v}{W_{v,L}}
====================

\frac{N!}{\prod_cW_c!}.
]

Thus, under joint ideal enumeration, the wire is

[
B_E
===

\log_2\frac{N!}{\prod_cW_c!}
+
\sum_cD_cW_c.
]

The corresponding no-flat baseline is

[
B_{E,0}
=======

\log_2\frac{N!}{\prod_sn_s!}.
]

Hence the exact flat penalty is

[
\boxed{
G_E
===

\sum_c
\left[
D_cW_c-
\log_2\frac{W_c!}{\prod_{s\in c}n_s!}
\right].
}
]

Equivalently,

[
G_E
===

\sum_c\left[D_cW_c-\log_2(W_c!)\right]
+
\sum_s\log_2(n_s!).
]

This is the exact-count analogue of your §3 algebra.

Two consequences matter:

* A uniform flat has (G_H=0), but generally has (G_E>0). For an exactly balanced binary split,
  [
  W-\log_2\binom{W}{W/2}
  \approx
  \frac12\log_2\frac{\pi W}{2}.
  ]
* A standalone enumerative rank needs an integer number of bits. A simple fixed-length rank uses
  [
  \left\lceil\log_2\binom WK\right\rceil,
  ]
  not exactly (\log_2\binom WK). Joint arithmetic coding or one mixed-radix rank can amortize most of that rounding.

Cover’s enumerative source coding is the classical rank/unrank construction underlying this model. ([IEEE Xplore][1])

I would name these two models explicitly:

* **H-model:** (W H(q)), uniform flats asymptotically free.
* **E-model:** (\log\binom WK), conditioned on exact counts, with sublinear type-class corrections.

Your current FSE measurements are much closer conceptually to the H-model plus setup, quantization, and termination overhead.

---

# 2. The code-length claim is true, with qualifications

Ignoring flats, node records, integer termination, and count-model costs:

[
\boxed{\text{routing rate is independent of tree shape}.}
]

But this does not quite mean that the production code lengths are rate-irrelevant, because the lengths determine:

* which flat chunks are forced;
* which symbols share each flat;
* the number of chunks and merge-node records;
* the canonical skeleton arrangement;
* mandatory fused pairs, if fusion cannot be disabled.

The correct production decomposition is therefore

[
\boxed{
B_{\rm production}
==================

N H(p)
+
G_H(\text{flats})
+
R_{\rm coder}(\text{skeleton})
+
B_{\rm records}.
}
]

Here

[
R_{\rm coder}
=============

\sum_{v\in\text{skeleton}}
\left[
B_v-W_vH_2(q_v)
\right]
]

is the coder’s residual above ideal entropy.

At PH,

[
B_v=W_v,
]

so

[
R_{\rm coder}
=============

\sum_vW_v\bigl(1-H_2(q_v)\bigr).
]

At an ideal arithmetic coder,

[
R_{\rm coder}\simeq0.
]

This decomposition is, in my view, the central result of the sequel.

---

# 3. Q1: contiguity survives — and the proof is short

For a group (c) of size (m_c=2^{D_c}), its entropy gap is

[
G_c
===

\sum_{s\in c}
n_s\log_2\frac{n_s}{W_c/m_c}.
]

This is the scalar generalized-KL/Bregman clustering cost from the counts (n_s) to their group mean (W_c/m_c).

Dropping the partition-independent term (\sum_sn_s\log n_s), minimizing total gap is equivalent to maximizing

[
\Phi
====

\sum_c
W_c\log_2\frac{W_c}{m_c}.
]

Define

[
\phi_c(W)=W\log_2(W/m_c).
]

Then

[
\phi_c'(W)
==========

\log_2(W/m_c)+\frac1{\ln 2},
\qquad
\phi_c''(W)>0.
]

Suppose two groups (A,B) currently satisfy

[
\frac{W_A}{m_A}\ge\frac{W_B}{m_B},
]

but (A) contains a smaller count (x) and (B) contains a larger count (y>x). Let (\delta=y-x). Swapping those symbols changes the objective by

[
\Delta
======

\int_0^\delta
\left[
\phi_A'(W_A+t)
--------------

\phi_B'(W_B-t)
\right]dt.
]

At (t=0), the integrand is nonnegative by the fullness ordering. As (t) increases, the first derivative rises and the second falls. Therefore

[
\Delta>0
]

unless the exchanged counts are equal.

So an optimum cannot have such an inversion.

## Contiguity theorem

For fixed group sizes, there exists a gap-minimizing assignment in which:

1. groups are ordered by descending fullness (W_c/m_c);
2. every count in an earlier group is at least every count in a later group;
3. consequently, every group is a contiguous block of frequency-sorted symbols.

Thus your contiguity conjecture is correct.

## The theorem survives the decode terms

For a fixed group type (j=(r_j,D_j)), drop global constants and write its full ideal-objective contribution as

[
C_j(W)
======

A_jW-\lambda'W\log_2W,
]

where

[
A_j
===

\hat\mu r_j+\hat\kappa_{D_j}+\lambda'D_j.
]

This is concave in (W). Define the marginal score

[
S_j(W)
======

# -C_j'(W)

\lambda'
\left(
\log_2W+\frac1{\ln2}
\right)
-A_j.
]

The same exchange argument shows:

> In an optimum, groups ordered by decreasing (S_j(W_j)) receive contiguous decreasing-frequency blocks.

For gap alone, this reduces to ordering by (W_j/2^{D_j}).

The exact-enumerative objective also has the property. Replace (W\log W) by (\log\Gamma(W+1)); log-gamma is convex, so the same exchange argument applies using the digamma derivative.

## There is no universal order by flat size

The ordering depends on the block sum (W), not only on (D).

With group sizes (1) and (2):

* counts ((100,1,1)) optimally give the singleton (100) and pair ((1,1));
* counts ((100,99,1)) optimally give the pair ((100,99)) and singleton (1).

So “larger chunks first” and “smaller chunks first” both fail as universal rules.

---

# 4. Q2: an exact DP does survive, but it is a segmentation DP

The nonlinear objective eliminates the previous global item-cost order, but contiguity gives something nearly as good.

Sort

[
n_1\ge n_2\ge\cdots\ge n_\sigma
]

and define prefixes

[
P[k]=\sum_{i=1}^kn_i,
\qquad
Q[k]=\sum_{i=1}^kn_i\log_2n_i.
]

Let the maximum root depth be (R), so Kraft mass is represented in units of (2^{-R}). A candidate group type is

[
t=(r,D),
]

with

[
m_t=2^D,
\qquad
a_t=2^{R-r}.
]

Enforce any applicable length cap, for example

[
r+D\le L_{\max}.
]

For a segment beginning after (k) and containing the next (m_t) symbols, let

[
j=k+m_t,
\qquad
W=P[j]-P[k].
]

Its entropy gap is

[
g_H(k,t)
========

D W-W\log_2W+Q[j]-Q[k].
]

This is computable in (O(1)).

Define

[
F[k,q]
======

\text{minimum cost for the first (k) sorted symbols using Kraft mass (q)}.
]

The transition is

[
\begin{aligned}
F[k+m_t,q+a_t]
==============

\min\bigg{
&F[k+m_t,q+a_t],\
&F[k,q]
+W(\hat\mu r+\hat\kappa_D)
+\lambda' g_H(k,t)
+h_t
+\hat\gamma
\bigg}.
\end{aligned}
]

The terminal is

[
F[\sigma,2^R]-\hat\gamma,
]

because (C) chunks create (C-1) skeleton merges.

For exact enumerative gaps, replace (g_H) by

[
g_E(k,t)
========

D W-\log_2\Gamma(W+1)
+
Q_![j]-Q_![k],
]

where

[
Q_![k]
======

\sum_{i=1}^k\log_2\Gamma(n_i+1).
]

## Correctness

Every feasible DP path defines:

* a partition into contiguous frequency blocks;
* a flat depth (D) and root depth (r) for each block;
* total Kraft mass exactly one.

Conversely, the contiguity theorem says that some global optimum has exactly that form. Kraft equality guarantees a prefix code for the chunk roots.

Therefore:

[
\boxed{
\text{The DP is exact for the free-assignment, free-chunk-multiplicity §4 problem.}
}
]

Its complexity is

[
O!\left(\sigma,2^R,|\mathcal T|\right),
]

and memory is (O(\sigma2^R)). This is pseudo-polynomial in the depth cap, but with (R=11) or (15) it is a small fixed grid.

For gap alone, with no root-depth optimization, it reduces to a simple one-dimensional segmentation DP of roughly (9\sigma) transitions.

---

# 5. The free-partition §4 is not yet the production-wire problem

The original lengths-only wire imposes additional constraints that §4 currently omits.

For a chunk with root depth (r) and flat depth (D), define

[
L=r+D.
]

The original wire permits at most one chunk for each pair ((L,D)), because the (D)-bit of (c_L) is either set or unset. Thus two identical ((r,D)) chunks are generally inexpressible.

The production feasible set also includes:

* within each (L), selected chunks ordered by decreasing (D);
* symbols dealt according to value order, not frequency order;
* canonical root assignment;
* any maximum code-length bound;
* mandatory (D=1) fusion, unless the format is changed to make it optional.

The segmentation DP therefore gives:

* an exact algorithm for a natural free-partition variant;
* a lower bound for the original lengths-only wire.

## A practical exact solver for the frequency-assignment wire variant

Precompute one edge for every possible:

[
(\text{segment start }k,\ r,\ D).
]

Use binary variables saying whether that edge appears in a source-to-sink segmentation path. Add:

* ordinary path-flow constraints;
* exact Kraft mass;
* at-most-one-use constraints for every ((L,D));
* precedence constraints for the largest-first rule within each (L).

All nonlinear costs are already precomputed per edge, so this is a compact 0/1 MILP or CP-SAT model.

There are only on the order of

[
\sigma|\mathcal T|
]

binary variables. It has no obvious polynomiality guarantee, but at (\sigma\le256) it should be a very realistic exact benchmark. The unconstrained DP supplies a strong lower bound and warm start.

For true production variant A, value-order grouping destroys the frequency-contiguous representation. That remains the hard coupling.

---

# 6. Q3: the ideal frontier has a clean local calculus

The KL expression gives a useful global interpretation:

[
\frac{G_H}{N}
=============

D_{\rm KL}(p\Vert\bar p).
]

But an even more useful operational identity appears when two equal-depth flats are joined.

Suppose flats (A) and (B), each of depth (D), are separated by one surviving merge. Let

[
W=W_A+W_B,
\qquad
q=W_A/W.
]

Replacing the two (D)-flats plus their parent merge by one ((D+1))-flat changes the entropy gap by

[
\boxed{
\Delta G_H
==========

W\bigl(1-H_2(q)\bigr).
}
]

This is independent of the internal distributions within (A) and (B).

Under exact enumeration,

[
\boxed{
\Delta G_E
==========

W-\log_2\binom{W}{W_A}.
}
]

The corresponding decode-time saving is

[
\boxed{
\Delta T_{\rm save}
===================

W\left(
\hat\mu+\hat\kappa_D-\hat\kappa_{D+1}
\right),
}
]

before fixed startup terms.

Therefore the local entropy-gap price per time saved is

[
\frac{1-H_2(q)}
{\hat\mu+\hat\kappa_D-\hat\kappa_{D+1}},
]

whenever the denominator is positive.

Near balance, with (q=\tfrac12+\delta),

[
1-H_2(q)
========

\frac{2}{\ln2}\delta^2+O(\delta^4).
]

This explains the knee:

* nearly balanced decisions are quadratically cheap to flatten;
* strongly skewed decisions are expensive to replace with a raw bit;
* nonmonotone (\kappa_D) can make some larger flats slower even at zero gap.

For a fixed perfect hierarchy, the supported frontier can be computed by a tree DP selecting contractions. A hard gap budget requires Pareto labels; as in the first problem, a (\lambda')-sweep finds only supported frontier points.

---

# 7. Dyadic distributions: exact zero-gap characterization

Under the entropy model,

[
G_H(c)=0
]

if and only if all leaves within flat (c) have equal probability.

Suppose

[
p_s=2^{-\ell_s}.
]

A group of (2^D) equal-probability symbols has mass

[
P_c
===

# 2^D2^{-\ell}

2^{-(\ell-D)}.
]

Thus assigning it root depth

[
r=\ell-D
]

is exactly Kraft-compatible, and every symbol retains total depth

[
r+D=\ell.
]

So, for dyadic (p):

> Zero-gap flats are precisely contractions of equal-probability perfect subalphabets.

If

[
\hat\mu+\hat\kappa_D\ge\hat\kappa_{D+1},
]

merging two such equal groups is also non-worsening in time. If this condition fails, the zero-gap time optimum may deliberately stop at (D=4), for example, rather than using a slower (D=5) kernel.

## The zero-gap knee need not be near the global minimum

Uniform 256-way data is the best case:

[
p_s=1/256,
]

so one (D=8) flat has zero entropy gap and zero merge passes.

At the opposite extreme, consider the complete dyadic distribution

[
\left(
\frac12,\frac14,\frac18,\frac1{16},
\frac1{32},\frac1{64},\frac1{128},
\frac1{256},\frac1{256}
\right).
]

Only the final two symbols have equal probability. The only nontrivial free flat is their fused pair, whose total mass is (1/128). The free reduction in average merge passes is therefore only (1/128).

Thus there is no universal theorem that “maximal uniform flats already get close to minimum passes.” It ranges from essentially no speedup to all merges removed.

## Finite-block correction

Even if the true conditional distribution is uniform, empirical block counts will not usually be exactly equal. For a uniform (m)-symbol flat of large weight (W),

[
\mathbb E!\left[
W(D-H(\hat p))
\right]
\approx
\frac{m-1}{2\ln2}
\quad\text{bits}.
]

For (m=256), that is about 184 bits, or 23 bytes, per empirical block. This vanishes per symbol as (W) grows, but it is visible at very small table cadences.

---

# 8. Q4: the coder dial needs bits and time separated

The §5 expression currently puts a decode-time tax (\tau W) inside a bit-count minimum. Those are different dimensions.

A cleaner model is

[
B_v(\eta)
=========

\begin{cases}
W_v,&W_v<G,[3pt]
\min\left{
W_v,;
h_b+\rho W_v+\dfrac{W_vH_2(q_v)}{\eta}
\right},
& W_v\ge G,
\end{cases}
]

where:

* (h_b) is marker/termination overhead in bits;
* (\rho W) is rate inefficiency in bits;
* (G) is the minimum-size gate.

Then separately,

[
T_v
===

\hat\mu W_v
+
z_v(h_t+\tau W_v),
]

where (z_v) indicates that the coded branch was selected.

If the implementation commits solely when bytes shrink, (z_v) is determined by (B_v), not by the combined objective. If the encoder is permitted to decline coding for speed, the branch decision should instead minimize the full bit/time objective.

## Per-node commit threshold

Ignoring the size gate, coding wins in bytes when

[
h_b+\rho W+\frac{W H_2(q)}{\eta}<W.
]

Equivalently,

[
\eta

>

# \eta_v^{\rm commit}

\frac{H_2(q)}
{1-\rho-h_b/W},
]

provided the denominator is positive.

For a fixed candidate tree, the set of coded nodes changes only at these thresholds. Between thresholds, its cost is of the form

[
A+\frac{B}{\eta}.
]

Different candidate trees can cross several times, so there is no universal single (\eta^*).

---

# 9. A local theorem about flattening under a real coder

Suppose one surviving merge joins two (D)-flats and can be replaced by one ((D+1))-flat.

Before flattening, the extra routing level costs (B_v(\eta)) bits. After flattening, the flat stores one raw bit per occurrence. Therefore

[
\boxed{
\Delta B_{\rm flatten}
======================

W_v-B_v(\eta)
+\Delta B_{\rm records}.
}
]

More generally, flattening an entire perfect subtree with internal merge set (\mathcal I) costs

[
\boxed{
\Delta B_{\rm flatten}
======================

\sum_{v\in\mathcal I}
\left[W_v-B_v(\eta)\right]
+
\Delta B_{\rm records}.
}
]

This is the coder savings that flattening forfeits.

Consequences:

* If a bitmap remains raw, (B_v=W_v), so flattening it has zero first-order ratio cost.
* If it compresses well, (B_v\ll W_v), so flattening it has a large ratio cost.
* As the coder becomes better, the **wire-only** penalty for flattening is nondecreasing.

This is opposite to the simplest reading of “a better coder lets us flatten more.” A better coder makes it cheaper to **retain** nonuniform routing decisions.

There is an opposing decode effect: when a node switches to coded form, its FSE decode tax may make flattening more attractive in time. Thus the full (J) can jump when coding activates, but the rate part has the monotonic direction above.

## Why your observed surviving merges become more skewed

The local formula supplies a direct explanation.

An entropy-aware optimizer preferentially flattens nodes for which

[
W_v-B_v
]

is small—that is, balanced or otherwise uncompressible bitmaps. It preferentially retains nodes for which (W_v-B_v) is large—that is, skewed and highly compressible bitmaps.

So the observation is real, but it is a **selection effect**:

> Flattening removes the least coder-valuable routing decisions and leaves the most coder-valuable ones.

This is much stronger and more useful than saying that flattening mechanically increases skew.

The minimum-size gate reinforces it. For (W<G), (B_v=W), so those nodes are ratio-free to flatten. That predicts PH-like aggressive flattening at small windows and more selective flattening once nodes cross the coder’s useful-size threshold.

---

# 10. Q5: the proposed skew theorem is false

For a fixed flat partition, let (P_c=W_c/N) be the chunk distribution. Over the surviving skeleton,

[
\sum_vW_vH_2(q_v)
=================

N H(P_1,\ldots,P_C).
]

Also,

[
\sum_vW_v
=========

\sum_cW_cr_c,
]

because each chunk occurrence crosses (r_c) surviving merges.

Therefore

[
\boxed{
\sum_vW_v\bigl(1-H_2(q_v)\bigr)
===============================

## \sum_cW_cr_c

N H(P_1,\ldots,P_C).
}
]

This gives two cases.

## If root depths are fixed

Both terms on the right are fixed. Therefore

[
\sum_vW_v(1-H_2(q_v))
]

is exactly invariant under all sibling pairings and skeleton arrangements.

No construction can maximize it; all arrangements tie.

## If root depths are not fixed

The entropy term is still fixed, so maximizing aggregate skew redundancy is exactly equivalent to maximizing

[
\sum_cW_cr_c,
]

the weighted merge-pass count.

Thus the tree with maximum aggregate skew is a deliberately bad, deep tree. The Huffman skeleton, which minimizes passes, also minimizes this aggregate raw-over-entropy redundancy.

A simple counterexample uses chunk weights

[
\left(\frac12,\frac14,\frac14\right).
]

Putting (1/2) at depth one and the two (1/4) chunks at depth two gives

[
L=1.5,\qquad H=1.5,
]

so aggregate skew redundancy is zero.

Putting a (1/4) chunk at depth one and the (1/2,1/4) chunks at depth two gives

[
L=1.75,\qquad H=1.5,
]

so aggregate skew redundancy is (0.25N). It is more skew-compressible precisely because it performs more raw merge work.

## The correct coder objective

For the production coder, the relevant quantity is not

[
\sum_vW_v(1-H(q_v)),
]

but the actual savings

[
\boxed{
S_{\rm coder}
=============

\sum_v\bigl(W_v-B_v\bigr).
}
]

Fixed markers, gates, quantization, and termination mean that the distribution of the theoretical redundancy across nodes matters. Concentrating a fixed amount of redundancy into fewer large nodes can help amortize headers, but there is no general proof that the canonical flat-first tree produces the best concentration.

---

# 11. Root-tree consequence under ideal coding

For a fixed flat partition, ideal routing rate is independent of the skeleton shape. If merge time is simply (\hat\mu W_v), the optimal skeleton is therefore the minimum-weighted-path-length tree over the chunk weights (W_c): the ordinary Huffman skeleton.

Coder reality modifies the merge cost to something like

[
g(a,b)
======

\hat\mu(a+b)
+
\lambda'B_\eta(a+b,a/(a+b))
+
\text{fixed terms}.
]

With this general merge cost, the usual Huffman greedy proof no longer applies. If the production wire fixes the canonical tree, this is only a scoring issue. If skeleton arrangement becomes selectable, it is a nonstandard optimal-merge-pattern problem.

For small chunk counts an exact subset recurrence is possible; for large counts, a beam search or generalized-Huffman heuristic followed by exact scoring is more realistic.

---

# 12. Q6: enumerative coding is exact, but naïve decoding is serial

A length-(W) bitmap with (K) ones can be ranked among

[
\binom WK
]

constant-weight strings.

There is also a sequential interpretation. With (w,k) remaining, assign

[
\Pr(1)=k/w,\qquad
\Pr(0)=(w-k)/w.
]

For any complete bitmap with (K) ones, the product of the sequential probabilities is

[
\frac{K!(W-K)!}{W!}
===================

\frac1{\binom WK}.
]

Therefore an arithmetic coder using this changing “without replacement” model realizes the enumerative rate without ever holding the enormous rank integer.

Arithmetic coding naturally permits symbol-by-symbol changing probability models, although finite-precision implementations require interval updates, renormalization, and careful arithmetic. ([arXiv][2])

## Why standard FSE is not exact enumerative coding

A static binary FSE table based on (q=K/W) approximates the Bernoulli cost

[
W H_2(q),
]

not the type-class cost

[
\log_2\binom WK.
]

The exact without-replacement probability changes after every decoded bit. A single static tANS/FSE table cannot express that changing model exactly.

FSE’s speed comes from its finite-state table transition: decode a symbol from the current state, consume a small number of bits, and update the state from a power-of-two-sized table. ([RFC Editor][3])

For interior (q),

[
W H_2(q)-\log_2\binom WK
========================

\frac12\log_2!\bigl(2\pi Wq(1-q)\bigr)
+O(1/W).
]

So enumerative coding can actually beat the H-model by several bits per node when the exact count is already free side information. In that sense it is not merely (\eta=1); it lies slightly beyond the empirical-entropy endpoint.

Whether that gain remains after charging the (K)-records is a wire-accounting question.

---

# 13. The most promising enumerative kernel: block hypergeometric decoding

I would not begin with one adaptive arithmetic operation per routing bit. That is likely to erase much of the merge-speed gain.

Instead, decode routing masks in blocks of (B=8) or (16) bits.

With (W,K) remaining, let (T) be the number of ones in the next (B)-bit mask. Then

[
\Pr(T=t)
========

\frac{
\binom Bt
\binom{W-B}{K-t}
}{
\binom WK
}.
]

Conditional on (T=t), each of the

[
\binom Bt
]

masks of weight (t) is equiprobable.

So the decoder can:

1. entropy-decode (t) under the hypergeometric distribution;
2. decode a rank in ([0,\binom Bt));
3. table-lookup the complete (B)-bit mask;
4. hand that packed mask to the existing merge kernel;
5. update (W\leftarrow W-B,\ K\leftarrow K-t).

For (B=8), the largest mask class has only

[
\binom84=70
]

entries. For (B=16), the largest has 12,870 entries.

This design has several attractions:

* one entropy decision per 8 or 16 routed occurrences, rather than one per occurrence;
* the merge kernel still consumes ordinary packed masks;
* no huge integers;
* exact count conditioning is retained if the (t)-sequence is coded accurately;
* several node streams can be interleaved to hide coder latency.

This is the enumerative design I would benchmark first.

---

# 14. A second option: one coder stream for the entire skeleton

The one-byte marker and per-node startup are probably more damaging than the first-order entropy inefficiency at modest (W).

A single arithmetic/range stream can process all surviving node bitmaps in a deterministic traversal order, switching the binary model at node boundaries. Arithmetic coding supports changing distributions during a stream, so only one termination state is needed. ([arXiv][2])

Possible organizations are:

* one global stream;
* one stream per tree level;
* four or eight interleaved streams;
* one stream per independent decode lane.

This would:

* amortize termination and markers across the whole skeleton;
* make balanced nodes cost approximately one bit without a raw/coded branch;
* compress skewed nodes automatically;
* largely remove the minimum-size gate.

The cost is greater serial dependence and a more expensive per-bit decode than raw packed masks. A small codebook of quantized binary ANS tables may provide a middle point, although exact switching behavior and redundancy would need benchmarking.

---

# 15. A useful architectural control: entropy-coded flats

A flat region is simply a sequence over an alphabet of size (2^D). Instead of storing that sequence in raw (D)-bit form, it could be decoded by an (m)-ary ANS/FSE/range kernel.

That would replace

[
W D
]

by approximately

[
W H(p\mid c)
]

while still avoiding all interior merges.

In the ideal limit, flat placement would then truly be rate-free and the optimization would become almost pure decode time.

The costs are substantial:

* a model or counts for each flat;
* table setup and memory;
* a slower flat kernel;
* for one (D=8) flat, this is essentially an ordinary 256-symbol entropy decoder.

Still, it is an important benchmark. An all-flat, entropy-coded (D=8) stream tells you the performance ceiling of “discard the routing tree and entropy-decode symbols directly.”

---

# 16. How enumerative coding changes Q4

An exact enumerative bitmap coder lowers the cost of retained routing below (WH(q)), conditioned on (K). Thus it increases

[
W-B_v
]

and makes replacing that bitmap with a raw flat bit more expensive in ratio.

So, all else equal, enumerative coding pushes toward:

* keeping skewed routing decisions;
* flattening balanced decisions;
* using the skeleton shape for speed because its rate is almost shape-invariant;
* avoiding nonuniform large flats unless their decode saving is compelling.

The opposing force is decoder cost. If enumerative decoding is materially slower than a raw merge, the time saving from removing a coded merge becomes larger.

Thus the answer is determined by the measured local inequality

[
\begin{aligned}
&\text{time saved by flattening}\
&\qquad >
\lambda'
\left(
\text{enumerative bits forfeited}
\right).
\end{aligned}
]

For a single (D\to D+1) contraction, including coder tax,

[
\begin{aligned}
\Delta J
={}&
W\left[
\hat\kappa_{D+1}
-\hat\kappa_D
-\hat\mu
-z\tau
\right]\
&+
\lambda'
\left[
W-B_v
+\Delta B_{\rm records}
\right]
+\Delta h_t.
\end{aligned}
]

That is the local decision rule I would fit from measurements.

---

# 17. Recommended optimization pipeline

## Stage 1: separate the rate oracles

Implement three scorers:

[
\begin{array}{ll}
\text{PH:}&B_v=W_v,[2pt]
\text{H-ideal:}&B_v=W_vH_2(q_v),[2pt]
\text{E-ideal:}&B_v=\log_2\binom{W_v}{W_{v,L}}.
\end{array}
]

Then add the measured FSE scorer as a fourth oracle.

This will reveal whether the important correction comes from first-order entropy, exact count conditioning, or merely amortizing node overheads.

## Stage 2: solve the free-partition ideal problem exactly

Use the contiguous segmentation/Kraft DP. It gives:

* an exact ideal frontier for the relaxed model;
* a lower bound for every production-wire scheme;
* a candidate generator for exact rescoring.

Maintain Pareto labels if the real requirement is a hard ratio budget rather than a supported (\lambda')-point.

## Stage 3: reimpose the production wire

Use the edge-path MILP/CP-SAT formulation with:

* Kraft equality;
* one use per ((L,D));
* largest-first precedence;
* maximum length;
* any mandatory fusion rule.

For value-order production, exact-score the selected candidates and use the ideal DP objective as a lower bound in branch-and-bound.

## Stage 4: model local contractions directly

For every eligible contraction, record:

[
\begin{aligned}
\text{pass saving}&=W,\
\text{kernel delta}&=W(\kappa_D-\kappa_{D+1}),\
\text{entropy price}&=W(1-H_2(q)),\
\text{actual price}&=W-B_v,\
\text{record delta}&=\Delta B_{\rm records}.
\end{aligned}
]

This will show immediately whether the current operating point lies before or after the real knee.

## Stage 5: benchmark coder architectures

The most informative comparison is:

1. raw packed bitmap;
2. current per-node FSE;
3. block-global binary range/ANS;
4. 8-bit block-hypergeometric enumerative;
5. 16-bit block-hypergeometric enumerative;
6. direct (m)-ary entropy-coded flat.

Benchmark across (W) and (q), not only file averages, because the optimizer needs the surface

[
(B(W,q),T(W,q)).
]

---

# Bottom line

The earlier raw-length objective does dissolve, but it is replaced by a very structured problem rather than no problem:

[
\boxed{
\text{choose flats to trade decode time against }
N,D_{\rm KL}(p\Vert\text{piecewise-uniform }p).
}
]

The strongest conclusions are:

* **Contiguity is true.**
* **The relaxed ideal problem has an exact segmentation/Kraft DP.**
* **Uniform dyadic subalphabets are the zero-gap opportunities, but they need not remove many passes.**
* **The skew-maximization theorem is false; aggregate redundancy is determined by weighted path length.**
* **Entropy-aware flattening naturally removes balanced bitmaps and preserves skewed ones.**
* **A better bitmap coder does not automatically imply more flattening; in wire terms it usually makes retaining nonuniform merges more attractive.**
* **Block-hypergeometric enumeration and a block-global coder are the two most promising next experiments.**

My strongest architectural prediction is not “flatten everything once PHA improves.” It is:

[
\boxed{
\text{flatten balanced structure aggressively,
retain and entropy-code skewed structure,
and make the skeleton as pass-efficient as the wire permits.}
}
]

[1]: https://ieeexplore.ieee.org/iel5/18/22667/01054929.pdf "https://ieeexplore.ieee.org/iel5/18/22667/01054929.pdf"
[2]: https://arxiv.org/pdf/2302.00819 "https://arxiv.org/pdf/2302.00819"
[3]: https://www.rfc-editor.org/rfc/rfc8478.txt "https://www.rfc-editor.org/rfc/rfc8478.txt"

# Flat placement under entropy-coded routing — problem statement

Self-contained, written for review without codebase knowledge.  Sequel
to docs/JOINT-LENGTHS.md (whose review, RESPONSE.md, already proved
the germ of this as its §8 "bonus identity"); this document takes that
identity to its conclusion, which appears to dissolve the objective
the previous problem was built on.  Status: proposed reformulation;
we want the claims checked and the open questions attacked.

## 1. Setting (recap; §1–2 of JOINT-LENGTHS.md in brief)

We entropy-code blocks of `N` symbols from an alphabet of `σ ≤ 256`
symbols with counts `n_s` (`Σ n_s = N`, weights `W` below are count
sums).  The decoder walks a binary prefix-code tree **bottom-up**:

* Each internal node not inside a flat subtree performs a **merge**:
  one pass over the `W_v` symbol occurrences routed through it,
  interleaving its children's streams by a transmitted **routing
  bitmap** (one bit per occurrence).  Cost ≈ `μ` per occurrence.
* A **flat subtree** (all `2^D` leaves exactly `D` levels below its
  root, `2 ≤ D ≤ 8`) is decoded by table lookup over a packed
  `D`-bit-per-occurrence region — no interior bitmaps.  Cost `κ_D`
  per occurrence, `κ_D ≪ μ`, not monotone in D.
* An occurrence of symbol `s` in a flat rooted at depth `r` therefore
  costs `r` merge passes + one flat lookup.

Wire: a 128-byte code-lengths header, per-merge-node records
(header + bitmap), per-flat packed regions.  Each merge bitmap may
optionally be entropy-coded (per-node FSE with a 1-byte marker,
committed only when it wins) — "PHA"; "PH" ships all bitmaps raw.

JOINT-LENGTHS.md optimized the code lengths under the objective
`Σ n_s·ℓ_s + λ·(merge passes)`, i.e. it priced ratio as **raw code
bits**.  This document is about what happens to that objective when
the bitmaps are entropy-coded.

## 2. The identity: routing information is shape-invariant

**Exact form.**  For ANY full binary tree over the alphabet, the
product over internal nodes of the bitmap arrangement counts
telescopes to the multinomial:

```
Π_v C(W_v, W_left(v))  =  N! / Π_s n_s!
```

(Each node splits its multiset; induction from the root.)  So if
every bitmap were coded to its exact arrangement count (enumerative /
adaptive arithmetic coding conditioned on the counts), total routing
bits = `log2( N! / Π n_s! )` — **independent of the tree shape**.

**Entropy form.**  First-order version: a node splitting `W_v`
occurrences with left fraction `q_v` ideally costs `W_v·H2(q_v)`
bits, and

```
Σ_v W_v·H2(q_v)  =  N·H(p)        (p_s = n_s/N)
```

by the entropy chain rule down the tree.  Again shape-invariant.

**Corollary (the point).**  Under ideal routing coding, *code lengths
do not affect compressed size at all*.  Any complete tree, Huffman or
wildly distorted, transmits `N·H(p)` routing bits plus per-node
constants.  The only structural choice with a ratio consequence is
which subtrees are FLAT, because a flat stores its region **raw**:

```
flat c with leaf set F, |F| = 2^D, weight W_c = Σ_{s∈F} n_s:
  raw cost           =  W_c · D
  ideal routing cost =  W_c · H(p|F)      (conditional entropy)
  ENTROPY GAP(c)     =  W_c · (D − H(p|F))  =  W_c · KL(p|F ‖ Uniform(2^D))  ≥ 0
```

and total wire (ideal coding, constants aside) is

```
Wire  =  N·H(p)  +  Σ_flats GAP(c).
```

**Sanity anchors.**  (a) The all-flat tree (σ = 256, one D = 8 flat)
has no bitmaps, so ideal and raw models agree exactly there: gap =
N·(8 − H), i.e. fixed 8-bit coding.  Our λ-sweep approaches this
limit from below (passes ×0.094 remaining at λ = 1 for +8.9 % raw
bits).  (b) A flat over equal-probability leaves has gap 0: flats
covering uniform sub-alphabets are FREE under ideal coding.  (c)
Fused pairs (D = 1) have gap `W·(1 − H2(split))` — nonzero unless
50/50, so even the baseline tree is already paying gaps today
wherever bitmaps go uncoded.

## 3. The gap algebra is clean

Summing gaps over a partition of the alphabet into flats (a lone
leaf is D = 0, gap 0):

```
Σ_c GAP(c)  =  Σ_c W_c·(D_c − log2 W_c)  +  Σ_s n_s·log2 n_s
```

The second term is partition-independent.  So minimizing total gap
means maximizing `Σ_c W_c · log2(W_c / 2^{D_c})` — each flat wants
its "fullness" `W_c / 2^{D_c}` (average count per leaf slot) large,
i.e. flats want to be small-and-heavy or uniform-and-anything.

## 4. The reformulated problem

Choose a partition of the alphabet into groups of dyadic sizes
`2^{D_c}` (`0 ≤ D_c ≤ 8`) and a root depth `r_c` for each, subject to
root Kraft equality `Σ_c 2^{−r_c} = 1`, minimizing

```
J'  =  Σ_c W_c·( μ̂·r_c + κ̂_{D_c} )        (decode time: passes + kernels)
     + λ'·Σ_c W_c·( D_c − log2 W_c )        (ratio: entropy gaps, + const)
     [ + coder-reality terms, §5 ]
```

Structurally this is JOINT-LENGTHS.md with `ℓ_s = r_c + D_c` — but
the ratio term changed from LINEAR in the prefix-sum weights
(`Σ n_s ℓ_s`) to the **concave-in-W** form `−W log W`-plus-linear.
The linearity is exactly what our exact DP's sorted-matching argument
(rearrangement inequality over constant per-slot costs) leaned on.

## 5. Coder reality: interpolating between the two objectives

Ideal coding is the η = 1 end of a dial.  The production coder is
per-node FSE with: a marker byte per merge node, a commit-only-if-
smaller rule, a minimum useful size (at G ≤ 16 K windows the
baseline's FSE never fired at all — 0.000 pp on every file; ~+0.4 pp
average at 128 K, up to +1.8 pp per file), a decode-time tax on
FSE'd nodes,
and 0th-order-with-block-structure efficiency < 1.  PH is η = 0,
where JOINT-LENGTHS.md's raw-bits objective is exact.

A workable middle model: each merge node v pays
`min( W_v, marker + η⁻¹-adjusted W_v·H2(q_v) + τ·W_v )` — the
commit rule makes the objective non-separable and slightly
non-monotone, but per-node it is still a simple function of
`(W_v, q_v)`.

One measured phenomenon the model should reproduce: distorted
(flattened) trees make the SURVIVING merges more skewed — flats
absorb the balanced structure, so residual bitmaps are exactly the
FSE-able ones (observed: FSE fires more under joint lengths at
small G than under Huffman).

## 6. Questions for review

1. **Contiguity.**  For fixed group sizes, is the gap-minimizing
   partition contiguous in frequency-sorted order?  (The exchange
   argument that worked for linear costs needs `−W log W` handled;
   we believe contiguity survives via a majorization argument but
   have not proved it.  If false, the wire's value-order dealing
   constraint bites harder than before.)
2. **Exact algorithm.**  Our slot-ledger DP transitions cost
   `rate × (P[k+2^b] − P[k])` — linear, enabling the global-cost-
   order exactness argument.  With per-chunk cost
   `f(W) = W·(D − log2 W)` the per-transition value is still O(1)
   from prefix sums, but the sorted-matching argument breaks: chunks
   no longer have a well-defined per-slot cost.  Does the DP remain
   exact under some processing order (the concavity of −W log W
   suggests an exchange argument ordering chunks by size within
   level)?  Is the problem still polynomial?
3. **Frontier shape.**  Under ideal coding the ratio axis is pure
   gap.  For an iid source, characterize the passes-vs-gap Pareto
   frontier analytically (e.g. dyadic p: is the frontier's knee at
   "flats = maximal uniform sub-alphabets, gap 0, passes already
   near-minimal"?).  Empirically our λ-sweep endpoint (all-flat,
   gap = N(8−H)) and start (Huffman, gap ≈ small) bracket it.
4. **The dial.**  With the §5 commit rule (min of raw and coded),
   the objective interpolates PH → PHA → ideal.  How sensitive is
   the optimal flat partition to η and the minimum-size gate?  Is
   there a threshold η* above which "flatten aggressively, let the
   coder mop up the skeleton" strictly dominates the current
   raw-bits-priced optimum?
5. **Skew theorem.**  Prove (or refute): among trees realizing the
   same flat partition, the joint/flat-first construction maximizes
   Σ_v W_v·(1 − H2(q_v)) over surviving merges — i.e. flattening
   necessarily concentrates routing information in fewer, more
   skewed, more compressible bitmaps.
6. **Enumerative option.**  The exact-count identity suggests coding
   bitmaps enumeratively (transmit nothing beyond counts once K_right
   is known per node — the bitmap costs `log2 C(W, W_L)` exactly).
   K_right headers already exist on the wire.  What decode-speed
   price would an enumerative/ANS bitmap coder pay, and does it
   change the answer to Q4?

## 7. Why we care (practical stakes)

At the 5–100 KB adaptive-table cadence, PH + joint lengths currently
wins +17–68 % decode e2e at ≤ +0.2 pp ratio (flat placement priced
in raw bits).  If Q4's answer is that a cheap bitmap coder shifts
the optimum toward substantially flatter trees at the SAME true
ratio, the decode ceiling rises again — the λ-sweep shows 91 % of
merge passes are removable if the gap bits become free.  Conversely
if the frontier knee (Q3) is where we already operate, this document
closes the question and PH stays the shipping configuration.

# Making Huffman trees fast, not just small

*A plain-language tour of the joint code-length / tree-shape optimizer
(branches `joint-flat-lengths` and `joint-flat-lengths-plus-fast-tables`)
and the cost-model work that followed it (branch `joint-cost-model`).
Technical companion docs: `docs/JOINT-LENGTHS.md` (problem statement +
full results) and `docs/KAPPA-COSTS.md` on the follow-on branch.  All
numbers below are Apple M-class measurements on LZ4 literal streams
unless stated otherwise; see "The data" near the end.*

---

## The one-sentence version

Huffman coding picks code lengths that make the file as small as
possible; we deliberately bend those lengths a little — at a bounded,
explicitly-priced cost in compression — so that the resulting tree has
a *shape* our decoder is much faster at, and we do it with three
interchangeable algorithms that cost ~100 µs, ~10 µs, or ~2 µs of
encode-side time per table.

The punchline, measured end-to-end against the pre-existing main
branch on realistic table-rebuild cadences: the cheapest rung makes
**both encoding and decoding faster at almost identical compression**,
and the expensive rungs buy up to **2.3× decode speed** at small table
periods — with compression *better* than baseline at the smallest ones.

---

## 1. Two ways to judge a prefix code

A Huffman code assigns every byte value a codeword: frequent bytes get
short codewords, rare bytes get long ones.  The classic Huffman
algorithm produces the assignment with the smallest possible total
size.  That's one way to judge a code: *how many bits does it write?*

There is a second question that classic Huffman never asks: *how much
work is it to decode?*  For a traditional bit-by-bit decoder the two
questions have nearly the same answer — every bit gets looked at once
either way.  PIVCO's decoder is different, and for it the two
questions come apart dramatically: two trees within a tenth of a
percent of each other in compressed size can differ by 2× in decode
speed.

Classic Huffman optimizes the first number and is blind to the second.
This work optimizes both at once.  Hence "joint".

## 2. How PIVCO decodes, and why shape matters

PIVCO doesn't walk the code tree bit by bit.  It decodes a whole block
(8,192 symbols) level by level, bottom-up, using SIMD:

* Every internal tree node stores a **bitmap** in the compressed
  stream: one bit per symbol routed through that node, saying "left
  child or right child".  Decoding that node means one linear pass
  that interleaves the two children's outputs according to the bitmap.
  We call this a **merge**.  Cost: proportional to how many symbol
  occurrences pass through the node.
* A **flat subtree** — an internal node whose 2^D leaves all sit
  exactly D levels below it — skips all of that.  Its D levels are
  stored as one packed D-bits-per-symbol region and decoded by a
  single table-lookup pass, whatever D is (2 ≤ D ≤ 8).  One pass
  instead of D passes.
* A two-leaf pair (D = 1) is fused into its parent's merge for free,
  and a lone leaf (D = 0) is just a byte fill.

So the decode cost of one symbol occurrence is essentially: **the
number of merge levels above it**.  A symbol with a 7-bit code that
lives inside a depth-6 flat subtree costs one merge pass plus one
cheap lookup.  A symbol with a 7-bit code that lives under seven
plain merge nodes costs seven passes.  Same compressed size.  Seven
times the memory traffic.

There's a tidy way to see this that's worth internalizing: in this
codec, **compressed bits and decode work are the same thing viewed
from two sides**.  Every "routing" bit — a bit that steers a symbol
through a merge node — is a bitmap bit that must be both *stored* and
*walked over* at decode time.  A bit inside a flat region is only
stored; the lookup swallows all D of them at once.  Moving bits from
"routing" to "flat" barely changes the file size, but it deletes real
decode work.

## 3. Round numbers make good trees

Here's the crucial mechanism.  The decoder rebuilds the whole tree
from just the 256 code lengths (a 128-byte header).  The rule it uses
is simple counting:

1. Count how many symbols have each code length.  Call the count for
   length L: c(L).
2. Write each count in binary.  **Every 1-bit becomes a "chunk"**: a
   set bit at position b contributes one flat block of 2^b codewords.
3. Those chunks are then stitched together with merge nodes.

The binary decomposition is provably the best possible grouping for a
*given* set of counts (biggest flat coverage, fewest chunks, fewest
merges — see JOINT-LENGTHS.md §3).  But it makes tree quality hostage
to the *binary texture of the counts*:

```
c(7) = 64          -> 1000000  -> one flat block of 64.  One chunk.
c(7) = 60          -> 0111100  -> blocks of 32+16+8+4.   Four chunks,
                                  three extra merge nodes to stitch.
c(7) = 61          -> 0111101  -> those four PLUS a lone leaf.
```

A length class of 64 symbols is one beautiful flat block.  A class of
60 — a count Huffman is perfectly happy to produce, since it only
cares about sizes — shatters into four blocks and extra merges.  Every
symbol in the class pays for that shattering on every decode.

So the lever is obvious once you see it: **nudge a few symbols to
slightly longer or shorter codes so the class counts land on round
binary numbers.**  Push four symbols from length 6 into that class of
60 (with a compensating change elsewhere — more on the bookkeeping in
a moment) and the whole class collapses into one block.

## 4. Pricing the trade

Making a code longer than Huffman-optimal costs bits.  Deleting merge
passes saves time.  These are different currencies, so we set an
explicit exchange rate, called **λ (the price)**:

> total cost = compressed bits + λ × merge passes

Production uses λ = 0.1: one merge pass is worth a tenth of a bit.
Turn the price up and the optimizer flattens more aggressively (at
λ = 1 the "optimal" tree is nearly one giant flat block — it has
essentially reinvented fixed 8-bit coding).  Turn it to zero and you
get plain optimal length-limited Huffman back, which doubles as a
built-in correctness check.

A price, not a hard rule, is the right mechanism because the trade is
smooth: sweeping λ traces a frontier where you can *always* buy more
decode speed — the exchange rate just keeps worsening.  On real data
the sweet spot under our adoption rules sits at λ ≈ 0.10–0.14, and
being a little off costs almost nothing (the landscape near the
optimum is nearly flat — a fact that one of the three algorithms
exploits directly).

## 5. The rules of the game: the slot budget

One hard constraint governs which sets of code lengths are legal at
all, and it shapes all three algorithms, so it's worth two paragraphs.

Think of building the tree level by level from the top, tracking how
many **open slots** you have.  You start with one slot (the root).  At
each level, every open slot either gets *used* — it becomes a leaf, a
pair, or the root of a flat block — or it *splits* into two open slots
one level down.  That's the whole game.  A legal code is exactly one
where every slot eventually gets used and every symbol eventually gets
a slot (a "complete" tree — no wasted space, nothing left over).

This gives each level a **feasibility window**:

* You can't use *too many* slots at this level: the symbols you
  haven't placed yet still need room below, and slots left open double
  at every level going down — spend too much now and the leftovers
  can't fit.
* You can't use *too few* either: every slot you leave open splits in
  two, and each half must eventually receive at least one symbol.
  Leave too many slots open and there aren't enough symbols left to
  satisfy them all.

Every count vector inside these windows is a legal tree; everything
outside is not.  (Readers who know the term: this is the Kraft
equality, dressed as inventory management.)

## 6. Algorithm 1: the exact solver (~100 µs worst case)

The most expensive rung of the ladder provably finds *the* best legal
set of code lengths under the priced objective.  It rests on two
observations that turn an intimidating search into bookkeeping.

**Observation 1: given the shopping list, the assignment is trivial.**
Suppose you've already decided *which* chunks the tree will contain —
say "one block of 64 at length 7, one pair at length 5, one lone leaf
at length 2...".  Each chunk has a fixed per-occurrence cost (its code
length in bits, plus λ times the merge passes above it, plus kernel
costs — a number you can just compute).  Then the best way to fill the
chunks with actual symbols is exactly what you'd guess: sort symbols
by frequency, sort chunks by cost, and deal the most frequent symbols
into the cheapest chunks, in order.  Like boarding your most frequent
flyers into the best seats.  (This is the step the reviewers checked
hardest — it's an exchange argument: any other assignment can be
improved by swapping two symbols, so sorted-into-sorted is optimal.)

**Observation 2: the shopping list is a walk down the levels.**  So
the only real decision is which chunks to buy, and that decision
unfolds level by level under the slot budget of §5: at each level,
which block sizes do you take?  Note a lovely accident: since chunk
sizes at one level come from the binary representation of that level's
count, "at most one chunk of each size per level" is *automatically*
the only option — the wire format's rule and the optimizer's choice
space coincide perfectly.

Put together, that's a textbook **dynamic program**: process levels
top to bottom, and for every combination of (symbols placed so far,
slots currently open) remember the cheapest way to reach it.  Take a
block → advance the symbol counter, consume a slot; descend a level →
open slots double.  At the bottom, read off the cheapest complete
state.  The table is small (≤ 256 symbols × ≤ 256 open slots × 11
levels), every entry is filled once, and the answer is exact — no
heuristics anywhere.

Getting it from "exact but 6 milliseconds" (fine for experiments,
absurd for production) down to production speed was its own small
saga, all pure implementation: sweeping the table along diagonals that
stay resident in L1 cache, skipping states that can't exist for parity
reasons, and clipping levels to the band of slot counts that are
actually reachable.  Final numbers on M1: worst case (all 256 byte
values in use) **127 µs**, typical English-text alphabets ~8 µs,
skewed alphabets ~2.4 µs.  Verified exact against an independent
slower solver on tens of thousands of instances, structural validation
(budget exactly spent, symbol counts, cost re-derived from scratch)
included.

One piece of fine print that matters later: the "deal into cheapest
chunks first" argument needs chunk costs to sort *consistently* — a
chunk at a deeper level must never be cheaper than one at a shallower
level, or the level-by-level walk and the global cost order disagree.
With λ ≤ 1/7 that's guaranteed, and production's λ = 0.1 sits safely
inside.  Keep this in mind for the κ̂ section below.

## 7. Algorithm 2: the same solver, zoomed out (~10 µs)

Studying what the exact solver actually *does* to trees revealed two
things.  First, its moves are humble: 98.4 % of moved symbols shift by
exactly one level — but in coordinated runs of dozens, which is why
naive "search near the baseline" heuristics fail.  Second, and more
usefully: **the cost landscape is nearly degenerate** — huge numbers
of almost-equally-good trees sit near the optimum.  You don't need
*the* best tree; you need any member of a large club.

That licenses a beautifully lazy trick: **solve a smaller version of
the same problem.**  Bundle the frequency-sorted symbols into groups
of g = 2, 4, or 8 and treat each group as one atom.  A group of 4 is
just a miniature depth-2 flat block, so the coarse problem *is* the
original problem with a smaller alphabet and correspondingly fewer
levels — same solver, same code, quarter or sixteenth the states.  (If
the alphabet size doesn't divide evenly, pad with ghost symbols of
zero frequency.)

The auto rule picks the granularity by alphabet size — exact up to 64
symbols, groups of 2 up to 128, groups of 4 above — which keeps the
solve at ~10 µs or less for *every* alphabet.  Measured against exact
across all test windows: average cost gap 0.1–0.4 %, and the
end-to-end decode wins are statistically indistinguishable from the
exact solver's.  The degenerate landscape really is that forgiving.

## 8. Algorithm 3: the greedy nudger (~2 µs, no table at all)

The cheapest rung came from a challenge: forget solvers — can "dumb
code that rounds class counts to a few powers of two" capture a
meaningful slice of the win in essentially no time?

Answer: yes, about half of it.  The nudger is one single pass down the
levels, no memory of alternatives, using the same slot budget:

1. At each level, compute the feasibility window of §5 and clamp the
   baseline Huffman count into it.
2. Consider just **five candidates** for this level's count: the
   clamped baseline itself; the baseline rounded down to its top one
   or top two binary digits (e.g. 60 → 32 or 48); the next power of
   two *up* (60 → 64); and zero (kill the level entirely, pushing its
   symbols deeper).
3. Score each candidate as: the true cost of this level's resulting
   chunks, plus a quick estimate of the rest of the tree — namely
   "pretend every remaining level just takes its clamped baseline" and
   add that up.  This *rollout* matters: scoring with only a one-level
   lookahead was tried first and systematically walked into bad
   corners (it captured only 9–14 % of the available win; the rollout
   lifted that to 43–50 %).
4. Keep the cheapest candidate, update the slot ledger, move down.

One deliberate bit of dishonesty: the nudger scores its candidates
with the price inflated to 1.5λ.  Greedy passes systematically
*under*-flatten compared to the full solver, and the final accept/
reject decision (next section) is judged at the honest λ — so biasing
the search toward flattening costs nothing and recovers some of the
gap.  Total cost: about 2 µs per table, ~45× cheaper than the exact
solve.  It captures 43–50 % of the exact solver's realized
improvement.

That completes the ladder: **off → nudge (~2 µs) → auto DP (~10 µs) →
exact (~100 µs)** — pick a rung by how much encode-side time the
table-build budget can afford.

## 9. The guard: never ship a worse tree

Whatever the rung, the result is not adopted blindly.  For every
table, both trees — plain Huffman and the optimizer's proposal — are
priced under a cost model, and the proposal ships only if it's
modeled at least 10 % faster to decode **and** at most 1.5 % larger.
Otherwise the baseline ships, and the entire feature cost that window
a few microseconds and nothing else.

Two things about the guard turned out to matter more than expected.

First: for the exact solver the guard is *mathematically redundant* —
in-model, the priced optimum is never worse than the baseline, because
the baseline is itself one of the candidates.  Its real jobs are (a)
skipping wins too small to bother shipping, and (b) being a firewall
against the model being *wrong about reality*.

Second: a firewall built from the same blueprints as the thing it
guards has a blind spot.  If the cost model misprices a tree, the
guard — using the same model — mispraises it identically and waves it
through.  That is not hypothetical; it's exactly what happened on one
distribution, and it's what the entire follow-on branch is about
(§11).

## 10. Results: the ladder, measured honestly

Methodology in one breath: real workloads rebuild entropy tables every
few-to-hundred kilobytes, so we measure *end-to-end* — histogram +
table build + optimizer + encode kernels on one side, per-window table
reconstruction + decode kernels on the other — sweeping the table
period G from 4 KB to 128 KB, on LZ4 literal streams from the Silesia
corpus (the byte streams an LZ-front-ended entropy coder actually
sees), each of 12 files measured separately, geometric mean reported,
compression ratio *including* the 128-byte-per-window header.  PH
codec (no per-node FSE; see §11 for why that matters).

Against the pre-existing `main` (so these numbers include the
rank-range fast-table rebase this branch sits on — "off" shows that
share alone):

| G     | rung  | encode e2e | decode e2e | ratio vs old main |
|-------|-------|-----------|------------|--------------------|
| 4 K   | off   | +19.8 %   | +39.6 %    |  0.000 pp          |
|       | nudge | +9.5 %    | +95.4 %    | −0.258 pp (smaller!) |
|       | auto  | −31.2 %   | +133.2 %   | −0.266 pp          |
|       | exact | −75.6 %   | +134.0 %   | −0.406 pp          |
| 16 K  | off   | +6.9 %    | +26.8 %    |  0.000 pp          |
|       | nudge | +4.8 %    | +57.5 %    | +0.090 pp          |
|       | auto  | −19.5 %   | +82.9 %    | +0.214 pp          |
| 64 K  | off   | +1.0 %    | +6.2 %     |  0.000 pp          |
|       | nudge | +8.5 %    | +25.1 %    | +0.101 pp          |
|       | auto  | −1.1 %    | +50.4 %    | +0.202 pp          |
| 128 K | off   | +7.8 %    | +5.6 %     |  0.000 pp          |
|       | nudge | +15.4 %   | +26.1 %    | +0.102 pp          |
|       | auto  | +11.9 %   | +45.3 %    | +0.180 pp          |
|       | exact | −23.3 %   | +47.2 %    | +0.099 pp          |

How to read it:

* **The nudge rung is simply free money**: faster encode than the old
  main at every G (even 4 K), decode +25 % to +95 %, compression
  within ±0.1 pp — *better* at 4–8 K, where fewer chunk records also
  shrink the wire.
* **Auto doubles decode at small windows** (2.08 → 4.85 GB/s at 4 K)
  and pays for itself on encode at 128 K.
* **Exact is for encode-once-decode-many**: at 4 K it gives up ~76 %
  of encode speed to make decode 2.34× faster *with 0.4 pp better
  compression*.  Absolute decode with auto/exact at 64–128 K:
  11.0–11.3 GB/s on M4.

Yes, we also asked "is this too good to be true?"  See §12.

## 11. When the model lies: the follow-on branch (`joint-cost-model`)

On synthetic test distributions, one case blew up: **geometric** (a
smooth exponential fall-off in symbol frequencies).  The optimizer
shipped a tree 10.6 % smaller and **28 % slower to decode** — and the
guard approved it, because the guard priced it with the same model
that made the mistake.  Real corpora only ever showed miniature
versions of this (worst measured: −2.8 %, at noise scale), but a knob
whose contract is "makes decode faster" deserves a model that can't be
fooled by construction.  Running the mechanism down took three fixes —
and, satisfyingly, the prime suspect turned out to be innocent.

### Fix 1: kernel price tags (κ̂)

The original model priced flat-block kernels at zero.  Real kernels
aren't free, and — surprisingly — their cost isn't ordered by size:
depth-4 and depth-8 blocks decode cheapest per symbol, depth-7
dearest, all still several times cheaper than a merge pass.  The fix
lets every solver and the guard carry a measured per-depth cost table
(κ̂, in merge-pass units — fitted values on M1 range from 0.49 to
1.88 depending on depth).

The subtle part was keeping the exact solver exact.  Its optimality
argument (§6 fine print) needs chunk costs to sort consistently across
levels, and an external review had concluded that this required the
kernel table to be "monotone-ish" — a condition nothing guarantees a
real table satisfies.  The resolution: the solver's within-level sweep
order was
never actually forced — sweep each level's block sizes in *cost*
order instead of *size* order and the questionable condition
evaporates, leaving only a mild requirement ("the per-depth cost
corrections must stay smaller than about one bit's worth"), which
every plausible kernel table satisfies at production λ.  Verified by
brute-force cross-checking two independent solvers on 16,000
instances, half with adversarially random non-monotone tables.

And the punchline: with real numbers in hand, κ̂ turned out **not** to
be the geometric culprit — the kernels are too cheap to explain −28 %.
The mechanism ships (honest infrastructure, settable per platform);
the witness pointed elsewhere.

### Fix 2: count the actual work (the simulator, and γ̂)

The guard's time model was a smooth formula.  It's now a simulator:
it reconstructs the exact skeleton of the schedule the decoder will
run — how many merges of each kind, how many flat blocks, how many
records — and prices those.  Along the way, per-record fixed costs
became measurable: each schedule record costs about **7.75 ns per
block** merely to exist (dispatch, wire header), independent of how
many symbols it covers.

That constant (γ̂, ~170 merge-pass-equivalents) was fed back into all
three algorithms as a per-chunk surcharge — "every chunk you add costs
this much per block, on top of everything else".  It's the kind of
correction that sounds too small to matter and then doesn't: at 4 K
windows (where a table serves few blocks, so fixed costs loom large)
it lifted the nudge rung's decode win from +38 % to +48 % and the
exact rung's from +58 % to +65 % — *while improving compression*,
because fewer records is also fewer wire bytes.  Fewer, bigger pieces:
a cost the decoder and the file format both charge for, now priced.

### Fix 3: the FSE tax (the actual culprit)

The real geometric story turned out to be an interaction with an
optional feature.  The codec has a mode (PHA) where each merge node's
bitmap may additionally be entropy-coded with FSE (a tANS entropy
coder) when the bitmap is skewed enough to shrink.  Two facts
collided:

* The optimizer's favorite trick — collapse structure into flats —
  leaves *fewer, larger* merges whose bitmaps are *more skewed*
  (that's precisely the structure it couldn't flatten).  Skewed is
  exactly what FSE likes to commit on.
* An FSE-coded bitmap decodes about **4× slower per element** than a
  raw one.

So on geometric data the optimizer built a tree whose remaining
bitmaps all got FSE'd: genuinely smaller on the wire (−10.6 %), much
slower to walk (−28 %) — and invisible to a model that assumed every
bitmap decodes at raw speed.  With FSE off, the same joint tree is
strictly *better* than baseline (+6.6 % decode, −14 % size); the tree
was never the problem.

The fix: the guard's time model now predicts which bitmaps the coder
will commit (mirroring the coder's own size rule) and taxes each one
at τ = 4 passes per element.  Result: geometric flips to *rejected*
(−0.2 % instead of a shipped −28 %), a second synthetic case likewise,
every legitimate win keeps its adoption (including a +920 % and a
+716 % on skewed synthetics), and the realistic-workload numbers are
unchanged.  The tax is inert in PH mode, where no bitmap is ever FSE'd.

### Sidebar: two measurement traps

Both fixes above were nearly derailed by benchmarking artifacts worth
recording.  Repeating one block through a decoder lets the branch
predictor *memorize the walk* — a tree pair that measured +6 % under
single-block repetition measured −27 % with fresh data per block.  And
an output buffer that fits in L1 cache hides the cost of streaming
stores.  Fresh multi-block data, streamed outputs, and ≥10 ms timed
regions (the timer quantizes near 256 µs) are all load-bearing.

## 12. Did we just overfit?  (the Goodhart checks)

Every constant above — the price, the nudger's candidate list and 1.5λ
bias, the auto-granularity schedule, the guard thresholds — was tuned
while staring at Silesia literals.  Three checks that we didn't simply
memorize the benchmark:

* **Frozen re-run on a different corpus** (Calgary, 18 files, smaller
  and noisier): the ladder holds — nudge +15–20 %, auto +37–40 %,
  exact +39–41 % decode at every G, and compression *improves at every
  rung and every G* (smaller windows mean raggeder baseline trees,
  so the record savings dominate).
* **Raw, non-literal files** (whole Silesia + Calgary members —
  skewed, structured, "wrong" data for this workload): reduced but
  uniformly positive magnitudes; worst single observation anywhere is
  a −2.8 % decode at noise scale.  Notably `pic` — the real-world file
  closest to the geometric shape — *gains* 4–12 %, because real
  windows have a dominant-symbol-plus-ragged-tail shape the model
  prices fine; the smooth synthetic pathology doesn't occur in the
  wild.
* **The exact rung is structurally safe**: it optimizes a true
  objective under a guard, not a benchmark score; its failure modes
  live in the cost model, which is exactly what §11 hardened.

## 13. Where everything lives

* **`joint-flat-lengths`** — the minimal, PR-shaped version: the three
  solvers, the guard, the build hooks.  Ten code-only commits.
* **`joint-flat-lengths-plus-fast-tables`** — the same work rebased
  onto the rank-range codec-table branch (~1.5 KB tables, sub-µs
  decode-table rebuilds), which is what all end-to-end numbers above
  are measured on.  Bonus of the marriage: the table build hands its
  already-sorted symbol array to the optimizer, so the nudge rung's
  total encode-side increment is ~1.8 µs per window.
* **`joint-cost-model`** — κ̂ kernel tables + the reordered-sweep
  exactness proof, the kind-aware guard simulator, γ̂ per-record
  costs, and the FSE tax.
* Knobs: `pivco_huffman_set_joint_lambda` (the price, default 0.1),
  `..._set_joint_granularity` (−1 nudge / 0 auto / 1 exact / 2, 4, 8
  fixed), `..._set_joint_guard` (defaults: adopt at ≤ 0.90× modeled
  time, ≤ 1.015× bits), and on the cost-model branch
  `..._set_joint_kappa`, `..._set_joint_gamma`, `..._set_joint_fse_tax`,
  `..._set_joint_merge_costs`.

### The data, precisely

"Silesia literals" = all 12 Silesia-corpus files preprocessed by
LZ4HC level 9, literal runs concatenated per file (32.9 MB total) —
the stream an LZ front end hands its entropy coder.  Files are
windowed independently at each G; speeds are unweighted geometric
means across files (per-file best-of-repeats), ratios unweighted
means, always including the per-window header.  Raw outputs and
per-file tables live under `results/` with dates and commit hashes.

### Open threads

Per-architecture κ̂/γ̂/τ fits (current constants are Apple-M-class;
the mechanisms are knobs precisely so other platforms can be measured
in), a defaults strategy for shipping measured κ̂ tables, a λ-aware
FSE commit rule (the encoder controls both the tree and the commit
decisions — taxing commits at choice time, not just guard time, turns
the PHA interaction from a hazard into another knob), and amortizing
the solve across windows by re-running only on histogram drift.

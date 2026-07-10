# joint-flat-lengths — M4 A/B (2026-07-10)

Joint code-length / flat-shape optimization (docs/JOINT-LENGTHS.md):
an exact 0/1-knapsack DP over (symbols placed, Kraft mass) chooses the
length histogram AND its induced flat structure together, at price
lambda bits per merge pass, with an adoption guard (take DP lengths
only when modeled passes drop >=10% at <=+1.5% modeled bits).
Encoder-side only — any decoder reads the output.  bench 20 --all,
best-of-2 interleaved, lambda=0.1 vs same-binary baseline.

## Headline: decode geomean +27.8 %, compressed size geomean −0.36 %

| dist         | decode   | size    |
|--------------|----------|---------|
| image_jpeg   | +883.7 % | +1.05 % |  <- tree goes fully flat (D=8 ~ raw bytes)
| bell_s80     | +730.8 % | +0.46 % |  <- same
| json_api     | +75.5 %  | +0.22 % |
| log_apache   | +65.5 %  | +0.34 % |
| bell_s30     | +54.8 %  | −0.65 % |
| prose_pride  | +50.4 %  | +0.12 % |
| chinese_text | +49.4 %  | −1.14 % |
| source_c     | +48.7 %  | +0.61 % |
| proba14/02   | +30/28 % | +0.5/0 % |
| html_wiki    | +9.9 %   | −1.40 % |

Several rows get FASTER decode and SMALLER output simultaneously —
partly fewer per-node wire records under flatter trees, partly the DP
being an exact length-limiter where production's limit_code_lengths
is a heuristic (geometric: −10.6 % size available!).

## Known issues
- geometric −26.4 % decode (at −10.6 % size): the guard adopted a
  modeled-pass win that doesn't realize — the model weighs all merge
  passes equally, but the production shape's deep HALF chains use the
  cheap cst_vec kernels.  Next step: per-merge-type pass weights in
  the DP cost (cst vs vec) and/or FSE-ability of the resulting
  bitmaps.
- Rows at exactly +0.00 % size with small negative decode deltas are
  byte-identical streams (guard kept production) = run noise.
- Assignment variant A vs B and per-D kernel costs kappa_D not yet
  modeled; docs/JOINT-LENGTHS.md §6 lists the open questions for
  review.

Raw captures: m4-*-joint-{l01,base}.txt.  Correctness: distorted
streams roundtrip through decode-side table rebuild + scalar + NEON
(test_joint_lengths); full suite + strict GuardMalloc green.

## Encode side (same M4 session, purpose-built A/B: table build + block encode)

| dist         | encode   | table build 0 -> J |
|--------------|----------|--------------------|
| image_jpeg   | +228.4 % | 3 us -> 7.8 ms     |
| json_api     | +72.8 %  | 3 us -> 3.3 ms     |
| chinese_text | +45.6 %  | 5 us -> 5.1 ms     |
| prose_pride  | +35.9 %  | 3 us -> 3.4 ms     |
| html_wiki    | +15.2 %  | 4 us -> 7.4 ms     |
| proba80      | +12.8 %  | 2 us -> 96 us      |
| english/dna  | ~0 %     | 1 us -> 0.8/1.2 ms (guard kept production; DP cost still paid) |

Encode mirrors decode (partitions instead of merges; flat pack instead
of flat unpack) — no encode-throughput regressions.  THE real cost of
the feature is table-build latency: the DP is O(items x sigma x 2^L)
~= 40 M cells at sigma=256, i.e. 1-8 ms per build vs ~4 us baseline.
Fine amortized over block streams and large files; dominant for small
files.  Mitigations, in order of effort: (a) gate the DP on input size
(e.g. only for >= a few MB); (b) drop the 16 B/state choice masks
(2/3 of memory traffic) via divide-and-conquer backtracking; (c) the
hierarchical/dyadic-mass DP — in slot units at level L the ledger is
bounded by remaining symbols, shrinking states ~100x, but it conflicts
with the global-cost-order argument that makes the current DP exact;
open question alongside doc §6.

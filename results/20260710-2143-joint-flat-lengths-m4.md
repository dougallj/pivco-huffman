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

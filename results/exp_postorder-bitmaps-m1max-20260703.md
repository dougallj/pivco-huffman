# Experiment: postorder-bitmaps (wire v0.6) — Apple M1 Max, 2026-07-03

Branch `postorder-bitmaps` off main @ a3464c6 (created as
`postorder-kleaf`, renamed 2026-07-03 — see design note).  Post-order
records, pre-order splits: each non-flat internal node's [marker][bitmap]
record is emitted AFTER its children's regions, exactly where the BU merge
consumes it, so the decoder reads the compressed stream strictly forward
and touches every byte once.  The K_right split survives (one LEB128
varint per recursion site, consumed at node entry to size the children) —
so no popcount is reintroduced.

Design note: this is an Euler-tour hybrid, not pure post-order — the
split varints sit at pre-order positions, because strictly-forward
parsing of variable-size regions requires sizing information in prefix
position.  The genuinely post-order alternative (the original "kleaf"
proposal) makes subtrees self-delimiting via a length at every terminal:
more slots than one-per-recursion-site, plus explicit zero-lengths for
symbols absent from a block, but zero downward information flow in the
decoder.  The varint replaces v0.5's fixed u16, which is where the small
ratio win comes from.

Motivation: v0.5's pre-order layout loads each bitmap only at merge time,
long after the cursor passed it — a backwards touch that keeps the whole
compressed block cache-resident.  Post-order makes the stream a moving
window, which should matter more at larger block sizes and on x86 L1s
(32-48 KB vs Apple's 128 KB).  Decode-side stack also shrinks: the
bm_scratch VLA (FSE path) no longer lives across the subtree recursion.

## Setup

- Host: Apple M1 Max (local laptop — NOT M4; not comparable to README/EC2
  numbers), DVFS warmup run before the measured sweep.
- `pivco_huffman_bench 50 --all`, block 16384, freq drift ≤0.7%.
- Baseline: same-day main sweep (shared with the popcount-lengths
  experiment).  Raw outputs:
  `exp_postorder-bitmaps-m1max-20260703-{main-baseline,branch}.txt`
- popcount-lengths (pcl) columns below are from `exp_popcount-lengths-m1max-20260702.md`
  (headers dropped entirely, decoder popcounts).

## Headline (vs main v0.5)

- **NEON BU decode: mean +0.96%, median +0.85%** — post-order is FASTER
  than the header-assisted pre-order baseline on M1 Max.  Best:
  proba02 +7.4%, source_c +3.6%, proba50 +2.8%.  Flat-heavy dists ~+0.8%.
  Only two_sym_eq (−2.2%, root fast path — identical code+stream, i.e.
  noise on a 34 GB/s dist) and a few ±1% noise entries dip.
- **Ratio: −0.020% total** (varint vs u16 on the same slot set; the
  full-header removal of popcount-lengths was −0.163%).
- Keeps v0.5's no-popcount property, so unlike popcount-lengths it does NOT risk the
  historical x86 popcount tax (K_right headers were worth +22–57% decode
  on c8i/c8a/c4/c8g when introduced, +0% on M4).  Expectation for EC2:
  v0.6 ≥ v0.5 everywhere, with the layout win growing with block size.

## Per-distribution (compressed bytes for 4M input; NEON BU decode M/s)

| dist          | size v0.5 | size v0.6 | Δsize   | (pcl Δ) | BU v0.5 | BU v0.6 | Δspeed | (pcl Δ) |
|---------------|-----------|-----------|---------|----------|---------|---------|--------|----------|
| proba80       |   484,753 |   484,432 | −0.07%  | −0.42%   |    4671 |    4632 |  −0.8% |  −1.0%   |
| proba50       | 1,057,537 | 1,056,647 | −0.08%  | −0.48%   |    9849 |   10129 |  +2.8% |  −6.2%   |
| proba14       | 2,220,081 | 2,219,326 | −0.03%  | −0.32%   |    5160 |    5107 |  −1.0% |  −5.8%   |
| proba02       | 3,745,119 | 3,743,855 | −0.03%  | −0.23%   |    4226 |    4537 |  +7.4% |  +1.3%   |
| bell_s10      | 2,984,714 | 2,983,434 | −0.04%  | −0.17%   |    5133 |    5164 |  +0.6% |  −2.2%   |
| bell_s30      | 3,731,184 | 3,730,416 | −0.02%  | −0.19%   |    4525 |    4532 |  +0.2% |  −3.9%   |
| bell_s80      | 4,175,454 | 4,175,454 |  0.00%  | −0.04%   |    6432 |    6500 |  +1.1% |  −0.1%   |
| uniform       | 4,194,816 | 4,194,816 |  0.00%  |  0.00%   |   10498 |   10440 |  −0.6% |  −1.0%   |
| english       | 2,236,347 | 2,236,086 | −0.01%  | −0.18%   |    6514 |    6600 |  +1.3% |  −3.5%   |
| zipfian       | 3,292,424 | 3,291,656 | −0.02%  | −0.20%   |    4481 |    4595 |  +2.5% |  −2.5%   |
| sparse_4      | 1,049,088 | 1,049,088 |  0.00%  |  0.00%   |   34476 |   34239 |  −0.7% |  −1.0%   |
| sparse_16     | 2,097,664 | 2,097,664 |  0.00%  |  0.00%   |   33869 |   34295 |  +1.3% |  +1.6%   |
| geometric     | 1,230,187 | 1,229,931 | −0.02%  | −0.37%   |    4887 |    4954 |  +1.4% |  −1.0%   |
| two_sym_eq    |   525,056 |   525,056 |  0.00%  |  0.00%   |   34784 |   34012 |  −2.2% |  −0.0%   |
| two_sym_90/10 |   252,357 |   252,357 |  0.00%  |  0.00%   |    7817 |    7738 |  −1.0% |  +0.4%   |
| flat_M3       | 1,573,376 | 1,573,376 |  0.00%  |  0.00%   |   24239 |   24428 |  +0.8% |  +0.8%   |
| flat_M5       | 2,621,952 | 2,621,952 |  0.00%  |  0.00%   |   24080 |   24292 |  +0.9% |  +2.2%   |
| flat_M6       | 3,146,240 | 3,146,240 |  0.00%  |  0.00%   |   20329 |   20500 |  +0.8% |  +1.4%   |
| flat_M7       | 3,670,528 | 3,670,528 |  0.00%  |  0.00%   |   12928 |   13025 |  +0.8% |  +2.2%   |
| html_wiki     | 2,961,365 | 2,960,199 | −0.04%  | −0.29%   |    3968 |    4027 |  +1.5% |  −6.2%   |
| prose_pride   | 2,426,919 | 2,426,257 | −0.03%  | −0.30%   |    4665 |    4705 |  +0.9% |  −6.9%   |
| image_jpeg    | 4,151,227 | 4,150,881 | −0.01%  | −0.10%   |    5304 |    5338 |  +0.6% |  −2.4%   |
| json_api      | 2,764,881 | 2,763,929 | −0.03%  | −0.31%   |    4323 |    4359 |  +0.8% |  −6.4%   |
| source_c      | 2,632,737 | 2,631,210 | −0.06%  | −0.35%   |    4587 |    4750 |  +3.6% |  −4.8%   |
| log_apache    | 2,918,802 | 2,917,804 | −0.03%  | −0.30%   |    4335 |    4403 |  +1.6% |  −6.4%   |
| dna_fasta     | 1,103,411 | 1,103,119 | −0.03%  | −0.19%   |    5640 |    5700 |  +1.1% |  −0.7%   |
| csv_numeric   | 1,681,359 | 1,680,258 | −0.07%  | −0.35%   |    3588 |    3600 |  +0.3% |  −3.8%   |
| gzip_random   | 4,194,816 | 4,194,816 |  0.00%  |  0.00%   |   10378 |   10426 |  +0.5% |  +0.6%   |
| chinese_text  | 3,137,169 | 3,136,924 | −0.01%  | −0.23%   |    4191 |    4229 |  +0.9% |  −4.7%   |
| calgary_pic   |   591,079 |   590,198 | −0.15%  | −0.78%   |    3684 |    3741 |  +1.5% |  −3.8%   |

Totals: v0.5 72,852,642 → v0.6 72,837,909 (**−0.020%**); popcount-lengths was
72,734,010 (−0.163%).

## Not measured / next

- Encode speed: the encoder now stages each node's bitmap on the stack
  and memcpys it into the stream after the children (its final position
  depends on FSE-variable child sizes) — an extra ≤N/8-byte copy per
  non-flat node.  bench_main is decode-only; run pivco_fair_bench for
  encode columns.
- x86 / AVX-512 / Graviton (EC2): the interesting hosts — v0.6 keeps the
  popcount elimination that motivated v0.5's headers, so the layout win
  should stack on top rather than trade against it.
- Block-size scaling: the L1 working-set argument predicts the v0.6 gap
  widens at 32K+ blocks; measure alongside the scratch-optimisation
  branch (in-place merge + page-hazard carving) whose RSS reductions it
  should compound with.

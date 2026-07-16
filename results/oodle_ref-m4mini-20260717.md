# Oodle 2.9.3 reference numbers (Selkie / Mermaid / Kraken), M4 mini

`extras/bench/bench_oodle_ref.c` against the Oodle 2.9.3 SDK
(liboo2coremac64.a arm64 slice, staged at ~/oodle-2.9.3 on the mini;
build line in the source header).  Same methodology as bench_lz4pv /
bench_lz4_enc: same 12 silesia files, >=100ms DVFS warm loop,
best-of-reps (enc 3, dec 8), single-threaded, fuzz-safe decode with
PREALLOCATED decoder memory (malloc-free timing, like our benches).
Raw per-file output: `oodle_ref-m4mini-20260717.txt`.

Geomeans/12:

| config             | ratio | enc MB/s | dec MB/s |
|--------------------|-------|----------|----------|
| Selkie HyperFast1  | 2.233 | 578      | 15021*   |
| Selkie VeryFast    | 2.320 | 485      | 13513*   |
| Selkie Normal      | 2.699 | 129      | 11779    |
| Selkie Optimal2    | 3.059 | 7        | 9652     |
| Mermaid HyperFast1 | 2.846 | 491      | 5033     |
| Mermaid VeryFast   | 2.995 | 421      | 4886     |
| Mermaid Normal     | 3.274 | 117      | 5679     |
| Mermaid Optimal2   | 3.721 | 5        | 5223     |
| Kraken HyperFast1  | 3.038 | 544      | 2783     |
| Kraken VeryFast    | 3.354 | 334      | 2961     |
| Kraken Normal      | 3.702 | 84       | 3165     |
| Kraken Optimal2    | 4.197 | 6        | 3262     |

\* Selkie HyperFast/VeryFast store sao + x-ray essentially raw
(ratio 1.00, "decode" 47-75 GB/s = memcpy), inflating those geomeans;
Selkie Normal's worst real rows are ~8-12 GB/s.

## Like-for-like vs our codecs (same files/mini/methodology)

| codec                    | ratio | enc MB/s | dec MB/s |
|--------------------------|-------|----------|----------|
| lz4pv HC-1 FASTEST       | 2.826 | 398      | 5278     |
| **Mermaid HyperFast1**   | 2.846 | 491      | 5033     |
| lz4pv HC-4 FASTEST       | 3.122 | 128      | 5895     |
| **Mermaid Normal**       | 3.274 | 117      | 5679     |
| lz4pv HC-12 FASTEST      | 3.219 | 25       | 6154     |
| **Mermaid Optimal2**     | 3.721 | 5        | 5223     |
| zstd-pivcoh L1           | 2.893 | 1457     | 2824     |
| **Kraken HyperFast1**    | 3.038 | 544      | 2783     |
| plain LZ4HC-9            | 2.729 | 59       | 5958     |
| **Selkie Normal**        | 2.699 | 129      | 11779    |

Readings:

* **lz4pv HC-1 is Mermaid-HyperFast1-shaped**: ratio within 0.7%,
  decode +5% for us, encode −19%.  A bolt-on entropy stage over stock
  LZ4HC lands on RAD's mid codec at its fast-encode point.
* At matched encode effort in the mid band (lz4pv HC-4 vs Mermaid
  Normal, ~120 MB/s both): Mermaid leads ratio +4.9%, we lead decode
  +3.8%.
* At the high-effort end Mermaid pulls away on ratio (Optimal2 3.721
  vs our HC-12 3.219, +15.6%) — a real optimal parse + rep-match
  machinery vs our stock HC parse; we keep +18% decode.  The parse,
  not the entropy stage, is lz4pv's ratio ceiling.
* **Selkie is the decode king** (9.7-15 GB/s): pure-LZ, no entropy
  passes.  It needs Optimal2 (7 MB/s enc) to reach ratio 3.06; lz4pv
  HC-4 gets 3.12 at 128 MB/s enc — but decodes at half Selkie's speed.
  Selkie Normal strictly dominates plain LZ4HC-9 (same ratio, 2x
  encode, 2x decode).
* **Kraken HyperFast1 vs zstd-pivcoh L1**: ratio +5% Kraken, decode
  parity (2783 vs 2824), encode 2.7x zstd-pivcoh.  At its Normal/
  Optimal levels Kraken's ratio-decode frontier (3.7-4.2 @ ~3.2 GB/s)
  is beyond both our stacks — that's the two-stage
  Huffman-with-fastpath + optimal parse combination.
* Encode-cost context: Oodle Optimal2 runs 5-7 MB/s on this corpus —
  the reference frontier is priced in minutes-per-GB territory, where
  our joint solve + parse costs are noise.

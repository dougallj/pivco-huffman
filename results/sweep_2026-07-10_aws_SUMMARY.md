# AWS sweep 2026-07-10 — rank-range-codec branch vs main (0a378fb)

Hosts: c8i.large (Granite Rapids, AVX-512), m7a.large (Zen 4), c8g.large
(Graviton 4).  Ubuntu 24.04, gcc 13.3, spot, taskset -c 1, per ~/AWS.md.
Branch tip: 46cf182 (schedule codec + decode_table_t + SIMD histo/sort +
bounds checks).  Raw logs: sweep_2026-07-10_*_aws_*.txt (r1 = tests/fuzz/
build-time/decode, r2 = wire cross-check/table-lifetime/interleaved decode).

## Correctness
- Tests pass on all hosts, both trees.  Fuzzer: 100K corrupted streams per
  host, guards intact (bounds checks working on x86 + arm/gcc).
- Wire byte-identical old<->new and old decodes new's files, all hosts,
  3 Silesia literal files (r1's "DIFFER" was a harness bug: missing files).

## Decode-side table setup (bench_build_time, median ns/build)
             c8i: main -> new (speedup)   m7a              c8g
  skew6        466 -> 152  (3.07x)      883 -> 156 (5.66x)  760 -> 191 (3.98x)
  text90      1017 -> 302  (3.36x)     1308 -> 283 (4.62x) 1200 -> 346 (3.47x)
  zipf256     1601 -> 371  (4.32x)     1664 -> 376 (4.43x) 1870 -> 502 (3.72x)
  uniform256  1412 -> 256  (5.53x)     1248 -> 269 (4.64x) 1748 -> 399 (4.38x)
  SIMD histo/sort vs portable: SSE2 +13..61%, NEON(G4) +33..50% -> ifdef justified.

## Table-lifetime (LZ4HC Silesia literals, decode incl. per-segment build)
  geomean end-to-end speedup (dickens/osdb/xml):
             G=5K   10K    20K    50K    100K
  c8i        1.45x  1.38x  1.24x  1.11x  1.02x
  m7a        1.63x  1.51x  1.31x  1.10x  1.02x
  c8g        1.35x  1.25x  1.15x  1.05x  1.04x

## Decode bandwidth (pivco_bu, best of 3 interleaved rounds)
  c8g: +0.3% geomean (neutral, matches M4/clang +1.0%).
  c8i: -3.4% geomean, m7a: -3.8% — consistent across dists (english -4.1/-4.5%,
  json_api -5.8%, prose_pride -5.3/-6.7%).  OPEN ITEM: x86+gcc-13 only.
  TRIAGED (c8i, sweep_..._c8i_triage.txt): NOT a gcc-13 artifact
  (gcc-14 -3.6%, clang-18 -1.6%), and no single culprit — it accumulates
  across the codec restructures (schedule walk -1.4%, decode-table split
  another -0.9%, bounds checks ~0, tip -3.0% under gcc-13).  A cursor-
  by-value/return experiment did not recover it (gcc -3.1%, clang -1.1%).
  Per-dist shape: flat on narrow alphabets (dna 0%, proba80 -1%),
  -5%ish on wide ones (json_api, prose_pride) — consistent with per-node
  walk overhead being RELATIVELY costlier on x86, where AVX-512 merges
  are much faster per symbol than NEON's (ARM hosts are neutral).
  Verdict: accepted for now — the table-lifetime regime (the actual
  target) nets 1.45-1.63x on these same hosts INCLUDING this effect;
  the ~2-3% only bites unbounded-lifetime single-table streams.
  Recovery ideas if it matters later: fuse the per-record kr/marker/
  bitmap bounds checks into one span check, flatten dispatch on x86,
  inline leaf-adjacent child handling into the parent node.

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
  Candidates: gcc-13 hot-loop layout sensitivity (see ~/AWS.md caveat —
  gcc-13 deltas of 6-12% measured elsewhere that vanish under gcc-15),
  the schedule-walk restructure's codegen on the x86 backends, or the
  bounds-check commit (c2f3455, never x86-perf-validated in isolation).
  Needs a gcc-15/clang re-run and a per-commit bisect on one x86 host.

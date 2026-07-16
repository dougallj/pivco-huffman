# zstd with pivcoh literals: full-codec before/after

pivcoh.h (wire v4.1) integrated into zstd v1.5.7 (f8745da) as the
literals entropy stage, replacing HUF — compressed literals become
[pivcoh lens wire][pivcoh blocks <= 32767 syms], treeless literals
reuse the previous pivcoh table (coverage-checked, estimate-compared);
raw/RLE literals, all sequence coding, matching, and framing untouched.
Effort maps from zstd strategy: < greedy -> SIMPLEST, >= greedy ->
BALANCED (joint coarse tier).  The pivcoh table (956 B) lives inside
the existing HUF entropy regions with a magic guard; frames are
NON-STANDARD (only this build decodes them).  Integration:
`extras/zstd_pivcoh_literals.patch` on a clone at `ext/zstd-pivcoh`
(untracked; `cp extras/pivcoh.h lib/common/` + apply patch + build
with `CPPFLAGS=-DZSTD_PIVCOH_LITERALS`).  Dictionaries and
targetCBlockSize are out of scope.

`zstd -b` on silesia.tar (212 MB), M4 mini, i4 (i2 at 19):

| lvl | ratio v -> p        | comp MB/s v -> p   | **decomp MB/s v -> p**   |
|-----|---------------------|--------------------|--------------------------|
| 1   | 2.892 -> 2.886 (+0.18%) | 1623 -> 1580 (-2.7%) | 2003 -> **2415 (+20.6%)** |
| 2   | 3.053 -> 3.048 (+0.14%) | 1224 -> 1191 (-2.7%) | 1844 -> **2060 (+11.7%)** |
| 3   | 3.200 -> 3.195 (+0.16%) | 963 -> 931 (-3.4%)   | 1789 -> **1925 (+7.6%)**  |
| 6   | 3.453 -> 3.444 (+0.25%) | 355 -> 342 (-3.7%)   | 1886 -> **2018 (+7.0%)**  |
| 9   | 3.582 -> 3.574 (+0.23%) | 224 -> 218 (-2.8%)   | 1979 -> **2128 (+7.5%)**  |
| 12  | 3.652 -> 3.643 (+0.23%) | 108 -> 106 (-1.8%)   | 2057 -> **2220 (+7.9%)**  |
| 19  | 4.010 -> 4.003 (+0.17%) | 8.78 -> 8.72 (-0.7%) | 1781 -> **1861 (+4.5%)**  |

(size deltas are compressed-size increases; roundtrip verified by the
bench itself plus explicit compress/decompress/cmp at levels 1/3/19,
including mozilla's split-litBuffer paths.)

Reading:

* The entropy-stage numbers translate: literals decode is the largest
  single share of zstd decompression at low levels, and swapping huf0
  (~1.4 GB/s) for pivcoh (~7-9 GB/s at this cadence) moves the WHOLE
  codec +21% at level 1, settling to +7-8% at mid levels (fewer
  literals) and +4.5% at 19.
* Size costs +0.14-0.25% — the entropy h2h gap (~0.8% of literals)
  diluted by literals' share of the stream; dropping K_rights
  (measured at 0.2-0.4% of literal bytes) would cover roughly half
  of it.
* Compression pays -0.7..-3.7%: histogram + table build + joint pass
  per fresh table, plus pivcoh encode ~0.8x huf0 at BALANCED.  The
  repeat path (estimate-compared, coverage-checked) keeps it modest;
  SIMPLEST at levels 1-3 keeps the fast path fast.
* zstd's optimal-parser literal-cost estimates still model HUF; at 19
  the parser slightly misprices literals for the actual backend —
  untouched, and evidently benign.

## Addendum: levels 1-3 with BALANCED everywhere (-DZSTD_PIVCOH_ALL_BALANCED)

|     | vanilla            | pivcoh SIMPLEST      | pivcoh ALL_BALANCED     |
|-----|--------------------|----------------------|-------------------------|
| 1   | 2.892 / 1623 / 2003 | 2.886 / 1580 / 2415 | 2.882 / **1620** / **2594** |
| 2   | 3.053 / 1224 / 1844 | 3.048 / 1191 / 2060 | 3.045 / **1225** / **2171** |
| 3   | 3.200 /  963 / 1789 | 3.195 /  931 / 1925 | 3.191 /  946  / **2011** |

BALANCED at the fast levels decompresses another +5-7% over SIMPLEST
(+29.5% / +17.7% / +12.4% over vanilla) and — the twist — COMPRESSES
faster than the SIMPLEST build, back to vanilla parity at levels 1-2:

RE-VERIFIED 2026-07-17 (methodology note: `zstd -b` self-warms — its
timed windows sit inside seconds of continuous load, so the DVFS
concern that bit bench_lz4pv does not apply here).  Alternating
vanilla/pivco -b -i4 pairs: L1 +26.0/+27.8/+28.4%, L2 +16.5/+17.4%,
L3 +12.3/+13.4% — L2/L3 match the table; L1's +29.5% was the cool end
of the band.  ABSOLUTE MB/s on the mini sag ~4% as the chassis warms
under sustained benching (both builds together: vanilla dec ran
1906-2004, pivco-bal 2448-2494 across one session), so cite the L1
headline as ~+27% (2.45-2.6 GB/s vs 1.9-2.0), not the 2594 point.
the joint solve (~2-4 us/table) is repaid by pivcoh's flatter trees
encoding faster (fewer partition passes), the same
whole-file-compress effect measured back on the frame API.  Cost:
size +0.33/+0.26/+0.26 % vs vanilla (about +0.1-0.15 pp over
SIMPLEST).  ALL_BALANCED looks like the better default unless the
extra ~0.1 % size matters.

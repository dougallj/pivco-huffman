# Verification: postorder-bitmaps (wire v0.6) vs main @ a3464c6 — 4 hosts, 2026-07-06

Independent A/B verification of the `postorder-bitmaps` branch against
its parent `a3464c6` on main — an A/B of the post-order-records wire
change.  (Run before the squash, as two commits of which the second was
rename/doc-only, and before the v0.7→v0.6 renumbering: the raw captures
and older write-ups may carry the old v0.7 label for the same wire
format.)

## Setup

- Hosts: **Apple M1 Max** (local, macOS/AppleClang), **Apple M4** (Mac
  mini over LAN, macOS/AppleClang, thermally constrained — see caveats),
  **Sapphire Rapids** (EC2 c7i.large spot, Xeon Platinum 8488C, Ubuntu
  24.04 / gcc 13.3, pinned `taskset -c 1`), **Zen 4** (EC2 m7a.large
  spot, EPYC 9R14, same OS/compiler/pinning).
- `pivco_huffman_bench 20 --all` (29 distributions), 3 rounds per build,
  **interleaved** head/base, per-cell best-of-rounds.  Blocks: 16384
  (Apple), 32768 (x86), identical within each host pair.
- Tests (`pivco_huffman_tests`) pass on all four hosts, both builds.
- Raw captures: `exp_postorder-bitmaps-verify-{m1max,m4,c7i,m7a}-20260706-{head,base}-r{1,2,3}.txt`
  plus `...-c7i-clang-...` for the compiler cross-check.

## Headline

| host | ΔBU geomean | ΔN geomean | Δscalar geomean | Δsize total |
|---|---:|---:|---:|---:|
| M1 Max (clang) | +0.02% | +0.20% | −0.02% | −0.020% |
| M4 (clang) | **+1.12%** | **+1.35%** | +0.06% | −0.020% |
| SPR c7i (gcc 13.3) | **+1.09%** | **+1.05%** | −8.97% † | −0.005% |
| Zen 4 m7a (gcc 13.3) | **+0.82%** | **+0.92%** | −13.56% † | −0.005% |
| SPR c7i (clang 18, clean rounds) | +0.78% | +0.79% | +2.63% † | — |

The branch write-up's prediction (`exp_postorder-bitmaps-m1max-20260703.md`:
"Expectation for EC2: v0.6 ≥ v0.5 everywhere") **verifies**: SIMD decode
is ~+1% geomean on M4/SPR/Zen 4, ~flat on M1 Max, and compressed size is
smaller in total on every host (worst single dist: +66 bytes english,
+54 bytes chinese_text at block 32768 — noise-level slot rounding).

† **The scalar geomean swings are the known binary lottery, not a real
regression.**  Under gcc, pivco_s drops 20–37% on the whole flat-tree
family (uniform, sparse_4/16, gzip_random, flat_M3–M7) on BOTH x86
hosts — but those dists produce **byte-identical streams** (Δbytes = +0)
and the diff never touches `primitives_scalar.h`; the two hosts run the
same gcc binary, so it's one layout roll observed twice.  Rebuilding
both trees with clang 18 on the same c7i host flips it: scalar geomean
+2.6%, flat_M6 **+36.2%** in head's favor (mirror image of gcc's
−34.5%), uniform/sparse/gzip_random within ±0.1%.  Same mechanism as
upstream `34a72de` ("Zen binary lottery": per-build encoding/schedule
rolls, ±15–22%); the scalar flat loop is simply layout-sensitive on
x86 in both compilers' rolls.  SIMD columns agree across compilers
(gcc +1.05/+1.09%, clang +0.78/+0.79% on the same box), so the +~1%
SIMD win is compiler-stable.

Other caveats observed:

- The M4 mini throttles (bench logged "CPU freq dropped 26–49%",
  drift up to +194%): its bell_s10 −7.2% BU outlier shrank to −1.8%
  in a 3-round `--tdbu` recheck; treat |Δ| ≲ 2% on M4 as noise.
- One clang round on c7i (base-r2) ran ~12% faster across ALL
  implementations including the untouched trad/huf0 comparators
  (spot-VM turbo window) — reported clang numbers use the clean
  r1-vs-r1 pair (drift +0.1%/+0.2%).  Best-of-rounds is not robust to
  a whole-round frequency excursion; check comparator columns before
  believing a uniform shift.
- The `+hdr_est` ≈ 2.3e18 values in the size table are pre-existing on
  main (u64 underflow when flat-aware `vIN` goes negative); display-only.

## Per-distribution decode M/s (best of 3 interleaved rounds, head=v0.6 vs base=v0.5)

### Apple M1 Max (clang, block 16384)

| dist | BU v0.5 | BU v0.6 | ΔBU | N v0.5 | N v0.6 | ΔN | Δscalar | Δbytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proba80 | 4633 | 4549 | -1.8% | 4712 | 4543 | -3.6% | -1.2% | -321 |
| proba50 | 9962 | 9786 | -1.8% | 9627 | 9500 | -1.3% | -0.6% | -890 |
| proba14 | 5120 | 5076 | -0.9% | 5093 | 5157 | +1.3% | -0.4% | -755 |
| proba02 | 4554 | 4482 | -1.6% | 4509 | 4480 | -0.6% | -0.4% | -1264 |
| bell_s10 | 5029 | 5113 | +1.7% | 5128 | 5050 | -1.5% | -0.8% | -1280 |
| bell_s30 | 4480 | 4560 | +1.8% | 4453 | 4460 | +0.2% | +0.0% | -768 |
| bell_s80 | 6318 | 6404 | +1.4% | 6314 | 6435 | +1.9% | -1.0% | +0 |
| uniform | 10435 | 10284 | -1.4% | 10194 | 10274 | +0.8% | +1.8% | +0 |
| english | 6506 | 6681 | +2.7% | 6413 | 6505 | +1.4% | -0.6% | -261 |
| zipfian | 4547 | 4474 | -1.6% | 4521 | 4683 | +3.6% | -0.9% | -768 |
| sparse_4 | 35025 | 34045 | -2.8% | 34114 | 34197 | +0.2% | +0.0% | +0 |
| sparse_16 | 34464 | 34521 | +0.2% | 34851 | 34450 | -1.2% | +1.0% | +0 |
| geometric | 4893 | 4891 | -0.0% | 4881 | 4909 | +0.6% | +0.0% | -256 |
| two_sym_eq | 34909 | 33798 | -3.2% | 34836 | 33880 | -2.7% | -1.0% | +0 |
| two_sym_90/10 | 7899 | 7877 | -0.3% | 7792 | 7782 | -0.1% | -3.5% | +0 |
| flat_M3 | 24217 | 24175 | -0.2% | 24308 | 24168 | -0.6% | +1.5% | +0 |
| flat_M5 | 24877 | 24716 | -0.6% | 24196 | 24210 | +0.1% | +1.1% | +0 |
| flat_M6 | 20570 | 20267 | -1.5% | 20292 | 20470 | +0.9% | +1.0% | +0 |
| flat_M7 | 13233 | 13028 | -1.5% | 13265 | 13024 | -1.8% | +0.2% | +0 |
| html_wiki | 3999 | 4022 | +0.6% | 4004 | 3984 | -0.5% | -0.4% | -1166 |
| prose_pride | 4581 | 4669 | +1.9% | 4618 | 4683 | +1.4% | -0.4% | -662 |
| image_jpeg | 5284 | 5270 | -0.3% | 5283 | 5330 | +0.9% | -0.6% | -346 |
| json_api | 4296 | 4308 | +0.3% | 4229 | 4290 | +1.4% | +2.5% | -952 |
| source_c | 4691 | 4804 | +2.4% | 4736 | 4750 | +0.3% | +0.0% | -1527 |
| log_apache | 4253 | 4350 | +2.3% | 4309 | 4318 | +0.2% | +0.4% | -998 |
| dna_fasta | 5616 | 5717 | +1.8% | 5564 | 5654 | +1.6% | +0.6% | -292 |
| csv_numeric | 3556 | 3582 | +0.7% | 3570 | 3695 | +3.5% | -0.3% | -1101 |
| gzip_random | 10318 | 10356 | +0.4% | 10360 | 10338 | -0.2% | +1.9% | +0 |
| chinese_text | 4063 | 4161 | +2.4% | 4135 | 4116 | -0.5% | +0.4% | -245 |
| calgary_pic | 3678 | 3682 | +0.1% | 3593 | 3624 | +0.9% | -0.8% | -881 |
| **geomean** | **7649** | **7651** | **+0.02%** | **7618** | **7634** | **+0.20%** | **−0.02%** | |

### Apple M4 Mac mini (clang, block 16384; throttling — ±2% is noise)

| dist | BU v0.5 | BU v0.6 | ΔBU | N v0.5 | N v0.6 | ΔN | Δscalar | Δbytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proba80 | 7683 | 7974 | +3.8% | 7711 | 8005 | +3.8% | -0.4% | -321 |
| proba50 | 17524 | 18992 | +8.4% | 17307 | 18775 | +8.5% | +0.0% | -890 |
| proba14 | 9069 | 9601 | +5.9% | 8897 | 9556 | +7.4% | +3.4% | -755 |
| proba02 | 8210 | 8227 | +0.2% | 8155 | 8207 | +0.6% | -1.3% | -1264 |
| bell_s10 | 8628 | 8009 | -7.2%* | 8601 | 8267 | -3.9% | -1.8% | -1280 |
| bell_s30 | 7991 | 7959 | -0.4% | 7720 | 8013 | +3.8% | -2.7% | -768 |
| bell_s80 | 10504 | 10466 | -0.4% | 10584 | 10637 | +0.5% | -1.4% | +0 |
| uniform | 15255 | 15128 | -0.8% | 15378 | 15238 | -0.9% | +1.4% | +0 |
| english | 11224 | 11057 | -1.5% | 11207 | 11074 | -1.2% | -1.0% | -261 |
| zipfian | 7644 | 7972 | +4.3% | 7666 | 7964 | +3.9% | -2.2% | -768 |
| sparse_4 | 55080 | 54971 | -0.2% | 55007 | 54935 | -0.1% | +0.0% | +0 |
| sparse_16 | 55297 | 55334 | +0.1% | 55297 | 54225 | -1.9% | +0.6% | +0 |
| geometric | 7576 | 7898 | +4.3% | 7585 | 7989 | +5.3% | +0.0% | -256 |
| two_sym_eq | 54085 | 54792 | +1.3% | 54085 | 54792 | +1.3% | +0.0% | +0 |
| two_sym_90/10 | 12811 | 12913 | +0.8% | 12860 | 12817 | -0.3% | +1.2% | +0 |
| flat_M3 | 37855 | 37600 | -0.7% | 37787 | 37567 | -0.6% | +0.7% | +0 |
| flat_M5 | 37250 | 36857 | -1.1% | 37533 | 36744 | -2.1% | +1.4% | +0 |
| flat_M6 | 29465 | 29465 | +0.0% | 29321 | 29198 | -0.4% | +2.0% | +0 |
| flat_M7 | 19508 | 19527 | +0.1% | 19618 | 19522 | -0.5% | +2.4% | +0 |
| html_wiki | 7016 | 7109 | +1.3% | 6995 | 7075 | +1.1% | +1.5% | -1166 |
| prose_pride | 8142 | 8287 | +1.8% | 8170 | 8263 | +1.1% | +0.8% | -662 |
| image_jpeg | 8546 | 8890 | +4.0% | 8427 | 8928 | +5.9% | +0.0% | -346 |
| json_api | 7421 | 7683 | +3.5% | 7407 | 7823 | +5.6% | -1.1% | -952 |
| source_c | 8518 | 8661 | +1.7% | 8417 | 8593 | +2.1% | -1.3% | -1527 |
| log_apache | 7709 | 7817 | +1.4% | 7639 | 7749 | +1.4% | -1.1% | -998 |
| dna_fasta | 9009 | 9183 | +1.9% | 9040 | 9204 | +1.8% | +0.7% | -292 |
| csv_numeric | 5834 | 5928 | +1.6% | 5821 | 5760 | -1.0% | +0.3% | -1101 |
| gzip_random | 15255 | 15128 | -0.8% | 15409 | 15211 | -1.3% | +0.9% | +0 |
| chinese_text | 7143 | 7224 | +1.1% | 7116 | 7194 | +1.1% | -1.1% | -245 |
| calgary_pic | 5826 | 5847 | +0.4% | 5796 | 5827 | +0.5% | +0.3% | -881 |
| **geomean** | **12580** | **12721** | **+1.12%** | **12546** | **12715** | **+1.35%** | **+0.06%** | |

\* bell_s10 −7.2% coincided with a logged throttle event; a 3-round
`--tdbu` recheck gives −1.8%/−1.4% (within the mini's jitter), and the
recheck geomeans are +2.11%/+2.22% BU/N.

### Sapphire Rapids, EC2 c7i.large (gcc 13.3, block 32768, pinned)

| dist | BU v0.5 | BU v0.6 | ΔBU | N v0.5 | N v0.6 | ΔN | Δscalar† | Δbytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proba80 | 4076 | 4034 | -1.0% | 4074 | 4032 | -1.0% | +19.1% | -128 |
| proba50 | 11880 | 11924 | +0.4% | 11868 | 11925 | +0.5% | +2.1% | -259 |
| proba14 | 6235 | 6400 | +2.6% | 6264 | 6394 | +2.1% | -3.3% | -66 |
| proba02 | 5326 | 5429 | +1.9% | 5203 | 5388 | +3.6% | -4.9% | -311 |
| bell_s10 | 5659 | 5728 | +1.2% | 5657 | 5730 | +1.3% | -5.2% | -128 |
| bell_s30 | 5370 | 5505 | +2.5% | 5355 | 5512 | +2.9% | -5.0% | -382 |
| bell_s80 | 7855 | 8026 | +2.2% | 7836 | 7990 | +2.0% | -6.7% | +0 |
| uniform | 11572 | 11585 | +0.1% | 11575 | 11581 | +0.1% | -16.3% | +0 |
| english | 7703 | 7972 | +3.5% | 7691 | 7967 | +3.6% | -3.9% | +66 |
| zipfian | 5590 | 5774 | +3.3% | 5592 | 5766 | +3.1% | -3.6% | -256 |
| sparse_4 | 18736 | 18755 | +0.1% | 18741 | 18771 | +0.2% | -14.0% | +0 |
| sparse_16 | 16107 | 16134 | +0.2% | 16129 | 16126 | -0.0% | -15.2% | +0 |
| geometric | 4921 | 4928 | +0.1% | 4922 | 4925 | +0.1% | +1.0% | -57 |
| two_sym_eq | 22293 | 22275 | -0.1% | 22308 | 22267 | -0.2% | +0.0% | +0 |
| two_sym_90/10 | 6374 | 6316 | -0.9% | 6375 | 6309 | -1.0% | +0.0% | +0 |
| flat_M3 | 17692 | 17717 | +0.1% | 17700 | 17699 | -0.0% | -36.6% | +0 |
| flat_M5 | 14742 | 14776 | +0.2% | 14779 | 14785 | +0.0% | -24.6% | +0 |
| flat_M6 | 14012 | 13987 | -0.2% | 14063 | 14112 | +0.3% | -34.5% | +0 |
| flat_M7 | 12754 | 12732 | -0.2% | 12750 | 12737 | -0.1% | -30.9% | +0 |
| html_wiki | 4473 | 4490 | +0.4% | 4530 | 4441 | -2.0% | -5.8% | -294 |
| prose_pride | 5686 | 5900 | +3.8% | 5673 | 5912 | +4.2% | -3.4% | -134 |
| image_jpeg | 6781 | 6946 | +2.4% | 6771 | 6944 | +2.6% | -5.4% | -5 |
| json_api | 5296 | 5415 | +2.2% | 5284 | 5413 | +2.4% | -5.0% | -269 |
| source_c | 6053 | 6215 | +2.7% | 6072 | 6202 | +2.1% | -3.8% | -400 |
| log_apache | 5278 | 5449 | +3.2% | 5262 | 5444 | +3.5% | -8.2% | -204 |
| dna_fasta | 5237 | 5357 | +2.3% | 5225 | 5353 | +2.4% | -1.1% | -139 |
| csv_numeric | 3992 | 3972 | -0.5% | 3988 | 3975 | -0.3% | -6.4% | -336 |
| gzip_random | 12205 | 12205 | +0.0% | 12189 | 12187 | -0.0% | -18.2% | +0 |
| chinese_text | 4737 | 4702 | -0.7% | 4749 | 4708 | -0.9% | -4.4% | +54 |
| calgary_pic | 3709 | 3743 | +0.9% | 3715 | 3739 | +0.6% | -1.6% | -302 |
| **geomean** | **7581** | **7663** | **+1.09%** | **7578** | **7658** | **+1.05%** | **−8.97%†** | |

### Zen 4, EC2 m7a.large (gcc 13.3, block 32768, pinned)

| dist | BU v0.5 | BU v0.6 | ΔBU | N v0.5 | N v0.6 | ΔN | Δscalar† | Δbytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proba80 | 5341 | 5282 | -1.1% | 5343 | 5282 | -1.1% | +0.5% | -128 |
| proba50 | 15155 | 15465 | +2.0% | 15144 | 15478 | +2.2% | +1.6% | -259 |
| proba14 | 8544 | 8661 | +1.4% | 8529 | 8671 | +1.7% | -1.3% | -66 |
| proba02 | 7489 | 7590 | +1.3% | 7535 | 7589 | +0.7% | -4.9% | -311 |
| bell_s10 | 7876 | 8062 | +2.4% | 7948 | 8029 | +1.0% | -6.2% | -128 |
| bell_s30 | 7418 | 7496 | +1.1% | 7380 | 7560 | +2.4% | -3.8% | -382 |
| bell_s80 | 11964 | 11999 | +0.3% | 11939 | 12054 | +1.0% | -7.8% | +0 |
| uniform | 38488 | 38532 | +0.1% | 38423 | 38660 | +0.6% | -37.4% | +0 |
| english | 11003 | 11181 | +1.6% | 11035 | 11153 | +1.1% | -5.8% | +66 |
| zipfian | 7328 | 7497 | +2.3% | 7348 | 7501 | +2.1% | -5.4% | -256 |
| sparse_4 | 59125 | 58961 | -0.3% | 59638 | 60214 | +1.0% | -37.4% | +0 |
| sparse_16 | 54396 | 54248 | -0.3% | 54286 | 54284 | -0.0% | -37.4% | +0 |
| geometric | 6624 | 6653 | +0.4% | 6638 | 6666 | +0.4% | +0.8% | -57 |
| two_sym_eq | 67003 | 67038 | +0.1% | 66936 | 67271 | +0.5% | +0.0% | +0 |
| two_sym_90/10 | 8521 | 8508 | -0.2% | 8534 | 8498 | -0.4% | +0.0% | +0 |
| flat_M3 | 56568 | 56948 | +0.7% | 56384 | 56427 | +0.1% | -32.1% | +0 |
| flat_M5 | 48986 | 48912 | -0.2% | 49059 | 48605 | -0.9% | -27.7% | +0 |
| flat_M6 | 45444 | 45333 | -0.2% | 45294 | 45276 | -0.0% | -30.9% | +0 |
| flat_M7 | 43678 | 43460 | -0.5% | 43321 | 43359 | +0.1% | -22.7% | +0 |
| html_wiki | 6424 | 6507 | +1.3% | 6425 | 6507 | +1.3% | -11.3% | -294 |
| prose_pride | 7720 | 7854 | +1.7% | 7671 | 7863 | +2.5% | -3.9% | -134 |
| image_jpeg | 10008 | 10096 | +0.9% | 9975 | 10072 | +1.0% | -7.2% | -5 |
| json_api | 7177 | 7297 | +1.7% | 7165 | 7297 | +1.8% | -13.8% | -269 |
| source_c | 8045 | 8189 | +1.8% | 8010 | 8190 | +2.2% | -8.3% | -400 |
| log_apache | 7423 | 7538 | +1.5% | 7386 | 7558 | +2.3% | -16.2% | -204 |
| dna_fasta | 7325 | 7432 | +1.5% | 7325 | 7435 | +1.5% | -0.8% | -139 |
| csv_numeric | 5892 | 5969 | +1.3% | 5896 | 5972 | +1.3% | -9.3% | -336 |
| gzip_random | 38807 | 38943 | +0.4% | 38834 | 38871 | +0.1% | -37.4% | +0 |
| chinese_text | 6601 | 6672 | +1.1% | 6602 | 6659 | +0.9% | -5.6% | +54 |
| calgary_pic | 4740 | 4775 | +0.7% | 4746 | 4774 | +0.6% | -1.2% | -302 |
| **geomean** | **13514** | **13625** | **+0.82%** | **13508** | **13633** | **+0.92%** | **−13.56%†** | |

## Conclusion

wire v0.6 verifies on all four hosts: SIMD decode +0.8–1.4% geomean on
M4/SPR/Zen 4 and neutral on M1 Max, total compressed size smaller
everywhere, tests green.  The only large per-file regressions (x86
scalar flat-tree dists, −20…−37% under gcc) occur on byte-identical
streams with untouched scalar source, and invert under clang on the
same host (+5…+36%) — the per-build code-layout lottery documented in
upstream `34a72de`, not an effect of this change.

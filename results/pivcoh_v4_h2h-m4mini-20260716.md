# pivcoh.h wire v4 vs huf0: the small-chunk head-to-head

`bench_huf0_lits` on the M4 mini (reps=8): huf0 (zstd's Huffman,
HUF_compress/HUF_decompress incl. its weights header + raw fallback) vs
pivcoh.h at SIMPLEST and BALANCED effort, one fresh table per zstd BLOCK
(the .litblk cadence — the smallest realistic chunking).  pivcoh size =
payload + its per-window table header; decode timings include parsing it.
"base" = wire v3 (production-compatible: 128 B nibble header, u16 K_right +
FSE marker per node, no raw fallback); "v4" = the redesigned wire.
Raw output: `pivcoh_v4_h2h-m4mini-20260716.txt`.

Geomeans (enc MB/s / dec MB/s / ratio):

## silesia-zstd-lits (12 files, ~14 KB windows)

|            | huf0             | pivcoh SIMPLEST   | pivcoh BALANCED   |
|------------|------------------|-------------------|-------------------|
| v3 (base)  | 1185 / 1356 / 1.28 | 1450 / 7300 / 1.25 | 1029 / 9930 / 1.25 |
| v4         | 1182 / 1355 / 1.28 | 1364 / 7206 / **1.27** | 984 / 8980 / **1.27** |

## prague-zstd-lits (30 files, smaller/rougher windows)

|            | huf0             | pivcoh SIMPLEST   | pivcoh BALANCED   |
|------------|------------------|-------------------|-------------------|
| v3 (base)  | 1067 / 1423 / 1.25 | 1294 / 6647 / 1.18 | 860 / 9571 / 1.19 |
| v4         | 1067 / 1422 / 1.25 | 1217 / 6721 / **1.23** | 830 / 8597 / **1.23** |

## What v4 changed

* code lengths: 128 B flat nibble header -> RLE nibble wire
  (`pivcoh_lens_wire_*`), ~46 B mean on zstd-lits tables (max 129, hard
  bound), decoded in ~90 ns
* per merge node: FSE marker byte deleted; K_right is 1 byte when the
  node's symbol count fits (the decoder always knows it) — ~2 B/node
* frame: u64 size -> varint; u32 per-block length prefixes dropped
  (segments self-delimit); raw-store segments (u16 top bit) for spans
  coding does not shrink — huf0-style incompressible fallback
* the bench itself: pivcoh windows fall back to raw store like huf0's,
  and the decode pass parses the lens wire per window (header costs are
  now inside the timings on both sides)

## Reading

* The ratio gap to huf0 closes from -2.3 % to **-0.8 %** on silesia and
  from -5.0 % to **-1.6 %** on prague, at per-BLOCK cadence — the
  worst case for header overheads.  At table-lifetime cadence the gap
  is proportionally smaller still.
* Decode stays 5.3-6.6x huf0; the ~2-9 % dip vs the v3 rows is honest
  accounting (v3 rows excluded pivcoh's header parse; huf0's number
  always included its own), concentrated on files with tiny windows.
* BALANCED now costs literally nothing in size vs SIMPLEST (both 1.27 /
  1.23) while decoding ~25 % faster — the joint pass plus the repeat-run
  token in the lens wire (big equal-length classes RLE well) cancel out.
* v4 streams are NOT production-decodable (and vice versa); trees remain
  production-identical (`bench_pivcoh_check` gates tables + roundtrip,
  ASan-clean, plus a standalone wire fuzz harness: truncations, bit
  flips, hand-built frames).

## Addendum: lens wire mode 2 (fixed-prefix delta tokens, commit ec882f7)

Lengths coded as deltas from the previous symbol's length under a fixed
canonical prefix code (trained jointly on both corpora's table
populations, 2.95 b/token vs 2.92 b entropy); unified 512-entry decode
LUT emits up to four 2-bit delta codes per lookup.  Cost-aware writer:
emitted only when it beats mode 1 by >= 16 B (dense tables save 30-45 B);
`PIVCOH_LENS_WIRE_NO_MODE2` suppresses emission.

Corpus table means: silesia 66.0 -> 57.3 B, prague 65.6 -> 51.2 B
(huf0 weight headers: 47.8 / 56.6).  Head-to-head (v43 rows in the txt):

|            | huf0             | pivcoh SIMPLEST   | pivcoh BALANCED   |
|------------|------------------|-------------------|-------------------|
| silesia    | 1171 / 1354 / 1.28 | 1280 / 6701 / **1.28** | 952 / 8598 / 1.27 |
| prague     | 1059 / 1421 / 1.25 | 1149 / 6408 / 1.23 | 800 / 8184 / 1.23 |

SIMPLEST reaches huf0 ratio parity on silesia at 4.9x its decode; the
lens parse costs ~4-7% decode at this extreme per-block cadence
(mozilla, dense 256-symbol tables, is the worst case at ~ -6%) and
amortizes away at table-lifetime cadence.  Residual size gap on the
dense tables is per-table adaptivity, unreachable without carrying an
entropy table on the wire.

# Silesia LZ4HC-9 literal streams

The 12 Silesia-corpus files preprocessed by `extras/lz4_lits.c`:
compress with LZ4HC level 9, walk the LZ4 block format, and concatenate
the literal runs — i.e. exactly the byte stream an LZ-front-ended
entropy coder sees.  32.9 MB total; per-file literal fractions range
from 3.5 % (xml) to ~22 % (mozilla).

This is the primary realistic benchmark corpus for the per-window
table-rebuild benches (`extras/bench/bench_lits_windows.c`,
`bench_table_lifetime.c`) and all the joint-lengths results in
`docs/JOINT-LENGTHS.md`.  Aggregation convention: files are windowed
independently (never concatenated), speeds summarized as the unweighted
geomean across files, ratios as the unweighted mean including the
128-byte-per-window lengths header.

## Provenance

* Source: the Silesia corpus, Sebastian Deorowicz, Silesian University
  of Technology — <https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia>
  (`silesia.zip`).  Input hashes: `SHA256SUMS.inputs`.
* Extraction: `extras/lz4_lits.c` built against **liblz4 1.10.0**,
  `LZ4_compress_HC(..., 9)`.  Output hashes: `SHA256SUMS`.

The committed `.lits` files are the reference corpus — every published
number was measured on these exact bytes.  `extras/fetch_silesia_lits.sh`
reproduces them from the original corpus and verifies against
`SHA256SUMS`.  Caveat: LZ4HC's literal/match split is not contractually
stable across liblz4 versions, so a hash mismatch after regeneration
usually means a different liblz4, not corruption; the committed files
win in that case.

This directory lives on its own branch (`silesia-lits`, off `main`) so
the ~31 MB of binary data is easy to cull or rebase away if it ever
becomes a burden.

## Reproducing the per-window (chunked/strided) numbers

The harnesses are included on this branch for reference —
`extras/bench/bench_lits_windows.c` (windowed e2e encode/decode/ratio;
`--ladder`, `--lams=` frontier sweep, `--guard=` override) and
`extras/bench/bench_prof_shares.c` (decode-kernel time decomposition
via `PIVCO_PROF`) — but they build against the codec-table API
(`build_codec_table` / `encode_ct` / `decode_dt`), which lives on the
`joint-flat-lengths-plus-fast-tables` branch, not on `main`.  So:

```sh
git checkout joint-flat-lengths-plus-fast-tables
git checkout silesia-lits -- testdata            # the corpus
rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
cc -O2 -Iinclude extras/bench/bench_lits_windows.c \
   build/libpivco_huffman.a -o blw -lm

# ladder (off/nudge/auto/exact at lambda=0.1, production guard):
./blw --G=64 --ladder --fse=0 --reps=8 testdata/silesia-lits/*.lits
# candidate default, per-file:
./blw --G=64 --joint=0.1 --fse=0 --gran=1 --reps=8 testdata/silesia-lits/*.lits
# ratio-vs-decode-speed frontier (guard off; mass-DP fallback above 1/7):
./blw --G=64 --lams=0.1,0.143,0.2,0.3,0.5,0.75,1,1.5,2.5,5 --guard=off \
      --fse=0 --gran=1 --reps=8 testdata/silesia-lits/*.lits
```

`bench_prof_shares.c` needs the library built with `-DPIVCO_PROF=1`
(build line in its header).  Raw captures of the published runs live
in `results/m4-2026071*-lamsweep-*` / `-prof-shares-*` on the joint
branch.  Numbers quoted in docs/JOINT-LENGTHS.md are Apple M4, PH
(`--fse=0`); expect ~3 % run-to-run thermal grain between batches.

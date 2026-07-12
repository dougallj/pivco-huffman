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

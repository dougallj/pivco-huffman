# jl-solver optimizations: encode e2e before/after per effort level

Branch `pivcoh-jl-opt` (12acf6a: 5/9-group register-resident take sweeps,
per-row band dispatch, 2x-unrolled generic sweep, iterative jl_sim) vs its
branch point `pivcoh-neon` (dc1a0f8).  Same production lib, same
`bench_pivcoh_lits --tables --reps=5` on the M4 mini: windows are the REAL
zstd HUF-table lifetimes from the `.lithdr` sidecars (a fresh pivcoh table
wherever zstd built a fresh HUF table).  Joint knobs = the effort-level
defaults (`PIVCOH_JOINT_DEFAULTS`: lambda 0.1, gamma 170, kappa 0).
enc-e2e = histogram + table build + joint pass + encode kernels.
Raw output: `jlopt_zstd_enc-m4mini-20260716-12acf6a.txt`.

Geomean enc-e2e MB/s (pivcoh side; `x prod` = vs the production codec's
encode in the same run):

## silesia-zstd-lits (12 files, 32.9 MB, 2343 table windows ≈ 14 KB/table)

| effort (gran)              | base | opt  | Δ enc  | x prod    | ebuild/w      | adopt     |
|----------------------------|------|------|--------|-----------|---------------|-----------|
| SIMPLEST (off)             | 1511 | 1523 | +0.8%  | 1.23x     | 1.24 us       | —         |
| BALANCED (coarse, -1)      | 1085 | 1133 | +4.4%  | 0.88→0.93x| 4.25→3.83 us  | 1421/2343 |
| FASTER_DECOMPRESS (auto, 0)| 801  | 891  | +11.2% | 0.66→0.73x| 7.75→6.29 us  | 1658/2343 |
| FASTEST_DECOMPRESS (exact) | 389  | 439  | +12.9% | 0.32→0.36x| 26.8→23.1 us  | 2150/2343 |

## prague-zstd-lits (30 files, 32 MB, 1033 table windows ≈ 31 KB/table)

| effort (gran)              | base | opt  | Δ enc  | x prod    | ebuild/w      | adopt     |
|----------------------------|------|------|--------|-----------|---------------|-----------|
| SIMPLEST (off)             | 1308 | 1322 | +1.1%  | 1.14x     | 1.14 us       | —         |
| BALANCED (coarse, -1)      | 858  | 906  | +5.6%  | 0.76→0.79x| 4.16→3.75 us  | 905/1033  |
| FASTER_DECOMPRESS (auto, 0)| 615  | 693  | +12.7% | 0.54→0.61x| 7.76→6.26 us  | 957/1033  |
| FASTEST_DECOMPRESS (exact) | 255  | 290  | +13.7% | 0.22→0.26x| 30.8→26.5 us  | 985/1033  |

Decode e2e and compression are unchanged (identical trees by construction —
the solver rewrites are bit-identity-gated; dec-e2e differences are run
noise, pivcoh ~10.0-10.6 GB/s at the shaped tiers, 1.22-1.25x prod).
Compression deltas vs plain Huffman: silesia +0.155/+0.072/+0.077 pp,
prague +0.093/+0.048/-0.012 pp (coarse/auto/exact).

Reading: the solver work lands where solver time dominates the per-window
encode budget — off is untouched (no solver), BALANCED gains ~5% e2e,
FASTER/FASTEST ~11-14%.  At BALANCED, pivcoh encode is now within 7-21% of
the production codec's (which runs no shaping here) while decoding ~1.23x
faster than it.

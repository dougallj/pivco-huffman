# Why dna_fasta is cursed on M4 (2026-07-10)

Investigation on tail-free-ping-pong-rewrite @ f067465, M4 mini, using
the new bench --blk / --no-fse knobs, xctrace, and an env-gated arena
offset (PIVCO_SCRATCH_OFF).  dna_fasta pivco_bu spanned 7.9–9.3 GB/s
(14.6 %) across today's configurations while being stable (1–2 %)
across processes of any one build — i.e. deterministic layout
sensitivity, not noise.  Three stacked causes:

## 1. dna is ~43 % FSE at defaults, and FSE is its own hazard
--no-fse streams decode at 15.5 GB/s vs 8.9 (proba80: 8→32 GB/s,
calgary_pic: 6→23 — the skewed MAIN dists are FSE-DOMINATED).  The FSE
path is extremely layout-sensitive: proba80 oscillates 4.6↔7.9 GB/s
(±42 %) with period 64 in --blk (fast only at blk ≡ 0 mod 64); with
FSE off the swing collapses to 8 %.  Any recompile or stream-size
change re-rolls this dice for every FSE-heavy dist.  ==> The FSE
decoder (bit-reader / payload alignment) deserves its own session;
it is the single biggest source of "cursed" variance on M4.

## 2. FSE-free dna sits on an N-keyed cache-conflict resonance
- --blk sweep (no-fse): razor cliff 16368→16369 (ONE symbol,
  17.5→15.6 GB/s), slow band ≈ [16369, 16600), recovery by 16640.
  english swings only 3–6 % over the same sweep.
- xctrace at 16368 vs 16376: the regression is in the INTERIOR merges
  (codec_decode_subtree +30 % absolute), not the root merge, not the
  unaligned-N epilogue (~2 %).
- Decisive: PIVCO_SCRATCH_OFF=256 at blk=16376 restores 14.65→17.06
  GB/s; every offset 256..3840 is fast, only 0 is slow; offsets are
  neutral at fast-N.  Arena-internal distances don't change when the
  base shifts, so the conflict is arena vs an EXTERNAL 16 KB-aligned
  allocation (enc stream bitmap loads or dec_buf stores) — a 4K-class
  store/load conflict whose relative page phase is a pure function of
  N (both allocations are 16 KB-aligned on macOS, so it is exactly
  reproducible per build and flips with any size change).
- Disproven: per-block K%4096 resonance (fast at blk 16640–16768 where
  the root split crosses 8192), bitmap page-crossing, output-buffer
  alignment (blk=16384 has perfectly aligned outputs and is slow).

## 3. Why dna specifically
Its tree splits nearly 50/50 at the top (root K_small = 8192−134 at
N=16384; next split 4096−89): the interior has the two largest
possible merges plus a deep HALF_RIGHT chain — the maximum concurrent
stream traffic of any MAIN dist, with natural buffer distances parked
next to 4 K multiples.  Everything nudges it; nothing nudges english.

## Recommendations
- Treat dna_fasta (and proba80/calgary/csv with FSE on) deltas under
  ~5 % on M4 as layout artifacts unless confirmed via a
  PIVCO_SCRATCH_OFF sweep or --blk jitter.
- Candidate cheap fix for the merge-side resonance: offset the decode
  arena base by a small constant (256 recovers dna fully; english
  −1 %) or rotate the offset per block — needs a full-sweep A/B before
  adopting (large offsets hurt english up to −7 %).
- FSE alignment investigation is the higher-value follow-up.

Raw data: inv_*/invnf_*/invb_*/off_* in the session scratchpad;
diagnostic knobs committed (bench --blk was already on the branch;
PIVCO_SCRATCH_OFF + bench_profile_english argv[2] land with this doc).

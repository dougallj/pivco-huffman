# Tail-free NEON decode (branch: tail-free)

Experiment: remove ALL tail handling from the NEON decompression
primitives — every decode loop runs full-width straight past the end of
its region instead of dropping to narrower / scalar mop-up ladders.

## What changed

`src/pivco_huffman_primitives_neon.h` (decode side only; encode untouched):

| primitive          | before                                        | after |
|--------------------|-----------------------------------------------|-------|
| `merge_vec_vec`    | 64-wide + V4 stride-16 + 8-wide + scalar      | 64-wide + 16-wide straight past K |
| `merge_cst_vec`    | 64-wide + 16-wide + scalar                    | 64-wide + 16-wide straight past K |
| `merge_vec_cst`    | 64-wide + 16-wide + scalar                    | 64-wide + 16-wide straight past K |
| `merge_cst_cst`    | 16-wide + 8-wide + scalar                     | ONE 16-wide loop |
| `merge_flat_d2`    | 64-wide + 16-wide + 4-wide + scalar           | 64-wide + 16-wide straight past n |
| `merge_flat_d3`    | 32-wide (n−16 bound) + 16-wide block + 8-safe + scalar | 32-wide + 16-wide straight past n |
| `merge_flat_d4`    | 32-wide + 16-wide + 2-wide + scalar           | 32-wide + 16-wide straight past n |
| `merge_flat_d5`    | gated blocks + 8-safe + scalar                | ONE 16-wide loop |
| `merge_flat_d6`    | gated blocks + 8-safe + scalar                | ONE 16-wide loop |
| `merge_flat_d7`    | 16-fast (2 loads) + 8-fast + 8-safe + scalar  | ONE 16-wide loop, 1 load/16 codes (new +7 hi-shuffle table) |
| `popcount_K_right` | dead since K_right moved to the wire — deleted | — |

Merge tails run the same two-table SABD 16-byte kernel as the ryg main
loop; garbage mask bits ≥ K only steer lanes/cursors that are never
consumed again, so correctness inside [0, K) is automatic.

## Safety contract

Defined in `include/pivco_huffman.h`:

* `PIVCO_DECODE_DST_PAD` (16): decode dst needs N+16 writable bytes;
  [N, N+16) may receive garbage.
* `PIVCO_DECODE_SRC_PAD` (16): the input buffer needs 16 readable bytes
  past the consumed stream.

Supporting pieces:

* decode scratch arena +128 B slack (codec.c) — absorbs the 16 B store
  spill and the 64+16 B merge-source overread for every packed child
  buffer (interior spills land in sibling scratch that is rewritten or
  already consumed, ordered left-then-right).
* file codec keeps its EXACT external contract: `pivcohuf_compress`
  appends a 16-byte pad after the body (outside body_len, +16 in
  compress_bound; old decoders ignore it), and `pivcohuf_decompress`
  bounce-buffers any block whose remaining output slack is < B+16
  through the padded `block_buf`.  Old (pre-pad) .pvh files are
  rejected with TOO_SHORT instead of overreading.
* tests: `test_tailfree_sizes` NEON-decodes 30 block sizes × 5 tree
  shapes into a canaried buffer and fails if any store lands past
  DST_PAD.  The whole suite also passes under macOS GuardMalloc
  (page-end allocations, catches any out-of-allocation read/write).

## Results (bench 20 --all, best of 2 interleaved A/B pairs per host)

Decode `pivco_bu`:

* **M1 Max** (machine under light load): **geomean +5.0 %**.
  Merge-heavy dists +8–13 % (source_c +11.5 %, html_wiki +12.2 %,
  chinese_text +11.8 %, json_api/log_apache/prose_pride ~+10 %,
  proba50 +12.7 %, calgary_pic +8 %); flat-dominated +0–4 %.
* **M4 mini** (idle): **geomean +2.9 %**, max +8 % (flat_M5, proba50),
  merge-heavy dists +4–7 %.  Nominal negatives (proba80 −1.0 %,
  csv_numeric −2.3 %) sit inside that host's 5 % A/A run spread.
  See results/20260710-1142-5827214-tailfree-m4.md.
* **Graviton 4** (c8g.large spot, gcc 13.3, pinned, ~0.5 % A/A noise):
  **geomean +1.8 %**, two_sym_eq +9.4 %, merge-heavy +2–5 %.  Two real
  regressions on this host/compiler: sparse_16 −5.5 % (D=4 flat root;
  the executed kernel is IDENTICAL to main for 32-aligned n, so this
  is gcc-13 code-layout fallout, not the scheme — M4 shows +0.4 %) and
  flat_M7 −1.3 % (one-load d7 kernel slightly worse on Neoverse V2).
  See results/20260710-1211-5827214-tailfree-c8g.md.
* code size: `codec_decode_subtree` 6916 → 4552 B (−34 %), NEON codec
  TU text 15.9 → 12.9 KB (−19 %, encode untouched)

## Options to make this shippable

1. **Contract + capacity param** (bench branch ships docs-only): add
   `dst_capacity` to the decode API; if capacity ≥ N+16 run tail-free,
   else fall back to the careful path (compile both from the same TU).
   Callers that can afford 16 spare bytes get full speed; nobody is
   unsafe.  This is the LZ4/zstd wild-copy pattern.
2. **Exact-end root merge**: only the root call writes user memory.
   Pass K_left down and recompute the final 16-byte chunk backwards
   from the buffer end (cursors are known: left+K_left / right+K_right;
   the mask needs a shifted 32-bit load since K isn't byte-aligned).
   Zero overwrite, ~10 extra instructions once per block, keeps every
   interior merge tail-free into library-owned scratch.  Needs a K<16
   micro-fallback.
3. **Bounce root** (simplest): decode the root into scratch, memcpy N
   out.  Zero contract change but costs a block-sized memcpy (~4–8 %
   at these speeds) — worse than (2) for no benefit beyond simplicity.
4. **Src side**: the stream-level 16-byte pad (already implemented in
   the file codec) generalizes: spec the pad as part of the container.
   Raw block-API users keep the SRC_PAD readable-slack rule, which any
   framed format satisfies for free for all but the last block.

Recommended ship shape: (1) + (2) — tail-free everywhere, exact-end
handling only at the root when the caller can't provide slack, stream
pad in the container.

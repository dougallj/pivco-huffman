# Unaligned-N cliff: measured and removed (M4, 2026-07-10)

`bench 20 --blk=16383` (unaligned) vs `--blk=16384` (aligned), MAIN
9-dist set, one run per cell.  "tfpr-slab" = tail-free-pr before the
exact-tail epilogue (root output bounced through an arena slab + full
N-byte memcpy for unaligned N).

| variant                    | unaligned / aligned geomean |
|----------------------------|------------------------------|
| main (exact-tail kernels)  | +0.7 % (no cliff)            |
| tail-free-pr, slab bounce  | **−10.7 %** (the cliff; image_jpeg −13 %, prose_pride −13 %, english −11 %) |
| tail-free-pr, epilogue     | −0.6 %                       |
| tail-free-ping-pong, epilogue | +0.1 %                    |

The epilogue splits the root merge at n1 = N & ~15: chunk-aligned
prefix direct into `symbols` (exact stores; bit n1 is byte-aligned in
the bitmap), then the 1..15 tail symbols as an independent sub-merge
into a 16-byte temp — cursors recovered O(1) from the wire's K_right
plus a popcount of the <=15 tail bits.  Flat roots split the same way
with a stack-copied source tail; D == 8 stays a plain memcpy.  Aligned
throughput is unchanged (the root-tail check is a compare against
NULL / a false flag on production block sizes).

The bench gained `--blk=N` in the same series — the 4M sequence
divides the default block sizes exactly, so this path had never been
timed before.  Raw captures: m4-*-cliff-*.txt.

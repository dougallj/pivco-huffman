/* dump_joint_lens — emit the production build's code lengths plus the
 * joint-optimizer rungs' lengths per bench distribution, as a JS data
 * sidecar for figures/tree_viz.html (lengths= URL param) and
 * figures/tree_compare.html.
 *
 * Usage:
 *   ./pivco_dump_joint_lens > figures/tree_viz_joint.js
 *
 * Output:
 *   window.PIVCO_JOINT_LENGTHS = {
 *     "proba80": { "base": [..256..], "nudge": null, "auto": null, "exact": null },
 *     ...
 *   };
 *
 * base = production build lengths (two-queue + limit heuristic) — NOT
 * necessarily the in-page JS Huffman.  Rungs run at lambda = 0.1 with
 * the production guard and this branch's cost-model defaults; null =
 * the rung's lengths equal base (solver no-op or guard veto).  Freqs
 * are scaled to ~4M total symbols first so per-block guard terms see
 * the same regime as the 4M-symbol dist-ladder benches. */
#include "pivco_huffman.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void            bench_init(void);
extern int             bench_num_distributions(void);
extern const char     *bench_dist_name(int idx);
extern const uint64_t *bench_dist_freq(int idx);

static void emit_lens(const uint8_t lens[256])
{
    printf("[");
    for (int s = 0; s < 256; s++) printf("%s%d", s ? "," : "", lens[s]);
    printf("]");
}

int main(void)
{
    bench_init();
    pivco_huffman_set_fse_enabled(1);
    printf("window.PIVCO_JOINT_LENGTHS = {\n");
    for (int d = 0; d < bench_num_distributions(); d++) {
        const uint64_t *raw = bench_dist_freq(d);
        uint64_t total = 0;
        for (int s = 0; s < 256; s++) total += raw[s];
        uint64_t scale = total ? ((uint64_t)4 << 20) / total : 1;
        if (!scale) scale = 1;
        uint64_t freq[256];
        for (int s = 0; s < 256; s++) freq[s] = raw[s] * scale;

        pivco_huffman_set_joint_lambda(0.0);
        pivco_huffman_codec_table_t base;
        pivco_huffman_build_codec_table(freq, &base);

        printf("%s\"%s\": {\"base\":", d ? ",\n" : "", bench_dist_name(d));
        emit_lens(base.code_len);

        struct { const char *name; int gran; } rungs[] = {
            { "nudge", -1 }, { "auto", 0 }, { "exact", 1 },
        };
        for (int r = 0; r < 3; r++) {
            pivco_huffman_set_joint_lambda(0.1);
            pivco_huffman_set_joint_granularity(rungs[r].gran);
            pivco_huffman_codec_table_t ct;
            pivco_huffman_build_codec_table(freq, &ct);
            printf(",\"%s\":", rungs[r].name);
            if (memcmp(base.code_len, ct.code_len, 256) != 0) emit_lens(ct.code_len);
            else printf("null");
        }
        printf("}");
    }
    printf("\n};\n");
    pivco_huffman_set_joint_lambda(0.0);
    pivco_huffman_set_joint_granularity(1);
    return 0;
}

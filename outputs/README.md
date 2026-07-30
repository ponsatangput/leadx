# Research artifacts (benchmarks, raw results, reports)

Every performance claim in the main README/DESIGN_NOTES points back to a raw
CSV in this directory, produced by the script next to it. Reports state their
own caveats (split counts, draw counts, provenance) — read those before
quoting numbers.

Start with `CRITEO_MULTISPLIT_GATE_REPORT.md`: it is the newest validation and
supersedes the original fixed-split Criteo accuracy interpretation. Its raw,
metadata, and compact summary artifacts share the
`criteo_uplift_multisplit_gate_*` prefix.

`CRITEO_SUPERVISED_IMBALANCE_REPORT.md` is the clean imbalance-only check: all
models receive complete labels. It should be read before claiming that low
prevalence by itself is a LeadX advantage.

`CRITEO_ULTRA_IMBALANCE_REPORT.md` continues that protocol at 1:1,000 and
1:10,000 on 10M rows. It records a preliminary crossover only at 1:10,000,
with no tight-shortlist or speed win.

**Note on paths:** these are research scripts, not part of the installable
package. Dataset paths default to the author's machine layout; every recent
script accepts `--data`/CLI overrides (see each script's `--help`). Datasets
themselves are **not** included in the repo (Kaggle terms / Criteo
CC BY-NC-SA) — download them yourself and point the scripts at them.

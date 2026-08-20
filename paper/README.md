# Manuscript verification

Everything in this directory is specific to *Quantifying the Recoverability of V and
J Genes from TCR CDR3 Sequences Using Generative Repertoire Models*. It is **not part
of the tool**: installing `supervdj` never requires anything here, and nothing in
`supervdj/` or `scripts/` imports from this directory.

`verify_manuscript_numbers.py` recomputes every quantity the manuscript states, from
the deposited posterior table and the ingest artifacts, and checks each one against
the value printed in the text. It prints a quoted-against-recomputed table and exits
non-zero if any row disagrees.

```bash
pip install -e .
PYTHONPATH=. python paper/verify_manuscript_numbers.py
```

```
163/163 claims reproduce
```

The total is fixed at 163 whatever is present locally. Two of the checks read the
manuscript's LaTeX sources, which are not deposited; without them those rows are
emitted as skips and the line reads
`163/163 claims reproduce (3 skipped, inputs not present here)`, so the number never
depends on what happens to be on disk.

The 163 checks cover the ingest and canonical counts, the conditional entropies,
top-k coverage, the high-confidence fractions, the confusion grouping and its
adjusted Rand indices, the selection shift with its permutation null and bootstrap
intervals, the matched-n and cross-chain comparisons, the held-out cohort
replication, the calibration split, the V-usage reweighting, and every cell of the
six supplementary tables.

## Inputs

The script reads the per-sequence posterior table and the analysis artifacts under
`results/`, none of which are in git. Download the deposited analysis record, unpack
it so `results/` sits at the repository root, and run the command above. If the
posterior table is elsewhere, pass `--posteriors /path/to/posteriors.tsv`.

> The record is not deposited yet, so the DOI the script prints is a placeholder
> (`10.5281/zenodo.XXXXXXX`). It is set as `ZENODO` at the top of the script.

Any missing input is reported up front, as a list, naming the record to download.

`--resample` recomputes the 2,000-permutation test and the 1,000-replicate bootstrap
instead of reading their stored results; it takes considerably longer and should
reproduce the same numbers.

The three word-count claims compare the manuscript's stated counts against a fresh
count of the LaTeX sources. Those sources are not deposited, so on any clone the three
appear as `SKIP` rows rather than being dropped from the total.

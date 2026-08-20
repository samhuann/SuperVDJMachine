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
`results/`, none of which are in git. They are deposited as a single archive:

**https://doi.org/10.5281/zenodo.22024426**

Download the archive and unpack it at the repository root:

```bash
unzip supervdjmachine-results.zip   # 138 MB, 1.1 GB unpacked
PYTHONPATH=. python paper/verify_manuscript_numbers.py
```

The archive's top-level directory is `supervdjmachine-results/`, not `results/`. The
script finds it either way: it looks for whichever directory here holds
`posteriors.tsv` and points `results/` at it, printing which one it used. Name it
explicitly with `--results DIR` if you unpacked somewhere unusual, or point at a
relocated table with `--posteriors /path/to/posteriors.tsv`.

Any missing input is reported up front, as a list, naming the record to download.

`--resample` recomputes the 2,000-permutation test and the 1,000-replicate bootstrap
instead of reading their stored results; it takes considerably longer and should
reproduce the same numbers.

The three word-count claims compare the manuscript's stated counts against a fresh
count of the LaTeX sources. Those sources are not deposited, so on any clone the three
appear as `SKIP` rows rather than being dropped from the total.

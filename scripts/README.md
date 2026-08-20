# Analysis scripts

Analyses that run on top of the pipeline, useful with any posterior table. Anything
tied to the manuscript specifically, including the script that verifies its numbers,
is in `paper/` instead.

They read a per-sequence posterior table from `supervdj.run`, expected at
`results/posteriors.tsv` unless a script says otherwise. Run them from the repository
root with `PYTHONPATH=.`.

`supervdj/aggregate.py` holds the metric definitions they share: conditional entropy,
top-k coverage, confidence fractions, the V-by-V leakage matrix and its clustering,
the usage-controlled selection shift, and the resampling tests.

| script | what it does |
|---|---|
| `export_confusion_groups.py` | derives the confusion groups and calibration curves shipped in `supervdj/data/`, so the tables the candidate-set utility depends on can be rebuilt from scratch |
| `validate_calibration.py` | fits the coverage curve on half a set and measures achieved coverage on the other half, split by a hash of the CDR3 |
| `validate_cohort.py` | repeats the recoverability measurements on an independent cohort and scores its confusion grouping against a reference one |
| `cohort_usage_reweight.py` | isolates how much of a difference between two sets is explained by their V-usage marginals, by importance reweighting at fixed posteriors |

`validate_cohort.py` reads cohorts from `data/cohorts/`, overridable with
`SUPERVDJ_COHORT_DIR`.

The permutation tests default to 2,000 permutations, which floors a two-sided
p-value at 1/2001; `alpha_bootstrap` uses 1,000 bootstrap replicates. Both are the
values behind the published numbers, so defaults reproduce them.

Plotting is an optional extra, since `aggregate.py` can also draw the figures behind
these analyses:

```bash
pip install -e ".[figures]"
```

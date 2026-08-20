# SuperVDJMachine / supervdj

Posteriors over the V and J gene of a TCR, given only the CDR3 amino-acid sequence.

`supervdj` computes, for one CDR3, a distribution over the candidate V genes and
over the candidate J genes, derived analytically from a pretrained generative model
of V(D)J recombination. Nothing is trained and nothing is predicted in the
machine-learning sense. Because a single gene call is often not supportable, the
main entry point returns a calibrated candidate set rather than one gene.

## Install

```bash
conda env create -f environment.yml
conda activate supervdj
pip install -e .
```

Without conda: `pip install -r requirements.lock.txt && pip install -e .` (python 3.11).

Both files pin exact versions. **olga** and **sonnia** determine every posterior, so
a different version of either gives different numbers. SONNIA runs on Keras 3 and
the code sets `KERAS_BACKEND=torch` at import; nothing to export by hand.

## Candidate sets

```bash
python -m supervdj.candidates --chain TRB --cdr3 CASSLGQAYEQYF --coverage 0.9
python -m supervdj.candidates --chain TRA --input cdr3s.txt --out sets.tsv
```

```python
from supervdj.candidates import calibrated_candidates

r = calibrated_candidates("CASSLGQAYEQYF", "TRB", coverage=0.9)
r["n_candidates"], r["achieved_mass"]
r["groups"][0]     # {'group': 1, 'mass': 0.576, 'genes': ['TRBV5-1', 'TRBV7-9', ...]}
```

`--coverage 0.9` asks for a set that contains the true gene about 90% of the time.
That is not the same as taking 0.9 of the posterior mass: the raw posterior is
over-confident, so the set is sized from a calibration curve shipped in
`supervdj/data/calibration_{chain}.tsv`. Use `--mass` instead to take the posterior
at face value, in which case the result reports the `expected_coverage` that mass is
worth.

Candidates are also grouped by mutually confusable genes, read from
`supervdj/data/confusion_groups_{chain}.tsv`. Genes with too little support have no
group and report `group: null`.

## Posteriors for a dataset

```bash
# pre- and post-selection, both J-handling modes, gene and family resolution
python -m supervdj.run --input my_tcrs.tsv --out results/posteriors.tsv \
    --post-modes grid fixed --resolutions gene family

# OLGA-only pre-selection pass, no SONNIA
python -m supervdj.run --input my_tcrs.tsv --out results/pre.tsv --no-sonia

# custom gene groupings
python -m supervdj.run --input my_tcrs.tsv --out results/grouped.tsv \
    --resolutions gene group --group-map data/gene_groups.tsv
```

Input is any TSV/CSV with chain, CDR3, V and J columns, under any of the usual
spellings (`Gene/CDR3/V/J`, `locus/junction_aa/v_call/j_call`, `chain/cdr3/v/j`).
Output has one row per
(sequence x axis x model x resolution):

| column | meaning |
|---|---|
| `top1_label` | top-1 gene/group |
| `true_rank` | rank of the annotated gene, blank if it is not a candidate |
| `top1_mass` | posterior mass on the top gene/group |
| `entropy_nats` | posterior entropy |
| `posterior_json` | the full posterior |
| `pgen_total`, `status` | OLGA `Pgen(CDR3)` and a per-row status flag |

## Building an analysis set

`supervdj.canonical` turns one or more CDR3 tables into a set that is consistent
with the models: gene names reconciled to the OLGA gene set, CDR3s cut to the IMGT
junction convention, duplicates collapsed within and across inputs on
`(cdr3_aa, v_gene, j_gene, chain)`, and sequences OLGA calls impossible
(`Pgen == 0`) dropped, since their posteriors are empty by construction.

```bash
python -m supervdj.canonical --input my_tcrs.tsv --out results/canonical
python -m supervdj.canonical --input run1.tsv run2.csv cohort.xlsx --chains TRB
```

`--input` takes the same tables `supervdj.run` does: TSV, CSV or XLSX with chain,
CDR3, V and J columns under any of the usual spellings (`Gene/CDR3/V/J`,
`locus/junction_aa/v_call/j_call`, `chain/cdr3/v/j`). It writes
`canonical_{chain}.tsv`, a summary, and `rejected.tsv` giving every dropped record
with its reason. `--chains` restricts the run, `--workers` defaults to
`cpu_count - 1`, and runtime is dominated by the Pgen pass.

Each file is its own source, so a rearrangement keeps a record of which inputs it
came from. To see how much two datasets share before combining them, run the
normalization on its own:

```bash
python -m ingest.run_ingest --input cohort_a.tsv cohort_b.csv --out-dir results/ingest
```

which writes the deduplicated table plus a pairwise overlap report.

IMGT germline references, used to place the junction boundaries, ship in
`data/imgt/` and can be pointed elsewhere with `--imgt-dir`.

## Models

Both are the pretrained defaults that ship with the packages; neither is retrained.

| chain | OLGA model | recombination type | SONNIA selection model |
|---|---|---|---|
| TRA (alpha) | `human_T_alpha` | VJ (`GenerationProbabilityVJ`), no D segment | linear `Sonia(ppost_model="human_T_alpha")` |
| TRB (beta) | `human_T_beta` | VDJ (`GenerationProbabilityVDJ`) | linear `Sonia(ppost_model="human_T_beta")` |

Pre-selection is `P(V | CDR3) = Pgen(CDR3, V) / Pgen(CDR3)`, and likewise for J.
Post-selection reweights by the SONNIA selection factor `Q`:
`Ppost(CDR3, V, J) ∝ Pgen(CDR3, V, J) · Q(CDR3, V, J)`, marginalized to one axis.
Both arms range over the same candidates, taken from the loaded OLGA model. Only the
linear SONNIA models ship with weights; the deep `SoNNia` variant would need
training. Allele suffixes are stripped, so candidates are gene-level (`TRBV19*01`
becomes `TRBV19`).

## Analyses and figures

`supervdj/aggregate.py` computes the recoverability metrics: conditional entropy,
top-k coverage, confidence fractions, the V-by-V leakage matrix and its clustering,
the usage-controlled selection shift, and the resampling tests.
`aggregate.build_figures(df)` draws the figure set from a posterior table.

```bash
pip install -e ".[figures]"
python -c "import supervdj.aggregate as A; A.build_figures(A.load())"
```

`scripts/` holds analyses that run on top of a posterior table and are useful with
any dataset: deriving confusion groups and calibration curves, checking coverage on
held-out data, replicating measurements on an independent cohort, and decomposing a
difference between two sets by V usage.

`paper/` holds what is specific to the manuscript this pipeline was written for, at
present the script that re-derives every number it states and checks it against the
text. Nothing in the tool depends on that directory.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

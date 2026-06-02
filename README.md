# SuperVDJMachine / `supervdj`

Rank plausible **TRAV/TRAJ/TRBV/TRBJ** gene pairs for a given TCR CDR3 amino-acid
sequence using germline-anchor filters, motif priors, gene-usage priors, and
damped OLGA log-Pgen + SONIA selection scores.

> A CDR3 amino-acid sequence rarely uniquely identifies V and J — junctional
> additions and shared anchor motifs destroy that information. This tool
> therefore returns a **calibrated ranking**, not a single gene call.

## Install

```bash
pip install -e .          # runtime deps incl. OLGA + SONIA
pip install -e ".[dev]"   # + pytest
```

OLGA and SONIA are required dependencies. If their model files fail to load,
the wrappers degrade to neutral scores with a warning rather than crashing.

## Usage

```bash
# Top-5 TRB ranking
supervdj predict --chain TRB --cdr3 CASSIRSSYEQYF --top-k 5

# TRA, CSV to file
supervdj predict --chain TRA --cdr3 CAVRDSNYQLIW --top-k 20 \
    --format csv --out results.csv

# Skip OLGA/SONIA
supervdj predict --chain TRB --cdr3 CASSIRSSYEQYF --no-olga --no-sonia
```

Equivalent module form: `python -m supervdj predict ...`.

Output columns: `rank, chain, cdr3, v_gene, j_gene, log_pgen, log_selection,
motif_score, gene_usage_prior, penalty, final_log_score, posterior_probability,
explanation`. The pretty format adds a `confidence` label and any warnings.

### Python API

```python
from supervdj import predict, PredictionConfig

result = predict(
    cdr3="CASSIRSSYEQYF",
    chain="TRB",
    config=PredictionConfig(top_k=10),
)
for c in result.candidates[:5]:
    print(c.v_gene, c.j_gene, round(c.posterior_probability, 3))
```

## Scoring

```
final_log_score = w_olga * log_pgen + w_sonia * log_selection
                + log_motif_prior + log_gene_usage_prior - penalty
```

Posteriors are a numerically stable softmax over surviving candidates.
Confidence is `high` / `medium` / `low` based on top-posterior mass, anchor
strength, and whether OLGA/SONIA were available.

### Filtering

1. CDR3 sanity: length, canonical residues, conserved `C` start and `F`/`W` end.
2. V- and J-anchor compatibility against the first/last `n_anchor_match` residues.
3. If all candidates are filtered out, filters are auto-relaxed by one position
   and the result is marked low confidence.

## Reference data

Packaged IMGT human germline FASTAs live at `supervdj/data/imgt/`
(`TRAV/TRAJ/TRBV/TRBJ.fasta`). `supervdj/imgt.py` parses them, honours IMGT
`codon_start`, extracts CDR3 anchors (V: suffix from the conserved Cys; J:
prefix up to the F/W of the F/W-G-X-G motif), and collapses alleles to one
entry per gene. V genes keep the longest extracted anchor across alleles, with
`*01` used only as a tie-breaker. Pseudogenes/ORFs are dropped by default; pass
`--include-nonfunctional` to keep them.

Override at the CLI with a TSV (`gene/chain/gene_type/functional/anchor/usage_prior`):

```bash
supervdj predict --chain TRB --cdr3 CASSIRSSYEQYF \
    --v-ref trbv.tsv --j-ref trbj.tsv
```

## VDJdb integration

`supervdj usage` prints empirical V/J usage frequencies, and `supervdj eval`
benchmarks ranking accuracy (top-1/5/10 for V, J, and joint V+J) against any
VDJdb-style `gene/cdr3/v/j` table:

```bash
# Empirical V/J usage table
supervdj usage --vdjdb vdjdb.tsv --chain TRB --top 10

# Accuracy benchmark, optionally with the trained V model
supervdj eval --vdjdb vdjdb.tsv --chain TRB --limit 1000 --no-olga --no-sonia \
    --v-cnn-dir supervdj/data/v_cnn_trb
```

## Supervised V-gene model (CNN)

V is far less determined by the CDR3 than J, so the strongest signal comes from
a supervised sequence model. `VGeneCNN` (in `supervdj/v_model.py`) is a 1D-CNN
over residue embeddings. The marginal V ranking (`result.v_ranking`) is driven
by this model plus the anchor motif and usage prior; OLGA/SONIA are excluded
from V ranking (they encode the same germline signal at ~10^4x the cost) but
still drive the joint (V, J) `candidates` and the Pgen=0 impossibility filter.
TensorFlow is an optional dependency:

```bash
pip install -e ".[cnn]"

# Train once on a repertoire/labelled table and save the artifact
python scripts/train_v_cnn.py --train supervdj/data/PRJNA280417.tsv --out supervdj/data/v_cnn_trb

# Use it at predict time (loading is cheap; training is not)
supervdj predict --chain TRB --cdr3 CASSIRSSYEQYF --v-cnn-dir supervdj/data/v_cnn_trb
```

### How good can CDR3 -> TRBV get?

Benchmarked on PRJNA280417 (bulk TRB) with a CDR3-disjoint grouped split,
instance-weighted recall of the true V gene among 49 functional TRBV genes:

| ranker | Top-1 | Top-5 | Top-10 |
| --- | --- | --- | --- |
| rule pipeline (motif+usage) | 28 | 44 | 57 |
| char-ngram + LinearSVC | 39 | 62 | 74 |
| char-ngram + logistic reg. | 39 | 65 | 79 |
| **VGeneCNN** | **41** | **68** | **82** |
| oracle: P(V \| exact CDR3) | 87 | 96 | 99 |

Adding model capacity (6x) or training data (4x) each move Top-10 by <1 point,
so the CNN is at the **information limit of the amino-acid CDR3**: residual error
is intrinsic sequence ambiguity (~5% of CDR3s map to >1 V gene, concentrated in
short junctions), not a modelling gap. The oracle reaches 99% only by memorizing
exact sequences seen before — unavailable for novel CDR3s. The anchor filter
does not narrow TRB candidates (all 49 V anchors are compatible with `CAS(S)…`).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests do not require OLGA or SONIA model files.

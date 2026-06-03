# SuperVDJMachine / `supervdj`

Rank plausible **TRAV/TRAJ/TRBV/TRBJ** gene pairs for a given TCR CDR3 amino-acid
sequence using germline-anchor filters, motif priors, gene-usage priors, and
damped OLGA log-Pgen + SONIA selection scores.

> A CDR3 amino-acid sequence rarely uniquely identifies V and J because
> junctional additions and shared anchor motifs remove some of that information.
> This tool therefore returns a calibrated ranking, not a single definitive call.

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

```text
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

## Reference Data

Packaged IMGT human germline FASTAs live at `supervdj/data/imgt/`
(`TRAV/TRAJ/TRBV/TRBJ.fasta`). `supervdj/imgt.py` parses them, honors IMGT
`codon_start`, extracts CDR3 anchors (V: suffix from the conserved Cys; J:
prefix up to the F/W of the F/W-G-X-G motif), and collapses alleles to one entry
per gene. V genes keep the longest extracted anchor across alleles, with `*01`
used only as a tie-breaker. Pseudogenes/ORFs are dropped by default; pass
`--include-nonfunctional` to keep them.

Override at the CLI with a TSV (`gene/chain/gene_type/functional/anchor/usage_prior`):

```bash
supervdj predict --chain TRB --cdr3 CASSIRSSYEQYF \
    --v-ref trbv.tsv --j-ref trbj.tsv
```

## VDJdb Integration

`supervdj usage` prints empirical V/J usage frequencies, and `supervdj eval`
benchmarks ranking accuracy against any VDJdb-style `gene/cdr3/v/j` table:

```bash
# Empirical V/J usage table
supervdj usage --vdjdb vdjdb.tsv --chain TRB --top 10

# Accuracy benchmark
supervdj eval --vdjdb vdjdb.tsv --chain TRB --limit 1000 --no-olga --no-sonia
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests do not require OLGA or SONIA model files.

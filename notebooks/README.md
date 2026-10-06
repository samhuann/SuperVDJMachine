# Notebooks

`supervdj_demo.ipynb` takes one CDR3 amino acid sequence and a chain and returns the
posterior over V genes and over J genes, the calibrated candidate set for a requested
coverage, and the confusion group of the top-ranked V gene.

It calls `supervdj.posterior.preselection_posterior`, the same code the manuscript used,
rather than reimplementing anything, so the numbers it prints are the paper's numbers. It
runs in Colab with no local installation; the setup cell clones this repository and
installs `olga`.

For more than a handful of sequences the command line is much faster:

```
PYTHONPATH=. python -m supervdj.run --help
```

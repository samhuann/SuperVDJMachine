"""ingest: normalize heterogeneous TCR sources into one canonical table.

Canonical schema: ``cdr3_aa, v_gene, j_gene, chain, source`` (see
:mod:`ingest.schema`).  One adapter per source (:mod:`ingest.adapters`) feeds
the shared normalization (:mod:`ingest.normalize`) -- gene-name reconciliation
onto the OLGA gene set and CDR3 boundary normalization to OLGA's junction
convention -- then deduplication (:mod:`ingest.dedup`).  The canonical table is
read by ``supervdj.io.load_dataset`` unchanged.
"""

__all__ = ["schema", "gene_names", "imgt_boundaries", "normalize", "dedup", "adapters"]

"""IMGT germline FASTA loader and CDR3-anchor extraction.

The IMGT human-germline FASTAs ship in ``supervdj/data/imgt/`` as
``TRAV.fasta``, ``TRAJ.fasta``, ``TRBV.fasta``, ``TRBJ.fasta``. Each
record carries a pipe-delimited header of the form::

    >accession|gene*allele|species|functional|REGION|...|<nt>nt|<codon_start>|...

Key responsibilities of this module:

* Parse the IMGT header and skip non-canonical residues / IMGT gap dots
  (``.``) before translating.
* Translate respecting the IMGT ``codon_start`` field (1, 2, or 3).
* For V genes, extract the CDR3 N-terminal anchor as the protein suffix
  beginning at the conserved Cys at IMGT position 104 — in practice the
  last ``C`` in the V-REGION protein.
* For J genes, extract the CDR3 C-terminal anchor as the protein prefix
  ending at (and including) the conserved Phe/Trp of the F/W-G-X-G
  motif.
* Collapse multiple alleles to one gene-level entry. V genes keep the
  longest extracted CDR3 anchor available across alleles; ties prefer
  ``*01`` and then the lexicographically first allele.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from supervdj.models import GeneReference

# Standard codon table.
_CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_FGXG_MOTIF = re.compile(r"([FW])G.G")


@dataclass
class ImgtRecord:
    """One IMGT FASTA record parsed into headed metadata + raw NT."""

    accession: str
    gene: str  # gene name including allele, e.g., "TRBV6-5*01"
    species: str
    functional: str  # F / P / ORF / etc.
    region: str
    codon_start: int
    nt: str
    protein: str

    @property
    def gene_name(self) -> str:
        """Gene name without the ``*allele`` suffix."""
        return self.gene.split("*", 1)[0]


def _translate(nt: str, codon_start: int) -> str:
    """Translate nucleotides to protein, respecting IMGT ``codon_start``."""
    seq = nt.upper().replace("U", "T")
    seq = "".join(c for c in seq if c.isalpha())  # strip dots / whitespace
    offset = max(0, codon_start - 1)
    seq = seq[offset:]
    aa = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if len(codon) < 3:
            break
        aa.append(_CODON_TABLE.get(codon, "X"))
    protein = "".join(aa)
    # Stop at first stop codon so spurious frame artefacts do not leak in.
    stop = protein.find("*")
    if stop != -1:
        protein = protein[:stop]
    return protein


def parse_fasta(path: Path) -> List[ImgtRecord]:
    """Parse a single IMGT FASTA file into a list of :class:`ImgtRecord`."""
    records: List[ImgtRecord] = []
    header: Optional[str] = None
    nt_buf: List[str] = []

    def flush(header: Optional[str], nt_buf: List[str]) -> None:
        if header is None:
            return
        fields = [f.strip() for f in header.lstrip(">").split("|")]
        if len(fields) < 8:
            return  # malformed header
        accession = fields[0]
        gene = fields[1]
        species = fields[2]
        functional = fields[3]
        region = fields[4]
        try:
            codon_start = int(fields[7])
        except ValueError:
            codon_start = 1
        nt = "".join(nt_buf)
        protein = _translate(nt, codon_start)
        records.append(
            ImgtRecord(
                accession=accession,
                gene=gene,
                species=species,
                functional=functional,
                region=region,
                codon_start=codon_start,
                nt=nt,
                protein=protein,
            )
        )

    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                flush(header, nt_buf)
                header = line
                nt_buf = []
            else:
                nt_buf.append(line)
        flush(header, nt_buf)
    return records


def _v_anchor(protein: str, max_len: int = 0) -> Optional[str]:
    """Return V CDR3 anchor: suffix beginning at the last Cys.

    ``max_len <= 0`` keeps the full suffix.
    """
    idx = protein.rfind("C")
    if idx == -1:
        return None
    anchor = protein[idx:]
    return anchor[:max_len] if max_len > 0 else anchor


def _j_anchor(protein: str, max_len: int = 10) -> Optional[str]:
    """Return J CDR3 anchor: prefix up to and including the F/W of the FGXG motif."""
    m = _FGXG_MOTIF.search(protein)
    if not m:
        return None
    end = m.start() + 1  # include the F/W
    anchor = protein[:end]
    if max_len > 0 and len(anchor) > max_len:
        anchor = anchor[-max_len:]
    return anchor


def records_to_references(
    records: Iterable[ImgtRecord],
    chain: str,
    gene_type: str,
    include_nonfunctional: bool = False,
    v_anchor_len: int = 0,
    j_anchor_len: int = 10,
) -> List[GeneReference]:
    """Collapse IMGT records (gene*allele) into one ``GeneReference`` per gene.

    Alleles are merged at the gene level. V genes keep the longest extracted
    anchor. Ties prefer ``*01``; if no ``*01`` exists, the lexicographically
    first allele wins.
    """
    by_gene: Dict[str, Tuple[ImgtRecord, str]] = {}
    for rec in records:
        if rec.region not in {"V-REGION", "J-REGION"}:
            continue
        if not include_nonfunctional and rec.functional not in {"F", "(F)"}:
            continue
        if gene_type == "V":
            anchor = _v_anchor(rec.protein, max_len=v_anchor_len)
        else:
            anchor = _j_anchor(rec.protein, max_len=j_anchor_len)
        if not anchor:
            continue

        existing = by_gene.get(rec.gene_name)
        if existing is None:
            by_gene[rec.gene_name] = (rec, anchor)
            continue
        existing_rec, existing_anchor = existing
        longer_v_anchor = gene_type == "V" and len(anchor) > len(existing_anchor)
        tie_or_j = gene_type != "V" or len(anchor) == len(existing_anchor)
        preferred_allele = (
            tie_or_j
            and rec.gene.endswith("*01")
            and not existing_rec.gene.endswith("*01")
        )
        earlier_allele = (
            tie_or_j
            and rec.gene < existing_rec.gene
            and not existing_rec.gene.endswith("*01")
        )
        if longer_v_anchor or preferred_allele or earlier_allele:
            by_gene[rec.gene_name] = (rec, anchor)

    refs: List[GeneReference] = []
    for gene_name, (rec, anchor) in sorted(by_gene.items()):
        refs.append(
            GeneReference(
                gene=gene_name,
                chain=chain,
                gene_type=gene_type,
                functional=rec.functional,
                anchor=anchor,
                usage_prior=0.0,
            )
        )
    return refs


def _packaged_imgt_path(filename: str) -> Path:
    with resources.as_file(
        resources.files("supervdj").joinpath("data", "imgt", filename)
    ) as p:
        return Path(p)


def load_imgt_reference(
    chain: str,
    imgt_dir: Optional[Path] = None,
    include_nonfunctional: bool = False,
) -> Tuple[List[GeneReference], List[GeneReference]]:
    """Load V and J references for ``chain`` from IMGT FASTAs.

    Args:
        chain: ``TRA`` or ``TRB``.
        imgt_dir: Optional override of the IMGT FASTA directory. Defaults
            to the packaged ``supervdj/data/imgt/`` directory.
        include_nonfunctional: If True, include pseudogenes and ORFs.

    Returns:
        Tuple of ``(v_refs, j_refs)``.
    """
    chain = chain.upper()
    if chain == "TRA":
        v_name, j_name = "TRAV.fasta", "TRAJ.fasta"
    elif chain == "TRB":
        v_name, j_name = "TRBV.fasta", "TRBJ.fasta"
    else:
        raise ValueError(f"Unsupported chain: {chain!r}")

    if imgt_dir is None:
        v_path = _packaged_imgt_path(v_name)
        j_path = _packaged_imgt_path(j_name)
    else:
        v_path = Path(imgt_dir) / v_name
        j_path = Path(imgt_dir) / j_name

    v_records = parse_fasta(v_path)
    j_records = parse_fasta(j_path)
    v_refs = records_to_references(
        v_records, chain=chain, gene_type="V",
        include_nonfunctional=include_nonfunctional,
    )
    j_refs = records_to_references(
        j_records, chain=chain, gene_type="J",
        include_nonfunctional=include_nonfunctional,
    )
    return v_refs, j_refs

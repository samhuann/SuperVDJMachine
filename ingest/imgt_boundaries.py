"""IMGT germline FASTAs as the CDR3 *boundary* reference (no scoring).

This is one of the two sanctioned uses of the IMGT FASTAs: establish the
conserved CDR3 boundary residues so every source can be normalized to the
exact convention OLGA expects.  (The other use, later, is the alignment
baseline for the confusion-group analysis.)  This is NOT the old heuristic
reference-scoring layer — it only reads germline anchor residues.

Convention OLGA expects (the IMGT junction):

    CDR3 begins at the V-region conserved cysteine (C, IMGT position 104)
    and ends at the J-region conserved phenylalanine/tryptophan (F or W of
    the F/W-G-X-G motif), both included.

So canonical CDR3 == ``C ... [FW]``.  ``normalize_cdr3`` validates a CDR3
against this convention (the trailing residue is taken gene-aware from the
J germline anchor), routing non-conforming rows to the rejected file rather
than fabricating residues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# Default location of the IMGT FASTAs supplied as an input to this task.
DEFAULT_IMGT_DIR = Path(
    "/10TBDrive1/sam/SuperVDJMachine-OLD-/supervdj/data/imgt"
)

_CODONS = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_FGXG = re.compile(r"([FW])G.G")
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _translate(nt: str, codon_start: int) -> str:
    seq = "".join(c for c in nt.upper().replace("U", "T") if c.isalpha())
    seq = seq[max(0, codon_start - 1):]
    aa = []
    for i in range(0, len(seq) - 2, 3):
        aa.append(_CODONS.get(seq[i:i + 3], "X"))
    prot = "".join(aa)
    stop = prot.find("*")
    return prot[:stop] if stop != -1 else prot


def _gene_of(name: str) -> str:
    return name.split("*", 1)[0]


@dataclass
class ImgtBoundaries:
    """Per-gene conserved CDR3 boundary residues from the IMGT germline."""

    v_end_residue: Dict[str, str] = field(default_factory=dict)   # gene -> 'C'
    j_end_residue: Dict[str, str] = field(default_factory=dict)   # gene -> 'F'/'W'

    @classmethod
    def from_dir(cls, imgt_dir: Path = DEFAULT_IMGT_DIR) -> "ImgtBoundaries":
        b = cls()
        for fname in ("TRAV.fasta", "TRBV.fasta"):
            for gene, prot in cls._records(Path(imgt_dir) / fname):
                idx = prot.rfind("C")
                if idx != -1:
                    b.v_end_residue.setdefault(gene, "C")
        for fname in ("TRAJ.fasta", "TRBJ.fasta"):
            for gene, prot in cls._records(Path(imgt_dir) / fname):
                m = _FGXG.search(prot)
                if m:
                    b.j_end_residue.setdefault(gene, m.group(1))
        return b

    @staticmethod
    def _records(path: Path):
        if not path.is_file():
            return
        header, buf = None, []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        yield ImgtBoundaries._emit(header, buf)
                    header, buf = line, []
                else:
                    buf.append(line)
        if header is not None:
            yield ImgtBoundaries._emit(header, buf)

    @staticmethod
    def _emit(header: str, buf) -> Tuple[str, str]:
        f = [x.strip() for x in header.lstrip(">").split("|")]
        gene = _gene_of(f[1]) if len(f) > 1 else ""
        try:
            cs = int(f[7])
        except (IndexError, ValueError):
            cs = 1
        return gene, _translate("".join(buf), cs)

    def expected_end(self, j_gene: str) -> str:
        """Conserved trailing residue for ``j_gene`` (F or W); '' if unknown."""
        return self.j_end_residue.get(j_gene, "")

    def normalize_cdr3(self, cdr3: str, j_gene: str) -> Tuple[Optional[str], str]:
        """Validate a CDR3 against the OLGA junction convention (C ... F/W).

        Returns ``(cdr3 or None, reason)``.  The trailing residue is checked
        gene-aware against the J germline; a generic F/W is accepted when the
        J germline residue is unknown.
        """
        s = (cdr3 or "").strip().upper()
        if not s:
            return None, "empty_cdr3"
        bad = set(s) - _VALID_AA
        if bad:
            return None, f"cdr3:nonstandard_residue:{''.join(sorted(bad))}"
        if not s.startswith("C"):
            return None, "boundary:missing_conserved_C"
        end = self.expected_end(j_gene)
        if end:
            if not s.endswith(end):
                # Accept the other canonical residue too (F<->W) before rejecting.
                if s[-1] in ("F", "W"):
                    return s, ""
                return None, f"boundary:expected_end_{end}_got_{s[-1]}"
            return s, ""
        if s[-1] not in ("F", "W"):
            return None, f"boundary:missing_conserved_FW_got_{s[-1]}"
        return s, ""

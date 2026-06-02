"""Train and save a VGeneCNN V-gene model from a gene/cdr3/v table.

The CNN is the strongest CDR3-only V ranker found (see exp_trb_models.py /
eval_prjna_recall.py). Training takes ~40s, so the deployed model is trained
once here and loaded at predict time via ``--v-cnn-dir``.

Source columns may be either the cleaned ``gene/cdr3/v`` schema or a raw
repertoire table (``Vregion/Jregion/AASeq``); the latter is normalized first.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def _rows(path: Path, chain: str):
    df = pd.read_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t",
                     dtype=str, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    if "vregion" in cols and "aaseq" in cols:  # raw repertoire table
        v = df[cols["vregion"]]
        cdr3 = df[cols["aaseq"]]
        gene = pd.Series(chain, index=df.index)
    else:
        v = df[cols["v"]]
        cdr3 = df[cols["cdr3"]]
        gene = df[cols.get("gene", cols.get("v"))].str.upper() if "gene" in cols else \
            pd.Series(chain, index=df.index)
    out = pd.DataFrame({"cdr3": cdr3.str.strip().str.upper(),
                        "v": v.str.split("*").str[0].str.upper(),
                        "gene": gene})
    out = out[out["gene"].str.upper() == chain.upper()]
    out = out.dropna(subset=["cdr3", "v"])
    out = out[out["cdr3"].str.match(r"^C.*[FW]$")]
    out = out.drop_duplicates(["cdr3", "v"])
    return list(zip(out["cdr3"], out["v"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("supervdj/data/PRJNA280417.tsv"))
    p.add_argument("--chain", default="TRB")
    p.add_argument("--out", type=Path, default=Path("supervdj/data/v_cnn_trb"))
    p.add_argument("--epochs", type=int, default=12)
    args = p.parse_args(argv)

    from supervdj.v_model import VGeneCNN

    rows = _rows(args.train, args.chain)
    print(f"training VGeneCNN on {len(rows)} unique (cdr3,v) rows for {args.chain}...")
    model = VGeneCNN.from_rows(rows, chain=args.chain, epochs=args.epochs)
    model.save(args.out)
    print(f"saved {len(model.classes)}-class CNN to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

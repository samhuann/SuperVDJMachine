"""Train and save a VGeneCNN V-gene model from a gene/cdr3/v table."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def _rows(path: Path, chain: str):
    df = pd.read_csv(
        path,
        sep="," if path.suffix.lower() == ".csv" else "\t",
        dtype=str,
        low_memory=False,
    )
    cols = {c.lower(): c for c in df.columns}
    if "vregion" in cols and "aaseq" in cols:
        v = df[cols["vregion"]]
        cdr3 = df[cols["aaseq"]]
        gene = pd.Series(chain, index=df.index)
    else:
        v = df[cols["v"]]
        cdr3 = df[cols["cdr3"]]
        gene = (
            df[cols.get("gene", cols.get("v"))].str.upper()
            if "gene" in cols
            else pd.Series(chain, index=df.index)
        )
    out = pd.DataFrame(
        {
            "cdr3": cdr3.str.strip().str.upper(),
            "v": v.str.split("*").str[0].str.upper(),
            "gene": gene,
        }
    )
    out = out[out["gene"].str.upper() == chain.upper()]
    out = out.dropna(subset=["cdr3", "v"])
    out = out[out["cdr3"].str.match(r"^C.*[FW]$")]
    out = out.drop_duplicates(["cdr3", "v"])
    return list(zip(out["cdr3"], out["v"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("supervdj/data/PRJNA280417.tsv"))
    parser.add_argument("--chain", default="TRB")
    parser.add_argument("--out", type=Path, default=Path("supervdj/data/v_cnn_trb"))
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args(argv)

    from supervdj.v_model import VGeneCNN

    rows = _rows(args.train, args.chain)
    print(f"training VGeneCNN on {len(rows)} unique (cdr3,v) rows for {args.chain}...")
    model = VGeneCNN.from_rows(rows, chain=args.chain, epochs=args.epochs)
    model.save(args.out)
    print(f"saved {len(model.classes)}-class CNN to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

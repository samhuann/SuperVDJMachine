"""Plot V-model ablation curves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(results_dir: Path, outdir: Path, category: str = "V exact") -> Path:
    curve = pd.read_csv(results_dir / "topk_curve_TRB.tsv", sep="\t")
    curve = curve[curve["category"] == category].copy()
    preferred = [
        "olga_sonia",
        "v_model",
        "olga_sonia_plus_v_model_w0.02",
        "olga_sonia_plus_v_model_w0.05",
        "olga_sonia_plus_v_model_w0.1",
    ]
    arms = [arm for arm in preferred if arm in set(curve["arm"])]
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for arm in arms:
        sub = curve[curve["arm"] == arm]
        label = arm.replace("olga_sonia", "OLGA+SONIA").replace("_plus_v_model_w", " + V model w=")
        label = label.replace("v_model", "V model")
        ax.plot(sub["k"], sub["fraction_recovered"], label=label)
    ax.set_xlabel("Number of candidates considered (k)")
    ax.set_ylabel("Fraction recovered")
    ax.set_title(f"TRB {category} ablation")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = outdir / "v_model_ablation_TRB.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot V-model ablation benchmark.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/v_model_ablation"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--category", default="V exact")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = plot_summary(args.results_dir, args.outdir, args.category)
    print(f"wrote figure to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot manuscript-ready recovery curves from benchmark TSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

CATEGORIES = (
    "V exact",
    "J exact",
    "V+J exact",
    "V family",
    "J family",
    "V+J family",
)


def _style_axes(ax, title: str = "") -> None:
    ax.set_xlabel("Number of candidates considered (k)")
    ax.set_ylabel("Fraction recovered")
    if title:
        ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linewidth=0.5, alpha=0.35)


def plot_chain_curve(results_dir: Path, outdir: Path, chain: str) -> None:
    df = pd.read_csv(results_dir / f"topk_curve_{chain}.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for category in CATEGORIES:
        sub = df[df["category"] == category]
        ax.plot(sub["k"], sub["fraction_recovered"], marker=None, label=category)
    _style_axes(ax, f"{chain} V/J recovery")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"topk_recovery_{chain}.png")
    plt.close(fig)


def plot_combined(results_dir: Path, outdir: Path) -> None:
    chains = [
        path.stem.replace("topk_curve_", "")
        for path in sorted(results_dir.glob("topk_curve_*.tsv"))
    ]
    fig, axes = plt.subplots(1, len(chains), figsize=(7.2 * len(chains), 4.8), dpi=160)
    if len(chains) == 1:
        axes = [axes]
    for ax, chain in zip(axes, chains):
        df = pd.read_csv(results_dir / f"topk_curve_{chain}.tsv", sep="\t")
        for category in CATEGORIES:
            sub = df[df["category"] == category]
            ax.plot(sub["k"], sub["fraction_recovered"], marker=None, label=category)
        _style_axes(ax, f"{chain} V/J recovery")
    axes[-1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "topk_recovery_combined.png")
    plt.close(fig)


def plot_v_family_vs_exact(results_dir: Path, outdir: Path) -> None:
    df = pd.read_csv(results_dir / "recovery_by_chain.tsv", sep="\t")
    chains = sorted(df["chain"].unique())
    rows = []
    for chain in chains:
        for category in ("V exact", "V family"):
            sub = df[(df["chain"] == chain) & (df["category"] == category)]
            if not sub.empty:
                rows.append(
                    {
                        "chain": chain,
                        "category": category,
                        "top1": float(sub.iloc[0].get("top1", 0.0)),
                        "top5": float(sub.iloc[0].get("top5", 0.0)),
                        "top10": float(sub.iloc[0].get("top10", 0.0)),
                    }
                )
    plot_df = pd.DataFrame(rows)
    metrics = ["top1", "top5", "top10"]
    labels = [m.replace("top", "Top-") for m in metrics]

    fig, axes = plt.subplots(1, len(chains), figsize=(5.6 * len(chains), 4.6), dpi=160)
    if len(chains) == 1:
        axes = [axes]
    width = 0.38
    x = range(len(metrics))
    for ax, chain in zip(axes, chains):
        exact = plot_df[(plot_df["chain"] == chain) & (plot_df["category"] == "V exact")]
        family = plot_df[(plot_df["chain"] == chain) & (plot_df["category"] == "V family")]
        exact_values = [float(exact.iloc[0][m]) for m in metrics] if not exact.empty else [0, 0, 0]
        family_values = [float(family.iloc[0][m]) for m in metrics] if not family.empty else [0, 0, 0]
        ax.bar([i - width / 2 for i in x], exact_values, width=width, label="V exact")
        ax.bar([i + width / 2 for i in x], family_values, width=width, label="V family")
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Fraction recovered")
        ax.set_title(chain)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "v_family_vs_exact_accuracy.png")
    plt.close(fig)


def plot_confusion(results_dir: Path, outdir: Path, chain: str, max_labels: int = 15) -> None:
    path = results_dir / f"confusion_V_family_{chain}.tsv"
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
        ax.text(0.5, 0.5, "No family-level V confusions", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / f"v_confusion_family_{chain}.png")
        plt.close(fig)
        return

    df = df.sort_values("n", ascending=False).head(max_labels)
    truth = list(dict.fromkeys(df["truth"].tolist()))
    predicted = list(dict.fromkeys(df["predicted"].tolist()))
    matrix = pd.DataFrame(0, index=truth, columns=predicted, dtype=float)
    for _, row in df.iterrows():
        matrix.loc[row["truth"], row["predicted"]] = row["n"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(range(len(predicted)), predicted, rotation=45, ha="right")
    ax.set_yticks(range(len(truth)), truth)
    ax.set_xlabel("Predicted V family")
    ax.set_ylabel("Annotated V family")
    ax.set_title(f"{chain} V-family top-1 confusions")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(outdir / f"v_confusion_family_{chain}.png")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot VDJdb recovery benchmark curves.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/vdjdb_recovery"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    chains = [
        path.stem.replace("topk_curve_", "")
        for path in sorted(args.results_dir.glob("topk_curve_*.tsv"))
    ]
    for chain in chains:
        plot_chain_curve(args.results_dir, args.outdir, chain)
        plot_confusion(args.results_dir, args.outdir, chain)
    plot_combined(args.results_dir, args.outdir)
    plot_v_family_vs_exact(args.results_dir, args.outdir)
    print(f"wrote figures to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

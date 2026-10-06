"""Render paper-style figures from newly computed summaries."""
from pathlib import Path
from ..utils.io import read_json


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(args.output)
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    if args.kind == "threshold":
        summary = read_json(root / "evaluation.json")["conditions"]
        configs = read_json(root / "manifest.json")["conditions"]
        selected = sorted((c["margin"], c["name"]) for c in configs
                          if c["name"] == "care" or c["name"].startswith("delta_0"))
        selected = [(d, name) for d, name in selected if 0.4 <= d <= 0.9]
        if not selected:
            raise ValueError("No threshold sweep results")
        x = [d for d, _ in selected]
        rows = [summary[name]["summary"]["her"] for _, name in selected]
        y = np.array([r["mean"] * 100 for r in rows])
        errors = np.array([[r["mean"] - r["min"] for r in rows], [r["max"] - r["mean"] for r in rows]]) * 100
        fig, ax = plt.subplots(figsize=(6.4, 4), layout="constrained")
        ax.errorbar(x, y, yerr=errors, color="#287d76", marker="o", capsize=4)
        ax.set(xlabel="Confidence threshold", ylabel="HER (%)", title="Threshold sensitivity (with mask)")
        ax.set_xticks(x)
        name = "threshold"
    else:
        summary = read_json(root / "case_study" / args.condition / "summary.json")
        categories = ["drug", "symptom", "parameter", "other"]
        labels = ["Drug names", "Symptoms", "Parameter-related", "Other"]
        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        bottom = np.zeros(4)
        series = [("correct", "Correct", "#388c87"), ("harmful", "Harmful", "#c26a56"), ("ineffective", "Ineffective", "#d6d9dc")]
        if summary["nogold_policy"] == "separate":
            series.append(("unmatched", "Unmatched", "#8c77a5"))
        for field, label, color in series:
            values = np.array([summary["mean"][c][field] for c in categories])
            ax.bar(range(4), values, bottom=bottom, label=label, color=color, width=.6)
            bottom += values
        ax.set_xticks(range(4), labels)
        ax.set(ylabel="Accepted proposals (mean count)", title="Outcomes of accepted proposals")
        ax.legend(frameon=False)
        name = "case_study"
    ax.spines[["top", "right"]].set_visible(False)
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"{name}.{suffix}", dpi=200)
    plt.close(fig)
    print(f"Saved figures to {destination}")
